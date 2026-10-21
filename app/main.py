from __future__ import annotations

import asyncio
import json
import os

import httpx
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from .web.session import Forbidden, RedirectToLogin, SetupWizardRequired, current_user
from .web.accounts import get_auth_settings, router as accounts_router
from .web.routes import router as web_router
from .web.ops import router as ops_router
from .auto_route import auto_alias_list_entries
from .model_aliases import alias_list_entries, apply_client_model_rewrites
from .auth.check import authorize, extract_model, extract_request_model
from .auth.concurrency import ConcurrencyLease, release_concurrency_lease
from .data.backends import (
    address_for_source,
    get_source_by_name,
)
from .routing import resolve_routed_source
from .config import (
    MODEL_ROUTE_KINDS,
    SESSION_COOKIE_NAME,
    session_cookie_https_only,
    get_settings,
    kind_from_upstream_path,
    map_upstream_path,
    onprem_api_port,
    split_source_path,
    upstream_path_for_proxy,
)
from .data.db import get_db, init_db
from .anthropic_messages import (
    ANTHROPIC_SSE_PING_INTERVAL_SEC,
    AnthropicTranslateError,
    OpenAIToAnthropicStream,
    anthropic_error_body,
    anthropic_request_to_openai,
    anthropic_sse_error,
    anthropic_sse_ping,
    is_anthropic_messages_path,
)
from .vision_route import (
    mint_forward_ticket,
    parse_forward_ticket,
    request_needs_vision,
    resolve_vl_model,
    rewrite_json_model,
)


def _unresolved_model_response(kind: str, model: str | None) -> JSONResponse:
    """No enabled catalog match — do not dump the request on a kind-default box."""
    if not (model or "").strip():
        return JSONResponse({"error": "missing_model", "kind": kind}, status_code=400)
    return JSONResponse(
        {"error": "unknown_model", "kind": kind, "model": model},
        status_code=404,
    )


def _lease_from_ticket(payload: dict) -> ConcurrencyLease | None:
    kid = payload.get("kid")
    if kid is None:
        return None
    try:
        key_id = int(kid)
    except (TypeError, ValueError):
        return None
    owner = payload.get("owner")
    try:
        user_id = int(owner) if owner is not None else None
    except (TypeError, ValueError):
        user_id = None
    mdl = str(payload.get("mdl") or "").strip() or None
    sk = str(payload.get("sk") or "").strip() or None
    return ConcurrencyLease(key_id=key_id, user_id=user_id, model=mdl, source_key=sk)


def _attach_source_key(
    lease: ConcurrencyLease | None, backend: str
) -> ConcurrencyLease | None:
    if lease is None:
        return None
    key = (backend or "").strip() or None
    return ConcurrencyLease(
        key_id=lease.key_id,
        user_id=lease.user_id,
        model=lease.model,
        source_key=key,
    )


def _source_admission_block(
    *,
    auth_cfg,
    src,
    backend: str,
    kind: str,
    model: str | None,
    service: str,
    lease: ConcurrencyLease | None,
    priority: int,
) -> JSONResponse | None:
    if auth_cfg is None or not getattr(auth_cfg, "source_admission_enabled", True):
        return None
    if src is None or lease is None:
        return None
    from .auth.source_admission import try_acquire_source_admission

    outcome = try_acquire_source_admission(
        src=src,
        backend=backend,
        kind=kind,
        model=model,
        key_id=lease.key_id,
        priority=priority,
        platform_queue_timeout_sec=int(
            getattr(auth_cfg, "source_queue_timeout_sec", 30) or 30
        ),
        enabled=True,
        max_hold_sec=int(getattr(auth_cfg, "source_admission_max_hold_sec", 0) or 0),
    )
    if outcome.acquired:
        return None
    release_concurrency_lease(lease)
    return JSONResponse(
        {
            "error": outcome.reason,
            "service": service,
            "limit": outcome.limit,
            "retry_after": outcome.retry_after,
        },
        status_code=503,
        headers={"Retry-After": str(outcome.retry_after)},
    )


def _preflight_block(
    *,
    auth_cfg,
    backend: str,
    kind: str,
    model: str | None,
    service: str,
    lease: ConcurrencyLease | None,
    src=None,
) -> JSONResponse | None:
    if auth_cfg is None or not getattr(auth_cfg, "preflight_upstream", False):
        return None
    from .upstream_preflight import preflight_upstream

    pf = preflight_upstream(
        backend=backend,
        kind=kind,
        model=model,
        engine_override=getattr(src, "engine_override", None) if src is not None else None,
        detected_engine=getattr(src, "detected_engine", None) if src is not None else None,
    )
    if pf.ok:
        return None
    release_concurrency_lease(lease)
    # Busy/loading → 503 + Retry-After (agents may retry).
    # Unreachable/probe fail → 502, not retryable (stop endless 503 loops).
    body = {
        "error": pf.reason,
        "service": service,
        "engine": pf.engine,
        "detail": pf.detail,
        "retryable": pf.retryable,
    }
    headers: dict[str, str] = {}
    if pf.retryable and pf.retry_after > 0:
        body["retry_after"] = pf.retry_after
        headers["Retry-After"] = str(pf.retry_after)
        status = 503
    else:
        status = 502
    return JSONResponse(body, status_code=status, headers=headers)


async def _source_admission_block_async(**kwargs) -> JSONResponse | None:
    """Admission may wait up to queue_timeout (default 30s) — never on the event loop."""
    return await asyncio.to_thread(lambda: _source_admission_block(**kwargs))


async def _preflight_block_async(**kwargs) -> JSONResponse | None:
    """Upstream probe is sync httpx — never on the event loop."""
    return await asyncio.to_thread(lambda: _preflight_block(**kwargs))

def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="OnPrem AI Gateway", docs_url=None, redoc_url=None)
    app.state.settings = settings
    init_db(settings)

    @app.on_event("startup")
    def _log_public_urls() -> None:
        port = os.getenv("PORT", "8080")
        web_port = os.getenv("WEB_PORT") or port
        api_port = onprem_api_port()
        base = f"http://127.0.0.1:{web_port}"
        print(f"Web UI:       {base}", flush=True)
        print(f"  Platform:   {base}/          (platform admin ops)", flush=True)
        print(f"  User:       {base}/me        (team members)", flush=True)
        print(f"  Login:      {base}/login", flush=True)
        print(f"  Account:    {base}/account   (password / email)", flush=True)
        print(f"  Forgot:     {base}/forgot", flush=True)
        print(f"  Register:   {base}/register (if enabled in Settings)", flush=True)
        print(f"OnPrem API:   http://127.0.0.1:{api_port}", flush=True)
        ph = (settings.public_host or "").strip() or f"127.0.0.1:{api_port}"
        print(f"  example:    http://{ph}/v1/chat/completions  (+ X-Api-Key)", flush=True)
        from .catalog_auto_sync import start_catalog_auto_sync
        from .data.fleet_status import start_fleet_status_worker

        start_catalog_auto_sync()
        start_fleet_status_worker()

    static_dir = Path(__file__).parent / "web" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Pure ASGI — do NOT use @app.middleware("http") / BaseHTTPMiddleware here.
    # That middleware buffers the receive channel and breaks request.body() for
    # nginx auth_request POSTs (hang → ClientDisconnect → 504 on /_auth).
    _PW_ALLOW = frozenset(
        {
            "/login",
            "/logout",
            "/forgot",
            "/reset",
            "/register",
            "/account",
            "/account/update",
            "/account/password",
            "/account/email",
            "/privacy",
            "/privacy/wipe-usage",
            "/healthz",
            "/api/hello",
        }
    )

    class _ForcePasswordChangeASGI:
        def __init__(self, app_):
            self.app = app_

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return
            path = scope.get("path") or ""
            if path.startswith("/static") or path.startswith("/v1/") or path.startswith("/legal") or path in _PW_ALLOW:
                await self.app(scope, receive, send)
                return
            uid = (scope.get("session") or {}).get("user_id")
            if uid:
                from .data.db import SessionLocal
                from .data.models import WebUser

                if SessionLocal is not None:
                    db = SessionLocal()
                    try:
                        u = db.get(WebUser, uid)
                        if u and u.must_change_password:
                            resp = RedirectResponse("/account", status_code=303)
                            await resp(scope, receive, send)
                            return
                    finally:
                        db.close()
            await self.app(scope, receive, send)

    # First added = innermost; SessionMiddleware last = outermost (sees cookies first).
    app.add_middleware(_ForcePasswordChangeASGI)
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        session_cookie=SESSION_COOKIE_NAME,
        max_age=settings.session_max_age,
        same_site="lax",
        https_only=session_cookie_https_only(settings),
    )

    @app.exception_handler(RedirectToLogin)
    async def _redirect_login(_request: Request, _exc: RedirectToLogin):
        return RedirectResponse("/login", status_code=303)

    @app.exception_handler(SetupWizardRequired)
    async def _redirect_setup(_request: Request, exc: SetupWizardRequired):
        return RedirectResponse(exc.redirect_to, status_code=303)

    from .web.templating import make_templates

    ui_templates = make_templates()

    @app.exception_handler(Forbidden)
    async def _forbidden(request: Request, _exc: Forbidden):
        user = None
        from .data.db import SessionLocal

        if SessionLocal is not None:
            db = SessionLocal()
            try:
                user = current_user(request, db)
            except Exception:
                user = None
            finally:
                db.close()
        return ui_templates.TemplateResponse(
            request,
            "forbidden.html",
            {"user": user, "nav": ""},
            status_code=403,
        )

    @app.get("/healthz")
    async def healthz():
        # async: never waits on the sync-endpoint threadpool (UI probes / DB work).
        from . import __version__

        return {"status": "ok", "version": __version__}

    @app.api_route("/api/hello", methods=["GET", "HEAD"])
    async def api_hello():
        """Claude Code / Anthropic clients probe this path on the API host."""
        return {"ok": True, "service": "onprem-auth"}

    @app.get("/v1/models")
    @app.get("/v1/onprem/models")
    def onprem_models(request: Request, db: Session = Depends(get_db)):
        """OpenAI-compatible model list (public path /v1/models; nginx → /v1/onprem/models)."""
        from .data.catalog import (
            load_api_key,
            models_list_kinds,
            models_visible_for_key,
            openai_models_payload,
        )
        from .data.models import utcnow

        raw_key = request.headers.get("x-api-key") or ""
        auth = request.headers.get("authorization") or ""
        if not raw_key and auth.lower().startswith("bearer "):
            raw_key = auth[7:].strip()
        api_key = load_api_key(db, raw_key)
        if api_key is None:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if api_key.expires_at and api_key.expires_at < utcnow():
            return JSONResponse({"error": "expired_api_key"}, status_code=401)
        rows = models_visible_for_key(
            db,
            api_key,
            kinds=models_list_kinds(request.query_params.get("kinds")),
        )
        from .auth.check import _models_for_key, _services_for_key
        from .data.source_capacity import capacities_by_source
        from .model_aliases import hidden_catalog_ids, list_aliases

        allowed_sources = _services_for_key(api_key, db)
        # Model allowlists are per-source; for alias visibility we union chat-like
        # grants. Unrestricted (None) on any granted chat source → show all aliases.
        allow_union: set[str] | None = None
        restricted = False
        for src in allowed_sources:
            mset = _models_for_key(api_key, src, db)
            if mset is None:
                allow_union = None
                restricted = False
                break
            restricted = True
            allow_union = (allow_union or set()) | set(mset)
        if not restricted:
            allow_union = None

        # Capacity from load_cache only (fleet worker / admission keep it warm).
        # Never block /v1/models on multi-second upstream timeouts.
        slot_sources: set[str] = {r.source_name for r in rows}
        from .data.models import CatalogModel

        for a in list_aliases(db, enabled_only=True):
            pref = (a.preferred_source or "").strip()
            if pref:
                slot_sources.add(pref)
                continue
            target = (a.target_model_id or "").strip()
            if not target:
                continue
            cat = (
                db.query(CatalogModel)
                .filter(
                    CatalogModel.model_id == target,
                    CatalogModel.enabled.is_(True),
                )
                .first()
            )
            if cat is not None:
                slot_sources.add(cat.source_name)
        capacity = capacities_by_source(db, slot_sources, probe=False)

        payload = openai_models_payload(rows, capacity_by_source=capacity)
        hidden = hidden_catalog_ids(db)
        if hidden:
            payload["data"] = [
                x
                for x in (payload.get("data") or [])
                if x.get("id") not in hidden
            ]
        aliases = auto_alias_list_entries(get_auth_settings(db)) + alias_list_entries(
            db,
            allow=allow_union,
            allowed_sources=allowed_sources if allowed_sources else None,
            capacity_by_source=capacity,
        )
        if aliases:
            seen = {x.get("id") for x in payload.get("data") or []}
            extra = [a for a in aliases if a["id"] not in seen]
            payload["data"] = extra + list(payload.get("data") or [])
        return payload

    @app.api_route("/v1/auth/check", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"])
    async def auth_check(request: Request, db: Session = Depends(get_db)):
        settings = request.app.state.settings
        host = request.headers.get("x-original-host") or request.headers.get("host") or ""
        uri = request.headers.get("x-original-uri") or request.url.path
        method = request.headers.get("x-original-method") or request.method
        client_ip = request.headers.get("x-real-ip") or (
            request.client.host if request.client else ""
        )
        raw_key = request.headers.get("x-api-key")
        named, upstream_uri = split_source_path(uri)
        kind = None
        service = None
        body = None
        content_type = request.headers.get("content-type")
        method_u = method.upper()
        vl_rewrite: str | None = None

        if named is not None:
            src = get_source_by_name(db, named)
            if src is None:
                return JSONResponse(
                    {"error": "unknown_source", "source": named, "path": uri},
                    status_code=403,
                )
            service = src.name
            kind = src.kind
            upstream_uri = upstream_uri or "/"
            if kind in MODEL_ROUTE_KINDS and method_u in {"POST", "PUT", "PATCH"}:
                body = await request.body()
            if kind == "chat" and body:
                asked = extract_model(body, content_type)
                body, rewritten, _pref = apply_client_model_rewrites(
                    db, body, asked=asked, auth=get_auth_settings(db)
                )
                if rewritten:
                    vl_rewrite = rewritten
        else:
            upstream_uri = upstream_path_for_proxy(uri)
            kind = kind_from_upstream_path(upstream_uri)
            if kind is None:
                return JSONResponse(
                    {"error": "unknown_route", "host": host, "path": uri},
                    status_code=403,
                )
            # Read body early so model routing can pick the source
            if kind in MODEL_ROUTE_KINDS and method_u in {"POST", "PUT", "PATCH"}:
                body = await request.body()
            alias_pref: str | None = None
            if kind == "chat" and body:
                asked = extract_model(body, content_type)
                body, rewritten, alias_pref = apply_client_model_rewrites(
                    db, body, asked=asked, auth=get_auth_settings(db)
                )
                if rewritten:
                    vl_rewrite = rewritten
            model = extract_request_model(kind, body, content_type) if body else None
            auth_cfg = get_auth_settings(db)
            src = resolve_routed_source(
                db,
                kind,
                model=model,
                raw_key=raw_key,
                auth=auth_cfg,
                preferred_source=alias_pref,
            )
            if src is None:
                return _unresolved_model_response(kind, model)
            service = src.name

        # Optional VL: detect image parts → authorize + forward as vision sibling
        if (
            kind == "chat"
            and body
            and get_auth_settings(db).auto_vl_routing
            and request_needs_vision(body, content_type)
        ):
            asked = extract_model(body, content_type)
            sibling = resolve_vl_model(db, service, asked)
            if sibling and sibling != asked:
                body = rewrite_json_model(body, sibling)
                vl_rewrite = sibling

        result = authorize(
            db,
            raw_key=raw_key,
            service=service,
            kind=kind,
            method=method,
            path=upstream_uri,
            host=host,
            client_ip=client_ip,
            body=body,
            content_type=content_type,
            # VL forward hop finalizes Wh; other auth_request paths do not meter.
            defer_metering=bool(vl_rewrite),
        )

        headers = {}
        if result.remaining_rpm is not None:
            headers["X-RateLimit-Remaining"] = str(result.remaining_rpm)
        headers["X-Priority"] = str(result.priority)
        headers["X-Auth-Reason"] = result.reason

        if result.status != 204:
            return JSONResponse(
                {"error": result.reason, "service": service},
                status_code=result.status,
                headers=headers,
            )

        backend = address_for_source(db, service)
        if not backend:
            release_concurrency_lease(result.concurrency_lease)
            return JSONResponse(
                {"error": "backend_not_configured", "service": service},
                status_code=503,
                headers=headers,
            )

        headers["X-Backend"] = backend
        # Always rewrite so nginx can map OpenAI client paths → backend dialect
        # (piper /audio/speech, whisper.cpp /inference, /s/{name}/ strip, …)
        src_row = get_source_by_name(db, service)
        api_style = getattr(src_row, "api_style", None) if src_row is not None else None
        rewrite_uri = map_upstream_path(
            upstream_uri, kind=kind, api_style=api_style
        )
        headers["X-Rewrite-Uri"] = rewrite_uri

        # VL-only: body rewrite needs a second hop (stock nginx cannot patch JSON).
        # Chat metering uses /v1/onprem/entry (see nginx) — not auth_request+forward.
        if vl_rewrite:
            pf_block = await _preflight_block_async(
                auth_cfg=get_auth_settings(db),
                backend=backend,
                kind=kind,
                model=result.model,
                service=service,
                lease=result.concurrency_lease,
                src=src_row,
            )
            if pf_block is not None:
                return pf_block
            adm_block = await _source_admission_block_async(
                auth_cfg=get_auth_settings(db),
                src=src_row,
                backend=backend,
                kind=kind,
                model=result.model,
                service=service,
                lease=result.concurrency_lease,
                priority=result.priority,
            )
            if adm_block is not None:
                return adm_block
            vl_lease = _attach_source_key(result.concurrency_lease, backend)
            headers["X-OnPrem-Proxy"] = "1"
            headers["X-OnPrem-Ticket"] = mint_forward_ticket(
                secret=settings.session_secret,
                service=service,
                backend=backend,
                rewrite_uri=rewrite_uri,
                rewrite_model=vl_rewrite,
                usage_id=result.usage_event_id,
                concurrency_lease=vl_lease,
            )
            headers["X-Auth-Reason"] = f"ok;vl={vl_rewrite}"

        return Response(status_code=204, headers=headers)

    async def _stream_upstream_metered(
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        usage_id: int | None,
        probe_url: str = "",
        concurrency_lease: ConcurrencyLease | None = None,
        max_hold_sec: int = 0,
        anthropic_client: bool = False,
        anthropic_model: str | None = None,
        anthropic_stream: bool = False,
        pre_upstream=None,
        prepare=None,
    ):
        """Shared streaming proxy + per-source GPU sample → finalize UsageEvent.

        max_hold_sec > 0: wall-clock kill of the upstream stream, then finally
        releases the admission/concurrency lease (no silent slot steal).

        anthropic_client: translate OpenAI upstream bytes → Anthropic Messages
        for the client; metering still parses the OpenAI buffer.
        anthropic_stream: client asked for SSE (prefer event-stream CT).

        Anthropic SSE: return HTTP 200 + event-stream **before** waiting on
        upstream headers, and emit pings while llama.cpp is still connecting /
        queueing — otherwise Claude Code idle-aborts (multi-minute retry).

        Auth is NEVER skipped. ``prepare`` (and/or ``pre_upstream``) still run
        ``authorize`` + routing + admission inside the generator; failed auth
        yields an SSE ``error`` event and never opens upstream.

        prepare: optional async () -> dict | str. Runs after bootstrap+ping.
        On success returns connection fields (url/headers/body/usage_id/…).
        On failure returns an error string (SSE error, no upstream).
        pre_upstream: optional async () -> str | None for deferred gates only
        when url is already known (legacy path).
        """
        import asyncio
        import time as time_mod
        from contextlib import suppress

        from .audit import finalize_usage_metering
        from .auth.source_admission import clamp_max_hold_sec
        from .data.db import SessionLocal
        from .metering_parse import parse_upstream_metrics
        from .usage_pool import fetch_gpu_watts, watt_hours_from_samples

        # Mutable connection state — ``prepare`` may fill these after SSE starts.
        conn: dict = {
            "url": url,
            "headers": dict(headers) if headers else {},
            "body": body,
            "usage_id": usage_id,
            "concurrency_lease": concurrency_lease,
            "hold_sec": clamp_max_hold_sec(max_hold_sec),
            "probe": (probe_url or "").strip(),
            "anthropic_model": anthropic_model,
        }
        hold_sec = conn["hold_sec"]

        # Cap buffered body for usage/timings parse (keep full for small JSON; tail for huge SSE).
        _BUF_MAX = 2 * 1024 * 1024
        _BUF_TAIL = 256 * 1024

        samples: list[float] = []
        probe = conn["probe"]
        if probe:
            power_status = "unreachable"  # until we get a sample
        else:
            power_status = "no_probe"

        async def _sample() -> None:
            nonlocal power_status
            p = conn["probe"]
            if not p:
                return
            w = await asyncio.to_thread(fetch_gpu_watts, p)
            if w is not None and w > 0:
                samples.append(float(w))
                power_status = "metered"

        t0 = time_mod.perf_counter()
        # Cap total/read wait at max hold when set; otherwise long streams OK.
        if hold_sec > 0:
            client_timeout = httpx.Timeout(hold_sec, connect=10.0)
        else:
            client_timeout = httpx.Timeout(3600.0, connect=10.0)

        early_anthropic_sse = bool(anthropic_client and anthropic_stream)
        # Anthropic SSE must return StreamingResponse with zero awaits first —
        # even a short probe delay postpones the first client byte.
        if not early_anthropic_sse:
            await _sample()

        def _finalize(
            *,
            buf: bytes,
            upstream_status: int,
            content_type: str | None,
        ) -> None:
            # Called from async finally via sync DB — keep sync like before.
            uid = conn["usage_id"]
            if uid is None or SessionLocal is None:
                return
            duration_ms = (time_mod.perf_counter() - t0) * 1000.0
            avg_w, wh = watt_hours_from_samples(
                samples=samples,
                duration_sec=max(0.001, duration_ms / 1000.0),
            )
            status = "metered" if samples else power_status
            metrics = parse_upstream_metrics(
                buf,
                duration_ms=duration_ms,
                content_type=content_type,
            )
            db = SessionLocal()
            try:
                finalize_usage_metering(
                    db,
                    uid,
                    duration_ms=duration_ms,
                    watts=avg_w,
                    watt_hours=wh,
                    upstream_status=upstream_status,
                    power_status=status,
                    tokens_in=metrics.tokens_in,
                    tokens_out=metrics.tokens_out,
                    pp_tok_s=metrics.pp_tok_s,
                    tg_tok_s=metrics.tg_tok_s,
                )
            finally:
                db.close()

        if early_anthropic_sse:
            import logging

            logging.getLogger("uvicorn.error").info(
                "anthropic_sse early start model=%s",
                (anthropic_model or "")[:80],
            )
            sse_headers = {
                "content-type": "text/event-stream",
                "cache-control": "no-cache",
                "x-accel-buffering": "no",
                "x-onprem-dialect": "anthropic-sse",
            }

            async def _stream_early():
                nonlocal power_status
                import logging as _log

                _sse_log = _log.getLogger("uvicorn.error")
                # Long timeout until prepare fills hold_sec; recreate if needed.
                client = httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=10.0))
                upstream = None
                upstream_status = 502
                content_type: str | None = None
                buf = bytearray()
                truncated = False
                watchdog: asyncio.Task | None = None
                pending: asyncio.Task | None = None
                send_task: asyncio.Task | None = None
                prep_task: asyncio.Task | None = None
                gate_task: asyncio.Task | None = None
                xform = OpenAIToAnthropicStream(model=conn["anthropic_model"])
                last_sample = time_mod.perf_counter()

                def _meter_chunk(chunk: bytes) -> None:
                    nonlocal truncated
                    if not truncated:
                        if len(buf) + len(chunk) <= _BUF_MAX:
                            buf.extend(chunk)
                        else:
                            buf.extend(chunk)
                            if len(buf) > _BUF_TAIL:
                                del buf[: len(buf) - _BUF_TAIL]
                                truncated = True
                    else:
                        buf.extend(chunk)
                        if len(buf) > _BUF_TAIL:
                            del buf[: len(buf) - _BUF_TAIL]

                async def _ping_while(task: asyncio.Task):
                    nonlocal last_sample
                    while not task.done():
                        done, _ = await asyncio.wait(
                            {task},
                            timeout=ANTHROPIC_SSE_PING_INTERVAL_SEC,
                        )
                        if not done:
                            yield anthropic_sse_ping()
                            now = time_mod.perf_counter()
                            if conn["probe"] and (now - last_sample) >= 1.0:
                                await _sample()
                                last_sample = now

                async def _reap_side_task(task: asyncio.Task | None) -> None:
                    """Cancel orphaned prepare/gate; release lease if never adopted."""
                    if task is None:
                        return
                    if not task.done():
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
                        return
                    with suppress(Exception):
                        result = task.result()
                        # Prepare finished after client disconnect and "transferred"
                        # the lease into the dict — release if stream never adopted it.
                        if (
                            isinstance(result, dict)
                            and conn.get("concurrency_lease") is None
                        ):
                            release_concurrency_lease(result.get("concurrency_lease"))

                try:
                    for piece in xform.bootstrap():
                        yield piece
                    yield anthropic_sse_ping()

                    # Auth + routing + admission MUST still run — just inside SSE
                    # so Claude gets keepalive bytes while authorize/DB/queue work.
                    if prepare is not None:
                        prep_task = asyncio.create_task(prepare())
                        async for ping in _ping_while(prep_task):
                            yield ping
                        prep = prep_task.result()
                        prep_task = None  # consumed; finally must not re-release
                        if isinstance(prep, str):
                            yield anthropic_sse_error(prep, err_type="api_error")
                            for piece in xform.flush():
                                yield piece
                            return
                        if not isinstance(prep, dict):
                            yield anthropic_sse_error(
                                "auth_setup_failed", err_type="api_error"
                            )
                            for piece in xform.flush():
                                yield piece
                            return
                        conn["url"] = prep["url"]
                        conn["headers"] = prep["headers"]
                        conn["body"] = prep["body"]
                        conn["usage_id"] = prep.get("usage_id")
                        conn["concurrency_lease"] = prep.get("concurrency_lease")
                        conn["hold_sec"] = clamp_max_hold_sec(
                            int(prep.get("max_hold_sec") or 0)
                        )
                        conn["probe"] = (prep.get("probe_url") or "").strip()
                        if conn["probe"]:
                            power_status = "unreachable"
                        if prep.get("anthropic_model"):
                            conn["anthropic_model"] = prep["anthropic_model"]
                        hs = conn["hold_sec"]
                        if hs > 0:
                            await client.aclose()
                            client = httpx.AsyncClient(
                                timeout=httpx.Timeout(hs, connect=10.0)
                            )

                    # Admission/preflight may block up to ~30s — ping while waiting
                    # or Claude Code idle-aborts at ~20s before we ever reach upstream.
                    if pre_upstream is not None:
                        gate_task = asyncio.create_task(pre_upstream())
                        async for ping in _ping_while(gate_task):
                            yield ping
                        gate_err = gate_task.result()
                        gate_task = None
                        if gate_err:
                            yield anthropic_sse_error(
                                str(gate_err), err_type="api_error"
                            )
                            for piece in xform.flush():
                                yield piece
                            return

                    if not (conn["url"] or "").strip():
                        yield anthropic_sse_error(
                            "unauthorized", err_type="authentication_error"
                        )
                        for piece in xform.flush():
                            yield piece
                        return

                    _sse_log.info(
                        "anthropic_sse upstream connect url=%s",
                        (conn["url"] or "")[:120],
                    )
                    send_task = asyncio.create_task(
                        client.send(
                            client.build_request(
                                method,
                                conn["url"],
                                headers=conn["headers"],
                                content=conn["body"],
                            ),
                            stream=True,
                        )
                    )
                    # Ping while waiting for upstream response headers (queue / load).
                    async for ping in _ping_while(send_task):
                        yield ping

                    try:
                        upstream = send_task.result()
                    except Exception as exc:
                        _sse_log.warning(
                            "anthropic_sse upstream_unreachable: %s", exc
                        )
                        yield anthropic_sse_error(
                            f"upstream_unreachable: {exc}", err_type="api_error"
                        )
                        for piece in xform.flush():
                            yield piece
                        return

                    upstream_status = upstream.status_code
                    content_type = upstream.headers.get("content-type")
                    _sse_log.info(
                        "anthropic_sse upstream status=%s ct=%s",
                        upstream_status,
                        (content_type or "")[:60],
                    )

                    if upstream_status >= 400:
                        err_raw = b""
                        with suppress(Exception):
                            err_raw = await upstream.aread()
                        _meter_chunk(err_raw)
                        msg = f"upstream_status_{upstream_status}"
                        with suppress(Exception):
                            import json as _json

                            obj = _json.loads(err_raw.decode("utf-8", errors="replace"))
                            if isinstance(obj, dict):
                                err = obj.get("error")
                                if isinstance(err, dict) and err.get("message"):
                                    msg = str(err["message"])
                                elif isinstance(err, str) and err.strip():
                                    msg = err.strip()
                                elif obj.get("detail"):
                                    msg = str(obj["detail"])
                        yield anthropic_sse_error(msg, err_type="api_error")
                        for piece in xform.flush():
                            yield piece
                        return

                    hs = int(conn["hold_sec"] or 0)
                    if hs > 0:

                        async def _kill_after_hold() -> None:
                            await asyncio.sleep(hs)
                            with suppress(Exception):
                                await upstream.aclose()

                        watchdog = asyncio.create_task(_kill_after_hold())

                    aiter = upstream.aiter_raw().__aiter__()
                    first_up = True
                    while True:
                        if pending is None:
                            pending = asyncio.ensure_future(aiter.__anext__())
                        done, _ = await asyncio.wait(
                            {pending},
                            timeout=ANTHROPIC_SSE_PING_INTERVAL_SEC,
                        )
                        if not done:
                            yield anthropic_sse_ping()
                            now = time_mod.perf_counter()
                            if conn["probe"] and (now - last_sample) >= 1.0:
                                await _sample()
                                last_sample = now
                            continue
                        try:
                            chunk = pending.result()
                        except StopAsyncIteration:
                            pending = None
                            break
                        pending = None
                        if first_up:
                            first_up = False
                            _sse_log.info(
                                "anthropic_sse upstream first_byte bytes=%s",
                                len(chunk),
                            )
                        now = time_mod.perf_counter()
                        if conn["probe"] and (now - last_sample) >= 1.0:
                            await _sample()
                            last_sample = now
                        _meter_chunk(chunk)
                        for piece in xform.feed(chunk):
                            yield piece
                    for piece in xform.flush():
                        yield piece
                    _sse_log.info(
                        "anthropic_sse stream done status=%s", upstream_status
                    )
                except (asyncio.CancelledError, GeneratorExit):
                    _sse_log.warning("anthropic_sse client_disconnect")
                    raise
                except Exception:
                    _sse_log.exception("anthropic_sse stream_error")
                    with suppress(Exception):
                        for piece in xform.flush():
                            yield piece
                    raise
                finally:
                    await _reap_side_task(prep_task)
                    await _reap_side_task(gate_task)
                    if send_task is not None:
                        if not send_task.done():
                            send_task.cancel()
                            with suppress(asyncio.CancelledError):
                                await send_task
                        elif upstream is None:
                            # Client disconnect during header wait: Response orphaned.
                            with suppress(Exception):
                                orphan = send_task.result()
                                await orphan.aclose()
                    if pending is not None and not pending.done():
                        pending.cancel()
                        with suppress(asyncio.CancelledError):
                            await pending
                    if watchdog is not None:
                        watchdog.cancel()
                        with suppress(asyncio.CancelledError):
                            await watchdog
                    if upstream is not None:
                        with suppress(Exception):
                            await upstream.aclose()
                    with suppress(Exception):
                        await client.aclose()
                    release_concurrency_lease(conn["concurrency_lease"])
                    await _sample()
                    _finalize(
                        buf=bytes(buf),
                        upstream_status=upstream_status,
                        content_type=content_type,
                    )

            return StreamingResponse(
                _stream_early(),
                status_code=200,
                headers=sse_headers,
                media_type="text/event-stream",
            )

        client = httpx.AsyncClient(timeout=client_timeout)
        try:
            upstream = await client.send(
                client.build_request(method, url, headers=headers, content=body),
                stream=True,
            )
        except Exception as exc:
            await client.aclose()
            release_concurrency_lease(concurrency_lease)
            duration_ms = (time_mod.perf_counter() - t0) * 1000.0
            if usage_id is not None and SessionLocal is not None:
                avg_w, wh = watt_hours_from_samples(
                    samples=samples, duration_sec=duration_ms / 1000.0
                )
                db = SessionLocal()
                try:
                    finalize_usage_metering(
                        db,
                        usage_id,
                        duration_ms=duration_ms,
                        watts=avg_w,
                        watt_hours=wh,
                        upstream_status=502,
                        power_status=power_status if not samples else "metered",
                    )
                finally:
                    db.close()
            err_body: dict = {
                "error": "upstream_unreachable",
                "detail": str(exc),
                "retryable": False,
            }
            if anthropic_client:
                err_body = anthropic_error_body(
                    f"upstream_unreachable: {exc}", err_type="api_error"
                )
            return JSONResponse(err_body, status_code=502)

        out_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower()
            not in {"transfer-encoding", "connection", "content-length", "content-encoding"}
        }
        upstream_status = upstream.status_code
        content_type = upstream.headers.get("content-type")
        media_type = content_type
        if anthropic_client:
            ct_l = (content_type or "").lower()
            use_sse = anthropic_stream or ("text/event-stream" in ct_l)
            # Errors stay JSON even when the client asked for a stream.
            if use_sse and upstream_status < 400:
                media_type = "text/event-stream"
                out_headers["content-type"] = "text/event-stream"
                out_headers["cache-control"] = "no-cache"
                out_headers["x-accel-buffering"] = "no"
            else:
                media_type = "application/json"
                out_headers["content-type"] = "application/json"

        async def _stream():
            last_sample = time_mod.perf_counter()
            buf = bytearray()
            truncated = False
            watchdog: asyncio.Task | None = None
            xform = (
                OpenAIToAnthropicStream(model=anthropic_model)
                if anthropic_client
                else None
            )
            use_pings = bool(
                xform is not None
                and media_type
                and "event-stream" in str(media_type).lower()
            )

            async def _kill_after_hold() -> None:
                await asyncio.sleep(hold_sec)
                with suppress(Exception):
                    await upstream.aclose()

            def _meter_chunk(chunk: bytes) -> None:
                nonlocal truncated
                if not truncated:
                    if len(buf) + len(chunk) <= _BUF_MAX:
                        buf.extend(chunk)
                    else:
                        buf.extend(chunk)
                        if len(buf) > _BUF_TAIL:
                            del buf[: len(buf) - _BUF_TAIL]
                            truncated = True
                else:
                    buf.extend(chunk)
                    if len(buf) > _BUF_TAIL:
                        del buf[: len(buf) - _BUF_TAIL]

            if hold_sec > 0:
                watchdog = asyncio.create_task(_kill_after_hold())
            pending: asyncio.Task | None = None
            try:
                if use_pings and xform is not None:
                    for piece in xform.bootstrap():
                        yield piece
                    yield anthropic_sse_ping()
                aiter = upstream.aiter_raw().__aiter__()
                while True:
                    if pending is None:
                        pending = asyncio.ensure_future(aiter.__anext__())
                    wait_timeout = (
                        ANTHROPIC_SSE_PING_INTERVAL_SEC if use_pings else None
                    )
                    done, _ = await asyncio.wait({pending}, timeout=wait_timeout)
                    if not done:
                        yield anthropic_sse_ping()
                        now = time_mod.perf_counter()
                        if probe and (now - last_sample) >= 1.0:
                            await _sample()
                            last_sample = now
                        continue
                    try:
                        chunk = pending.result()
                    except StopAsyncIteration:
                        pending = None
                        break
                    pending = None
                    now = time_mod.perf_counter()
                    if probe and (now - last_sample) >= 1.0:
                        await _sample()
                        last_sample = now
                    _meter_chunk(chunk)
                    if xform is None:
                        yield chunk
                    else:
                        for piece in xform.feed(chunk):
                            yield piece
                if xform is not None:
                    for piece in xform.flush():
                        yield piece
            except Exception:
                if xform is not None:
                    with suppress(Exception):
                        for piece in xform.flush():
                            yield piece
                raise
            finally:
                if pending is not None and not pending.done():
                    pending.cancel()
                    with suppress(asyncio.CancelledError):
                        await pending
                if watchdog is not None:
                    watchdog.cancel()
                    with suppress(asyncio.CancelledError):
                        await watchdog
                await upstream.aclose()
                await client.aclose()
                release_concurrency_lease(concurrency_lease)
                await _sample()
                _finalize(
                    buf=bytes(buf),
                    upstream_status=upstream_status,
                    content_type=content_type,
                )

        return StreamingResponse(
            _stream(),
            status_code=upstream.status_code,
            headers=out_headers,
            media_type=media_type,
        )

    @app.api_route("/v1/onprem/entry", methods=["POST", "PUT", "PATCH"])
    async def onprem_entry(request: Request, db: Session = Depends(get_db)):
        """Auth + upstream proxy in one hop (avoids nginx auth_request body hang).

        Used for chat (and optionally embed) so duration/Wh metering works.
        Anthropic clients hit nginx ``/v1/messages`` → this entry.
        """
        settings = request.app.state.settings
        host = request.headers.get("x-original-host") or request.headers.get("host") or ""
        uri = request.headers.get("x-original-uri") or request.url.path
        method = request.headers.get("x-original-method") or request.method
        client_ip = request.headers.get("x-real-ip") or (
            request.client.host if request.client else ""
        )
        raw_key = request.headers.get("x-api-key")
        if not raw_key:
            auth_h = request.headers.get("authorization") or ""
            if auth_h.lower().startswith("bearer "):
                raw_key = auth_h[7:].strip()
        content_type = request.headers.get("content-type")
        body = await request.body()

        # Anthropic Messages client dialect: translate before aliases/VL/auth.
        # nginx rewrites to /v1/onprem/entry — prefer X-Original-URI; also accept
        # anthropic-version (Claude Code always sends it) so we never miss SSE pings.
        anthropic_client = is_anthropic_messages_path(uri) or (
            bool(request.headers.get("anthropic-version"))
            and (
                is_anthropic_messages_path(request.url.path)
                or str(request.url.path).rstrip("/").endswith("/entry")
            )
        )
        anthropic_model: str | None = None
        anthropic_stream = False
        if anthropic_client:
            if not body:
                return JSONResponse(
                    anthropic_error_body(
                        "request body is required",
                        err_type="invalid_request_error",
                    ),
                    status_code=400,
                )
            try:
                body, anthropic_stream = anthropic_request_to_openai(body)
                content_type = "application/json"
                anthropic_model = extract_model(body, content_type)
            except AnthropicTranslateError as exc:
                return JSONResponse(
                    anthropic_error_body(str(exc), err_type=exc.err_type),
                    status_code=400,
                )

        def _entry_err(payload: dict, status: int) -> JSONResponse:
            if anthropic_client:
                msg = str(payload.get("error") or payload.get("detail") or "error")
                return JSONResponse(
                    anthropic_error_body(msg, err_type="api_error"),
                    status_code=status,
                )
            return JSONResponse(payload, status_code=status)

        def _maybe_anthropic_json(resp: JSONResponse | None) -> JSONResponse | None:
            if resp is None or not anthropic_client:
                return resp
            try:
                data = json.loads(resp.body.decode("utf-8") if resp.body else b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                data = {}
            if not isinstance(data, dict):
                data = {}
            msg = str(data.get("error") or data.get("detail") or "error")
            return JSONResponse(
                anthropic_error_body(msg, err_type="api_error"),
                status_code=resp.status_code,
            )

        hop = {
            "content-type",
            "content-length",
            "host",
            "connection",
            "transfer-encoding",
            "x-onprem-ticket",
            "x-onprem-proxy",
            # Anthropic client headers — not for OpenAI/llama.cpp upstreams
            "anthropic-version",
            "anthropic-beta",
            "anthropic-dangerous-direct-browser-access",
        }
        req_headers = {k: v for k, v in request.headers.items()}
        req_query = str(request.url.query or "")

        # Anthropic SSE: return event-stream NOW (first byte), but NEVER skip
        # authorize — routing/auth/admission run inside the pinging generator.
        # Failed auth → SSE error event; upstream opens only after authorize 204.
        if anthropic_client and anthropic_stream:
            openai_body = body
            ct = content_type or "application/json"
            early_model = anthropic_model

            def _prepare_auth_and_route_sync() -> dict | str:
                """Sync authorize+route+gates — runs in a worker thread.

                Must not touch the event loop: Claude Code idle-aborts if SSE
                pings stop while sync DB/auth/admission blocks the loop.
                """
                import logging

                from .data.db import SessionLocal as _SL
                from .usage_pool import probe_url_for_source

                log = logging.getLogger("uvicorn.error")
                if _SL is None:
                    return "db_unavailable"
                db2 = _SL()
                lease_hold: ConcurrencyLease | None = None
                try:
                    named, upstream_uri = split_source_path(uri)
                    auth_cfg = get_auth_settings(db2)
                    auto_rewrite: str | None = None
                    alias_pref: str | None = None
                    work_body = openai_body
                    if named is not None:
                        src = get_source_by_name(db2, named)
                        if src is None:
                            return f"unknown_source: {named}"
                        service = src.name
                        kind = src.kind
                        upstream_uri = upstream_uri or "/"
                        if kind == "chat" and work_body:
                            asked = extract_model(work_body, ct)
                            work_body, auto_rewrite, _pref = apply_client_model_rewrites(
                                db2, work_body, asked=asked, auth=auth_cfg
                            )
                    else:
                        upstream_uri = upstream_path_for_proxy(uri)
                        kind = kind_from_upstream_path(upstream_uri)
                        if kind is None:
                            return f"unknown_route: {uri}"
                        if kind == "chat" and work_body:
                            asked = extract_model(work_body, ct)
                            work_body, auto_rewrite, alias_pref = (
                                apply_client_model_rewrites(
                                    db2, work_body, asked=asked, auth=auth_cfg
                                )
                            )
                        model = extract_request_model(kind, work_body, ct)
                        src = resolve_routed_source(
                            db2,
                            kind,
                            model=model,
                            raw_key=raw_key,
                            auth=auth_cfg,
                            preferred_source=alias_pref,
                        )
                        if src is None:
                            if not (model or "").strip():
                                return "model is required"
                            return f"unknown_model: {model}"
                        service = src.name

                    vl_rewrite: str | None = None
                    if (
                        kind == "chat"
                        and work_body
                        and auth_cfg.auto_vl_routing
                        and request_needs_vision(work_body, ct)
                    ):
                        asked = extract_model(work_body, ct)
                        sibling = resolve_vl_model(db2, service, asked)
                        if sibling and sibling != asked:
                            work_body = rewrite_json_model(work_body, sibling)
                            vl_rewrite = sibling

                    # AUTHORIZE — never skipped. No upstream without 204.
                    result = authorize(
                        db2,
                        raw_key=raw_key,
                        service=service,
                        kind=kind,
                        method=method,
                        path=upstream_uri,
                        host=host,
                        client_ip=client_ip,
                        body=work_body,
                        content_type=ct,
                        defer_metering=True,
                    )
                    lease_hold = result.concurrency_lease
                    if result.status != 204:
                        release_concurrency_lease(result.concurrency_lease)
                        lease_hold = None
                        log.info(
                            "anthropic_sse prepare auth_fail reason=%s",
                            (result.reason or "unauthorized")[:80],
                        )
                        return str(result.reason or "unauthorized")

                    backend = address_for_source(db2, service)
                    if not backend:
                        release_concurrency_lease(result.concurrency_lease)
                        lease_hold = None
                        return "backend_not_configured"

                    src_row = get_source_by_name(db2, service)
                    entry_lease = _attach_source_key(result.concurrency_lease, backend)
                    lease_hold = entry_lease

                    # Sync gates (may wait up to source_queue_timeout) — OK in thread.
                    pf_block = _preflight_block(
                        auth_cfg=auth_cfg,
                        backend=backend,
                        kind=kind,
                        model=result.model,
                        service=service,
                        lease=result.concurrency_lease,
                        src=src_row,
                    )
                    if pf_block is not None:
                        try:
                            data = json.loads(
                                pf_block.body.decode("utf-8")
                                if pf_block.body
                                else b"{}"
                            )
                        except (
                            json.JSONDecodeError,
                            UnicodeDecodeError,
                            AttributeError,
                        ):
                            data = {}
                        if not isinstance(data, dict):
                            data = {}
                        lease_hold = None
                        return str(
                            data.get("error")
                            or data.get("detail")
                            or "preflight_failed"
                        )

                    adm_block = _source_admission_block(
                        auth_cfg=auth_cfg,
                        src=src_row,
                        backend=backend,
                        kind=kind,
                        model=result.model,
                        service=service,
                        lease=result.concurrency_lease,
                        priority=result.priority,
                    )
                    if adm_block is not None:
                        try:
                            data = json.loads(
                                adm_block.body.decode("utf-8")
                                if adm_block.body
                                else b"{}"
                            )
                        except (
                            json.JSONDecodeError,
                            UnicodeDecodeError,
                            AttributeError,
                        ):
                            data = {}
                        if not isinstance(data, dict):
                            data = {}
                        lease_hold = None
                        return str(
                            data.get("error")
                            or data.get("detail")
                            or "admission_failed"
                        )

                    api_style = (
                        getattr(src_row, "api_style", None)
                        if src_row is not None
                        else None
                    )
                    rewrite_uri = map_upstream_path(
                        upstream_uri, kind=kind, api_style=api_style
                    )
                    final_model = vl_rewrite or auto_rewrite
                    if final_model:
                        work_body = rewrite_json_model(work_body, final_model)
                    if (
                        kind == "chat"
                        and work_body
                        and getattr(auth_cfg, "soft_sampling_defaults", False)
                    ):
                        from .sampling_merge import soft_sampling_body

                        mid = (
                            final_model
                            or result.model
                            or extract_model(work_body, ct)
                        )
                        key_prof = ""
                        if result.api_key is not None:
                            key_prof = (
                                getattr(result.api_key, "sampling_profile", "") or ""
                            )
                        filled = soft_sampling_body(
                            db2,
                            work_body,
                            ct,
                            model_id=mid,
                            headers=req_headers,
                            enabled=True,
                            key_profile=key_prof,
                        )
                        if filled is not None:
                            work_body = filled

                    up_url = f"http://{backend}{rewrite_uri}"
                    up_headers = {
                        k: v
                        for k, v in req_headers.items()
                        if k.lower() not in hop
                    }
                    up_headers["content-type"] = ct
                    up_headers["content-length"] = str(len(work_body))
                    lease_hold = None  # ownership transfers to stream finally
                    out_model = early_model or final_model or result.model
                    log.info(
                        "anthropic_sse prepare ok service=%s model=%s",
                        service,
                        (out_model or "")[:80],
                    )
                    return {
                        "url": up_url,
                        "headers": up_headers,
                        "body": work_body,
                        "usage_id": result.usage_event_id,
                        "probe_url": probe_url_for_source(src_row),
                        "concurrency_lease": entry_lease,
                        "max_hold_sec": int(
                            getattr(auth_cfg, "source_admission_max_hold_sec", 0) or 0
                        ),
                        "anthropic_model": out_model,
                    }
                except Exception:
                    import logging as _logging

                    _logging.getLogger("uvicorn.error").exception(
                        "anthropic_sse prepare auth_setup_failed"
                    )
                    return "auth_setup_failed"
                finally:
                    release_concurrency_lease(lease_hold)
                    db2.close()

            async def _prepare_auth_and_route() -> dict | str:
                # Off the event loop so SSE pings keep flushing during auth/queue.
                return await asyncio.to_thread(_prepare_auth_and_route_sync)

            return await _stream_upstream_metered(
                method=method,
                url="",
                headers={},
                body=openai_body,
                usage_id=None,
                anthropic_client=True,
                anthropic_model=early_model,
                anthropic_stream=True,
                prepare=_prepare_auth_and_route,
            )

        named, upstream_uri = split_source_path(uri)
        auth_cfg = get_auth_settings(db)
        auto_rewrite: str | None = None
        alias_pref: str | None = None
        if named is not None:
            src = get_source_by_name(db, named)
            if src is None:
                return _entry_err(
                    {"error": "unknown_source", "source": named},
                    403,
                )
            service = src.name
            kind = src.kind
            upstream_uri = upstream_uri or "/"
            if kind == "chat" and body:
                asked = extract_model(body, content_type)
                body, auto_rewrite, _pref = apply_client_model_rewrites(
                    db, body, asked=asked, auth=auth_cfg
                )
        else:
            upstream_uri = upstream_path_for_proxy(uri)
            kind = kind_from_upstream_path(upstream_uri)
            if kind is None:
                return _entry_err({"error": "unknown_route", "path": uri}, 403)
            if kind == "chat" and body:
                asked = extract_model(body, content_type)
                body, auto_rewrite, alias_pref = apply_client_model_rewrites(
                    db, body, asked=asked, auth=auth_cfg
                )
            model = extract_request_model(kind, body, content_type)
            src = resolve_routed_source(
                db,
                kind,
                model=model,
                raw_key=raw_key,
                auth=auth_cfg,
                preferred_source=alias_pref,
            )
            if src is None:
                if anthropic_client:
                    if not (model or "").strip():
                        return JSONResponse(
                            anthropic_error_body(
                                "model is required", err_type="invalid_request_error"
                            ),
                            status_code=400,
                        )
                    return JSONResponse(
                        anthropic_error_body(
                            f"unknown_model: {model}", err_type="not_found_error"
                        ),
                        status_code=404,
                    )
                return _unresolved_model_response(kind, model)
            service = src.name

        vl_rewrite: str | None = None
        if (
            kind == "chat"
            and body
            and auth_cfg.auto_vl_routing
            and request_needs_vision(body, content_type)
        ):
            asked = extract_model(body, content_type)
            sibling = resolve_vl_model(db, service, asked)
            if sibling and sibling != asked:
                body = rewrite_json_model(body, sibling)
                vl_rewrite = sibling

        result = authorize(
            db,
            raw_key=raw_key,
            service=service,
            kind=kind,
            method=method,
            path=upstream_uri,
            host=host,
            client_ip=client_ip,
            body=body,
            content_type=content_type,
            defer_metering=True,
        )
        if result.status != 204:
            release_concurrency_lease(result.concurrency_lease)
            return _entry_err(
                {"error": result.reason, "service": service},
                result.status,
            )

        backend = address_for_source(db, service)
        if not backend:
            release_concurrency_lease(result.concurrency_lease)
            return _entry_err({"error": "backend_not_configured"}, 503)

        src_row = get_source_by_name(db, service)
        entry_lease = _attach_source_key(result.concurrency_lease, backend)
        pf_block = await _preflight_block_async(
            auth_cfg=auth_cfg,
            backend=backend,
            kind=kind,
            model=result.model,
            service=service,
            lease=result.concurrency_lease,
            src=src_row,
        )
        pf_block = _maybe_anthropic_json(pf_block)
        if pf_block is not None:
            return pf_block
        adm_block = await _source_admission_block_async(
            auth_cfg=auth_cfg,
            src=src_row,
            backend=backend,
            kind=kind,
            model=result.model,
            service=service,
            lease=result.concurrency_lease,
            priority=result.priority,
        )
        adm_block = _maybe_anthropic_json(adm_block)
        if adm_block is not None:
            return adm_block
        api_style = getattr(src_row, "api_style", None) if src_row is not None else None
        rewrite_uri = map_upstream_path(upstream_uri, kind=kind, api_style=api_style)
        final_model = vl_rewrite or auto_rewrite
        if final_model:
            body = rewrite_json_model(body, final_model)
        if kind == "chat" and body and getattr(auth_cfg, "soft_sampling_defaults", False):
            from .sampling_merge import soft_sampling_body

            mid = final_model or result.model or extract_model(body, content_type)
            key_prof = ""
            if result.api_key is not None:
                key_prof = getattr(result.api_key, "sampling_profile", "") or ""
            filled = soft_sampling_body(
                db,
                body,
                content_type,
                model_id=mid,
                headers=req_headers,
                enabled=True,
                key_profile=key_prof,
            )
            if filled is not None:
                body = filled

        url = f"http://{backend}{rewrite_uri}"
        # Claude Code sends ?beta=true on /v1/messages — do not forward to OpenAI/llama.cpp.
        if req_query and not anthropic_client:
            url = f"{url}?{req_query}"
        up_headers = {
            k: v for k, v in req_headers.items() if k.lower() not in hop
        }
        up_headers["content-type"] = content_type or "application/json"
        up_headers["content-length"] = str(len(body))

        from .usage_pool import probe_url_for_source

        return await _stream_upstream_metered(
            method=method,
            url=url,
            headers=up_headers,
            body=body,
            usage_id=result.usage_event_id,
            probe_url=probe_url_for_source(src_row),
            concurrency_lease=entry_lease,
            max_hold_sec=int(getattr(auth_cfg, "source_admission_max_hold_sec", 0) or 0),
            anthropic_client=anthropic_client,
            anthropic_model=anthropic_model
            or final_model
            or result.model,
            anthropic_stream=anthropic_stream,
        )

    @app.api_route("/v1/onprem/forward", methods=["POST", "PUT", "PATCH"])
    async def onprem_forward(request: Request):
        """Legacy VL hop after auth_request — ticket required."""
        settings = request.app.state.settings
        ticket = request.headers.get("x-onprem-ticket") or ""
        payload = parse_forward_ticket(ticket, settings.session_secret)
        if payload is None:
            return JSONResponse({"error": "invalid_onprem_ticket"}, status_code=403)

        backend = str(payload.get("backend") or "").strip()
        rewrite_uri = str(payload.get("uri") or "/").strip() or "/"
        rewrite_model = str(payload.get("model") or "").strip()
        usage_id = payload.get("uid")
        lease = _lease_from_ticket(payload)
        try:
            usage_id_int = int(usage_id) if usage_id is not None else None
        except (TypeError, ValueError):
            usage_id_int = None
        if not backend:
            release_concurrency_lease(lease)
            return JSONResponse({"error": "invalid_onprem_ticket"}, status_code=403)

        body = await request.body()
        if rewrite_model:
            body = rewrite_json_model(body, rewrite_model)
        url = f"http://{backend}{rewrite_uri}"
        if request.url.query:
            url = f"{url}?{request.url.query}"

        hop = {
            "content-type",
            "content-length",
            "host",
            "connection",
            "transfer-encoding",
            "x-onprem-ticket",
            "x-onprem-proxy",
        }
        up_headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in hop
        }
        content_type = request.headers.get("content-type") or "application/json"
        up_headers["content-type"] = content_type
        up_headers["content-length"] = str(len(body))

        from .data.backends import get_source_by_name
        from .data.db import SessionLocal
        from .usage_pool import probe_url_for_source

        probe = ""
        kind = "chat"
        auth_cfg = None
        src = None
        service_name = str(payload.get("service") or "")
        if SessionLocal is not None:
            pdb = SessionLocal()
            try:
                src = get_source_by_name(pdb, service_name)
                probe = probe_url_for_source(src)
                if src is not None:
                    kind = src.kind or "chat"
                auth_cfg = get_auth_settings(pdb)
                if (
                    kind == "chat"
                    and body
                    and auth_cfg is not None
                    and getattr(auth_cfg, "soft_sampling_defaults", False)
                ):
                    from .data.models import ApiKey
                    from .sampling_merge import soft_sampling_body

                    mid = rewrite_model or str(payload.get("mdl") or "") or None
                    if not mid:
                        mid = extract_model(body, content_type)
                    key_prof = ""
                    kid = getattr(lease, "key_id", None) if lease is not None else None
                    if kid:
                        krow = pdb.get(ApiKey, int(kid))
                        if krow is not None:
                            key_prof = getattr(krow, "sampling_profile", "") or ""
                    filled = soft_sampling_body(
                        pdb,
                        body,
                        content_type,
                        model_id=mid,
                        headers=request.headers,
                        enabled=True,
                        key_profile=key_prof,
                    )
                    if filled is not None:
                        body = filled
                        up_headers["content-length"] = str(len(body))
            finally:
                pdb.close()

        if lease is None or not lease.source_key:
            pf_block = await _preflight_block_async(
                auth_cfg=auth_cfg,
                backend=backend,
                kind=kind,
                model=rewrite_model or str(payload.get("mdl") or "") or None,
                service=service_name,
                lease=lease,
                src=src,
            )
            if pf_block is not None:
                return pf_block

        return await _stream_upstream_metered(
            method=request.method,
            url=url,
            headers=up_headers,
            body=body,
            usage_id=usage_id_int,
            probe_url=probe,
            concurrency_lease=lease,
            max_hold_sec=int(
                getattr(auth_cfg, "source_admission_max_hold_sec", 0) or 0
            )
            if auth_cfg is not None
            else 0,
        )

    app.include_router(web_router)
    app.include_router(ops_router)
    app.include_router(accounts_router)
    return app


app = create_app()
