from __future__ import annotations

import asyncio
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
    return JSONResponse(
        {
            "error": pf.reason,
            "service": service,
            "engine": pf.engine,
            "detail": pf.detail,
            "retry_after": pf.retry_after,
        },
        status_code=503,
        headers={"Retry-After": str(pf.retry_after)},
    )


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

        start_catalog_auto_sync()

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

        # Slot fields from load-cache / admission only — no live probes on list.
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
    ):
        """Shared streaming proxy + per-source GPU sample → finalize UsageEvent."""
        import asyncio
        import time as time_mod

        from .audit import finalize_usage_metering
        from .data.db import SessionLocal
        from .metering_parse import parse_upstream_metrics
        from .usage_pool import fetch_gpu_watts, watt_hours_from_samples

        # Cap buffered body for usage/timings parse (keep full for small JSON; tail for huge SSE).
        _BUF_MAX = 2 * 1024 * 1024
        _BUF_TAIL = 256 * 1024

        samples: list[float] = []
        probe = (probe_url or "").strip()
        if probe:
            power_status = "unreachable"  # until we get a sample
        else:
            power_status = "no_probe"

        async def _sample() -> None:
            nonlocal power_status
            if not probe:
                return
            w = await asyncio.to_thread(fetch_gpu_watts, probe)
            if w is not None and w > 0:
                samples.append(float(w))
                power_status = "metered"

        await _sample()
        t0 = time_mod.perf_counter()
        client = httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=10.0))
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
            return JSONResponse(
                {"error": "upstream_unreachable", "detail": str(exc)},
                status_code=502,
            )

        out_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower()
            not in {"transfer-encoding", "connection", "content-length", "content-encoding"}
        }
        upstream_status = upstream.status_code
        content_type = upstream.headers.get("content-type")

        async def _stream():
            last_sample = time_mod.perf_counter()
            buf = bytearray()
            truncated = False
            try:
                async for chunk in upstream.aiter_raw():
                    now = time_mod.perf_counter()
                    if probe and (now - last_sample) >= 1.0:
                        await _sample()
                        last_sample = now
                    if not truncated:
                        if len(buf) + len(chunk) <= _BUF_MAX:
                            buf.extend(chunk)
                        else:
                            # Keep a tail window for SSE usage/timings.
                            buf.extend(chunk)
                            if len(buf) > _BUF_TAIL:
                                del buf[: len(buf) - _BUF_TAIL]
                                truncated = True
                    else:
                        buf.extend(chunk)
                        if len(buf) > _BUF_TAIL:
                            del buf[: len(buf) - _BUF_TAIL]
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()
                release_concurrency_lease(concurrency_lease)
                await _sample()
                duration_ms = (time_mod.perf_counter() - t0) * 1000.0
                if usage_id is not None and SessionLocal is not None:
                    avg_w, wh = watt_hours_from_samples(
                        samples=samples,
                        duration_sec=max(0.001, duration_ms / 1000.0),
                    )
                    status = "metered" if samples else power_status
                    metrics = parse_upstream_metrics(
                        bytes(buf),
                        duration_ms=duration_ms,
                        content_type=content_type,
                    )
                    db = SessionLocal()
                    try:
                        finalize_usage_metering(
                            db,
                            usage_id,
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

        return StreamingResponse(
            _stream(),
            status_code=upstream.status_code,
            headers=out_headers,
            media_type=upstream.headers.get("content-type"),
        )

    @app.api_route("/v1/onprem/entry", methods=["POST", "PUT", "PATCH"])
    async def onprem_entry(request: Request, db: Session = Depends(get_db)):
        """Auth + upstream proxy in one hop (avoids nginx auth_request body hang).

        Used for chat (and optionally embed) so duration/Wh metering works.
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

        named, upstream_uri = split_source_path(uri)
        auth_cfg = get_auth_settings(db)
        auto_rewrite: str | None = None
        alias_pref: str | None = None
        if named is not None:
            src = get_source_by_name(db, named)
            if src is None:
                return JSONResponse(
                    {"error": "unknown_source", "source": named},
                    status_code=403,
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
                return JSONResponse({"error": "unknown_route", "path": uri}, status_code=403)
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
            return JSONResponse(
                {"error": result.reason, "service": service},
                status_code=result.status,
            )

        backend = address_for_source(db, service)
        if not backend:
            release_concurrency_lease(result.concurrency_lease)
            return JSONResponse({"error": "backend_not_configured"}, status_code=503)

        src_row = get_source_by_name(db, service)
        pf_block = await _preflight_block_async(
            auth_cfg=auth_cfg,
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
            return adm_block
        entry_lease = _attach_source_key(result.concurrency_lease, backend)
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
                headers=request.headers,
                enabled=True,
                key_profile=key_prof,
            )
            if filled is not None:
                body = filled

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
            k: v for k, v in request.headers.items() if k.lower() not in hop
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
        )

    app.include_router(web_router)
    app.include_router(ops_router)
    app.include_router(accounts_router)
    return app


app = create_app()
