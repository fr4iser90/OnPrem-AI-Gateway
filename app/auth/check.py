from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import httpx
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from ..audit import bump_usage_daily, check_quota_alert, maybe_alert
from ..config import MODEL_CHECK_KINDS, MODEL_REQUIRED_KINDS, get_settings
from ..data.db import hash_api_key
from ..data.models import WebUser, ApiKey, ModelLimit, Team, UsageDaily, UsageEvent, utcnow
from .concurrency import ConcurrencyLease, release_concurrency_lease
from .priority import priority_gate
from .rate_limit import rate_limiter

_STREAM_KINDS = frozenset({"chat", "embed", "stt", "tts"})


@dataclass
class AuthResult:
    status: int
    reason: str = "ok"
    remaining_rpm: int | None = None
    priority: int = 0
    api_key: ApiKey | None = None
    model: str | None = None
    usage_event_id: int | None = None
    # Chat proxied via /v1/onprem/forward for real duration / Wh metering
    meter_proxy: bool = False
    concurrency_lease: ConcurrencyLease | None = None


def check_temperature(db: Session, service: str) -> AuthResult | None:
    settings = get_settings()
    if settings.temp_guard_disabled:
        return None
    from ..data.backends import get_source_by_name
    from ..usage_pool import temp_guard_url_for_source

    src = get_source_by_name(db, service)
    guard_url = temp_guard_url_for_source(src)
    if not guard_url:
        return None
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(guard_url)
    except Exception as exc:
        if settings.temp_guard_fail_open:
            return None
        return AuthResult(status=503, reason=f"temp_guard_unreachable:{exc}")

    if resp.status_code == 204:
        return None
    if resp.status_code == 403:
        return AuthResult(status=503, reason="local_temperature_above_limit")
    if settings.temp_guard_fail_open:
        return None
    return AuthResult(status=503, reason=f"temp_guard_status_{resp.status_code}")


_JSON_STRING_FIELD_SCAN = 32768
_JSON_FULL_PARSE_LIMIT = 2_000_000


def _decode_json_string_literal(raw: str) -> str:
    try:
        decoded = json.loads(f'"{raw}"')
    except Exception:
        decoded = raw
    return decoded if isinstance(decoded, str) else raw


def _extract_json_string_field(body: bytes, field: str) -> str | None:
    """Read one top-level string field without parsing megabyte-scale JSON bodies."""
    if not body or body[:1] not in (b"{",):
        return None
    if len(body) <= _JSON_FULL_PARSE_LIMIT:
        try:
            payload = json.loads(body.decode("utf-8", errors="ignore"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            val = payload.get(field)
            if isinstance(val, str):
                s = val.strip()
                return s or None
    key = field.encode("ascii")
    m = re.search(
        rb'"' + key + rb'"\s*:\s*"((?:[^"\\]|\\.)*)"',
        body[:_JSON_STRING_FIELD_SCAN],
    )
    if not m:
        return None
    s = _decode_json_string_literal(m.group(1).decode("utf-8", errors="replace")).strip()
    return s or None


def extract_model(body: bytes | None, content_type: str | None) -> str | None:
    if not body:
        return None
    ct = (content_type or "").lower()
    if "multipart/form-data" in ct:
        # Whisper-style: form field "model"
        m = re.search(
            rb'Content-Disposition:[^\n]*name="model"[^\n]*\r?\n\r?\n([^\r\n]+)',
            body[:65536],
            re.IGNORECASE,
        )
        if m:
            return m.group(1).decode("utf-8", errors="ignore").strip() or None
        return None
    if "application/json" not in ct and body[:1] not in (b"{", b"["):
        return None
    return _extract_json_string_field(body, "model")


def extract_voice(body: bytes | None, content_type: str | None) -> str | None:
    """Piper TTS often sends voice, not model."""
    if not body:
        return None
    ct = (content_type or "").lower()
    if "application/json" not in ct and body[:1] not in (b"{", b"["):
        return None
    return _extract_json_string_field(body, "voice")


def extract_request_model(
    kind: str,
    body: bytes | None,
    content_type: str | None,
) -> str | None:
    """Model id for routing + allowlist (TTS falls back to voice)."""
    model = extract_model(body, content_type)
    if kind == "tts" and not model:
        model = extract_voice(body, content_type)
    return model


def _services_for_key(api_key: ApiKey, db: Session | None = None) -> set[str]:
    """Effective sources for a key (grant ceiling ∩ key subset / inherit)."""
    if db is None:
        # Legacy fallback without ceiling (tests that don't pass db)
        key_services = {g.service for g in api_key.service_grants}
        if key_services:
            return key_services
        if api_key.team:
            return {g.service for g in api_key.team.service_grants}
        return set()
    from ..data.grants import effective_services

    return effective_services(db, api_key)


def _models_for_key(
    api_key: ApiKey, service: str, db: Session | None = None
) -> set[str] | None:
    if db is None:
        key_models = [m.model_name for m in api_key.model_allowlists if m.service == service]
        if key_models:
            return set(key_models)
        if api_key.team:
            team_models = [
                m.model_name for m in api_key.team.model_allowlists if m.service == service
            ]
            if team_models:
                return set(team_models)
        return None
    from ..data.grants import effective_models

    return effective_models(db, api_key, service)


def _pick_model_limit(api_key: ApiKey, service: str, model: str | None) -> ModelLimit | None:
    if not model:
        return None
    for lim in api_key.model_limits:
        if lim.service == service and lim.model_name == model:
            return lim
    if api_key.team:
        for lim in api_key.team.model_limits:
            if lim.service == service and lim.model_name == model:
                return lim
    return None


def _effective_limits(api_key: ApiKey) -> tuple[int | None, int | None, int, int | None, int | None]:
    rpm = api_key.rpm_limit
    concurrency = api_key.concurrency_limit
    priority = api_key.priority if api_key.priority is not None else 0
    daily = api_key.daily_quota
    monthly = None
    if api_key.team:
        if rpm is None:
            rpm = api_key.team.rpm_limit
        if concurrency is None:
            concurrency = api_key.team.concurrency_limit
        if api_key.priority is None:
            priority = api_key.team.priority or 0
        if daily is None:
            daily = api_key.team.daily_quota
        monthly = api_key.team.monthly_quota
    return rpm, concurrency, priority, daily, monthly


def _owner_ceilings(api_key: ApiKey) -> tuple[int | None, int | None, int | None, int | None]:
    """User-level caps across all keys. Platform admins: no user ceiling."""
    owner = api_key.owner
    if owner is None or owner.is_platform_admin:
        return None, None, None, None
    return (
        owner.id,
        owner.rpm_limit,
        owner.concurrency_limit,
        owner.daily_quota,
    )


def _monthly_ok_count(db: Session, team_id: int | None) -> int:
    if not team_id:
        return 0
    today = utcnow().date()
    month_start = today.replace(day=1)
    total = (
        db.query(func.coalesce(func.sum(UsageDaily.ok_count), 0))
        .filter(UsageDaily.team_id == team_id, UsageDaily.day >= month_start)
        .scalar()
    )
    return int(total or 0)


def log_usage(
    db: Session,
    *,
    api_key: ApiKey | None,
    service: str,
    method: str,
    path: str,
    host: str,
    client_ip: str,
    model: str | None,
    status: int,
    result: str,
    duration_ms: float | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    audio_seconds: float | None = None,
    response_chars: int | None = None,
    body: bytes | None = None,
    kind: str | None = None,
    watts: float | None = None,
    watt_hours: float | None = None,
    pool_cost: float | None = None,
    defer_metering: bool = False,
    power_status: str = "",
) -> UsageEvent:
    from ..web.accounts import get_auth_settings
    from ..privacy import anonymize_ip, estimate_prompt_tokens

    auth = get_auth_settings(db)
    ip = client_ip or ""
    if auth.anonymize_client_ip:
        ip = anonymize_ip(ip)

    if tokens_in is None and result == "ok" and kind in MODEL_CHECK_KINDS:
        tokens_in = estimate_prompt_tokens(body)

    # Deferred: entry/forward patches real duration + Wh after upstream ends.
    # Auth-time global GPU sample removed — Wh is per-source probe only.
    if defer_metering:
        watts = None
        watt_hours = None
        duration_ms = None
        if not power_status:
            power_status = ""
    else:
        watts = None
        watt_hours = None

    team_id = api_key.team_id if api_key else None
    key_id = api_key.id if api_key else None
    key_label = api_key.label if api_key else ""
    team_name = api_key.team.name if api_key and api_key.team else ""
    event = UsageEvent(
        api_key_id=key_id,
        team_id=team_id,
        key_label=key_label,
        team_name=team_name,
        service=service,
        method=method,
        path=path[:512],
        host=host[:256],
        client_ip=ip[:64],
        model=model,
        status=status,
        result=result,
        duration_ms=duration_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        audio_seconds=audio_seconds,
        response_chars=response_chars,
        watts=watts,
        watt_hours=watt_hours,
        power_status=(power_status or "")[:32],
        pool_cost=pool_cost,
        is_demo=False,
    )
    db.add(event)
    db.flush()
    bump_usage_daily(
        db,
        team_id=team_id,
        api_key_id=key_id,
        team_name=team_name,
        key_label=key_label,
        service=service,
        model=model,
        result=result,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        audio_seconds=audio_seconds,
        response_chars=response_chars,
        duration_ms=duration_ms,
        watt_hours=watt_hours,
        pool_cost=pool_cost,
    )
    if result == "rate_limit":
        maybe_alert(
            db,
            event="rate_limit",
            message=f"key={key_label or '?'} service={service} model={model or '-'}",
        )
    return event


def authorize(
    db: Session,
    *,
    raw_key: str | None,
    service: str,
    kind: str,
    method: str,
    path: str,
    host: str,
    client_ip: str,
    body: bytes | None = None,
    content_type: str | None = None,
    defer_metering: bool = False,
) -> AuthResult:
    started = time.perf_counter()
    model = None
    if kind in MODEL_CHECK_KINDS:
        model = extract_request_model(kind, body, content_type)

    def _fail(status: int, reason: str, api_key: ApiKey | None = None, **kw) -> AuthResult:
        log_usage(
            db,
            api_key=api_key,
            service=service,
            kind=kind,
            method=method,
            path=path,
            host=host,
            client_ip=client_ip,
            model=model,
            status=status,
            result="rate_limit" if status == 429 else "deny",
            duration_ms=(time.perf_counter() - started) * 1000,
            body=body,
        )
        db.commit()
        return AuthResult(status=status, reason=reason, api_key=api_key, model=model, **kw)

    if not raw_key:
        return _fail(401, "missing_api_key")

    key_hash = hash_api_key(raw_key)
    api_key = (
        db.query(ApiKey)
        .options(
            joinedload(ApiKey.team).joinedload(Team.service_grants),
            joinedload(ApiKey.team).joinedload(Team.model_allowlists),
            joinedload(ApiKey.team).joinedload(Team.model_limits),
            joinedload(ApiKey.owner).joinedload(WebUser.service_grants),
            joinedload(ApiKey.owner).joinedload(WebUser.model_allowlists),
            joinedload(ApiKey.service_grants),
            joinedload(ApiKey.model_allowlists),
            joinedload(ApiKey.model_limits),
        )
        .filter(ApiKey.key_hash == key_hash)
        .first()
    )

    if api_key is None or not api_key.is_active:
        return _fail(401, "invalid_api_key")

    if api_key.expires_at and api_key.expires_at < utcnow():
        return _fail(401, "expired_api_key", api_key)

    if service not in _services_for_key(api_key, db):
        return _fail(403, "service_not_allowed", api_key)

    if kind in MODEL_CHECK_KINDS:
        writing = method.upper() in {"POST", "PUT", "PATCH"}
        if kind in MODEL_REQUIRED_KINDS and writing and not model:
            return _fail(400, "missing_model", api_key)
        if model:
            allow = _models_for_key(api_key, service, db)
            if allow is not None and model not in allow:
                from ..model_aliases import model_allowed_via_alias

                # Allowlist may name the public alias (qwen3.6); request is rewritten
                # to the active target before authorize — treat that as a match.
                if not model_allowed_via_alias(db, allow, model):
                    return _fail(403, "model_not_allowed", api_key)
            from ..data.catalog import is_model_globally_enabled

            if not is_model_globally_enabled(db, service, model):
                return _fail(403, "model_disabled", api_key)

    rpm, concurrency, priority, daily_quota, monthly_quota = _effective_limits(api_key)
    mlim = _pick_model_limit(api_key, service, model)
    owner_id, user_rpm, user_conc, user_daily = _owner_ceilings(api_key)

    if monthly_quota and monthly_quota > 0 and api_key.team_id:
        if _monthly_ok_count(db, api_key.team_id) >= monthly_quota:
            return _fail(429, "team_monthly_quota_exceeded", api_key, priority=priority)

    if not priority_gate.acquire(key_id=api_key.id, priority=priority, timeout=2.0):
        return _fail(429, "priority_queue_timeout", api_key, priority=priority)

    decision = rate_limiter.check_and_acquire(
        key_id=api_key.id,
        team_id=api_key.team_id,
        rpm=rpm,
        concurrency=concurrency,
        model=model,
        model_rpm=mlim.rpm_limit if mlim else None,
        model_concurrency=mlim.concurrency_limit if mlim else None,
        key_daily_quota=api_key.daily_quota,
        team_daily_quota=api_key.team.daily_quota if api_key.team else None,
        model_daily_quota=mlim.daily_quota if mlim else None,
        user_id=owner_id,
        user_rpm=user_rpm,
        user_concurrency=user_conc,
        user_daily_quota=user_daily,
    )
    if not decision.allowed:
        priority_gate.release(api_key.id)
        return _fail(
            429,
            decision.reason,
            api_key,
            remaining_rpm=decision.remaining_rpm,
            priority=priority,
        )

    hold_concurrency = bool(defer_metering) and kind in _STREAM_KINDS
    lease: ConcurrencyLease | None = None
    if hold_concurrency:
        lease = ConcurrencyLease(
            key_id=api_key.id,
            user_id=owner_id,
            model=model,
        )
    else:
        rate_limiter.release(api_key.id, model=model, user_id=owner_id)
        priority_gate.release(api_key.id)

    from ..web.accounts import get_auth_settings
    from ..usage_pool import check_and_consume_pool

    pool = check_and_consume_pool(
        db,
        api_key=api_key,
        auth=get_auth_settings(db),
        service=service,
        model=model,
        body=body,
    )
    if not pool.allowed:
        release_concurrency_lease(lease)
        return _fail(
            429,
            pool.reason,
            api_key,
            remaining_rpm=decision.remaining_rpm,
            priority=priority,
        )

    temp_block = check_temperature(db, service)
    if temp_block is not None:
        release_concurrency_lease(lease)
        return _fail(temp_block.status, temp_block.reason, api_key, priority=priority)

    # Only /v1/onprem/entry (or VL forward hop) finalizes duration/Wh.
    # Plain auth_request must not leave orphan deferred events.
    meter_proxy = hold_concurrency

    api_key.last_used_at = utcnow()
    event = log_usage(
        db,
        api_key=api_key,
        service=service,
        kind=kind,
        method=method,
        path=path,
        host=host,
        client_ip=client_ip,
        model=model,
        status=204,
        result="ok",
        duration_ms=(time.perf_counter() - started) * 1000,
        body=body,
        watts=None if meter_proxy else pool.watts,
        watt_hours=None if meter_proxy else pool.watt_hours,
        pool_cost=pool.cost if pool.cost else None,
        defer_metering=meter_proxy,
    )
    check_quota_alert(
        db,
        key_id=api_key.id,
        team_id=api_key.team_id,
        key_q=api_key.daily_quota,
        team_q=api_key.team.daily_quota if api_key.team else None,
        user_id=owner_id,
        user_q=user_daily,
        label=api_key.label,
    )
    db.commit()

    return AuthResult(
        status=204,
        reason="ok",
        remaining_rpm=decision.remaining_rpm,
        priority=priority,
        api_key=api_key,
        model=model,
        usage_event_id=event.id if meter_proxy else None,
        meter_proxy=meter_proxy,
        concurrency_lease=lease,
    )
