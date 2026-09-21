"""Shared builders for You (/usage) and Ops (/ops/usage) pages."""

from __future__ import annotations

import csv
import io
from datetime import timedelta
from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi.responses import StreamingResponse
from pydantic import BeforeValidator
from sqlalchemy.orm import Session, joinedload

from ..data.backends import source_names
from ..data.models import ApiKey, Team, UsageDaily, UsageEvent, WebUser, utcnow
from ..stats import (
    daily_traffic_chart_svg,
    days_window_start,
    enrich_energy_with_owners,
    energy_by_owner,
    model_perf_averages,
    usage_stats,
    zone_from_request,
)
from .accounts import get_auth_settings, user_display_name
from .session import Forbidden, user_team_ids, user_teams
from .shared import _gpu_power_enabled, _teams_on


def _blank_query_to_none(v: object) -> object:
    """HTML selects submit '' for 'all' — treat as missing optional int."""
    if v is None:
        return None
    if isinstance(v, str) and not v.strip():
        return None
    return v


OptionalQueryInt = Annotated[int | None, BeforeValidator(_blank_query_to_none)]


def resolve_usage_range(
    raw: str | None, retention_days: int
) -> tuple[int, str, str]:
    """(days, range_key, short_label) for KPI / chart lookback."""
    key = (raw or "7").strip().lower()
    ret = max(0, int(retention_days or 0))
    if key in {"30", "30d"}:
        return 30, "30", "30d"
    if key in {"retention", "all", "full"}:
        days = 365 if ret <= 0 else min(ret, 3650)
        label = f"{days}d" if ret > 0 else f"{days}d"
        return days, "retention", label
    return 7, "7", "7d"


def own_keys_query(db: Session, user: WebUser):
    """Keys owned by this user only (You portal — never fleet)."""
    return db.query(ApiKey).filter(ApiKey.owner_user_id == user.id)


def own_key_ids(db: Session, user: WebUser) -> list[int]:
    return [int(kid) for (kid,) in own_keys_query(db, user).with_entities(ApiKey.id).all()]


def owner_meta_by_key_ids(db: Session, key_ids: list[int]) -> dict[int, dict[str, Any]]:
    """api_key_id → {owner_user_id, owner_name}."""
    if not key_ids:
        return {}
    keys = (
        db.query(ApiKey)
        .options(joinedload(ApiKey.owner))
        .filter(ApiKey.id.in_(key_ids))
        .all()
    )
    out: dict[int, dict[str, Any]] = {}
    for k in keys:
        if k.owner is None:
            name = f"user#{k.owner_user_id}" if k.owner_user_id else "—"
        else:
            name = user_display_name(k.owner) or "—"
        out[int(k.id)] = {
            "owner_user_id": k.owner_user_id,
            "owner_name": name,
        }
    return out


def _range_options(retention_days: int) -> list[dict[str, str]]:
    return [
        {"key": "7", "label": "7 days"},
        {"key": "30", "label": "30 days"},
        {
            "key": "retention",
            "label": (
                f"Retention ({retention_days}d)"
                if retention_days > 0
                else "Retention (1y window)"
            ),
        },
    ]


def _filters_dict(
    *,
    service: str | None,
    team_id: int | None,
    key_id: int | None,
    result: str | None,
    range_key: str,
    owner_user_id: int | None,
) -> dict[str, Any]:
    return {
        "service": service or "",
        "team_id": team_id or "",
        "key_id": key_id or "",
        "result": result or "",
        "range": range_key,
        "owner_user_id": owner_user_id or "",
    }


def filter_query_string(filters: dict[str, Any], *, drop: set[str] | None = None) -> str:
    skip = drop or set()
    bits = {
        k: v
        for k, v in filters.items()
        if k not in skip and v not in ("", None)
    }
    return urlencode(bits)


def _resolve_scope(
    db: Session,
    user: WebUser,
    *,
    ops_mode: bool,
    team_id: int | None,
    owner_user_id: int | None,
    teams_on: bool,
) -> tuple[list[ApiKey], list[int], list[Team], list[WebUser]]:
    if ops_mode:
        keys_q = db.query(ApiKey)
        if owner_user_id:
            keys_q = keys_q.filter(ApiKey.owner_user_id == owner_user_id)
        scoped_keys = keys_q.order_by(ApiKey.label).all()
        teams = db.query(Team).order_by(Team.name).all() if teams_on else []
        owners = (
            db.query(WebUser)
            .filter(WebUser.is_active.is_(True))
            .order_by(WebUser.username)
            .all()
        )
    else:
        if team_id and teams_on and team_id not in user_team_ids(user):
            raise Forbidden()
        scoped_keys = own_keys_query(db, user).order_by(ApiKey.label).all()
        teams = user_teams(user) if teams_on else []
        owners = []
    key_ids = [k.id for k in scoped_keys]
    return scoped_keys, key_ids, teams, owners


def _filter_events_q(
    q,
    *,
    key_ids: list[int] | None,
    service: str | None,
    team_id: int | None,
    key_id: int | None,
    result: str | None,
    teams_on: bool,
):
    if key_ids is not None:
        q = q.filter(UsageEvent.api_key_id.in_(key_ids)) if key_ids else q.filter(False)
    if service:
        q = q.filter(UsageEvent.service == service)
    if team_id and teams_on:
        q = q.filter(UsageEvent.team_id == team_id)
    if key_id:
        q = q.filter(UsageEvent.api_key_id == key_id)
    if result:
        q = q.filter(UsageEvent.result == result)
    return q


def _filter_daily_q(
    q,
    *,
    key_ids: list[int] | None,
    service: str | None,
    team_id: int | None,
    key_id: int | None,
    teams_on: bool,
    day_from=None,
):
    if key_ids is not None:
        q = q.filter(UsageDaily.api_key_id.in_(key_ids)) if key_ids else q.filter(False)
    if service:
        q = q.filter(UsageDaily.service == service)
    if team_id and teams_on:
        q = q.filter(UsageDaily.team_id == team_id)
    if key_id:
        q = q.filter(UsageDaily.api_key_id == key_id)
    if day_from is not None:
        q = q.filter(UsageDaily.day >= day_from)
    return q


def build_usage_page_context(
    request,
    db: Session,
    user: WebUser,
    *,
    ops_mode: bool,
    service: str | None = None,
    team_id: int | None = None,
    key_id: int | None = None,
    result: str | None = None,
    range_raw: str | None = None,
    owner_user_id: int | None = None,
) -> dict[str, Any]:
    teams_on = _teams_on(db)
    auth = get_auth_settings(db)
    range_days, range_key, range_label = resolve_usage_range(
        range_raw, auth.retention_days
    )
    base_path = "/ops/usage" if ops_mode else "/usage"

    scoped_keys, key_ids, teams, owners = _resolve_scope(
        db,
        user,
        ops_mode=ops_mode,
        team_id=team_id,
        owner_user_id=owner_user_id,
        teams_on=teams_on,
    )

    q = db.query(UsageEvent).order_by(UsageEvent.created_at.desc())
    q = _filter_events_q(
        q,
        key_ids=key_ids,
        service=service,
        team_id=team_id,
        key_id=key_id,
        result=result,
        teams_on=teams_on,
    )
    events = q.limit(200).all()

    zone = zone_from_request(request, user)
    now = utcnow()
    day_ago = now - timedelta(days=1)
    range_start = days_window_start(zone, days=range_days)
    day = usage_stats(db, since=day_ago, key_ids=key_ids, tz=zone)
    period = usage_stats(db, since=range_start, key_ids=key_ids, tz=zone)
    model_avgs = model_perf_averages(
        db, key_ids=key_ids, lookback_days=range_days
    )
    today = now.astimezone(zone).date()
    daily_q = db.query(UsageDaily).filter(UsageDaily.day == today)
    daily_q = _filter_daily_q(
        daily_q,
        key_ids=key_ids,
        service=service,
        team_id=team_id,
        key_id=key_id,
        teams_on=teams_on,
    )
    daily_rows = daily_q.order_by(UsageDaily.ok_count.desc()).limit(50).all()

    owner_by_key = owner_meta_by_key_ids(
        db,
        list(
            {
                *(e.api_key_id for e in events if e.api_key_id is not None),
                *(r.api_key_id for r in daily_rows if r.api_key_id is not None),
                *(k.id for k in scoped_keys),
            }
        ),
    )

    gpu_on = _gpu_power_enabled(request, db)
    energy_keys = enrich_energy_with_owners(db, list(period["energy_by_key"]))
    energy_owners = energy_by_owner(energy_keys) if ops_mode else []
    live_slots = []
    if ops_mode:
        from ..auth.source_admission import live_admission_rows

        live_slots = live_admission_rows(db)

    filters = _filters_dict(
        service=service,
        team_id=team_id,
        key_id=key_id,
        result=result,
        range_key=range_key,
        owner_user_id=owner_user_id,
    )
    export_qs = filter_query_string(filters)
    subnav_qs = filter_query_string(filters)
    retention_days = int(auth.retention_days or 0)

    return {
        "user": user,
        "events": events,
        "services": source_names(db),
        "teams": teams,
        "keys": scoped_keys,
        "owners": owners,
        "owner_by_key": owner_by_key,
        "filters": filters,
        "ok": day["total_ok"],
        "deny": day["denies"],
        "rate": day["rate_limits"],
        "tokens_in": day["tokens_in"],
        "latency_p95": day["latency_p95"],
        "watt_hours_day": day["watt_hours"],
        "watt_hours_period": period["watt_hours"],
        "energy_by_key": energy_keys,
        "energy_by_owner": energy_owners,
        "chart_daily": daily_traffic_chart_svg(
            period["daily_series"],
            tz_label=str(zone),
            days_label=range_label,
        ),
        "model_avgs": model_avgs,
        "daily_rows": daily_rows,
        "live_slots": live_slots,
        "nav": "ops-usage" if ops_mode else "usage",
        "is_admin": user.is_platform_admin,
        "ops_mode": ops_mode,
        "base_path": base_path,
        "export_href": (
            f"{base_path}/export.csv?{export_qs}" if export_qs else f"{base_path}/export.csv"
        ),
        "events_href": f"{base_path}?{subnav_qs}" if subnav_qs else base_path,
        "daily_href": (
            f"{base_path}/daily?{subnav_qs}" if subnav_qs else f"{base_path}/daily"
        ),
        "filter_action": base_path,
        "show_result_filter": True,
        "range_days": range_days,
        "range_key": range_key,
        "range_label": range_label,
        "range_options": _range_options(retention_days),
        "retention_days": retention_days,
        "teams_enabled": teams_on,
        "gpu_power_enabled": gpu_on,
    }


def build_daily_page_context(
    request,
    db: Session,
    user: WebUser,
    *,
    ops_mode: bool,
    service: str | None = None,
    team_id: int | None = None,
    key_id: int | None = None,
    range_raw: str | None = None,
    owner_user_id: int | None = None,
) -> dict[str, Any]:
    teams_on = _teams_on(db)
    auth = get_auth_settings(db)
    range_days, range_key, range_label = resolve_usage_range(
        range_raw, auth.retention_days
    )
    base_path = "/ops/usage" if ops_mode else "/usage"

    scoped_keys, key_ids, teams, owners = _resolve_scope(
        db,
        user,
        ops_mode=ops_mode,
        team_id=team_id,
        owner_user_id=owner_user_id,
        teams_on=teams_on,
    )

    zone = zone_from_request(request, user)
    day_from = days_window_start(zone, days=range_days).date()

    q = db.query(UsageDaily).order_by(UsageDaily.day.desc(), UsageDaily.ok_count.desc())
    q = _filter_daily_q(
        q,
        key_ids=key_ids,
        service=service,
        team_id=team_id,
        key_id=key_id,
        teams_on=teams_on,
        day_from=day_from,
    )
    rows = q.limit(300).all()
    owner_by_key = owner_meta_by_key_ids(
        db,
        list({r.api_key_id for r in rows if r.api_key_id is not None}),
    )

    filters = _filters_dict(
        service=service,
        team_id=team_id,
        key_id=key_id,
        result=None,
        range_key=range_key,
        owner_user_id=owner_user_id,
    )
    subnav_qs = filter_query_string(filters)
    retention_days = int(auth.retention_days or 0)

    return {
        "user": user,
        "rows": rows,
        "services": source_names(db),
        "teams": teams,
        "keys": scoped_keys,
        "owners": owners,
        "owner_by_key": owner_by_key,
        "filters": filters,
        "nav": "ops-usage" if ops_mode else "usage",
        "is_admin": user.is_platform_admin,
        "ops_mode": ops_mode,
        "base_path": base_path,
        "events_href": f"{base_path}?{subnav_qs}" if subnav_qs else base_path,
        "daily_href": (
            f"{base_path}/daily?{subnav_qs}" if subnav_qs else f"{base_path}/daily"
        ),
        "filter_action": f"{base_path}/daily",
        "show_result_filter": False,
        "range_days": range_days,
        "range_key": range_key,
        "range_label": range_label,
        "range_options": _range_options(retention_days),
        "retention_days": retention_days,
        "teams_enabled": teams_on,
    }


def usage_csv_response(
    db: Session,
    user: WebUser,
    *,
    ops_mode: bool,
    service: str | None = None,
    team_id: int | None = None,
    key_id: int | None = None,
    result: str | None = None,
    owner_user_id: int | None = None,
) -> StreamingResponse:
    teams_on = _teams_on(db)
    _, key_ids, _, _ = _resolve_scope(
        db,
        user,
        ops_mode=ops_mode,
        team_id=team_id,
        owner_user_id=owner_user_id,
        teams_on=teams_on,
    )

    q = db.query(UsageEvent).order_by(UsageEvent.created_at.desc())
    q = _filter_events_q(
        q,
        key_ids=key_ids,
        service=service,
        team_id=team_id,
        key_id=key_id,
        result=result,
        teams_on=teams_on,
    )
    events = q.limit(5000).all()
    owner_by_key = (
        owner_meta_by_key_ids(
            db, [e.api_key_id for e in events if e.api_key_id is not None]
        )
        if ops_mode
        else {}
    )

    buf = io.StringIO()
    writer = csv.writer(buf)
    header = [
        "created_at",
        *(["owner"] if ops_mode else []),
        *(["team"] if teams_on else []),
        "key",
        "service",
        "method",
        "path",
        "host",
        "client_ip",
        "model",
        "status",
        "result",
        "duration_ms",
        "tokens_in",
        "tokens_out",
        "audio_seconds",
        "response_chars",
        "watts",
        "watt_hours",
        "power_status",
        "pool_cost",
        "is_demo",
    ]
    writer.writerow(header)
    for e in events:
        owner_name = ""
        if ops_mode:
            meta = owner_by_key.get(e.api_key_id) if e.api_key_id is not None else None
            owner_name = (meta or {}).get("owner_name") or ""
        row = [
            e.created_at.isoformat() if e.created_at else "",
            *([owner_name] if ops_mode else []),
            *([e.team_name] if teams_on else []),
            e.key_label,
            e.service,
            e.method,
            e.path,
            e.host,
            e.client_ip,
            e.model or "",
            e.status,
            e.result,
            e.duration_ms or "",
            e.tokens_in or "",
            e.tokens_out or "",
            e.audio_seconds or "",
            e.response_chars or "",
            e.watts if e.watts is not None else "",
            e.watt_hours if e.watt_hours is not None else "",
            getattr(e, "power_status", "") or "",
            e.pool_cost if e.pool_cost is not None else "",
            1 if e.is_demo else 0,
        ]
        writer.writerow(row)
    buf.seek(0)
    filename = "ops-usage.csv" if ops_mode else "usage.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
