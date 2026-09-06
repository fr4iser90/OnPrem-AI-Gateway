from __future__ import annotations

import csv
import io
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session

from ...data.backends import source_names
from ...data.db import get_db
from ...data.models import WebUser, ApiKey, Team, UsageEvent, utcnow
from ..session import Forbidden, require_user, scope_keys_query, scoped_key_ids, user_team_ids, user_teams
from ..shared import templates, _gpu_power_enabled, _teams_on

router = APIRouter()


@router.get("/usage", response_class=HTMLResponse)
def usage_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_user)],
    service: str | None = None,
    team_id: int | None = None,
    key_id: int | None = None,
    result: str | None = None,
):
    from ...data.models import UsageDaily
    from ...stats import (
        daily_traffic_chart_svg,
        enrich_energy_with_owners,
        energy_by_owner,
        model_perf_averages,
        usage_stats,
        week_window_start,
        zone_from_request,
    )

    teams_on = _teams_on(db)
    q = db.query(UsageEvent).order_by(UsageEvent.created_at.desc())
    visible_ids = scoped_key_ids(db, user, teams_enabled=teams_on)
    if not user.is_platform_admin:
        q = q.filter(UsageEvent.api_key_id.in_(visible_ids)) if visible_ids else q.filter(False)
        if team_id and teams_on and team_id not in user_team_ids(user):
            raise Forbidden()
    if service:
        q = q.filter(UsageEvent.service == service)
    if team_id and teams_on:
        q = q.filter(UsageEvent.team_id == team_id)
    if key_id:
        q = q.filter(UsageEvent.api_key_id == key_id)
    if result:
        q = q.filter(UsageEvent.result == result)
    events = q.limit(200).all()
    teams = (
        (
            db.query(Team).order_by(Team.name).all()
            if user.is_platform_admin
            else user_teams(user)
        )
        if teams_on
        else []
    )
    keys_q = scope_keys_query(db.query(ApiKey), user, teams_enabled=teams_on)
    scoped_keys = keys_q.order_by(ApiKey.label).all()
    key_ids = [k.id for k in scoped_keys]
    zone = zone_from_request(request, user)
    now = utcnow()
    day_ago = now - timedelta(days=1)
    week_ago = week_window_start(zone)
    day = usage_stats(db, since=day_ago, key_ids=key_ids, tz=zone)
    week = usage_stats(db, since=week_ago, key_ids=key_ids, tz=zone)
    model_avgs = model_perf_averages(db, key_ids=key_ids, lookback_days=7)
    today = now.astimezone(zone).date()
    daily_q = db.query(UsageDaily).filter(UsageDaily.day == today)
    if key_ids:
        daily_q = daily_q.filter(UsageDaily.api_key_id.in_(key_ids))
    else:
        daily_q = daily_q.filter(False)
    daily_rows = daily_q.order_by(UsageDaily.ok_count.desc()).limit(50).all()
    gpu_on = _gpu_power_enabled(request, db)
    live_slots = []
    energy_keys = enrich_energy_with_owners(db, list(week["energy_by_key"]))
    energy_owners = energy_by_owner(energy_keys) if user.is_platform_admin else []
    if user.is_platform_admin:
        from ...auth.source_admission import live_admission_rows

        live_slots = live_admission_rows(db)
    return templates.TemplateResponse(
        request,
        "usage.html",
        {
            "user": user,
            "events": events,
            "services": source_names(db),
            "teams": teams,
            "keys": scoped_keys,
            "filters": {
                "service": service or "",
                "team_id": team_id or "",
                "key_id": key_id or "",
                "result": result or "",
            },
            "ok": day["total_ok"],
            "deny": day["denies"],
            "rate": day["rate_limits"],
            "tokens_in": day["tokens_in"],
            "latency_p95": day["latency_p95"],
            "watt_hours_day": day["watt_hours"],
            "watt_hours_week": week["watt_hours"],
            "energy_by_key": energy_keys,
            "energy_by_owner": energy_owners,
            "chart_daily": daily_traffic_chart_svg(week["daily_series"], tz_label=str(zone)),
            "model_avgs": model_avgs,
            "daily_rows": daily_rows,
            "live_slots": live_slots,
            "nav": "usage",
            "is_admin": user.is_platform_admin,
            "teams_enabled": teams_on,
            "gpu_power_enabled": gpu_on,
        },
    )


@router.get("/usage/export.csv")
def usage_export(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_user)],
    service: str | None = None,
    team_id: int | None = None,
    key_id: int | None = None,
    result: str | None = None,
):
    teams_on = _teams_on(db)
    q = db.query(UsageEvent).order_by(UsageEvent.created_at.desc())
    if not user.is_platform_admin:
        visible_ids = scoped_key_ids(db, user, teams_enabled=teams_on)
        q = q.filter(UsageEvent.api_key_id.in_(visible_ids)) if visible_ids else q.filter(False)
    if service:
        q = q.filter(UsageEvent.service == service)
    if team_id and teams_on:
        q = q.filter(UsageEvent.team_id == team_id)
    if key_id:
        q = q.filter(UsageEvent.api_key_id == key_id)
    if result:
        q = q.filter(UsageEvent.result == result)
    events = q.limit(5000).all()

    buf = io.StringIO()
    writer = csv.writer(buf)
    header = [
        "created_at",
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
        row = [
            e.created_at.isoformat() if e.created_at else "",
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
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=usage.csv"},
    )
