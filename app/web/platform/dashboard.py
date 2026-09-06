from __future__ import annotations

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from ...data.backends import hardware_labels
from ...data.db import get_db
from ...data.models import WebUser, ApiKey, Team, utcnow
from ..dashboard_ops import attention_items, fleet_statuses, fleet_summary
from ..session import require_user
from .setup import _wizard_or_next
from ..shared import templates, _gpu_power_enabled, _teams_on

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_user)],
):
    if not user.is_platform_admin:
        return RedirectResponse("/me", status_code=303)
    redir = _wizard_or_next(db, user)
    if redir is not None:
        return redir

    from ...stats import (
        bar_chart_svg,
        daily_traffic_chart_svg,
        usage_stats,
        week_window_start,
        zone_from_request,
        pulse_stats,
        area_chart_svg,
    )
    from ..dashboard_ops import attention_items, fleet_statuses, fleet_summary
    from ..setup import setup_status
    from ...data.backends import hardware_labels
    from ...auth.source_admission import live_admission_rows
    from ...data.source_capacity import format_slots_label
    from ...stats import enrich_energy_with_owners, energy_by_owner

    zone = zone_from_request(request, user)
    now = utcnow()
    day_ago = now - timedelta(days=1)
    week_ago = week_window_start(zone)
    day = usage_stats(db, since=day_ago, tz=zone)
    week = usage_stats(db, since=week_ago, tz=zone)
    fleet = fleet_statuses(db)
    gpu_on = _gpu_power_enabled(request, db)
    tz_label = str(zone)
    active_keys = db.query(func.count(ApiKey.id)).filter(ApiKey.is_active.is_(True)).scalar() or 0
    teams_on = _teams_on(db)
    teams_count = (
        (db.query(func.count(Team.id)).scalar() or 0) if teams_on else 0
    )
    flash_ok = request.session.pop("flash_ok", None)
    setup = setup_status(db)
    attention = attention_items(
        db,
        fleet=fleet,
        denies_24h=day["denies"],
        rate_limits_24h=day["rate_limits"],
    )
    pulse = pulse_stats(db)
    if not fleet:
        pulse_status = "Idle"
    elif (fleet_summary(fleet) or {}).get("down"):
        pulse_status = "Degraded"
    else:
        pulse_status = "Healthy"
    live_slots = live_admission_rows(db)
    energy_keys = enrich_energy_with_owners(db, list(week["energy_by_key"]))
    energy_owners = energy_by_owner(energy_keys)

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "user": user,
            "total_day": day["total_ok"],
            "total_week": week["total_ok"],
            "denies": day["denies"],
            "rate_limits": day["rate_limits"],
            "top_keys": day["top_keys"],
            "by_service": day["by_service"],
            "by_model": day["by_model"],
            "tokens_in": day["tokens_in"],
            "watt_hours_day": day["watt_hours"],
            "watt_hours_week": week["watt_hours"],
            "energy_by_key": energy_keys,
            "energy_by_owner": energy_owners,
            "latency_p95": day["latency_p95"],
            "chart_service": bar_chart_svg(day["by_service"], unit="requests"),
            "chart_model": bar_chart_svg([(m, c) for m, c in day["by_model"]], unit="requests"),
            "chart_keys": bar_chart_svg(day["top_keys"], unit="requests"),
            "chart_daily": daily_traffic_chart_svg(week["daily_series"], tz_label=tz_label),
            "pulse": pulse,
            "pulse_status": pulse_status,
            "pulse_cta": None,
            "chart_pulse": area_chart_svg(
                pulse["series"], fill_id="gw-pulse", aria="Throughput last 60 minutes"
            ),
            "active_keys": active_keys,
            "teams_count": teams_count,
            "teams_enabled": teams_on,
            "display_timezone": tz_label,
            "flash_ok": flash_ok,
            "nav": "dashboard",
            "is_admin": True,
            "setup": setup,
            "gpu_power_enabled": gpu_on,
            "fleet": fleet,
            "fleet_summary": fleet_summary(fleet),
            "hardware_by_name": hardware_labels(db),
            "format_slots_label": format_slots_label,
            "attention": attention,
            "live_slots": live_slots,
        },
    )
