"""Admin ops: model limits, audit, alerts, me-dashboard."""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from ..audit import verify_audit_chain, write_audit
from ..data.backends import source_names
from ..data.db import get_db
from ..data.models import (
    WebUser,
    AlertConfig,
    ApiKey,
    AuditLog,
    ModelLimit,
    Team,
    utcnow,
)
from .session import (
    Forbidden,
    can_access_team,
    require_platform_admin,
    require_user,
    user_teams,
)
from .templating import make_templates

templates = make_templates()
router = APIRouter()


def _parse_model_limits(raw: str, allowed: set[str] | list[str]) -> list[dict]:
    """
    Lines: source:model:rpm:concurrency:daily
    Example: chat:llama3.2:30:2:500
    """
    allowed_set = set(allowed)
    out = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(":")]
        if len(parts) < 2:
            continue
        service, model = parts[0], parts[1]
        if service not in allowed_set:
            continue
        rpm = int(parts[2]) if len(parts) > 2 and parts[2] else None
        conc = int(parts[3]) if len(parts) > 3 and parts[3] else None
        daily = int(parts[4]) if len(parts) > 4 and parts[4] else None
        out.append(
            {
                "service": service,
                "model_name": model,
                "rpm_limit": rpm,
                "concurrency_limit": conc,
                "daily_quota": daily,
            }
        )
    return out


def _format_model_limits(limits: list[ModelLimit]) -> str:
    lines = []
    for lim in limits:
        lines.append(
            f"{lim.service}:{lim.model_name}:"
            f"{lim.rpm_limit or ''}:"
            f"{lim.concurrency_limit or ''}:"
            f"{lim.daily_quota or ''}"
        )
    return "\n".join(lines)


def sync_key_model_limits(db: Session, api_key: ApiKey, raw: str) -> None:
    db.query(ModelLimit).filter(ModelLimit.api_key_id == api_key.id).delete()
    for item in _parse_model_limits(raw, source_names(db)):
        db.add(ModelLimit(api_key_id=api_key.id, **item))


def sync_team_model_limits(db: Session, team: Team, raw: str) -> None:
    db.query(ModelLimit).filter(ModelLimit.team_id == team.id).delete()
    for item in _parse_model_limits(raw, source_names(db)):
        db.add(ModelLimit(team_id=team.id, **item))


@router.get("/me", response_class=HTMLResponse)
def me_dashboard(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_user)],
):
    """User dashboard — owner-scoped or team-scoped depending on settings."""
    import os

    from ..config import public_api_base
    from ..stats import area_chart_svg, pulse_stats
    from .accounts import teams_feature_enabled
    from .session import scope_keys_query, scoped_key_ids

    teams_on = teams_feature_enabled(db)
    keys_q = scope_keys_query(db.query(ApiKey), user, teams_enabled=teams_on)
    keys = keys_q.order_by(ApiKey.created_at.desc()).limit(20).all()
    visible_ids = scoped_key_ids(db, user, teams_enabled=teams_on)
    pulse = pulse_stats(db, key_ids=None if user.is_platform_admin else visible_ids)
    api_base = public_api_base()
    return templates.TemplateResponse(
        request,
        "me.html",
        {
            "user": user,
            "api_base": api_base,
            "keys": keys,
            "flash_ok": request.session.pop("flash_ok", None),
            "flash_err": request.session.pop("flash_err", None),
            "pulse": pulse,
            "pulse_status": "Healthy" if pulse["count"] else "Idle",
            "pulse_cta": {"href": "/keys/new", "label": "New key"},
            "chart_pulse": area_chart_svg(
                pulse["series"], fill_id="gw-pulse", aria="Throughput last 60 minutes"
            ),
            "nav": "me",
            "is_admin": user.is_platform_admin,
            "teams_enabled": teams_on,
        },
    )


@router.get("/audit", response_class=HTMLResponse)
def audit_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    rows = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(200).all()
    chain_ok, chain_msg = verify_audit_chain(db)
    return templates.TemplateResponse(
        request,
        "audit.html",
        {
            "user": user,
            "rows": rows,
            "nav": "audit",
            "is_admin": True,
            "chain_ok": chain_ok,
            "chain_msg": chain_msg,
        },
    )


@router.get("/alerts", response_class=HTMLResponse)
def alerts_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    cfg = db.query(AlertConfig).first()
    return templates.TemplateResponse(
        request,
        "alerts.html",
        {"user": user, "cfg": cfg, "nav": "alerts", "is_admin": True},
    )


@router.post("/alerts")
async def alerts_save(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    form = await request.form()
    cfg = db.query(AlertConfig).first()
    if cfg is None:
        cfg = AlertConfig()
        db.add(cfg)
    cfg.webhook_url = str(form.get("webhook_url") or "").strip()
    cfg.enabled = form.get("enabled") == "on"
    cfg.alert_on_quota = form.get("alert_on_quota") == "on"
    cfg.alert_on_rate_limit = form.get("alert_on_rate_limit") == "on"
    cfg.quota_warn_pct = int(form.get("quota_warn_pct") or 80)
    write_audit(
        db,
        actor=user,
        action="alerts.update",
        entity_type="alert_config",
        entity_id=cfg.id or 0,
        detail=f"enabled={cfg.enabled} url_set={bool(cfg.webhook_url)}",
    )
    db.commit()
    return RedirectResponse("/alerts", status_code=303)


@router.post("/alerts/test")
def alerts_test(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ..audit import maybe_alert

    cfg = db.query(AlertConfig).first()
    if cfg:
        # temporarily force enabled path
        was = cfg.enabled
        cfg.enabled = True
        db.flush()
        maybe_alert(db, event="quota", message=f"test webhook by {user.username}", quota_pct=100)
        cfg.enabled = was
        write_audit(db, actor=user, action="alerts.test", entity_type="alert_config", detail="sent")
        db.commit()
    return RedirectResponse("/alerts", status_code=303)
