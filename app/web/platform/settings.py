from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from ...audit import write_audit
from ...data.backends import source_chip_rows, source_names
from ...data.db import get_db
from ...data.models import WebUser, Team
from ..session import require_platform_admin, require_user
from ..shared import (
    templates,
    _collect_models_from_form,
    _gpu_power_enabled,
    _parse_services,
    _settings,
    _temp_guard_enabled,
)

router = APIRouter()


@router.get("/settings")
def settings_root(
    user: Annotated[WebUser, Depends(require_user)],
):
    if user.is_platform_admin:
        return RedirectResponse("/settings/access", status_code=303)
    return RedirectResponse("/settings/system", status_code=303)


def _settings_page_context(
    request: Request,
    db: Session,
    user: WebUser,
    *,
    settings_tab: str,
) -> dict:
    import os

    from ...config import onprem_api_port
    from ...data.backends import source_rows
    from ...data.grants import (
        AccessCeiling,
        catalog_groups_for_ceiling,
        display_default_models,
        display_default_sources,
    )
    from ...data.routing_strategy import routing_strategy_choices
    from ...data.usage_weights import catalog_weight_suggestions
    from ..accounts import get_auth_settings

    settings = _settings(request)
    public_host = request.headers.get("x-forwarded-host") or request.headers.get("host") or "127.0.0.1"
    host_only = public_host.split(":")[0]
    admin_url = str(request.base_url).rstrip("/")
    gw_port = onprem_api_port()
    api_base_url = f"http://{host_only}:{gw_port}"

    backend_rows_data = source_rows(db, settings)
    auth = get_auth_settings(db) if user.is_platform_admin else None
    teams = (
        db.query(Team).order_by(Team.name).all() if user.is_platform_admin else []
    )
    flash_ok = request.session.pop("flash_ok", None)
    flash_err = request.session.pop("flash_err", None)
    observed_tok_s = None
    if auth is not None:
        from ...usage_pool import migrate_pool_to_token_budget, observed_tokens_per_sec

        if migrate_pool_to_token_budget(db, auth):
            db.commit()
            db.refresh(auth)
        observed_tok_s = observed_tokens_per_sec(db)
    weight_status = catalog_weight_suggestions(db)
    grant_ctx: dict = {}
    if auth is not None:
        grant_ctx = {
            "source_chips": source_chip_rows(db),
            "selected_services": display_default_sources(db),
            "catalog_groups": catalog_groups_for_ceiling(
                db, AccessCeiling(unrestricted=True)
            ),
            "selected_models": display_default_models(db),
        }

    return {
        "user": user,
        "settings": settings,
        "nav": "settings",
        "settings_tab": settings_tab,
        "session_max_age": settings.session_max_age,
        "temp_guard_enabled": _temp_guard_enabled(request, db),
        "temp_guard_disabled": settings.temp_guard_disabled,
        "admin_url": admin_url,
        "api_base_url": api_base_url,
        "backend_rows": backend_rows_data,
        "auth": auth,
        "teams": teams,
        "flash_ok": flash_ok,
        "flash_err": flash_err,
        "gpu_power_enabled": _gpu_power_enabled(request, db),
        "observed_tok_s": observed_tok_s,
        "weight_status": weight_status,
        "routing_strategies": routing_strategy_choices(),
        **grant_ctx,
    }


def _render_settings(
    request: Request,
    db: Session,
    user: WebUser,
    tab: str,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "settings.html",
        _settings_page_context(request, db, user, settings_tab=tab),
    )


@router.get("/settings/access", response_class=HTMLResponse)
def settings_access_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    return _render_settings(request, db, user, "access")


@router.get("/settings/limits", response_class=HTMLResponse)
def settings_limits_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    return _render_settings(request, db, user, "limits")


@router.get("/settings/routing", response_class=HTMLResponse)
def settings_routing_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    return _render_settings(request, db, user, "routing")


@router.get("/settings/privacy", response_class=HTMLResponse)
def settings_privacy_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    return _render_settings(request, db, user, "privacy")


@router.get("/settings/system", response_class=HTMLResponse)
def settings_system_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_user)],
):
    return _render_settings(request, db, user, "system")


@router.post("/settings/default-grant")
async def settings_default_grant_save(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...data.grants import save_default_grant

    form = await request.form()
    names = source_names(db)
    services = _parse_services(form.getlist("services"), names)
    models = [(s, m) for s, m in _collect_models_from_form(form, db) if s in services]
    save_default_grant(db, services, models)
    write_audit(
        db,
        actor=user,
        action="settings.default_grant",
        entity_type="auth_settings",
        detail=f"services={services} models={len(models)}",
    )
    db.commit()
    if services:
        request.session["flash_ok"] = "Default access for new users saved."
    else:
        request.session["flash_ok"] = "Saved: new users get no sources until you grant them."
    return RedirectResponse("/settings/access", status_code=303)


def _settings_redirect(request: Request, tab: str, message: str) -> RedirectResponse:
    request.session["flash_ok"] = message
    return RedirectResponse(f"/settings/{tab}", status_code=303)


@router.post("/settings/access")
def settings_access_save(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
    allow_self_registration: str = Form(""),
    teams_enabled: str = Form(""),
    default_team_id: str = Form(""),
    max_keys_per_user: str = Form("3"),
):
    from ..accounts import get_auth_settings

    auth = get_auth_settings(db)
    auth.allow_self_registration = allow_self_registration == "on"
    auth.require_email = True
    auth.teams_enabled = teams_enabled == "on"
    try:
        auth.max_keys_per_user = max(0, min(100, int(max_keys_per_user or "3")))
    except ValueError:
        auth.max_keys_per_user = 3
    tid = default_team_id.strip()
    if auth.teams_enabled and tid:
        team = db.get(Team, int(tid))
        auth.default_team_id = team.id if team else None
    else:
        auth.default_team_id = None
    write_audit(
        db,
        actor=user,
        action="settings.access",
        entity_type="auth_settings",
        entity_id=auth.id,
        detail=f"register={auth.allow_self_registration} teams={auth.teams_enabled} max_keys={auth.max_keys_per_user}",
    )
    db.commit()
    return _settings_redirect(request, "access", "Access & registration saved.")


@router.post("/settings/limits")
def settings_limits_save(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
    pool_window_hours: str = Form("5"),
    pool_min_cost: str = Form("1"),
    pool_model_weights_enabled: str = Form(""),
):
    from ..accounts import get_auth_settings
    from ...usage_pool import migrate_pool_to_token_budget, resolve_tokens_per_sec

    auth = get_auth_settings(db)
    migrate_pool_to_token_budget(db, auth)
    try:
        auth.pool_window_hours = max(0, min(8760, int(pool_window_hours or "5")))
    except ValueError:
        auth.pool_window_hours = 5
    auth.pool_tokens_per_unit = 1
    try:
        auth.pool_min_cost = max(0.0, float(pool_min_cost or "1"))
    except ValueError:
        auth.pool_min_cost = 1.0
    auth.pool_watt_weight = 0.0
    auth.pool_tokens_per_sec = resolve_tokens_per_sec(db)
    auth.pool_model_weights_enabled = pool_model_weights_enabled == "on"
    write_audit(
        db,
        actor=user,
        action="settings.limits",
        entity_type="auth_settings",
        entity_id=auth.id,
        detail=f"pool_h={auth.pool_window_hours} pool_weights={auth.pool_model_weights_enabled}",
    )
    db.commit()
    return _settings_redirect(request, "limits", "Limits & budget saved.")


@router.post("/settings/routing")
def settings_routing_save(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
    auto_vl_routing: str = Form(""),
    preflight_upstream: str = Form(""),
    routing_strategy: str = Form("load_aware"),
    source_admission_enabled: str = Form(""),
    catalog_prune_on_sync: str = Form(""),
    catalog_auto_sync: str = Form(""),
    catalog_auto_sync_minutes: str = Form("30"),
    soft_sampling_defaults: str = Form(""),
    source_queue_timeout_sec: str = Form("30"),
    auto_model_default: str = Form(""),
    auto_model_quality: str = Form(""),
    auto_model_long: str = Form(""),
):
    from ..accounts import get_auth_settings

    from ...catalog_auto_sync import clamp_auto_sync_minutes
    from ...data.routing_strategy import normalize_routing_strategy

    auth = get_auth_settings(db)
    auth.auto_vl_routing = auto_vl_routing == "on"
    auth.preflight_upstream = preflight_upstream == "on"
    strategy = normalize_routing_strategy(routing_strategy)
    auth.routing_strategy = strategy
    auth.load_aware_routing = strategy == "load_aware"
    auth.source_admission_enabled = source_admission_enabled == "on"
    auth.catalog_prune_on_sync = catalog_prune_on_sync == "on"
    auth.catalog_auto_sync = catalog_auto_sync == "on"
    auth.catalog_auto_sync_minutes = clamp_auto_sync_minutes(catalog_auto_sync_minutes)
    auth.soft_sampling_defaults = soft_sampling_defaults == "on"
    try:
        auth.source_queue_timeout_sec = max(
            1, min(600, int(source_queue_timeout_sec or "30"))
        )
    except ValueError:
        auth.source_queue_timeout_sec = 30
    auth.auto_model_default = (auto_model_default or "").strip()[:256]
    auth.auto_model_quality = (auto_model_quality or "").strip()[:256]
    auth.auto_model_long = (auto_model_long or "").strip()[:256]
    write_audit(
        db,
        actor=user,
        action="settings.routing",
        entity_type="auth_settings",
        entity_id=auth.id,
        detail=(
            f"auto_vl={auth.auto_vl_routing} preflight={auth.preflight_upstream} "
            f"routing={auth.routing_strategy} admission={auth.source_admission_enabled} "
            f"catalog_prune={auth.catalog_prune_on_sync} "
            f"catalog_auto_sync={auth.catalog_auto_sync}/{auth.catalog_auto_sync_minutes}m "
            f"soft_sampling={auth.soft_sampling_defaults} "
            f"queue_s={auth.source_queue_timeout_sec}"
        ),
    )
    db.commit()
    return _settings_redirect(request, "routing", "Routing saved.")


@router.post("/settings/privacy")
def settings_privacy_save(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
    anonymize_client_ip: str = Form(""),
    retention_days: str = Form("30"),
):
    from ..accounts import get_auth_settings
    from ...privacy import purge_old_usage

    auth = get_auth_settings(db)
    auth.anonymize_client_ip = anonymize_client_ip == "on"
    try:
        auth.retention_days = max(0, min(3650, int(retention_days or "30")))
    except ValueError:
        auth.retention_days = 30
    purged = purge_old_usage(db, auth.retention_days)
    write_audit(
        db,
        actor=user,
        action="settings.privacy",
        entity_type="auth_settings",
        entity_id=auth.id,
        detail=f"anon_ip={auth.anonymize_client_ip} retention={auth.retention_days} purged={purged}",
    )
    db.commit()
    msg = (
        f"Privacy saved. Purged {purged} old usage events."
        if purged
        else "Privacy & retention saved."
    )
    return _settings_redirect(request, "privacy", msg)


@router.post("/settings/system")
def settings_system_save(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
    operator_name: str = Form(""),
    operator_address: str = Form(""),
    operator_email: str = Form(""),
    operator_phone: str = Form(""),
):
    from ..accounts import DEFAULT_OPERATOR_EMAIL, get_auth_settings

    auth = get_auth_settings(db)
    auth.operator_name = operator_name.strip()
    auth.operator_address = operator_address.strip()
    auth.operator_email = operator_email.strip() or DEFAULT_OPERATOR_EMAIL
    auth.operator_phone = operator_phone.strip()
    write_audit(
        db,
        actor=user,
        action="settings.system",
        entity_type="auth_settings",
        entity_id=auth.id,
        detail=f"operator_email={auth.operator_email}",
    )
    db.commit()
    return _settings_redirect(request, "system", "Operator / Impressum saved.")
