from __future__ import annotations

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from ...audit import write_audit
from ...data.backends import source_chip_rows, source_names
from ...data.db import (
    generate_api_key,
    get_db,
    hash_api_key,
    key_display_prefix,
)
from ...data.models import WebUser, ApiKey, Team, TeamMember, UsageEvent, utcnow
from ...data.grants import configured_default_sources
from ..ops import sync_key_model_limits, sync_team_model_limits, _format_model_limits
from ..session import (
    Forbidden,
    can_access_key,
    can_access_team,
    require_platform_admin,
    require_user,
    scope_keys_query,
    scoped_key_ids,
    user_team_ids,
    user_teams,
)
from ..shared import (
    templates,
    _collect_models_from_form,
    _gpu_power_enabled,
    _key_form_context,
    _parse_services,
    _parse_model_checks,
    _resolve_ceiling,
    _selected_model_keys,
    _source_tips,
    _sync_key_grants,
    _sync_key_models,
    _sync_team_grants,
    _sync_team_models,
    _sync_team_members,
    _teams_on,
)

# _key_teams_owners and _load_editable_key below


router = APIRouter()


def _keys_list_url(*, owner_user_id: int | None = None, api_key: ApiKey | None = None) -> str:
    oid = owner_user_id if owner_user_id is not None else (
        api_key.owner_user_id if api_key else None
    )
    if oid:
        return f"/keys?owner_user_id={oid}"
    return "/keys"


def _apply_key_routing_from_form(api_key: ApiKey, form, db: Session) -> None:
    from ...data.routing_strategy import normalize_routing_strategy
    from ...sampling_merge import normalize_key_sampling_profile

    names = set(source_names(db))
    raw_strat = (form.get("routing_strategy") or "").strip()
    api_key.routing_strategy = (
        normalize_routing_strategy(raw_strat) if raw_strat else None
    )
    pref = str(form.get("preferred_source") or "").strip().lower()
    api_key.preferred_source = pref if pref in names else None
    api_key.sampling_profile = normalize_key_sampling_profile(
        str(form.get("sampling_profile") or "")
    )


@router.get("/keys", response_class=HTMLResponse)
def keys_list(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_user)],
    owner_user_id: int | None = None,
):
    from ...config import public_api_base

    teams_on = _teams_on(db)
    q = db.query(ApiKey).options(
        joinedload(ApiKey.team),
        joinedload(ApiKey.service_grants),
        joinedload(ApiKey.owner),
    )
    q = scope_keys_query(q, user, teams_enabled=teams_on)
    filter_owner: WebUser | None = None
    if owner_user_id is not None:
        if not user.is_platform_admin and owner_user_id != user.id:
            raise Forbidden()
        filter_owner = db.get(WebUser, owner_user_id)
        if filter_owner is None:
            request.session["flash_err"] = "User not found."
            return RedirectResponse("/keys", status_code=303)
        q = q.filter(ApiKey.owner_user_id == owner_user_id)
    keys = q.order_by(ApiKey.created_at.desc()).all()
    # All-owners admin view: group by owner so identical labels (e.g. main) stay distinct.
    if user.is_platform_admin and owner_user_id is None and not teams_on:
        from ..accounts import user_display_name

        keys.sort(
            key=lambda k: (
                (user_display_name(k.owner) if k.owner else "").lower(),
                (k.label or "").lower(),
                -(k.created_at.timestamp() if k.created_at else 0),
            )
        )
    key_ids = [k.id for k in keys]
    day_ago = utcnow() - timedelta(days=1)
    usage_24h: dict[int, int] = {}
    last_used: dict[int, object] = {}
    if key_ids:
        usage_rows = (
            db.query(UsageEvent.api_key_id, func.count(UsageEvent.id))
            .filter(
                UsageEvent.api_key_id.in_(key_ids),
                UsageEvent.created_at >= day_ago,
            )
            .group_by(UsageEvent.api_key_id)
            .all()
        )
        usage_24h = {int(kid): int(n) for kid, n in usage_rows if kid is not None}
        last_rows = (
            db.query(UsageEvent.api_key_id, func.max(UsageEvent.created_at))
            .filter(UsageEvent.api_key_id.in_(key_ids))
            .group_by(UsageEvent.api_key_id)
            .all()
        )
        last_used = {int(kid): ts for kid, ts in last_rows if kid is not None and ts is not None}
    from ...stats import energy_wh_by_key_ids
    from ..shared import _gpu_power_enabled

    energy_7d = energy_wh_by_key_ids(db, key_ids, lookback_days=7) if key_ids else {}
    gpu_on = _gpu_power_enabled(request, db)
    active_count = sum(1 for k in keys if k.is_active)
    teams, owners = _key_teams_owners(db, user, teams_on)
    return templates.TemplateResponse(
        request,
        "keys.html",
        {
            "user": user,
            "keys": keys,
            "services": source_names(db),
            "teams": teams,
            "owners": owners,
            "filter_owner": filter_owner,
            "filter_owner_user_id": owner_user_id,
            "flash_key": request.session.pop("flash_key", None),
            "flash_key_services": request.session.pop("flash_key_services", None),
            "flash_ok": request.session.pop("flash_ok", None),
            "flash_err": request.session.pop("flash_err", None),
            "api_base": public_api_base(),
            "active_count": active_count,
            "usage_24h": usage_24h,
            "last_used": last_used,
            "energy_7d": energy_7d,
            "gpu_power_enabled": gpu_on,
            "source_tips": _source_tips(db),
            "nav": "keys",
            "is_admin": user.is_platform_admin,
            "teams_enabled": teams_on,
        },
    )


def _key_teams_owners(db: Session, user: WebUser, teams_on: bool) -> tuple[list, list]:
    teams = (
        (
            db.query(Team).order_by(Team.name).all()
            if user.is_platform_admin
            else user_teams(user)
        )
        if teams_on
        else []
    )
    owners = (
        db.query(WebUser).order_by(WebUser.username).all()
        if user.is_platform_admin and not teams_on
        else []
    )
    return teams, owners


def _load_editable_key(db: Session, key_id: int):
    return (
        db.query(ApiKey)
        .options(
            joinedload(ApiKey.service_grants),
            joinedload(ApiKey.model_allowlists),
            joinedload(ApiKey.model_limits),
            joinedload(ApiKey.team).joinedload(Team.service_grants),
            joinedload(ApiKey.team).joinedload(Team.model_allowlists),
            joinedload(ApiKey.owner).joinedload(WebUser.service_grants),
            joinedload(ApiKey.owner).joinedload(WebUser.model_allowlists),
        )
        .filter(ApiKey.id == key_id)
        .first()
    )


@router.get("/keys/new", response_class=HTMLResponse)
def keys_new(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_user)],
    owner_user_id: int | None = None,
):
    teams_on = _teams_on(db)
    teams = (
        (
            db.query(Team).order_by(Team.name).all()
            if user.is_platform_admin
            else user_teams(user)
        )
        if teams_on
        else []
    )
    owners = (
        db.query(WebUser).order_by(WebUser.username).all()
        if user.is_platform_admin and not teams_on
        else []
    )
    default_team = teams[0].id if teams_on and len(teams) == 1 else None
    default_owner = user.id
    if (
        not teams_on
        and user.is_platform_admin
        and owner_user_id is not None
        and db.get(WebUser, owner_user_id) is not None
    ):
        default_owner = owner_user_id
    ctx = _key_form_context(
        db,
        user=user,
        teams_on=teams_on,
        api_key=None,
        teams=teams,
        owners=owners,
        team_id=default_team,
        owner_user_id=default_owner,
    )
    return templates.TemplateResponse(request, "key_form.html", ctx)


@router.post("/keys/new")
async def keys_create(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_user)],
):
    from ...data.grants import clamp_models, clamp_services, normalize_model_allowlist
    from ..accounts import assert_can_create_key

    teams_on = _teams_on(db)
    form = await request.form()
    label = str(form.get("label") or "").strip() or "unnamed"
    team_id = None
    owner_user_id = user.id
    if teams_on:
        team_id_raw = form.get("team_id")
        team_id = int(team_id_raw) if team_id_raw else None
        if not user.is_platform_admin:
            if team_id is None or not can_access_team(user, team_id):
                raise Forbidden()
        owner_user_id = user.id
    else:
        if user.is_platform_admin and form.get("owner_user_id"):
            owner_user_id = int(form.get("owner_user_id"))
        else:
            owner_user_id = user.id
        team_id = None

    owner = db.get(WebUser, owner_user_id) if owner_user_id else None
    blocked = assert_can_create_key(db, owner)
    if blocked:
        request.session["flash_err"] = blocked
        return RedirectResponse("/keys/new", status_code=303)

    ceil = _resolve_ceiling(
        db,
        teams_on=teams_on,
        acting_user=user,
        team_id=team_id,
        owner_user_id=owner_user_id,
    )
    names = source_names(db)
    services = clamp_services(
        _parse_services(form.getlist("services"), names), ceil, db
    )
    models = clamp_models(_collect_models_from_form(form, db), ceil)
    models = normalize_model_allowlist(db, services, models)
    rpm = form.get("rpm_limit")
    concurrency = form.get("concurrency_limit")
    priority = form.get("priority")
    daily = form.get("daily_quota")

    raw = generate_api_key()
    api_key = ApiKey(
        label=label,
        key_hash=hash_api_key(raw),
        key_prefix=key_display_prefix(raw),
        team_id=team_id,
        owner_user_id=owner_user_id,
        is_active=True,
        rpm_limit=int(rpm) if rpm else None,
        concurrency_limit=int(concurrency) if concurrency else None,
        daily_quota=int(daily) if daily else None,
        priority=int(priority) if priority not in (None, "") else None,
    )
    db.add(api_key)
    db.flush()
    _apply_key_routing_from_form(api_key, form, db)
    _sync_key_grants(db, api_key, services)
    _sync_key_models(db, api_key, models)
    sync_key_model_limits(db, api_key, str(form.get("model_limits") or ""))
    write_audit(
        db,
        actor=user,
        action="key.create",
        entity_type="api_key",
        entity_id=api_key.id,
        detail=f"label={label} team_id={team_id} owner={owner_user_id}",
    )
    db.commit()
    request.session["flash_key"] = raw
    request.session["flash_key_services"] = (
        ", ".join(services) if services else "all from grant"
    )
    return RedirectResponse(_keys_list_url(owner_user_id=owner_user_id), status_code=303)


@router.get("/keys/{key_id}", response_class=HTMLResponse)
def keys_edit(
    key_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_user)],
):
    teams_on = _teams_on(db)
    api_key = db.get(ApiKey, key_id)
    if not api_key:
        return RedirectResponse("/keys", status_code=303)
    if not can_access_key(user, api_key, teams_enabled=teams_on):
        raise Forbidden()
    return RedirectResponse(f"/keys?edit={key_id}", status_code=303)


@router.get("/keys/{key_id}/partial", response_class=HTMLResponse)
def keys_edit_partial(
    key_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_user)],
):
    teams_on = _teams_on(db)
    api_key = _load_editable_key(db, key_id)
    if not api_key:
        return HTMLResponse("Key not found", status_code=404)
    if not can_access_key(user, api_key, teams_enabled=teams_on):
        raise Forbidden()
    teams, owners = _key_teams_owners(db, user, teams_on)
    ctx = _key_form_context(
        db,
        user=user,
        teams_on=teams_on,
        api_key=api_key,
        teams=teams,
        owners=owners,
    )
    return templates.TemplateResponse(request, "_key_edit_expand.html", ctx)


@router.post("/keys/{key_id}")
async def keys_update(
    key_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_user)],
):
    from ...data.grants import clamp_models, clamp_services, normalize_model_allowlist

    teams_on = _teams_on(db)
    api_key = db.get(ApiKey, key_id)
    if not api_key:
        return RedirectResponse("/keys", status_code=303)
    if not can_access_key(user, api_key, teams_enabled=teams_on):
        raise Forbidden()
    form = await request.form()
    api_key.label = str(form.get("label") or "").strip() or api_key.label
    if teams_on:
        team_id = form.get("team_id")
        new_team = int(team_id) if team_id else None
        if not user.is_platform_admin and new_team and not can_access_team(user, new_team):
            raise Forbidden()
        api_key.team_id = new_team
    else:
        api_key.team_id = None
        if user.is_platform_admin and form.get("owner_user_id"):
            api_key.owner_user_id = int(form.get("owner_user_id"))
        elif not api_key.owner_user_id:
            api_key.owner_user_id = user.id
    api_key.is_active = form.get("is_active") == "on"
    rpm = form.get("rpm_limit")
    concurrency = form.get("concurrency_limit")
    priority = form.get("priority")
    daily = form.get("daily_quota")
    api_key.rpm_limit = int(rpm) if rpm else None
    api_key.concurrency_limit = int(concurrency) if concurrency else None
    api_key.priority = int(priority) if priority not in (None, "") else None
    api_key.daily_quota = int(daily) if daily else None
    _apply_key_routing_from_form(api_key, form, db)

    ceil = _resolve_ceiling(
        db,
        teams_on=teams_on,
        acting_user=user,
        team_id=api_key.team_id,
        owner_user_id=api_key.owner_user_id,
        api_key=api_key,
    )
    # Refresh team/owner relationships for ceiling after team change
    if api_key.team_id:
        db.refresh(api_key)
        team = (
            db.query(Team)
            .options(
                joinedload(Team.service_grants),
                joinedload(Team.model_allowlists),
            )
            .filter(Team.id == api_key.team_id)
            .first()
        )
        api_key.team = team
        from ...data.grants import ceiling_from_team

        ceil = ceiling_from_team(team) if team else ceil
    else:
        from ...data.grants import ceiling_from_user, load_user_with_grants

        owner = load_user_with_grants(db, api_key.owner_user_id)
        if owner:
            ceil = ceiling_from_user(owner)

    names = source_names(db)
    services = clamp_services(
        _parse_services(form.getlist("services"), names), ceil, db
    )
    models = clamp_models(_collect_models_from_form(form, db), ceil)
    models = normalize_model_allowlist(db, services, models)
    _sync_key_grants(db, api_key, services)
    _sync_key_models(db, api_key, models)
    if "model_limits" in form:
        sync_key_model_limits(db, api_key, str(form.get("model_limits") or ""))
    write_audit(
        db,
        actor=user,
        action="key.update",
        entity_type="api_key",
        entity_id=api_key.id,
        detail=api_key.label,
    )
    db.commit()
    request.session["flash_ok"] = f"Saved {api_key.label}."
    return RedirectResponse(_keys_list_url(api_key=api_key), status_code=303)


@router.post("/keys/{key_id}/rotate")
def keys_rotate(
    key_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_user)],
):
    teams_on = _teams_on(db)
    api_key = db.get(ApiKey, key_id)
    if not api_key:
        return RedirectResponse("/keys", status_code=303)
    if not can_access_key(user, api_key, teams_enabled=teams_on):
        raise Forbidden()
    raw = generate_api_key()
    api_key.key_hash = hash_api_key(raw)
    api_key.key_prefix = key_display_prefix(raw)
    write_audit(
        db, actor=user, action="key.rotate", entity_type="api_key", entity_id=api_key.id
    )
    db.commit()
    request.session["flash_key"] = raw
    return RedirectResponse(_keys_list_url(api_key=api_key), status_code=303)


@router.post("/keys/{key_id}/revoke")
def keys_revoke(
    key_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_user)],
):
    teams_on = _teams_on(db)
    api_key = db.get(ApiKey, key_id)
    if api_key:
        if not can_access_key(user, api_key, teams_enabled=teams_on):
            raise Forbidden()
        api_key.is_active = False
        write_audit(
            db, actor=user, action="key.revoke", entity_type="api_key", entity_id=api_key.id
        )
        db.commit()
    return RedirectResponse(_keys_list_url(api_key=api_key), status_code=303)
