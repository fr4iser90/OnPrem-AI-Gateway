from __future__ import annotations

import csv
import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session, joinedload

from ...audit import write_audit
from ...data.backends import source_chip_rows, source_names
from ...data.catalog import list_catalog
from ...data.db import get_db, hash_password
from ...data.grants import configured_default_sources
from ...data.models import WebUser, ApiKey
from ..session import require_platform_admin
from ..shared import (
    templates,
    _collect_models_from_form,
    _parse_model_checks,
    _parse_services,
    _teams_on,
)

router = APIRouter()


@router.get("/users", response_class=HTMLResponse)
def users_list(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...data.grants import (
        AccessCeiling,
        catalog_groups_for_ceiling,
        ceiling_from_user,
        display_default_models,
        display_default_sources,
        grant_summary,
    )
    from ...data.usage_weights import catalog_weight_suggestions
    from ..accounts import (
        active_key_count,
        assert_can_create_key,
        get_auth_settings,
        user_display_name,
        user_needs_username,
    )
    from ..invites import pending_invites
    from ...mailer import get_smtp, smtp_ready
    from ..user_limits import user_limits_summary

    users = (
        db.query(WebUser)
        .options(
            joinedload(WebUser.service_grants),
            joinedload(WebUser.model_allowlists),
        )
        .order_by(WebUser.username)
        .all()
    )
    flash = request.session.pop("flash_ok", None)
    err = request.session.pop("flash_err", None)
    grant_labels = {u.id: grant_summary(ceiling_from_user(u)) for u in users}
    grant_sources = {
        u.id: sorted({g.service for g in u.service_grants}) for u in users
    }
    all_names = source_names(db)
    ungranted_sources = {
        u.id: [n for n in all_names if n not in set(grant_sources.get(u.id, []))]
        for u in users
    }
    auth = get_auth_settings(db)
    pool_window = int(getattr(auth, "pool_window_hours", 0) or 0)
    limit_summaries = {
        u.id: user_limits_summary(u, pool_window_hours=pool_window) for u in users
    }
    weight_status = catalog_weight_suggestions(db)
    pending = pending_invites(db)
    creator_ids = {i.created_by_id for i in pending}
    invite_creators: dict[int, str] = {}
    if creator_ids:
        invite_creators = {
            u.id: u.username
            for u in db.query(WebUser).filter(WebUser.id.in_(creator_ids)).all()
        }
    key_create_blocked: dict[int, str | None] = {
        u.id: assert_can_create_key(db, u) for u in users
    }
    user_ids = [u.id for u in users]
    user_keys: dict[int, list[ApiKey]] = {uid: [] for uid in user_ids}
    if user_ids and not _teams_on(db):
        for key in (
            db.query(ApiKey)
            .filter(ApiKey.owner_user_id.in_(user_ids))
            .order_by(ApiKey.created_at.desc())
            .all()
        ):
            if key.owner_user_id is not None:
                user_keys[key.owner_user_id].append(key)

    return templates.TemplateResponse(
        request,
        "users.html",
        {
            "user": user,
            "users": users,
            "nav": "users",
            "is_admin": True,
            "flash_ok": flash,
            "flash_err": err,
            "smtp_ok": smtp_ready(get_smtp(db)),
            "teams_enabled": _teams_on(db),
            "grant_labels": grant_labels,
            "grant_sources": grant_sources,
            "ungranted_sources": ungranted_sources,
            "limit_summaries": limit_summaries,
            "weight_status": weight_status,
            "pool_model_weights_enabled": bool(auth.pool_model_weights_enabled),
            "source_chips": source_chip_rows(db),
            "selected_services": display_default_sources(db),
            "catalog_groups": catalog_groups_for_ceiling(
                db, AccessCeiling(unrestricted=True)
            ),
            "selected_models": display_default_models(db),
            "pending_invites": pending,
            "invite_creators": invite_creators,
            "user_counts": {
                "total": len(users),
                "active": sum(1 for u in users if u.is_active),
                "admins": sum(1 for u in users if u.is_platform_admin),
                "must_change": sum(1 for u in users if u.must_change_password),
            },
            "active_key_counts": {u.id: active_key_count(db, u.id) for u in users},
            "user_keys": user_keys,
            "key_create_blocked": key_create_blocked,
            "display_names": {u.id: user_display_name(u) for u in users},
            "needs_username": {u.id: user_needs_username(u) for u in users},
        },
    )


def _load_grant_target(db: Session, user_id: int) -> WebUser | None:
    return (
        db.query(WebUser)
        .options(
            joinedload(WebUser.service_grants),
            joinedload(WebUser.model_allowlists),
        )
        .filter(WebUser.id == user_id)
        .first()
    )


@router.get("/users/{user_id}/grant/sources/partial", response_class=HTMLResponse)
def users_grant_sources_partial(
    user_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    """HTML fragment: pick ungranted sources to add under the user row."""
    if _teams_on(db):
        return HTMLResponse("Teams mode — edit grants on Teams.", status_code=400)
    target = _load_grant_target(db, user_id)
    if target is None:
        return HTMLResponse("User not found", status_code=404)
    if target.is_platform_admin:
        return HTMLResponse("Platform admins have full access.", status_code=400)
    granted = {g.service for g in target.service_grants}
    chips = [c for c in source_chip_rows(db) if c["name"] not in granted]
    return templates.TemplateResponse(
        request,
        "_user_grant_sources_expand.html",
        {"user": user, "target": target, "source_chips": chips},
    )


@router.post("/users/{user_id}/grant/sources/add")
async def users_grant_sources_add(
    user_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...data.grants import sync_user_grants

    if _teams_on(db):
        request.session["flash_err"] = "Teams are enabled — edit the Team grant instead."
        return RedirectResponse("/users", status_code=303)
    target = _load_grant_target(db, user_id)
    if not target or target.is_platform_admin:
        return RedirectResponse("/users", status_code=303)
    form = await request.form()
    names = set(source_names(db))
    granted = {g.service for g in target.service_grants}
    to_add = _parse_services(form.getlist("services"), names)
    if not to_add:
        request.session["flash_err"] = "Check at least one source to add."
        return RedirectResponse("/users", status_code=303)
    merged = sorted(granted | set(to_add))
    sync_user_grants(db, target, merged)
    write_audit(
        db,
        actor=user,
        action="user.grant.sources_add",
        entity_type="user",
        entity_id=target.id,
        detail=f"added={to_add} total={merged}",
    )
    db.commit()
    request.session["flash_ok"] = (
        f"Added {', '.join(to_add)} for {target.username}."
        if len(to_add) > 1
        else f"Added {to_add[0]} for {target.username}."
    )
    return RedirectResponse("/users", status_code=303)


@router.post("/users/{user_id}/grant/source/remove")
async def users_grant_source_remove(
    user_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
    source: str = Form(""),
):
    from ...data.grants import sync_user_grants
    from ...data.models import ModelAllowlist

    if _teams_on(db):
        request.session["flash_err"] = "Teams are enabled — edit the Team grant instead."
        return RedirectResponse("/users", status_code=303)
    target = _load_grant_target(db, user_id)
    if not target or target.is_platform_admin:
        return RedirectResponse("/users", status_code=303)
    name = source.strip()
    names = set(source_names(db))
    if not name or name not in names:
        request.session["flash_err"] = "Unknown source."
        return RedirectResponse("/users", status_code=303)
    granted = {g.service for g in target.service_grants}
    if name not in granted:
        request.session["flash_err"] = f"Source '{name}' was not granted."
        return RedirectResponse("/users", status_code=303)
    remaining = sorted(granted - {name})
    sync_user_grants(db, target, remaining)
    db.query(ModelAllowlist).filter(
        ModelAllowlist.user_id == target.id,
        ModelAllowlist.service == name,
    ).delete()
    write_audit(
        db,
        actor=user,
        action="user.grant.source_remove",
        entity_type="user",
        entity_id=target.id,
        detail=f"removed={name} remaining={remaining}",
    )
    db.commit()
    request.session["flash_ok"] = f"Removed {name} from {target.username}."
    return RedirectResponse("/users", status_code=303)


@router.get("/users/{user_id}/grant/limits/partial", response_class=HTMLResponse)
def users_grant_limits_partial(
    user_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    if _teams_on(db):
        return HTMLResponse("Teams mode — edit grants on Teams.", status_code=400)
    target = _load_grant_target(db, user_id)
    if target is None:
        return HTMLResponse("User not found", status_code=404)
    if target.is_platform_admin:
        return HTMLResponse("Platform admins have full access.", status_code=400)
    from ..accounts import get_auth_settings

    auth = get_auth_settings(db)
    return templates.TemplateResponse(
        request,
        "_user_grant_limits_expand.html",
        {
            "user": user,
            "target": target,
            "pool_window_hours": int(getattr(auth, "pool_window_hours", 0) or 0),
        },
    )


@router.post("/users/{user_id}/grant/limits")
async def users_grant_limits_save(
    user_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ..user_limits import apply_user_limits_from_form

    if _teams_on(db):
        request.session["flash_err"] = "Teams are enabled — edit the Team grant instead."
        return RedirectResponse("/users", status_code=303)
    target = _load_grant_target(db, user_id)
    if not target or target.is_platform_admin:
        return RedirectResponse("/users", status_code=303)
    form = await request.form()
    apply_user_limits_from_form(target, form)
    write_audit(
        db,
        actor=user,
        action="user.grant.limits",
        entity_type="user",
        entity_id=target.id,
        detail=(
            f"rpm={target.rpm_limit} conc={target.concurrency_limit} "
            f"daily={target.daily_quota} pool={target.pool_limit}"
        ),
    )
    db.commit()
    request.session["flash_ok"] = f"Limits saved for {target.username}."
    return RedirectResponse("/users", status_code=303)


@router.get("/users/{user_id}/grant/partial", response_class=HTMLResponse)
def users_grant_partial(
    user_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    """HTML fragment: models for one source, under that user row."""
    if _teams_on(db):
        return HTMLResponse("Teams mode — edit grants on Teams.", status_code=400)
    source = (request.query_params.get("source") or "").strip()
    names = set(source_names(db))
    if not source or source not in names:
        return HTMLResponse("Unknown source", status_code=400)
    target = (
        db.query(WebUser)
        .options(
            joinedload(WebUser.service_grants),
            joinedload(WebUser.model_allowlists),
        )
        .filter(WebUser.id == user_id)
        .first()
    )
    if target is None:
        return HTMLResponse("User not found", status_code=404)
    if target.is_platform_admin:
        return HTMLResponse("Platform admins have full access.", status_code=400)
    granted = {g.service for g in target.service_grants}
    if source not in granted:
        return HTMLResponse("Source not granted", status_code=400)
    models = [
        row
        for row in list_catalog(db)
        if row.enabled and row.source_name == source
    ]
    listed = {m.model_name for m in target.model_allowlists if m.service == source}
    if listed:
        selected = listed
    else:
        selected = {m.model_id for m in models}
    return templates.TemplateResponse(
        request,
        "_user_grant_expand.html",
        {
            "user": user,
            "target": target,
            "source": source,
            "models": models,
            "selected": selected,
        },
    )


@router.post("/users/{user_id}/grant/source")
async def users_grant_source_save(
    user_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...data.grants import sync_user_models_for_service

    if _teams_on(db):
        request.session["flash_err"] = "Teams are enabled — edit the Team grant instead."
        return RedirectResponse("/users", status_code=303)
    target = (
        db.query(WebUser)
        .options(joinedload(WebUser.service_grants))
        .filter(WebUser.id == user_id)
        .first()
    )
    if not target or target.is_platform_admin:
        return RedirectResponse("/users", status_code=303)
    form = await request.form()
    source = str(form.get("source") or "").strip()
    names = set(source_names(db))
    granted = {g.service for g in target.service_grants}
    if source not in names or source not in granted:
        request.session["flash_err"] = "Unknown or ungranted source."
        return RedirectResponse("/users", status_code=303)
    catalog_ids = [
        row.model_id
        for row in list_catalog(db)
        if row.enabled and row.source_name == source
    ]
    picked = [m for s, m in _parse_model_checks(form.getlist("models")) if s == source]
    if not picked or (catalog_ids and set(picked) >= set(catalog_ids)):
        model_names = None
    else:
        model_names = picked
    sync_user_models_for_service(db, target, source, model_names)
    write_audit(
        db,
        actor=user,
        action="user.grant.source",
        entity_type="user",
        entity_id=target.id,
        detail=f"source={source} models={model_names if model_names is not None else 'all'}",
    )
    db.commit()
    request.session["flash_ok"] = f"Saved {source} models for {target.username}."
    return RedirectResponse("/users", status_code=303)


def _welcome_mail_body(
    *,
    target: WebUser,
    email: str,
    password: str,
    login_url: str,
    keys_url: str,
    api_base: str,
    must_change_password: bool,
    personal_note: str = "",
) -> str:
    """Industry-style welcome: login + docs, no API key secret."""
    from ..accounts import user_needs_username

    greeting = (
        "Hi,\n\n"
        if user_needs_username(target)
        else f"Hi {target.username},\n\n"
    )
    if must_change_password:
        setup = (
            "On first login you must set a new password.\n"
            "Username is optional — you can set one later in Profile (email stays your login).\n\n"
        )
    elif user_needs_username(target):
        setup = (
            "Username is optional — you can set one later in Profile (email stays your login).\n\n"
        )
    else:
        setup = ""

    if target.is_platform_admin:
        access = "Access: platform admin (full access).\n"
    else:
        sources = sorted({g.service for g in target.service_grants})
        if sources:
            access = "Enabled sources: " + ", ".join(sources) + ".\n"
            models = sorted(
                {f"{m.service}/{m.model_name}" for m in target.model_allowlists}
            )
            if models:
                shown = models[:20]
                access += "Model allowlist: " + ", ".join(shown)
                if len(models) > 20:
                    access += f" (+{len(models) - 20} more)"
                access += ".\n"
            else:
                access += "Models: all enabled models for those sources (see GET /v1/models).\n"
        else:
            access = "Access: no sources granted yet — ask an admin if calls fail.\n"

    note = (personal_note or "").strip()
    note_block = f"Message from your admin:\n{note}\n\n" if note else ""

    return (
        f"{greeting}"
        f"{note_block}"
        "An account was created for you on OnPrem AI Gateway.\n\n"
        f"Login:\n{login_url}\n\n"
        f"Email: {email}\n"
        f"Temporary password: {password}\n\n"
        f"{setup}"
        "API (OpenAI-compatible) — all sources are merged under one base URL:\n"
        f"  Base URL: {api_base}\n"
        "  Auth header: Authorization: Bearer YOUR_API_KEY\n"
        "  (or: X-Api-Key: YOUR_API_KEY)\n\n"
        f"{access}\n"
        "API key:\n"
        "  After login, open API Keys and create a key (shown once), or your admin will send one separately.\n"
        f"  Keys page: {keys_url}\n"
        "  Never share your key; rotate it if it leaks.\n"
    )


@router.post("/users/new")
async def users_create(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...config import public_api_base
    from ...data.grants import sync_user_grants, sync_user_models
    from ...mailer import MailError, get_smtp, send_mail, smtp_ready
    from ..accounts import (
        pending_username_for_id,
        user_display_name,
        validate_username,
    )

    form = await request.form()
    username_raw = str(form.get("username") or "").strip()
    password = str(form.get("password") or "")
    email_n = str(form.get("email") or "").strip().lower() or None
    is_platform_admin = form.get("is_platform_admin") == "on"
    must_change_password = form.get("must_change_password") == "on"
    send_welcome = form.get("send_welcome_email") == "on"
    welcome_note = str(form.get("welcome_note") or "").strip()[:2000]
    if not email_n:
        request.session["flash_err"] = "Email is required (login + password reset)."
        return RedirectResponse("/users", status_code=303)
    if not password:
        request.session["flash_err"] = "Password is required."
        return RedirectResponse("/users", status_code=303)
    if len(password) < 8:
        request.session["flash_err"] = "Password min 8 characters."
        return RedirectResponse("/users", status_code=303)
    if db.query(WebUser).filter(WebUser.email == email_n).first():
        request.session["flash_err"] = "Email already exists."
        return RedirectResponse("/users", status_code=303)

    claim_later = not username_raw
    if username_raw:
        err = validate_username(username_raw)
        if err:
            request.session["flash_err"] = err
            return RedirectResponse("/users", status_code=303)
        if db.query(WebUser).filter(WebUser.username == username_raw).first():
            request.session["flash_err"] = "Username already exists."
            return RedirectResponse("/users", status_code=303)
    else:
        # Temporary unique handle until first login (email is the login).
        must_change_password = True

    # Placeholder until flush assigns id (replaced immediately below when claim_later).
    bootstrap_username = username_raw or f"pending-tmp-{secrets.token_hex(8)}"
    target = WebUser(
        username=bootstrap_username,
        email=email_n,
        password_hash=hash_password(password),
        is_active=True,
        is_platform_admin=is_platform_admin,
        must_change_password=must_change_password,
    )
    db.add(target)
    db.flush()
    if claim_later:
        target.username = pending_username_for_id(target.id)
    if not target.is_platform_admin and not _teams_on(db):
        names = source_names(db)
        services = _parse_services(form.getlist("services"), names)
        if not services:
            services = configured_default_sources(db)
        sync_user_grants(db, target, services)
        models = [(s, m) for s, m in _collect_models_from_form(form, db) if s in services]
        if models:
            sync_user_models(db, target, models)
    write_audit(
        db,
        actor=user,
        action="user.create",
        entity_type="user",
        detail=user_display_name(target),
    )

    mailed = False
    mail_err: str | None = None
    if send_welcome:
        cfg = get_smtp(db)
        if not smtp_ready(cfg):
            mail_err = "SMTP is not ready — account created, welcome mail skipped."
        else:
            assert cfg is not None
            base = (cfg.public_base_url or str(request.base_url)).rstrip("/")
            api_base = public_api_base()
            if api_base.startswith("http://localhost") and base.startswith("https://"):
                api_base = f"{base}/v1"
            try:
                send_mail(
                    db,
                    to_email=email_n,
                    subject="Your OnPrem AI Gateway account",
                    body_text=_welcome_mail_body(
                        target=target,
                        email=email_n,
                        password=password,
                        login_url=f"{base}/login",
                        keys_url=f"{base}/keys",
                        api_base=api_base,
                        must_change_password=must_change_password,
                        personal_note=welcome_note,
                    ),
                )
                mailed = True
                write_audit(
                    db,
                    actor=user,
                    action="user.welcome_mail",
                    entity_type="user",
                    entity_id=target.id,
                    detail=email_n,
                )
            except MailError as exc:
                mail_err = str(exc)

    db.commit()

    if target.is_platform_admin:
        msg = f"Admin {user_display_name(target)} created (full access)."
    elif claim_later:
        msg = (
            f"Created {email_n} — they log in with email"
            + (" and must set a new password." if must_change_password else ".")
        )
    else:
        n = len(target.service_grants)
        msg = (
            f"User {target.username} created with {n} source(s). "
            f"Fine-tune models under Edit grant if needed."
        )
    if mailed:
        msg += f" Welcome email sent to {email_n}."
    elif mail_err:
        request.session["flash_err"] = mail_err
    request.session["flash_ok"] = msg
    return RedirectResponse("/users", status_code=303)


@router.post("/users/{user_id}/toggle")
def users_toggle(
    user_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    target = db.get(WebUser, user_id)
    if target and target.id != user.id:
        target.is_active = not target.is_active
        write_audit(
            db,
            actor=user,
            action="user.toggle",
            entity_type="user",
            entity_id=target.id,
            detail=f"active={target.is_active}",
        )
        db.commit()
    return RedirectResponse("/users", status_code=303)


@router.post("/users/{user_id}/update")
async def users_update(
    user_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    target = db.get(WebUser, user_id)
    if not target:
        return RedirectResponse("/users", status_code=303)
    form = await request.form()
    email_n = str(form.get("email") or "").strip().lower() or None
    if email_n:
        other = (
            db.query(WebUser)
            .filter(WebUser.email == email_n, WebUser.id != target.id)
            .first()
        )
        if other:
            request.session["flash_err"] = "Email already in use."
            return RedirectResponse("/users", status_code=303)
    target.email = email_n
    if target.id != user.id:
        target.is_platform_admin = form.get("is_platform_admin") == "on"
    new_pw = str(form.get("new_password") or "")
    if new_pw:
        if len(new_pw) < 8:
            request.session["flash_err"] = "Password min 8 characters."
            return RedirectResponse("/users", status_code=303)
        target.password_hash = hash_password(new_pw)
        target.must_change_password = form.get("must_change_password") == "on"
    elif form.get("must_change_password") == "on":
        target.must_change_password = True
    write_audit(
        db,
        actor=user,
        action="user.update",
        entity_type="user",
        entity_id=target.id,
        detail=f"email={email_n or '-'}",
    )
    db.commit()
    request.session["flash_ok"] = f"Updated {target.username}."
    return RedirectResponse("/users", status_code=303)


@router.post("/users/{user_id}/send-reset")
def users_send_reset(
    user_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ..accounts import create_reset_token
    from ...mailer import MailError, get_smtp, send_mail, smtp_ready

    target = db.get(WebUser, user_id)
    if not target or not target.email:
        request.session["flash_err"] = "User needs an email address."
        return RedirectResponse("/users", status_code=303)
    cfg = get_smtp(db)
    if not smtp_ready(cfg):
        request.session["flash_err"] = "Configure SMTP first (/smtp)."
        return RedirectResponse("/users", status_code=303)
    try:
        raw = create_reset_token(db, target, by_platform_admin=True)
        assert cfg is not None
        link = f"{cfg.public_base_url.rstrip('/')}/reset?token={raw}"
        send_mail(
            db,
            to_email=target.email,
            subject="Password reset — OnPrem AI Gateway",
            body_text=(
                f"Hi {target.username},\n\n"
                f"An admin requested a password reset. Link (1 hour):\n{link}\n"
            ),
        )
        write_audit(
            db,
            actor=user,
            action="user.send_reset",
            entity_type="user",
            entity_id=target.id,
        )
        db.commit()
        request.session["flash_ok"] = f"Reset mail sent to {target.email}."
    except MailError as exc:
        request.session["flash_err"] = str(exc)
    return RedirectResponse("/users", status_code=303)


@router.post("/users/invites")
def users_invite_create(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
    email: str = Form(""),
    note: str = Form(""),
):
    from ..invites import create_registration_invite, invite_register_url
    from ...mailer import MailError, get_smtp, send_mail, smtp_ready

    raw = create_registration_invite(db, created_by=user, note=note)
    cfg = get_smtp(db)
    base = str(request.base_url).rstrip("/")
    if cfg and cfg.public_base_url:
        base = cfg.public_base_url.rstrip("/")
    link = invite_register_url(base, raw)
    email_n = email.strip().lower()
    mailed = False
    if email_n and smtp_ready(cfg):
        try:
            assert cfg is not None
            send_mail(
                db,
                to_email=email_n,
                subject="You're invited — OnPrem AI Gateway",
                body_text=(
                    "You've been invited to create an account on OnPrem AI Gateway.\n\n"
                    f"One-time signup link (7 days):\n{link}\n\n"
                    "After signup you'll get the default access configured in Settings."
                ),
            )
            mailed = True
        except MailError as exc:
            request.session["flash_err"] = str(exc)
    write_audit(
        db,
        actor=user,
        action="user.invite_create",
        entity_type="registration_invite",
        detail=email_n or note or "link",
    )
    db.commit()
    if mailed:
        request.session["flash_ok"] = f"Invite sent to {email_n}."
    else:
        request.session["flash_ok"] = f"Invite link (7 days, one use): {link}"
    return RedirectResponse("/users", status_code=303)


@router.post("/users/invites/{invite_id}/revoke")
def users_invite_revoke(
    invite_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...data.models import RegistrationInvite

    inv = db.get(RegistrationInvite, invite_id)
    if inv is None or inv.used_at is not None:
        request.session["flash_err"] = "Invite not found or already used."
        return RedirectResponse("/users", status_code=303)
    write_audit(
        db,
        actor=user,
        action="user.invite_revoke",
        entity_type="registration_invite",
        entity_id=inv.id,
    )
    db.delete(inv)
    db.commit()
    request.session["flash_ok"] = "Invite revoked."
    return RedirectResponse("/users", status_code=303)
