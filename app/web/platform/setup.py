from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from ...audit import write_audit
from ...config import KINDS, onprem_api_port, public_api_base
from ...data.backends import get_source_by_name, source_chip_rows, source_names
from ...data.db import generate_api_key, get_db, hash_api_key, key_display_prefix
from ...data.models import WebUser, ApiKey
from ..session import require_platform_admin
from ..shared import (
    templates,
    _collect_models_from_form,
    _parse_services,
    _sync_key_grants,
    _sync_key_models,
)

router = APIRouter()


def _wizard_or_next(db: Session, user: WebUser):
    from ..setup import needs_setup_wizard, wizard_progress

    if not needs_setup_wizard(db, user):
        return None
    nxt = wizard_progress(db)["next"]
    return RedirectResponse((nxt["path"] if nxt else "/setup/done"), status_code=303)


# ---- First-run setup wizard ----


def _wizard_sync_catalog(db: Session, request: Request, user: WebUser) -> None:
    from ...data.catalog import sync_catalog_from_sources

    stats = sync_catalog_from_sources(db)
    write_audit(
        db,
        actor=user,
        action="setup.catalog.sync",
        entity_type="catalog_models",
        detail=str(stats),
    )
    db.commit()
    seen = int(stats.get("seen") or 0)
    if seen == 0:
        request.session["flash_err"] = (
            "Sync found 0 models — check source addresses. Retry later under Models."
        )
    else:
        created = int(stats.get("created") or 0)
        request.session["flash_ok"] = f"Synced {seen} models ({created} new)."


def _ensure_catalog_synced(db: Session, request: Request, user: WebUser) -> None:
    """Backends seeded from .env skip Step 1 Save — sync before access/key pickers."""
    from ..setup import wizard_progress

    wiz = wizard_progress(db)
    if wiz["has_sources"] and not wiz["has_models"]:
        _wizard_sync_catalog(db, request, user)


@router.get("/setup", response_class=HTMLResponse)
def setup_root(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ..setup import wizard_progress

    wiz = wizard_progress(db)
    if wiz["complete"]:
        return RedirectResponse("/setup/done", status_code=303)
    return RedirectResponse(wiz["next"]["path"], status_code=303)


@router.get("/setup/sources", response_class=HTMLResponse)
def setup_sources_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...config import KINDS
    from ..setup import wizard_progress

    wiz = wizard_progress(db)
    return templates.TemplateResponse(
        request,
        "setup_sources.html",
        {
            "user": user,
            "nav": "setup",
            "wizard": wiz,
            "step_id": "sources",
            "step_title": "Step 1 · Backends",
            "step_lede": "Point OnPrem AI Gateway at your local chat / embed / extractor / STT / TTS servers.",
            "kinds": KINDS,
            "flash_ok": request.session.pop("flash_ok", None),
            "flash_err": request.session.pop("flash_err", None),
            "is_admin": True,
        },
    )


@router.post("/setup/sources")
def setup_sources_save(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
    name: str = Form(""),
    kind: str = Form("chat"),
    address: str = Form(""),
):
    from ...data.backends import (
        upsert_source,
        validate_backend,
        validate_kind,
        validate_source_name,
    )

    err = validate_source_name(name) or validate_kind(kind) or validate_backend(address)
    if err:
        request.session["flash_err"] = err
        return RedirectResponse("/setup/sources", status_code=303)
    existing = get_source_by_name(db, name.strip().lower())
    if existing:
        request.session["flash_err"] = f"Source '{name}' already exists — change the name or edit under Services."
        return RedirectResponse("/setup/sources", status_code=303)

    src = upsert_source(
        db,
        name=name,
        kind=kind,
        address=address,
        route_models="",
        api_style="auto",
    )
    write_audit(
        db,
        actor=user,
        action="setup.source",
        entity_type="backend_source",
        entity_id=src.id,
        detail=f"{src.name}@{src.address}",
    )
    db.commit()
    request.session["flash_ok"] = f"Added {src.name} → {src.address}. Add more or continue."
    return RedirectResponse("/setup/sources", status_code=303)


@router.post("/setup/sources/save-all")
async def setup_sources_save_all(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...data.backends import apply_source_row_edits, list_sources
    from ...data.models import BackendSource

    form = await request.form()
    sources = [s for s in list_sources(db) if (s.address or "").strip()]
    edits: list[tuple[BackendSource, str, str]] = []
    for src in sources:
        name = str(form.get(f"name_{src.id}") or src.name)
        address = str(form.get(f"address_{src.id}") or src.address)
        edits.append((src, name, address))
    err = apply_source_row_edits(db, edits)
    if err:
        request.session["flash_err"] = err
        return RedirectResponse("/setup/sources", status_code=303)
    write_audit(
        db,
        actor=user,
        action="setup.sources.save",
        entity_type="backend_source",
        detail=f"n={len(edits)}",
    )
    db.commit()
    _wizard_sync_catalog(db, request, user)
    return RedirectResponse("/setup/access", status_code=303)


@router.post("/setup/sources/{source_id}/delete")
def setup_sources_delete(
    source_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...data.backends import delete_source
    from ...data.models import BackendSource

    src = db.get(BackendSource, source_id)
    if src is None:
        request.session["flash_err"] = "Source not found"
        return RedirectResponse("/setup/sources", status_code=303)
    name = src.name
    delete_source(db, src)
    write_audit(
        db,
        actor=user,
        action="source.delete",
        entity_type="backend_source",
        entity_id=source_id,
        detail=name,
    )
    db.commit()
    request.session["flash_ok"] = f"Source '{name}' deleted."
    return RedirectResponse("/setup/sources", status_code=303)


@router.get("/setup/models", response_class=HTMLResponse)
def setup_models_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    """Old wizard step — catalog now syncs from Backends → Default access."""
    from ..setup import wizard_progress

    if not wizard_progress(db)["has_sources"]:
        return RedirectResponse("/setup/sources", status_code=303)
    _wizard_sync_catalog(db, request, user)
    return RedirectResponse("/setup/access", status_code=303)


@router.post("/setup/models")
def setup_models_sync(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    return setup_models_page(request, db, user)


@router.get("/setup/access", response_class=HTMLResponse)
def setup_access_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...data.grants import (
        AccessCeiling,
        catalog_groups_for_ceiling,
        display_default_models,
        display_default_sources,
        display_enabled_models_for_services,
    )
    from ..setup import wizard_progress

    wiz = wizard_progress(db)
    if not wiz["has_sources"]:
        return RedirectResponse("/setup/sources", status_code=303)
    _ensure_catalog_synced(db, request, user)
    wiz = wizard_progress(db)
    ceil = AccessCeiling(unrestricted=True, label="setup")
    selected_services = display_default_sources(db)
    catalog_groups = catalog_groups_for_ceiling(db, ceil)
    selected_models = display_default_models(db)
    if catalog_groups and selected_services and not selected_models:
        selected_models = display_enabled_models_for_services(db, selected_services)
    return templates.TemplateResponse(
        request,
        "setup_access.html",
        {
            "user": user,
            "nav": "setup",
            "wizard": wiz,
            "step_id": "access",
            "step_title": "Step 2 · Default access",
            "step_lede": "What new (non-admin) users get automatically. Everything starts checked — uncheck what they should not get.",
            "source_chips": source_chip_rows(db),
            "selected_services": selected_services,
            "catalog_groups": catalog_groups,
            "selected_models": selected_models,
            "flash_ok": request.session.pop("flash_ok", None),
            "flash_err": request.session.pop("flash_err", None),
            "is_admin": True,
        },
    )


@router.post("/setup/access")
async def setup_access_save(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...data.grants import save_default_grant
    from ..setup import wizard_progress

    wiz = wizard_progress(db)
    if not wiz["has_sources"]:
        return RedirectResponse("/setup/sources", status_code=303)
    form = await request.form()
    names = source_names(db)
    services = _parse_services(form.getlist("services"), names)
    models = [(s, m) for s, m in _collect_models_from_form(form, db) if s in services]
    save_default_grant(db, services, models)
    write_audit(
        db,
        actor=user,
        action="setup.default_grant",
        entity_type="auth_settings",
        detail=f"services={services} models={len(models)}",
    )
    db.commit()
    if services:
        request.session["flash_ok"] = "Default access saved for new users."
    else:
        request.session["flash_ok"] = "Saved: new users get no sources until you grant them."
    return RedirectResponse("/setup/key", status_code=303)


@router.get("/setup/key", response_class=HTMLResponse)
def setup_key_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    import os

    from ...config import onprem_api_port, public_api_base
    from ...data.grants import (
        AccessCeiling,
        catalog_groups_for_ceiling,
        display_enabled_models_for_services,
    )
    from ...vision_route import group_models_vl_pairs
    from ..accounts import get_auth_settings
    from ..setup import wizard_progress

    wiz = wizard_progress(db)
    if not wiz["has_sources"]:
        return RedirectResponse("/setup/sources", status_code=303)
    _ensure_catalog_synced(db, request, user)
    wiz = wizard_progress(db)
    created = request.session.pop("flash_key", None)
    created_summary = request.session.pop("flash_key_services", None)
    created_models_n = request.session.pop("flash_key_models_n", None)
    ceil = AccessCeiling(unrestricted=True, label="setup")
    auth = get_auth_settings(db)
    catalog_groups = catalog_groups_for_ceiling(db, ceil)
    catalog_vl_groups = (
        [(src, group_models_vl_pairs(models)) for src, models in catalog_groups]
        if auth.auto_vl_routing
        else None
    )
    service_names = [s.name for s in wiz["sources"]] if wiz.get("sources") else []
    gw_port = onprem_api_port()
    return templates.TemplateResponse(
        request,
        "setup_key.html",
        {
            "user": user,
            "nav": "setup",
            "wizard": wiz,
            "step_id": "key",
            "step_title": "Step 3 · Your API key",
            "step_lede": "One admin key to verify OnPrem AI Gateway works. Uncheck only what this key should not use.",
            "created_key": created,
            "created_summary": created_summary,
            "created_models_n": created_models_n,
            "sources": wiz["sources"],
            "source_chips": source_chip_rows(db),
            "selected_services": service_names,
            "selected_models": display_enabled_models_for_services(db, service_names),
            "catalog_groups": catalog_groups,
            "catalog_vl_groups": catalog_vl_groups,
            "api_port": gw_port,
            "api_base": public_api_base(api_port=gw_port),
            "flash_ok": request.session.pop("flash_ok", None),
            "flash_err": request.session.pop("flash_err", None),
            "is_admin": True,
        },
    )


@router.post("/setup/key")
async def setup_key_create(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...data.grants import AccessCeiling, clamp_models, clamp_services, normalize_model_allowlist
    from ..setup import wizard_progress

    wiz = wizard_progress(db)
    if not wiz["has_sources"]:
        return RedirectResponse("/setup/sources", status_code=303)

    form = await request.form()
    label = str(form.get("label") or "").strip() or "main"
    names = source_names(db)
    ceil = AccessCeiling(unrestricted=True, label="setup")
    services = clamp_services(
        _parse_services(form.getlist("services"), names), ceil, db
    )
    if not services:
        request.session["flash_err"] = "Check at least one source for this key."
        return RedirectResponse("/setup/key", status_code=303)

    models = clamp_models(_collect_models_from_form(form, db), ceil)
    models = [(s, m) for s, m in models if s in services]
    models = normalize_model_allowlist(db, services, models)

    raw = generate_api_key()
    api_key = ApiKey(
        label=label,
        key_hash=hash_api_key(raw),
        key_prefix=key_display_prefix(raw),
        owner_user_id=user.id,
        is_active=True,
        priority=0,
    )
    db.add(api_key)
    db.flush()
    _sync_key_grants(db, api_key, services)
    _sync_key_models(db, api_key, models)
    write_audit(
        db,
        actor=user,
        action="setup.key",
        entity_type="api_key",
        entity_id=api_key.id,
        detail=f"{api_key.label} services={services} models={len(models)}",
    )
    db.commit()
    request.session["flash_key"] = raw
    request.session["flash_key_services"] = ", ".join(services)
    request.session["flash_key_models_n"] = len(models) if models else 0
    return RedirectResponse("/setup/key", status_code=303)


@router.get("/setup/done", response_class=HTMLResponse)
def setup_done_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ..setup import wizard_progress

    if not wizard_progress(db)["complete"]:
        return RedirectResponse("/setup", status_code=303)
    return templates.TemplateResponse(
        request,
        "setup_done.html",
        {"user": user, "nav": "setup", "is_admin": True},
    )
