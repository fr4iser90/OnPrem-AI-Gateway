from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from ...audit import write_audit
from ...config import API_STYLES, KINDS, dialect_choices
from ...data.backends import get_source_by_name, source_chip_rows
from ...data.dialects import dialect_blurb_for_kind
from ...data.db import get_db
from ...data.models import WebUser
from ..session import require_platform_admin
from ..shared import templates, _settings

router = APIRouter()


@router.get("/services", response_class=HTMLResponse)
def services_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...data.backends import catalog_route_models, hardware_labels, source_rows
    from ...data.capabilities import engine_choices
    from ...data.probe import probe_all
    from ...data.source_capacity import format_slots_label

    settings = _settings(request)
    rows = source_rows(db, settings)
    statuses = probe_all(db)
    catalog_by_source = {src.name: catalog_route_models(db, src.name) for src, _ in rows}
    engine_by = {s.service: s.engine for s in statuses}
    dialect_blurbs = {
        src.name: dialect_blurb_for_kind(src.kind, src.api_style) for src, _ in rows
    }
    flash_ok = request.session.pop("flash_ok", None)
    flash_err = request.session.pop("flash_err", None)
    return templates.TemplateResponse(
        request,
        "services.html",
        {
            "user": user,
            "rows": rows,
            "statuses": statuses,
            "catalog_by_source": catalog_by_source,
            "hardware_by_name": hardware_labels(db),
            "engine_by": engine_by,
            "kinds": KINDS,
            "api_styles": dialect_choices(),
            "engine_choices": engine_choices(),
            "dialect_blurbs": dialect_blurbs,
            "domain": settings.domain,
            "nav": "services",
            "flash_ok": flash_ok,
            "flash_err": flash_err,
            "can_edit": True,
            "format_slots_label": format_slots_label,
        },
    )


@router.get("/services/status")
def services_status_json(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...data.probe import probe_all

    return {"services": [s.to_dict() for s in probe_all(db)]}


@router.post("/services")
def services_create(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
    name: str = Form(""),
    kind: str = Form("chat"),
    address: str = Form(""),
    api_style: str = Form("auto"),
    engine_override: str = Form(""),
    hardware: str = Form(""),
    temp_guard_enabled: str | None = Form(None),
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
        return RedirectResponse("/services", status_code=303)
    if get_source_by_name(db, name.strip().lower()):
        request.session["flash_err"] = f"Source '{name}' already exists — edit address below"
        return RedirectResponse("/services", status_code=303)

    src = upsert_source(
        db,
        name=name,
        kind=kind,
        address=address,
        route_models="",
        api_style=api_style,
        temp_guard_enabled=temp_guard_enabled is not None,
        hardware=hardware,
    )
    write_audit(
        db,
        actor=user,
        action="source.create",
        entity_type="backend_source",
        entity_id=src.id,
        detail=f"{src.name} kind={src.kind} style={src.api_style}",
    )
    db.commit()
    request.session["flash_ok"] = f"Source '{src.name}' added."
    return RedirectResponse("/services", status_code=303)


@router.post("/services/{source_id}/save")
def services_update(
    source_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
    address: str = Form(""),
    api_style: str = Form("auto"),
    engine_override: str = Form(""),
    max_concurrency: str = Form(""),
    queue_timeout_sec: str = Form(""),
    name: str = Form(""),
    hardware: str = Form(""),
    temp_guard_enabled: str | None = Form(None),
):
    from ...data.backends import (
        normalize_backend,
        normalize_hardware_label,
        rename_source,
        validate_backend,
    )
    from ...data.capabilities import ENGINE_PROFILES
    from ...data.models import BackendSource

    src = db.get(BackendSource, source_id)
    if src is None:
        request.session["flash_err"] = "Source not found"
        return RedirectResponse("/services", status_code=303)
    err = validate_backend(address)
    if err:
        request.session["flash_err"] = f"{src.name}: {err}"
        return RedirectResponse("/services", status_code=303)

    style = (api_style or "auto").strip().lower()
    if style not in API_STYLES:
        style = "auto"
    src.address = normalize_backend(address)
    src.gpu_power_url = ""  # always derived from address host:9105
    src.temp_guard_enabled = temp_guard_enabled is not None
    src.route_models = ""  # merge targets come from catalog sync only
    src.api_style = style
    override = (engine_override or "").strip()
    src.engine_override = override if override in ENGINE_PROFILES else ""

    def _opt_int_field(raw: str) -> int | None:
        s = (raw or "").strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            return None

    mc = _opt_int_field(max_concurrency)
    src.max_concurrency = mc if mc is not None else None
    qt = _opt_int_field(queue_timeout_sec)
    src.queue_timeout_sec = qt if qt and qt > 0 else None
    src.hardware = normalize_hardware_label(hardware)
    err = rename_source(db, src, name or src.name)
    if err:
        request.session["flash_err"] = err
        return RedirectResponse("/services", status_code=303)
    write_audit(
        db,
        actor=user,
        action="source.update",
        entity_type="backend_source",
        entity_id=src.id,
        detail=(
            f"{src.name} addr={'set' if src.address else 'empty'} "
            f"style={src.api_style}"
        ),
    )
    db.commit()
    request.session["flash_ok"] = f"Source '{src.name}' saved."
    return RedirectResponse("/services", status_code=303)


@router.post("/services/{source_id}/delete")
def services_delete(
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
        return RedirectResponse("/services", status_code=303)
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
    return RedirectResponse("/services", status_code=303)
