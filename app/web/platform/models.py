from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from sqlalchemy.orm import Session, joinedload

from ...audit import write_audit
from ...data.db import get_db
from ...data.models import ModelAlias, WebUser
from ..session import require_platform_admin
from ..shared import templates, _gpu_power_enabled

router = APIRouter()


def _alias_card_context(db: Session, row: ModelAlias, *, open: bool = False) -> dict:
    from ...model_aliases import (
        candidate_model_ids,
        catalog_ids_for_picker,
        preferred_sources_for_model,
    )

    return {
        "a": row,
        "cands": candidate_model_ids(row),
        "catalog_model_ids": catalog_ids_for_picker(db),
        "source_names": preferred_sources_for_model(db, row.target_model_id),
        "open": open,
    }


def _render_alias_card(request: Request, db: Session, row: ModelAlias, *, open: bool = False) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/_alias_card.html",
        _alias_card_context(db, row, open=open),
    )


@router.get("/models", response_class=HTMLResponse)
def models_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...data.catalog import (
        TAG_SUGGESTIONS,
        catalog_grouped_by_kind,
        format_bytes,
        format_param_count,
        list_catalog,
        split_sync_stale_pairs,
        suggest_docs_url,
    )
    from ...data.source_capacity import capacities_by_source, format_slots_label
    from ...data.usage_weights import catalog_weight_suggestions
    from ...model_aliases import (
        candidate_model_ids,
        catalog_ids_for_picker,
        alias_picker_meta,
        list_aliases,
        preferred_sources_for_model,
    )
    from ...stats import model_perf_averages, model_perf_by_id
    from ...vision_route import group_kind_rows_vl_pairs
    from ..accounts import get_auth_settings

    flash_ok = request.session.pop("flash_ok", None)
    flash_err = request.session.pop("flash_err", None)
    # Loaded badges: last Sync (no full discover on every page load).
    # Slot labels: short /slots?model= probes (≤1.5s) — llama-router requires a model id.
    rows = list_catalog(db)
    auth = get_auth_settings(db)
    auto_vl = bool(auth.auto_vl_routing)
    pool_weights = bool(auth.pool_model_weights_enabled)
    weight_status = catalog_weight_suggestions(db)
    observed = model_perf_by_id(model_perf_averages(db, key_ids=None, lookback_days=7))
    cap_source_names = {r.source_name for r in rows if (r.source_name or "").strip()}
    capacity = capacities_by_source(db, cap_source_names, probe=True)
    groups = []
    for kind, kind_rows in catalog_grouped_by_kind(rows):
        pairs = group_kind_rows_vl_pairs(
            kind_rows,
            pair=auto_vl and kind == "chat",
        )
        active, stale = split_sync_stale_pairs(pairs)
        groups.append((kind, active, stale, len(kind_rows)))
    aliases = list_aliases(db)
    picker_meta = alias_picker_meta(db)
    catalog_model_ids = catalog_ids_for_picker(db)
    alias_candidates = {a.id: candidate_model_ids(a) for a in aliases}
    alias_pref_sources = {
        a.id: preferred_sources_for_model(db, a.target_model_id) for a in aliases
    }

    return templates.TemplateResponse(
        request,
        "models.html",
        {
            "user": user,
            "nav": "models",
            "rows": rows,
            "groups": groups,
            "observed": observed,
            "capacity": capacity,
            "auto_vl_routing": auto_vl,
            "pool_model_weights_enabled": pool_weights,
            "weight_status": weight_status,
            "suggest_docs_url": suggest_docs_url,
            "format_param_count": format_param_count,
            "format_bytes": format_bytes,
            "format_slots_label": format_slots_label,
            "tag_suggestions": TAG_SUGGESTIONS,
            "aliases": aliases,
            "alias_candidates": alias_candidates,
            "alias_pref_sources": alias_pref_sources,
            "catalog_model_ids": catalog_model_ids,
            "alias_picker_meta": picker_meta,
            "source_names": [],  # add-form: filled by JS from Active model
            "flash_ok": flash_ok,
            "flash_err": flash_err,
            "gpu_power_enabled": _gpu_power_enabled(request, db),
        },
    )


@router.post("/models/sync")
def models_sync(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...data.catalog import sync_catalog_from_sources

    stats = sync_catalog_from_sources(db)
    write_audit(
        db,
        actor=user,
        action="catalog.sync",
        entity_type="catalog_models",
        detail=str(stats),
    )
    db.commit()
    pruned = int(stats.get("pruned") or 0)
    msg = (
        f"Synced: {stats['seen']} models from {stats['sources']} sources "
        f"({stats['created']} new, {stats.get('tagged', 0)} auto-tagged, "
        f"{stats.get('meta', 0)} with meta"
    )
    if pruned:
        msg += f", {pruned} disabled (not on upstream)"
    msg += ")."
    request.session["flash_ok"] = msg
    return RedirectResponse("/models", status_code=303)


@router.post("/models/apply-suggested-weights")
def models_apply_suggested_weights(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...data.usage_weights import apply_weight_suggestions, catalog_weight_suggestions
    from ..accounts import get_auth_settings

    auth = get_auth_settings(db)
    if not auth.pool_model_weights_enabled:
        request.session["flash_err"] = (
            "Enable per-model budget factors in Settings first — suggestions never apply automatically."
        )
        return RedirectResponse("/models", status_code=303)
    status = catalog_weight_suggestions(db)
    if not status.ready:
        request.session["flash_err"] = status.message
        return RedirectResponse("/models", status_code=303)
    n = apply_weight_suggestions(db, status.suggestions)
    write_audit(
        db,
        actor=user,
        action="catalog.suggest_weights",
        entity_type="catalog_models",
        detail=f"updated={n} baseline_tg={status.baseline_tg_tok_s}",
    )
    db.commit()
    if n:
        request.session["flash_ok"] = (
            f"Updated {n} budget factor(s) from 7d usage "
            f"(baseline TG ~{status.baseline_tg_tok_s} tok/s)."
        )
    else:
        request.session["flash_ok"] = "Weights already match suggestions — nothing changed."
    return RedirectResponse("/models", status_code=303)


@router.post("/models/save")
async def models_save(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...data.catalog import list_catalog, update_catalog_meta
    from ..accounts import get_auth_settings

    def _parse_weight(raw) -> float:
        try:
            return max(0.01, float(raw or 1))
        except (TypeError, ValueError):
            return 1.0

    form = await request.form()
    enabled_ids = {int(x) for x in form.getlist("enabled") if str(x).isdigit()}
    weights_on = bool(get_auth_settings(db).pool_model_weights_enabled)
    changed = 0
    for row in list_catalog(db):
        want = row.id in enabled_ids
        if row.enabled != want:
            row.enabled = want
            row.disabled_by = "" if want else "admin"
            changed += 1
        meta_kw: dict = {
            "tags": str(form.get(f"tags_{row.id}") or ""),
            "short_note": str(form.get(f"note_{row.id}") or ""),
            "docs_url": str(form.get(f"docs_{row.id}") or ""),
            "recommended_sampling": str(form.get(f"sampling_{row.id}") or ""),
        }
        if weights_on and f"weight_{row.id}" in form:
            meta_kw["usage_weight"] = _parse_weight(form.get(f"weight_{row.id}"))
        try:
            update_catalog_meta(db, row.id, **meta_kw)
        except ValueError as exc:
            db.rollback()
            request.session["flash_err"] = str(exc)
            return RedirectResponse("/models", status_code=303)
    write_audit(
        db,
        actor=user,
        action="catalog.update",
        entity_type="catalog_models",
        detail=f"toggled={changed}",
    )
    db.commit()
    request.session["flash_ok"] = f"Saved ({changed} enable toggles; metadata updated)."
    return RedirectResponse("/models", status_code=303)


@router.post("/models/import-sampling")
async def models_import_sampling(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    """One-shot import of recommended_sampling JSON keyed by model_id substrings."""
    import json

    from ...data.catalog import import_recommended_sampling

    form = await request.form()
    raw = str(form.get("sampling_json") or "").strip()
    overwrite = str(form.get("overwrite") or "") in ("1", "on", "true", "yes")
    if not raw:
        request.session["flash_err"] = "Paste a JSON object to import."
        return RedirectResponse("/models", status_code=303)
    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError:
        request.session["flash_err"] = "Invalid JSON."
        return RedirectResponse("/models", status_code=303)
    if not isinstance(mapping, dict) or not mapping:
        request.session["flash_err"] = "JSON must be a non-empty object."
        return RedirectResponse("/models", status_code=303)
    stats = import_recommended_sampling(db, mapping, overwrite=overwrite)
    write_audit(
        db,
        actor=user,
        action="catalog.import_sampling",
        entity_type="catalog_models",
        detail=f"updated={stats['updated']} skipped={stats['skipped']} unmatched={stats['unmatched']} overwrite={overwrite}",
    )
    db.commit()
    request.session["flash_ok"] = (
        f"Sampling import: updated {stats['updated']}, "
        f"skipped {stats['skipped']}, unmatched {stats['unmatched']}."
    )
    return RedirectResponse("/models", status_code=303)


@router.post("/models/aliases/detect")
async def models_alias_detect(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    """Return similar catalog ids for Active model (Detect button; no DB write)."""
    from ...model_aliases import detect_similar_models

    form = await request.form()
    model_id = str(form.get("model_id") or form.get("target_model_id") or "").strip()
    alias_id = str(form.get("alias_id") or "").strip()
    kind = str(form.get("kind") or "").strip()
    if not model_id:
        return JSONResponse({"error": "Active model required."}, status_code=400)
    prefix, models = detect_similar_models(
        db, model_id=model_id, kind=kind or None, alias_id=alias_id or None
    )
    return JSONResponse({"prefix": prefix, "models": models, "count": len(models)})


@router.post("/models/aliases/add")
async def models_alias_add(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...model_aliases import upsert_alias, validate_alias_id

    form = await request.form()
    alias_id = str(form.get("alias_id") or "")
    target_model_id = str(form.get("target_model_id") or "")
    preferred_source = str(form.get("preferred_source") or "")
    # New aliases always start: on + hide list + show target.
    hide_candidates = True
    show_backend = True

    def _err(msg: str, code: int = 400):
        if request.headers.get("x-requested-with") == "fetch":
            return JSONResponse({"error": msg}, status_code=code)
        request.session["flash_err"] = msg
        return RedirectResponse("/models", status_code=303)

    if validate_alias_id(alias_id) is None:
        return _err(
            "Invalid alias name (lowercase a-z0-9._-, not auto/auto-quality/auto-long)."
        )
    mid = validate_alias_id(alias_id)
    existing = db.query(ModelAlias).filter(ModelAlias.alias_id == mid).first()
    if existing is not None:
        return _err(f"Alias '{mid}' already exists.")

    row = upsert_alias(
        db,
        alias_id=alias_id,
        target_model_id=target_model_id,
        preferred_source=preferred_source,
        show_backend=show_backend,
        hide_candidates=hide_candidates,
        candidates=[target_model_id.strip()] if target_model_id.strip() else None,
        enabled=True,
    )
    if row is None:
        return _err("Alias needs an active model.")

    write_audit(
        db,
        actor=user,
        action="alias.create",
        entity_type="model_alias",
        entity_id=row.id,
        detail=f"{row.alias_id}→{row.target_model_id}",
    )
    db.commit()
    row = (
        db.query(ModelAlias)
        .options(joinedload(ModelAlias.candidates))
        .filter(ModelAlias.id == row.id)
        .first()
    )
    if request.headers.get("x-requested-with") == "fetch":
        return _render_alias_card(request, db, row, open=True)
    request.session["flash_ok"] = f"Alias '{row.alias_id}' added."
    return RedirectResponse("/models", status_code=303)


@router.post("/models/aliases/{alias_row_id}/save")
async def models_alias_save_one(
    alias_row_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...model_aliases import (
        kind_for_model,
        normalize_preferred_source,
        rename_alias,
        set_candidates,
        validate_alias_id,
    )

    row = (
        db.query(ModelAlias)
        .options(joinedload(ModelAlias.candidates))
        .filter(ModelAlias.id == alias_row_id)
        .first()
    )
    if row is None:
        return JSONResponse({"error": "Alias not found."}, status_code=404)

    form = await request.form()
    new_name = str(form.get("alias_id") or "").strip()
    target = str(form.get("target_model_id") or "").strip()
    if not target:
        return JSONResponse({"error": "Active model required."}, status_code=400)
    if validate_alias_id(new_name) is None:
        return JSONResponse({"error": "Invalid alias name."}, status_code=400)
    if rename_alias(db, row, new_name) is None:
        return JSONResponse({"error": "Alias name taken or invalid."}, status_code=400)

    row.target_model_id = target[:256]
    row.kind = kind_for_model(db, target)[:16]
    row.preferred_source = normalize_preferred_source(
        db, target, str(form.get("preferred_source") or "")
    )[:64]
    row.enabled = "enabled" in form
    row.hide_candidates = "hide_candidates" in form
    row.show_backend = "show_backend" in form

    cands = [c.strip() for c in form.getlist("candidates") if str(c).strip()]
    if not cands:
        cands = [target]
    set_candidates(db, row, cands, ensure_target=True)

    write_audit(
        db,
        actor=user,
        action="alias.update",
        entity_type="model_alias",
        entity_id=row.id,
        detail=f"{row.alias_id}→{row.target_model_id}",
    )
    db.commit()
    row = (
        db.query(ModelAlias)
        .options(joinedload(ModelAlias.candidates))
        .filter(ModelAlias.id == alias_row_id)
        .first()
    )
    return _render_alias_card(request, db, row, open=True)


@router.post("/models/aliases/{alias_row_id}/delete")
def models_alias_delete_one(
    alias_row_id: int,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    row = db.get(ModelAlias, alias_row_id)
    if row is None:
        if request.headers.get("x-requested-with") == "fetch":
            return JSONResponse({"error": "Alias not found."}, status_code=404)
        request.session["flash_err"] = "Alias not found."
        return RedirectResponse("/models", status_code=303)
    alias = row.alias_id
    db.delete(row)
    write_audit(
        db,
        actor=user,
        action="alias.delete",
        entity_type="model_alias",
        detail=alias,
    )
    db.commit()
    if request.headers.get("x-requested-with") == "fetch":
        return Response(status_code=204)
    request.session["flash_ok"] = f"Alias '{alias}' deleted."
    return RedirectResponse("/models", status_code=303)


# Legacy form endpoints kept as thin redirects for bookmarks / no-JS.
@router.post("/models/aliases/delete")
def models_alias_delete_legacy(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
    delete_id: str = Form(""),
):
    try:
        aid = int(delete_id)
    except (TypeError, ValueError):
        request.session["flash_err"] = "Bad alias id."
        return RedirectResponse("/models", status_code=303)
    return models_alias_delete_one(aid, request, db, user)
