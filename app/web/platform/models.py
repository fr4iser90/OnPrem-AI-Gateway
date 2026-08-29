from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from ...audit import write_audit
from ...data.db import get_db
from ...data.models import WebUser
from ..session import require_platform_admin
from ..shared import templates, _gpu_power_enabled

router = APIRouter()


def _parse_candidate_lines(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in (raw or "").replace(",", "\n").splitlines():
        mid = line.strip()
        if mid and mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out


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
    from ...data.usage_weights import catalog_weight_suggestions
    from ...model_aliases import (
        candidate_model_ids,
        catalog_ids_for_picker,
        list_aliases,
    )
    from ...stats import model_perf_averages, model_perf_by_id
    from ...vision_route import group_kind_rows_vl_pairs
    from ..accounts import get_auth_settings

    flash_ok = request.session.pop("flash_ok", None)
    flash_err = request.session.pop("flash_err", None)
    rows = list_catalog(db)
    auth = get_auth_settings(db)
    auto_vl = bool(auth.auto_vl_routing)
    pool_weights = bool(auth.pool_model_weights_enabled)
    weight_status = catalog_weight_suggestions(db)
    observed = model_perf_by_id(model_perf_averages(db, key_ids=None, lookback_days=7))
    groups = []
    for kind, kind_rows in catalog_grouped_by_kind(rows):
        pairs = group_kind_rows_vl_pairs(
            kind_rows,
            pair=auto_vl and kind == "chat",
        )
        active, stale = split_sync_stale_pairs(pairs)
        groups.append((kind, active, stale, len(kind_rows)))
    from ...data.backends import list_sources

    aliases = list_aliases(db)
    catalog_model_ids = catalog_ids_for_picker(db)
    alias_candidates: dict[int, list[str]] = {}
    alias_target_options: dict[int, list[str]] = {}
    for a in aliases:
        cands = candidate_model_ids(a)
        alias_candidates[a.id] = cands
        # Dropdown: candidates first, then other catalog ids (admin can promote new active).
        seen: set[str] = set()
        opts: list[str] = []
        for mid in cands + catalog_model_ids:
            if mid and mid not in seen:
                seen.add(mid)
                opts.append(mid)
        if a.target_model_id and a.target_model_id not in seen:
            opts.insert(0, a.target_model_id)
        alias_target_options[a.id] = opts

    return templates.TemplateResponse(
        request,
        "models.html",
        {
            "user": user,
            "nav": "models",
            "rows": rows,
            "groups": groups,
            "observed": observed,
            "auto_vl_routing": auto_vl,
            "pool_model_weights_enabled": pool_weights,
            "weight_status": weight_status,
            "suggest_docs_url": suggest_docs_url,
            "format_param_count": format_param_count,
            "format_bytes": format_bytes,
            "tag_suggestions": TAG_SUGGESTIONS,
            "aliases": aliases,
            "alias_candidates": alias_candidates,
            "alias_target_options": alias_target_options,
            "catalog_model_ids": catalog_model_ids,
            "source_names": [s.name for s in list_sources(db)],
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
        }
        if weights_on and f"weight_{row.id}" in form:
            meta_kw["usage_weight"] = _parse_weight(form.get(f"weight_{row.id}"))
        update_catalog_meta(db, row.id, **meta_kw)
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


@router.post("/models/aliases/add")
def models_alias_add(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
    alias_id: str = Form(""),
    target_model_id: str = Form(""),
    preferred_source: str = Form(""),
    description: str = Form(""),
    family_prefix: str = Form(""),
    show_backend: str | None = Form(None),
    hide_candidates: str | None = Form(None),
    suggest_family: str | None = Form(None),
    sort_order: str = Form("0"),
):
    from ...model_aliases import upsert_alias, validate_alias_id

    if validate_alias_id(alias_id) is None:
        request.session["flash_err"] = (
            "Invalid alias id (lowercase a-z0-9._-, not auto/auto-quality/auto-long)."
        )
        return RedirectResponse("/models", status_code=303)
    try:
        order = int(sort_order or 0)
    except (TypeError, ValueError):
        order = 0
    row = upsert_alias(
        db,
        alias_id=alias_id,
        target_model_id=target_model_id,
        preferred_source=preferred_source,
        description=description,
        show_backend=show_backend is not None,
        hide_candidates=hide_candidates is not None,
        family_prefix=family_prefix,
        suggest_family=suggest_family is not None,
        enabled=True,
        sort_order=order,
    )
    if row is None:
        request.session["flash_err"] = "Alias needs a target model id."
        return RedirectResponse("/models", status_code=303)
    write_audit(
        db,
        actor=user,
        action="alias.create",
        entity_type="model_alias",
        entity_id=row.id,
        detail=f"{row.alias_id}→{row.target_model_id}",
    )
    db.commit()
    n = len(row.candidates or [])
    request.session["flash_ok"] = (
        f"Alias '{row.alias_id}' added"
        + (f" ({n} candidates)." if n else ".")
    )
    return RedirectResponse("/models", status_code=303)


@router.post("/models/aliases/save")
async def models_alias_save(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
):
    from ...model_aliases import list_aliases, set_candidates

    form = await request.form()
    for row in list_aliases(db):
        target = str(form.get(f"target_{row.id}") or "").strip()
        if not target:
            continue
        row.target_model_id = target[:256]
        row.preferred_source = str(form.get(f"source_{row.id}") or "").strip().lower()[:64]
        row.description = str(form.get(f"desc_{row.id}") or "").strip()[:512]
        row.family_prefix = str(form.get(f"family_{row.id}") or "").strip()[:128]
        row.show_backend = f"show_{row.id}" in form
        row.enabled = f"enabled_{row.id}" in form
        row.hide_candidates = f"hide_{row.id}" in form
        try:
            row.sort_order = int(form.get(f"sort_{row.id}") or 0)
        except (TypeError, ValueError):
            row.sort_order = 0
        cands = _parse_candidate_lines(str(form.get(f"candidates_{row.id}") or ""))
        set_candidates(db, row, cands, ensure_target=True)
    write_audit(
        db,
        actor=user,
        action="alias.update",
        entity_type="model_alias",
        detail="bulk",
    )
    db.commit()
    request.session["flash_ok"] = "Aliases saved."
    return RedirectResponse("/models", status_code=303)


@router.post("/models/aliases/suggest")
def models_alias_suggest(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
    alias_row_id: str = Form(""),
):
    from ...data.models import ModelAlias
    from ...model_aliases import suggest_and_set_candidates

    try:
        aid = int(alias_row_id)
    except (TypeError, ValueError):
        request.session["flash_err"] = "Bad alias id."
        return RedirectResponse("/models", status_code=303)
    row = db.get(ModelAlias, aid)
    if row is None:
        request.session["flash_err"] = "Alias not found."
        return RedirectResponse("/models", status_code=303)
    matches = suggest_and_set_candidates(db, row)
    write_audit(
        db,
        actor=user,
        action="alias.suggest_family",
        entity_type="model_alias",
        entity_id=row.id,
        detail=f"{row.alias_id} family={row.family_prefix} n={len(matches)}",
    )
    db.commit()
    request.session["flash_ok"] = (
        f"Suggested {len(matches)} candidate(s) for '{row.alias_id}' "
        f"(prefix {row.family_prefix or '—'})."
    )
    return RedirectResponse("/models", status_code=303)


@router.post("/models/aliases/delete")
def models_alias_delete(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
    delete_id: str = Form(""),
):
    from ...data.models import ModelAlias

    try:
        aid = int(delete_id)
    except (TypeError, ValueError):
        request.session["flash_err"] = "Bad alias id."
        return RedirectResponse("/models", status_code=303)
    row = db.get(ModelAlias, aid)
    if row is None:
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
    request.session["flash_ok"] = f"Alias '{alias}' deleted."
    return RedirectResponse("/models", status_code=303)
