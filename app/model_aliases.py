"""Public model aliases: stable client ids → real catalog model + candidate pool.

Option B: each alias is a slot (e.g. qwen3.6) with an active target and a pool of
candidates. When hide_candidates is on, targets/candidates are filtered from
GET /v1/models so clients see one id per slot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy.orm import Session, joinedload

from .data.models import CatalogModel, ModelAlias, ModelAliasCandidate, utcnow
from .vision_route import rewrite_json_model

_ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,126}$")
# Size token that usually ends the family prefix (Qwen3.6-35B-… → Qwen3.6).
_FAMILY_SIZE_RE = re.compile(
    r"(?i)[-_.](?:\d+\s*[Bb](?:-A\d+[Bb])?|A\d+[Bb])\b"
)
_SLOT_TAG_RE = re.compile(r"(?i)(?:^|[,;\s])slot:([a-z0-9][a-z0-9._-]{0,126})")


@dataclass(frozen=True)
class AliasResolution:
    alias_id: str
    target_model_id: str
    preferred_source: str | None = None


def normalize_alias_id(raw: str | None) -> str:
    return (raw or "").strip().lower()


def validate_alias_id(raw: str | None) -> str | None:
    """Return normalized id or None if invalid."""
    mid = normalize_alias_id(raw)
    if not mid or not _ALIAS_RE.match(mid):
        return None
    # Reserved auto slots stay on auto_route.py
    if mid in {"auto", "auto-quality", "auto-long"}:
        return None
    return mid


def infer_family_prefix(model_id: str | None) -> str:
    """Guess family stem from a catalog id (Qwen3.6-35B-… → Qwen3.6)."""
    mid = (model_id or "").strip()
    if not mid:
        return ""
    m = _FAMILY_SIZE_RE.search(mid)
    if m:
        return mid[: m.start()].rstrip("-_.")
    # Fallback: first hyphen segment if it looks versioned (qwen3.6)
    head = mid.split("-", 1)[0]
    return head if head else mid


def get_alias(db: Session, alias_id: str | None) -> ModelAlias | None:
    mid = normalize_alias_id(alias_id)
    if not mid:
        return None
    return (
        db.query(ModelAlias)
        .options(joinedload(ModelAlias.candidates))
        .filter(ModelAlias.alias_id == mid, ModelAlias.enabled.is_(True))
        .first()
    )


def resolve_alias(db: Session, model: str | None) -> AliasResolution | None:
    row = get_alias(db, model)
    if row is None:
        return None
    target = (row.target_model_id or "").strip()
    if not target:
        return None
    pref = (row.preferred_source or "").strip().lower() or None
    return AliasResolution(
        alias_id=row.alias_id,
        target_model_id=target,
        preferred_source=pref,
    )


def rewrite_model_alias(
    db: Session,
    body: bytes | None,
    *,
    asked: str | None,
) -> tuple[bytes | None, AliasResolution | None]:
    """Rewrite JSON body when asked id is a configured alias."""
    resolved = resolve_alias(db, asked)
    if resolved is None or not body:
        return body, None
    if asked and asked.strip() == resolved.target_model_id:
        return body, resolved
    return rewrite_json_model(body, resolved.target_model_id), resolved


def apply_client_model_rewrites(
    db: Session,
    body: bytes | None,
    *,
    asked: str | None,
    auth,
) -> tuple[bytes | None, str | None, str | None]:
    """Apply auto-* then custom aliases. Returns (body, rewritten_id, preferred_source)."""
    from .auto_route import rewrite_auto_model

    body, auto_to = rewrite_auto_model(body, asked=asked, auth=auth)
    next_ask = auto_to or asked
    body, alias = rewrite_model_alias(db, body, asked=next_ask)
    rewritten = (alias.target_model_id if alias else None) or auto_to
    pref = alias.preferred_source if alias else None
    return body, rewritten, pref


def list_aliases(db: Session, *, enabled_only: bool = False) -> list[ModelAlias]:
    q = (
        db.query(ModelAlias)
        .options(joinedload(ModelAlias.candidates))
        .order_by(ModelAlias.sort_order, ModelAlias.alias_id)
    )
    if enabled_only:
        q = q.filter(ModelAlias.enabled.is_(True))
    return q.all()


def candidate_model_ids(row: ModelAlias) -> list[str]:
    """Ordered unique candidate ids; always includes active target if set."""
    seen: set[str] = set()
    out: list[str] = []
    target = (row.target_model_id or "").strip()
    if target:
        seen.add(target)
        out.append(target)
    for c in row.candidates or []:
        mid = (c.model_id or "").strip()
        if mid and mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out


def hidden_catalog_ids(db: Session) -> set[str]:
    """Catalog model ids that should not appear in GET /v1/models for clients."""
    hidden: set[str] = set()
    for row in list_aliases(db, enabled_only=True):
        if not getattr(row, "hide_candidates", True):
            continue
        for mid in candidate_model_ids(row):
            hidden.add(mid)
    return hidden


def family_matches(
    db: Session,
    *,
    prefix: str | None = None,
    alias_id: str | None = None,
    kind: str = "chat",
) -> list[str]:
    """Catalog model ids matching family prefix and/or slot:alias tag."""
    pref = (prefix or "").strip()
    aid = normalize_alias_id(alias_id)
    if not pref and not aid:
        return []
    found: list[str] = []
    seen: set[str] = set()
    q = db.query(CatalogModel).filter(CatalogModel.enabled.is_(True))
    if kind:
        q = q.filter(CatalogModel.kind == kind)
    for row in q.order_by(CatalogModel.model_id).all():
        mid = (row.model_id or "").strip()
        if not mid or mid in seen:
            continue
        match = False
        if pref and (mid == pref or mid.startswith(pref + "-") or mid.startswith(pref + "_")):
            match = True
        if not match and aid:
            tags = row.tags or ""
            for m in _SLOT_TAG_RE.finditer(tags):
                if m.group(1).lower() == aid:
                    match = True
                    break
        if match:
            seen.add(mid)
            found.append(mid)
    return found


def set_candidates(
    db: Session,
    row: ModelAlias,
    model_ids: list[str] | None,
    *,
    ensure_target: bool = True,
) -> None:
    """Replace candidate rows. Active target stays on ModelAlias.target_model_id."""
    ids: list[str] = []
    seen: set[str] = set()
    for raw in model_ids or []:
        mid = (raw or "").strip()
        if mid and mid not in seen:
            seen.add(mid)
            ids.append(mid[:256])
    if ensure_target:
        target = (row.target_model_id or "").strip()
        if target and target not in seen:
            ids.insert(0, target[:256])
            seen.add(target)
    row.candidates.clear()
    db.flush()
    for i, mid in enumerate(ids):
        row.candidates.append(
            ModelAliasCandidate(model_id=mid, sort_order=i)
        )


def suggest_and_set_candidates(
    db: Session,
    row: ModelAlias,
    *,
    prefix: str | None = None,
) -> list[str]:
    """Fill candidates from family heuristic; keep current target active."""
    fam = (prefix or row.family_prefix or "").strip()
    if not fam:
        fam = infer_family_prefix(row.target_model_id)
    if fam:
        row.family_prefix = fam[:128]
    matches = family_matches(db, prefix=fam, alias_id=row.alias_id, kind=row.kind or "chat")
    set_candidates(db, row, matches, ensure_target=True)
    return candidate_model_ids(row)


def upsert_alias(
    db: Session,
    *,
    alias_id: str,
    target_model_id: str,
    preferred_source: str = "",
    description: str = "",
    show_backend: bool = True,
    enabled: bool = True,
    kind: str = "chat",
    sort_order: int = 0,
    hide_candidates: bool = True,
    family_prefix: str = "",
    candidates: list[str] | None = None,
    suggest_family: bool = False,
) -> ModelAlias | None:
    mid = validate_alias_id(alias_id)
    target = (target_model_id or "").strip()
    if not mid or not target:
        return None
    row = (
        db.query(ModelAlias)
        .options(joinedload(ModelAlias.candidates))
        .filter(ModelAlias.alias_id == mid)
        .first()
    )
    if row is None:
        row = ModelAlias(alias_id=mid, created_at=utcnow())
        db.add(row)
        db.flush()
    row.target_model_id = target[:256]
    row.preferred_source = (preferred_source or "").strip().lower()[:64]
    row.description = (description or "").strip()[:512]
    row.show_backend = bool(show_backend)
    row.enabled = bool(enabled)
    row.kind = (kind or "chat").strip().lower()[:16] or "chat"
    row.sort_order = int(sort_order)
    row.hide_candidates = bool(hide_candidates)
    fam = (family_prefix or "").strip() or infer_family_prefix(target)
    row.family_prefix = fam[:128]
    if suggest_family:
        suggest_and_set_candidates(db, row, prefix=fam or None)
    elif candidates is not None:
        set_candidates(db, row, candidates, ensure_target=True)
    elif not row.candidates:
        # First create without explicit list: at least the active target.
        set_candidates(db, row, [target], ensure_target=True)
    else:
        # Ensure active target stays in the pool when only target changes.
        set_candidates(db, row, candidate_model_ids(row), ensure_target=True)
    return row


def delete_alias(db: Session, alias_id: str) -> bool:
    mid = normalize_alias_id(alias_id)
    row = db.query(ModelAlias).filter(ModelAlias.alias_id == mid).first()
    if row is None:
        return False
    db.delete(row)
    return True


def _catalog_row_for_target(db: Session, target: str) -> CatalogModel | None:
    return (
        db.query(CatalogModel)
        .filter(CatalogModel.model_id == target, CatalogModel.enabled.is_(True))
        .order_by(CatalogModel.source_name)
        .first()
    )


def model_allowed_via_alias(db: Session, allow: set[str], model: str | None) -> bool:
    """True if allowlists an alias whose active target is ``model``."""
    mid = (model or "").strip()
    if not mid or not allow:
        return False
    for row in list_aliases(db, enabled_only=True):
        if row.alias_id in allow and (row.target_model_id or "").strip() == mid:
            return True
    return False


def alias_visible_for_allow(
    row: ModelAlias,
    *,
    allow: set[str] | None,
) -> bool:
    """Whether this alias should appear for a key with the given model allow set.

    ``allow is None`` means unrestricted. Alias shows if alias_id or active
    target is listed (so admins can grant either the stable id or the backend).
    """
    if allow is None:
        return True
    if row.alias_id in allow:
        return True
    target = (row.target_model_id or "").strip()
    return bool(target and target in allow)


def alias_list_entries(
    db: Session,
    *,
    allow: set[str] | None = None,
    allowed_sources: set[str] | None = None,
    capacity_by_source: dict | None = None,
) -> list[dict]:
    """Synthetic /v1/models rows for enabled aliases (clients pick stable ids)."""
    from .data.source_capacity import SourceCapacity, attach_capacity

    out: list[dict] = []
    for row in list_aliases(db, enabled_only=True):
        if not alias_visible_for_allow(row, allow=allow):
            continue
        pref = (row.preferred_source or "").strip().lower()
        if (
            allowed_sources is not None
            and pref
            and pref not in allowed_sources
        ):
            # Preferred source not granted — skip unless unrestricted sources.
            continue
        target = (row.target_model_id or "").strip()
        if not target:
            continue
        cat = _catalog_row_for_target(db, target)
        if (
            allowed_sources is not None
            and cat is not None
            and cat.source_name not in allowed_sources
            and (not pref or pref not in allowed_sources)
        ):
            continue
        desc = (row.description or "").strip()
        if row.show_backend:
            backend_bit = f"→ {target}"
            desc = f"{desc} {backend_bit}".strip() if desc else backend_bit
        entry: dict = {
            "id": row.alias_id,
            "object": "model",
            "owned_by": "onprem",
            "created": int((row.created_at or utcnow()).timestamp()),
            "description": desc or row.alias_id,
        }
        if cat is not None:
            from .data.catalog import (
                architecture_for_openai_payload,
                context_length_for_model,
                tags_for_openai_payload,
            )

            ctx = context_length_for_model(cat)
            if ctx is not None:
                entry["context_length"] = ctx
            if cat.ctx_size is not None:
                entry["ctx_size"] = cat.ctx_size
            tags = tags_for_openai_payload(cat)
            if tags:
                entry["tags"] = tags
            arch = architecture_for_openai_payload(cat)
            if arch:
                entry["architecture"] = arch
        # Slots from preferred source, else the catalog row's source.
        src_for_slots = pref or (cat.source_name if cat is not None else "")
        if capacity_by_source and src_for_slots:
            cap = capacity_by_source.get(src_for_slots)
            if isinstance(cap, SourceCapacity):
                attach_capacity(entry, cap)
        out.append(entry)
    return out


def catalog_ids_for_picker(db: Session) -> list[str]:
    """Distinct enabled chat/embed model ids for alias target dropdowns."""
    rows = (
        db.query(CatalogModel.model_id)
        .filter(
            CatalogModel.enabled.is_(True),
            CatalogModel.kind.in_(("chat", "embed")),
        )
        .order_by(CatalogModel.model_id)
        .distinct()
        .all()
    )
    return [r[0] for r in rows if r[0]]
