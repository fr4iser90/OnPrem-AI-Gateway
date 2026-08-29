"""Access ceilings (grants): admin-defined source + model sets for users/teams.

Keys may only use a subset of their ceiling. Empty key fields inherit the grant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session, joinedload

from ..config import MODEL_CHECK_KINDS
from .backends import source_names
from .catalog import list_catalog
from .models import (
    WebUser,
    ApiKey,
    AuthSettings,
    CatalogModel,
    ModelAllowlist,
    ServiceGrant,
    Team,
)


@dataclass
class AccessCeiling:
    """Resolved grant for a key owner/team.

    unrestricted: platform admin with no need for grant rows.
    services: allowed source names (empty = no API access).
    models: per-source allowlist; missing key or None value = all ON models for that source.
    """

    unrestricted: bool = False
    services: set[str] = field(default_factory=set)
    # service -> set of model ids; None means unrestricted models for that service
    models: dict[str, set[str] | None] = field(default_factory=dict)
    label: str = ""  # "team X" / "user Y" / "platform admin"

    def models_for(self, service: str) -> set[str] | None:
        if self.unrestricted:
            return None
        if service not in self.services:
            return set()
        if service not in self.models:
            return None
        return self.models[service]


def _models_map_from_rows(
    rows: list[ModelAllowlist], services: set[str]
) -> dict[str, set[str] | None]:
    by: dict[str, set[str]] = {}
    for row in rows:
        if row.service not in services:
            continue
        by.setdefault(row.service, set()).add(row.model_name)
    # Explicit empty allowlist rows for a service aren't distinguishable from
    # "no rows". Convention: no rows for service => None (all models).
    out: dict[str, set[str] | None] = {}
    for svc in services:
        if svc in by:
            out[svc] = by[svc]
        else:
            out[svc] = None
    return out


def ceiling_from_team(team: Team) -> AccessCeiling:
    services = {g.service for g in (team.service_grants or [])}
    return AccessCeiling(
        unrestricted=False,
        services=services,
        models=_models_map_from_rows(list(team.model_allowlists or []), services),
        label=f"team:{team.name}",
    )


def ceiling_from_user(user: WebUser) -> AccessCeiling:
    if user.is_platform_admin:
        return AccessCeiling(unrestricted=True, label=f"admin:{user.username}")
    services = {g.service for g in (user.service_grants or [])}
    return AccessCeiling(
        unrestricted=False,
        services=services,
        models=_models_map_from_rows(list(user.model_allowlists or []), services),
        label=f"user:{user.username}",
    )


def load_user_with_grants(db: Session, user_id: int | None) -> WebUser | None:
    if not user_id:
        return None
    return (
        db.query(WebUser)
        .options(
            joinedload(WebUser.service_grants),
            joinedload(WebUser.model_allowlists),
        )
        .filter(WebUser.id == user_id)
        .first()
    )


def ceiling_for_key(db: Session, api_key: ApiKey) -> AccessCeiling:
    """Team key → team grant. Owner user → user grant. Else legacy unrestricted."""
    if api_key.team is not None:
        return ceiling_from_team(api_key.team)
    owner = api_key.owner
    if owner is None and api_key.owner_user_id:
        owner = load_user_with_grants(db, api_key.owner_user_id)
    if owner is not None:
        return ceiling_from_user(owner)
    # Keys without owner/team (tests / pre-grant installs): no ceiling.
    return AccessCeiling(unrestricted=True, label="legacy")


def effective_services(db: Session, api_key: ApiKey) -> set[str]:
    ceil = ceiling_for_key(db, api_key)
    key_services = {g.service for g in (api_key.service_grants or [])}

    if ceil.unrestricted:
        if key_services:
            return key_services
        # Admin key with nothing checked: no implicit "all sources" — require explicit
        # OR inherit nothing. Prefer: if unrestricted and empty key → all known sources
        # so admin UX "none checked = all" still works for admin-owned keys.
        return set(source_names(db))

    if not ceil.services:
        return set()
    if key_services:
        return key_services & ceil.services
    return set(ceil.services)


def effective_models(db: Session, api_key: ApiKey, service: str) -> set[str] | None:
    """None = unrestricted models for this service (still subject to catalog ON)."""
    if service not in effective_services(db, api_key):
        return set()

    ceil = ceiling_for_key(db, api_key)
    ceil_models = ceil.models_for(service)

    key_models = [
        m.model_name for m in (api_key.model_allowlists or []) if m.service == service
    ]
    if key_models:
        key_set = set(key_models)
        if ceil_models is None:
            return key_set
        return key_set & ceil_models

    # Inherit grant model set (None = all ON models for source)
    return ceil_models


def clamp_services(selected: list[str], ceil: AccessCeiling, db: Session) -> list[str]:
    if ceil.unrestricted:
        allowed = set(source_names(db))
        return [s for s in selected if s in allowed]
    if not ceil.services:
        return []
    return [s for s in selected if s in ceil.services]


def clamp_models(
    pairs: list[tuple[str, str]], ceil: AccessCeiling
) -> list[tuple[str, str]]:
    if ceil.unrestricted:
        return list(pairs)
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for svc, name in pairs:
        if svc not in ceil.services:
            continue
        allowed = ceil.models_for(svc)
        if allowed is not None and name not in allowed:
            continue
        pair = (svc, name)
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def catalog_groups_for_ceiling(
    db: Session, ceil: AccessCeiling, *, kinds: set[str] | None = None
) -> list[tuple[str, list[CatalogModel]]]:
    """Enabled catalog rows visible under a grant (for key/grant UIs)."""
    from ..model_aliases import list_aliases

    kinds = kinds or set(MODEL_CHECK_KINDS)
    allowed_sources = set(services_for_ceiling(db, ceil))
    by_source: dict[str, list[CatalogModel]] = {}
    for row in list_catalog(db):
        if not row.enabled:
            continue
        if row.kind not in kinds:
            continue
        if row.source_name not in allowed_sources:
            continue
        allowed = None if ceil.unrestricted else ceil.models_for(row.source_name)
        if allowed is not None and row.model_id not in allowed:
            continue
        by_source.setdefault(row.source_name, []).append(row)

    alias_inject: dict[str, list[CatalogModel]] = {}
    for a in list_aliases(db, enabled_only=True):
        if (a.kind or "chat") not in kinds:
            continue
        svc = (a.preferred_source or "").strip() or "chat"
        if svc not in allowed_sources:
            continue
        if not ceil.unrestricted:
            allowed = ceil.models_for(svc)
            target = (a.target_model_id or "").strip()
            if allowed is not None and a.alias_id not in allowed and target not in allowed:
                continue
        alias_inject.setdefault(svc, []).append(
            CatalogModel(
                source_name=svc,
                kind=a.kind or "chat",
                model_id=a.alias_id,
                enabled=True,
                short_note=(
                    (a.description or "").strip()
                    or f"alias → {a.target_model_id}"
                )[:512],
                tags="alias",
            )
        )

    for svc, rows in alias_inject.items():
        by_source[svc] = rows + by_source.get(svc, [])

    return sorted(by_source.items(), key=lambda x: x[0])


def services_for_ceiling(db: Session, ceil: AccessCeiling) -> list[str]:
    if ceil.unrestricted:
        return list(source_names(db))
    names = source_names(db)
    return [n for n in names if n in ceil.services]


def sync_user_grants(db: Session, user: WebUser, services: list[str]) -> None:
    db.query(ServiceGrant).filter(ServiceGrant.user_id == user.id).delete()
    for s in services:
        db.add(ServiceGrant(user_id=user.id, service=s))


def sync_user_models(
    db: Session, user: WebUser, models: list[tuple[str, str]]
) -> None:
    db.query(ModelAllowlist).filter(ModelAllowlist.user_id == user.id).delete()
    for svc, name in models:
        db.add(ModelAllowlist(user_id=user.id, service=svc, model_name=name))


def sync_user_models_for_service(
    db: Session, user: WebUser, service: str, model_names: list[str] | None
) -> None:
    """Replace allowlist for one source only. None = all models (no rows)."""
    db.query(ModelAllowlist).filter(
        ModelAllowlist.user_id == user.id,
        ModelAllowlist.service == service,
    ).delete()
    if model_names is None:
        return
    for name in model_names:
        db.add(ModelAllowlist(user_id=user.id, service=service, model_name=name))


def _auth_row(db: Session) -> AuthSettings | None:
    return db.query(AuthSettings).first()


def parse_source_model_lines(raw: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        svc, name = line.split(":", 1)
        svc, name = svc.strip(), name.strip()
        if not svc or not name:
            continue
        pair = (svc, name)
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out


# Explicit "new users get nothing". Empty DB column used to mean "fall back to
# kind-default sources"; that mixed routing with grants. '-' cannot be a source name.
DEFAULT_GRANT_NONE = "-"


def configured_default_sources(db: Session) -> list[str]:
    """Sources new non-admin users get. Unset or none → []. Independent of source routing."""
    auth = _auth_row(db)
    raw = (getattr(auth, "default_grant_sources", None) or "").strip() if auth else ""
    if not raw or raw == DEFAULT_GRANT_NONE:
        return []
    names = source_names(db)
    wanted = {
        p.strip()
        for p in raw.split(",")
        if p.strip() and p.strip() != DEFAULT_GRANT_NONE
    }
    return [n for n in names if n in wanted]


def default_grant_was_saved(db: Session) -> bool:
    """True once default access was saved (including explicit none)."""
    auth = _auth_row(db)
    raw = (getattr(auth, "default_grant_sources", None) or "").strip() if auth else ""
    return bool(raw)


def display_default_sources(db: Session) -> list[str]:
    """Sources to show checked in default-access UI (opt-out until first save)."""
    if not default_grant_was_saved(db):
        return source_names(db)
    return configured_default_sources(db)


def configured_default_models(db: Session) -> list[tuple[str, str]]:
    """Optional model ceiling for new users. Empty = all ON models."""
    auth = _auth_row(db)
    raw = (getattr(auth, "default_grant_models", None) or "") if auth else ""
    allowed = set(configured_default_sources(db))
    return [(s, m) for s, m in parse_source_model_lines(raw) if s in allowed]


def _enabled_catalog_models_for_services(
    db: Session, services: set[str]
) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for row in list_catalog(db):
        if not row.enabled or row.kind not in MODEL_CHECK_KINDS:
            continue
        if row.source_name in services:
            out.add((row.source_name, row.model_id))
    return out


def display_default_models(db: Session) -> set[str]:
    """Model keys (source:model) checked in default-access UI."""
    services = list(display_default_sources(db))
    if not services:
        return set()
    all_enabled = display_enabled_models_for_services(db, services)
    if not all_enabled:
        return set()

    if not default_grant_was_saved(db):
        return all_enabled

    auth = _auth_row(db)
    raw = (getattr(auth, "default_grant_models", None) or "").strip() if auth else ""
    if not raw:
        return all_enabled

    configured = configured_default_models(db)
    if not configured:
        return all_enabled

    selected = {f"{s}:{m}" for s, m in configured}
    live = selected & all_enabled
    return live if live else all_enabled


def display_enabled_models_for_services(db: Session, services: list[str]) -> set[str]:
    """All enabled catalog models for sources — opt-out UI default."""
    names = set(services)
    if not names:
        return set()
    return {f"{s}:{m}" for s, m in _enabled_catalog_models_for_services(db, names)}


def normalize_model_allowlist(
    db: Session, services: list[str], models: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Empty list = unrestricted (all ON models) when every enabled model is selected."""
    service_set = set(services)
    kept = [(s, m) for s, m in models if s in service_set]
    all_enabled = _enabled_catalog_models_for_services(db, service_set)
    if kept and set(kept) == all_enabled:
        return []
    return kept


def save_default_grant(
    db: Session, services: list[str], models: list[tuple[str, str]]
) -> None:
    from .models import AuthSettings as _AS

    auth = _auth_row(db)
    if auth is None:
        auth = _AS()
        db.add(auth)
        db.flush()
    names = set(source_names(db))
    services = [s for s in services if s in names]
    auth.default_grant_sources = ",".join(services) if services else DEFAULT_GRANT_NONE
    kept = normalize_model_allowlist(db, services, models)
    auth.default_grant_models = "\n".join(f"{s}:{m}" for s, m in kept)


def apply_default_grant(db: Session, user: WebUser) -> None:
    services = configured_default_sources(db)
    sync_user_grants(db, user, services)
    models = configured_default_models(db)
    if models:
        sync_user_models(db, user, models)


def grant_summary(ceil: AccessCeiling) -> str:
    if ceil.unrestricted:
        return "Full access (platform admin)"
    if not ceil.services:
        return "No sources granted"
    parts = []
    for svc in sorted(ceil.services):
        mods = ceil.models_for(svc)
        if mods is None:
            parts.append(f"{svc}: all models")
        else:
            parts.append(f"{svc}: {len(mods)} model(s)")
    return " · ".join(parts)
