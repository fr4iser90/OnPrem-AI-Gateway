from __future__ import annotations

from fastapi import Request
from sqlalchemy.orm import Session, joinedload

from ..config import MODEL_CHECK_KINDS, Settings
from ..data.backends import list_sources, source_chip_rows, source_names
from ..data.catalog import list_catalog
from ..data.models import (
    WebUser,
    ApiKey,
    CatalogModel,
    ModelAllowlist,
    ServiceGrant,
    Team,
    TeamMember,
)
_templates = None


def __getattr__(name: str):
    global _templates
    if name == "templates":
        if _templates is None:
            from .templating import make_templates

            _templates = make_templates()
        return _templates
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _source_tips(db: Session) -> dict[str, str]:
    from sqlalchemy import func

    from ..data.models import CatalogModel

    counts = dict(
        db.query(CatalogModel.source_name, func.count(CatalogModel.id))
        .filter(CatalogModel.enabled.is_(True))
        .group_by(CatalogModel.source_name)
        .all()
    )
    tips: dict[str, str] = {}
    for row in source_chip_rows(db):
        n = int(counts.get(row["name"]) or 0)
        extra = f"{n} models" if n != 1 else "1 model"
        if n == 0:
            extra = ""
        tips[row["name"]] = " · ".join(p for p in (row.get("tooltip"), extra) if p) or row["name"]
    return tips


def _sync_team_members(db: Session, team: Team, member_ids: list, owner_ids: list) -> None:
    members = {int(uid) for uid in member_ids}
    owners = {int(uid) for uid in owner_ids}
    keep = members | owners
    db.query(TeamMember).filter(TeamMember.team_id == team.id).delete()
    for uid in sorted(keep):
        db.add(
            TeamMember(
                team_id=team.id,
                user_id=uid,
                role="owner" if uid in owners else "member",
            )
        )


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _gpu_power_enabled(request: Request, db: Session | None = None) -> bool:
    """Energy UI when env set, or any source would auto/explicit-probe."""
    if (_settings(request).gpu_power_url or "").strip():
        return True
    if db is not None:
        from ..data.models import BackendSource
        from ..usage_pool import probe_url_for_source

        for src in db.query(BackendSource).all():
            if probe_url_for_source(src):
                return True
    return False


def _temp_guard_enabled(request: Request, db: Session | None = None) -> bool:
    """Thermal UI when globally on and any source has sidecar checks enabled."""
    if _settings(request).temp_guard_disabled:
        return False
    if db is not None:
        from ..data.models import BackendSource
        from ..usage_pool import temp_guard_url_for_source

        for src in db.query(BackendSource).all():
            if temp_guard_url_for_source(src):
                return True
    return False


def _parse_services(form_list: list[str] | None, allowed: set[str] | list[str]) -> list[str]:
    allowed_set = set(allowed)
    if not form_list:
        return []
    return [s for s in form_list if s in allowed_set]


def _parse_models(raw: str, default_service: str, db: Session) -> list[tuple[str, str]]:
    """Parse lines like 'chat:llama3' or bare names for a given default source."""
    by_name = {s.name: s for s in list_sources(db)}
    out: list[tuple[str, str]] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            svc, name = line.split(":", 1)
            svc, name = svc.strip(), name.strip()
            src = by_name.get(svc)
            if src and src.kind in MODEL_CHECK_KINDS and name:
                out.append((svc, name))
        else:
            src = by_name.get(default_service)
            if src and src.kind in MODEL_CHECK_KINDS:
                out.append((default_service, line))
    return out


def _parse_model_checks(form_list: list[str] | None) -> list[tuple[str, str]]:
    """Parse checkbox values 'source:model'."""
    out: list[tuple[str, str]] = []
    for raw in form_list or []:
        raw = (raw or "").strip()
        if ":" not in raw:
            continue
        svc, name = raw.split(":", 1)
        svc, name = svc.strip(), name.strip()
        if svc and name:
            out.append((svc, name))
    return out


def _collect_models_from_form(form, db: Session) -> list[tuple[str, str]]:
    """Checkbox allowlist from catalog. Empty = unrestricted."""
    _ = db
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for pair in _parse_model_checks(form.getlist("models")):
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def _selected_model_keys(rows: list[ModelAllowlist]) -> set[str]:
    return {f"{m.service}:{m.model_name}" for m in rows}


def _catalog_for_allowlist(
    db: Session, selected: set[str]
) -> list[tuple[str, list[CatalogModel]]]:
    """Enabled catalog rows grouped by source; orphans from allowlist appended.

    Public aliases are injected under their preferred source (or ``chat``) so
    grants store ``chat:qwen3.6`` and authorize matches via alias rewrite.
    """
    from ..model_aliases import list_aliases

    by_source: dict[str, list[CatalogModel]] = {}
    seen: set[str] = set()
    for row in list_catalog(db):
        if not row.enabled and f"{row.source_name}:{row.model_id}" not in selected:
            continue
        by_source.setdefault(row.source_name, []).append(row)
        seen.add(f"{row.source_name}:{row.model_id}")

    # Aliases first within each source group.
    alias_inject: dict[str, list[CatalogModel]] = {}
    for a in list_aliases(db, enabled_only=True):
        svc = (a.preferred_source or "").strip() or "chat"
        key = f"{svc}:{a.alias_id}"
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
        seen.add(key)

    for svc, rows in alias_inject.items():
        by_source[svc] = rows + by_source.get(svc, [])

    for key in sorted(selected - seen):
        if ":" not in key:
            continue
        svc, mid = key.split(":", 1)
        by_source.setdefault(svc, []).append(
            CatalogModel(source_name=svc, kind="chat", model_id=mid, enabled=True)
        )
    return sorted(by_source.items(), key=lambda x: x[0])


def _sync_key_grants(db: Session, api_key: ApiKey, services: list[str]) -> None:
    db.query(ServiceGrant).filter(ServiceGrant.api_key_id == api_key.id).delete()
    for s in services:
        db.add(ServiceGrant(api_key_id=api_key.id, service=s))


def _sync_key_models(db: Session, api_key: ApiKey, models: list[tuple[str, str]]) -> None:
    db.query(ModelAllowlist).filter(ModelAllowlist.api_key_id == api_key.id).delete()
    for svc, name in models:
        db.add(ModelAllowlist(api_key_id=api_key.id, service=svc, model_name=name))


def _sync_team_grants(db: Session, team: Team, services: list[str]) -> None:
    db.query(ServiceGrant).filter(ServiceGrant.team_id == team.id).delete()
    for s in services:
        db.add(ServiceGrant(team_id=team.id, service=s))


def _sync_team_models(db: Session, team: Team, models: list[tuple[str, str]]) -> None:
    db.query(ModelAllowlist).filter(ModelAllowlist.team_id == team.id).delete()
    for svc, name in models:
        db.add(ModelAllowlist(team_id=team.id, service=svc, model_name=name))


def _resolve_ceiling(
    db: Session,
    *,
    teams_on: bool,
    acting_user: WebUser,
    team_id: int | None = None,
    owner_user_id: int | None = None,
    api_key: ApiKey | None = None,
):
    from ..data.grants import (
        ceiling_for_key,
        ceiling_from_team,
        ceiling_from_user,
        load_user_with_grants,
    )

    if api_key is not None:
        return ceiling_for_key(db, api_key)
    if teams_on:
        if team_id:
            team = (
                db.query(Team)
                .options(
                    joinedload(Team.service_grants),
                    joinedload(Team.model_allowlists),
                )
                .filter(Team.id == team_id)
                .first()
            )
            if team:
                return ceiling_from_team(team)
        from ..data.grants import AccessCeiling

        return AccessCeiling(unrestricted=False, services=set(), label="no-team")
    uid = owner_user_id or acting_user.id
    owner = load_user_with_grants(db, uid) or acting_user
    return ceiling_from_user(owner)


def _key_form_context(
    db: Session,
    *,
    user: WebUser,
    teams_on: bool,
    api_key: ApiKey | None,
    teams: list,
    owners: list,
    selected_services: list[str] | None = None,
    team_id: int | None = None,
    owner_user_id: int | None = None,
) -> dict:
    from ..data.grants import (
        catalog_groups_for_ceiling,
        display_enabled_models_for_services,
        grant_summary,
        services_for_ceiling,
    )
    from .ops import _format_model_limits

    if api_key is not None:
        team_id = api_key.team_id
        owner_user_id = api_key.owner_user_id
    ceil = _resolve_ceiling(
        db,
        teams_on=teams_on,
        acting_user=user,
        team_id=team_id,
        owner_user_id=owner_user_id,
        api_key=api_key,
    )
    services = services_for_ceiling(db, ceil)
    if selected_services is None:
        if api_key is not None:
            key_svcs = [g.service for g in api_key.service_grants]
            selected_services = key_svcs if key_svcs else list(services)
        else:
            selected_services = list(services)
    if api_key is not None:
        selected_models = _selected_model_keys(list(api_key.model_allowlists))
        if not selected_models:
            selected_models = display_enabled_models_for_services(db, selected_services)
    else:
        selected_models = display_enabled_models_for_services(db, selected_services)
    resolved_owner = (
        api_key.owner_user_id
        if api_key is not None
        else (owner_user_id or user.id)
    )
    from ..data.routing_strategy import routing_strategy_choices

    return {
        "user": user,
        "key": api_key,
        "services": services,
        "source_chips": source_chip_rows(db, services),
        "selected_services": selected_services,
        "teams": teams,
        "owners": owners,
        "catalog_groups": catalog_groups_for_ceiling(db, ceil),
        "selected_models": selected_models,
        "model_limits_text": (
            _format_model_limits(api_key.model_limits) if api_key else ""
        ),
        "nav": "keys",
        "is_admin": user.is_platform_admin,
        "teams_enabled": teams_on,
        "ceiling": ceil,
        "grant_summary": grant_summary(ceil),
        "grant_empty": (not ceil.unrestricted and not ceil.services),
        "default_owner_user_id": resolved_owner,
        "routing_strategies": routing_strategy_choices(include_inherit=True),
    }


def _teams_on(db: Session) -> bool:
    from .accounts import teams_feature_enabled

    return teams_feature_enabled(db)
