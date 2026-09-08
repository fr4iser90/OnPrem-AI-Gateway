"""First-run setup checklist + guided wizard."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session, joinedload

from ..data.backends import list_sources
from ..data.models import WebUser, ApiKey, AuthSettings, CatalogModel, Team
from .accounts import teams_feature_enabled

# Wizard: sources → default access → admin API key (catalog syncs on Save backends)
WIZARD_STEP_IDS = ("sources", "access", "key")


@dataclass
class SetupStep:
    id: str
    title: str
    detail: str
    href: str
    done: bool
    cta: str


def _addressed_sources(db: Session) -> list:
    return [s for s in list_sources(db) if (s.address or "").strip()]


def wizard_progress(db: Session) -> dict:
    """Minimum path for a working OnPrem AI Gateway install (admin solo)."""
    addressed = _addressed_sources(db)
    catalog_n = db.query(CatalogModel).count()
    active_keys = db.query(ApiKey).filter(ApiKey.is_active.is_(True)).count()
    has_sources = bool(addressed)
    has_models = catalog_n > 0
    auth = db.query(AuthSettings).first()
    has_access = bool(auth and (auth.default_grant_sources or "").strip())
    has_keys = active_keys > 0
    steps = [
        {
            "id": "sources",
            "num": 1,
            "title": "Backends",
            "done": has_sources,
            "path": "/setup/sources",
        },
        {
            "id": "access",
            "num": 2,
            "title": "Default access",
            "done": has_access,
            "path": "/setup/access",
        },
        {
            "id": "key",
            "num": 3,
            "title": "API key",
            "done": has_keys,
            "path": "/setup/key",
        },
    ]
    next_step = next((s for s in steps if not s["done"]), None)
    return {
        "steps": steps,
        "has_sources": has_sources,
        "has_models": has_models,
        "has_access": has_access,
        "has_keys": has_keys,
        "catalog_n": catalog_n,
        "source_count": len(addressed),
        "complete": has_sources and has_keys,
        "next": next_step,
        "sources": addressed,
    }


def needs_setup_wizard(db: Session, user: WebUser) -> bool:
    """Platform admins must finish the wizard once on a fresh install."""
    if not user.is_platform_admin:
        return False
    return not wizard_progress(db)["complete"]


def setup_status(db: Session) -> dict:
    sources = list_sources(db)
    addressed = [s for s in sources if (s.address or "").strip()]
    catalog_n = db.query(CatalogModel).count()
    teams_on = teams_feature_enabled(db)
    active_keys = db.query(ApiKey).filter(ApiKey.is_active.is_(True)).count()
    wiz = wizard_progress(db)

    if teams_on:
        teams = (
            db.query(Team)
            .options(joinedload(Team.service_grants), joinedload(Team.members))
            .all()
        )
        grant_ok = any(t.service_grants for t in teams)
        grant_detail = (
            "Create a team, set Grant · sources (+ optional models), add members."
            if not grant_ok
            else f"{sum(1 for t in teams if t.service_grants)} team(s) with grants."
        )
        grant_href = "/teams"
        grant_cta = "Open teams"
    else:
        users = (
            db.query(WebUser)
            .options(joinedload(WebUser.service_grants))
            .filter(WebUser.is_platform_admin.is_(False))
            .all()
        )
        extra_users = len(users)
        granted = sum(1 for u in users if u.service_grants)
        if extra_users == 0:
            grant_ok = True
            grant_detail = (
                "Optional: add users under Users → Edit grant. "
                "You (admin) already have full access for your own keys."
            )
        else:
            grant_ok = granted > 0
            grant_detail = (
                f"{granted}/{extra_users} non-admin user(s) have a grant."
                if grant_ok
                else "Non-admin users need Users → Edit grant (sources + models)."
            )
        grant_href = "/users"
        grant_cta = "Users / grants"

    steps = [
        SetupStep(
            id="sources",
            title="1 · Sources",
            detail=(
                f"{len(addressed)} source(s) with address."
                if addressed
                else "No backends yet — add chat/embed/extractor/stt/tts addresses."
            ),
            href="/setup/sources" if not wiz["has_sources"] else "/services",
            done=bool(addressed),
            cta="Add sources",
        ),
        SetupStep(
            id="models",
            title="2 · Models",
            detail=(
                f"{catalog_n} model(s) in catalog."
                if catalog_n
                else "Sync from Services later, or wait — wizard syncs on Save backends."
            ),
            href="/models",
            done=catalog_n > 0,
            cta="Open models",
        ),
        SetupStep(
            id="keys",
            title="3 · API key",
            detail=(
                f"{active_keys} active key(s)."
                if active_keys
                else "Create a key for yourself."
            ),
            href="/setup/key" if not wiz["has_keys"] else "/keys",
            done=active_keys > 0,
            cta="Create key",
        ),
        SetupStep(
            id="grants",
            title="4 · Grants (optional)",
            detail=grant_detail,
            href=grant_href,
            done=grant_ok,
            cta=grant_cta,
        ),
    ]

    done_n = sum(1 for s in steps if s.done)
    return {
        "steps": steps,
        "done_n": done_n,
        "total_n": len(steps),
        "complete": wiz["complete"],  # wizard minimum; grants optional
        "wizard": wiz,
        "teams_on": teams_on,
        "has_sources": bool(addressed),
        "catalog_n": catalog_n,
    }
