from __future__ import annotations

from fastapi import Depends, Request
from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload
from typing import Annotated

from ..data.db import get_db
from ..data.models import WebUser, ApiKey, Team, TeamMember


class RedirectToLogin(Exception):
    pass


class Forbidden(Exception):
    pass


class SetupWizardRequired(Exception):
    """Platform admin must finish first-run wizard before using the rest of the UI."""

    def __init__(self, redirect_to: str):
        self.redirect_to = redirect_to


def _setup_path_allowed(path: str) -> bool:
    """Routes reachable while the first-run wizard is incomplete."""
    p = path or "/"
    if p.startswith("/setup"):
        return True
    if p.startswith("/account"):
        return True
    if p.startswith("/static"):
        return True
    if p.startswith("/legal"):
        return True
    if p in {"/logout", "/login", "/forgot", "/register", "/healthz"}:
        return True
    return False


def current_user(
    request: Request, db: Annotated[Session, Depends(get_db)]
) -> WebUser | None:
    uid = request.session.get("user_id")
    if not uid:
        return None
    user = (
        db.query(WebUser)
        .options(joinedload(WebUser.memberships).joinedload(TeamMember.team))
        .filter(WebUser.id == uid)
        .first()
    )
    if user is None or not user.is_active:
        return None
    return user


def require_user(
    request: Request, db: Annotated[Session, Depends(get_db)]
) -> WebUser:
    user = current_user(request, db)
    if user is None:
        raise RedirectToLogin()

    # First-login onboarding: password change and/or username claim.
    from .accounts import user_needs_onboarding

    if user_needs_onboarding(user) and not _setup_path_allowed(request.url.path):
        raise SetupWizardRequired("/account")

    # First-run: lock platform ops UI until sources → models → key are done.
    if user.is_platform_admin:
        from .setup import needs_setup_wizard, wizard_progress

        if needs_setup_wizard(db, user):
            request.state.setup_incomplete = True
            if not _setup_path_allowed(request.url.path):
                nxt = wizard_progress(db)["next"]
                raise SetupWizardRequired((nxt["path"] if nxt else "/setup"))
        else:
            request.state.setup_incomplete = False

    return user


def require_platform_admin(user: Annotated[WebUser, Depends(require_user)]) -> WebUser:
    if not user.is_platform_admin:
        raise Forbidden()
    return user


def user_team_ids(user: WebUser) -> set[int]:
    return {m.team_id for m in user.memberships}


def owned_team_ids(user: WebUser) -> set[int]:
    return {m.team_id for m in user.memberships if m.role == "owner"}


def user_teams(user: WebUser) -> list[Team]:
    return [m.team for m in user.memberships if m.team]


def can_access_team(user: WebUser, team_id: int | None) -> bool:
    if user.is_platform_admin:
        return True
    if team_id is None:
        return False
    return team_id in user_team_ids(user)


def can_access_key(user: WebUser, api_key, *, teams_enabled: bool) -> bool:
    """Keys: own keys always; team owners also see every key on that team.

    Platform admins may access any key (Ops → Users → manage that user's keys).
    """
    if user.is_platform_admin:
        return True
    if api_key.owner_user_id == user.id:
        return True
    if teams_enabled and api_key.team_id is not None:
        return api_key.team_id in owned_team_ids(user)
    return False


def scope_keys_query(
    q,
    user: WebUser,
    *,
    teams_enabled: bool,
    owner_user_id: int | None = None,
):
    """Scope key lists for the portal.

    You (/keys, /me): always the caller's own keys (admins included) — never the
    whole fleet. Ops managing another user passes owner_user_id (admin only).
    """
    if user.is_platform_admin and owner_user_id is not None:
        return q.filter(ApiKey.owner_user_id == int(owner_user_id))
    if not teams_enabled:
        return q.filter(ApiKey.owner_user_id == user.id)
    owned = owned_team_ids(user)
    if owned:
        return q.filter(
            or_(ApiKey.owner_user_id == user.id, ApiKey.team_id.in_(owned))
        )
    return q.filter(ApiKey.owner_user_id == user.id)


def scoped_key_ids(db: Session, user: WebUser, *, teams_enabled: bool) -> list[int]:
    rows = scope_keys_query(db.query(ApiKey.id), user, teams_enabled=teams_enabled).all()
    return [int(kid) for (kid,) in rows]


def is_team_owner(user: WebUser, team_id: int) -> bool:
    if user.is_platform_admin:
        return True
    for m in user.memberships:
        if m.team_id == team_id and m.role == "owner":
            return True
    return False
