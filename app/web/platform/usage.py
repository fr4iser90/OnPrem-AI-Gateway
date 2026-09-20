"""Ops usage — fleet-wide energy and events (platform admin only)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from ...data.db import get_db
from ...data.models import WebUser
from ..session import require_platform_admin
from ..shared import templates
from ..usage_pages import (
    build_daily_page_context,
    build_usage_page_context,
    usage_csv_response,
)

router = APIRouter()


@router.get("/ops/usage", response_class=HTMLResponse)
def ops_usage_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
    service: str | None = None,
    team_id: int | None = None,
    key_id: int | None = None,
    result: str | None = None,
    range: str | None = None,
    owner_user_id: int | None = None,
):
    ctx = build_usage_page_context(
        request,
        db,
        user,
        ops_mode=True,
        service=service,
        team_id=team_id,
        key_id=key_id,
        result=result,
        range_raw=range,
        owner_user_id=owner_user_id,
    )
    return templates.TemplateResponse(request, "usage.html", ctx)


@router.get("/ops/usage/daily", response_class=HTMLResponse)
def ops_usage_daily_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
    service: str | None = None,
    team_id: int | None = None,
    key_id: int | None = None,
    range: str | None = None,
    owner_user_id: int | None = None,
):
    ctx = build_daily_page_context(
        request,
        db,
        user,
        ops_mode=True,
        service=service,
        team_id=team_id,
        key_id=key_id,
        range_raw=range,
        owner_user_id=owner_user_id,
    )
    return templates.TemplateResponse(request, "usage_daily.html", ctx)


@router.get("/ops/usage/export.csv")
def ops_usage_export(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_platform_admin)],
    service: str | None = None,
    team_id: int | None = None,
    key_id: int | None = None,
    result: str | None = None,
    owner_user_id: int | None = None,
):
    return usage_csv_response(
        db,
        user,
        ops_mode=True,
        service=service,
        team_id=team_id,
        key_id=key_id,
        result=result,
        owner_user_id=owner_user_id,
    )
