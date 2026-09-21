from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from ...data.db import get_db
from ...data.models import WebUser
from ..session import require_user
from ..shared import templates
from ..usage_pages import (
    OptionalQueryInt,
    build_daily_page_context,
    build_usage_page_context,
    usage_csv_response,
)

router = APIRouter()


@router.get("/usage", response_class=HTMLResponse)
def usage_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_user)],
    service: str | None = None,
    team_id: OptionalQueryInt = None,
    key_id: OptionalQueryInt = None,
    result: str | None = None,
    range: str | None = None,
):
    ctx = build_usage_page_context(
        request,
        db,
        user,
        ops_mode=False,
        service=service,
        team_id=team_id,
        key_id=key_id,
        result=result,
        range_raw=range,
    )
    return templates.TemplateResponse(request, "usage.html", ctx)


@router.get("/usage/daily", response_class=HTMLResponse)
def usage_daily_page(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_user)],
    service: str | None = None,
    team_id: OptionalQueryInt = None,
    key_id: OptionalQueryInt = None,
    range: str | None = None,
):
    ctx = build_daily_page_context(
        request,
        db,
        user,
        ops_mode=False,
        service=service,
        team_id=team_id,
        key_id=key_id,
        range_raw=range,
    )
    return templates.TemplateResponse(request, "usage_daily.html", ctx)


@router.get("/usage/export.csv")
def usage_export(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[WebUser, Depends(require_user)],
    service: str | None = None,
    team_id: OptionalQueryInt = None,
    key_id: OptionalQueryInt = None,
    result: str | None = None,
):
    return usage_csv_response(
        db,
        user,
        ops_mode=False,
        service=service,
        team_id=team_id,
        key_id=key_id,
        result=result,
    )
