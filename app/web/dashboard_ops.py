"""Dashboard helpers: fleet health summary and actionable attention items."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from ..data.models import WebUser
from ..data.probe import ServiceStatus, probe_all
from ..mailer import get_smtp, smtp_ready
from .accounts import get_auth_settings
from .invites import pending_invites
from .setup import setup_status


@dataclass
class AttentionItem:
    severity: str  # error | warn | info
    title: str
    detail: str
    href: str
    cta: str


def fleet_statuses(db: Session) -> list[ServiceStatus]:
    return probe_all(db)


def fleet_summary(statuses: list[ServiceStatus]) -> dict[str, int]:
    counts = {"ok": 0, "loading": 0, "down": 0, "other": 0}
    for s in statuses:
        if s.state == "ok":
            counts["ok"] += 1
        elif s.state in {"loading", "busy"}:
            counts["loading"] += 1
        elif s.state == "down":
            counts["down"] += 1
        else:
            counts["other"] += 1
    return counts


def attention_items(
    db: Session,
    *,
    fleet: list[ServiceStatus],
    denies_24h: int = 0,
    rate_limits_24h: int = 0,
) -> list[AttentionItem]:
    items: list[AttentionItem] = []

    if not fleet:
        items.append(
            AttentionItem(
                severity="warn",
                title="No backends configured",
                detail="Add chat/embed/extractor/stt/tts sources before routing traffic.",
                href="/services",
                cta="Add sources",
            )
        )
    for s in fleet:
        if s.state == "down":
            items.append(
                AttentionItem(
                    severity="error",
                    title=f"Source down · {s.service}",
                    detail=s.detail or s.backend or "Probe failed",
                    href="/services",
                    cta="Services",
                )
            )
        elif s.state in {"loading", "busy"}:
            items.append(
                AttentionItem(
                    severity="warn",
                    title=f"Source loading · {s.service}",
                    detail=s.detail or "Model may still be loading",
                    href="/services",
                    cta="Services",
                )
            )

    setup = setup_status(db)
    todo_step = next((s for s in setup["steps"] if not s.done), None)
    if todo_step is not None:
        items.append(
            AttentionItem(
                severity="info",
                title=f"Setup · {todo_step.title}",
                detail=todo_step.detail,
                href=todo_step.href,
                cta=todo_step.cta,
            )
        )

    if not smtp_ready(get_smtp(db)):
        items.append(
            AttentionItem(
                severity="info",
                title="SMTP not configured",
                detail="Password-reset and invite emails won't send until SMTP is ready.",
                href="/smtp",
                cta="Configure SMTP",
            )
        )

    invites = pending_invites(db)
    if invites:
        n = len(invites)
        items.append(
            AttentionItem(
                severity="info",
                title=f"{n} open invite{'s' if n != 1 else ''}",
                detail="One-time signup links waiting to be used or revoked.",
                href="/users",
                cta="Users",
            )
        )

    auth = get_auth_settings(db)
    pool_window = int(getattr(auth, "pool_window_hours", 0) or 0)
    users = (
        db.query(WebUser)
        .filter(
            WebUser.is_platform_admin.is_(False),
            WebUser.is_active.is_(True),
            WebUser.pool_limit.isnot(None),
            WebUser.pool_limit > 0,
        )
        .all()
    )
    for u in users:
        limit = int(u.pool_limit or 0)
        used = float(u.pool_used or 0)
        if limit <= 0:
            continue
        pct = used / limit
        if pct >= 0.85:
            win = f" / {pool_window}h window" if pool_window else ""
            items.append(
                AttentionItem(
                    severity="warn" if pct >= 0.95 else "info",
                    title=f"Pool budget · {u.username}",
                    detail=f"{int(used):,} / {limit:,} tokens used{win}",
                    href="/users",
                    cta="Users",
                )
            )

    if denies_24h > 0:
        items.append(
            AttentionItem(
                severity="warn" if denies_24h >= 10 else "info",
                title=f"{denies_24h:,} denied request{'s' if denies_24h != 1 else ''} · 24h",
                detail="Auth failures, quota, or model allowlist blocks.",
                href="/usage?result=deny",
                cta="Usage",
            )
        )
    if rate_limits_24h > 0:
        items.append(
            AttentionItem(
                severity="warn" if rate_limits_24h >= 10 else "info",
                title=f"{rate_limits_24h:,} rate-limited · 24h",
                detail="HTTP 429 — RPM, concurrency, or daily quota.",
                href="/usage?result=rate_limit",
                cta="Usage",
            )
        )

    order = {"error": 0, "warn": 1, "info": 2}
    items.sort(key=lambda i: order.get(i.severity, 9))
    return items
