"""Demo users + usage for local UI review."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy.orm import Session

from .audit import bump_usage_daily
from .data.db import generate_api_key, hash_api_key, hash_password, key_display_prefix
from .data.grants import sync_user_grants
from .data.models import WebUser, ApiKey, UsageDaily, UsageEvent, utcnow

SERVICES_DEMO = [
    ("chat", ["/v1/chat/completions", "llama3.2"], "tokens"),
    ("embed", ["/v1/embeddings", "nomic-embed-text"], "tokens"),
    ("extractor", ["/v1/extract", "agents-k1"], "tokens"),
    ("stt", ["/v1/audio/transcriptions", "whisper-large"], "audio"),
    ("tts", ["/v1/audio/speech", "piper-en"], "tts"),
]

RESULTS = ["ok"] * 85 + ["deny"] * 8 + ["rate_limit"] * 7

DEMO_PASSWORD = "Demo-user-123!"
DEMO_PREFIX = "demo-"


@dataclass(frozen=True)
class DemoUserSpec:
    username: str
    email: str
    services: tuple[str, ...] | None  # None = all configured sources
    weight: float
    must_change_password: bool = False


DEMO_USERS: tuple[DemoUserSpec, ...] = (
    DemoUserSpec("demo-alice", "alice@demo.local", ("chat", "embed"), 1.0),
    DemoUserSpec("demo-bob", "bob@demo.local", ("chat", "embed", "stt"), 2.5),
    DemoUserSpec("demo-heavy", "heavy@demo.local", None, 5.0),
    DemoUserSpec("demo-new", "new@demo.local", ("chat",), 0.4, must_change_password=True),
)


def _demo_usernames() -> list[str]:
    return [spec.username for spec in DEMO_USERS]


def ensure_demo_users(db: Session, *, source_names: list[str]) -> tuple[list[WebUser], list[ApiKey]]:
    users: list[WebUser] = []
    keys: list[ApiKey] = []
    for spec in DEMO_USERS:
        user = db.query(WebUser).filter(WebUser.username == spec.username).first()
        if user is None:
            user = WebUser(
                username=spec.username,
                email=spec.email,
                password_hash=hash_password(DEMO_PASSWORD),
                is_active=True,
                is_platform_admin=False,
                must_change_password=spec.must_change_password,
            )
            db.add(user)
            db.flush()
        else:
            user.email = spec.email
            user.is_active = True
            user.is_platform_admin = False
            user.must_change_password = spec.must_change_password
            if not user.password_hash:
                user.password_hash = hash_password(DEMO_PASSWORD)

        services = list(spec.services) if spec.services is not None else list(source_names)
        services = [s for s in services if s in source_names]
        if services:
            sync_user_grants(db, user, services)

        key = (
            db.query(ApiKey)
            .filter(ApiKey.owner_user_id == user.id, ApiKey.label == "demo")
            .first()
        )
        if key is None:
            raw = generate_api_key()
            key = ApiKey(
                label="demo",
                key_hash=hash_api_key(raw),
                key_prefix=key_display_prefix(raw),
                owner_user_id=user.id,
                is_active=True,
            )
            db.add(key)
            db.flush()
        else:
            key.is_active = True

        users.append(user)
        keys.append(key)
    return users, keys


def clear_demo_usage(db: Session) -> int:
    n = db.query(UsageEvent).filter(UsageEvent.is_demo.is_(True)).delete()
    _recompute_daily(db, days=14)
    return int(n or 0)


def clear_demo_users(db: Session) -> int:
    removed = 0
    for username in _demo_usernames():
        user = db.query(WebUser).filter(WebUser.username == username).first()
        if user is None:
            continue
        db.query(ApiKey).filter(ApiKey.owner_user_id == user.id).delete()
        db.delete(user)
        removed += 1
    return removed


def clear_demo_world(db: Session) -> dict[str, int]:
    return {
        "events": clear_demo_usage(db),
        "users": clear_demo_users(db),
    }


def _recompute_daily(db: Session, *, days: int = 14) -> None:
    cutoff = utcnow() - timedelta(days=days)
    db.query(UsageDaily).filter(UsageDaily.day >= cutoff.date()).delete()
    events = (
        db.query(UsageEvent)
        .filter(UsageEvent.created_at >= cutoff)
        .order_by(UsageEvent.created_at.asc())
        .all()
    )
    for e in events:
        bump_usage_daily(
            db,
            team_id=e.team_id,
            api_key_id=e.api_key_id,
            team_name=e.team_name,
            key_label=e.key_label,
            service=e.service,
            model=e.model,
            result=e.result,
            tokens_in=e.tokens_in,
            tokens_out=e.tokens_out,
            audio_seconds=e.audio_seconds,
            response_chars=e.response_chars,
            duration_ms=e.duration_ms,
            day=e.created_at.date() if e.created_at else None,
        )


def _weighted_key(keys: list[ApiKey], weights: dict[int, float]) -> ApiKey | None:
    if not keys:
        return None
    bag: list[ApiKey] = []
    for key in keys:
        w = max(0.1, float(weights.get(key.id, 1.0)))
        bag.extend([key] * max(1, int(w * 10)))
    return random.choice(bag)


def seed_demo_usage(
    db: Session,
    *,
    count: int = 120,
    keys: list[ApiKey] | None = None,
    key_weights: dict[int, float] | None = None,
) -> int:
    """Insert synthetic usage events tagged is_demo=True (replaces prior demo)."""
    clear_demo_usage(db)
    keys = keys or db.query(ApiKey).filter(ApiKey.is_active.is_(True)).all()
    if not keys:
        keys = [None]  # type: ignore[list-item]

    now = utcnow()
    created = 0
    for i in range(count):
        svc, (path, model), kind = random.choice(SERVICES_DEMO)
        if key_weights:
            key = _weighted_key([k for k in keys if k is not None], key_weights)
        else:
            key = random.choice(keys)
        result = random.choice(RESULTS)
        if i < 21:
            day_offset = i % 7
            age_h = day_offset * 24 + random.uniform(1, 20)
        else:
            age_h = random.uniform(0, 24 * 6.5)
        created_at = now - timedelta(hours=age_h)
        duration = random.uniform(40, 2800) if result == "ok" else random.uniform(5, 120)

        tokens_in = tokens_out = None
        audio_seconds = response_chars = None
        if result == "ok":
            if kind == "tokens":
                tokens_in = random.randint(80, 2500)
                tokens_out = random.randint(20, 900) if svc != "embed" else 0
            elif kind == "audio":
                audio_seconds = round(random.uniform(1.5, 90.0), 2)
            elif kind == "tts":
                response_chars = random.randint(40, 1200)
                audio_seconds = round(response_chars / 14.0, 2)

        ip = f"10.0.{random.randint(0, 5)}.{random.randint(1, 254)}"
        db.add(
            UsageEvent(
                created_at=created_at,
                api_key_id=key.id if key else None,
                team_id=key.team_id if key else None,
                key_label=key.label if key else "demo-key",
                team_name=key.team.name if key and key.team else "",
                service=svc,
                method="POST",
                path=path,
                host=f"{svc}.demo.local",
                client_ip=ip,
                model=model,
                status=204 if result == "ok" else (429 if result == "rate_limit" else 403),
                result=result,
                duration_ms=duration,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                audio_seconds=audio_seconds,
                response_chars=response_chars,
                is_demo=True,
            )
        )
        created += 1
        bump_usage_daily(
            db,
            team_id=key.team_id if key else None,
            api_key_id=key.id if key else None,
            team_name=key.team.name if key and key.team else "",
            key_label=key.label if key else "demo-key",
            service=svc,
            model=model,
            result=result,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            audio_seconds=audio_seconds,
            response_chars=response_chars,
            duration_ms=duration,
            day=created_at.date(),
        )
    return created


def seed_demo_world(db: Session, *, count: int = 280) -> dict:
    """Create demo users/keys and spread tagged usage across them."""
    from .web.accounts import get_auth_settings
    from .data.backends import source_names

    names = source_names(db)
    users, keys = ensure_demo_users(db, source_names=names)
    weights = {
        keys[i].id: spec.weight
        for i, spec in enumerate(DEMO_USERS)
        if i < len(keys)
    }
    events = seed_demo_usage(db, count=count, keys=keys, key_weights=weights)

    auth = get_auth_settings(db)
    auth.show_global_stats = True

    return {
        "events": events,
        "users": [
            {
                "username": spec.username,
                "password": DEMO_PASSWORD,
                "must_change_password": spec.must_change_password,
            }
            for spec in DEMO_USERS
        ],
    }
