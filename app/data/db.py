from __future__ import annotations

import hashlib
import secrets
from pathlib import Path

import bcrypt
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from ..config import KINDS, Settings
from .models import (
    WebUser,
    AlertConfig,
    ApiKey,
    AuthSettings,
    Base,
    ServiceGrant,
    SmtpConfig,
    Team,
    make_engine,
    make_session_factory,
)

engine = None
SessionLocal = None


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    return f"gw_{secrets.token_urlsafe(32)}"


def key_display_prefix(raw: str) -> str:
    """Safe UI prefix: gw_xxxx…xxxx (never middle of secret)."""
    if raw.startswith("gw_") and len(raw) > 12:
        return f"{raw[:7]}…{raw[-4:]}"
    digest = hash_api_key(raw)[:8]
    return f"key_{digest}"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def _ensure_columns(eng) -> None:
    """SQLite lightweight migrations for new columns."""
    specs = {
        "password_reset_tokens": [
            ("created_by_platform_admin", "BOOLEAN DEFAULT 0"),
        ],
        "web_users": [
            ("is_platform_admin", "BOOLEAN DEFAULT 0"),
            ("email", "VARCHAR(255)"),
            ("must_change_password", "BOOLEAN DEFAULT 0"),
            ("timezone", "VARCHAR(64)"),
            ("rpm_limit", "INTEGER"),
            ("concurrency_limit", "INTEGER"),
            ("daily_quota", "INTEGER"),
            ("pool_limit", "INTEGER"),
            ("pool_used", "FLOAT DEFAULT 0"),
            ("pool_window_start", "DATETIME"),
        ],
        "teams": [
            ("daily_quota", "INTEGER"),
            ("monthly_quota", "INTEGER"),
            ("routing_strategy", "VARCHAR(32) DEFAULT ''"),
            ("preferred_source", "VARCHAR(64) DEFAULT ''"),
        ],
        "usage_events": [
            ("tokens_in", "INTEGER"),
            ("tokens_out", "INTEGER"),
            ("audio_seconds", "FLOAT"),
            ("response_chars", "INTEGER"),
            ("is_demo", "BOOLEAN DEFAULT 0"),
            ("watts", "FLOAT"),
            ("watt_hours", "FLOAT"),
            ("pool_cost", "FLOAT"),
            ("power_status", "VARCHAR(32) DEFAULT ''"),
            ("pp_tok_s", "FLOAT"),
            ("tg_tok_s", "FLOAT"),
        ],
        "usage_daily": [
            ("tokens_in", "INTEGER DEFAULT 0"),
            ("tokens_out", "INTEGER DEFAULT 0"),
            ("audio_seconds", "FLOAT DEFAULT 0"),
            ("response_chars", "INTEGER DEFAULT 0"),
            ("latency_sum_ms", "FLOAT DEFAULT 0"),
            ("latency_count", "INTEGER DEFAULT 0"),
            ("watt_hours", "FLOAT DEFAULT 0"),
            ("pool_cost", "FLOAT DEFAULT 0"),
        ],
        "api_keys": [
            ("daily_quota", "INTEGER"),
            ("owner_user_id", "INTEGER"),
            ("routing_strategy", "VARCHAR(32)"),
            ("preferred_source", "VARCHAR(64)"),
        ],
        "auth_settings": [
            ("teams_enabled", "BOOLEAN DEFAULT 0"),
            ("allow_self_registration", "BOOLEAN DEFAULT 0"),
            ("require_email", "BOOLEAN DEFAULT 1"),
            ("default_team_id", "INTEGER"),
            ("anonymize_client_ip", "BOOLEAN DEFAULT 1"),
            ("retention_days", "INTEGER DEFAULT 30"),
            ("auto_vl_routing", "BOOLEAN DEFAULT 0"),
            ("auto_model_default", "VARCHAR(256) DEFAULT ''"),
            ("auto_model_quality", "VARCHAR(256) DEFAULT ''"),
            ("auto_model_long", "VARCHAR(256) DEFAULT ''"),
            ("max_keys_per_user", "INTEGER DEFAULT 3"),
            ("pool_window_hours", "INTEGER DEFAULT 5"),
            ("pool_tokens_per_unit", "INTEGER DEFAULT 1"),
            ("pool_min_cost", "FLOAT DEFAULT 1.0"),
            ("pool_watt_weight", "FLOAT DEFAULT 0"),
            ("pool_tokens_per_sec", "FLOAT DEFAULT 50"),
            ("pool_model_weights_enabled", "BOOLEAN DEFAULT 0"),
            ("preflight_upstream", "BOOLEAN DEFAULT 1"),
            ("load_aware_routing", "BOOLEAN DEFAULT 1"),
            ("routing_strategy", "VARCHAR(32) DEFAULT 'load_aware'"),
            ("source_admission_enabled", "BOOLEAN DEFAULT 1"),
            ("source_queue_timeout_sec", "INTEGER DEFAULT 30"),
            ("catalog_prune_on_sync", "BOOLEAN DEFAULT 1"),
            ("default_grant_sources", "TEXT DEFAULT ''"),
            ("default_grant_models", "TEXT DEFAULT ''"),
            ("show_global_stats", "BOOLEAN DEFAULT 0"),
            ("operator_name", "VARCHAR(255) DEFAULT ''"),
            ("operator_address", "TEXT DEFAULT ''"),
            ("operator_email", "VARCHAR(255) DEFAULT 'support@fr4iser.com'"),
            ("operator_phone", "VARCHAR(64) DEFAULT ''"),
        ],
        "audit_logs": [
            ("prev_hash", "VARCHAR(64) DEFAULT ''"),
            ("entry_hash", "VARCHAR(64) DEFAULT ''"),
        ],
        "backend_config": [
            ("chat", "VARCHAR(255) DEFAULT ''"),
            ("chat2", "VARCHAR(255) DEFAULT ''"),
        ],
        "backend_sources": [
            ("route_models", "TEXT DEFAULT ''"),
            ("api_style", "VARCHAR(32) DEFAULT 'auto'"),
            ("gpu_power_url", "VARCHAR(512) DEFAULT ''"),
            ("temp_guard_enabled", "BOOLEAN DEFAULT 1"),
            ("hardware", "VARCHAR(64) DEFAULT ''"),
            ("detected_engine", "VARCHAR(64) DEFAULT ''"),
            ("engine_override", "VARCHAR(64) DEFAULT ''"),
            ("max_concurrency", "INTEGER"),
            ("queue_timeout_sec", "INTEGER"),
        ],
        "service_grants": [
            ("user_id", "INTEGER"),
        ],
        "model_allowlists": [
            ("user_id", "INTEGER"),
        ],
        "catalog_models": [
            ("tags", "VARCHAR(512) DEFAULT ''"),
            ("short_note", "VARCHAR(512) DEFAULT ''"),
            ("docs_url", "VARCHAR(512) DEFAULT ''"),
            ("upstream_status", "VARCHAR(32) DEFAULT ''"),
            ("ctx_size", "INTEGER"),
            ("n_ctx", "INTEGER"),
            ("n_ctx_train", "INTEGER"),
            ("n_embd", "INTEGER"),
            ("n_params", "INTEGER"),
            ("model_size", "INTEGER"),
            ("modalities_in", "VARCHAR(128) DEFAULT ''"),
            ("modalities_out", "VARCHAR(128) DEFAULT ''"),
            ("upstream_meta_at", "DATETIME"),
            ("usage_weight", "FLOAT DEFAULT 1.0"),
            ("disabled_by", "VARCHAR(16) DEFAULT ''"),
        ],
        "model_aliases": [
            ("hide_candidates", "BOOLEAN DEFAULT 1"),
            ("family_prefix", "VARCHAR(128) DEFAULT ''"),
            ("kind", "VARCHAR(16) DEFAULT 'chat'"),
            ("sort_order", "INTEGER DEFAULT 0"),
            ("show_backend", "BOOLEAN DEFAULT 1"),
            ("enabled", "BOOLEAN DEFAULT 1"),
            ("description", "VARCHAR(512) DEFAULT ''"),
            ("preferred_source", "VARCHAR(64) DEFAULT ''"),
        ],
    }
    insp = inspect(eng)
    with eng.begin() as conn:
        for table, cols in specs.items():
            if table not in insp.get_table_names():
                continue
            existing = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in cols:
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))

        # Retired: model_favorites (removed feature).
        tables = set(inspect(eng).get_table_names())
        if "model_favorites" in tables:
            conn.execute(text("DROP TABLE model_favorites"))

        # Retired: source.isolated (grants decide access; do not re-add).
        if "backend_sources" in tables:
            src_cols = {c["name"] for c in inspect(eng).get_columns("backend_sources")}
            if "isolated" in src_cols:
                conn.execute(text("ALTER TABLE backend_sources DROP COLUMN isolated"))
        if "backend_config" in tables:
            cols = {c["name"] for c in inspect(eng).get_columns("backend_config")}
            if "chat" in cols:
                if "llm" in cols:
                    conn.execute(
                        text(
                            "UPDATE backend_config SET chat = llm "
                            "WHERE (chat IS NULL OR chat = '') AND llm IS NOT NULL AND llm != ''"
                        )
                    )
                if "ollama" in cols:
                    conn.execute(
                        text(
                            "UPDATE backend_config SET chat = ollama "
                            "WHERE (chat IS NULL OR chat = '') AND ollama IS NOT NULL AND ollama != ''"
                        )
                    )
        _migrate_service_name_to_chat(conn, tables)


def _migrate_service_name_to_chat(conn, tables: set[str]) -> None:
    """Fold llm/ollama into chat without UNIQUE collisions."""
    if "service_grants" in tables:
        rows = conn.execute(
            text(
                "SELECT id, api_key_id, team_id, service FROM service_grants "
                "WHERE service IN ('llm', 'ollama', 'chat')"
            )
        ).fetchall()
        # Prefer keeping an existing chat row; else keep one llm/ollama row to rename
        keep: set[int] = set()
        seen_key: set[int] = set()
        seen_team: set[int] = set()
        # Pass 1: mark chat rows as keepers
        for rid, key_id, team_id, svc in rows:
            if svc != "chat":
                continue
            keep.add(rid)
            if key_id is not None:
                seen_key.add(key_id)
            if team_id is not None:
                seen_team.add(team_id)
        # Pass 2: one legacy row per key/team if no chat yet (prefer llm over ollama)
        for prefer in ("llm", "ollama"):
            for rid, key_id, team_id, svc in rows:
                if svc != prefer or rid in keep:
                    continue
                if key_id is not None:
                    if key_id in seen_key:
                        continue
                    seen_key.add(key_id)
                    keep.add(rid)
                elif team_id is not None:
                    if team_id in seen_team:
                        continue
                    seen_team.add(team_id)
                    keep.add(rid)
        drop_ids = [rid for rid, _, _, svc in rows if svc in ("llm", "ollama") and rid not in keep]
        for rid in drop_ids:
            conn.execute(text("DELETE FROM service_grants WHERE id = :id"), {"id": rid})
        conn.execute(
            text("UPDATE service_grants SET service = 'chat' WHERE service IN ('llm', 'ollama')")
        )

    if "model_allowlists" in tables:
        rows = conn.execute(
            text(
                "SELECT id, api_key_id, team_id, service, model_name FROM model_allowlists "
                "WHERE service IN ('llm', 'ollama', 'chat')"
            )
        ).fetchall()
        keep = set()
        seen: set[tuple] = set()
        for rid, key_id, team_id, svc, model in rows:
            if svc != "chat":
                continue
            keep.add(rid)
            seen.add((key_id, team_id, model))
        for prefer in ("llm", "ollama"):
            for rid, key_id, team_id, svc, model in rows:
                if svc != prefer or rid in keep:
                    continue
                sig = (key_id, team_id, model)
                if sig in seen:
                    continue
                seen.add(sig)
                keep.add(rid)
        for rid, _, _, svc, _ in rows:
            if svc in ("llm", "ollama") and rid not in keep:
                conn.execute(text("DELETE FROM model_allowlists WHERE id = :id"), {"id": rid})
        conn.execute(
            text("UPDATE model_allowlists SET service = 'chat' WHERE service IN ('llm', 'ollama')")
        )

    if "model_limits" in tables:
        rows = conn.execute(
            text(
                "SELECT id, api_key_id, team_id, service, model_name FROM model_limits "
                "WHERE service IN ('llm', 'ollama', 'chat')"
            )
        ).fetchall()
        keep = set()
        seen: set[tuple] = set()
        for rid, key_id, team_id, svc, model in rows:
            if svc != "chat":
                continue
            keep.add(rid)
            seen.add((key_id, team_id, model))
        for prefer in ("llm", "ollama"):
            for rid, key_id, team_id, svc, model in rows:
                if svc != prefer or rid in keep:
                    continue
                sig = (key_id, team_id, model)
                if sig in seen:
                    continue
                seen.add(sig)
                keep.add(rid)
        for rid, _, _, svc, _ in rows:
            if svc in ("llm", "ollama") and rid not in keep:
                conn.execute(text("DELETE FROM model_limits WHERE id = :id"), {"id": rid})
        conn.execute(
            text("UPDATE model_limits SET service = 'chat' WHERE service IN ('llm', 'ollama')")
        )

    if "usage_events" in tables:
        conn.execute(
            text("UPDATE usage_events SET service = 'chat' WHERE service IN ('llm', 'ollama')")
        )

    if "usage_daily" in tables:
        rows = conn.execute(
            text(
                "SELECT id, day, api_key_id, team_id, service, model FROM usage_daily "
                "WHERE service IN ('llm', 'ollama', 'chat')"
            )
        ).fetchall()
        keep = set()
        seen: set[tuple] = set()
        for rid, day, key_id, team_id, svc, model in rows:
            if svc != "chat":
                continue
            keep.add(rid)
            seen.add((day, key_id, team_id, model))
        for prefer in ("llm", "ollama"):
            for rid, day, key_id, team_id, svc, model in rows:
                if svc != prefer or rid in keep:
                    continue
                sig = (day, key_id, team_id, model)
                if sig in seen:
                    continue
                seen.add(sig)
                keep.add(rid)
        for rid, _, _, _, svc, _ in rows:
            if svc in ("llm", "ollama") and rid not in keep:
                conn.execute(text("DELETE FROM usage_daily WHERE id = :id"), {"id": rid})
        conn.execute(
            text("UPDATE usage_daily SET service = 'chat' WHERE service IN ('llm', 'ollama')")
        )


def _ensure_audit_immutable(eng) -> None:
    """Block UPDATE/DELETE on audit_logs (append-only)."""
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TRIGGER IF NOT EXISTS audit_logs_no_update
                BEFORE UPDATE ON audit_logs
                BEGIN
                  SELECT RAISE(ABORT, 'audit_logs is immutable');
                END;
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TRIGGER IF NOT EXISTS audit_logs_no_delete
                BEFORE DELETE ON audit_logs
                BEGIN
                  SELECT RAISE(ABORT, 'audit_logs is immutable');
                END;
                """
            )
        )


def init_db(settings: Settings) -> None:
    global engine, SessionLocal
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    from ..config import DB_FILENAME

    db_path = data_dir / DB_FILENAME
    engine = make_engine(str(db_path))
    SessionLocal = make_session_factory(engine)
    Base.metadata.create_all(bind=engine)
    _ensure_columns(engine)
    _ensure_audit_immutable(engine)

    with SessionLocal() as db:
        bootstrap_platform_admin(db, settings)
        migrate_legacy_keys(db, settings)
        if db.query(AlertConfig).first() is None:
            db.add(AlertConfig(enabled=False, webhook_url=""))
        if db.query(SmtpConfig).first() is None:
            db.add(SmtpConfig(enabled=False))
        if db.query(AuthSettings).first() is None:
            db.add(
                AuthSettings(
                    allow_self_registration=False,
                    require_email=True,
                    teams_enabled=False,
                    default_team_id=None,
                    anonymize_client_ip=True,
                    retention_days=30,
                    auto_vl_routing=False,
                    auto_model_default="",
                    auto_model_quality="",
                    auto_model_long="",
                    max_keys_per_user=3,
                    pool_window_hours=5,
                    pool_tokens_per_unit=1,
                    pool_min_cost=1.0,
                    pool_watt_weight=0.0,
                    pool_tokens_per_sec=50.0,
                    pool_model_weights_enabled=False,
                    preflight_upstream=True,
                    load_aware_routing=True,
                    routing_strategy="load_aware",
                    source_admission_enabled=True,
                    source_queue_timeout_sec=30,
                    catalog_prune_on_sync=True,
                )
            )
        from .backends import seed_backends_from_env
        from ..privacy import purge_old_usage
        from ..web.accounts import get_auth_settings

        seed_backends_from_env(db, settings)
        auth = get_auth_settings(db)
        if auth.teams_enabled:
            _ensure_default_team_source_grants(db)
        else:
            prune_orphan_default_team(db)
        purged = purge_old_usage(db, auth.retention_days or 0)
        if purged:
            print(f"Purged {purged} usage events older than {auth.retention_days} days", flush=True)
        db.commit()


def get_db():
    assert SessionLocal is not None
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def bootstrap_platform_admin(db: Session, settings: Settings) -> None:
    """Ensure the bootstrap platform-admin account exists on empty DB."""
    user = db.query(WebUser).filter(WebUser.username == settings.admin_bootstrap_user).first()
    if user is None:
        db.add(
            WebUser(
                username=settings.admin_bootstrap_user,
                password_hash=hash_password(settings.admin_bootstrap_password),
                is_active=True,
                is_platform_admin=True,
            )
        )
    else:
        user.is_platform_admin = True


def _ensure_key(
    db: Session,
    *,
    label: str,
    raw_key: str,
    services: list[str],
) -> None:
    key_hash = hash_api_key(raw_key)
    existing = db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()
    if existing:
        existing.key_prefix = key_display_prefix(raw_key)
        return
    api_key = ApiKey(
        label=label,
        key_hash=key_hash,
        key_prefix=key_display_prefix(raw_key),
        is_active=True,
        priority=0,
    )
    db.add(api_key)
    db.flush()
    for service in services:
        db.add(ServiceGrant(api_key_id=api_key.id, service=service))


def migrate_legacy_keys(db: Session, settings: Settings) -> None:
    mapping = [
        ("migrated-chat", settings.llm_api_key or settings.ollama_api_key, ["chat"]),
        ("migrated-embed", settings.embed_api_key, ["embed"]),
        ("migrated-stt", settings.stt_api_key, ["stt"]),
        ("migrated-tts", settings.tts_api_key, ["tts"]),
    ]
    # Prefer separate migrated keys when both legacy keys exist
    if settings.llm_api_key and settings.ollama_api_key and settings.llm_api_key != settings.ollama_api_key:
        mapping = [
            ("migrated-chat", settings.llm_api_key, ["chat"]),
            ("migrated-chat-ollama", settings.ollama_api_key, ["chat"]),
            ("migrated-embed", settings.embed_api_key, ["embed"]),
            ("migrated-stt", settings.stt_api_key, ["stt"]),
            ("migrated-tts", settings.tts_api_key, ["tts"]),
        ]
    for label, raw, services in mapping:
        if raw:
            _ensure_key(db, label=label, raw_key=raw, services=services)


def prune_orphan_default_team(db: Session) -> None:
    """Remove leftover 'default' team when teams feature is off and unused."""
    from ..web.accounts import get_auth_settings

    auth = get_auth_settings(db)
    if auth.teams_enabled:
        return
    team = db.query(Team).filter(Team.name == "default").first()
    if team is None:
        return
    if db.query(ApiKey).filter(ApiKey.team_id == team.id).first():
        return
    db.delete(team)


def _ensure_default_team_source_grants(db: Session) -> None:
    from .grants import configured_default_sources

    team = db.query(Team).filter(Team.name == "default").first()
    if team is None:
        return
    existing = {g.service for g in team.service_grants}
    for name in configured_default_sources(db):
        if name not in existing:
            db.add(ServiceGrant(team_id=team.id, service=name))
