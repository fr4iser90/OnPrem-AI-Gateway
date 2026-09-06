from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class WebUser(Base):
    __tablename__ = "web_users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_platform_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False)
    # IANA name cached from browser detection (cookie gw_tz). Not a manual setting.
    timezone: Mapped[str | None] = mapped_column(String(64), nullable=True, default=None)
    # Rate ceiling across all of this user's keys (None = no user-level cap).
    rpm_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    concurrency_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    daily_quota: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Usage pool (compute units). None limit = pool off for this user.
    pool_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pool_used: Mapped[float] = mapped_column(Float, default=0.0)
    pool_window_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    memberships: Mapped[list[TeamMember]] = relationship(back_populates="user")
    service_grants: Mapped[list[ServiceGrant]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    model_allowlists: Mapped[list[ModelAllowlist]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    rpm_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    concurrency_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    daily_quota: Mapped[int | None] = mapped_column(Integer, nullable=True)
    monthly_quota: Mapped[int | None] = mapped_column(Integer, nullable=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    # Empty = inherit platform routing_strategy from auth_settings.
    routing_strategy: Mapped[str] = mapped_column(String(32), default="")
    # Pin tied-route picks to this source name when the model exists there.
    preferred_source: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    members: Mapped[list[TeamMember]] = relationship(
        back_populates="team", cascade="all, delete-orphan"
    )
    keys: Mapped[list[ApiKey]] = relationship(back_populates="team")
    service_grants: Mapped[list[ServiceGrant]] = relationship(
        back_populates="team", cascade="all, delete-orphan"
    )
    model_allowlists: Mapped[list[ModelAllowlist]] = relationship(
        back_populates="team", cascade="all, delete-orphan"
    )
    model_limits: Mapped[list[ModelLimit]] = relationship(
        back_populates="team", cascade="all, delete-orphan"
    )


class TeamMember(Base):
    __tablename__ = "team_members"
    __table_args__ = (UniqueConstraint("team_id", "user_id", name="uq_team_user"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id", ondelete="CASCADE"))
    user_id: Mapped[int] = mapped_column(ForeignKey("web_users.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(32), default="member")  # owner|member

    team: Mapped[Team] = relationship(back_populates="members")
    user: Mapped[WebUser] = relationship(back_populates="memberships")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String(128))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    key_prefix: Mapped[str] = mapped_column(String(32), default="")
    team_id: Mapped[int | None] = mapped_column(
        ForeignKey("teams.id", ondelete="SET NULL"), nullable=True
    )
    owner_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("web_users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rpm_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    concurrency_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    daily_quota: Mapped[int | None] = mapped_column(Integer, nullable=True)
    priority: Mapped[int | None] = mapped_column(Integer, nullable=True)
    routing_strategy: Mapped[str | None] = mapped_column(String(32), nullable=True)
    preferred_source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Soft-fill profile when client omits sampling: "" = catalog default_profile.
    sampling_profile: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    team: Mapped[Team | None] = relationship(back_populates="keys")
    owner: Mapped[WebUser | None] = relationship(foreign_keys=[owner_user_id])
    service_grants: Mapped[list[ServiceGrant]] = relationship(
        back_populates="api_key", cascade="all, delete-orphan"
    )
    model_allowlists: Mapped[list[ModelAllowlist]] = relationship(
        back_populates="api_key", cascade="all, delete-orphan"
    )
    model_limits: Mapped[list[ModelLimit]] = relationship(
        back_populates="api_key", cascade="all, delete-orphan"
    )


class ServiceGrant(Base):
    __tablename__ = "service_grants"
    __table_args__ = (
        UniqueConstraint("api_key_id", "service", name="uq_key_service"),
        UniqueConstraint("team_id", "service", name="uq_team_service"),
        UniqueConstraint("user_id", "service", name="uq_user_service"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    service: Mapped[str] = mapped_column(String(64), index=True)
    api_key_id: Mapped[int | None] = mapped_column(
        ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=True
    )
    team_id: Mapped[int | None] = mapped_column(
        ForeignKey("teams.id", ondelete="CASCADE"), nullable=True
    )
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("web_users.id", ondelete="CASCADE"), nullable=True
    )

    api_key: Mapped[ApiKey | None] = relationship(back_populates="service_grants")
    team: Mapped[Team | None] = relationship(back_populates="service_grants")
    user: Mapped[WebUser | None] = relationship(back_populates="service_grants")


class ModelAllowlist(Base):
    __tablename__ = "model_allowlists"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    service: Mapped[str] = mapped_column(String(64), index=True)
    model_name: Mapped[str] = mapped_column(String(256))
    api_key_id: Mapped[int | None] = mapped_column(
        ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=True
    )
    team_id: Mapped[int | None] = mapped_column(
        ForeignKey("teams.id", ondelete="CASCADE"), nullable=True
    )
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("web_users.id", ondelete="CASCADE"), nullable=True
    )

    api_key: Mapped[ApiKey | None] = relationship(back_populates="model_allowlists")
    team: Mapped[Team | None] = relationship(back_populates="model_allowlists")
    user: Mapped[WebUser | None] = relationship(back_populates="model_allowlists")


class ModelLimit(Base):
    """Per-model RPM / concurrency / daily quota for a team or key."""

    __tablename__ = "model_limits"
    __table_args__ = (
        UniqueConstraint(
            "api_key_id", "service", "model_name", name="uq_key_model_limit"
        ),
        UniqueConstraint(
            "team_id", "service", "model_name", name="uq_team_model_limit"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    service: Mapped[str] = mapped_column(String(64), index=True)
    model_name: Mapped[str] = mapped_column(String(256), index=True)
    rpm_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    concurrency_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    daily_quota: Mapped[int | None] = mapped_column(Integer, nullable=True)
    api_key_id: Mapped[int | None] = mapped_column(
        ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=True
    )
    team_id: Mapped[int | None] = mapped_column(
        ForeignKey("teams.id", ondelete="CASCADE"), nullable=True
    )

    api_key: Mapped[ApiKey | None] = relationship(back_populates="model_limits")
    team: Mapped[Team | None] = relationship(back_populates="model_limits")


class UsageEvent(Base):
    __tablename__ = "usage_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    api_key_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    team_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    key_label: Mapped[str] = mapped_column(String(128), default="")
    team_name: Mapped[str] = mapped_column(String(128), default="")
    service: Mapped[str] = mapped_column(String(32), index=True)
    method: Mapped[str] = mapped_column(String(16), default="")
    path: Mapped[str] = mapped_column(String(512), default="")
    host: Mapped[str] = mapped_column(String(256), default="")
    client_ip: Mapped[str] = mapped_column(String(64), default="")
    model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    status: Mapped[int] = mapped_column(Integer, default=204)
    result: Mapped[str] = mapped_column(String(32), default="ok")
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    audio_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    response_chars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Estimated GPU sample at auth (optional sidecar)
    watts: Mapped[float | None] = mapped_column(Float, nullable=True)
    watt_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    # metered | no_probe | unreachable | ""
    power_status: Mapped[str] = mapped_column(String(32), default="")
    # llama.cpp timings / derived throughput (tok/s)
    pp_tok_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    tg_tok_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    pool_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)


class UsageDaily(Base):
    __tablename__ = "usage_daily"
    __table_args__ = (
        UniqueConstraint(
            "day",
            "team_id",
            "api_key_id",
            "service",
            "model",
            name="uq_usage_daily",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    day: Mapped[datetime] = mapped_column(Date, index=True)
    team_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    api_key_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    team_name: Mapped[str] = mapped_column(String(128), default="")
    key_label: Mapped[str] = mapped_column(String(128), default="")
    service: Mapped[str] = mapped_column(String(32), default="")
    model: Mapped[str] = mapped_column(String(256), default="")
    ok_count: Mapped[int] = mapped_column(Integer, default=0)
    deny_count: Mapped[int] = mapped_column(Integer, default=0)
    rate_limit_count: Mapped[int] = mapped_column(Integer, default=0)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    audio_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    response_chars: Mapped[int] = mapped_column(Integer, default=0)
    latency_sum_ms: Mapped[float] = mapped_column(Float, default=0.0)
    latency_count: Mapped[int] = mapped_column(Integer, default=0)
    watt_hours: Mapped[float] = mapped_column(Float, default=0.0)
    pool_cost: Mapped[float] = mapped_column(Float, default=0.0)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    actor_user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actor_username: Mapped[str] = mapped_column(String(128), default="")
    action: Mapped[str] = mapped_column(String(64), index=True)
    entity_type: Mapped[str] = mapped_column(String(64), default="")
    entity_id: Mapped[str] = mapped_column(String(64), default="")
    detail: Mapped[str] = mapped_column(Text, default="")
    prev_hash: Mapped[str] = mapped_column(String(64), default="")
    entry_hash: Mapped[str] = mapped_column(String(64), default="", index=True)


class AlertConfig(Base):
    __tablename__ = "alert_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    webhook_url: Mapped[str] = mapped_column(Text, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    alert_on_quota: Mapped[bool] = mapped_column(Boolean, default=True)
    alert_on_rate_limit: Mapped[bool] = mapped_column(Boolean, default=False)
    quota_warn_pct: Mapped[int] = mapped_column(Integer, default=80)


class SmtpConfig(Base):
    __tablename__ = "smtp_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    host: Mapped[str] = mapped_column(String(255), default="")
    port: Mapped[int] = mapped_column(Integer, default=587)
    username: Mapped[str] = mapped_column(String(255), default="")
    password: Mapped[str] = mapped_column(Text, default="")
    from_email: Mapped[str] = mapped_column(String(255), default="")
    from_name: Mapped[str] = mapped_column(String(255), default="OnPrem AI Gateway")
    use_tls: Mapped[bool] = mapped_column(Boolean, default=True)
    use_ssl: Mapped[bool] = mapped_column(Boolean, default=False)
    public_base_url: Mapped[str] = mapped_column(String(512), default="")  # https://ai.example.com


class BackendConfig(Base):
    """Legacy singleton row — migrated once into BackendSource. Do not use in new code."""

    __tablename__ = "backend_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat: Mapped[str] = mapped_column(String(255), default="")
    chat2: Mapped[str] = mapped_column(String(255), default="")
    embed: Mapped[str] = mapped_column(String(255), default="")
    stt: Mapped[str] = mapped_column(String(255), default="")
    tts: Mapped[str] = mapped_column(String(255), default="")


class BackendSource(Base):
    """Named upstream: many sources per kind (chat/embed/stt/tts)."""

    __tablename__ = "backend_sources"
    __table_args__ = (UniqueConstraint("name", name="uq_backend_source_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)  # chat|embed|stt|tts
    address: Mapped[str] = mapped_column(String(255), default="")
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    # Newline-separated model patterns for /v1 routing (exact, prefix*, or name → name:tag)
    route_models: Mapped[str] = mapped_column(Text, default="")
    # auto|openai|piper|whisper_cpp — how to map /v1 client paths to upstream
    api_style: Mapped[str] = mapped_column(String(32), default="auto")
    # Optional source-sidecar is always co-located: http://<address-host>:9105/power
    # (column kept for migrations; not used for overrides)
    gpu_power_url: Mapped[str] = mapped_column(String(512), default="")
    # Thermal guard check is also co-located per source: http://<address-host>:9105/check
    temp_guard_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # Optional operator label: "Strix Halo", "Jetson Orin Super" — display only.
    hardware: Mapped[str] = mapped_column(String(64), default="")
    # Last probe fingerprint (llama.cpp, ollama, vllm, …). Empty until probed.
    detected_engine: Mapped[str] = mapped_column(String(64), default="")
    # Admin override; empty = auto (use detected_engine or live discover).
    engine_override: Mapped[str] = mapped_column(String(64), default="")
    # Gateway admission slots for this source. NULL = auto from engine /slots or /props.
    # 0 = admission gate off for this source.
    max_concurrency: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Per-source queue wait (sec). NULL = platform default in auth_settings.
    queue_timeout_sec: Mapped[int | None] = mapped_column(Integer, nullable=True)


class ModelAlias(Base):
    """Public model id → real catalog id (+ optional source pin + candidate pool)."""

    __tablename__ = "model_aliases"
    __table_args__ = (UniqueConstraint("alias_id", name="uq_model_alias_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alias_id: Mapped[str] = mapped_column(String(128), index=True)
    target_model_id: Mapped[str] = mapped_column(String(256), default="")
    preferred_source: Mapped[str] = mapped_column(String(64), default="")
    description: Mapped[str] = mapped_column(String(512), default="")
    show_backend: Mapped[bool] = mapped_column(Boolean, default=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    kind: Mapped[str] = mapped_column(String(16), default="chat")
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    # Hide target + candidates from GET /v1/models (clients see the alias only).
    hide_candidates: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    candidates: Mapped[list[ModelAliasCandidate]] = relationship(
        back_populates="alias",
        cascade="all, delete-orphan",
        order_by="ModelAliasCandidate.sort_order, ModelAliasCandidate.model_id",
    )


class ModelAliasCandidate(Base):
    """Swap pool for one public alias (admin picks active via ModelAlias.target_model_id)."""

    __tablename__ = "model_alias_candidates"
    __table_args__ = (
        UniqueConstraint("alias_row_id", "model_id", name="uq_alias_candidate"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alias_row_id: Mapped[int] = mapped_column(
        ForeignKey("model_aliases.id", ondelete="CASCADE"), index=True
    )
    model_id: Mapped[str] = mapped_column(String(256), index=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)

    alias: Mapped[ModelAlias] = relationship(back_populates="candidates")


class CatalogModel(Base):
    """Discovered upstream model; admin can disable globally for listing + inference."""

    __tablename__ = "catalog_models"
    __table_args__ = (
        UniqueConstraint("source_name", "model_id", name="uq_catalog_source_model"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_name: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(16), index=True, default="chat")
    model_id: Mapped[str] = mapped_column(String(256), index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    # Admin decision aids (not permissions)
    tags: Mapped[str] = mapped_column(String(512), default="")  # comma-separated: tools,vision,…
    short_note: Mapped[str] = mapped_column(String(512), default="")
    docs_url: Mapped[str] = mapped_column(String(512), default="")  # HF / docs link
    # Admin-curated JSON for clients (GET /v1/models). Never applied to inference.
    recommended_sampling: Mapped[str] = mapped_column(Text, default="")
    # Last-known fields from upstream /v1/models (llama.cpp etc.). Never invent:
    # ctx_size from status.args --ctx-size; n_parallel from --parallel;
    # meta.* only when status was loaded.
    upstream_status: Mapped[str] = mapped_column(String(32), default="")  # loaded|unloaded|…
    ctx_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_parallel: Mapped[int | None] = mapped_column(Integer, nullable=True)  # --parallel
    n_ctx: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_ctx_train: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_embd: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_params: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_size: Mapped[int | None] = mapped_column(Integer, nullable=True)  # bytes
    modalities_in: Mapped[str] = mapped_column(String(128), default="")
    modalities_out: Mapped[str] = mapped_column(String(128), default="")
    # Multiplier for usage-pool cost (1.0 = default). Heavy models → higher.
    usage_weight: Mapped[float] = mapped_column(Float, default=1.0)
    upstream_meta_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Why disabled: "" | admin (manual) | sync (missing on upstream after sync).
    disabled_by: Mapped[str] = mapped_column(String(16), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuthSettings(Base):
    """Singleton: public auth / registration / teams / privacy policy."""

    __tablename__ = "auth_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    allow_self_registration: Mapped[bool] = mapped_column(Boolean, default=False)
    require_email: Mapped[bool] = mapped_column(Boolean, default=True)
    teams_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    default_team_id: Mapped[int | None] = mapped_column(
        ForeignKey("teams.id", ondelete="SET NULL"), nullable=True
    )
    anonymize_client_ip: Mapped[bool] = mapped_column(Boolean, default=True)
    retention_days: Mapped[int] = mapped_column(Integer, default=30)
    # When on: chat requests with image parts rewrite model → VL sibling (if found).
    auto_vl_routing: Mapped[bool] = mapped_column(Boolean, default=False)
    # OnPrem aliases auto / auto-quality / auto-long. Empty → built-in Halo defaults.
    auto_model_default: Mapped[str] = mapped_column(String(256), default="")
    auto_model_quality: Mapped[str] = mapped_column(String(256), default="")
    auto_model_long: Mapped[str] = mapped_column(String(256), default="")
    # Max active API keys per non-admin user (0 = unlimited). Platform admins exempt.
    max_keys_per_user: Mapped[int] = mapped_column(Integer, default=3)
    # Usage pool: window length in hours (Claude-style rolling reset). 0 = disabled globally.
    pool_window_hours: Mapped[int] = mapped_column(Integer, default=5)
    # Token budget: cost ~= max(min_cost, tokens/pool_tokens_per_unit)
    # Optional × factor only when pool_model_weights_enabled (default off).
    pool_tokens_per_unit: Mapped[int] = mapped_column(Integer, default=1)
    pool_min_cost: Mapped[float] = mapped_column(Float, default=1.0)
    # Multiply estimated Wh into pool cost (0 = ignore sidecar even if up).
    pool_watt_weight: Mapped[float] = mapped_column(Float, default=0.0)
    # Rough tok/s for Wh estimate at auth time (no end-of-request hook yet).
    pool_tokens_per_sec: Mapped[float] = mapped_column(Float, default=50.0)
    # Off by default: budget charge = tokens. On: Models → factor may scale charge only.
    pool_model_weights_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    # Before proxy: capability-driven upstream check → 503 (strict, no fail-open).
    preflight_upstream: Mapped[bool] = mapped_column(Boolean, default=True)
    # Among tied catalog matches: pick source with most idle capacity (cached probe).
    load_aware_routing: Mapped[bool] = mapped_column(Boolean, default=True)
    # load_aware | round_robin | name — tie-break when same model on multiple sources.
    routing_strategy: Mapped[str] = mapped_column(String(32), default="load_aware")
    # Per-source admission queue (after strict preflight, before proxy).
    source_admission_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    source_queue_timeout_sec: Mapped[int] = mapped_column(Integer, default=30)
    # After sync: disable catalog rows no longer listed upstream (per source).
    catalog_prune_on_sync: Mapped[bool] = mapped_column(Boolean, default=True)
    # When on: chat requests missing temperature/etc. get catalog recommended_sampling
    # (default_profile / client hint). Never overrides keys the client already sent.
    soft_sampling_defaults: Mapped[bool] = mapped_column(Boolean, default=False)
    # Periodic catalog sync from sources (discover new / prune stale). Off by default.
    catalog_auto_sync: Mapped[bool] = mapped_column(Boolean, default=False)
    catalog_auto_sync_minutes: Mapped[int] = mapped_column(Integer, default=30)
    # New users (and self-register): comma source names; empty = sources marked default.
    default_grant_sources: Mapped[str] = mapped_column(Text, default="")
    # Newline "source:model"; empty = all ON models for those sources.
    default_grant_models: Mapped[str] = mapped_column(Text, default="")
    # Optional user-facing panel on Overview with aggregate gateway stats.
    show_global_stats: Mapped[bool] = mapped_column(Boolean, default=False)
    # Instance operator (Impressum) — filled by the person who runs this install.
    operator_name: Mapped[str] = mapped_column(String(255), default="")
    operator_address: Mapped[str] = mapped_column(Text, default="")
    operator_email: Mapped[str] = mapped_column(String(255), default="support@fr4iser.com")
    operator_phone: Mapped[str] = mapped_column(String(64), default="")


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("web_users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_by_platform_admin: Mapped[bool] = mapped_column(Boolean, default=False)


class RegistrationInvite(Base):
    """One-time signup token when public self-registration is off."""

    __tablename__ = "registration_invites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    used_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("web_users.id", ondelete="SET NULL"), nullable=True
    )
    created_by_id: Mapped[int] = mapped_column(
        ForeignKey("web_users.id", ondelete="CASCADE"), index=True
    )
    note: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def make_engine(db_path: str):
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def make_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
