from __future__ import annotations

import re

from sqlalchemy.orm import Session

from ..config import (
    KINDS,
    SOURCE_NAME_RE,
    Settings,
    public_route_for_source,
)
from .models import BackendConfig, BackendSource

# host or host:port (IPv4 / hostname); empty allowed
_BACKEND_RE = re.compile(
    r"^("
    r"(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)"
    r"|(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r")"
    r"(?::(?:[1-9]\d{0,4}|0))?"
    r"$"
)


def list_sources(db: Session) -> list[BackendSource]:
    return (
        db.query(BackendSource)
        .order_by(BackendSource.kind, BackendSource.name)
        .all()
    )


def source_names(db: Session) -> list[str]:
    return [s.name for s in list_sources(db)]


def default_grant_source_names(db: Session) -> list[str]:
    """Addressed sources (all equal). Grants use Settings → Default access, not this."""
    return [s.name for s in list_sources(db) if (s.address or "").strip()]


def source_chip_rows(
    db: Session, names: list[str] | None = None
) -> list[dict]:
    """UI rows for source checkboxes: name, kind, address. All sources are equal."""
    wanted = set(names) if names is not None else None
    rows: list[dict] = []
    for s in list_sources(db):
        if wanted is not None and s.name not in wanted:
            continue
        hw = (s.hardware or "").strip()
        tip = " · ".join(p for p in (s.kind, hw, (s.address or "").strip()) if p)
        rows.append(
            {
                "name": s.name,
                "kind": s.kind,
                "address": s.address or "",
                "hardware": hw,
                "hint": "",
                "tooltip": tip,
            }
        )
    return rows


def normalize_hardware_label(raw: str | None) -> str:
    """Optional box label (Strix Halo, Jetson …). Display only — not used for routing."""
    return " ".join((raw or "").split())[:64]


def hardware_labels(db: Session) -> dict[str, str]:
    return {
        s.name: (s.hardware or "").strip()
        for s in list_sources(db)
        if (s.hardware or "").strip()
    }


def get_source_by_name(db: Session, name: str) -> BackendSource | None:
    return db.query(BackendSource).filter(BackendSource.name == name.lower()).first()


def default_source_for_kind(db: Session, kind: str) -> BackendSource | None:
    row = (
        db.query(BackendSource)
        .filter(BackendSource.kind == kind, BackendSource.is_default.is_(True))
        .first()
    )
    if row:
        return row
    return (
        db.query(BackendSource)
        .filter(BackendSource.kind == kind)
        .order_by(BackendSource.name)
        .first()
    )


def parse_route_models(raw: str) -> list[str]:
    out: list[str] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def normalize_route_models(raw: str) -> str:
    return "\n".join(parse_route_models(raw))


def catalog_route_models(db: Session, source_name: str) -> list[str]:
    """Enabled catalog model ids for a source (auto merge targets after Models → Sync)."""
    from .models import CatalogModel

    rows = (
        db.query(CatalogModel.model_id)
        .filter(
            CatalogModel.source_name == source_name,
            CatalogModel.enabled.is_(True),
        )
        .order_by(CatalogModel.model_id)
        .all()
    )
    return [r[0] for r in rows]


def route_patterns_for_source(db: Session, src: BackendSource) -> list[str]:
    """Enabled catalog model ids for this source (Models → Sync)."""
    return catalog_route_models(db, src.name)


def model_match_score(pattern: str, model: str) -> int | None:
    """Higher = better. None = no match."""
    if not pattern or not model:
        return None
    if pattern == model:
        return 1000 + len(pattern)
    if pattern.endswith("*") and len(pattern) > 1:
        prefix = pattern[:-1]
        if model.startswith(prefix):
            return 500 + len(prefix)
        return None
    # bare name matches name and name:tag (Ollama-style)
    if ":" not in pattern and (model == pattern or model.startswith(pattern + ":")):
        return 400 + len(pattern)
    return None


def resolve_source_for_kind(
    db: Session,
    kind: str,
    *,
    model: str | None = None,
    allowed_services: set[str] | None = None,
    routing_strategy: str = "load_aware",
    preferred_source: str | None = None,
    load_aware: bool | None = None,
) -> BackendSource | None:
    """Pick upstream for a /v1 request by enabled catalog model.

    No kind fallback: unknown, disabled, or missing model → None.
    When several sources tie on pattern score, break ties via routing_strategy.
    """
    from .routing_strategy import normalize_routing_strategy, round_robin_state

    if load_aware is not None:
        strategy = "load_aware" if load_aware else "name"
    else:
        strategy = normalize_routing_strategy(routing_strategy)
    if not (model or "").strip():
        return None
    sources = (
        db.query(BackendSource)
        .filter(BackendSource.kind == kind)
        .order_by(BackendSource.name)
        .all()
    )
    if not sources:
        return None

    ranked: list[tuple[int, str, BackendSource]] = []
    for src in sources:
        if allowed_services is not None and src.name not in allowed_services:
            continue
        if not (src.address or "").strip():
            continue
        patterns = route_patterns_for_source(db, src)
        if not patterns:
            continue
        best_score = None
        for pat in patterns:
            s = model_match_score(pat, model)
            if s is not None and (best_score is None or s > best_score):
                best_score = s
        if best_score is None:
            continue
        ranked.append((best_score, src.name, src))
    if not ranked:
        return None

    ranked.sort(key=lambda t: (-t[0], t[1]))
    top_score = ranked[0][0]
    tied = [t for t in ranked if t[0] == top_score]
    if len(tied) == 1:
        return tied[0][2]

    pref = (preferred_source or "").strip().lower()
    if pref:
        for _score, name, src in tied:
            if name == pref:
                return src

    if strategy == "round_robin":
        names = [t[1] for t in tied]
        pick_name = round_robin_state.pick(kind=kind, model=model or "", source_names=names)
        for _score, name, src in tied:
            if name == pick_name:
                return src
        return tied[0][2]

    if strategy == "name":
        return tied[0][2]

    # load_aware (default)
    from .source_load import load_cache, load_sort_key

    def _pick_key(t: tuple[int, str, BackendSource]) -> tuple:
        src = t[2]
        snap = load_cache.snapshot_for(src, kind=kind, model=model)
        return (load_sort_key(snap), t[1])

    tied.sort(key=_pick_key)
    return tied[0][2]


def address_for_source(db: Session, name: str) -> str:
    src = get_source_by_name(db, name)
    return (src.address if src else "").strip()


def normalize_backend(raw: str) -> str:
    return (raw or "").strip()


def validate_backend(raw: str) -> str | None:
    """Return error message or None if OK. Empty is allowed (source disabled)."""
    value = normalize_backend(raw)
    if not value:
        return None
    if not _BACKEND_RE.match(value):
        return "Use host or host:port (e.g. 192.168.1.10:8080 or localai:8080)"
    if ":" in value:
        port = int(value.rsplit(":", 1)[-1])
        if port < 1 or port > 65535:
            return "Port must be 1–65535"
    return None


def validate_source_name(name: str) -> str | None:
    n = (name or "").strip().lower()
    if not n:
        return "Name is required"
    if not SOURCE_NAME_RE.match(n):
        return "Name: start with a–z, then a–z / 0–9 / _ / - (max 32)"
    if n == "s":
        return "Name 's' is reserved"
    return None


def rename_source(db: Session, src: BackendSource, new_name: str) -> str | None:
    """Rename a box. Kind stays. Rewrites grants/catalog/usage that stored the slug."""
    err = validate_source_name(new_name)
    if err:
        return err
    new_name = new_name.strip().lower()
    old = src.name
    if new_name == old:
        return None
    if get_source_by_name(db, new_name) is not None:
        return f"Source '{new_name}' already exists"

    from .models import (
        AuthSettings,
        CatalogModel,
        ModelAllowlist,
        ModelLimit,
        ServiceGrant,
        UsageDaily,
        UsageEvent,
    )

    def _swap(model, col) -> None:
        db.query(model).filter(col == old).update({col: new_name}, synchronize_session=False)

    _swap(CatalogModel, CatalogModel.source_name)
    _swap(ServiceGrant, ServiceGrant.service)
    _swap(ModelAllowlist, ModelAllowlist.service)
    _swap(ModelLimit, ModelLimit.service)
    _swap(UsageEvent, UsageEvent.service)
    _swap(UsageDaily, UsageDaily.service)

    auth = db.query(AuthSettings).first()
    if auth is not None:
        raw = (auth.default_grant_sources or "").strip()
        if raw and raw != "-":
            parts = [
                new_name if p.strip() == old else p.strip()
                for p in raw.split(",")
                if p.strip()
            ]
            auth.default_grant_sources = ",".join(parts)
        prefix = old + ":"
        auth.default_grant_models = "\n".join(
            (new_name + ":" + line[len(prefix) :]) if line.startswith(prefix) else line
            for line in (auth.default_grant_models or "").splitlines()
        )
    src.name = new_name
    db.flush()
    return None


def apply_source_row_edits(
    db: Session, edits: list[tuple[BackendSource, str, str]]
) -> str | None:
    """Apply name+address edits for several boxes. Two-phase rename avoids unique clashes."""
    prepared: list[tuple[BackendSource, str, str]] = []
    seen: set[str] = set()
    for src, raw_name, raw_addr in edits:
        err = validate_source_name(raw_name) or validate_backend(raw_addr)
        if err:
            return f"{src.name}: {err}"
        name = raw_name.strip().lower()
        addr = normalize_backend(raw_addr)
        if name in seen:
            return f"Duplicate name '{name}'"
        seen.add(name)
        prepared.append((src, name, addr))

    occupied = {
        s.name
        for s in list_sources(db)
        if s.id not in {row.id for row, _, _ in prepared}
    }
    for _, name, _ in prepared:
        if name in occupied:
            return f"Source '{name}' already exists"

    for src, _, addr in prepared:
        src.address = addr
    db.flush()

    need_rename = [(src, name) for src, name, _ in prepared if src.name != name]
    if not need_rename:
        return None

    parked: list[tuple[BackendSource, str]] = []
    used = {s.name for s in list_sources(db)} | seen
    n = 0
    for src, final in need_rename:
        temp = None
        while temp is None or temp in used:
            temp = f"zz{n}"
            n += 1
        used.add(temp)
        err = rename_source(db, src, temp)
        if err:
            return err
        parked.append((src, final))
    for src, final in parked:
        err = rename_source(db, src, final)
        if err:
            return err
    return None


def validate_kind(kind: str) -> str | None:
    if kind not in KINDS:
        return f"Kind must be one of: {', '.join(KINDS)}"
    return None


def clear_default_for_kind(db: Session, kind: str, *, except_id: int | None = None) -> None:
    q = db.query(BackendSource).filter(
        BackendSource.kind == kind, BackendSource.is_default.is_(True)
    )
    if except_id is not None:
        q = q.filter(BackendSource.id != except_id)
    for row in q.all():
        row.is_default = False


def upsert_source(
    db: Session,
    *,
    name: str,
    kind: str,
    address: str,
    is_default: bool = False,
    route_models: str = "",
    api_style: str = "auto",
    gpu_power_url: str = "",
    temp_guard_enabled: bool = True,
    hardware: str | None = None,
) -> BackendSource:
    from ..config import API_STYLES

    name = name.strip().lower()
    address = normalize_backend(address)
    models = normalize_route_models(route_models)
    style = (api_style or "auto").strip().lower()
    if style not in API_STYLES:
        style = "auto"
    gpu = (gpu_power_url or "").strip()
    temp_on = bool(temp_guard_enabled)
    label = None if hardware is None else normalize_hardware_label(hardware)
    existing = get_source_by_name(db, name)
    if is_default:
        clear_default_for_kind(db, kind, except_id=existing.id if existing else None)
    if existing:
        existing.kind = kind
        existing.address = address
        existing.route_models = models
        existing.api_style = style
        existing.gpu_power_url = gpu
        existing.temp_guard_enabled = temp_on
        if label is not None:
            existing.hardware = label
        return existing
    src = BackendSource(
        name=name,
        kind=kind,
        address=address,
        is_default=is_default,
        route_models=models,
        api_style=style,
        gpu_power_url=gpu,
        temp_guard_enabled=temp_on,
        hardware=label or "",
    )
    db.add(src)
    db.flush()
    return src


def delete_source(db: Session, src: BackendSource) -> None:
    db.delete(src)
    db.flush()


def migrate_backend_config_to_sources(db: Session) -> None:
    """One-shot: copy legacy backend_config columns into backend_sources."""
    if db.query(BackendSource).count() > 0:
        return
    cfg = db.query(BackendConfig).first()
    if cfg is None:
        return
    mapping = [
        ("chat", "chat", (cfg.chat or "").strip(), True),
        ("chat2", "chat", (getattr(cfg, "chat2", None) or "").strip(), False),
        ("embed", "embed", (cfg.embed or "").strip(), True),
        ("stt", "stt", (cfg.stt or "").strip(), True),
        ("tts", "tts", (cfg.tts or "").strip(), True),
    ]
    for name, kind, address, is_default in mapping:
        if not address:
            continue
        upsert_source(db, name=name, kind=kind, address=address, is_default=is_default)


def seed_backends_from_env(db: Session, settings: Settings) -> None:
    """One-time bootstrap: legacy migration, then env seeds only on an empty DB."""
    migrate_backend_config_to_sources(db)
    if db.query(BackendSource).count() > 0:
        return
    seeds: list[tuple[str, str, str, bool]] = [
        (
            "chat",
            "chat",
            settings.chat_source
            or settings.chat_backend
            or settings.llm_backend
            or settings.ollama_backend,
            True,
        ),
        ("embed", "embed", settings.embed_source or settings.embed_backend, True),
        (
            "extractor",
            "extractor",
            settings.extractor_source or settings.extractor_backend,
            True,
        ),
        ("stt", "stt", settings.stt_source or settings.stt_backend, True),
        ("tts", "tts", settings.tts_source or settings.tts_backend, True),
        ("chat2", "chat", settings.chat2_source or settings.chat2_backend, False),
    ]
    for name, kind, address, is_default in seeds:
        address = normalize_backend(address)
        if not address:
            continue
        existing = get_source_by_name(db, name)
        if existing:
            if not existing.address:
                existing.address = address
            continue
        # Don't steal default if kind already has one
        has_default = (
            db.query(BackendSource)
            .filter(BackendSource.kind == kind, BackendSource.is_default.is_(True))
            .first()
            is not None
        )
        upsert_source(
            db,
            name=name,
            kind=kind,
            address=address,
            is_default=is_default and not has_default,
        )


def source_rows(
    db: Session, settings: Settings | None = None
) -> list[tuple[BackendSource, str]]:
    """(source, public_route_display)."""
    rows = []
    for src in list_sources(db):
        route = public_route_for_source(src.name, src.kind, settings=settings)
        rows.append((src, route))
    return rows


# --- deprecated helpers (thin wrappers) ---


def backend_for_service(db: Session, service: str) -> str:
    return address_for_source(db, service)
