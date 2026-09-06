"""Per-source capacity (slots) for client-facing model listings."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from sqlalchemy.orm import Session

from .backends import list_sources
from .models import BackendSource, CatalogModel
from .source_load import SourceLoadSnapshot, load_cache


@dataclass(frozen=True)
class SourceCapacity:
    source_name: str
    slots_total: int | None = None
    slots_idle: int | None = None
    slots_busy: int | None = None
    state: str = "unknown"


def format_slots_label(busy: int | None, total: int | None) -> str:
    """Human-readable slot load for UI (not a boolean busy flag)."""
    if total is None:
        return "—"
    if busy is None:
        return f"slots {total}"
    b = max(0, min(int(busy), int(total)))
    t = max(0, int(total))
    if t > 0 and b >= t:
        return f"all slots full ({b}/{t})"
    return f"{b}/{t} in use"


def capacity_fields(cap: SourceCapacity | None) -> dict:
    """OpenAI-style extra fields for GET /v1/models entries."""
    if cap is None:
        return {}
    out: dict = {}
    if cap.slots_total is not None:
        out["slots_total"] = cap.slots_total
    if cap.slots_idle is not None:
        out["slots_idle"] = cap.slots_idle
    if cap.slots_busy is not None:
        out["slots_busy"] = cap.slots_busy
    if cap.state and cap.state != "unknown":
        out["load_state"] = cap.state
    return out


def probe_model_for_source(db: Session, source_name: str) -> str | None:
    """Pick a catalog model id for llama-router ``/slots?model=`` (needs a name)."""
    name = (source_name or "").strip()
    if not name:
        return None
    rows = (
        db.query(CatalogModel)
        .filter(
            CatalogModel.source_name == name,
            CatalogModel.enabled.is_(True),
        )
        .all()
    )
    if not rows:
        return None

    def _rank(row: CatalogModel) -> tuple:
        st = (row.upstream_status or "").lower()
        status_rank = {"loaded": 0, "sleeping": 1, "loading": 2}.get(st, 3)
        # Prefer rows that know --parallel (router slot count).
        has_parallel = 0 if row.n_parallel else 1
        return (status_rank, has_parallel, row.model_id or "")

    rows.sort(key=_rank)
    mid = (rows[0].model_id or "").strip()
    return mid or None


def _merge_capacity(
    src: BackendSource,
    snap: SourceLoadSnapshot | None,
) -> SourceCapacity:
    """Combine probe snapshot, admin max_concurrency, and gateway admission inflight."""
    from ..auth.source_admission import resolve_admission_limit, source_admission_gate

    addr = (src.address or "").strip()
    probe_total = snap.slots_total if snap else None
    probe_idle = snap.slots_idle if snap else None
    state = (snap.state if snap else None) or "unknown"
    engine = (snap.engine if snap else None) or (src.detected_engine or "") or (
        src.engine_override or ""
    )

    limit = resolve_admission_limit(
        max_concurrency=src.max_concurrency,
        engine=engine,
        slots_total=probe_total,
    )
    total = limit if limit is not None else probe_total

    gw_inflight, gw_limit = source_admission_gate.snapshot(addr)
    if gw_limit is not None and gw_limit > 0 and total is None:
        total = gw_limit

    busy: int | None = None
    idle: int | None = None

    # Prefer live upstream /slots over gateway inflight (external clients / other
    # hops can occupy slots the gateway is not tracking).
    if probe_idle is not None and total is not None:
        idle = max(0, min(probe_idle, total))
        busy = max(0, total - idle)
    elif probe_idle is not None:
        idle = max(0, probe_idle)
        if probe_total is not None:
            busy = max(0, probe_total - idle)
            if total is None:
                total = probe_total
    elif gw_inflight is not None and total is not None:
        busy = max(0, min(gw_inflight, total))
        idle = max(0, total - busy)
    elif probe_total is not None and total is not None:
        # Have total but no idle/busy signal yet.
        pass

    return SourceCapacity(
        source_name=src.name,
        slots_total=total,
        slots_idle=idle,
        slots_busy=busy,
        state=state,
    )


def capacity_for_source(
    src: BackendSource,
    *,
    probe: bool = True,
    model: str | None = None,
) -> SourceCapacity:
    """Capacity for one source; uses load cache (TTL ~3s), probes on miss when probe=True."""
    addr = (src.address or "").strip()
    kind = src.kind or "chat"
    snap: SourceLoadSnapshot | None = None
    if addr:
        snap = load_cache.get(addr, kind, model)
        if snap is None and model:
            # Router probes are model-keyed; also accept a bare cache from prior hops.
            snap = load_cache.get(addr, kind, None)
        if snap is None and probe:
            snap = load_cache.snapshot_for(src, kind=kind, model=model)
    return _merge_capacity(src, snap)


def capacities_by_source(
    db: Session,
    names: set[str] | None = None,
    *,
    probe: bool = True,
) -> dict[str, SourceCapacity]:
    """Map source name → capacity. Probes unique sources in parallel on cache miss."""
    sources = list_sources(db)
    wanted = {n.strip() for n in (names or set()) if n and n.strip()}
    selected = [
        s
        for s in sources
        if (s.address or "").strip() and (not wanted or s.name in wanted)
    ]
    if not selected:
        return {}

    models: dict[str, str | None] = {}
    if probe:
        for src in selected:
            models[src.name] = probe_model_for_source(db, src.name)

    out: dict[str, SourceCapacity] = {}
    misses: list[tuple[BackendSource, str | None]] = []
    for src in selected:
        addr = (src.address or "").strip()
        kind = src.kind or "chat"
        model = models.get(src.name) if probe else None
        snap = None
        if addr:
            if model:
                snap = load_cache.get(addr, kind, model)
            if snap is None:
                snap = load_cache.get(addr, kind, None)
        if snap is not None or not probe:
            out[src.name] = _merge_capacity(src, snap)
        else:
            misses.append((src, model))

    if not misses:
        return out

    def _one(item: tuple[BackendSource, str | None]) -> tuple[str, SourceCapacity]:
        src, model = item
        return src.name, capacity_for_source(src, probe=True, model=model)

    workers = min(8, len(misses))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, item) for item in misses]
        for fut in as_completed(futs):
            name, cap = fut.result()
            out[name] = cap
    return out


def attach_capacity(entry: dict, cap: SourceCapacity | None) -> dict:
    """Mutate and return entry with slot fields."""
    if not entry or cap is None:
        return entry
    entry.update(capacity_fields(cap))
    return entry
