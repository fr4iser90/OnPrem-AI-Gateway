"""Per-source admission: semaphore + priority queue until stream ends."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from ..data.capabilities import engine_profile, probe_engine_state, resolve_engine_for_source
from ..data.models import BackendSource
from ..data.source_load import load_cache


@dataclass(order=True)
class _Waiter:
    sort_key: tuple
    event: threading.Event = field(compare=False)
    key_id: int = field(compare=False)


@dataclass
class _Holder:
    key_id: int
    held_at: float


@dataclass
class _GateState:
    limit: int
    inflight: int = 0
    holders: list[_Holder] = field(default_factory=list)
    waiters: list[_Waiter] = field(default_factory=list)


@dataclass(frozen=True)
class SourceAdmissionOutcome:
    acquired: bool
    reason: str = ""
    retry_after: int = 15
    limit: int | None = None
    queued: bool = False


@dataclass(frozen=True)
class AdmissionSnapshot:
    source_key: str
    inflight: int
    limit: int
    holder_key_ids: tuple[int, ...]
    queued_key_ids: tuple[int, ...]


def resolve_admission_limit(
    *,
    max_concurrency: int | None,
    engine: str,
    slots_total: int | None,
) -> int | None:
    """Slot limit for gateway admission, or None when gate does not apply."""
    if max_concurrency is not None:
        if max_concurrency <= 0:
            return None
        return max_concurrency
    if slots_total is not None and slots_total > 0:
        profile = engine_profile(engine)
        if profile and profile.slot_admission:
            return slots_total
    return None


def resolve_queue_timeout(
    *,
    source_timeout: int | None,
    platform_timeout: int,
) -> float:
    if source_timeout is not None and source_timeout > 0:
        return float(source_timeout)
    return float(max(1, platform_timeout))


def clamp_max_hold_sec(raw: object) -> int:
    """0 = disabled; otherwise 1..86400 seconds (stream wall-clock kill)."""
    try:
        value = int(raw or 0)
    except (TypeError, ValueError):
        return 0
    if value <= 0:
        return 0
    return max(1, min(86400, value))


class SourceAdmissionGate:
    """One semaphore per upstream address (host:port).

    Slots are only freed via release() (stream end / abort). There is no
    silent counter reclaim — max hold is enforced by killing the stream
    (see main._stream_upstream_metered), which then releases the lease.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gates: dict[str, _GateState] = {}
        self._max_hold_sec = 0

    def configure(self, *, max_hold_sec: int | None = None) -> None:
        if max_hold_sec is not None:
            self._max_hold_sec = clamp_max_hold_sec(max_hold_sec)

    @property
    def max_hold_sec(self) -> int:
        return self._max_hold_sec

    def _promote_waiters(self, gate: _GateState, now: float) -> None:
        while gate.waiters and gate.inflight < gate.limit:
            next_waiter = gate.waiters.pop(0)
            gate.inflight += 1
            gate.holders.append(_Holder(key_id=int(next_waiter.key_id), held_at=now))
            next_waiter.event.set()

    def acquire(
        self,
        source_key: str,
        *,
        limit: int,
        key_id: int,
        priority: int,
        timeout: float,
        max_hold_sec: int | None = None,
    ) -> bool:
        key = (source_key or "").strip()
        if not key or limit <= 0:
            return True

        if max_hold_sec is not None:
            self._max_hold_sec = clamp_max_hold_sec(max_hold_sec)
        deadline = time.time() + timeout
        event = threading.Event()
        waiter = _Waiter(sort_key=(-priority, time.time()), event=event, key_id=key_id)

        with self._lock:
            gate = self._gates.setdefault(key, _GateState(limit=limit))
            gate.limit = limit
            now = time.time()
            if gate.inflight < gate.limit and not gate.waiters:
                gate.inflight += 1
                gate.holders.append(_Holder(key_id=int(key_id), held_at=now))
                return True
            gate.waiters.append(waiter)
            gate.waiters.sort()

        remaining = deadline - time.time()
        if remaining <= 0 or not event.wait(timeout=max(0.0, remaining)):
            with self._lock:
                gate = self._gates.get(key)
                if gate and waiter in gate.waiters:
                    gate.waiters.remove(waiter)
            return False
        return True

    def release(self, source_key: str, key_id: int | None = None) -> None:
        key = (source_key or "").strip()
        if not key:
            return
        with self._lock:
            gate = self._gates.get(key)
            if gate is None:
                return
            if key_id is not None:
                kid = int(key_id)
                for i, holder in enumerate(gate.holders):
                    if holder.key_id == kid:
                        gate.holders.pop(i)
                        break
                else:
                    # Double-release — do not steal another holder's slot.
                    return
            elif gate.holders:
                gate.holders.pop(0)
            else:
                return
            if gate.inflight > 0:
                gate.inflight -= 1
            self._promote_waiters(gate, time.time())

    def snapshot(self, source_key: str) -> tuple[int | None, int | None]:
        """Return (inflight, limit) for this upstream address, or (None, None)."""
        key = (source_key or "").strip()
        if not key:
            return None, None
        with self._lock:
            gate = self._gates.get(key)
            if gate is None:
                return None, None
            return gate.inflight, gate.limit

    def snapshot_detail(self, source_key: str) -> AdmissionSnapshot | None:
        key = (source_key or "").strip()
        if not key:
            return None
        with self._lock:
            gate = self._gates.get(key)
            if gate is None:
                return None
            return AdmissionSnapshot(
                source_key=key,
                inflight=gate.inflight,
                limit=gate.limit,
                holder_key_ids=tuple(h.key_id for h in gate.holders),
                queued_key_ids=tuple(w.key_id for w in gate.waiters),
            )

    def all_snapshots(self) -> list[AdmissionSnapshot]:
        with self._lock:
            out: list[AdmissionSnapshot] = []
            for addr, gate in self._gates.items():
                if gate.inflight <= 0 and not gate.waiters:
                    continue
                out.append(
                    AdmissionSnapshot(
                        source_key=addr,
                        inflight=gate.inflight,
                        limit=gate.limit,
                        holder_key_ids=tuple(h.key_id for h in gate.holders),
                        queued_key_ids=tuple(w.key_id for w in gate.waiters),
                    )
                )
            return out


source_admission_gate = SourceAdmissionGate()


def try_acquire_source_admission(
    *,
    src: BackendSource,
    backend: str,
    kind: str,
    model: str | None,
    key_id: int,
    priority: int,
    platform_queue_timeout_sec: int,
    enabled: bool = True,
    max_hold_sec: int = 0,
) -> SourceAdmissionOutcome:
    """Acquire a per-source slot. No gate when limit cannot be resolved."""
    if not enabled:
        return SourceAdmissionOutcome(acquired=True)

    addr = (backend or "").strip()
    if not addr:
        return SourceAdmissionOutcome(
            acquired=False,
            reason="no_backend",
            retry_after=15,
        )

    engine_id = resolve_engine_for_source(
        backend=addr,
        kind=kind,
        engine_override=src.engine_override or None,
        detected_engine=src.detected_engine or None,
    )

    cached = load_cache.get(addr, kind, model)
    slots_total = cached.slots_total if cached is not None else None
    engine = cached.engine if cached and cached.engine else engine_id

    if slots_total is None or not engine:
        snap = probe_engine_state(
            backend=addr,
            kind=kind,
            model=model,
            engine=engine_id,
        )
        slots_total = snap.slots_total
        engine = snap.engine or engine_id
        from ..data.capabilities import engine_state_to_load_snapshot

        load_cache.put(addr, kind, model, engine_state_to_load_snapshot(snap))

    limit = resolve_admission_limit(
        max_concurrency=src.max_concurrency,
        engine=engine,
        slots_total=slots_total,
    )
    if limit is None:
        return SourceAdmissionOutcome(acquired=True, limit=None)

    timeout = resolve_queue_timeout(
        source_timeout=src.queue_timeout_sec,
        platform_timeout=platform_queue_timeout_sec,
    )
    source_admission_gate.configure(max_hold_sec=max_hold_sec)
    ok = source_admission_gate.acquire(
        addr,
        limit=limit,
        key_id=key_id,
        priority=priority,
        timeout=timeout,
        max_hold_sec=max_hold_sec,
    )
    if ok:
        return SourceAdmissionOutcome(acquired=True, limit=limit)
    return SourceAdmissionOutcome(
        acquired=False,
        reason="source_queue_timeout",
        retry_after=max(5, int(timeout)),
        limit=limit,
        queued=True,
    )


def live_admission_rows(db) -> list[dict]:
    """Admin view: gateway-held slots per source (not upstream-only clients)."""
    from ..data.backends import list_sources
    from ..data.models import ApiKey
    from ..data.source_capacity import format_slots_label
    from ..web.accounts import get_auth_settings

    by_addr = {
        (s.address or "").strip(): s
        for s in list_sources(db)
        if (s.address or "").strip()
    }
    auth = get_auth_settings(db)
    source_admission_gate.configure(
        max_hold_sec=clamp_max_hold_sec(getattr(auth, "source_admission_max_hold_sec", 0))
    )
    snaps = source_admission_gate.all_snapshots()
    if not snaps:
        return []

    key_ids: set[int] = set()
    for snap in snaps:
        key_ids.update(snap.holder_key_ids)
        key_ids.update(snap.queued_key_ids)
    labels: dict[int, str] = {}
    if key_ids:
        for row in db.query(ApiKey).filter(ApiKey.id.in_(key_ids)).all():
            labels[row.id] = row.label or f"key#{row.id}"

    def _fmt(ids: tuple[int, ...]) -> list[str]:
        return [labels.get(i, f"key#{i}") for i in ids]

    rows: list[dict] = []
    for snap in snaps:
        src = by_addr.get(snap.source_key)
        name = src.name if src is not None else snap.source_key
        rows.append(
            {
                "source": name,
                "address": snap.source_key,
                "inflight": snap.inflight,
                "limit": snap.limit,
                "label": format_slots_label(snap.inflight, snap.limit),
                "holders": _fmt(snap.holder_key_ids),
                "queued": _fmt(snap.queued_key_ids),
            }
        )
    rows.sort(key=lambda r: (-r["inflight"], r["source"]))
    return rows
