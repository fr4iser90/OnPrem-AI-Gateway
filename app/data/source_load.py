"""Cached upstream load snapshots for routing among tied sources."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .backends import BackendSource
from .capabilities import engine_state_to_load_snapshot, probe_engine_state

# Align with fleet-status background refresh (~10s) so UI/API cache-only
# reads stay warm between ticks without re-probing on every page load.
_CACHE_TTL_SEC = 15.0
_PROBE_TIMEOUT = 1.5

# Lower sort key = preferred for routing.
_STATE_RANK = {
    "ok": 0,
    "busy": 1,
    "loading": 2,
    "unknown": 3,
    "down": 4,
}


@dataclass(frozen=True)
class SourceLoadSnapshot:
    state: str = "unknown"
    slots_total: int | None = None
    slots_idle: int | None = None
    model_loaded: bool | None = None
    probed_at: float = 0.0
    engine: str = ""


def probe_source_load(
    *,
    backend: str,
    kind: str,
    model: str | None = None,
    timeout: float = _PROBE_TIMEOUT,
    engine: str | None = None,
    engine_override: str | None = None,
    detected_engine: str | None = None,
) -> SourceLoadSnapshot:
    """Capability-driven load probe (≤ timeout)."""
    state = probe_engine_state(
        backend=backend,
        kind=kind,
        model=model,
        engine=engine,
        timeout=timeout,
    )
    snap = engine_state_to_load_snapshot(state)
    return SourceLoadSnapshot(
        state=snap.state,
        slots_total=snap.slots_total,
        slots_idle=snap.slots_idle,
        model_loaded=snap.model_loaded,
        probed_at=state.probed_at,
        engine=state.engine,
    )


def load_sort_key(snapshot: SourceLoadSnapshot | None) -> tuple:
    """Lower = less loaded / preferred."""
    if snapshot is None:
        return (_STATE_RANK["unknown"], 1, 0)
    state = _STATE_RANK.get(snapshot.state, _STATE_RANK["unknown"])
    loaded_penalty = 0
    if snapshot.model_loaded is False:
        loaded_penalty = 1
    idle = snapshot.slots_idle
    if idle is None:
        idle_score = 0
    else:
        idle_score = -idle
    return (state, loaded_penalty, idle_score)


class LoadCache:
    def __init__(self, ttl_sec: float = _CACHE_TTL_SEC) -> None:
        self._ttl = ttl_sec
        self._lock = threading.Lock()
        self._entries: dict[str, SourceLoadSnapshot] = {}

    @staticmethod
    def _key(backend: str, kind: str, model: str | None) -> str:
        return f"{backend}|{kind}|{(model or '').strip()}"

    def get(
        self, backend: str, kind: str, model: str | None = None
    ) -> SourceLoadSnapshot | None:
        key = self._key(backend, kind, model)
        with self._lock:
            snap = self._entries.get(key)
            if snap is None:
                return None
            if time.time() - snap.probed_at > self._ttl:
                del self._entries[key]
                return None
            return snap

    def put(
        self,
        backend: str,
        kind: str,
        model: str | None,
        snapshot: SourceLoadSnapshot,
    ) -> None:
        key = self._key(backend, kind, model)
        with self._lock:
            self._entries[key] = snapshot

    def snapshot_for(
        self,
        src: BackendSource,
        *,
        kind: str,
        model: str | None = None,
    ) -> SourceLoadSnapshot:
        backend = (src.address or "").strip()
        cached = self.get(backend, kind, model)
        if cached is not None:
            return cached
        snap = probe_source_load(
            backend=backend,
            kind=kind,
            model=model,
            engine_override=src.engine_override or None,
            detected_engine=src.detected_engine or None,
        )
        self.put(backend, kind, model, snap)
        return snap


load_cache = LoadCache()
