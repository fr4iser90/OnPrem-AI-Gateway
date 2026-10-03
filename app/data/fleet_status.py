"""Cached fleet probe results — UI never waits on live upstream probes.

Background worker refreshes every few seconds; request handlers only read the
cache (or lightweight placeholders when still cold).
"""

from __future__ import annotations

import threading
import time

from sqlalchemy.orm import Session

from .backends import list_sources
from .probe import ServiceStatus, probe_all

_REFRESH_SEC = 10.0
_stop = threading.Event()
_thread: threading.Thread | None = None
_start_lock = threading.Lock()
_cache_lock = threading.Lock()
_refresh_lock = threading.Lock()

_cached: list[ServiceStatus] = []
_cached_at: float = 0.0
_refreshing = False


def cache_age_sec() -> float | None:
    with _cache_lock:
        if _cached_at <= 0:
            return None
        return max(0.0, time.time() - _cached_at)


def cached_fleet_statuses() -> list[ServiceStatus]:
    with _cache_lock:
        return list(_cached)


def _placeholders(db: Session) -> list[ServiceStatus]:
    out: list[ServiceStatus] = []
    for src in list_sources(db):
        addr = (src.address or "").strip()
        if not addr:
            out.append(
                ServiceStatus(
                    service=src.name,
                    backend="",
                    kind=src.kind or "",
                    state="unset",
                    detail="not configured",
                )
            )
        else:
            out.append(
                ServiceStatus(
                    service=src.name,
                    backend=addr,
                    kind=src.kind or "",
                    state="unknown",
                    detail="probing…",
                    engine=(src.detected_engine or ""),
                )
            )
    return out


def _seed_load_cache(statuses: list[ServiceStatus]) -> None:
    """Keep capacity/routing cache warm without per-request probes."""
    from .source_load import SourceLoadSnapshot, load_cache

    now = time.time()
    for st in statuses:
        addr = (st.backend or "").strip()
        if not addr:
            continue
        kind = (st.kind or "chat").strip() or "chat"
        state = (st.state or "unknown").strip() or "unknown"
        if state in {"unset"}:
            continue
        load_cache.put(
            addr,
            kind,
            None,
            SourceLoadSnapshot(
                state=state if state in {"ok", "busy", "loading", "down", "unknown"} else "unknown",
                slots_total=st.slots_total,
                slots_idle=st.slots_idle,
                model_loaded=True if state in {"ok", "busy"} else (
                    False if state == "down" else None
                ),
                probed_at=now,
                engine=(st.engine or ""),
            ),
        )


def refresh_fleet_now(db: Session) -> list[ServiceStatus]:
    """Blocking live probe — background worker / tests only."""
    statuses = probe_all(db)
    _seed_load_cache(statuses)
    with _cache_lock:
        global _cached, _cached_at
        _cached = list(statuses)
        _cached_at = time.time()
    return list(statuses)


def get_fleet_statuses(db: Session) -> list[ServiceStatus]:
    """UI/API: cache only. Never blocks on upstream HTTP."""
    cached = cached_fleet_statuses()
    if cached:
        return cached
    kick_refresh()
    return _placeholders(db)


def kick_refresh() -> None:
    """Start a non-blocking refresh if one is not already running."""
    global _refreshing
    with _refresh_lock:
        if _refreshing:
            return
        _refreshing = True

    def _run() -> None:
        global _refreshing
        try:
            from .db import SessionLocal

            if SessionLocal is None:
                return
            db = SessionLocal()
            try:
                refresh_fleet_now(db)
            except Exception as exc:  # noqa: BLE001
                print(f"fleet status refresh failed: {exc}", flush=True)
            finally:
                db.close()
        finally:
            with _refresh_lock:
                _refreshing = False

    threading.Thread(target=_run, name="fleet-status-refresh", daemon=True).start()


def _loop() -> None:
    # Immediate first refresh (UI may already be serving placeholders).
    kick_refresh()
    while not _stop.wait(_REFRESH_SEC):
        kick_refresh()


def start_fleet_status_worker() -> None:
    """Idempotent: start daemon refresher once per process."""
    global _thread
    with _start_lock:
        if _thread is not None and _thread.is_alive():
            return
        _stop.clear()
        _thread = threading.Thread(
            target=_loop,
            name="fleet-status",
            daemon=True,
        )
        _thread.start()
        print("fleet status worker started (UI reads cache only)", flush=True)


def stop_fleet_status_worker() -> None:
    global _thread
    with _start_lock:
        _stop.set()
        t = _thread
        _thread = None
    if t is not None:
        t.join(timeout=2.0)


def reset_fleet_status_cache_for_tests() -> None:
    """Clear cache + stop worker (unit tests)."""
    stop_fleet_status_worker()
    with _cache_lock:
        global _cached, _cached_at
        _cached = []
        _cached_at = 0.0
    with _refresh_lock:
        global _refreshing
        _refreshing = False
    _stop.clear()
