"""Optional background catalog sync (admin flag catalog_auto_sync)."""

from __future__ import annotations

import threading
import time
from typing import Callable

_MIN_MINUTES = 5
_MAX_MINUTES = 24 * 60
_DEFAULT_MINUTES = 30

_stop = threading.Event()
_thread: threading.Thread | None = None
_lock = threading.Lock()


def clamp_auto_sync_minutes(raw) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = _DEFAULT_MINUTES
    return max(_MIN_MINUTES, min(_MAX_MINUTES, n))


def _session_factory():
    from .data.db import SessionLocal

    return SessionLocal


def _run_once() -> dict | None:
    SessionLocal = _session_factory()
    if SessionLocal is None:
        return None
    from .data.catalog import sync_catalog_from_sources
    from .web.accounts import get_auth_settings

    db = SessionLocal()
    try:
        auth = get_auth_settings(db)
        if not getattr(auth, "catalog_auto_sync", False):
            return None
        stats = sync_catalog_from_sources(db)
        db.commit()
        return stats
    except Exception as exc:
        db.rollback()
        print(f"catalog auto-sync failed: {exc}", flush=True)
        return None
    finally:
        db.close()


def _loop(sleep_fn: Callable[[float], None] = time.sleep) -> None:
    # Short initial delay so startup/logging finishes first.
    if _stop.wait(15):
        return
    while not _stop.is_set():
        minutes = _DEFAULT_MINUTES
        SessionLocal = _session_factory()
        if SessionLocal is not None:
            db = SessionLocal()
            try:
                from .web.accounts import get_auth_settings

                auth = get_auth_settings(db)
                enabled = bool(getattr(auth, "catalog_auto_sync", False))
                minutes = clamp_auto_sync_minutes(
                    getattr(auth, "catalog_auto_sync_minutes", _DEFAULT_MINUTES)
                )
            except Exception as exc:
                print(f"catalog auto-sync settings read failed: {exc}", flush=True)
                enabled = False
            finally:
                db.close()
        else:
            enabled = False

        if enabled:
            stats = _run_once()
            if stats is not None:
                print(
                    "catalog auto-sync: "
                    f"seen={stats.get('seen', 0)} created={stats.get('created', 0)} "
                    f"pruned={stats.get('pruned', 0)} tagged={stats.get('tagged', 0)}",
                    flush=True,
                )

        # Re-read interval each cycle; wake early on stop.
        _stop.wait(minutes * 60)


def start_catalog_auto_sync() -> None:
    """Idempotent: start daemon thread once per process."""
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _stop.clear()
        _thread = threading.Thread(
            target=_loop,
            name="catalog-auto-sync",
            daemon=True,
        )
        _thread.start()
        print("catalog auto-sync worker started (honors Settings flag)", flush=True)


def stop_catalog_auto_sync() -> None:
    global _thread
    with _lock:
        _stop.set()
        t = _thread
        _thread = None
    if t is not None:
        t.join(timeout=2.0)
