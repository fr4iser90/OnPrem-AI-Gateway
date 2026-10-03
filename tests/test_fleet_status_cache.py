"""Fleet status cache: UI must not block on upstream probe timeouts."""

from __future__ import annotations

import time
from unittest.mock import patch

from app.data.fleet_status import (
    get_fleet_statuses,
    refresh_fleet_now,
    reset_fleet_status_cache_for_tests,
)
from app.data.probe import ServiceStatus, _HostDead, _get
from app.data.source_load import load_cache
from app.web.dashboard_ops import fleet_statuses


def setup_function():
    reset_fleet_status_cache_for_tests()


def teardown_function():
    reset_fleet_status_cache_for_tests()


def test_get_returns_cache_without_calling_probe_all():
    cached = [
        ServiceStatus(
            service="chat",
            backend="10.0.0.1:11535",
            kind="chat",
            state="down",
            detail="unreachable",
        )
    ]
    with patch(
        "app.data.fleet_status.probe_all",
        side_effect=AssertionError("probe_all must not run on cache hit"),
    ):
        # Seed via refresh path's cache write
        from app.data import fleet_status as fs

        with fs._cache_lock:
            fs._cached = list(cached)
            fs._cached_at = time.time()

        out = get_fleet_statuses(db=None)  # type: ignore[arg-type]
        assert out[0].service == "chat"
        assert out[0].state == "down"


def test_fleet_statuses_dashboard_helper_is_cache_only():
    from app.data import fleet_status as fs

    with fs._cache_lock:
        fs._cached = [
            ServiceStatus(
                service="embed",
                backend="10.0.0.2:8080",
                kind="embed",
                state="ok",
                detail="ok",
            )
        ]
        fs._cached_at = time.time()

    with patch(
        "app.data.fleet_status.probe_all",
        side_effect=AssertionError("live probe"),
    ):
        rows = fleet_statuses(db=None)  # type: ignore[arg-type]
    assert rows[0].state == "ok"


def test_refresh_seeds_load_cache():
    status = ServiceStatus(
        service="chat",
        backend="192.168.1.50:11535",
        kind="chat",
        state="ok",
        detail="ok",
        slots_total=4,
        slots_idle=2,
        slots_busy=2,
        engine="llama.cpp",
    )

    def fake_probe_all(_db):
        return [status]

    with patch("app.data.fleet_status.probe_all", side_effect=fake_probe_all):
        out = refresh_fleet_now(db=None)  # type: ignore[arg-type]
    assert out[0].slots_total == 4
    snap = load_cache.get("192.168.1.50:11535", "chat", None)
    assert snap is not None
    assert snap.state == "ok"
    assert snap.slots_idle == 2


def test_host_dead_raised_on_connect_timeout():
    import httpx

    class BoomClient:
        def get(self, url):
            raise httpx.ConnectTimeout("timed out")

    try:
        _get(BoomClient(), "http://example.invalid/health")
        assert False, "expected _HostDead"
    except _HostDead:
        pass


def test_ui_path_fast_when_probe_would_be_slow():
    """Simulates saturated upstream: probe_all sleeps; cache read stays fast."""

    def slow_probe(_db):
        time.sleep(2.0)
        return [
            ServiceStatus(
                service="chat",
                backend="10.0.0.9:1",
                kind="chat",
                state="down",
                detail="unreachable",
            )
        ]

    from app.data import fleet_status as fs

    with fs._cache_lock:
        fs._cached = [
            ServiceStatus(
                service="chat",
                backend="10.0.0.9:1",
                kind="chat",
                state="down",
                detail="stale-ok",
            )
        ]
        fs._cached_at = time.time()

    with patch("app.data.fleet_status.probe_all", side_effect=slow_probe):
        t0 = time.monotonic()
        rows = get_fleet_statuses(db=None)  # type: ignore[arg-type]
        elapsed = time.monotonic() - t0
    assert elapsed < 0.2
    assert rows[0].detail == "stale-ok"
