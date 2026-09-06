"""Per-source admission gate."""

from __future__ import annotations

import threading
import time

from app.auth.source_admission import (
    SourceAdmissionGate,
    resolve_admission_limit,
    resolve_queue_timeout,
    try_acquire_source_admission,
)
from app.data.models import BackendSource


def test_resolve_admission_limit_explicit():
    assert resolve_admission_limit(max_concurrency=2, engine="llama.cpp", slots_total=4) == 2
    assert resolve_admission_limit(max_concurrency=0, engine="llama.cpp", slots_total=4) is None


def test_resolve_admission_limit_auto_from_slots():
    assert (
        resolve_admission_limit(max_concurrency=None, engine="llama.cpp", slots_total=3) == 3
    )
    assert (
        resolve_admission_limit(max_concurrency=None, engine="ollama", slots_total=3) is None
    )


def test_gate_priority_queue():
    gate = SourceAdmissionGate()
    started: list[int] = []
    lock = threading.Lock()

    def worker(priority: int, tag: int):
        ok = gate.acquire("h:1", limit=1, key_id=tag, priority=priority, timeout=2.0)
        with lock:
            started.append((tag, ok))
        if ok:
            time.sleep(0.05)
            gate.release("h:1")

    t_low = threading.Thread(target=worker, args=(0, 1))
    t_low.start()
    time.sleep(0.01)
    t_high = threading.Thread(target=worker, args=(10, 2))
    t_high.start()
    t_low.join(timeout=3)
    t_high.join(timeout=3)

    assert started[0] == (1, True)
    assert any(tag == 2 and ok for tag, ok in started)


def test_try_acquire_uses_explicit_limit():
    src = BackendSource(
        name="chat",
        kind="chat",
        address="127.0.0.1:8080",
        max_concurrency=1,
    )
    gate = SourceAdmissionGate()
    import app.auth.source_admission as mod

    orig = mod.source_admission_gate
    mod.source_admission_gate = gate
    try:
        first = try_acquire_source_admission(
            src=src,
            backend="127.0.0.1:8080",
            kind="chat",
            model="m",
            key_id=1,
            priority=0,
            platform_queue_timeout_sec=1,
        )
        second = try_acquire_source_admission(
            src=src,
            backend="127.0.0.1:8080",
            kind="chat",
            model="m",
            key_id=2,
            priority=0,
            platform_queue_timeout_sec=0,
        )
    finally:
        mod.source_admission_gate = orig

    assert first.acquired is True
    assert first.limit == 1
    assert second.acquired is False
    assert second.reason == "source_queue_timeout"
    gate.release("127.0.0.1:8080")


def test_gate_tracks_holders():
    gate = SourceAdmissionGate()
    assert gate.acquire("h:1", limit=2, key_id=10, priority=0, timeout=0.1)
    assert gate.acquire("h:1", limit=2, key_id=20, priority=0, timeout=0.1)
    detail = gate.snapshot_detail("h:1")
    assert detail is not None
    assert detail.inflight == 2
    assert detail.holder_key_ids == (10, 20)
    gate.release("h:1", key_id=10)
    detail = gate.snapshot_detail("h:1")
    assert detail.inflight == 1
    assert detail.holder_key_ids == (20,)
    gate.release("h:1", key_id=20)
    assert gate.all_snapshots() == []


def test_resolve_queue_timeout():
    assert resolve_queue_timeout(source_timeout=10, platform_timeout=30) == 10.0
    assert resolve_queue_timeout(source_timeout=None, platform_timeout=25) == 25.0
