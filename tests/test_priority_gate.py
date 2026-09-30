"""Priority gate: optional global slots; 0 = unlimited."""

from __future__ import annotations

import threading
import time

from app.auth.priority import PriorityGate


def test_priority_gate_disabled_never_blocks():
    gate = PriorityGate(global_slots=0)
    for i in range(100):
        assert gate.acquire(key_id=i, priority=0, timeout=0.01) is True
    # release is a no-op when disabled
    gate.release(1)


def test_priority_gate_hard_cap_blocks():
    gate = PriorityGate(global_slots=1)
    assert gate.acquire(key_id=1, priority=0, timeout=0.05) is True
    assert gate.acquire(key_id=2, priority=0, timeout=0.05) is False
    gate.release(1)
    assert gate.acquire(key_id=2, priority=0, timeout=0.05) is True
    gate.release(2)


def test_priority_gate_wakes_higher_priority():
    gate = PriorityGate(global_slots=1)
    started: list[int] = []
    lock = threading.Lock()

    def worker(priority: int, tag: int):
        ok = gate.acquire(key_id=tag, priority=priority, timeout=2.0)
        with lock:
            started.append((tag, ok))
        if ok:
            time.sleep(0.05)
            gate.release(tag)

    t_low = threading.Thread(target=worker, args=(0, 1))
    t_low.start()
    time.sleep(0.01)
    t_high = threading.Thread(target=worker, args=(10, 2))
    t_high.start()
    t_low.join(timeout=3)
    t_high.join(timeout=3)

    assert started[0] == (1, True)
    assert any(tag == 2 and ok for tag, ok in started)
