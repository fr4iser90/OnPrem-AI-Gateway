from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(order=True)
class _Waiter:
    # lower sort key = higher priority (we invert priority)
    sort_key: tuple
    event: threading.Event = field(compare=False)
    key_id: int = field(compare=False)


class PriorityGate:
    """
    Optional global concurrency slots with priority fairness.

    global_slots <= 0: disabled (always acquire) — no fleet-wide 429.
    global_slots > 0: hard cap; higher priority values are served first when a slot frees.
    """

    def __init__(self, global_slots: int = 0) -> None:
        self.global_slots = int(global_slots)
        self._lock = threading.Lock()
        self._inflight = 0
        self._waiters: list[_Waiter] = []
        self._key_inflight: dict[int, int] = defaultdict(int)

    def acquire(self, *, key_id: int, priority: int, timeout: float = 2.0) -> bool:
        if self.global_slots <= 0:
            return True

        deadline = time.time() + timeout
        event = threading.Event()
        # Negate priority so higher priority sorts first; then FIFO by time
        waiter = _Waiter(sort_key=(-priority, time.time()), event=event, key_id=key_id)

        with self._lock:
            if self._inflight < self.global_slots and not self._waiters:
                self._inflight += 1
                self._key_inflight[key_id] += 1
                return True
            self._waiters.append(waiter)
            self._waiters.sort()

        remaining = deadline - time.time()
        if remaining <= 0 or not event.wait(timeout=max(0.0, remaining)):
            with self._lock:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
            return False
        return True

    def release(self, key_id: int) -> None:
        if self.global_slots <= 0:
            return
        with self._lock:
            if self._key_inflight[key_id] > 0:
                self._key_inflight[key_id] -= 1
            if self._inflight > 0:
                self._inflight -= 1
            self._wake_next()

    def _wake_next(self) -> None:
        while self._waiters and self._inflight < self.global_slots:
            waiter = self._waiters.pop(0)
            self._inflight += 1
            self._key_inflight[waiter.key_id] += 1
            waiter.event.set()


# Default off: a hard fleet-wide slot cap was returning 429 to every key
# (including those with no RPM) when streams leaked. Per-key / team / user
# limits and source-admission priority ordering still apply.
priority_gate = PriorityGate(global_slots=0)
