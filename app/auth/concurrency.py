"""Hold RPM/concurrency slots for the duration of an upstream stream."""

from __future__ import annotations

from dataclasses import dataclass

from .priority import priority_gate
from .rate_limit import rate_limiter


@dataclass(frozen=True)
class ConcurrencyLease:
    key_id: int
    user_id: int | None = None
    model: str | None = None
    source_key: str | None = None  # upstream host:port — held in source admission gate


def release_concurrency_lease(lease: ConcurrencyLease | None) -> None:
    if lease is None:
        return
    rate_limiter.release(lease.key_id, model=lease.model, user_id=lease.user_id)
    priority_gate.release(lease.key_id)
    if lease.source_key:
        from .source_admission import source_admission_gate

        source_admission_gate.release(lease.source_key, key_id=lease.key_id)
