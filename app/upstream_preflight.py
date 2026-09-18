"""Upstream preflight before proxying — strict, capability-driven."""

from __future__ import annotations

from dataclasses import dataclass

from .data.capabilities import (
    Admission,
    admission_reason,
    engine_state_to_load_snapshot,
    probe_engine_state,
    resolve_engine_for_source,
)
from .data.source_load import load_cache


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    reason: str = ""
    retry_after: int = 15
    engine: str = ""
    detail: str = ""
    retryable: bool = True


def preflight_upstream(
    *,
    backend: str,
    kind: str,
    model: str | None = None,
    timeout: float = 2.0,
    engine: str | None = None,
    engine_override: str | None = None,
    detected_engine: str | None = None,
) -> PreflightResult:
    """Return not-ok when backend cannot admit the request.

    Strict: probe errors, unreachable backends, and missing capabilities block.
    """
    addr = (backend or "").strip()
    if not addr:
        return PreflightResult(
            False, "no_backend", retry_after=0, retryable=False
        )

    engine_id = resolve_engine_for_source(
        backend=addr,
        kind=kind,
        engine_override=engine_override or engine,
        detected_engine=detected_engine,
        timeout=timeout,
    )

    state = probe_engine_state(
        backend=addr,
        kind=kind,
        model=model,
        engine=engine_id,
        timeout=timeout,
    )
    load_cache.put(addr, kind, model, engine_state_to_load_snapshot(state))

    if state.admission == Admission.OK:
        return PreflightResult(
            ok=True, engine=state.engine, detail=state.detail
        )

    reason, retry, retryable = admission_reason(state)
    return PreflightResult(
        ok=False,
        reason=reason,
        retry_after=retry,
        engine=state.engine,
        detail=state.detail,
        retryable=retryable,
    )
