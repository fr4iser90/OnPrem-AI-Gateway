"""Upstream preflight before proxy."""

from __future__ import annotations

from unittest.mock import patch

from app.data.capabilities import Admission, EngineState
from app.upstream_preflight import preflight_upstream


def test_preflight_loading_from_state():
    state = EngineState("llama.cpp", Admission.LOADING, "health_loading", probed_at=1.0)
    with patch("app.upstream_preflight.probe_engine_state", return_value=state):
        pf = preflight_upstream(
            backend="127.0.0.1:8080", kind="chat", model="jarvis", engine="llama.cpp"
        )
    assert not pf.ok
    assert pf.reason == "model_initializing"


def test_preflight_busy():
    state = EngineState("llama.cpp", Admission.BUSY, "all_slots_full", probed_at=1.0)
    with patch("app.upstream_preflight.probe_engine_state", return_value=state):
        pf = preflight_upstream(
            backend="127.0.0.1:8080", kind="chat", model="jarvis", engine="llama.cpp"
        )
    assert not pf.ok
    assert pf.reason == "all_slots_full"
    assert pf.retry_after == 5


def test_preflight_model_not_loaded():
    state = EngineState(
        "ollama", Admission.LOADING, "model_cold_start", model_loaded=False, probed_at=1.0
    )
    with patch("app.upstream_preflight.probe_engine_state", return_value=state):
        pf = preflight_upstream(
            backend="127.0.0.1:11434", kind="chat", model="jarvis", engine="ollama"
        )
    assert not pf.ok
    assert pf.reason == "model_initializing"


def test_preflight_strict_on_down():
    state = EngineState("llama.cpp", Admission.DOWN, "unreachable", probed_at=1.0)
    with patch("app.upstream_preflight.probe_engine_state", return_value=state):
        pf = preflight_upstream(
            backend="127.0.0.1:8080", kind="chat", model="jarvis", engine="llama.cpp"
        )
    assert not pf.ok
    assert pf.reason == "backend_unreachable"
