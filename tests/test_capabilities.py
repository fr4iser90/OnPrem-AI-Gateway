"""Capability registry + strict preflight."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from app.data.capabilities import (
    Admission,
    admission_reason,
    probe_engine_state,
)
from app.upstream_preflight import preflight_upstream


def _mock_client(responses: dict[str, httpx.Response]) -> MagicMock:
    client = MagicMock()

    def _get(url: str) -> httpx.Response | None:
        for path, resp in responses.items():
            if url.endswith(path) or path in url:
                return resp
        return None

    client.get.side_effect = _get
    return client


def test_llama_slots_busy_blocks():
    responses = {
        "/health": httpx.Response(200, json={"status": "ok"}),
        "/props": httpx.Response(200, json={"total_slots": 2, "is_sleeping": False}),
        "/slots": httpx.Response(
            200,
            json=[
                {"id": 0, "is_processing": True},
                {"id": 1, "is_processing": True},
            ],
        ),
    }
    with patch("app.data.capabilities.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value = _mock_client(responses)
        state = probe_engine_state(
            backend="127.0.0.1:8080",
            kind="chat",
            engine="llama.cpp",
        )
    assert state.admission == Admission.BUSY
    assert state.slots_idle == 0


def test_llama_fail_on_no_slot_503():
    responses = {
        "/health": httpx.Response(200, json={"status": "ok"}),
        "/props": httpx.Response(200, json={"total_slots": 1, "is_sleeping": False}),
        "fail_on_no_slot=1": httpx.Response(503, json={"error": {"code": 503}}),
        "/slots": httpx.Response(200, json=[{"id": 0, "is_processing": False}]),
    }
    with patch("app.data.capabilities.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value = _mock_client(responses)
        state = probe_engine_state(
            backend="127.0.0.1:8080",
            kind="chat",
            engine="llama.cpp",
        )
    assert state.admission == Admission.BUSY
    assert state.detail == "no_idle_slots"


def test_llama_router_unloaded_allows_on_demand_load():
    model = "Qwen-VL"
    responses = {
        "/health": httpx.Response(200, json={"status": "ok"}),
        "/props": httpx.Response(200, json={"total_slots": 1, "role": "router"}),
        f"/models?model={model}": httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": model,
                        "status": {"value": "unloaded", "args": ["--ctx-size", "8192"]},
                    }
                ]
            },
        ),
    }
    with patch("app.data.capabilities.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value = _mock_client(responses)
        state = probe_engine_state(
            backend="127.0.0.1:8080",
            kind="chat",
            model=model,
            engine="llama-router",
        )
    assert state.admission == Admission.OK
    assert state.detail == "model_on_demand"
    assert state.model_loaded is False


def test_llama_router_loading_blocks():
    model = "Qwen-VL"
    responses = {
        "/health": httpx.Response(200, json={"status": "ok"}),
        "/props": httpx.Response(200, json={"total_slots": 1, "role": "router"}),
        f"/models?model={model}": httpx.Response(
            200,
            json={"data": [{"id": model, "status": {"value": "loading"}}]},
        ),
    }
    with patch("app.data.capabilities.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value = _mock_client(responses)
        state = probe_engine_state(
            backend="127.0.0.1:8080",
            kind="chat",
            model=model,
            engine="llama-router",
        )
    assert state.admission == Admission.LOADING
    assert state.detail == "model_loading"


def test_llama_health_loading():
    responses = {
        "/health": httpx.Response(
            503,
            json={"error": {"message": "Loading model", "code": 503}},
        ),
    }
    with patch("app.data.capabilities.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value = _mock_client(responses)
        state = probe_engine_state(
            backend="127.0.0.1:8080",
            kind="chat",
            engine="llama.cpp",
        )
    assert state.admission == Admission.LOADING


def test_ollama_model_mismatch():
    responses = {
        "/api/ps": httpx.Response(
            200,
            json={"models": [{"name": "llama3:latest"}]},
        ),
    }
    with patch("app.data.capabilities.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value = _mock_client(responses)
        state = probe_engine_state(
            backend="127.0.0.1:11434",
            kind="chat",
            model="qwen2.5",
            engine="ollama",
        )
    assert state.admission == Admission.MODEL_MISMATCH


def test_ollama_requested_model_loaded():
    responses = {
        "/api/ps": httpx.Response(
            200,
            json={"models": [{"name": "jarvis:latest"}]},
        ),
    }
    with patch("app.data.capabilities.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value = _mock_client(responses)
        state = probe_engine_state(
            backend="127.0.0.1:11434",
            kind="chat",
            model="jarvis",
            engine="ollama",
        )
    assert state.admission == Admission.OK
    assert state.model_loaded is True


def test_strict_preflight_blocks_unreachable():
    with patch("app.upstream_preflight.probe_engine_state") as probe:
        from app.data.capabilities import EngineState

        probe.return_value = EngineState(
            "llama.cpp",
            Admission.DOWN,
            "connection_refused",
            probed_at=1.0,
        )
        pf = preflight_upstream(backend="127.0.0.1:8080", kind="chat", engine="llama.cpp")
    assert not pf.ok
    assert pf.reason == "backend_unreachable"


def test_strict_preflight_no_fail_open_on_down():
    with patch("app.upstream_preflight.probe_engine_state") as probe:
        from app.data.capabilities import EngineState

        probe.return_value = EngineState(
            "",
            Admission.PROBE_FAILED,
            "engine_undetected",
            probed_at=1.0,
        )
        pf = preflight_upstream(backend="127.0.0.1:8080", kind="chat")
    assert not pf.ok
    assert pf.reason == "probe_failed"


def test_admission_reason_mapping():
    from app.data.capabilities import EngineState

    reason, retry, retryable = admission_reason(
        EngineState("ollama", Admission.MODEL_MISMATCH, probed_at=1.0)
    )
    assert reason == "model_mismatch"
    assert retry == 15
    assert retryable is True

    reason, retry, retryable = admission_reason(
        EngineState("llama.cpp", Admission.DOWN, probed_at=1.0)
    )
    assert reason == "backend_unreachable"
    assert retry == 0
    assert retryable is False

    reason, retry, retryable = admission_reason(
        EngineState("llama.cpp", Admission.BUSY, probed_at=1.0)
    )
    assert reason == "all_slots_full"
    assert retry == 5
    assert retryable is True
