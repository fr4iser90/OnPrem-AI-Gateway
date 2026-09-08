"""Auth model/voice extraction and allowlist kinds."""

from __future__ import annotations

import json

from app.auth.check import extract_model, extract_request_model, extract_voice
from app.config import MODEL_CHECK_KINDS, MODEL_REQUIRED_KINDS


def test_model_check_kinds_include_stt_tts():
    assert MODEL_CHECK_KINDS == frozenset({"chat", "embed", "extractor", "stt", "tts"})
    assert MODEL_REQUIRED_KINDS == frozenset({"chat", "embed", "extractor"})


def test_extract_voice_from_tts_json():
    body = json.dumps(
        {"input": "hello", "voice": "de_DE-thorsten-high", "response_format": "wav"}
    ).encode()
    assert extract_model(body, "application/json") is None
    assert extract_voice(body, "application/json") == "de_DE-thorsten-high"


def test_extract_request_model_uses_voice_for_tts():
    body = json.dumps({"input": "hi", "voice": "de_DE-thorsten-high"}).encode()
    assert extract_request_model("tts", body, "application/json") == "de_DE-thorsten-high"
    assert extract_request_model("chat", body, "application/json") is None


def test_extract_model_prefers_model_field():
    body = json.dumps(
        {"model": "tts-1", "input": "hello", "voice": "de_DE-thorsten-high"}
    ).encode()
    assert extract_model(body, "application/json") == "tts-1"
    assert extract_voice(body, "application/json") == "de_DE-thorsten-high"


def test_extract_model_from_large_chat_body():
    """model must be found when messages exceed the old 64KiB json.loads window."""
    model = "Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL"
    huge = "x" * 200_000
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": huge}]}).encode(
        "utf-8"
    )
    assert len(body) > 65536
    assert extract_model(body, "application/json") == model
    assert extract_request_model("chat", body, "application/json") == model
