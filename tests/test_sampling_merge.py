"""Soft-merge catalog recommended_sampling into chat bodies (missing keys only)."""

from __future__ import annotations

import json

from app.sampling_merge import (
    apply_soft_sampling,
    merge_missing_sampling,
    resolve_sampling_profile,
)


_PROFILES = {
    "default_profile": "thinking",
    "thinking": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": "medium",
    },
    "instruct": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "enable_thinking": False,
        "preserve_thinking": False,
    },
}


def test_resolve_default_profile():
    p = resolve_sampling_profile(_PROFILES)
    assert p["temperature"] == 1.0


def test_resolve_key_profile_over_catalog_default():
    p = resolve_sampling_profile(_PROFILES, key_profile="instruct")
    assert p["temperature"] == 0.7


def test_resolve_header_beats_key_profile():
    p = resolve_sampling_profile(_PROFILES, hint="thinking", key_profile="instruct")
    assert p["temperature"] == 1.0


def test_resolve_enable_thinking_beats_key_profile():
    p = resolve_sampling_profile(
        _PROFILES, enable_thinking=False, key_profile="thinking"
    )
    assert p["temperature"] == 0.7


def test_normalize_key_sampling_profile():
    from app.sampling_merge import normalize_key_sampling_profile

    assert normalize_key_sampling_profile("") == ""
    assert normalize_key_sampling_profile("inherit") == ""
    assert normalize_key_sampling_profile("Instruct") == "instruct"
    assert normalize_key_sampling_profile("thinking") == "thinking"


def test_apply_soft_sampling_uses_key_profile():
    body = json.dumps({"model": "x", "messages": []}).encode()
    out = apply_soft_sampling(
        body, "application/json", profiles=_PROFILES, key_profile="instruct"
    )
    data = json.loads(out)
    assert data["temperature"] == 0.7
    assert data["chat_template_kwargs"]["enable_thinking"] is False


def test_merge_does_not_override_client():
    payload = {"model": "x", "temperature": 0.2, "messages": []}
    changed = merge_missing_sampling(payload, _PROFILES["thinking"])
    assert changed is True
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.95
    assert payload["chat_template_kwargs"]["enable_thinking"] is True
    assert payload["chat_template_kwargs"]["preserve_thinking"] is True
    assert payload["chat_template_kwargs"]["reasoning_effort"] == "medium"


def test_merge_preserves_client_template_kwargs():
    payload = {
        "model": "x",
        "messages": [],
        "chat_template_kwargs": {"preserve_thinking": False, "foo": 1},
    }
    changed = merge_missing_sampling(payload, _PROFILES["thinking"])
    assert changed is True
    assert payload["chat_template_kwargs"]["preserve_thinking"] is False
    assert payload["chat_template_kwargs"]["foo"] == 1
    assert payload["chat_template_kwargs"]["enable_thinking"] is True
    assert payload["chat_template_kwargs"]["reasoning_effort"] == "medium"


def test_apply_soft_sampling_roundtrip():
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": "hi"}]}).encode()
    out = apply_soft_sampling(body, "application/json", profiles=_PROFILES)
    assert out is not None
    data = json.loads(out)
    assert data["temperature"] == 1.0
    assert data["top_p"] == 0.95
    assert data["chat_template_kwargs"]["enable_thinking"] is True
    assert data["chat_template_kwargs"]["preserve_thinking"] is True
    assert data["chat_template_kwargs"]["reasoning_effort"] == "medium"


def test_apply_noop_when_client_set_all():
    body = json.dumps(
        {
            "model": "x",
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 10,
            "chat_template_kwargs": {
                "enable_thinking": False,
                "preserve_thinking": False,
                "reasoning_effort": "low",
            },
            "messages": [],
        }
    ).encode()
    out = apply_soft_sampling(body, "application/json", profiles=_PROFILES, profile_hint="instruct")
    assert out is body
