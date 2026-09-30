"""Unit tests for upstream engine fingerprinting."""

from app.data.probe import fingerprint_engine, _models_hint


def test_llama_cpp_via_slots():
    assert (
        fingerprint_engine(kind="chat", probes_ok=["/health", "/slots"], slots_total=4)
        == "llama.cpp"
    )


def test_llama_router_via_props():
    assert (
        fingerprint_engine(
            kind="chat",
            probes_ok=["/health", "/props", "/v1/models"],
            slots_total=None,
            body_hints='role:router build_info:b8771 model_alias:llama-server',
        )
        == "llama-router"
    )


def test_llama_cpp_via_props_without_router_role():
    assert (
        fingerprint_engine(
            kind="embed",
            probes_ok=["/props", "/v1/models"],
            slots_total=None,
            body_hints="build_info:b123 llama.cpp",
        )
        == "llama.cpp"
    )


def test_ollama_via_api_tags():
    assert (
        fingerprint_engine(kind="chat", probes_ok=["/api/tags"], slots_total=None) == "ollama"
    )


def test_vllm_owned_by():
    assert (
        fingerprint_engine(
            kind="chat",
            probes_ok=["/v1/models"],
            slots_total=None,
            body_hints="owned_by:vllm id:meta-llama",
        )
        == "vllm"
    )


def test_vllm_version_body():
    assert (
        fingerprint_engine(
            kind="chat",
            probes_ok=["/health", "/version"],
            slots_total=None,
            body_hints='{"version":"0.6.0","vllm":true}',
        )
        == "vllm"
    )


def test_localai_header():
    assert (
        fingerprint_engine(
            kind="chat",
            probes_ok=["/v1/models"],
            slots_total=None,
            header_hints="x-localai-version:2.20.1 server:localai",
        )
        == "localai"
    )


def test_lmstudio_model_id():
    assert (
        fingerprint_engine(
            kind="chat",
            probes_ok=["/v1/models"],
            slots_total=None,
            body_hints="id:lmstudio-community/qwen2.5 owned_by:organization",
        )
        == "lmstudio"
    )


def test_tei_info_path():
    assert (
        fingerprint_engine(
            kind="embed",
            probes_ok=["/info"],
            slots_total=None,
            body_hints="model_id:BAAI/bge-small",
        )
        == "tei"
    )


def test_tei_body_token():
    assert (
        fingerprint_engine(
            kind="embed",
            probes_ok=["/health"],
            slots_total=None,
            body_hints="text-embeddings-inference",
        )
        == "tei"
    )


def test_gufo_owned_by_beats_placeholder_slots():
    # Gufo advertises llama-shaped /slots that must not drive admission.
    assert (
        fingerprint_engine(
            kind="chat",
            probes_ok=["/health", "/slots", "/v1/models"],
            slots_total=3,
            body_hints="owned_by:gufo id:Qwen3.8-Flash-Next",
        )
        == "gufo"
    )


def test_gufo_without_slots():
    assert (
        fingerprint_engine(
            kind="chat",
            probes_ok=["/health", "/v1/models"],
            slots_total=None,
            body_hints="owned_by:gufo id:chat",
        )
        == "gufo"
    )


def test_openai_api_fallback():
    assert (
        fingerprint_engine(kind="chat", probes_ok=["/v1/models"], slots_total=None)
        == "openai-api"
    )


def test_tei_requires_real_info_not_just_path_typo():
    # /v1/models alone on embed → openai-api, not tei
    assert (
        fingerprint_engine(
            kind="embed",
            probes_ok=["/v1/models"],
            slots_total=None,
            body_hints="id:bge-m3-Q4_K_M",
        )
        == "openai-api"
    )


def test_stt_tts_soft():
    assert fingerprint_engine(kind="stt", probes_ok=["/health"], slots_total=None) == "whisper.cpp?"
    assert fingerprint_engine(kind="tts", probes_ok=["/"], slots_total=None) == "piper?"
    assert (
        fingerprint_engine(
            kind="stt",
            probes_ok=["/health"],
            slots_total=None,
            body_hints="faster-whisper",
        )
        == "faster-whisper"
    )


def test_models_hint_extracts_owned_by():
    hint = _models_hint(
        {
            "object": "list",
            "data": [{"id": "llama-3", "owned_by": "vllm", "object": "model"}],
        }
    )
    assert "owned_by:vllm" in hint
    assert "id:llama-3" in hint


def test_priority_llama_beats_vllm_hint():
    # Real llama.cpp with /slots wins even if models body mentions unrelated tokens
    assert (
        fingerprint_engine(
            kind="chat",
            probes_ok=["/slots", "/v1/models"],
            slots_total=2,
            body_hints="owned_by:vllm",
        )
        == "llama.cpp"
    )
