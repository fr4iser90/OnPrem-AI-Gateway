from app.config import map_upstream_path, resolve_api_style
from app.data.dialects import API_STYLES, DIALECTS, dialect_choices


def test_auto_styles():
    assert resolve_api_style("tts", "auto") == "piper"
    assert resolve_api_style("stt", None) == "whisper_cpp"
    assert resolve_api_style("chat", "auto") == "openai"
    assert resolve_api_style("extractor", "auto") == "openai"


def test_map_piper_and_whisper():
    assert map_upstream_path("/v1/audio/speech", kind="tts") == "/audio/speech"
    assert map_upstream_path("/v1/audio/transcriptions", kind="stt") == "/inference"
    assert map_upstream_path("/v1/audio/translations", kind="stt") == "/inference"
    assert map_upstream_path("/v1/chat/completions", kind="chat") == "/v1/chat/completions"
    assert map_upstream_path("/v1/extract", kind="extractor") == "/v1/chat/completions"


def test_openai_style_keeps_paths():
    assert (
        map_upstream_path("/v1/audio/speech", kind="tts", api_style="openai")
        == "/v1/audio/speech"
    )
    assert (
        map_upstream_path("/v1/audio/transcriptions", kind="stt", api_style="openai")
        == "/v1/audio/transcriptions"
    )


def test_registry_has_expected_ids():
    assert "auto" in API_STYLES
    assert set(DIALECTS) == {"openai", "piper", "whisper_cpp"}
    ids = [c["id"] for c in dialect_choices()]
    assert ids[0] == "auto"
    assert "piper" in ids
