import pytest

from app.config import (
    KINDS,
    SOURCE_NAME_RE,
    kind_from_upstream_path,
    public_route_for_source,
    split_source_path,
    upstream_path_for_proxy,
)


def test_kinds_are_functional():
    assert KINDS == ("chat", "embed", "extractor", "stt", "tts")


def test_source_name_slug():
    assert SOURCE_NAME_RE.match("chat")
    assert SOURCE_NAME_RE.match("ollama")
    assert SOURCE_NAME_RE.match("gpu2")
    assert not SOURCE_NAME_RE.match("Chat")
    assert not SOURCE_NAME_RE.match("2gpu")


@pytest.mark.parametrize(
    "path,expected_kind",
    [
        ("/v1/chat/completions", "chat"),
        ("/v1/completions", "chat"),
        ("/v1/models", "chat"),
        ("/api/chat", "chat"),
        ("/api/tags", "chat"),
        ("/v1/embeddings", "embed"),
        ("/v1/extract", "extractor"),
        ("/v1/audio/transcriptions", "stt"),
        ("/v1/audio/translations", "stt"),
        ("/v1/audio/speech", "tts"),
        ("/unknown", None),
        ("/", None),
    ],
)
def test_kind_from_upstream_path(path: str, expected_kind: str | None):
    assert kind_from_upstream_path(path) == expected_kind


def test_named_source_prefix():
    assert split_source_path("/s/ollama/v1/models") == ("ollama", "/v1/models")
    assert split_source_path("/s/gpu2/api/tags") == ("gpu2", "/api/tags")
    assert upstream_path_for_proxy("/s/ollama/api/tags?x=1") == "/api/tags"
    assert upstream_path_for_proxy("/v1/models") == "/v1/models"
    # reserved / invalid slug rejected
    assert split_source_path("/s/Bad/v1/x")[0] == "bad"  # normalized lower
    assert split_source_path("/s/2bad/v1/x")[0] is None


def test_public_route_same_for_every_source_of_kind():
    chat = public_route_for_source("chat", "chat")
    test = public_route_for_source("test", "chat")
    assert chat == test
    assert "/v1/chat/completions" in chat
    assert "/s/" not in chat
    assert "/s/" not in public_route_for_source("embed-lab", "embed")
    assert "/v1/extract" in public_route_for_source("extractor", "extractor")
