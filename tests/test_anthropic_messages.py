"""Anthropic Messages client dialect: request/response/stream translation."""

from __future__ import annotations

import json

import pytest

from app.anthropic_messages import (
    ANTHROPIC_SSE_PING_INTERVAL_SEC,
    OpenAIToAnthropicStream,
    anthropic_request_to_openai,
    anthropic_sse_ping,
    is_anthropic_messages_path,
    openai_response_to_anthropic,
)
from app.config import kind_from_upstream_path, map_upstream_path


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/v1/messages", True),
        ("/v1/messages/", True),
        ("/v1/messages?x=1", True),
        ("/s/gpu1/v1/messages", True),
        ("/v1/chat/completions", False),
        ("/v1/models", False),
    ],
)
def test_is_anthropic_messages_path(path: str, expected: bool):
    assert is_anthropic_messages_path(path) is expected


def test_anthropic_sse_ping_format():
    ping = anthropic_sse_ping()
    assert ping.startswith(b"event: ping\n")
    assert b'"type": "ping"' in ping
    assert ping.endswith(b"\n\n")
    assert 0 < ANTHROPIC_SSE_PING_INTERVAL_SEC <= 5.0


def test_stream_bootstrap_emits_message_start_once():
    xform = OpenAIToAnthropicStream(model="m1")
    first = b"".join(xform.bootstrap())
    assert b"event: message_start" in first
    assert b'"model": "m1"' in first
    # Second bootstrap must be a no-op (already started).
    assert xform.bootstrap() == []
    # Later feed must not duplicate message_start.
    chunk = (
        b'data: {"id":"1","object":"chat.completion.chunk","model":"m1",'
        b'"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
    )
    joined = b"".join(xform.feed(chunk))
    assert b"event: message_start" not in joined
    assert b"text_delta" in joined


def test_stream_bootstrap_then_empty_flush_has_content_block():
    xform = OpenAIToAnthropicStream(model="m1")
    assert b"message_start" in b"".join(xform.bootstrap())
    closed = b"".join(xform.flush())
    assert b"content_block_start" in closed
    assert b"content_block_stop" in closed
    assert b"message_stop" in closed


def test_kind_and_path_map_for_messages():
    assert kind_from_upstream_path("/v1/messages") == "chat"
    assert (
        map_upstream_path("/v1/messages", kind="chat", api_style="openai")
        == "/v1/chat/completions"
    )
    assert (
        map_upstream_path("/v1/messages", kind="chat", api_style="auto")
        == "/v1/chat/completions"
    )


def test_request_text_and_system():
    raw = json.dumps(
        {
            "model": "qwen-test",
            "max_tokens": 128,
            "system": "Be brief.",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.2,
            "stop_sequences": ["END"],
        }
    ).encode()
    out, stream = anthropic_request_to_openai(raw)
    assert stream is False
    data = json.loads(out)
    assert data["model"] == "qwen-test"
    assert data["max_tokens"] == 128
    assert data["temperature"] == 0.2
    assert data["stop"] == ["END"]
    assert data["messages"][0] == {"role": "system", "content": "Be brief."}
    assert data["messages"][1] == {"role": "user", "content": "Hi"}


def test_request_system_role_inside_messages():
    """Claude Code mid-conversation system → user wrap (llama.cpp rejects mid system)."""
    raw = json.dumps(
        {
            "model": "local-alias",
            "max_tokens": 256,
            "thinking": {"type": "adaptive"},
            "metadata": {"user_id": "x"},
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "system", "content": "Stay concise."},
                {"role": "user", "content": "again"},
            ],
        }
    ).encode()
    out, _ = anthropic_request_to_openai(raw)
    msgs = json.loads(out)["messages"]
    assert msgs[0] == {"role": "user", "content": "hi"}
    assert msgs[1] == {"role": "assistant", "content": "hello"}
    assert msgs[2]["role"] == "user"
    assert "Stay concise." in msgs[2]["content"]
    assert "<system>" in msgs[2]["content"]
    assert msgs[3] == {"role": "user", "content": "again"}
    assert "thinking" not in json.loads(out)


def test_request_leading_system_in_messages_stays_system():
    raw = json.dumps(
        {
            "model": "x",
            "max_tokens": 10,
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ],
        }
    ).encode()
    msgs = json.loads(anthropic_request_to_openai(raw)[0])["messages"]
    assert msgs[0] == {"role": "system", "content": "You are helpful."}
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_request_empty_system_message_skipped():
    raw = json.dumps(
        {
            "model": "x",
            "max_tokens": 10,
            "messages": [
                {"role": "system", "content": []},
                {"role": "user", "content": "hi"},
            ],
        }
    ).encode()
    msgs = json.loads(anthropic_request_to_openai(raw)[0])["messages"]
    assert msgs == [{"role": "user", "content": "hi"}]


def test_request_default_max_tokens_and_image():
    raw = json.dumps(
        {
            "model": "vl",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "aaa",
                            },
                        },
                    ],
                }
            ],
        }
    ).encode()
    out, stream = anthropic_request_to_openai(raw)
    assert stream is False
    data = json.loads(out)
    assert data["max_tokens"] == 4096
    content = data["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "what?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,aaa")


def test_request_tools_and_tool_choice():
    raw = json.dumps(
        {
            "model": "x",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Weather lookup",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            "tool_choice": {"type": "any"},
        }
    ).encode()
    out, _ = anthropic_request_to_openai(raw)
    data = json.loads(out)
    assert data["tool_choice"] == "required"
    assert data["tools"][0]["type"] == "function"
    assert data["tools"][0]["function"]["name"] == "get_weather"
    assert data["tools"][0]["function"]["parameters"]["required"] == ["city"]


def test_request_tool_use_and_tool_result_history():
    raw = json.dumps(
        {
            "model": "x",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "temp?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"city": "Berlin"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "12C",
                        }
                    ],
                },
            ],
        }
    ).encode()
    out, _ = anthropic_request_to_openai(raw)
    msgs = json.loads(out)["messages"]
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["tool_calls"][0]["id"] == "toolu_1"
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(msgs[1]["tool_calls"][0]["function"]["arguments"]) == {
        "city": "Berlin"
    }
    assert msgs[2] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "12C",
    }


def test_response_tool_calls_to_tool_use():
    openai = {
        "id": "chatcmpl-1",
        "model": "qwen-test",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Berlin"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    out = json.loads(openai_response_to_anthropic(json.dumps(openai).encode()))
    assert out["stop_reason"] == "tool_use"
    assert out["content"] == [
        {
            "type": "tool_use",
            "id": "call_abc",
            "name": "get_weather",
            "input": {"city": "Berlin"},
        }
    ]


def test_response_roundtrip_shape():
    openai = {
        "id": "chatcmpl-1",
        "model": "qwen-test",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }
    out = json.loads(openai_response_to_anthropic(json.dumps(openai).encode()))
    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["content"] == [{"type": "text", "text": "Hello!"}]
    assert out["stop_reason"] == "end_turn"
    assert out["usage"] == {"input_tokens": 3, "output_tokens": 2}
    assert out["model"] == "qwen-test"
    assert out["id"].startswith("msg_")


def test_openai_string_error_to_anthropic():
    raw = json.dumps(
        {"error": "upstream_unreachable", "detail": "connection refused"}
    ).encode()
    out = json.loads(openai_response_to_anthropic(raw))
    assert out["type"] == "error"
    assert "upstream_unreachable" in out["error"]["message"]
    assert "connection refused" in out["error"]["message"]


def test_stream_utf8_split_safe():
    """Multi-byte UTF-8 must survive chunk boundaries."""
    xform = OpenAIToAnthropicStream(model="m1")
    # "é" is C3 A9 — split after C3
    prefix = b'data: {"choices":[{"delta":{"content":"'
    mid = "caf\u00e9".encode("utf-8")
    suffix = b'"},"finish_reason":null}]}\n\n'
    assert mid[3:4] == b"\xc3"
    events = b"".join(xform.feed(prefix + mid[:4]))  # ends mid codepoint
    events += b"".join(xform.feed(mid[4:] + suffix))
    events += b"".join(xform.feed(b"data: [DONE]\n\n"))
    events += b"".join(xform.flush())
    assert "caf\u00e9" in events.decode("utf-8")


def test_stream_openai_to_anthropic_events():
    xform = OpenAIToAnthropicStream(model="m1")
    chunks = [
        b'data: {"id":"c1","model":"m1","choices":[{"delta":{"content":"He"},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"llo"},"finish_reason":null}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\n',
        b"data: [DONE]\n\n",
    ]
    events: list[str] = []
    for c in chunks:
        for piece in xform.feed(c):
            events.append(piece.decode())
    for piece in xform.flush():
        events.append(piece.decode())
    joined = "".join(events)
    assert "event: message_start" in joined
    assert "event: content_block_delta" in joined
    assert '"text": "He"' in joined
    assert '"text": "llo"' in joined
    assert "event: message_stop" in joined
    assert "end_turn" in joined


def test_stream_tool_calls():
    xform = OpenAIToAnthropicStream(model="m1")
    chunks = [
        (
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
            b'"type":"function","function":{"name":"get_weather","arguments":""}}]},'
            b'"finish_reason":null}]}\n\n'
        ),
        (
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            b'"function":{"arguments":"{\\"city\\":\\"Berlin\\"}"}}]},'
            b'"finish_reason":null}]}\n\n'
        ),
        b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    joined = "".join(
        piece.decode() for c in chunks for piece in xform.feed(c)
    ) + "".join(piece.decode() for piece in xform.flush())
    assert "tool_use" in joined
    assert "get_weather" in joined
    assert "input_json_delta" in joined
    assert "Berlin" in joined
    assert "tool_use" in joined
    assert '"stop_reason": "tool_use"' in joined
