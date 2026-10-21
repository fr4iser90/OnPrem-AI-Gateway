"""Anthropic Messages client dialect → OpenAI chat.completions upstream.

Clients may POST /v1/messages (Anthropic shape). The gateway translates to
OpenAI for local backends, meters on the OpenAI response, then translates back.
Tools / tool_use are out of scope for MVP.
"""

from __future__ import annotations

import codecs
import json
import re
import uuid
from typing import Any, Iterator


class AnthropicTranslateError(ValueError):
    """Invalid or unsupported Anthropic Messages request."""

    def __init__(self, message: str, *, err_type: str = "invalid_request_error"):
        super().__init__(message)
        self.err_type = err_type


# Path ends with /v1/messages (optional trailing slash; query stripped by callers)
_MESSAGES_SUFFIX = re.compile(r"/v1/messages/?$", re.IGNORECASE)


def is_anthropic_messages_path(path: str | None) -> bool:
    """True when the client URI is Anthropic Messages (incl. /s/{name}/…)."""
    p = (path or "").split("?")[0].rstrip("/") or "/"
    return bool(_MESSAGES_SUFFIX.search(p))


def anthropic_error_body(message: str, *, err_type: str = "api_error") -> dict[str, Any]:
    return {"type": "error", "error": {"type": err_type, "message": message}}


def _msg_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def _finish_to_stop_reason(finish: str | None) -> str:
    f = (finish or "").strip().lower()
    if f in {"length", "max_tokens"}:
        return "max_tokens"
    if f in {"tool_calls", "function_call"}:
        return "tool_use"
    if f in {"content_filter"}:
        return "refusal"
    return "end_turn"


def _system_to_text(system: Any) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        t = system.strip()
        return t or None
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if isinstance(block, str):
                if block.strip():
                    parts.append(block)
            elif isinstance(block, dict):
                if str(block.get("type") or "text").lower() == "text":
                    tx = block.get("text")
                    if isinstance(tx, str) and tx:
                        parts.append(tx)
        joined = "\n\n".join(parts).strip()
        return joined or None
    return None


def _anthropic_image_to_openai(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    st = str(source.get("type") or "").strip().lower()
    if st == "url":
        url = source.get("url")
        if isinstance(url, str) and url.strip():
            return {"type": "image_url", "image_url": {"url": url.strip()}}
        return None
    if st == "base64":
        data = source.get("data")
        media = str(source.get("media_type") or "image/png").strip() or "image/png"
        if isinstance(data, str) and data.strip():
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{media};base64,{data.strip()}"},
            }
    return None


def _content_to_openai(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return str(content)
    parts: list[Any] = []
    text_only: list[str] = []
    has_non_text = False
    for block in content:
        if isinstance(block, str):
            text_only.append(block)
            parts.append({"type": "text", "text": block})
            continue
        if not isinstance(block, dict):
            continue
        bt = str(block.get("type") or "text").strip().lower()
        if bt == "text":
            tx = block.get("text")
            if isinstance(tx, str):
                text_only.append(tx)
                parts.append({"type": "text", "text": tx})
        elif bt == "image":
            has_non_text = True
            img = _anthropic_image_to_openai(block)
            if img is not None:
                parts.append(img)
        elif bt in {"tool_use", "tool_result", "thinking", "redacted_thinking"}:
            raise AnthropicTranslateError(
                f"content block type '{bt}' is not supported yet",
                err_type="invalid_request_error",
            )
        else:
            tx = block.get("text")
            if isinstance(tx, str):
                text_only.append(tx)
                parts.append({"type": "text", "text": tx})
    if not has_non_text:
        return "".join(text_only)
    return parts if parts else ""


def anthropic_request_to_openai(body: bytes) -> tuple[bytes, bool]:
    """Translate Anthropic Messages JSON → OpenAI chat.completions JSON.

    Returns (openai_body, stream).
    """
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnthropicTranslateError(f"invalid JSON body: {exc}") from exc
    if not isinstance(data, dict):
        raise AnthropicTranslateError("request body must be a JSON object")

    if data.get("tools") is not None or data.get("tool_choice") is not None:
        raise AnthropicTranslateError(
            "tools / tool_choice are not supported on this gateway yet",
            err_type="invalid_request_error",
        )

    model = data.get("model")
    if not isinstance(model, str) or not model.strip():
        raise AnthropicTranslateError("model is required")

    messages_in = data.get("messages")
    if not isinstance(messages_in, list) or not messages_in:
        raise AnthropicTranslateError("messages must be a non-empty array")

    out_messages: list[dict[str, Any]] = []
    system_text = _system_to_text(data.get("system"))
    if system_text:
        out_messages.append({"role": "system", "content": system_text})

    for msg in messages_in:
        if not isinstance(msg, dict):
            raise AnthropicTranslateError("each message must be an object")
        role = str(msg.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            raise AnthropicTranslateError(
                f"unsupported message role '{role}' "
                "(use user/assistant; system is top-level)"
            )
        out_messages.append(
            {"role": role, "content": _content_to_openai(msg.get("content"))}
        )

    max_tokens = data.get("max_tokens")
    if max_tokens is None:
        max_tokens = 4096
    try:
        max_tokens_i = int(max_tokens)
    except (TypeError, ValueError) as exc:
        raise AnthropicTranslateError("max_tokens must be an integer") from exc
    if max_tokens_i < 1:
        raise AnthropicTranslateError("max_tokens must be >= 1")

    stream = bool(data.get("stream") or False)
    openai: dict[str, Any] = {
        "model": model.strip(),
        "messages": out_messages,
        "max_tokens": max_tokens_i,
        "stream": stream,
    }
    if "temperature" in data and data["temperature"] is not None:
        openai["temperature"] = data["temperature"]
    if "top_p" in data and data["top_p"] is not None:
        openai["top_p"] = data["top_p"]
    if "top_k" in data and data["top_k"] is not None:
        openai["top_k"] = data["top_k"]
    stops = data.get("stop_sequences")
    if isinstance(stops, list) and stops:
        openai["stop"] = stops
    if stream:
        # Ask OpenAI-shaped upstreams for usage on the final chunk when supported.
        openai["stream_options"] = {"include_usage": True}

    return json.dumps(openai, ensure_ascii=False).encode("utf-8"), stream


def openai_response_to_anthropic(
    body: bytes,
    *,
    model: str | None = None,
) -> bytes:
    """Translate a non-stream OpenAI chat.completion JSON → Anthropic message."""
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body
    if not isinstance(data, dict):
        return body
    # Already Anthropic-shaped
    if data.get("type") == "error" and isinstance(data.get("error"), dict):
        return body

    # OpenAI / gateway error shapes (dict or string error, optional detail)
    if "choices" not in data and (
        "error" in data or data.get("type") == "error" or "detail" in data
    ):
        err = data.get("error")
        et = "api_error"
        if isinstance(err, dict):
            msg = str(err.get("message") or err.get("code") or err)
            et = str(err.get("type") or et)
        elif isinstance(err, str) and err.strip():
            msg = err.strip()
            detail = data.get("detail")
            if detail is not None and str(detail).strip():
                msg = f"{msg}: {detail}"
        else:
            msg = str(data.get("detail") or data.get("error") or "upstream error")
        return json.dumps(anthropic_error_body(msg, err_type=et)).encode("utf-8")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return body
    choice0 = choices[0] if isinstance(choices[0], dict) else {}
    message = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}
    content = message.get("content")
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    elif isinstance(content, list):
        bits: list[str] = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                bits.append(p["text"])
            elif isinstance(p, str):
                bits.append(p)
        text = "".join(bits)
    else:
        text = str(content)

    usage_in = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    mid = data.get("id")
    if not isinstance(mid, str) or not mid.startswith("msg_"):
        mid = _msg_id()
    out_model = model or data.get("model") or ""
    out = {
        "id": mid,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": out_model,
        "stop_reason": _finish_to_stop_reason(choice0.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage_in.get("prompt_tokens") or 0),
            "output_tokens": int(usage_in.get("completion_tokens") or 0),
        },
    }
    return json.dumps(out, ensure_ascii=False).encode("utf-8")


def _sse_data_lines(chunk_text: str) -> Iterator[str]:
    """Yield payload strings from OpenAI-style ``data:`` lines in a text buffer."""
    for line in chunk_text.splitlines():
        if line.startswith("data:"):
            yield line[5:].strip()


class OpenAIToAnthropicStream:
    """Incremental OpenAI chat.completion SSE → Anthropic Messages SSE."""

    def __init__(self, *, model: str | None = None):
        self.model = (model or "").strip()
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._buf = ""
        self._started = False
        self._block_open = False
        self._msg_id = _msg_id()
        self._input_tokens = 0
        self._output_tokens = 0
        self._stop_reason = "end_turn"
        self._json_mode = False
        self._json_buf = bytearray()
        self._done = False
        self._saw_bytes = False

    def feed(self, chunk: bytes) -> list[bytes]:
        if self._done:
            return []
        if not chunk:
            return []
        if not self._saw_bytes and not self._json_mode:
            head = chunk.lstrip()[:1]
            if head == b"{":
                self._json_mode = True
        self._saw_bytes = True
        if self._json_mode:
            self._json_buf.extend(chunk)
            return []

        self._buf += self._decoder.decode(chunk, final=False)
        out: list[bytes] = []
        while True:
            if "\n" not in self._buf:
                break
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            if payload == "[DONE]":
                out.extend(self._close_events())
                self._done = True
                return out
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            out.extend(self._handle_chunk(obj))
        return out

    def flush(self) -> list[bytes]:
        if self._done:
            return []
        if self._json_mode:
            self._done = True
            return [
                openai_response_to_anthropic(
                    bytes(self._json_buf), model=self.model or None
                )
            ]
        # Flush any partial UTF-8 sequence held by the incremental decoder.
        self._buf += self._decoder.decode(b"", final=True)
        out: list[bytes] = []
        if self._buf.strip():
            for payload in _sse_data_lines(self._buf):
                if payload == "[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    out.extend(self._handle_chunk(obj))
            self._buf = ""
        out.extend(self._close_events())
        self._done = True
        return out

    def _emit(self, event: str, data: dict[str, Any]) -> bytes:
        return (
            f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode(
                "utf-8"
            )
        )

    def _ensure_started(self) -> list[bytes]:
        if self._started:
            return []
        self._started = True
        events = [
            self._emit(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": self._msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            ),
            self._emit(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
        ]
        self._block_open = True
        return events

    def _handle_chunk(self, obj: dict[str, Any]) -> list[bytes]:
        out: list[bytes] = []
        if not self.model and isinstance(obj.get("model"), str):
            self.model = obj["model"]
        usage = obj.get("usage")
        if isinstance(usage, dict):
            if usage.get("prompt_tokens") is not None:
                try:
                    self._input_tokens = int(usage["prompt_tokens"])
                except (TypeError, ValueError):
                    pass
            if usage.get("completion_tokens") is not None:
                try:
                    self._output_tokens = int(usage["completion_tokens"])
                except (TypeError, ValueError):
                    pass

        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices:
            return out
        choice0 = choices[0] if isinstance(choices[0], dict) else {}
        finish = choice0.get("finish_reason")
        if finish:
            self._stop_reason = _finish_to_stop_reason(str(finish))

        delta = choice0.get("delta") if isinstance(choice0.get("delta"), dict) else {}
        if not delta and isinstance(choice0.get("message"), dict):
            content = choice0["message"].get("content")
            if isinstance(content, str) and content:
                out.extend(self._ensure_started())
                out.append(
                    self._emit(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": content},
                        },
                    )
                )
            return out

        text = delta.get("content")
        if isinstance(text, str) and text:
            out.extend(self._ensure_started())
            out.append(
                self._emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
            )
        return out

    def _close_events(self) -> list[bytes]:
        out: list[bytes] = []
        if not self._started:
            out.extend(self._ensure_started())
        if self._block_open:
            out.append(
                self._emit(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": 0},
                )
            )
            self._block_open = False
        usage_delta: dict[str, int] = {"output_tokens": self._output_tokens}
        if self._input_tokens:
            usage_delta["input_tokens"] = self._input_tokens
        out.append(
            self._emit(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": self._stop_reason,
                        "stop_sequence": None,
                    },
                    "usage": usage_delta,
                },
            )
        )
        out.append(self._emit("message_stop", {"type": "message_stop"}))
        return out
