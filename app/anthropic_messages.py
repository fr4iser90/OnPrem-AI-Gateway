"""Anthropic Messages client dialect → OpenAI chat.completions upstream.

Full client-facing Anthropic Messages compatibility for gateways (Claude Code,
SDKs, etc.): accept mid-conversation ``role: system``, tools, vision, streaming,
and ignore Anthropic-only extras (``thinking``, ``cache_control``, betas) that
local OpenAI/llama.cpp upstreams do not understand — never 400 on them.

Translate request → OpenAI, meter on OpenAI upstream, translate response back.
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


# Anthropic Messages SSE keepalive (Claude Code stream idle watchdog).
# Emit while local upstreams are silent (thinking / prefill).
ANTHROPIC_SSE_PING = b'event: ping\ndata: {"type": "ping"}\n\n'
# SSE comment lines are ignored by the event parser but still count as wire
# bytes for Claude Code's byte-level idle watchdog (gateway / custom base URL).
ANTHROPIC_SSE_COMMENT_PING = b": keepalive\n\n"
# Claude Code shows "Waiting for API response" after ~20s wire silence.
ANTHROPIC_SSE_PING_INTERVAL_SEC = 2.0


def anthropic_sse_ping() -> bytes:
    """Keepalive: SSE comment bytes + Anthropic ``event: ping`` frame."""
    return ANTHROPIC_SSE_COMMENT_PING + ANTHROPIC_SSE_PING


def anthropic_sse_error(message: str, *, err_type: str = "api_error") -> bytes:
    """SSE ``error`` event (used when HTTP status is already 200 after early SSE)."""
    payload = anthropic_error_body(message, err_type=err_type)
    return (
        f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode(
            "utf-8"
        )
    )


def _msg_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def _tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


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


def _tool_result_content_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        try:
            return json.dumps(content, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(content)
    return str(content)


def _anthropic_tools_to_openai(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        raise AnthropicTranslateError("tools must be an array")
    out: list[dict[str, Any]] = []
    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue  # skip junk; don't 400 whole request
        ttype = str(tool.get("type") or "").strip().lower()
        # Server / beta tool types Claude Code may send — skip for local upstreams
        if ttype and ttype not in {"", "function", "custom"} and not tool.get("name"):
            continue
        if ttype.startswith("advisor") or "tool_reference" in ttype:
            continue
        # Already OpenAI-shaped
        if ttype == "function" and isinstance(tool.get("function"), dict):
            fn_in = tool["function"]
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(fn_in.get("name") or f"tool_{i}"),
                        "description": fn_in.get("description") or "",
                        "parameters": fn_in.get("parameters")
                        if isinstance(fn_in.get("parameters"), dict)
                        else {"type": "object", "properties": {}},
                    },
                }
            )
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        desc = tool.get("description")
        schema = tool.get("input_schema")
        if schema is None:
            schema = tool.get("parameters") or {"type": "object", "properties": {}}
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        fn: dict[str, Any] = {
            "name": name.strip(),
            "parameters": schema,
        }
        if isinstance(desc, str) and desc:
            fn["description"] = desc
        out.append({"type": "function", "function": fn})
    return out


def _anthropic_tool_choice_to_openai(choice: Any) -> Any:
    if choice is None:
        return None
    if isinstance(choice, str):
        c = choice.strip().lower()
        if c in {"auto", "none", "required"}:
            return c
        if c == "any":
            return "required"
        return "auto"  # unknown string → don't 400
    if not isinstance(choice, dict):
        return "auto"
    ct = str(choice.get("type") or "").strip().lower()
    if ct == "auto":
        return "auto"
    if ct == "any":
        return "required"
    if ct == "none":
        return "none"
    if ct == "tool":
        name = choice.get("name")
        if isinstance(name, str) and name.strip():
            return {"type": "function", "function": {"name": name.strip()}}
        return "auto"
    return "auto"


def _message_to_openai_messages(role: str, content: Any) -> list[dict[str, Any]]:
    """Expand one Anthropic message into one or more OpenAI chat messages."""
    if content is None:
        return [{"role": role, "content": ""}]
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return [{"role": role, "content": str(content)}]

    text_parts: list[Any] = []
    text_only: list[str] = []
    has_media = False
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    for block in content:
        if isinstance(block, str):
            text_only.append(block)
            text_parts.append({"type": "text", "text": block})
            continue
        if not isinstance(block, dict):
            continue
        bt = str(block.get("type") or "text").strip().lower()
        if bt == "text":
            tx = block.get("text")
            if isinstance(tx, str):
                text_only.append(tx)
                text_parts.append({"type": "text", "text": tx})
        elif bt == "image":
            has_media = True
            img = _anthropic_image_to_openai(block)
            if img is not None:
                text_parts.append(img)
        elif bt == "tool_use":
            tid = block.get("id")
            name = block.get("name")
            if not isinstance(name, str) or not name.strip():
                raise AnthropicTranslateError("tool_use.name is required")
            if not isinstance(tid, str) or not tid.strip():
                tid = _tool_call_id()
            inp = block.get("input")
            if inp is None:
                args = "{}"
            elif isinstance(inp, str):
                args = inp
            else:
                try:
                    args = json.dumps(inp, ensure_ascii=False)
                except (TypeError, ValueError) as exc:
                    raise AnthropicTranslateError(
                        "tool_use.input must be JSON-serializable"
                    ) from exc
            tool_calls.append(
                {
                    "id": tid.strip(),
                    "type": "function",
                    "function": {"name": name.strip(), "arguments": args},
                }
            )
        elif bt == "tool_result":
            tid = block.get("tool_use_id")
            if not isinstance(tid, str) or not tid.strip():
                raise AnthropicTranslateError("tool_result.tool_use_id is required")
            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": tid.strip(),
                    "content": _tool_result_content_to_str(block.get("content")),
                }
            )
        elif bt in {"thinking", "redacted_thinking"}:
            # Keep thinking text in history as plain text so context isn't lost.
            tx = block.get("thinking") or block.get("text")
            if isinstance(tx, str) and tx.strip():
                text_only.append(tx)
                text_parts.append({"type": "text", "text": tx})
        elif bt in {"document", "file"}:
            # Best-effort: flatten document text / title into text content.
            title = block.get("title")
            if isinstance(title, str) and title.strip():
                text_only.append(f"[document: {title.strip()}]")
                text_parts.append(
                    {"type": "text", "text": f"[document: {title.strip()}]"}
                )
            src = block.get("source") if isinstance(block.get("source"), dict) else {}
            if str(src.get("type") or "") == "text" and isinstance(src.get("data"), str):
                text_only.append(src["data"])
                text_parts.append({"type": "text", "text": src["data"]})
        else:
            # Unknown block: keep any text; never 400 the whole request.
            tx = block.get("text") or block.get("thinking")
            if isinstance(tx, str) and tx:
                text_only.append(tx)
                text_parts.append({"type": "text", "text": tx})

    out: list[dict[str, Any]] = []

    if role == "system":
        # Caller decides leading vs mid-conversation (see anthropic_request_to_openai).
        text = "".join(text_only) if not has_media else ""
        if has_media and text_parts:
            bits = []
            for p in text_parts:
                if isinstance(p, dict) and p.get("type") == "text":
                    bits.append(str(p.get("text") or ""))
            text = "".join(bits)
        out.append({"role": "system", "content": text})
        return out

    if role == "assistant":
        msg: dict[str, Any] = {"role": "assistant"}
        if has_media:
            msg["content"] = text_parts if text_parts else None
        elif text_only:
            msg["content"] = "".join(text_only)
        else:
            msg["content"] = None if tool_calls else ""
        if tool_calls:
            msg["tool_calls"] = tool_calls
            if msg.get("content") == "":
                msg["content"] = None
        out.append(msg)
        return out

    # user (or other): tool_results become separate role=tool messages
    if tool_results:
        out.extend(tool_results)
    if text_parts or text_only or has_media:
        if has_media:
            body: Any = text_parts if text_parts else ""
        else:
            body = "".join(text_only)
        out.append({"role": role, "content": body})
    elif not tool_results:
        out.append({"role": role, "content": ""})
    return out


def _openai_has_dialogue(messages: list[dict[str, Any]]) -> bool:
    """True once a user/assistant/tool turn exists (system may no longer be prepended)."""
    return any(
        isinstance(m, dict) and str(m.get("role") or "") not in {"", "system"}
        for m in messages
    )


def _mid_system_as_user(text: str) -> dict[str, Any]:
    """llama.cpp / OpenAI: system role only at the start — remap mid-turn system."""
    body = (text or "").strip()
    if body:
        return {"role": "user", "content": f"<system>\n{body}\n</system>"}
    return {"role": "user", "content": "<system></system>"}


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
            continue
        role = str(msg.get("role") or "").strip().lower()
        # Claude Code / newer Anthropic: mid-conversation role=system in messages[].
        if role not in {"user", "assistant", "system"}:
            role = "user"
        content = msg.get("content")
        # Empty system messages (Claude Code effort markers) → skip
        if role == "system" and (
            content is None
            or content == ""
            or (isinstance(content, list) and len(content) == 0)
        ):
            continue
        if role == "system":
            pieces = _message_to_openai_messages("system", content)
            text = ""
            if pieces and isinstance(pieces[0], dict):
                text = str(pieces[0].get("content") or "")
            # OpenAI/llama.cpp: "System message must be at the beginning."
            # Leading system stays system; after dialogue → user-wrapped (Claude Code retry shape).
            if _openai_has_dialogue(out_messages):
                out_messages.append(_mid_system_as_user(text))
            else:
                out_messages.append({"role": "system", "content": text})
            continue
        out_messages.extend(_message_to_openai_messages(role, content))

    max_tokens = data.get("max_tokens")
    if max_tokens is None:
        max_tokens = 4096
    try:
        max_tokens_i = int(max_tokens)
    except (TypeError, ValueError) as exc:
        raise AnthropicTranslateError("max_tokens must be an integer") from exc
    # Anthropic allows 0 (cache warm); OpenAI upstreams need >= 1 for generation.
    if max_tokens_i < 0:
        raise AnthropicTranslateError("max_tokens must be >= 0")
    if max_tokens_i == 0:
        max_tokens_i = 1

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

    tools = data.get("tools")
    if tools is not None:
        mapped = _anthropic_tools_to_openai(tools)
        if mapped:
            openai["tools"] = mapped
    if "tool_choice" in data and data["tool_choice"] is not None:
        openai["tool_choice"] = _anthropic_tool_choice_to_openai(data["tool_choice"])

    # Anthropic-only body fields (thinking, cache_control, output_config,
    # context_management, metadata, …) are intentionally dropped — local
    # OpenAI/llama.cpp upstreams reject them with 400.

    if stream:
        # Ask OpenAI-shaped upstreams for usage on the final chunk when supported.
        openai["stream_options"] = {"include_usage": True}

    if not out_messages:
        raise AnthropicTranslateError("messages must contain at least one usable turn")

    return json.dumps(openai, ensure_ascii=False).encode("utf-8"), stream


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_raw": parsed}


def _openai_message_content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        bits: list[str] = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                bits.append(p["text"])
            elif isinstance(p, str):
                bits.append(p)
        if bits:
            blocks.append({"type": "text", "text": "".join(bits)})

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = fn.get("name") if isinstance(fn.get("name"), str) else ""
            tid = tc.get("id")
            if not isinstance(tid, str) or not tid.strip():
                tid = _tool_call_id()
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tid.strip(),
                    "name": name or "tool",
                    "input": _parse_tool_arguments(fn.get("arguments")),
                }
            )
    return blocks


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
    blocks = _openai_message_content_blocks(message)
    if not blocks:
        blocks = [{"type": "text", "text": ""}]

    usage_in = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    mid = data.get("id")
    if not isinstance(mid, str) or not mid.startswith("msg_"):
        mid = _msg_id()
    out_model = model or data.get("model") or ""
    stop = _finish_to_stop_reason(choice0.get("finish_reason"))
    if any(b.get("type") == "tool_use" for b in blocks) and stop == "end_turn":
        stop = "tool_use"
    out = {
        "id": mid,
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "model": out_model,
        "stop_reason": stop,
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
        self._text_index: int | None = None
        self._text_open = False
        self._next_block_index = 0
        # OpenAI tool_call index → Anthropic content block index
        self._tool_index_map: dict[int, int] = {}
        self._tool_open: set[int] = set()
        self._msg_id = _msg_id()
        self._input_tokens = 0
        self._output_tokens = 0
        self._stop_reason = "end_turn"
        self._json_mode = False
        self._json_buf = bytearray()
        self._done = False
        self._saw_bytes = False

    def bootstrap(self) -> list[bytes]:
        """Emit ``message_start`` before the first upstream token.

        Claude Code and similar clients treat prolonged silence after HTTP 200
        headers as a dead stream. Starting the Anthropic envelope immediately
        (then SSE pings) keeps idle timers reset during llama.cpp prefill.
        """
        if self._done:
            return []
        return self._ensure_message_start()

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

    def _ensure_message_start(self) -> list[bytes]:
        if self._started:
            return []
        self._started = True
        return [
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
            )
        ]

    def _ensure_text_block(self) -> list[bytes]:
        out = self._ensure_message_start()
        if self._text_open:
            return out
        idx = self._next_block_index
        self._next_block_index += 1
        self._text_index = idx
        self._text_open = True
        out.append(
            self._emit(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        return out

    def _close_text_block(self) -> list[bytes]:
        if not self._text_open or self._text_index is None:
            return []
        idx = self._text_index
        self._text_open = False
        return [
            self._emit(
                "content_block_stop",
                {"type": "content_block_stop", "index": idx},
            )
        ]

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
            # Non-incremental full message in a chunk
            msg = choice0["message"]
            content = msg.get("content")
            if isinstance(content, str) and content:
                out.extend(self._ensure_text_block())
                out.append(
                    self._emit(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self._text_index,
                            "delta": {"type": "text_delta", "text": content},
                        },
                    )
                )
            tcs = msg.get("tool_calls")
            if isinstance(tcs, list):
                for i, tc in enumerate(tcs):
                    if isinstance(tc, dict):
                        out.extend(self._handle_tool_call_delta(i, tc))
            return out

        text = delta.get("content")
        if isinstance(text, str) and text:
            out.extend(self._ensure_text_block())
            out.append(
                self._emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._text_index,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
            )

        tool_deltas = delta.get("tool_calls")
        if isinstance(tool_deltas, list):
            for td in tool_deltas:
                if not isinstance(td, dict):
                    continue
                try:
                    idx = int(td.get("index", 0))
                except (TypeError, ValueError):
                    idx = 0
                out.extend(self._handle_tool_call_delta(idx, td))
        return out

    def _handle_tool_call_delta(self, openai_index: int, td: dict[str, Any]) -> list[bytes]:
        out: list[bytes] = []
        out.extend(self._ensure_message_start())
        # Close text before first tool block
        if openai_index not in self._tool_index_map and self._text_open:
            out.extend(self._close_text_block())

        if openai_index not in self._tool_index_map:
            aidx = self._next_block_index
            self._next_block_index += 1
            self._tool_index_map[openai_index] = aidx
            tid = td.get("id")
            if not isinstance(tid, str) or not tid.strip():
                tid = _tool_call_id()
            fn = td.get("function") if isinstance(td.get("function"), dict) else {}
            name = fn.get("name") if isinstance(fn.get("name"), str) else ""
            out.append(
                self._emit(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": aidx,
                        "content_block": {
                            "type": "tool_use",
                            "id": tid.strip(),
                            "name": name or "tool",
                            "input": {},
                        },
                    },
                )
            )
            self._tool_open.add(aidx)
            # Name may arrive on a later delta — patch via input only; name is set at start.
            # If name was empty and arrives later, emit nothing (Anthropic requires name at start).
            args = fn.get("arguments")
            if isinstance(args, str) and args:
                out.append(
                    self._emit(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": aidx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": args,
                            },
                        },
                    )
                )
            return out

        aidx = self._tool_index_map[openai_index]
        fn = td.get("function") if isinstance(td.get("function"), dict) else {}
        # Late name with empty start: rare; ignore if already started.
        args = fn.get("arguments")
        if isinstance(args, str) and args:
            out.append(
                self._emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": aidx,
                        "delta": {"type": "input_json_delta", "partial_json": args},
                    },
                )
            )
        return out

    def _close_events(self) -> list[bytes]:
        out: list[bytes] = []
        if not self._started:
            out.extend(self._ensure_message_start())
        # Empty / aborted completion (incl. after bootstrap()): Anthropic clients
        # expect at least one content block before message_delta/stop.
        if not self._text_open and not self._tool_open and self._text_index is None:
            out.extend(self._ensure_text_block())
        out.extend(self._close_text_block())
        for aidx in sorted(self._tool_open):
            out.append(
                self._emit(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": aidx},
                )
            )
        self._tool_open.clear()
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
