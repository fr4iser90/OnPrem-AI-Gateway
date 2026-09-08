"""Upstream API dialects: OpenAI client paths → backend-native paths.

Clients always call OpenAI-style routes (/v1/…). The DB stores ``api_style`` per
source; this registry defines how (and if) paths are rewritten for nginx.
Add a new dialect here when you wire a backend with a different path shape.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ApiDialect:
    """One upstream wire format OnPrem AI Gateway knows how to talk to."""

    id: str
    label: str
    summary: str
    kinds: tuple[str, ...]  # which source kinds this dialect suits
    # client path prefix → upstream path (first matching prefix wins)
    path_map: tuple[tuple[str, str], ...] = ()
    examples: str = ""  # short UI hint


# Concrete dialects (not including "auto")
DIALECTS: dict[str, ApiDialect] = {
    "openai": ApiDialect(
        id="openai",
        label="OpenAI / llama.cpp",
        summary="Same /v1/… paths as the client (llama.cpp server, Ollama OpenAI API, most embeds). Extractor rewrites to chat completions.",
        kinds=("chat", "embed", "extractor", "stt", "tts"),
        path_map=(("/v1/extract", "/v1/chat/completions"),),
        examples="/v1/chat/completions, /v1/embeddings; /v1/extract → /v1/chat/completions",
    ),
    "piper": ApiDialect(
        id="piper",
        label="Piper TTS",
        summary="Self-hosted Piper (or similar) that exposes POST /audio/speech instead of /v1/audio/speech.",
        kinds=("tts",),
        path_map=(("/v1/audio/speech", "/audio/speech"),),
        examples="Client /v1/audio/speech → upstream /audio/speech",
    ),
    "whisper_cpp": ApiDialect(
        id="whisper_cpp",
        label="whisper.cpp",
        summary="Official whisper.cpp HTTP server: multipart POST /inference (not /v1/audio/transcriptions).",
        kinds=("stt",),
        path_map=(
            ("/v1/audio/transcriptions", "/inference"),
            ("/v1/audio/translations", "/inference"),
        ),
        examples="Client /v1/audio/transcriptions → upstream /inference",
    ),
}

# Default dialect when api_style is "auto" or empty, by source kind
AUTO_DIALECT_BY_KIND: dict[str, str] = {
    "chat": "openai",
    "embed": "openai",
    "extractor": "openai",
    "tts": "piper",
    "stt": "whisper_cpp",
}

# Values allowed in DB / forms (auto + registry keys)
API_STYLES: tuple[str, ...] = ("auto", *DIALECTS.keys())


def dialect_choices() -> list[dict[str, str]]:
    """UI dropdown rows: id, label, summary."""
    rows = [
        {
            "id": "auto",
            "label": "Auto (by kind)",
            "summary": "Not a live probe — chat/embed/extractor→OpenAI/llama.cpp paths, stt→whisper.cpp, tts→Piper. Override only if your server differs.",
        }
    ]
    for d in DIALECTS.values():
        rows.append({"id": d.id, "label": d.label, "summary": d.summary})
    return rows


def resolve_api_style(kind: str, api_style: str | None) -> str:
    style = (api_style or "").strip().lower() or "auto"
    if style == "auto":
        return AUTO_DIALECT_BY_KIND.get(kind, "openai")
    if style in DIALECTS:
        return style
    return "openai"


def get_dialect(dialect_id: str) -> ApiDialect:
    return DIALECTS.get(dialect_id) or DIALECTS["openai"]


def effective_dialect_label(kind: str, api_style: str | None) -> str:
    """Human label for the dialect actually used for path rewrite."""
    return get_dialect(resolve_api_style(kind, api_style)).label


def map_upstream_path(client_path: str, *, kind: str, api_style: str | None = None) -> str:
    """Translate OpenAI client paths to backend-native paths for the resolved dialect."""
    p = (client_path or "").split("?")[0] or "/"
    dialect = get_dialect(resolve_api_style(kind, api_style))
    for prefix, target in dialect.path_map:
        if p == prefix or p.startswith(prefix + "/"):
            return target
    return p


def dialect_blurb_for_kind(kind: str, api_style: str | None = None) -> str:
    """One-line help for Services UI (shows saved mode + effective dialect)."""
    saved = (api_style or "auto").strip().lower() or "auto"
    effective = resolve_api_style(kind, saved)
    d = get_dialect(effective)
    if saved == "auto":
        return f"Auto → uses {d.label}" + (f" · {d.examples}" if d.examples else "")
    return f"Forced {d.label}" + (f" · {d.examples}" if d.examples else "")
