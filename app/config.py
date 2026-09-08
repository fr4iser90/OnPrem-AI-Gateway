from functools import lru_cache
import re

from pydantic_settings import BaseSettings, SettingsConfigDict

from .data.dialects import (  # noqa: F401 — re-export
    API_STYLES,
    dialect_choices,
    map_upstream_path,
    resolve_api_style,
)


# Functional kinds (path families). Source *names* are free-form slugs in the DB.
KINDS = ("chat", "embed", "extractor", "stt", "tts")
MODEL_CHECK_KINDS = frozenset(KINDS)
MODEL_REQUIRED_KINDS = frozenset({"chat", "embed", "extractor"})
MODEL_ROUTE_KINDS = frozenset(KINDS)

SOURCE_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

KIND_PATH_HINTS = {
    "chat": "/v1/chat/completions · /api/*",
    "embed": "/v1/embeddings",
    "extractor": "/v1/extract",
    "stt": "/v1/audio/transcriptions",
    "tts": "/v1/audio/speech",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    domain: str = "localhost"
    public_host: str = ""
    data_dir: str = "/data"
    session_secret: str = "change-me-session-secret"
    admin_bootstrap_user: str = "admin"
    admin_bootstrap_password: str = "changeme"
    session_max_age: int = 60 * 60 * 12
    # None = auto: Secure cookie when PUBLIC_HOST is set (Traefik/TLS), else off for local HTTP.
    session_cookie_secure: bool | None = None

    llm_api_key: str | None = None
    ollama_api_key: str | None = None
    embed_api_key: str | None = None
    extractor_api_key: str | None = None
    stt_api_key: str | None = None
    tts_api_key: str | None = None

    chat_source: str = ""
    chat2_source: str = ""
    embed_source: str = ""
    extractor_source: str = ""
    stt_source: str = ""
    tts_source: str = ""
    chat_backend: str = ""
    chat2_backend: str = ""
    llm_backend: str = ""
    ollama_backend: str = ""
    embed_backend: str = ""
    extractor_backend: str = ""
    stt_backend: str = ""
    tts_backend: str = ""

    temp_max_c: str = "30"
    temp_guard_disabled: bool = False
    # If the sidecar is unreachable or returns an unexpected status:
    # - true  → allow traffic (fail-open)
    # - false → reject traffic (fail-closed)
    temp_guard_fail_open: bool = True
    # Deprecated legacy override; normal behavior derives /check from each source host.
    temp_guard_url: str = "http://source-sidecar:8080/check"
    # Optional source-sidecar power probe. Empty = off.
    gpu_power_url: str = ""
    display_timezone: str = "UTC"
    # Impressum / operator of this install (public /legal/*). Env wins over the Settings form.
    operator_name: str = ""
    operator_address: str = ""
    operator_email: str = ""
    operator_phone: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()


def split_source_path(path: str) -> tuple[str | None, str]:
    """If path is /s/{name}/..., return (name, upstream_path). Else (None, path)."""
    p = (path or "").split("?")[0]
    parts = p.split("/")
    if len(parts) >= 3 and parts[1].lower() == "s":
        name = parts[2].lower()
        if SOURCE_NAME_RE.match(name):
            rest = "/" + "/".join(parts[3:]) if len(parts) > 3 else "/"
            return name, rest
    return None, p


def kind_from_upstream_path(upstream: str) -> str | None:
    """Map stripped upstream path → kind (for default sources)."""
    p = (upstream or "").split("?")[0].lower()
    if p.startswith("/api/"):
        return "chat"
    if p.startswith("/v1/embeddings"):
        return "embed"
    if p.startswith("/v1/extract"):
        return "extractor"
    if p.startswith("/v1/audio/transcriptions") or p.startswith("/v1/audio/translations"):
        return "stt"
    if p.startswith("/v1/audio/speech"):
        return "tts"
    if (
        p.startswith("/v1/chat/completions")
        or p.startswith("/v1/completions")
        or p.startswith("/completion")
        or p.startswith("/v1/models")
        or p.startswith("/v1/")
        or p.startswith("/chat")
    ):
        return "chat"
    return None


def upstream_path_for_proxy(path: str) -> str:
    """Path sent to the upstream (strips /s/{name} when present)."""
    named, upstream = split_source_path(path)
    if named is not None:
        return upstream or "/"
    return (path or "").split("?")[0] or "/"


def public_route_for_source(
    name: str,
    kind: str,
    *,
    is_default: bool = False,
    settings: Settings | None = None,
) -> str:
    """Client path for this kind. All sources of a kind share /v1; model picks the box."""
    del name, is_default  # equal sources — catalog merge, not /s/{name} vs primary
    settings = settings or get_settings()
    base = (settings.public_host or "").strip() or "onprem-api"
    hint = KIND_PATH_HINTS.get(kind, "/")
    return f"{base}{hint}"


SERVICES = KINDS
MODEL_CHECK_SERVICES = MODEL_CHECK_KINDS

SESSION_COOKIE_NAME = "onprem_session"
DB_FILENAME = "onprem.db"
DEFAULT_API_PORT = "9081"


def session_cookie_https_only(settings: Settings | None = None) -> bool:
    """Secure (HTTPS-only) session cookie — auto off for local HTTP, on for PUBLIC_HOST/TLS."""
    import os

    settings = settings or get_settings()
    if settings.session_cookie_secure is not None:
        return bool(settings.session_cookie_secure)
    raw = (os.getenv("SESSION_COOKIE_SECURE") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    host = (settings.public_host or os.getenv("PUBLIC_HOST") or "").strip()
    return bool(host and host != "_")


def onprem_api_port(explicit: str | None = None) -> str:
    """Published local API port (compose maps ONPREM_API_PORT → nginx)."""
    import os

    if explicit:
        return explicit.strip()
    return (os.getenv("ONPREM_API_PORT") or "").strip() or DEFAULT_API_PORT


def public_api_base(*, api_port: str | None = None) -> str:
    """Client-facing OpenAI-compatible base (…/v1), no trailing slash after v1."""
    import os

    settings = get_settings()
    host = (settings.public_host or os.getenv("PUBLIC_HOST") or "").strip()
    if host and host != "_":
        host = host.split("/")[0].strip()
        if host.startswith("http://") or host.startswith("https://"):
            return f"{host.rstrip('/')}/v1"
        # Homelab public names are almost always TLS via reverse proxy
        return f"https://{host}/v1"
    port = onprem_api_port(api_port)
    return f"http://localhost:{port}/v1"
