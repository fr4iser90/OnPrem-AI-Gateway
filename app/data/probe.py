"""Best-effort upstream probes (llama health/slots, ollama, generic /health)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
from sqlalchemy.orm import Session

from ..config import MODEL_CHECK_KINDS
from .backends import list_sources
from .models import BackendSource

_TIMEOUT = 2.0
_loading_since: dict[str, float] = {}


@dataclass
class ServiceStatus:
    service: str
    backend: str
    state: str = "down"
    detail: str = ""
    kind: str = ""
    slots_total: int | None = None
    slots_busy: int | None = None
    slots_idle: int | None = None
    models: list[str] = field(default_factory=list)
    probes_ok: list[str] = field(default_factory=list)
    gpu_power: str = "off"  # ok | unreachable | off
    gpu_watts: float | None = None
    gpu_power_url: str = ""
    # Best-effort fingerprint from probe paths / headers /models (≠ api_style dialect)
    engine: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _base_url(backend: str) -> str:
    return f"http://{backend}"


def _get(client: httpx.Client, url: str) -> httpx.Response | None:
    try:
        return client.get(url)
    except Exception:
        return None


def _health_state(resp: httpx.Response) -> tuple[str, str] | None:
    if resp.status_code >= 500:
        return None
    if 200 <= resp.status_code < 500:
        try:
            data = resp.json()
            if isinstance(data, dict):
                status = str(data.get("status") or data.get("state") or "").lower()
                if status in {"ok", "healthy", "ready"}:
                    return "ok", status or "ok"
                if status in {"loading", "starting"}:
                    return "loading", status
        except Exception:
            pass
        if resp.status_code == 200:
            return "ok", f"HTTP {resp.status_code}"
        return "unknown", f"HTTP {resp.status_code}"
    return None


def _parse_slots(data: Any) -> tuple[int, int, int] | None:
    if not isinstance(data, list):
        return None
    total = len(data)
    busy = 0
    for slot in data:
        if isinstance(slot, dict) and slot.get("is_processing"):
            busy += 1
    return total, busy, total - busy


def _headers_hint(resp: httpx.Response | None) -> str:
    if resp is None:
        return ""
    bits: list[str] = []
    for key, val in resp.headers.items():
        kl = key.lower()
        if kl in {
            "server",
            "x-powered-by",
            "x-localai-version",
            "x-request-id",
            "openai-processing-ms",
            "openai-version",
            "x-stainless-lang",
        } or any(s in kl for s in ("vllm", "localai", "lmstudio", "tei", "openai")):
            bits.append(f"{kl}:{val}")
    return " ".join(bits).lower()


def _models_hint(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    parts: list[str] = []
    for key in ("object", "owned_by", "id", "root", "model", "model_id"):
        if key in data and data[key] is not None:
            parts.append(f"{key}:{data[key]}")
    for row in data.get("data") or []:
        if not isinstance(row, dict):
            continue
        for key in ("id", "owned_by", "root", "object"):
            if key in row and row[key] is not None:
                parts.append(f"{key}:{row[key]}")
    return " ".join(str(p) for p in parts).lower()


def fingerprint_engine(
    *,
    kind: str,
    probes_ok: list[str],
    slots_total: int | None,
    header_hints: str = "",
    body_hints: str = "",
) -> str:
    """Resolve a short engine label for the Services UI.

    Order: strong path signals → header/body tokens → soft kind guesses.
    """
    hints = f"{header_hints} {body_hints}".lower()
    paths = set(probes_ok)

    if slots_total is not None or "/slots" in paths:
        return "llama.cpp"
    # llama.cpp router mode: GET /props → {"role":"router", ...}
    if "/props" in paths or '"role":"router"' in hints or "role:router" in hints:
        if "role:router" in hints or '"role":"router"' in hints:
            return "llama-router"
        return "llama.cpp"
    if "llama.cpp" in hints or "llamacpp" in hints or "llama-server" in hints:
        return "llama.cpp"
    if "/api/ps" in paths or "/api/tags" in paths:
        return "ollama"
    if "/info" in paths or "text-embeddings-inference" in hints or "tei-" in hints:
        return "tei"
    if "vllm" in hints:
        return "vllm"
    if "localai" in hints or "x-localai" in hints:
        return "localai"
    if "lmstudio" in hints or "lm.studio" in hints or "lm-studio" in hints:
        return "lmstudio"
    if kind == "stt":
        if "faster-whisper" in hints or "faster_whisper" in hints:
            return "faster-whisper"
        return "whisper.cpp?"
    if kind == "tts":
        return "piper" if "piper" in hints else "piper?"
    if "/v1/models" in paths or "/v1/embeddings" in paths:
        return "openai-api"
    return ""


def probe_source(src: BackendSource) -> ServiceStatus:
    service = src.name
    kind = src.kind
    backend = (src.address or "").strip()
    if not backend:
        return ServiceStatus(
            service=service, backend="", kind=kind, state="unset", detail="not configured"
        )

    key = f"{service}:{backend}"
    base = _base_url(backend)
    status = ServiceStatus(
        service=service, backend=backend, kind=kind, state="down", detail="unreachable"
    )
    header_hints = ""
    body_hints = ""

    try:
        with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
            health_paths = ["/health", "/v1/health"]
            if kind == "chat":
                health_paths = ["/health", "/v1/health", "/api/tags"]
            if kind == "extractor":
                # Same OpenAI chat surface as chat; no TEI /info fingerprint.
                health_paths = ["/health", "/v1/health", "/v1/models"]
            if kind == "embed":
                # TEI: probe /info first (strong fingerprint); then /health
                health_paths = ["/info", "/health", "/v1/health"]

            for path in health_paths:
                resp = _get(client, base + path)
                if resp is None:
                    continue
                header_hints += " " + _headers_hint(resp)
                # /info is TEI-only. Never treat a 404/HTML /info as generic health —
                # that used to mark state=unknown and fingerprint as tei.
                if path == "/info":
                    if resp.status_code == 200:
                        try:
                            info = resp.json()
                            if isinstance(info, dict) and (
                                info.get("model_id")
                                or info.get("model_sha")
                                or "model_id" in info
                            ):
                                body_hints += " " + _models_hint(info)
                                mid = info.get("model_id") or info.get("model_sha")
                                if mid and not status.models:
                                    status.models = [str(mid)]
                                status.probes_ok.append("/info")
                                status.state = "ok"
                                status.detail = "tei /info"
                                body_hints += " text-embeddings-inference"
                                break
                        except Exception:
                            pass
                    continue
                parsed = _health_state(resp)
                if parsed is None:
                    continue
                state, detail = parsed
                status.state = state
                status.detail = detail
                status.probes_ok.append(path)
                if state != "down":
                    break

            if status.state == "down":
                resp = _get(client, base + "/")
                if resp is not None:
                    header_hints += " " + _headers_hint(resp)
                    status.state = "unknown"
                    status.detail = f"reachable (HTTP {resp.status_code})"
                    status.probes_ok.append("/")

            if kind in ("chat", "embed", "extractor") and status.state != "down":
                resp = _get(client, base + "/slots")
                if resp is not None and resp.status_code == 200:
                    header_hints += " " + _headers_hint(resp)
                    try:
                        data = resp.json()
                    except Exception:
                        data = None
                    parsed_slots = _parse_slots(data)
                    if parsed_slots:
                        total, busy, idle = parsed_slots
                        status.slots_total = total
                        status.slots_busy = busy
                        status.slots_idle = idle
                        status.probes_ok.append("/slots")
                        if busy and status.state == "ok":
                            status.state = "busy"
                            from .source_capacity import format_slots_label

                            status.detail = format_slots_label(busy, total)
                        elif status.state == "ok":
                            status.detail = f"{total} slots, all idle"

                # llama.cpp server (incl. router mode) exposes /props
                props = _get(client, base + "/props")
                if props is not None and props.status_code == 200:
                    header_hints += " " + _headers_hint(props)
                    try:
                        pdata = props.json()
                        if isinstance(pdata, dict):
                            body_hints += " " + _models_hint(pdata)
                            role = str(pdata.get("role") or "").lower()
                            if role:
                                body_hints += f" role:{role}"
                            if pdata.get("build_info") or pdata.get("model_alias") or role:
                                status.probes_ok.append("/props")
                                body_hints += " llama.cpp"
                                if role == "router" and status.state == "ok" and "/slots" not in status.probes_ok:
                                    status.detail = "llama-router /props"
                    except Exception:
                        pass

            if kind == "chat" and status.state != "down" and not status.models:
                resp = _get(client, base + "/api/ps")
                if resp is not None and resp.status_code == 200:
                    header_hints += " " + _headers_hint(resp)
                    try:
                        data = resp.json()
                        models = []
                        for m in data.get("models") or []:
                            name = m.get("name") or m.get("model")
                            if name:
                                models.append(str(name))
                        status.models = models
                        status.probes_ok.append("/api/ps")
                        if models:
                            status.detail = f"{len(models)} model(s) loaded"
                    except Exception:
                        pass
                if not status.models:
                    tags = _get(client, base + "/api/tags")
                    if tags is not None and tags.status_code == 200:
                        header_hints += " " + _headers_hint(tags)
                        try:
                            data = tags.json()
                            status.models = [
                                str(m.get("name"))
                                for m in (data.get("models") or [])
                                if m.get("name")
                            ][:12]
                            status.probes_ok.append("/api/tags")
                        except Exception:
                            pass

            # OpenAI /v1/models — also used for vLLM / local inference stacks / LM Studio hints
            if kind in MODEL_CHECK_KINDS and status.state not in (
                "down",
                "unset",
            ):
                if "/v1/models" not in status.probes_ok:
                    resp = _get(client, base + "/v1/models")
                    if resp is not None and resp.status_code == 200:
                        header_hints += " " + _headers_hint(resp)
                        try:
                            data = resp.json()
                            body_hints += " " + _models_hint(data)
                            ids = [
                                str(m.get("id"))
                                for m in (data.get("data") or [])
                                if isinstance(m, dict) and m.get("id")
                            ]
                            if ids and not status.models:
                                status.models = ids[:12]
                            status.probes_ok.append("/v1/models")
                            # Reachable OpenAI catalog beats a prior non-200 /health
                            if status.state == "unknown":
                                status.state = "ok"
                                status.detail = "ok"
                        except Exception:
                            pass

            # vLLM often exposes /version
            if kind in ("chat", "embed", "extractor") and status.state not in ("down", "unset"):
                ver = _get(client, base + "/version")
                if ver is not None and ver.status_code == 200:
                    header_hints += " " + _headers_hint(ver)
                    try:
                        text = ver.text.lower()
                        body_hints += " " + text[:200]
                        if "vllm" in text:
                            status.probes_ok.append("/version")
                    except Exception:
                        pass

            if status.state == "loading":
                _loading_since.setdefault(key, time.time())
            else:
                _loading_since.pop(key, None)

            status.engine = fingerprint_engine(
                kind=kind,
                probes_ok=status.probes_ok,
                slots_total=status.slots_total,
                header_hints=header_hints,
                body_hints=body_hints,
            )

    except Exception as exc:  # noqa: BLE001
        status.state = "down"
        status.detail = str(exc)[:120]

    from ..usage_pool import check_probe, probe_url_for_source

    probe = probe_url_for_source(src)
    status.gpu_power_url = probe
    st, watts, _ = check_probe(probe)
    status.gpu_power = st
    status.gpu_watts = watts

    return status


def probe_all(db: Session) -> list[ServiceStatus]:
    sources = list_sources(db)
    if not sources:
        return []
    results: dict[str, ServiceStatus] = {}
    with ThreadPoolExecutor(max_workers=max(4, len(sources))) as pool:
        futs = {pool.submit(probe_source, s): s for s in sources}
        for fut in as_completed(futs):
            src = futs[fut]
            st = fut.result()
            results[st.service] = st
            engine = (st.engine or "").strip()
            if engine and engine != (src.detected_engine or ""):
                src.detected_engine = engine
    db.commit()
    return [results[s.name] for s in sources if s.name in results]
