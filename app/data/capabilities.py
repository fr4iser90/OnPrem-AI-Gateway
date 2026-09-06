"""Engine capability registry — strict admission checks per upstream type.

Each engine profile declares which HTTP APIs prove readiness. Preflight never
assumes a source is free when a required probe fails or is missing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from urllib.parse import quote

import httpx

from .probe import fingerprint_engine

_PROBE_TIMEOUT = 2.0


class Admission(str, Enum):
    OK = "ok"
    BUSY = "busy"
    LOADING = "loading"
    DOWN = "down"
    PROBE_FAILED = "probe_failed"
    CAPABILITY_MISSING = "capability_missing"
    MODEL_MISMATCH = "model_mismatch"


@dataclass(frozen=True)
class EngineState:
    engine: str
    admission: Admission
    detail: str = ""
    slots_total: int | None = None
    slots_idle: int | None = None
    model_loaded: bool | None = None
    probed_at: float = 0.0
    checks_run: tuple[str, ...] = ()


@dataclass(frozen=True)
class EngineProfile:
    id: str
    label: str
    kinds: frozenset[str]
    # True → must prove idle inference capacity before admit (slots/metrics).
    slot_admission: bool = False
    # True → chat requests must confirm model residency (Ollama / router).
    model_admission: bool = False


ENGINE_PROFILES: dict[str, EngineProfile] = {
    "llama.cpp": EngineProfile(
        id="llama.cpp",
        label="llama.cpp server",
        kinds=frozenset({"chat", "embed"}),
        slot_admission=True,
    ),
    "llama-router": EngineProfile(
        id="llama-router",
        label="llama.cpp router",
        kinds=frozenset({"chat", "embed"}),
        slot_admission=True,
        model_admission=True,
    ),
    "ollama": EngineProfile(
        id="ollama",
        label="Ollama",
        kinds=frozenset({"chat", "embed"}),
        model_admission=True,
    ),
    "vllm": EngineProfile(
        id="vllm",
        label="vLLM",
        kinds=frozenset({"chat", "embed"}),
        slot_admission=True,
    ),
    "tei": EngineProfile(
        id="tei",
        label="Text Embeddings Inference",
        kinds=frozenset({"embed"}),
    ),
    "localai": EngineProfile(
        id="localai",
        label="LocalAI",
        kinds=frozenset({"chat", "embed", "stt", "tts"}),
    ),
    "lmstudio": EngineProfile(
        id="lmstudio",
        label="LM Studio",
        kinds=frozenset({"chat", "embed"}),
    ),
    "openai-api": EngineProfile(
        id="openai-api",
        label="OpenAI-compatible (no slot API)",
        kinds=frozenset({"chat", "embed", "stt", "tts"}),
    ),
    "faster-whisper": EngineProfile(
        id="faster-whisper",
        label="faster-whisper",
        kinds=frozenset({"stt"}),
    ),
    "whisper.cpp?": EngineProfile(
        id="whisper.cpp?",
        label="whisper.cpp",
        kinds=frozenset({"stt"}),
    ),
    "piper": EngineProfile(
        id="piper",
        label="Piper TTS",
        kinds=frozenset({"tts"}),
    ),
    "piper?": EngineProfile(
        id="piper?",
        label="Piper TTS (unconfirmed)",
        kinds=frozenset({"tts"}),
    ),
}


def engine_profile(engine_id: str | None) -> EngineProfile | None:
    if not engine_id:
        return None
    return ENGINE_PROFILES.get(engine_id.strip())


def engine_choices() -> list[dict[str, str]]:
    """Admin dropdown: auto + known engines."""
    rows = [{"id": "", "label": "Auto (probe)", "summary": "Detect from live probe"}]
    for p in ENGINE_PROFILES.values():
        bits = []
        if p.slot_admission:
            bits.append("slots")
        if p.model_admission:
            bits.append("model")
        cap = ", ".join(bits) if bits else "reachability"
        rows.append({"id": p.id, "label": p.label, "summary": f"Preflight: {cap}"})
    return rows


def _base_url(backend: str) -> str:
    return f"http://{(backend or '').strip()}"


def _get(client: httpx.Client, url: str) -> httpx.Response | None:
    try:
        return client.get(url)
    except Exception:
        return None


def _health_loading(resp: httpx.Response) -> bool:
    if resp.status_code == 503:
        return True
    try:
        data = resp.json()
        if isinstance(data, dict):
            status = str(data.get("status") or data.get("state") or "").lower()
            if status in {"loading", "starting"}:
                return True
            err = data.get("error")
            if isinstance(err, dict):
                msg = str(err.get("message") or "").lower()
                if "loading" in msg:
                    return True
    except Exception:
        pass
    return False


def _slots_path(*, model: str | None, router: bool) -> str:
    if router and (model or "").strip():
        return f"/slots?model={quote(model.strip(), safe='')}"
    return "/slots"


def _ollama_loaded(model: str, loaded: list[str]) -> bool:
    if not loaded:
        return False
    want = model.strip()
    if not want:
        return True
    base = want.split(":", 1)[0]
    for name in loaded:
        n = name.strip()
        if n == want or n.startswith(want + ":") or n == base or n.startswith(base + ":"):
            return True
    return False


def _parse_slots_response(data: object) -> tuple[int, int, int] | None:
    if not isinstance(data, list):
        return None
    total = len(data)
    busy = sum(
        1 for slot in data if isinstance(slot, dict) and slot.get("is_processing")
    )
    return total, busy, total - busy


def _prometheus_gauge(text: str, name: str) -> float | None:
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(name + " ") or line.startswith(name + "{"):
            try:
                return float(line.split()[-1])
            except ValueError:
                continue
    return None


def discover_engine(
    *,
    backend: str,
    kind: str,
    timeout: float = _PROBE_TIMEOUT,
) -> str:
    """Probe upstream once and fingerprint the engine (no admission decision)."""
    addr = (backend or "").strip()
    if not addr:
        return ""
    base = _base_url(addr)
    probes_ok: list[str] = []
    slots_total: int | None = None
    header_hints = ""
    body_hints = ""

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            for path in ("/health", "/v1/health", "/api/tags", "/info"):
                resp = _get(client, base + path)
                if resp is None:
                    continue
                if resp.status_code == 200:
                    probes_ok.append(path)
                if path == "/info" and resp.status_code == 200:
                    try:
                        info = resp.json()
                        if isinstance(info, dict) and (
                            info.get("model_id") or info.get("model_sha")
                        ):
                            body_hints += " text-embeddings-inference"
                            break
                    except Exception:
                        pass

            slots_resp = _get(client, base + "/slots")
            if slots_resp is not None and slots_resp.status_code == 200:
                probes_ok.append("/slots")
                parsed = _parse_slots_response(slots_resp.json())
                if parsed:
                    slots_total = parsed[0]

            props = _get(client, base + "/props")
            if props is not None and props.status_code == 200:
                probes_ok.append("/props")
                try:
                    pdata = props.json()
                    if isinstance(pdata, dict):
                        role = str(pdata.get("role") or "").lower()
                        if role:
                            body_hints += f" role:{role}"
                        body_hints += " llama.cpp"
                except Exception:
                    pass

            ps = _get(client, base + "/api/ps")
            if ps is not None and ps.status_code == 200:
                probes_ok.append("/api/ps")

            tags = _get(client, base + "/api/tags")
            if tags is not None and tags.status_code == 200:
                probes_ok.append("/api/tags")

            models = _get(client, base + "/v1/models")
            if models is not None and models.status_code == 200:
                probes_ok.append("/v1/models")

            ver = _get(client, base + "/version")
            if ver is not None and ver.status_code == 200:
                probes_ok.append("/version")
                body_hints += " " + (ver.text or "")[:200].lower()

    except Exception:
        return ""

    return fingerprint_engine(
        kind=kind,
        probes_ok=probes_ok,
        slots_total=slots_total,
        header_hints=header_hints,
        body_hints=body_hints,
    )


def _check_llama(
    client: httpx.Client,
    base: str,
    *,
    kind: str,
    model: str | None,
    router: bool,
) -> EngineState:
    engine = "llama-router" if router else "llama.cpp"
    checks: list[str] = []

    health = _get(client, base + "/health")
    checks.append("health")
    if health is None:
        return EngineState(
            engine, Admission.PROBE_FAILED, "health_unreachable", checks_run=tuple(checks)
        )
    if _health_loading(health):
        return EngineState(
            engine, Admission.LOADING, "health_loading", checks_run=tuple(checks)
        )
    if health.status_code != 200:
        return EngineState(
            engine,
            Admission.PROBE_FAILED,
            f"health_http_{health.status_code}",
            checks_run=tuple(checks),
        )

    props = _get(client, base + "/props")
    checks.append("props")
    total_slots: int | None = None
    if props is not None and props.status_code == 200:
        try:
            pdata = props.json()
            if isinstance(pdata, dict):
                if pdata.get("is_sleeping"):
                    return EngineState(
                        engine,
                        Admission.LOADING,
                        "model_sleeping",
                        checks_run=tuple(checks),
                    )
                ts = pdata.get("total_slots")
                if isinstance(ts, int):
                    total_slots = ts
                dgs = pdata.get("default_generation_settings")
                if isinstance(dgs, dict) and dgs.get("is_processing"):
                    return EngineState(
                        engine,
                        Admission.BUSY,
                        "props_is_processing",
                        slots_total=total_slots,
                        slots_idle=0,
                        checks_run=tuple(checks),
                    )
        except Exception:
            return EngineState(
                engine,
                Admission.PROBE_FAILED,
                "props_parse_error",
                checks_run=tuple(checks),
            )

    if router and kind == "chat" and (model or "").strip():
        checks.append("models")
        models_url = f"{base}/models?model={quote(model.strip(), safe='')}"
        models_resp = _get(client, models_url)
        if models_resp is None:
            return EngineState(
                engine,
                Admission.PROBE_FAILED,
                "models_unreachable",
                checks_run=tuple(checks),
            )
        if models_resp.status_code != 200:
            return EngineState(
                engine,
                Admission.PROBE_FAILED,
                f"models_http_{models_resp.status_code}",
                checks_run=tuple(checks),
            )
        try:
            mdata = models_resp.json()
            rows = mdata.get("data") if isinstance(mdata, dict) else None
            if isinstance(rows, list) and rows:
                status = rows[0].get("status") if isinstance(rows[0], dict) else None
                if isinstance(status, dict):
                    value = str(status.get("value") or "").lower()
                    # Router keeps models unloaded until the first request; blocking
                    # here would prevent on-demand load from ever starting.
                    if value == "unloaded":
                        return EngineState(
                            engine,
                            Admission.OK,
                            "model_on_demand",
                            model_loaded=False,
                            checks_run=tuple(checks),
                        )
                    if value in {"loading", "sleeping"}:
                        return EngineState(
                            engine,
                            Admission.LOADING,
                            f"model_{value}",
                            checks_run=tuple(checks),
                        )
                    if value == "failed" or status.get("failed"):
                        return EngineState(
                            engine,
                            Admission.PROBE_FAILED,
                            "model_load_failed",
                            checks_run=tuple(checks),
                        )
        except Exception:
            return EngineState(
                engine,
                Admission.PROBE_FAILED,
                "models_parse_error",
                checks_run=tuple(checks),
            )

    slots_base = base + _slots_path(model=model, router=router)
    checks.append("slots")
    slots_fail = _get(client, f"{slots_base}&fail_on_no_slot=1" if "?" in slots_base else f"{slots_base}?fail_on_no_slot=1")
    if slots_fail is not None and slots_fail.status_code == 503:
        return EngineState(
            engine,
            Admission.BUSY,
            "no_idle_slots",
            slots_total=total_slots,
            slots_idle=0,
            checks_run=tuple(checks),
        )

    slots_resp = slots_fail if slots_fail is not None and slots_fail.status_code == 200 else _get(client, slots_base)
    if slots_resp is None:
        return EngineState(
            engine,
            Admission.PROBE_FAILED,
            "slots_unreachable",
            checks_run=tuple(checks),
        )
    if slots_resp.status_code == 501:
        return EngineState(
            engine,
            Admission.CAPABILITY_MISSING,
            "slots_endpoint_disabled",
            checks_run=tuple(checks),
        )
    if slots_resp.status_code != 200:
        return EngineState(
            engine,
            Admission.PROBE_FAILED,
            f"slots_http_{slots_resp.status_code}",
            checks_run=tuple(checks),
        )

    try:
        parsed = _parse_slots_response(slots_resp.json())
    except Exception:
        return EngineState(
            engine,
            Admission.PROBE_FAILED,
            "slots_parse_error",
            checks_run=tuple(checks),
        )
    if not parsed:
        return EngineState(
            engine,
            Admission.PROBE_FAILED,
            "slots_empty_response",
            checks_run=tuple(checks),
        )
    total, _busy, idle = parsed
    if total > 0 and idle <= 0:
        return EngineState(
            engine,
            Admission.BUSY,
            "all_slots_full",
            slots_total=total,
            slots_idle=0,
            checks_run=tuple(checks),
        )
    return EngineState(
        engine,
        Admission.OK,
        "ready",
        slots_total=total,
        slots_idle=idle,
        model_loaded=True if router else None,
        checks_run=tuple(checks),
    )


def _check_ollama(
    client: httpx.Client,
    base: str,
    *,
    model: str | None,
) -> EngineState:
    engine = "ollama"
    checks = ["api/ps"]
    ps = _get(client, base + "/api/ps")
    if ps is None:
        return EngineState(
            engine, Admission.PROBE_FAILED, "api_ps_unreachable", checks_run=tuple(checks)
        )
    if ps.status_code != 200:
        return EngineState(
            engine,
            Admission.PROBE_FAILED,
            f"api_ps_http_{ps.status_code}",
            checks_run=tuple(checks),
        )
    try:
        data = ps.json()
    except Exception:
        return EngineState(
            engine, Admission.PROBE_FAILED, "api_ps_parse_error", checks_run=tuple(checks)
        )
    names: list[str] = []
    if isinstance(data, dict):
        for row in data.get("models") or []:
            if isinstance(row, dict):
                n = row.get("name") or row.get("model")
                if n:
                    names.append(str(n))
    want = (model or "").strip()
    if not want:
        return EngineState(
            engine,
            Admission.OK,
            "ready",
            model_loaded=None,
            checks_run=tuple(checks),
        )
    if not names:
        return EngineState(
            engine,
            Admission.LOADING,
            "model_cold_start",
            model_loaded=False,
            checks_run=tuple(checks),
        )
    if _ollama_loaded(want, names):
        return EngineState(
            engine,
            Admission.OK,
            "ready",
            model_loaded=True,
            checks_run=tuple(checks),
        )
    return EngineState(
        engine,
        Admission.MODEL_MISMATCH,
        "other_model_loaded",
        model_loaded=False,
        checks_run=tuple(checks),
    )


def _check_vllm(client: httpx.Client, base: str) -> EngineState:
    engine = "vllm"
    checks = ["metrics"]
    metrics = _get(client, base + "/metrics")
    if metrics is None:
        return EngineState(
            engine,
            Admission.PROBE_FAILED,
            "metrics_unreachable",
            checks_run=tuple(checks),
        )
    if metrics.status_code in {404, 501}:
        return EngineState(
            engine,
            Admission.CAPABILITY_MISSING,
            "metrics_endpoint_disabled",
            checks_run=tuple(checks),
        )
    if metrics.status_code != 200:
        return EngineState(
            engine,
            Admission.PROBE_FAILED,
            f"metrics_http_{metrics.status_code}",
            checks_run=tuple(checks),
        )
    running = _prometheus_gauge(metrics.text, "vllm:num_requests_running")
    waiting = _prometheus_gauge(metrics.text, "vllm:num_requests_waiting")
    if running is None and waiting is None:
        return EngineState(
            engine,
            Admission.CAPABILITY_MISSING,
            "metrics_missing_gauges",
            checks_run=tuple(checks),
        )
    run_n = int(running or 0)
    wait_n = int(waiting or 0)
    if wait_n > 0:
        return EngineState(
            engine,
            Admission.BUSY,
            "requests_waiting",
            slots_idle=0,
            checks_run=tuple(checks),
        )
    return EngineState(
        engine,
        Admission.OK,
        "ready",
        slots_total=run_n if run_n > 0 else None,
        checks_run=tuple(checks),
    )


def _check_reachability(
    client: httpx.Client,
    base: str,
    *,
    kind: str,
    engine: str,
    paths: tuple[str, ...],
) -> EngineState:
    checks: list[str] = []
    for path in paths:
        checks.append(path.lstrip("/"))
        resp = _get(client, base + path)
        if resp is None:
            continue
        if _health_loading(resp):
            return EngineState(
                engine,
                Admission.LOADING,
                f"{path}_loading",
                checks_run=tuple(checks),
            )
        if resp.status_code == 200:
            return EngineState(
                engine, Admission.OK, "ready", checks_run=tuple(checks)
            )
    return EngineState(
        engine,
        Admission.PROBE_FAILED,
        "reachability_failed",
        checks_run=tuple(checks),
    )


# Fix _check_vllm - I used slots_busy which doesn't exist on EngineState. Use slots_total for running count or just omit.

def _run_profile_check(
    profile: EngineProfile,
    client: httpx.Client,
    base: str,
    *,
    kind: str,
    model: str | None,
) -> EngineState:
    if profile.id in {"llama.cpp", "llama-router"}:
        return _check_llama(
            client,
            base,
            kind=kind,
            model=model,
            router=profile.id == "llama-router",
        )
    if profile.id == "ollama":
        return _check_ollama(client, base, model=model)
    if profile.id == "vllm":
        return _check_vllm(client, base)
    if profile.id == "tei":
        return _check_reachability(
            client, base, kind=kind, engine=profile.id, paths=("/info", "/health")
        )
    if profile.id in {"localai", "lmstudio", "openai-api"}:
        paths = ("/health", "/v1/health", "/v1/models")
        if kind == "embed":
            paths = ("/health", "/v1/health", "/info", "/v1/models")
        return _check_reachability(
            client, base, kind=kind, engine=profile.id, paths=paths
        )
    if profile.id in {"faster-whisper", "whisper.cpp?"}:
        return _check_reachability(
            client, base, kind=kind, engine=profile.id, paths=("/health", "/")
        )
    if profile.id in {"piper", "piper?"}:
        return _check_reachability(
            client, base, kind=kind, engine=profile.id, paths=("/health", "/")
        )
    return EngineState(
        profile.id,
        Admission.CAPABILITY_MISSING,
        "unknown_engine_profile",
    )


def resolve_engine_for_source(
    *,
    backend: str,
    kind: str,
    engine_override: str | None = None,
    detected_engine: str | None = None,
    timeout: float = _PROBE_TIMEOUT,
) -> str:
    override = (engine_override or "").strip()
    if override and override in ENGINE_PROFILES:
        return override
    detected = (detected_engine or "").strip()
    if detected and detected in ENGINE_PROFILES:
        return detected
    return discover_engine(backend=backend, kind=kind, timeout=timeout)


def probe_engine_state(
    *,
    backend: str,
    kind: str,
    model: str | None = None,
    engine: str | None = None,
    timeout: float = _PROBE_TIMEOUT,
) -> EngineState:
    """Strict engine probe — never assumes OK on failure."""
    addr = (backend or "").strip()
    now = time.time()
    if not addr:
        return EngineState("", Admission.DOWN, "no_backend", probed_at=now)

    engine_id = (engine or "").strip()
    if not engine_id or engine_id not in ENGINE_PROFILES:
        engine_id = discover_engine(backend=addr, kind=kind, timeout=timeout)
    if not engine_id:
        return EngineState(
            "",
            Admission.PROBE_FAILED,
            "engine_undetected",
            probed_at=now,
            checks_run=("discover",),
        )

    profile = ENGINE_PROFILES.get(engine_id)
    if profile is None:
        return EngineState(
            engine_id,
            Admission.CAPABILITY_MISSING,
            "unknown_engine",
            probed_at=now,
        )
    if kind not in profile.kinds:
        return EngineState(
            engine_id,
            Admission.CAPABILITY_MISSING,
            f"engine_not_for_kind_{kind}",
            probed_at=now,
        )

    base = _base_url(addr)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            state = _run_profile_check(
                profile, client, base, kind=kind, model=model
            )
    except Exception as exc:
        return EngineState(
            engine_id,
            Admission.DOWN,
            str(exc)[:120],
            probed_at=now,
        )

    return EngineState(
        state.engine,
        state.admission,
        state.detail,
        slots_total=state.slots_total,
        slots_idle=state.slots_idle,
        model_loaded=state.model_loaded,
        probed_at=now,
        checks_run=state.checks_run,
    )


def engine_state_to_load_snapshot(state: EngineState):
    """Map strict engine state → load-routing snapshot."""
    from .source_load import SourceLoadSnapshot
    if state.admission == Admission.OK:
        load_state = "ok"
    elif state.admission == Admission.BUSY:
        load_state = "busy"
    elif state.admission in {
        Admission.LOADING,
        Admission.MODEL_MISMATCH,
    }:
        load_state = "loading"
    elif state.admission == Admission.DOWN:
        load_state = "down"
    else:
        load_state = "unknown"
    return SourceLoadSnapshot(
        state=load_state,
        slots_total=state.slots_total,
        slots_idle=state.slots_idle,
        model_loaded=state.model_loaded,
        probed_at=state.probed_at,
    )


def admission_reason(state: EngineState) -> tuple[str, int]:
    """Map admission → (error_code, retry_after_sec) for 503 responses."""
    mapping: dict[Admission, tuple[str, int]] = {
        Admission.BUSY: ("all_slots_full", 5),
        Admission.LOADING: ("model_initializing", 15),
        Admission.DOWN: ("backend_unreachable", 15),
        Admission.PROBE_FAILED: ("probe_failed", 10),
        Admission.CAPABILITY_MISSING: ("capability_missing", 30),
        Admission.MODEL_MISMATCH: ("model_mismatch", 15),
    }
    return mapping.get(state.admission, ("backend_unavailable", 15))
