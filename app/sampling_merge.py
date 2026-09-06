"""Soft-fill missing chat sampling fields from catalog recommended_sampling.

Admin opt-in only (AuthSettings.soft_sampling_defaults). Never overrides keys
the client already sent. Does not invent thinking vs instruct without a signal
or default_profile in the catalog JSON.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from .data.catalog import parse_recommended_sampling
from .data.models import CatalogModel

# Top-level OpenAI / llama.cpp sampling scalars we may fill when absent.
_SAMPLING_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "repeat_penalty",
    }
)

# Merged into chat_template_kwargs when missing there and at top level.
_TEMPLATE_KEYS = frozenset(
    {
        "enable_thinking",
        "preserve_thinking",
        "reasoning_effort",
    }
)

_META_KEYS = frozenset({"default_profile", "source", "notes", "note", "docs"})

_PROFILE_HEADER = "x-onprem-sampling-profile"

# Key / form values: empty = inherit catalog default_profile.
_KEY_PROFILE_ALIASES = {
    "": "",
    "inherit": "",
    "default": "",
    "catalog": "",
    "thinking": "thinking",
    "instruct": "instruct",
}


def normalize_key_sampling_profile(value: str | None) -> str:
    """Return '' (inherit) or a profile name like thinking|instruct."""
    raw = (value or "").strip().lower()
    if raw in _KEY_PROFILE_ALIASES:
        return _KEY_PROFILE_ALIASES[raw]
    # Allow custom profile names that exist in catalog JSON (e.g. coding).
    if raw and raw.replace("-", "").replace("_", "").isalnum():
        return raw[:32]
    return ""


def sampling_profile_choices() -> list[dict[str, str]]:
    return [
        {
            "id": "",
            "label": "Inherit (catalog default_profile)",
            "summary": "Use the model’s recommended_sampling default_profile",
        },
        {
            "id": "thinking",
            "label": "Thinking",
            "summary": "Soft-fill the thinking profile when the client omits sampling",
        },
        {
            "id": "instruct",
            "label": "Instruct",
            "summary": "Soft-fill the instruct profile when the client omits sampling",
        },
    ]


def recommended_sampling_for_model(db: Session, model_id: str | None) -> dict[str, Any] | None:
    mid = (model_id or "").strip()
    if not mid:
        return None
    row = (
        db.query(CatalogModel)
        .filter(CatalogModel.model_id == mid, CatalogModel.enabled.is_(True))
        .order_by(CatalogModel.id.asc())
        .first()
    )
    if row is None:
        row = (
            db.query(CatalogModel)
            .filter(CatalogModel.model_id == mid)
            .order_by(CatalogModel.id.asc())
            .first()
        )
    if row is None:
        return None
    return parse_recommended_sampling(row.recommended_sampling)


def profile_hint_from_headers(headers) -> str | None:
    """Optional client signal: X-OnPrem-Sampling-Profile: thinking|instruct|…"""
    if headers is None:
        return None
    raw = headers.get(_PROFILE_HEADER) or headers.get(_PROFILE_HEADER.title())
    if raw is None and hasattr(headers, "get"):
        # Starlette Headers is case-insensitive; plain dict may not be.
        for k, v in getattr(headers, "items", lambda: [])():
            if str(k).lower() == _PROFILE_HEADER:
                raw = v
                break
    name = str(raw or "").strip().lower()
    return name or None


def _profile_names(profiles: dict[str, Any]) -> list[str]:
    return [k for k, v in profiles.items() if k not in _META_KEYS and isinstance(v, dict)]


def resolve_sampling_profile(
    profiles: dict[str, Any],
    *,
    hint: str | None = None,
    enable_thinking: bool | None = None,
    key_profile: str | None = None,
) -> dict[str, Any] | None:
    """Pick one profile dict from recommended_sampling.

    Priority: client header hint → body enable_thinking → key profile →
    catalog default_profile → single/fallback profile.
    """
    names = _profile_names(profiles)
    if not names:
        return None

    if hint:
        h = hint.strip().lower()
        if h in profiles and isinstance(profiles[h], dict):
            return profiles[h]

    if enable_thinking is True and "thinking" in profiles and isinstance(profiles["thinking"], dict):
        return profiles["thinking"]
    if enable_thinking is False and "instruct" in profiles and isinstance(profiles["instruct"], dict):
        return profiles["instruct"]

    key = normalize_key_sampling_profile(key_profile)
    if key and key in profiles and isinstance(profiles[key], dict):
        return profiles[key]

    default = profiles.get("default_profile")
    if isinstance(default, str):
        d = default.strip().lower()
        if d in profiles and isinstance(profiles[d], dict):
            return profiles[d]

    if len(names) == 1:
        return profiles[names[0]]

    for preferred in ("instruct", "thinking"):
        if preferred in profiles and isinstance(profiles[preferred], dict):
            return profiles[preferred]
    return profiles[names[0]]


def _enable_thinking_from_body(payload: dict) -> bool | None:
    kwargs = payload.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and "enable_thinking" in kwargs:
        return bool(kwargs.get("enable_thinking"))
    if "enable_thinking" in payload:
        return bool(payload.get("enable_thinking"))
    return None


def merge_missing_sampling(
    payload: dict[str, Any],
    profile: dict[str, Any],
) -> bool:
    """Fill absent sampling / template keys from profile. Returns True if changed."""
    changed = False
    for key, value in profile.items():
        if key in _TEMPLATE_KEYS:
            continue
        if key not in _SAMPLING_KEYS:
            continue
        if key in payload:
            continue
        payload[key] = value
        changed = True

    template_updates = {
        key: profile[key] for key in _TEMPLATE_KEYS if key in profile
    }
    if not template_updates:
        return changed

    kwargs = payload.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
    else:
        kwargs = dict(kwargs)

    for key, value in template_updates.items():
        if key in kwargs or key in payload:
            continue
        kwargs[key] = value
        changed = True

    if kwargs != (payload.get("chat_template_kwargs") or {}):
        payload["chat_template_kwargs"] = kwargs
    return changed


def apply_soft_sampling(
    body: bytes | None,
    content_type: str | None,
    *,
    profiles: dict[str, Any] | None,
    profile_hint: str | None = None,
    key_profile: str | None = None,
) -> bytes | None:
    """Return body with missing sampling filled, or original body if nothing to do."""
    if not body or not profiles:
        return body
    ct = (content_type or "").lower()
    if "multipart/form-data" in ct:
        return body
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return body
    if not isinstance(payload, dict):
        return body

    profile = resolve_sampling_profile(
        profiles,
        hint=profile_hint,
        enable_thinking=_enable_thinking_from_body(payload),
        key_profile=key_profile,
    )
    if not profile:
        return body
    if not merge_missing_sampling(payload, profile):
        return body
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def soft_sampling_body(
    db: Session,
    body: bytes | None,
    content_type: str | None,
    *,
    model_id: str | None,
    headers=None,
    enabled: bool = False,
    key_profile: str | None = None,
) -> bytes | None:
    """Catalog lookup + soft merge when admin flag is on."""
    if not enabled or not body:
        return body
    profiles = recommended_sampling_for_model(db, model_id)
    if not profiles:
        return body
    return apply_soft_sampling(
        body,
        content_type,
        profiles=profiles,
        profile_hint=profile_hint_from_headers(headers),
        key_profile=key_profile,
    )
