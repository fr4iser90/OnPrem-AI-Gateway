"""Client-facing API base URL for key UI / docs snippets.

Covers app.config.public_api_base and app.web.shared.api_base_for_ui.
No prior dedicated tests. User: public domain should show when key is created.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.config import is_loopback_api_base, public_api_base


def test_public_api_base_uses_public_host(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PUBLIC_HOST", "ai.example.com")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        assert public_api_base() == "https://ai.example.com/v1"
    finally:
        get_settings.cache_clear()


def test_public_api_base_fallback_origin(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PUBLIC_HOST", raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        assert is_loopback_api_base(public_api_base())
        assert (
            public_api_base(fallback_origin="https://gw.home.arpa")
            == "https://gw.home.arpa/v1"
        )
        assert (
            public_api_base(fallback_origin="https://gw.home.arpa/v1")
            == "https://gw.home.arpa/v1"
        )
    finally:
        get_settings.cache_clear()


def test_api_base_for_ui_uses_browser_host(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PUBLIC_HOST", raising=False)
    from app.config import get_settings
    from app.web.shared import api_base_for_ui

    get_settings.cache_clear()
    try:
        req = MagicMock()
        req.headers.get.side_effect = lambda k, d=None: {
            "x-forwarded-host": "ai.example.com",
            "x-forwarded-proto": "https",
            "host": "onprem-auth:8080",
        }.get(k, d)
        req.url.scheme = "http"
        assert api_base_for_ui(req, db=None) == "https://ai.example.com/v1"
    finally:
        get_settings.cache_clear()
