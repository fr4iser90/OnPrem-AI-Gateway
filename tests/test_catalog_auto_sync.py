"""Catalog auto-sync helper tests."""

from __future__ import annotations

from app.catalog_auto_sync import clamp_auto_sync_minutes


def test_clamp_auto_sync_minutes():
    assert clamp_auto_sync_minutes(30) == 30
    assert clamp_auto_sync_minutes(1) == 5
    assert clamp_auto_sync_minutes(99999) == 1440
    assert clamp_auto_sync_minutes("nope") == 30
    assert clamp_auto_sync_minutes(None) == 30
