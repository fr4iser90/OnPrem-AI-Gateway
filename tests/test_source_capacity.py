"""Source capacity (slots) exposed on GET /v1/models."""

from __future__ import annotations

from pathlib import Path

from app.auth.source_admission import source_admission_gate
from app.data.backends import upsert_source
from app.data.catalog import openai_models_payload
from app.data.models import Base, CatalogModel, make_engine, make_session_factory
from app.data.source_capacity import (
    SourceCapacity,
    _merge_capacity,
    attach_capacity,
    capacity_fields,
    format_slots_label,
)
from app.data.source_load import SourceLoadSnapshot
from app.model_aliases import alias_list_entries, upsert_alias


def _session(tmp_path: Path):
    eng = make_engine(str(tmp_path / "cap.db"))
    Base.metadata.create_all(bind=eng)
    return make_session_factory(eng)()


def test_format_slots_label():
    assert format_slots_label(0, 4) == "0/4 in use"
    assert format_slots_label(2, 4) == "2/4 in use"
    assert format_slots_label(4, 4) == "all slots full (4/4)"
    assert format_slots_label(None, 4) == "slots 4"
    assert format_slots_label(1, None) == "—"


def test_capacity_fields_and_attach():
    cap = SourceCapacity(
        source_name="chat",
        slots_total=3,
        slots_idle=2,
        slots_busy=1,
        state="ok",
    )
    fields = capacity_fields(cap)
    assert fields == {
        "slots_total": 3,
        "slots_idle": 2,
        "slots_busy": 1,
        "load_state": "ok",
    }
    entry = {"id": "qwen3.6"}
    attach_capacity(entry, cap)
    assert entry["slots_total"] == 3
    assert entry["slots_idle"] == 2


def test_merge_prefers_probe_idle_over_gateway_inflight(tmp_path: Path):
    db = _session(tmp_path)
    src = upsert_source(
        db,
        name="chat",
        kind="chat",
        address="127.0.0.1:11535",
    )
    src.max_concurrency = 3
    db.commit()
    snap = SourceLoadSnapshot(
        state="ok",
        slots_total=8,  # engine reports more; admin cap wins for total
        slots_idle=1,  # upstream: 2 of 3 busy under admin cap
        probed_at=0,
        engine="llamacpp",
    )
    # Gateway only sees 1 stream; upstream probe is authoritative for UI.
    assert source_admission_gate.acquire(
        "127.0.0.1:11535", limit=3, key_id=1, priority=0, timeout=0.1
    )
    try:
        cap = _merge_capacity(src, snap)
        assert cap.slots_total == 3
        assert cap.slots_busy == 2
        assert cap.slots_idle == 1
    finally:
        source_admission_gate.release("127.0.0.1:11535")


def test_probe_model_for_source_prefers_loaded(tmp_path: Path):
    from app.data.source_capacity import probe_model_for_source

    db = _session(tmp_path)
    upsert_source(db, name="coder", kind="chat", address="127.0.0.1:11538")
    db.add(
        CatalogModel(
            source_name="coder",
            kind="chat",
            model_id="other",
            enabled=True,
            upstream_status="unloaded",
            n_parallel=4,
        )
    )
    db.add(
        CatalogModel(
            source_name="coder",
            kind="chat",
            model_id="Qwen3-Coder",
            enabled=True,
            upstream_status="loaded",
            n_parallel=2,
        )
    )
    db.commit()
    assert probe_model_for_source(db, "coder") == "Qwen3-Coder"


def test_openai_payload_includes_slots(tmp_path: Path):
    db = _session(tmp_path)
    row = CatalogModel(
        source_name="chat",
        kind="chat",
        model_id="alpha",
        enabled=True,
        ctx_size=65536,
    )
    db.add(row)
    db.commit()
    capacity = {
        "chat": SourceCapacity(
            source_name="chat",
            slots_total=3,
            slots_idle=3,
            slots_busy=0,
            state="ok",
        )
    }
    item = openai_models_payload([row], capacity_by_source=capacity)["data"][0]
    assert item["id"] == "alpha"
    assert item["slots_total"] == 3
    assert item["slots_idle"] == 3
    assert item["context_length"] == 65536


def test_alias_list_entries_include_slots_from_preferred_source(tmp_path: Path):
    db = _session(tmp_path)
    upsert_source(db, name="chat", kind="chat", address="127.0.0.1:11535")
    db.add(
        CatalogModel(
            source_name="chat",
            kind="chat",
            model_id="Qwen-VL",
            enabled=True,
            ctx_size=131072,
            modalities_in="text,image",
            modalities_out="text",
        )
    )
    upsert_alias(
        db,
        alias_id="qwen3.6",
        target_model_id="Qwen-VL",
        preferred_source="chat",
        show_backend=True,
    )
    db.commit()
    capacity = {
        "chat": SourceCapacity(
            source_name="chat",
            slots_total=3,
            slots_idle=2,
            slots_busy=1,
            state="ok",
        )
    }
    entries = alias_list_entries(db, capacity_by_source=capacity)
    assert entries[0]["id"] == "qwen3.6"
    assert entries[0]["slots_total"] == 3
    assert entries[0]["slots_idle"] == 2
    assert entries[0]["slots_busy"] == 1
    assert entries[0]["context_length"] == 131072


def test_merge_probe_idle_without_gateway(tmp_path: Path):
    db = _session(tmp_path)
    src = upsert_source(
        db, name="chat2", kind="chat", address="127.0.0.1:11537"
    )
    db.commit()
    snap = SourceLoadSnapshot(
        state="busy",
        slots_total=1,
        slots_idle=0,
        probed_at=0,
        engine="llamacpp",
    )
    # Ensure no leftover gate state for this address.
    source_admission_gate.release("127.0.0.1:11537")
    cap = _merge_capacity(src, snap)
    assert cap.slots_total == 1
    assert cap.slots_idle == 0
    assert cap.slots_busy == 1
    assert cap.state == "busy"
