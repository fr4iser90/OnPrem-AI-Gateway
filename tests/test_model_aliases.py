"""Public model aliases (slots + candidate pools)."""

from __future__ import annotations

import json
from pathlib import Path

from app.data.backends import resolve_source_for_kind, upsert_source
from app.data.models import Base, CatalogModel, make_engine, make_session_factory
from app.model_aliases import (
    alias_list_entries,
    apply_client_model_rewrites,
    candidate_model_ids,
    detect_similar_models,
    family_matches,
    hidden_catalog_ids,
    infer_family_prefix,
    model_allowed_via_alias,
    resolve_alias,
    upsert_alias,
    validate_alias_id,
)


def _session(tmp_path: Path):
    eng = make_engine(str(tmp_path / "alias.db"))
    Base.metadata.create_all(bind=eng)
    return make_session_factory(eng)()


def test_validate_alias_id():
    assert validate_alias_id("qwen3.6") == "qwen3.6"
    assert validate_alias_id("AUTO") is None
    assert validate_alias_id("Bad Name") is None


def test_infer_family_prefix():
    assert infer_family_prefix("Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL-VL") == "Qwen3.6"
    assert infer_family_prefix("Qwen3.8-27B-UD-Q4_K_XL-MTP") == "Qwen3.8"
    assert infer_family_prefix("Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL") == "Qwen3-Coder"


def test_detect_similar_models_includes_vl(tmp_path: Path):
    db = _session(tmp_path)
    for mid in (
        "Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL-VL",
        "Qwen3.6-35B-A3B-MTP-UD-Q5_K_XL-VL",
        "Qwen3.6-35B-A3B-MTP-UD-Q4_K_M",
        "Qwen3.8-27B-UD-Q4_K_XL-MTP",
        "Other-Model-7B",
    ):
        db.add(
            CatalogModel(
                source_name="chat",
                kind="chat",
                model_id=mid,
                enabled=True,
            )
        )
    db.commit()
    prefix, models = detect_similar_models(
        db,
        model_id="Qwen3.6-35B-A3B-MTP-UD-Q5_K_XL-VL",
        alias_id="qwen3.6",
    )
    assert prefix == "Qwen3.6"
    assert "Qwen3.6-35B-A3B-MTP-UD-Q5_K_XL-VL" in models
    assert "Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL-VL" in models
    assert "Qwen3.6-35B-A3B-MTP-UD-Q4_K_M" in models
    assert "Qwen3.8-27B-UD-Q4_K_XL-MTP" not in models
    assert "Other-Model-7B" not in models


def test_family_matches_slot_tag(tmp_path: Path):
    db = _session(tmp_path)
    db.add(
        CatalogModel(
            source_name="chat",
            kind="chat",
            model_id="WeirdName-Q4",
            enabled=True,
            tags="slot:smart, tools",
        )
    )
    db.add(
        CatalogModel(
            source_name="chat",
            kind="chat",
            model_id="Unrelated-7B",
            enabled=True,
            tags="tools",
        )
    )
    db.commit()
    assert family_matches(db, alias_id="smart") == ["WeirdName-Q4"]


def test_resolve_and_rewrite(tmp_path: Path):
    db = _session(tmp_path)
    upsert_alias(
        db,
        alias_id="qwen3.6",
        target_model_id="Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL-VL",
        preferred_source="chat",
        description="Daily VL",
    )
    db.commit()
    got = resolve_alias(db, "qwen3.6")
    assert got is not None
    assert got.target_model_id.endswith("-VL")
    assert got.preferred_source == "chat"

    body = json.dumps({"model": "qwen3.6", "messages": []}).encode()
    new_body, rewritten, pref = apply_client_model_rewrites(
        db, body, asked="qwen3.6", auth=None
    )
    assert rewritten.endswith("-VL")
    assert pref == "chat"
    assert json.loads(new_body)["model"].endswith("-VL")


def test_alias_list_entries_include_architecture(tmp_path: Path):
    db = _session(tmp_path)
    db.add(
        CatalogModel(
            source_name="chat",
            kind="chat",
            model_id="Qwen-VL",
            enabled=True,
            modalities_in="text,image",
            modalities_out="text",
            tags="vision",
            ctx_size=131072,
        )
    )
    upsert_alias(db, alias_id="qwen3.6", target_model_id="Qwen-VL", show_backend=True)
    db.commit()
    entries = alias_list_entries(db)
    assert entries[0]["id"] == "qwen3.6"
    assert "Qwen-VL" in entries[0]["description"]
    assert entries[0]["architecture"]["input_modalities"] == ["text", "image"]
    assert "vision" in entries[0]["tags"]


def test_alias_preferred_source_routes_to_named_upstream(tmp_path: Path):
    db = _session(tmp_path)
    upsert_source(db, name="chat", kind="chat", address="127.0.0.1:11535")
    upsert_source(db, name="chat2", kind="chat", address="127.0.0.1:11537")
    mid = "shared-model"
    db.add(CatalogModel(source_name="chat", kind="chat", model_id=mid, enabled=True))
    db.add(CatalogModel(source_name="chat2", kind="chat", model_id=mid, enabled=True))
    upsert_alias(
        db,
        alias_id="qwen3.6",
        target_model_id=mid,
        preferred_source="chat2",
    )
    db.commit()
    resolved = resolve_alias(db, "qwen3.6")
    assert resolved is not None
    assert resolved.preferred_source == "chat2"
    pick = resolve_source_for_kind(
        db,
        "chat",
        model=mid,
        routing_strategy="name",
        preferred_source=resolved.preferred_source,
    )
    assert pick is not None
    assert pick.name == "chat2"


def test_preferred_sources_only_hosts_of_active_model(tmp_path: Path):
    from app.model_aliases import (
        normalize_preferred_source,
        preferred_sources_for_model,
    )

    db = _session(tmp_path)
    db.add(CatalogModel(source_name="chat", kind="chat", model_id="M", enabled=True))
    db.add(CatalogModel(source_name="chat2", kind="chat", model_id="M", enabled=True))
    db.add(CatalogModel(source_name="stt", kind="stt", model_id="whisper", enabled=True))
    db.commit()
    assert preferred_sources_for_model(db, "M") == ["chat", "chat2"]
    assert preferred_sources_for_model(db, "whisper") == ["stt"]
    assert normalize_preferred_source(db, "M", "stt") == ""
    assert normalize_preferred_source(db, "M", "chat2") == "chat2"
    row = upsert_alias(
        db, alias_id="w", target_model_id="whisper", preferred_source="chat"
    )
    assert row is not None
    assert row.preferred_source == ""
    assert row.kind == "stt"


def test_manual_candidates_and_hide(tmp_path: Path):
    db = _session(tmp_path)
    for mid in ("A-Q4", "A-Q5", "Other-7B"):
        db.add(CatalogModel(source_name="chat", kind="chat", model_id=mid, enabled=True))
    row = upsert_alias(
        db,
        alias_id="humbuck",
        target_model_id="A-Q4",
        candidates=["A-Q4", "A-Q5"],
        hide_candidates=True,
    )
    db.commit()
    assert candidate_model_ids(row) == ["A-Q4", "A-Q5"]
    hidden = hidden_catalog_ids(db)
    assert "A-Q4" in hidden and "A-Q5" in hidden
    assert "Other-7B" not in hidden


def test_switch_active_candidate(tmp_path: Path):
    db = _session(tmp_path)
    for mid in ("A-Q4", "A-Q5", "A-Q8"):
        db.add(CatalogModel(source_name="chat", kind="chat", model_id=mid, enabled=True))
    row = upsert_alias(
        db,
        alias_id="smart",
        target_model_id="A-Q4",
        candidates=["A-Q4", "A-Q5", "A-Q8"],
        hide_candidates=True,
    )
    db.commit()
    from app.model_aliases import set_candidates

    row.target_model_id = "A-Q8"
    set_candidates(db, row, ["A-Q4", "A-Q5", "A-Q8"], ensure_target=True)
    db.commit()
    got = resolve_alias(db, "smart")
    assert got is not None
    assert got.target_model_id == "A-Q8"


def test_model_allowed_via_alias(tmp_path: Path):
    db = _session(tmp_path)
    upsert_alias(
        db,
        alias_id="qwen3.6",
        target_model_id="Qwen-Real",
        preferred_source="chat",
    )
    db.commit()
    assert model_allowed_via_alias(db, {"qwen3.6"}, "Qwen-Real")
    assert not model_allowed_via_alias(db, {"qwen3.6"}, "Other")
    assert not model_allowed_via_alias(db, {"other"}, "Qwen-Real")


def test_alias_list_respects_allow(tmp_path: Path):
    db = _session(tmp_path)
    upsert_alias(db, alias_id="qwen3.6", target_model_id="T1", preferred_source="chat")
    upsert_alias(db, alias_id="smart", target_model_id="T2", preferred_source="chat2")
    db.commit()
    ids = [e["id"] for e in alias_list_entries(db, allow={"qwen3.6"})]
    assert ids == ["qwen3.6"]
    ids_all = [e["id"] for e in alias_list_entries(db, allow=None)]
    assert set(ids_all) == {"qwen3.6", "smart"}
