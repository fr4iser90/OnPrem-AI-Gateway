"""Catalog metadata: tags, notes, docs links."""

from __future__ import annotations

from pathlib import Path

from app.data.catalog import (
    format_tags,
    openai_models_payload,
    parse_tags,
    suggest_docs_url,
    update_catalog_meta,
)
from app.data.models import Base, CatalogModel, make_engine, make_session_factory


def _session(tmp_path: Path):
    eng = make_engine(str(tmp_path / "meta.db"))
    Base.metadata.create_all(bind=eng)
    return make_session_factory(eng)()


def test_parse_and_format_tags():
    assert parse_tags("Tools, vision; CODE  ") == ["tools", "vision", "code"]
    assert format_tags("Fast, tools, tools") == "fast,tools"


def test_suggest_docs_url():
    assert suggest_docs_url("meta-llama/Llama-3.2-3B") == (
        "https://huggingface.co/meta-llama/Llama-3.2-3B"
    )
    assert suggest_docs_url("meta-llama/Llama-3.2-3B:Q4") == (
        "https://huggingface.co/meta-llama/Llama-3.2-3B"
    )
    assert suggest_docs_url("llama3.2:latest") == ""
    assert suggest_docs_url("https://example.com/x") == ""


def test_infer_tags():
    from app.data.catalog import infer_tags

    assert "vision" in infer_tags("Qwen3.6-35B-A3B-UD-Q4_K_M-VL", "chat")
    assert "medium" in infer_tags("Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL", "chat")
    assert "slow" in infer_tags("Qwen3.8-27B-Q8_0-MTP", "chat")
    assert "fast" in infer_tags("Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL", "chat")
    assert "embed" in infer_tags("bge-m3-Q4_K_M", "embed")
    assert "extractor" in infer_tags("agents-k1", "extractor")
    assert "code" in infer_tags("GemCod-R-Sapphire-270M", "chat")
    assert "stt" in infer_tags("stt", "stt")


def test_update_catalog_meta_and_payload_description(tmp_path: Path):
    db = _session(tmp_path)
    row = CatalogModel(
        source_name="chat",
        kind="chat",
        model_id="alpha",
        enabled=True,
    )
    db.add(row)
    db.commit()

    update_catalog_meta(
        db,
        row.id,
        tags="tools, code",
        short_note="Solid default",
        docs_url="https://huggingface.co/org/alpha",
        recommended_sampling='{"instruct":{"temperature":0.7,"top_p":0.8}}',
    )
    db.commit()
    db.refresh(row)
    assert row.tags == "tools,code"
    assert row.short_note == "Solid default"
    assert row.docs_url.startswith("https://")
    assert '"temperature":0.7' in row.recommended_sampling

    update_catalog_meta(db, row.id, docs_url="not-a-url")
    db.commit()
    db.refresh(row)
    assert row.docs_url == ""

    payload = openai_models_payload([row])
    assert payload["data"][0]["description"] == "Solid default"
    assert payload["data"][0]["recommended_sampling"]["instruct"]["temperature"] == 0.7


def test_recommended_sampling_rejects_non_object(tmp_path: Path):
    db = _session(tmp_path)
    row = CatalogModel(source_name="chat", kind="chat", model_id="x", enabled=True)
    db.add(row)
    db.commit()
    try:
        update_catalog_meta(db, row.id, recommended_sampling="[1,2]")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_import_recommended_sampling_longest_key(tmp_path: Path):
    from app.data.catalog import import_recommended_sampling

    db = _session(tmp_path)
    rows = [
        CatalogModel(source_name="chat", kind="chat", model_id="Qwen3.8-Flash-Next-UD-Q4_K_XL", enabled=True),
        CatalogModel(source_name="chat", kind="chat", model_id="Qwen3.8-27B-UD-Q4_K_XL", enabled=True),
        CatalogModel(source_name="chat", kind="chat", model_id="Qwen3-8B-UD-Q4_K_XL", enabled=True),
        CatalogModel(source_name="chat", kind="chat", model_id="Qwen3.8-mmproj-F16", enabled=True),
    ]
    db.add_all(rows)
    db.commit()

    stats = import_recommended_sampling(
        db,
        {
            "Qwen3.8-Flash-Next": {"thinking": {"temperature": 1.0}},
            "Qwen3.8-27B": {"instruct": {"temperature": 0.7}},
            "Qwen3-8B": {"thinking": {"temperature": 0.6}},
        },
    )
    db.commit()
    assert stats["updated"] == 3

    by_id = {r.model_id: r for r in db.query(CatalogModel).all()}
    assert '"temperature":1.0' in by_id["Qwen3.8-Flash-Next-UD-Q4_K_XL"].recommended_sampling
    assert '"temperature":0.7' in by_id["Qwen3.8-27B-UD-Q4_K_XL"].recommended_sampling
    assert '"temperature":0.6' in by_id["Qwen3-8B-UD-Q4_K_XL"].recommended_sampling
    assert (by_id["Qwen3.8-mmproj-F16"].recommended_sampling or "") == ""

    # Second import without overwrite skips filled rows
    stats2 = import_recommended_sampling(
        db,
        {"Qwen3.8-27B": {"instruct": {"temperature": 0.9}}},
        overwrite=False,
    )
    assert stats2["updated"] == 0
    assert stats2["skipped"] == 1



def test_parse_openai_model_item_loaded_and_unloaded():
    from app.data.catalog import parse_openai_model_item

    loaded = parse_openai_model_item(
        {
            "id": "GemCod",
            "status": {
                "value": "loaded",
                "args": ["--ctx-size", "8192", "--parallel", "4", "--model", "/x.gguf"],
            },
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            },
            "meta": {
                "n_ctx": 4096,
                "n_ctx_train": 32768,
                "n_embd": 640,
                "n_params": 268098816,
                "size": 536309504,
            },
        }
    )
    assert loaded is not None
    assert loaded.model_id == "GemCod"
    assert loaded.upstream_status == "loaded"
    assert loaded.ctx_size == 8192
    assert loaded.n_parallel == 4
    assert loaded.has_meta is True
    assert loaded.n_ctx == 4096
    assert loaded.n_ctx_train == 32768
    assert loaded.n_embd == 640
    assert loaded.n_params == 268098816
    assert loaded.model_size == 536309504
    assert loaded.modalities_in == "text"

    unloaded = parse_openai_model_item(
        {
            "id": "big",
            "status": {
                "value": "unloaded",
                "args": ["--ctx-size", "131072", "--parallel", "2"],
            },
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            },
        }
    )
    assert unloaded is not None
    assert unloaded.upstream_status == "unloaded"
    assert unloaded.ctx_size == 131072
    assert unloaded.n_parallel == 2
    assert unloaded.has_meta is False
    assert unloaded.n_embd is None


def test_parallel_from_args():
    from app.data.catalog import parallel_from_args

    assert parallel_from_args(["--parallel", "8"]) == 8
    assert parallel_from_args(["-np", "3"]) == 3
    assert parallel_from_args(["--parallel", "0"]) is None
    assert parallel_from_args(["--ctx-size", "8192"]) is None
    assert parallel_from_args(None) is None


def test_last_known_meta_retained_on_unload(tmp_path: Path):
    from app.data.catalog import apply_discovered_fields, parse_openai_model_item
    from app.data.models import utcnow

    db = _session(tmp_path)
    row = CatalogModel(source_name="chat", kind="chat", model_id="GemCod", enabled=True)
    db.add(row)
    db.commit()

    loaded = parse_openai_model_item(
        {
            "id": "GemCod",
            "status": {"value": "loaded", "args": ["--ctx-size", "8192", "--parallel", "4"]},
            "meta": {"n_embd": 640, "n_ctx": 4096, "n_params": 100},
        }
    )
    assert loaded is not None
    apply_discovered_fields(row, loaded, now=utcnow())
    db.commit()
    db.refresh(row)
    assert row.n_embd == 640
    assert row.n_ctx == 4096
    assert row.ctx_size == 8192
    assert row.n_parallel == 4
    assert row.upstream_status == "loaded"

    unloaded = parse_openai_model_item(
        {
            "id": "GemCod",
            "status": {"value": "unloaded", "args": ["--ctx-size", "16384", "--parallel", "2"]},
        }
    )
    assert unloaded is not None
    apply_discovered_fields(row, unloaded, now=utcnow())
    db.commit()
    db.refresh(row)
    assert row.upstream_status == "unloaded"
    assert row.ctx_size == 16384  # args always refresh
    assert row.n_parallel == 2
    assert row.n_embd == 640  # last known retained
    assert row.n_ctx == 4096

    payload = openai_models_payload([row])
    assert payload["data"][0]["n_embd"] == 640
    assert payload["data"][0]["ctx_size"] == 8192  # per-slot for clients
    assert payload["data"][0]["ctx_pool"] == 16384  # raw --ctx-size
    assert payload["data"][0]["n_parallel"] == 2
    assert payload["data"][0]["context_length"] == 8192  # per-slot: ctx_size // n_parallel
    assert payload["data"][0]["status"] == "unloaded"


def test_context_length_divides_by_parallel(tmp_path: Path):
    db = _session(tmp_path)
    row = CatalogModel(
        source_name="chat",
        kind="chat",
        model_id="vl-262k",
        enabled=True,
        ctx_size=262144,
        n_parallel=3,
        n_ctx_train=262144,
    )
    db.add(row)
    db.commit()
    item = openai_models_payload([row])["data"][0]
    assert item["ctx_pool"] == 262144
    assert item["ctx_size"] == 87381
    assert item["n_parallel"] == 3
    assert item["context_length"] == 87381  # 262144 // 3
    assert item["n_ctx_train"] == 87381  # capped so Hermes won't read 262k


def test_refresh_catalog_load_status_updates_badge_only(tmp_path: Path, monkeypatch):
    from app.data import catalog as catalog_mod
    from app.data.backends import upsert_source
    from app.data.catalog import (
        DiscoveredModel,
        SourceDiscovery,
        refresh_catalog_load_status,
    )

    db = _session(tmp_path)
    upsert_source(db, name="lab", kind="chat", address="127.0.0.1:11537")
    row = CatalogModel(
        source_name="lab",
        kind="chat",
        model_id="Nemotron-x",
        enabled=True,
        upstream_status="unloaded",
    )
    db.add(row)
    db.commit()

    def fake_discover(address: str, kind: str) -> SourceDiscovery:
        assert "11537" in address
        return SourceDiscovery(
            [
                DiscoveredModel(
                    model_id="Nemotron-x",
                    upstream_status="loaded",
                )
            ],
            ok=True,
        )

    monkeypatch.setattr(catalog_mod, "discover_models_for_source", fake_discover)
    n = refresh_catalog_load_status(db)
    assert n == 1
    db.refresh(row)
    assert row.upstream_status == "loaded"


def test_context_length_from_ctx_size_only(tmp_path: Path):
    db = _session(tmp_path)
    row = CatalogModel(
        source_name="chat",
        kind="chat",
        model_id="big-128k",
        enabled=True,
        ctx_size=131072,
        n_ctx_train=32768,
    )
    db.add(row)
    db.commit()
    payload = openai_models_payload([row])
    assert payload["data"][0]["context_length"] == 131072


def test_openai_payload_includes_tags_and_architecture(tmp_path: Path):
    db = _session(tmp_path)
    row = CatalogModel(
        source_name="chat",
        kind="chat",
        model_id="Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL-VL",
        enabled=True,
        tags="tools,medium",
        modalities_in="text,image",
        modalities_out="text",
    )
    db.add(row)
    db.commit()
    payload = openai_models_payload([row])
    item = payload["data"][0]
    assert item["tags"] == ["tools", "medium", "vision"]
    assert item["architecture"] == {
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
    }


def test_openai_payload_vision_tag_from_modalities_only(tmp_path: Path):
    db = _session(tmp_path)
    row = CatalogModel(
        source_name="chat",
        kind="chat",
        model_id="custom-vl",
        enabled=True,
        modalities_in="text,image",
        modalities_out="text",
    )
    db.add(row)
    db.commit()
    item = openai_models_payload([row])["data"][0]
    assert item["tags"] == ["vision"]
    assert item["architecture"]["input_modalities"] == ["text", "image"]


def test_context_length_falls_back_to_n_ctx_train(tmp_path: Path):
    db = _session(tmp_path)
    row = CatalogModel(
        source_name="chat",
        kind="chat",
        model_id="m",
        enabled=True,
        n_ctx_train=65536,
    )
    db.add(row)
    db.commit()
    payload = openai_models_payload([row])
    assert payload["data"][0]["context_length"] == 65536
    assert payload["data"][0]["ctx_size"] == 65536  # same budget for clients that read ctx_size
    assert "ctx_pool" not in payload["data"][0]


def test_fetch_piper_voices_returns_discovered():
    from app.data.catalog import _fetch_piper_voices
    from unittest.mock import MagicMock, patch

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "ok": True,
        "service": "piper-tts",
        "voices": ["de_DE-thorsten-high"],
    }
    with patch("app.data.catalog.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
        got, ok = _fetch_piper_voices("http://tts:9001")
        assert ok is True
        assert [d.model_id for d in got] == ["de_DE-thorsten-high"]
