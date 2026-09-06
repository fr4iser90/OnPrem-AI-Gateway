"""Energy rollups must key by api_key_id, not shared labels like 'main'."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.data.models import ApiKey, Base, UsageEvent, WebUser, make_engine, make_session_factory, utcnow
from app.stats import energy_by_key, energy_by_owner, energy_wh_by_key_ids, enrich_energy_with_owners


def _session(tmp_path: Path):
    eng = make_engine(str(tmp_path / "energy.db"))
    Base.metadata.create_all(bind=eng)
    return make_session_factory(eng)()


def test_energy_by_key_separates_same_label_different_ids():
    events = [
        SimpleNamespace(
            result="ok",
            api_key_id=1,
            key_label="main",
            watt_hours=1.0,
            watts=100.0,
        ),
        SimpleNamespace(
            result="ok",
            api_key_id=2,
            key_label="main",
            watt_hours=2.0,
            watts=200.0,
        ),
        SimpleNamespace(
            result="ok",
            api_key_id=1,
            key_label="main",
            watt_hours=0.5,
            watts=50.0,
        ),
    ]
    rows = energy_by_key(events)
    assert len(rows) == 2
    by_id = {r["api_key_id"]: r for r in rows}
    assert by_id[1]["watt_hours"] == 1.5
    assert by_id[1]["ok_count"] == 2
    assert by_id[2]["watt_hours"] == 2.0
    assert by_id[2]["key_label"] == "main"
    assert rows[0]["api_key_id"] == 2  # sorted by Wh desc


def test_energy_by_owner_rolls_up_keys():
    key_rows = [
        {
            "api_key_id": 1,
            "key_label": "main",
            "owner_user_id": 10,
            "owner_name": "alice",
            "ok_count": 2,
            "watt_hours": 1.5,
            "watts_avg": 75.0,
        },
        {
            "api_key_id": 2,
            "key_label": "main",
            "owner_user_id": 20,
            "owner_name": "bob",
            "ok_count": 1,
            "watt_hours": 2.0,
            "watts_avg": 200.0,
        },
        {
            "api_key_id": 3,
            "key_label": "bot",
            "owner_user_id": 10,
            "owner_name": "alice",
            "ok_count": 3,
            "watt_hours": 0.5,
            "watts_avg": 40.0,
        },
    ]
    owners = energy_by_owner(key_rows)
    assert len(owners) == 2
    by_name = {r["owner_name"]: r for r in owners}
    assert by_name["alice"]["watt_hours"] == 2.0
    assert by_name["alice"]["keys"] == 2
    assert by_name["alice"]["ok_count"] == 5
    assert by_name["bob"]["keys"] == 1
    assert owners[0]["owner_name"] in ("alice", "bob")


def test_enrich_and_wh_by_key_ids(tmp_path: Path):
    db = _session(tmp_path)
    alice = WebUser(username="alice", password_hash="x", is_platform_admin=False)
    bob = WebUser(username="bob", password_hash="x", is_platform_admin=False)
    db.add_all([alice, bob])
    db.flush()
    k1 = ApiKey(
        label="main",
        key_prefix="gw_a",
        key_hash="ha",
        is_active=True,
        owner_user_id=alice.id,
    )
    k2 = ApiKey(
        label="main",
        key_prefix="gw_b",
        key_hash="hb",
        is_active=True,
        owner_user_id=bob.id,
    )
    db.add_all([k1, k2])
    db.flush()
    now = utcnow()
    db.add_all(
        [
            UsageEvent(
                created_at=now,
                api_key_id=k1.id,
                key_label="main",
                service="chat",
                method="POST",
                path="/v1/chat/completions",
                host="h",
                client_ip="",
                model="m",
                status=200,
                result="ok",
                watt_hours=1.25,
                watts=100.0,
            ),
            UsageEvent(
                created_at=now,
                api_key_id=k2.id,
                key_label="main",
                service="chat",
                method="POST",
                path="/v1/chat/completions",
                host="h",
                client_ip="",
                model="m",
                status=200,
                result="ok",
                watt_hours=3.0,
                watts=150.0,
            ),
        ]
    )
    db.commit()

    wh = energy_wh_by_key_ids(db, [k1.id, k2.id], lookback_days=7)
    assert wh[k1.id] == 1.25
    assert wh[k2.id] == 3.0

    rows = energy_by_key(
        [
            SimpleNamespace(
                result="ok",
                api_key_id=k1.id,
                key_label="main",
                watt_hours=1.25,
                watts=100.0,
            ),
            SimpleNamespace(
                result="ok",
                api_key_id=k2.id,
                key_label="main",
                watt_hours=3.0,
                watts=150.0,
            ),
        ]
    )
    enrich_energy_with_owners(db, rows)
    by_id = {r["api_key_id"]: r for r in rows}
    assert by_id[k1.id]["owner_name"]
    assert by_id[k2.id]["owner_name"]
    assert by_id[k1.id]["owner_user_id"] == alice.id
    assert by_id[k2.id]["owner_user_id"] == bob.id
    assert by_id[k1.id]["owner_name"] != by_id[k2.id]["owner_name"]
