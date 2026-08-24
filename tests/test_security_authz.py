"""HTTP authz matrix — roles, IDOR, API grants (TestClient, no browser/LAN).

Not full browser E2E; exercises real routes, sessions, and authorize() paths in CI.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.data.backends import upsert_source
from app.data.db import hash_api_key, hash_password
from app.data.grants import sync_user_grants
from app.data.models import WebUser, ApiKey, CatalogModel, ModelAllowlist, ServiceGrant
from tests.constants import (
    ALICE_API_KEY,
    ALICE_NAME,
    ALICE_PASSWORD,
    BOB_API_KEY,
    BOB_NAME,
    BOB_PASSWORD,
    BOOTSTRAP_PASSWORD,
    BOOTSTRAP_USER,
)

pytestmark = pytest.mark.security


@dataclass
class SecurityWorld:
    admin_id: int
    alice_id: int
    bob_id: int
    alice_key_id: int
    bob_key_id: int


def _login(client, username: str, password: str) -> None:
    resp = client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=False,
    )
    assert resp.status_code == 303, resp.text


def _seed_security_world() -> SecurityWorld:
    from app.data import db as dbmod

    assert dbmod.SessionLocal is not None
    with dbmod.SessionLocal() as db:
        admin = (
            db.query(WebUser).filter(WebUser.username == BOOTSTRAP_USER).first()
        )
        assert admin is not None

        upsert_source(db, name="chat", kind="chat", address="127.0.0.1:1", is_default=True)
        upsert_source(db, name="embed", kind="embed", address="127.0.0.1:2", is_default=True)
        db.add(
            CatalogModel(source_name="chat", kind="chat", model_id="m1", enabled=True)
        )
        db.add(
            CatalogModel(source_name="chat", kind="chat", model_id="m2", enabled=True)
        )
        db.add(
            CatalogModel(source_name="embed", kind="embed", model_id="e1", enabled=True)
        )

        if not db.query(ApiKey).filter(ApiKey.owner_user_id == admin.id).first():
            db.add(
                ApiKey(
                    label="admin-setup",
                    key_hash=hash_api_key("gw_sec_admin_setup"),
                    key_prefix="gw_sec",
                    owner_user_id=admin.id,
                    is_active=True,
                )
            )

        alice = WebUser(
            username=ALICE_NAME,
            password_hash=hash_password(ALICE_PASSWORD),
            is_platform_admin=False,
            is_active=True,
        )
        bob = WebUser(
            username=BOB_NAME,
            password_hash=hash_password(BOB_PASSWORD),
            is_platform_admin=False,
            is_active=True,
        )
        db.add(alice)
        db.add(bob)
        db.flush()

        sync_user_grants(db, alice, ["chat"])
        sync_user_grants(db, bob, ["chat"])

        alice_key = ApiKey(
            label="alice",
            key_hash=hash_api_key(ALICE_API_KEY),
            key_prefix="gw_sec",
            owner_user_id=alice.id,
            is_active=True,
        )
        bob_key = ApiKey(
            label="bob",
            key_hash=hash_api_key(BOB_API_KEY),
            key_prefix="gw_sec",
            owner_user_id=bob.id,
            is_active=True,
        )
        db.add(alice_key)
        db.add(bob_key)
        db.flush()
        db.add(ServiceGrant(api_key_id=alice_key.id, service="chat"))
        db.add(ServiceGrant(api_key_id=bob_key.id, service="chat"))
        db.add(
            ModelAllowlist(api_key_id=alice_key.id, service="chat", model_name="m1")
        )
        db.commit()

        return SecurityWorld(
            admin_id=admin.id,
            alice_id=alice.id,
            bob_id=bob.id,
            alice_key_id=alice_key.id,
            bob_key_id=bob_key.id,
        )


@pytest.fixture()
def sec(gateway_client):
    world = _seed_security_world()
    return gateway_client, world


@pytest.mark.parametrize(
    "path",
    [
        "/users",
        "/settings/access",
        "/services",
        "/smtp",
        "/audit",
    ],
)
def test_anonymous_admin_routes_redirect_login(gateway_client, path: str):
    resp = gateway_client.get(path, follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/login")


def _auth_check(client, *, uri: str, api_key: str | None = None, payload: dict | None = None):
    """Simulate nginx auth_request → /v1/auth/check."""
    headers = {
        "X-Original-Uri": uri,
        "X-Original-Method": "POST",
    }
    if api_key:
        headers["X-Api-Key"] = api_key
    return client.post(
        "/v1/auth/check",
        headers=headers,
        json=payload or {},
        follow_redirects=False,
    )


@pytest.mark.parametrize(
    "path",
    [
        "/users",
        "/settings/access",
        "/settings/limits",
        "/models",
        "/services",
        "/smtp",
        "/alerts",
        "/audit",
    ],
)
def test_regular_user_forbidden_on_admin_routes(sec, path: str):
    client, _world = sec
    _login(client, ALICE_NAME, ALICE_PASSWORD)
    resp = client.get(path, follow_redirects=False)
    assert resp.status_code == 403
    assert "Forbidden" in resp.text


def test_regular_user_forbidden_on_services(sec):
    client, _world = sec
    _login(client, ALICE_NAME, ALICE_PASSWORD)
    assert client.get("/services", follow_redirects=False).status_code == 403
    assert client.get("/services/status", follow_redirects=False).status_code == 403
    assert client.post("/services", data={}, follow_redirects=False).status_code == 403


def test_regular_user_can_use_self_service_pages(sec):
    client, _world = sec
    _login(client, ALICE_NAME, ALICE_PASSWORD)
    for path in ("/me", "/keys", "/account", "/privacy"):
        resp = client.get(path, follow_redirects=False)
        assert resp.status_code == 200, path


def test_admin_can_open_ops_pages(sec):
    client, _world = sec
    _login(client, BOOTSTRAP_USER, BOOTSTRAP_PASSWORD)
    for path in ("/users", "/settings/access", "/services", "/smtp"):
        resp = client.get(path, follow_redirects=False)
        assert resp.status_code == 200, path


def test_users_page_links_create_key_for_members(sec):
    client, world = sec
    _login(client, BOOTSTRAP_USER, BOOTSTRAP_PASSWORD)
    resp = client.get("/users", follow_redirects=False)
    assert resp.status_code == 200
    assert f'/keys/new?owner_user_id={world.alice_id}' in resp.text
    assert f'/keys/new?owner_user_id={world.bob_id}' in resp.text
    assert f'/keys?owner_user_id={world.alice_id}' in resp.text


def test_keys_list_filter_by_owner(sec):
    client, world = sec
    _login(client, BOOTSTRAP_USER, BOOTSTRAP_PASSWORD)
    resp = client.get(f"/keys?owner_user_id={world.bob_id}", follow_redirects=False)
    assert resp.status_code == 200
    assert "Showing keys for" in resp.text
    assert ">bob<" in resp.text or 'label">bob' in resp.text
    assert ">alice<" not in resp.text.split("keys-table")[1] if "keys-table" in resp.text else True


def test_regular_user_cannot_filter_keys_by_other_owner(sec):
    client, world = sec
    _login(client, ALICE_NAME, ALICE_PASSWORD)
    resp = client.get(f"/keys?owner_user_id={world.bob_id}", follow_redirects=False)
    assert resp.status_code == 403


def test_admin_keys_new_honors_owner_query(sec):
    client, world = sec
    _login(client, BOOTSTRAP_USER, BOOTSTRAP_PASSWORD)
    resp = client.get(
        f"/keys/new?owner_user_id={world.bob_id}", follow_redirects=False
    )
    assert resp.status_code == 200
    assert f'value="{world.bob_id}" selected' in resp.text


def test_regular_user_cannot_prefill_owner_on_keys_new(sec):
    client, world = sec
    _login(client, ALICE_NAME, ALICE_PASSWORD)
    resp = client.get(
        f"/keys/new?owner_user_id={world.bob_id}", follow_redirects=False
    )
    assert resp.status_code == 200
    assert 'id="key-owner"' not in resp.text
    assert f"owner_user_id={world.bob_id}" not in resp.text


def test_setup_wizard_blocks_admin_until_minimum_done(gateway_client):
    _login(gateway_client, BOOTSTRAP_USER, BOOTSTRAP_PASSWORD)
    resp = gateway_client.get("/users", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/setup")


def test_user_cannot_open_other_users_key(sec):
    client, world = sec
    _login(client, ALICE_NAME, ALICE_PASSWORD)
    resp = client.get(f"/keys/{world.bob_key_id}", follow_redirects=False)
    assert resp.status_code == 403
    resp = client.get(f"/keys/{world.bob_key_id}/partial", follow_redirects=False)
    assert resp.status_code == 403


def test_user_cannot_post_admin_settings(sec):
    client, _world = sec
    _login(client, ALICE_NAME, ALICE_PASSWORD)
    resp = client.post(
        "/settings/access",
        data={"max_keys_per_user": "99", "allow_self_registration": "on"},
        follow_redirects=False,
    )
    assert resp.status_code == 403


def test_user_cannot_create_users(sec):
    client, _world = sec
    _login(client, ALICE_NAME, ALICE_PASSWORD)
    resp = client.post(
        "/users/new",
        data={"username": "evil", "password": "EvilPass123!", "password2": "EvilPass123!"},
        follow_redirects=False,
    )
    assert resp.status_code == 403


def test_api_missing_key_is_unauthorized(sec):
    client, _world = sec
    resp = client.get("/v1/models", follow_redirects=False)
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"


def test_api_invalid_key_is_unauthorized(sec):
    client, _world = sec
    resp = client.get(
        "/v1/models",
        headers={"X-Api-Key": "not-a-real-key"},
        follow_redirects=False,
    )
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"


def test_api_key_denied_ungranted_service(sec):
    client, _world = sec
    resp = _auth_check(
        client,
        uri="/s/embed/v1/embeddings",
        api_key=ALICE_API_KEY,
        payload={"model": "e1", "input": "hello"},
    )
    assert resp.status_code == 403
    assert resp.json()["error"] == "service_not_allowed"


def test_api_key_denied_model_not_on_key_allowlist(sec):
    client, _world = sec
    resp = _auth_check(
        client,
        uri="/v1/chat/completions",
        api_key=ALICE_API_KEY,
        payload={
            "model": "m2",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 403
    assert resp.json()["error"] == "model_not_allowed"
