from __future__ import annotations

from starlette.testclient import TestClient

from app.data.backends import upsert_source
from app.data.db import hash_api_key, hash_password
from app.data.models import WebUser, ApiKey, AuthSettings, UsageEvent
from tests.constants import (
    BOOTSTRAP_PASSWORD,
    BOOTSTRAP_USER,
    FORCED_NEW_PASSWORD,
    FORCED_OLD_PASSWORD,
    FORCED_USER_NAME,
    USER1_NAME,
    USER1_PASSWORD,
)


def _login(client: TestClient, username: str, password: str):
    return client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=False,
    )


def _seed_ui_world():
    from app.data import db as dbmod

    assert dbmod.SessionLocal is not None
    with dbmod.SessionLocal() as db:
        admin = db.query(WebUser).filter(WebUser.username == BOOTSTRAP_USER).first()
        assert admin is not None

        upsert_source(db, name="chat", kind="chat", address="127.0.0.1:1", is_default=True)

        auth = db.query(AuthSettings).first()
        if auth is None:
            auth = AuthSettings()
            db.add(auth)
            db.flush()
        auth.show_global_stats = True

        if not db.query(ApiKey).filter(ApiKey.owner_user_id == admin.id).first():
            db.add(
                ApiKey(
                    label="admin-seed",
                    key_hash=hash_api_key("gw_ui_admin_seed"),
                    key_prefix="gw_ui",
                    owner_user_id=admin.id,
                    is_active=True,
                )
            )

        user = db.query(WebUser).filter(WebUser.username == USER1_NAME).first()
        if user is None:
            user = WebUser(
                username=USER1_NAME,
                password_hash=hash_password(USER1_PASSWORD),
                is_platform_admin=False,
                is_active=True,
            )
            db.add(user)
            db.flush()
            db.add(
                ApiKey(
                    label="user1-key",
                    key_hash=hash_api_key("gw_ui_user1_key"),
                    key_prefix="gw_ui",
                    owner_user_id=user.id,
                    is_active=True,
                )
            )

        forced = db.query(WebUser).filter(WebUser.username == FORCED_USER_NAME).first()
        if forced is None:
            forced = WebUser(
                username=FORCED_USER_NAME,
                password_hash=hash_password(FORCED_OLD_PASSWORD),
                is_platform_admin=False,
                is_active=True,
                must_change_password=True,
            )
            db.add(forced)

        db.commit()


def test_admin_can_view_ops_pages_with_demo_usage(data_dir):
    from app.config import get_settings
    from app.data import db as dbmod
    from app.demo_seed import seed_demo_usage
    from app.main import create_app

    get_settings.cache_clear()
    with TestClient(create_app()) as gateway_client:
        _seed_ui_world()

        assert dbmod.SessionLocal is not None
        with dbmod.SessionLocal() as db:
            seed_demo_usage(db, count=120)
            db.commit()

        login = _login(gateway_client, BOOTSTRAP_USER, BOOTSTRAP_PASSWORD)
        assert login.status_code == 303

        overview = gateway_client.get("/", follow_redirects=False)
        assert overview.status_code == 200
        assert "Ops Overview" in overview.text
        assert "Seed demo" not in overview.text

        page = gateway_client.get("/usage", follow_redirects=False)
        assert page.status_code == 200
        assert "Usage" in page.text
        assert "Model averages" in page.text
        assert "Energy by owner" not in page.text

        ops_usage = gateway_client.get("/ops/usage", follow_redirects=False)
        assert ops_usage.status_code == 200
        assert "Ops usage" in ops_usage.text
        assert 'name="owner_user_id"' in ops_usage.text
        assert 'name="range"' in ops_usage.text
        assert "Fleet-wide" in ops_usage.text

        ops_30 = gateway_client.get("/ops/usage?range=30", follow_redirects=False)
        assert ops_30.status_code == 200
        assert "30d" in ops_30.text or "30 days" in ops_30.text

    with dbmod.SessionLocal() as db:
        assert db.query(UsageEvent).filter(UsageEvent.is_demo.is_(True)).count() == 120


def test_user_overview_shows_pulse(gateway_client):
    _seed_ui_world()

    login = _login(gateway_client, USER1_NAME, USER1_PASSWORD)
    assert login.status_code == 303
    assert login.headers["location"] == "/me"

    page = gateway_client.get("/me", follow_redirects=False)
    assert page.status_code == 200
    assert "Overview" in page.text
    assert "Requests · 60 min" in page.text
    assert "OnPrem AI Gateway pulse" in page.text


def test_first_login_password_change_flow(gateway_client):
    _seed_ui_world()

    login = _login(gateway_client, FORCED_USER_NAME, FORCED_OLD_PASSWORD)
    assert login.status_code == 303
    assert login.headers["location"] == "/account"

    forced_redirect = gateway_client.get("/me", follow_redirects=False)
    assert forced_redirect.status_code == 303
    assert forced_redirect.headers["location"] == "/account"

    account = gateway_client.get("/account", follow_redirects=False)
    assert account.status_code == 200
    assert "You must change your password before continuing." in account.text

    update = gateway_client.post(
        "/account/update",
        data={
            "password": FORCED_NEW_PASSWORD,
            "password2": FORCED_NEW_PASSWORD,
        },
        follow_redirects=False,
    )
    assert update.status_code == 303
    assert update.headers["location"] == "/me"

    page = gateway_client.get("/me", follow_redirects=False)
    assert page.status_code == 200
    assert "Password updated" in page.text
    assert "You must change your password before continuing." not in page.text

    unlocked = gateway_client.get("/me", follow_redirects=False)
    assert unlocked.status_code == 200


def test_account_email_unchanged_skips_password(gateway_client):
    _seed_ui_world()

    login = _login(gateway_client, USER1_NAME, USER1_PASSWORD)
    assert login.status_code == 303

    resp = gateway_client.post(
        "/account/update",
        data={"email": "", "current_password": "", "password": "", "password2": ""},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/account"

    page = gateway_client.get("/account", follow_redirects=False)
    assert page.status_code == 200
    assert "No changes to save" in page.text
