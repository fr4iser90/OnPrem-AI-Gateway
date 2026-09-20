"""Web UI bounded context — browser session, HTML pages, static assets.

Every signed-in person is a WebUser (table web_users). Platform ops require
is_platform_admin=True — enforced in session.py, not by package name.

URLs live at the site root (/me, /keys, /login), never under /admin.

Layout (DDD-light):
  public/     — login, logout
  portal/     — self-service (all users): me, keys, teams, usage
  platform/   — ops (platform admin only): dashboard, users, services, settings, ops/usage
  session.py  — require_user / require_platform_admin
  accounts.py — register, profile, SMTP
  shared.py   — helpers + templates
  usage_pages.py — shared You vs Ops usage builders

See docs/ARCHITECTURE.md.
"""