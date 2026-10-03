"""Pulse design contract — every HTML surface, not a subset of ops pages.

A failure here means either:

* a **new page/template was added** and never classified (so it can silently
  ship the old mint/IBM/card-soup look), or
* an **existing page drifted** (old brand color, old type, hidden eyebrows,
  inline HTML error pages, legacy LocalAI / Gateway chrome).

Assertion messages say *why* the hit is a drift.

Surfaces in scope: every ``*.html`` under ``app/``, ``style.css``, ``favicon.svg``,
and any Python ``HTMLResponse`` that contains markup. Out of scope: JSON API,
nginx error JSON, email subjects, and ``tests/integration_helpers.py`` (those
HTML files are offline test artifacts, not served by OnPrem AI Gateway).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
TEMPLATES = APP / "web" / "templates"
STYLE = APP / "web" / "static" / "style.css"
FAVICON = APP / "web" / "static" / "favicon.svg"

# Pulse tokens (dark canvas). Light theme may remap accent to teal #0d9488.
DARK_BG = "#0e1116"
DARK_ACCENT = "#5eead4"

# Old mint / IBM face — leftover from the previous visual system.
MINT_HEX = re.compile(r"#3d9a7a|#2d7a62|#b8f0d6|#04140f", re.I)
PLEX_SANS = re.compile(r"IBM Plex Sans")
LOCALAI_CHROME = re.compile(r"LocalAI(\s|<span)")

# kind → how the file must participate in the Pulse system
#   auth     — Mode C: login-wrap card, no sidebar
#   browse   — Mode A: compact page-head, column scroll
#   task     — Mode B: content--fill + one fill panel
#   wizard   — first-run steps; inherit setup_base
#   error    — 403 (and future status pages); Pulse chrome, not raw HTML
#   shell    — base.html
#   fragment — include-only partials (not a route by themselves)
PAGE_KIND: dict[str, str] = {
    "base.html": "shell",
    # Auth (user-facing, no admin chrome)
    "login.html": "auth",
    "register.html": "auth",
    "forgot.html": "auth",
    "reset.html": "auth",
    # User + ops browse
    "me.html": "browse",
    "dashboard.html": "browse",
    "keys.html": "browse",
    "usage.html": "browse",
    "usage_daily.html": "browse",
    "users.html": "browse",
    "teams.html": "browse",
    "models.html": "browse",
    "services.html": "browse",
    "settings.html": "browse",
    "smtp.html": "browse",
    "alerts.html": "browse",
    "audit.html": "browse",
    "setup_done.html": "browse",
    "legal_imprint.html": "legal",
    "legal_privacy.html": "legal",
    "legal_cookies.html": "legal",
    # Task
    "account.html": "task",
    "privacy.html": "task",
    "key_form.html": "task",
    "team_form.html": "task",
    # Wizard
    "setup_base.html": "wizard",
    "setup_sources.html": "wizard",
    "setup_access.html": "wizard",
    "setup_key.html": "wizard",
    # Error
    "forbidden.html": "error",
    # Fragments
    "_flash.html": "fragment",
    "_legal_links.html": "fragment",
    "_legal_imprint_body.html": "fragment",
    "_legal_privacy_body.html": "fragment",
    "_legal_cookies_body.html": "fragment",
    "_pulse.html": "fragment",
    "_pw_rules.html": "fragment",
    "_source_chips.html": "fragment",
    "_settings_subnav.html": "fragment",
    "_settings_access.html": "fragment",
    "_settings_limits.html": "fragment",
    "_settings_routing.html": "fragment",
    "_settings_privacy.html": "fragment",
    "_settings_system.html": "fragment",
    "_key_edit_expand.html": "fragment",
    "_user_grant_form.html": "fragment",
    "_user_grant_expand.html": "fragment",
    "_user_grant_limits_expand.html": "fragment",
    "_user_grant_sources_expand.html": "fragment",
    "_catalog_model_row.html": "fragment",
    "_usage_filters.html": "fragment",
    "partials/model_picker.html": "fragment",
    "partials/model_meta.html": "fragment",
    "partials/_alias_card.html": "fragment",
}

# SMTP From display name is email identity, not UI chrome.
SMTP_FROM_ALLOW = "cfg.from_name if cfg else 'OnPrem AI Gateway'"

# HTMX/partial error strings (plain text, no tags) are not pages.
HTML_MARKUP_IN_PY = re.compile(
    r"""HTMLResponse\(\s*(?:b?['"][^'"]*<[^'"]*['"]|['"][^'"]*<html)""",
    re.I,
)


def _rel_templates() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in TEMPLATES.rglob("*.html"):
        out[path.relative_to(TEMPLATES).as_posix()] = path
    return out


def _css_token(name: str) -> str:
    text = STYLE.read_text(encoding="utf-8")
    m = re.search(rf"{re.escape(name)}\s*:\s*([^;]+);", text)
    assert m, f"CSS token {name} missing — Pulse dark tokens must live on :root."
    return m.group(1).strip()


def test_every_html_template_is_classified():
    """New files must be added to PAGE_KIND or they skip the design audit."""
    found = _rel_templates()
    missing = sorted(set(found) - set(PAGE_KIND))
    extra = sorted(set(PAGE_KIND) - set(found))
    assert not missing, (
        "Untracked HTML templates — these pages/partials were never classified, "
        "so Pulse drift on them would not fail CI. Add each to PAGE_KIND in "
        f"tests/test_design_contract.py and apply the contract: {missing}"
    )
    assert not extra, (
        "PAGE_KIND lists deleted files — remove the stale entries: " + ", ".join(extra)
    )


def test_pulse_tokens_on_canvas():
    assert _css_token("--bg") == DARK_BG, (
        "Dark canvas drifted — Pulse spec is anthrazit #0e1116, not the old charcoal/mint face."
    )
    assert _css_token("--accent") == DARK_ACCENT, (
        "Accent drifted — Pulse is one cyan (#5eead4 dark / #0d9488 light), not mint or indigo."
    )
    assert "Inter" in _css_token("--font"), (
        "Type drifted — Pulse body face is Inter, not IBM Plex Sans."
    )
    css = STYLE.read_text(encoding="utf-8")
    assert "--accent: #0d9488" in css, (
        "Light theme must keep the teal accent (#0d9488), not mint."
    )


def test_no_old_mint_or_plex_sans_on_served_ui():
    """#3d9a7a / IBM Plex Sans is the previous brand. Any leftover is visible drift."""
    hits: list[str] = []
    for path in [STYLE, FAVICON, *TEMPLATES.rglob("*.html")]:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT).as_posix()
        if MINT_HEX.search(text):
            hits.append(f"{rel}: old mint hex (Pulse uses cyan #5eead4 / #0d9488)")
        if PLEX_SANS.search(text):
            hits.append(f"{rel}: IBM Plex Sans (Pulse body face is Inter; Mono is still OK)")
    assert not hits, "Visual drift:\n  " + "\n  ".join(hits)


def test_favicon_is_pulse_cyan():
    svg = FAVICON.read_text(encoding="utf-8")
    assert DARK_ACCENT in svg, (
        "Favicon still uses another accent — mark/stroke must be Pulse cyan #5eead4."
    )
    assert "#3d9a7a" not in svg.lower()


def test_base_shell_is_pulse():
    base = (TEMPLATES / "base.html").read_text(encoding="utf-8")
    assert "Inter" in _css_token("--font"), (
        "Type drifted — Pulse body face is Inter, not IBM Plex Sans."
    )
    css = STYLE.read_text(encoding="utf-8")
    assert "@font-face" in css and "/static/fonts/inter-latin-400.woff2" in css, (
        "Inter must be self-hosted. Loading fonts.googleapis.com sends visitor IPs to Google (DSGVO)."
    )
    assert 'class="topbar-brand">OnPrem AI Gateway</span>' in base
    assert re.search(r'<p class="brand">\s*.*?OnPrem AI Gateway', base, re.S)
    assert "LocalAI" not in base, (
        "Shell chrome still says LocalAI — wordmark must be OnPrem AI Gateway "
        "(sidebar, topbar). Product name in emails is a separate surface."
    )
    assert 'src="/static/theme.js"' in base
    # Auth pages (else branch) must get theme.js too — login toggle + stored theme.
    else_at = base.index("{% else %}")
    assert 'src="/static/theme.js"' in base[else_at:], (
        "Auth/error pages (login, register, forgot, reset, logged-out 403) "
        "did not load theme.js — they would keep OS theme but the toggle would be dead."
    )


def test_no_eyebrow_chrome():
    hits = [
        p.relative_to(TEMPLATES).as_posix()
        for p in TEMPLATES.rglob("*.html")
        if 'class="eyebrow"' in p.read_text(encoding="utf-8")
    ]
    assert not hits, (
        "Eyebrow labels are hidden leftover chrome from the old page-head. "
        f"Remove them: {hits}"
    )


def test_auth_pages_use_login_card():
    for name in ("login.html", "register.html", "forgot.html", "reset.html"):
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        assert "block body" in text or "{% block body %}" in text, (
            f"{name}: Auth pages must use {{% block body %}} (Mode C, no sidebar). "
            "If this extends content instead, a logged-out user gets a broken shell."
        )
        assert "login-wrap" in text and "login-card" in text, (
            f"{name}: Mode C drift — auth must be a centered login-card on the Pulse canvas, "
            "not a full app layout and not a naked <h1>."
        )
        assert "LocalAI" not in text, (
            f"{name}: visible LocalAI copy — wordmark must be OnPrem AI Gateway."
        )


def test_error_page_is_pulse_not_raw_html():
    text = (TEMPLATES / "forbidden.html").read_text(encoding="utf-8")
    assert "{% extends \"base.html\" %}" in text
    assert "login-wrap" in text, (
        "Logged-out 403 must use the auth card so it matches login, not browser-default HTML."
    )
    assert "page-head--compact" in text, (
        "Logged-in 403 must use compact page-head like every other app page."
    )


def test_no_inline_html_pages_in_python():
    """Raw HTMLResponse('<h1>…') bypasses CSS/tokens — that is how 403 used to look."""
    hits: list[str] = []
    for path in APP.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if HTML_MARKUP_IN_PY.search(text) or re.search(
            r"""HTMLResponse\(\s*["'][^"']*<[hHp]""", text
        ):
            hits.append(path.relative_to(ROOT).as_posix())
    assert not hits, (
        "Python still returns markup HTML without a template — that page will not "
        "get Inter/cyan/anthrazit. Render a classified template instead: "
        + ", ".join(hits)
    )


def test_template_response_names_are_classified():
    """Routes cannot render a template that PAGE_KIND does not know."""
    names: set[str] = set()
    for path in APP.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        names.update(re.findall(r'TemplateResponse\([^)]*?["\']([\w./-]+\.html)["\']', text, re.S))
    unknown = sorted(names - set(PAGE_KIND))
    assert not unknown, (
        "TemplateResponse renders unclassified files — treat them as new pages and "
        f"add Pulse checks: {unknown}"
    )


def test_browse_and_task_headers_are_compact():
    for name, kind in PAGE_KIND.items():
        if kind not in {"browse", "task", "error", "legal"}:
            continue
        if name == "forbidden.html":
            continue  # checked separately (dual body/content)
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        assert "page-head--compact" in text, (
            f"{name} ({kind}): missing compact page-head — Pulse pages use h1 + optional "
            "page-meta, not essay ledes / eyebrows."
        )


def test_task_pages_fill_the_pane():
    for name, kind in PAGE_KIND.items():
        if kind != "task":
            continue
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        assert "content--fill" in text, (
            f"{name}: Task page must set content--fill (Mode B) so the form fills the pane "
            "instead of a thin strip plus empty canvas."
        )
        assert "panel--fill" in text, f"{name}: Task page needs panel--fill."


def test_browse_pages_do_not_fill():
    bad = [
        name
        for name, kind in PAGE_KIND.items()
        if kind == "browse" and "content--fill" in (TEMPLATES / name).read_text(encoding="utf-8")
    ]
    assert not bad, (
        "Browse pages must scroll the main column (Mode A), not lock one fill panel: "
        + ", ".join(bad)
    )


def test_pulse_partial_on_overviews():
    me = (TEMPLATES / "me.html").read_text(encoding="utf-8")
    ops = (TEMPLATES / "dashboard.html").read_text(encoding="utf-8")
    pulse = (TEMPLATES / "_pulse.html").read_text(encoding="utf-8")
    assert '{% include "_pulse.html" %}' in me, (
        "User Overview (/me) must include the Pulse block (status, 60-min count, p95, area). "
        "Metric-card strips are the old face."
    )
    assert '{% include "_pulse.html" %}' in ops, (
        "Ops Overview (/) must include the same Pulse block as /me, not a separate hero."
    )
    assert "Requests · 60 min" in pulse, (
        "Pulse copy must stay honest — count in 60 minutes, not a fake req/min."
    )
    assert "class=\"pulse\"" in pulse or "class='pulse'" in pulse or 'class="pulse"' in pulse


def test_usage_reuses_pulse_surface_not_stat_strip():
    text = (TEMPLATES / "usage.html").read_text(encoding="utf-8")
    assert 'class="pulse"' in text, (
        "Usage must use the Pulse surface (KPIs + area), not a 5-cell stat-strip of cards."
    )
    assert "stat-strip" not in text
    assert "cards-hero" not in text


def test_localai_not_in_visible_templates():
    hits: list[str] = []
    for rel, path in _rel_templates().items():
        text = path.read_text(encoding="utf-8")
        if SMTP_FROM_ALLOW in text:
            text = text.replace(SMTP_FROM_ALLOW, "")
        if LOCALAI_CHROME.search(text) or "LocalAI Gateway" in text:
            hits.append(rel)
    assert not hits, (
        "Visible UI still says LocalAI Gateway — Pulse chrome is OnPrem AI Gateway. "
        f"(SMTP From name default is the only allowed leftover.) Drift in: {hits}"
    )


def test_stat_strip_and_hero_cards_not_used_on_pages():
    hits = [
        rel
        for rel, path in _rel_templates().items()
        if re.search(r"stat-strip|cards-hero", path.read_text(encoding="utf-8"))
    ]
    assert not hits, (
        "Metric-card soup (stat-strip / cards-hero) is the old overview. "
        f"Pulse uses one surface: {hits}"
    )


def test_no_google_fonts_on_served_ui():
    """Google Fonts CDN sends visitor IPs to Google — a common DSGVO fail in DE."""
    hits: list[str] = []
    for path in (APP / "web").rglob("*"):
        if path.suffix.lower() not in {".html", ".css", ".js", ".svg"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "fonts.googleapis.com" in text or "fonts.gstatic.com" in text:
            hits.append(path.relative_to(ROOT).as_posix())
    assert not hits, (
        "Served UI still loads Google Fonts. Self-host under /static/fonts: "
        + ", ".join(hits)
    )


def test_legal_pages_cover_de_duties():
    fonts = ROOT / "app" / "web" / "static" / "fonts"
    assert (fonts / "inter-latin-400.woff2").is_file()
    imprint = (TEMPLATES / "legal_imprint.html").read_text(encoding="utf-8")
    privacy = (TEMPLATES / "legal_privacy.html").read_text(encoding="utf-8")
    cookies = (TEMPLATES / "_legal_cookies_body.html").read_text(encoding="utf-8")
    links = (TEMPLATES / "_legal_links.html").read_text(encoding="utf-8")
    base = (TEMPLATES / "base.html").read_text(encoding="utf-8")
    assert "legal-wrap" in imprint and "page-head--compact" in imprint
    assert "Art. 13" in privacy or "DSGVO" in privacy
    assert "GDPR" in (TEMPLATES / "_legal_privacy_body.html").read_text(encoding="utf-8")
    assert "onprem_session" in cookies
    assert "Consent-Banner" in cookies or "keine</strong> Analyse" in cookies
    assert "no consent banner" in cookies.lower()
    assert "/legal/imprint" in links and "/legal/privacy" in links and "/legal/cookies" in links
    assert "Imprint" in links and "Privacy" in links
    assert '_legal_links.html' in base
    assert "support@fr4iser.com" in (TEMPLATES / "_settings_system.html").read_text(
        encoding="utf-8"
    )
