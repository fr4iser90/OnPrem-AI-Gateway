"""Web UI layout conventions — structural + CSS token assertions.

Industry-aligned web UI shell rules we enforce:

1. One content column that fills the main pane (``--page-max: none``).
2. No half-viewport page wrappers (``.stack--form`` / ``.form-narrow`` / ``.panel-chart``
   must not introduce a narrower max-width than the page column).
3. Authenticated app pages use ``page-head page-head--compact`` (not bare ``<h1>``).
4. ``base.html`` wraps content in ``.content-inner``.
5. Charts fill their panel (``.chart`` / ``.panel-chart`` → ``max-width: none``).

Layout modes (Browse / Task / Auth) are documented in ``docs/LAYOUT.md`` and enforced
below via ``test_task_pages_use_content_fill_contract`` and
``test_browse_pages_do_not_use_content_fill``.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "app" / "web" / "templates"
STYLE = ROOT / "app" / "web" / "static" / "style.css"
LAYOUT_DOC = ROOT / "docs" / "LAYOUT.md"

# Mode B — Task pages: fixed shell, one panel--fill, sticky actions (see docs/LAYOUT.md)
TASK_PAGES = frozenset({
    "account.html",
    "privacy.html",
    "key_form.html",
    "team_form.html",
})

# Mode A — Browse pages: main column scrolls; must not use content--fill
BROWSE_PAGES = frozenset({
    "me.html",
    "usage.html",
    "usage_daily.html",
    "keys.html",
    "users.html",
    "teams.html",
    "models.html",
    "services.html",
    "dashboard.html",
    "audit.html",
    "settings.html",
    "smtp.html",
    "alerts.html",
    "setup_done.html",
    "legal_imprint.html",
    "legal_privacy.html",
    "legal_cookies.html",
})

# App pages that render inside the authenticated shell (extend base + block content)
APP_PAGES = sorted(
    p
    for p in TEMPLATES.glob("*.html")
    if p.name
    not in {
        "base.html",
        "setup_base.html",  # uses page-head via its own content
        "_pw_rules.html",
        "login.html",
        "register.html",
        "forgot.html",
        "reset.html",
        # wizard bodies are partials included by setup_base
        "setup_key.html",
        "setup_sources.html",
        "setup_access.html",
    }
)

# Setup partials are OK without their own page-head (parent provides it)
PARTIALS = {
    "setup_key.html",
    "setup_sources.html",
    "setup_access.html",
    "_pw_rules.html",
}


def _css_token(name: str) -> str:
    text = STYLE.read_text(encoding="utf-8")
    # Prefer :root / dark theme block
    m = re.search(rf"{re.escape(name)}\s*:\s*([^;]+);", text)
    assert m, f"CSS token {name} not found in style.css"
    return m.group(1).strip()


def _rule_max_width(selector: str) -> str | None:
    """Return max-width for a rule that includes ``selector`` (comma lists OK)."""
    text = STYLE.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    for m in re.finditer(r"([^{}]+)\{([^}]+)\}", text):
        sel_list = m.group(1)
        body = m.group(2)
        parts = [p.strip() for p in sel_list.split(",") if p.strip()]
        if selector not in parts:
            continue
        mw = re.search(r"max-width\s*:\s*([^;]+);", body)
        if mw:
            return mw.group(1).strip()
    return None


def test_layout_tokens_fill_main_column():
    assert _css_token("--page-max") == "none"
    assert _css_token("--sidebar-width") == "240px"
    assert _css_token("--bg") == "#0e1116"
    assert _css_token("--accent") == "#5eead4"
    assert "Inter" in _css_token("--font")
    pad_x = _css_token("--page-pad-x")
    assert pad_x.endswith("rem") or pad_x.endswith("px")
    # Meaningful horizontal padding (≥ 1rem)
    if pad_x.endswith("rem"):
        assert float(pad_x[:-3]) >= 1.0


def test_templates_drop_eyebrow_chrome():
    hits = []
    for path in TEMPLATES.rglob("*.html"):
        if 'class="eyebrow"' in path.read_text(encoding="utf-8"):
            hits.append(path.name)
    assert not hits, f"Remove hidden eyebrow chrome from: {', '.join(hits)}"


def test_page_wrappers_are_full_width():
    for sel in (".stack--form", ".form-narrow", ".panel-chart", ".chart", ".content-inner"):
        mw = _rule_max_width(sel)
        assert mw is not None, f"{sel} should declare max-width"
        assert mw in {"none", "var(--page-max)"}, (
            f"{sel} max-width must be none or var(--page-max), got {mw!r}"
        )


def test_base_wraps_content_inner():
    base = (TEMPLATES / "base.html").read_text(encoding="utf-8")
    assert 'class="content-inner"' in base
    assert "{% block content %}" in base
    # content block must live inside content-inner
    inner_at = base.index('class="content-inner"')
    block_at = base.index("{% block content %}")
    assert block_at > inner_at


def test_app_pages_use_compact_page_head():
    missing: list[str] = []
    for path in APP_PAGES:
        text = path.read_text(encoding="utf-8")
        if "{% block content %}" not in text and "{% block wizard_body %}" not in text:
            continue
        if "page-head page-head--compact" not in text and "page-head--compact" not in text:
            # setup_done etc. must still have it
            missing.append(path.name)
    assert not missing, (
        "App pages missing page-head--compact: " + ", ".join(missing)
    )


def test_no_bare_h1_outside_page_head_on_app_pages():
    """Bare top-level <h1> without page-head chrome is a layout drift."""
    bad: list[str] = []
    for path in APP_PAGES:
        text = path.read_text(encoding="utf-8")
        if "page-head" in text:
            continue
        if re.search(r"<h1[\s>]", text):
            bad.append(path.name)
    assert not bad, f"Pages with bare <h1> (use page-head): {', '.join(bad)}"


def test_no_form_narrow_class_in_templates():
    hits = []
    for path in TEMPLATES.glob("*.html"):
        text = path.read_text(encoding="utf-8")
        if "form-narrow" in text:
            hits.append(path.name)
    assert not hits, f"Remove form-narrow from: {', '.join(hits)}"


def test_no_stack_form_narrowing_in_templates():
    """stack--form is deprecated as a width mode; plain stack is OK."""
    hits = []
    for path in TEMPLATES.glob("*.html"):
        if "stack--form" in path.read_text(encoding="utf-8"):
            hits.append(path.name)
    assert not hits, f"Replace stack--form with stack: {', '.join(hits)}"


def test_layout_doc_exists():
    assert LAYOUT_DOC.is_file()
    text = LAYOUT_DOC.read_text(encoding="utf-8")
    assert "Mode A" in text and "Mode B" in text and "content--fill" in text


def _task_page_source(name: str) -> str:
    text = (TEMPLATES / name).read_text(encoding="utf-8")
    return text


def test_task_pages_use_content_fill_contract():
    for name in sorted(TASK_PAGES):
        text = _task_page_source(name)
        assert "content--fill" in text, f"{name} must set content--fill"
        assert "panel--fill" in text, f"{name} must use panel--fill"
        assert "wizard-form--fill" in text or "profile-form--fill" in text or "privacy-fill" in text, name
        assert "wizard-actions--sticky" in text or "profile-actions" in text, name
    setup = (TEMPLATES / "setup_base.html").read_text(encoding="utf-8")
    assert "content--fill" in setup


def test_browse_pages_do_not_use_content_fill():
    violations = []
    for name in sorted(BROWSE_PAGES):
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        if "content--fill" in text:
            violations.append(name)
    assert not violations, f"Browse pages must not use content--fill: {', '.join(violations)}"


def test_account_uses_content_fill_single_form():
    account = (TEMPLATES / "account.html").read_text(encoding="utf-8")
    assert "content--fill" in account
    assert "panel--fill" in account
    assert 'action="/account/update"' in account
    assert account.count('type="submit"') == 1
    assert "details" not in account
    assert "stack--form" not in account
    assert 'style="' not in account


def test_privacy_uses_content_fill():
    text = (TEMPLATES / "privacy.html").read_text(encoding="utf-8")
    assert "content--fill" in text
    assert "panel--fill" in text
    assert "privacy-fill" in text
    assert "form-narrow" not in text
    assert 'style="' not in text


def test_layout_grid_uses_sidebar_token():
    text = STYLE.read_text(encoding="utf-8")
    assert "var(--sidebar-width)" in text
    assert re.search(
        r"\.layout\s*\{[^}]*grid-template-columns:\s*var\(--sidebar-width\)",
        text,
        re.DOTALL,
    )


def test_expected_page_max_and_chart_contract_values():
    """Pin the concrete contract values used in the design scorecard."""
    assert _css_token("--page-max") == "none"
    assert _rule_max_width(".panel-chart") == "none"
    assert _rule_max_width(".chart") == "none"
    assert _rule_max_width(".stack--form") == "none"
    assert _rule_max_width(".form-narrow") == "none"
    assert _rule_max_width(".content-inner") == "var(--page-max)"


def test_field_max_tokens():
    assert _css_token("--field-max") == "32rem"
    assert _css_token("--field-max-sm") == "12.5rem"
    assert _css_token("--field-max-xs") == "9rem"
    css = STYLE.read_text(encoding="utf-8")
    assert "max-width: var(--field-max)" in css
    assert ".field--xs" in css
    assert ".field-gap" in css
    assert "focus-visible" in css


def test_settings_uses_section_labels_not_inline_h3():
    for path in sorted(TEMPLATES.glob("_settings_*.html")):
        if path.name == "_settings_subnav.html":
            continue
        text = path.read_text(encoding="utf-8")
        assert "h3 style=" not in text, path.name
        if re.search(r"<h3[\s>]", text):
            assert "section-label" in text, f"{path.name} missing section-label"


def test_core_pages_inline_style_budget():
    """Showcase / form pages: keep inline styles near zero.

    Dense UIs (models catalog, setup_key) may still use a few layout styles.
    """
    budget = {
        "account.html": 0,
        "privacy.html": 0,
        "login.html": 0,
        "register.html": 0,
        "settings.html": 2,
        "alerts.html": 2,
        "keys.html": 2,
        "smtp.html": 4,
        "team_form.html": 3,
    }
    over: list[str] = []
    for name, max_n in budget.items():
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        n = len(re.findall(r'\sstyle="', text))
        if n > max_n:
            over.append(f"{name}: {n} > {max_n}")
    assert not over, "Inline style budget exceeded: " + "; ".join(over)


def test_services_live_status_uses_panel_head():
    text = (TEMPLATES / "services.html").read_text(encoding="utf-8")
    assert "panel-head" in text
    assert "Live status" in text
    assert 'style="display:flex;align-items:baseline' not in text


def test_a11y_shell_contracts():
    base = (TEMPLATES / "base.html").read_text(encoding="utf-8")
    assert 'class="skip-link"' in base
    assert 'href="#main-content"' in base
    assert 'id="main-content"' in base
    assert 'aria-label="Primary"' in base
    assert "aria-current=\"page\"" in base
    assert "data-nav-toggle" in base
    assert "data-nav-backdrop" in base
    assert 'src="/static/nav.js"' in base
    css = STYLE.read_text(encoding="utf-8")
    assert ".skip-link" in css
    assert ".layout.nav-open .sidebar" in css
    assert ".empty-state" in css
    assert (ROOT / "app" / "web" / "static" / "nav.js").is_file()


def test_models_setup_key_services_inline_budget():
    budget = {
        "models.html": 0,
        "setup_key.html": 0,
        "services.html": 0,
    }
    over = []
    for name, max_n in budget.items():
        n = len(re.findall(r'\sstyle="', (TEMPLATES / name).read_text(encoding="utf-8")))
        if n > max_n:
            over.append(f"{name}: {n} > {max_n}")
    assert not over, "Inline style budget exceeded: " + "; ".join(over)


def test_empty_state_pattern_present():
    models = (TEMPLATES / "models.html").read_text(encoding="utf-8")
    keys = (TEMPLATES / "keys.html").read_text(encoding="utf-8")
    services = (TEMPLATES / "services.html").read_text(encoding="utf-8")
    assert "empty-state" in models and "No models yet" in models
    assert "empty-state" in keys
    assert "empty-state" in services


def test_list_pages_use_empty_state():
    pages = {
        "users.html": "No users yet",
        "audit.html": "No audit events yet",
        "teams.html": "No teams yet",
        "usage.html": "No events",
        "usage_daily.html": "No aggregates yet",
        "me.html": "No keys yet",
        "keys.html": "No API keys yet",
        "models.html": "No models yet",
        "services.html": "No sources to probe",
    }
    missing = []
    for name, phrase in pages.items():
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        if "empty-state" not in text or phrase not in text:
            missing.append(name)
    assert not missing, f"Missing empty-state pattern: {', '.join(missing)}"


def test_a11y_usage_subnav_and_forms():
    usage = (TEMPLATES / "usage.html").read_text(encoding="utf-8")
    daily = (TEMPLATES / "usage_daily.html").read_text(encoding="utf-8")
    filters = (TEMPLATES / "_usage_filters.html").read_text(encoding="utf-8")
    assert 'aria-current="page"' in usage
    assert 'aria-current="page"' in daily
    assert 'role="search"' in filters
    assert 'include "_usage_filters.html"' in usage
    assert 'include "_usage_filters.html"' in daily
    assert "for=\"usage-service\"" in filters
    assert "for=\"usage-owner\"" in filters
    assert "{% if ops_mode %}<th>Owner</th>{% endif %}" in usage or "Owner" in usage
    assert "Owner" in daily
    users = (TEMPLATES / "users.html").read_text(encoding="utf-8")
    assert 'for="new-username"' in users
    assert "details-edit" in users
    audit = (TEMPLATES / "audit.html").read_text(encoding="utf-8")
    assert 'role="status"' in audit or 'role="alert"' in audit
    assert "sr-only" in audit


def test_core_ops_pages_inline_budget_tight():
    budget = {
        "users.html": 0,
        "audit.html": 0,
        "teams.html": 0,
        "usage.html": 0,
        "usage_daily.html": 0,
        "alerts.html": 0,
        "smtp.html": 0,
        "me.html": 0,
        "key_form.html": 0,
        "setup_done.html": 0,
        "setup_sources.html": 0,
        "setup_access.html": 0,
        "_pw_rules.html": 0,
    }
    over = []
    for name, max_n in budget.items():
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        n = len(re.findall(r'\sstyle="', text))
        if n > max_n:
            over.append(f"{name}: {n} > {max_n}")
    assert not over, "Inline style budget exceeded: " + "; ".join(over)


def test_users_grant_expand_clicks_are_not_intercepted():
    """Checkboxes in the grant expand row must not hit the badge click handler."""
    js = (ROOT / "app" / "web" / "static" / "users_grant.js").read_text(encoding="utf-8")
    assert 'closest(".grant-expand-row")' in js
    assert "a[data-grant-user][data-grant-source]" in js
    expand = (TEMPLATES / "_user_grant_expand.html").read_text(encoding="utf-8")
    assert 'name="models"' in expand
    assert 'type="checkbox"' in expand
    assert "data-grant-collapse" in expand
    users = (TEMPLATES / "users.html").read_text(encoding="utf-8")
    assert "data-grant-add-user" in users
    assert "grant/source/remove" in users
    assert "data-grant-disable-all" in expand
    assert "data-grant-enable-all" in expand
    assert "data-grant-disable-all" in js


def test_keys_edit_expands_inline_on_list():
    keys = (TEMPLATES / "keys.html").read_text(encoding="utf-8")
    assert 'class="keys-table"' in keys
    assert "data-key-edit" in keys
    assert 'href="/keys/{{ k.id }}"' not in keys
    assert "/static/keys_edit.js" in keys
    js = (ROOT / "app" / "web" / "static" / "keys_edit.js").read_text(encoding="utf-8")
    assert "/keys/" in js and "/partial" in js
    assert "initModelPickers" in js
    expand = (TEMPLATES / "_key_edit_expand.html").read_text(encoding="utf-8")
    assert '_source_chips.html' in expand
    assert "model_picker.html" in expand
    assert "data-key-collapse" in expand
    picker = (ROOT / "app" / "web" / "static" / "picker.js").read_text(encoding="utf-8")
    assert "window.initModelPickers" in picker


def test_usage_daily_team_column_gated_on_teams_enabled():
    text = (TEMPLATES / "usage_daily.html").read_text(encoding="utf-8")
    assert "{% if teams_enabled %}<th>Team</th>{% endif %}" in text
    assert "{% if teams_enabled %}<td>{{ r.team_name or '—' }}</td>{% endif %}" in text
    assert "<th>Team</th>" not in text.replace("{% if teams_enabled %}<th>Team</th>{% endif %}", "")

    text = (TEMPLATES / "setup_sources.html").read_text(encoding="utf-8")
    assert 'action="/setup/sources/{{ s.id }}/delete"' in text
    assert "btn-icon-del" in text
    assert 'form="setup-del-{{ s.id }}"' in text


def test_wizard_access_keeps_actions_visible_and_picker_grids():
    access = (TEMPLATES / "setup_access.html").read_text(encoding="utf-8")
    key = (TEMPLATES / "setup_key.html").read_text(encoding="utf-8")
    chips = (TEMPLATES / "_source_chips.html").read_text(encoding="utf-8")
    js = (ROOT / "app" / "web" / "static" / "picker.js").read_text(encoding="utf-8")
    css = STYLE.read_text(encoding="utf-8")
    assert "wizard-form--fill" in access and "wizard-grow" in access
    assert "wizard-actions--sticky" in access
    assert "wizard-actions--sticky" in key
    assert 'chips_for_picker=\'setup-access\'' in access
    assert 'chips_for_picker=\'setup-key\'' in key
    assert "auto_vl_routing" not in key
    assert "Everything starts checked" in key
    assert "data-source-chips-for" in chips
    assert "data-source-chip" in chips
    assert "linkSourceChips" in js
    assert "repeat(auto-fill, minmax(" in css
    assert ".wizard-actions--sticky" in css
    picker = (TEMPLATES / "partials" / "model_picker.html").read_text(encoding="utf-8")
    assert 'data-kind="{{ m.kind }}"' in picker
    assert "--card-chat-bg" in css
    assert ".model-picker-item[data-kind=\"chat\"]" in css
