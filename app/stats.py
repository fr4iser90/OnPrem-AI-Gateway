"""Usage aggregations and readable SVG charts for dashboards."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from html import escape
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from .data.models import UsageEvent, utcnow


def display_zone(name: str | None = None) -> ZoneInfo:
    """Resolve an IANA timezone name (invalid/empty → UTC)."""
    raw = (name or "").strip()
    if not raw:
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(raw)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def is_valid_timezone(name: str) -> bool:
    raw = (name or "").strip()
    if not raw:
        return False
    try:
        ZoneInfo(raw)
        return True
    except ZoneInfoNotFoundError:
        return False


def zone_from_request(request=None, user=None) -> ZoneInfo:
    """Browser cookie gw_tz (auto-detected) → last saved user.timezone → UTC.

    No manual settings. Every logged-in UI user gets their browser zone.
    """
    raw = ""
    if request is not None:
        try:
            raw = (request.cookies.get("gw_tz") or "").strip()
        except Exception:
            raw = ""
    if not is_valid_timezone(raw):
        raw = ""
    if not raw and user is not None:
        raw = (getattr(user, "timezone", None) or "").strip()
    if not is_valid_timezone(raw):
        raw = ""
    return display_zone(raw or "UTC")


# Back-compat alias
def zone_for_user(user=None, *, fallback: str | None = None) -> ZoneInfo:
    if fallback and is_valid_timezone(fallback):
        if user is None or not (getattr(user, "timezone", None) or "").strip():
            return display_zone(fallback)
    return zone_from_request(None, user)

def week_window_start(tz: ZoneInfo | None = None):
    """Start of the oldest day in the 7-day chart (local midnight in display TZ)."""
    zone = tz or display_zone()
    today = utcnow().astimezone(zone).date()
    start = today - timedelta(days=6)
    return datetime(start.year, start.month, start.day, tzinfo=zone)


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _fmt_int(n: int | float) -> str:
    return f"{int(n):,}"


def usage_stats(
    db: Session,
    *,
    since=None,
    key_ids: list[int] | None = None,
    team_ids: set[int] | None = None,
    tz: ZoneInfo | None = None,
) -> dict:
    zone = tz or display_zone()
    since = since or (utcnow() - timedelta(days=1))
    q = db.query(UsageEvent).filter(UsageEvent.created_at >= since)
    if key_ids is not None:
        q = q.filter(UsageEvent.api_key_id.in_(key_ids)) if key_ids else q.filter(False)
    if team_ids is not None:
        q = q.filter(UsageEvent.team_id.in_(team_ids)) if team_ids else q.filter(False)

    events = q.all()
    ok = [e for e in events if e.result == "ok"]
    denies = sum(1 for e in events if e.result == "deny")
    rates = sum(1 for e in events if e.result == "rate_limit")

    by_service: dict[str, int] = {}
    by_model: dict[str, int] = {}
    top_keys: dict[str, int] = {}
    tokens_in = tokens_out = 0
    audio_seconds = 0.0
    response_chars = 0
    watt_hours = 0.0
    pool_cost = 0.0
    latencies: list[float] = []
    watts_samples: list[float] = []

    # Complete 7 calendar days ending today in display timezone
    today = utcnow().astimezone(zone).date()
    daily_map: dict[str, dict[str, float | int]] = {}
    day_labels: list[tuple[str, str]] = []  # (key, display)
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        key = d.isoformat()
        day_labels.append((key, d.strftime("%a %d")))
        daily_map[key] = {"ok": 0, "deny": 0, "rate_limit": 0, "wh": 0.0}

    for e in events:
        if e.result == "ok":
            by_service[e.service] = by_service.get(e.service, 0) + 1
            if e.model:
                by_model[e.model] = by_model.get(e.model, 0) + 1
            top_keys[e.key_label or "(unknown)"] = top_keys.get(e.key_label or "(unknown)", 0) + 1
            tokens_in += e.tokens_in or 0
            tokens_out += e.tokens_out or 0
            audio_seconds += e.audio_seconds or 0.0
            response_chars += e.response_chars or 0
            watt_hours += float(getattr(e, "watt_hours", None) or 0)
            pool_cost += float(getattr(e, "pool_cost", None) or 0)
            if getattr(e, "watts", None) is not None:
                watts_samples.append(float(e.watts))
        if e.duration_ms is not None:
            latencies.append(float(e.duration_ms))
        if e.created_at:
            created = e.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            dkey = created.astimezone(zone).date().isoformat()
            if dkey in daily_map:
                bucket = e.result if e.result in ("ok", "deny", "rate_limit") else "deny"
                if bucket == "rate_limit":
                    daily_map[dkey]["rate_limit"] = int(daily_map[dkey]["rate_limit"]) + 1
                elif bucket == "ok":
                    daily_map[dkey]["ok"] = int(daily_map[dkey]["ok"]) + 1
                    daily_map[dkey]["wh"] = float(daily_map[dkey]["wh"]) + float(
                        getattr(e, "watt_hours", None) or 0
                    )
                else:
                    daily_map[dkey]["deny"] = int(daily_map[dkey]["deny"]) + 1

    latencies.sort()
    daily_series = [
        {
            "day": key,
            "label": label,
            "ok": int(daily_map[key]["ok"]),
            "deny": int(daily_map[key]["deny"]),
            "rate_limit": int(daily_map[key]["rate_limit"]),
            "watt_hours": round(float(daily_map[key]["wh"]), 4),
            "total": int(daily_map[key]["ok"])
            + int(daily_map[key]["deny"])
            + int(daily_map[key]["rate_limit"]),
        }
        for key, label in day_labels
    ]

    return {
        "total_ok": len(ok),
        "denies": denies,
        "rate_limits": rates,
        "demo_count": sum(1 for e in events if getattr(e, "is_demo", False)),
        "by_service": sorted(by_service.items(), key=lambda x: -x[1]),
        "by_model": sorted(by_model.items(), key=lambda x: -x[1])[:12],
        "top_keys": sorted(top_keys.items(), key=lambda x: -x[1])[:10],
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "audio_seconds": round(audio_seconds, 1),
        "response_chars": response_chars,
        "watt_hours": round(watt_hours, 4),
        "pool_cost": round(pool_cost, 2),
        "watts_avg": (sum(watts_samples) / len(watts_samples)) if watts_samples else None,
        "latency_p50": _percentile(latencies, 0.5),
        "latency_p95": _percentile(latencies, 0.95),
        "latency_avg": (sum(latencies) / len(latencies)) if latencies else None,
        "daily_series": daily_series,
        "event_count": len(events),
        "timezone": str(zone),
        "energy_by_key": energy_by_key(events),
    }


def energy_by_key(events: list) -> list[dict]:
    """Approx Wh rollup per API key id (not label — many keys share 'main')."""
    buckets: dict[int | str, dict] = {}
    for e in events:
        if getattr(e, "result", None) != "ok":
            continue
        wh = float(getattr(e, "watt_hours", None) or 0)
        w = getattr(e, "watts", None)
        if wh <= 0 and w is None:
            continue
        kid = getattr(e, "api_key_id", None)
        bucket_key: int | str = int(kid) if kid is not None else (
            (getattr(e, "key_label", None) or "").strip() or "(unknown)"
        )
        label = (getattr(e, "key_label", None) or "").strip() or "(unknown)"
        b = buckets.setdefault(
            bucket_key,
            {
                "key_label": label,
                "api_key_id": int(kid) if kid is not None else None,
                "ok_count": 0,
                "watt_hours": 0.0,
                "watts": [],
            },
        )
        b["ok_count"] += 1
        b["watt_hours"] += wh
        if label and b["key_label"] == "(unknown)":
            b["key_label"] = label
        if w is not None and float(w) > 0:
            b["watts"].append(float(w))
    out: list[dict] = []
    for b in buckets.values():
        samples = b.pop("watts")
        out.append(
            {
                "key_label": b["key_label"],
                "api_key_id": b["api_key_id"],
                "owner_user_id": None,
                "owner_name": "",
                "ok_count": b["ok_count"],
                "watt_hours": round(float(b["watt_hours"]), 4),
                "watts_avg": (sum(samples) / len(samples)) if samples else None,
            }
        )
    out.sort(
        key=lambda r: (
            -r["watt_hours"],
            -r["ok_count"],
            r["owner_name"],
            r["key_label"],
        )
    )
    return out


def enrich_energy_with_owners(db: Session, rows: list[dict]) -> list[dict]:
    """Attach owner display names from ApiKey.owner_user_id."""
    from .data.models import ApiKey

    ids = [r["api_key_id"] for r in rows if r.get("api_key_id") is not None]
    if not ids:
        return rows
    keys = (
        db.query(ApiKey)
        .options(joinedload(ApiKey.owner))
        .filter(ApiKey.id.in_(ids))
        .all()
    )
    meta: dict[int, tuple[int | None, str]] = {}
    for k in keys:
        owner = k.owner
        oid = k.owner_user_id
        if owner is None:
            name = f"user#{oid}" if oid else "—"
        else:
            uname = (owner.username or "").strip()
            if uname.startswith("pending-") or not uname:
                name = (owner.email or uname or f"user-{owner.id}").strip()
            else:
                name = uname
        meta[k.id] = (oid, name)
    for r in rows:
        kid = r.get("api_key_id")
        if kid is None or kid not in meta:
            r["owner_user_id"] = r.get("owner_user_id")
            r["owner_name"] = r.get("owner_name") or "—"
            continue
        oid, name = meta[kid]
        r["owner_user_id"] = oid
        r["owner_name"] = name
    rows.sort(
        key=lambda r: (
            -r["watt_hours"],
            -r["ok_count"],
            r.get("owner_name") or "",
            r.get("key_label") or "",
        )
    )
    return rows


def energy_by_owner(key_rows: list[dict]) -> list[dict]:
    """Roll key-level Wh up to owner (admin overview)."""
    buckets: dict[str, dict] = {}
    for r in key_rows:
        oid = r.get("owner_user_id")
        name = (r.get("owner_name") or "").strip() or "—"
        key = str(oid) if oid is not None else f"name:{name}"
        b = buckets.setdefault(
            key,
            {
                "owner_user_id": oid,
                "owner_name": name,
                "ok_count": 0,
                "watt_hours": 0.0,
                "watts": [],
                "key_count": 0,
            },
        )
        b["ok_count"] += int(r.get("ok_count") or 0)
        b["watt_hours"] += float(r.get("watt_hours") or 0)
        b["key_count"] += 1
        if r.get("watts_avg") is not None:
            b["watts"].append(float(r["watts_avg"]))
    out: list[dict] = []
    for b in buckets.values():
        samples = b.pop("watts")
        out.append(
            {
                "owner_user_id": b["owner_user_id"],
                "owner_name": b["owner_name"],
                # key_count — not "keys" (Jinja dict.keys method collision)
                "key_count": b["key_count"],
                "ok_count": b["ok_count"],
                "watt_hours": round(float(b["watt_hours"]), 4),
                "watts_avg": (sum(samples) / len(samples)) if samples else None,
            }
        )
    out.sort(key=lambda r: (-r["watt_hours"], -r["ok_count"], r["owner_name"]))
    return out


def energy_wh_by_key_ids(
    db: Session,
    key_ids: list[int],
    *,
    lookback_days: int = 7,
) -> dict[int, float]:
    """Map api_key_id → approx Wh over lookback (for Keys table)."""
    if not key_ids:
        return {}
    since = utcnow() - timedelta(days=max(1, int(lookback_days)))
    rows = (
        db.query(UsageEvent.api_key_id, func.sum(UsageEvent.watt_hours))
        .filter(
            UsageEvent.api_key_id.in_(key_ids),
            UsageEvent.created_at >= since,
            UsageEvent.result == "ok",
            UsageEvent.watt_hours.isnot(None),
            UsageEvent.watt_hours > 0,
        )
        .group_by(UsageEvent.api_key_id)
        .all()
    )
    return {int(kid): round(float(wh or 0), 4) for kid, wh in rows if kid is not None}


def model_perf_averages(
    db: Session,
    *,
    key_ids: list[int] | None = None,
    lookback_days: int = 7,
    min_samples: int = 1,
) -> list[dict]:
    """Per-model averages from OK usage events (W, Wh, PP, TG, latency).

    ``key_ids=None`` → fleet (all keys). ``key_ids=[]`` → empty. Otherwise filter.
    """
    since = utcnow() - timedelta(days=max(1, int(lookback_days)))
    q = db.query(UsageEvent).filter(
        UsageEvent.created_at >= since,
        UsageEvent.result == "ok",
        UsageEvent.model.isnot(None),
        UsageEvent.model != "",
    )
    if key_ids is not None:
        if not key_ids:
            return []
        q = q.filter(UsageEvent.api_key_id.in_(key_ids))

    buckets: dict[str, dict[str, list[float]]] = {}
    counts: dict[str, int] = {}
    for e in q.all():
        model = (e.model or "").strip()
        if not model:
            continue
        counts[model] = counts.get(model, 0) + 1
        b = buckets.setdefault(
            model,
            {
                "watts": [],
                "watt_hours": [],
                "pp_tok_s": [],
                "tg_tok_s": [],
                "duration_ms": [],
            },
        )
        if e.watts is not None:
            b["watts"].append(float(e.watts))
        if e.watt_hours is not None and float(e.watt_hours) > 0:
            b["watt_hours"].append(float(e.watt_hours))
        if getattr(e, "pp_tok_s", None) is not None:
            b["pp_tok_s"].append(float(e.pp_tok_s))
        if getattr(e, "tg_tok_s", None) is not None:
            b["tg_tok_s"].append(float(e.tg_tok_s))
        if e.duration_ms is not None:
            b["duration_ms"].append(float(e.duration_ms))

    def _avg(vals: list[float]) -> float | None:
        return round(sum(vals) / len(vals), 3) if vals else None

    rows: list[dict] = []
    for model, n in counts.items():
        if n < min_samples:
            continue
        b = buckets[model]
        rows.append(
            {
                "model": model,
                "n": n,
                "watts_avg": _avg(b["watts"]),
                "watt_hours_avg": _avg(b["watt_hours"]),
                "pp_tok_s_avg": _avg(b["pp_tok_s"]),
                "tg_tok_s_avg": _avg(b["tg_tok_s"]),
                "duration_ms_avg": _avg(b["duration_ms"]),
            }
        )
    rows.sort(key=lambda r: (-r["n"], r["model"]))
    return rows


def model_perf_by_id(
    rows: list[dict],
) -> dict[str, dict]:
    """Index averages by model_id for catalog templates."""
    return {str(r["model"]): r for r in rows}


def bar_chart_svg(
    rows: list[tuple[str, float | int]],
    *,
    width: int = 560,
    unit: str = "requests",
) -> str:
    """Horizontal bar chart with axis max and value labels."""
    if not rows:
        return '<div class="chart-empty">No data in this window.</div>'
    max_v = max(float(v) for _, v in rows) or 1.0
    row_h = 28
    left = 132
    right = 56
    top = 8
    plot_w = width - left - right
    height = top + row_h * len(rows) + 28
    lines = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Bar chart of {escape(unit)}">'
        f'<text x="{left}" y="14" class="chart-axis-title">{escape(unit)} · max {_fmt_int(max_v)}</text>'
    ]
    for i, (label, val) in enumerate(rows):
        y = top + 18 + i * row_h
        w = max(4, int((float(val) / max_v) * plot_w))
        safe = escape(str(label)[:26])
        lines.append(
            f'<text x="{left - 8}" y="{y + 12}" text-anchor="end" class="chart-label">{safe}</text>'
            f'<rect x="{left}" y="{y}" width="{plot_w}" height="16" rx="3" class="chart-track"/>'
            f'<rect x="{left}" y="{y}" width="{w}" height="16" rx="3" class="chart-bar"/>'
            f'<text x="{left + w + 6}" y="{y + 12}" class="chart-val">{_fmt_int(val)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def pulse_stats(
    db: Session,
    *,
    key_ids: list[int] | None = None,
    minutes: int = 60,
    buckets: int = 24,
) -> dict:
    """Last-hour throughput for the Overview pulse (honest counts, not fake RPM)."""
    now = utcnow()
    since = now - timedelta(minutes=minutes)
    q = db.query(UsageEvent).filter(UsageEvent.created_at >= since)
    if key_ids is not None:
        q = q.filter(UsageEvent.api_key_id.in_(key_ids)) if key_ids else q.filter(False)
    events = q.all()
    width_s = (minutes * 60) / max(buckets, 1)
    counts = [0] * buckets
    latencies: list[float] = []
    live_cut = now - timedelta(minutes=5)
    live = False
    for e in events:
        created = e.created_at
        if created is None:
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        delta = (created - since).total_seconds()
        i = int(delta / width_s) if width_s else 0
        i = max(0, min(buckets - 1, i))
        counts[i] += 1
        if e.duration_ms is not None:
            latencies.append(float(e.duration_ms))
        if created >= live_cut:
            live = True
    latencies.sort()
    return {
        "count": len(events),
        "live": live,
        "latency_p95": _percentile(latencies, 0.95) if latencies else None,
        "series": [{"label": "", "total": n} for n in counts],
    }


def area_chart_svg(
    series: list[dict],
    *,
    width: int = 720,
    fill_id: str = "gw-area",
    aria: str = "Throughput",
) -> str:
    """Thin cyan area-line — no axis chrome, no point labels."""
    if not series:
        series = [{"total": 0}]
    n = len(series)
    left, right, top, bottom = 2, 2, 8, 4
    plot_h = 108
    plot_w = width - left - right
    height = top + plot_h + bottom
    raw_max = max(int(s.get("total") or 0) for s in series)
    max_v = max(raw_max, 1)
    xs = [left + plot_w / 2] if n == 1 else [left + i * plot_w / (n - 1) for i in range(n)]
    base = top + plot_h
    ys = [base - (int(s.get("total") or 0) / max_v) * plot_h for s in series]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    fill = f"{xs[0]:.1f},{base:.1f} {line} {xs[-1]:.1f},{base:.1f}"
    mid = top + plot_h * 0.5
    return (
        f'<svg class="chart chart-area" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(aria)}">'
        f'<defs><linearGradient id="{escape(fill_id)}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" class="chart-area-stop-a"/>'
        f'<stop offset="100%" class="chart-area-stop-b"/>'
        f"</linearGradient></defs>"
        f'<line x1="{left}" y1="{mid:.1f}" x2="{left + plot_w}" y2="{mid:.1f}" class="chart-rule-faint"/>'
        f'<polygon points="{fill}" fill="url(#{escape(fill_id)})"/>'
        f'<polyline points="{line}" class="chart-area-line"/>'
        f"</svg>"
    )


def _nice_ceiling(v: int) -> int:
    """Round an axis max so ticks land on even steps (5 → 6, 53 → 60)."""
    v = max(1, int(v))
    if v <= 4:
        return v
    if v <= 6:
        return 6
    if v <= 8:
        return 8
    if v <= 10:
        return 10
    if v <= 50:
        step = 5
    elif v <= 100:
        step = 10
    elif v <= 250:
        step = 25
    else:
        step = 50
    return ((v + step - 1) // step) * step


def _axis_ticks(max_v: int) -> list[int]:
    if max_v <= 4:
        return list(range(0, max_v + 1))
    for n in (4, 3, 5, 2):
        if max_v % n == 0:
            step = max_v // n
            return [i * step for i in range(n + 1)]
    return [0, max_v // 2, max_v]


def daily_traffic_chart_svg(
    series: list[dict], *, width: int = 720, tz_label: str = "UTC"
) -> str:
    """7-day throughput as a thin area-line (not stacked bars)."""
    if not series:
        return '<div class="chart-empty">No data in this window.</div>'

    total_all = sum(int(s.get("total") or 0) for s in series)
    if total_all <= 0:
        tz_safe = escape(tz_label or "UTC")
        return (
            '<div class="chart-empty chart-empty--soft">'
            f"<strong>No traffic yet</strong>"
            f"<span>Last 7 days ({tz_safe}) — the line appears after the first request.</span>"
            "</div>"
        )

    svg = area_chart_svg(series, fill_id="gw-day", aria="Requests per day, last 7 days")
    rows_html = "".join(
        f"<tr><td>{escape(s['label'])}</td>"
        f"<td class='num'>{s['ok']}</td>"
        f"<td class='num'>{s['deny']}</td>"
        f"<td class='num'>{s['rate_limit']}</td>"
        f"<td class='num'><strong>{s['total']}</strong></td></tr>"
        for s in series
    )
    table = (
        '<details class="chart-details">'
        "<summary>Exact daily counts</summary>"
        '<table class="chart-table"><thead><tr>'
        "<th>Day</th><th>OK</th><th>Deny</th><th>429</th><th>Total</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table></details>"
    )
    return f'<div class="chart-frame">{svg}{table}</div>'
