"""Founder-led and personalized circle-of-competence research maps.

These lists prioritize diligence.  They never create a security recommendation,
portfolio allocation, order, or broker action.
"""

from __future__ import annotations

import html
import json
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CURRENT_ROOT, REPORT_ROOT, ROOT


REFERENCE_ROOT = Path(__file__).resolve().parent / "reference"
FOUNDER_SOURCE = REFERENCE_ROOT / "founder_ceos.json"
CIRCLE_SOURCE = REFERENCE_ROOT / "circle_of_competence.json"
FMP_CURRENT = CURRENT_ROOT / "fmp_latest.parquet"
BROAD_UNIVERSE = CURRENT_ROOT / "broad_universe_latest.parquet"
CANDIDATES = CURRENT_ROOT / "candidates_latest.parquet"
DEFAULT_OUTPUT = REPORT_ROOT / "company_maps.html"


def _esc(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return html.escape(str(value))


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower()


def _money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(number):
        return "—"
    if abs(number) >= 1_000_000_000_000:
        return f"${number / 1_000_000_000_000:.1f}T"
    if abs(number) >= 1_000_000_000:
        return f"${number / 1_000_000_000:.1f}B"
    if abs(number) >= 1_000_000:
        return f"${number / 1_000_000:.0f}M"
    return f"${number:,.0f}"


def load_company_map_sources(
    founder_path: Path = FOUNDER_SOURCE,
    circle_path: Path = CIRCLE_SOURCE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    founder = json.loads(founder_path.read_text(encoding="utf-8"))
    circle = json.loads(circle_path.read_text(encoding="utf-8"))
    validate_company_map_sources(founder, circle)
    return founder, circle


def validate_company_map_sources(founder: dict[str, Any], circle: dict[str, Any]) -> None:
    active = founder.get("active") or []
    removed = founder.get("recent_removals") or []
    companies = circle.get("companies") or []
    excluded = circle.get("excluded") or []
    if len(active) < 25:
        raise ValueError("founder-CEO roster is unexpectedly small")
    for label, rows in (("founder", active), ("circle", companies), ("excluded", excluded)):
        tickers = [str(row.get("ticker") or "").upper() for row in rows]
        if any(not ticker for ticker in tickers):
            raise ValueError(f"{label} rows require tickers")
        if len(tickers) != len(set(tickers)):
            raise ValueError(f"duplicate ticker in {label} source")
        for row in rows:
            url = str(row.get("source_url") or "")
            if not url.startswith("https://"):
                raise ValueError(f"{label} source requires https URL: {row.get('ticker')}")
    active_tickers = {str(row["ticker"]).upper() for row in active}
    removed_tickers = {str(row["ticker"]).upper() for row in removed}
    overlap = active_tickers & removed_tickers
    if overlap:
        raise ValueError(f"founder roster contains removed tickers: {sorted(overlap)}")
    circle_tickers = {str(row["ticker"]).upper() for row in companies}
    excluded_tickers = {str(row["ticker"]).upper() for row in excluded}
    overlap = circle_tickers & excluded_tickers
    if overlap:
        raise ValueError(f"personal map contains active and excluded tickers: {sorted(overlap)}")
    for row in companies:
        score = int(row.get("fit_score") or 0)
        if not 1 <= score <= 10:
            raise ValueError(f"circle fit score out of range: {row.get('ticker')}")


def _load_optional_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _latest_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "endpoint" not in frame.columns:
        return pd.DataFrame()
    profiles = frame[frame["endpoint"].eq("profile")].copy()
    if profiles.empty:
        return profiles
    profiles["ticker"] = profiles["ticker"].astype(str).str.upper()
    sort_cols = [col for col in ("snapshot_as_of", "fetched_at") if col in profiles.columns]
    if sort_cols:
        profiles = profiles.sort_values(sort_cols)
    return profiles.drop_duplicates("ticker", keep="last").set_index("ticker")


def _by_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "ticker" not in frame.columns:
        return pd.DataFrame()
    indexed = frame.copy()
    indexed["ticker"] = indexed["ticker"].astype(str).str.upper()
    return indexed.drop_duplicates("ticker", keep="last").set_index("ticker")


def prepare_founder_rows(
    source: dict[str, Any],
    profiles: pd.DataFrame,
    universe: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    as_of: str,
) -> list[dict[str, Any]]:
    profile_ix = _latest_profiles(profiles)
    universe_ix = _by_ticker(universe)
    candidate_ix = _by_ticker(candidates)
    rows: list[dict[str, Any]] = []
    cutoff = pd.Timestamp(as_of)
    for raw in source.get("active") or []:
        row = dict(raw)
        ticker = str(row["ticker"]).upper()
        current_ceo = None
        profile_as_of = None
        if not profile_ix.empty and ticker in profile_ix.index:
            profile = profile_ix.loc[ticker]
            current_ceo = profile.get("ceo")
            profile_as_of = profile.get("snapshot_as_of")
        tokens = [_normalize(token) for token in row.get("match_tokens") or []]
        normalized_ceo = _normalize(current_ceo)
        if current_ceo and any(token and token in normalized_ceo for token in tokens):
            verification = "Double-checked"
            verification_class = "verified"
        elif current_ceo:
            verification = "CEO mismatch — recheck"
            verification_class = "mismatch"
        else:
            verification = "Primary source"
            verification_class = "source-only"
        source_date = pd.to_datetime(row.get("source_date"), errors="coerce")
        if pd.notna(source_date) and (cutoff - source_date).days > 400:
            verification = "Source stale — recheck"
            verification_class = "mismatch"
        market = universe_ix.loc[ticker] if not universe_ix.empty and ticker in universe_ix.index else pd.Series(dtype=object)
        candidate = candidate_ix.loc[ticker] if not candidate_ix.empty and ticker in candidate_ix.index else pd.Series(dtype=object)
        row.update({
            "ticker": ticker,
            "current_ceo": current_ceo,
            "profile_as_of": profile_as_of,
            "verification": verification,
            "verification_class": verification_class,
            "company_name": market.get("company_name", row.get("company_name")),
            "sector": market.get("sector", candidate.get("sector")),
            "market_cap": market.get("market_cap", candidate.get("market_cap")),
            "research_eligible": market.get("research_eligible"),
            "research_lane": market.get("research_lane", candidate.get("research_lane")),
        })
        rows.append(row)
    return rows


def prepare_circle_rows(
    source: dict[str, Any],
    founder_tickers: set[str],
    universe: pd.DataFrame,
    candidates: pd.DataFrame,
) -> list[dict[str, Any]]:
    universe_ix = _by_ticker(universe)
    candidate_ix = _by_ticker(candidates)
    rows: list[dict[str, Any]] = []
    for raw in source.get("companies") or []:
        row = dict(raw)
        ticker = str(row["ticker"]).upper()
        market = universe_ix.loc[ticker] if not universe_ix.empty and ticker in universe_ix.index else pd.Series(dtype=object)
        candidate = candidate_ix.loc[ticker] if not candidate_ix.empty and ticker in candidate_ix.index else pd.Series(dtype=object)
        row.update({
            "ticker": ticker,
            "company_name": market.get("company_name", row.get("company_name")),
            "sector": market.get("sector", candidate.get("sector")),
            "market_cap": market.get("market_cap", candidate.get("market_cap")),
            "trend_state": market.get("trend_state", candidate.get("trend_state")),
            "research_priority": candidate.get("research_priority"),
            "founder_led": ticker in founder_tickers,
        })
        rows.append(row)
    return sorted(
        rows,
        key=lambda item: (
            0 if item.get("starting_point_rank") else 1,
            int(item.get("starting_point_rank") or 999),
            -int(item["fit_score"]),
            item["ticker"],
        ),
    )


def build_company_maps_report(
    *,
    as_of: str,
    output_path: Path = DEFAULT_OUTPUT,
    fmp_frame: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    candidates: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    founder_source, circle_source = load_company_map_sources()
    fmp_frame = _load_optional_parquet(FMP_CURRENT) if fmp_frame is None else fmp_frame
    universe = _load_optional_parquet(BROAD_UNIVERSE) if universe is None else universe
    candidates = _load_optional_parquet(CANDIDATES) if candidates is None else candidates
    founder_rows = prepare_founder_rows(
        founder_source, fmp_frame, universe, candidates, as_of=as_of
    )
    founder_active = [row for row in founder_rows if row["verification_class"] != "mismatch"]
    mismatches = [row for row in founder_rows if row["verification_class"] == "mismatch"]
    founder_tickers = {row["ticker"] for row in founder_active}
    circle_rows = prepare_circle_rows(circle_source, founder_tickers, universe, candidates)
    excluded = circle_source.get("excluded") or []
    overlap = sorted(row["ticker"] for row in circle_rows if row["founder_led"])

    direct_cards = "".join(
        f"""<article class="fit-card"><div class="card-top"><a href="{_esc(row['source_url'])}">{_esc(row['ticker'])}</a><span>{_esc(row['fit_score'])}/10</span></div>
        <h3>{_esc(row['company_name'])}</h3><p>{_esc(row['why_fit'])}</p><small>Hard part: {_esc(row['hard_part'])}</small></article>"""
        for row in circle_rows[:4]
    )
    circle_table = "".join(
        f"""<tr><td><a href="{_esc(row['source_url'])}"><b>{_esc(row['ticker'])}</b></a><small>{_esc(row['company_name'])}</small></td>
        <td><span class="score">{_esc(row['fit_score'])}/10</span><small>{_esc(row['tier'])}</small></td>
        <td>{_esc(row['why_fit'])}<small>{_esc(row['basis'])}</small></td>
        <td>{_esc(row['hard_part'])}</td><td>{_esc(row['first_kpis'])}</td>
        <td>{'<span class="founder">Founder-led</span>' if row['founder_led'] else '—'}</td></tr>"""
        for row in circle_rows
    )
    excluded_table = "".join(
        f"""<tr><td><a href="{_esc(row['source_url'])}"><b>{_esc(row['ticker'])}</b></a><small>{_esc(row['company_name'])}</small></td>
        <td>{_esc(row['familiarity'])}</td><td>{_esc(row['reason'])}</td><td>{_esc(row['what_would_change_mind'])}</td></tr>"""
        for row in excluded
    )
    founder_table = "".join(
        f"""<tr data-search="{_esc(' '.join([row['ticker'], str(row.get('company_name') or ''), str(row.get('founder_ceo') or ''), str(row.get('sector') or '')]).lower())}">
        <td><a href="{_esc(row['source_url'])}"><b>{_esc(row['ticker'])}</b></a><small>{_esc(row.get('company_name'))}</small></td>
        <td>{_esc(row.get('founder_ceo'))}<small>{_esc(row.get('founder_role'))}</small></td>
        <td><span class="check {_esc(row['verification_class'])}">{_esc(row['verification'])}</span><small>{_esc(row.get('current_ceo') or 'No local profile; primary source controls')}</small></td>
        <td>{_esc(row.get('sector'))}<small>{_money(row.get('market_cap'))}</small></td>
        <td>{_esc(row.get('source_date'))}</td><td>{'<span class="overlap">In your circle</span>' if row['ticker'] in overlap else '—'}</td></tr>"""
        for row in founder_active
    )
    mismatch_rows = "".join(
        f"<li><b>{_esc(row['ticker'])}</b> — {_esc(row['verification'])}; source record is held out until reverified.</li>"
        for row in mismatches
    ) or "<li>None. Every seeded founder record passed the current checks available locally.</li>"
    removed_rows = "".join(
        f"""<tr><td><a href="{_esc(row['source_url'])}"><b>{_esc(row['ticker'])}</b></a><small>{_esc(row['company_name'])}</small></td>
        <td>{_esc(row['founder'])}</td><td>{_esc(row['reason'])}</td><td>{_esc(row['effective_date'])}</td></tr>"""
        for row in founder_source.get("recent_removals") or []
    )
    source_only = sum(row["verification_class"] == "source-only" for row in founder_active)
    double_checked = sum(row["verification_class"] == "verified" for row in founder_active)
    top_circle = sum(int(row["fit_score"]) >= 8 for row in circle_rows)
    report_meta = {
        "as_of": as_of,
        "founder_active": len(founder_active),
        "founder_double_checked": double_checked,
        "founder_primary_source_only": source_only,
        "founder_held_for_recheck": len(mismatches),
        "circle_count": len(circle_rows),
        "circle_excluded_count": len(excluded),
        "circle_score_8_plus": top_circle,
        "founder_circle_overlap": overlap,
        "live_actions_enabled": False,
    }
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Founder-Led & Circle-of-Competence Company Maps — {_esc(as_of)}</title>
<style>
:root{{--bg:#09101d;--panel:#121c2f;--panel2:#182640;--text:#eef3fb;--muted:#9caac0;--line:#2b3c58;--blue:#79baff;--green:#72dfad;--amber:#ffd071;--red:#ff9e9e}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(180deg,#08101d,#0e182b 50%,#09101d);color:var(--text);font:15px/1.5 Inter,Segoe UI,Arial,sans-serif}}
main{{max-width:1240px;margin:auto;padding:38px 28px 64px}}a{{color:var(--blue);text-decoration:none}}a:hover{{text-decoration:underline}}.eyebrow{{font-size:11px;letter-spacing:.8px;text-transform:uppercase;color:var(--muted)}}
h1{{font-size:42px;line-height:1.06;letter-spacing:-1px;margin:7px 0 12px;max-width:900px}}h2{{font-size:25px;margin:40px 0 10px}}h3{{margin:5px 0 8px}}p{{margin:7px 0}}small{{display:block;color:var(--muted);margin-top:3px}}
.warning{{padding:11px 14px;border-radius:9px;background:#302817;border:1px solid #8b6b2c;color:#ffdc8a;font-size:13px}}.answer{{background:linear-gradient(135deg,#162640,#101a2d);border:1px solid #3a5577;border-radius:14px;padding:21px 23px;margin:20px 0}}
.answer p{{font-size:17px;max-width:950px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}}.stat{{background:var(--panel2);border-radius:10px;padding:13px}}.stat b{{display:block;font-size:24px}}.stat span{{color:var(--muted);font-size:12px}}
.fit-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.fit-card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}.card-top{{display:flex;justify-content:space-between;font-weight:850}}.card-top a{{font-size:21px}}.card-top span,.score{{color:var(--green);font-weight:850}}
.panel{{background:rgba(18,28,47,.93);border:1px solid var(--line);border-radius:12px;padding:10px 16px;overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:980px}}th{{padding:10px 9px;text-align:left;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid var(--line)}}td{{padding:11px 9px;border-bottom:1px solid #22314a;vertical-align:top}}td:first-child{{min-width:130px}}
.founder,.overlap,.check{{display:inline-block;border-radius:999px;padding:3px 7px;font-size:10px;font-weight:850;white-space:nowrap}}.founder,.overlap,.verified{{background:#163a31;color:#79e5b7}}.source-only{{background:#263650;color:#b9d7ff}}.mismatch{{background:#49292d;color:#ffb1b1}}
.nav{{display:flex;gap:9px;flex-wrap:wrap;margin:18px 0}}.nav a{{background:var(--panel2);border:1px solid var(--line);padding:8px 11px;border-radius:8px;font-weight:750}}.search{{width:100%;max-width:420px;background:#0d1728;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:8px 0 14px}}
details{{margin-top:24px;background:rgba(18,28,47,.72);border:1px solid var(--line);border-radius:12px;padding:14px 17px}}summary{{cursor:pointer;font-weight:800;color:#d5e2f2}}footer{{color:var(--muted);font-size:12px;margin-top:30px}}
@media(max-width:900px){{.fit-grid,.stats{{grid-template-columns:1fr 1fr}}h1{{font-size:35px}}}}@media(max-width:600px){{main{{padding:25px 15px}}.fit-grid,.stats{{grid-template-columns:1fr}}h1{{font-size:31px}}}}
</style></head><body><main>
<div class="eyebrow">Fundamental research offshoot · {_esc(as_of)}</div><h1>Founder-led companies and your circle of competence</h1>
<div class="warning">Research universes only. Inclusion says nothing about valuation, expected return, position size, or whether a stock should be owned.</div>
<section class="answer"><div class="eyebrow">The useful conclusion</div><p>Your circle should start with <b>consumer products whose behavior explains the business</b>—not simply products you happen to touch. The strongest current founder-led overlap is <b>{_esc(', '.join(overlap))}</b>. These names are easier places to begin research, not investment theses.</p></section>
<nav class="nav"><a href="#circle">Product circle</a><a href="#excluded">Familiar but opaque</a><a href="#founders">Founder-CEO roster</a><a href="#changes">Recent removals</a><a href="fundamental_daily.html">Daily fundamental brief</a></nav>
<section class="stats"><div class="stat"><b>{len(founder_active)}</b><span>current founder-CEO records</span></div><div class="stat"><b>{len(circle_rows)}</b><span>consumer-product candidates</span></div><div class="stat"><b>{len(excluded)}</b><span>familiar but economically opaque</span></div><div class="stat"><b>{len(overlap)}</b><span>founder + product-circle overlap</span></div></section>

<section id="circle"><h2>Best product-led starting points</h2><p class="muted">These are the first businesses to test against your actual customer experience. No direct use is assumed unless you have said so. Understandability is separate from price and valuation.</p><div class="fit-grid">{direct_cards}</div>
<h2>Full consumer-product map</h2><div class="panel"><table><thead><tr><th>Company</th><th>Fit</th><th>Why product observation helps</th><th>What could fool you</th><th>First operating KPIs</th><th>Founder overlap</th></tr></thead><tbody>{circle_table}</tbody></table></div></section>

<section id="excluded"><h2>Familiar product, still an opaque business</h2><p>These names fail the new test. Touching the product does not yet give you a reliable view of the variables that drive revenue, margins and cash flow.</p><div class="panel"><table><thead><tr><th>Company</th><th>Why it feels familiar</th><th>Why it does not qualify</th><th>What would make it understandable</th></tr></thead><tbody>{excluded_table}</tbody></table></div></section>

<section id="founders"><h2>Current founder-CEO roster</h2><p>This is strict: founder-chair-only companies are excluded. “Double-checked” means the primary filing and current local CEO snapshot agree; “Primary source” means the filing or official leadership page controls because the company is outside the current local baseline.</p>
<input id="founder-search" class="search" type="search" placeholder="Filter by ticker, company, founder or sector…" aria-label="Filter founder roster">
<div class="panel"><table id="founder-table"><thead><tr><th>Company</th><th>Founder CEO</th><th>Current check</th><th>Sector / size</th><th>Source date</th><th>Personal overlap</th></tr></thead><tbody>{founder_table}</tbody></table></div>
<details><summary>Held out for recheck ({len(mismatches)})</summary><ul>{mismatch_rows}</ul></details></section>

<section id="changes"><h2>Recent removals and common false positives</h2><p>A running list is only useful if names leave when the founder stops being CEO.</p><div class="panel"><table><thead><tr><th>Company</th><th>Founder</th><th>Why excluded</th><th>Effective</th></tr></thead><tbody>{removed_rows}</tbody></table></div></section>

<details><summary>Method and evidence posture</summary><p><b>Founder roster:</b> current proxy/annual filing or official leadership page, reviewed through {_esc(as_of)}. It is research-grade but curated, not globally exhaustive. A CEO-name mismatch automatically holds a seeded record out for recheck.</p><p><b>Personal map:</b> PM judgment centered on consumer products and observable behavior. A company qualifies only when the customer experience can help explain demand, frequency, pricing, retention, throughput, unit economics or brand strength.</p><p><b>Important limit:</b> the list does not assume you personally use every product. Large platforms, enterprise tools, exchanges and financial infrastructure are excluded when familiarity does not reveal the causal earnings engine.</p></details>
<footer>Founder leadership is a governance and incentive attribute, not a quality factor by itself. Live actions are disabled.</footer>
<script>const q=document.getElementById('founder-search');q.addEventListener('input',()=>{{const s=q.value.toLowerCase().trim();document.querySelectorAll('#founder-table tbody tr').forEach(r=>r.hidden=s&&!r.dataset.search.includes(s));}});</script>
<script type="application/json" id="report-meta">{html.escape(json.dumps(report_meta))}</script>
</main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    support_path = CURRENT_ROOT / "company_maps_latest.json"
    support_path.parent.mkdir(parents=True, exist_ok=True)
    support_path.write_text(json.dumps({
        "meta": report_meta,
        "founder_definition": founder_source.get("definition"),
        "founder_rows": founder_rows,
        "recent_removals": founder_source.get("recent_removals") or [],
        "circle_method": circle_source.get("method"),
        "circle_rows": circle_rows,
        "circle_excluded": excluded,
        "live_actions_enabled": False,
    }, indent=2, default=str), encoding="utf-8")
    return output_path, support_path
