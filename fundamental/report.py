"""A concise PM-facing brief backed by the full fundamental research funnel."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import POLICY_VERSION


def _esc(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return html.escape(str(value))


def _pct(value: Any) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _num(value: Any, digits: int = 1) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _priority_class(priority: str) -> str:
    if priority.startswith("A"):
        return "look"
    if priority.startswith("B"):
        return "watch"
    if priority.startswith("C"):
        return "background"
    return "pass"


def _short_status(priority: str) -> str:
    if priority.startswith("A"):
        return "QUICK REVIEW"
    if priority.startswith("B"):
        return "KEEP DIGGING"
    if priority.startswith("C"):
        return "BACKGROUND"
    return "PASS"


def _next_check(row: pd.Series) -> str:
    if pd.notna(row.get("share_count_cagr_3y")) and float(row["share_count_cagr_3y"]) > 0.03:
        return "Separate operating growth from acquisition or commodity effects and test whether per-share value grows after dilution."
    if pd.isna(row.get("roic")):
        return "Normalize the capital base and verify whether the cash economics are durable."
    if pd.notna(row.get("fcf_yield")) and float(row["fcf_yield"]) < 0.025:
        return "Build the reverse DCF and determine how much growth the current price already requires."
    if (
        pd.notna(row.get("fcf_yield"))
        and float(row["fcf_yield"]) > 0.15
        and str(row.get("trend_state")) == "RED"
    ):
        return "Verify normalized cash flow and the estimate path; a high trailing yield is not enough while the price trend deteriorates."
    if str(row.get("trend_state")) != "GREEN":
        return "Test the next earnings and estimate path before treating mixed trend as an opportunity."
    return "Tie the operating drivers to consensus and look for a specific expectations gap."


def _survival_read(row: pd.Series) -> str:
    pieces = []
    if pd.notna(row.get("research_score")):
        pieces.append(f"score {_num(row.get('research_score'))}")
    if pd.notna(row.get("fcf_yield")):
        pieces.append(f"{_pct(row.get('fcf_yield'))} FCF yield")
    if pd.notna(row.get("latest_revenue_growth")):
        pieces.append(f"{_pct(row.get('latest_revenue_growth'))} latest revenue growth")
    pieces.append(f"{str(row.get('trend_state') or 'unknown').lower()} trend")
    return ", ".join(pieces).capitalize() + "."


def render_candidate_report(
    candidates: pd.DataFrame,
    health: dict,
    output_path: Path,
    tearsheet_links: dict[str, str] | None = None,
    underwrite_decisions: list[dict[str, Any]] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tearsheet_links = tearsheet_links or {}
    underwrite_decisions = underwrite_decisions or []
    counts = candidates["research_priority"].value_counts().to_dict() if not candidates.empty else {}
    total = len(candidates)
    quick_records = [
        record for record in underwrite_decisions
        if str(record.get("decision") or "").upper() == "QUICK_REVIEW"
    ][:3]
    active_records = [
        record for record in underwrite_decisions
        if str(record.get("decision") or "").upper() in {"WAIT_FOR_PROOF", "WAIT_FOR_EVENT"}
    ][:3]
    a_count = len(quick_records)
    b_count = len(active_records)
    sec_loaded = int(candidates["source_posture"].str.startswith("Research-grade").sum()) if total else 0
    universe = health.get("universe") or {}
    discovered = int(universe.get("discovered", total))
    research_eligible = int(universe.get("research_eligible", total))
    fundamental_covered = int(universe.get("fundamental_covered", total))
    specialist_queue = int(universe.get("specialist_queue", 0))

    if a_count:
        headline = f"{a_count} idea{'s' if a_count != 1 else ''} worth a quick check"
        conclusion = (
            "The highlighted name(s) cleared a completed, source-backed underwrite—not merely the numerical screen. "
            "The brief below is the only company-level work that needs your attention."
        )
        user_instruction = "Review only the QUICK REVIEW item(s). Everything else remains my research work."
    else:
        headline = "Nothing needs your attention today"
        conclusion = (
            f"No company has cleared the full research bar. {b_count} name{'s are' if b_count != 1 else ' is'} "
            "in active underwriting, but none currently has enough evidence to justify your time."
        )
        user_instruction = "You do not need to review anything. I will keep narrowing the universe and doing the diligence."

    brief_cards = []
    indexed = (
        candidates.drop_duplicates("ticker").set_index("ticker")
        if not candidates.empty else pd.DataFrame()
    )
    for record in quick_records:
        ticker = str(record.get("ticker") or "").upper()
        if indexed.empty or ticker not in indexed.index:
            continue
        row = indexed.loc[ticker]
        link = (
            f'<a class="deep-link" href="{_esc(tearsheet_links[ticker])}">Open the completed underwrite →</a>'
            if ticker in tearsheet_links else ""
        )
        brief_cards.append(f"""
        <article class="idea">
          <div class="idea-top"><div><span class="ticker">{_esc(ticker)}</span>
            <span class="company">{_esc(row.get('company_name'))}</span></div>
            <span class="status look">QUICK REVIEW</span></div>
          <div class="one-line">{_esc(record.get('verdict'))}</div>
          <div class="decision-grid">
            <div><span>Why it may be mispriced</span><p>{_esc(record.get('mispricing'))}</p></div>
            <div><span>What to review</span><p>{_esc(record.get('review_request') or record.get('next_review'))}</p></div>
          </div>
          {link}
        </article>""")

    quick_section = (
        f'<h2>Quick review</h2><section class="ideas">{"".join(brief_cards)}</section>'
        if brief_cards else ""
    )
    active_items = []
    for record in active_records:
        ticker = str(record.get("ticker") or "").upper()
        link = (
            f' <a href="{_esc(tearsheet_links[ticker])}">underwrite</a>'
            if ticker in tearsheet_links else ""
        )
        active_items.append(
            f"<li><b>{_esc(ticker)}</b> — {_esc(record.get('verdict'))}{link}</li>"
        )
    active_details = (
        f'<details class="research-progress"><summary>Research in progress — optional ({len(active_items)})</summary>'
        f'<p class="muted">These names are my work, not your reading list.</p><ul>{"".join(active_items)}</ul></details>'
        if active_items else ""
    )

    table_rows = []
    for _, row in candidates.head(250).iterrows():
        ticker = str(row.get("ticker") or "").upper()
        table_rows.append(f"""
        <tr><td><b>{_esc(ticker)}</b><small>{_esc(row.get('company_name'))}</small></td>
        <td>{_short_status(str(row.get('research_priority') or ''))}</td>
        <td class="num">{_num(row.get('research_score'))}</td>
        <td>{_esc(row.get('trend_state'))}</td><td>{_esc(row.get('sector'))}</td>
        <td>{_esc(row.get('first_rejection'))}</td></tr>""")

    source_rows = "".join(
        f"<tr><td>{_esc(x.get('source'))}</td><td>{_esc(x.get('as_of'))}</td>"
        f"<td>{_esc(x.get('posture'))}</td><td>{_esc(x.get('use'))}</td></tr>"
        for x in health.get("sources", [])
    )
    size_rows = "".join(
        f"<tr><td>{_esc(str(name).title())}</td><td class='num'>{int(count):,}</td></tr>"
        for name, count in (universe.get("by_size") or {}).items()
    )
    lane_rows = "".join(
        f"<tr><td>{_esc(str(name).replace('_', ' ').title())}</td><td class='num'>{int(count):,}</td></tr>"
        for name, count in (universe.get("by_lane") or {}).items()
    )
    gaps = "".join(f"<li>{_esc(item)}</li>" for item in health.get("gaps", []))

    payload = {
        "policy_version": POLICY_VERSION,
        "as_of": health.get("as_of"),
        "candidate_count": total,
        "quick_review_count": a_count,
        "internal_diligence_count": b_count,
        "completed_underwrite_count": len(underwrite_decisions),
        "live_actions_enabled": False,
    }
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fundamental Research Brief — {_esc(health.get('as_of'))}</title>
<style>
:root{{--bg:#0a1020;--panel:#121b2e;--panel2:#18243b;--text:#ecf1fa;--muted:#9ba9bd;--line:#293955;--green:#67dca8;--amber:#ffd070;--red:#ff9a9a;--blue:#78baff}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(180deg,#09101d,#0d1628 46%,#0a1020);color:var(--text);font:15px/1.5 Inter,Segoe UI,Arial,sans-serif}}
main{{max-width:1050px;margin:auto;padding:36px 28px}}.eyebrow{{color:var(--muted);font-size:12px;letter-spacing:.8px;text-transform:uppercase}}h1{{font-size:38px;letter-spacing:-.8px;line-height:1.08;margin:7px 0 12px}}h2{{font-size:21px;margin:32px 0 12px}}p{{margin:6px 0}}.muted,small{{color:var(--muted)}}
.answer{{background:linear-gradient(135deg,#15243c,#111a2d);border:1px solid #365172;border-radius:14px;padding:20px 22px;margin:20px 0}}.answer p{{font-size:17px}}.instruction{{color:#cfe4ff;font-weight:750;margin-top:12px}}
.warning{{padding:10px 14px;border-radius:9px;background:#302817;border:1px solid #8b6b2c;color:#ffdc8a;font-size:13px}}
.ideas{{display:grid;gap:12px}}.idea{{background:rgba(18,27,46,.96);border:1px solid var(--line);border-radius:13px;padding:18px 20px}}.idea-top{{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}}.ticker{{font-size:24px;font-weight:850;margin-right:8px}}.company{{color:var(--muted)}}
.status{{padding:5px 9px;border-radius:999px;font-size:11px;font-weight:850;letter-spacing:.35px;white-space:nowrap}}.look{{background:#173b31;color:#7be7b7}}.watch{{background:#40351e;color:#ffdb84}}.background{{background:#20334c;color:#9bcaff}}.pass{{background:#45272a;color:#ffb0b0}}
.one-line{{font-size:16px;font-weight:700;margin:12px 0;border-top:1px solid var(--line);padding-top:12px}}.decision-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.decision-grid span{{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px;font-weight:750}}.decision-grid p{{margin-top:4px}}.deep-link{{display:inline-block;margin-top:10px;color:var(--blue);text-decoration:none;font-weight:750}}.deep-link:hover{{text-decoration:underline}}.empty{{padding:18px;background:var(--panel);border-radius:12px;color:var(--muted)}}
.workflow{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.workflow div{{background:var(--panel2);border-left:4px solid var(--blue);border-radius:8px;padding:13px}}.workflow b{{display:block;margin-bottom:3px}}
.map-link{{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-top:14px;padding:14px 16px;background:var(--panel);border:1px solid var(--line);border-radius:10px;color:var(--blue);text-decoration:none;font-weight:800}}.map-link span{{display:block;color:var(--muted);font-weight:400;font-size:13px}}.map-link:hover{{border-color:#4b6f99}}
details{{margin-top:28px;background:rgba(18,27,46,.72);border:1px solid var(--line);border-radius:12px;padding:14px 17px}}summary{{cursor:pointer;font-weight:800;color:#c9d7e9}}.diag{{padding-top:10px}}.diag-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0}}.diag-grid div{{background:var(--panel2);padding:10px;border-radius:7px}}.diag-grid b{{display:block;font-size:19px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.panel{{background:var(--panel);padding:14px;border-radius:9px;overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:760px}}table.compact{{min-width:0}}th{{text-align:left;font-size:10px;text-transform:uppercase;color:var(--muted);letter-spacing:.4px;padding:8px;border-bottom:1px solid var(--line)}}td{{padding:9px 8px;border-bottom:1px solid #22304a;vertical-align:top}}td.num{{text-align:right}}td small{{display:block}}ul{{padding-left:19px}}footer{{color:var(--muted);font-size:12px;margin-top:28px}}
@media(max-width:760px){{main{{padding:24px 16px}}h1{{font-size:31px}}.decision-grid,.workflow,.diag-grid,.two{{grid-template-columns:1fr}}.idea-top{{display:block}}.status{{display:inline-block;margin-top:8px}}}}
</style></head><body><main>
<div class="eyebrow">Fundamental research brief · {_esc(health.get('as_of'))}</div>
<h1>{_esc(headline)}</h1>
<div class="warning">Research only. No position, allocation, order, or live action is generated here.</div>
<section class="answer"><div class="eyebrow">My conclusion</div><p>{_esc(conclusion)}</p><p class="instruction">{_esc(user_instruction)}</p></section>

{quick_section}
{active_details}

<h2>How this will reach you</h2>
<section class="workflow"><div><b>LOOK</b>One to three names that deserve a quick check.</div><div><b>WATCH</b>I am still doing the work; no review needed.</div><div><b>PASS</b>Rejected quietly unless the reason teaches us something.</div></section>
<a class="map-link" href="company_maps.html"><div>Founder-led + your circle-of-competence maps<span>Separate running research universes; not recommendations.</span></div><b>Open →</b></a>

<details><summary>Research engine diagnostics — optional</summary><div class="diag">
<p class="muted">The broad universe stays here for audit and debugging. It is not your daily reading list.</p>
<section class="diag-grid"><div><b>{discovered:,}</b>discovered</div><div><b>{research_eligible:,}</b>eligible</div><div><b>{fundamental_covered:,}</b>covered</div><div><b>{sec_loaded:,}</b>SEC packages</div></section>
<section class="two"><div class="panel"><b>By size</b><table class="compact"><tbody>{size_rows}</tbody></table></div><div class="panel"><b>By research lane</b><table class="compact"><tbody>{lane_rows}</tbody></table><small>{specialist_queue:,} require specialist scorecards.</small></div></section>
<h2>Full candidate table</h2><div class="panel"><table><thead><tr><th>Company</th><th>Status</th><th>Score</th><th>Trend</th><th>Sector</th><th>First rejection</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<h2>Evidence gaps</h2><div class="panel"><ul>{gaps}</ul></div>
<h2>Source register</h2><div class="panel"><table><thead><tr><th>Source</th><th>As of</th><th>Posture</th><th>Use</th></tr></thead><tbody>{source_rows}</tbody></table></div>
</div></details>
<footer>The research engine can remain broad; the reader-facing brief will remain narrow. Live actions are disabled.</footer>
<script type="application/json" id="report-meta">{html.escape(json.dumps(payload))}</script>
</main></body></html>"""
    output_path.write_text(document, encoding="utf-8")
    return output_path
