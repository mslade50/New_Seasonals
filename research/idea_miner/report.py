"""Standalone HTML renderer for the weekly hypothesis inbox."""

from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _pill(status: str) -> str:
    labels = {
        "advance_to_preregistration": "Ready for preregistration",
        "needs_source_diligence": "Needs source diligence",
        "duplicate_prior_hypothesis": "Already in registry",
    }
    return f'<span class="pill {_esc(status)}">{_esc(labels.get(status, status))}</span>'


def render_weekly_inbox(queue: Mapping[str, Any]) -> str:
    funnel = queue.get("funnel", {})
    coverage = queue.get("coverage", {})
    selected = queue.get("selected", [])

    cards: list[str] = []
    for number, card in enumerate(selected, start=1):
        instruments = ", ".join(card.get("instruments", []))
        requirements = ", ".join(card.get("data_requirements", []))
        source_url = _esc(card.get("source_url"))
        cards.append(
            f"""
            <article class="idea-card">
              <header>
                <div><span class="rank">{number:02d}</span><span class="lane">{_esc(card.get('archetype'))}</span></div>
                {_pill(str(card.get('status', '')))}
              </header>
              <h2>{_esc(card.get('claim'))}</h2>
              <p class="mechanism"><strong>Proposed mechanism.</strong> {_esc(card.get('mechanism'))}</p>
              <dl>
                <div><dt>Instruments / horizon</dt><dd>{_esc(instruments)} · {_esc(card.get('horizon'))}</dd></div>
                <div><dt>Actionability</dt><dd>{_esc(card.get('actionability'))}</dd></div>
                <div><dt>Variant wedge</dt><dd>{_esc(card.get('variant_wedge'))}</dd></div>
                <div><dt>Why now</dt><dd>{_esc(card.get('why_now'))}</dd></div>
                <div><dt>Falsifiable test</dt><dd>{_esc(card.get('falsifiable_test'))}</dd></div>
                <div><dt>First rejection</dt><dd>{_esc(card.get('first_rejection'))}</dd></div>
                <div><dt>What would kill it</dt><dd>{_esc(card.get('what_would_kill_it'))}</dd></div>
                <div><dt>Data required</dt><dd>{_esc(requirements)}</dd></div>
                <div><dt>Next workflow</dt><dd>{_esc(card.get('next_workflow'))}</dd></div>
              </dl>
              <footer>
                <span>Research-priority score {_esc(card.get('research_priority_score'))}/10</span>
                <a href="{source_url}">Source: {_esc(card.get('source_type', '').upper())}</a>
              </footer>
            </article>
            """
        )

    if not cards:
        cards.append(
            "<section class='empty'><h2>No hypothesis cleared the bounded review queue.</h2>"
            "<p>This is a valid result. Review source coverage and prior-registry duplicates.</p></section>"
        )

    source_rows = []
    for source in queue.get("sources", []):
        source_rows.append(
            "<tr>"
            f"<td>{_esc(source.get('source_type', '').upper())}</td>"
            f"<td><a href='{_esc(source.get('url'))}'>{_esc(source.get('title') or source.get('url'))}</a></td>"
            f"<td>{_esc(source.get('published_at') or 'not supplied')}</td>"
            f"<td>{_esc(source.get('retrieved_at'))}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weekly External Research Inbox · {_esc(queue.get('as_of'))}</title>
<style>
:root {{ --bg:#0b1018; --panel:#131b27; --panel2:#182333; --line:#2a394d; --text:#edf3fb; --muted:#9baabd; --blue:#72b7ff; --green:#6bd6a3; --amber:#f2c777; --red:#f18b92; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:linear-gradient(160deg,#0b1018 0%,#0e1723 55%,#0b1018 100%); color:var(--text); font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
a {{ color:var(--blue); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
main {{ width:min(1180px,calc(100% - 32px)); margin:0 auto; padding:54px 0 80px; }}
.eyebrow {{ color:var(--blue); text-transform:uppercase; letter-spacing:.14em; font-size:12px; font-weight:750; }}
h1 {{ font-size:clamp(34px,5vw,62px); line-height:1.02; max-width:850px; margin:10px 0 16px; letter-spacing:-.035em; }}
.lede {{ max-width:810px; color:var(--muted); font-size:18px; margin:0 0 30px; }}
.guardrail {{ border:1px solid var(--line); border-left:4px solid var(--amber); background:rgba(242,199,119,.06); padding:13px 16px; border-radius:8px; margin:0 0 24px; }}
.tiles {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:42px; }}
.tile {{ background:rgba(19,27,39,.88); border:1px solid var(--line); border-radius:12px; padding:17px; }}
.tile .value {{ display:block; font-size:30px; font-weight:760; letter-spacing:-.03em; }} .tile .label {{ color:var(--muted); font-size:13px; }}
.section-head {{ display:flex; justify-content:space-between; align-items:end; gap:20px; margin:32px 0 15px; }}
.section-head h2 {{ margin:0; font-size:25px; }} .section-head p {{ margin:0; color:var(--muted); }}
.ideas {{ display:grid; gap:16px; }}
.idea-card {{ background:linear-gradient(145deg,rgba(24,35,51,.98),rgba(19,27,39,.98)); border:1px solid var(--line); border-radius:14px; padding:22px; box-shadow:0 18px 50px rgba(0,0,0,.18); }}
.idea-card header,.idea-card footer {{ display:flex; justify-content:space-between; align-items:center; gap:16px; }}
.rank {{ color:var(--muted); font-variant-numeric:tabular-nums; margin-right:10px; }} .lane {{ color:var(--blue); font-size:12px; text-transform:uppercase; letter-spacing:.1em; font-weight:750; }}
.pill {{ border:1px solid var(--line); border-radius:99px; padding:5px 9px; font-size:12px; color:var(--muted); white-space:nowrap; }}
.pill.advance_to_preregistration {{ color:var(--green); border-color:rgba(107,214,163,.45); }} .pill.needs_source_diligence {{ color:var(--amber); border-color:rgba(242,199,119,.45); }}
.idea-card h2 {{ margin:16px 0 8px; line-height:1.25; font-size:23px; }} .mechanism {{ margin:0 0 18px; color:#c8d3e1; }}
dl {{ display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--line); border:1px solid var(--line); border-radius:10px; overflow:hidden; margin:0; }}
dl div {{ background:var(--panel); padding:13px 15px; }} dt {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; font-weight:750; }} dd {{ margin:4px 0 0; }}
dl div:last-child:nth-child(odd) {{ grid-column:1 / -1; }}
.idea-card footer {{ border-top:1px solid var(--line); margin-top:17px; padding-top:14px; color:var(--muted); font-size:13px; }}
.source-table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }} th,td {{ text-align:left; padding:11px 12px; border-bottom:1px solid var(--line); vertical-align:top; }} th {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }}
.footnote {{ color:var(--muted); font-size:13px; margin-top:16px; }} .empty {{ border:1px dashed var(--line); border-radius:12px; padding:30px; color:var(--muted); }}
@media(max-width:800px) {{ .tiles {{ grid-template-columns:1fr 1fr; }} dl {{ grid-template-columns:1fr; }} .idea-card header,.idea-card footer,.section-head {{ align-items:flex-start; flex-direction:column; }} }}
</style>
</head>
<body><main>
  <div class="eyebrow">Research-only hypothesis intake · {_esc(queue.get('as_of'))}</div>
  <h1>Weekly external research inbox</h1>
  <p class="lede">A bounded, deduplicated queue from X, SSRN and other supplied source records. Selection allocates research attention; it is not an investment recommendation, approval, or order.</p>
  <div class="guardrail"><strong>No production path.</strong> Every selected item still needs a preregistered test, point-in-time data, an explicit trial budget, independent validation, and a human promotion decision.</div>
  <section class="tiles">
    <div class="tile"><span class="value">{_esc(coverage.get('source_rows_received', 0))}</span><span class="label">source rows received</span></div>
    <div class="tile"><span class="value">{_esc(funnel.get('hypotheses_created', 0))}</span><span class="label">unique hypotheses</span></div>
    <div class="tile"><span class="value">{_esc(funnel.get('selected_for_review', 0))}</span><span class="label">bounded review queue</span></div>
    <div class="tile"><span class="value">{_esc(funnel.get('duplicate_prior', 0))}</span><span class="label">prior-registry duplicates</span></div>
  </section>
  <div class="section-head"><h2>Research-priority queue</h2><p>Archetype-balanced; maximum {_esc(queue.get('methodology', {}).get('max_candidates'))}</p></div>
  <section class="ideas">{''.join(cards)}</section>
  <div class="section-head"><h2>Source ledger</h2><p>Direct links and supplied timestamps</p></div>
  <table class="source-table"><thead><tr><th>Type</th><th>Source</th><th>Published</th><th>Retrieved</th></tr></thead><tbody>{''.join(source_rows)}</tbody></table>
  <p class="footnote">The process does not fetch sources, infer current prices, or validate paper results. Missing source detail remains a blocker rather than being filled from memory.</p>
</main></body></html>"""
