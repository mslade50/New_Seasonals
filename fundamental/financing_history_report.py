"""Standalone, evidence-linked report for a bounded historical research pilot."""
from __future__ import annotations

from html import escape as e
import json


GROUPS = {"strong_short": "Strength + short runway", "strong_longer": "Strength + longer runway",
          "strong_no_burn": "Strength + no observed burn", "no_strength_short": "Short runway without strength"}


def render(result):
    counts = result["counts"]
    cards = [("Frozen companies", counts["companies"]), ("Monthly observations", counts["observations"]),
             ("Strength + short runway", counts["target_observations"]), ("Matched longer-runway controls", counts["matched_longer"])]
    metrics = "".join(f'<div class="metric"><strong>{v:,}</strong><span>{e(k)}</span></div>' for k, v in cards)
    groups = ""
    for name, stats in result["groups"].items():
        rate = f'{stats["rate"]:.1%}' if stats["rate"] is not None else "Withheld"
        groups += f'<tr><td>{e(GROUPS[name])}</td><td>{stats["observations"]}</td><td>{stats["total_issuers"]}</td><td>{stats["positives"]}</td><td>{stats["unknown"]}</td><td>{rate}</td></tr>'
    signals = ""
    for r in result["signals"]:
        runway = min(v for v in [r.get("runway_operating"), r.get("runway_capex")] if v is not None)
        flag = r.get("review_note") or ("Cash-only lower bound; investments unverified" if r.get("cash_only") else "Reported runway; funding history needs review")
        outcome = {1: "Confirmed announcement", 0: "Reviewed: no announcement"}.get(r.get("outcome_60"), "Unknown / not fully audited")
        if r.get("outcome_status_60") == "same_day_date_only":
            outcome = "Same-day timing unresolved"
        link = r.get("review_source")
        if link:
            flag = f'<a href="{e(link, quote=True)}">{e(flag)}</a>'
        else:
            flag = e(flag)
        signals += f'<tr data-search="{e(str(r.get("ticker")) + " " + r["session"], quote=True)}"><td><b>{e(str(r.get("ticker")))}</b></td><td>{r["session"]}</td><td>{runway:.1f}m</td><td>{e(outcome)}</td><td>{flag}</td></tr>'
    events = ""
    for r in result["events"]:
        sources = " · ".join(f'<a href="{e(u, quote=True)}">Source {i+1}</a>' for i, u in enumerate(r["sources"]))
        events += f'<tr><td>{e(r.get("ticker_at_event", ""))}</td><td>{e(r["announcement_date"])}</td><td>{e(r["type"])}</td><td>{e(r["status"])}</td><td>{sources}<br><small>{e(r.get("caveat", ""))}</small></td></tr>'
    coverage = "".join(f'<tr><td>{e(r["label"])}</td><td>{r["value"]:,}</td><td>{e(r["note"])}</td></tr>' for r in result["coverage"])
    sources = [
        ("SEC financial statement data and historical revisions", "https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets"),
        ("SEC API documentation and historical submission files", "https://www.sec.gov/search-filings/edgar-application-programming-interfaces"),
        ("SEC full-text search scope", "https://www.sec.gov/edgar/search/efts-faq.html")]
    source_links = " · ".join(f'<a href="{url}">{e(label)}</a>' for label, url in sources)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Financing research · Historical pilot</title><style>
:root{{--ink:#19303c;--muted:#59707b;--line:#d9e2e6;--bg:#f2f5f6;--accent:#086c6c;--amber:#805616}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}main{{max-width:1280px;margin:auto;padding:34px 28px 60px}}header{{margin-bottom:25px}}.eyebrow{{text-transform:uppercase;letter-spacing:.14em;font-size:12px;color:var(--accent);font-weight:750}}h1{{font-size:35px;line-height:1.15;max-width:870px;margin:13px 0}}h2{{font-size:22px;margin:0 0 12px}}p{{max-width:990px;margin:10px 0}}.sub,small{{color:var(--muted)}}.tag{{display:inline-block;background:#fff0d6;color:var(--amber);padding:5px 10px;border-radius:5px;font-size:12px;font-weight:750}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}}.metric{{background:white;border:1px solid var(--line);border-radius:10px;padding:19px}}.metric strong{{font-size:36px;display:block;color:var(--accent)}}.metric span{{font-size:13px}}section{{background:white;border:1px solid var(--line);border-radius:10px;padding:24px;margin:18px 0}}.notice{{border-left:4px solid #c88f2b;background:#fff8eb;padding:14px 18px;margin:15px 0}}.scroll{{overflow:auto;max-width:100%}}.scroll table{{min-width:780px}}#signals{{min-width:1060px}}#signals td:nth-child(2){{white-space:nowrap}}#event-audit table{{min-width:1000px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th{{text-align:left;background:#eef4f4;color:var(--muted);font-size:12px}}td,th{{padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top}}a{{color:var(--accent);text-underline-offset:3px}}input{{padding:10px 12px;border:1px solid #aabdc4;border-radius:5px;font:inherit;width:280px;max-width:100%;margin-bottom:13px}}details>summary{{cursor:pointer;font-weight:650}}.downloads{{display:flex;gap:14px;flex-wrap:wrap}}footer{{font-size:12px;color:var(--muted)}}.columns{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}li{{margin:8px 0}}@media(max-width:700px){{main{{padding:22px 14px}}h1{{font-size:28px}}.metrics{{grid-template-columns:repeat(2,1fr);gap:9px}}.metric{{padding:13px}}.metric strong{{font-size:28px}}section{{padding:17px}}.columns{{grid-template-columns:1fr}}td,th{{padding:9px 10px}}}}
</style></head><body><main>
<header><div class="eyebrow">Offering research · Free-data historical pilot · 2023–2025</div><h1>The pilot is too thin to validate the offering signal.</h1><p class="sub">A frozen historical sample tests whether liquid rallies plus short reported cash runway anticipate primary equity offerings. This run establishes feasibility and exposes the gaps that must be fixed before an edge claim.</p><span class="tag">Research only · Predictive edge unproven · No trading simulation</span></header>
<div class="metrics">{metrics}</div>
<section><h2>What this run tells us</h2><p><b>{counts["target_observations"]} qualifying stock-months come from {counts["target_issuers"]} companies.</b> They produce {counts["matched_longer"]} outcome-blind pair with a strong, longer-runway company under the frozen matching rules. The comparison against short-runway stocks without strength produces {counts["matched_no_strength"]} pairs. Repeated signals are not independent samples.</p>
<div class="notice"><b>The stricter sensitivity leaves {counts['strict_target']} observations across {counts['strict_issuers']} companies.</b> No defensible offering-rate lift or trading edge can be reported: negative event coverage is incomplete, the matched sample is sparse, and free historical prices lose many former listings. Missing history remains unknown.</div>
<p>There is a useful design finding: reported cash can already be obsolete when a rally triggers the screen. The event audit identifies raises announced after the balance date but before the signal. Those observations are flagged for exclusion from the stricter sensitivity analysis; proceeds are never simply added to reported cash.</p></section>
<section><h2>Signal groups and outcome coverage</h2><div class="scroll"><table id="groups"><thead><tr><th>Group</th><th>Stock-months</th><th>Companies</th><th>Confirmed 60d positives</th><th>Unknown outcomes</th><th>60d incidence</th></tr></thead><tbody>{groups}</tbody></table></div><p class="sub">A zero in confirmed positives is not evidence of zero offerings. Rates and confidence intervals are withheld when negative labels are incomplete. A single offering can be inside more than one monthly window.</p></section>
<section><h2>The {counts["target_observations"]} qualifying observations</h2><p class="sub">Runway shown is the smaller available operating or PP&amp;E-capex-inclusive six-month estimate. It starts at the reported balance date.</p><input id="search" aria-label="Filter signal observations" placeholder="Filter ticker or date"><div class="scroll"><table id="signals"><thead><tr><th>Company</th><th>Signal session</th><th>Runway</th><th>Next 60 days</th><th>Funding-history check</th></tr></thead><tbody>{signals}</tbody></table></div><p id="empty" hidden>No matching observations.</p></section>
<section><h2>Coverage and attrition</h2><div class="scroll"><table><thead><tr><th>Check</th><th>Count</th><th>Meaning</th></tr></thead><tbody>{coverage}</tbody></table></div><p class="sub">No missing company was replaced. Price acquisition failures and unresolved ticker aliases remain in the denominator of the frozen 90-name cohort. Current Yahoo histories and reconstructed as-filed SEC data are not untouched historical database snapshots.</p></section>
<section><h2>Rules were fixed before outcomes</h2><div class="columns"><div><b>Sample and timing</b><ul><li>US-incorporated, nonfinancial Q4 2022 filers with $50m–$5b reported assets. Deterministic CIK-hash draw: 30 healthcare, 30 technology, 30 other.</li><li>Month-end NYSE sessions in 2023–2025; cutoff 30 minutes after the close. 2023–2024 development, 2025 evaluation; no threshold fitting.</li><li>Historical filings joined by accession and acceptance time. Later restatements cannot enter earlier observations.</li></ul></div><div><b>Signal and event definition</b><ul><li>Original price ≥$3, 20-session dollar volume ≥$5m; fixed sustained-strength or fresh-rally conditions.</li><li>Reported runway ≤24 months; balance ≤150 days old. Missing investments/capex remain explicit.</li><li>First public announcement of primary common/pre-funded equity cash raise within 60 calendar days; 30/90-day secondary windows. ATM, shelf, resale-only, merger financing, debt and exercises are separate.</li><li>Controls match month, sector, assets and liquidity; strong controls also match 60-day momentum.</li></ul></div></div><p><a href="protocol.json">Frozen protocol</a> · <a href="cohort_manifest.json">Sample hash and stratum populations</a></p></section>
<section><details id="event-audit"><summary>Audited offering examples ({len(result["events"])} events; partial registry)</summary><p class="sub">Deduplicated launch/pricing/closing records. “Provisional” means the first public announcement date is not established well enough for outcome labeling. Date-only same-day announcements remain ambiguous.</p><div class="scroll"><table><thead><tr><th>Ticker at event</th><th>Announcement date</th><th>Type</th><th>Date audit</th><th>Sources and limits</th></tr></thead><tbody>{events}</tbody></table></div></details></section>
<section><h2>Decision and next research gate</h2><p><b>Continue data research; do not promote this into a trading strategy.</b> Expand the cohort selected before the test period, repair historical security aliases and delisted coverage, and finish first-announcement and negative-window audits. Then rerun the fixed comparison with enough independent issuers and matched events. Preserve a fresh evaluation period when revising the signal.</p><p>Only after an issuance-prediction effect survives those checks should a separate trading study test entry timing, borrow availability, squeeze exposure, slippage and costs. Predicting an offering does not establish a profitable short.</p></section>
<section><h2>Download the research evidence</h2><div class="downloads"><a href="cohort.csv">Frozen cohort</a><a href="observations.csv">Signal observations</a><a href="labeled_observations.csv">Outcome coverage</a><a href="event_candidates.csv">Discovery candidates</a><a href="event_reviews.json">Audited events</a><a href="analysis.json">Analysis and counts</a><a href="manifest.json">Run manifest</a></div></section>
<footer><p>Built {e(result["generated_at"])}. Sample rates would describe this deliberately balanced pilot, not the wider stock market. Asset size is not market capitalization.</p><p>{source_links}</p></footer>
</main><script>const input=document.querySelector('#search');input.addEventListener('input',()=>{{let n=0;for(const row of document.querySelectorAll('#signals tbody tr')){{const ok=row.dataset.search.toLowerCase().includes(input.value.toLowerCase());row.hidden=!ok;if(ok)n++;}}document.querySelector('#empty').hidden=n!==0;}});</script></body></html>'''
