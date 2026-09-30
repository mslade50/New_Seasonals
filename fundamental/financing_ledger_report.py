"""Standalone, source-linked funding reconciliation report."""
from __future__ import annotations

from html import escape


STATUS = {
    "cash_received_since_balance": "Received since balance",
    "funding_announced_receipt_unconfirmed": "Announced; receipt unconfirmed",
    "same_day_timing_unresolved": "Timing needs review",
    "receipt_allocation_needs_review": "Receipt needs allocation",
    "no_documented_intervening_funding_coverage_incomplete": "Funding audit incomplete",
    "financial_balance_unavailable": "Balance unavailable",
}


def money(value):
    return "—" if value is None else f"${value / 1e6:,.1f}m"


def text(value):
    return escape(str(value if value is not None else "—"), quote=True)


def links(urls):
    return " ".join(f'<a href="{text(u)}" target="_blank" rel="noopener">Source {i + 1}</a>' for i, u in enumerate(urls))


def render(result):
    s = result["stats"]
    record_map = {r["record_id"]: r for r in result["records"]}
    signal_rows = []
    for r in sorted(result["signals"], key=lambda r: (r["ticker"], r["session"])):
        sources = sorted({url for rid in r["source_record_ids"] for url in record_map[rid]["sources"]})
        if r.get("prior_review_source") and r["prior_review_source"] not in sources:
            sources.append(r["prior_review_source"])
        net = r["net_receipts_usd"] + r["estimated_net_receipts_usd"]
        note = r.get("prior_review_note") or "No complete funding audit yet; absence of a receipt record does not establish no financing."
        funding = "<strong>Net " + money(net) + "</strong>" if net else "No quantified net receipt"
        if r["bridge_includes_estimated_net"]:
            funding += "<small>Includes issuer estimate</small>"
        if r["gross_receipts_usd"]:
            funding += f'<small>Gross only: {money(r["gross_receipts_usd"])}; excluded from net bridge</small>'
        if r["unspecified_receipts_usd"]:
            funding += f'<small>Other received: {money(r["unspecified_receipts_usd"])}; basis unresolved</small>'
        runway = "—" if r["original_runway"] is None else f'{r["original_runway"]:.1f} months'
        signal_rows.append(f'''<tr data-search="{text(r['ticker'] + ' ' + r['session'] + ' ' + STATUS[r['status']])}">
<td><b>{text(r['ticker'])}</b><small>{text(r['session'])}</small></td>
<td>{money(r['reported_liquidity'])}<small>As of {text(r['balance_date'])}</small><small>Original operating runway: {runway}</small>{'<small class="warn">Cash-only coverage</small>' if r['cash_only'] else ''}<small>{links([r['reported_liquidity_source']]) if r.get('reported_liquidity_source') else ''}</small></td>
<td><span class="badge">{STATUS[r['status']]}</span><div class="receipt">{funding}</div></td>
<td>{text(note)}<div class="sources">{links(sources)}</div></td></tr>''')
    ledger_rows = []
    for r in sorted(result["records"], key=lambda r: (r["ticker"], (r.get("available_at") or r.get("available_date", "")), r["record_id"])):
        cash = r.get("cash_start", "")
        if r.get("cash_end") and r["cash_end"] != cash:
            cash += " to " + r["cash_end"]
        timing = cash or r.get("event_date", "—")
        known = r.get("available_at") or (r["available_date"] + " · time unknown")
        ledger_rows.append(f'''<tr data-search="{text(r['ticker'] + ' ' + r['stage'] + ' ' + r['kind'])}">
<td><b>{text(r['ticker'])}</b><small>{text(r['kind'].replace('_', ' '))}</small></td>
<td>{text(r['stage'].capitalize())}<small>{text(r['status'])}</small></td>
<td>{text(timing)}<small>Public by: {text(known)}</small></td>
<td>{money(r.get('amount_usd'))}<small>{text(r['amount_basis'].replace('_', ' '))}</small></td>
<td>{text(r.get('note', ''))}<div class="sources">{links(r['sources'])}</div></td></tr>''')
    queue = "".join(f'<tr><td>{text(r["name"])}</td><td>{r.get("financing_fact_rows", 0):,}</td><td>{text(r.get("filings_to_review"))}</td><td>{text(r.get("filings_without_search_capture"))}</td></tr>' for r in result["queue"] if r["priority"])
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Financing history · Funding reconciliation</title><style>
:root{color-scheme:light;--ink:#172f3c;--muted:#5c6e78;--line:#d6e0e4;--green:#11675e;--amber:#8b541d}
*{box-sizing:border-box}body{margin:0;background:#f2f5f5;color:var(--ink);font:15px/1.5 system-ui,sans-serif}main{max-width:1370px;margin:auto;padding:40px 30px 70px}header{padding:8px 0 24px}.eyebrow{font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--green)}h1{font-size:clamp(29px,4vw,46px);max-width:1050px;line-height:1.13;letter-spacing:-.035em;margin:15px 0}h2{font-size:23px;margin:0 0 12px}.lede{max-width:930px;font-size:18px;color:var(--muted)}.meta{font-size:13px;color:var(--muted)}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:10px 0 25px}.metric{padding:19px;background:white;border:1px solid var(--line);border-radius:8px}.metric b{display:block;font-size:32px;line-height:1.2}.metric span{font-size:13px;color:var(--muted)}section,details.panel{padding:25px;background:#fff;border:1px solid var(--line);border-radius:8px;margin-bottom:23px}.callout{border-left:4px solid var(--green);padding:14px 20px;background:#edf5f2;margin:15px 0 24px}.toolbar{display:flex;gap:15px;align-items:center;margin:19px 0}input{font:inherit;padding:10px 13px;border:1px solid #a7bbc4;border-radius:6px;width:340px;max-width:100%}.scroll{overflow-x:auto;max-width:100%}table{border-collapse:collapse;width:100%;text-align:left;min-width:980px}th{font-size:12px;text-transform:uppercase;letter-spacing:.05em;background:#f1f5f6;color:var(--muted)}th,td{padding:14px 13px;border-bottom:1px solid var(--line);vertical-align:top}#signals td:nth-child(1){width:115px}#signals td:nth-child(2){width:200px}#signals td:nth-child(3){width:255px}small{display:block;font-size:12px;color:var(--muted);margin-top:5px}.badge{display:inline-block;background:#edf2f5;padding:4px 8px;border-radius:5px;font-size:12px;font-weight:650}.receipt{margin-top:11px;font-size:13px}.sources{margin-top:8px;display:flex;gap:12px;flex-wrap:wrap;font-size:12px}a{color:var(--green);text-underline-offset:3px}.warn{color:var(--amber)}summary{font-size:21px;font-weight:650;cursor:pointer}.downloads{display:flex;gap:17px;flex-wrap:wrap}.empty{padding:20px;color:var(--muted)}footer{font-size:13px;color:var(--muted)}.methods{display:grid;grid-template-columns:1fr 1fr;gap:26px}.methods p{margin:0 0 12px}#ledger td:last-child{min-width:370px}
@media(max-width:700px){main{padding:22px 14px 45px}.metrics{grid-template-columns:1fr 1fr;gap:9px}.metric{padding:14px}.metric b{font-size:28px}section,details.panel{padding:17px}.methods{grid-template-columns:1fr;gap:8px}.lede{font-size:16px}.toolbar{display:block}input{width:100%}.toolbar span{display:block;margin-top:8px}}
</style></head><body><main><header><div class="eyebrow">Offering research / Financing ledger</div>
<h1>Check the funding history before trusting the cash-runway flag.</h1>
<p class="lede">The ledger separates announced deals from cash actually received, then checks what was public at each historical signal. This improves the inputs. It does not establish an offering forecast or a profitable short strategy.</p>
<div class="meta">Frozen 2023–2025 pilot · 90 issuers · Free SEC and issuer sources · Updated __DATE__</div></header>
<div class="metrics"><div class="metric"><b>__SIGNALS__</b><span>Original strength + short-runway observations</span></div><div class="metric"><b>__FLAGGED__</b><span>With documented funding since the cash balance</span></div><div class="metric"><b>__RECORDS__</b><span>Dated funding assertions, including revisions</span></div><div class="metric"><b>__FACTS__</b><span>As-filed financing facts across __ISSUERS__ issuers</span></div></div>
<div class="callout"><b>What changes:</b> a launch or pricing announcement can flag an existing financing plan, but it cannot increase cash. Only confirmed receipts enter the receipt history. Missing closing evidence remains unresolved.</div>
<section><h2>The original signals, reconciled to known funding</h2><p>All 20 observations remain visible. “Funding audit incomplete” is a research queue, not clearance to short. Net receipts below exclude cash already inside the reported balance.</p>
<div class="toolbar"><input id="search" aria-label="Filter signals" placeholder="Search ticker, date, or funding status"><span id="count" class="meta"></span></div>
<div class="scroll"><table id="signals"><thead><tr><th>Signal</th><th>Reported liquidity</th><th>Funding known at signal</th><th>Interpretation and evidence</th></tr></thead><tbody>__SIGNAL_ROWS__</tbody></table></div><div id="empty" class="empty" hidden>No matching signals.</div></section>
<section><h2>How to read the ledger</h2><div class="methods"><div><p><b>Two dates matter.</b> A cash receipt has an economic date and a later public confirmation date. A retrospective filing cannot make its information available at an earlier signal.</p><p><b>One financing, several stages.</b> Launch, pricing, closing and revised proceeds share a funding identity. Separate option tranches are recorded incrementally; repeated disclosures do not create extra cash.</p></div><div><p><b>Reported cash is still historical.</b> Any reported-balance-plus-net-receipts bridge excludes subsequent burn and other flows. It is not current cash, and no adjusted runway is promoted as a verified estimate.</p><p><b>Capacity is not cash.</b> Shelves, ATM capacity, undrawn credit and contingent tranches need their own records. XBRL financing series can overlap and are never blindly summed or treated as offering announcements.</p></div></div></section>
<details class="panel" id="ledger-panel"><summary>Funding stages and source evidence</summary><p>__EVENTS__ financing identities; __RECEIPTS__ receipt assertions include later revisions of the same cash. Blank launch amounts prevent later pricing terms leaking into earlier dates.</p><input id="ledger-search" aria-label="Filter funding ledger" placeholder="Search ticker or stage"><div class="scroll"><table id="ledger"><thead><tr><th>Issuer / kind</th><th>Stage</th><th>Economic date / public availability</th><th>Amount / basis</th><th>Evidence and caveat</th></tr></thead><tbody>__LEDGER_ROWS__</tbody></table></div><div id="ledger-empty" class="empty" hidden>No matching funding records.</div></details>
<details class="panel" id="coverage"><summary>Coverage and remaining source review</summary><p>Financing statements are inventoried for all 90 issuers. The event ledger remains a reviewed subset. No no-offering window is certified by this build; offering incidence and trading returns remain withheld. Existing historical price gaps also remain.</p><p>__ARCHIVED__ of __SOURCES__ ledger source URLs have hash-verified raw captures. Source download failures are visible in the source index. A keyword hit or captured filing is not evidence of a completed audit.</p><div class="scroll"><table><thead><tr><th>Signal issuer</th><th>Financing fact vintages</th><th>Filings in review inventory</th><th>Without prior search capture</th></tr></thead><tbody>__QUEUE__</tbody></table></div></details>
<section><h2>Research files</h2><div class="downloads"><a href="ledger.csv">Funding ledger</a><a href="signal_reconciliation.csv">Signal reconciliation</a><a href="financing_facts.csv">Financing fact vintages</a><a href="source_index.csv">Source index</a><a href="review_queue.json">Remaining filing reviews</a><a href="manifest.json">Run evidence</a></div></section>
<footer>Research only. Next: complete issuer funding and no-event coverage before expanding the frozen sample or measuring offering incidence. The watchlist and production strategies are unchanged.</footer>
</main><script>
function wire(input,table,empty,count){const rows=[...document.querySelectorAll(table+' tbody tr')];function filter(){const q=document.querySelector(input).value.trim().toLowerCase();let n=0;for(const r of rows){r.hidden=!r.dataset.search.toLowerCase().includes(q);if(!r.hidden)n++;}document.querySelector(empty).hidden=n>0;if(count)document.querySelector(count).textContent=n+' of '+rows.length+' observations';}document.querySelector(input).addEventListener('input',filter);filter();}
wire('#search','#signals','#empty','#count');wire('#ledger-search','#ledger','#ledger-empty');
</script></body></html>'''
    values = dict(DATE=text(result["generated_at"][:10]), SIGNALS=s["signals"], FLAGGED=s["signals_with_documented_funding"],
        RECORDS=s["assertions"], FACTS=f'{s["financing_facts"]:,}', ISSUERS=s["fact_issuers"], SIGNAL_ROWS="".join(signal_rows),
        EVENTS=s["funding_events"], RECEIPTS=s["receipt_assertions"], LEDGER_ROWS="".join(ledger_rows),
        ARCHIVED=s["archived_sources"], SOURCES=s["sources"], QUEUE=queue)
    for key, value in values.items():
        html = html.replace("__" + key + "__", str(value))
    return html
