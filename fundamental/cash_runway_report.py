"""Standalone local research report; never a private-site production payload."""
from __future__ import annotations

from collections import Counter
from html import escape
import json


def money(value):
    return "—" if value is None else f"${value / 1e6:,.2f}m"


def months(value):
    return "—" if value is None else f"{value:,.1f}m"


def e(value):
    return escape(str(value if value is not None else "—"), quote=True)


def render_report(rows, manifest):
    ordered = sorted(rows, key=lambda r: (r.get("status") != "calculated", r.get("bucket") == "Stale",
                    r.get("runway_6m") is None, r.get("runway_6m") or 0, r["ticker"]))
    counts = Counter(r.get("bucket", "Unavailable") for r in rows)
    short = counts["Under 3m"] + counts["3–6m"]
    audited = sum(bool(r.get("manual_review")) for r in rows)
    table, details = [], []
    for r in ordered:
        ticker = e(r["ticker"])
        scope = r.get("liquidity_basis", "Unavailable")
        bucket = r.get("bucket", "Unavailable")
        review = r.get("manual_review")
        marker = " <span class='verified'>checked</span>" if review else ""
        table.append(f"""<tr data-ticker='{ticker}' data-bucket='{e(bucket)}'>
          <td><a class='symbol' href='#{ticker}'>{ticker}</a>{marker}<small>{e(r.get('company_name'))}</small></td>
          <td title='{e(scope)}'>{money(r.get('reported_liquidity'))}<small>{'Cash only*' if 'Cash only' in scope else 'Cash + current investments' if r.get('reported_liquidity') is not None else 'Unavailable'}</small></td>
          <td>{money(r.get('monthly_burn_6m'))}</td>
          <td class='runway'>{months(r.get('runway_6m'))}<small>{e(bucket)}</small></td>
          <td>{months(r.get('runway_3m'))}</td>
          <td>{months(r.get('runway_with_capex_6m'))}</td>
          <td>{e(r.get('balance_date'))}<small>{e(r.get('balance_age_days'))} days old</small></td>
          <td class='funding'>{e(review.get('financing_note') if review else r.get('financing_status'))}</td>
        </tr>""")
        sources = []
        for label, source in r.get("flow_sources", {}).items():
            if not source:
                continue
            inputs = " · ".join(f"<a href='{e('https://www.sec.gov/Archives/edgar/data/' + str(r['cik']) + '/' + f['accession'].replace('-', '') + '/' + f['accession'] + '-index.htm')}' target='_blank' rel='noopener'>{e(f['start'])} → {e(f['end'])}: {money(f['value'])}</a>" for f in source["inputs"])
            sources.append(f"<tr><td>{e(label)}</td><td>{money(source['value'])}</td><td>{e(source['method'])}</td><td>{inputs}</td></tr>")
        docs = []
        for d in r.get("financing_documents", []):
            snippets = "".join(f"<blockquote>{e(t)}</blockquote>" for t in d["excerpts"])
            docs.append(f"<details><summary><a href='{e(d['url'])}' target='_blank' rel='noopener'>{e(d['form'])}</a> · {e(d['accepted_at'])}</summary>{snippets or '<p>No financing keyword hit. This is not proof of no financing.</p>'}</details>")
        notes = "".join(f"<li>{e(w)}</li>" for w in r["warnings"])
        audit = ""
        if review:
            audit_sources = " · ".join(f"<a href='{e(s['url'])}' target='_blank' rel='noopener'>{e(s['label'])}</a>" for s in review.get("sources", []))
            audit = f"""<div class='audit'><b>Manual source check: {e(review.get('check_status'))}</b>
            <p>{e(review.get('accounting_note'))}</p><p><b>Management outlook:</b> {e(review.get('management_runway'))}</p>
            <p><b>Since the balance date:</b> {e(review.get('financing_note'))}</p><p>{audit_sources}</p></div>"""
        details.append(f"""<details class='company' id='{ticker}'><summary><b>{ticker}</b> · {e(r.get('company_name'))} · {months(r.get('runway_6m'))} operating runway</summary>
          <p>{e(scope)}. {e(r.get('balance_date'))} balance: cash {money(r.get('cash'))}; current investments {money(r.get('current_investments'))}.</p>
          <p><a href='{e(r.get('filing_url', '#'))}' target='_blank' rel='noopener'>Latest financial filing</a> · Accepted {e(r.get('filing_accepted_at'))}</p>
          {audit}<ul class='warnings'>{notes}</ul>
          <div class='scroll'><table class='inputs'><thead><tr><th>Input</th><th>Amount</th><th>Method</th><th>Original periods / filing links</th></tr></thead><tbody>{''.join(sources)}</tbody></table></div>
          <details><summary>Financing discovery documents ({len(docs)}) — context may refer to older transactions</summary>{''.join(docs)}<p>{e(r.get('financing_status'))}</p></details>
        </details>""")
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
    <title>Cash runway | SEC research pilot</title><style>
    :root{{--ink:#182b3a;--muted:#627282;--line:#dce3e9;--blue:#135985;--paper:#fff;--bg:#f2f5f7}}
    *{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}
    main{{max-width:1530px;margin:auto;padding:36px 38px 60px}}a{{color:var(--blue)}}h1{{font-size:35px;letter-spacing:-1px;margin:5px 0 8px}}h2{{font-size:22px;margin:28px 0 12px}}
    .eyebrow{{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--blue);font-weight:750}}.lede{{max-width:1020px;font-size:17px;margin:0 0 13px}}.meta{{color:var(--muted);font-size:12px}}
    .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}}.card{{background:white;border:1px solid var(--line);padding:16px 20px;border-radius:9px}}.card strong{{font-size:31px;display:block;line-height:1.2}}.card span{{color:var(--muted);font-size:13px}}
    .notice{{border-left:4px solid #a77124;background:#fff8e9;padding:14px 18px;margin:16px 0;color:#61491e;font-size:14px}}
    .toolbar{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px}}input,select{{border:1px solid #bac8d2;border-radius:6px;padding:10px 12px;background:#fff;color:var(--ink);font:inherit}}input{{width:300px}}.scroll{{overflow:auto;background:white;border:1px solid var(--line);border-radius:8px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th{{font-size:11px;text-transform:uppercase;letter-spacing:.5px;background:#e8eef2;text-align:left;padding:13px 12px;white-space:nowrap}}td{{padding:13px 12px;border-top:1px solid var(--line);vertical-align:top;white-space:nowrap}}td:first-child{{min-width:170px;max-width:230px}}small{{display:block;font-size:11px;color:var(--muted);white-space:normal;line-height:1.35;margin-top:4px}}.symbol{{font-weight:750;text-decoration:none;font-size:16px}}.runway{{font-weight:700}}.funding{{min-width:230px;max-width:345px;white-space:normal;font-size:12px}}.verified{{font-size:10px;background:#e5f1eb;color:#246244;padding:2px 5px;border-radius:4px}}
    .company{{background:white;border:1px solid var(--line);border-radius:8px;margin:9px 0;padding:13px 17px}}summary{{cursor:pointer}}.company>summary{{font-size:15px}}.company>summary>b{{color:var(--blue)}}.company p{{font-size:14px}}.company .inputs td{{font-size:12px;white-space:normal}}.company .inputs td:last-child{{min-width:350px}}.warnings{{color:#6a5a3c;font-size:13px;padding-left:20px}}blockquote{{margin:10px 0;padding:9px 14px;border-left:3px solid #c8d3dc;font-size:12px;background:#f6f8fa}}.audit{{border:1px solid #b9d5c6;background:#f3faf6;padding:12px 16px;border-radius:7px;margin:15px 0}}.method{{max-width:1050px}}.method li{{margin:7px 0}}footer{{margin-top:28px;font-size:12px;color:var(--muted)}}
    @media(max-width:760px){{main{{padding:22px 15px}}h1{{font-size:28px}}.cards{{grid-template-columns:repeat(2,1fr);gap:9px}}.card{{padding:12px}}.card strong{{font-size:25px}}input{{width:100%}}.lede{{font-size:15px}}}}
    </style></head><body><main><div class='eyebrow'>New Seasonals · Free SEC research pilot</div><h1>Cash runway watchlist</h1>
    <p class='lede'>Which companies are consuming cash faster than their reported liquidity can sustain? Compare recent burn, inspect the funding context, and follow each calculation back to its filing.</p>
    <div class='meta'>Research cutoff: {e(manifest['as_of'])} · USD · Deliberate sample; current listing checks, no price/liquidity filter · Research flags only</div>
    <div class='cards'><div class='card'><strong>{len(rows)}</strong><span>companies attempted</span></div><div class='card'><strong>{manifest['calculated']}</strong><span>runway calculations available</span></div><div class='card'><strong>{short}</strong><span>under 6 months at balance date*</span></div><div class='card'><strong>{audited}</strong><span>companies checked against filings</span></div></div>
    <div class='notice'><b>These are balance-date estimates, not cash balances today.</b> *Cash-only rows are lower bounds when investments remain unverified. Subsequent financing and management spending plans can change the result. Missing capex is not zero. A short runway is a research flag, not a forecast offering date.</div>
    <div class='toolbar'><input id='search' aria-label='Search ticker or company' placeholder='Search ticker or company'><select id='filter' aria-label='Filter runway'><option value='all'>All companies</option><option value='short'>Under 6 months</option><option value='6–12m'>6–12 months</option><option value='12m+'>12 months or more</option><option value='No observed burn'>No observed operating burn</option><option value='Unavailable'>Unavailable</option></select><span class='meta' id='count'>{len(rows)} companies shown</span><a href='watchlist.csv' download>Download CSV</a></div>
    <div class='scroll'><table id='watchlist'><thead><tr><th>Company / source detail</th><th>Reported liquidity</th><th>Monthly burn<br>6-month average</th><th>Operating runway<br>6-month burn</th><th>Runway<br>latest quarter</th><th>Runway + PP&amp;E<br>6-month burn</th><th>Balance date</th><th>Subsequent funding review</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>
    <p class='meta'>Click a ticker for exact input periods, calculation provenance, manual checks and financing evidence. Zero observed burn has no finite runway estimate.</p>
    <h2>Source checks and company details</h2>{''.join(details)}
    <section class='method'><h2>How to read this pilot</h2><ul>
    <li>Operating burn = max(0, −operating cash flow) / months in the fiscal interval. PP&amp;E-inclusive burn = max(0, PP&amp;E cash capex − operating cash flow) / months. Runway divides reported liquidity by that burn.</li>
    <li>Cash excludes explicitly restricted balances. A combined cash/short-term-investments tag is used once; otherwise cash and a reported current-investment balance are added. Long-term securities are excluded from the automated total, even if marketable; manual notes identify material examples.</li>
    <li>Quarterly cash-flow figures are reconstructed from actual start/end dates and YTD differences. Fiscal years need not follow calendar years. Missing or conflicting facts remain unavailable; no revenue history is required.</li>
    <li>Only known SEC acceptance timestamps at or before the cutoff enter calculations. This current sample is not a survivorship-safe or dissemination-time-validated historical backtest.</li>
    <li>Financing discovery reads the latest financial report and up to {manifest.get('max_filings', 12)} subsequent relevant filings per company. Keyword excerpts can describe old transactions. Empty hits, inaccessible documents and unused shelf capacity do not prove the absence or completion of a raise. No keyword match changes cash.</li>
    <li>PP&amp;E capex is not total investment spending. Capitalized software, patents, project equipment, acquisitions, debt payments, one-time receipts and changes in working capital require judgment. Management's outlook is shown separately from calculated runway.</li>
    <li>{e(manifest['scope'])}</li></ul>
    <p>Sources: <a href='https://www.sec.gov/search-filings/edgar-application-programming-interfaces'>SEC Company Facts / Submissions</a> · <a href='https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs'>Nasdaq listing directory</a>. Raw source captures and SHA-256 digests are retained with this report.</p></section>
    <footer>Local research artifact · {e(manifest['version'])} · No trade recommendations or automatic execution.</footer></main>
    <script>const rows=[...document.querySelectorAll('#watchlist tbody tr')];function filterRows(){{const q=document.querySelector('#search').value.toLowerCase();const f=document.querySelector('#filter').value;let n=0;for(const r of rows){{const match=r.textContent.toLowerCase().includes(q)&&(f==='all'||f===r.dataset.bucket||(f==='short'&&['Under 3m','3–6m'].includes(r.dataset.bucket)));r.hidden=!match;if(match)n++;}}document.querySelector('#count').textContent=n+' companies shown';}}document.querySelector('#search').addEventListener('input',filterRows);document.querySelector('#filter').addEventListener('change',filterRows);function expandHash(){{const d=document.getElementById(location.hash.slice(1));if(d&&d.tagName==='DETAILS')d.open=true;}}addEventListener('hashchange',expandHash);expandHash();</script></body></html>"""
