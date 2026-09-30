"""Standalone research watchlist; screen flags and verified disclosures stay separate."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
from html import escape
import json
import math
from io import StringIO
from pathlib import Path

import pandas as pd

from fundamental.financing_opportunity import VERSION, join_funding


def esc(value):
    return escape(str(value if value is not None else ""), quote=True)


def num(value, kind="number"):
    if value is None:
        return "—"
    if kind == "pct": return f"{value:+.1%}"
    if kind == "money":
        return f"${value / 1e9:,.2f}B" if abs(value) >= 1e9 else f"${value / 1e6:,.1f}M"
    return f"{value:,.1f}"


def link(url, label):
    if not url or not str(url).startswith("https://"):
        return esc(label)
    return f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(label)} ↗</a>'


def apply_reviews(rows, reviews, manifest):
    by_ticker = {r["ticker"]: r for r in rows}
    seen = set()
    for review in reviews:
        ticker = review["ticker"]
        if ticker in seen:
            raise ValueError("Duplicate manual review")
        seen.add(ticker)
        row = by_ticker[ticker]
        if review["cik"] != row["cik"] or review["as_of"] != manifest["as_of"]:
            raise ValueError(f"Review issuer/cutoff mismatch for {ticker}")
        if review["balance_date"] != row.get("financial", {}).get("balance_date"):
            raise ValueError(f"Review financial period mismatch for {ticker}")
        if not review.get("sources") or any(not s["url"].startswith("https://") for s in review["sources"]):
            raise ValueError("Source-linked reviews required")
        if any(s.get("published_date", "9999") > manifest["as_of"][:10] for s in review["sources"]):
            raise ValueError("Review source published after cutoff")
        for source in review["sources"]:
            datetime.strptime(source["published_date"], "%Y-%m-%d")
            if source["published_date"] == manifest["as_of"][:10]:
                published = datetime.fromisoformat(source.get("published_at", ""))
                if published.tzinfo is None or published > datetime.fromisoformat(manifest["as_of"]):
                    raise ValueError("Same-day review sources need a timestamp before the cutoff")
        for key, expected in review.get("expected_financials", {}).items():
            actual = row["financial"].get(key)
            if actual is None or not math.isclose(actual, expected, rel_tol=0, abs_tol=1):
                raise ValueError(f"Manual financial tie-out failed for {ticker}: {key}")
        row["review"] = review
    return rows


def detail(row):
    f = row.get("financial") or {}
    review = row.get("review")
    financial = "<p>Financials have not been checked. This is a price-screen result only.</p>"
    if f:
        period = (f.get("flow_sources", {}).get("ocf_6m") or {})
        financial = f'''<div class="facts">
          <div><small>Balance date</small>{esc(f.get('balance_date', 'Unavailable'))}</div>
          <div><small>Reported liquidity</small>{num(f.get('reported_liquidity'), 'money')}</div>
          <div><small>Operating burn / month</small>{num(f.get('monthly_burn_6m'), 'money')}</div>
          <div><small>With PP&amp;E capex / month</small>{num(f.get('monthly_burn_with_capex_6m'), 'money')}</div></div>
          <p>{esc(f.get('liquidity_basis', 'Financial coverage unavailable'))}. Six-month cash-flow interval: {esc(period.get('start', '—'))} to {esc(period.get('end', f.get('balance_date','—')))}.
          {link(f.get('filing_url'), 'Financial filing')} · accepted {esc(f.get('filing_accepted_at','unknown'))}.</p>
          <p>Recent-quarter operating runway: {num(f.get('runway_3m'))} months; six-month operating runway: {num(f.get('runway_6m'))} months. Both use the same reported balance; differences reflect changing historical burn.</p>
          <p class="muted">{esc(' · '.join(f.get('warnings', [])))}</p>'''
    evidence = ""
    if review:
        items = "".join(f"<p><strong>{esc(label)}:</strong> {esc(review.get(key, 'Unknown'))}</p>" for key,label in [
            ("shelf", "Primary shelf"), ("atm", "ATM"), ("prior_raise", "Previous financing"),
            ("warrants", "Warrants / converts"), ("subsequent_financing", "After the cash balance"),
            ("first_rejection", "Why this could be a false positive"), ("next_check", "Next check")])
        sources = " · ".join(link(s["url"], s["label"]) for s in review["sources"])
        evidence = f'<div class="review"><h3>Financing source check</h3><p><strong>{esc(review["verdict"])}</strong></p>{items}<p>{sources}</p></div>'
    else:
        evidence = '<p class="notice">Financing readiness unverified. No inference of an available shelf, remaining capacity, current ATM sales or imminent offering.</p>'
    docs = ""
    for doc in f.get("financing_documents", []):
        docs += f'<p>{link(doc.get("url"), doc.get("form", "Filing"))} · accepted {esc(doc.get("accepted_at", "unknown"))}</p>'
    if docs:
        docs = f'<details><summary>Automated filing search — requires interpretation</summary><p>This bounded search covers the latest financial report and up to 12 later financing-related filings. Linked exhibits are not automatically read.</p>{docs}<p>Truncated: {esc(f.get("financing_scan_truncated"))}. Fetch errors: {len(f.get("financing_errors", []))}.</p></details>'
    warnings = ""
    if row.get("split_recent"):
        warnings += "Recent stock split: review capital structure and historical offering-price comparisons. "
    if row.get("listing_warning"):
        warnings += "Listing warning: " + row["listing_warning"]
    return f'''<details class="company" id="{esc(row['ticker'])}"><summary><b>{esc(row['ticker'])}</b> {esc(row['company_name'])}<span>{esc(row.get('funding_status'))}</span></summary>
      <div class="companybody"><p>{esc(' + '.join(row['setups']))} · {link(row.get('price_source'), 'Price history')} · {esc(row.get('price_as_of'))}</p>
      <p>60d return minus SPY: {num(row.get('relative_60'), 'pct')} points · largest opening gap / 5d: {num(row.get('max_gap_5'), 'pct')} · latest RVOL: {num(row.get('volume_ratio'))}×.</p>
      <p class="muted">{esc(warnings)}</p>{financial}{evidence}{docs}</div></details>'''


def render(rows, manifest):
    setup_rows = [r for r in rows if r.get("setups")]
    setup_rows.sort(key=lambda r: (not r.get("candidate"), not bool(r.get("review")), -(r.get("return_20") or 0), r["ticker"]))
    reviewed = [r for r in setup_rows if r.get("review")]
    counts = manifest["counts"]
    cards = "".join(f'''<article class="reviewcard"><div class="eyebrow">SOURCE CHECKED · {esc(' + '.join(r['setups']))}</div>
      <h3><a href="#{esc(r['ticker'])}">{esc(r['ticker'])}</a><span>{num(r.get('return_20'),'pct')} / 20d</span></h3>
      <p>{esc(r['review']['verdict'])}</p><p class="muted">{esc(r['review']['summary'])}</p></article>''' for r in reviewed)
    tr = []
    for r in setup_rows:
        f = r.get("financial") or {}
        status = r["review"]["verdict"] if r.get("review") else r.get("funding_status", "Not checked")
        tr.append(f'''<tr data-candidate="{int(r.get('candidate',False))}" data-reviewed="{int(bool(r.get('review')))}" data-cohort="{esc('|'.join(r['setups']))}" data-search="{esc((r['ticker']+' '+r['company_name']).lower())}">
          <td class="ticker"><a href="#{esc(r['ticker'])}">{esc(r['ticker'])}</a><small>{esc(r['company_name'])}</small></td>
          <td>{esc(' + '.join(r['setups']))}</td><td>${r['close']:.2f}</td><td>{num(r.get('return_5'),'pct')}</td>
          <td>{num(r.get('return_20'),'pct')}</td><td>{num(r.get('return_60'),'pct')}</td>
          <td>{num(r.get('dollar_volume_20'),'money')}</td><td>{num(r.get('max_volume_ratio_5'))}×</td>
          <td>{num(f.get('runway_6m'))}<small>{esc(f.get('balance_date','Not checked'))}</small></td>
          <td>{num(f.get('runway_with_capex_6m'))}</td><td class="status">{esc(status)}<small>{'Source check attached' if r.get('review') else 'Financing readiness unverified'}</small></td></tr>''')
    details = "".join(detail(r) for r in setup_rows)
    rules = manifest["policy"]
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Financing opportunity watchlist</title><style>
    :root{--ink:#182b3b;--muted:#607180;--paper:#f4f6f7;--line:#dbe2e7;--blue:#195b89;--warm:#fff4df}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,Segoe UI,sans-serif}main{max-width:1550px;margin:0 auto;padding:36px 34px 70px}a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}.eyebrow{font-size:11px;letter-spacing:.13em;font-weight:750;color:var(--muted)}h1{font-size:36px;line-height:1.14;margin:10px 0 14px;letter-spacing:-1px}h2{font-size:23px;margin:0 0 14px}h3{margin:8px 0;font-size:19px}p{margin:10px 0}.lead{font-size:18px;max-width:1000px}.muted,small{color:var(--muted)}.meta{font-size:12px;margin:16px 0 22px}.funnel{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:24px 0}.tile{background:white;border:1px solid var(--line);border-top:3px solid var(--blue);padding:15px 18px;border-radius:8px}.tile b{display:block;font-size:27px}.tile span{font-size:12px}.notice{background:var(--warm);border-left:3px solid #ba8a33;padding:11px 15px;font-size:13px;border-radius:3px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:22px 0}.reviewcard{background:white;border:1px solid var(--line);border-radius:9px;padding:20px}.reviewcard h3{display:flex;align-items:center;justify-content:space-between}.reviewcard h3 span{font-size:13px;font-weight:500}.reviewcard p{font-size:13px}section{margin:30px 0}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:15px 0}input,select{font:inherit;border:1px solid #b9c9d5;background:white;border-radius:6px;padding:9px 11px;max-width:100%}input{width:260px}.tablewrap{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:8px}table{width:100%;border-collapse:collapse;font-size:12px;min-width:1280px}th{text-align:left;background:#eaf0f4;color:#38536a;padding:12px 10px;white-space:nowrap}td{padding:12px 10px;border-top:1px solid var(--line);vertical-align:top;font-variant-numeric:tabular-nums}td small{display:block;font-size:10px;margin-top:4px}.ticker{width:150px;position:sticky;left:0;background:white}.ticker a{font-weight:800;font-size:15px}.status{min-width:205px;max-width:250px}tr:hover td{background:#f6f9fb}.company{border:1px solid var(--line);border-radius:7px;background:white;margin:10px 0}.company>summary{padding:14px 18px;cursor:pointer}.company summary b{margin-right:10px}.company summary span{float:right;font-size:12px;color:var(--muted)}.companybody{padding:0 20px 20px}.facts{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0}.facts div{font-size:20px}.facts small{display:block;font-size:11px}.review{border-top:1px solid var(--line);margin-top:15px;padding-top:10px}.review p{font-size:13px}blockquote{margin:8px 0;padding:10px 14px;background:#f4f6f8;font-size:12px}details.method{background:white;padding:18px;border:1px solid var(--line);border-radius:7px}details.method summary{cursor:pointer;font-weight:650}li{margin:8px 0}footer{border-top:1px solid var(--line);padding-top:18px;font-size:12px}.empty{padding:20px;display:none}@media(max-width:750px){main{padding:24px 16px}h1{font-size:30px}.lead{font-size:16px}.funnel{grid-template-columns:repeat(2,1fr)}.cards{grid-template-columns:1fr}.facts{grid-template-columns:repeat(2,1fr)}.company summary span{float:none;display:block;margin-top:5px}.controls>*{width:100%}.tile b{font-size:24px}}
    </style></head><body><main>''' + f'''
    <div class="eyebrow">NEW SEASONALS · EQUITY FINANCING RESEARCH</div><h1>Who has a window to raise?</h1>
    <p class="lead">{counts['funding_flags']} trading setups also show reported cash runway of 24 months or less. Investigate funding need, available issuance capacity and the quality of the rally.</p>
    <div class="meta">Completed price session: <b>{manifest['price_session']}</b> · SEC information cutoff: {esc(manifest['as_of'])} · Free public sources · Research candidates, no trade signals</div>
    <div class="funnel">''' + "".join(f'<div class="tile"><b>{v:,}</b><span>{esc(label)}</span></div>' for v,label in [
        (counts['universe'],'Listed symbols screened'),(counts['price_liquid'],'Pass price & liquidity'),
        (counts['setups'],'Strength / rally matches'),(counts['financial_checked'],'Financial checks'),
        (counts['funding_flags'],'Reported runway flags')]) + f'''</div>
    <p class="notice"><strong>Reported runway is measured at the balance-sheet date.</strong> It is not cash remaining today. Later financing can invalidate the apparent urgency. Shelf capacity, warrants and conditional funding are not cash; unknown capacity stays unknown.</p>
    <section><h2>Financing checks</h2><p class="muted">{len(reviewed)} source-checked examples. Findings can weaken a screen flag as well as support further research.</p><div class="cards">{cards or '<p>No manual financing checks attached yet.</p>'}</div></section>
    <section><h2>Trading setups and funding flags</h2><p class="muted">Cohorts can overlap. Default view shows reported runway flags; other matches remain available for comparison. Click a ticker for financial periods and source evidence.</p>
    <div class="controls"><input id="search" placeholder="Search ticker or company" aria-label="Search ticker or company"><select id="view" aria-label="Research group"><option value="funding">Reported runway flags</option><option value="all">All trading setups</option><option value="reviewed">Source checked</option><option value="comparison">Other / unchecked setups</option></select>
    <select id="cohort" aria-label="Trading setup"><option value="all">Both setups</option><option>Sustained strength</option><option>Fresh rally</option></select><span id="count"></span></div>
    <div class="tablewrap"><table id="watchlist"><thead><tr><th>Ticker / company</th><th>Setup</th><th>Close</th><th>5d return</th><th>20d return</th><th>60d return</th><th>Avg $ vol / 20d</th><th>Peak RVOL / 5d</th><th>Op. runway / mo</th><th>+ capex / mo</th><th>Research status</th></tr></thead><tbody>{''.join(tr)}</tbody></table><div class="empty" id="empty">No rows match these filters.</div></div>
    <p class="meta">Returns use adjusted close. Dollar volume uses Close × Volume. RVOL uses the preceding 20 sessions. Runway uses the latest available six-month cash-flow interval. — means unavailable or no finite runway; see details.</p></section>
    <section><details class="method"><summary>Screen rules, coverage and next test</summary>
    <p>These thresholds are initial research assumptions, not calibrated offering probabilities.</p><ul>
    <li>Price ≥ ${rules['min_price']:.0f}; average 20-session dollar volume ≥ ${rules['min_dollar_volume_20']/1e6:.0f}M; at least {rules['min_history']} consecutive recent benchmark sessions. Recent IPOs and sparse histories are excluded.</li>
    <li><strong>Sustained strength:</strong> 20d return ≥ {rules['sustained_return_20']:.0%}, 60d return ≥ {rules['sustained_return_60']:.0%}, 60d return exceeds SPY by ≥ {rules['sustained_relative_60']:.0%} points, and adjusted close is above its 50-session average.</li>
    <li><strong>Fresh rally:</strong> 5d return ≥ {rules['fresh_return_5']:.0%} OR an opening gap ≥ {rules['fresh_gap_5']:.0%} in the last five sessions; at least one session in that window has RVOL ≥ {rules['fresh_relative_volume']:.1f}×. The price move and volume spike need not occur on the same day.</li>
    <li><strong>Funding flag:</strong> positive six-month operating burn or PP&amp;E-capex-inclusive burn gives reported runway ≤24 months; balance sheet age ≤150 days. Cash-only calculations are lower bounds where investment coverage is missing. Longer-runway companies can still raise opportunistically.</li>
    <li>Financial checks alternate cohorts, ranking sustained strength by 20d return and fresh rallies by the larger of 5d return / recent positive gap. {counts['financial_unchecked']} trading setups remain unchecked under the bounded financial queue. Comparison names are selected screen matches, not a representative control sample.</li>
    <li>Current listing scope: {esc(manifest['scope'])} Listing abnormalities remain visible; current listings do not establish US domicile. Banks/financial SICs and unmapped foreign financial taxonomies remain coverage gaps.</li>
    <li>Coverage: {counts['price_unavailable']} histories unavailable, stale or incomplete; {counts['financial_gaps']} checked financials incomplete or unavailable; {counts['financial_stale']} stale financial balances. Fresh daily bars are unofficial Yahoo observations and may be revised.</li>
    <li>Next test: freeze this snapshot, label subsequent announced primary offerings separately from ATM usage, and compare 30/60/90-day event rates and price responses. No future outcomes, historical performance or predictive accuracy have been measured.</li></ul>
    <p>Downloads: <a href="watchlist.csv">Trading setups CSV</a> · <a href="coverage.csv">Full universe coverage CSV</a> · <a href="watchlist.json">Financial evidence JSON</a> · <a href="observations.csv">Prospective observations</a></p></details></section>
    <section><h2>Company evidence</h2>{details}</section>
    <footer>Sources: <a href="https://www.sec.gov/search-filings/edgar-application-programming-interfaces">SEC EDGAR</a> · <a href="https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs">Nasdaq listing directory</a> · <a href="https://finance.yahoo.com/quote/SPY/history/">Yahoo Finance / SPY benchmark</a>. Source retrieval timestamps and hashes are retained with this snapshot. No borrow availability or execution costs have been checked.</footer>
    <script>
    const rows=[...document.querySelectorAll('#watchlist tbody tr')];
    function filter(){{let n=0;const q=document.querySelector('#search').value.toLowerCase();const v=document.querySelector('#view').value;const c=document.querySelector('#cohort').value;for(const r of rows){{const ok=r.dataset.search.includes(q)&&(c==='all'||r.dataset.cohort.includes(c))&&(v==='all'||v==='funding'&&r.dataset.candidate==='1'||v==='reviewed'&&r.dataset.reviewed==='1'||v==='comparison'&&r.dataset.candidate!=='1');r.hidden=!ok;if(ok)n++;}}document.querySelector('#count').textContent=n+' companies';document.querySelector('#empty').style.display=n?'none':'block';}}
    document.querySelector('#search').addEventListener('input',filter);for(const id of ['view','cohort'])document.querySelector('#'+id).addEventListener('change',filter);
    function expandHash(){{const id=decodeURIComponent(location.hash.slice(1));const d=document.getElementById(id);if(d&&d.classList.contains('company')){{d.open=true;d.scrollIntoView({{block:'start'}});}}}}window.addEventListener('hashchange',expandHash);document.addEventListener('click',e=>{{const a=e.target.closest('a[href^="#"]');if(a){{const d=document.getElementById(a.hash.slice(1));if(d)d.open=true;}}}});filter();expandHash();
    </script></main></body></html>'''


def freeze_observations(path, setups, manifest):
    """Preserve labels and reject changes to an already frozen observation cohort."""
    observations = []
    for r in setups:
        f = r.get("financial") or {}
        known = r["funding_status"] not in {"Not checked", "Stale financials", "Financial coverage gap"}
        observations.append(dict(ticker=r["ticker"], cik=r["cik"], observed_at=manifest["as_of"],
            price_session=manifest["price_session"], setups=" | ".join(r["setups"]),
            funding_flag=r["candidate"] if known else None, funding_status=r["funding_status"],
            financial_checked=bool(f), balance_date=f.get("balance_date"),
            runway_6m=f.get("runway_6m"), runway_with_capex_6m=f.get("runway_with_capex_6m"),
            offering_30d=None, offering_60d=None, offering_90d=None,
            outcome_status="Pending; no follow-up performed"))
    body = pd.DataFrame(observations).to_csv(index=False)
    if path.exists():
        old = list(csv.DictReader(StringIO(path.read_text(encoding="utf-8"))))
        new = list(csv.DictReader(StringIO(body)))
        baseline = [k for k in new[0] if not k.startswith("offering_") and k != "outcome_status"] if new else []
        if len(old) != len(new) or any(any(a.get(k) != b[k] for k in baseline) for a,b in zip(old,new)):
            raise ValueError("Frozen observations differ; use a new snapshot directory to revise the cohort")
    else:
        path.write_text(body, encoding="utf-8")


def build_report(output, reviews_path=None):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    prices = json.loads((output / "price_screen.json").read_text(encoding="utf-8"))
    financials = json.loads((output / "financials.json").read_text(encoding="utf-8"))
    rows = [join_funding(r, financials.get(r["ticker"])) for r in prices]
    reviews = json.loads(reviews_path.read_text(encoding="utf-8")) if reviews_path else []
    apply_reviews(rows, reviews, manifest)
    setups = [r for r in rows if r.get("setups")]
    freeze_observations(output / "observations.csv", setups, manifest)
    manifest["counts"] = dict(universe=len(rows), price_liquid=sum(bool(r.get("tradable_filter")) for r in rows),
        setups=len(setups), funding_flags=sum(r["candidate"] for r in rows),
        financial_checked=sum(r.get("financial") is not None for r in setups),
        financial_unchecked=sum(r.get("financial") is None for r in setups),
        price_unavailable=sum(r["price_status"] != "complete" for r in rows),
        financial_gaps=sum(r["funding_status"] == "Financial coverage gap" for r in rows),
        financial_stale=sum((r.get("financial") or {}).get("balance_age_days", 0) > 150 for r in rows), manual_checks=len(reviews))
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["run_id"] = output.name
    manifest["input_hashes"] = {p: hashlib.sha256((output / p).read_bytes()).hexdigest() for p in
        ["universe.json", "price_captures.json", "price_screen.json", "financials.json", "sec/captures.json"]}
    root = Path(__file__).resolve().parents[1]
    manifest["source_hashes"] = {p:hashlib.sha256((root / p).read_bytes()).hexdigest() for p in [
        "fundamental/financing_opportunity.py", "fundamental/financing_opportunity_report.py", "scripts/build_financing_watchlist.py", "fundamental/cash_runway.py"]}
    if reviews_path:
        manifest["reviews_sha256"] = hashlib.sha256(reviews_path.read_bytes()).hexdigest()
    for name, data in [("watchlist.json", rows),("manual_reviews.json",reviews)]:
        (output / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    flat = []
    for r in rows:
        f = r.get("financial") or {}
        item = {k:v for k,v in r.items() if not isinstance(v,(dict,list))}
        item.update(setups=" | ".join(r["setups"]), price_reasons=" | ".join(r["price_reasons"]),
            runway_6m=f.get("runway_6m"), runway_with_capex_6m=f.get("runway_with_capex_6m"),
            balance_date=f.get("balance_date"), financial_filing=f.get("filing_url"),
            manual_verdict=r.get("review",{}).get("verdict"), review_status="Source checked" if r.get("review") else "Unverified")
        flat.append(item)
    pd.DataFrame(flat).to_csv(output / "coverage.csv", index=False)
    pd.DataFrame([r for r in flat if r["setups"]]).to_csv(output / "watchlist.csv", index=False)
    (output / "financing_watchlist.html").write_text(render(rows,manifest), encoding="utf-8")
    manifest["outputs"] = {"report": {"path":str(output / "financing_watchlist.html"),
        "sha256":hashlib.sha256((output / "financing_watchlist.html").read_bytes()).hexdigest()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(manifest["counts"],indent=2))
