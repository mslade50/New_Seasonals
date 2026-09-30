"""Free, resumable historical financing research. All outputs are task artifacts.

Run --stage universe first to freeze a sample before inspecting event outcomes.
No strategy, canonical cache, portfolio, or production-state writes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlencode, urljoin, urlparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from lxml import etree

from fundamental.sec import SECClient
from fundamental.cash_runway import filing_rows, filing_url


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str,
                               allow_nan=False), encoding="utf-8")


def preserve_attempt(path):
    """Retain a failed/older derived attempt before a resumable replacement."""
    if path.exists():
        dest = path.parent / "prior_attempts"
        dest.mkdir(exist_ok=True)
        data = path.read_bytes()
        copy = dest / (path.stem + "-" + hashlib.sha256(data).hexdigest()[:16] + ".json")
        if not copy.exists():
            copy.write_bytes(data)


def now():
    return datetime.now(timezone.utc).isoformat()


class Capture:
    """URL-addressed immutable source bytes with resumable hash verification."""
    def __init__(self, output):
        self.root = output / "capture"
        self.root.mkdir(exist_ok=True)
        self.sec = SECClient(sleep_seconds=.3)

    def get(self, url):
        key = hashlib.sha256(url.encode()).hexdigest()
        body, meta = self.root / f"{key}.bin", self.root / f"{key}.json"
        if body.exists() and meta.exists():
            data = body.read_bytes()
            if hashlib.sha256(data).hexdigest() != json.loads(meta.read_text())["sha256"]:
                raise ValueError(f"Capture hash mismatch: {body}")
            return data
        for attempt in range(3):
            time.sleep(.4 + attempt * 2)
            host = urlparse(url).hostname or ""
            user_agent = self.sec.user_agent if host == "sec.gov" or host.endswith(".sec.gov") else "NewSeasonalsResearch/1.0"
            response = self.sec.session.get(url, headers={"User-Agent": user_agent}, timeout=50)
            if response.status_code not in (429, 500, 502, 503, 504):
                break
        response.raise_for_status()
        data = response.content
        if len(data) > 50_000_000:
            raise ValueError("Document exceeds explicit 50MB capture limit")
        body.write_bytes(data)
        save(meta, dict(url=url, sha256=hashlib.sha256(data).hexdigest(),
                        fetched_at=now(), bytes=len(data), status=response.status_code))
        return data

    def json(self, url):
        result = json.loads(self.get(url))
        if not isinstance(result, dict):
            raise ValueError("Expected JSON object")
        return result


def stratum(sic):
    sic = int(sic)
    if 2830 <= sic <= 2839 or 3840 <= sic <= 3859 or 8000 <= sic <= 8099:
        return "health"
    if 3570 <= sic <= 3579 or 3660 <= sic <= 3699 or 7370 <= sic <= 7379:
        return "technology"
    return "other"


EVENT_FORMS = ["8-K", "8-K/A"] + [f"424B{n}" for n in range(1, 9)] + ["FWP"]


def event_search_query(cik, start, end):
    # The endpoint takes root forms, not literal amended-form names.
    return dict(q='"offering" OR "private placement" OR "purchase agreement" OR "sales agreement"',
                dateRange="custom", startdt=start, enddt=end, ciks=f"{int(cik):010d}",
                forms=",".join(f for f in EVENT_FORMS if not f.endswith("/A")))


def universe(out):
    if (out / "cohort.csv").exists():
        print("Frozen cohort exists; leaving membership unchanged", flush=True)
        return
    protocol = dict(version="financing-history.v1", frozen_at=now(),
        seed="financing-history-2023-2025-v1", sample_per_stratum=30,
        cohort_cutoff="2022-12-31T23:59:59-05:00",
        universe="US-incorporated nonfinancial 10-Q/10-K issuers in SEC 2022Q4, consolidated USD assets $50m-$5b; historical exchange listing verified after selection; failures retained",
        baseline_selection="Latest reported period then acceptance, before cutoff; no prevrpt filter",
        strata="health: SIC 2830-2839/3840-3859/8000-8099; technology: 3570-3579/3660-3699/7370-7379; other: remaining nonfinancial SIC",
        fsds_vintage="SEC 2022Q4 as-filed reconstruction, reprocessed by SEC in 2024; original filing identities retained",
        observation_schedule="Last NYSE session of each month, 2023-2025, 30 minutes after official close",
        development_years=[2023, 2024], evaluation_years=[2025],
        heldout_rule="No threshold fitting; evaluate fixed original watchlist rules; 2025 evaluation frozen in advance, not a pristine independent external validation",
        primary_outcome="First public announcement of primary common/pre-funded equity cash raise within next 60 calendar days",
        secondary_horizons=[30, 90], outcome_capture_end="2026-03-31",
        event_types="Underwritten, registered direct, standalone cash PIPE; exclude ATM facility/sales, shelf-only, secondary-only resale, convertible/preferred debt, exercise and merger financing",
        same_day_rule="Date-only event on observation date is ambiguous; exclude that observation",
        competing_events="No assumed negative following missing coverage or delisting; no replacement of unavailable issuers",
        comparisons="Strong+short-runway vs strong+longer-runway and short-runway+no-strength; sector/month matches within 4x reported assets and dollar volume, momentum caliper 30 percentage points for strength comparison",
        financial_rule="As-public acceptance timestamps; six-month operating OR PP&E-capex runway <=24 months; balance age <=150 days. Cash-only lower bounds flagged, missing capex not zero.",
        strict_sensitivity="Exclude cash-only short-runway flags and intervening announced primary equity raises after the reported balance date",
        inference="Sample rates only (equal strata not population-weighted); issuer-cluster uncertainty, distinct issuer and event counts; no trading P&L or fitted offering probabilities",
        limitations="Monthly snapshots can miss brief rallies. Small cohort is a feasibility study. Yahoo missing/delisted histories and event-recall uncertainty can prevent a predictive conclusion.")
    save(out / "protocol.json", protocol)
    sub = pd.read_csv(out / "sec_2022q4_sub.tsv", sep="\t", dtype=str, keep_default_na=False)
    sub["sic_n"] = pd.to_numeric(sub.sic, errors="coerce").fillna(0).astype(int)
    sub = sub[(sub.countryinc == "US") & sub.form.isin(["10-Q", "10-K", "10-Q/A", "10-K/A"])
              & ~sub.sic_n.between(6000, 6999) & sub.sic_n.gt(0)
              & (sub.accepted < "2023-01-01")]
    sub = sub.sort_values(["period", "accepted"]).drop_duplicates("cik", keep="last")
    selected = set(sub.adsh)
    assets = []
    with zipfile.ZipFile(out / "sec_2022q4.zip") as z:
        for chunk in pd.read_csv(z.open("num.txt"), sep="\t", dtype=str, keep_default_na=False, chunksize=200000):
            rows = chunk[(chunk.tag == "Assets") & (chunk.uom == "USD") & (chunk.qtrs == "0")
                         & (chunk.coreg == "") & (chunk.segments == "") & chunk.adsh.isin(selected)]
            assets.append(rows[["adsh", "ddate", "value"]])
    nums = pd.concat(assets, ignore_index=True).drop_duplicates()
    merged = sub.merge(nums, left_on=["adsh", "period"], right_on=["adsh", "ddate"])
    conflicting = set(merged.groupby("adsh").value.nunique().loc[lambda x: x > 1].index)
    merged = merged[~merged.adsh.isin(conflicting)].copy()
    merged["baseline_assets"] = pd.to_numeric(merged.value, errors="coerce")
    merged = merged[merged.baseline_assets.between(50e6, 5e9)].copy()
    merged["stratum"] = merged.sic_n.map(stratum)
    merged["draw_key"] = merged.cik.map(lambda c: hashlib.sha256((protocol["seed"] + ":" + str(int(c))).encode()).hexdigest())
    columns = ["cik", "name", "sic", "stratum", "adsh", "period", "accepted", "instance", "baseline_assets", "draw_key"]
    merged[columns].to_csv(out / "eligible_population.csv", index=False)
    cohort = merged.sort_values("draw_key").groupby("stratum", sort=True).head(30)
    cohort[columns].sort_values(["stratum", "draw_key"]).to_csv(out / "cohort.csv", index=False)
    save(out / "cohort_manifest.json", dict(frozen_at=now(), eligible_by_stratum=merged.stratum.value_counts().to_dict(),
         sample_by_stratum=cohort.stratum.value_counts().to_dict(), conflicting_asset_accessions=sorted(conflicting),
         cohort_sha256=hashlib.sha256((out / "cohort.csv").read_bytes()).hexdigest(),
         protocol_sha256=hashlib.sha256((out / "protocol.json").read_bytes()).hexdigest()))
    print(json.loads((out / "cohort_manifest.json").read_text()), flush=True)


def merge_submissions(main, archives):
    rows = filing_rows(main)
    for archive in archives:
        rows.extend(filing_rows({"filings": {"recent": archive}}))
    by_acc = {}
    for row in rows:
        by_acc.setdefault(row["accessionNumber"], row)
    rows = sorted(by_acc.values(), key=lambda x: (x.get("filingDate", ""), x["accessionNumber"]))
    keys = set().union(*(r.keys() for r in rows)) if rows else set()
    return dict(main, filings={"recent": {k: [r.get(k) for r in rows] for k in keys}, "files": main["filings"].get("files", [])})


def baseline_securities(data):
    root = etree.fromstring(data, parser=etree.XMLParser(resolve_entities=False, no_network=True))
    fields = {"TradingSymbol", "SecurityExchangeName", "Security12bTitle"}
    contexts = {}
    for node in root.iter():
        local = etree.QName(node).localname if isinstance(node.tag, str) else ""
        if local in fields:
            contexts.setdefault(node.get("contextRef"), {}).setdefault(local, []).append("".join(node.itertext()).strip())
    securities = []
    for context, vals in contexts.items():
        for symbol in vals.get("TradingSymbol", []):
            securities.append(dict(context=context, ticker=symbol, exchange=" | ".join(vals.get("SecurityExchangeName", [])),
                                   title=" | ".join(vals.get("Security12bTitle", []))))
    return securities


def capture_sec(out):
    cap = Capture(out)
    directory = out / "issuers"
    directory.mkdir(exist_ok=True)
    cohort = pd.read_csv(out / "cohort.csv", keep_default_na=False)
    for i, base in enumerate(cohort.to_dict("records")):
        cik = int(base["cik"])
        dest = directory / str(cik)
        dest.mkdir(exist_ok=True)
        if (dest / "coverage.json").exists():
            old = json.loads((dest / "coverage.json").read_text(encoding="utf-8"))
            if not old.get("errors"):
                continue
            preserve_attempt(dest / "coverage.json")
        coverage = dict(cik=cik, baseline=base, errors=[], captured_at=now())
        try:
            url = filing_url(cik, base["adsh"], base["instance"])
            securities = baseline_securities(cap.get(url))
            coverage.update(baseline_source=url, securities=securities)
            common = [s for s in securities if re.search(r"common|ordinary|capital stock", s["title"], re.I)
                      and not re.search(r"warrant|preferred|depositary|unit", s["title"], re.I)
                      and re.search(r"NASDAQ|NYSE|New York|American", s["exchange"], re.I)
                      and re.fullmatch(r"[A-Z][A-Z.\-]{0,7}", s["ticker"])]
            # Multiple share classes need explicit identity review, not arbitrary selection.
            coverage["identity_status"] = "verified" if len(common) == 1 else "manual_review"
            coverage["historical_ticker"] = common[0]["ticker"] if len(common) == 1 else None
        except Exception as exc:
            coverage["errors"].append("baseline: " + str(exc))
        try:
            main = cap.json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
            archives = []
            archive_names = []
            for f in main.get("filings", {}).get("files", []):
                if f.get("filingTo", "9999") >= "2021-01-01" and f.get("filingFrom", "0000") <= "2026-03-31":
                    archives.append(cap.json("https://data.sec.gov/submissions/" + f["name"]))
                    archive_names.append(f["name"])
            merged = merge_submissions(main, archives)
            merged["sic"] = str(base["sic"])
            save(dest / "submissions.json", merged)
            coverage.update(submission_archives=archive_names, filings=len(filing_rows(merged)),
                            current_tickers=main.get("tickers", []))
            save(dest / "companyfacts.json", cap.json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"))
        except Exception as exc:
            coverage["errors"].append("sec: " + str(exc))
        save(dest / "coverage.json", coverage)
        print(f"SEC {i+1}/{len(cohort)} {cik} {coverage.get('historical_ticker')} {coverage['errors']}", flush=True)


def capture_prices(out):
    import yfinance as yf
    from fundamental.financing_opportunity import ticker_bars
    dest = out / "prices"
    dest.mkdir(exist_ok=True)
    cache = out / "yfinance_cache"
    cache.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(cache))
    names = {"SPY"}
    for path in (out / "issuers").glob("*/coverage.json"):
        cov = json.loads(path.read_text(encoding="utf-8"))
        if cov.get("historical_ticker"):
            names.add(cov["historical_ticker"].replace(".", "-"))
    for i, symbol in enumerate(sorted(names)):
        meta = dest / f"{symbol}.json"
        previous = []
        if meta.exists():
            old = json.loads(meta.read_text(encoding="utf-8"))
            if old.get("status") == "captured":
                continue
            previous = old.pop("previous_attempts", []) + [old]
        record = dict(ticker=symbol, fetched_at=now(), provider="Yahoo/yfinance", start="2022-01-01",
                      end_exclusive=datetime.now(timezone.utc).date().isoformat(), auto_adjust=False, actions=True,
                      previous_attempts=previous)
        try:
            raw = yf.download(symbol, start=record["start"], end=record["end_exclusive"], auto_adjust=False,
                              actions=True, progress=False, threads=False, timeout=30)
            bars = ticker_bars(raw, symbol).dropna(subset=["Close"]) if not raw.empty else pd.DataFrame()
            if bars.empty or "Stock Splits" not in bars.columns:
                raise ValueError("No history or missing split-action coverage")
            bars.to_parquet(dest / f"{symbol}.parquet")
            record.update(status="captured", rows=len(bars), first=str(bars.index[0].date()), last=str(bars.index[-1].date()),
                          sha256=hashlib.sha256((dest / f"{symbol}.parquet").read_bytes()).hexdigest())
        except Exception as exc:
            record.update(status="unavailable", error=str(exc))
        save(meta, record)
        print(f"Prices {i+1}/{len(names)} {symbol} {record['status']}", flush=True)


def capture_events(out, only_cik=None):
    """Broad, complete paginated discovery plus all prospectuses, retaining gaps.

    Search hits and classifiers do not themselves label any outcome as negative.
    Human-reviewed event/coverage files are a separate mandatory gate.
    """
    from bs4 import BeautifulSoup
    from fundamental.financing_history import event_triage
    cap = Capture(out)
    dest = out / "event_discovery"
    dest.mkdir(exist_ok=True)
    forms = EVENT_FORMS
    cohort = pd.read_csv(out / "cohort.csv").to_dict("records")
    if only_cik is not None:
        cohort = [r for r in cohort if int(r["cik"]) == only_cik]
    for i, base in enumerate(cohort):
        cik = int(base["cik"])
        final = dest / f"{cik}.json"
        if final.exists():
            old = json.loads(final.read_text(encoding="utf-8"))
            if old.get("version") == 2 and not old.get("errors"):
                continue
            preserve_attempt(final)
        cov = dict(version=2, cik=cik, searched_at=now(), start="2022-10-01", through="2026-03-31", errors=[], documents=[])
        query = event_search_query(cik, cov["start"], cov["through"])
        # EFTS expects root forms. Including literal 8-K/A silently returns zero
        # for the entire query; root 8-K includes amendments in observed results.
        pending = {}
        try:
            hits, offset, total = [], 0, None
            while total is None or offset < total:
                result = cap.json("https://efts.sec.gov/LATEST/search-index?" + urlencode(dict(query, **{"from": offset, "size": 100})))
                if result.get("timed_out") or result.get("_shards", {}).get("failed", 0):
                    raise ValueError("Incomplete EFTS query")
                count = result["hits"]["total"]
                if count["relation"] != "eq" or count["value"] > 10000:
                    raise ValueError("Truncated EFTS total")
                total = count["value"]
                batch = result["hits"]["hits"]
                if not batch and offset < total:
                    raise ValueError("Premature EFTS pagination end")
                hits.extend(batch)
                offset += len(batch)
            if len({h["_id"] for h in hits}) != total:
                raise ValueError("Duplicate EFTS pages/documents; search coverage is incomplete")
            cov["search_hits"] = total
            for hit in hits:
                src = hit["_source"]
                acc, filename = hit["_id"].split(":", 1)
                kind = src.get("file_type", "")
                if kind in forms or kind.upper().startswith(("EX-99", "EX-10")):
                    url = filing_url(cik, acc, filename)
                    pending[url] = dict(url=url, accession=acc, form=src.get("form"), file_type=kind,
                                        filing_date=src.get("file_date"), discovery="EFTS")
        except Exception as exc:
            cov["errors"].append("search: " + str(exc))
        subpath = out / "issuers" / str(cik) / "submissions.json"
        if subpath.exists():
            filings = filing_rows(json.loads(subpath.read_text(encoding="utf-8")))
            inventory = [f for f in filings if f.get("form") in forms and cov["start"] <= f.get("filingDate", "") <= cov["through"]]
            cov["inventory_filings"] = len(inventory)
            cov["inventory_forms"] = pd.Series([f["form"] for f in inventory], dtype=str).value_counts().to_dict()
            accepted = {f["accessionNumber"]: f.get("acceptanceDateTime") for f in inventory}
            for f in inventory:
                if f["form"].startswith("424") or f["form"] == "FWP":
                    url = filing_url(cik, f["accessionNumber"], f["primaryDocument"])
                    pending.setdefault(url, dict(url=url, accession=f["accessionNumber"], form=f["form"], file_type=f["form"],
                                                 filing_date=f["filingDate"], discovery="all_prospectus_inventory"))
        else:
            accepted = {}
            cov["errors"].append("missing submissions inventory")
        done = set()
        while pending:
            url, row = pending.popitem()
            if url in done:
                continue
            done.add(url)
            row["accepted_at"] = accepted.get(row["accession"])
            try:
                data = cap.get(url)
                soup = BeautifulSoup(data, "lxml")
                for tag in soup(["script", "style", "ix:hidden"]):
                    tag.decompose()
                text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
                digest = hashlib.sha256(url.encode()).hexdigest()
                textpath = dest / f"{digest}.txt"
                textpath.write_text(text, encoding="utf-8")
                row.update(text_file=textpath.name, text_chars=len(text), excerpt=text[:1800],
                           **event_triage(text, row["file_type"]))
                # Exhibits often carry earlier release dates than the 8-K itself.
                if row["file_type"] in ("8-K", "8-K/A"):
                    for link in soup.find_all("a", href=True):
                        href = link["href"]
                        label = link.get_text(" ", strip=True)
                        if re.search(r"99[._-]?\d|ex(?:hibit)?[._-]?99", href, re.I) or re.search(r"99\.\d|press release", label, re.I):
                            target = urljoin(url, href).split("#")[0]
                            if target.startswith(url.rsplit("/", 1)[0] + "/") and target not in done and target != url:
                                pending.setdefault(target, dict(url=target, accession=row["accession"], form=row["form"],
                                    file_type="EX-99-linked", filing_date=row["filing_date"], discovery="8K_exhibit_link"))
            except Exception as exc:
                row["error"] = str(exc)
                cov["errors"].append(f"document: {url}: {exc}")
            cov["documents"].append(row)
        covered_accessions = {d["accession"] for d in cov["documents"] if not d.get("error")}
        cov["unsearched_8k_accessions"] = [f["accessionNumber"] for f in inventory if f["form"].startswith("8-K") and f["accessionNumber"] not in covered_accessions] if subpath.exists() else []
        cov["negative_status"] = "unreviewed; keyword discovery alone cannot establish a negative"
        save(final, cov)
        candidates = sum(d.get("triage") == "primary_equity_candidate" for d in cov["documents"])
        print(f"Events {i+1}/90 {cik}: {len(done)} docs, {candidates} candidates, {len(cov['errors'])} errors", flush=True)


def events_parallel(out):
    # Three independent sessions, each <=2.5 requests/second: aggregate <=7.5/s.
    # No unbounded workers and no other concurrent SEC acquisition in this stage.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cohort = pd.read_csv(out / "cohort.csv")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(capture_events, out, int(c)): int(c) for c in cohort.cik}
        for future in as_completed(futures):
            future.result()


def observations(out):
    import exchange_calendars as xcals
    from fundamental.financing_history import funding_state, small_facts, historical_metrics, group_name
    cal = xcals.get_calendar("XNYS")
    sessions = cal.sessions_in_range("2023-01-01", "2025-12-31")
    dates = pd.Series(sessions, index=sessions).groupby(sessions.strftime("%Y-%m")).last().tolist()
    benchmark = pd.read_parquet(out / "prices" / "SPY.parquet")
    rows, financials = [], []
    for base in pd.read_csv(out / "cohort.csv").to_dict("records"):
        cik = int(base["cik"])
        source = out / "issuers" / str(cik)
        cov = json.loads((source / "coverage.json").read_text(encoding="utf-8"))
        ticker = cov.get("historical_ticker")
        price_identity = "baseline_and_current_SEC_agree" if ticker and ticker in cov.get("current_tickers", []) else "manual_alias_review_required"
        pricefile = out / "prices" / f"{str(ticker).replace('.', '-')}.parquet"
        bars = pd.read_parquet(pricefile) if pricefile.exists() and price_identity == "baseline_and_current_SEC_agree" else pd.DataFrame()
        factsfile = source / "companyfacts.json"
        subfile = source / "submissions.json"
        payload = small_facts(json.loads(factsfile.read_text(encoding="utf-8"))) if factsfile.exists() else None
        subs = json.loads(subfile.read_text(encoding="utf-8")) if subfile.exists() else None
        for session in dates:
            as_of = (cal.session_close(session) + pd.Timedelta(minutes=30)).isoformat()
            row = dict(cik=cik, ticker=ticker, name=base["name"], stratum=base["stratum"], session=str(session.date()), as_of=as_of,
                       financial_status="unavailable", funding_group="unknown", identity_status=cov.get("identity_status", "unknown"),
                       price_identity_status=price_identity)
            row.update(historical_metrics(bars, benchmark, row["session"]))
            if payload and subs:
                fin = funding_state(payload, subs, ticker, as_of)
                financials.append(fin)
                row.update(financial_status=fin["status"], funding_group=fin["funding_group"], assets=fin.get("assets"),
                           cash_only=fin.get("cash_only"), balance_date=fin.get("balance_date"),
                           runway_operating=fin.get("runway_6m"), runway_capex=fin.get("runway_with_capex_6m"),
                           financial_accepted_at=fin.get("filing_accepted_at"))
            row["group"] = group_name(row)
            rows.append(row)
        print(f"Observed {cik} {ticker}", flush=True)
    save(out / "financial_vintages.json", financials)
    save(out / "observations.json", rows)
    pd.DataFrame(rows).to_csv(out / "observations.csv", index=False)
    print(pd.DataFrame(rows).group.value_counts().to_dict(), flush=True)


def analyze(out):
    from collections import Counter
    from fundamental.financing_history import label_window, rate_summary, match_controls, intervening_announcement
    from fundamental.financing_history_report import GROUPS, render
    rows = json.loads((out / "observations.json").read_text(encoding="utf-8"))
    review_path = out / "event_reviews.json"
    reviews = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {"events": [], "windows": [], "signals": []}
    events = reviews["events"]
    if len({r["event_id"] for r in events}) != len(events):
        raise ValueError("Duplicate event IDs in audited registry")
    signal_reviews = {(int(r["cik"]), r["session"]): r for r in reviews.get("signals", [])}
    candidates, discoveries = [], []
    for path in sorted((out / "event_discovery").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("version") != 2:
            continue
        discoveries.append(d)
        for r in d["documents"]:
            candidates.append(dict(cik=d["cik"], **r))
    pd.DataFrame(candidates).to_csv(out / "event_candidates.csv", index=False)
    for row in rows:
        covs = [c for c in reviews.get("windows", []) if c.get("cik") == row["cik"] and c.get("start", "9999") <= row["session"] <= c.get("through", "0000")]
        cov = max(covs, key=lambda c: c.get("through", "")) if covs else {}
        for n in (30, 60, 90):
            value, status, event_id = label_window(row, events, cov, n)
            row.update({f"outcome_{n}": value, f"outcome_status_{n}": status, f"event_{n}": event_id})
        prior = [v for v in events if intervening_announcement(v, row)]
        manual = signal_reviews.get((row["cik"], row["session"]), {})
        row["prior_raise_since_balance"] = bool(prior)
        row["strict_sensitivity_eligible"] = (row["group"] == "strong_short" and not row.get("cash_only")
                    and not prior and not manual.get("exclude_strict") and row.get("outcome_status_60") != "same_day_date_only")
        if manual:
            row["review_note"] = manual["note"]
            row["review_source"] = manual.get("source")
        elif prior:
            row["review_note"] = "Equity raise already announced since balance date; exclude from strict sensitivity"
            row["review_source"] = prior[-1]["sources"][0]
    groups = {}
    for group in GROUPS:
        group_rows = [r for r in rows if r["group"] == group]
        groups[group] = dict(rate_summary(group_rows), total_issuers=len({r["cik"] for r in group_rows}))
    pairs = {c: match_controls(rows, control=c) for c in ("strong_longer", "no_strength_short")}
    cohorts = pd.read_csv(out / "cohort.csv")
    issuer_cov = [json.loads(p.read_text(encoding="utf-8")) for p in (out / "issuers").glob("*/coverage.json")]
    target = [r for r in rows if r["group"] == "strong_short"]
    price_issuers = {r["cik"] for r in rows if r["price_status"] == "complete"}
    counts = dict(companies=len(cohorts), observations=len(rows), target_observations=len(target), target_issuers=len({r["cik"] for r in target}),
                  matched_longer=len(pairs["strong_longer"]), matched_no_strength=len(pairs["no_strength_short"]),
                  usable_price_issuers=len(price_issuers), event_discovery_issuers=len(discoveries),
                  event_documents=len(candidates), audited_events=len(events), strict_target=sum(r["strict_sensitivity_eligible"] for r in rows),
                  strict_issuers=len({r["cik"] for r in rows if r["strict_sensitivity_eligible"]}))
    coverage = [
        dict(label="Frozen historical cohort", value=len(cohorts), note="30 per stratum; selected before outcome collection"),
        dict(label="Original exchange/common-stock identities verified", value=sum(c.get("identity_status") == "verified" for c in issuer_cov), note="Unverified listing/security types retained as gaps"),
        dict(label="SEC issuer and financial captures without errors", value=sum(not c["errors"] for c in issuer_cov), note="Historical submission archives included; not every financial observation is usable"),
        dict(label="Companies with usable historical prices", value=len(price_issuers), note="Original/current SEC symbols agree; enough sessions for at least one signal date"),
        dict(label="Companies without usable historical prices", value=len(cohorts)-len(price_issuers), note="Includes delistings, unavailable symbols and unresolved identities; not replaced"),
        dict(label="Stock-months passing price/liquidity filter", value=sum(bool(r.get("tradable_filter")) for r in rows), note="Historical $3 floor, $5m mean dollar turnover; split correction applied"),
        dict(label="Stock-months with unknown financial group", value=sum(r["funding_group"] == "unknown" for r in rows), note="Missing/stale facts or incomplete capex information; no zero imputation"),
        dict(label="Issuer event searches captured", value=len(discoveries), note="Complete pagination plus prospectus inventory; keyword search is not a negative-label audit"),
        dict(label="Captured event-search documents", value=len(candidates), note="Includes duplicate stages, shelves, ATM, resale and irrelevant hits"),
        dict(label="Event document/search errors", value=sum(len(d["errors"]) for d in discoveries), note="Any unresolved source gaps prevent negative labels"),
        dict(label="Target observations left in strict sensitivity", value=counts["strict_target"], note="Excludes cash-only lower bounds, known intervening funding and ambiguous same-day events; still not a verified equity-need estimate")]
    result = dict(generated_at=now(), conclusion="Inconclusive: sparse matched sample and incomplete event-negative coverage; no validated predictive or trading edge",
                  counts=counts, groups=groups, signals=target, events=events, coverage=coverage, matches=pairs,
                  yearly_target_counts=dict(Counter(r["session"][:4] for r in target)),
                  yearly_groups={year: {g: rate_summary([r for r in rows if r["group"] == g and r["session"].startswith(year)]) for g in GROUPS} for year in ["2023", "2024", "2025"]},
                  strict_sensitivity=rate_summary([r for r in rows if r["strict_sensitivity_eligible"]]))
    save(out / "labeled_observations.json", rows)
    pd.DataFrame(rows).to_csv(out / "labeled_observations.csv", index=False)
    save(out / "analysis.json", result)
    report = out / "financing_history.html"
    report.write_text(render(result), encoding="utf-8")
    paths = [Path("fundamental/financing_history.py"), Path("fundamental/financing_history_report.py"),
             Path("scripts/build_financing_history.py"), Path("fundamental/cash_runway.py"), Path("fundamental/financing_opportunity.py")]
    save(out / "manifest.json", dict(run_id="financing-history-" + out.name, generated_at=now(), scope="local research artifacts only",
        source_hashes={str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        outputs={"report": {"path": str(report), "sha256": hashlib.sha256(report.read_bytes()).hexdigest()}},
        artifact_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("*") if p.is_file() and p.suffix in (".json", ".csv") and p.name != "manifest.json"},
        counts=counts, completion_status="RESEARCH_COMPLETE_QA_PENDING", qa={"visual_status": "PENDING"}))
    print(json.dumps(counts, indent=2), flush=True)


def verify_frozen_inputs(out):
    manifest = json.loads((out / "cohort_manifest.json").read_text(encoding="utf-8"))
    for name in ("cohort", "protocol"):
        suffix = ".csv" if name == "cohort" else ".json"
        if hashlib.sha256((out / (name + suffix)).read_bytes()).hexdigest() != manifest[name + "_sha256"]:
            raise ValueError("Frozen " + name + " changed; use a new study directory")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    stages = {"universe": universe, "sec": capture_sec, "prices": capture_prices, "events": events_parallel, "observations": observations, "report": analyze}
    p.add_argument("--stage", choices=list(stages), required=True)
    args = p.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / "artifacts"):
        p.error("Output must be under this repository's artifacts directory")
    out.mkdir(parents=True, exist_ok=True)
    if args.stage != "universe":
        verify_frozen_inputs(out)
    stages[args.stage](out)


if __name__ == "__main__":
    main()
