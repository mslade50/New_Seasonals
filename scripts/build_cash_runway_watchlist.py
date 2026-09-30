"""Build a bounded free SEC cash-runway pilot under artifacts, or replay captures.

No FMP, broker, email, uploads, deployment, or authoritative research-state writes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests
from bs4 import BeautifulSoup

from fundamental.sec import SECClient
from fundamental.cash_runway import (VERSION, apply_manual_review, calculate_runway, filing_rows,
    filing_url, financing_excerpts, utc)
from public_market_sources import parse_listing_directory


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


class Capture:
    def __init__(self, output, source_dir=None):
        self.output = output
        self.raw = output / "raw"
        self.raw.mkdir()
        self.entries = []
        self.source_dir = source_dir
        self.client = None if source_dir else SECClient(sleep_seconds=0.3)
        self.previous = {}
        if source_dir:
            previous = json.loads((source_dir / "captures.json").read_text(encoding="utf-8"))
            self.previous = {r["name"]: r for r in previous}

    def get(self, name, url):
        if self.source_dir:
            entry = self.previous[name]
            raw = (self.source_dir / "raw" / name).read_bytes()
            if entry["source"] != url or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError(f"Replay provenance mismatch for {name}")
            entry = dict(entry)
        else:
            # No retry storm. One bounded retry for a transient server/rate error.
            for attempt in range(2):
                response = self.client.session.get(url,
                    headers={"User-Agent": self.client.user_agent, "Accept-Encoding": "gzip, deflate"},
                    timeout=30)
                time.sleep(0.3)
                if response.status_code in {429, 500, 502, 503, 504} and attempt == 0:
                    time.sleep(2)
                    continue
                response.raise_for_status()
                break
            raw = response.content
            if len(raw) > 20_000_000:
                raise ValueError("Document exceeds 20 MB bounded capture")
            entry = dict(name=name, source=url, sha256=hashlib.sha256(raw).hexdigest(),
                         fetched_at=datetime.now(timezone.utc).isoformat())
        (self.raw / name).write_bytes(raw)
        self.entries.append(entry)
        dump(self.output / "captures.json", self.entries)
        return raw

    def get_json(self, name, url):
        return json.loads(self.get(name, url))


def plain_html(raw):
    soup = BeautifulSoup(raw, "html.parser")
    for node in soup(["script", "style", "ix:header"]):
        node.decompose()
    return soup.get_text(" ", strip=True)


def scan_financing(capture, row, submissions, as_of, max_filings):
    if not row.get("balance_date"):
        return
    eligible = []
    for item in filing_rows(submissions):
        try:
            stamp = utc(item["acceptanceDateTime"])
        except (KeyError, ValueError, TypeError):
            continue
        if not row["balance_date"] < stamp.date().isoformat() or stamp > utc(as_of):
            continue
        if item.get("form") in {"8-K", "8-K/A", "6-K", "6-K/A", "FWP"} or item.get("form", "").startswith(("424B", "S-3", "F-3", "S-1", "F-1")):
            eligible.append(item)
    # Latest financial report includes subsequent-events notes, which can record
    # cash receipts before its own publication; include it in discovery too.
    sources, errors = [], []
    docs = [(row["filing_url"], "latest financial report", row["filing_accepted_at"])]
    for item in sorted(eligible, key=lambda x: x["acceptanceDateTime"], reverse=True)[:max_filings]:
        docs.append((filing_url(row["cik"], item["accessionNumber"], item.get("primaryDocument")), item["form"], item["acceptanceDateTime"]))
    for i, (url, form, stamp) in enumerate(docs):
        name = f"{row['ticker']}_document_{i:02d}.html"
        try:
            text = plain_html(capture.get(name, url))
            sources.append(dict(url=url, form=form, accepted_at=stamp,
                capture=name, excerpts=financing_excerpts(text), text_length=len(text)))
        except Exception as exc:
            errors.append(dict(url=url, error=type(exc).__name__))
    row["financing_documents"] = sources
    row["financing_errors"] = errors
    row["financing_filings_found"] = len(eligible)
    row["financing_filings_checked"] = max(0, len(sources) - int(bool(sources) and sources[0]["form"] == "latest financial report"))
    row["financing_scan_truncated"] = len(eligible) > max_filings
    row["financing_status"] = "Document review required" if any(d["excerpts"] for d in sources) else "No keyword hit; not verified"
    if errors or row["financing_scan_truncated"]:
        row["financing_status"] = "Incomplete document coverage; review required"


def run(args):
    output = args.output_dir.resolve()
    artifact_root = (ROOT / "artifacts").resolve()
    if artifact_root not in output.parents:
        raise ValueError("Output must be a new directory below repository artifacts/")
    output.mkdir(parents=True, exist_ok=False)
    capture = Capture(output, args.source_dir)
    config = json.loads(args.universe.read_text(encoding="utf-8"))
    tickers = args.tickers or config["tickers"]
    if len(tickers) > 150 or len(tickers) != len(set(tickers)) or any(not re.fullmatch(r"[A-Z]{1,5}(?:-[AB])?", t) for t in tickers):
        raise ValueError("Use up to 150 unique explicit common-stock symbols")
    if args.source_dir:
        source_manifest = json.loads((args.source_dir / "manifest.json").read_text(encoding="utf-8"))
        as_of = args.as_of or source_manifest["as_of"]
    else:
        as_of = args.as_of or datetime.now(timezone.utc).isoformat()
    utc(as_of)
    dump(output / "universe.json", {"description": config["description"], "tickers": tickers})
    exchange = capture.get_json("sec_exchange.json", "https://www.sec.gov/files/company_tickers_exchange.json")
    mapping = {r["ticker"].replace(".", "-"): r for r in
               (dict(zip(exchange["fields"], values)) for values in exchange["data"])}
    directories = []
    for name, nasdaq in [("nasdaqlisted.txt", True), ("otherlisted.txt", False)]:
        raw = capture.get(name, f"https://www.nasdaqtrader.com/dynamic/SymDir/{name}")
        directories.extend(parse_listing_directory(raw.decode("utf-8"), nasdaq=nasdaq).to_dict("records"))
    listings = {r["ticker"]: r for r in directories}
    rows = []
    reviews = {r["ticker"]: r for r in json.loads(args.reviews.read_text(encoding="utf-8"))} if args.reviews else {}
    audits = []
    for i, ticker in enumerate(tickers):
        row = dict(ticker=ticker, company_name=mapping.get(ticker, {}).get("name", ticker),
                   status="unavailable", warnings=[], financing_status="Not reviewed")
        try:
            if ticker not in mapping or ticker not in listings:
                raise ValueError("Current SEC/listing identity unavailable; no silent ticker substitution")
            listing = listings[ticker]
            reasons = set(filter(None, listing["listing_reasons"].split(";")))
            # Distressed companies are intentionally relevant. Abnormal Nasdaq
            # financial status is a warning, not an automatic research exclusion.
            if reasons - {"financial_status_not_normal"}:
                raise ValueError("Common-stock identity not established: " + listing["listing_reasons"])
            age = utc(as_of) - listing["listing_as_of"].to_pydatetime()
            if abs(age.total_seconds()) > 4 * 86400:
                raise ValueError("Listing snapshot is too far from research cutoff")
            cik = int(mapping[ticker]["cik"])
            submissions = capture.get_json(f"{ticker}_submissions.json", f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
            if ticker not in {t.replace(".", "-") for t in submissions.get("tickers", [])}:
                raise ValueError("SEC submissions ticker identity mismatch")
            facts = capture.get_json(f"{ticker}_companyfacts.json", f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
            if int(facts["cik"]) != cik:
                raise ValueError("SEC companyfacts identity mismatch")
            row = calculate_runway(facts, submissions, ticker=ticker, as_of=as_of)
            row.update(exchange=listing["exchange"], listing_as_of=listing["listing_as_of"].isoformat(),
                       listing_warning=listing["listing_reasons"])
            if args.documents:
                scan_financing(capture, row, submissions, as_of, args.max_filings)
            if ticker in reviews:
                audits.append(apply_manual_review(row, reviews[ticker]))
        except Exception as exc:
            # Retain coverage failures as rows. Never silently lose a distressed
            # or delisted issuer because its source cannot be fetched/normalized.
            row["warnings"].append(str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
        rows.append(row)
        dump(output / "watchlist.json", rows)
        print(f"{i+1}/{len(tickers)} {ticker}: {row['status']}; {row.get('bucket', 'Unavailable')}", flush=True)
    manifest = dict(version=VERSION, as_of=as_of, completed_at=datetime.now(timezone.utc).isoformat(),
                    requested=len(tickers), calculated=sum(r["status"] == "calculated" for r in rows),
                    source_mode="archived replay" if args.source_dir else "live free SEC + Nasdaq listing directory",
                    documents_checked=args.documents, max_filings=args.max_filings,
                    research_only=True, historical_backtest_ready=False,
                    scope=config["description"], capture_count=len(capture.entries))
    manifest["manual_checks"] = len(audits)
    manifest["manual_checks_passed"] = sum(r["check_status"] == "PASS" for r in audits)
    manifest["source_hashes"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ["fundamental/cash_runway.py", "fundamental/cash_runway_report.py",
                     "scripts/build_cash_runway_watchlist.py"]}
    manifest["universe_sha256"] = hashlib.sha256(args.universe.read_bytes()).hexdigest()
    manifest["review_sha256"] = hashlib.sha256(args.reviews.read_bytes()).hexdigest() if args.reviews else None
    if args.reviews:
        dump(output / "manual_reviews.json", audits)
    dump(output / "manifest.json", manifest)
    flat = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} |
            {"warnings": "; ".join(r["warnings"]),
             "manual_check_status": r.get("manual_review", {}).get("check_status"),
             "manual_accounting_note": r.get("manual_review", {}).get("accounting_note"),
             "management_runway": r.get("manual_review", {}).get("management_runway"),
             "post_balance_financing_note": r.get("manual_review", {}).get("financing_note"),
             "manual_source_urls": " | ".join(s["url"] for s in r.get("manual_review", {}).get("sources", []))} for r in rows]
    pd.DataFrame(flat).to_csv(output / "watchlist.csv", index=False)
    from fundamental.cash_runway_report import render_report
    (output / "cash_runway.html").write_text(render_report(rows, manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["calculated"] and manifest["manual_checks"] == manifest["manual_checks_passed"] and len(audits) == len(reviews) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--universe", type=Path, default=ROOT / "config/cash_runway_pilot.json")
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--source-dir", type=Path, help="Prior complete run; validates capture hashes and reuses its cutoff")
    parser.add_argument("--reviews", type=Path, help="Dated, source-linked manual checks; mismatches remain visible")
    parser.add_argument("--as-of", help="Timezone-aware cutoff; defaults to current capture time")
    parser.add_argument("--documents", action="store_true", help="Scan latest financial report and subsequent financing-related filings")
    parser.add_argument("--max-filings", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.max_filings <= 30:
        parser.error("--max-filings must be 1..30")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
