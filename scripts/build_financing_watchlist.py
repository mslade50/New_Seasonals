"""Free local financing-opportunity research screen, with resumable captures.

Stage prices first, fundamentals second, then a source-linked review report.
Uses artifacts only; never changes the strategy book or canonical data caches.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import exchange_calendars as xc
import yfinance as yf

from fundamental.cash_runway import calculate_runway
from fundamental.financing_opportunity import (
    VERSION, ScreenPolicy, choose_financial_queue, join_funding, price_metrics, ticker_bars,
)
from public_market_sources import parse_listing_directory
from fundamental.sec import SECClient
from scripts.build_cash_runway_watchlist import Capture, dump, scan_financing


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def last_completed_session(as_of):
    now = pd.Timestamp(as_of)
    if now.tzinfo is None:
        raise ValueError("Research cutoff must have a timezone")
    schedule = xc.get_calendar("XNYS").schedule.loc[
        str((now - pd.Timedelta(days=14)).date()):str(now.date())]
    completed = schedule[schedule["close"] + pd.Timedelta(minutes=30) <= now]
    return completed.index[-1].date().isoformat()


class ResumableCapture(Capture):
    def __init__(self, output):
        self.output = output
        self.raw = output / "raw"
        self.raw.mkdir(exist_ok=True)
        self.entries = []
        self.source_dir = None
        self.previous = {}
        self.client = SECClient(sleep_seconds=.3)


def establish(output):
    output.mkdir(parents=True, exist_ok=False)
    capture = Capture(output)
    exchange = capture.get_json("sec_exchange.json", "https://www.sec.gov/files/company_tickers_exchange.json")
    mapping = {r["ticker"].replace(".", "-"): r for r in
        (dict(zip(exchange["fields"], values)) for values in exchange["data"])}
    listings = []
    for name, nasdaq in [("nasdaqlisted.txt", True), ("otherlisted.txt", False)]:
        raw = capture.get(name, f"https://www.nasdaqtrader.com/dynamic/SymDir/{name}")
        listings += parse_listing_directory(raw.decode("utf-8"), nasdaq=nasdaq).to_dict("records")
    rows = []
    for r in listings:
        reasons = set(filter(None, r["listing_reasons"].split(";")))
        if reasons - {"financial_status_not_normal"} or r["ticker"] not in mapping:
            continue
        rows.append(dict(ticker=r["ticker"], company_name=mapping[r["ticker"]]["name"],
            cik=int(mapping[r["ticker"]]["cik"]), exchange=r["exchange"],
            listing_warning=r["listing_reasons"], listing_as_of=r["listing_as_of"].isoformat()))
    rows.sort(key=lambda r: r["ticker"])
    dump(output / "universe.json", rows)
    now = datetime.now(timezone.utc).isoformat()
    manifest = dict(version=VERSION, as_of=now, price_session=last_completed_session(now),
        discovered=len(rows), policy=ScreenPolicy().to_dict(),
        source="SEC + Nasdaq Trader listings + Yahoo Finance via yfinance",
        research_only=True, historical_backtest_ready=False,
        scope="Current Nasdaq/NYSE/AMEX common/ordinary shares mapped to SEC; excludes funds, depositary securities, units and unverified types.")
    dump(output / "manifest.json", manifest)


def download_prices(output, batch_size=75):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    universe = json.loads((output / "universe.json").read_text(encoding="utf-8"))
    price_dir = output / "prices"
    price_dir.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(output / "yahoo_cache"))
    tickers = ["SPY"] + [r["ticker"] for r in universe if r["ticker"] != "SPY"]
    end = pd.Timestamp(manifest["price_session"]) + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=390)
    entries = json.loads((output / "price_captures.json").read_text(encoding="utf-8")) if (output / "price_captures.json").exists() else {}
    for pos in range(0, len(tickers), batch_size):
        group = [t for t in tickers[pos:pos + batch_size] if t not in entries]
        if not group:
            continue
        try:
            data = yf.download(group, start=start.date().isoformat(), end=end.date().isoformat(),
                auto_adjust=False, actions=True, threads=4, progress=False, timeout=20)
            for ticker in group:
                bars = ticker_bars(data, ticker)
                if bars.empty or "Close" not in bars or bars.Close.dropna().empty:
                    entries[ticker] = dict(status="unavailable", fetched_at=datetime.now(timezone.utc).isoformat())
                    continue
                path = price_dir / f"{ticker}.parquet"
                bars.to_parquet(path)
                entries[ticker] = dict(status="captured", sha256=sha(path),
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    source=f"https://finance.yahoo.com/quote/{ticker}/history/", start=str(start.date()), end_exclusive=str(end.date()))
        except Exception as exc:
            for ticker in group:
                entries[ticker] = dict(status="unavailable", error=type(exc).__name__)
        dump(output / "price_captures.json", entries)
        print(f"Prices {min(pos+batch_size,len(tickers))}/{len(tickers)}; captured {sum(e['status']=='captured' for e in entries.values())}", flush=True)
        time.sleep(.5)


def screen_prices(output):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    captures = json.loads((output / "price_captures.json").read_text(encoding="utf-8"))
    def read(ticker):
        entry = captures.get(ticker, {})
        if entry.get("status") != "captured":
            return pd.DataFrame()
        path = output / "prices" / f"{ticker}.parquet"
        if sha(path) != entry["sha256"]:
            raise ValueError(f"Price capture hash mismatch: {ticker}")
        return pd.read_parquet(path)
    benchmark = read("SPY").dropna(subset=["Adj Close"])
    rows = []
    for row in json.loads((output / "universe.json").read_text(encoding="utf-8")):
        row.update(price_metrics(read(row["ticker"]), benchmark, session=manifest["price_session"]))
        row["price_source"] = captures.get(row["ticker"], {}).get("source")
        row["price_fetched_at"] = captures.get(row["ticker"], {}).get("fetched_at")
        rows.append(row)
    dump(output / "price_screen.json", rows)
    print(json.dumps({"universe":len(rows), "complete":sum(r["price_status"]=="complete" for r in rows),
        "tradable":sum(bool(r.get("tradable_filter")) for r in rows),
        "setups":sum(bool(r["setups"]) for r in rows)}, indent=2))
    return rows


def fundamentals(output, limit, documents):
    rows = screen_prices(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    selected = choose_financial_queue(rows, limit)
    dump(output / "financial_queue.json", selected)
    folder = output / "sec"
    folder.mkdir(exist_ok=True)
    capture = ResumableCapture(folder)
    # Resume only hash-verified captures; never overwrite captured bytes.
    previous = json.loads((folder / "captures.json").read_text(encoding="utf-8")) if (folder / "captures.json").exists() else []
    capture.entries = previous
    old_get = capture.get
    def get(name, url):
        known = next((e for e in capture.entries if e["name"] == name), None)
        if known:
            path = folder / "raw" / name
            if known["source"] != url or sha(path) != known["sha256"]:
                raise ValueError(f"SEC capture mismatch: {name}")
            return path.read_bytes()
        return old_get(name, url)
    capture.get = get
    financials = json.loads((output / "financials.json").read_text(encoding="utf-8")) if (output / "financials.json").exists() else {}
    by_ticker = {r["ticker"]: r for r in rows}
    for i, ticker in enumerate(selected):
        if ticker in financials:
            continue
        cik = by_ticker[ticker]["cik"]
        try:
            sub = capture.get_json(f"{ticker}_submissions.json", f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
            if ticker not in {t.replace(".", "-") for t in sub.get("tickers", [])}:
                raise ValueError("Current SEC ticker mismatch")
            facts = capture.get_json(f"{ticker}_companyfacts.json", f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
            if int(facts["cik"]) != cik:
                raise ValueError("CIK mismatch")
            financial = calculate_runway(facts, sub, ticker=ticker, as_of=manifest["as_of"])
            if documents and join_funding(by_ticker[ticker], financial)["candidate"]:
                scan_financing(capture, financial, sub, manifest["as_of"], max_filings=12)
        except Exception as exc:
            financial = dict(ticker=ticker, status="unavailable", warnings=[str(exc) if isinstance(exc, ValueError) else type(exc).__name__])
        financials[ticker] = financial
        dump(output / "financials.json", financials)
        print(f"SEC {i+1}/{len(selected)} {ticker}: {financial['status']}; {financial.get('runway_6m')}", flush=True)
    return [join_funding(row, financials.get(row["ticker"])) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["prices", "fundamentals", "report"], required=True)
    parser.add_argument("--financial-limit", type=int, default=150)
    parser.add_argument("--documents", action="store_true")
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (ROOT / "artifacts").resolve() not in output.parents:
        parser.error("Output must be under repository artifacts/")
    if not 1 <= args.financial_limit <= 300:
        parser.error("Financial limit must be 1..300")
    if args.stage == "prices":
        if not output.exists():
            establish(output)
        download_prices(output)
        screen_prices(output)
    elif args.stage == "fundamentals":
        rows = fundamentals(output, args.financial_limit, args.documents)
        dump(output / "watchlist.json", rows)
    else:
        from fundamental.financing_opportunity_report import build_report
        build_report(output, args.reviews)


if __name__ == "__main__":
    main()
