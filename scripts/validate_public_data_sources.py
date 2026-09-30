"""Bounded FMP-free listing/research/earnings evidence validation under artifacts.

Writes raw captures, candidate statements, metrics, and coverage gaps. Does not
promote research, change trading universes, upload data, or confirm earnings from
an estimated date. Yahoo news/consensus remain optional secondary inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fundamental.sec import SECClient
from fundamental.sec_statements import annual_statement_bundle, filing_news
from fundamental.metrics import calculate_ticker_metrics
from public_market_sources import (parse_listing_directory, symbol_candidate,
    normalize_yahoo_news, share_structure, reported_earnings_dates)
from scripts.build_official_macro_releases import artifact_output


def run(output, tickers, *, source_dir=None, baseline_symbols=None):
    now = pd.Timestamp.now(tz="UTC")
    raw_dir = output / "raw"
    raw_dir.mkdir()
    yf.set_tz_cache_location(str(output / "yf_cache"))
    captures, symbols, metrics, bundles, summaries = [], [], [], [], []
    client = SECClient() if not source_dir else None

    def capture(name, source, fetch):
        if source_dir:
            raw = (Path(source_dir) / name).read_bytes()
            prior = json.loads((Path(source_dir).parent / "manifest.json").read_text(encoding="utf-8"))
            entry = next(v for v in prior["captures"] if v["name"] == name)
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError(f"replay digest mismatch: {name}")
            stamp = pd.Timestamp(entry["fetched_at"])
        else:
            value = fetch()
            raw = value if isinstance(value, bytes) else json.dumps(value, default=str).encode()
            stamp = pd.Timestamp.now(tz="UTC")
        (raw_dir / name).write_bytes(raw)
        captures.append(dict(name=name, source=source, sha256=hashlib.sha256(raw).hexdigest(), fetched_at=stamp.isoformat()))
        return raw, stamp

    def http(url):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content

    listings = []
    for filename, nasdaq in [("nasdaqlisted.txt", True), ("otherlisted.txt", False)]:
        url = f"https://www.nasdaqtrader.com/dynamic/SymDir/{filename}"
        raw, _ = capture(filename, url, lambda: http(url))
        listings.append(parse_listing_directory(raw.decode(), nasdaq=nasdaq))
    listed = pd.concat(listings, ignore_index=True)
    if listed.ticker.duplicated().any():
        raise ValueError("cross-directory symbol collision")
    listed.to_parquet(output / "listing_candidates.parquet", index=False)
    raw, _ = capture("sec_tickers.json", "https://www.sec.gov/files/company_tickers.json",
                     lambda: client.ticker_map())
    ciks = json.loads(raw)
    for ticker in tickers:
        summary = {"ticker": ticker, "status": "partial", "gaps": []}
        yahoo = yf.Ticker(ticker)
        info = {}
        try:
            raw, fetched = capture(f"{ticker}_info.json", f"https://finance.yahoo.com/quote/{ticker}/profile/", yahoo.get_info)
            info = json.loads(raw)
            match = listed[listed.ticker.eq(ticker)]
            if len(match) != 1:
                raise ValueError("ticker missing or ambiguous in listing directory")
            candidate = symbol_candidate(match.iloc[0].to_dict(), info, fetched_at=fetched)
            symbols.append(candidate)
            summary["universe_eligible"] = candidate["eligible"]
            summary["share_structure"] = share_structure(info)
        except Exception as exc:
            summary["gaps"].append(f"metadata: {type(exc).__name__}: {exc}")
        try:
            cik = ciks[ticker]
            raw_sub, _ = capture(f"{ticker}_submissions.json", f"https://data.sec.gov/submissions/CIK{cik:010d}.json", lambda: client.submissions(cik))
            raw_facts, stamp = capture(f"{ticker}_facts.json", f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", lambda: client.companyfacts(cik))
            submissions, facts = json.loads(raw_sub), json.loads(raw_facts)
            bundle, report = annual_statement_bundle(facts, submissions, ticker=ticker,
                as_of=stamp, digest=hashlib.sha256(raw_facts).hexdigest())
            bundles.append(bundle)
            metric = calculate_ticker_metrics(bundle, ticker=ticker,
                market_cap=info.get("marketCap"), company_name=info.get("longName"),
                sector=info.get("sector"), industry=info.get("industry"))
            metrics.append(metric)
            summary["financials"] = report
            summary["missing_metrics"] = [k for k in ("roic", "net_debt_to_ebitda", "revenue_cagr_3y", "fcf_margin", "share_count_cagr_3y")
                if pd.isna(metric.get(k))]
            (output / f"{ticker}_filing_news.json").write_text(json.dumps(filing_news(submissions, ticker=ticker, as_of=stamp), indent=2), encoding="utf-8")
        except Exception as exc:
            summary["gaps"].append(f"SEC financials: {type(exc).__name__}: {exc}")
        for kind, fetch in [("news", lambda: yahoo.get_news(count=10)),
                            ("estimates", lambda: json.loads(yahoo.get_earnings_estimate().to_json(orient="table", date_format="iso"))),
                            ("revenue_estimates", lambda: json.loads(yahoo.get_revenue_estimate().to_json(orient="table", date_format="iso"))),
                            ("earnings_dates", lambda: json.loads(yahoo.get_earnings_dates(limit=24).to_json(orient="table", date_format="iso")))]:
            try:
                raw, stamp = capture(f"{ticker}_{kind}.json", f"https://finance.yahoo.com/quote/{ticker}/", fetch)
                payload = json.loads(raw)
                if kind == "news":
                    normalized = normalize_yahoo_news(payload, ticker=ticker,
                        company_name=info.get("shortName", ticker), as_of=stamp)
                    summary["news"] = {"raw_items": len(payload), "issuer_relevant_items": len(normalized)}
                    (output / f"{ticker}_news.json").write_text(json.dumps(normalized, indent=2), encoding="utf-8")
                elif kind in {"estimates", "revenue_estimates"}:
                    annual = [r for r in payload.get("data", []) if r.get("period") in {"0y", "+1y"} or r.get("index") in {"0y", "+1y"}]
                    summary[f"annual_{kind}_rows"] = len(annual)
                    summary["estimate_scope"] = "current/next FY EPS and revenue; not a ten-year vendor forecast history"
                    if len(annual) != 2 or any(r.get("avg") is None or not r.get("numberOfAnalysts") for r in annual):
                        summary["gaps"].append(f"annual {kind} consensus incomplete")
                else:
                    from io import StringIO
                    frame = pd.read_json(StringIO(raw.decode()), orient="table")
                    confirmed = reported_earnings_dates(frame, as_of=stamp)
                    (output / f"{ticker}_reported_dates.json").write_text(json.dumps(confirmed, indent=2), encoding="utf-8")
                    summary["secondary_reported_dates"] = len(confirmed)
                    if not confirmed: summary["gaps"].append("no secondary reported earnings dates")
            except Exception as exc:
                summary["gaps"].append(f"{kind}: {type(exc).__name__}: {exc}")
        summary["status"] = "captured_with_gaps" if summary["gaps"] or summary.get("missing_metrics") else "captured"
        summaries.append(summary)
        print(f"{ticker}: {summary['status']}; {len(summary['gaps'])} source gaps", flush=True)
    if symbols: pd.DataFrame(symbols).to_parquet(output / "symbol_master_sample.parquet", index=False)
    if bundles: pd.concat(bundles, ignore_index=True).to_parquet(output / "sec_statements.parquet", index=False)
    if metrics: pd.DataFrame(metrics).to_parquet(output / "sec_metrics.parquet", index=False)
    coverage = None
    if baseline_symbols:
        baseline = pd.read_parquet(baseline_symbols)
        joined = baseline[["ticker"]].merge(listed, on="ticker", how="left", indicator=True)
        joined.to_csv(output / "baseline_listing_coverage.csv", index=False)
        coverage = {"baseline_symbols": len(baseline), "listed": int(joined._merge.eq("both").sum()),
            "common_stock_verified": int(joined.listing_eligible.eq(True).sum()),
            "warning": "listing identity only; full-universe fresh metadata/price validation still required"}
    report = {"captured_at": now.isoformat(), "captures": captures, "companies": summaries,
        "listing_rows": len(listed), "listing_candidates": int(listed.listing_eligible.sum()),
        "baseline_coverage": coverage, "production_ready": False,
        "pending_gates": ["broad universe validation", "issuer accounting mappings and missing metrics",
            "primary earnings confirmation/date-change adjudication", "historical/new-listing bootstrap",
            "scheduled reliability and reviewed provider activation"]}
    (output / "manifest.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--source-dir", type=Path)
    ap.add_argument("--baseline-symbols", type=Path)
    args = ap.parse_args()
    tickers = sorted({t.upper().replace(".", "-") for t in args.tickers})
    if len(tickers) > 25 or any(not re.fullmatch(r"[A-Z]{1,5}(?:-[AB])?", t) for t in tickers):
        raise SystemExit("use at most 25 explicit common-stock tickers per validation run")
    report = run(artifact_output(args.output_dir), tickers, source_dir=args.source_dir, baseline_symbols=args.baseline_symbols)
    print(json.dumps({k: report[k] for k in ("listing_rows", "listing_candidates", "baseline_coverage", "production_ready")}, indent=2))
    return 0 if all(not c["gaps"] for c in report["companies"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
