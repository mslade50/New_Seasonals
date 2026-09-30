"""Bounded primary earnings-date collector, with no FMP/R2/production writes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import time
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fundamental.sec import SECClient
from fundamental.sec_earnings import parse_earnings_8k
from scripts.build_official_macro_releases import artifact_output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--filings-per-ticker", type=int, default=4)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    tickers = sorted(set(t.upper().replace(".", "-") for t in args.tickers))
    if len(tickers) > 25 or not 1 <= args.filings_per_ticker <= 20 or any(not re.fullmatch(r"[A-Z]{1,5}(?:-[AB])?", t) for t in tickers):
        raise SystemExit("bounded validation requires <=25 common-stock tickers and <=20 filings each")
    out = artifact_output(args.output_dir); client = SECClient(); ciks = client.ticker_map()
    rows, gaps = [], []
    for ticker in tickers:
        try:
            cik = ciks[ticker]; submissions = client.submissions(cik)
            (out / f"{ticker}_submissions.json").write_text(json.dumps(submissions), encoding="utf-8")
            recent = submissions.get("filings", {}).get("recent", {})
            indices = [i for i, form in enumerate(recent.get("form", []))
                       if form == "8-K" and "2.02" in recent["items"][i]][:args.filings_per_ticker]
            if not indices: gaps.append(f"{ticker}: no recent 8-K Item 2.02; foreign/other forms require separate parsing")
            for i in indices:
                accession = recent["accessionNumber"][i]
                url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{recent['primaryDocument'][i]}"
                try:
                    response = client.session.get(url, headers={"User-Agent": client.user_agent}, timeout=client.timeout)
                    time.sleep(client.sleep_seconds)
                    response.raise_for_status()
                    (out / f"{ticker}_{accession}.html").write_bytes(response.content)
                    rows.append(parse_earnings_8k(response.content, ticker=ticker, source_url=url,
                        accepted_at=recent["acceptanceDateTime"][i], captured_at=pd.Timestamp.now(tz="UTC")))
                except Exception as exc:
                    gaps.append(f"{ticker} {accession}: {type(exc).__name__}: {exc}")
        except Exception as exc:
            gaps.append(f"{ticker}: {type(exc).__name__}: {exc}")
    if rows:
        frame = pd.DataFrame(rows).drop_duplicates(["ticker", "date", "fiscalDateEnding"])
        frame.to_parquet(out / "confirmed_earnings.parquet", index=False)
    report = {"confirmations": len(rows), "gaps": gaps, "requested_tickers": tickers,
              "production_ready": False, "scope": "explicit announcement dates; no EPS/revenue estimates or surprise data"}
    (out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 2 if gaps else 0


if __name__ == "__main__":
    raise SystemExit(main())
