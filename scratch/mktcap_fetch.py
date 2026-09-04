"""Fetch daily historical market cap from FMP for the equity universe.

One profile call per ticker (skip ETFs/funds/trusts), then two
historical-market-capitalization windows (endpoint caps at 5,000 rows).
Writes scratch/mktcap_history.parquet + scratch/mktcap_skiplist.csv.
Resumable: tickers already in the output parquet are skipped on rerun.
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from scripts.build_earnings_calendar import load_env
from strategy_config import CSV_UNIVERSE

KEY = load_env()
BASE = "https://financialmodelingprep.com/stable"
OUT = Path(_ROOT) / "scratch" / "mktcap_history.parquet"
SKIP_OUT = Path(_ROOT) / "scratch" / "mktcap_skiplist.csv"
SLEEP = 0.12
WINDOWS = [("2000-01-01", "2009-12-31"), ("2010-01-01", "2026-07-16")]


def get(path: str, **params) -> list | dict | None:
    params["apikey"] = KEY
    for attempt in range(3):
        try:
            r = requests.get(f"{BASE}/{path}", params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if r.ok:
                return r.json()
            return None
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return None


def main() -> None:
    tickers = sorted({t for t in CSV_UNIVERSE
                      if not t.startswith("^") and "=" not in t and "-USD" not in t})
    done: set[str] = set()
    frames: list[pd.DataFrame] = []
    if OUT.exists():
        prior = pd.read_parquet(OUT)
        frames.append(prior)
        done |= set(prior["ticker"].unique())
    skips: list[dict] = []
    if SKIP_OUT.exists():
        prior_skips = pd.read_csv(SKIP_OUT)
        skips = prior_skips.to_dict("records")
        done |= set(prior_skips["ticker"])

    todo = [t for t in tickers if t not in done]
    print(f"universe {len(tickers)}, already done {len(done)}, fetching {len(todo)}")

    for n, ticker in enumerate(todo, 1):
        profile = get("profile", symbol=ticker)
        time.sleep(SLEEP)
        prof = profile[0] if isinstance(profile, list) and profile else {}
        if not prof:
            skips.append({"ticker": ticker, "reason": "no_profile"})
            continue
        if prof.get("isEtf") or prof.get("isFund"):
            skips.append({"ticker": ticker, "reason": "etf_or_fund"})
            continue

        rows: list[dict] = []
        for start, end in WINDOWS:
            data = get("historical-market-capitalization",
                       symbol=ticker, limit=5000, **{"from": start, "to": end})
            time.sleep(SLEEP)
            if isinstance(data, list):
                rows.extend(data)
        if not rows:
            skips.append({"ticker": ticker, "reason": "no_mktcap"})
            continue
        df = pd.DataFrame(rows)[["date", "marketCap"]]
        df["ticker"] = ticker
        frames.append(df)

        if n % 50 == 0 or n == len(todo):
            combined = pd.concat(frames, ignore_index=True).drop_duplicates(
                ["ticker", "date"])
            combined.to_parquet(OUT, index=False)
            pd.DataFrame(skips).to_csv(SKIP_OUT, index=False)
            print(f"  {n}/{len(todo)} fetched "
                  f"({len(combined['ticker'].unique())} tickers, "
                  f"{len(combined):,} rows, {len(skips)} skipped)")

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(["ticker", "date"])
    combined.to_parquet(OUT, index=False)
    pd.DataFrame(skips).to_csv(SKIP_OUT, index=False)
    print(f"final: {len(combined['ticker'].unique())} tickers, "
          f"{len(combined):,} rows -> {OUT}")
    if skips:
        reasons = pd.DataFrame(skips)["reason"].value_counts()
        print("skips:", reasons.to_dict())


if __name__ == "__main__":
    main()
