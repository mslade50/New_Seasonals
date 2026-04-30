"""
Build earnings calendar parquet from FMP /stable/earnings endpoint.

Pulls full historical earnings (announcement date, EPS actual/estimate,
revenue actual/estimate) for every ticker in CSV_UNIVERSE and writes to
data/earnings_calendar.parquet for the backtester to consume.

Usage:
    python scripts/build_earnings_calendar.py                # full universe
    python scripts/build_earnings_calendar.py --tickers AAPL NUE
    python scripts/build_earnings_calendar.py --refresh     # rebuild from scratch

API key: read from FMP_API_KEY env var or .env at project root.
Rate limit: ~750 calls/min on FMP Premium → script paces at ~10/sec.
"""
import argparse
import os
import sys
import time
import requests
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from strategy_config import CSV_UNIVERSE

OUTPUT_PATH = os.path.join(parent_dir, "data", "earnings_calendar.parquet")
ENV_PATH = os.path.join(parent_dir, ".env")
ENDPOINT = "https://financialmodelingprep.com/stable/earnings"

# ~750 calls/min on Premium. Pace at 10/sec to stay well under limit.
SLEEP_BETWEEN_CALLS = 0.1
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3


def load_env():
    """Load FMP_API_KEY from .env into os.environ if not already set."""
    if "FMP_API_KEY" in os.environ:
        return os.environ["FMP_API_KEY"]
    if not os.path.exists(ENV_PATH):
        raise SystemExit(f"FMP_API_KEY not in env and no .env at {ENV_PATH}")
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    key = os.environ.get("FMP_API_KEY")
    if not key:
        raise SystemExit("FMP_API_KEY not found in .env")
    return key


def fetch_ticker(symbol, api_key):
    """Pull historical earnings for one ticker. Returns list of dicts or None on hard failure."""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(
                ENDPOINT,
                params={"symbol": symbol, "apikey": api_key},
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    return data
                # Dict response = error message
                return []
            if r.status_code == 429:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            return []
        except requests.exceptions.RequestException:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return None
    return None


def build_calendar(tickers, api_key, output_path):
    print(f"Building earnings calendar for {len(tickers)} tickers...")
    print(f"Output: {output_path}\n")

    rows = []
    failures = []
    empty = []
    t0 = time.time()

    for i, sym in enumerate(tickers, start=1):
        if i % 50 == 0 or i == 1 or i == len(tickers):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(tickers) - i) / rate if rate > 0 else 0
            print(f"  [{i:>4}/{len(tickers)}] {sym:<8}  rows={len(rows):>6}  "
                  f"rate={rate:.1f}/s  ETA={eta:.0f}s")
        data = fetch_ticker(sym, api_key)
        if data is None:
            failures.append(sym)
            continue
        if not data:
            empty.append(sym)
            continue
        for r in data:
            rows.append({
                "ticker": sym.upper(),
                "date": r.get("date"),
                "eps_actual": r.get("epsActual"),
                "eps_est": r.get("epsEstimated"),
                "revenue_actual": r.get("revenueActual"),
                "revenue_est": r.get("revenueEstimated"),
                "last_updated": r.get("lastUpdated"),
            })
        time.sleep(SLEEP_BETWEEN_CALLS)

    if not rows:
        print("\nNo rows pulled — aborting write.")
        return

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Rows written:    {len(df):,}")
    print(f"  Tickers covered: {df['ticker'].nunique()}/{len(tickers)}")
    print(f"  Date range:      {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  Empty results:   {len(empty)}")
    if empty[:10]:
        print(f"    sample: {empty[:10]}")
    print(f"  Failures (network): {len(failures)}")
    if failures[:10]:
        print(f"    sample: {failures[:10]}")
    print(f"\nSaved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Backfill earnings calendar from FMP.")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Specific tickers (default: full CSV_UNIVERSE)")
    parser.add_argument("--output", default=OUTPUT_PATH,
                        help=f"Output parquet path (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    api_key = load_env()
    tickers = args.tickers if args.tickers else sorted(set(CSV_UNIVERSE))
    if not tickers:
        raise SystemExit("No tickers to process.")

    build_calendar(tickers, api_key, args.output)


if __name__ == "__main__":
    main()
