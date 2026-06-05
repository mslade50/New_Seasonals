"""
Build earnings calendar parquet from FMP /stable/earnings endpoint.

Pulls full historical earnings (announcement date, EPS actual/estimate,
revenue actual/estimate) for every ticker in CSV_UNIVERSE and writes to
data/earnings_calendar.parquet for the backtester to consume.

Derived columns (computed locally — no extra API calls):
    eps_surprise_pct  = (eps_actual - eps_est) / |eps_est|
                        — NaN when |eps_est| < EPS_EST_MIN (avoids divide-by-near-zero)
    rev_surprise_pct  = (revenue_actual - revenue_est) / revenue_est
    eps_yoy           = (eps_actual - eps_actual.shift(4)) / |eps_actual.shift(4)|
                        — only computed when prior row is 330-400 days back
                        (skips data gaps where shift(4) would not be ~1 yr earlier)
    rev_yoy           = (revenue_actual - revenue_actual.shift(4)) / revenue_actual.shift(4)
                        — same date-gap guard

Usage:
    python scripts/build_earnings_calendar.py                # full universe
    python scripts/build_earnings_calendar.py --tickers AAPL NUE
    python scripts/build_earnings_calendar.py --derive-only  # no API calls, just
                                                             # add/refresh derived cols

API key: read from FMP_API_KEY env var or .env at project root.
Rate limit: ~750 calls/min on FMP Premium -> script paces at ~10/sec.
"""
import argparse
import os
import sys
import time
import numpy as np
import requests
import pandas as pd

# EPS estimates below this absolute value get NaN surprise % — the ratio is
# unstable when the denominator is near zero (a $0.01 beat on a $0.02 est is
# 50%, which is mathematically true but uninformative for filtering).
EPS_EST_MIN = 0.05

# YoY shift(4) only counts when the prior row is roughly one calendar year
# back. A wider window absorbs reporting calendar drift; outside of it we
# probably crossed a missing quarter or a calendar shift and should NaN out.
YOY_DAYS_LO = 330
YOY_DAYS_HI = 400

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


def compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add eps_surprise_pct, rev_surprise_pct, eps_yoy, rev_yoy to an
    earnings DataFrame in place (returns same df). Pure function — no API
    calls. Safe to run repeatedly; existing derived columns are overwritten.

    Input must have: ticker, date, eps_actual, eps_est, revenue_actual, revenue_est.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    eps_est_safe = df["eps_est"].where(df["eps_est"].abs() >= EPS_EST_MIN)
    df["eps_surprise_pct"] = (df["eps_actual"] - eps_est_safe) / eps_est_safe.abs()

    rev_est_safe = df["revenue_est"].where(df["revenue_est"] > 0)
    df["rev_surprise_pct"] = (df["revenue_actual"] - rev_est_safe) / rev_est_safe

    grp = df.groupby("ticker", sort=False)
    prior_eps = grp["eps_actual"].shift(4)
    prior_rev = grp["revenue_actual"].shift(4)
    prior_date = grp["date"].shift(4)
    days_gap = (df["date"] - prior_date).dt.days
    yoy_window = days_gap.between(YOY_DAYS_LO, YOY_DAYS_HI)

    eps_yoy_raw = (df["eps_actual"] - prior_eps) / prior_eps.abs()
    df["eps_yoy"] = eps_yoy_raw.where(yoy_window & (prior_eps.abs() >= EPS_EST_MIN))

    prior_rev_safe = prior_rev.where(prior_rev > 0)
    rev_yoy_raw = (df["revenue_actual"] - prior_rev_safe) / prior_rev_safe
    df["rev_yoy"] = rev_yoy_raw.where(yoy_window)

    return df


def upload_to_r2(output_path: str, key: str = "earnings_calendar.parquet"):
    """Push the parquet to R2 so cloud workflows + Streamlit Cloud can read it."""
    try:
        from cache_io import upload_from_local
        upload_from_local(output_path, key)
    except Exception as e:
        print(f"[r2 upload] non-fatal error: {e}")


def derive_only(output_path: str):
    """Read existing parquet, recompute derived columns, write back, upload to R2.

    Used when we change the derivation logic and want to refresh without
    burning ~1060 FMP calls.
    """
    if not os.path.exists(output_path):
        raise SystemExit(f"--derive-only needs an existing parquet at {output_path}")
    df = pd.read_parquet(output_path)
    print(f"Loaded {len(df):,} rows from {output_path}")
    df = compute_derived_columns(df)
    df.to_parquet(output_path, index=False)
    derived_cols = ["eps_surprise_pct", "rev_surprise_pct", "eps_yoy", "rev_yoy"]
    print("Derived column non-null counts:")
    for c in derived_cols:
        print(f"  {c:<20} {df[c].notna().sum():>7,} / {len(df):,}")
    upload_to_r2(output_path)


def build_calendar(tickers, api_key, output_path, r2_key="earnings_calendar.parquet"):
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
    df = compute_derived_columns(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Rows written:    {len(df):,}")
    print(f"  Tickers covered: {df['ticker'].nunique()}/{len(tickers)}")
    print(f"  Date range:      {df['date'].min().date()} -> {df['date'].max().date()}")
    derived_cols = ["eps_surprise_pct", "rev_surprise_pct", "eps_yoy", "rev_yoy"]
    print("  Derived column non-null counts:")
    for c in derived_cols:
        print(f"    {c:<20} {df[c].notna().sum():>7,}")
    print(f"  Empty results:   {len(empty)}")
    if empty[:10]:
        print(f"    sample: {empty[:10]}")
    print(f"  Failures (network): {len(failures)}")
    if failures[:10]:
        print(f"    sample: {failures[:10]}")
    print(f"\nSaved: {output_path}")

    upload_to_r2(output_path, key=r2_key)


def main():
    parser = argparse.ArgumentParser(description="Backfill earnings calendar from FMP.")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Specific tickers (default: full CSV_UNIVERSE)")
    parser.add_argument("--with-symbol-master", action="store_true",
                        help="Also include data/symbol_master.parquet tickers (overflow Layer A) so "
                             "overflow OVS names have earnings coverage for the blackout filter.")
    parser.add_argument("--output", default=OUTPUT_PATH,
                        help=f"Output parquet path (default: {OUTPUT_PATH})")
    parser.add_argument("--overflow-staging", action="store_true",
                        help="ISOLATED mode: build earnings for ONLY the new overflow names "
                             "(symbol_master - CSV_UNIVERSE) into a separate staging parquet "
                             "(data/earnings_calendar_overflow.parquet, R2 key "
                             "earnings_calendar_overflow.parquet). Production earnings_calendar is "
                             "never touched, so the daily CSV_UNIVERSE rebuild can't wipe these.")
    parser.add_argument("--derive-only", action="store_true",
                        help="Skip API calls; recompute derived columns on the existing parquet only.")
    args = parser.parse_args()

    if args.derive_only and args.overflow_staging:
        raise SystemExit("--derive-only cannot be combined with --overflow-staging "
                         "(derive-only would re-upload the PRODUCTION parquet).")
    if args.derive_only:
        derive_only(args.output)
        return

    api_key = load_env()
    r2_key = "earnings_calendar.parquet"
    output = args.output

    if args.overflow_staging:
        # New overflow names only -> isolated staging parquet + R2 key.
        output = os.path.join(parent_dir, "data", "earnings_calendar_overflow.parquet")
        r2_key = "earnings_calendar_overflow.parquet"
        _sm = os.path.join(parent_dir, "data", "symbol_master.parquet")
        if not os.path.exists(_sm):
            raise SystemExit(f"--overflow-staging needs {_sm} (run build_symbol_master.py first).")
        _extra = set(pd.read_parquet(_sm, columns=["ticker"])["ticker"].astype(str).str.upper())
        tickers = sorted(_extra - set(t.upper() for t in CSV_UNIVERSE))
        print(f"[overflow-staging] {len(tickers)} new names -> {output} (R2: {r2_key})")
    elif args.tickers:
        tickers = args.tickers
    else:
        _u = set(CSV_UNIVERSE)
        if args.with_symbol_master:
            _sm = os.path.join(parent_dir, "data", "symbol_master.parquet")
            if os.path.exists(_sm):
                try:
                    _extra = pd.read_parquet(_sm, columns=["ticker"])["ticker"].astype(str).str.upper().tolist()
                    _u |= set(_extra)
                    print(f"[universe] +symbol_master -> {len(_u)} tickers")
                except Exception as _e:
                    print(f"[universe] warn: could not read {_sm}: {_e}")
            else:
                print(f"[universe] warn: --with-symbol-master set but {_sm} missing")
        tickers = sorted(_u)
    if not tickers:
        raise SystemExit("No tickers to process.")

    build_calendar(tickers, api_key, output, r2_key=r2_key)


if __name__ == "__main__":
    main()
