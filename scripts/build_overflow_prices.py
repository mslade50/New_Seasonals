"""Build the ISOLATED staging price cache for new overflow candidate tickers.

Writes data/overflow_prices.parquet (R2 key: overflow_prices.parquet) holding
OHLCV history for the symbol_master candidates that are NOT already in
production master_prices. This keeps the live daily price pipeline
(update_master_prices / portfolio_report / earnings) completely untouched while
the dynamic universe is built and backtested.

  * READS production master_prices.parquet (to know which names already exist) —
    never writes it.
  * Fetches only the genuinely-new names from yfinance.
  * Same long schema as master_prices: ticker, date, Open, High, Low, Close,
    Volume (OHLC float32, Volume float64).

At PROMOTE time these rows get merged into master_prices (and the daily updater
takes over); until then nothing in production changes.

Usage:
    python scripts/build_overflow_prices.py                     # backfill new names + R2 upload
    python scripts/build_overflow_prices.py --dry-run           # plan only, no fetch/write
    python scripts/build_overflow_prices.py --backfill-start 2010-01-01
    python scripts/build_overflow_prices.py --exclude-today     # drop today's partial bar
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PARENT)
sys.path.insert(0, _THIS_DIR)  # so `import build_master_prices` resolves

from overflow_universe import (  # noqa: E402
    OVERFLOW_PRICES_PATH, OVERFLOW_PRICES_R2_KEY, SYMBOL_MASTER_PATH,
)

MASTER_PRICES_PATH = os.path.join(_PARENT, "data", "master_prices.parquet")
ENV_PATH = os.path.join(_PARENT, ".env")
CHUNK_SIZE = 50


def load_env() -> None:
    """Load .env into os.environ (so cache_io sees R2_* creds). No-op if absent."""
    if not os.path.exists(ENV_PATH):
        return
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def ensure_master_prices() -> None:
    """Pull master_prices from R2 if it's not already local (read-only seed)."""
    if os.path.exists(MASTER_PRICES_PATH):
        return
    try:
        from cache_io import download_to_local
        if download_to_local("master_prices.parquet", MASTER_PRICES_PATH):
            print("  pulled master_prices.parquet from R2 (read-only seed)")
    except Exception as e:  # noqa: BLE001
        print(f"  note: could not pull master_prices from R2: {e}")


def _norm(t: str) -> str:
    return str(t).upper().strip().replace(".", "-")


def load_candidates() -> list:
    if not os.path.exists(SYMBOL_MASTER_PATH):
        raise SystemExit(f"symbol_master.parquet missing at {SYMBOL_MASTER_PATH} "
                         "— run build_symbol_master.py first.")
    df = pd.read_parquet(SYMBOL_MASTER_PATH, columns=["ticker"])
    return sorted({_norm(t) for t in df["ticker"].tolist() if str(t).strip()})


def master_ticker_set() -> set:
    if not os.path.exists(MASTER_PRICES_PATH):
        print("  note: master_prices.parquet not found locally — treating all "
              "candidates as new (pull it from R2 first to avoid re-fetching).")
        return set()
    df = pd.read_parquet(MASTER_PRICES_PATH, columns=["ticker"])
    return {_norm(t) for t in df["ticker"].unique().tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build isolated overflow staging price cache.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan (new-ticker count) only - no fetch, no write, no upload.")
    ap.add_argument("--backfill-start", default="2005-01-01",
                    help="Start date for new tickers' first backfill (default 2005-01-01).")
    ap.add_argument("--buffer-days", type=int, default=5,
                    help="Re-fetch this many days back for tickers already staged (catches splits).")
    ap.add_argument("--exclude-today", action="store_true",
                    help="Drop bars dated today (Eastern) - for pre-market refreshes.")
    args = ap.parse_args()

    load_env()              # make R2 creds available to cache_io
    ensure_master_prices()  # pull master_prices from R2 if not local (read-only seed)

    # Liquid set is excluded from overflow entirely.
    from strategy_config import LIQUID_PLUS_COMMODITIES
    liquid = {_norm(t) for t in LIQUID_PLUS_COMMODITIES}

    candidates = load_candidates()
    in_master = master_ticker_set()

    # Existing staging cache (incremental update of already-staged names).
    staged = {}
    if os.path.exists(OVERFLOW_PRICES_PATH):
        try:
            _ex = pd.read_parquet(OVERFLOW_PRICES_PATH)
            _ex["ticker"] = [_norm(t) for t in _ex["ticker"].tolist()]
            staged = {t: g for t, g in _ex.groupby("ticker")}
        except Exception as e:  # noqa: BLE001
            print(f"  warn: could not read existing staging cache: {e}")

    # New names = candidates not in production master, not liquid.
    new_names = sorted(set(candidates) - in_master - liquid)
    # Split into first-time backfill vs incremental refresh.
    to_backfill = [t for t in new_names if t not in staged]
    to_refresh = [t for t in new_names if t in staged]

    print(f"Candidates (symbol_master):     {len(candidates):,}")
    print(f"Already in master_prices:       {len(in_master & set(candidates)):,}")
    print(f"New overflow names (staging):    {len(new_names):,}")
    print(f"  first-time backfill:           {len(to_backfill):,}  (from {args.backfill_start})")
    print(f"  incremental refresh:           {len(to_refresh):,}  (buffer {args.buffer_days}d)")

    if args.dry_run:
        print("\n[dry-run] no fetch, no write, no upload.")
        return

    # yfinance fetch helpers from the production builder (no side effects on import).
    from build_master_prices import download_chunk

    frames = []
    if staged:
        frames.extend(staged.values())

    def _fetch(tickers, start):
        for i in range(0, len(tickers), CHUNK_SIZE):
            chunk = tickers[i:i + CHUNK_SIZE]
            print(f"  [{i+1:>5}-{min(i+CHUNK_SIZE, len(tickers)):>5} / {len(tickers)}] start={start}", flush=True)
            res = download_chunk(chunk, start)
            for t, df in res.items():
                df = df.copy()
                df["ticker"] = _norm(t)
                df = df.reset_index().rename(columns={"index": "date", "Date": "date"})
                frames.append(df)
            time.sleep(0.3)

    if to_refresh:
        # earliest stale among refresh names
        last_dates = {t: staged[t]["date"].max() for t in to_refresh}
        earliest = min(pd.to_datetime(list(last_dates.values())))
        refresh_start = (pd.to_datetime(earliest) - pd.Timedelta(days=args.buffer_days)).strftime("%Y-%m-%d")
        print(f"\nRefreshing {len(to_refresh)} staged names from {refresh_start} ...")
        _fetch(to_refresh, refresh_start)
    if to_backfill:
        print(f"\nBackfilling {len(to_backfill)} new names from {args.backfill_start} ...")
        _fetch(to_backfill, args.backfill_start)

    if not frames:
        print("Nothing fetched — aborting write.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    if args.exclude_today:
        today_e = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
        before = len(combined)
        combined = combined[combined["date"] < today_e]
        print(f"  --exclude-today: dropped {before - len(combined):,} rows >= {today_e.date()}")
    combined = combined.dropna(subset=["Close"])
    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    for c in ["Open", "High", "Low", "Close"]:
        if c in combined.columns:
            combined[c] = combined[c].astype("float32")
    if "Volume" in combined.columns:
        combined["Volume"] = combined["Volume"].astype("float64")
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OVERFLOW_PRICES_PATH), exist_ok=True)
    combined.to_parquet(OVERFLOW_PRICES_PATH, index=False)
    print(f"\nWrote {OVERFLOW_PRICES_PATH}: {combined['ticker'].nunique():,} tickers, {len(combined):,} rows")

    try:
        from cache_io import upload_from_local
        upload_from_local(OVERFLOW_PRICES_PATH, OVERFLOW_PRICES_R2_KEY)
    except Exception as e:  # noqa: BLE001
        print(f"[r2 upload] non-fatal error: {e}")


if __name__ == "__main__":
    main()
