"""Build the candidate symbol master (Layer A) from the FMP company screener.

Pulls all actively-traded US common stocks on NASDAQ / NYSE / AMEX above a
rough share-volume and price pre-filter, and writes:

    data/symbol_master.parquet   ticker, exchange, company_name, market_cap,
                                 avg_volume, price, sector, as_of

…then uploads to R2. This is the *candidate* pool; the precise dollar-volume /
ATR screen happens in scripts/build_overflow_universe.py against our own
master_prices data. The FMP pre-filter only bounds how many names we maintain
price/earnings/seasonal history for.

FMP notes (verified during review):
  * /stable/company-screener is authoritative. /stable/stock-list is the legacy
    v3 path and /stable/company-symbols-list lacks exchange/type — neither is a
    usable fallback, so we rely on the screener.
  * `exchange` takes a SINGLE value per call → we loop one call per exchange.
  * The screener has a default result cap → we pass an explicit high `limit` and
    assert the returned count is under it (warn + tighten if hit).

API key: FMP_API_KEY env var or .env at project root (same as build_earnings_calendar).

Usage:
    python scripts/build_symbol_master.py            # build + write + R2 upload
    python scripts/build_symbol_master.py --dry-run  # fetch + print, NO write/upload
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import requests

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PARENT)

OUTPUT_PATH = os.path.join(_PARENT, "data", "symbol_master.parquet")
ENV_PATH = os.path.join(_PARENT, ".env")
SCREENER_ENDPOINT = "https://financialmodelingprep.com/stable/company-screener"

EXCHANGES = ["NASDAQ", "NYSE", "AMEX"]
PRICE_MORE_THAN = 3          # penny guard (final price gate is in Layer C)
VOLUME_MORE_THAN = 300_000   # share-volume pre-filter (final $-vol gate is in Layer C)
RESULT_LIMIT = 10_000        # explicit cap; assert returned count is under it
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.2

# yfinance can't resolve warrants/units/preferreds/test issues. Drop obvious
# non-common-stock suffixes and malformed tickers (logged).
_DENY_SUFFIXES = ("W", "WS", "U", "R", "RT")  # checked only after a '.'/'-' class sep


def load_env() -> str:
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


def fetch_exchange(exchange: str, api_key: str,
                   price_more_than=PRICE_MORE_THAN,
                   volume_more_than=VOLUME_MORE_THAN) -> list:
    """One screener call for a single exchange. Returns list of dicts."""
    params = {
        "exchange": exchange,
        "isEtf": "false",
        "isFund": "false",
        "isActivelyTrading": "true",
        "priceMoreThan": price_more_than,
        "volumeMoreThan": volume_more_than,
        "country": "US",
        "limit": RESULT_LIMIT,
        "apikey": api_key,
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(SCREENER_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    if len(data) >= RESULT_LIMIT:
                        print(f"  WARN: {exchange}: hit result cap ({RESULT_LIMIT}) - "
                              f"some names may be truncated; tighten filters or paginate.")
                    return data
                print(f"  WARN: {exchange}: unexpected response shape: {str(data)[:200]}")
                return []
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            print(f"  WARN: {exchange}: HTTP {r.status_code}: {r.text[:200]}")
            return []
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"  WARN: {exchange}: request failed: {e}")
            return []
    return []


def _is_clean_common(symbol: str) -> bool:
    s = str(symbol).strip().upper()
    if not s or len(s) > 6:
        return False
    if any(ch in s for ch in ("^", "/", " ", "$")):
        return False
    # Drop class/warrant/unit suffixes after a separator (e.g. BRK.B kept as
    # BRK-B is fine, but FOO.WS / FOO.U are warrants/units → drop).
    for sep in (".", "-"):
        if sep in s:
            tail = s.split(sep)[-1]
            if tail in _DENY_SUFFIXES:
                return False
    return True


def normalize_rows(raw: list, exchange: str) -> list:
    rows = []
    for d in raw:
        sym = d.get("symbol")
        if not sym or not _is_clean_common(sym):
            continue
        rows.append(
            {
                "ticker": str(sym).upper().strip().replace(".", "-"),
                "exchange": d.get("exchangeShortName") or exchange,
                "company_name": d.get("companyName"),
                "market_cap": d.get("marketCap"),
                "avg_volume": d.get("volume") or d.get("avgVolume"),
                "price": d.get("price"),
                "sector": d.get("sector"),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the FMP candidate symbol master.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch + print only - no parquet write, no R2 upload.")
    ap.add_argument("--volume-more-than", type=int, default=VOLUME_MORE_THAN,
                    help=f"FMP share-volume pre-filter (default {VOLUME_MORE_THAN}). "
                         "Lower = wider candidate pool = heavier backfill.")
    ap.add_argument("--price-more-than", type=float, default=PRICE_MORE_THAN,
                    help=f"FMP price pre-filter (default {PRICE_MORE_THAN}).")
    args = ap.parse_args()

    api_key = load_env()
    print(f"Filters: price > {args.price_more_than}, volume > {args.volume_more_than:,} shares/day")
    all_rows = []
    for ex in EXCHANGES:
        print(f"Fetching {ex} ...")
        raw = fetch_exchange(ex, api_key,
                             price_more_than=args.price_more_than,
                             volume_more_than=args.volume_more_than)
        rows = normalize_rows(raw, ex)
        print(f"  {ex}: {len(raw):,} raw -> {len(rows):,} clean common stocks")
        all_rows.extend(rows)
        time.sleep(SLEEP_BETWEEN_CALLS)

    if not all_rows:
        raise SystemExit("No symbols fetched - aborting (check FMP key / plan tier).")

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    as_of = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    df["as_of"] = as_of
    print(f"\nTotal unique candidate symbols: {len(df):,}")
    print(df.head(15).to_string(index=False))

    if args.dry_run:
        print("\n[dry-run] no parquet written, no R2 upload.")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {OUTPUT_PATH} ({len(df):,} symbols)")

    try:
        from cache_io import upload_from_local
        upload_from_local(OUTPUT_PATH, "symbol_master.parquet")
    except Exception as e:  # noqa: BLE001
        print(f"[r2 upload] non-fatal error: {e}")


if __name__ == "__main__":
    main()
