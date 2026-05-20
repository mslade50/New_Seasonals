"""
Push data/intraday/{TICKER}_{interval}.parquet files to R2.

Mirrors the same data under R2 keys `intraday/{interval}/{TICKER}.parquet` and
writes a `_meta.parquet` index so callers can discover what's available without
listing the bucket.

Usage:
    python scripts/upload_intraday_to_r2.py
    python scripts/upload_intraday_to_r2.py --interval 15min
    python scripts/upload_intraday_to_r2.py --tickers XLK SPY
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cache_io import upload_from_local, is_configured  # noqa: E402

INTRADAY_DIR = os.path.join(ROOT, "data", "intraday")
META_LOCAL = os.path.join(INTRADAY_DIR, "_meta.parquet")


def collect_files(interval, tickers_filter):
    """Return list of (ticker, interval, local_path) tuples to upload."""
    pattern = os.path.join(INTRADAY_DIR, f"*_{interval}.parquet")
    found = []
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        # Match TICKER_INTERVAL.parquet — ticker may contain hyphens (BTC-USD)
        # or carets/dots/equals (e.g. ^GSPC, DX-Y.NYB, USDEUR=X) so be liberal.
        m = re.match(r"(.+)_" + re.escape(interval) + r"\.parquet$", fname)
        if not m:
            continue
        ticker = m.group(1)
        if tickers_filter and ticker not in tickers_filter:
            continue
        found.append((ticker, interval, path))
    return found


def build_meta(files):
    """Inspect each parquet, build a metadata table for the index."""
    rows = []
    for ticker, interval, path in files:
        try:
            df = pd.read_parquet(path, columns=["ts"])
        except Exception as e:
            print(f"   ! could not read {path}: {e}")
            continue
        if df.empty:
            continue
        rows.append({
            "ticker": ticker,
            "interval": interval,
            "n_bars": int(len(df)),
            "first_ts": df["ts"].min(),
            "last_ts": df["ts"].max(),
            "n_days": int(df["ts"].dt.date.nunique()),
            "size_kb": round(os.path.getsize(path) / 1024, 1),
        })
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", default="15min")
    p.add_argument("--tickers", nargs="*", default=None)
    args = p.parse_args()

    if not is_configured():
        print("R2 credentials missing in environment. Aborting.")
        sys.exit(1)

    tickers_filter = set(args.tickers) if args.tickers else None
    files = collect_files(args.interval, tickers_filter)
    if not files:
        print("No matching intraday parquet files found.")
        sys.exit(1)

    print(f"Found {len(files)} {args.interval} parquet(s) to upload")

    n_ok = 0
    for ticker, interval, path in files:
        key = f"intraday/{interval}/{ticker}.parquet"
        if upload_from_local(path, key):
            n_ok += 1

    # Rebuild meta from *all* local files (not just the filtered upload set)
    # so partial uploads don't shrink the index.
    all_files = collect_files(args.interval, tickers_filter=None)
    meta = build_meta(all_files)
    if meta.empty:
        print("Meta inspection failed.")
        sys.exit(1)
    meta.to_parquet(META_LOCAL, index=False)
    upload_from_local(META_LOCAL, f"intraday/{args.interval}/_meta.parquet")

    print(f"\nUploaded {n_ok}/{len(files)} files. Meta index covers {len(meta)} tickers.")
    print(meta.to_string(index=False))


if __name__ == "__main__":
    main()
