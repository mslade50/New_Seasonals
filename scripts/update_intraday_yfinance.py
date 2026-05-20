"""Incremental intraday updater — appends recent 15min bars from yfinance to
every parquet in data/intraday/. Designed as the long-term maintenance tool
that replaces FMP for keeping the cache fresh.

yfinance gives ~60 days of rolling 15min history per request, which is plenty
of headroom as long as this script runs at least every ~50 days. The seam
between FMP (historical) and yfinance (incremental) has been validated:
median close drift ~0.001%, max ~0.07% — statistically equivalent.

Usage:
    python scripts/update_intraday_yfinance.py                  # all local files
    python scripts/update_intraday_yfinance.py --tickers SPY    # subset
    python scripts/update_intraday_yfinance.py --upload         # also push to R2
    python scripts/update_intraday_yfinance.py --buffer-days 3  # default 7
    python scripts/update_intraday_yfinance.py --period 60d     # full window per fetch

Behavior:
  - Reads existing data/intraday/{TICKER}_{interval}.parquet
  - Fetches last `period` (or smart-buffer) of 15min bars from yfinance
  - Converts UTC -> America/New_York -> strips tz (matches FMP convention)
  - Filters to regular session (09:30-16:00 ET) to match FMP coverage
  - Appends, dedupes on ts, rewrites parquet
  - Rebuilds _meta.parquet
  - Optional --upload pushes changed files + meta to R2
"""
import argparse
import glob
import os
import re
import sys
import time
from typing import Optional

import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

INTRADAY_DIR = os.path.join(ROOT, "data", "intraday")
META_LOCAL = os.path.join(INTRADAY_DIR, "_meta.parquet")

ET = "America/New_York"


def _existing_files(interval: str, tickers_filter: Optional[set]):
    pattern = os.path.join(INTRADAY_DIR, f"*_{interval}.parquet")
    out = []
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        if fname == "_meta.parquet":
            continue
        m = re.match(r"(.+)_" + re.escape(interval) + r"\.parquet$", fname)
        if not m:
            continue
        ticker = m.group(1)
        if tickers_filter and ticker not in tickers_filter:
            continue
        out.append((ticker, path))
    return out


def _normalize_yf_intraday(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance 15m -> the schema FMP uses on disk."""
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    ts_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df["ts"] = pd.to_datetime(df[ts_col])
    if df["ts"].dt.tz is None:
        # yfinance usually returns tz-aware; if not, assume UTC.
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    df["ts"] = df["ts"].dt.tz_convert(ET).dt.tz_localize(None)

    df = df.rename(columns={c: c.lower() for c in df.columns})
    keep = ["ts", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.dropna(subset=["close"])

    # Regular session only (FMP convention is 09:30-16:00 ET, exclusive end).
    tod = df["ts"].dt.time
    mask = (tod >= pd.Timestamp("09:30").time()) & (tod < pd.Timestamp("16:00").time())
    df = df.loc[mask].copy()

    df["open"] = df["open"].astype("float32")
    df["high"] = df["high"].astype("float32")
    df["low"] = df["low"].astype("float32")
    df["close"] = df["close"].astype("float32")
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0).astype("uint32", errors="ignore")
    return df.sort_values("ts").reset_index(drop=True)


def _fetch_one(ticker: str, period: str) -> pd.DataFrame:
    try:
        raw = yf.download(
            ticker, period=period, interval="15m",
            auto_adjust=False, progress=False, threads=False,
        )
    except Exception as e:
        print(f"   ! {ticker} yfinance error: {e}")
        return pd.DataFrame()
    return _normalize_yf_intraday(raw)


def _merge_and_write(ticker: str, path: str, fresh: pd.DataFrame) -> dict:
    if fresh.empty:
        return {"ticker": ticker, "added": 0, "status": "no-fresh"}
    existing = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
    if existing.empty:
        merged = fresh
    else:
        existing["ts"] = pd.to_datetime(existing["ts"])
        merged = pd.concat([existing, fresh], ignore_index=True)
        merged = merged.drop_duplicates(subset=["ts"], keep="last")
        merged = merged.sort_values("ts").reset_index(drop=True)
    added = len(merged) - len(existing)
    merged.to_parquet(path, index=False, compression="zstd")
    return {
        "ticker": ticker, "added": int(added),
        "n_bars": int(len(merged)),
        "first_ts": merged["ts"].min(), "last_ts": merged["ts"].max(),
        "status": "ok",
    }


def _rebuild_meta(interval: str) -> pd.DataFrame:
    files = _existing_files(interval, None)
    rows = []
    for ticker, path in files:
        df = pd.read_parquet(path, columns=["ts"])
        if df.empty:
            continue
        rows.append({
            "ticker": ticker, "interval": interval,
            "n_bars": int(len(df)),
            "first_ts": df["ts"].min(), "last_ts": df["ts"].max(),
            "n_days": int(df["ts"].dt.date.nunique()),
            "size_kb": round(os.path.getsize(path) / 1024, 1),
        })
    meta = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    meta.to_parquet(META_LOCAL, index=False)
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="*", default=None,
                   help="Subset to update. Default: every parquet in data/intraday/.")
    p.add_argument("--interval", default="15min")
    p.add_argument("--period", default=None,
                   help="yfinance period (e.g. 60d). Default: derived from buffer-days.")
    p.add_argument("--buffer-days", type=int, default=7,
                   help="When --period unset, fetch max(buffer, gap-since-last-bar+2).")
    p.add_argument("--upload", action="store_true",
                   help="Push updated parquets + _meta.parquet to R2.")
    p.add_argument("--missing-ok", action="store_true",
                   help="When --tickers is set, create new parquets for any not yet on disk.")
    args = p.parse_args()

    if args.interval != "15min":
        # yfinance supports 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h. Only 15m for now.
        print(f"WARN: interval {args.interval} not validated, proceeding anyway")

    yf_interval_label = args.interval.replace("min", "m") if args.interval.endswith("min") else args.interval
    if yf_interval_label != "15m":
        print(f"yfinance interval label: {yf_interval_label}")

    tickers_filter = set(t.upper() for t in args.tickers) if args.tickers else None
    files = _existing_files(args.interval, tickers_filter)

    if not files and args.missing_ok and tickers_filter:
        files = [(t, os.path.join(INTRADAY_DIR, f"{t}_{args.interval}.parquet")) for t in tickers_filter]

    if not files:
        print("No matching intraday parquet files found.")
        return 1

    print(f"Updating {len(files)} parquet(s), interval={args.interval}")
    t0 = time.time()
    results = []
    os.makedirs(INTRADAY_DIR, exist_ok=True)

    for i, (ticker, path) in enumerate(files, 1):
        # Decide fetch window
        period = args.period
        if period is None:
            if os.path.exists(path):
                existing = pd.read_parquet(path, columns=["ts"])
                if not existing.empty:
                    last = pd.to_datetime(existing["ts"].max())
                    gap_days = (pd.Timestamp.now() - last).days
                    days_needed = max(args.buffer_days, gap_days + 2)
                    # yfinance hard-caps 15m history at ~60d. Clip to be safe.
                    days_needed = min(days_needed, 59)
                else:
                    days_needed = 59
            else:
                days_needed = 59
            period = f"{days_needed}d"

        fresh = _fetch_one(ticker, period)
        r = _merge_and_write(ticker, path, fresh)
        print(f"  [{i:>3}/{len(files)}] {ticker:<7} period={period:<5} +{r['added']:>5} bars  last_ts={r.get('last_ts')}")
        results.append(r)
        time.sleep(0.15)

    print()
    print(f"Done in {time.time()-t0:.1f}s. {sum(1 for r in results if r['added']>0)} files modified.")

    meta = _rebuild_meta(args.interval)
    print(f"Meta rebuilt: {len(meta)} tickers")

    if args.upload:
        try:
            from cache_io import is_configured, upload_from_local
        except ImportError:
            print("cache_io not importable; skipping upload.")
            return 0
        if not is_configured():
            print("R2 not configured; skipping upload.")
            return 0
        n_ok = 0
        for ticker, path in _existing_files(args.interval, None):
            # Only upload tickers we touched this run, plus everyone if we rebuilt fresh
            if tickers_filter is None or ticker in tickers_filter:
                if upload_from_local(path, f"intraday/{args.interval}/{ticker}.parquet"):
                    n_ok += 1
        upload_from_local(META_LOCAL, f"intraday/{args.interval}/_meta.parquet")
        print(f"R2 upload: {n_ok} parquet(s) + meta")

    return 0


if __name__ == "__main__":
    sys.exit(main())
