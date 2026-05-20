"""
FMP intraday validator — pull 15min bars for a small ticker set and validate
quality against yfinance daily aggregates before committing to a full backfill.

What it does:
  1. Backwards-walks FMP's /stable/historical-chart/{interval} endpoint to get
     full history per ticker (the endpoint caps each response at ~30 trading
     days, so we anchor on the earliest returned timestamp and recurse back).
  2. Writes data/intraday/{TICKER}_{interval}.parquet (ts, OHLCV).
  3. Aggregates to daily OHLCV and compares against yfinance daily for the last
     N days. Reports max close drift and volume drift per ticker.

Usage:
  python scripts/pull_intraday_validator.py
  python scripts/pull_intraday_validator.py --tickers XLK SPY
  python scripts/pull_intraday_validator.py --interval 30min --start 2020-01-01

Notes:
  - Requires FMP_API_KEY in .env. Plan tier dictates rate limit; we self-throttle
    at 5 req/s by default to avoid 429s on Starter.
  - Validation uses the last 60 trading days. Drift > 0.5% on a single day is
    flagged; persistent multi-day drift suggests adjustment-vs-unadjusted mismatch.
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

FMP_KEY = os.getenv("FMP_API_KEY")
FMP_URL = "https://financialmodelingprep.com/stable/historical-chart"
OUT_DIR = os.path.join(ROOT, "data", "intraday")
os.makedirs(OUT_DIR, exist_ok=True)

DEFAULT_TICKERS = ["XLK", "SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
DEFAULT_INTERVAL = "15min"
DEFAULT_START = "2003-01-01"
THROTTLE_SECONDS = 0.2


def fetch_chunk(symbol, interval, frm, to):
    """Single FMP call. Returns list of bar dicts (newest first) or [] on empty."""
    params = {"symbol": symbol, "from": frm, "to": to, "apikey": FMP_KEY}
    r = requests.get(f"{FMP_URL}/{interval}", params=params, timeout=60)
    if r.status_code == 200:
        d = r.json()
        if isinstance(d, list):
            return d
    if r.status_code == 429:
        time.sleep(5.0)
        return fetch_chunk(symbol, interval, frm, to)
    print(f"   ! {symbol} {frm}->{to}: HTTP {r.status_code} {r.text[:120]}")
    return []


def pull_full_history(symbol, interval, start_date):
    """Walk backwards from today to start_date in <=30-trading-day chunks.

    Anchors each request on the earliest timestamp returned by the previous
    one, so we don't have to guess the FMP cap exactly.
    """
    all_rows = []
    to = datetime.now().date().isoformat()
    seen = set()
    n_calls = 0
    while True:
        rows = fetch_chunk(symbol, interval, start_date, to)
        n_calls += 1
        if not rows:
            break
        # Dedupe in case FMP returns overlapping bars across windows
        new_rows = [r for r in rows if r["date"] not in seen]
        if not new_rows:
            break
        seen.update(r["date"] for r in new_rows)
        all_rows.extend(new_rows)
        # Anchor next request: the earliest bar returned, minus one day.
        earliest = min(r["date"] for r in new_rows)
        earliest_d = datetime.strptime(earliest[:10], "%Y-%m-%d").date()
        if earliest_d.isoformat() <= start_date:
            break
        to = (earliest_d - timedelta(days=1)).isoformat()
        time.sleep(THROTTLE_SECONDS)
    return all_rows, n_calls


def to_df(rows):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["date"])
    df = df[["ts", "open", "high", "low", "close", "volume"]].sort_values("ts").reset_index(drop=True)
    df["open"] = df["open"].astype("float32")
    df["high"] = df["high"].astype("float32")
    df["low"] = df["low"].astype("float32")
    df["close"] = df["close"].astype("float32")
    df["volume"] = df["volume"].astype("uint32", errors="ignore")
    return df


def aggregate_daily(intraday_df):
    """Roll 15min bars up to a daily OHLCV frame for validation."""
    if intraday_df.empty:
        return pd.DataFrame()
    g = intraday_df.assign(date=intraday_df["ts"].dt.date).groupby("date")
    daily = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    })
    daily.index = pd.to_datetime(daily.index)
    return daily


def validate_against_yfinance(ticker, daily_from_intraday, lookback_days=60):
    """Compare last N days of FMP-aggregated daily vs yfinance daily.

    Returns dict with max/median % drift on close and volume.
    """
    try:
        import yfinance as yf
    except ImportError:
        return {"yf_available": False}
    if daily_from_intraday.empty:
        return {"yf_available": True, "n_compared": 0}

    end = daily_from_intraday.index.max() + pd.Timedelta(days=1)
    start = daily_from_intraday.index.max() - pd.Timedelta(days=lookback_days * 2)
    yf_df = yf.download(ticker, start=start, end=end, auto_adjust=False,
                        progress=False, threads=False)
    if isinstance(yf_df.columns, pd.MultiIndex):
        yf_df.columns = yf_df.columns.get_level_values(0)
    if yf_df.empty:
        return {"yf_available": True, "n_compared": 0}

    yf_df.index = pd.to_datetime(yf_df.index).normalize()
    fmp = daily_from_intraday.copy()
    fmp.index = fmp.index.normalize()

    joined = fmp.join(yf_df[["Close", "Volume"]], how="inner", rsuffix="_yf").tail(lookback_days)
    if joined.empty:
        return {"yf_available": True, "n_compared": 0}

    close_drift_pct = ((joined["close"] - joined["Close"]) / joined["Close"] * 100).abs()
    vol_drift_pct = ((joined["volume"] - joined["Volume"]) / joined["Volume"] * 100).abs()
    return {
        "yf_available": True,
        "n_compared": int(len(joined)),
        "max_close_drift_pct": float(close_drift_pct.max()),
        "median_close_drift_pct": float(close_drift_pct.median()),
        "max_vol_drift_pct": float(vol_drift_pct.max()),
        "median_vol_drift_pct": float(vol_drift_pct.median()),
        "days_close_drift_over_05pct": int((close_drift_pct > 0.5).sum()),
    }


def process_ticker(ticker, interval, start_date):
    print(f"\n=== {ticker} ===")
    t0 = time.time()
    rows, n_calls = pull_full_history(ticker, interval, start_date)
    df = to_df(rows)
    if df.empty:
        print(f"   no data returned in {n_calls} calls")
        return None

    out_path = os.path.join(OUT_DIR, f"{ticker}_{interval}.parquet")
    df.to_parquet(out_path, index=False, compression="zstd")
    elapsed = time.time() - t0

    print(f"   {len(df):,} bars over {df['ts'].dt.date.nunique()} days "
          f"({df['ts'].iloc[0]} -> {df['ts'].iloc[-1]})")
    print(f"   {n_calls} API calls, {elapsed:.1f}s wall, "
          f"file size: {os.path.getsize(out_path)/1024:.1f} KB")

    daily = aggregate_daily(df)
    val = validate_against_yfinance(ticker, daily)
    if val.get("n_compared", 0):
        print(f"   validation vs yfinance ({val['n_compared']} days):")
        print(f"     close drift: median {val['median_close_drift_pct']:.3f}%, "
              f"max {val['max_close_drift_pct']:.3f}%, "
              f"days >0.5%: {val['days_close_drift_over_05pct']}")
        print(f"     volume drift: median {val['median_vol_drift_pct']:.2f}%, "
              f"max {val['max_vol_drift_pct']:.2f}%")
    else:
        print("   yfinance comparison skipped (no overlap)")

    return {
        "ticker": ticker, "n_bars": len(df), "n_days": df["ts"].dt.date.nunique(),
        "first_bar": df["ts"].iloc[0], "last_bar": df["ts"].iloc[-1],
        "n_api_calls": n_calls, "elapsed_sec": round(elapsed, 1),
        "file_kb": round(os.path.getsize(out_path) / 1024, 1),
        **{k: v for k, v in val.items() if k != "yf_available"},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--tickers-file", default=None,
                   help="Path to file containing whitespace-separated tickers. Used when --tickers not given.")
    p.add_argument("--interval", default=DEFAULT_INTERVAL)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip tickers that already have a parquet on disk.")
    args = p.parse_args()

    if not FMP_KEY:
        print("FMP_API_KEY missing from .env")
        sys.exit(1)

    if args.tickers:
        tickers = args.tickers
    elif args.tickers_file:
        with open(args.tickers_file) as f:
            tickers = f.read().split()
    else:
        tickers = DEFAULT_TICKERS

    if args.skip_existing:
        before = len(tickers)
        tickers = [
            t for t in tickers
            if not os.path.exists(os.path.join(OUT_DIR, f"{t}_{args.interval}.parquet"))
        ]
        print(f"--skip-existing: {before - len(tickers)} ticker(s) already on disk, processing {len(tickers)}")

    args.tickers = tickers
    print(f"Tickers: {args.tickers}")
    print(f"Interval: {args.interval} | start: {args.start}")
    print(f"Output: {OUT_DIR}")

    results = []
    for t in args.tickers:
        try:
            r = process_ticker(t, args.interval, args.start)
            if r:
                results.append(r)
        except Exception as e:
            print(f"   ! {t} crashed: {e}")

    if results:
        print("\n=== SUMMARY ===")
        summary = pd.DataFrame(results)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
