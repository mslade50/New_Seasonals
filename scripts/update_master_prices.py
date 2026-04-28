"""Daily incremental update — fetches recent bars for every ticker in
data/master_prices.parquet, appends, dedupes, and writes back.

Idempotent: re-running on the same day is safe (drops duplicate (ticker, date)).
Run after market close. Wire into Task Scheduler for unattended daily updates.

Usage:
    python scripts/update_master_prices.py [--buffer-days 5]

The buffer pulls a few extra days back from the earliest stale max-date so
late-reporting bars or splits get refreshed.
"""
import argparse
import os
import sys
import time
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data")
PATH = os.path.join(DATA_DIR, "master_prices.parquet")
CHUNK_SIZE = 50


def _normalize_ticker_df(t_df):
    if isinstance(t_df.columns, pd.MultiIndex):
        t_df.columns = t_df.columns.get_level_values(0)
    if "Close" not in t_df.columns or t_df["Close"].dropna().empty:
        return None
    if t_df.index.tz is not None:
        t_df.index = t_df.index.tz_localize(None)
    t_df.index = pd.to_datetime(t_df.index).normalize()
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in t_df.columns]
    return t_df[cols].dropna(subset=["Close"])


def download_chunk(tickers, start_date):
    out = {}
    try:
        df = yf.download(
            tickers, start=start_date, group_by="ticker",
            auto_adjust=True, progress=False, threads=True,
        )
        if df.empty:
            return out
        if len(tickers) == 1:
            t = tickers[0]
            norm = _normalize_ticker_df(df.copy())
            if norm is not None:
                out[t] = norm
        else:
            top_levels = df.columns.levels[0] if isinstance(df.columns, pd.MultiIndex) else []
            for t in tickers:
                if t not in top_levels:
                    continue
                try:
                    norm = _normalize_ticker_df(df[t].copy())
                    if norm is not None:
                        out[t] = norm
                except Exception:
                    continue
    except Exception as e:
        print(f"  chunk error: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer-days", type=int, default=5)
    args = ap.parse_args()

    if not os.path.exists(PATH):
        print(f"ERROR: {PATH} missing — run scripts/build_master_prices.py first")
        return 1

    print(f"Loading {PATH}...")
    master = pd.read_parquet(PATH)
    universe = sorted(master["ticker"].unique().tolist())
    last_dates = master.groupby("ticker")["date"].max()
    earliest_stale = last_dates.min()
    today = pd.Timestamp.today().normalize()

    fetch_start = (earliest_stale - pd.Timedelta(days=args.buffer_days)).strftime("%Y-%m-%d")
    print(f"  tickers:        {len(universe)}")
    print(f"  earliest stale: {earliest_stale.date()}")
    print(f"  fetch from:     {fetch_start}")
    print(f"  today:          {today.date()}\n")

    all_frames = []
    t_start = time.time()
    for i in range(0, len(universe), CHUNK_SIZE):
        chunk = universe[i:i + CHUNK_SIZE]
        print(f"[{i+1:>5}-{min(i+CHUNK_SIZE, len(universe)):>5} / {len(universe)}] updating...", flush=True)
        result = download_chunk(chunk, fetch_start)
        for t, df in result.items():
            df = df.copy()
            df["ticker"] = t
            df = df.reset_index().rename(columns={"index": "date", "Date": "date"})
            all_frames.append(df)
        time.sleep(0.3)

    if not all_frames:
        print("no updates")
        return 0

    new_data = pd.concat(all_frames, ignore_index=True)
    combined = pd.concat([master, new_data], ignore_index=True)
    combined = combined.dropna(subset=["Close"])
    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    for c in ["Open", "High", "Low", "Close"]:
        if c in combined.columns:
            combined[c] = combined[c].astype("float32")
    if "Volume" in combined.columns:
        combined["Volume"] = combined["Volume"].astype("float64")
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined[["ticker", "date", "Open", "High", "Low", "Close", "Volume"]]
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    added = len(combined) - len(master)
    combined.to_parquet(PATH, compression="snappy", index=False)

    elapsed = time.time() - t_start
    new_max = combined.groupby("ticker")["date"].max().max()
    print(f"\nDone in {elapsed:.1f}s. Added {added:,} rows. New max date: {new_max.date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
