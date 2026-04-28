"""One-time backfill — pulls full OHLCV history from yfinance for the union
universe and writes data/master_prices.parquet (long format).

Universe = CSV_UNIVERSE ∪ seasonal_ranks.csv ∪ sznl_ranks.csv ∪ INDICES_AND_ETFS.

Run from repo root:
    python scripts/build_master_prices.py [--start 2000-01-01]

Re-running will OVERWRITE the existing master file. For incremental updates,
use scripts/update_master_prices.py instead.
"""
import argparse
import os
import sys
import time
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from strategy_config import CSV_UNIVERSE  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
OUT_PATH = os.path.join(DATA_DIR, "master_prices.parquet")
DEFAULT_START = "2000-01-01"
CHUNK_SIZE = 50
RETRY_SLEEP = 2.0
MAX_RETRIES = 2

# Indices, sector ETFs, and cross-asset proxies pulled by the dashboards/scans.
# Kept here so the master parquet covers everything any page might need.
INDICES_AND_ETFS = [
    "SPY", "^VIX", "^VIX3M", "^VVIX", "^GSPC", "^TNX", "^IRX", "^MOVE", "^SKEW",
    "UUP", "LQD", "HYG", "IEF",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE",
    "IBB", "IHI", "ITA", "ITB", "IYR", "KRE", "OIH", "SMH", "VNQ",
    "QQQ", "IWM", "DIA", "EFA", "EEM", "GLD", "SLV", "USO", "TLT",
]


def load_csv_tickers(path):
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path, usecols=["ticker"])
        return set(df["ticker"].astype(str).str.upper().str.strip())
    except Exception as e:
        print(f"  warn: could not read {path}: {e}")
        return set()


def build_universe():
    u = set()
    u.update(t.upper() for t in CSV_UNIVERSE)
    u.update(load_csv_tickers(os.path.join(ROOT, "sznl_ranks.csv")))
    u.update(load_csv_tickers(os.path.join(ROOT, "seasonal_ranks.csv")))
    u.update(INDICES_AND_ETFS)
    u.discard("")
    return sorted(u)


def _normalize_ticker_df(t_df):
    """Return df with [Open, High, Low, Close, Volume] columns and tz-naive Date index."""
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
    """Return {ticker: normalized df} for a chunk. Retries on failure."""
    out = {}
    for attempt in range(MAX_RETRIES + 1):
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
            return out
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP * (attempt + 1))
            else:
                print(f"  chunk failed after {MAX_RETRIES} retries: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_START)
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    universe = build_universe()
    total = len(universe)
    print(f"Universe: {total} tickers")
    print(f"Output:   {OUT_PATH}")
    print(f"Period:   {args.start} -> today\n")

    all_frames = []
    failed = []
    t_start = time.time()

    for i in range(0, total, CHUNK_SIZE):
        chunk = universe[i:i + CHUNK_SIZE]
        print(f"[{i+1:>5}-{min(i+CHUNK_SIZE, total):>5} / {total}] downloading ({len(chunk)} tickers)...", flush=True)
        result = download_chunk(chunk, args.start)
        for t, df in result.items():
            df = df.copy()
            df["ticker"] = t
            df = df.reset_index().rename(columns={"index": "date", "Date": "date"})
            all_frames.append(df)
        for t in chunk:
            if t not in result:
                failed.append(t)
        time.sleep(0.3)

    if not all_frames:
        print("\nERROR: no data downloaded")
        return 1

    print("\nConsolidating...")
    master = pd.concat(all_frames, ignore_index=True)
    master = master.dropna(subset=["Close"])
    master = master.drop_duplicates(subset=["ticker", "date"], keep="last")
    for c in ["Open", "High", "Low", "Close"]:
        if c in master.columns:
            master[c] = master[c].astype("float32")
    if "Volume" in master.columns:
        master["Volume"] = master["Volume"].astype("float64")
    master["date"] = pd.to_datetime(master["date"])
    master = master[["ticker", "date", "Open", "High", "Low", "Close", "Volume"]]
    master = master.sort_values(["ticker", "date"]).reset_index(drop=True)

    master.to_parquet(OUT_PATH, compression="snappy", index=False)

    elapsed = time.time() - t_start
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  rows:    {len(master):,}")
    print(f"  tickers: {master['ticker'].nunique()}")
    print(f"  range:   {master['date'].min().date()} -> {master['date'].max().date()}")
    print(f"  file:    {size_mb:.1f} MB ({OUT_PATH})")
    if failed:
        print(f"\nFailed ({len(failed)}): {failed[:30]}{' ...' if len(failed) > 30 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
