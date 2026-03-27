"""
Build seasonal rank forecast for the sector trends page.

Generates 2026 (and optionally 2025) forecast ranks for all SECTOR_ETFS tickers.
Writes to a local SQLite DB (gitignored) and exports a CSV for the repo.

Usage:
    python build_sznl_forecast.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
import datetime as dt
from pandas.tseries.offsets import CustomBusinessDay
import sys
import os

# Import SECTOR_ETFS from sector_trends
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "pages"))
from sector_trends import SECTOR_ETFS

# --- CONFIGURATION ---
DB_PATH = os.path.join(current_dir, "data", "sznl_forecast.db")
CSV_PATH = os.path.join(current_dir, "sznl_sector_forecast.csv")
FORECAST_YEAR = 2026

# NYSE holidays for forecast years — prevents day_count drift vs historical data
NYSE_HOLIDAYS = [
    # 2025
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18",
    "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
    "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-06-19", "2026-07-03",
    "2026-09-07", "2026-11-26", "2026-12-25",
]


def get_forward_returns(df, windows=[5, 10, 21]):
    """Calculate forward log returns using trading-day indexing."""
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["year"] = df.index.year
    df["day_count"] = df.groupby("year").cumcount() + 1

    for w in windows:
        df[f"Fwd_{w}d"] = np.log(df["Close"].shift(-w) / df["Close"])

    return df


def calculate_forecast_profile(df, target_year):
    """
    Build seasonal rank profile for target_year using data strictly PRIOR to target_year.
    Walk-forward safe: no future data leakage.
    75% cycle-specific, 25% all-years weighting. 5d centered smoothing.
    Uses trading day count (not calendar day).
    """
    df = df[df["year"] < target_year].copy()
    if df.empty:
        return None

    fwd_cols = [f"Fwd_{w}d" for w in [5, 10, 21]]

    # All-years profile by trading day
    stats_all = df.groupby("day_count")[fwd_cols].mean()
    rank_all = stats_all.rank(pct=True) * 100

    # Cycle-specific profile
    cycle_remainder = target_year % 4
    cycle_data = df[df["year"] % 4 == cycle_remainder]

    if cycle_data.empty or len(cycle_data["year"].unique()) < 2:
        rank_cycle = rank_all.copy()
    else:
        stats_cycle = cycle_data.groupby("day_count")[fwd_cols].mean()
        rank_cycle = stats_cycle.rank(pct=True) * 100

    # Reindex to cover full range of trading days (max ~253)
    max_day = max(df["day_count"].max(), 253)
    full_idx = pd.RangeIndex(start=1, stop=max_day + 1)
    rank_all = rank_all.reindex(full_idx).interpolate(method="nearest").fillna(50)
    rank_cycle = rank_cycle.reindex(full_idx).interpolate(method="nearest").fillna(50)

    # Weighted average: 25% all years, 75% cycle
    avg_all = rank_all.mean(axis=1)
    avg_cycle = rank_cycle.mean(axis=1)
    final = (avg_all + 3 * avg_cycle) / 4

    # Smooth
    return final.rolling(5, center=True, min_periods=1).mean()


def generate_forecast_dates(year):
    """Generate trading days for a given year, skipping NYSE holidays."""
    market_day = CustomBusinessDay(holidays=NYSE_HOLIDAYS)
    dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq=market_day)
    forecast_df = pd.DataFrame({"Date": dates})
    forecast_df["day_count"] = range(1, len(dates) + 1)
    return forecast_df


def process_ticker(ticker):
    """Download history, build profile, return forecast DataFrame."""
    try:
        raw = yf.download(ticker, period="max", progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw.empty or len(raw) < 252:
            print(f"  {ticker}: insufficient data ({len(raw)} rows), skipping")
            return None

        df = get_forward_returns(raw)
        profile = calculate_forecast_profile(df, FORECAST_YEAR)

        forecast_dates = generate_forecast_dates(FORECAST_YEAR)
        forecast_dates["seasonal_rank"] = forecast_dates["day_count"].map(profile)
        forecast_dates["ticker"] = ticker

        return forecast_dates[["Date", "seasonal_rank", "ticker"]]

    except Exception as e:
        print(f"  {ticker}: error — {e}")
        return None


def main():
    tickers = sorted(set(SECTOR_ETFS))
    print(f"Building {FORECAST_YEAR} seasonal forecast for {len(tickers)} tickers...\n")

    # Ensure data dir exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    all_results = []
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        result = process_ticker(ticker)
        if result is not None:
            all_results.append(result)
            print(f"OK ({len(result)} days)")
        else:
            print("SKIP")

    if not all_results:
        print("\nNo results. Exiting.")
        return

    combined = pd.concat(all_results, ignore_index=True)

    # Write to SQLite (local, gitignored)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS seasonal_ranks")
    conn.execute(
        """CREATE TABLE seasonal_ranks (
            Date TIMESTAMP,
            seasonal_rank REAL,
            ticker TEXT,
            PRIMARY KEY (Date, ticker)
        )"""
    )
    combined.to_sql("seasonal_ranks", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print(f"\nSQLite DB written to {DB_PATH}")

    # Export CSV (committed to repo)
    combined.to_csv(CSV_PATH, index=False)
    print(f"CSV exported to {CSV_PATH} ({len(combined)} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
