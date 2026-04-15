"""
Build ATR-Normalized Seasonal Ranks
====================================
Computes per-ticker, per-day-of-year seasonal ranks using ATR-normalized
forward returns instead of raw percentage returns. This equalizes the
contribution of high-vol and low-vol regimes.

For each ticker on each trading day (day_count), ranks the historical average
ATR-normalized forward return against all other day_counts. Walk-forward safe:
ranks for year Y use only data from years < Y.

Output columns:
    atr_sznl_5d, atr_sznl_10d, atr_sznl_21d, atr_sznl_63d, atr_sznl_126d, atr_sznl_252d

Forward returns cross year boundaries (day 240 with a 63d window reaches
into the next year).

Usage:
    python build_atr_seasonal_ranks.py                    # 2026 only (live scan)
    python build_atr_seasonal_ranks.py --years 2025 2026  # specific years
    python build_atr_seasonal_ranks.py --full              # 2001-2026 (backtester)
    python build_atr_seasonal_ranks.py --tickers AAPL MSFT # specific tickers
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import sys
import time
import argparse
from pandas.tseries.offsets import CustomBusinessDay

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from strategy_config import CSV_UNIVERSE

# --- Config ---
CACHE_DIR = os.path.join(current_dir, "data")
OVERFLOW_CACHE = os.path.join(CACHE_DIR, "overflow_price_cache.parquet")
OUTPUT_PATH = os.path.join(current_dir, "atr_seasonal_ranks.parquet")
OUTPUT_CSV = os.path.join(current_dir, "atr_seasonal_ranks.csv")

ATR_WINDOW = 14
FWD_WINDOWS = [5, 10, 21, 63, 126, 252]
MAX_DAY_COUNT = 251  # Cap: day_counts above this have too few samples

DEFAULT_YEAR = 2026
FULL_START_YEAR = 2001


# ============================================================================
# DATA LOADING
# ============================================================================

def load_overflow_cache():
    """Load tickers from the overflow price cache if it exists."""
    if not os.path.exists(OVERFLOW_CACHE):
        return {}
    try:
        store = pd.read_parquet(OVERFLOW_CACHE)
        data_dict = {}
        for ticker in store['ticker'].unique():
            t_df = store[store['ticker'] == ticker].drop(columns=['ticker']).copy()
            t_df.index = pd.to_datetime(t_df['date'])
            t_df = t_df.drop(columns=['date']).sort_index()
            data_dict[ticker] = t_df
        print(f"   Loaded {len(data_dict)} tickers from overflow cache")
        return data_dict
    except Exception as e:
        print(f"   Overflow cache load failed: {e}")
        return {}


def download_tickers(tickers, start_date="1990-01-01"):
    """Download OHLCV for a list of tickers. Returns {ticker: DataFrame}."""
    data_dict = {}
    CHUNK_SIZE = 20
    total = len(tickers)

    for i in range(0, total, CHUNK_SIZE):
        chunk = tickers[i:i + CHUNK_SIZE]
        batch_num = i // CHUNK_SIZE + 1
        total_batches = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"   Batch {batch_num}/{total_batches} ({len(chunk)} tickers)...")

        for attempt in range(3):
            try:
                df = yf.download(
                    chunk, start=start_date, group_by='ticker',
                    auto_adjust=True, progress=False, threads=True
                )
                if df.empty:
                    break

                if len(chunk) == 1:
                    t = chunk[0]
                    if isinstance(df.columns, pd.MultiIndex):
                        # Could be (Price, Ticker) or (Ticker, Price) — pick the level with price names
                        lvl0 = df.columns.get_level_values(0).unique().tolist()
                        price_cols = {'Open', 'High', 'Low', 'Close', 'Volume',
                                      'open', 'high', 'low', 'close', 'volume'}
                        if set(lvl0) & price_cols:
                            df.columns = df.columns.get_level_values(0)
                        else:
                            df.columns = df.columns.get_level_values(1)
                    df.columns = [c.capitalize() for c in df.columns]
                    if 'Close' in df.columns and not df['Close'].dropna().empty:
                        df.index = df.index.normalize()
                        if df.index.tz is not None:
                            df.index = df.index.tz_localize(None)
                        data_dict[t] = df
                else:
                    if isinstance(df.columns, pd.MultiIndex):
                        lvl0 = df.columns.get_level_values(0).unique().tolist()
                        price_cols = {'Open', 'High', 'Low', 'Close', 'Volume',
                                      'open', 'high', 'low', 'close', 'volume'}
                        if set(lvl0) & price_cols:
                            # (Price, Ticker) format
                            for t in df.columns.get_level_values(1).unique():
                                try:
                                    t_df = df.xs(t, level=1, axis=1).copy()
                                    t_df.columns = [c.capitalize() for c in t_df.columns]
                                    if 'Close' in t_df.columns and not t_df['Close'].dropna().empty:
                                        t_df.index = t_df.index.normalize()
                                        if t_df.index.tz is not None:
                                            t_df.index = t_df.index.tz_localize(None)
                                        data_dict[str(t).upper()] = t_df.dropna(subset=['Close'])
                                except Exception:
                                    continue
                        else:
                            # (Ticker, Price) format
                            for t in lvl0:
                                try:
                                    t_df = df[t].copy()
                                    t_df.columns = [c.capitalize() for c in t_df.columns]
                                    if 'Close' in t_df.columns and not t_df['Close'].dropna().empty:
                                        t_df.index = t_df.index.normalize()
                                        if t_df.index.tz is not None:
                                            t_df.index = t_df.index.tz_localize(None)
                                        data_dict[str(t).upper()] = t_df.dropna(subset=['Close'])
                                except Exception:
                                    continue
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                else:
                    print(f"      Batch failed: {e}")

        time.sleep(1.0)

    return data_dict


# ============================================================================
# ATR & FORWARD RETURN COMPUTATION
# ============================================================================

def prepare_ticker_data(df):
    """Compute ATR, day_count, and forward ATR returns for a single ticker."""
    df = df.copy()
    if len(df) < ATR_WINDOW + 50:
        return None

    # ATR
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(ATR_WINDOW).mean()

    # Year and day_count — cap at MAX_DAY_COUNT so late-year days
    # with sparse samples don't produce noisy ranks
    df['year'] = df.index.year
    df['day_count'] = df.groupby('year').cumcount() + 1
    df['day_count'] = df['day_count'].clip(upper=MAX_DAY_COUNT)

    # Forward ATR returns — these naturally cross year boundaries
    for w in FWD_WINDOWS:
        fwd_price = df['Close'].shift(-w)
        df[f'fwd_atr_{w}d'] = (fwd_price - df['Close']) / df['ATR']

    # Drop rows without ATR
    df = df.dropna(subset=['ATR'])

    return df


# ============================================================================
# RANK COMPUTATION (walk-forward, cycle-weighted)
# ============================================================================

def compute_ranks_for_year(df, target_year):
    """
    Compute ATR seasonal ranks for a target year using only data < target_year.
    Returns a Series-per-window dict {window: Series indexed by day_count}.

    Weighting: 75% presidential cycle, 25% all-years.
    Smoothing: 5-day centered rolling.
    """
    hist = df[df['year'] < target_year].copy()
    if hist.empty or len(hist['year'].unique()) < 3:
        return None

    fwd_cols = [f'fwd_atr_{w}d' for w in FWD_WINDOWS]

    # All-years average by day_count
    avg_all = hist.groupby('day_count')[fwd_cols].mean()
    rank_all = avg_all.rank(pct=True) * 100

    # Cycle-specific (presidential cycle: same year % 4)
    cycle_remainder = target_year % 4
    cycle_data = hist[hist['year'] % 4 == cycle_remainder]

    if cycle_data.empty or len(cycle_data['year'].unique()) < 2:
        rank_cycle = rank_all.copy()
    else:
        avg_cycle = cycle_data.groupby('day_count')[fwd_cols].mean()
        rank_cycle = avg_cycle.rank(pct=True) * 100

    # Reindex to cover full day_count range (capped)
    full_idx = pd.RangeIndex(start=1, stop=MAX_DAY_COUNT + 1)
    rank_all = rank_all.reindex(full_idx).interpolate(method='nearest').fillna(50)
    rank_cycle = rank_cycle.reindex(full_idx).interpolate(method='nearest').fillna(50)

    # Weighted: 25% all-years + 75% cycle
    final = (rank_all + 3 * rank_cycle) / 4

    # Smooth each column independently
    for col in final.columns:
        final[col] = final[col].rolling(5, center=True, min_periods=1).mean()

    # Rename columns to output names
    final.columns = [f'atr_sznl_{w}d' for w in FWD_WINDOWS]

    return final


def generate_trading_dates(year):
    """Generate trading dates for a year using the actual NYSE calendar.
    Falls back to pandas business days with US federal holidays for years
    outside the exchange_calendars range."""
    try:
        import exchange_calendars as xcals
        nyse = xcals.get_calendar('XNYS')
        sessions = nyse.sessions_in_range(f"{year}-01-01", f"{year}-12-31")
        dates = sessions.tz_localize(None)
    except Exception:
        # Fallback for years outside exchange_calendars range
        from pandas.tseries.holiday import USFederalHolidayCalendar
        bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq=bday)
    day_counts = [min(i + 1, MAX_DAY_COUNT) for i in range(len(dates))]
    return pd.DataFrame({
        'Date': dates,
        'day_count': day_counts
    })


# ============================================================================
# MAIN BUILD
# ============================================================================

def build_atr_ranks(tickers, target_years, output_path=OUTPUT_PATH):
    """Build ATR seasonal ranks for all tickers and target years."""
    print(f"Building ATR seasonal ranks")
    print(f"  Tickers: {len(tickers)}")
    print(f"  Years: {target_years}")
    print(f"  Windows: {FWD_WINDOWS}")
    print()

    # 1. Load price data
    print("Loading price data...")
    clean_tickers = [t.replace('.', '-') for t in tickers]

    # Only load the heavy overflow cache for large runs
    if len(clean_tickers) > 50:
        cached = load_overflow_cache()
    else:
        cached = {}

    # 2. Download missing tickers
    missing = [t for t in clean_tickers if t not in cached]
    if missing:
        print(f"   Downloading {len(missing)} tickers...")
        fresh = download_tickers(missing)
        cached.update(fresh)
        print(f"   Total available: {len(cached)} tickers")
    else:
        print(f"   All {len(cached)} tickers in cache")

    # 3. Process each ticker
    print("\nComputing ranks...")
    all_results = []
    success = 0
    errors = 0

    for i, ticker in enumerate(clean_tickers):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"   [{i+1}/{len(clean_tickers)}] {ticker}...")

        raw_df = cached.get(ticker)
        if raw_df is None or raw_df.empty:
            errors += 1
            continue

        # Ensure column names are correct
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
        raw_df.columns = [c.capitalize() for c in raw_df.columns]

        if 'Close' not in raw_df.columns:
            errors += 1
            continue

        # Prepare data (ATR, forward returns)
        prepped = prepare_ticker_data(raw_df)
        if prepped is None:
            errors += 1
            continue

        # Compute ranks for each target year
        for year in target_years:
            ranks = compute_ranks_for_year(prepped, year)
            if ranks is None:
                continue

            # Map ranks to trading dates
            dates_df = generate_trading_dates(year)
            merged = dates_df.merge(
                ranks, left_on='day_count', right_index=True, how='left'
            )
            merged['ticker'] = ticker
            merged = merged.drop(columns=['day_count'])

            # Fill any gaps
            rank_cols = [f'atr_sznl_{w}d' for w in FWD_WINDOWS]
            merged[rank_cols] = merged[rank_cols].fillna(50.0).round(1)

            all_results.append(merged)

        success += 1

    if not all_results:
        print("No results generated!")
        return

    # 4. Combine and save
    result_df = pd.concat(all_results, ignore_index=True)
    result_df['Date'] = pd.to_datetime(result_df['Date'])

    print(f"\nSaving to {output_path}...")
    result_df.to_parquet(output_path, index=False)
    print(f"   {len(result_df):,} rows, {result_df['ticker'].nunique()} tickers")

    # Also save CSV for inspection
    csv_path = output_path.replace('.parquet', '.csv')
    result_df.to_csv(csv_path, index=False)
    print(f"   CSV copy: {csv_path}")

    print(f"\nDone: {success} tickers OK, {errors} errors")

    # Show sample
    sample_ticker = 'AAPL' if 'AAPL' in result_df['ticker'].values else result_df['ticker'].iloc[0]
    sample = result_df[result_df['ticker'] == sample_ticker].head(10)
    print(f"\nSample ({sample_ticker}):")
    print(sample.to_string(index=False))


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ATR-normalized seasonal ranks")
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Target years (default: 2026)")
    parser.add_argument("--full", action="store_true",
                        help=f"Compute all years from {FULL_START_YEAR} to {DEFAULT_YEAR}")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Specific tickers (default: full CSV_UNIVERSE)")
    args = parser.parse_args()

    if args.full:
        years = list(range(FULL_START_YEAR, DEFAULT_YEAR + 1))
    elif args.years:
        years = args.years
    else:
        years = [DEFAULT_YEAR]

    tickers = args.tickers if args.tickers else CSV_UNIVERSE

    build_atr_ranks(tickers, years)
