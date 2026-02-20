"""
Risk Dashboard V2 — Executive Summary + Absorption Ratio
=========================================================
Standalone market risk monitor.
6 validated signals from event study backtesting, flat signal board
with metric summaries, individual signal charts, and fragility dial.

Data: yfinance only. No broker connections. No strategy imports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import sys
import os
import plotly.graph_objects as go
import json

# ---------------------------------------------------------------------------
# PAGE CONFIG (must be first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Risk Dashboard V2",
    page_icon="\U0001f4ca",
    layout="wide",
)

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
VOL_TICKERS = ["SPY", "^VIX", "^VIX3M", "^VVIX"]
SECTOR_ETFS = [
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK",
    "XLP", "XLRE", "XLU", "XLV", "XLY",
]

# Cross-asset tickers (kept for future use — cached and cheap)
CROSS_ASSET_TICKERS = ['LQD', 'HYG', 'IEF', '^MOVE']

DATA_DIR = os.path.join(parent_dir, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SIGNAL_CACHE_PATH = os.path.join(DATA_DIR, "risk_dashboard_signal_state.json")

# Parquet cache paths
CACHE_SPY_OHLC = os.path.join(DATA_DIR, "rd2_spy_ohlc.parquet")
CACHE_CLOSES   = os.path.join(DATA_DIR, "rd2_closes.parquet")
CACHE_SP500    = os.path.join(DATA_DIR, "rd2_sp500_closes.parquet")

RISK_CLASSIFICATION_PATH = os.path.join(DATA_DIR, "sp500_risk_classification.csv")
HORIZON_STATS_PATH = os.path.join(DATA_DIR, "signal_horizon_stats.json")
SEASONAL_RANKS_PRIMARY = os.path.join(parent_dir, "sznl_ranks.csv")
SEASONAL_RANKS_BACKUP = os.path.join(parent_dir, "seasonal_ranks.csv")

ALL_SIGNAL_TICKERS = sorted(set(VOL_TICKERS + SECTOR_ETFS + CROSS_ASSET_TICKERS))

# FOMC announcement dates (for pre-FOMC rally signal)
FOMC_DATES = pd.to_datetime([
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17",
    "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
])


# ---------------------------------------------------------------------------
# DATA DOWNLOAD & CACHE LAYER
# ---------------------------------------------------------------------------

def _download_sp500_closes(tickers: list, start_date: str,
                           chunk_size: int = 50, sleep_between: float = 0.3,
                           progress_callback=None) -> pd.DataFrame:
    """
    Batch-download close prices for ~505 S&P 500 constituents.
    Returns wide DataFrame: index=date, columns=tickers, values=Close.
    """
    import time as _time

    clean = sorted(set(str(t).strip().upper().replace(".", "-") for t in tickers))
    total_batches = (len(clean) + chunk_size - 1) // chunk_size
    frames = []

    for i in range(0, len(clean), chunk_size):
        chunk = clean[i : i + chunk_size]
        batch_num = i // chunk_size + 1
        if progress_callback:
            progress_callback(batch_num, total_batches)
        try:
            raw = yf.download(chunk, start=start_date,
                              auto_adjust=True, progress=False, threads=True)
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                lvl0 = raw.columns.get_level_values(0).unique().tolist()
                key = "Close" if "Close" in lvl0 else ("close" if "close" in lvl0 else None)
                if key:
                    close_df = raw[key].copy()
                else:
                    continue
            else:
                cols = [str(c).capitalize() for c in raw.columns]
                raw.columns = cols
                if "Close" in raw.columns:
                    close_df = raw[["Close"]].copy()
                    close_df.columns = [chunk[0]] if len(chunk) == 1 else ["UNKNOWN"]
                else:
                    continue
            close_df.columns = [str(c).strip().upper() for c in close_df.columns]
            if close_df.index.tz is not None:
                close_df.index = close_df.index.tz_localize(None)
            frames.append(close_df)
        except Exception as e:
            print(f"Warning: S&P 500 batch {batch_num} failed: {e}")
            if "rate" in str(e).lower():
                _time.sleep(5)
        _time.sleep(sleep_between)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    if combined.index.tz is not None:
        combined.index = combined.index.tz_localize(None)
    return combined


def load_cached_data():
    """
    Load dashboard data from parquet cache.
    Returns (spy_df, closes, sp500_closes).
    spy_df: DataFrame with OHLC for SPY.
    closes: wide DataFrame of Close prices for all signal tickers.
    sp500_closes: wide DataFrame of Close prices for S&P 500 (or None).
    Returns (None, None, None) if core caches are missing.
    """
    if not os.path.exists(CACHE_SPY_OHLC) or not os.path.exists(CACHE_CLOSES):
        return None, None, None

    spy_df = pd.read_parquet(CACHE_SPY_OHLC)
    closes = pd.read_parquet(CACHE_CLOSES)

    sp500_closes = None
    if os.path.exists(CACHE_SP500):
        sp500_closes = pd.read_parquet(CACHE_SP500)

    return spy_df, closes, sp500_closes


def refresh_all_data(start_date: str, progress_callback=None):
    """
    Download all data fresh and save to parquet.
    progress_callback(step_text, pct) for Streamlit progress bar.
    """
    def _progress(text, pct):
        if progress_callback:
            progress_callback(text, pct)

    # Step 1: Download ~26 signal tickers (VOL + SECTOR + CROSS-ASSET)
    _progress("Downloading signal tickers...", 0.05)
    all_data = _download_ticker_group(ALL_SIGNAL_TICKERS, start_date)

    if "SPY" not in all_data:
        raise RuntimeError("Could not download SPY data. Check your internet connection.")

    # Step 2: Extract SPY OHLC and save
    _progress("Saving SPY OHLC...", 0.15)
    spy_df = all_data["SPY"]
    spy_df.to_parquet(CACHE_SPY_OHLC)

    # Step 3: Build wide closes DataFrame for all signal tickers and save
    _progress("Saving signal closes...", 0.20)
    close_frames = {}
    for ticker, df in all_data.items():
        if "Close" in df.columns:
            close_frames[ticker] = df["Close"]
    closes = pd.DataFrame(close_frames)
    closes.to_parquet(CACHE_CLOSES)

    # Step 4: Batch-download S&P 500 close prices
    sp500_tickers = None
    try:
        from abs_return_dispersion import SP500_TICKERS as _sp500
        sp500_tickers = _sp500
    except ImportError:
        pass

    if sp500_tickers and len(sp500_tickers) > 50:
        def _sp500_progress(batch_num, total_batches):
            pct = 0.25 + 0.70 * (batch_num / total_batches)
            _progress(f"S&P 500 batch {batch_num}/{total_batches}...", pct)

        sp500_closes = _download_sp500_closes(
            sp500_tickers, start_date, progress_callback=_sp500_progress
        )
        if not sp500_closes.empty:
            sp500_closes.to_parquet(CACHE_SP500)
            _progress("S&P 500 data saved.", 0.98)
        else:
            _progress("S&P 500 download returned no data.", 0.98)
    else:
        _progress("S&P 500 ticker list unavailable — skipping.", 0.98)

    _progress("Done.", 1.0)


def _download_ticker_group(tickers: list, start_date: str) -> dict:
    """
    Download OHLC data for a list of tickers.
    Returns dict of {ticker: DataFrame with [Open, High, Low, Close, Volume]}.
    Handles yfinance MultiIndex columns.
    """
    result = {}
    try:
        raw = yf.download(tickers, start=start_date, auto_adjust=True, threads=True)
    except Exception as e:
        st.warning(f"yfinance bulk download failed: {e}")
        return result

    if raw is None or raw.empty:
        return result

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                if isinstance(raw.columns, pd.MultiIndex):
                    df = raw.xs(ticker, level="Ticker", axis=1)
                else:
                    df = raw.copy()

            # Flatten any remaining MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = [c.capitalize() for c in df.columns]
            df = df.dropna(how="all")

            if not df.empty:
                result[ticker] = df
        except Exception as e:
            print(f"Warning: Could not process {ticker}: {e}")

    return result


# ---------------------------------------------------------------------------
# ABSORPTION RATIO (core computation — used by Low AR signal)
# ---------------------------------------------------------------------------

def compute_absorption_ratio(sector_returns_df: pd.DataFrame, window: int = 63) -> pd.Series:
    """
    Rolling absorption ratio: fraction of total variance
    explained by first principal component.
    """
    ar_series = pd.Series(dtype=float, index=sector_returns_df.index)

    for i in range(window, len(sector_returns_df)):
        window_data = sector_returns_df.iloc[i - window:i].dropna(axis=1)
        if window_data.shape[1] < 5:
            continue
        try:
            cov_matrix = window_data.cov()
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]
            ar_series.iloc[i] = eigenvalues[0] / eigenvalues.sum()
        except Exception:
            continue

    return ar_series


# ---------------------------------------------------------------------------
# PERCENTILE HELPERS
# ---------------------------------------------------------------------------

def expanding_percentile(series: pd.Series, min_periods: int = 252) -> pd.Series:
    """Expanding percentile rank (0-100)."""
    def _pctile(x):
        if len(x) < min_periods:
            return np.nan
        return (x.values[:-1] < x.values[-1]).sum() / (len(x) - 1) * 100
    return series.expanding(min_periods=min_periods).apply(_pctile, raw=False)


def _rolling_percentile(series: pd.Series, lookback: int) -> pd.Series:
    """Fixed-lookback rolling percentile rank (0-100)."""
    arr = series.values
    out = np.full(len(arr), np.nan)
    min_valid = int(lookback * 0.8)
    for i in range(lookback, len(arr)):
        window = arr[i - lookback : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < min_valid:
            continue
        out[i] = (valid[:-1] < valid[-1]).sum() / (len(valid) - 1) * 100
    return pd.Series(out, index=series.index)


# ---------------------------------------------------------------------------
# 6 VALIDATED SIGNAL FUNCTIONS
# Each returns a dict with: on, detail, summary, and chart series.
# ---------------------------------------------------------------------------

def compute_da_signal(spy_df: pd.DataFrame) -> dict:
    """
    Distribution/Accumulation signal.
    Institutional selling beneath the surface near highs.

    Two levels:
      - WARNING: ratio > 3.75, SPY within 2% of 52w high, SPY > 50d SMA
      - ELEVATED: ratio > 6.0, SPY within 2% of 52w high (special risk-off)
    """
    vol_mult = 1.15
    window = 63
    threshold = 3.75
    elevated_threshold = 6.0
    near_high_pct = 2.0

    close = spy_df["Close"]
    open_ = spy_df["Open"]
    volume = spy_df["Volume"]

    avg_vol_63 = volume.rolling(63).mean()
    vol_above_avg = volume > avg_vol_63
    vol_surge_vs_prev = volume > (volume.shift(1) * vol_mult)
    vol_qualified = vol_above_avg & vol_surge_vs_prev

    dist_days = vol_qualified & (close < open_)
    accum_days = vol_qualified & (close > open_)

    dist_count = dist_days.astype(int).rolling(window).sum()
    accum_count = accum_days.astype(int).rolling(window).sum()

    # When accum > 0: ratio = dist / accum
    # When accum = 0: ratio = dist - accum + 1 (so 4D/0A = 5)
    da_ratio = pd.Series(
        np.where(accum_count > 0, dist_count / accum_count, dist_count + 1),
        index=dist_count.index,
    )

    # Filters
    sma_50 = close.rolling(50).mean()
    high_52w = close.rolling(252, min_periods=60).max()
    near_high = close >= high_52w * (1 - near_high_pct / 100)

    signal = (da_ratio > threshold) & (close > sma_50) & near_high
    elevated = (da_ratio > elevated_threshold) & near_high

    latest_ratio = float(da_ratio.iloc[-1]) if len(da_ratio) > 0 and not np.isnan(da_ratio.iloc[-1]) else 0.0
    signal_on = bool(signal.iloc[-1]) if len(signal) > 0 and not pd.isna(signal.iloc[-1]) else False
    elevated_on = bool(elevated.iloc[-1]) if len(elevated) > 0 and not pd.isna(elevated.iloc[-1]) else False

    recent_dist = int(dist_days.iloc[-window:].sum())
    recent_accum = int(accum_days.iloc[-window:].sum())

    detail = ""
    if elevated_on:
        detail = (
            f"D/A ratio at {latest_ratio:.1f} \u2014 ELEVATED RISK. "
            f"{recent_dist} distribution vs {recent_accum} accumulation days "
            f"over {window} sessions while SPY within {near_high_pct:.0f}% of 52w high. "
            f"Extreme institutional selling near highs \u2014 consider reducing exposure."
        )
    elif signal_on:
        detail = (
            f"D/A ratio at {latest_ratio:.1f} over last {window} sessions "
            f"({recent_dist} distribution vs {recent_accum} accumulation days). "
            f"SPY within {near_high_pct:.0f}% of 52w high \u2014 "
            f"volume-confirmed selling beneath the surface."
        )

    # Combine: elevated overrides base signal
    is_on = signal_on or elevated_on

    summary = (
        f"D/A ratio: {latest_ratio:.1f} (threshold: {threshold}) "
        f"\u2014 {recent_dist}D / {recent_accum}A last {window}d"
    )
    if elevated_on:
        summary += f" \u2014 ABOVE {elevated_threshold:.0f} ELEVATED THRESHOLD"

    return {
        'on': is_on,
        'elevated': elevated_on,
        'detail': detail,
        'summary': summary,
        'da_ratio': da_ratio,
        'signal_history': signal | elevated,
    }


def compute_vix_range_compression(vix_close: pd.Series) -> dict:
    """
    VIX Range Compression signal.
    VIX in a tight squeeze — eventual breakout tends to be violent.
    """
    empty = {
        'on': False, 'detail': '', 'summary': 'VIX data unavailable',
        'compression_pctile': pd.Series(dtype=float),
        'signal_history': pd.Series(dtype=bool),
    }
    if len(vix_close) < 504:
        return empty

    range_window = 21
    pctile_threshold = 15
    min_vix = 13
    lookback = 504
    sma_period = 20

    compression_metric = vix_close.rolling(range_window).max() - vix_close.rolling(range_window).min()
    compression_pctile = _rolling_percentile(compression_metric, lookback)

    vix_sma = vix_close.rolling(sma_period, min_periods=int(sma_period * 0.8)).mean()

    signal = (compression_pctile < pctile_threshold) & (vix_close > min_vix) & (vix_close > vix_sma)

    latest_pctile = float(compression_pctile.iloc[-1]) if not np.isnan(compression_pctile.iloc[-1]) else 50.0
    signal_on = bool(signal.iloc[-1]) if len(signal) > 0 and not pd.isna(signal.iloc[-1]) else False
    cur_vix = float(vix_close.iloc[-1])

    detail = ""
    if signal_on:
        detail = (
            f"VIX {range_window}d range at {latest_pctile:.0f}th percentile "
            f"(threshold: {pctile_threshold}th). VIX at {cur_vix:.1f} is compressed \u2014 "
            f"eventual breakout tends to be violent."
        )

    summary = (
        f"{range_window}d range: {latest_pctile:.0f}th pctile "
        f"(fires below {pctile_threshold}th) \u2014 VIX at {cur_vix:.1f}"
    )

    return {
        'on': signal_on,
        'detail': detail,
        'summary': summary,
        'compression_pctile': compression_pctile,
        'signal_history': signal,
    }


def compute_defensive_leadership(sp500_closes, spy_close: pd.Series) -> dict:
    """
    Defensive Leadership signal — two tiers.

    WARNING: 50d spread < -10pp AND 200d spread < 0pp AND SPY within 2% of 52w high.
    DIRE:    Same conditions but SPY within 1% of 52w high — auto put/put-spread territory.
    """
    empty = {
        'on': False, 'elevated': False, 'detail': '', 'summary': 'S&P 500 data unavailable',
        'spread': pd.Series(dtype=float),
        'on_breadth': pd.Series(dtype=float),
        'off_breadth': pd.Series(dtype=float),
        'signal_history': pd.Series(dtype=bool),
    }
    if sp500_closes is None or (hasattr(sp500_closes, 'empty') and sp500_closes.empty):
        return empty

    if not os.path.exists(RISK_CLASSIFICATION_PATH):
        empty['summary'] = 'Risk classification file missing'
        return empty

    classification = pd.read_csv(RISK_CLASSIFICATION_PATH)
    on_tickers = classification.loc[classification["label"] == "risk_on", "ticker"].tolist()
    off_tickers = classification.loc[classification["label"] == "risk_off", "ticker"].tolist()

    on_cols = [t for t in on_tickers if t in sp500_closes.columns]
    off_cols = [t for t in off_tickers if t in sp500_closes.columns]

    if len(on_cols) < 10 or len(off_cols) < 10:
        empty['summary'] = f'Too few classified stocks ({len(on_cols)} on / {len(off_cols)} off)'
        return empty

    # 50d breadth (primary signal)
    sma_50 = sp500_closes.rolling(50, min_periods=40).mean()
    on_above_50 = (sp500_closes[on_cols] > sma_50[on_cols]).sum(axis=1) / len(on_cols) * 100
    off_above_50 = (sp500_closes[off_cols] > sma_50[off_cols]).sum(axis=1) / len(off_cols) * 100
    spread_50 = on_above_50 - off_above_50

    # 200d breadth (confirmation)
    sma_200 = sp500_closes.rolling(200, min_periods=160).mean()
    on_above_200 = (sp500_closes[on_cols] > sma_200[on_cols]).sum(axis=1) / len(on_cols) * 100
    off_above_200 = (sp500_closes[off_cols] > sma_200[off_cols]).sum(axis=1) / len(off_cols) * 100
    spread_200 = on_above_200 - off_above_200

    # SPY proximity to 52w high
    high_52w = spy_close.rolling(252, min_periods=60).max()
    near_high_2pct = spy_close >= high_52w * 0.98  # within 2%
    near_high_1pct = spy_close >= high_52w * 0.99  # within 1%

    # Base conditions: 50d spread < -10pp AND 200d spread < 0pp
    base_condition = (spread_50 < -10) & (spread_200 < 0)

    # WARNING: base + within 2% of high
    signal = base_condition & near_high_2pct
    # DIRE: base + within 1% of high
    dire = base_condition & near_high_1pct

    latest_spread_50 = float(spread_50.iloc[-1]) if len(spread_50) > 0 and not np.isnan(spread_50.iloc[-1]) else 0.0
    latest_spread_200 = float(spread_200.iloc[-1]) if len(spread_200) > 0 and not np.isnan(spread_200.iloc[-1]) else 0.0
    signal_on = bool(signal.iloc[-1]) if len(signal) > 0 and not pd.isna(signal.iloc[-1]) else False
    dire_on = bool(dire.iloc[-1]) if len(dire) > 0 and not pd.isna(dire.iloc[-1]) else False

    on_pct_50 = float(on_above_50.iloc[-1]) if len(on_above_50) > 0 and not np.isnan(on_above_50.iloc[-1]) else 0.0
    off_pct_50 = float(off_above_50.iloc[-1]) if len(off_above_50) > 0 and not np.isnan(off_above_50.iloc[-1]) else 0.0

    detail = ""
    if dire_on:
        detail = (
            f"DIRE: Risk-off leading on BOTH timeframes while SPY within 1% of 52w high. "
            f"50d spread: {latest_spread_50:+.0f}pp | 200d spread: {latest_spread_200:+.0f}pp. "
            f"Institutional rotation to safety at all-time highs \u2014 "
            f"consider 3-month put / put-spread."
        )
    elif signal_on:
        detail = (
            f"Risk-off stocks leading: 50d spread {latest_spread_50:+.0f}pp, "
            f"200d spread {latest_spread_200:+.0f}pp. "
            f"SPY within 2% of 52w high \u2014 defensive rotation while index holds."
        )

    is_on = signal_on or dire_on

    summary = (
        f"50d spread: {latest_spread_50:+.0f}pp (fires < -10) | "
        f"200d spread: {latest_spread_200:+.0f}pp (fires < 0)"
    )
    if dire_on:
        summary += " \u2014 DIRE: within 1% of highs"

    return {
        'on': is_on,
        'elevated': dire_on,
        'detail': detail,
        'summary': summary,
        'spread': spread_50,
        'on_breadth': on_above_50,
        'off_breadth': off_above_50,
        'signal_history': signal | dire,
    }


def compute_fomc_signal(spy_close: pd.Series) -> dict:
    """
    Pre-FOMC Rally signal.
    Strong run-up into FOMC meeting increases reversal risk.
    """
    today = spy_close.index[-1]
    pre_window = 5
    pctile_threshold = 75
    lookback = 504

    # Check if we're within ~5 trading days before any FOMC date
    in_pre_fomc = False
    next_fomc = None

    for fomc_date in FOMC_DATES:
        days_ahead = (fomc_date - today).days
        if 0 <= days_ahead <= 8:  # ~5 trading days = 7-8 calendar days
            in_pre_fomc = True
            next_fomc = fomc_date
            break

    # Also find the next upcoming FOMC for summary even when outside window
    next_upcoming = None
    for fomc_date in FOMC_DATES:
        if fomc_date >= today:
            next_upcoming = fomc_date
            break

    # Compute 5d trailing return percentile
    pre_return = spy_close.pct_change(pre_window)
    pre_pctile = _rolling_percentile(pre_return, lookback)

    latest_pctile = float(pre_pctile.iloc[-1]) if len(pre_pctile) > 0 and not np.isnan(pre_pctile.iloc[-1]) else 0.0

    signal_on = in_pre_fomc and latest_pctile > pctile_threshold

    # Historical signal fire dates: FOMC dates where 5d return pctile > threshold
    signal_dates = []
    for fomc_date in FOMC_DATES:
        if fomc_date not in pre_pctile.index:
            # Snap to nearest trading day
            future = pre_pctile.index[pre_pctile.index >= fomc_date]
            if len(future) > 0 and (future[0] - fomc_date).days <= 5:
                snap = future[0]
            else:
                continue
        else:
            snap = fomc_date
        pct_val = pre_pctile.get(snap, np.nan)
        if not np.isnan(pct_val) and pct_val > pctile_threshold:
            signal_dates.append(snap)

    detail = ""
    if signal_on:
        fomc_str = next_fomc.strftime('%b %d') if next_fomc is not None else "upcoming"
        detail = (
            f"FOMC on {fomc_str} \u2014 SPY trailing 5d return at "
            f"{latest_pctile:.0f}th percentile. Strong pre-FOMC rally increases "
            f"'buy the rumor, sell the news' reversal risk."
        )

    if in_pre_fomc:
        fomc_str = next_fomc.strftime('%b %d')
        days_cal = (next_fomc - today).days
        summary = (
            f"Next FOMC: {fomc_str} ({days_cal}d away) \u2014 "
            f"5d return: {latest_pctile:.0f}th pctile (fires above {pctile_threshold}th)"
        )
    elif next_upcoming is not None:
        days_cal = (next_upcoming - today).days
        summary = (
            f"Next FOMC: {next_upcoming.strftime('%b %d')} ({days_cal}d away) \u2014 "
            f"5d return: {latest_pctile:.0f}th pctile (outside window)"
        )
    else:
        summary = "No upcoming FOMC dates in calendar"

    # Build boolean signal_history from point events
    # Each FOMC signal fire covers the ~5 trading days leading up to the FOMC date
    signal_history = pd.Series(False, index=spy_close.index)
    for sd in signal_dates:
        # Mark the 5 trading days up to and including the signal date
        mask = (spy_close.index <= sd) & (spy_close.index >= sd - pd.Timedelta(days=8))
        signal_history.loc[mask] = True

    return {
        'on': signal_on,
        'detail': detail,
        'summary': summary,
        'signal_dates': signal_dates,
        'signal_history': signal_history,
    }


def compute_low_ar_signal(sector_returns: pd.DataFrame, spy_close: pd.Series) -> dict:
    """
    Low Absorption Ratio signal.
    Low AR = sectors moving independently (Minsky quiet phase, vol suppression).
    When combined with SPY near highs, indicates complacency / fragility.
    """
    pca_window = 21
    pctile_lookback = 504
    pctile_threshold = 10
    near_high_pct = 2.0

    empty = {
        'on': False, 'detail': '', 'summary': 'AR data unavailable',
        'ar_series': pd.Series(dtype=float), 'ar_pctile': pd.Series(dtype=float),
        'signal_history': pd.Series(dtype=bool),
    }

    if len(sector_returns.columns) < 5 or len(sector_returns) < pca_window + 10:
        return empty

    ar_series = compute_absorption_ratio(sector_returns, window=pca_window)
    ar_valid = ar_series.dropna()
    if len(ar_valid) < pctile_lookback:
        return {**empty, 'ar_series': ar_series,
                'summary': f'AR: insufficient history ({len(ar_valid)} days, need {pctile_lookback})'}

    ar_pctile = _rolling_percentile(ar_valid, pctile_lookback)

    # Filters
    high_52w = spy_close.rolling(252, min_periods=60).max()
    near_high = spy_close >= high_52w * (1 - near_high_pct / 100)

    # Signal: low AR percentile while SPY near highs (full history + latest)
    ar_pctile_aligned = ar_pctile.reindex(spy_close.index)
    signal_full = (ar_pctile_aligned < pctile_threshold) & near_high

    latest_ar = float(ar_valid.iloc[-1])
    latest_pctile = float(ar_pctile.iloc[-1]) if len(ar_pctile) > 0 and not np.isnan(ar_pctile.iloc[-1]) else 50.0
    latest_near_high = bool(near_high.iloc[-1]) if len(near_high) > 0 and not pd.isna(near_high.iloc[-1]) else False

    signal_on = latest_pctile < pctile_threshold and latest_near_high

    detail = ""
    if signal_on:
        detail = (
            f"Absorption Ratio at {latest_ar:.3f} ({latest_pctile:.0f}th percentile) "
            f"\u2014 below {pctile_threshold}th threshold. Sectors moving independently "
            f"while SPY near 52w high \u2014 vol suppression through diversification. "
            f"Historically precedes below-average forward returns."
        )

    summary = (
        f"AR: {latest_ar:.3f} ({latest_pctile:.0f}th pctile, fires below "
        f"{pctile_threshold}th) \u2014 near high: {'yes' if latest_near_high else 'no'}"
    )

    return {
        'on': signal_on,
        'detail': detail,
        'summary': summary,
        'ar_series': ar_series,
        'ar_pctile': ar_pctile,
        'signal_history': signal_full,
    }


@st.cache_data(ttl=86400)
def _load_seasonal_spread() -> pd.Series | None:
    """
    Load seasonal rank spread (risk-off avg - risk-on avg) from CSV.
    Ticker universe locked to seasonal_ranks.csv (backtested set).
    sznl_ranks.csv used only for 2026 date coverage with same tickers.
    Cached for 24h since files are updated annually.
    """
    if not os.path.exists(RISK_CLASSIFICATION_PATH):
        return None
    if not os.path.exists(SEASONAL_RANKS_BACKUP):
        return None

    # Load backtested universe to lock ticker set
    sr_base = pd.read_csv(SEASONAL_RANKS_BACKUP, parse_dates=['Date'])
    base_tickers = set(sr_base['ticker'].unique())

    # Extend with sznl_ranks.csv for 2026 dates, restricted to same tickers
    if os.path.exists(SEASONAL_RANKS_PRIMARY):
        sr_ext = pd.read_csv(SEASONAL_RANKS_PRIMARY, parse_dates=['Date'])
        sr_ext = sr_ext[sr_ext['ticker'].isin(base_tickers)]
        sr = pd.concat([sr_base, sr_ext], ignore_index=True)
        sr = sr.drop_duplicates(subset=['Date', 'ticker'], keep='first')
    else:
        sr = sr_base

    rc = pd.read_csv(RISK_CLASSIFICATION_PATH)

    # Add tickers not in S&P 500 classification
    extra = pd.DataFrame([
        {'ticker': 'HIG', 'label': 'risk_off'},
        {'ticker': 'K', 'label': 'risk_off'},
        {'ticker': 'LEG', 'label': 'risk_off'},
    ])
    rc = pd.concat([rc, extra], ignore_index=True)
    label_map = dict(zip(rc['ticker'], rc['label']))

    # Individual stocks only — skip ETFs, indices, commodities
    skip = {
        'VFC', 'SPY', 'QQQ', 'DIA', 'IWM', '^GSPC', '^NDX',
        'GLD', 'SLV', 'USO', 'UNG', 'UVXY', 'VNQ', 'IYR',
        'SMH', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK',
        'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOP', 'XRT',
        'IBB', 'IHI', 'ITA', 'ITB', 'KRE', 'OIH', 'CEF', 'PSA', 'SPG',
    }
    sr_c = sr[sr['ticker'].isin(label_map) & ~sr['ticker'].isin(skip)].copy()
    sr_c['label'] = sr_c['ticker'].map(label_map)

    pivot = sr_c.pivot_table(index='Date', columns='ticker', values='seasonal_rank')
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()

    on_cols = [t for t in sr_c[sr_c['label'] == 'risk_on']['ticker'].unique() if t in pivot.columns]
    off_cols = [t for t in sr_c[sr_c['label'] == 'risk_off']['ticker'].unique() if t in pivot.columns]

    if len(on_cols) < 10 or len(off_cols) < 10:
        return None

    return pivot[off_cols].mean(axis=1) - pivot[on_cols].mean(axis=1)


def compute_seasonal_divergence_signal(spy_close: pd.Series) -> dict:
    """
    Seasonal Rank Divergence signal.
    Risk-off stocks have stronger seasonals than risk-on while SPY near highs.
    Short-to-intermediate term signal (5d-21d edge).
    """
    threshold = 10  # pp spread
    near_high_pct = 2.0

    empty = {
        'on': False, 'detail': '', 'summary': 'Seasonal rank data unavailable',
        'spread': pd.Series(dtype=float),
        'signal_history': pd.Series(dtype=bool),
    }

    spread = _load_seasonal_spread()
    if spread is None or len(spread) < 252:
        return empty

    # Align with SPY
    common = spread.index.intersection(spy_close.index)
    if len(common) < 252:
        return empty
    spread_aligned = spread.loc[common]
    spy = spy_close.loc[common]

    # Near-high filter
    high_52w = spy.rolling(252, min_periods=60).max()
    near_high = spy >= high_52w * (1 - near_high_pct / 100)

    # Signal
    signal = (spread_aligned > threshold) & near_high

    latest_spread = float(spread_aligned.iloc[-1])
    signal_on = bool(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else False
    latest_near_high = bool(near_high.iloc[-1]) if not pd.isna(near_high.iloc[-1]) else False

    detail = ""
    if signal_on:
        detail = (
            f"Seasonal rank spread at {latest_spread:+.1f}pp (risk-off minus risk-on). "
            f"Defensive stocks have stronger seasonal tailwinds while SPY near 52w high "
            f"\u2014 historically precedes below-average short-term returns."
        )

    summary = (
        f"Spread: {latest_spread:+.1f}pp (fires above {threshold}) "
        f"\u2014 near high: {'yes' if latest_near_high else 'no'}"
    )

    return {
        'on': signal_on,
        'detail': detail,
        'summary': summary,
        'spread': spread_aligned,
        'signal_history': signal,
    }


# ---------------------------------------------------------------------------
# SIGNAL FRAMEWORK
# ---------------------------------------------------------------------------

def compute_price_context(spy_close: pd.Series) -> dict:
    """
    Compute the three dimensions of price context.
    Returns dict with all values needed for the banner.
    """
    latest = float(spy_close.iloc[-1])

    # Trailing 12-month return
    if len(spy_close) >= 252:
        ret_12m = (latest / float(spy_close.iloc[-252]) - 1)
    else:
        ret_12m = None

    # Extension vs 200d SMA
    sma_200 = float(spy_close.rolling(200).mean().iloc[-1]) if len(spy_close) >= 200 else None
    extension_200d = (latest / sma_200 - 1) if sma_200 else None

    # Drawdown from 52-week high
    high_52w = float(spy_close.rolling(252).max().iloc[-1]) if len(spy_close) >= 252 else None
    drawdown = (latest / high_52w - 1) if high_52w else None

    # Price regime label
    if ret_12m is not None and extension_200d is not None and drawdown is not None:
        if drawdown < -0.10:
            regime_label = "Significant drawdown in progress"
        elif drawdown < -0.05:
            regime_label = "Correction underway"
        elif ret_12m > 0.20 and extension_200d > 0.08:
            regime_label = "Extended uptrend, stretched above trend"
        elif ret_12m > 0.15 and extension_200d > 0.05:
            regime_label = "Strong uptrend, moderately extended"
        elif ret_12m > 0.10:
            regime_label = "Healthy uptrend"
        elif ret_12m > 0:
            regime_label = "Modest gains, near trend"
        elif ret_12m > -0.05:
            regime_label = "Flat to slightly negative"
        else:
            regime_label = "Downtrend"
    else:
        regime_label = "Insufficient data"

    return {
        'price': latest,
        'ret_12m': ret_12m,
        'extension_200d': extension_200d,
        'drawdown': drawdown,
        'sma_200': sma_200,
        'high_52w': high_52w,
        'regime_label': regime_label,
    }


def compute_regime_multiplier(price_ctx: dict) -> float:
    """
    Amplify or dampen fragility signals based on price context.

    At extended highs: fragility signals are MORE dangerous -> multiplier > 1
    In corrections: fragility signals are LESS dangerous -> multiplier < 1

    Returns multiplier between 0.6 and 1.8.
    """
    m = 1.0

    ret = price_ctx.get('ret_12m')
    ext = price_ctx.get('extension_200d')
    dd = price_ctx.get('drawdown')

    if ret is not None:
        if ret > 0.25:
            m += 0.25
        elif ret > 0.15:
            m += 0.10
        elif ret < -0.05:
            m -= 0.15

    if ext is not None:
        if ext > 0.10:
            m += 0.25
        elif ext > 0.05:
            m += 0.10
        elif ext < -0.02:
            m -= 0.15

    if dd is not None:
        if dd > -0.02:      # Near highs
            m += 0.10
        elif dd < -0.10:    # Deep drawdown
            m -= 0.20

    return max(0.6, min(1.8, m))


# ---------------------------------------------------------------------------
# EXECUTIVE SUMMARY RENDERING
# ---------------------------------------------------------------------------

def render_price_context(price_ctx: dict):
    """Render the price context banner."""
    p = price_ctx

    ret_str = f"{p['ret_12m']:+.1%}" if p['ret_12m'] is not None else "N/A"
    ext_str = f"{p['extension_200d']:+.1%}" if p['extension_200d'] is not None else "N/A"
    dd_str = f"{p['drawdown']:+.1%}" if p['drawdown'] is not None else "N/A"

    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
                padding: 10px 16px; border-radius: 6px; margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="font-size: 15px; font-weight: 600;">SPY: ${p['price']:.2f}</span>
                <span style="font-size: 13px; color: #999; margin-left: 12px;">
                    {ret_str} 12mo &nbsp;|&nbsp; {ext_str} vs 200d &nbsp;|&nbsp; {dd_str} from high
                </span>
            </div>
            <div style="font-size: 13px; color: #bbb;">
                {p['regime_label']}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_signal_board(signals: dict, price_ctx: dict):
    """Render flat signal list with metric summaries (4-tier: ELEVATED/ON/DECAYING/OFF)."""
    dd = price_ctx.get('drawdown')
    spy_pct_from_high = abs(dd) if dd is not None else 0.0

    for name, sig in signals.items():
        on = sig['on']
        elevated = sig.get('elevated', False)
        summary = sig['summary']
        detail = sig.get('detail', '')

        if elevated:
            st.markdown(
                f"<div style='padding: 6px 10px; margin-bottom: 6px; "
                f"background: rgba(204,0,0,0.15); border-left: 4px solid #FF0000; "
                f"border-radius: 0 4px 4px 0;'>"
                f"<span style='font-size: 13px;'>\U0001f6a8 <strong>{name}</strong> "
                f"<span style='color: #FF4444; font-size: 11px; font-weight: 600;'>ELEVATED</span>"
                f"</span><br>"
                f"<span style='font-size: 12px; color: #bbb;'>{detail}</span><br>"
                f"<span style='font-size: 11px; color: #888;'>{summary}</span></div>",
                unsafe_allow_html=True
            )
        elif on:
            st.markdown(
                f"<div style='padding: 6px 10px; margin-bottom: 6px; "
                f"background: rgba(204,0,0,0.08); border-left: 3px solid #CC0000; "
                f"border-radius: 0 4px 4px 0;'>"
                f"<span style='font-size: 13px;'>\U0001f534 <strong>{name}</strong></span><br>"
                f"<span style='font-size: 12px; color: #bbb;'>{detail}</span><br>"
                f"<span style='font-size: 11px; color: #888;'>{summary}</span></div>",
                unsafe_allow_html=True
            )
        else:
            decay_meta = _compute_decay_metadata(sig, spy_pct_from_high)
            if decay_meta is not None:
                ds = decay_meta['days_since']
                mr = decay_meta['max_remaining_days']
                # Build per-horizon weight summary
                h_parts = []
                for h_label in ('5d', '21d', '63d'):
                    h = decay_meta['horizons'][h_label]
                    if h['weight'] == 0.0:
                        h_parts.append(f"{h_label}: expired")
                    else:
                        h_parts.append(f"{h_label}: {h['weight']:.0%}")
                h_line = ' | '.join(h_parts)

                st.markdown(
                    f"<div style='padding: 6px 10px; margin-bottom: 6px; "
                    f"background: rgba(255,183,0,0.10); border-left: 3px solid #FFB700; "
                    f"border-radius: 0 4px 4px 0;'>"
                    f"<span style='font-size: 13px;'>\U0001f7e1 <strong>{name}</strong> "
                    f"<span style='color: #FFB700; font-size: 11px; font-weight: 600;'>DECAYING</span> "
                    f"<span style='font-size: 11px; color: #999;'>fired {ds}d ago — expires in {mr}d</span>"
                    f"</span><br>"
                    f"<span style='font-size: 11px; color: #BB9500;'>Remaining weight: {h_line}</span><br>"
                    f"<span style='font-size: 11px; color: #888;'>{summary}</span></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='padding: 4px 10px; margin-bottom: 4px;'>"
                    f"<span style='font-size: 13px; color: #aaa;'>\U0001f7e2 {name}</span><br>"
                    f"<span style='font-size: 11px; color: #666; margin-left: 4px;'>{summary}</span></div>",
                    unsafe_allow_html=True
                )


# ---------------------------------------------------------------------------
# SIGNAL STATE PERSISTENCE
# ---------------------------------------------------------------------------

def load_previous_signal_state() -> dict:
    """Load yesterday's signal states from cache file."""
    try:
        if os.path.exists(SIGNAL_CACHE_PATH):
            with open(SIGNAL_CACHE_PATH, 'r') as f:
                data = json.load(f)
            # Only use if from a previous calendar date
            cached_date = data.get('date', '')
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            if cached_date != today:
                return data
        return {}
    except Exception:
        return {}


def save_current_signal_state(signal_states: dict):
    """Save current signal states for tomorrow's comparison."""
    try:
        signal_states['date'] = datetime.datetime.now().strftime('%Y-%m-%d')
        with open(SIGNAL_CACHE_PATH, 'w') as f:
            json.dump(signal_states, f)
    except Exception:
        pass


def compute_changes(current_states: dict, previous_states: dict) -> list:
    """
    Compare current vs previous signal states.
    Returns list of change description strings.
    """
    changes = []
    prev_signals = previous_states.get('signals', {})

    for name, cur_active in current_states.get('signals', {}).items():
        prev_active = prev_signals.get(name, False)
        if cur_active and not prev_active:
            changes.append(f"\U0001f534 **{name}** activated")
        elif not cur_active and prev_active:
            changes.append(f"\U0001f7e2 **{name}** deactivated")

    return changes


# ---------------------------------------------------------------------------
# FRAGILITY SCORE & RISK DIAL
# ---------------------------------------------------------------------------

def load_horizon_stats() -> dict | None:
    """Load backtested signal horizon stats from JSON."""
    if not os.path.exists(HORIZON_STATS_PATH):
        return None
    with open(HORIZON_STATS_PATH, 'r') as f:
        return json.load(f)


def _signal_edge(stats: dict, signal_key: str, horizon: str) -> float:
    """Return the downside edge (positive = worse) for a signal at a horizon."""
    sig = stats.get('signals', {}).get(signal_key, {})
    dm = sig.get('horizons', {}).get(horizon, {}).get('diff_mean', 0)
    if dm is None:
        return 0.0
    return max(0.0, -dm)


HORIZON_DAYS = {'5d': 5, '21d': 21, '63d': 63}


def _days_since_last_fire(signal_history: pd.Series) -> int | None:
    """
    Return trading days since the signal last fired (was True).
    Returns 0 if signal is currently ON, None if it never fired.
    """
    if signal_history is None or signal_history.empty:
        return None
    try:
        fire_mask = signal_history.astype(bool)
    except (ValueError, TypeError):
        return None
    if not fire_mask.any():
        return None
    last_fire_idx = fire_mask[fire_mask].index[-1]
    # Count trading days from last fire to end of series
    return len(signal_history.loc[last_fire_idx:]) - 1


def _signal_decay_weight(sig: dict, horizon: str, spy_pct_from_high: float) -> float:
    """
    Compute effective weight (0-1) for a signal on a given horizon dial.

    - If currently ON: 1.0 (full weight)
    - If OFF: linear decay based on remaining fraction of the horizon window,
      modulated by SPY proximity to highs.

    spy_pct_from_high: positive value, e.g. 0.03 means SPY is 3% below 52w high.
    """
    if sig.get('on'):
        return 1.0

    days_since = _days_since_last_fire(sig.get('signal_history'))
    if days_since is None or days_since == 0:
        return 0.0

    h_days = HORIZON_DAYS.get(horizon, 21)
    remaining_frac = max(0.0, (h_days - days_since) / h_days)
    if remaining_frac == 0.0:
        return 0.0

    # SPY proximity: scales from 1.0 (at highs) to 0.0 (10%+ from highs)
    spy_factor = max(0.0, 1.0 - (spy_pct_from_high / 0.10))

    return remaining_frac * spy_factor


def _compute_decay_metadata(sig: dict, spy_pct_from_high: float) -> dict | None:
    """
    Return decay metadata for a signal that is OFF but still contributing weight.

    Returns None if signal is ON, never fired, or fully expired on all horizons.
    Otherwise returns {days_since, horizons: {5d/21d/63d: {weight, remaining_days}}, max_remaining_days}.
    """
    if sig.get('on'):
        return None

    days_since = _days_since_last_fire(sig.get('signal_history'))
    if days_since is None or days_since == 0:
        return None

    horizons = {}
    for h_label, h_days in HORIZON_DAYS.items():
        w = _signal_decay_weight(sig, h_label, spy_pct_from_high)
        remaining = max(0, h_days - days_since)
        horizons[h_label] = {'weight': w, 'remaining_days': remaining}

    # If no horizon still has weight, signal is fully expired
    if all(h['weight'] == 0.0 for h in horizons.values()):
        return None

    max_remaining = max(h['remaining_days'] for h in horizons.values() if h['weight'] > 0)
    return {
        'days_since': days_since,
        'horizons': horizons,
        'max_remaining_days': max_remaining,
    }


def compute_horizon_fragility(
    signals_ordered: dict,
    regime_mult: float,
    horizon_stats: dict,
    price_ctx: dict,
) -> dict:
    """
    Compute 0-100 fragility scores for 3 horizons (5d, 21d, 63d).

    Each signal's contribution is weighted by its backtested edge
    (how much worse than baseline forward returns are when signal active).
    Elevated/dire tiers use their own (typically larger) weights.

    Signals that recently turned OFF decay linearly over their horizon window,
    modulated by SPY proximity to highs.
    """
    horizons = ['5d', '21d', '63d']
    stats = horizon_stats

    da = signals_ordered.get('Distribution Dominance', {})
    vrc = signals_ordered.get('VIX Range Compression', {})
    dl = signals_ordered.get('Defensive Leadership', {})
    fomc = signals_ordered.get('Pre-FOMC Rally', {})
    ar = signals_ordered.get('Low Absorption Ratio', {})
    srd = signals_ordered.get('Seasonal Rank Divergence', {})

    # SPY distance from highs (positive = below high)
    dd = price_ctx.get('drawdown')
    spy_pct_from_high = abs(dd) if dd is not None and dd < 0 else 0.0

    scores = {}
    for h in horizons:
        # Max weight: all signals at normal tier
        max_weight = (
            _signal_edge(stats, 'Distribution Dominance', h)
            + _signal_edge(stats, 'VIX Range Compression', h)
            + _signal_edge(stats, 'Defensive Leadership', h)
            + _signal_edge(stats, 'Pre-FOMC Rally', h)
            + _signal_edge(stats, 'Low Absorption Ratio', h)
            + _signal_edge(stats, 'Seasonal Rank Divergence', h)
        )

        active_weight = 0.0

        # D/A — elevated tier has different (usually worse) forward stats
        da_w = _signal_decay_weight(da, h, spy_pct_from_high)
        if da_w > 0:
            if da.get('elevated'):
                active_weight += _signal_edge(stats, 'Distribution Dominance (Elevated)', h) * da_w
            else:
                active_weight += _signal_edge(stats, 'Distribution Dominance', h) * da_w

        vrc_w = _signal_decay_weight(vrc, h, spy_pct_from_high)
        if vrc_w > 0:
            active_weight += _signal_edge(stats, 'VIX Range Compression', h) * vrc_w

        dl_w = _signal_decay_weight(dl, h, spy_pct_from_high)
        if dl_w > 0:
            active_weight += _signal_edge(stats, 'Defensive Leadership', h) * dl_w

        fomc_w = _signal_decay_weight(fomc, h, spy_pct_from_high)
        if fomc_w > 0:
            active_weight += _signal_edge(stats, 'Pre-FOMC Rally', h) * fomc_w

        ar_w = _signal_decay_weight(ar, h, spy_pct_from_high)
        if ar_w > 0:
            active_weight += _signal_edge(stats, 'Low Absorption Ratio', h) * ar_w

        srd_w = _signal_decay_weight(srd, h, spy_pct_from_high)
        if srd_w > 0:
            active_weight += _signal_edge(stats, 'Seasonal Rank Divergence', h) * srd_w

        if max_weight > 0:
            score = (active_weight / max_weight) * 80 * regime_mult
        else:
            score = 0.0

        scores[h] = min(100.0, max(0.0, score))

    return scores


def build_risk_dial(fragility_score: float, title: str = "") -> go.Figure:
    """Build a Robust -> Fragile dial gauge."""

    if fragility_score < 20:
        bar_color = "#00CC00"
    elif fragility_score < 40:
        bar_color = "#7FCC00"
    elif fragility_score < 60:
        bar_color = "#FFD700"
    elif fragility_score < 80:
        bar_color = "#FF8C00"
    else:
        bar_color = "#CC0000"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fragility_score,
        title={'text': title, 'font': {'size': 13}},
        number={'suffix': '', 'font': {'size': 32}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickvals': [0, 50, 100],
                'ticktext': ['Robust', 'Neutral', 'Fragile'],
                'tickfont': {'size': 9},
            },
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, 20], 'color': 'rgba(0,204,0,0.12)'},
                {'range': [20, 40], 'color': 'rgba(0,204,0,0.06)'},
                {'range': [40, 60], 'color': 'rgba(255,215,0,0.08)'},
                {'range': [60, 80], 'color': 'rgba(255,140,0,0.10)'},
                {'range': [80, 100], 'color': 'rgba(204,0,0,0.12)'},
            ],
            'threshold': {
                'line': {'color': bar_color, 'width': 3},
                'thickness': 0.8,
                'value': fragility_score,
            },
        },
    ))

    fig.update_layout(
        height=180,
        margin=dict(l=15, r=15, t=35, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# PLOTLY CHART HELPERS
# ---------------------------------------------------------------------------

CHART_HEIGHT = 250
CHART_MARGIN = dict(l=10, r=10, t=30, b=10)


def _add_signal_vlines(fig: go.Figure, signal_history: pd.Series):
    """Overlay semi-transparent red vertical lines on dates where a signal fired."""
    if signal_history is None or signal_history.empty:
        return
    fire_dates = signal_history[signal_history.astype(bool)].index
    for dt in fire_dates:
        fig.add_vline(x=dt, line_color="rgba(204,0,0,0.35)", line_width=1)


def _base_layout(title: str = "") -> dict:
    return dict(
        height=CHART_HEIGHT,
        margin=CHART_MARGIN,
        hovermode="x unified",
        title=dict(text=title, font=dict(size=13)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
    )


def _dual_y_layout(title: str, y1_title: str = "", y2_title: str = "") -> dict:
    layout = _base_layout(title)
    layout['height'] = 300
    if y1_title:
        layout['yaxis'] = dict(title=y1_title, showgrid=True, gridcolor="rgba(128,128,128,0.2)")
    layout['yaxis2'] = dict(
        overlaying="y", side="right", showgrid=False,
        title=y2_title if y2_title else "",
    )
    return layout


def chart_absorption_ratio(ar_series: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ar_series.index, y=ar_series,
        name="Absorption Ratio",
        line=dict(width=1.5, color="#0066CC"),
    ))
    fig.add_hline(y=0.4, line_dash="dot", line_color="#CC0000", line_width=1,
                  annotation_text="0.40", annotation_position="right")
    fig.update_layout(**_base_layout("Absorption Ratio (PCA on Sector ETFs)"))
    return fig


# ---------------------------------------------------------------------------
# SIGNAL CHARTS
# ---------------------------------------------------------------------------

def _spy_y2_range(spy_close: pd.Series, days: int = 730, pad_pct: float = 0.03) -> list | None:
    """Compute ±pad% range for SPY within the default view window."""
    cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    window = spy_close[spy_close.index >= cutoff]
    if len(window) == 0:
        return None
    lo = float(window.min())
    hi = float(window.max())
    return [lo * (1 - pad_pct), hi * (1 + pad_pct)]


def chart_da_ratio(da_ratio: pd.Series, spy_close: pd.Series) -> go.Figure:
    """D/A ratio with threshold lines and SPY overlay."""
    fig = go.Figure()
    clean = da_ratio.dropna()
    fig.add_trace(go.Scatter(
        x=clean.index, y=clean,
        name="D/A Ratio", line=dict(width=1.5, color="#e74c3c"),
    ))
    fig.add_hline(y=3.75, line_dash="dash", line_color="#FFD700", line_width=1)
    fig.add_hline(y=6.0, line_dash="dash", line_color="#CC0000", line_width=1)
    fig.add_trace(go.Scatter(
        x=spy_close.index, y=spy_close,
        name="SPY", line=dict(width=1, color="rgba(100,100,100,0.4)"),
        yaxis="y2",
    ))
    fig.update_layout(**_dual_y_layout("Distribution / Accumulation Ratio", "D/A Ratio", "SPY"))
    two_yr_ago = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
    fig.update_xaxes(range=[two_yr_ago, datetime.datetime.now().strftime("%Y-%m-%d")])
    spy_range = _spy_y2_range(spy_close)
    if spy_range:
        fig.update_layout(yaxis2=dict(range=spy_range))
    return fig


def chart_vix_compression(vix_close: pd.Series, compression_pctile: pd.Series) -> go.Figure:
    """VIX + compression percentile with threshold."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vix_close.index, y=vix_close,
        name="VIX", line=dict(width=1.5, color="#3498db"),
    ))
    pctile_clean = compression_pctile.dropna()
    fig.add_trace(go.Scatter(
        x=pctile_clean.index, y=pctile_clean,
        name="Range Percentile", line=dict(width=1, color="#e67e22"),
        yaxis="y2",
    ))
    fig.add_hline(y=15, line_dash="dash", line_color="#FFD700", line_width=1,
                  annotation_text="Threshold: 15th", yref="y2",
                  annotation_position="right")
    fig.update_layout(**_dual_y_layout("VIX Level + Range Compression Percentile", "VIX", "Range Pctile"))
    two_yr_ago = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
    fig.update_xaxes(range=[two_yr_ago, datetime.datetime.now().strftime("%Y-%m-%d")])
    return fig


def chart_leadership(on_breadth: pd.Series, off_breadth: pd.Series,
                     spy_close: pd.Series) -> go.Figure:
    """Risk-on vs risk-off breadth with SPY overlay."""
    fig = go.Figure()
    on_clean = on_breadth.dropna()
    off_clean = off_breadth.dropna()
    fig.add_trace(go.Scatter(
        x=on_clean.index, y=on_clean,
        name="Risk-On % > 50d", line=dict(width=1.5, color="#2ecc71"),
    ))
    fig.add_trace(go.Scatter(
        x=off_clean.index, y=off_clean,
        name="Risk-Off % > 50d", line=dict(width=1.5, color="#e74c3c"),
    ))
    fig.add_trace(go.Scatter(
        x=spy_close.index, y=spy_close,
        name="SPY", line=dict(width=1, color="rgba(100,100,100,0.4)"),
        yaxis="y2",
    ))
    fig.update_layout(**_dual_y_layout("Risk-On vs Risk-Off Breadth", "% Above 50d SMA", "SPY"))
    two_yr_ago = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
    fig.update_xaxes(range=[two_yr_ago, datetime.datetime.now().strftime("%Y-%m-%d")])
    spy_range = _spy_y2_range(spy_close)
    if spy_range:
        fig.update_layout(yaxis2=dict(range=spy_range))
    return fig


def chart_ar_signal(ar_pctile: pd.Series, spy_close: pd.Series) -> go.Figure:
    """AR percentile with threshold line and SPY overlay."""
    fig = go.Figure()
    clean = ar_pctile.dropna()
    fig.add_trace(go.Scatter(
        x=clean.index, y=clean,
        name="AR Percentile", line=dict(width=1.5, color="#9b59b6"),
    ))
    fig.add_hline(y=10, line_dash="dash", line_color="#CC0000", line_width=1)
    fig.add_trace(go.Scatter(
        x=spy_close.index, y=spy_close,
        name="SPY", line=dict(width=1, color="rgba(100,100,100,0.4)"),
        yaxis="y2",
    ))
    fig.update_layout(**_dual_y_layout("Absorption Ratio Percentile (PCA w=21)", "AR Pctile", "SPY"))
    two_yr_ago = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
    fig.update_xaxes(range=[two_yr_ago, datetime.datetime.now().strftime("%Y-%m-%d")])
    spy_range = _spy_y2_range(spy_close)
    if spy_range:
        fig.update_layout(yaxis2=dict(range=spy_range))
    return fig


def chart_fomc_signals(spy_close: pd.Series, signal_dates: list) -> go.Figure:
    """SPY price with vertical red lines at pre-FOMC signal fire dates."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spy_close.index, y=spy_close,
        name="SPY", line=dict(width=1.5, color="rgba(150,150,150,0.8)"),
    ))

    for dt in signal_dates:
        fig.add_vline(x=dt, line_color="rgba(204,0,0,0.6)", line_width=1)

    layout = _base_layout(f"Pre-FOMC Rally Signal ({len(signal_dates)} events)")
    layout['height'] = 300
    fig.update_layout(**layout)
    two_yr_ago = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
    fig.update_xaxes(range=[two_yr_ago, datetime.datetime.now().strftime("%Y-%m-%d")])
    spy_range = _spy_y2_range(spy_close)
    if spy_range:
        fig.update_layout(yaxis=dict(range=spy_range))
    return fig


def chart_seasonal_divergence(spread: pd.Series, spy_close: pd.Series) -> go.Figure:
    """Seasonal rank spread (risk-off - risk-on) with SPY overlay."""
    fig = go.Figure()
    clean = spread.dropna()
    fig.add_trace(go.Scatter(
        x=clean.index, y=clean,
        name="Seasonal Spread", line=dict(width=1.5, color="#1abc9c"),
    ))
    fig.add_hline(y=10, line_dash="dash", line_color="#CC0000", line_width=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(128,128,128,0.4)", line_width=1)
    fig.add_trace(go.Scatter(
        x=spy_close.index, y=spy_close,
        name="SPY", line=dict(width=1, color="rgba(100,100,100,0.4)"),
        yaxis="y2",
    ))
    fig.update_layout(**_dual_y_layout("Seasonal Rank Divergence (Risk-Off \u2212 Risk-On)", "Spread (pp)", "SPY"))
    two_yr_ago = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
    fig.update_xaxes(range=[two_yr_ago, datetime.datetime.now().strftime("%Y-%m-%d")])
    spy_range = _spy_y2_range(spy_close)
    if spy_range:
        fig.update_layout(yaxis2=dict(range=spy_range))
    return fig


def _signal_periods(signal_history: pd.Series) -> list:
    """Convert boolean series to list of (start_date, end_date) tuples for contiguous active periods."""
    if signal_history is None or signal_history.empty:
        return []
    try:
        mask = signal_history.astype(bool)
    except (ValueError, TypeError):
        return []
    changes = mask.astype(int).diff().fillna(mask.astype(int))
    starts = changes[changes == 1].index.tolist()
    ends = changes[changes == -1].index.tolist()
    # If signal is still on at end, close the period
    if len(starts) > len(ends):
        ends.append(mask.index[-1])
    return list(zip(starts, ends))


SIGNAL_COLORS = {
    'Distribution Dominance': '#e74c3c',
    'VIX Range Compression': '#e67e22',
    'Defensive Leadership': '#2ecc71',
    'Pre-FOMC Rally': '#3498db',
    'Low Absorption Ratio': '#9b59b6',
    'Seasonal Rank Divergence': '#1abc9c',
}


def chart_signal_overlay(spy_close: pd.Series, signals_ordered: dict) -> go.Figure:
    """
    Composite chart: SPY price on top, signal activity Gantt-style timeline on bottom.
    Each signal is a colored strip at its own y-level. Overlaps are visually obvious.
    """
    from plotly.subplots import make_subplots

    sig_names = list(signals_ordered.keys())
    n_sigs = len(sig_names)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.70, 0.30], vertical_spacing=0.04,
    )

    # Top: SPY
    fig.add_trace(go.Scatter(
        x=spy_close.index, y=spy_close,
        name="SPY", line=dict(width=1.5, color="rgba(180,180,180,0.9)"),
        showlegend=False,
    ), row=1, col=1)

    # Bottom: signal activity strips
    for i, name in enumerate(sig_names):
        sig = signals_ordered[name]
        periods = _signal_periods(sig.get('signal_history'))
        color = SIGNAL_COLORS.get(name, '#888888')

        if not periods:
            # Add invisible trace for legend
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=8, color=color, symbol='square'),
                name=name, showlegend=True,
            ), row=2, col=1)
            continue

        # First period gets the legend entry
        for j, (start, end) in enumerate(periods):
            fig.add_trace(go.Scatter(
                x=[start, end, end, start, start],
                y=[i - 0.35, i - 0.35, i + 0.35, i + 0.35, i - 0.35],
                fill='toself', fillcolor=color,
                line=dict(width=0, color=color),
                marker=dict(color=color),
                opacity=0.7,
                name=name if j == 0 else name,
                showlegend=(j == 0),
                legendgroup=name,
                hoverinfo='name',
            ), row=2, col=1)

    # Vertical red lines on SPY chart for dates with 3+ concurrent signals
    histories = []
    for sig in signals_ordered.values():
        h = sig.get('signal_history')
        if h is not None and not h.empty:
            try:
                histories.append(h.astype(int))
            except (ValueError, TypeError):
                pass
    if histories:
        combined = pd.concat(histories, axis=1).fillna(0)
        overlap_count = combined.sum(axis=1)
        overlap_dates = overlap_count[overlap_count >= 2].index
        for dt in overlap_dates:
            fig.add_vline(x=dt, line_color="rgba(204,0,0,0.5)", line_width=1.5, row=1, col=1)

    # Configure axes
    two_yr_ago = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
    fig.update_xaxes(range=[two_yr_ago, datetime.datetime.now().strftime("%Y-%m-%d")])

    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(128,128,128,0.2)",
        row=1, col=1,
    )
    fig.update_yaxes(
        tickvals=list(range(n_sigs)),
        ticktext=[n.replace(' ', '\n') if len(n) > 12 else n for n in sig_names],
        tickfont=dict(size=9),
        range=[-0.5, n_sigs - 0.5],
        showgrid=True, gridcolor="rgba(128,128,128,0.1)",
        row=2, col=1,
    )

    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=35, b=10),
        hovermode="x unified",
        title=dict(text="Signal Activity Overlay", font=dict(size=13)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
        xaxis=dict(showgrid=False),
    )

    spy_range = _spy_y2_range(spy_close)
    if spy_range:
        fig.update_yaxes(range=spy_range, row=1, col=1)

    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def _cache_age_str() -> str:
    """Return human-readable age of the parquet cache."""
    if not os.path.exists(CACHE_SPY_OHLC):
        return "No cache"
    mtime = os.path.getmtime(CACHE_SPY_OHLC)
    age_sec = datetime.datetime.now().timestamp() - mtime
    if age_sec < 60:
        return "Just now"
    elif age_sec < 3600:
        return f"{int(age_sec / 60)}m ago"
    elif age_sec < 86400:
        return f"{age_sec / 3600:.1f}h ago"
    else:
        return f"{age_sec / 86400:.1f}d ago"


def main():
    st.title("\U0001f4ca Risk Dashboard V2")

    # --- Sidebar ---
    with st.sidebar:
        st.header("\u2699\ufe0f Dashboard Settings")
        lookback_years = st.slider("History (years)", 5, 15, 10)
        st.divider()
        refresh = st.button("\U0001f504 Refresh Data", type="primary", use_container_width=True)
        st.caption(f"Last refreshed: {_cache_age_str()}")

    start_date = (datetime.datetime.now() - datetime.timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")

    # Enforce minimum start for computation stability
    if start_date > "2010-01-01":
        dl_start = start_date
    else:
        dl_start = "2010-01-01"

    # --- Refresh data (manual) ---
    if refresh:
        progress_bar = st.progress(0, text="Starting refresh...")
        def _update_progress(text, pct):
            progress_bar.progress(min(pct, 1.0), text=text)
        try:
            refresh_all_data(dl_start, progress_callback=_update_progress)
            progress_bar.empty()
            st.rerun()
        except RuntimeError as e:
            progress_bar.empty()
            st.error(str(e))
            st.stop()

    # --- Load cached data ---
    spy_df, closes, sp500_closes = load_cached_data()
    if spy_df is None:
        st.info("No cached data found. Click **Refresh Data** in the sidebar to initialize.")
        st.stop()

    spy_close = spy_df["Close"]

    # Sector returns (needed for Absorption Ratio signal)
    sector_cols = [c for c in SECTOR_ETFS if c in closes.columns]
    sector_closes = closes[sector_cols].dropna(axis=1, how="all")
    sector_returns = sector_closes.pct_change().dropna(how="all")

    # -------------------------------------------------------------------
    # COMPUTE 5 VALIDATED SIGNALS
    # -------------------------------------------------------------------
    da = compute_da_signal(spy_df)

    vix_close = closes["^VIX"].dropna() if "^VIX" in closes.columns else pd.Series(dtype=float)
    vrc = compute_vix_range_compression(vix_close)

    dl = compute_defensive_leadership(sp500_closes, spy_close)

    fomc = compute_fomc_signal(spy_close)

    ar = compute_low_ar_signal(sector_returns, spy_close)

    srd = compute_seasonal_divergence_signal(spy_close)

    # Build ordered signal dict for rendering + persistence
    signals_ordered = {
        'Distribution Dominance': da,
        'VIX Range Compression': vrc,
        'Defensive Leadership': dl,
        'Pre-FOMC Rally': fomc,
        'Low Absorption Ratio': ar,
        'Seasonal Rank Divergence': srd,
    }
    signals_bool = {name: sig['on'] for name, sig in signals_ordered.items()}

    # -------------------------------------------------------------------
    # EXECUTIVE SUMMARY
    # -------------------------------------------------------------------

    # Price context + regime
    price_ctx = compute_price_context(spy_close)
    regime_mult = compute_regime_multiplier(price_ctx)
    render_price_context(price_ctx)

    # What Changed
    prev_state = load_previous_signal_state()
    current_state = {'signals': signals_bool}
    changes = compute_changes(current_state, prev_state)

    if changes:
        changes_text = " \u00b7 ".join(changes)
        st.markdown(f"<div style='font-size: 13px; padding: 4px 0 8px 0;'>Since last session: {changes_text}</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size: 12px; color: #555; padding: 4px 0 8px 0;'>No signal changes since last session.</div>",
                    unsafe_allow_html=True)

    save_current_signal_state(current_state)

    # Signal board
    render_signal_board(signals_ordered, price_ctx)

    # Risk horizon dials
    horizon_stats = load_horizon_stats()
    active_count = sum(1 for v in signals_bool.values() if v)
    total_count = len(signals_bool)

    if horizon_stats is not None:
        h_scores = compute_horizon_fragility(signals_ordered, regime_mult, horizon_stats, price_ctx)

        dial_c1, dial_c2, dial_c3 = st.columns(3)
        with dial_c1:
            st.plotly_chart(build_risk_dial(h_scores['5d'], 'Short-Term (5d)'), use_container_width=True)
        with dial_c2:
            st.plotly_chart(build_risk_dial(h_scores['21d'], 'Intermediate (21d)'), use_container_width=True)
        with dial_c3:
            st.plotly_chart(build_risk_dial(h_scores['63d'], 'Long-Term (63d)'), use_container_width=True)

        if active_count > 0:
            st.markdown(
                f"<p style='text-align: center; font-size: 12px; color: #888; margin-top: -8px;'>"
                f"{active_count} of {total_count} signals active — dials weighted by backtested forward returns</p>",
                unsafe_allow_html=True,
            )
    else:
        st.warning("Horizon stats file missing — using equal-weight fallback.")
        fallback = (active_count / total_count * 80 * regime_mult) if total_count > 0 else 0
        st.plotly_chart(build_risk_dial(min(100, fallback), 'Fragility'), use_container_width=True)

    # -------------------------------------------------------------------
    # SIGNAL CHARTS (3x2 grid)
    # -------------------------------------------------------------------
    st.divider()

    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        if len(da['da_ratio'].dropna()) > 0:
            fig = chart_da_ratio(da['da_ratio'], spy_close)
            _add_signal_vlines(fig, da.get('signal_history'))
            st.plotly_chart(fig, use_container_width=True)

    with row1_c2:
        if len(vrc['compression_pctile'].dropna()) > 0:
            fig = chart_vix_compression(vix_close, vrc['compression_pctile'])
            _add_signal_vlines(fig, vrc.get('signal_history'))
            st.plotly_chart(fig, use_container_width=True)
        elif len(vix_close) > 0:
            st.info("VIX compression data requires 504+ days of history.")

    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        if len(dl['spread'].dropna()) > 0:
            fig = chart_leadership(dl['on_breadth'], dl['off_breadth'], spy_close)
            _add_signal_vlines(fig, dl.get('signal_history'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Defensive Leadership requires S&P 500 data + risk classification.")

    with row2_c2:
        if len(fomc['signal_dates']) > 0:
            st.plotly_chart(chart_fomc_signals(spy_close, fomc['signal_dates']), use_container_width=True)

    row3_c1, row3_c2 = st.columns(2)

    with row3_c1:
        if len(ar.get('ar_pctile', pd.Series(dtype=float)).dropna()) > 0:
            fig = chart_ar_signal(ar['ar_pctile'], spy_close)
            _add_signal_vlines(fig, ar.get('signal_history'))
            st.plotly_chart(fig, use_container_width=True)
        elif len(ar.get('ar_series', pd.Series(dtype=float)).dropna()) > 0:
            st.plotly_chart(chart_absorption_ratio(ar['ar_series']), use_container_width=True)

    with row3_c2:
        if len(srd.get('spread', pd.Series(dtype=float)).dropna()) > 0:
            fig = chart_seasonal_divergence(srd['spread'], spy_close)
            _add_signal_vlines(fig, srd.get('signal_history'))
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------
    # COMPOSITE SIGNAL OVERLAY
    # -------------------------------------------------------------------
    st.divider()
    st.plotly_chart(chart_signal_overlay(spy_close, signals_ordered), use_container_width=True)


main()
