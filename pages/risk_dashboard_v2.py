"""
Risk Dashboard V2 — Executive Summary + Absorption Ratio
=========================================================
Standalone market risk monitor.
Signal-based three-question framework with fragility dial.
4 validated signals from event study backtesting.

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
    "2026-01-28",
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
# ABSORPTION RATIO (kept — display-only structural context)
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
# 4 VALIDATED SIGNAL FUNCTIONS
# ---------------------------------------------------------------------------

def compute_da_signal(spy_df: pd.DataFrame) -> tuple:
    """
    Distribution/Accumulation signal.
    Institutional selling beneath the surface during an uptrend.
    """
    vol_mult = 1.15
    window = 21
    threshold = 5.0

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

    da_ratio = dist_count / accum_count.replace(0, np.nan)

    signal = da_ratio > threshold

    # Require uptrend (SPY > 50d SMA)
    sma_50 = close.rolling(50).mean()
    signal = signal & (close > sma_50)

    latest_ratio = float(da_ratio.iloc[-1]) if len(da_ratio) > 0 and not np.isnan(da_ratio.iloc[-1]) else 0.0
    signal_on = bool(signal.iloc[-1]) if len(signal) > 0 and not pd.isna(signal.iloc[-1]) else False

    detail = ""
    if signal_on:
        recent_dist = int(dist_days.iloc[-window:].sum())
        recent_accum = int(accum_days.iloc[-window:].sum())
        detail = (
            f"D/A ratio at {latest_ratio:.1f} over last {window} sessions "
            f"({recent_dist} distribution vs {recent_accum} accumulation days). "
            f"Volume-confirmed selling exceeds buying during an uptrend."
        )

    return signal_on, latest_ratio, detail


def compute_vix_range_compression(vix_close: pd.Series) -> tuple:
    """
    VIX Range Compression signal.
    VIX in a tight squeeze — eventual breakout tends to be violent.
    """
    if len(vix_close) < 504:
        return False, 0.0, ""

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

    detail = ""
    if signal_on:
        cur_vix = float(vix_close.iloc[-1])
        detail = (
            f"VIX {range_window}d range at {latest_pctile:.0f}th percentile "
            f"(threshold: {pctile_threshold}th). VIX at {cur_vix:.1f} is compressed \u2014 "
            f"eventual breakout tends to be violent."
        )

    return signal_on, latest_pctile, detail


def compute_defensive_leadership(sp500_closes, spy_close: pd.Series) -> tuple:
    """
    Defensive Leadership signal.
    Risk-off stocks leading while SPY near highs = institutional rotation to safety.
    """
    if sp500_closes is None or (hasattr(sp500_closes, 'empty') and sp500_closes.empty):
        return False, 0.0, ""

    if not os.path.exists(RISK_CLASSIFICATION_PATH):
        return False, 0.0, ""

    classification = pd.read_csv(RISK_CLASSIFICATION_PATH)
    on_tickers = classification.loc[classification["label"] == "risk_on", "ticker"].tolist()
    off_tickers = classification.loc[classification["label"] == "risk_off", "ticker"].tolist()

    on_cols = [t for t in on_tickers if t in sp500_closes.columns]
    off_cols = [t for t in off_tickers if t in sp500_closes.columns]

    if len(on_cols) < 10 or len(off_cols) < 10:
        return False, 0.0, ""

    sma_200 = sp500_closes.rolling(200, min_periods=160).mean()
    on_above = (sp500_closes[on_cols] > sma_200[on_cols]).sum(axis=1) / len(on_cols) * 100
    off_above = (sp500_closes[off_cols] > sma_200[off_cols]).sum(axis=1) / len(off_cols) * 100

    spread = on_above - off_above  # negative = risk-off leading

    # SPY within 5% of 52w high
    high_52w = spy_close.rolling(252, min_periods=60).max()
    near_high = spy_close >= high_52w * 0.95

    signal = (spread < -10) & near_high

    latest_spread = float(spread.iloc[-1]) if len(spread) > 0 and not np.isnan(spread.iloc[-1]) else 0.0
    signal_on = bool(signal.iloc[-1]) if len(signal) > 0 and not pd.isna(signal.iloc[-1]) else False

    detail = ""
    if signal_on:
        on_pct = float(on_above.iloc[-1])
        off_pct = float(off_above.iloc[-1])
        detail = (
            f"Risk-off stocks leading: {off_pct:.0f}% above 200d SMA vs "
            f"{on_pct:.0f}% for risk-on (spread: {latest_spread:+.0f}pp). "
            f"SPY near 52w high \u2014 defensive rotation while index holds."
        )

    return signal_on, latest_spread, detail


def compute_fomc_signal(spy_close: pd.Series) -> tuple:
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

    # Compute 5d trailing return percentile
    pre_return = spy_close.pct_change(pre_window)
    pre_pctile = _rolling_percentile(pre_return, lookback)

    latest_pctile = float(pre_pctile.iloc[-1]) if len(pre_pctile) > 0 and not np.isnan(pre_pctile.iloc[-1]) else 0.0

    signal_on = in_pre_fomc and latest_pctile > pctile_threshold

    detail = ""
    if signal_on:
        fomc_str = next_fomc.strftime('%b %d') if next_fomc is not None else "upcoming"
        detail = (
            f"FOMC on {fomc_str} \u2014 SPY trailing 5d return at "
            f"{latest_pctile:.0f}th percentile. Strong pre-FOMC rally increases "
            f"'buy the rumor, sell the news' reversal risk."
        )

    return signal_on, latest_pctile, detail


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


def compute_condition_signals(metrics: dict, price_ctx: dict) -> dict:
    """
    Compute the three-question signal framework using 4 validated signals.

    Returns dict of:
    {
        'signals': {name: bool},
        'details': {name: str},
        'questions': {question: [signal_names]},
    }
    """
    signals = {}
    details = {}

    # Question 1: Is liquidity real? (1 signal)
    signals['VIX Range Compression'] = metrics.get('vrc_on', False)
    if metrics.get('vrc_detail'):
        details['VIX Range Compression'] = metrics['vrc_detail']

    # Question 2: Is everyone on the same side? (2 signals)
    signals['Distribution Dominance'] = metrics.get('da_on', False)
    if metrics.get('da_detail'):
        details['Distribution Dominance'] = metrics['da_detail']

    signals['Defensive Leadership'] = metrics.get('dl_on', False)
    if metrics.get('dl_detail'):
        details['Defensive Leadership'] = metrics['dl_detail']

    # Question 3: Are correlations stable? (1 signal)
    signals['Pre-FOMC Rally'] = metrics.get('fomc_on', False)
    if metrics.get('fomc_detail'):
        details['Pre-FOMC Rally'] = metrics['fomc_detail']

    questions = {
        'Is liquidity real?': ['VIX Range Compression'],
        'Is everyone on the same side?': ['Distribution Dominance', 'Defensive Leadership'],
        'Are correlations stable?': ['Pre-FOMC Rally'],
    }

    return {
        'signals': signals,
        'details': details,
        'questions': questions,
    }


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


def render_three_questions(signal_result: dict):
    """Render the three-question signal board."""

    question_icons = {
        'Is liquidity real?': '\U0001f4a7',
        'Is everyone on the same side?': '\U0001f465',
        'Are correlations stable?': '\U0001f517',
    }

    for question, signal_names in signal_result['questions'].items():
        icon = question_icons.get(question, '\u2753')
        active_in_group = sum(1 for s in signal_names if signal_result['signals'].get(s, False))

        # Question header with count
        if active_in_group == 0:
            q_color = "#00CC00"
            q_badge = "CLEAR"
        elif active_in_group == 1:
            q_color = "#FFD700"
            q_badge = "WATCH"
        else:
            q_color = "#CC0000"
            q_badge = "WARNING"

        st.markdown(
            f"<div style='margin-bottom: 4px;'>"
            f"<span style='font-size: 14px; font-weight: 600;'>{icon} {question}</span>"
            f"&nbsp;&nbsp;<span style='background: {q_color}30; color: {q_color}; "
            f"padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600;'>"
            f"{q_badge}</span></div>",
            unsafe_allow_html=True
        )

        # Individual signals
        for name in signal_names:
            active = signal_result['signals'].get(name, False)
            if active:
                detail = signal_result['details'].get(name, '')
                st.markdown(
                    f"<div style='margin-left: 20px; padding: 6px 10px; margin-bottom: 4px; "
                    f"background: rgba(204,0,0,0.08); border-left: 3px solid #CC0000; "
                    f"border-radius: 0 4px 4px 0;'>"
                    f"<span style='font-size: 13px;'>\U0001f534 <strong>{name}</strong></span><br>"
                    f"<span style='font-size: 12px; color: #bbb;'>{detail}</span></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='margin-left: 20px; padding: 4px 10px; margin-bottom: 2px;'>"
                    f"<span style='font-size: 13px; color: #666;'>\U0001f7e2 {name}</span></div>",
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

def compute_fragility_score(signal_result: dict, regime_multiplier: float) -> float:
    """
    Compute 0-100 fragility score.

    Base: each active signal contributes equally.
    Multiplied by price regime context.
    """
    total_signals = len(signal_result['signals'])
    active_count = sum(1 for v in signal_result['signals'].values() if v)

    if total_signals == 0:
        return 0.0

    # Base score: linear from 0 (none active) to 80 (all active)
    # Each signal = 20 base pts (4 signals total)
    base = (active_count / total_signals) * 80

    # Apply regime multiplier
    adjusted = base * regime_multiplier

    return min(100, max(0, adjusted))


def build_risk_dial(fragility_score: float) -> go.Figure:
    """Build the Robust -> Fragile dial."""

    # Color based on score
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
        number={'suffix': '', 'font': {'size': 40}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickvals': [0, 50, 100],
                'ticktext': ['Robust', 'Neutral', 'Fragile'],
                'tickfont': {'size': 10},
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
        height=190,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# PLOTLY CHART HELPERS
# ---------------------------------------------------------------------------

CHART_HEIGHT = 250
CHART_MARGIN = dict(l=10, r=10, t=30, b=10)


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

    # -------------------------------------------------------------------
    # COMPUTE 4 VALIDATED SIGNALS
    # -------------------------------------------------------------------
    da_on, da_ratio_val, da_detail = compute_da_signal(spy_df)

    vix_close = closes["^VIX"].dropna() if "^VIX" in closes.columns else pd.Series(dtype=float)
    vrc_on, vrc_pctile_val, vrc_detail = compute_vix_range_compression(vix_close)

    dl_on, dl_spread_val, dl_detail = compute_defensive_leadership(sp500_closes, spy_close)

    fomc_on, fomc_pctile_val, fomc_detail = compute_fomc_signal(spy_close)

    # -------------------------------------------------------------------
    # EXECUTIVE SUMMARY
    # -------------------------------------------------------------------

    # Price context + regime
    price_ctx = compute_price_context(spy_close)
    regime_mult = compute_regime_multiplier(price_ctx)
    render_price_context(price_ctx)

    # Signal framework
    signal_metrics = {
        'da_on': da_on, 'da_detail': da_detail,
        'vrc_on': vrc_on, 'vrc_detail': vrc_detail,
        'dl_on': dl_on, 'dl_detail': dl_detail,
        'fomc_on': fomc_on, 'fomc_detail': fomc_detail,
    }
    signal_result = compute_condition_signals(signal_metrics, price_ctx)

    # What Changed
    prev_state = load_previous_signal_state()
    current_state = {'signals': signal_result['signals']}
    changes = compute_changes(current_state, prev_state)

    if changes:
        changes_text = " \u00b7 ".join(changes)
        st.markdown(f"<div style='font-size: 13px; padding: 4px 0 8px 0;'>Since last session: {changes_text}</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size: 12px; color: #555; padding: 4px 0 8px 0;'>No signal changes since last session.</div>",
                    unsafe_allow_html=True)

    # Save current state for next session's comparison
    save_current_signal_state(current_state)

    # Section B: Three Questions + Risk Dial
    questions_col, dial_col = st.columns([3, 1])

    with questions_col:
        render_three_questions(signal_result)

    # Fragility score
    fragility = compute_fragility_score(signal_result, regime_mult)
    active_count = sum(1 for v in signal_result['signals'].values() if v)
    total_count = len(signal_result['signals'])

    with dial_col:
        fig_dial = build_risk_dial(fragility)
        st.plotly_chart(fig_dial, use_container_width=True)

        if active_count == 0:
            dial_label = "No warning signals active"
        else:
            dial_label = f"{active_count} of {total_count} signals active"

        st.markdown(f"<p style='text-align: center; font-size: 13px; margin-top: -8px;'>{dial_label}</p>",
                    unsafe_allow_html=True)

    # -------------------------------------------------------------------
    # ABSORPTION RATIO CHART
    # -------------------------------------------------------------------
    sector_cols = [c for c in SECTOR_ETFS if c in closes.columns]
    sector_closes = closes[sector_cols].dropna(axis=1, how="all")
    sector_returns = sector_closes.pct_change().dropna(how="all")

    if len(sector_returns.columns) >= 5:
        with st.spinner("Computing absorption ratio..."):
            ar_series = compute_absorption_ratio(sector_returns, window=63)
        if len(ar_series.dropna()) > 0:
            st.divider()
            st.plotly_chart(chart_absorption_ratio(ar_series), use_container_width=True)


main()
