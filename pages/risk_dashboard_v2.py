"""
Risk Dashboard V2 — Phases 1 & 2 (Layers 0-4)
===============================================
Standalone market risk monitor.
Layer 0: Composite regime verdict (rules-based point system)
Layer 1: Volatility state (HAR-RV, VRP, VIX term structure, VVIX)
Layer 2: Equity market internals (breadth, absorption ratio, dispersion, Hurst, complacency counters)
Layer 3: Cross-asset plumbing (credit spreads, yield curve, MOVE, dollar)
Layer 4: Tail risk & protection cost (SKEW, protection cost proxy, hedge recommendation)

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
from plotly.subplots import make_subplots
import math
import json

try:
    from scipy.stats import norm as scipy_norm
except ImportError:
    class _NormFallback:
        @staticmethod
        def cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    scipy_norm = _NormFallback()

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

# Try to import SP500_TICKERS for breadth (optional)
try:
    from abs_return_dispersion import SP500_TICKERS
except ImportError:
    SP500_TICKERS = None

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
VOL_TICKERS = ["SPY", "^VIX", "^VIX3M", "^VVIX"]
SECTOR_ETFS = [
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK",
    "XLP", "XLRE", "XLU", "XLV", "XLY",
]

# Cross-asset tickers (Layer 3)
CROSS_ASSET_TICKERS = ['LQD', 'HYG', 'IEF', 'UUP', '^MOVE', '^TNX', '^IRX']

# Tail risk tickers (Layer 4)
TAIL_RISK_TICKERS = ['^SKEW']

REGIME_COLORS = {
    "Normal": "#00CC00",
    "Caution": "#FFD700",
    "Stress": "#FF8C00",
    "Crisis": "#CC0000",
}
REGIME_EMOJI = {
    "Normal": "\U0001f7e2",
    "Caution": "\U0001f7e1",
    "Stress": "\U0001f7e0",
    "Crisis": "\U0001f534",
}
REGIME_MULTIPLIER = {
    "Normal": 1.00,
    "Caution": 0.75,
    "Stress": 0.50,
    "Crisis": 0.25,
}

DATA_DIR = os.path.join(parent_dir, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SIGNAL_CACHE_PATH = os.path.join(DATA_DIR, "risk_dashboard_signal_state.json")

# ---------------------------------------------------------------------------
# DATA DOWNLOAD
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Downloading volatility data...")
def download_vol_data(start_date: str = "2010-01-01") -> dict:
    """Download OHLC data for volatility tickers."""
    return _download_ticker_group(VOL_TICKERS, start_date)


@st.cache_data(ttl=3600, show_spinner="Downloading sector ETF data...")
def download_sector_data(start_date: str = "2010-01-01") -> dict:
    """Download OHLC data for sector ETFs."""
    return _download_ticker_group(SECTOR_ETFS, start_date)


@st.cache_data(ttl=3600, show_spinner="Downloading cross-asset data...")
def download_cross_asset_data(start_date: str = "2010-01-01") -> dict:
    """Download OHLC data for cross-asset tickers (Layer 3)."""
    return _download_ticker_group(CROSS_ASSET_TICKERS, start_date)


@st.cache_data(ttl=3600, show_spinner="Downloading tail risk data...")
def download_tail_risk_data(start_date: str = "2010-01-01") -> dict:
    """Download OHLC data for tail risk tickers (Layer 4)."""
    return _download_ticker_group(TAIL_RISK_TICKERS, start_date)


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
# LAYER 1 COMPUTATIONS
# ---------------------------------------------------------------------------

def yang_zhang_vol(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Yang-Zhang (2000) volatility estimator using OHLC data.
    Returns annualized volatility series.
    For window=1, falls back to per-bar Rogers-Satchell estimator
    (rolling variance is undefined for a single observation).
    """
    log_ho = np.log(df["High"] / df["Open"])
    log_lo = np.log(df["Low"] / df["Open"])
    log_co = np.log(df["Close"] / df["Open"])

    # Rogers-Satchell component
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    if window <= 1:
        # Single-bar: use Rogers-Satchell directly (no rolling variance possible)
        return np.sqrt(rs.abs()) * np.sqrt(252)

    log_oc = np.log(df["Open"] / df["Close"].shift(1))

    # Overnight component
    sigma_o = log_oc.rolling(window).var()

    # Close-to-open component
    sigma_c = log_co.rolling(window).var()

    # Rogers-Satchell component
    sigma_rs = rs.rolling(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    yang_zhang = np.sqrt(sigma_o + k * sigma_c + (1 - k) * sigma_rs) * np.sqrt(252)
    return yang_zhang


def compute_vrp(vix_close: pd.Series, rv_22d: pd.Series) -> pd.Series:
    """
    Variance Risk Premium = (VIX/100)^2 - RV_22d^2
    Both sides annualized.
    """
    implied_var = (vix_close / 100.0) ** 2
    realized_var = rv_22d ** 2
    return implied_var - realized_var


def compute_vix_term_structure(vix: pd.Series, vix3m: pd.Series) -> pd.Series:
    """VIX / VIX3M ratio. > 1 means backwardation."""
    return vix / vix3m


# ---------------------------------------------------------------------------
# LAYER 2 COMPUTATIONS
# ---------------------------------------------------------------------------

def compute_breadth_sector_proxy(sector_data: dict, window: int = 200) -> pd.DataFrame:
    """
    Compute % of sector ETFs above their N-day SMA.
    Returns DataFrame with pct_above_200, pct_above_50 columns.
    """
    # Build close price matrix
    closes = pd.DataFrame({t: d["Close"] for t, d in sector_data.items() if "Close" in d.columns})
    if closes.empty:
        return pd.DataFrame()

    sma200 = closes.rolling(200).mean()
    sma50 = closes.rolling(50).mean()

    pct_above_200 = (closes > sma200).sum(axis=1) / closes.notna().sum(axis=1) * 100
    pct_above_50 = (closes > sma50).sum(axis=1) / closes.notna().sum(axis=1) * 100

    result = pd.DataFrame({
        "pct_above_200": pct_above_200,
        "pct_above_50": pct_above_50,
    })
    return result.dropna()


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


def compute_cross_sectional_dispersion(daily_returns_df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Cross-sectional std of rolling N-day returns across sectors."""
    rolling_rets = daily_returns_df.rolling(window).sum()
    dispersion = rolling_rets.std(axis=1)
    return dispersion


def compute_avg_pairwise_correlation(daily_returns_df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Rolling mean pairwise correlation across sectors."""
    avg_corr = pd.Series(dtype=float, index=daily_returns_df.index)
    for i in range(window, len(daily_returns_df)):
        corr_matrix = daily_returns_df.iloc[i - window:i].corr()
        n = len(corr_matrix)
        if n < 2:
            continue
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        pairwise = corr_matrix.values[mask]
        avg_corr.iloc[i] = np.nanmean(pairwise)
    return avg_corr


def compute_hurst_dfa(series: pd.Series, window: int = 126) -> pd.Series:
    """
    Detrended Fluctuation Analysis to estimate Hurst exponent.
    Returns rolling H estimate. Uses 126-day window and wider box sizes
    to reduce finite-sample bias.
    """
    hurst_series = pd.Series(dtype=float, index=series.index)

    for i in range(window, len(series)):
        segment = series.iloc[i - window:i].values
        N = len(segment)
        Y = np.cumsum(segment - np.mean(segment))

        box_sizes = [8, 16, 32, 48, 63]
        box_sizes = [b for b in box_sizes if b <= N // 2]

        if len(box_sizes) < 3:
            continue

        flucts = []
        for bs in box_sizes:
            n_boxes = N // bs
            rms_list = []
            for j in range(n_boxes):
                box = Y[j * bs:(j + 1) * bs]
                x = np.arange(bs)
                coeffs = np.polyfit(x, box, 1)
                trend = np.polyval(coeffs, x)
                rms_list.append(np.sqrt(np.mean((box - trend) ** 2)))
            if rms_list:
                flucts.append(np.mean(rms_list))

        if len(flucts) < 2:
            continue

        log_bs = np.log(box_sizes[: len(flucts)])
        log_fl = np.log(flucts)

        if not np.any(np.isnan(log_fl)) and not np.any(np.isinf(log_fl)):
            try:
                slope, _ = np.polyfit(log_bs, log_fl, 1)
                hurst_series.iloc[i] = slope
            except Exception:
                continue

    return hurst_series


def compute_days_since(spy_close: pd.Series, threshold_pct: float = 0.05) -> pd.Series:
    """
    For each day, compute trading days since the last drawdown >= threshold_pct.
    Drawdown = peak-to-trough decline from trailing high.
    """
    trailing_high = spy_close.expanding().max()
    drawdown = (spy_close - trailing_high) / trailing_high

    # Find days where drawdown breached threshold
    breach_dates = set(drawdown.index[drawdown <= -threshold_pct])

    # For each day, count business days since last breach
    days_since = pd.Series(dtype=float, index=spy_close.index)
    last_breach = None
    for date in spy_close.index:
        if date in breach_dates:
            last_breach = date
        if last_breach is not None:
            days_since[date] = np.busday_count(
                last_breach.date(), date.date()
            )
        else:
            days_since[date] = np.nan  # No correction yet in history

    return days_since


def compute_days_since_vix_spike(vix_close: pd.Series, threshold: float = 28.0) -> pd.Series:
    """
    For each day, count trading days since VIX last closed above threshold.
    """
    above_threshold = vix_close >= threshold

    days_since = pd.Series(0, index=vix_close.index, dtype=int)
    count = 0
    for i in range(len(vix_close)):
        if above_threshold.iloc[i]:
            count = 0
        else:
            count += 1
        days_since.iloc[i] = count

    return days_since


# ---------------------------------------------------------------------------
# LAYER 3 COMPUTATIONS
# ---------------------------------------------------------------------------

def compute_credit_spread_proxy(lqd_close, hyg_close, ief_close):
    """
    Proxy for credit spreads using ETF price ratios.
    When credit spreads widen, LQD and HYG fall relative to IEF.
    We INVERT the ratio so that higher = wider spreads = more stress.
    """
    ig_spread = -(lqd_close / ief_close)
    hy_spread = -(hyg_close / ief_close)

    # Normalize to z-scores for comparability
    ig_z = (ig_spread - ig_spread.rolling(63).mean()) / ig_spread.rolling(63).std()
    hy_z = (hy_spread - hy_spread.rolling(63).mean()) / hy_spread.rolling(63).std()

    return ig_z, hy_z


def compute_yield_curve(tnx_series, irx_series):
    """
    10Y - 3M yield curve spread.
    ^TNX and ^IRX are both in percentage points (e.g. 4.5 = 4.5%).
    """
    spread = tnx_series - irx_series
    spread_21d_change = spread.diff(21)
    spread_z = (spread_21d_change - spread_21d_change.rolling(252).mean()) / spread_21d_change.rolling(252).std()

    return spread, spread_z


def compute_dollar_momentum(uup_close):
    """
    21-day rate of change of UUP (dollar bull ETF) as DXY proxy.
    """
    pct_change_21d = uup_close.pct_change(21) * 100  # As percentage
    return pct_change_21d


# ---------------------------------------------------------------------------
# LAYER 4 COMPUTATIONS
# ---------------------------------------------------------------------------

def compute_protection_cost(vix3m_series, skew_series):
    """
    Proxy for 3-month OTM put cost.
    Higher = more expensive protection.
    """
    if skew_series is None or (hasattr(skew_series, 'empty') and skew_series.empty):
        cost = vix3m_series.copy()
    else:
        # Align on common index
        common = vix3m_series.dropna().index.intersection(skew_series.dropna().index)
        cost = vix3m_series.reindex(common) * (skew_series.reindex(common) / 130)

    # Percentile rank against trailing 5-year history (1260 trading days)
    cost_percentile = cost.rolling(1260, min_periods=252).rank(pct=True) * 100

    return cost, cost_percentile


def generate_hedge_recommendation(regime: str, protection_percentile: float) -> tuple:
    """
    Returns (recommendation_text, detail_text, color).
    """
    if protection_percentile is None or np.isnan(protection_percentile):
        return ("Protection cost data unavailable",
                "Cannot generate recommendation without protection cost percentile.",
                "#888888")

    if protection_percentile < 20:
        rec = "Protection is historically cheap"
        detail = ("3-month ~5% OTM index puts are priced below the 20th percentile of history. "
                  "Consider allocating 1-2% of NAV to put protection. This is a positive expected "
                  "value allocation at current pricing regardless of market outlook.")
        color = "#00CC00"
    elif regime in ("Caution", "Stress") and protection_percentile < 60:
        rec = "Protection is fairly priced and conditions warrant it"
        detail = ("Consider 0.5-1% of NAV on 3-month puts. "
                  "Alternatively, reduce gross exposure to 0.75x.")
        color = "#FFD700"
    elif regime in ("Stress", "Crisis") and protection_percentile >= 60 and protection_percentile < 85:
        rec = "Protection is moderately expensive — prefer collars or exposure reduction"
        detail = ("Consider a collar: buy 5% OTM put, sell 3-5% OTM call on SPY for near-zero "
                  "net premium. Or simply reduce gross exposure to 0.50x.")
        color = "#FF8C00"
    elif protection_percentile >= 85:
        rec = "Protection is expensive — reduce exposure directly"
        detail = ("Buying puts at this pricing is likely negative EV. Reduce gross exposure to "
                  "0.50x or lower. Hold existing hedges but don't add.")
        color = "#CC0000"
    else:
        rec = "No hedge action needed"
        detail = ("Market conditions are normal and protection is not attractively priced. "
                  "Standard operations.")
        color = "#888888"

    return rec, detail, color


# ---------------------------------------------------------------------------
# PERCENTILE HELPER
# ---------------------------------------------------------------------------

def expanding_percentile(series: pd.Series, min_periods: int = 252) -> pd.Series:
    """Expanding percentile rank (0-100)."""
    def _pctile(x):
        if len(x) < min_periods:
            return np.nan
        return (x.values[:-1] < x.values[-1]).sum() / (len(x) - 1) * 100
    return series.expanding(min_periods=min_periods).apply(_pctile, raw=False)


# ---------------------------------------------------------------------------
# ALERT SCORING (Layer 0 point system)
# ---------------------------------------------------------------------------

def score_alerts(metrics: dict) -> tuple:
    """
    Compute composite alert score from individual metrics.
    Returns (total_points, breakdown_dict).

    Each metric in alert range = +1, alarm range = +2.
    0 = Normal, 1-2 = Caution, 3-4 = Stress, 5+ = Crisis.
    """
    points = {}

    # 1A: HAR-RV spike — alert: RV_1d > 2x RV_22d; alarm: RV_22d > 75th pctile AND rising
    rv1d = metrics.get("rv_1d")
    rv22d = metrics.get("rv_22d")
    rv22d_pctile = metrics.get("rv_22d_pctile")
    rv22d_prev = metrics.get("rv_22d_prev")
    if rv1d is not None and rv22d is not None and rv22d > 0:
        if rv1d > 2 * rv22d:
            points["HAR-RV (1d spike)"] = 1
    if rv22d_pctile is not None and rv22d_prev is not None:
        if rv22d_pctile > 75 and rv22d is not None and rv22d_prev is not None and rv22d > rv22d_prev:
            points["HAR-RV (22d elevated & rising)"] = 2

    # 1B: VRP — alert: < 25th pctile; alarm: negative
    vrp = metrics.get("vrp")
    vrp_pctile = metrics.get("vrp_pctile")
    if vrp is not None:
        if vrp < 0:
            points["VRP (negative)"] = 2
        elif vrp_pctile is not None and vrp_pctile < 25:
            points["VRP (low pctile)"] = 1

    # 1C: VIX Term Structure — alert: ratio > 0.95; alarm: > 1.0
    ts_ratio = metrics.get("vix_ts_ratio")
    if ts_ratio is not None:
        if ts_ratio > 1.0:
            points["VIX Term Str (backwardation)"] = 2
        elif ts_ratio > 0.95:
            points["VIX Term Str (elevated)"] = 1

    # 1D: VVIX — alert: > 100; alarm: > 120
    vvix = metrics.get("vvix")
    if vvix is not None:
        if vvix > 120:
            points["VVIX (>120)"] = 2
        elif vvix > 100:
            points["VVIX (>100)"] = 1

    # 2A: Breadth — alert: %> 200d < 60% while SPY near high; alarm: < 40%
    pct200 = metrics.get("pct_above_200")
    spy_near_high = metrics.get("spy_near_52w_high")
    if pct200 is not None:
        if pct200 < 40:
            points["Breadth (<40% above 200d)"] = 2
        elif pct200 < 60 and spy_near_high:
            points["Breadth (divergence)"] = 1

    # 2B: Absorption Ratio — removed from composite scoring (under review)

    # 2C: Dispersion — uses the 2x2 grid logic
    disp_high = metrics.get("dispersion_high")
    corr_high = metrics.get("correlation_high")
    if disp_high is not None and corr_high is not None:
        if disp_high and corr_high:
            points["Dispersion (high disp + high corr)"] = 2
        elif disp_high:
            points["Dispersion (elevated)"] = 1

    # 2D: Hurst — alert: H > 80th pctile; alarm: H > 95th pctile (empirical)
    delta_h = metrics.get("hurst_delta_5d")
    hurst_pctile = metrics.get("hurst_pctile")
    if hurst_pctile is not None:
        if hurst_pctile > 95:
            points["Hurst (>95th pctile)"] = 2
        elif hurst_pctile > 80:
            points["Hurst (>80th pctile)"] = 1
    if delta_h is not None and delta_h > 0.05 and "Hurst" not in "".join(points.keys()):
        points["Hurst (rising fast)"] = 1

    # 2E: Days-Since Complacency — compound scoring
    # Alert (+1): Either counter > 80th percentile
    # Alarm (+2): BOTH counters > 80th percentile simultaneously
    ds_5pct_pctile = metrics.get("days_since_5pct_pctile")
    ds_vix_pctile = metrics.get("days_since_vix_spike_pctile")
    ds_5_high = ds_5pct_pctile is not None and ds_5pct_pctile > 80
    ds_vix_high = ds_vix_pctile is not None and ds_vix_pctile > 80
    if ds_5_high and ds_vix_high:
        points["Complacency (compound calm)"] = 2
    elif ds_5_high:
        points["Calm Streak (5% drawdown >80th)"] = 1
    elif ds_vix_high:
        points["Calm Streak (VIX spike >80th)"] = 1

    # --- LAYER 3: Cross-Asset Plumbing ---

    # 3A: Credit Spreads
    # Alert: IG z > 1.0 OR HY z > 1.0 (+1)
    # Alarm: BOTH IG and HY z > 1.5 (+2)
    ig_z = metrics.get("credit_ig_z")
    hy_z = metrics.get("credit_hy_z")
    ig_wide = ig_z is not None and ig_z > 1.0
    hy_wide = hy_z is not None and hy_z > 1.0
    ig_alarm = ig_z is not None and ig_z > 1.5
    hy_alarm = hy_z is not None and hy_z > 1.5
    if ig_alarm and hy_alarm:
        points["Credit (IG+HY both >1.5\u03c3)"] = 2
    elif ig_wide or hy_wide:
        points["Credit (spread widening)"] = 1

    # 3B: Yield Curve
    # Alert: Inverted OR 21d change z < -1.5 (+1)
    # Alarm: Inverted AND flattening accelerating (z < -2.0) (+2)
    yc_spread = metrics.get("yield_curve_spread")
    yc_z = metrics.get("yield_curve_z")
    yc_inverted = yc_spread is not None and yc_spread < 0
    yc_flattening = yc_z is not None and yc_z < -1.5
    yc_accel = yc_z is not None and yc_z < -2.0
    if yc_inverted and yc_accel:
        points["Yield Curve (inverted + accelerating)"] = 2
    elif yc_inverted or yc_flattening:
        points["Yield Curve (warning)"] = 1

    # 3C: MOVE (only if data available)
    # Alert: > 120 (+1); Alarm: > 150 (+2)
    move_val = metrics.get("move")
    if move_val is not None:
        if move_val > 150:
            points["MOVE (>150)"] = 2
        elif move_val > 120:
            points["MOVE (>120)"] = 1

    # 3D: Dollar
    # Alert: |21d change| > 3% (+1); Alarm: |21d change| > 5% (+2)
    dollar_21d = metrics.get("dollar_21d_pct")
    if dollar_21d is not None:
        if abs(dollar_21d) > 5:
            points["Dollar (>5% move)"] = 2
        elif abs(dollar_21d) > 3:
            points["Dollar (>3% move)"] = 1

    total = sum(points.values())
    return total, points


def classify_regime(total_points: int) -> str:
    if total_points == 0:
        return "Normal"
    elif total_points <= 2:
        return "Caution"
    elif total_points <= 4:
        return "Stress"
    else:
        return "Crisis"


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
    Compute the three-question signal framework.

    Returns dict of:
    {
        'signals': {name: bool},        # ON/OFF for each signal
        'details': {name: str},          # Brief explanation when active
        'questions': {                   # Grouped by question
            'Is liquidity real?': [signal_names],
            'Is everyone on the same side?': [signal_names],
            'Are correlations stable?': [signal_names],
        }
    }
    """
    signals = {}
    details = {}

    # =====================================================================
    # QUESTION 1: IS LIQUIDITY REAL?
    # =====================================================================

    # Signal 1A: Vol Suppression
    ar_pctile = metrics.get('ar_pctile')
    rv22d_pctile = metrics.get('rv_22d_pctile')

    vol_suppression_on = False
    if ar_pctile is not None and rv22d_pctile is not None:
        vol_suppression_on = (ar_pctile < 25) and (rv22d_pctile < 35)
    signals['Vol Suppression'] = vol_suppression_on
    if vol_suppression_on:
        details['Vol Suppression'] = (
            f"Absorption ratio at {ar_pctile:.0f}th pctile while realized vol at "
            f"{rv22d_pctile:.0f}th pctile. Index vol is likely being suppressed by "
            f"systematic selling \u2014 apparent calm masks thin underlying liquidity."
        )

    # Signal 1B: VRP Compression
    vrp = metrics.get('vrp')
    vrp_pctile = metrics.get('vrp_pctile')

    vrp_compression_on = False
    if vrp is not None and vrp_pctile is not None:
        vrp_compression_on = (vrp < 0) or (vrp_pctile < 15)
    signals['VRP Compression'] = vrp_compression_on
    if vrp_compression_on:
        if vrp is not None and vrp < 0:
            details['VRP Compression'] = (
                f"VRP is negative ({vrp:.4f}) \u2014 realized vol exceeds implied. "
                f"The market was underpricing risk. Options are too cheap to hedge."
            )
        else:
            details['VRP Compression'] = (
                f"VRP at {vrp_pctile:.0f}th percentile \u2014 very little risk premium. "
                f"Market is complacent about future vol."
            )

    # =====================================================================
    # QUESTION 2: IS EVERYONE ON THE SAME SIDE?
    # =====================================================================

    # Signal 2A: Breadth Divergence
    pct200 = metrics.get('pct_above_200')
    spy_near_high = metrics.get('spy_near_52w_high')

    breadth_div_on = False
    if pct200 is not None and spy_near_high is not None:
        breadth_div_on = spy_near_high and (pct200 < 55)
    signals['Breadth Divergence'] = breadth_div_on
    if breadth_div_on:
        details['Breadth Divergence'] = (
            f"SPY is near its 52-week high but only {pct200:.0f}% of sectors are "
            f"above their 200d SMA. The index is being held up by a few names \u2014 "
            f"the army is retreating while the flag is still flying."
        )

    # Signal 2B: Extended Calm (compound complacency)
    ds_5_pctile = metrics.get('days_since_5pct_pctile')
    ds_vix_pctile = metrics.get('days_since_vix_spike_pctile')

    extended_calm_on = False
    both_elevated = (ds_5_pctile is not None and ds_5_pctile > 70 and
                     ds_vix_pctile is not None and ds_vix_pctile > 70)
    either_extreme = ((ds_5_pctile is not None and ds_5_pctile > 85) or
                      (ds_vix_pctile is not None and ds_vix_pctile > 85))
    extended_calm_on = both_elevated or either_extreme
    signals['Extended Calm'] = extended_calm_on
    if extended_calm_on:
        ds5_str = f"{metrics.get('days_since_5pct', 0):.0f} days ({ds_5_pctile:.0f}th)" if ds_5_pctile else "N/A"
        dsv_str = f"{metrics.get('days_since_vix_spike', 0):.0f} days ({ds_vix_pctile:.0f}th)" if ds_vix_pctile else "N/A"
        details['Extended Calm'] = (
            f"Days since 5% correction: {ds5_str}. Days since VIX > 28: {dsv_str}. "
            f"Leveraged and systematic positions haven't been cleared in a long time."
        )

    # Signal 2C: Vol Compression Duration
    rv22d_series = metrics.get('rv_22d_series')
    vol_compress_on = False
    vol_compress_days = 0
    vol_compress_depth = 0.0
    if rv22d_series is not None and len(rv22d_series.dropna()) > 252:
        rv_median = rv22d_series.expanding(min_periods=252).median()
        below_median = rv22d_series < rv_median

        # Count consecutive days below median at end of series
        clean = below_median.dropna()
        if len(clean) > 0 and clean.iloc[-1]:
            count = 0
            for i in range(len(clean) - 1, -1, -1):
                if clean.iloc[i]:
                    count += 1
                else:
                    break
            vol_compress_days = count

            # Depth: how far below median is current RV?
            cur_rv = rv22d_series.dropna().iloc[-1]
            cur_median = rv_median.dropna().iloc[-1]
            if cur_median > 0:
                vol_compress_depth = 1.0 - (cur_rv / cur_median)

        vol_compress_on = vol_compress_days > 60

    signals['Vol Compression'] = vol_compress_on
    if vol_compress_on:
        details['Vol Compression'] = (
            f"Realized vol has been below its median for {vol_compress_days} consecutive days "
            f"(currently {vol_compress_depth:.0%} below median). Participants have adapted: "
            f"reduced hedges, increased leverage, sold options."
        )

    # =====================================================================
    # QUESTION 3: ARE CORRELATIONS STABLE?
    # =====================================================================

    # Signal 3A: Credit-Equity Divergence
    hy_z = metrics.get('credit_hy_z')
    spy_21d_ret = metrics.get('spy_21d_return')

    credit_eq_div_on = False
    if hy_z is not None and spy_21d_ret is not None:
        credit_eq_div_on = (hy_z > 0.75) and (spy_21d_ret > -0.02)
    signals['Credit-Equity Divergence'] = credit_eq_div_on
    if credit_eq_div_on:
        details['Credit-Equity Divergence'] = (
            f"HY credit spreads widening (z: {hy_z:+.1f}\u03c3) while SPX is stable "
            f"({spy_21d_ret:+.1%} over 21d). Credit is sniffing risk that equity "
            f"hasn't priced. Historically leads equity by 2-6 weeks."
        )

    # Signal 3B: Rates-Equity Vol Gap
    move_val = metrics.get('move')
    move_pctile = metrics.get('move_pctile')
    vix_val = metrics.get('vix')
    vix_pctile = metrics.get('vix_pctile')

    rates_eq_gap_on = False
    if move_pctile is not None and vix_pctile is not None:
        rates_eq_gap_on = (move_pctile > 70) and (vix_pctile < 40)
    elif move_val is not None and vix_val is not None:
        rates_eq_gap_on = (move_val > 100) and (vix_val < 18)
    signals['Rates-Equity Vol Gap'] = rates_eq_gap_on
    if rates_eq_gap_on:
        move_str = f"{move_val:.0f}" if move_val else "N/A"
        vix_str = f"{vix_val:.1f}" if vix_val else "N/A"
        details['Rates-Equity Vol Gap'] = (
            f"MOVE at {move_str} (elevated) while VIX at {vix_str} (calm). "
            f"Rates vol transmits to equity vol via dealer balance sheets. "
            f"This gap tends to close via equity vol rising."
        )

    # Signal 3C: VIX Uncertainty
    vvix_val = metrics.get('vvix')
    vvix_vix_ratio = None
    if vvix_val is not None and vix_val is not None and vix_val > 0:
        vvix_vix_ratio = vvix_val / vix_val

    vvix_vix_pctile = metrics.get('vvix_vix_ratio_pctile')

    vol_uncertainty_on = False
    if vvix_vix_pctile is not None:
        vol_uncertainty_on = vvix_vix_pctile > 80
    elif vvix_vix_ratio is not None:
        vol_uncertainty_on = vvix_vix_ratio > 7.5
    signals['Vol Uncertainty'] = vol_uncertainty_on
    if vol_uncertainty_on:
        ratio_str = f"{vvix_vix_ratio:.1f}" if vvix_vix_ratio else "N/A"
        details['Vol Uncertainty'] = (
            f"VVIX/VIX ratio at {ratio_str} (elevated). The options market is "
            f"pricing wide uncertainty around the vol path \u2014 the market doesn't "
            f"trust the current calm. Explosive move potential in either direction."
        )

    # Group signals by question
    questions = {
        'Is liquidity real?': ['Vol Suppression', 'VRP Compression'],
        'Is everyone on the same side?': ['Breadth Divergence', 'Extended Calm', 'Vol Compression'],
        'Are correlations stable?': ['Credit-Equity Divergence', 'Rates-Equity Vol Gap', 'Vol Uncertainty'],
    }

    return {
        'signals': signals,
        'details': details,
        'questions': questions,
        'vol_compress_days': vol_compress_days,
        'vol_compress_depth': vol_compress_depth,
    }


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
    # Reserve 80-100 for extreme regime-amplified readings
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


def chart_har_rv(spy_df: pd.DataFrame) -> go.Figure:
    rv1 = yang_zhang_vol(spy_df, 1)
    rv5 = yang_zhang_vol(spy_df, 5)
    rv22 = yang_zhang_vol(spy_df, 22)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rv1.index, y=rv1, name="RV 1d", line=dict(width=1, color="#FF6B6B")))
    fig.add_trace(go.Scatter(x=rv5.index, y=rv5, name="RV 5d", line=dict(width=1.5, color="#4ECDC4")))
    fig.add_trace(go.Scatter(x=rv22.index, y=rv22, name="RV 22d", line=dict(width=2, color="#0066CC")))
    layout = _base_layout("HAR-RV Decomposition (Yang-Zhang)")
    one_year_ago = (pd.Timestamp.now() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    layout["xaxis"] = dict(showgrid=False, range=[one_year_ago, None])
    fig.update_layout(**layout)
    fig.update_yaxes(tickformat=".0%")
    return fig, rv1, rv5, rv22


def chart_vrp(vrp_series: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vrp_series.index, y=vrp_series,
        name="VRP",
        line=dict(width=1.5, color="#0066CC"),
        fill="tozeroy",
        fillcolor="rgba(0,102,204,0.1)",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#CC0000", line_width=1)
    layout = _base_layout("Variance Risk Premium")
    one_year_ago = (pd.Timestamp.now() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    layout["xaxis"] = dict(showgrid=False, range=[one_year_ago, None])
    fig.update_layout(**layout)
    return fig


def chart_term_structure(ts_ratio: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_ratio.index, y=ts_ratio,
        name="VIX/VIX3M",
        line=dict(width=1.5, color="#0066CC"),
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#888888", line_width=1)
    fig.add_hrect(y0=1.0, y1=ts_ratio.max() * 1.05 if len(ts_ratio) > 0 else 1.5,
                  fillcolor="rgba(204,0,0,0.08)", line_width=0)
    fig.update_layout(**_base_layout("VIX Term Structure (VIX / VIX3M)"))
    return fig


def chart_vvix(vvix_series: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vvix_series.index, y=vvix_series,
        name="VVIX",
        line=dict(width=1.5, color="#0066CC"),
    ))
    for level, color in [(80, "#00CC00"), (100, "#FFD700"), (120, "#CC0000")]:
        fig.add_hline(y=level, line_dash="dot", line_color=color, line_width=1,
                      annotation_text=str(level), annotation_position="right")
    fig.update_layout(**_base_layout("VVIX (Volatility of Volatility)"))
    return fig


def chart_breadth(breadth_df: pd.DataFrame, spy_close: pd.Series) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=breadth_df.index, y=breadth_df["pct_above_200"], name="% > 200d SMA",
                   line=dict(width=1.5, color="#0066CC")),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=breadth_df.index, y=breadth_df["pct_above_50"], name="% > 50d SMA",
                   line=dict(width=1.5, color="#4ECDC4")),
        secondary_y=False,
    )
    # Align SPY to breadth index
    spy_aligned = spy_close.reindex(breadth_df.index).dropna()
    fig.add_trace(
        go.Scatter(x=spy_aligned.index, y=spy_aligned, name="SPY",
                   line=dict(width=1, color="rgba(128,128,128,0.4)")),
        secondary_y=True,
    )
    fig.add_hline(y=60, line_dash="dot", line_color="#FFD700", line_width=1, secondary_y=False)
    fig.add_hline(y=40, line_dash="dot", line_color="#CC0000", line_width=1, secondary_y=False)
    layout = _base_layout("Market Breadth (Sector ETF Proxy)")
    layout.pop("yaxis", None)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="% Above SMA", secondary_y=False)
    fig.update_yaxes(title_text="SPY", showgrid=False, secondary_y=True)
    return fig


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


def chart_dispersion(disp_series: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=disp_series.index, y=disp_series,
        name="Cross-Sectional Dispersion",
        line=dict(width=1.5, color="#0066CC"),
        fill="tozeroy",
        fillcolor="rgba(0,102,204,0.1)",
    ))
    fig.update_layout(**_base_layout("Cross-Sectional Return Dispersion (21d)"))
    fig.update_yaxes(tickformat=".1%")
    return fig


def chart_hurst(hurst_series: pd.Series, p20: float = None, p80: float = None) -> go.Figure:
    fig = go.Figure()
    # Background shading using empirical percentile bands
    h_min = hurst_series.dropna().min() if len(hurst_series.dropna()) > 0 else 0.3
    h_max = hurst_series.dropna().max() if len(hurst_series.dropna()) > 0 else 0.8
    if p20 is not None and p80 is not None:
        fig.add_hrect(y0=h_min - 0.05, y1=p20, fillcolor="rgba(0,204,0,0.05)", line_width=0)
        fig.add_hrect(y0=p20, y1=p80, fillcolor="rgba(128,128,128,0.05)", line_width=0)
        fig.add_hrect(y0=p80, y1=h_max + 0.05, fillcolor="rgba(204,0,0,0.05)", line_width=0)
    fig.add_trace(go.Scatter(
        x=hurst_series.index, y=hurst_series,
        name="Hurst (DFA)",
        line=dict(width=1.5, color="#0066CC"),
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#888888", line_width=1,
                  annotation_text="0.5", annotation_position="right")
    if p20 is not None:
        fig.add_hline(y=p20, line_dash="dot", line_color="#00CC00", line_width=1,
                      annotation_text=f"P20: {p20:.2f}", annotation_position="left")
    if p80 is not None:
        fig.add_hline(y=p80, line_dash="dot", line_color="#CC0000", line_width=1,
                      annotation_text=f"P80: {p80:.2f}", annotation_position="left")
    fig.update_layout(**_base_layout("Hurst Exponent (DFA, 126d rolling)"))
    return fig


def render_2x2_grid(disp_high: bool, corr_high: bool):
    """Render the dispersion x correlation 2x2 grid."""
    cells = [
        ("Low Disp / Low Corr", not disp_high and not corr_high, "Normal: idiosyncratic, diversified", "#00CC00"),
        ("Low Disp / High Corr", not disp_high and corr_high, "Quiet but correlated: watch for breakout", "#FFD700"),
        ("High Disp / Low Corr", disp_high and not corr_high, "Stock-pickers market: alpha opportunity", "#4ECDC4"),
        ("High Disp / High Corr", disp_high and corr_high, "Stress: high dispersion + high correlation", "#CC0000"),
    ]

    cols = st.columns(2)
    for idx, (label, active, desc, color) in enumerate(cells):
        col = cols[idx % 2]
        bg = f"{color}30" if active else "#33333320"
        border = f"3px solid {color}" if active else "1px solid #55555540"
        with col:
            st.markdown(
                f'<div style="background:{bg}; border:{border}; border-radius:6px; '
                f'padding:10px; margin-bottom:8px; min-height:60px;">'
                f'<strong>{label}</strong>{"  ◀ CURRENT" if active else ""}<br>'
                f'<small>{desc}</small></div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# LAYER 3 / 4 CHART HELPERS
# ---------------------------------------------------------------------------

CHART_HEIGHT_SMALL = 200


def chart_credit_spreads(ig_z: pd.Series, hy_z: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ig_z.index, y=ig_z, name="IG z-score",
                             line=dict(width=1.5, color="#0066CC")))
    fig.add_trace(go.Scatter(x=hy_z.index, y=hy_z, name="HY z-score",
                             line=dict(width=1.5, color="#FF6B6B")))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#FFD700", line_width=1,
                  annotation_text="1.0\u03c3", annotation_position="right")
    fig.add_hline(y=2.0, line_dash="dot", line_color="#CC0000", line_width=1,
                  annotation_text="2.0\u03c3", annotation_position="right")
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(128,128,128,0.3)", line_width=1)
    layout = _base_layout("Credit Spread Proxy (z-score)")
    layout["height"] = CHART_HEIGHT_SMALL
    fig.update_layout(**layout)
    return fig


def chart_yield_curve(spread: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spread.index, y=spread, name="10Y-3M Spread",
        line=dict(width=1.5, color="#0066CC"),
        fill="tozeroy",
        fillcolor="rgba(0,102,204,0.08)",
    ))
    fig.add_hline(y=0, line_dash="solid", line_color="#CC0000", line_width=1.5,
                  annotation_text="Inversion", annotation_position="right")
    # Shade below zero red
    fig.add_hrect(y0=spread.min() - 0.5 if len(spread.dropna()) > 0 else -2,
                  y1=0, fillcolor="rgba(204,0,0,0.08)", line_width=0)
    layout = _base_layout("Yield Curve (10Y - 3M)")
    layout["height"] = CHART_HEIGHT_SMALL
    fig.update_layout(**layout)
    return fig


def chart_move(move_series: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=move_series.index, y=move_series, name="MOVE",
        line=dict(width=1.5, color="#0066CC"),
    ))
    fig.add_hline(y=80, line_dash="dot", line_color="#00CC00", line_width=1,
                  annotation_text="80", annotation_position="right")
    fig.add_hline(y=120, line_dash="dot", line_color="#FFD700", line_width=1,
                  annotation_text="120", annotation_position="right")
    fig.add_hline(y=150, line_dash="dot", line_color="#CC0000", line_width=1,
                  annotation_text="150", annotation_position="right")
    fig.add_hrect(y0=120, y1=150, fillcolor="rgba(255,215,0,0.08)", line_width=0)
    fig.add_hrect(y0=150,
                  y1=move_series.max() * 1.05 if len(move_series.dropna()) > 0 else 200,
                  fillcolor="rgba(204,0,0,0.08)", line_width=0)
    layout = _base_layout("MOVE Index")
    layout["height"] = CHART_HEIGHT_SMALL
    fig.update_layout(**layout)
    return fig


def chart_dollar(pct_change_21d: pd.Series) -> go.Figure:
    fig = go.Figure()
    colors = []
    for v in pct_change_21d.dropna():
        if abs(v) > 5:
            colors.append("#CC0000")
        elif abs(v) > 3:
            colors.append("#FF8C00")
        else:
            colors.append("#0066CC")
    fig.add_trace(go.Bar(
        x=pct_change_21d.dropna().index, y=pct_change_21d.dropna(),
        name="21d % Chg", marker_color=colors,
    ))
    fig.add_hline(y=3, line_dash="dot", line_color="#FF8C00", line_width=1)
    fig.add_hline(y=-3, line_dash="dot", line_color="#FF8C00", line_width=1)
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(128,128,128,0.3)", line_width=1)
    layout = _base_layout("Dollar (UUP) 21d Momentum (%)")
    layout["height"] = CHART_HEIGHT_SMALL
    fig.update_layout(**layout)
    return fig


def chart_days_since_sawtooth(days_series: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days_series.index, y=days_series, name="Days Since",
        line=dict(width=1.5, color="#0066CC"),
        fill="tozeroy",
        fillcolor="rgba(0,102,204,0.08)",
    ))
    layout = _base_layout(title)
    layout["height"] = CHART_HEIGHT_SMALL
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# METRIC STATUS BADGE
# ---------------------------------------------------------------------------

def status_badge(label: str, value, fmt: str = ".2f", alert: bool = False, alarm: bool = False) -> str:
    if alarm:
        color = "#CC0000"
        icon = "\U0001f534"
    elif alert:
        color = "#FFD700"
        icon = "\U0001f7e1"
    else:
        color = "#00CC00"
        icon = "\U0001f7e2"
    val_str = f"{value:{fmt}}" if isinstance(value, (int, float)) and not np.isnan(value) else "N/A"
    return f"{icon} **{label}:** {val_str}"


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("\U0001f4ca Risk Dashboard V2")

    # --- Sidebar ---
    with st.sidebar:
        st.header("\u2699\ufe0f Dashboard Settings")
        lookback_years = st.slider("History (years)", 5, 15, 10)
        st.divider()
        st.subheader("Alert Thresholds")
        st.caption("Defaults based on academic literature. Adjust with caution.")
        # TODO: Add editable thresholds in Phase 2
        st.divider()
        refresh = st.button("\U0001f504 Refresh Data", type="primary", use_container_width=True)

    start_date = (datetime.datetime.now() - datetime.timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")

    # Enforce minimum start for computation stability
    if start_date > "2010-01-01":
        dl_start = start_date
    else:
        dl_start = "2010-01-01"

    # --- Download data ---
    if refresh:
        st.cache_data.clear()

    vol_data = download_vol_data(dl_start)
    sector_data = download_sector_data(dl_start)
    cross_asset_data = download_cross_asset_data(dl_start)
    tail_risk_data = download_tail_risk_data(dl_start)

    # Validate minimum data
    if "SPY" not in vol_data:
        st.error("Could not download SPY data. Check your internet connection.")
        st.stop()

    spy_df = vol_data["SPY"]

    # -----------------------------------------------------------------------
    # LAYER 1 COMPUTATIONS
    # -----------------------------------------------------------------------
    # 1A: HAR-RV
    rv_1d = yang_zhang_vol(spy_df, 1)
    rv_5d = yang_zhang_vol(spy_df, 5)
    rv_22d = yang_zhang_vol(spy_df, 22)
    rv_22d_pctile = expanding_percentile(rv_22d)

    # 1B: VRP
    vix_available = "^VIX" in vol_data
    vrp_series = pd.Series(dtype=float)
    vrp_pctile_series = pd.Series(dtype=float)
    if vix_available:
        vix_close = vol_data["^VIX"]["Close"]
        # Align VIX and RV on common index
        common_idx = rv_22d.dropna().index.intersection(vix_close.dropna().index)
        if len(common_idx) > 0:
            vrp_series = compute_vrp(vix_close.reindex(common_idx), rv_22d.reindex(common_idx))
            vrp_pctile_series = expanding_percentile(vrp_series)

    # 1C: VIX Term Structure
    vix3m_available = "^VIX3M" in vol_data
    ts_ratio_series = pd.Series(dtype=float)
    if vix_available and vix3m_available:
        vix_c = vol_data["^VIX"]["Close"]
        vix3m_c = vol_data["^VIX3M"]["Close"]
        common_idx = vix_c.dropna().index.intersection(vix3m_c.dropna().index)
        if len(common_idx) > 0:
            ts_ratio_series = compute_vix_term_structure(
                vix_c.reindex(common_idx), vix3m_c.reindex(common_idx)
            )

    # 1D: VVIX
    vvix_available = "^VVIX" in vol_data
    vvix_series = pd.Series(dtype=float)
    if vvix_available:
        vvix_series = vol_data["^VVIX"]["Close"].dropna()

    # -----------------------------------------------------------------------
    # LAYER 2 COMPUTATIONS
    # -----------------------------------------------------------------------
    # Build sector close and return matrices
    sector_closes = pd.DataFrame(
        {t: d["Close"] for t, d in sector_data.items() if "Close" in d.columns}
    )
    sector_returns = sector_closes.pct_change().dropna(how="all")

    active_sectors = sector_returns.columns.tolist()
    if len(active_sectors) < 8:
        st.warning(f"Only {len(active_sectors)} sector ETFs available (need 8+). Some metrics may be degraded.")

    # 2A: Breadth
    breadth_df = compute_breadth_sector_proxy(sector_data)

    # 2B: Absorption Ratio
    ar_series = pd.Series(dtype=float)
    if len(active_sectors) >= 5:
        with st.spinner("Computing absorption ratio (PCA)..."):
            ar_series = compute_absorption_ratio(sector_returns, window=63)

    # 2C: Dispersion + Correlation
    disp_series = pd.Series(dtype=float)
    corr_series = pd.Series(dtype=float)
    if len(active_sectors) >= 5:
        disp_series = compute_cross_sectional_dispersion(sector_returns, window=21)
        with st.spinner("Computing pairwise correlations..."):
            corr_series = compute_avg_pairwise_correlation(sector_returns, window=21)

    # 2D: Hurst (smoothed: 11d rolling median → 15d EMA)
    spy_returns = spy_df["Close"].pct_change().dropna()
    hurst_raw = pd.Series(dtype=float)
    with st.spinner("Computing Hurst exponent (DFA)..."):
        hurst_raw = compute_hurst_dfa(spy_returns, window=126)
    hurst_series = hurst_raw.rolling(11, center=True).median().ewm(span=15).mean()
    hurst_pctile_series = expanding_percentile(hurst_series, min_periods=252)

    # 2E: Days Since Correction (5% and 10%)
    spy_close = spy_df["Close"]
    days_since_5_series = compute_days_since(spy_close, threshold_pct=0.05)
    days_since_5_pctile_series = expanding_percentile(days_since_5_series, min_periods=252)
    days_since_10_series = compute_days_since(spy_close, threshold_pct=0.10)
    days_since_10_pctile_series = expanding_percentile(days_since_10_series, min_periods=252)

    # 2E addl: Days since VIX > 28
    days_since_vix_series = pd.Series(dtype=float)
    days_since_vix_pctile_series = pd.Series(dtype=float)
    if vix_available:
        vix_c_for_spike = vol_data["^VIX"]["Close"].dropna()
        days_since_vix_series = compute_days_since_vix_spike(vix_c_for_spike, threshold=28.0)
        days_since_vix_pctile_series = expanding_percentile(days_since_vix_series, min_periods=252)

    # -----------------------------------------------------------------------
    # LAYER 3 COMPUTATIONS
    # -----------------------------------------------------------------------
    # 3A: Credit Spread Proxy
    ig_z_series = pd.Series(dtype=float)
    hy_z_series = pd.Series(dtype=float)
    lqd_ok = 'LQD' in cross_asset_data
    hyg_ok = 'HYG' in cross_asset_data
    ief_ok = 'IEF' in cross_asset_data
    if lqd_ok and hyg_ok and ief_ok:
        try:
            lqd_c = cross_asset_data['LQD']['Close']
            hyg_c = cross_asset_data['HYG']['Close']
            ief_c = cross_asset_data['IEF']['Close']
            common_credit = lqd_c.dropna().index.intersection(
                hyg_c.dropna().index).intersection(ief_c.dropna().index)
            if len(common_credit) > 63:
                ig_z_series, hy_z_series = compute_credit_spread_proxy(
                    lqd_c.reindex(common_credit),
                    hyg_c.reindex(common_credit),
                    ief_c.reindex(common_credit),
                )
        except Exception as e:
            print(f"Warning: Credit spread computation failed: {e}")

    # 3B: Yield Curve
    yc_spread_series = pd.Series(dtype=float)
    yc_z_series = pd.Series(dtype=float)
    tnx_ok = '^TNX' in cross_asset_data
    irx_ok = '^IRX' in cross_asset_data
    if tnx_ok and irx_ok:
        try:
            tnx_c = cross_asset_data['^TNX']['Close']
            irx_c = cross_asset_data['^IRX']['Close']
            common_yc = tnx_c.dropna().index.intersection(irx_c.dropna().index)
            if len(common_yc) > 252:
                yc_spread_series, yc_z_series = compute_yield_curve(
                    tnx_c.reindex(common_yc), irx_c.reindex(common_yc))
        except Exception as e:
            print(f"Warning: Yield curve computation failed: {e}")

    # 3C: MOVE Index
    move_series = pd.Series(dtype=float)
    if '^MOVE' in cross_asset_data:
        try:
            move_series = cross_asset_data['^MOVE']['Close'].dropna()
        except Exception:
            pass

    # 3D: Dollar Dynamics
    dollar_21d_series = pd.Series(dtype=float)
    if 'UUP' in cross_asset_data:
        try:
            uup_c = cross_asset_data['UUP']['Close'].dropna()
            dollar_21d_series = compute_dollar_momentum(uup_c)
        except Exception as e:
            print(f"Warning: Dollar computation failed: {e}")

    # -----------------------------------------------------------------------
    # LAYER 4 COMPUTATIONS
    # -----------------------------------------------------------------------
    skew_series = pd.Series(dtype=float)
    if '^SKEW' in tail_risk_data:
        try:
            skew_series = tail_risk_data['^SKEW']['Close'].dropna()
        except Exception:
            pass

    # Protection cost proxy
    prot_cost_series = pd.Series(dtype=float)
    prot_pctile_series = pd.Series(dtype=float)
    if vix3m_available:
        vix3m_for_prot = vol_data["^VIX3M"]["Close"].dropna()
        skew_for_prot = skew_series if len(skew_series.dropna()) > 0 else None
        try:
            prot_cost_series, prot_pctile_series = compute_protection_cost(
                vix3m_for_prot, skew_for_prot)
        except Exception as e:
            print(f"Warning: Protection cost computation failed: {e}")

    # -----------------------------------------------------------------------
    # COLLECT CURRENT READINGS FOR LAYER 0
    # -----------------------------------------------------------------------
    def _last_valid(s):
        if s is None or len(s) == 0:
            return None
        v = s.dropna()
        return float(v.iloc[-1]) if len(v) > 0 else None

    cur_rv1d = _last_valid(rv_1d)
    cur_rv22d = _last_valid(rv_22d)
    cur_rv22d_pctile = _last_valid(rv_22d_pctile)

    # Previous day's RV22d for "rising" check
    rv22d_clean = rv_22d.dropna()
    cur_rv22d_prev = float(rv22d_clean.iloc[-2]) if len(rv22d_clean) >= 2 else None

    cur_vrp = _last_valid(vrp_series)
    cur_vrp_pctile = _last_valid(vrp_pctile_series)

    cur_ts_ratio = _last_valid(ts_ratio_series)
    cur_vvix = _last_valid(vvix_series)

    cur_pct200 = _last_valid(breadth_df["pct_above_200"]) if not breadth_df.empty else None

    # SPY near 52-week high?
    spy_52w_high = spy_close.rolling(252).max()
    spy_near_high = False
    if len(spy_52w_high.dropna()) > 0:
        latest_price = float(spy_close.iloc[-1])
        latest_high = float(spy_52w_high.iloc[-1])
        if latest_high > 0:
            spy_near_high = latest_price >= latest_high * 0.95

    cur_ar = _last_valid(ar_series)
    ar_clean = ar_series.dropna()

    # Was AR < 0.4 at any point in the last 10 days?
    # Dispersion & correlation thresholds (75th percentile)
    disp_75 = disp_series.quantile(0.75) if len(disp_series.dropna()) > 100 else None
    corr_75 = corr_series.quantile(0.75) if len(corr_series.dropna()) > 100 else None
    cur_disp = _last_valid(disp_series)
    cur_corr = _last_valid(corr_series)
    disp_high = cur_disp is not None and disp_75 is not None and cur_disp > disp_75
    corr_high = cur_corr is not None and corr_75 is not None and cur_corr > corr_75

    # Hurst
    cur_hurst = _last_valid(hurst_series)
    cur_hurst_pctile = _last_valid(hurst_pctile_series)
    hurst_clean = hurst_series.dropna()
    hurst_delta_5d = None
    if len(hurst_clean) >= 6:
        hurst_delta_5d = float(hurst_clean.iloc[-1] - hurst_clean.iloc[-6])
    # Empirical percentile bands for chart
    hurst_p20 = float(hurst_clean.quantile(0.20)) if len(hurst_clean) > 252 else None
    hurst_p80 = float(hurst_clean.quantile(0.80)) if len(hurst_clean) > 252 else None

    # Days Since Correction
    cur_days_since_5 = _last_valid(days_since_5_series)
    cur_days_since_5_pctile = _last_valid(days_since_5_pctile_series)
    cur_days_since_10 = _last_valid(days_since_10_series)
    cur_days_since_10_pctile = _last_valid(days_since_10_pctile_series)
    cur_days_since_vix = _last_valid(days_since_vix_series)
    cur_days_since_vix_pctile = _last_valid(days_since_vix_pctile_series)

    # Layer 3 current readings
    cur_ig_z = _last_valid(ig_z_series)
    cur_hy_z = _last_valid(hy_z_series)
    cur_yc_spread = _last_valid(yc_spread_series)
    cur_yc_z = _last_valid(yc_z_series)
    cur_move = _last_valid(move_series)
    cur_dollar_21d = _last_valid(dollar_21d_series)

    # Layer 4 current readings
    cur_skew = _last_valid(skew_series)
    cur_prot_pctile = _last_valid(prot_pctile_series)

    # --- Additional percentiles for signal framework ---
    ar_pctile_series = expanding_percentile(ar_series) if len(ar_series.dropna()) > 0 else pd.Series(dtype=float)
    cur_ar_pctile = _last_valid(ar_pctile_series)

    disp_pctile_series = expanding_percentile(disp_series) if len(disp_series.dropna()) > 0 else pd.Series(dtype=float)
    cur_disp_pctile = _last_valid(disp_pctile_series)

    corr_pctile_series = expanding_percentile(corr_series) if len(corr_series.dropna()) > 0 else pd.Series(dtype=float)
    cur_corr_pctile = _last_valid(corr_pctile_series)

    move_pctile_series = expanding_percentile(move_series) if len(move_series.dropna()) > 0 else pd.Series(dtype=float)
    cur_move_pctile = _last_valid(move_pctile_series)

    # VIX current value and percentile
    cur_vix = _last_valid(vol_data["^VIX"]["Close"]) if "^VIX" in vol_data else None
    vix_pctile_series = expanding_percentile(vol_data["^VIX"]["Close"]) if "^VIX" in vol_data else pd.Series(dtype=float)
    cur_vix_pctile = _last_valid(vix_pctile_series)

    # SPY 21-day return (for credit-equity divergence)
    spy_21d_return = float(spy_close.iloc[-1] / spy_close.iloc[-22] - 1) if len(spy_close) >= 22 else None

    # VVIX/VIX ratio percentile
    cur_vvix_vix_ratio_pctile = None
    if vvix_available and vix_available:
        vvix_c_ratio = vol_data["^VVIX"]["Close"].dropna()
        vix_c_ratio = vol_data["^VIX"]["Close"].dropna()
        common_ratio = vvix_c_ratio.index.intersection(vix_c_ratio.index)
        vvix_vix_ratio_series = vvix_c_ratio.reindex(common_ratio) / vix_c_ratio.reindex(common_ratio)
        vvix_vix_ratio_series = vvix_vix_ratio_series.replace([np.inf, -np.inf], np.nan).dropna()
        vvix_vix_ratio_pctile_series = expanding_percentile(vvix_vix_ratio_series) if len(vvix_vix_ratio_series) > 0 else pd.Series(dtype=float)
        cur_vvix_vix_ratio_pctile = _last_valid(vvix_vix_ratio_pctile_series)

    # --- Legacy scoring (kept for reference) ---
    metrics_for_scoring = {
        "rv_1d": cur_rv1d,
        "rv_22d": cur_rv22d,
        "rv_22d_pctile": cur_rv22d_pctile,
        "rv_22d_prev": cur_rv22d_prev,
        "vrp": cur_vrp,
        "vrp_pctile": cur_vrp_pctile,
        "vix_ts_ratio": cur_ts_ratio,
        "vvix": cur_vvix,
        "pct_above_200": cur_pct200,
        "spy_near_52w_high": spy_near_high,
        "dispersion_high": disp_high,
        "correlation_high": corr_high,
        "hurst_delta_5d": hurst_delta_5d,
        "hurst_pctile": cur_hurst_pctile,
        # 2E: compound complacency
        "days_since_5pct_pctile": cur_days_since_5_pctile,
        "days_since_vix_spike_pctile": cur_days_since_vix_pctile,
        # Layer 3
        "credit_ig_z": cur_ig_z,
        "credit_hy_z": cur_hy_z,
        "yield_curve_spread": cur_yc_spread,
        "yield_curve_z": cur_yc_z,
        "move": cur_move,
        "dollar_21d_pct": cur_dollar_21d,
    }

    # Keep regime for Layer 4 hedge recommendation
    total_pts, pt_breakdown = score_alerts(metrics_for_scoring)
    regime = classify_regime(total_pts)

    # ===================================================================
    # EXECUTIVE SUMMARY — Signal-Based Framework
    # ===================================================================

    # Compute price context
    price_ctx = compute_price_context(spy_close)
    regime_mult = compute_regime_multiplier(price_ctx)

    # Section A: Price Context Banner
    render_price_context(price_ctx)

    # Compute condition signals
    signal_metrics = {
        'rv_22d_pctile': cur_rv22d_pctile,
        'vrp': cur_vrp,
        'vrp_pctile': cur_vrp_pctile,
        'pct_above_200': cur_pct200,
        'spy_near_52w_high': spy_near_high,
        'days_since_5pct_pctile': cur_days_since_5_pctile,
        'days_since_vix_spike_pctile': cur_days_since_vix_pctile,
        'days_since_5pct': cur_days_since_5,
        'days_since_vix_spike': cur_days_since_vix,
        'credit_hy_z': cur_hy_z,
        'move': cur_move,
        'vvix': cur_vvix,
        'vix': cur_vix,
        'ar_pctile': cur_ar_pctile,
        'rv_22d_series': rv_22d,
        'spy_21d_return': spy_21d_return,
        'move_pctile': cur_move_pctile,
        'vix_pctile': cur_vix_pctile,
        'vvix_vix_ratio_pctile': cur_vvix_vix_ratio_pctile,
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

        # TODO: Replace placeholder hit rates with actual event study results.
        # These are rough estimates. Run the individual signal backtests to calibrate.
        if active_count == 0:
            dial_label = "No warning signals active"
            hit_rate_note = ""
        elif active_count <= 2:
            dial_label = f"{active_count} of {total_count} signals active"
            hit_rate_note = "Historically, 1-2 active signals precede corrections ~30% of the time."
        elif active_count <= 4:
            dial_label = f"{active_count} of {total_count} signals active"
            hit_rate_note = "3-4 signals active has preceded corrections within 6 weeks ~55% of the time."
        else:
            dial_label = f"{active_count} of {total_count} signals active"
            hit_rate_note = "5+ signals is rare and has historically preceded significant corrections."

        st.markdown(f"<p style='text-align: center; font-size: 13px; margin-top: -8px;'>{dial_label}</p>",
                    unsafe_allow_html=True)
        if hit_rate_note:
            st.markdown(f"<p style='text-align: center; font-size: 11px; color: #888; margin-top: -5px; font-style: italic;'>{hit_rate_note}</p>",
                        unsafe_allow_html=True)

    # Section C: Stored Energy (conditional — only when 2+ signals active)
    vol_compress_days = signal_result.get('vol_compress_days', 0)
    vol_compress_depth = signal_result.get('vol_compress_depth', 0.0)

    if active_count >= 2:
        st.markdown("---")
        st.markdown("##### If conditions deteriorate:")

        se_cols = st.columns(3)

        with se_cols[0]:
            if vol_compress_days > 0:
                st.metric(
                    "Vol Compression",
                    f"{vol_compress_days}d below median",
                    f"Depth: {vol_compress_depth:.0%} below",
                )
            else:
                st.metric("Vol Compression", "Not compressed", "")

        with se_cols[1]:
            ds5_val = int(cur_days_since_5) if cur_days_since_5 and not np.isnan(cur_days_since_5) else 0
            ds5_p = f"{cur_days_since_5_pctile:.0f}th" if cur_days_since_5_pctile and not np.isnan(cur_days_since_5_pctile) else "N/A"
            st.metric(
                "Calm Streak",
                f"{ds5_val}d since 5% DD",
                f"{ds5_p} percentile",
            )

        with se_cols[2]:
            ext = price_ctx.get('extension_200d', 0) or 0

            base_dd = 3.0
            if ext > 0.10:
                base_dd += 3.0
            elif ext > 0.05:
                base_dd += 1.5
            if vol_compress_days > 100:
                base_dd += 2.0
            elif vol_compress_days > 50:
                base_dd += 1.0
            if active_count >= 4:
                base_dd += 2.0

            low_est = base_dd
            high_est = base_dd + 4.0

            st.metric(
                "Potential Unwind",
                f"~{low_est:.0f}\u2013{high_est:.0f}%",
                "if correction materializes",
            )
            st.caption("Based on extension, compression duration, and active signal count. Rough estimate, not a prediction.")

    # Legacy scoring detail (point system)
    with st.expander("Legacy scoring detail (point system)", expanded=False):
        if pt_breakdown:
            for metric_name, pts in pt_breakdown.items():
                st.markdown(f"- **{metric_name}**: +{pts}")
        else:
            st.markdown("No legacy alerts triggered.")
        st.caption(f"Legacy regime: {regime} ({total_pts} pts) | 0 pts = Normal | 1-2 = Caution | 3-4 = Stress | 5+ = Crisis")

    st.divider()

    # ===================================================================
    # LAYERS 1 & 2: Two-column layout
    # ===================================================================
    col_l1, col_l2 = st.columns([1, 1])

    # -------------------------------------------------------------------
    # LAYER 1 (left column)
    # -------------------------------------------------------------------
    with col_l1:
        st.markdown("### Layer 1: Volatility State")

        # 1A: HAR-RV
        st.markdown("#### 1A. HAR-RV Decomposition")
        fig_rv, _, _, _ = chart_har_rv(spy_df)
        st.plotly_chart(fig_rv, use_container_width=True)

        # Current readings table
        rv_data = {
            "Horizon": ["1d", "5d", "22d"],
            "RV (ann.)": [
                f"{cur_rv1d:.1%}" if cur_rv1d is not None else "N/A",
                f"{_last_valid(rv_5d):.1%}" if _last_valid(rv_5d) is not None else "N/A",
                f"{cur_rv22d:.1%}" if cur_rv22d is not None else "N/A",
            ],
            "Percentile": [
                f"{_last_valid(expanding_percentile(rv_1d)):.0f}" if _last_valid(expanding_percentile(rv_1d)) is not None else "N/A",
                f"{_last_valid(expanding_percentile(rv_5d)):.0f}" if _last_valid(expanding_percentile(rv_5d)) is not None else "N/A",
                f"{cur_rv22d_pctile:.0f}" if cur_rv22d_pctile is not None else "N/A",
            ],
        }
        st.dataframe(pd.DataFrame(rv_data), hide_index=True, use_container_width=True)

        is_spike = cur_rv1d is not None and cur_rv22d is not None and cur_rv22d > 0 and cur_rv1d > 2 * cur_rv22d
        st.markdown(status_badge("RV Spike (1d > 2x 22d)", "YES" if is_spike else "NO",
                                 fmt="s", alert=is_spike, alarm=False))

        st.markdown("---")

        # 1B: VRP
        st.markdown("#### 1B. Variance Risk Premium")
        if len(vrp_series.dropna()) > 0:
            fig_vrp = chart_vrp(vrp_series)
            st.plotly_chart(fig_vrp, use_container_width=True)
            vrp_alert = cur_vrp_pctile is not None and cur_vrp_pctile < 25
            vrp_alarm = cur_vrp is not None and cur_vrp < 0
            st.markdown(status_badge("VRP", cur_vrp, fmt=".4f", alert=vrp_alert, alarm=vrp_alarm))
            if cur_vrp_pctile is not None:
                st.markdown(f"Percentile: **{cur_vrp_pctile:.0f}th**")
        else:
            st.info("VRP data unavailable (VIX download may have failed).")

        st.markdown("---")

        # 1C: VIX Term Structure
        st.markdown("#### 1C. VIX Term Structure")
        if len(ts_ratio_series.dropna()) > 0:
            fig_ts = chart_term_structure(ts_ratio_series)
            st.plotly_chart(fig_ts, use_container_width=True)
            ts_alert = cur_ts_ratio is not None and cur_ts_ratio > 0.95
            ts_alarm = cur_ts_ratio is not None and cur_ts_ratio > 1.0
            st.markdown(status_badge("VIX/VIX3M", cur_ts_ratio, fmt=".3f",
                                     alert=ts_alert, alarm=ts_alarm))
            if ts_alarm:
                st.markdown("\u26a0\ufe0f **Backwardation detected** — near-term fear exceeds medium-term.")
        else:
            st.info("VIX term structure data unavailable (^VIX3M may not be available).")

        st.markdown("---")

        # 1D: VVIX
        st.markdown("#### 1D. VVIX")
        if len(vvix_series.dropna()) > 0:
            fig_vvix = chart_vvix(vvix_series)
            st.plotly_chart(fig_vvix, use_container_width=True)
            vvix_alert = cur_vvix is not None and cur_vvix > 100
            vvix_alarm = cur_vvix is not None and cur_vvix > 120
            st.markdown(status_badge("VVIX", cur_vvix, fmt=".1f",
                                     alert=vvix_alert, alarm=vvix_alarm))
        else:
            st.info("VVIX data unavailable. ^VVIX may not be supported by yfinance.")

    # -------------------------------------------------------------------
    # LAYER 2 (right column)
    # -------------------------------------------------------------------
    with col_l2:
        st.markdown("### Layer 2: Equity Market Internals")

        # 2A: Breadth
        st.markdown("#### 2A. Market Breadth (Sector Proxy)")
        if not breadth_df.empty:
            fig_breadth = chart_breadth(breadth_df, spy_close)
            st.plotly_chart(fig_breadth, use_container_width=True)
            breadth_alert = cur_pct200 is not None and cur_pct200 < 60 and spy_near_high
            breadth_alarm = cur_pct200 is not None and cur_pct200 < 40
            st.markdown(status_badge("% > 200d SMA", cur_pct200, fmt=".0f",
                                     alert=breadth_alert, alarm=breadth_alarm))
            if breadth_alert and not breadth_alarm:
                st.markdown("\u26a0\ufe0f Breadth divergence: SPY near highs but breadth weakening.")
        else:
            st.info("Breadth data unavailable.")

        st.markdown("---")

        # 2B: Absorption Ratio
        st.markdown("#### 2B. Absorption Ratio")
        if len(ar_series.dropna()) > 0:
            fig_ar = chart_absorption_ratio(ar_series)
            st.plotly_chart(fig_ar, use_container_width=True)

            if cur_ar is not None:
                st.markdown(status_badge("AR", cur_ar, fmt=".3f",
                                         alert=False, alarm=False))
        else:
            st.info("Absorption ratio unavailable (insufficient sector data).")

        st.markdown("---")

        # 2C: Dispersion + 2x2 Grid
        st.markdown("#### 2C. Return Dispersion & Correlation")
        if len(disp_series.dropna()) > 0:
            fig_disp = chart_dispersion(disp_series)
            st.plotly_chart(fig_disp, use_container_width=True)
            disp_pctile = expanding_percentile(disp_series)
            cur_disp_pctile = _last_valid(disp_pctile)
            st.markdown(status_badge("Dispersion", cur_disp, fmt=".4f",
                                     alert=disp_high, alarm=(disp_high and corr_high)))
            if cur_disp_pctile is not None:
                st.markdown(f"Percentile: **{cur_disp_pctile:.0f}th**")

            st.markdown("**Dispersion x Correlation Grid:**")
            render_2x2_grid(disp_high, corr_high)
        else:
            st.info("Dispersion data unavailable.")

        st.markdown("---")

        # 2D: Hurst
        st.markdown("#### 2D. Hurst Exponent (DFA)")
        if len(hurst_series.dropna()) > 0:
            fig_hurst = chart_hurst(hurst_series, p20=hurst_p20, p80=hurst_p80)
            st.plotly_chart(fig_hurst, use_container_width=True)
            hurst_alert = cur_hurst_pctile is not None and cur_hurst_pctile > 80
            hurst_alarm = cur_hurst_pctile is not None and cur_hurst_pctile > 95
            st.markdown(status_badge("Hurst", cur_hurst, fmt=".3f",
                                     alert=hurst_alert, alarm=hurst_alarm))
            if cur_hurst_pctile is not None:
                st.markdown(f"Percentile: **{cur_hurst_pctile:.0f}th**")
            if hurst_delta_5d is not None:
                delta_sign = "+" if hurst_delta_5d > 0 else ""
                st.markdown(f"5d \u0394H: **{delta_sign}{hurst_delta_5d:.3f}**")
        else:
            st.info("Hurst exponent unavailable (insufficient SPY return data).")

        st.markdown("---")

        # 2E: Days Since Complacency Counters
        st.markdown("#### 2E. Complacency Counters")

        def _calm_style(pctile_val):
            """Return (color, label) for a calm streak percentile."""
            if pctile_val is None or np.isnan(pctile_val):
                return "#888888", "Percentile unavailable"
            if pctile_val > 95:
                return "#CC0000", "Historically rare calm"
            if pctile_val > 80:
                return "#FF8C00", "Extended calm"
            if pctile_val > 50:
                return "#FFD700", "Above-average calm"
            return "#00CC00", "Normal"

        has_5 = cur_days_since_5 is not None and not np.isnan(cur_days_since_5)
        has_vix = cur_days_since_vix is not None and not np.isnan(cur_days_since_vix)

        if has_5 or has_vix:
            rows_html = ""

            if has_5:
                d5 = int(cur_days_since_5)
                p5 = cur_days_since_5_pctile
                c5, l5 = _calm_style(p5)
                p5_str = f"{p5:.0f}" if p5 is not None and not np.isnan(p5) else "N/A"
                rows_html += (
                    f'<div style="margin-bottom: 8px;">'
                    f'<span style="font-size: 28px; font-weight: bold;">Day {d5}</span> '
                    f'since last <strong>5% SPX correction</strong> — '
                    f'<span style="color: {c5}; font-weight: bold;">'
                    f'{p5_str}th pctile</span> ({l5})</div>'
                )

            if has_vix:
                dv = int(cur_days_since_vix)
                pv = cur_days_since_vix_pctile
                cv, lv = _calm_style(pv)
                pv_str = f"{pv:.0f}" if pv is not None and not np.isnan(pv) else "N/A"
                rows_html += (
                    f'<div style="margin-bottom: 8px;">'
                    f'<span style="font-size: 28px; font-weight: bold;">Day {dv}</span> '
                    f'since last <strong>VIX &gt; 28</strong> — '
                    f'<span style="color: {cv}; font-weight: bold;">'
                    f'{pv_str}th pctile</span> ({lv})</div>'
                )

            # Also show 10% drawdown counter (for context)
            has_10 = cur_days_since_10 is not None and not np.isnan(cur_days_since_10)
            if has_10:
                d10 = int(cur_days_since_10)
                p10 = cur_days_since_10_pctile
                c10, l10 = _calm_style(p10)
                p10_str = f"{p10:.0f}" if p10 is not None and not np.isnan(p10) else "N/A"
                rows_html += (
                    f'<div>'
                    f'<span style="font-size: 28px; font-weight: bold;">Day {d10}</span> '
                    f'since last <strong>10% correction</strong> — '
                    f'<span style="color: {c10}; font-weight: bold;">'
                    f'{p10_str}th pctile</span> ({l10})</div>'
                )

            # Border color from worst percentile across the two primary counters
            p5_val = cur_days_since_5_pctile if has_5 and cur_days_since_5_pctile is not None and not np.isnan(cur_days_since_5_pctile) else 0
            pv_val = cur_days_since_vix_pctile if has_vix and cur_days_since_vix_pctile is not None and not np.isnan(cur_days_since_vix_pctile) else 0
            box_color, _ = _calm_style(max(p5_val, pv_val))

            st.markdown(
                f'<div style="background-color: {box_color}20; border-left: 4px solid {box_color}; '
                f'padding: 12px; border-radius: 6px;">{rows_html}</div>',
                unsafe_allow_html=True,
            )

            # Compound complacency badge
            both_high = p5_val > 80 and pv_val > 80
            either_high = p5_val > 80 or pv_val > 80
            st.markdown(status_badge(
                "Complacency",
                "COMPOUND" if both_high else ("ELEVATED" if either_high else "NORMAL"),
                fmt="s", alert=either_high and not both_high, alarm=both_high))

            # Sawtooth charts
            if has_5 and len(days_since_5_series.dropna()) > 0:
                fig_saw5 = chart_days_since_sawtooth(days_since_5_series, "Days Since 5% Drawdown")
                st.plotly_chart(fig_saw5, use_container_width=True)
            if has_vix and len(days_since_vix_series.dropna()) > 0:
                fig_sawv = chart_days_since_sawtooth(days_since_vix_series, "Days Since VIX > 28")
                st.plotly_chart(fig_sawv, use_container_width=True)
        else:
            st.info("No correction in available history — counter unavailable.")

    # ===================================================================
    # LAYER 3: CROSS-ASSET PLUMBING
    # ===================================================================
    st.divider()
    st.subheader("Layer 3: Cross-Asset Plumbing")
    l3_col1, l3_col2, l3_col3, l3_col4 = st.columns(4)

    # 3A: Credit Spreads
    with l3_col1:
        st.markdown("**3A. Credit Spreads**")
        if len(ig_z_series.dropna()) > 0 and len(hy_z_series.dropna()) > 0:
            fig_credit = chart_credit_spreads(ig_z_series, hy_z_series)
            st.plotly_chart(fig_credit, use_container_width=True)
            ig_str = f"{cur_ig_z:+.1f}\u03c3" if cur_ig_z is not None else "N/A"
            hy_str = f"{cur_hy_z:+.1f}\u03c3" if cur_hy_z is not None else "N/A"
            st.markdown(f"IG: **{ig_str}** | HY: **{hy_str}**")
            credit_alert = (cur_ig_z is not None and cur_ig_z > 1.0) or (cur_hy_z is not None and cur_hy_z > 1.0)
            credit_alarm = (cur_ig_z is not None and cur_ig_z > 1.5) and (cur_hy_z is not None and cur_hy_z > 1.5)
            st.markdown(status_badge("Credit", "STRESS" if credit_alarm else ("WIDENING" if credit_alert else "NORMAL"),
                                     fmt="s", alert=credit_alert and not credit_alarm, alarm=credit_alarm))
        else:
            st.caption("Credit spread data unavailable (LQD/HYG/IEF).")

    # 3B: Yield Curve
    with l3_col2:
        st.markdown("**3B. Yield Curve**")
        if len(yc_spread_series.dropna()) > 0:
            fig_yc = chart_yield_curve(yc_spread_series)
            st.plotly_chart(fig_yc, use_container_width=True)
            spread_str = f"{cur_yc_spread:+.2f}%" if cur_yc_spread is not None else "N/A"
            z_str = f"{cur_yc_z:+.1f}\u03c3" if cur_yc_z is not None else "N/A"
            st.markdown(f"Curve: **{spread_str}** | 21d chg z: **{z_str}**")
            yc_inverted = cur_yc_spread is not None and cur_yc_spread < 0
            yc_flat_fast = cur_yc_z is not None and cur_yc_z < -1.5
            yc_accel_disp = cur_yc_z is not None and cur_yc_z < -2.0
            yc_alarm = yc_inverted and yc_accel_disp
            yc_alert = yc_inverted or yc_flat_fast
            st.markdown(status_badge("Yield Curve",
                                     "INVERTED" if yc_inverted else ("FLATTENING" if yc_flat_fast else "NORMAL"),
                                     fmt="s", alert=yc_alert and not yc_alarm, alarm=yc_alarm))
        else:
            st.caption("Yield curve data unavailable (^TNX/^IRX).")

    # 3C: MOVE Index
    with l3_col3:
        st.markdown("**3C. MOVE Index**")
        if len(move_series.dropna()) > 0:
            fig_move = chart_move(move_series)
            st.plotly_chart(fig_move, use_container_width=True)
            if cur_move is not None:
                if cur_move > 150:
                    move_label = "Extreme"
                elif cur_move > 120:
                    move_label = "Elevated"
                else:
                    move_label = "Normal"
                st.markdown(f"MOVE: **{cur_move:.0f}** ({move_label})")
                move_alert = cur_move > 120
                move_alarm = cur_move > 150
                st.markdown(status_badge("MOVE", cur_move, fmt=".0f",
                                         alert=move_alert and not move_alarm, alarm=move_alarm))
            else:
                st.caption("MOVE: current reading unavailable.")
        else:
            st.caption("MOVE Index unavailable via yfinance — consider FRED as alternative data source.")

    # 3D: Dollar Dynamics
    with l3_col4:
        st.markdown("**3D. Dollar Dynamics**")
        if len(dollar_21d_series.dropna()) > 0:
            fig_dollar = chart_dollar(dollar_21d_series)
            st.plotly_chart(fig_dollar, use_container_width=True)
            if cur_dollar_21d is not None:
                if abs(cur_dollar_21d) > 5:
                    dollar_label = "Extreme"
                elif abs(cur_dollar_21d) > 3:
                    dollar_label = "Elevated"
                else:
                    dollar_label = "Normal"
                st.markdown(f"Dollar 21d: **{cur_dollar_21d:+.1f}%** ({dollar_label})")
                dollar_alert = abs(cur_dollar_21d) > 3
                dollar_alarm = abs(cur_dollar_21d) > 5
                st.markdown(status_badge("Dollar", f"{cur_dollar_21d:+.1f}%",
                                         fmt="s", alert=dollar_alert and not dollar_alarm, alarm=dollar_alarm))
            else:
                st.caption("Dollar momentum: current reading unavailable.")
        else:
            st.caption("Dollar data unavailable (UUP).")

    # ===================================================================
    # LAYER 4: TAIL RISK & COST OF PROTECTION
    # ===================================================================
    layer4_expanded = (active_count >= 2)

    with st.expander("Layer 4: Tail Risk & Cost of Protection", expanded=layer4_expanded):
        l4_col1, l4_col2, l4_col3 = st.columns([1, 1, 1])

        # 4A: SKEW Index
        with l4_col1:
            st.markdown("**4A. SKEW Index**")
            if len(skew_series.dropna()) > 0:
                fig_skew = go.Figure()
                fig_skew.add_trace(go.Scatter(
                    x=skew_series.index, y=skew_series, name="SKEW",
                    line=dict(width=1.5, color="#0066CC"),
                ))
                fig_skew.add_hline(y=120, line_dash="dot", line_color="#FFD700", line_width=1,
                                   annotation_text="120", annotation_position="right")
                fig_skew.add_hline(y=140, line_dash="dot", line_color="#CC0000", line_width=1,
                                   annotation_text="140", annotation_position="right")
                layout_skew = _base_layout("SKEW Index")
                layout_skew["height"] = CHART_HEIGHT
                fig_skew.update_layout(**layout_skew)
                st.plotly_chart(fig_skew, use_container_width=True)

                if cur_skew is not None:
                    skew_pctile_series = expanding_percentile(skew_series, min_periods=252)
                    cur_skew_pctile = _last_valid(skew_pctile_series)
                    pctile_str = f" ({cur_skew_pctile:.0f}th pctile)" if cur_skew_pctile is not None else ""
                    st.markdown(f"SKEW: **{cur_skew:.0f}**{pctile_str}")

                # Disorderly stress detection
                if vix_available and len(skew_series) > 5:
                    vix_for_skew = vol_data["^VIX"]["Close"].reindex(skew_series.index)
                    skew_falling = skew_series.diff(5) < -3
                    vix_rising = vix_for_skew.diff(5) > 3
                    disorderly = skew_falling & vix_rising
                    if len(disorderly.dropna()) > 0 and disorderly.iloc[-1]:
                        st.markdown(
                            '<div style="background:#CC000020; border-left:3px solid #CC0000; '
                            'padding:8px; border-radius:4px;">'
                            '\u26a0\ufe0f <strong>SKEW falling while VIX rising</strong> — '
                            'disorderly stress pattern detected.</div>',
                            unsafe_allow_html=True)
            else:
                st.caption("SKEW Index unavailable via yfinance.")

        # 4B: Cost of Protection Proxy
        with l4_col2:
            st.markdown("**4B. Protection Cost**")
            if len(prot_pctile_series.dropna()) > 0 and cur_prot_pctile is not None:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=cur_prot_pctile,
                    title={'text': "Protection Cost Percentile"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 20], 'color': '#00CC00'},
                            {'range': [20, 60], 'color': '#FFD700'},
                            {'range': [60, 85], 'color': '#FF8C00'},
                            {'range': [85, 100], 'color': '#CC0000'},
                        ],
                    }
                ))
                fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Time series below gauge
                fig_prot = go.Figure()
                fig_prot.add_trace(go.Scatter(
                    x=prot_cost_series.index, y=prot_cost_series, name="Protection Cost",
                    line=dict(width=1.5, color="#0066CC"),
                ))
                prot_layout = _base_layout("Protection Cost Proxy (time series)")
                prot_layout["height"] = CHART_HEIGHT_SMALL
                fig_prot.update_layout(**prot_layout)
                st.plotly_chart(fig_prot, use_container_width=True)
            else:
                st.caption("Protection cost data unavailable (requires VIX3M).")

        # 4C: Hedge Recommendation
        with l4_col3:
            st.markdown("**4C. Hedge Recommendation**")
            prot_pctile_for_rec = cur_prot_pctile if cur_prot_pctile is not None else 50.0
            rec, detail, rec_color = generate_hedge_recommendation(regime, prot_pctile_for_rec)

            st.markdown(f"""
            <div style="background-color: {rec_color}20; border-left: 4px solid {rec_color};
                        padding: 16px; border-radius: 8px;">
                <h4 style="margin: 0; color: {rec_color};">{rec}</h4>
                <p style="margin: 8px 0 0 0;">{detail}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.caption("Regime: **{}** | Protection pctile: **{:.0f}**".format(
                regime, prot_pctile_for_rec))

    # ===================================================================
    # FUTURE WORK
    # ===================================================================
    # TODO: Full Bayesian composite to replace the simple point system
    # TODO: Full S&P 500 breadth (all ~500 constituents) instead of sector ETF proxy
    # TODO: Historical regime validation / backtesting


main()
