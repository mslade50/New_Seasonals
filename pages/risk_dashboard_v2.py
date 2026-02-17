"""
Risk Dashboard V2 — Executive Summary + Absorption Ratio
=========================================================
Standalone market risk monitor.
Signal-based three-question framework with fragility dial.
Absorption ratio chart for structural regime context.

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

# Cross-asset tickers (credit + MOVE only — signals need these)
CROSS_ASSET_TICKERS = ['LQD', 'HYG', 'IEF', '^MOVE']

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
    """Download OHLC data for cross-asset tickers."""
    return _download_ticker_group(CROSS_ASSET_TICKERS, start_date)


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
# LAYER 1 COMPUTATIONS (feed signals — no charts)
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


# ---------------------------------------------------------------------------
# LAYER 2 COMPUTATIONS (feed signals + AR chart)
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
# LAYER 3 COMPUTATIONS (feed signals — no charts)
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

    # Validate minimum data
    if "SPY" not in vol_data:
        st.error("Could not download SPY data. Check your internet connection.")
        st.stop()

    spy_df = vol_data["SPY"]
    spy_close = spy_df["Close"]

    # -------------------------------------------------------------------
    # COMPUTATIONS FOR SIGNAL FRAMEWORK
    # -------------------------------------------------------------------

    # Layer 1: Realized vol + VRP (feed signals, no charts)
    rv_22d = yang_zhang_vol(spy_df, 22)
    rv_22d_pctile = expanding_percentile(rv_22d)

    vix_available = "^VIX" in vol_data
    vrp_series = pd.Series(dtype=float)
    vrp_pctile_series = pd.Series(dtype=float)
    if vix_available:
        vix_close = vol_data["^VIX"]["Close"]
        common_idx = rv_22d.dropna().index.intersection(vix_close.dropna().index)
        if len(common_idx) > 0:
            vrp_series = compute_vrp(vix_close.reindex(common_idx), rv_22d.reindex(common_idx))
            vrp_pctile_series = expanding_percentile(vrp_series)

    vvix_available = "^VVIX" in vol_data

    # Layer 2: Breadth, AR, complacency (AR gets a chart)
    sector_closes = pd.DataFrame(
        {t: d["Close"] for t, d in sector_data.items() if "Close" in d.columns}
    )
    sector_returns = sector_closes.pct_change().dropna(how="all")

    active_sectors = sector_returns.columns.tolist()
    if len(active_sectors) < 8:
        st.warning(f"Only {len(active_sectors)} sector ETFs available (need 8+). Some metrics may be degraded.")

    breadth_df = compute_breadth_sector_proxy(sector_data)

    ar_series = pd.Series(dtype=float)
    if len(active_sectors) >= 5:
        with st.spinner("Computing absorption ratio (PCA)..."):
            ar_series = compute_absorption_ratio(sector_returns, window=63)

    days_since_5_series = compute_days_since(spy_close, threshold_pct=0.05)
    days_since_5_pctile_series = expanding_percentile(days_since_5_series, min_periods=252)

    days_since_vix_series = pd.Series(dtype=float)
    days_since_vix_pctile_series = pd.Series(dtype=float)
    if vix_available:
        vix_c_for_spike = vol_data["^VIX"]["Close"].dropna()
        days_since_vix_series = compute_days_since_vix_spike(vix_c_for_spike, threshold=28.0)
        days_since_vix_pctile_series = expanding_percentile(days_since_vix_series, min_periods=252)

    # Layer 3: Credit spreads + MOVE (feed signals, no charts)
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

    move_series = pd.Series(dtype=float)
    if '^MOVE' in cross_asset_data:
        try:
            move_series = cross_asset_data['^MOVE']['Close'].dropna()
        except Exception:
            pass

    # -------------------------------------------------------------------
    # COLLECT CURRENT READINGS
    # -------------------------------------------------------------------
    def _last_valid(s):
        if s is None or len(s) == 0:
            return None
        v = s.dropna()
        return float(v.iloc[-1]) if len(v) > 0 else None

    cur_rv22d_pctile = _last_valid(rv_22d_pctile)
    cur_vrp = _last_valid(vrp_series)
    cur_vrp_pctile = _last_valid(vrp_pctile_series)
    cur_vvix = _last_valid(vol_data["^VVIX"]["Close"]) if vvix_available else None
    cur_pct200 = _last_valid(breadth_df["pct_above_200"]) if not breadth_df.empty else None

    # SPY near 52-week high?
    spy_52w_high = spy_close.rolling(252).max()
    spy_near_high = False
    if len(spy_52w_high.dropna()) > 0:
        latest_price = float(spy_close.iloc[-1])
        latest_high = float(spy_52w_high.iloc[-1])
        if latest_high > 0:
            spy_near_high = latest_price >= latest_high * 0.95

    cur_days_since_5 = _last_valid(days_since_5_series)
    cur_days_since_5_pctile = _last_valid(days_since_5_pctile_series)
    cur_days_since_vix = _last_valid(days_since_vix_series)
    cur_days_since_vix_pctile = _last_valid(days_since_vix_pctile_series)
    cur_hy_z = _last_valid(hy_z_series)
    cur_move = _last_valid(move_series)

    # Additional percentiles for signal framework
    ar_pctile_series = expanding_percentile(ar_series) if len(ar_series.dropna()) > 0 else pd.Series(dtype=float)
    cur_ar_pctile = _last_valid(ar_pctile_series)

    move_pctile_series = expanding_percentile(move_series) if len(move_series.dropna()) > 0 else pd.Series(dtype=float)
    cur_move_pctile = _last_valid(move_pctile_series)

    cur_vix = _last_valid(vol_data["^VIX"]["Close"]) if "^VIX" in vol_data else None
    vix_pctile_series = expanding_percentile(vol_data["^VIX"]["Close"]) if "^VIX" in vol_data else pd.Series(dtype=float)
    cur_vix_pctile = _last_valid(vix_pctile_series)

    spy_21d_return = float(spy_close.iloc[-1] / spy_close.iloc[-22] - 1) if len(spy_close) >= 22 else None

    cur_vvix_vix_ratio_pctile = None
    if vvix_available and vix_available:
        vvix_c_ratio = vol_data["^VVIX"]["Close"].dropna()
        vix_c_ratio = vol_data["^VIX"]["Close"].dropna()
        common_ratio = vvix_c_ratio.index.intersection(vix_c_ratio.index)
        vvix_vix_ratio_series = vvix_c_ratio.reindex(common_ratio) / vix_c_ratio.reindex(common_ratio)
        vvix_vix_ratio_series = vvix_vix_ratio_series.replace([np.inf, -np.inf], np.nan).dropna()
        vvix_vix_ratio_pctile_series = expanding_percentile(vvix_vix_ratio_series) if len(vvix_vix_ratio_series) > 0 else pd.Series(dtype=float)
        cur_vvix_vix_ratio_pctile = _last_valid(vvix_vix_ratio_pctile_series)

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

    # ===================================================================
    # ABSORPTION RATIO CHART
    # ===================================================================
    st.divider()
    st.subheader("Absorption Ratio")

    if len(ar_series.dropna()) > 0:
        fig_ar = chart_absorption_ratio(ar_series)
        st.plotly_chart(fig_ar, use_container_width=True)

        cur_ar = _last_valid(ar_series)
        if cur_ar is not None:
            st.markdown(status_badge("AR", cur_ar, fmt=".3f", alert=False, alarm=False))
    else:
        st.info("Absorption ratio unavailable (insufficient sector data).")


main()
