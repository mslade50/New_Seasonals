"""
Risk Dashboard V2 — Phase 1 (Layers 0, 1, 2)
=============================================
Standalone market risk monitor.
Layer 0: Composite regime verdict (rules-based point system)
Layer 1: Volatility state (HAR-RV, VRP, VIX term structure, VVIX)
Layer 2: Equity market internals (breadth, absorption ratio, dispersion, Hurst)

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

    # 2E: Days Since Correction — alert: > 80th pctile; alarm: > 95th pctile
    calm_pctile = metrics.get("days_since_pctile")
    if calm_pctile is not None:
        if calm_pctile > 95:
            points["Calm Streak (>95th pctile)"] = 2
        elif calm_pctile > 80:
            points["Calm Streak (>80th pctile)"] = 1

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


def generate_summary(regime: str, points: dict) -> str:
    if regime == "Normal":
        return "All monitored risk metrics are within normal ranges. Full position sizing."
    elif regime == "Caution":
        triggers = ", ".join(points.keys())
        return f"Mild caution warranted. Flagged: {triggers}. Consider modest size reduction."
    elif regime == "Stress":
        triggers = ", ".join(points.keys())
        return f"Elevated stress detected across multiple dimensions: {triggers}. Reduce exposure."
    else:
        triggers = ", ".join(points.keys())
        return f"Crisis-level readings. {triggers}. Minimize directional exposure and consider hedges."


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

    # --- Score ---
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
        "days_since_pctile": cur_days_since_5_pctile,
    }

    total_pts, pt_breakdown = score_alerts(metrics_for_scoring)
    regime = classify_regime(total_pts)
    multiplier = REGIME_MULTIPLIER[regime]
    summary = generate_summary(regime, pt_breakdown)
    color = REGIME_COLORS[regime]
    emoji = REGIME_EMOJI[regime]

    # ===================================================================
    # LAYER 0: THE VERDICT
    # ===================================================================
    st.markdown(
        f"""
        <div style="background-color: {color}20; border-left: 4px solid {color};
                    padding: 16px; border-radius: 8px; margin-bottom: 16px;">
            <h2 style="margin: 0; color: {color};">{emoji} {regime.upper()}</h2>
            <p style="font-size: 24px; margin: 8px 0;">Sizing: <strong>{multiplier}x</strong></p>
            <p style="margin: 0;">{summary}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Point breakdown expander
    with st.expander(f"Score breakdown: {total_pts} points"):
        if pt_breakdown:
            for metric_name, pts in pt_breakdown.items():
                st.markdown(f"- **{metric_name}**: +{pts}")
        else:
            st.markdown("No alerts triggered.")
        st.caption("0 pts = Normal | 1-2 = Caution | 3-4 = Stress | 5+ = Crisis")

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

        # 2E: Days Since Correction
        st.markdown("#### 2E. Days Since Correction")

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
        has_10 = cur_days_since_10 is not None and not np.isnan(cur_days_since_10)

        if has_5 or has_10:
            # Build HTML rows for both thresholds in one box
            rows_html = ""

            if has_5:
                d5 = int(cur_days_since_5)
                p5 = cur_days_since_5_pctile
                c5, l5 = _calm_style(p5)
                p5_str = f"{p5:.0f}" if p5 is not None and not np.isnan(p5) else "N/A"
                rows_html += (
                    f'<div style="margin-bottom: 8px;">'
                    f'<span style="font-size: 28px; font-weight: bold;">Day {d5}</span> '
                    f'since last <strong>5%</strong> correction — '
                    f'<span style="color: {c5}; font-weight: bold;">'
                    f'{p5_str}th pctile</span> ({l5})</div>'
                )

            if has_10:
                d10 = int(cur_days_since_10)
                p10 = cur_days_since_10_pctile
                c10, l10 = _calm_style(p10)
                p10_str = f"{p10:.0f}" if p10 is not None and not np.isnan(p10) else "N/A"
                rows_html += (
                    f'<div>'
                    f'<span style="font-size: 28px; font-weight: bold;">Day {d10}</span> '
                    f'since last <strong>10%</strong> correction — '
                    f'<span style="color: {c10}; font-weight: bold;">'
                    f'{p10_str}th pctile</span> ({l10})</div>'
                )

            # Use the worse (higher) percentile for the box border color
            p5_val = cur_days_since_5_pctile if has_5 and cur_days_since_5_pctile is not None and not np.isnan(cur_days_since_5_pctile) else 0
            p10_val = cur_days_since_10_pctile if has_10 and cur_days_since_10_pctile is not None and not np.isnan(cur_days_since_10_pctile) else 0
            box_color, _ = _calm_style(max(p5_val, p10_val))

            st.markdown(
                f'<div style="background-color: {box_color}20; border-left: 4px solid {box_color}; '
                f'padding: 12px; border-radius: 6px;">{rows_html}</div>',
                unsafe_allow_html=True,
            )

            # Alert badge based on 5% streak (primary signal)
            if has_5:
                calm_alert = p5_val > 80
                calm_alarm = p5_val > 95
                st.markdown(status_badge("Calm Streak (5%)", f"{p5_val:.0f}th pctile",
                                         fmt="s", alert=calm_alert, alarm=calm_alarm))
        else:
            st.info("No correction in available history — counter unavailable.")

    # ===================================================================
    # PHASE 2 PLACEHOLDERS
    # ===================================================================
    # TODO: Layer 3 — Credit (LQD/HYG spread), Yield Curve (^TNX/^IRX), MOVE, Dollar (UUP)
    # TODO: Layer 4 — SKEW, Protection Cost Proxy, Hedge Recommendation Engine
    # TODO: Full Bayesian composite to replace the simple point system
    # TODO: Full S&P 500 breadth (all ~500 constituents) instead of sector ETF proxy
    # TODO: Historical regime validation / backtesting


main()
