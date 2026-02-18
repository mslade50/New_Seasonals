import streamlit as st

st.set_page_config(page_title="Signal Backtester 2", page_icon="\U0001f52c", layout="wide")

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import sys
import os
import plotly.graph_objects as go
from scipy import stats

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

DATA_DIR = os.path.join(parent_dir, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_SPY_OHLC = os.path.join(DATA_DIR, "rd2_spy_ohlc.parquet")
CACHE_CLOSES = os.path.join(DATA_DIR, "rd2_closes.parquet")


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    """Load SPY OHLC and closes from parquet cache, or download fresh."""
    if os.path.exists(CACHE_SPY_OHLC) and os.path.exists(CACHE_CLOSES):
        spy_df = pd.read_parquet(CACHE_SPY_OHLC)
        closes = pd.read_parquet(CACHE_CLOSES)
        return spy_df, closes

    st.warning("No cached data found. Downloading fresh (this may take a minute)...")
    tickers = ["SPY", "QQQ", "IWM", "^VIX", "^VIX3M",
               "XLB", "XLC", "XLE", "XLF", "XLI", "XLK",
               "XLP", "XLRE", "XLU", "XLV", "XLY"]
    raw = yf.download(tickers, start="2010-01-01", auto_adjust=True, threads=True)

    if isinstance(raw.columns, pd.MultiIndex):
        spy_df = raw.xs("SPY", level="Ticker", axis=1).copy()
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.get_level_values(0)
        spy_df.columns = [c.capitalize() for c in spy_df.columns]

        lvl0 = raw.columns.get_level_values(0).unique().tolist()
        close_key = "Close" if "Close" in lvl0 else ("close" if "close" in lvl0 else lvl0[0])
        closes = raw[close_key].copy()
        closes.columns = [str(c) for c in closes.columns]
    else:
        spy_df = raw.copy()
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.get_level_values(0)
        spy_df.columns = [c.capitalize() for c in spy_df.columns]
        closes = spy_df[["Close"]].copy()
        closes.columns = ["SPY"]

    if spy_df.index.tz is not None:
        spy_df.index = spy_df.index.tz_localize(None)
    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)

    return spy_df, closes


@st.cache_data(ttl=3600, show_spinner="Downloading IWM and QQQ...")
def download_extra_tickers():
    """Download IWM and QQQ if not in the cache."""
    try:
        raw = yf.download(["IWM", "QQQ"], start="2010-01-01",
                          auto_adjust=True, threads=True)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = raw.columns.get_level_values(0).unique().tolist()
            close_key = "Close" if "Close" in lvl0 else ("close" if "close" in lvl0 else lvl0[0])
            extra_close = raw[close_key].copy()
            extra_close.columns = [str(c) for c in extra_close.columns]
        else:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.columns = [c.capitalize() for c in raw.columns]
            extra_close = raw[["Close"]].copy()
            extra_close.columns = ["IWM"]

        if extra_close.index.tz is not None:
            extra_close.index = extra_close.index.tz_localize(None)
        return extra_close
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SHARED EVENT STUDY ENGINE (copied from signal_backtester.py)
# ---------------------------------------------------------------------------
def run_event_study(
    signal_series: pd.Series,
    price_series: pd.Series,
    forward_windows: list = None,
    signal_name: str = "Signal",
) -> dict:
    """Run a standard event study."""
    if forward_windows is None:
        forward_windows = [5, 10, 21, 42, 63]

    common = signal_series.dropna().index.intersection(price_series.dropna().index)
    signal = signal_series.reindex(common)
    price = price_series.reindex(common)
    signal_dates = signal[signal == True].index

    results = {}
    fwd_rets_signal = {}
    fwd_rets_all = {}

    for w in forward_windows:
        fwd = price.shift(-w) / price - 1
        fwd = fwd.dropna()
        signal_fwd = fwd.reindex(signal_dates).dropna()

        fwd_rets_signal[w] = signal_fwd
        fwd_rets_all[w] = fwd

        if len(signal_fwd) > 2:
            _, p_val = stats.ttest_ind(signal_fwd.values, fwd.values, equal_var=False)
        else:
            p_val = np.nan

        results[f"{w}d"] = {
            "Signal Mean": signal_fwd.mean() if len(signal_fwd) > 0 else np.nan,
            "Signal Median": signal_fwd.median() if len(signal_fwd) > 0 else np.nan,
            "Unconditional Mean": fwd.mean(),
            "Unconditional Median": fwd.median(),
            "Difference (Mean)": (signal_fwd.mean() - fwd.mean()) if len(signal_fwd) > 0 else np.nan,
            "Difference (Median)": (signal_fwd.median() - fwd.median()) if len(signal_fwd) > 0 else np.nan,
            "Signal Worst": signal_fwd.min() if len(signal_fwd) > 0 else np.nan,
            "Signal Best": signal_fwd.max() if len(signal_fwd) > 0 else np.nan,
            "Hit Rate (neg fwd ret)": (signal_fwd < 0).mean() if len(signal_fwd) > 0 else np.nan,
            "p-value": p_val,
        }

    results_df = pd.DataFrame(results).T

    return {
        "signal_dates": signal_dates,
        "n_activations": len(signal_dates),
        "results": results_df,
        "forward_returns": fwd_rets_signal,
        "unconditional_returns": fwd_rets_all,
    }


def render_event_study(study: dict, signal_name: str, signal_series: pd.Series,
                       price_series: pd.Series):
    """Render the full event study output for one signal tab."""
    n_total = len(signal_series.dropna())
    st.markdown(
        f"**Activation count:** {study['n_activations']} days "
        f"({study['n_activations'] / n_total * 100:.1f}% of trading days)"
        if n_total > 0 else "**No data.**"
    )

    if study["n_activations"] < 5:
        st.warning("Too few activations for meaningful analysis. Try relaxing the thresholds.")
        return

    # --- Results table ---
    results_display = study["results"].copy()
    for col in results_display.index:
        for metric in results_display.columns:
            val = results_display.loc[col, metric]
            if metric == "p-value":
                results_display.loc[col, metric] = f"{val:.3f}" if not np.isnan(val) else "N/A"
            elif "Hit Rate" in metric:
                results_display.loc[col, metric] = f"{val:.1%}" if not np.isnan(val) else "N/A"
            else:
                results_display.loc[col, metric] = f"{val:+.2%}" if not np.isnan(val) else "N/A"

    st.dataframe(results_display, use_container_width=True)

    # --- Minimum bar check ---
    med_21d = study["results"].loc["21d", "Difference (Median)"]
    if not np.isnan(med_21d):
        if med_21d < -0.005:
            st.success(
                f"Passes minimum bar: median 21d forward return is {med_21d:+.2%} "
                f"vs unconditional (threshold: -0.50%)"
            )
        else:
            st.warning(
                f"Does NOT pass minimum bar: median 21d forward return is {med_21d:+.2%} "
                f"vs unconditional (threshold: -0.50%)"
            )

    # --- SPY chart with signal onset vertical lines ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_series.index, y=price_series,
        name="SPY", line=dict(width=1.5, color="rgba(150,150,150,0.8)"),
    ))

    signal_dates = study["signal_dates"]
    onset_dates = []
    last_onset = None
    for dt in signal_dates:
        if last_onset is None or (dt - last_onset).days > 10:
            onset_dates.append(dt)
            last_onset = dt

    for dt in onset_dates:
        fig.add_vline(x=dt, line_color="rgba(204,0,0,0.6)", line_width=1, line_dash="solid")

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        xaxis=dict(showgrid=False),
        yaxis=dict(title="SPY"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text=f"{signal_name}: Signal Onsets on SPY ({len(onset_dates)} events)", font=dict(size=13)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Forward return distributions (21d) ---
    if 21 in study["forward_returns"] and len(study["forward_returns"][21]) > 5:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=study["unconditional_returns"][21],
            name="All Days", opacity=0.5,
            marker_color="rgba(100,100,100,0.5)",
            histnorm="probability density",
            nbinsx=50,
        ))
        fig_hist.add_trace(go.Histogram(
            x=study["forward_returns"][21],
            name="Signal Days", opacity=0.7,
            marker_color="rgba(204,0,0,0.7)",
            histnorm="probability density",
            nbinsx=30,
        ))
        fig_hist.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            barmode="overlay",
            title=dict(text="Forward 21d Return Distribution (density)", font=dict(size=13)),
            xaxis=dict(tickformat=".1%"),
            yaxis=dict(title="Density"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ---------------------------------------------------------------------------
# DUAL Y-AXIS LAYOUT HELPER
# ---------------------------------------------------------------------------
def _dual_y_layout(title: str, y1_title: str, y2_title: str) -> dict:
    return dict(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        yaxis=dict(title=y1_title),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title=y2_title),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text=title, font=dict(size=13)),
    )


# ---------------------------------------------------------------------------
# SIGNAL 1: CLOSE LOCATION VALUE
# ---------------------------------------------------------------------------
def compute_close_location(spy_df: pd.DataFrame,
                           window: int = 21,
                           threshold: float = 0.42,
                           require_uptrend: bool = True,
                           require_near_high: bool = True,
                           near_high_pct: float = 5) -> tuple:
    """
    Close location signal.

    Returns:
        signal_series, close_loc (raw daily 0-1), rolling_close_loc (smoothed)
    """
    high = spy_df['High']
    low = spy_df['Low']
    close = spy_df['Close']

    daily_range = high - low
    close_loc = ((close - low) / daily_range).replace([np.inf, -np.inf], np.nan)
    close_loc = close_loc.clip(0, 1)

    rolling_close_loc = close_loc.rolling(window).mean()

    signal = rolling_close_loc < threshold

    if require_uptrend:
        sma_50 = close.rolling(50).mean()
        signal = signal & (close > sma_50)

    if require_near_high:
        high_52w = close.rolling(252, min_periods=60).max()
        near_high = close >= high_52w * (1 - near_high_pct / 100)
        signal = signal & near_high

    return signal, close_loc, rolling_close_loc


# ---------------------------------------------------------------------------
# SIGNAL 2: EQUITY PUT/CALL RATIO
# ---------------------------------------------------------------------------
def load_putcall_data() -> pd.Series | None:
    """Load equity put/call ratio from user-uploaded CSV."""
    uploaded = st.file_uploader(
        "Upload equity put/call ratio CSV (columns: date, ratio)",
        type=["csv"],
        key="pcr_upload",
        help="Download from CBOE or your data provider. Need columns: date and equity put/call ratio.",
    )

    if uploaded is None:
        st.info(
            "**Data required:** Equity put/call ratio is not available via yfinance.\n\n"
            "**Where to get it:**\n"
            "- CBOE: https://www.cboe.com/us/options/market_statistics/\n"
            "- Look for 'Equity Put/Call Ratio' (NOT total, NOT index-only)\n"
            "- Download as CSV, then upload here.\n\n"
            "The CSV should have a date column and a ratio column. "
            "Column names will be auto-detected."
        )
        return None

    try:
        df = pd.read_csv(uploaded, parse_dates=True)

        # Auto-detect date column
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or df[col].dtype == 'datetime64[ns]':
                date_col = col
                break
        if date_col is None:
            try:
                df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
                date_col = df.columns[0]
            except Exception:
                st.error("Could not identify date column. Ensure one column contains dates.")
                return None

        # Auto-detect ratio column
        ratio_col = None
        for col in df.columns:
            if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                if any(kw in col.lower() for kw in ['put', 'call', 'ratio', 'pc', 'p/c']):
                    ratio_col = col
                    break
        if ratio_col is None:
            for col in df.columns:
                if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                    ratio_col = col
                    break

        if ratio_col is None:
            st.error("Could not identify ratio column. Ensure one numeric column contains the put/call ratio.")
            return None

        df[date_col] = pd.to_datetime(df[date_col])
        series = df.set_index(date_col)[ratio_col].sort_index().dropna()

        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)

        st.success(
            f"Loaded {len(series)} days of put/call data "
            f"({series.index[0].strftime('%Y-%m-%d')} to {series.index[-1].strftime('%Y-%m-%d')})"
        )
        return series

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None


def compute_putcall_signal(pcr_series: pd.Series,
                           spy_close: pd.Series,
                           zscore_lookback: int = 63,
                           zscore_threshold: float = -1.5,
                           smoothing: int = 5,
                           require_uptrend: bool = True) -> tuple:
    """
    Equity put/call ratio signal.
    Low put/call = lots of calls = speculative excess = fragility.
    Signal fires when z-score is BELOW threshold.

    Returns:
        signal_series, pcr_smoothed, pcr_zscore
    """
    if smoothing > 1:
        pcr_smoothed = pcr_series.rolling(smoothing).mean()
    else:
        pcr_smoothed = pcr_series.copy()

    pcr_mean = pcr_smoothed.rolling(zscore_lookback).mean()
    pcr_std = pcr_smoothed.rolling(zscore_lookback).std()
    pcr_zscore = (pcr_smoothed - pcr_mean) / pcr_std
    pcr_zscore = pcr_zscore.replace([np.inf, -np.inf], np.nan)

    signal = pcr_zscore < zscore_threshold

    if require_uptrend:
        spy_aligned = spy_close.reindex(pcr_series.index, method='ffill')
        sma_50 = spy_aligned.rolling(50).mean()
        signal = signal & (spy_aligned > sma_50)

    return signal, pcr_smoothed, pcr_zscore


# ---------------------------------------------------------------------------
# SIGNAL 3: SMALL CAP RELATIVE STRENGTH
# ---------------------------------------------------------------------------
def compute_smallcap_signal(closes: pd.DataFrame,
                            spy_close: pd.Series,
                            change_lookback: int = 42,
                            zscore_lookback: int = 252,
                            zscore_threshold: float = -1.0,
                            check_qqq: bool = False) -> tuple:
    """
    Small cap relative strength signal.

    Returns:
        signal_series, iwm_spy_ratio, ratio_change, ratio_change_z
    """
    if 'IWM' not in closes.columns:
        empty = pd.Series(dtype=float, index=closes.index)
        return pd.Series(False, index=closes.index), empty, empty, empty

    iwm = closes['IWM'].dropna()
    spy = spy_close.reindex(iwm.index)
    common = iwm.dropna().index.intersection(spy.dropna().index)

    iwm = iwm.reindex(common)
    spy = spy.reindex(common)

    iwm_spy_ratio = iwm / spy

    ratio_change = iwm_spy_ratio.pct_change(change_lookback)

    rc_mean = ratio_change.rolling(zscore_lookback).mean()
    rc_std = ratio_change.rolling(zscore_lookback).std()
    ratio_change_z = (ratio_change - rc_mean) / rc_std
    ratio_change_z = ratio_change_z.replace([np.inf, -np.inf], np.nan)

    signal = ratio_change_z < zscore_threshold

    if check_qqq and 'QQQ' in closes.columns:
        qqq = closes['QQQ'].dropna().reindex(common)
        qqq_spy_ratio = qqq / spy
        qqq_change = qqq_spy_ratio.pct_change(change_lookback)
        qqq_mean = qqq_change.rolling(zscore_lookback).mean()
        qqq_std = qqq_change.rolling(zscore_lookback).std()
        qqq_z = (qqq_change - qqq_mean) / qqq_std
        signal = signal & (qqq_z < zscore_threshold)

    return signal, iwm_spy_ratio, ratio_change, ratio_change_z


# ---------------------------------------------------------------------------
# SIGNAL 4: OVERNIGHT VS INTRADAY RETURNS
# ---------------------------------------------------------------------------
def compute_overnight_intraday(spy_df: pd.DataFrame,
                               cum_window: int = 21,
                               require_overnight_pos: bool = True,
                               intraday_threshold: float = 0.0,
                               divergence_min: float = 0.02,
                               require_near_high: bool = True,
                               near_high_pct: float = 5) -> tuple:
    """
    Overnight vs intraday return decomposition signal.

    Returns:
        signal_series, overnight_ret, intraday_ret, cum_overnight, cum_intraday
    """
    close = spy_df['Close']
    open_ = spy_df['Open']

    overnight_ret = open_ / close.shift(1) - 1
    intraday_ret = close / open_ - 1

    cum_overnight = overnight_ret.rolling(cum_window).sum()
    cum_intraday = intraday_ret.rolling(cum_window).sum()

    divergence = cum_overnight - cum_intraday

    signal = (cum_intraday < intraday_threshold) & (divergence > divergence_min)

    if require_overnight_pos:
        signal = signal & (cum_overnight > 0)

    if require_near_high:
        high_52w = close.rolling(252, min_periods=60).max()
        near_high = close >= high_52w * (1 - near_high_pct / 100)
        signal = signal & near_high

    return signal, overnight_ret, intraday_ret, cum_overnight, cum_intraday


# ===========================================================================
# MAIN PAGE
# ===========================================================================
st.title("\U0001f52c Signal Event Study Backtester \u2014 Round 2")
st.caption(
    "Four additional candidate signals for the risk dashboard. "
    "These are intentionally orthogonal to Round 1 \u2014 each taps a different "
    "information channel: intraday price action, options positioning, "
    "relative equity performance, and session decomposition."
)

spy_df, closes = load_data()
if spy_df is None or spy_df.empty:
    st.error("No data available. Run the Risk Dashboard V2 data refresh first to populate the parquet cache.")
    st.stop()

spy_close = spy_df["Close"]

# Ensure IWM/QQQ are in closes
if 'IWM' not in closes.columns or 'QQQ' not in closes.columns:
    extra = download_extra_tickers()
    if extra is not None:
        for col in extra.columns:
            if col not in closes.columns:
                closes[col] = extra[col]

tab1, tab2, tab3, tab4 = st.tabs([
    "\U0001f4cd Close Location",
    "\U0001f4de Equity Put/Call Ratio",
    "\U0001f426 Small Cap Relative Strength",
    "\U0001f319 Overnight vs Intraday",
])

# Store signal series for correlation matrix at the end
all_signals = {}

# ========================== TAB 1: CLOSE LOCATION ==========================
with tab1:
    st.markdown("### Close Location Value")
    st.markdown(
        "> Measures where in the daily range SPY closes, averaged over a rolling "
        "window. `close_location = (close - low) / (high - low)` where 1.0 = "
        "closed at the high and 0.0 = closed at the low. When the rolling average "
        "drifts below 0.5 while price is rising or flat, it means sellers are "
        "consistently winning the intraday battle even though the daily close-to-close "
        "chart looks fine. The market opens strong and fades \u2014 a distribution "
        "pattern invisible to daily bars. This is completely independent of "
        "volume-based distribution signals because it fires on every day regardless "
        "of volume."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        cl_window = st.slider("Rolling window (trading days)", 10, 63, 21, key="cl_win")
    with c2:
        cl_threshold = st.slider("Close location threshold", 0.30, 0.50, 0.42, 0.01, key="cl_thresh")
    with c3:
        cl_uptrend = st.checkbox("Require uptrend (SPY > 50d SMA)", value=True, key="cl_uptrend")

    c4, c5 = st.columns(2)
    with c4:
        cl_near_high = st.checkbox("Require near 52w high", value=True, key="cl_near_high")
    with c5:
        cl_near_pct = st.slider("Near-high threshold (%)", 2, 10, 5, key="cl_near_pct")

    signal_cl, close_loc, rolling_cl = compute_close_location(
        spy_df, cl_window, cl_threshold, cl_uptrend, cl_near_high, cl_near_pct,
    )
    all_signals["Close Location"] = signal_cl

    # Current reading
    cur_cl = float(rolling_cl.dropna().iloc[-1]) if len(rolling_cl.dropna()) > 0 else None
    if cur_cl is not None:
        if cur_cl > 0.50:
            cl_icon = "\U0001f7e2"
            cl_desc = "buyers in control"
        elif cur_cl > cl_threshold:
            cl_icon = "\U0001f7e1"
            cl_desc = "sellers gaining ground but not yet at threshold"
        else:
            cl_icon = "\U0001f534"
            cl_desc = "below threshold \u2014 sellers winning the intraday battle"
        st.markdown(
            f"**Current {cl_window}d avg close location:** {cl_icon} **{cur_cl:.3f}** "
            f"(threshold: {cl_threshold:.2f}) \u2014 {cl_desc}"
        )

    study_cl = run_event_study(signal_cl, spy_close, signal_name="Close Location")

    # Rolling close location chart with SPY overlay
    fig_cl = go.Figure()
    rcl_clean = rolling_cl.dropna()
    fig_cl.add_trace(go.Scatter(
        x=rcl_clean.index, y=rcl_clean,
        name=f"{cl_window}d Avg Close Loc", line=dict(width=1.5, color="#3498db"),
    ))
    fig_cl.add_hline(
        y=cl_threshold, line_dash="dash", line_color="#FFD700", line_width=1,
        annotation_text=f"Threshold: {cl_threshold}",
    )
    fig_cl.add_hline(y=0.5, line_dash="dot", line_color="rgba(150,150,150,0.5)", line_width=1)
    fig_cl.add_trace(go.Scatter(
        x=spy_close.index, y=spy_close,
        name="SPY", line=dict(width=1, color="rgba(100,100,100,0.4)"),
        yaxis="y2",
    ))
    fig_cl.update_layout(**_dual_y_layout("Rolling Close Location Value", "Close Location (0\u20131)", "SPY"))
    st.plotly_chart(fig_cl, use_container_width=True)

    render_event_study(study_cl, "Close Location", signal_cl, spy_close)

    # Scatter: rolling close location vs forward 21d return
    if len(rolling_cl.dropna()) > 252:
        fwd_21d = spy_close.shift(-21) / spy_close - 1
        scatter_df = pd.DataFrame({
            'close_loc': rolling_cl,
            'fwd_21d': fwd_21d,
        }).dropna()

        if len(scatter_df) > 100:
            slope, intercept, r, p_scatter, _ = stats.linregress(
                scatter_df['close_loc'], scatter_df['fwd_21d']
            )

            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=scatter_df['close_loc'], y=scatter_df['fwd_21d'],
                mode='markers', name='Days',
                marker=dict(size=2, color="rgba(52,152,219,0.3)"),
            ))
            x_line = np.linspace(scatter_df['close_loc'].min(), scatter_df['close_loc'].max(), 50)
            fig_scatter.add_trace(go.Scatter(
                x=x_line, y=slope * x_line + intercept,
                name=f"Regression (R\u00b2={r**2:.3f})", mode='lines',
                line=dict(color="#e74c3c", width=2),
            ))
            fig_scatter.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(title=f"{cl_window}d Avg Close Location"),
                yaxis=dict(title="Forward 21d SPY Return", tickformat=".1%"),
                title=dict(
                    text=f"Close Location vs Forward 21d Return (R\u00b2={r**2:.3f}, p={p_scatter:.4f})",
                    font=dict(size=13),
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)


# ========================== TAB 2: EQUITY PUT/CALL ==========================
with tab2:
    st.markdown("### Equity Put/Call Ratio")
    st.markdown(
        "> The equity put/call ratio measures speculative option positioning on "
        "individual stocks (not index options, which are dominated by institutional "
        "hedging). When the ratio is very low (heavy call buying relative to puts), "
        "speculative participants are aggressively bullish. This creates mechanical "
        "fragility: market makers short those calls and delta-hedge by buying stock, "
        "pushing prices up in a reflexive loop. But when the trade reverses, the "
        "delta-hedge unwind creates selling pressure. Extreme low readings identify "
        "speculative sentiment extremes AND mechanical fragility simultaneously."
    )

    pcr_series = load_putcall_data()

    if pcr_series is not None and len(pcr_series) > 0:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pc_lookback = st.slider("Z-score lookback (days)", 21, 252, 63, key="pc_lb")
        with c2:
            pc_threshold = st.slider("Z-score threshold", -2.5, -0.5, -1.5, 0.1, key="pc_thresh")
        with c3:
            pc_smooth = st.slider("Smoothing window (days)", 1, 21, 5, key="pc_smooth")
        with c4:
            pc_uptrend = st.checkbox("Require SPY uptrend", value=True, key="pc_uptrend")

        signal_pc, pcr_smooth, pcr_z = compute_putcall_signal(
            pcr_series, spy_close, pc_lookback, pc_threshold, pc_smooth, pc_uptrend,
        )
        all_signals["Equity Put/Call"] = signal_pc

        # Current reading
        cur_z = float(pcr_z.dropna().iloc[-1]) if len(pcr_z.dropna()) > 0 else None
        cur_raw = float(pcr_smooth.dropna().iloc[-1]) if len(pcr_smooth.dropna()) > 0 else None
        if cur_z is not None:
            z_icon = "\U0001f534" if cur_z < pc_threshold else ("\U0001f7e1" if cur_z < 0 else "\U0001f7e2")
            st.markdown(
                f"**Current put/call z-score:** {z_icon} **{cur_z:.2f}** "
                f"(threshold: {pc_threshold:.1f}) | "
                f"Smoothed ratio: {cur_raw:.3f}" if cur_raw else ""
            )

        study_pc = run_event_study(signal_pc, spy_close, signal_name="Equity Put/Call")

        # Z-score chart with SPY overlay
        fig_z = go.Figure()
        z_clean = pcr_z.dropna()
        fig_z.add_trace(go.Scatter(
            x=z_clean.index, y=z_clean,
            name="P/C Z-Score", line=dict(width=1.5, color="#9b59b6"),
        ))
        fig_z.add_hline(
            y=pc_threshold, line_dash="dash", line_color="#FF4444", line_width=1,
            annotation_text=f"Threshold: {pc_threshold}",
        )
        fig_z.add_hline(y=0, line_dash="dot", line_color="rgba(150,150,150,0.4)", line_width=1)
        fig_z.add_trace(go.Scatter(
            x=spy_close.index, y=spy_close,
            name="SPY", line=dict(width=1, color="rgba(100,100,100,0.4)"),
            yaxis="y2",
        ))
        fig_z.update_layout(**_dual_y_layout("Equity Put/Call Z-Score", "Z-Score", "SPY"))
        st.plotly_chart(fig_z, use_container_width=True)

        # Raw smoothed put/call ratio with bands
        fig_raw = go.Figure()
        pcr_clean = pcr_smooth.dropna()
        pcr_rm = pcr_smooth.rolling(pc_lookback).mean().dropna()
        pcr_rstd = pcr_smooth.rolling(pc_lookback).std().dropna()
        common_idx = pcr_clean.index.intersection(pcr_rm.index).intersection(pcr_rstd.index)

        fig_raw.add_trace(go.Scatter(
            x=pcr_clean.index, y=pcr_clean,
            name=f"Smoothed P/C ({pc_smooth}d)", line=dict(width=1.5, color="#3498db"),
        ))
        if len(common_idx) > 0:
            fig_raw.add_trace(go.Scatter(
                x=common_idx, y=pcr_rm.reindex(common_idx),
                name=f"Mean ({pc_lookback}d)", line=dict(width=1, color="rgba(150,150,150,0.6)", dash="dot"),
            ))
            upper = (pcr_rm + 1.5 * pcr_rstd).reindex(common_idx)
            lower = (pcr_rm - 1.5 * pcr_rstd).reindex(common_idx)
            fig_raw.add_trace(go.Scatter(
                x=common_idx, y=upper,
                name="+1.5\u03c3", line=dict(width=0.5, color="rgba(100,100,100,0.3)"),
                showlegend=False,
            ))
            fig_raw.add_trace(go.Scatter(
                x=common_idx, y=lower,
                name="-1.5\u03c3", line=dict(width=0.5, color="rgba(100,100,100,0.3)"),
                fill="tonexty", fillcolor="rgba(100,100,100,0.05)",
                showlegend=False,
            ))

        fig_raw.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            hovermode="x unified",
            yaxis=dict(title="Put/Call Ratio"),
            title=dict(text="Smoothed Equity Put/Call Ratio", font=dict(size=13)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_raw, use_container_width=True)

        render_event_study(study_pc, "Equity Put/Call", signal_pc, spy_close)

    else:
        st.markdown(
            "This signal requires externally sourced data. Upload the CBOE equity "
            "put/call ratio CSV to run the analysis. **The other three tabs on this "
            "page work with cached data.**"
        )


# ========================== TAB 3: SMALL CAP RELATIVE STRENGTH ==============
with tab3:
    st.markdown("### Small Cap Relative Strength")
    st.markdown(
        "> The IWM/SPY ratio measures risk appetite across the market cap spectrum. "
        "Small caps are where leveraged and speculative money concentrates \u2014 "
        "they're less liquid, have wider spreads, and more credit risk. When "
        "institutions derisk, they sell small caps first because they can, then "
        "move up the cap spectrum. Persistent IWM underperformance vs SPY (falling "
        "IWM/SPY ratio) is the canary in the coal mine \u2014 it means risk appetite "
        "is deteriorating at the margin even if large caps are still holding up. "
        "This signal has 2\u20134 weeks of lead time before large cap weakness materializes."
    )

    if 'IWM' not in closes.columns:
        st.error("IWM data not available. Cannot compute small cap relative strength.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            sc_change_lb = st.slider("Lookback for ratio change (days)", 10, 63, 42, key="sc_change")
        with c2:
            sc_z_lb = st.slider("Z-score lookback (days)", 126, 504, 252, key="sc_zlb")
        with c3:
            sc_z_thresh = st.slider("Z-score threshold", -2.0, -0.5, -1.0, 0.1, key="sc_zthresh")

        sc_qqq = st.checkbox("Also check QQQ/SPY divergence", value=False, key="sc_qqq")

        signal_sc, iwm_spy, ratio_chg, ratio_chg_z = compute_smallcap_signal(
            closes, spy_close, sc_change_lb, sc_z_lb, sc_z_thresh, sc_qqq,
        )
        all_signals["Small Cap RS"] = signal_sc

        # Current reading
        cur_z_sc = float(ratio_chg_z.dropna().iloc[-1]) if len(ratio_chg_z.dropna()) > 0 else None
        cur_ratio = float(iwm_spy.dropna().iloc[-1]) if len(iwm_spy.dropna()) > 0 else None
        if cur_z_sc is not None:
            z_icon = "\U0001f534" if cur_z_sc < sc_z_thresh else ("\U0001f7e1" if cur_z_sc < 0 else "\U0001f7e2")
            st.markdown(
                f"**IWM/SPY ratio change z-score:** {z_icon} **{cur_z_sc:.2f}** "
                f"(threshold: {sc_z_thresh:.1f}) | "
                f"Current IWM/SPY ratio: {cur_ratio:.4f}" if cur_ratio else ""
            )

        study_sc = run_event_study(signal_sc, spy_close, signal_name="Small Cap RS")

        # IWM/SPY ratio chart with SPY overlay
        fig_ratio = go.Figure()
        ratio_clean = iwm_spy.dropna()
        fig_ratio.add_trace(go.Scatter(
            x=ratio_clean.index, y=ratio_clean,
            name="IWM/SPY", line=dict(width=1.5, color="#e67e22"),
        ))
        fig_ratio.add_trace(go.Scatter(
            x=spy_close.index, y=spy_close,
            name="SPY", line=dict(width=1, color="rgba(100,100,100,0.4)"),
            yaxis="y2",
        ))
        fig_ratio.update_layout(**_dual_y_layout("IWM / SPY Ratio", "IWM/SPY", "SPY"))
        st.plotly_chart(fig_ratio, use_container_width=True)

        # Z-score chart
        fig_z_sc = go.Figure()
        z_clean = ratio_chg_z.dropna()
        fig_z_sc.add_trace(go.Scatter(
            x=z_clean.index, y=z_clean,
            name="Ratio Change Z-Score", line=dict(width=1.5, color="#2ecc71"),
        ))
        fig_z_sc.add_hline(
            y=sc_z_thresh, line_dash="dash", line_color="#FF4444", line_width=1,
            annotation_text=f"Threshold: {sc_z_thresh}",
        )
        fig_z_sc.add_hline(y=0, line_dash="dot", line_color="rgba(150,150,150,0.4)", line_width=1)
        fig_z_sc.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            hovermode="x unified",
            yaxis=dict(title="Z-Score"),
            title=dict(text=f"IWM/SPY {sc_change_lb}d Change Z-Score", font=dict(size=13)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_z_sc, use_container_width=True)

        # Rolling correlation: ratio_change vs forward 21d SPY return
        if len(ratio_chg.dropna()) > 504:
            fwd_21d = spy_close.shift(-21) / spy_close - 1
            rc_aligned = ratio_chg.reindex(spy_close.index)
            corr_df = pd.DataFrame({
                'ratio_chg': rc_aligned,
                'fwd_21d': fwd_21d,
            }).dropna()

            if len(corr_df) > 504:
                rolling_corr = corr_df['ratio_chg'].rolling(252).corr(corr_df['fwd_21d'])

                fig_corr = go.Figure()
                corr_clean = rolling_corr.dropna()
                fig_corr.add_trace(go.Scatter(
                    x=corr_clean.index, y=corr_clean,
                    name="Rolling 252d Corr", line=dict(width=1.5, color="#8e44ad"),
                ))
                fig_corr.add_hline(y=0, line_dash="dot", line_color="rgba(150,150,150,0.5)", line_width=1)
                fig_corr.update_layout(
                    height=250,
                    margin=dict(l=10, r=10, t=30, b=10),
                    hovermode="x unified",
                    yaxis=dict(title="Correlation", range=[-1, 1]),
                    title=dict(
                        text="Rolling Corr: IWM/SPY Change vs Forward 21d SPY Return",
                        font=dict(size=13),
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_corr, use_container_width=True)

        render_event_study(study_sc, "Small Cap RS", signal_sc, spy_close)


# ========================== TAB 4: OVERNIGHT VS INTRADAY ====================
with tab4:
    st.markdown("### Overnight vs Intraday Returns")
    st.markdown(
        "> Decomposes SPY's return into two sessions: overnight (today's open vs "
        "yesterday's close) and intraday (today's close vs today's open). These "
        "are driven by completely different participants \u2014 overnight returns "
        "by institutional/macro positioning, intraday by market makers, retail, "
        "and systematic flows. In a healthy market, both components are positive. "
        "In a fragile market approaching a top, you see a specific pattern: "
        "**strong overnight returns but weak/negative intraday returns.** The market "
        "gaps up every morning then sells off during the day. This means institutional "
        "overnight buyers are providing exit liquidity for daytime sellers \u2014 "
        "an invisible distribution pattern that doesn't show up on daily "
        "close-to-close charts."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        oi_window = st.slider("Cumulative window (trading days)", 10, 63, 21, key="oi_win")
    with c2:
        oi_overnight_pos = st.checkbox("Require overnight positive", value=True, key="oi_on_pos")
    with c3:
        oi_intra_thresh = st.slider(
            "Intraday cumulative threshold",
            -0.020, 0.010, 0.000, 0.002,
            format="%.3f", key="oi_intra",
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        oi_div_min = st.slider(
            "Divergence minimum (overnight - intraday)",
            0.010, 0.050, 0.020, 0.005,
            format="%.3f", key="oi_div",
        )
    with c5:
        oi_near_high = st.checkbox("Require near 52w high", value=True, key="oi_near")
    with c6:
        oi_near_pct = st.slider("Near-high threshold (%)", 2, 10, 5, key="oi_near_pct")

    signal_oi, on_ret, intra_ret, cum_on, cum_intra = compute_overnight_intraday(
        spy_df, oi_window, oi_overnight_pos, oi_intra_thresh, oi_div_min,
        oi_near_high, oi_near_pct,
    )
    all_signals["Overnight/Intraday"] = signal_oi

    # Current reading
    cur_cum_on = float(cum_on.dropna().iloc[-1]) if len(cum_on.dropna()) > 0 else None
    cur_cum_intra = float(cum_intra.dropna().iloc[-1]) if len(cum_intra.dropna()) > 0 else None
    if cur_cum_on is not None and cur_cum_intra is not None:
        div = cur_cum_on - cur_cum_intra
        if cur_cum_on > 0 and cur_cum_intra < 0:
            oi_icon = "\U0001f534"
            oi_desc = "overnight is carrying the market, intraday selling is persistent"
        elif cur_cum_intra < 0:
            oi_icon = "\U0001f7e1"
            oi_desc = "intraday returns negative, but overnight not strongly positive"
        else:
            oi_icon = "\U0001f7e2"
            oi_desc = "both sessions contributing positively"

        st.markdown(
            f"**Last {oi_window} trading days:** {oi_icon} Overnight: **{cur_cum_on:+.1%}** | "
            f"Intraday: **{cur_cum_intra:+.1%}** | "
            f"Divergence: **{div:+.1%}** \u2014 {oi_desc}"
        )

    study_oi = run_event_study(signal_oi, spy_close, signal_name="Overnight/Intraday")

    # Chart 1: Cumulative overnight vs intraday with SPY overlay
    fig_oi = go.Figure()
    cum_on_clean = cum_on.dropna()
    cum_intra_clean = cum_intra.dropna()
    fig_oi.add_trace(go.Scatter(
        x=cum_on_clean.index, y=cum_on_clean,
        name=f"Cum Overnight ({oi_window}d)", line=dict(width=1.5, color="#3498db"),
    ))
    fig_oi.add_trace(go.Scatter(
        x=cum_intra_clean.index, y=cum_intra_clean,
        name=f"Cum Intraday ({oi_window}d)", line=dict(width=1.5, color="#e74c3c"),
    ))
    fig_oi.add_trace(go.Scatter(
        x=spy_close.index, y=spy_close,
        name="SPY", line=dict(width=1, color="rgba(100,100,100,0.4)"),
        yaxis="y2",
    ))
    fig_oi.update_layout(**_dual_y_layout(
        f"Cumulative Overnight vs Intraday Returns ({oi_window}d window)",
        "Cumulative Return", "SPY",
    ))
    fig_oi.update_layout(yaxis=dict(tickformat=".1%"))
    st.plotly_chart(fig_oi, use_container_width=True)

    # Chart 2: Stacked contribution (63d window for visibility)
    contrib_window = 63
    cum_on_63 = on_ret.rolling(contrib_window).sum()
    cum_intra_63 = intra_ret.rolling(contrib_window).sum()
    fig_stack = go.Figure()
    c63_on = cum_on_63.dropna()
    c63_intra = cum_intra_63.dropna()
    fig_stack.add_trace(go.Scatter(
        x=c63_on.index, y=c63_on,
        name="Overnight (63d)", line=dict(width=0), mode='lines',
        fillcolor="rgba(52,152,219,0.3)", stackgroup='one',
    ))
    fig_stack.add_trace(go.Scatter(
        x=c63_intra.index, y=c63_intra,
        name="Intraday (63d)", line=dict(width=0), mode='lines',
        fillcolor="rgba(231,76,60,0.3)", stackgroup='one',
    ))
    fig_stack.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        yaxis=dict(title="Cumulative Return", tickformat=".1%"),
        title=dict(text="Return Decomposition: Overnight vs Intraday (63d cumulative)", font=dict(size=13)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    # Chart 3: Divergence (overnight - intraday cumulative)
    divergence = cum_on - cum_intra
    fig_div = go.Figure()
    div_clean = divergence.dropna()
    fig_div.add_trace(go.Scatter(
        x=div_clean.index, y=div_clean,
        name="Divergence", line=dict(width=1.5, color="#f39c12"),
    ))
    fig_div.add_hline(
        y=oi_div_min, line_dash="dash", line_color="#FF4444", line_width=1,
        annotation_text=f"Threshold: {oi_div_min:.3f}",
    )
    fig_div.add_hline(y=0, line_dash="dot", line_color="rgba(150,150,150,0.4)", line_width=1)
    fig_div.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        yaxis=dict(title="Overnight \u2212 Intraday", tickformat=".1%"),
        title=dict(text=f"Session Divergence ({oi_window}d cumulative)", font=dict(size=13)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_div, use_container_width=True)

    render_event_study(study_oi, "Overnight/Intraday", signal_oi, spy_close)


# ===========================================================================
# SIGNAL INDEPENDENCE CHECK (Correlation Matrix)
# ===========================================================================
st.divider()
st.subheader("Signal Independence Check")
st.markdown(
    "Pairwise correlation between signal activation series (daily boolean). "
    "Low correlations confirm the signals are tapping different information channels."
)

# Build correlation DataFrame from all computed signals
corr_signals = {}
for name, sig in all_signals.items():
    if sig is not None and len(sig.dropna()) > 0:
        corr_signals[name] = sig.astype(float)

if len(corr_signals) >= 2:
    corr_df = pd.DataFrame(corr_signals).dropna(how='all')
    corr_df = corr_df.fillna(0)
    corr_matrix = corr_df.corr()

    # Plotly heatmap
    labels = list(corr_matrix.columns)
    z_vals = corr_matrix.values

    # Color scale: green for low correlation, red for high
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#00CC00"],
            [0.25, "#7FCC00"],
            [0.5, "#FFD700"],
            [0.75, "#FF8C00"],
            [1.0, "#CC0000"],
        ],
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z_vals],
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
    ))
    fig_heatmap.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Signal Pairwise Correlation (Boolean Activation)", font=dict(size=13)),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Flag high correlations
    high_corr = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > 0.5:
                high_corr.append((labels[i], labels[j], val))

    if high_corr:
        st.warning(
            "**High correlations detected** (>0.5): "
            + ", ".join(f"{a} / {b}: {v:.2f}" for a, b, v in high_corr)
            + " \u2014 consider dropping redundant signals."
        )
    else:
        st.success("All pairwise correlations below 0.5 \u2014 signals appear independent.")
else:
    st.info("Need at least 2 signals computed to show the correlation matrix.")
