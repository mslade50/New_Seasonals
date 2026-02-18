import streamlit as st

st.set_page_config(page_title="Signal Backtester", page_icon="\U0001f52c", layout="wide")

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
CACHE_SP500 = os.path.join(DATA_DIR, "rd2_sp500_closes.parquet")
RISK_CLASSIFICATION = os.path.join(DATA_DIR, "sp500_risk_classification.csv")

# ---------------------------------------------------------------------------
# FOMC & CPI CALENDARS
# ---------------------------------------------------------------------------
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

CPI_DATES = pd.to_datetime([
    # 2020
    "2020-01-14", "2020-02-13", "2020-03-11", "2020-04-10",
    "2020-05-12", "2020-06-10", "2020-07-14", "2020-08-12",
    "2020-09-11", "2020-10-13", "2020-11-12", "2020-12-10",
    # 2021
    "2021-01-13", "2021-02-10", "2021-03-10", "2021-04-13",
    "2021-05-12", "2021-06-10", "2021-07-13", "2021-08-11",
    "2021-09-14", "2021-10-13", "2021-11-10", "2021-12-10",
    # 2022
    "2022-01-12", "2022-02-10", "2022-03-10", "2022-04-12",
    "2022-05-11", "2022-06-10", "2022-07-13", "2022-08-10",
    "2022-09-13", "2022-10-13", "2022-11-10", "2022-12-13",
    # 2023
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12",
    "2023-05-10", "2023-06-13", "2023-07-12", "2023-08-10",
    "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
    # 2024
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
    "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
    "2025-09-10", "2025-10-14", "2025-11-12", "2025-12-10",
    # 2026
    "2026-01-13", "2026-02-11",
])

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


@st.cache_data(ttl=3600)
def load_sp500_closes():
    """Load S&P 500 constituent closes from parquet cache."""
    if os.path.exists(CACHE_SP500):
        df = pd.read_parquet(CACHE_SP500)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    return None


@st.cache_data(ttl=3600)
def load_risk_classification():
    """Load risk-on/risk-off classification CSV."""
    if os.path.exists(RISK_CLASSIFICATION):
        df = pd.read_csv(RISK_CLASSIFICATION)
        return df
    return None


# ---------------------------------------------------------------------------
# SHARED EVENT STUDY ENGINE
# ---------------------------------------------------------------------------
def run_event_study(
    signal_series: pd.Series,
    price_series: pd.Series,
    forward_windows: list = None,
    signal_name: str = "Signal",
) -> dict:
    """
    Run a standard event study.

    For each day where signal_series is True, compute forward returns
    at each horizon.  Compare to unconditional forward returns.
    """
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

    # Deduplicate signal onsets: first activation in each cluster (10d lookback)
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

    # --- Forward return distributions (21d) — normalized density ---
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
# SIGNAL 1: DISTRIBUTION / ACCUMULATION RATIO
# ---------------------------------------------------------------------------
def compute_distribution_accumulation(spy_df: pd.DataFrame,
                                       vol_mult: float = 1.25,
                                       window: int = 21,
                                       ratio_threshold: float = 1.5,
                                       require_uptrend: bool = True) -> tuple:
    """
    Compute distribution/accumulation signal.

    Returns: (signal_series, da_ratio, dist_days, accum_days)
    """
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

    signal = da_ratio > ratio_threshold

    if require_uptrend:
        sma_50 = close.rolling(50).mean()
        signal = signal & (close > sma_50)

    return signal, da_ratio, dist_days, accum_days


# ---------------------------------------------------------------------------
# SIGNAL 2: VIX RANGE COMPRESSION
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_vix_ohlc():
    """Download VIX OHLC for ATR computation."""
    try:
        raw = yf.download("^VIX", start="2010-01-01", auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.capitalize() for c in raw.columns]
        if raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)
        return raw
    except Exception:
        return None


def _rolling_percentile(series: pd.Series, lookback: int) -> pd.Series:
    """Vectorised rolling percentile rank."""
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


def compute_vix_compression(closes: pd.DataFrame,
                             spy_close: pd.Series,
                             range_window: int = 21,
                             pctile_threshold: float = 20,
                             min_vix: float = 13,
                             pctile_lookback: int = 504,
                             require_above_sma: bool = True,
                             sma_period: int = 20,
                             metric: str = "Close Range",
                             atr_period: int = 14,
                             vix_ohlc: pd.DataFrame = None) -> tuple:
    """
    VIX compression signal using close-to-close range or ATR.

    Returns: (signal_series, compression_metric, compression_pctile)
    """
    if "^VIX" not in closes.columns:
        empty = pd.Series(dtype=float)
        return pd.Series(dtype=bool), empty, empty

    vix = closes["^VIX"].dropna()

    if metric == "ATR" and vix_ohlc is not None and not vix_ohlc.empty:
        high = vix_ohlc["High"]
        low = vix_ohlc["Low"]
        prev_close = vix_ohlc["Close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(atr_period, min_periods=max(1, atr_period // 2)).mean()
        compression_metric = atr.reindex(vix.index)
    else:
        compression_metric = vix.rolling(range_window).max() - vix.rolling(range_window).min()

    compression_pctile = _rolling_percentile(compression_metric, pctile_lookback)

    signal = (compression_pctile < pctile_threshold) & (vix > min_vix)

    if require_above_sma:
        vix_sma = vix.rolling(sma_period, min_periods=int(sma_period * 0.8)).mean()
        signal = signal & (vix > vix_sma)

    return signal, compression_metric, compression_pctile


# ---------------------------------------------------------------------------
# SIGNAL 3: LEADERSHIP QUALITY (STOCK-LEVEL RISK CLASSIFICATION)
# ---------------------------------------------------------------------------
def compute_leadership_quality(sp500_closes: pd.DataFrame,
                                classification: pd.DataFrame,
                                spy_close: pd.Series,
                                sma_period: int = 200,
                                def_lead_threshold: float = 10,
                                near_high_pct: float = 5,
                                use_rel_perf: bool = False,
                                rel_perf_window: int = 21) -> tuple:
    """
    Stock-level leadership quality signal using risk-on/risk-off classification.

    Returns: (signal_series, leadership_spread, risk_on_breadth, risk_off_breadth,
              n_risk_on, n_risk_off)
    """
    on_tickers = classification.loc[classification["label"] == "risk_on", "ticker"].tolist()
    off_tickers = classification.loc[classification["label"] == "risk_off", "ticker"].tolist()

    # Intersect with available price data
    on_cols = [t for t in on_tickers if t in sp500_closes.columns]
    off_cols = [t for t in off_tickers if t in sp500_closes.columns]

    if len(on_cols) < 10 or len(off_cols) < 10:
        empty = pd.Series(dtype=float, index=sp500_closes.index)
        return pd.Series(False, index=sp500_closes.index), empty, empty, empty, 0, 0

    high_52w = spy_close.rolling(252, min_periods=60).max()
    near_high = spy_close >= high_52w * (1 - near_high_pct / 100)

    if use_rel_perf:
        on_ret = sp500_closes[on_cols].pct_change(rel_perf_window).mean(axis=1)
        off_ret = sp500_closes[off_cols].pct_change(rel_perf_window).mean(axis=1)

        leadership_spread = (on_ret - off_ret) * 100
        risk_on_breadth = on_ret * 100
        risk_off_breadth = off_ret * 100

        signal = (leadership_spread < -def_lead_threshold / 10) & near_high
    else:
        sma = sp500_closes.rolling(sma_period, min_periods=int(sma_period * 0.8)).mean()

        on_above = (sp500_closes[on_cols] > sma[on_cols]).sum(axis=1) / len(on_cols) * 100
        off_above = (sp500_closes[off_cols] > sma[off_cols]).sum(axis=1) / len(off_cols) * 100

        leadership_spread = on_above - off_above
        risk_on_breadth = on_above
        risk_off_breadth = off_above

        signal = (leadership_spread < -def_lead_threshold) & near_high

    return signal, leadership_spread, risk_on_breadth, risk_off_breadth, len(on_cols), len(off_cols)


# ---------------------------------------------------------------------------
# SIGNAL 4: PRE-EVENT POSITIONING
# ---------------------------------------------------------------------------
def compute_pre_event_positioning(spy_close: pd.Series,
                                   pre_window: int = 5,
                                   pctile_threshold: float = 75,
                                   include_cpi: bool = True) -> tuple:
    """
    Pre-event positioning signal.

    Returns: (signal_series, event_df, all_event_dates)
    """
    event_dates_list = list(FOMC_DATES)
    event_labels = ["FOMC"] * len(FOMC_DATES)

    if include_cpi:
        event_dates_list += list(CPI_DATES)
        event_labels += ["CPI"] * len(CPI_DATES)

    combined = list(zip(event_dates_list, event_labels))
    combined.sort(key=lambda x: x[0])
    # Deduplicate by date
    seen = set()
    deduped = []
    for dt, lbl in combined:
        if dt not in seen:
            seen.add(dt)
            deduped.append((dt, lbl))
    event_dates_list = [x[0] for x in deduped]
    event_labels = [x[1] for x in deduped]

    # Snap to trading days
    adjusted_dates = []
    adjusted_labels = []
    for dt, label in zip(event_dates_list, event_labels):
        if dt in spy_close.index:
            adjusted_dates.append(dt)
            adjusted_labels.append(label)
        else:
            future = spy_close.index[spy_close.index >= dt]
            if len(future) > 0 and (future[0] - dt).days <= 5:
                adjusted_dates.append(future[0])
                adjusted_labels.append(label)

    pre_returns = spy_close.pct_change(pre_window)

    # Rolling percentile of pre-event return
    pre_ret_arr = pre_returns.values
    pre_ret_pctile_arr = np.full(len(pre_ret_arr), np.nan)
    lookback = 504
    for i in range(lookback, len(pre_ret_arr)):
        window = pre_ret_arr[i - lookback : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 100:
            continue
        pre_ret_pctile_arr[i] = (valid[:-1] < valid[-1]).sum() / (len(valid) - 1) * 100
    pre_ret_pctile = pd.Series(pre_ret_pctile_arr, index=pre_returns.index)

    rows = []
    for dt, label in zip(adjusted_dates, adjusted_labels):
        if dt not in spy_close.index:
            continue
        idx = spy_close.index.get_loc(dt)

        pre_ret = pre_returns.get(dt, np.nan)
        pre_pct = pre_ret_pctile.get(dt, np.nan)

        row = {
            "date": dt,
            "event": label,
            "pre_return": pre_ret,
            "pre_pctile": pre_pct,
            "signal_on": pre_pct > pctile_threshold if not np.isnan(pre_pct) else False,
        }

        for h in [1, 2, 3, 5, 10, 21]:
            if idx + h < len(spy_close):
                row[f"post_{h}d"] = spy_close.iloc[idx + h] / spy_close.iloc[idx] - 1
            else:
                row[f"post_{h}d"] = np.nan

        rows.append(row)

    event_df = pd.DataFrame(rows)

    signal_series = pd.Series(False, index=spy_close.index)
    if not event_df.empty:
        signal_dates = event_df.loc[event_df["signal_on"], "date"]
        for d in signal_dates:
            if d in signal_series.index:
                signal_series.loc[d] = True

    return signal_series, event_df, pd.DatetimeIndex(adjusted_dates)


# ===========================================================================
# MAIN PAGE
# ===========================================================================
st.title("\U0001f52c Signal Event Study Backtester")
st.caption(
    "Evaluating candidate signals for the risk dashboard. "
    "Each signal is tested independently: does it identify environments "
    "where forward equity returns are worse than average?"
)

spy_df, closes = load_data()
if spy_df is None or spy_df.empty:
    st.error("No data available. Run the Risk Dashboard V2 data refresh first to populate the parquet cache.")
    st.stop()

spy_close = spy_df["Close"]

tab1, tab2, tab3, tab4 = st.tabs([
    "Distribution / Accumulation",
    "VIX Range Compression",
    "Sector Leadership",
    "Pre-Event Positioning",
])

# ========================== TAB 1 ==========================
with tab1:
    st.markdown("### Distribution / Accumulation Ratio")
    st.markdown(
        "> A **distribution day** = SPY volume > trailing 63d avg volume "
        "AND volume > previous day's volume by a configurable multiplier (default 1.25x) "
        "AND close < open (intraday selling on a volume surge). "
        "An **accumulation day** = same volume thresholds but close > open. "
        "The signal tracks the rolling ratio of distribution days to accumulation days. "
        "When distribution meaningfully exceeds accumulation during an uptrend, "
        "institutional selling is occurring beneath the surface."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        da_vol_mult = st.slider("Prev-day volume multiplier", 1.0, 2.0, 1.25, 0.05, key="da_vol")
    with c2:
        da_window = st.slider("Rolling window (days)", 10, 63, 21, key="da_win")
    with c3:
        da_ratio_thresh = st.slider("D/A ratio threshold", 1.0, 6.0, 1.5, 0.25, key="da_thresh")
    with c4:
        da_uptrend = st.checkbox("Require uptrend (SPY > 50d SMA)", value=True, key="da_up")

    signal, da_ratio, dist_days, accum_days = compute_distribution_accumulation(
        spy_df, da_vol_mult, da_window, da_ratio_thresh, da_uptrend
    )

    # Current state summary
    recent_dist = int(dist_days.iloc[-da_window:].sum()) if len(dist_days) >= da_window else 0
    recent_accum = int(accum_days.iloc[-da_window:].sum()) if len(accum_days) >= da_window else 0
    current_ratio = da_ratio.iloc[-1] if len(da_ratio) > 0 and not np.isnan(da_ratio.iloc[-1]) else 0
    st.markdown(
        f"**Last {da_window} sessions:** Distribution days: {recent_dist} | "
        f"Accumulation days: {recent_accum} | Current ratio: {current_ratio:.2f}"
    )

    study = run_event_study(signal, spy_close, signal_name="Distribution/Accumulation")

    # D/A ratio chart with SPY overlay
    fig_da = go.Figure()
    da_clean = da_ratio.dropna()
    fig_da.add_trace(go.Scatter(
        x=da_clean.index, y=da_clean,
        name="D/A Ratio", line=dict(width=1.5, color="#e74c3c"),
    ))
    fig_da.add_hline(y=da_ratio_thresh, line_dash="dash", line_color="yellow",
                     annotation_text=f"Threshold: {da_ratio_thresh}")
    fig_da.add_trace(go.Scatter(
        x=spy_close.index, y=spy_close,
        name="SPY", line=dict(width=1, color="rgba(100,100,100,0.4)"),
        yaxis="y2",
    ))
    fig_da.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        yaxis=dict(title="D/A Ratio"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="SPY"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text="Distribution / Accumulation Ratio", font=dict(size=13)),
    )
    st.plotly_chart(fig_da, use_container_width=True)

    render_event_study(study, "Distribution/Accumulation", signal, spy_close)


# ========================== TAB 2 ==========================
with tab2:
    st.markdown("### VIX Range Compression")
    st.markdown(
        "> Measures whether VIX is in a squeeze — compressing into a tight range. "
        "When VIX makes lower highs and higher lows simultaneously, the eventual "
        "breakout tends to be violent. The mechanism is mechanical: compressed VIX "
        "range -> dealer gamma concentration -> explosive expansion when the range breaks."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        vc_metric = st.selectbox("Compression metric", ["Close Range", "ATR"], key="vc_metric")
    with c2:
        vc_pctile = st.slider("Compression percentile", 5, 40, 20, key="vc_pct")
    with c3:
        vc_min_vix = st.slider("Min VIX level", 10.0, 18.0, 13.0, 0.5, key="vc_min")
    with c4:
        vc_lookback = st.slider("Percentile lookback (days)", 252, 1260, 504, key="vc_lb")

    vc_range_win = 21
    vc_atr_period = 14
    c5, c6, c7 = st.columns(3)
    with c5:
        if vc_metric == "Close Range":
            vc_range_win = st.slider("Range window (days)", 10, 42, 21, key="vc_win")
        else:
            vc_atr_period = st.slider("ATR period (days)", 5, 30, 14, key="vc_atr_p")
    with c6:
        vc_require_sma = st.checkbox("Require VIX > SMA", value=True, key="vc_sma_on")
    with c7:
        vc_sma_period = st.selectbox("SMA period", [10, 20, 50, 100, 200], index=1, key="vc_sma_p")

    vix_ohlc = load_vix_ohlc() if vc_metric == "ATR" else None

    signal_vc, vix_compression, vix_compression_pctile = compute_vix_compression(
        closes, spy_close,
        range_window=vc_range_win if vc_metric == "Close Range" else 21,
        pctile_threshold=vc_pctile,
        min_vix=vc_min_vix,
        pctile_lookback=vc_lookback,
        require_above_sma=vc_require_sma,
        sma_period=vc_sma_period,
        metric=vc_metric,
        atr_period=vc_atr_period if vc_metric == "ATR" else 14,
        vix_ohlc=vix_ohlc,
    )

    if "^VIX" not in closes.columns:
        st.error("VIX data not available in cache.")
    else:
        study_vc = run_event_study(signal_vc, spy_close, signal_name="VIX Compression")

        # VIX + range percentile chart
        vix = closes["^VIX"].dropna()
        fig_vc = go.Figure()
        fig_vc.add_trace(go.Scatter(
            x=vix.index, y=vix,
            name="VIX", line=dict(width=1.5, color="#3498db"),
        ))
        metric_label = "ATR" if vc_metric == "ATR" else "Range"
        pctile_clean = vix_compression_pctile.dropna()
        fig_vc.add_trace(go.Scatter(
            x=pctile_clean.index, y=pctile_clean,
            name=f"{metric_label} Percentile", line=dict(width=1, color="#e67e22"),
            yaxis="y2",
        ))
        fig_vc.add_hline(y=vc_pctile, line_dash="dash", line_color="yellow",
                         annotation_text=f"Pctile threshold: {vc_pctile}",
                         yref="y2")

        signal_on = signal_vc.fillna(False).astype(bool)
        transitions = signal_on.astype(int).diff().fillna(0)
        starts = transitions[transitions == 1].index
        ends = transitions[transitions == -1].index
        if len(signal_on) > 0 and signal_on.iloc[0]:
            starts = starts.insert(0, signal_on.index[0])
        if len(signal_on) > 0 and signal_on.iloc[-1]:
            ends = ends.append(pd.DatetimeIndex([signal_on.index[-1]]))
        for s, e in zip(starts, ends):
            fig_vc.add_vrect(x0=s, x1=e, fillcolor="rgba(204,0,0,0.1)", line_width=0, layer="below")

        fig_vc.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            hovermode="x unified",
            yaxis=dict(title="VIX"),
            yaxis2=dict(overlaying="y", side="right", showgrid=False, title=f"{metric_label} Pctile"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title=dict(text=f"VIX Level + {metric_label} Compression Percentile", font=dict(size=13)),
        )
        st.plotly_chart(fig_vc, use_container_width=True)

        # Forward VIX analysis
        st.markdown("#### Forward VIX Expansion")
        vix_fwd_range = {}
        for w in [5, 10, 21, 42]:
            fwd_max = vix.rolling(w).max().shift(-w)
            fwd_expansion = fwd_max - vix
            sig_exp = fwd_expansion.reindex(signal_vc[signal_vc == True].index).dropna()
            all_exp = fwd_expansion.dropna()
            vix_fwd_range[f"{w}d"] = {
                "Signal Mean": f"{sig_exp.mean():.1f}" if len(sig_exp) > 0 else "N/A",
                "Unconditional Mean": f"{all_exp.mean():.1f}",
                "Difference": f"{sig_exp.mean() - all_exp.mean():+.1f}" if len(sig_exp) > 0 else "N/A",
            }
        st.dataframe(pd.DataFrame(vix_fwd_range).T, use_container_width=True)

        render_event_study(study_vc, "VIX Compression", signal_vc, spy_close)


# ========================== TAB 3 ==========================
with tab3:
    st.markdown("### Leadership Quality (Stock-Level)")
    st.markdown(
        "> Each S&P 500 constituent is classified as **risk-on** or **risk-off** "
        "using a beta-based first pass with manual overrides "
        "(see `data/sp500_risk_classification.csv`). "
        "When risk-off stocks have better breadth (% above SMA) than risk-on stocks "
        "while SPX is near highs, it signals institutional rotation into safety — "
        "the market is being held up by defensive names, not genuine risk appetite."
    )

    sp500_closes = load_sp500_closes()
    classification = load_risk_classification()

    if sp500_closes is None or classification is None:
        st.error(
            "Missing data. Requires `data/rd2_sp500_closes.parquet` "
            "(run Risk Dashboard V2 refresh) and `data/sp500_risk_classification.csv`."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sl_sma = st.selectbox("Breadth SMA", [50, 100, 200], index=2, key="sl_sma")
        with c2:
            sl_thresh = st.slider("Risk-off lead threshold (pct pts)", 0, 30, 10, 5, key="sl_thresh")
        with c3:
            sl_near_high = st.slider("SPX proximity to high (%)", 2, 10, 5, key="sl_high")
        with c4:
            sl_rel_perf = st.checkbox("Use relative performance instead", value=False, key="sl_rel")

        sl_rel_win = 21
        if sl_rel_perf:
            sl_rel_win = st.slider("Relative perf window (days)", 10, 63, 21, key="sl_relwin")

        signal_sl, spread, on_breadth, off_breadth, n_on, n_off = compute_leadership_quality(
            sp500_closes, classification, spy_close,
            sl_sma, sl_thresh, sl_near_high, sl_rel_perf, sl_rel_win,
        )

        n_neutral = len(classification[classification["label"] == "neutral"])
        st.markdown(
            f"**Universe:** {n_on} risk-on stocks | {n_off} risk-off stocks | "
            f"{n_neutral} neutral (excluded)"
        )

        study_sl = run_event_study(signal_sl, spy_close, signal_name="Leadership Quality")

        # Risk-on vs Risk-off breadth chart
        fig_sl = go.Figure()
        on_clean = on_breadth.dropna()
        off_clean = off_breadth.dropna()
        label_suffix = "Rel Perf %" if sl_rel_perf else "% > SMA"
        fig_sl.add_trace(go.Scatter(
            x=on_clean.index, y=on_clean,
            name=f"Risk-On {label_suffix}", line=dict(width=1.5, color="#2ecc71"),
        ))
        fig_sl.add_trace(go.Scatter(
            x=off_clean.index, y=off_clean,
            name=f"Risk-Off {label_suffix}", line=dict(width=1.5, color="#e74c3c"),
        ))
        fig_sl.add_trace(go.Scatter(
            x=spy_close.index, y=spy_close,
            name="SPY", line=dict(width=1, color="rgba(100,100,100,0.4)"),
            yaxis="y2",
        ))
        fig_sl.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            hovermode="x unified",
            yaxis=dict(title=label_suffix),
            yaxis2=dict(overlaying="y", side="right", showgrid=False, title="SPY"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title=dict(text="Risk-On vs Risk-Off Breadth", font=dict(size=13)),
        )
        st.plotly_chart(fig_sl, use_container_width=True)

        # Leadership spread filled area
        spread_clean = spread.dropna()
        fig_spread = go.Figure()
        colors = ["rgba(46,204,113,0.4)" if v >= 0 else "rgba(231,76,60,0.4)" for v in spread_clean.values]
        fig_spread.add_trace(go.Bar(
            x=spread_clean.index, y=spread_clean, name="Spread",
            marker_color=colors,
        ))
        fig_spread.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text="Leadership Spread (Risk-On - Risk-Off)", font=dict(size=13)),
            yaxis=dict(title="Spread (pct pts)"),
        )
        st.plotly_chart(fig_spread, use_container_width=True)

        render_event_study(study_sl, "Leadership Quality", signal_sl, spy_close)


# ========================== TAB 4 ==========================
with tab4:
    st.markdown("### Pre-Event Positioning")
    st.markdown(
        "> Measures whether the market has rallied strongly going into a known macro "
        "event (FOMC announcement, CPI release). When SPY has rallied >Nth percentile "
        "of trailing returns in the days before a major event, positioning is one-sided "
        "and vulnerable to a 'buy the rumor, sell the news' reversal."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pe_window = st.slider("Pre-event window (days)", 3, 10, 5, key="pe_win")
    with c2:
        pe_pctile = st.slider("Pre-event return pctile threshold", 60, 90, 75, key="pe_pct")
    with c3:
        pe_cpi = st.checkbox("Include CPI dates", value=True, key="pe_cpi")
    with c4:
        pe_post_win = st.slider("Post-event eval window (days)", 3, 21, 5, key="pe_post")

    signal_pe, event_df, all_events = compute_pre_event_positioning(
        spy_close, pe_window, pe_pctile, pe_cpi
    )

    if event_df.empty:
        st.warning("No event dates found in the data range.")
    else:
        n_events = len(event_df)
        n_triggered = event_df["signal_on"].sum()
        st.markdown(f"**Events:** {n_events} total | {n_triggered} triggered signal "
                    f"({n_triggered / n_events * 100:.0f}%)")

        # Scatter plot: pre-event pctile vs post-event return
        plot_df = event_df.dropna(subset=["pre_pctile", f"post_{pe_post_win}d"])
        if len(plot_df) > 5:
            fig_scatter = go.Figure()

            for evt_type, color in [("FOMC", "#3498db"), ("CPI", "#e67e22")]:
                subset = plot_df[plot_df["event"] == evt_type]
                if subset.empty:
                    continue
                fig_scatter.add_trace(go.Scatter(
                    x=subset["pre_pctile"], y=subset[f"post_{pe_post_win}d"],
                    mode="markers", name=evt_type,
                    marker=dict(size=6, color=color, opacity=0.7),
                ))

            # Regression line
            x_vals = plot_df["pre_pctile"].values
            y_vals = plot_df[f"post_{pe_post_win}d"].values
            mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
            if mask.sum() > 5:
                slope, intercept, r_val, _, _ = stats.linregress(x_vals[mask], y_vals[mask])
                x_line = np.linspace(np.nanmin(x_vals), np.nanmax(x_vals), 50)
                y_line = slope * x_line + intercept
                fig_scatter.add_trace(go.Scatter(
                    x=x_line, y=y_line,
                    mode="lines", name=f"Regression (R={r_val:.2f})",
                    line=dict(color="white", dash="dash", width=1),
                ))

            fig_scatter.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(title="Pre-Event Return Percentile"),
                yaxis=dict(title=f"Post-Event {pe_post_win}d Return", tickformat=".1%"),
                title=dict(text=f"Pre-Event Positioning vs Post-{pe_post_win}d Return", font=dict(size=13)),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Quartile breakdown
        st.markdown("#### Returns by Pre-Event Return Quartile")
        quartile_df = event_df.dropna(subset=["pre_pctile"]).copy()
        if len(quartile_df) > 10:
            quartile_df["quartile"] = pd.qcut(
                quartile_df["pre_pctile"], 4,
                labels=["Q1 (sold off)", "Q2", "Q3", "Q4 (rallied)"]
            )

            post_cols = [c for c in quartile_df.columns if c.startswith("post_")]
            q_summary = quartile_df.groupby("quartile")[post_cols].agg(["mean", "median", "count"])

            # Flatten column names
            q_display = pd.DataFrame()
            for col in post_cols:
                horizon = col.replace("post_", "")
                q_display[f"{horizon} Mean"] = q_summary[(col, "mean")].map(lambda v: f"{v:+.2%}" if not np.isnan(v) else "N/A")
                q_display[f"{horizon} Median"] = q_summary[(col, "median")].map(lambda v: f"{v:+.2%}" if not np.isnan(v) else "N/A")

            q_display["N"] = q_summary[(post_cols[0], "count")].astype(int)
            st.dataframe(q_display, use_container_width=True)

        # Bar chart: signal ON vs all others
        st.markdown(f"#### Mean Post-{pe_post_win}d Return: Strong Rally vs Others")
        post_col = f"post_{pe_post_win}d"
        if post_col in event_df.columns:
            sig_on = event_df[event_df["signal_on"]][post_col].dropna()
            sig_off = event_df[~event_df["signal_on"]][post_col].dropna()

            if len(sig_on) > 0 and len(sig_off) > 0:
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=["Strong Rally Into Event", "All Other Events"],
                    y=[sig_on.mean(), sig_off.mean()],
                    marker_color=["#e74c3c", "#2ecc71"],
                    text=[f"{sig_on.mean():+.2%} (N={len(sig_on)})",
                          f"{sig_off.mean():+.2%} (N={len(sig_off)})"],
                    textposition="outside",
                ))
                fig_bar.update_layout(
                    height=250,
                    margin=dict(l=10, r=10, t=30, b=10),
                    yaxis=dict(tickformat=".2%"),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # Also show standard event study for completeness
        with st.expander("Full Event Study (standard forward returns)"):
            study_pe = run_event_study(
                signal_pe, spy_close,
                forward_windows=[5, 10, 21, 42, 63],
                signal_name="Pre-Event Positioning",
            )
            render_event_study(study_pe, "Pre-Event Positioning", signal_pe, spy_close)
