"""
Leader vs Laggard Forward Returns by Fragility Regime
======================================================
Event study: do leaders or laggards underperform more
when the market enters fragile regimes?

Standalone page — no imports from risk_dashboard_v2 or strategy modules.
Reads cached parquets produced by Risk Dashboard V2.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.graph_objects as go
from scipy import stats

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="Leader vs Laggard Study",
        page_icon="\U0001f4e1",
        layout="wide",
    )
except st.errors.StreamlitAPIException:
    pass

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

DATA_DIR = os.path.join(parent_dir, "data")

CACHE_SPY_OHLC = os.path.join(DATA_DIR, "rd2_spy_ohlc.parquet")
CACHE_CLOSES = os.path.join(DATA_DIR, "rd2_closes.parquet")
CACHE_SP500 = os.path.join(DATA_DIR, "rd2_sp500_closes.parquet")
CACHE_FRAGILITY = os.path.join(DATA_DIR, "rd2_fragility_ts.parquet")

# ---------------------------------------------------------------------------
# CONSTANTS (inlined — standalone page)
# ---------------------------------------------------------------------------
SECTOR_ETFS = [
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK",
    "XLP", "XLRE", "XLU", "XLV", "XLY",
]

REGIME_BUCKETS = {
    "Robust":   (0, 20),
    "Calm":     (20, 40),
    "Neutral":  (40, 60),
    "Elevated": (60, 80),
    "Fragile":  (80, 100.01),
}

REGIME_ORDER = ["Robust", "Calm", "Neutral", "Elevated", "Fragile"]

REGIME_COLORS = {
    "Robust":   "#00CC00",
    "Calm":     "#7FCC00",
    "Neutral":  "#FFD700",
    "Elevated": "#FF8C00",
    "Fragile":  "#CC0000",
}

# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------
def _assign_regime_bucket(score: float) -> str:
    for name, (lo, hi) in REGIME_BUCKETS.items():
        if lo <= score < hi:
            return name
    return "Fragile" if score >= 80 else "Robust"


@st.cache_data(ttl=3600)
def load_fragility_ts():
    if not os.path.exists(CACHE_FRAGILITY):
        return None
    df = pd.read_parquet(CACHE_FRAGILITY)
    return df


@st.cache_data(ttl=3600)
def load_sector_closes():
    if not os.path.exists(CACHE_CLOSES):
        return None
    df = pd.read_parquet(CACHE_CLOSES)
    # Keep only sector ETFs + SPY
    cols = [c for c in SECTOR_ETFS + ["SPY"] if c in df.columns]
    return df[cols]


@st.cache_data(ttl=3600)
def load_sp500_closes():
    if not os.path.exists(CACHE_SP500):
        return None
    return pd.read_parquet(CACHE_SP500)


@st.cache_data(ttl=3600)
def load_spy_close():
    if not os.path.exists(CACHE_SPY_OHLC):
        return None
    df = pd.read_parquet(CACHE_SPY_OHLC)
    return df["Close"]


# ---------------------------------------------------------------------------
# CORE COMPUTATION
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def compute_study(
    closes: pd.DataFrame,
    spy_close: pd.Series,
    frag_ts: pd.DataFrame,
    frag_horizon: str,
    lookback: int,
    fwd_horizon: int,
    n_groups: int,
    excess: bool,
):
    """
    Vectorised leader/laggard event study.

    Returns
    -------
    result_df : DataFrame
        Columns: date, ticker, group, regime, fwd_ret
    """
    # --- fragility regime (lagged 1 day) ---
    if frag_horizon == "Average (21d + 63d)":
        frag = frag_ts[["21d", "63d"]].mean(axis=1)
    else:
        frag = frag_ts[frag_horizon]
    regime_series = frag.shift(1).dropna().apply(_assign_regime_bucket)

    # --- trailing returns (momentum) ---
    trail_ret = closes.pct_change(lookback, fill_method=None)

    # --- forward returns ---
    fwd_ret = closes.pct_change(fwd_horizon, fill_method=None).shift(-fwd_horizon)

    # --- excess vs SPY ---
    if excess:
        spy_fwd = spy_close.pct_change(fwd_horizon, fill_method=None).shift(-fwd_horizon)
        fwd_ret = fwd_ret.sub(spy_fwd, axis=0)

    # --- align dates ---
    common = trail_ret.index.intersection(fwd_ret.index).intersection(regime_series.index)
    trail_ret = trail_ret.loc[common]
    fwd_ret = fwd_ret.loc[common]
    regime_series = regime_series.loc[common]

    # --- cross-sectional percentile rank ---
    pct_rank = trail_ret.rank(axis=1, pct=True) * 100  # 0-100

    # --- assign groups (vectorised) ---
    if n_groups == 10:  # deciles for S&P 500
        # Use qcut-style binning: floor(rank * 10), clamp to 0-9
        decile_num = np.floor(pct_rank.values / 10.0).clip(0, 9).astype(float)
        # Preserve NaN
        decile_num = np.where(np.isnan(pct_rank.values), np.nan, decile_num)
        decile_df = pd.DataFrame(decile_num, index=pct_rank.index, columns=pct_rank.columns)
        # Map 0->D1, 1->D2, ..., 9->D10
        label_map = {float(i): f"D{i+1}" for i in range(10)}
        group_df = decile_df.stack().map(label_map).unstack()
    else:  # 3 groups for sectors
        # Tercile bins using rank thresholds
        def _assign_3(row):
            valid = row.dropna()
            if len(valid) < 3:
                return pd.Series(np.nan, index=row.index)
            q33 = valid.quantile(0.333)
            q67 = valid.quantile(0.667)
            out = pd.Series("Middle 5", index=valid.index)
            out[valid <= q33] = "Bottom 3"
            out[valid >= q67] = "Top 3"
            return out.reindex(row.index)

        group_df = pct_rank.apply(_assign_3, axis=1)

    # --- melt to long form (vectorised) ---
    fwd_long = fwd_ret.stack().rename("fwd_ret")
    group_long = group_df.stack().rename("group")
    merged = pd.concat([fwd_long, group_long], axis=1).dropna()
    merged = merged.reset_index()
    merged.columns = ["date", "ticker", "fwd_ret", "group"]
    merged["regime"] = merged["date"].map(regime_series)
    result_df = merged.dropna(subset=["regime"]).reset_index(drop=True)
    return result_df


@st.cache_data(ttl=3600)
def aggregate_by_regime_group(result_df: pd.DataFrame, metric: str):
    """Pivot table: regime × group → mean or median forward return."""
    agg_func = "mean" if metric == "Mean" else "median"
    pivot = result_df.pivot_table(
        values="fwd_ret", index="regime", columns="group", aggfunc=agg_func
    )
    return pivot * 100  # percent


@st.cache_data(ttl=3600)
def aggregate_counts(result_df: pd.DataFrame):
    """Sample size pivot."""
    pivot = result_df.pivot_table(
        values="fwd_ret", index="regime", columns="group", aggfunc="count"
    )
    return pivot


@st.cache_data(ttl=3600)
def compute_ttest_table(result_df: pd.DataFrame, leader_label: str, laggard_label: str):
    """Welch's t-test: leader vs laggard forward returns per regime."""
    rows = []
    for regime in REGIME_ORDER:
        sub = result_df[result_df["regime"] == regime]
        leaders = sub[sub["group"] == leader_label]["fwd_ret"].dropna() * 100
        laggards = sub[sub["group"] == laggard_label]["fwd_ret"].dropna() * 100
        if len(leaders) < 5 or len(laggards) < 5:
            rows.append({
                "Regime": regime,
                "Leader Mean (%)": np.nan,
                "Laggard Mean (%)": np.nan,
                "Diff (L-Lag)": np.nan,
                "t-stat": np.nan,
                "p-value": np.nan,
                "Sig": "",
            })
            continue
        t_stat, p_val = stats.ttest_ind(leaders, laggards, equal_var=False)
        diff = leaders.mean() - laggards.mean()
        sig = ""
        if p_val < 0.01:
            sig = "***"
        elif p_val < 0.05:
            sig = "**"
        elif p_val < 0.10:
            sig = "*"
        rows.append({
            "Regime": regime,
            "Leader Mean (%)": round(leaders.mean(), 3),
            "Laggard Mean (%)": round(laggards.mean(), 3),
            "Diff (L-Lag)": round(diff, 3),
            "t-stat": round(t_stat, 3),
            "p-value": round(p_val, 4),
            "Sig": sig,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RENDERING
# ---------------------------------------------------------------------------
def render_money_heatmap(pivot: pd.DataFrame, group_order: list, title: str):
    """Annotated heatmap: regime × group."""
    # Reindex to consistent order
    pivot = pivot.reindex(index=REGIME_ORDER, columns=group_order)
    z = pivot.values
    # Annotations
    annot = []
    for i, row in enumerate(z):
        annot_row = []
        for val in row:
            if pd.isna(val):
                annot_row.append("")
            else:
                annot_row.append(f"{val:+.2f}%")
        annot.append(annot_row)

    abs_max = np.nanmax(np.abs(z)) if np.any(~np.isnan(z)) else 1
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=group_order,
        y=REGIME_ORDER,
        colorscale="RdYlGn",
        zmin=-abs_max,
        zmax=abs_max,
        text=annot,
        texttemplate="%{text}",
        textfont=dict(size=12),
        hovertemplate="Regime: %{y}<br>Group: %{x}<br>Return: %{text}<extra></extra>",
        colorbar=dict(title="Fwd Ret %"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Momentum Group",
        yaxis_title="Fragility Regime",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_sample_heatmap(counts: pd.DataFrame, group_order: list):
    """Sample size companion heatmap."""
    counts = counts.reindex(index=REGIME_ORDER, columns=group_order)
    z = counts.values
    annot = []
    for row in z:
        annot.append([str(int(v)) if pd.notna(v) else "" for v in row])

    # Color low-N cells
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=group_order,
        y=REGIME_ORDER,
        colorscale=[[0, "#fff3e0"], [1, "#1565c0"]],
        text=annot,
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="Regime: %{y}<br>Group: %{x}<br>N: %{text}<extra></extra>",
        colorbar=dict(title="N"),
    ))
    fig.update_layout(
        title="Sample Size (N) — cells with N < 30 may be unreliable",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_hedge_answer(ttest_df: pd.DataFrame, fwd_horizon: int, leader_label: str, laggard_label: str):
    """Styled conclusion banner for Elevated+Fragile regimes."""
    stress = ttest_df[ttest_df["Regime"].isin(["Elevated", "Fragile"])]
    leader_mean = stress["Leader Mean (%)"].mean()
    laggard_mean = stress["Laggard Mean (%)"].mean()
    p_vals = stress["p-value"].dropna()
    avg_p = p_vals.mean() if len(p_vals) > 0 else 1.0

    if pd.isna(leader_mean) or pd.isna(laggard_mean):
        st.info("Insufficient data for Elevated/Fragile regimes to draw conclusions.")
        return

    diff = leader_mean - laggard_mean
    if avg_p < 0.10:
        if diff < -0.5:
            verdict = f"**Short leaders** ({leader_label}) — they give back more than laggards"
            color = "#e53935"
        elif diff > 0.5:
            verdict = f"**Short laggards** ({laggard_label}) — they continue to underperform"
            color = "#e53935"
        else:
            verdict = "**No significant edge** — leaders and laggards perform similarly"
            color = "#FFD700"
    else:
        verdict = "**No statistically significant difference** between leaders and laggards"
        color = "#9e9e9e"

    st.markdown(
        f"""<div style="border-left: 4px solid {color}; padding: 12px 16px;
        background: rgba(0,0,0,0.03); border-radius: 4px; margin-bottom: 16px;">
        <strong>Hedge Answer (Elevated + Fragile regimes, {fwd_horizon}d forward):</strong><br>
        Leaders avg <strong>{leader_mean:+.2f}%</strong> vs Laggards avg
        <strong>{laggard_mean:+.2f}%</strong> (avg p={avg_p:.3f})<br>
        {verdict}
        </div>""",
        unsafe_allow_html=True,
    )


def render_distribution_overlay(result_df: pd.DataFrame, leader_label: str, laggard_label: str, fwd_horizon: int):
    """Histogram overlay of leader vs laggard forward returns in stress regimes."""
    stress = result_df[result_df["regime"].isin(["Elevated", "Fragile"])]
    leaders = stress[stress["group"] == leader_label]["fwd_ret"].dropna() * 100
    laggards = stress[stress["group"] == laggard_label]["fwd_ret"].dropna() * 100

    if len(leaders) < 10 or len(laggards) < 10:
        st.info("Insufficient data for distribution overlay in stress regimes.")
        return

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=leaders, name=f"Leaders ({leader_label})",
        opacity=0.6, marker_color="#1565c0",
        nbinsx=40, histnorm="probability density",
    ))
    fig.add_trace(go.Histogram(
        x=laggards, name=f"Laggards ({laggard_label})",
        opacity=0.6, marker_color="#e53935",
        nbinsx=40, histnorm="probability density",
    ))
    fig.update_layout(
        title=f"Forward {fwd_horizon}d Return Distribution — Elevated + Fragile Regimes",
        xaxis_title=f"{fwd_horizon}d Forward Return (%)",
        yaxis_title="Density",
        barmode="overlay",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.98),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_spread_timeseries(
    result_df: pd.DataFrame,
    leader_label: str,
    laggard_label: str,
    fwd_horizon: int,
):
    """Time series of (leader avg fwd ret - laggard avg fwd ret) colored by regime."""
    leaders = (
        result_df[result_df["group"] == leader_label]
        .groupby("date")["fwd_ret"]
        .mean()
        * 100
    )
    laggards = (
        result_df[result_df["group"] == laggard_label]
        .groupby("date")["fwd_ret"]
        .mean()
        * 100
    )
    spread = (leaders - laggards).dropna()
    if len(spread) < 20:
        st.info("Insufficient data for spread time series.")
        return

    # Get regime per date
    regimes_by_date = result_df.drop_duplicates("date").set_index("date")["regime"]

    # Smooth for readability
    spread_smooth = spread.rolling(63, min_periods=21).mean()

    fig = go.Figure()

    # Background regime shading
    for regime in REGIME_ORDER:
        mask = regimes_by_date.reindex(spread_smooth.index) == regime
        dates_in_regime = spread_smooth.index[mask]
        if len(dates_in_regime) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=dates_in_regime,
            y=spread_smooth.loc[dates_in_regime],
            mode="markers",
            marker=dict(size=3, color=REGIME_COLORS[regime]),
            name=regime,
            showlegend=True,
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"Leader minus Laggard Avg {fwd_horizon}d Forward Return (63d smoothed)",
        xaxis_title="Date",
        yaxis_title="Spread (%)",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_fragility_timeseries(frag_ts: pd.DataFrame, spy_close: pd.Series, horizon: str):
    """Dual-axis chart: fragility score (area) + SPY (line). Matches dashboard chart."""
    from plotly.subplots import make_subplots

    h_labels = {"5d": "5-Day", "21d": "21-Day", "63d": "63-Day"}
    if horizon == "Average (21d + 63d)":
        frag = frag_ts[["21d", "63d"]].mean(axis=1).dropna()
        label = "Avg (21d+63d)"
    else:
        frag = frag_ts[horizon].dropna()
        label = h_labels.get(horizon, horizon)

    common = frag.index.intersection(spy_close.dropna().index).sort_values()
    frag = frag.reindex(common)
    spy = spy_close.reindex(common)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=common, y=frag.values,
            name=f"{label} Fragility",
            fill="tozeroy",
            fillcolor="rgba(255, 140, 0, 0.15)",
            line=dict(color="rgba(255, 140, 0, 0.8)", width=1),
            hovertemplate=f"{label} Fragility: %{{y:.1f}}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Regime threshold lines
    for thresh, color, lbl in [
        (50, "rgba(255, 215, 0, 0.4)", "50"),
        (70, "rgba(255, 69, 0, 0.4)", "70"),
        (100, "rgba(139, 0, 0, 0.5)", "100"),
    ]:
        fig.add_hline(
            y=thresh, line_dash="dot", line_color=color, line_width=1,
            annotation_text=lbl, annotation_position="left",
            secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(
            x=common, y=spy.values,
            name="SPY",
            line=dict(color="#4A90D9", width=1.5),
            hovertemplate="SPY: $%{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    y_max = max(105, frag.max() * 1.05) if len(frag) > 0 else 105
    spy_min, spy_max = spy.min() * 0.95, spy.max() * 1.05

    fig.update_layout(
        title=dict(text=f"{label} Fragility Score vs SPY", font=dict(size=13)),
        height=350,
        margin=dict(l=10, r=10, t=35, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
    )
    fig.update_yaxes(title_text="Fragility", range=[0, y_max], secondary_y=False)
    fig.update_yaxes(title_text="SPY", range=[spy_min, spy_max], secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)


def render_current_composition(
    closes: pd.DataFrame,
    lookback: int,
):
    """Show which tickers currently sit in D1 (worst) and D10 (best)."""
    trail_ret = closes.pct_change(lookback, fill_method=None).iloc[-1].dropna()
    if len(trail_ret) < 20:
        st.info("Insufficient data for current composition.")
        return

    pct_rank = trail_ret.rank(pct=True) * 100
    d1 = pct_rank[pct_rank <= 10].sort_values()
    d10 = pct_rank[pct_rank >= 90].sort_values(ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**D1 — Worst Momentum ({lookback}d)**")
        d1_df = pd.DataFrame({
            "Ticker": d1.index,
            f"{lookback}d Return (%)": (trail_ret.loc[d1.index] * 100).round(2).values,
            "Pctile": d1.round(1).values,
        })
        st.dataframe(d1_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown(f"**D10 — Best Momentum ({lookback}d)**")
        d10_df = pd.DataFrame({
            "Ticker": d10.index,
            f"{lookback}d Return (%)": (trail_ret.loc[d10.index] * 100).round(2).values,
            "Pctile": d10.round(1).values,
        })
        st.dataframe(d10_df, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    st.title("Leader vs Laggard Forward Returns by Fragility Regime")
    st.caption(
        "Event study: when the market is fragile, do recent winners or losers "
        "underperform more going forward? Uses cached data from Risk Dashboard V2."
    )

    # --- Load data ---
    frag_ts = load_fragility_ts()
    sector_closes = load_sector_closes()
    spy_close = load_spy_close()

    missing = []
    if frag_ts is None:
        missing.append("rd2_fragility_ts.parquet")
    if sector_closes is None:
        missing.append("rd2_closes.parquet")
    if spy_close is None:
        missing.append("rd2_spy_ohlc.parquet")

    if missing:
        st.warning(
            f"Missing data files: {', '.join(missing)}. "
            "Run **Risk Dashboard V2** first to populate caches."
        )
        st.stop()

    # --- Sidebar controls ---
    st.sidebar.header("Study Parameters")

    frag_horizon = st.sidebar.selectbox(
        "Fragility horizon",
        ["21d", "63d", "Average (21d + 63d)"],
        index=0,
    )

    lookback = st.sidebar.selectbox(
        "Momentum lookback (days)",
        [21, 63, 126],
        index=1,
    )

    fwd_horizon = st.sidebar.selectbox(
        "Forward return horizon (days)",
        [5, 21, 63],
        index=1,
    )

    metric = st.sidebar.radio("Aggregation metric", ["Mean", "Median"])
    excess = st.sidebar.radio("Return type", ["Raw", "Excess vs SPY"]) == "Excess vs SPY"

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Fragility scores are lagged 1 day (yesterday's score → today's regime). "
        "Forward returns may overlap at longer horizons — interpret effective sample size carefully."
    )

    # --- Fragility context chart ---
    with st.expander("Fragility Score Time Series", expanded=True):
        render_fragility_timeseries(frag_ts, spy_close, frag_horizon)

    # --- Tabs ---
    tab_sector, tab_sp500 = st.tabs(["Sector ETFs (11)", "S\u0026P 500 (~505)"])

    # ===========================
    # TAB 1: SECTOR ETFs
    # ===========================
    with tab_sector:
        sector_only = sector_closes[[c for c in SECTOR_ETFS if c in sector_closes.columns]]
        result_sector = compute_study(
            closes=sector_only,
            spy_close=spy_close,
            frag_ts=frag_ts,
            frag_horizon=frag_horizon,
            lookback=lookback,
            fwd_horizon=fwd_horizon,
            n_groups=3,
            excess=excess,
        )

        if len(result_sector) == 0:
            st.warning("No valid observations. Check data alignment.")
            st.stop()

        leader_label_s = "Top 3"
        laggard_label_s = "Bottom 3"
        group_order_s = ["Bottom 3", "Middle 5", "Top 3"]

        # 1. Hedge answer
        ttest_sector = compute_ttest_table(result_sector, leader_label_s, laggard_label_s)
        render_hedge_answer(ttest_sector, fwd_horizon, leader_label_s, laggard_label_s)

        # 2. Money chart heatmap
        pivot_sector = aggregate_by_regime_group(result_sector, metric)
        render_money_heatmap(
            pivot_sector, group_order_s,
            f"Sector ETF {metric} {fwd_horizon}d Forward Return (%) by Regime & Momentum Group",
        )

        # 3. Sample size
        with st.expander("Sample Size (N)"):
            counts_sector = aggregate_counts(result_sector)
            render_sample_heatmap(counts_sector, group_order_s)

        # 4. Comparison table
        st.subheader("Leader vs Laggard Statistical Comparison")
        st.dataframe(
            ttest_sector.style.format({
                "Leader Mean (%)": "{:+.3f}",
                "Laggard Mean (%)": "{:+.3f}",
                "Diff (L-Lag)": "{:+.3f}",
                "t-stat": "{:.3f}",
                "p-value": "{:.4f}",
            }, na_rep="—"),
            hide_index=True,
            use_container_width=True,
        )

        # 5. Distribution overlay
        render_distribution_overlay(result_sector, leader_label_s, laggard_label_s, fwd_horizon)

        # 6. Spread time series
        render_spread_timeseries(result_sector, leader_label_s, laggard_label_s, fwd_horizon)

    # ===========================
    # TAB 2: S&P 500
    # ===========================
    with tab_sp500:
        sp500_closes = load_sp500_closes()
        if sp500_closes is None:
            st.warning(
                "Missing rd2_sp500_closes.parquet. Run Risk Dashboard V2 first."
            )
            st.stop()

        result_sp = compute_study(
            closes=sp500_closes,
            spy_close=spy_close,
            frag_ts=frag_ts,
            frag_horizon=frag_horizon,
            lookback=lookback,
            fwd_horizon=fwd_horizon,
            n_groups=10,
            excess=excess,
        )

        if len(result_sp) == 0:
            st.warning("No valid observations. Check data alignment.")
            st.stop()

        leader_label_d = "D10"
        laggard_label_d = "D1"
        group_order_d = [f"D{i}" for i in range(1, 11)]

        # 1. Hedge answer
        ttest_sp = compute_ttest_table(result_sp, leader_label_d, laggard_label_d)
        render_hedge_answer(ttest_sp, fwd_horizon, leader_label_d, laggard_label_d)

        # 2. Money chart heatmap
        pivot_sp = aggregate_by_regime_group(result_sp, metric)
        render_money_heatmap(
            pivot_sp, group_order_d,
            f"S&P 500 {metric} {fwd_horizon}d Forward Return (%) by Regime & Decile",
        )

        # 3. Sample size
        with st.expander("Sample Size (N)"):
            counts_sp = aggregate_counts(result_sp)
            render_sample_heatmap(counts_sp, group_order_d)

        # 4. Comparison table
        st.subheader("Leader vs Laggard Statistical Comparison")
        st.dataframe(
            ttest_sp.style.format({
                "Leader Mean (%)": "{:+.3f}",
                "Laggard Mean (%)": "{:+.3f}",
                "Diff (L-Lag)": "{:+.3f}",
                "t-stat": "{:.3f}",
                "p-value": "{:.4f}",
            }, na_rep="—"),
            hide_index=True,
            use_container_width=True,
        )

        # 5. Distribution overlay
        render_distribution_overlay(result_sp, leader_label_d, laggard_label_d, fwd_horizon)

        # 6. Spread time series
        render_spread_timeseries(result_sp, leader_label_d, laggard_label_d, fwd_horizon)

        # 7. Current composition
        st.subheader(f"Current D1 / D10 Composition ({lookback}d lookback)")
        render_current_composition(sp500_closes, lookback)


main()
