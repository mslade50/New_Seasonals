import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import sys
import os
import plotly.graph_objects as go
from pathlib import Path

# -----------------------------------------------------------------------------
# PAGE CONFIG (must be first Streamlit command)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Risk Dashboard", page_icon="🛡️", layout="wide")

# -----------------------------------------------------------------------------
# IMPORT FROM PROJECT ROOT
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from abs_return_dispersion import download_and_compute_cached, SP500_TICKERS
except ImportError:
    st.error("abs_return_dispersion.py not found in project root.")
    st.stop()


# -----------------------------------------------------------------------------
# CACHING HELPERS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_dispersion_data(cache_path: str, start_date: str, window: int, max_cache_age_days: int):
    """Load or compute dispersion data with caching."""
    return download_and_compute_cached(
        cache_path=cache_path,
        start_date=start_date,
        window=window,
        max_cache_age_days=max_cache_age_days,
    )


@st.cache_data(ttl=3600)
def load_spy_data(start_date: str):
    """Download SPY close prices for forward return calculations."""
    spy = yf.download("SPY", start=start_date, auto_adjust=True, progress=False)
    if spy.empty:
        return None
    # Handle yfinance MultiIndex columns
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.columns = [c.capitalize() for c in spy.columns]
    # Normalize timezone
    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)
    return spy


def get_regime_label(percentile: float) -> tuple:
    """Return regime label and color based on percentile rank."""
    if percentile >= 95:
        return "EXTREME", "#ff4444"
    elif percentile >= 80:
        return "HIGH", "#ff8c00"
    elif percentile >= 50:
        return "ELEVATED", "#ffd93d"
    else:
        return "NORMAL", "#4ade80"


def compute_forward_returns(spy_df: pd.DataFrame, disp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forward returns and realized volatility for SPY at various horizons.
    Merge with dispersion data for bucketed analysis.
    """
    if spy_df is None or spy_df.empty:
        return pd.DataFrame()

    close = spy_df["Close"].copy()
    daily_ret = close.pct_change()

    horizons = [5, 10, 21, 63]
    fwd_data = pd.DataFrame(index=close.index)

    for h in horizons:
        # Forward return: pct_change shifted negative to align future return with today
        fwd_data[f"fwd_{h}d_ret"] = close.pct_change(h).shift(-h)
        # Forward realized vol: rolling std of daily returns, shifted forward
        fwd_data[f"fwd_{h}d_rvol"] = daily_ret.rolling(h).std().shift(-h) * np.sqrt(252)

    # Merge with dispersion rank
    merged = fwd_data.join(disp_df[["dispersion_rank"]], how="inner")
    merged = merged.dropna(subset=["dispersion_rank"])

    return merged


def bucket_analysis(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket historical data by dispersion percentile and compute summary stats.
    """
    if merged_df.empty:
        return pd.DataFrame()

    # Define percentile bins
    bins = [0, 25, 50, 75, 90, 95, 100]
    labels = ["0-25", "25-50", "50-75", "75-90", "90-95", "95-100"]

    merged_df = merged_df.copy()
    merged_df["bucket"] = pd.cut(
        merged_df["dispersion_rank"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    results = []
    for bucket in labels:
        bucket_data = merged_df[merged_df["bucket"] == bucket]
        n = len(bucket_data)
        if n == 0:
            continue

        row = {
            "Dispersion Regime": bucket,
            "N (days)": n,
            "Fwd 5d Avg": bucket_data["fwd_5d_ret"].mean() * 100 if "fwd_5d_ret" in bucket_data else np.nan,
            "Fwd 10d Avg": bucket_data["fwd_10d_ret"].mean() * 100 if "fwd_10d_ret" in bucket_data else np.nan,
            "Fwd 21d Avg": bucket_data["fwd_21d_ret"].mean() * 100 if "fwd_21d_ret" in bucket_data else np.nan,
            "Fwd 63d Avg": bucket_data["fwd_63d_ret"].mean() * 100 if "fwd_63d_ret" in bucket_data else np.nan,
            "Fwd 21d RVol": bucket_data["fwd_21d_rvol"].mean() * 100 if "fwd_21d_rvol" in bucket_data else np.nan,
            "Fwd 5d Win%": (bucket_data["fwd_5d_ret"] > 0).mean() * 100 if "fwd_5d_ret" in bucket_data else np.nan,
        }
        results.append(row)

    return pd.DataFrame(results)


def get_current_bucket(percentile: float) -> str:
    """Determine which bucket the current percentile falls into."""
    if percentile <= 25:
        return "0-25"
    elif percentile <= 50:
        return "25-50"
    elif percentile <= 75:
        return "50-75"
    elif percentile <= 90:
        return "75-90"
    elif percentile <= 95:
        return "90-95"
    else:
        return "95-100"


def style_bucket_table(df: pd.DataFrame, current_bucket: str) -> pd.io.formats.style.Styler:
    """Apply styling to the bucket analysis table, highlighting current regime."""
    def highlight_row(row):
        if row["Dispersion Regime"] == current_bucket:
            return ["background-color: rgba(255, 217, 61, 0.3)"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(highlight_row, axis=1)
    styled = styled.format({
        "N (days)": "{:,.0f}",
        "Fwd 5d Avg": "{:.2f}%",
        "Fwd 10d Avg": "{:.2f}%",
        "Fwd 21d Avg": "{:.2f}%",
        "Fwd 63d Avg": "{:.2f}%",
        "Fwd 21d RVol": "{:.1f}%",
        "Fwd 5d Win%": "{:.1f}%",
    })
    return styled


def create_dispersion_chart(disp_df: pd.DataFrame, view: str = "dispersion") -> go.Figure:
    """Create interactive Plotly chart for dispersion data."""
    df = disp_df.dropna(subset=["dispersion"]).copy()

    # Compute percentile thresholds
    p95 = df["dispersion"].quantile(0.95)
    p99 = df["dispersion"].quantile(0.99)

    fig = go.Figure()

    if view == "dispersion":
        # Main dispersion area chart
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["dispersion"] * 100,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(58, 58, 92, 0.5)",
            line=dict(color="#8888cc", width=1),
            name="Dispersion %",
            hovertemplate="%{x|%Y-%m-%d}<br>Dispersion: %{y:.1f}%<extra></extra>"
        ))

        # 95th percentile line
        fig.add_hline(
            y=p95 * 100,
            line_dash="dash",
            line_color="orange",
            line_width=1,
            annotation_text="95th pctl",
            annotation_position="right"
        )

        # 99th percentile line
        fig.add_hline(
            y=p99 * 100,
            line_dash="dash",
            line_color="red",
            line_width=1,
            annotation_text="99th pctl",
            annotation_position="right"
        )

        # Zero line
        fig.add_hline(y=0, line_color="#444", line_width=0.5)

        # Latest reading marker (red diamond)
        latest = df.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[df.index[-1]],
            y=[latest["dispersion"] * 100],
            mode="markers",
            marker=dict(symbol="diamond", size=12, color="red"),
            name="Latest",
            hovertemplate=f"Latest: {latest['dispersion']*100:.1f}%<extra></extra>"
        ))

        y_title = "Dispersion %"

    else:  # percentile rank view
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["dispersion_rank"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(42, 74, 122, 0.5)",
            line=dict(color="#4a8adf", width=1),
            name="Percentile Rank",
            hovertemplate="%{x|%Y-%m-%d}<br>Rank: %{y:.0f}th<extra></extra>"
        ))

        # Reference lines at 80 and 95
        fig.add_hline(y=80, line_dash="dash", line_color="orange", line_width=1)
        fig.add_hline(y=95, line_dash="dash", line_color="red", line_width=1)

        # Latest marker
        latest = df.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[df.index[-1]],
            y=[latest["dispersion_rank"]],
            mode="markers",
            marker=dict(symbol="diamond", size=12, color="red"),
            name="Latest",
            hovertemplate=f"Latest: {latest['dispersion_rank']:.0f}th<extra></extra>"
        ))

        y_title = "Percentile Rank"

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            title=""
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            title=y_title
        ),
        margin=dict(l=50, r=50, t=30, b=30),
        showlegend=False,
        height=400,
    )

    return fig


# =============================================================================
# MAIN PAGE
# =============================================================================
st.title("🛡️ Market Risk Dashboard")
st.markdown("**S&P 500 Absolute Return Dispersion** — Mean |Constituent Return| minus |Index Return|")

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    window = st.slider(
        "Return Window (days)",
        min_value=10,
        max_value=63,
        value=21,
        step=1,
        help="Lookback period for computing returns (default: 21 trading days = ~1 month)"
    )

    start_date = st.date_input(
        "Start Date",
        value=datetime.date(1998, 1, 1),
        min_value=datetime.date(1990, 1, 1),
        max_value=datetime.date.today(),
        help="Historical start date for analysis"
    )

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
try:
    with st.spinner("Loading dispersion data..."):
        cache_path = os.path.join(parent_dir, "data", "sp500_dispersion_prices.parquet")
        disp_df = load_dispersion_data(
            cache_path=cache_path,
            start_date=start_date.strftime("%Y-%m-%d"),
            window=window,
            max_cache_age_days=1
        )

        spy_df = load_spy_data(start_date.strftime("%Y-%m-%d"))

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.warning("Attempting to use any available cached data...")
    disp_df = pd.DataFrame()
    spy_df = None

# Validate data
if disp_df.empty:
    st.error("No dispersion data available. Please check your network connection and try refreshing.")
    st.stop()

disp_clean = disp_df.dropna(subset=["dispersion", "dispersion_rank"])

if len(disp_clean) < 252:
    st.warning("Less than 1 year of data available. Percentile ranks may be unreliable.")

# Get latest reading
latest = disp_clean.iloc[-1]
latest_date = disp_clean.index[-1]
regime_label, regime_color = get_regime_label(latest["dispersion_rank"])

# =============================================================================
# SECTION 1: CURRENT READING CALLOUT
# =============================================================================
st.markdown("---")
st.subheader("Current Market Reading")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Dispersion",
        value=f"{latest['dispersion'] * 100:.1f}%",
    )

with col2:
    rank_val = latest["dispersion_rank"]
    st.metric(
        label="Percentile Rank",
        value=f"{rank_val:.0f}th",
    )

with col3:
    st.metric(
        label="Avg Stock |Move|",
        value=f"{latest['avg_abs_ret'] * 100:.1f}%",
    )

with col4:
    st.metric(
        label="SPY |Move|",
        value=f"{latest['index_abs_ret'] * 100:.1f}%",
    )

with col5:
    st.metric(
        label="Constituents",
        value=f"{int(latest['n_constituents'])}",
    )

# Regime indicator
st.markdown(
    f"""
    <div style="
        background-color: {regime_color}22;
        border: 2px solid {regime_color};
        border-radius: 8px;
        padding: 12px 20px;
        margin: 10px 0;
        display: inline-block;
    ">
        <span style="font-size: 14px; color: #888;">Regime:</span>
        <span style="font-size: 20px; font-weight: bold; color: {regime_color}; margin-left: 10px;">
            {regime_label}
        </span>
        <span style="font-size: 12px; color: #888; margin-left: 15px;">
            as of {latest_date.strftime('%Y-%m-%d')}
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# SECTION 2: DISPERSION CHART
# =============================================================================
st.markdown("---")
st.subheader("Historical Dispersion")

chart_view = st.radio(
    "View",
    options=["Dispersion %", "Percentile Rank"],
    horizontal=True,
    label_visibility="collapsed"
)

view_key = "dispersion" if chart_view == "Dispersion %" else "rank"
fig = create_dispersion_chart(disp_clean, view=view_key)
st.plotly_chart(fig, use_container_width=True)

# Historical stats
st.markdown("**Historical Distribution**")
stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)

with stats_col1:
    st.metric("Mean", f"{disp_clean['dispersion'].mean() * 100:.1f}%")
with stats_col2:
    st.metric("Median", f"{disp_clean['dispersion'].median() * 100:.1f}%")
with stats_col3:
    st.metric("Std Dev", f"{disp_clean['dispersion'].std() * 100:.1f}%")
with stats_col4:
    st.metric("95th Pctl", f"{disp_clean['dispersion'].quantile(0.95) * 100:.1f}%")
with stats_col5:
    st.metric("99th Pctl", f"{disp_clean['dispersion'].quantile(0.99) * 100:.1f}%")

# =============================================================================
# SECTION 3: FORWARD RETURNS ANALYSIS TABLE
# =============================================================================
st.markdown("---")
st.subheader("Forward Returns Analysis by Dispersion Regime")

if spy_df is not None and not spy_df.empty:
    merged = compute_forward_returns(spy_df, disp_clean)

    if not merged.empty:
        bucket_df = bucket_analysis(merged)

        if not bucket_df.empty:
            current_bucket = get_current_bucket(latest["dispersion_rank"])
            styled_df = style_bucket_table(bucket_df, current_bucket)

            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Forward returns are computed on SPY from 1998-present. "
                "Realized vol is annualized. The current dispersion regime row is highlighted."
            )
        else:
            st.warning("Insufficient data for bucket analysis.")
    else:
        st.warning("Could not compute forward returns. Insufficient overlapping data.")
else:
    st.warning("SPY data not available. Forward returns analysis skipped.")

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"Data: {disp_clean.index[0].strftime('%Y-%m-%d')} to {disp_clean.index[-1].strftime('%Y-%m-%d')} "
    f"({len(disp_clean):,} trading days) | "
    f"Source: S&P 500 constituents via yfinance | "
    f"Methodology: Nomura Vol"
)
