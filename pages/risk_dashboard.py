import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import time
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
# CONSTANTS
# -----------------------------------------------------------------------------
# Sector ETFs for leadership analysis
LEADERSHIP_ETFS = [
    'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK',
    'XLP', 'XLU', 'XLV', 'XLY', 'SMH', 'XBI',
    'XHB', 'KRE', 'XME', 'XOP', 'XRT', 'XLRE', 'VNQ'
]

# Defensive/commodity names to flag when leading
DEFENSIVE_LEADERS = ['XLE', 'XLP', 'XLV']


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


# -----------------------------------------------------------------------------
# EVENT STUDY HELPERS (Section 5)
# -----------------------------------------------------------------------------
def find_crossing_dates(disp_df: pd.DataFrame, threshold: float, cooldown_days: int = 21) -> list:
    """
    Find dates where dispersion_rank crosses above threshold from below.

    A crossing = dispersion_rank >= threshold today AND dispersion_rank < threshold yesterday.
    After a crossing is detected, skip subsequent crossings within cooldown_days calendar days.

    Returns list of crossing dates (DatetimeIndex-compatible).
    """
    rank = disp_df['dispersion_rank'].dropna()

    above = rank >= threshold
    below = rank < threshold

    # Crossing = above today AND below yesterday
    crossings = above & below.shift(1)

    raw_dates = rank[crossings].index.tolist()

    # Enforce cooldown: skip crossings within cooldown_days of last accepted crossing
    filtered = []
    last_accepted = None
    for d in raw_dates:
        if last_accepted is None or (d - last_accepted).days >= cooldown_days:
            filtered.append(d)
            last_accepted = d

    return filtered


def collect_post_crossing_trades(
    sig_df: pd.DataFrame,
    crossing_dates: list,
    forward_days: int
) -> pd.DataFrame:
    """
    For each crossing date, collect trades entered within the next `forward_days` trading days.

    Uses trading day calendar (sig_df's own dates) rather than calendar days.
    """
    if not crossing_dates:
        return pd.DataFrame()

    # Build a set of all trade entry dates for fast lookup
    sig_df = sig_df.copy()
    if 'Date_normalized' not in sig_df.columns:
        sig_df['Date_normalized'] = pd.to_datetime(sig_df['Date']).dt.normalize()

    # Get sorted unique trading dates from sig_df
    all_trade_dates = sorted(sig_df['Date_normalized'].unique())

    collected = []

    for cross_date in crossing_dates:
        cross_date = pd.Timestamp(cross_date).normalize()

        # Find trade dates within forward_days TRADING days after crossing
        # Use forward_days * 2 calendar days as outer bound to handle weekends/holidays
        future_dates = [d for d in all_trade_dates
                        if d > cross_date and d <= cross_date + pd.Timedelta(days=forward_days * 2)]

        # Take only the first `forward_days` trading days
        future_trading_days = future_dates[:forward_days]

        if not future_trading_days:
            continue

        # Collect trades entered on these dates
        mask = sig_df['Date_normalized'].isin(future_trading_days)
        window_trades = sig_df[mask].copy()
        window_trades['crossing_date'] = cross_date
        window_trades['crossing_label'] = f"{cross_date.strftime('%Y-%m-%d')}"
        collected.append(window_trades)

    if not collected:
        return pd.DataFrame()

    return pd.concat(collected, ignore_index=True)


def compute_event_study(sig_df, disp_df, threshold, forward_windows=[5, 10, 21]):
    """
    Full event study for a single threshold.

    Returns a dict with:
    - n_events: number of crossing events
    - crossing_dates: list of dates
    - per-window metrics: {5: {trades, avg_r, win_rate, pf}, 10: {...}, 21: {...}}
    """
    crossing_dates = find_crossing_dates(disp_df, threshold)

    # R-Multiple for the full sig_df (needed for all windows)
    sig_df = sig_df.copy()
    sig_df['R_Multiple'] = sig_df['PnL'] / sig_df['Risk $']

    result = {
        'n_events': len(crossing_dates),
        'crossing_dates': crossing_dates,
        'windows': {}
    }

    for window in forward_windows:
        post_trades = collect_post_crossing_trades(sig_df, crossing_dates, window)

        if post_trades.empty:
            result['windows'][window] = {
                'trades': 0, 'avg_r': 0, 'win_rate': 0, 'profit_factor': 0
            }
            continue

        post_trades['R_Multiple'] = post_trades['PnL'] / post_trades['Risk $']

        n = len(post_trades)
        winners = post_trades[post_trades['PnL'] > 0]
        losers = post_trades[post_trades['PnL'] <= 0]

        gross_profit = winners['PnL'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['PnL'].sum()) if len(losers) > 0 else 0

        result['windows'][window] = {
            'trades': n,
            'avg_r': post_trades['R_Multiple'].mean(),
            'win_rate': len(winners) / n if n > 0 else 0,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 999
        }

    return result


def build_event_study_table(sig_df, disp_df, thresholds=[90, 95], forward_windows=[5, 10, 21]):
    """
    Build the summary DataFrame comparing post-crossing performance to baseline.
    """
    # Baseline: all trades
    sig_df_copy = sig_df.copy()
    sig_df_copy['R_Multiple'] = sig_df_copy['PnL'] / sig_df_copy['Risk $']

    baseline_n = len(sig_df_copy)
    baseline_avg_r = sig_df_copy['R_Multiple'].mean()
    baseline_wr = (sig_df_copy['PnL'] > 0).mean()

    rows = []

    for threshold in thresholds:
        study = compute_event_study(sig_df, disp_df, threshold, forward_windows)

        row = {
            'Signal': f'Crosses {threshold}th pctl',
            'Events': study['n_events'],
        }

        for w in forward_windows:
            metrics = study['windows'].get(w, {})
            row[f'Trades ({w}d)'] = metrics.get('trades', 0)
            row[f'Avg R ({w}d)'] = metrics.get('avg_r', 0)
            row[f'Win% ({w}d)'] = metrics.get('win_rate', 0)

        rows.append(row)

    # Baseline row
    baseline_row = {
        'Signal': 'Baseline (all trades)',
        'Events': '—',
    }
    for w in forward_windows:
        baseline_row[f'Trades ({w}d)'] = baseline_n
        baseline_row[f'Avg R ({w}d)'] = baseline_avg_r
        baseline_row[f'Win% ({w}d)'] = baseline_wr

    rows.append(baseline_row)

    return pd.DataFrame(rows)


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


# -----------------------------------------------------------------------------
# SECTOR LEADERSHIP HELPERS (Section 6)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def download_sector_prices(start_date: str):
    """Download sector ETF + SPY close prices."""
    tickers = LEADERSHIP_ETFS + ['SPY']
    raw = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)

    if raw.empty:
        return None

    # Handle yfinance MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close'].copy()
    else:
        prices = raw[['Close']].copy()
        prices.columns = ['SPY']

    # Normalize column names
    prices.columns = [str(c).strip().upper() for c in prices.columns]

    # Normalize timezone
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    return prices


def compute_sector_leadership(prices: pd.DataFrame, windows: list = [5, 10, 21]):
    """
    For each day and each lookback window, compute trailing returns for all sector ETFs
    and identify the leader (best performer).

    Returns a DataFrame with columns:
      - leader_{w}d: ticker of top-performing sector ETF over trailing w days
      - leader_{w}d_ret: the return of that leader
      - defensive_leader_{w}d: True if leader is in DEFENSIVE_LEADERS
    """
    sector_cols = [c for c in prices.columns if c in LEADERSHIP_ETFS]

    result = pd.DataFrame(index=prices.index)

    for w in windows:
        # Trailing returns for each sector ETF
        returns = prices[sector_cols].pct_change(w)

        # Leader = column with max return each day
        result[f'leader_{w}d'] = returns.idxmax(axis=1)
        result[f'leader_{w}d_ret'] = returns.max(axis=1)

        # Flag if leader is defensive
        result[f'defensive_leader_{w}d'] = result[f'leader_{w}d'].isin(DEFENSIVE_LEADERS)

    return result


def compute_leadership_forward_returns(
    prices: pd.DataFrame,
    leadership_df: pd.DataFrame,
    lookback_window: int,
    forward_windows: list = [5, 10, 21]
):
    """
    When XLE/XLP/XLV is the trailing leader vs when they're not,
    what are SPX forward returns?

    Returns a DataFrame with one row per condition (XLE leads, XLP leads, XLU leads,
    Any defensive leads, Non-defensive leads).
    """
    if 'SPY' not in prices.columns:
        return pd.DataFrame()

    spy = prices['SPY']

    # Compute SPY forward returns
    fwd_rets = {}
    for fw in forward_windows:
        fwd_rets[fw] = spy.pct_change(fw).shift(-fw)  # shift negative = forward

    leader_col = f'leader_{lookback_window}d'

    rows = []

    # Individual defensive leaders
    for ticker in DEFENSIVE_LEADERS:
        mask = leadership_df[leader_col] == ticker
        n_days = mask.sum()

        if n_days == 0:
            continue

        row = {'Leader': ticker, 'Days Leading': n_days}

        for fw in forward_windows:
            fw_subset = fwd_rets[fw][mask].dropna()
            row[f'SPX Fwd {fw}d Avg'] = fw_subset.mean() if len(fw_subset) > 0 else np.nan
            row[f'SPX Fwd {fw}d Win%'] = (fw_subset > 0).mean() if len(fw_subset) > 0 else np.nan

        rows.append(row)

    # Any defensive leader
    def_mask = leadership_df[f'defensive_leader_{lookback_window}d'] == True
    n_def = def_mask.sum()

    row_any_def = {'Leader': 'Any XLE/XLP/XLV', 'Days Leading': n_def}
    for fw in forward_windows:
        fw_subset = fwd_rets[fw][def_mask].dropna()
        row_any_def[f'SPX Fwd {fw}d Avg'] = fw_subset.mean() if len(fw_subset) > 0 else np.nan
        row_any_def[f'SPX Fwd {fw}d Win%'] = (fw_subset > 0).mean() if len(fw_subset) > 0 else np.nan
    rows.append(row_any_def)

    # Non-defensive (baseline comparison)
    nondef_mask = ~def_mask & leadership_df[leader_col].notna()
    n_nondef = nondef_mask.sum()

    row_nondef = {'Leader': 'Other sector leads', 'Days Leading': n_nondef}
    for fw in forward_windows:
        fw_subset = fwd_rets[fw][nondef_mask].dropna()
        row_nondef[f'SPX Fwd {fw}d Avg'] = fw_subset.mean() if len(fw_subset) > 0 else np.nan
        row_nondef[f'SPX Fwd {fw}d Win%'] = (fw_subset > 0).mean() if len(fw_subset) > 0 else np.nan
    rows.append(row_nondef)

    return pd.DataFrame(rows)


def compute_leadership_strategy_performance(
    sig_df: pd.DataFrame,
    leadership_df: pd.DataFrame,
    lookback_window: int
):
    """
    Strategy performance bucketed by sector leadership at time of trade entry.

    Returns DataFrame: one row per leader condition with Trades, Avg R, Win Rate, Profit Factor.
    """
    sig = sig_df.copy()
    if 'Date_normalized' not in sig.columns:
        sig['Date_normalized'] = pd.to_datetime(sig['Date']).dt.normalize()
    sig['R_Multiple'] = sig['PnL'] / sig['Risk $']

    leader_col = f'leader_{lookback_window}d'
    def_col = f'defensive_leader_{lookback_window}d'

    # Map leadership to each trade's entry date
    leader_lookup = leadership_df[leader_col].copy()
    leader_lookup.index = leader_lookup.index.normalize()
    def_lookup = leadership_df[def_col].copy()
    def_lookup.index = def_lookup.index.normalize()

    sig['leader'] = sig['Date_normalized'].map(leader_lookup)
    sig['defensive_leader'] = sig['Date_normalized'].map(def_lookup)

    rows = []

    def calc_row(label, subset):
        if subset.empty:
            return None
        n = len(subset)
        winners = subset[subset['PnL'] > 0]
        losers = subset[subset['PnL'] <= 0]
        gross_profit = winners['PnL'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['PnL'].sum()) if len(losers) > 0 else 0
        return {
            'Condition': label,
            'Trades': n,
            'Avg R': subset['R_Multiple'].mean(),
            'Win Rate': len(winners) / n,
            'Profit Factor': gross_profit / gross_loss if gross_loss > 0 else 999,
        }

    # Individual defensive leaders
    for ticker in DEFENSIVE_LEADERS:
        result = calc_row(f'{ticker} leading', sig[sig['leader'] == ticker])
        if result:
            rows.append(result)

    # Any defensive
    result = calc_row('Any XLE/XLP/XLV leading', sig[sig['defensive_leader'] == True])
    if result:
        rows.append(result)

    # Non-defensive baseline
    nondef = sig[(sig['defensive_leader'] == False) & sig['leader'].notna()]
    result = calc_row('Other sector leads', nondef)
    if result:
        rows.append(result)

    # Overall baseline
    result = calc_row('All trades (baseline)', sig[sig['leader'].notna()])
    if result:
        rows.append(result)

    return pd.DataFrame(rows)


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

# =============================================================================
# SECTION 4: STRATEGY PERFORMANCE BY DISPERSION REGIME
# =============================================================================
st.markdown("---")
st.subheader("📊 Strategy Performance by Dispersion Regime")


def load_cached_sig_df():
    """Load cached backtest results from strat_backtester.py."""
    cache_path = os.path.join(parent_dir, "data", "backtest_sig_df.parquet")
    if not os.path.exists(cache_path):
        return None, None

    mod_time = os.path.getmtime(cache_path)
    cache_date = datetime.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")

    sig_df = pd.read_parquet(cache_path)

    # Ensure Date column is datetime
    if 'Date' in sig_df.columns:
        sig_df['Date'] = pd.to_datetime(sig_df['Date'])

    return sig_df, cache_date


def compute_regime_performance(sig_df_with_disp):
    """Bucket trades by dispersion regime and compute aggregate metrics."""

    bins = [0, 25, 50, 75, 90, 95, 100.01]  # 100.01 to include rank=100
    labels = ['0-25 (Low)', '25-50', '50-75', '75-90', '90-95 (High)', '95-100 (Extreme)']

    df = sig_df_with_disp.copy()
    df['regime_bucket'] = pd.cut(
        df['dispersion_rank'],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    )

    # R-Multiple
    df['R_Multiple'] = df['PnL'] / df['Risk $']

    results = []
    total_trades = len(df)

    for label in labels:
        bucket = df[df['regime_bucket'] == label]
        if bucket.empty:
            results.append({
                'Dispersion Regime': label,
                'Trades': 0, 'Avg R': 0, 'Win Rate': 0,
                'Profit Factor': 0, 'Avg PnL': 0, '% of Trades': 0
            })
            continue

        n = len(bucket)
        winners = bucket[bucket['PnL'] > 0]
        losers = bucket[bucket['PnL'] <= 0]

        gross_profit = winners['PnL'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['PnL'].sum()) if len(losers) > 0 else 0

        results.append({
            'Dispersion Regime': label,
            'Trades': n,
            'Avg R': bucket['R_Multiple'].mean(),
            'Win Rate': len(winners) / n,
            'Profit Factor': gross_profit / gross_loss if gross_loss > 0 else 999,
            'Avg PnL': bucket['PnL'].mean(),
            '% of Trades': n / total_trades
        })

    return pd.DataFrame(results)


def get_current_regime_label(rank):
    """Map percentile rank to regime label."""
    if rank <= 25:
        return '0-25 (Low)'
    elif rank <= 50:
        return '25-50'
    elif rank <= 75:
        return '50-75'
    elif rank <= 90:
        return '75-90'
    elif rank <= 95:
        return '90-95 (High)'
    else:
        return '95-100 (Extreme)'


def highlight_current_regime(row, current_regime):
    """Style function to highlight the current regime row."""
    if row['Dispersion Regime'] == current_regime:
        return ['background-color: rgba(255, 217, 61, 0.2)'] * len(row)
    return [''] * len(row)


# Load cached backtest data
sig_df, cache_date = load_cached_sig_df()

if sig_df is None:
    st.info(
        "No cached backtest data found. Run a backtest in the **Strategy Backtester** page first — "
        "results will automatically appear here."
    )
else:
    st.caption(f"Using backtest cached on {cache_date} · {len(sig_df):,} trades")

    # Merge with dispersion data
    # Create a date → dispersion_rank lookup (normalize to date-only for clean merge)
    disp_lookup = disp_clean['dispersion_rank'].copy()
    disp_lookup.index = disp_lookup.index.normalize()

    sig_df['Date_normalized'] = pd.to_datetime(sig_df['Date']).dt.normalize()
    sig_df['dispersion_rank'] = sig_df['Date_normalized'].map(disp_lookup)

    # Drop trades where we don't have dispersion data (pre-1998 or gaps)
    sig_df_with_disp = sig_df.dropna(subset=['dispersion_rank'])

    # Coverage stats
    coverage = len(sig_df_with_disp) / len(sig_df) * 100
    st.caption(
        f"{len(sig_df_with_disp):,} of {len(sig_df):,} trades matched to dispersion data "
        f"({coverage:.0f}% coverage)"
    )

    if not sig_df_with_disp.empty:
        # Compute regime performance
        regime_df = compute_regime_performance(sig_df_with_disp)

        # Determine current regime for highlighting
        current_rank = disp_clean['dispersion_rank'].dropna().iloc[-1]
        current_regime = get_current_regime_label(current_rank)

        # Style and display
        styled = regime_df.style.apply(
            lambda row: highlight_current_regime(row, current_regime), axis=1
        ).format({
            'Trades': '{:,}',
            'Avg R': '{:+.3f}',
            'Win Rate': '{:.1%}',
            'Profit Factor': '{:.2f}',
            'Avg PnL': '${:,.0f}',
            '% of Trades': '{:.1%}'
        })

        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.caption(
            "Trades bucketed by dispersion percentile rank at entry. "
            "Current regime row highlighted. "
            "Source: strat_backtester.py cached results merged with abs_return_dispersion.py."
        )

        # Strategy-level breakdown
        with st.expander("📋 Breakdown by Strategy"):
            strategies_in_data = sorted(sig_df_with_disp['Strategy'].unique().tolist())
            selected_strat = st.selectbox(
                "Strategy",
                ['All Strategies'] + strategies_in_data,
                key="regime_strat_select"
            )

            if selected_strat != 'All Strategies':
                filtered = sig_df_with_disp[sig_df_with_disp['Strategy'] == selected_strat]
            else:
                filtered = sig_df_with_disp

            if not filtered.empty:
                strat_regime_df = compute_regime_performance(filtered)

                styled_strat = strat_regime_df.style.apply(
                    lambda row: highlight_current_regime(row, current_regime), axis=1
                ).format({
                    'Trades': '{:,}',
                    'Avg R': '{:+.3f}',
                    'Win Rate': '{:.1%}',
                    'Profit Factor': '{:.2f}',
                    'Avg PnL': '${:,.0f}',
                    '% of Trades': '{:.1%}'
                })

                st.dataframe(styled_strat, use_container_width=True, hide_index=True)
            else:
                st.warning("No trades found for selected strategy.")
    else:
        st.warning("No trades could be matched to dispersion data. Check date ranges.")

# =============================================================================
# SECTION 5: DISPERSION CROSSING EVENT STUDY
# =============================================================================
st.markdown("---")
st.subheader("⚡ Dispersion Crossing Event Study")
st.caption("When dispersion spikes above a threshold, does strategy performance degrade in the forward window?")

# Reuse sig_df and sig_df_with_disp from Section 4
# Use try/except since sig_df_with_disp is only defined if sig_df loaded successfully
try:
    _has_data = sig_df is not None and sig_df_with_disp is not None and not sig_df_with_disp.empty
except NameError:
    _has_data = False

if _has_data:

    event_table = build_event_study_table(
        sig_df_with_disp,
        disp_clean,
        thresholds=[90, 95],
        forward_windows=[5, 10, 21]
    )

    # Format the table
    format_dict = {}
    for w in [5, 10, 21]:
        format_dict[f'Avg R ({w}d)'] = '{:+.3f}'
        format_dict[f'Win% ({w}d)'] = '{:.1%}'

    # Highlight baseline row for comparison
    def highlight_baseline(row):
        if row['Signal'] == 'Baseline (all trades)':
            return ['background-color: rgba(100, 100, 255, 0.1)'] * len(row)
        return [''] * len(row)

    styled_event = event_table.style.apply(highlight_baseline, axis=1).format(
        format_dict, na_rep='—'
    )

    st.dataframe(styled_event, use_container_width=True, hide_index=True)

    # Interpretation helper
    st.caption(
        "A 'crossing' = dispersion_rank moves from below to at/above the threshold "
        "(prevents double-counting sustained high readings). "
        "Trades are collected within the forward window after each crossing. "
        "Compare Avg R and Win% against the baseline row. "
        "If post-spike Avg R is materially below baseline, that's a sizing signal with a specific duration."
    )

    # Show the individual crossing dates in an expander for transparency
    with st.expander("📅 Crossing Event Dates"):
        for threshold in [90, 95]:
            crossing_dates = find_crossing_dates(disp_clean, threshold)
            if crossing_dates:
                dates_str = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in crossing_dates]
                st.write(f"**{threshold}th percentile crossings ({len(dates_str)} events):**")
                # Display as a compact comma-separated list, not a huge table
                st.text(', '.join(dates_str))
            else:
                st.write(f"**{threshold}th percentile:** No crossings found in data range")

else:
    st.info("Run a backtest in the **Strategy Backtester** page first to enable the event study.")

# =============================================================================
# SECTION 6: SECTOR LEADERSHIP ANALYSIS
# =============================================================================
st.markdown("---")
st.subheader("🏭 Sector Leadership Analysis")
st.caption("When XLE, XLP, or XLU is the top-performing sector over trailing N days, what happens next?")

# Inline control for lookback window
leadership_window = st.radio(
    "Trailing lookback for leadership",
    [5, 10, 21],
    index=2,  # default to 21
    horizontal=True,
    key="leadership_window"
)

# Download sector prices (cached 1hr)
try:
    sector_prices = download_sector_prices(start_date=str(start_date))
    if sector_prices is not None and not sector_prices.empty:
        leadership_df = compute_sector_leadership(sector_prices, windows=[5, 10, 21])
    else:
        leadership_df = None
except Exception as e:
    st.error(f"Could not load sector data: {e}")
    sector_prices = None
    leadership_df = None

if leadership_df is not None and not leadership_df.empty:

    # Current state callout
    leader_col = f'leader_{leadership_window}d'
    leader_ret_col = f'leader_{leadership_window}d_ret'

    current_leader = leadership_df[leader_col].dropna().iloc[-1] if not leadership_df[leader_col].dropna().empty else None
    current_leader_ret = leadership_df[leader_ret_col].dropna().iloc[-1] if not leadership_df[leader_ret_col].dropna().empty else 0

    if current_leader:
        is_defensive = current_leader in DEFENSIVE_LEADERS

        if is_defensive:
            st.warning(
                f"**Current {leadership_window}d leader: {current_leader}** "
                f"({current_leader_ret:+.1%}) — defensive/commodity leadership"
            )
        else:
            st.success(
                f"**Current {leadership_window}d leader: {current_leader}** "
                f"({current_leader_ret:+.1%})"
            )

    # Table 1: SPX forward returns by leader
    st.markdown(f"**SPX Forward Returns by {leadership_window}d Sector Leader**")

    spx_table = compute_leadership_forward_returns(
        sector_prices, leadership_df, leadership_window, forward_windows=[5, 10, 21]
    )

    if not spx_table.empty:
        # Format
        spx_format = {'Days Leading': '{:,.0f}'}
        for fw in [5, 10, 21]:
            spx_format[f'SPX Fwd {fw}d Avg'] = '{:+.2%}'
            spx_format[f'SPX Fwd {fw}d Win%'] = '{:.1%}'

        # Highlight the "Any XLE/XLP/XLV" row
        def highlight_defensive_row(row):
            if row['Leader'] == 'Any XLE/XLP/XLV':
                return ['background-color: rgba(255, 160, 0, 0.15)'] * len(row)
            if row['Leader'] == 'Other sector leads':
                return ['background-color: rgba(100, 100, 255, 0.1)'] * len(row)
            return [''] * len(row)

        styled_spx = spx_table.style.apply(highlight_defensive_row, axis=1).format(
            spx_format, na_rep='—'
        )
        st.dataframe(styled_spx, use_container_width=True, hide_index=True)
    else:
        st.warning("Could not compute SPX forward returns by sector leader.")

    # Table 2: Strategy performance by leader (only if sig_df available)
    try:
        _has_sig_df = sig_df is not None and not sig_df.empty
    except NameError:
        _has_sig_df = False

    if _has_sig_df:
        st.markdown(f"**Strategy Performance by {leadership_window}d Sector Leader**")

        strat_table = compute_leadership_strategy_performance(
            sig_df, leadership_df, leadership_window
        )

        if not strat_table.empty:
            def highlight_strat_row(row):
                if row['Condition'] == 'Any XLE/XLP/XLV leading':
                    return ['background-color: rgba(255, 160, 0, 0.15)'] * len(row)
                if row['Condition'] == 'All trades (baseline)':
                    return ['background-color: rgba(100, 100, 255, 0.1)'] * len(row)
                return [''] * len(row)

            styled_strat = strat_table.style.apply(highlight_strat_row, axis=1).format({
                'Avg R': '{:+.3f}',
                'Win Rate': '{:.1%}',
                'Profit Factor': '{:.2f}',
            })

            st.dataframe(styled_strat, use_container_width=True, hide_index=True)

    # Leaderboard: how often each sector leads (for context)
    with st.expander("📊 Sector Leadership Frequency"):
        freq = leadership_df[leader_col].value_counts()
        total = freq.sum()
        freq_df = pd.DataFrame({
            'Sector': freq.index,
            'Days Leading': freq.values,
            '% of Days': freq.values / total
        })
        # Flag defensive
        freq_df['Type'] = freq_df['Sector'].apply(
            lambda x: '⚠️ Defensive' if x in DEFENSIVE_LEADERS else ''
        )

        styled_freq = freq_df.style.format({
            '% of Days': '{:.1%}'
        })
        st.dataframe(styled_freq, use_container_width=True, hide_index=True)

else:
    st.warning("Could not load sector leadership data. Check network connection.")

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
