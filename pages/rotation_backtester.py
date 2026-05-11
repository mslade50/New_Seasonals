"""
Cross-Asset Trend Backtester — vol-targeted time-series momentum.

Per asset, blends vol-scaled point-in-time log returns at multiple lookbacks
(default 1m / 3m / 12m, equal weights), caps the blended signal at +/-2, and
sizes each position to an equal vol contribution (target_vol_i = 10% /
sqrt(N)). The whole book is then scaled by a 252-day ex-ante covariance
estimate so the portfolio targets a fixed annualized vol (default 10%).

Rebalances monthly (last business day) by default. T+1 execution. Vol
estimate is EWMA of daily log returns, 60-day half-life, annualized by
sqrt(252).

Universe is free-text. master_prices.parquet is the primary source; missing
tickers (FX, some commodities, indices) fall back to yfinance and are cached
for the session.
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import sys
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# PATH SETUP
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import data_provider

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
DEFAULT_BASKET = [
    "QQQ", "SPY", "TLT", "GLD", "SLV", "USO", "WEAT", "UNG", "CPER",
    "BTC-USD", "^GDAXI", "^N225",
    "USDEUR=X", "USDJPY=X", "USDCHF=X", "USDMXN=X", "USDAUD=X", "USDCAD=X",
    "EEM",
]
PRESETS = {
    "Cross-asset starter (19)": DEFAULT_BASKET,
    "Concentrated (5)": ["SPY", "TLT", "GLD", "DBC", "UUP"],
    "Equity index ETFs (6)": ["SPY", "QQQ", "IWM", "EFA", "EEM", "VEA"],
    "(custom)": [],
}

st.set_page_config(page_title="Cross-Asset Trend Backtester", layout="wide")


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _yf_fetch(tickers_tuple, start, end):
    """Live yfinance fetch for tickers missing from master_prices.

    Returns {ticker: Series of close} indexed by date. Handles the MultiIndex
    columns yfinance returns for multi-ticker downloads.
    """
    if not tickers_tuple:
        return {}
    try:
        import yfinance as yf
    except ImportError:
        return {}
    raw = yf.download(
        list(tickers_tuple), start=start, end=end,
        auto_adjust=True, progress=False, threads=True,
    )
    if raw is None or raw.empty:
        return {}
    out = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for tk in tickers_tuple:
            try:
                df = raw.xs(tk, level=1, axis=1)
            except (KeyError, ValueError):
                continue
            if 'Close' not in df.columns:
                continue
            s = df['Close'].dropna()
            if s.empty:
                continue
            s.index = pd.to_datetime(s.index).normalize()
            if hasattr(s.index, 'tz') and s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            out[tk] = s
    else:
        # Single-ticker download
        if 'Close' in raw.columns and tickers_tuple:
            s = raw['Close'].dropna()
            s.index = pd.to_datetime(s.index).normalize()
            if hasattr(s.index, 'tz') and s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            out[tickers_tuple[0]] = s
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_basket_closes(tickers_tuple):
    """Return DataFrame of close prices, columns=tickers, index=dates.

    Tries master_prices first, then yfinance for missing tickers. Forward-fills
    so daily-frequency assets (FX has fewer holidays than equities) align.
    """
    tickers = list(tickers_tuple)
    if not tickers:
        return pd.DataFrame(), []
    closes = {}
    if data_provider.has_master():
        hist = data_provider.get_history(tickers)
        for tk in tickers:
            df = hist.get(tk)
            if df is None or df.empty:
                continue
            s = df['Close'].copy()
            s.index = pd.to_datetime(s.index).normalize()
            if hasattr(s.index, 'tz') and s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            closes[tk] = s.dropna()

    missing = [t for t in tickers if t not in closes]
    if missing:
        # yfinance fallback for FX / non-master tickers
        yf_data = _yf_fetch(tuple(missing), start="1995-01-01",
                            end=(datetime.date.today() + datetime.timedelta(days=1)).isoformat())
        for tk, s in yf_data.items():
            if not s.empty:
                closes[tk] = s

    failed = [t for t in tickers if t not in closes]
    if not closes:
        return pd.DataFrame(), failed
    df = pd.DataFrame(closes).sort_index()
    # Forward-fill to align trading days; cap at 5 days so a delisted/halted
    # asset doesn't contaminate later signals indefinitely.
    df = df.ffill(limit=5)
    return df, failed


# -----------------------------------------------------------------------------
# SIGNAL & VOL
# -----------------------------------------------------------------------------
def compute_log_returns(close_df):
    return np.log(close_df / close_df.shift(1))


def compute_ewma_vol(log_rets, halflife):
    """EWMA realized vol of daily log returns, annualized by sqrt(252).

    pandas .ewm(halflife=...) uses the standard formula; .std() returns the
    sample stdev with bias correction. We use that and annualize.
    """
    ewma_var = log_rets.ewm(halflife=halflife, min_periods=halflife).var()
    return np.sqrt(ewma_var) * np.sqrt(252)


def compute_signal(close_df, log_rets, sigma_ann, lookbacks, weights, cap):
    """Per-asset blended trend signal, capped at +/- cap.

    For each lookback w (in trading days), compute the point-in-time log
    return r_w = log(P_t / P_{t-w}), divide by annualized vol, and blend
    across lookbacks with the supplied weights. Then clip.

    Note on convention: this divides by annualized sigma (not horizon-scaled
    sigma). Long-lookback components are naturally larger in magnitude than
    short-lookback ones; the cap limits the blend rather than each component.
    Iterate the spec here if you want each lookback z-like instead.
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    components = []
    for w, wt in zip(lookbacks, weights):
        r_w = np.log(close_df / close_df.shift(w))
        scaled = r_w / sigma_ann
        components.append(scaled * wt)
    blended = sum(components)
    return blended.clip(lower=-cap, upper=cap)


# -----------------------------------------------------------------------------
# POSITION SIZING
# -----------------------------------------------------------------------------
def compute_per_asset_weights(signal_row, sigma_row, target_vol_per_asset):
    """w_i = signal_i * (target_vol_i / sigma_i). Returns Series (NaN-safe)."""
    valid = signal_row.notna() & sigma_row.notna() & (sigma_row > 0)
    w = pd.Series(0.0, index=signal_row.index)
    if not valid.any():
        return w
    w.loc[valid] = signal_row.loc[valid] * (target_vol_per_asset / sigma_row.loc[valid])
    return w


def scale_to_portfolio_vol(weights, cov_matrix_ann, target_vol):
    """Scale a weight vector so ex-ante portfolio vol = target.

    cov_matrix_ann is the annualized covariance matrix over the assets that
    are tradeable on this rebalance date. Returns scaled weights and the
    pre-scale ex-ante vol.
    """
    if cov_matrix_ann is None or cov_matrix_ann.empty:
        return weights, np.nan
    common = [t for t in weights.index if t in cov_matrix_ann.index]
    if not common:
        return weights, np.nan
    w_vec = weights.reindex(common).fillna(0.0).values
    cov = cov_matrix_ann.reindex(index=common, columns=common).values
    var = float(w_vec @ cov @ w_vec)
    if var <= 0 or not np.isfinite(var):
        return weights, np.nan
    ex_ante_vol = np.sqrt(var)
    if ex_ante_vol == 0:
        return weights, ex_ante_vol
    scaled = weights.copy()
    scaled.loc[common] = scaled.loc[common] * (target_vol / ex_ante_vol)
    return scaled, ex_ante_vol


# -----------------------------------------------------------------------------
# REBALANCE CALENDAR
# -----------------------------------------------------------------------------
def get_rebal_dates(date_index, freq):
    di = pd.DatetimeIndex(date_index)
    if di.empty:
        return di
    s = di.to_series()
    if freq == 'Monthly (last bday)':
        return pd.DatetimeIndex(sorted(s.groupby([s.dt.year.values, s.dt.month.values]).max().values))
    if freq == 'Weekly (Fri)':
        ic = s.dt.isocalendar()
        return pd.DatetimeIndex(sorted(s.groupby([ic.year.values, ic.week.values]).max().values))
    if freq == 'Quarterly (last bday)':
        q = ((s.dt.month - 1) // 3 + 1).values
        return pd.DatetimeIndex(sorted(s.groupby([s.dt.year.values, q]).max().values))
    if freq == 'Daily':
        return di
    return di


# -----------------------------------------------------------------------------
# CORE BACKTEST
# -----------------------------------------------------------------------------
def run_backtest(
    close_df, rebal_dates,
    lookbacks, lookback_weights, signal_cap, ewma_halflife,
    target_portfolio_vol, cov_window,
    starting_equity, execution_lag,
):
    """Walk rebal dates, build vol-targeted weights, simulate daily PnL.

    Returns dict with equity, weights timeline, ex-ante vol series, etc.
    """
    cal = close_df.index
    if cal.empty:
        return None

    log_rets = compute_log_returns(close_df)
    simple_rets = close_df.pct_change()
    sigma_ann = compute_ewma_vol(log_rets, ewma_halflife)
    signal_df = compute_signal(close_df, log_rets, sigma_ann, lookbacks, lookback_weights, signal_cap)

    weights = pd.DataFrame(0.0, index=cal, columns=close_df.columns)
    rebal_records = []

    pairs = []
    for rd in rebal_dates:
        if rd not in cal:
            continue
        if execution_lag:
            idx = cal.searchsorted(rd) + 1
            if idx >= len(cal):
                continue
            ed = cal[idx]
        else:
            ed = rd
        pairs.append((rd, ed))
    if not pairs:
        return None

    max_lookback = max(lookbacks)

    for i, (rd, ed) in enumerate(pairs):
        next_ed = pairs[i + 1][1] if i + 1 < len(pairs) else cal[-1] + pd.Timedelta(days=1)
        hold_mask = (cal >= ed) & (cal < next_ed)

        # Per-asset eligibility: need enough history for signal + sigma
        sig_row = signal_df.loc[rd] if rd in signal_df.index else None
        sig_row = sig_row.dropna() if sig_row is not None else pd.Series(dtype=float)
        sig_row = sig_row.reindex([c for c in close_df.columns if c in sig_row.index])

        sigma_row = sigma_ann.loc[rd] if rd in sigma_ann.index else pd.Series(dtype=float)

        eligible = [t for t in sig_row.index if pd.notna(sigma_row.get(t)) and sigma_row.get(t, 0) > 0]
        if not eligible:
            rebal_records.append({'date': ed, 'n_assets': 0, 'ex_ante_vol_pre': np.nan,
                                  'scale': np.nan, 'gross': 0.0, 'net': 0.0})
            continue

        n_eligible = len(eligible)
        target_vol_per_asset = target_portfolio_vol / np.sqrt(n_eligible)
        sig_e = sig_row.reindex(eligible)
        sigma_e = sigma_row.reindex(eligible)

        per_asset_w = compute_per_asset_weights(sig_e, sigma_e, target_vol_per_asset)

        # Annualized covariance matrix from a trailing window of log returns
        win_start_idx = max(0, cal.searchsorted(rd) - cov_window + 1)
        cov_slice = log_rets.iloc[win_start_idx:cal.searchsorted(rd) + 1][eligible]
        # Drop columns with too few observations
        valid_cols = [c for c in cov_slice.columns if cov_slice[c].notna().sum() >= max(20, cov_window // 4)]
        cov_slice = cov_slice[valid_cols].dropna(how='any')
        if len(cov_slice) >= max(20, cov_window // 4):
            cov_ann = cov_slice.cov() * 252
        else:
            cov_ann = pd.DataFrame()

        scaled_w, ex_ante = scale_to_portfolio_vol(per_asset_w, cov_ann, target_portfolio_vol)
        if not np.isfinite(ex_ante) or ex_ante == 0:
            scale = 1.0
        else:
            scale = target_portfolio_vol / ex_ante

        # Apply weights from ed to next_ed (exclusive)
        for t in scaled_w.index:
            weights.loc[hold_mask, t] = scaled_w[t]

        gross = float(scaled_w.abs().sum())
        net = float(scaled_w.sum())
        rebal_records.append({
            'date': ed, 'n_assets': int(n_eligible),
            'ex_ante_vol_pre': float(ex_ante) if np.isfinite(ex_ante) else np.nan,
            'scale': float(scale),
            'gross': gross, 'net': net,
        })

    # MTM: yesterday's weights drive today's return
    weights_lagged = weights.shift(1).fillna(0.0)
    daily_pnl_pct = (weights_lagged * simple_rets.fillna(0.0)).sum(axis=1)
    equity = starting_equity * (1 + daily_pnl_pct).cumprod()
    equity.iloc[0] = starting_equity

    # Equal-weight buy-and-hold benchmark across the same universe
    n_assets = close_df.shape[1]
    bench_w = pd.Series(1.0 / n_assets, index=close_df.columns)
    bench_ret = (simple_rets.fillna(0.0) * bench_w).sum(axis=1)
    bench_eq = starting_equity * (1 + bench_ret).cumprod()
    bench_eq.iloc[0] = starting_equity

    return {
        'equity': equity, 'benchmark': bench_eq,
        'port_ret': daily_pnl_pct, 'bench_ret': bench_ret,
        'weights': weights, 'signal': signal_df, 'sigma_ann': sigma_ann,
        'rebal_log': pd.DataFrame(rebal_records),
        'cal': cal,
    }


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("Cross-Asset Trend Backtester")
st.caption(
    "Per-asset trend signal (blend of vol-scaled returns at multiple lookbacks), "
    "vol-targeted positions sized to equal vol contribution, then portfolio "
    "scaled to a fixed ex-ante vol via rolling covariance."
)

# --- Sidebar: globals ---
st.sidebar.header("Globals")
starting_equity = st.sidebar.number_input("Starting Equity ($)", value=100000, step=10000, min_value=1000)
target_portfolio_vol_pct = st.sidebar.number_input(
    "Target portfolio vol (%, annualized)", value=10.0, min_value=1.0, max_value=50.0, step=0.5,
)
ewma_halflife = st.sidebar.number_input(
    "EWMA vol half-life (days)", value=60, min_value=10, max_value=252, step=5,
    help="Half-life for the EWMA realized-vol estimator on daily log returns.",
)
cov_window = st.sidebar.number_input(
    "Covariance window (days)", value=252, min_value=63, max_value=1260, step=21,
    help="Trailing window of daily log returns used for the ex-ante covariance estimate.",
)
signal_cap = st.sidebar.number_input(
    "Signal cap (+/-)", value=2.0, min_value=0.5, max_value=5.0, step=0.5,
)

st.sidebar.divider()
st.sidebar.subheader("Lookbacks")
lb_short = st.sidebar.number_input("Short (days)", value=21, min_value=5, max_value=63, step=1)
lb_med = st.sidebar.number_input("Medium (days)", value=63, min_value=10, max_value=252, step=1)
lb_long = st.sidebar.number_input("Long (days)", value=252, min_value=63, max_value=756, step=1)
weight_short = st.sidebar.number_input("Weight short", value=1.0, min_value=0.0, max_value=10.0, step=0.5)
weight_med = st.sidebar.number_input("Weight medium", value=1.0, min_value=0.0, max_value=10.0, step=0.5)
weight_long = st.sidebar.number_input("Weight long", value=1.0, min_value=0.0, max_value=10.0, step=0.5)

# --- 1. Universe ---
st.subheader("1. Universe")
preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
default_text = ", ".join(PRESETS[preset_name]) if PRESETS[preset_name] else ""
basket_text = st.text_area(
    "Tickers (comma-separated)", value=default_text, height=70,
    help="Master_prices is checked first, then yfinance for the rest. "
         "Use yfinance notation for FX (USDJPY=X), indices (^GDAXI), crypto (BTC-USD).",
)
basket = sorted({t.strip() for t in basket_text.replace('\n', ',').split(',') if t.strip()})
st.caption(f"{len(basket)} tickers parsed")

# --- 2. Date range ---
st.subheader("2. Backtest period")
c1, c2 = st.columns(2)
with c1:
    start_date = st.date_input("Start", value=datetime.date(2010, 1, 1),
                               min_value=datetime.date(1995, 1, 1))
with c2:
    end_date = st.date_input("End", value=datetime.date.today(),
                             min_value=start_date)
st.caption("400-day buffer is added before the start so EWMA vol and 12m returns warm up.")

# --- 3. Rebalance ---
st.subheader("3. Rebalance")
c1, c2 = st.columns(2)
with c1:
    rebal_freq = st.selectbox(
        "Frequency",
        ["Monthly (last bday)", "Weekly (Fri)", "Quarterly (last bday)", "Daily"],
        index=0,
    )
with c2:
    execution_lag = st.checkbox(
        "Execute next-day open (T+1 lag)", value=True,
        help="Recommended. Compute signal at rebal-day close, take position next bar.",
    )

# --- Run ---
st.divider()
run_btn = st.button("Run Backtest", type="primary")

if run_btn:
    if not basket:
        st.error("Add at least one ticker.")
        st.stop()

    with st.spinner("Loading prices (master_prices + yfinance fallback)..."):
        close_df, failed = load_basket_closes(tuple(basket))
    if close_df.empty:
        st.error(f"No data loaded. Failed tickers: {failed}")
        st.stop()
    if failed:
        st.warning(f"Dropped (no data anywhere): {', '.join(failed)}")
    missing_in_master = sorted(set(basket) - set(close_df.columns) - set(failed))
    if missing_in_master:
        st.info(f"Loaded via yfinance fallback (not in master): {', '.join(missing_in_master)}")

    buffer_start = pd.Timestamp(start_date) - pd.Timedelta(days=500)
    close_df = close_df[(close_df.index >= buffer_start) & (close_df.index <= pd.Timestamp(end_date))]
    if close_df.empty:
        st.error("No price data in the selected window.")
        st.stop()

    backtest_cal = close_df.index[close_df.index >= pd.Timestamp(start_date)]
    if backtest_cal.empty:
        st.error("No trading days in the selected window after applying the buffer.")
        st.stop()
    rebal_dates = get_rebal_dates(backtest_cal, rebal_freq)

    lookbacks = [int(lb_short), int(lb_med), int(lb_long)]
    lookback_weights = [float(weight_short), float(weight_med), float(weight_long)]
    if sum(lookback_weights) <= 0:
        st.error("Lookback weights must sum to > 0.")
        st.stop()

    target_vol = target_portfolio_vol_pct / 100.0

    with st.spinner(f"Running backtest over {len(rebal_dates)} rebalances..."):
        result = run_backtest(
            close_df=close_df.reindex(backtest_cal),
            rebal_dates=rebal_dates,
            lookbacks=lookbacks,
            lookback_weights=lookback_weights,
            signal_cap=float(signal_cap),
            ewma_halflife=int(ewma_halflife),
            target_portfolio_vol=target_vol,
            cov_window=int(cov_window),
            starting_equity=float(starting_equity),
            execution_lag=execution_lag,
        )
    if result is None:
        st.error("No rebalance dates resolved.")
        st.stop()

    # ---- RESULTS ----
    st.divider()
    st.subheader("Results")

    eq, bench = result['equity'], result['benchmark']
    rot_rets, bench_rets = result['port_ret'], result['bench_ret']
    n_years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    rot_total = float(eq.iloc[-1] / eq.iloc[0] - 1)
    bench_total = float(bench.iloc[-1] / bench.iloc[0] - 1)
    rot_cagr = (1 + rot_total) ** (1 / n_years) - 1 if rot_total > -1 else float('nan')
    bench_cagr = (1 + bench_total) ** (1 / n_years) - 1 if bench_total > -1 else float('nan')

    def _sharpe(r):
        return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float('nan')

    def _sortino(r):
        d = r[r < 0]
        return float(r.mean() / d.std() * np.sqrt(252)) if len(d) >= 2 and d.std() > 0 else float('nan')

    rot_sharpe, bench_sharpe = _sharpe(rot_rets), _sharpe(bench_rets)
    rot_sortino, bench_sortino = _sortino(rot_rets), _sortino(bench_rets)
    rot_dd = float((eq / eq.cummax() - 1).min())
    bench_dd = float((bench / bench.cummax() - 1).min())
    realized_vol = float(rot_rets.std() * np.sqrt(252))

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("CAGR", f"{rot_cagr*100:.1f}%", delta=f"{(rot_cagr-bench_cagr)*100:+.1f}% vs basket")
    s2.metric("Sharpe", f"{rot_sharpe:.2f}", delta=f"{rot_sharpe-bench_sharpe:+.2f}")
    s3.metric("Max DD", f"{rot_dd*100:.1f}%", delta=f"{(rot_dd-bench_dd)*100:+.1f}% vs basket",
              delta_color="inverse")
    s4.metric("Realized vol", f"{realized_vol*100:.1f}%",
              delta=f"target {target_portfolio_vol_pct:.1f}%", delta_color="off")

    s5, s6, s7, s8 = st.columns(4)
    s5.metric("Total return", f"{rot_total*100:.1f}%")
    s6.metric("Sortino", f"{rot_sortino:.2f}", delta=f"{rot_sortino-bench_sortino:+.2f}")
    s7.metric("Rebalances", f"{len(result['rebal_log'])}")
    s8.metric("Years", f"{n_years:.1f}")

    # Equity
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode='lines',
                             name='Trend portfolio', line=dict(color='#1E88E5', width=2)))
    fig.add_trace(go.Scatter(x=bench.index, y=bench.values, mode='lines',
                             name='Equal-weight basket', line=dict(color='#888', width=1, dash='dash')))
    fig.update_layout(height=420, yaxis_type='log', yaxis_title='Equity ($)',
                      margin=dict(l=10, r=10, t=30, b=10), title='Equity Curve (log scale)')
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown
    rot_dd_series = (eq / eq.cummax() - 1) * 100
    bench_dd_series = (bench / bench.cummax() - 1) * 100
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=rot_dd_series.index, y=rot_dd_series.values, mode='lines',
                                name='Trend', line=dict(color='#1E88E5', width=1.5),
                                fill='tozeroy'))
    fig_dd.add_trace(go.Scatter(x=bench_dd_series.index, y=bench_dd_series.values, mode='lines',
                                name='Basket', line=dict(color='#888', width=1, dash='dash')))
    fig_dd.update_layout(height=240, yaxis_title='Drawdown (%)',
                         margin=dict(l=10, r=10, t=30, b=10), title='Drawdown')
    st.plotly_chart(fig_dd, use_container_width=True)

    # Gross / net exposure over time
    weights = result['weights']
    gross = weights.abs().sum(axis=1)
    net = weights.sum(axis=1)
    fig_exp = go.Figure()
    fig_exp.add_trace(go.Scatter(x=gross.index, y=gross.values * 100, mode='lines',
                                 name='Gross', line=dict(color='#1E88E5', width=1.5)))
    fig_exp.add_trace(go.Scatter(x=net.index, y=net.values * 100, mode='lines',
                                 name='Net', line=dict(color='#43A047', width=1.5)))
    fig_exp.update_layout(height=260, yaxis_title='Exposure (% of equity)',
                          margin=dict(l=10, r=10, t=30, b=10), title='Gross / Net Exposure')
    st.plotly_chart(fig_exp, use_container_width=True)

    # Rebalance log
    with st.expander("Rebalance log", expanded=False):
        rl = result['rebal_log'].copy()
        if not rl.empty:
            rl['date'] = pd.to_datetime(rl['date']).dt.strftime('%Y-%m-%d')
            rl['ex_ante_vol_pre'] = (rl['ex_ante_vol_pre'] * 100).round(2)
            rl['scale'] = rl['scale'].round(3)
            rl['gross'] = (rl['gross'] * 100).round(1)
            rl['net'] = (rl['net'] * 100).round(1)
            rl = rl.rename(columns={
                'ex_ante_vol_pre': 'ex_ante_vol_pre (%)',
                'gross': 'gross (%)', 'net': 'net (%)',
            })
            st.dataframe(rl, hide_index=True, use_container_width=True)

    # Per-asset weight heatmap (monthly snapshot)
    with st.expander("Weight timeline (monthly)", expanded=False):
        if not weights.empty:
            try:
                import plotly.express as px
                monthly = weights.resample('ME').last()
                monthly = monthly.loc[:, (monthly.abs() > 1e-6).any(axis=0)]
                if not monthly.empty:
                    fig_hm = px.imshow(
                        monthly.T.values * 100,
                        x=monthly.index, y=monthly.columns,
                        aspect='auto', color_continuous_scale='RdBu', color_continuous_midpoint=0,
                        labels=dict(x='Date', y='Ticker', color='Weight (%)'),
                    )
                    fig_hm.update_layout(height=max(280, 26 * len(monthly.columns) + 80),
                                         margin=dict(l=80, r=10, t=20, b=40))
                    st.plotly_chart(fig_hm, use_container_width=True)
            except ImportError:
                pass

    # Final-state diagnostic table — last signal & sigma per asset
    with st.expander("Latest signal snapshot", expanded=False):
        sig_df = result['signal']
        sig_ann = result['sigma_ann']
        last_dt = sig_df.dropna(how='all').index.max()
        if pd.notna(last_dt):
            snap = pd.DataFrame({
                'signal': sig_df.loc[last_dt],
                'sigma_ann': sig_ann.loc[last_dt] * 100,
                'last_weight': weights.loc[last_dt] * 100,
            }).dropna(subset=['signal']).sort_values('signal', key=abs, ascending=False)
            snap.columns = ['signal', 'sigma_ann (%)', 'weight (%)']
            st.caption(f"As of {last_dt.strftime('%Y-%m-%d')}")
            st.dataframe(snap.round(3), use_container_width=True)
