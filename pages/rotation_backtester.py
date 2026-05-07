"""
Rotation Backtester — momentum / mean-reversion rotation across a basket
of tickers.

At each rebalance date, ranks the basket by N-day return (cross-sectional
within the basket OR time-series vs each ticker's own history), applies
optional eligibility filters, and rotates equal-weight into the top-N
(or bottom-N in reverse mode). Slack to cash when filters reject every
candidate.

Designed for "always invested, slow moving" experiments — pair a long
ranking window (63d / 252d) with a forgiving eligibility filter and a
weekly/monthly rebalance to size momentum exposure without chasing
extremes. Reverse mode covers the deep-oversold mean-reversion case.

Companion to pages/exposure_backtester.py — exposure_backtester scales
fixed weights via rules; this page picks weights dynamically by ranking.
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import sys
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# PATH SETUP
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import data_provider

try:
    from strategy_config import SECTOR_INDEX_ETFS, INDEX_ETFS, LIQUID_UNIVERSE
except ImportError:
    SECTOR_INDEX_ETFS, INDEX_ETFS, LIQUID_UNIVERSE = [], [], []

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
WINDOWS = [5, 10, 21, 63, 126, 252]
TS_RANK_DEFAULT_LOOKBACK = 252  # rolling lookback for time-series rank
LOGIC_OPS = ["<", ">", "Between"]
RANK_TYPES = ["xsec", "ts"]
RANK_TYPE_LABELS = {"xsec": "Cross-sectional (within basket)", "ts": "Time-series (own history)"}

st.set_page_config(page_title="Rotation Backtester", layout="wide")

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_basket_closes(tickers_tuple):
    """Pull close-price DataFrame for a basket. Cached on the tickers tuple."""
    tickers = list(tickers_tuple)
    if not tickers:
        return pd.DataFrame()
    if not data_provider.has_master():
        return pd.DataFrame()
    hist = data_provider.get_history(tickers)
    closes = {}
    for tk in tickers:
        df = hist.get(tk)
        if df is None or df.empty:
            continue
        s = df['Close'].copy()
        s.index = pd.to_datetime(s.index).normalize()
        if hasattr(s.index, 'tz') and s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        closes[tk] = s.dropna()
    if not closes:
        return pd.DataFrame()
    return pd.DataFrame(closes).sort_index().ffill()


# -----------------------------------------------------------------------------
# RANK COMPUTATION
# -----------------------------------------------------------------------------
def compute_returns(close_df, windows):
    """Return {window: DataFrame of N-day returns}."""
    return {w: close_df.pct_change(w) for w in windows}


def compute_xsec_ranks(rets_dict):
    """Cross-sectional rank within basket per row, expressed as 0-100 percentile.

    100 = best return today within basket. Tickers with NaN (insufficient
    history) get NaN rank — the filter+selection logic skips NaNs.
    """
    return {w: rets_dict[w].rank(axis=1, pct=True) * 100 for w in rets_dict}


def compute_ts_ranks(rets_dict, lookback):
    """Time-series rank for each ticker — its N-day return today vs its own
    last `lookback` trading days. 100 = best within the lookback window.

    Per-column rolling rank is O(n*lookback) but vectorized with .rank(pct=True).
    """
    out = {}
    min_p = max(20, int(lookback * 0.25))
    for w, ret_df in rets_dict.items():
        ts = pd.DataFrame(index=ret_df.index, columns=ret_df.columns, dtype=float)
        for col in ret_df.columns:
            ts[col] = ret_df[col].rolling(lookback, min_periods=min_p).rank(pct=True) * 100
        out[w] = ts
    return out


# -----------------------------------------------------------------------------
# REBALANCE CALENDAR
# -----------------------------------------------------------------------------
def get_rebal_dates(date_index, freq):
    """Resolve rebalance dates from a daily trading-day index.

    'Weekly (Mon)' — first trading day of each calendar week (Mon if open,
                     else Tue, etc.).
    'Monthly (EOM)' — last trading day of each calendar month.
    'Every N days' — every Nth row of the index.
    'Daily' — every trading day.
    """
    di = pd.DatetimeIndex(date_index)
    if di.empty:
        return di
    if freq == 'Weekly (Mon)':
        s = di.to_series()
        # Group by ISO year+week so we don't merge week 52 with the next year's week 1
        ic = s.dt.isocalendar()
        first = s.groupby([ic.year.values, ic.week.values]).min()
        return pd.DatetimeIndex(sorted(first.values))
    if freq == 'Monthly (EOM)':
        s = di.to_series()
        last = s.groupby([s.dt.year.values, s.dt.month.values]).max()
        return pd.DatetimeIndex(sorted(last.values))
    if freq == 'Daily':
        return di
    if freq.startswith('Every'):
        n = int(freq.split()[1])
        return di[::n]
    return di


# -----------------------------------------------------------------------------
# FILTER EVALUATION
# -----------------------------------------------------------------------------
def _filter_mask(filt, candidates_row):
    """Apply one filter spec to a Series of rank values keyed by ticker."""
    op = filt['logic']
    if op == '<':
        return candidates_row < filt['thresh']
    if op == '>':
        return candidates_row > filt['thresh']
    if op == 'Between':
        return (candidates_row >= filt['thresh_min']) & (candidates_row <= filt['thresh_max'])
    return pd.Series(False, index=candidates_row.index)


def evaluate_eligibility(filters, candidates, rebal_date, xsec_ranks, ts_ranks):
    """Return boolean Series — True for candidates that pass ALL filters."""
    if not filters:
        return pd.Series(True, index=candidates)
    mask = pd.Series(True, index=candidates)
    for f in filters:
        ranks_df = xsec_ranks[f['window']] if f['type'] == 'xsec' else ts_ranks[f['window']]
        if rebal_date not in ranks_df.index:
            return pd.Series(False, index=candidates)
        row = ranks_df.loc[rebal_date].reindex(candidates)
        m = _filter_mask(f, row).fillna(False)
        mask &= m
    return mask


def evaluate_reverse_trigger(filters, candidates, rebal_date, xsec_ranks, ts_ranks):
    """Return True iff EVERY candidate satisfies EVERY filter (AND across filters,
    AND across candidates). This is the "all members oversold" gate.
    """
    if not filters:
        return False
    for f in filters:
        ranks_df = xsec_ranks[f['window']] if f['type'] == 'xsec' else ts_ranks[f['window']]
        if rebal_date not in ranks_df.index:
            return False
        row = ranks_df.loc[rebal_date].reindex(candidates).dropna()
        if row.empty:
            return False
        m = _filter_mask(f, row)
        if not bool(m.all()):
            return False
    return True


# -----------------------------------------------------------------------------
# CORE BACKTEST
# -----------------------------------------------------------------------------
def run_backtest(
    close_df, rebal_dates,
    select_window, select_type, direction, top_n,
    eligibility_filters,
    reverse_enabled, reverse_filters, reverse_n,
    starting_equity, execution_lag, cash_apr,
    xsec_ranks, ts_ranks,
):
    """Walk rebal dates, pick winners, hold until next rebal, compute MTM.

    execution_lag=True → ranks computed at rebal_date close, position taken at
                        next trading day (T+1). More realistic.
    execution_lag=False → ranks AND position both at rebal_date close. Cleaner
                        backtest but introduces look-ahead since you can't act
                        on the close until the next bar.
    """
    cal = close_df.index
    if cal.empty:
        return None
    weights = pd.DataFrame(0.0, index=cal, columns=close_df.columns)
    cash_w = pd.Series(1.0, index=cal)
    trade_log = []

    # Resolve (decision_date, execution_date) pairs
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

    candidates = list(close_df.columns)

    for i, (rd, ed) in enumerate(pairs):
        # Holding period: [ed, next_ed)
        next_ed = pairs[i + 1][1] if i + 1 < len(pairs) else cal[-1] + pd.Timedelta(days=1)
        hold_mask = (cal >= ed) & (cal < next_ed)

        ranks_df = xsec_ranks[select_window] if select_type == 'xsec' else ts_ranks[select_window]
        if rd not in ranks_df.index:
            trade_log.append({'date': ed, 'mode': 'no-data', 'picks': []})
            continue
        rank_row = ranks_df.loc[rd].reindex(candidates).dropna()
        live_candidates = list(rank_row.index)
        if not live_candidates:
            trade_log.append({'date': ed, 'mode': 'no-candidates', 'picks': []})
            continue

        # Reverse mode takes precedence — fires only when ALL candidates satisfy.
        chosen_picks, mode = [], 'normal'
        if reverse_enabled and reverse_filters:
            if evaluate_reverse_trigger(reverse_filters, live_candidates, rd, xsec_ranks, ts_ranks):
                chosen_picks = rank_row.nsmallest(reverse_n).index.tolist()
                mode = 'reverse'

        if mode == 'normal':
            elig = evaluate_eligibility(eligibility_filters, live_candidates, rd, xsec_ranks, ts_ranks)
            keep = rank_row[elig.reindex(rank_row.index).fillna(False)]
            if keep.empty:
                trade_log.append({'date': ed, 'mode': 'cash', 'picks': []})
                continue
            chosen_picks = (keep.nlargest(top_n).index.tolist()
                            if direction == 'best'
                            else keep.nsmallest(top_n).index.tolist())

        if not chosen_picks:
            trade_log.append({'date': ed, 'mode': 'cash', 'picks': []})
            continue

        weight = 1.0 / len(chosen_picks)
        for p in chosen_picks:
            weights.loc[hold_mask, p] = weight
        cash_w.loc[hold_mask] = 0.0
        trade_log.append({'date': ed, 'mode': mode, 'picks': chosen_picks})

    # MTM — yesterday's weights drive today's return (no look-ahead)
    ret_df = close_df.pct_change().fillna(0.0)
    weights_lagged = weights.shift(1).fillna(0.0)
    cash_lagged = cash_w.shift(1).fillna(1.0)
    daily_cash = (cash_apr / 100.0) / 252.0
    port_ret = (weights_lagged * ret_df).sum(axis=1) + cash_lagged * daily_cash
    equity = starting_equity * (1 + port_ret).cumprod()
    equity.iloc[0] = starting_equity

    # Equal-weight basket benchmark
    bench_w = pd.Series(1.0 / len(candidates), index=candidates)
    bench_ret = (ret_df * bench_w).sum(axis=1)
    bench_eq = starting_equity * (1 + bench_ret).cumprod()
    bench_eq.iloc[0] = starting_equity

    return {
        'equity': equity, 'benchmark': bench_eq,
        'port_ret': port_ret, 'bench_ret': bench_ret,
        'weights': weights, 'cash_weight': cash_w,
        'trade_log': pd.DataFrame(trade_log),
        'cal': cal,
    }


# -----------------------------------------------------------------------------
# UI HELPERS
# -----------------------------------------------------------------------------
def _filter_widget(prefix, idx, default, list_key):
    """Render one filter row. Returns the filter dict."""
    cc = st.columns([1.6, 0.8, 0.8, 0.9, 0.9, 0.4])
    with cc[0]:
        t = st.selectbox(
            "Rank type", RANK_TYPES,
            index=RANK_TYPES.index(default.get('type', 'xsec')),
            format_func=lambda x: RANK_TYPE_LABELS[x],
            key=f'{prefix}_t_{idx}',
        )
    with cc[1]:
        w = st.selectbox(
            "Window (d)", WINDOWS,
            index=WINDOWS.index(default.get('window', 21)),
            key=f'{prefix}_w_{idx}',
        )
    with cc[2]:
        l = st.selectbox(
            "Logic", LOGIC_OPS,
            index=LOGIC_OPS.index(default.get('logic', '<')),
            key=f'{prefix}_l_{idx}',
        )
    if l == 'Between':
        with cc[3]:
            tmn = st.number_input(
                "Min", min_value=0.0, max_value=100.0, step=1.0,
                value=float(default.get('thresh_min', 0.0)),
                key=f'{prefix}_min_{idx}',
            )
        with cc[4]:
            tmx = st.number_input(
                "Max", min_value=0.0, max_value=100.0, step=1.0,
                value=float(default.get('thresh_max', 100.0)),
                key=f'{prefix}_max_{idx}',
            )
        thresh = 0.0
    else:
        with cc[3]:
            thresh = st.number_input(
                "Threshold", min_value=0.0, max_value=100.0, step=1.0,
                value=float(default.get('thresh', 90.0)),
                key=f'{prefix}_th_{idx}',
            )
        tmn, tmx = 0.0, 100.0
    with cc[5]:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("X", key=f'{prefix}_rm_{idx}'):
            st.session_state[list_key].pop(idx)
            st.rerun()
    return {'type': t, 'window': w, 'logic': l, 'thresh': thresh, 'thresh_min': tmn, 'thresh_max': tmx}


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("Rotation Backtester")
st.caption(
    "Pick top-N (or bottom-N in reverse mode) from a basket at each rebalance and "
    "hold until the next rebalance. Equal-weight within picks; slack goes to cash "
    "when filters reject every candidate."
)

# --- Sidebar ---
st.sidebar.header("Globals")
starting_equity = st.sidebar.number_input("Starting Equity ($)", value=100000, step=10000, min_value=1000)
cash_apr = st.sidebar.number_input(
    "Cash APR (%)", value=0.0, step=0.25, min_value=-5.0, max_value=10.0,
    help="Annualized return on cash slack. 0 = no return on uninvested capital.",
)
ts_lookback = st.sidebar.number_input(
    "Time-series rank lookback (trading days)", value=252, min_value=63, max_value=2520, step=21,
    help="Rolling window for the per-ticker time-series percentile rank. 252 ≈ 1 year.",
)

# --- 1. Basket ---
st.subheader("1. Basket")
preset = st.selectbox(
    "Preset", ["(custom)", "Sector + Index ETFs (26)", "Index ETFs only (5)"],
    index=1,
)
if preset.startswith("Sector + Index"):
    default_basket = ", ".join(SECTOR_INDEX_ETFS) if SECTOR_INDEX_ETFS else "SPY, QQQ, IWM"
elif preset.startswith("Index ETFs only"):
    default_basket = ", ".join(INDEX_ETFS) if INDEX_ETFS else "SPY, QQQ, IWM, DIA, SMH"
else:
    default_basket = "SPY, QQQ, IWM, DIA, SMH"
basket_text = st.text_area(
    "Tickers (comma-separated)", value=default_basket, height=70,
    help="Free-text. Anything not in master_prices.parquet is silently dropped.",
)
basket = sorted({t.strip().upper() for t in basket_text.replace('\n', ',').split(',') if t.strip()})
st.caption(f"{len(basket)} tickers parsed")

# --- 2. Date range ---
st.subheader("2. Backtest period")
c1, c2 = st.columns(2)
with c1:
    start_date = st.date_input("Start", value=datetime.date(2010, 1, 1),
                               min_value=datetime.date(1990, 1, 1))
with c2:
    end_date = st.date_input("End", value=datetime.date.today(),
                             min_value=start_date)
st.caption("400-day buffer is added before the start so rolling ranks have history.")

# --- 3. Rebalance ---
st.subheader("3. Rebalance")
c1, c2 = st.columns(2)
with c1:
    rebal_freq = st.selectbox(
        "Frequency",
        ["Weekly (Mon)", "Monthly (EOM)", "Every 5 days", "Every 10 days", "Every 21 days", "Daily"],
        index=0,
    )
with c2:
    execution_lag = st.checkbox(
        "Execute next-day open (T+1 lag)", value=True,
        help="True (recommended): rank at rebal-day close, take position next bar. "
             "False: rank AND take position at the same close (look-ahead in practice).",
    )

# --- 4. Selection rule ---
st.subheader("4. Selection rule")
c1, c2, c3, c4 = st.columns(4)
with c1:
    select_window = st.selectbox("Ranking window", WINDOWS, index=2)
with c2:
    select_type = st.radio(
        "Rank type", RANK_TYPES, index=0,
        format_func=lambda x: RANK_TYPE_LABELS[x],
        horizontal=False,
    )
with c3:
    direction = st.radio("Direction", ["best", "worst"], index=0,
                         format_func=lambda x: "Best (highest rank)" if x == 'best' else "Worst (lowest rank)")
with c4:
    top_n = int(st.number_input("How many to hold", value=1, min_value=1, max_value=10))

# --- 5. Eligibility filters ---
st.subheader("5. Eligibility filters")
st.caption(
    "Applied to candidates BEFORE selection. Each filter narrows the eligible "
    "pool (AND across filters). Empty = no filter, full basket eligible."
)
if 'rot_elig_filters' not in st.session_state:
    st.session_state.rot_elig_filters = []

ec1, ec2 = st.columns([1, 5])
with ec1:
    if st.button("Add eligibility filter"):
        st.session_state.rot_elig_filters.append(
            {'type': 'xsec', 'window': 21, 'logic': '<', 'thresh': 90.0,
             'thresh_min': 0.0, 'thresh_max': 100.0}
        )
        st.rerun()
with ec2:
    if st.button("Clear eligibility filters"):
        st.session_state.rot_elig_filters = []
        st.rerun()

eligibility_filters = []
for i, f in enumerate(list(st.session_state.rot_elig_filters)):
    eligibility_filters.append(_filter_widget('ef', i, f, 'rot_elig_filters'))
st.session_state.rot_elig_filters = eligibility_filters

# --- 6. Reverse mode ---
st.subheader("6. Reverse mode (optional)")
reverse_enabled = st.checkbox(
    "Enable reverse mode",
    value=False,
    help="When EVERY basket member satisfies the reverse-trigger conditions, "
         "rotate into the bottom-N (most-oversold) instead of the top-N. "
         "Useful for deep-oversold mean reversion gates.",
)

if 'rot_rev_filters' not in st.session_state:
    st.session_state.rot_rev_filters = []

if reverse_enabled:
    rc1, rc2, rc3 = st.columns([1, 1, 4])
    with rc1:
        if st.button("Add reverse condition"):
            st.session_state.rot_rev_filters.append(
                {'type': 'ts', 'window': 5, 'logic': '<', 'thresh': 15.0,
                 'thresh_min': 0.0, 'thresh_max': 100.0}
            )
            st.rerun()
    with rc2:
        if st.button("Clear reverse"):
            st.session_state.rot_rev_filters = []
            st.rerun()
    with rc3:
        reverse_n = int(st.number_input(
            "Bottom-N to hold when reverse fires", value=1, min_value=1, max_value=10,
            key='rev_n',
        ))
    st.caption(
        "Reverse fires when ALL filters below evaluate True for ALL basket members "
        "(AND across filters AND across members). Common pattern: time-series rank < 15 "
        "on a short window like 5d — every member is in the bottom 15% of its own history."
    )
    reverse_filters = []
    for i, f in enumerate(list(st.session_state.rot_rev_filters)):
        reverse_filters.append(_filter_widget('rf', i, f, 'rot_rev_filters'))
    st.session_state.rot_rev_filters = reverse_filters
else:
    reverse_filters = []
    reverse_n = 1

# --- Run ---
st.divider()
run_btn = st.button("Run Backtest", type="primary")

if run_btn:
    if not basket:
        st.error("Add at least one ticker to the basket.")
        st.stop()

    with st.spinner("Loading prices..."):
        close_df = load_basket_closes(tuple(basket))
    if close_df.empty:
        st.error("No data loaded for basket. Check tickers exist in master_prices.parquet.")
        st.stop()
    missing = sorted(set(basket) - set(close_df.columns))
    if missing:
        st.warning(f"Dropped (no data): {', '.join(missing)}")

    buffer_start = pd.Timestamp(start_date) - pd.Timedelta(days=400)
    close_df = close_df[(close_df.index >= buffer_start) & (close_df.index <= pd.Timestamp(end_date))]
    if close_df.empty:
        st.error("No price data in selected date range.")
        st.stop()

    with st.spinner("Computing ranks..."):
        rets = compute_returns(close_df, WINDOWS)
        xsec_ranks = compute_xsec_ranks(rets)
        ts_ranks = compute_ts_ranks(rets, int(ts_lookback))

    backtest_cal = close_df.index[close_df.index >= pd.Timestamp(start_date)]
    if backtest_cal.empty:
        st.error("No trading days in the selected window after applying the buffer.")
        st.stop()
    rebal_dates = get_rebal_dates(backtest_cal, rebal_freq)

    with st.spinner(f"Running backtest over {len(rebal_dates)} rebalances..."):
        result = run_backtest(
            close_df=close_df.reindex(backtest_cal),
            rebal_dates=rebal_dates,
            select_window=int(select_window),
            select_type=select_type,
            direction=direction,
            top_n=top_n,
            eligibility_filters=eligibility_filters,
            reverse_enabled=reverse_enabled,
            reverse_filters=reverse_filters,
            reverse_n=reverse_n,
            starting_equity=float(starting_equity),
            execution_lag=execution_lag,
            cash_apr=float(cash_apr),
            xsec_ranks=xsec_ranks,
            ts_ranks=ts_ranks,
        )
    if result is None:
        st.error("No rebalance dates resolved. Try a longer date range or different frequency.")
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
    cash_pct = float((result['cash_weight'] > 0.99).mean() * 100)
    n_rebals = len(result['trade_log'])

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("CAGR", f"{rot_cagr*100:.1f}%", delta=f"{(rot_cagr-bench_cagr)*100:+.1f}% vs basket")
    s2.metric("Sharpe", f"{rot_sharpe:.2f}", delta=f"{rot_sharpe-bench_sharpe:+.2f}")
    s3.metric("Max DD", f"{rot_dd*100:.1f}%", delta=f"{(rot_dd-bench_dd)*100:+.1f}% vs basket",
              delta_color="inverse")
    s4.metric("Time in cash", f"{cash_pct:.1f}%")

    s5, s6, s7, s8 = st.columns(4)
    s5.metric("Total return", f"{rot_total*100:.1f}%")
    s6.metric("Sortino", f"{rot_sortino:.2f}", delta=f"{rot_sortino-bench_sortino:+.2f}")
    s7.metric("Rebalances", f"{n_rebals}")
    s8.metric("Years", f"{n_years:.1f}")

    # Equity curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode='lines',
                             name='Rotation', line=dict(color='#1E88E5', width=2)))
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
                                name='Rotation', line=dict(color='#1E88E5', width=1.5),
                                fill='tozeroy'))
    fig_dd.add_trace(go.Scatter(x=bench_dd_series.index, y=bench_dd_series.values, mode='lines',
                                name='Basket', line=dict(color='#888', width=1, dash='dash')))
    fig_dd.update_layout(height=240, yaxis_title='Drawdown (%)',
                         margin=dict(l=10, r=10, t=30, b=10), title='Drawdown')
    st.plotly_chart(fig_dd, use_container_width=True)

    # Pick frequency
    pick_counts = {}
    mode_counts = {'normal': 0, 'reverse': 0, 'cash': 0, 'no-data': 0, 'no-candidates': 0}
    for _, row in result['trade_log'].iterrows():
        mode_counts[row['mode']] = mode_counts.get(row['mode'], 0) + 1
        for p in row['picks']:
            pick_counts[p] = pick_counts.get(p, 0) + 1
    st.subheader("Pick frequency")
    pf1, pf2 = st.columns([2, 1])
    with pf1:
        if pick_counts:
            pf_df = pd.DataFrame(
                sorted(pick_counts.items(), key=lambda x: -x[1]),
                columns=['Ticker', 'Times Picked'],
            )
            fig_pf = px.bar(pf_df, x='Ticker', y='Times Picked', text='Times Picked')
            fig_pf.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_pf, use_container_width=True)
        else:
            st.info("No tickers were picked — backtest stayed in cash the entire period.")
    with pf2:
        st.markdown("**Mode breakdown**")
        mb_df = pd.DataFrame(
            [(m, c) for m, c in mode_counts.items() if c > 0],
            columns=['Mode', 'Count'],
        )
        st.dataframe(mb_df, hide_index=True, use_container_width=True)

    # Trade log
    with st.expander("Trade log (every rebalance)", expanded=False):
        tl = result['trade_log'].copy()
        if not tl.empty:
            tl['picks'] = tl['picks'].apply(lambda x: ', '.join(x) if x else '(cash)')
            tl['date'] = pd.to_datetime(tl['date']).dt.strftime('%Y-%m-%d')
            st.dataframe(tl, hide_index=True, use_container_width=True)

    # Holdings timeline (sampled monthly)
    with st.expander("Holdings timeline (monthly snapshot)", expanded=False):
        weights = result['weights']
        if not weights.empty:
            monthly = weights.resample('ME').last()
            monthly = monthly.loc[:, (monthly > 0).any(axis=0)]  # drop never-held
            if not monthly.empty:
                fig_hm = px.imshow(
                    monthly.T.values,
                    x=monthly.index, y=monthly.columns,
                    aspect='auto', color_continuous_scale='Blues',
                    labels=dict(x='Date', y='Ticker', color='Weight'),
                )
                fig_hm.update_layout(height=max(280, 26 * len(monthly.columns) + 80),
                                     margin=dict(l=80, r=10, t=20, b=40))
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("No tickers were ever held.")
