"""
Fragility Sizing Lab — replay backtester trades with fragility-adjusted position sizing.

Inputs:
  - data/backtest_sig_df.parquet  (trade log from strat_backtester)
  - data/rd2_fragility.parquet    (daily fragility scores from daily_risk_report)

No price data needed — uses pre-computed entry/exit prices and PnL per share.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

st.set_page_config(page_title="Fragility Sizing Lab", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

DATA_DIR = os.path.join(parent_dir, "data")
SIG_DF_PATH = os.path.join(DATA_DIR, "backtest_sig_df.parquet")
FRAG_PATH = os.path.join(DATA_DIR, "rd2_fragility.parquet")

# Regime buckets (same as strat_backtester)
REGIME_BUCKETS = {
    'Robust': (0, 20), 'Calm': (20, 40), 'Neutral': (40, 60),
    'Elevated': (60, 80), 'Fragile': (80, 100.01),
}
REGIME_ORDER = ['Robust', 'Calm', 'Neutral', 'Elevated', 'Fragile']
REGIME_COLORS = {
    'Robust': '#00CC00', 'Calm': '#7FCC00', 'Neutral': '#FFD700',
    'Elevated': '#FF8C00', 'Fragile': '#CC0000',
}


def _assign_regime(score):
    for name, (lo, hi) in REGIME_BUCKETS.items():
        if lo <= score < hi:
            return name
    return 'Fragile' if score >= 80 else 'Robust'


# ─── Backward-compat: derive missing columns from old parquet ───────────────

def _ensure_columns(sig_df):
    """Add Exit Price / stop_atr / tgt_atr if missing (old parquet format)."""
    needs_exit_price = 'Exit Price' not in sig_df.columns
    needs_atr_cols = 'stop_atr' not in sig_df.columns or 'tgt_atr' not in sig_df.columns

    if needs_exit_price:
        # Derive: exit_price = entry_price + pnl/shares (BUY) or entry_price - pnl/shares (SHORT)
        pnl_per_share = sig_df['PnL'] / sig_df['Shares'].replace(0, np.nan)
        is_long = sig_df['Action'] == 'BUY'
        sig_df['Exit Price'] = np.where(is_long,
                                        sig_df['Price'] + pnl_per_share,
                                        sig_df['Price'] - pnl_per_share)

    if needs_atr_cols:
        # Look up from STRATEGY_BOOK by strategy name
        strat_lookup = {}
        try:
            from strategy_config import STRATEGY_BOOK
            for s in STRATEGY_BOOK:
                strat_lookup[s['name']] = (
                    s['execution'].get('stop_atr', 2.0),
                    s['execution'].get('tgt_atr', 4.0),
                )
        except Exception:
            pass

        def _lookup(row):
            name = row['Strategy']
            # Strip " (LOC)" suffix for lookup
            clean = name.replace(' (LOC)', '')
            sa, ta = strat_lookup.get(clean, (2.0, 4.0))
            return pd.Series({'stop_atr': sa, 'tgt_atr': ta})

        derived = sig_df.apply(_lookup, axis=1)
        sig_df['stop_atr'] = derived['stop_atr']
        sig_df['tgt_atr'] = derived['tgt_atr']

    return sig_df


# ─── Replay engine ──────────────────────────────────────────────────────────

def replay_equity(sig_df, frag_series, starting_equity, min_mult, max_mult, cutoff,
                  enabled_strategies, threshold=55, mode='linear',
                  scale_with_equity=False):
    """
    Replay trades with fragility adjustment.

    When scale_with_equity=False (default): every trade risks the same fixed
    dollar amount for clean avg-R analysis.
    When scale_with_equity=True: risk scales with running equity (baseline uses
    baseline equity, test uses test equity) to see compounding effects.

    Returns DataFrame with one row per trade: baseline + adjusted PnL, shares,
    multiplier, fragility score, regime, running equity.
    """
    df = sig_df.copy()
    df = df[df['Strategy'].isin(enabled_strategies)].sort_values('Exit Date').reset_index(drop=True)

    if df.empty:
        return df

    baseline_equity = starting_equity
    test_equity = starting_equity
    rows = []

    for _, row in df.iterrows():
        signal_date = row['Date']
        # Look up fragility
        frag_score = 0.0
        if frag_series is not None:
            fs = frag_series.asof(signal_date)
            if not pd.isna(fs):
                frag_score = float(fs)

        regime = _assign_regime(frag_score)

        # Risk per trade
        risk_bps = row.get('Risk bps', 20)
        base_risk_equity = baseline_equity if scale_with_equity else starting_equity
        test_risk_equity = test_equity if scale_with_equity else starting_equity
        base_risk_dollar = base_risk_equity * risk_bps / 10000
        test_risk_dollar = test_risk_equity * risk_bps / 10000
        atr = row['ATR']
        stop_atr = row.get('stop_atr', 2.0)
        dist = atr * stop_atr

        if pd.isna(dist) or dist <= 0:
            continue

        base_shares = int(base_risk_dollar / dist)
        if base_shares <= 0:
            continue

        # PnL per share from entry/exit prices
        is_long = row['Action'] == 'BUY'
        if is_long:
            pnl_per_share = row['Exit Price'] - row['Price']
        else:
            pnl_per_share = row['Price'] - row['Exit Price']

        base_pnl = round(pnl_per_share * base_shares, 0)
        baseline_equity += base_pnl

        # Fragility-adjusted sizing
        skipped = False
        if frag_score > cutoff:
            adj_mult = 0.0
            skipped = True
        elif mode == 'linear':
            if frag_score <= threshold:
                if threshold > 0:
                    adj_mult = max_mult - (frag_score / threshold) * (max_mult - 1.0)
                else:
                    adj_mult = max_mult
            else:
                adj_mult = max(min_mult, 1.0 - ((frag_score - threshold) / (100 - threshold)) * (1 - min_mult))
        else:
            adj_mult = 1.0

        test_base_shares = int(test_risk_dollar / dist)
        adj_shares = int(round(test_base_shares * adj_mult)) if not skipped else 0
        adj_pnl = round(pnl_per_share * adj_shares, 0)
        test_equity += adj_pnl

        rows.append({
            'Date': signal_date,
            'Entry Date': row['Entry Date'],
            'Exit Date': row['Exit Date'],
            'Strategy': row['Strategy'],
            'Ticker': row['Ticker'],
            'Action': row['Action'],
            'Price': row['Price'],
            'Exit Price': row['Exit Price'],
            'ATR': atr,
            'Fragility': frag_score,
            'Regime': regime,
            'Multiplier': adj_mult,
            'Skipped': skipped,
            'Orig Shares': base_shares,
            'Adj Shares': adj_shares,
            'Orig PnL': base_pnl,
            'Adj PnL': adj_pnl,
            'Risk $': test_risk_dollar,
            'Baseline Equity': baseline_equity,
            'Test Equity': test_equity,
        })

    return pd.DataFrame(rows)


# ─── Metrics helpers ────────────────────────────────────────────────────────

def _max_drawdown(equity_series):
    """Max drawdown from a series of equity values."""
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    return dd.min() * 100  # as pct


def _daily_sharpe(result_df, pnl_col, starting_equity, periods_per_year=252):
    """Annualized Sharpe from daily aggregate P&L (not per-trade).

    Aggregates trade PnL by exit date, fills non-trading days with 0,
    then computes Sharpe on daily portfolio returns.
    """
    daily_pnl = result_df.groupby('Exit Date')[pnl_col].sum()
    # Fill to calendar trading days so quiet days count as 0
    idx = pd.bdate_range(daily_pnl.index.min(), daily_pnl.index.max())
    daily_pnl = daily_pnl.reindex(idx, fill_value=0.0)
    daily_ret = daily_pnl / starting_equity
    if daily_ret.std() == 0:
        return 0.0
    return (daily_ret.mean() / daily_ret.std()) * np.sqrt(periods_per_year)


# ─── Main UI ────────────────────────────────────────────────────────────────

st.title("Fragility Sizing Lab")
st.caption("Replay backtester trades with fragility-adjusted position sizing. No re-download needed.")

# Check for data
if not os.path.exists(SIG_DF_PATH):
    st.error(f"Trade log not found: `data/backtest_sig_df.parquet`\n\nRun the Backtester first to generate trades.")
    st.stop()
if not os.path.exists(FRAG_PATH):
    st.error(f"Fragility data not found: `data/rd2_fragility.parquet`\n\nRun the Daily Risk Report to generate fragility scores.")
    st.stop()

# Load data
sig_df = pd.read_parquet(SIG_DF_PATH)
sig_df = _ensure_columns(sig_df)

frag_df = pd.read_parquet(FRAG_PATH)

# Filter to fragility coverage period only
frag_start = frag_df.index.min()
total_before_filter = len(sig_df)
sig_df = sig_df[pd.to_datetime(sig_df['Date']) >= frag_start].copy()
trades_dropped = total_before_filter - len(sig_df)

# ─── Sidebar ────────────────────────────────────────────────────────────────

st.sidebar.header("Sizing Parameters")
starting_equity = st.sidebar.number_input("Starting Equity ($)", value=750_000, step=50_000, min_value=10_000)

horizon = st.sidebar.selectbox("Fragility Horizon", ['21d', '5d', '63d'], index=0)
if horizon not in frag_df.columns:
    st.error(f"Horizon `{horizon}` not found in fragility data. Available: {list(frag_df.columns)}")
    st.stop()

smoothing = st.sidebar.selectbox("Fragility Smoothing", ['None', '10d MA', '21d MA'], index=0,
                                  help="Smooth fragility scores before applying sizing rules.")

frag_series = frag_df[horizon].dropna()
# Apply smoothing
if smoothing == '10d MA':
    frag_series = frag_series.rolling(10, min_periods=1).mean()
elif smoothing == '21d MA':
    frag_series = frag_series.rolling(21, min_periods=1).mean()
# Lag by 1 day — at entry we only know yesterday's closing score
frag_series = frag_series.shift(1).dropna()

st.sidebar.markdown("---")
st.sidebar.subheader("Linear Ramp")
threshold = st.sidebar.slider("Ramp threshold (neutral zone)", 0, 95, 55, 5,
                               help="Below this: size UP toward max. Above this: size DOWN toward min.")
max_mult = st.sidebar.slider("Max multiplier (at frag=0)", 1.0, 1.5, 1.0, 0.05,
                              help="Sizing boost when fragility is low. 1.0 = no boost.")
min_mult = st.sidebar.slider("Min multiplier (at frag=100)", 0.1, 1.0, 0.5, 0.05,
                              help=f"Sizing floor when fragility is high.")

st.sidebar.subheader("Step Function Cutoff")
cutoff = st.sidebar.slider("Skip trades above fragility", 0, 150, 150, 5,
                            help="Trades with fragility above this threshold are skipped entirely. Set to 150 to disable.")

scale_with_equity = st.sidebar.checkbox("Scale size with portfolio growth", value=False,
                                        help="When enabled, risk per trade scales with running equity instead of fixed starting equity.")

# ─── Sizing schedule preview ───────────────────────────────────────────────

def _compute_mult(score, threshold, max_mult, min_mult, cutoff):
    """Compute the sizing multiplier for a given fragility score."""
    if score > cutoff:
        return 0.0
    if score <= threshold:
        if threshold > 0:
            return max_mult - (score / threshold) * (max_mult - 1.0)
        return max_mult
    return max(min_mult, 1.0 - ((score - threshold) / (100 - threshold)) * (1 - min_mult))

# Sample points: regime boundaries + midpoints
_schedule_scores = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
_schedule_mults = [_compute_mult(s, threshold, max_mult, min_mult, cutoff) for s in _schedule_scores]
_schedule_regimes = [_assign_regime(s) for s in _schedule_scores]

_sched_df = pd.DataFrame({
    'Fragility': _schedule_scores,
    'Regime': _schedule_regimes,
    'Multiplier': _schedule_mults,
    'Sizing %': [f"{m:.0%}" for m in _schedule_mults],
})

st.sidebar.markdown("---")
st.sidebar.subheader("Sizing Schedule")
st.sidebar.dataframe(
    _sched_df[['Fragility', 'Regime', 'Sizing %']].style.map(
        lambda v: f'color: {REGIME_COLORS.get(v, "inherit")}',
        subset=['Regime'],
    ),
    hide_index=True,
    use_container_width=True,
    height=422,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Strategies")
all_strategies = sorted(sig_df['Strategy'].unique())
enabled = {}
for s in all_strategies:
    enabled[s] = st.sidebar.checkbox(s, value=True, key=f"strat_{s}")
enabled_strategies = [s for s, v in enabled.items() if v]

if not enabled_strategies:
    st.warning("No strategies enabled.")
    st.stop()

# ─── Run replay ─────────────────────────────────────────────────────────────

if trades_dropped > 0:
    st.info(f"Filtered to fragility coverage period ({frag_start.strftime('%Y-%m-%d')}+). {trades_dropped:,} earlier trades excluded.")

if sig_df.empty:
    st.warning("No trades fall within fragility data coverage period.")
    st.stop()

result = replay_equity(sig_df, frag_series, starting_equity, min_mult, max_mult, cutoff,
                       enabled_strategies, threshold=threshold, scale_with_equity=scale_with_equity)

if result.empty:
    st.warning("No trades to replay with selected strategies.")
    st.stop()

# ─── Section 1: Comparison Metrics ──────────────────────────────────────────

st.subheader("Comparison Metrics")

baseline_total_return = (result['Baseline Equity'].iloc[-1] / starting_equity - 1) * 100
test_total_return = (result['Test Equity'].iloc[-1] / starting_equity - 1) * 100

baseline_dd = _max_drawdown(result['Baseline Equity'])
test_dd = _max_drawdown(result['Test Equity'])

baseline_sharpe = _daily_sharpe(result, 'Orig PnL', starting_equity)
test_sharpe = _daily_sharpe(result, 'Adj PnL', starting_equity)

trades_skipped = result['Skipped'].sum()
total_trades = len(result)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Return", f"{test_total_return:.1f}%",
              delta=f"{test_total_return - baseline_total_return:+.1f}%")
    st.caption(f"Baseline: {baseline_total_return:.1f}%")
with c2:
    st.metric("Max Drawdown", f"{test_dd:.1f}%",
              delta=f"{test_dd - baseline_dd:+.1f}%")
    st.caption(f"Baseline: {baseline_dd:.1f}%")
with c3:
    st.metric("Sharpe (daily)", f"{test_sharpe:.2f}",
              delta=f"{test_sharpe - baseline_sharpe:+.2f}")
    st.caption(f"Baseline: {baseline_sharpe:.2f}")
with c4:
    st.metric("Trades Skipped", f"{trades_skipped:,} / {total_trades:,}")
    st.caption(f"{trades_skipped / total_trades * 100:.0f}% of trades" if total_trades > 0 else "")

# ─── Section 2: Equity Curves ───────────────────────────────────────────────

st.subheader("Equity Curves")

fig_eq = go.Figure()

fig_eq.add_trace(go.Scatter(
    x=result['Exit Date'], y=result['Baseline Equity'],
    mode='lines', name='Baseline (no adjustment)',
    line=dict(color='gray', width=1.5),
))
fig_eq.add_trace(go.Scatter(
    x=result['Exit Date'], y=result['Test Equity'],
    mode='lines', name=f'Test (min={min_mult}, cutoff={cutoff})',
    line=dict(color='#1f77b4', width=2),
))

fig_eq.update_layout(
    height=400,
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    yaxis_title="Equity ($)",
    yaxis_type="log",
    xaxis_title="",
    hovermode="x unified",
)
st.plotly_chart(fig_eq, use_container_width=True)

# ─── Section 3: Fragility Impact by Strategy ────────────────────────────────

st.subheader("Fragility Impact by Strategy")

impact = result.groupby('Strategy').agg(
    Orig_PnL=('Orig PnL', 'sum'),
    Adj_PnL=('Adj PnL', 'sum'),
    Trades=('Orig PnL', 'count'),
    Skipped=('Skipped', 'sum'),
).reset_index()
impact['Delta'] = impact['Adj_PnL'] - impact['Orig_PnL']
impact = impact.sort_values('Delta')

fig_impact = go.Figure()
fig_impact.add_trace(go.Bar(
    y=impact['Strategy'], x=impact['Delta'],
    orientation='h',
    marker_color=np.where(impact['Delta'] >= 0, '#2ca02c', '#d62728'),
    text=impact['Delta'].apply(lambda x: f"${x:+,.0f}"),
    textposition='auto',
))
fig_impact.update_layout(
    height=max(250, len(impact) * 35),
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis_title="PnL Delta ($)",
    yaxis_title="",
)
st.plotly_chart(fig_impact, use_container_width=True)

# Impact table
st.dataframe(
    impact.rename(columns={
        'Orig_PnL': 'Baseline PnL',
        'Adj_PnL': 'Adjusted PnL',
    }).style.format({
        'Baseline PnL': '${:,.0f}',
        'Adjusted PnL': '${:,.0f}',
        'Delta': '${:+,.0f}',
        'Trades': '{:,.0f}',
        'Skipped': '{:,.0f}',
    }),
    use_container_width=True,
    hide_index=True,
)

# ─── Section 4: Strategy Performance by Regime ──────────────────────────────

st.subheader("Strategy Performance by Regime")

regime_trades = result[~result['Skipped']].copy()

if not regime_trades.empty:
    # Compute R-multiple: PnL / risk$
    regime_trades['R'] = regime_trades['Adj PnL'] / regime_trades['Risk $'].replace(0, np.nan)

    # Pivot: Strategy x Regime → Avg R per trade
    pivot = regime_trades.pivot_table(
        index='Strategy', columns='Regime', values='R', aggfunc='mean', fill_value=0
    )
    # Reorder columns
    pivot = pivot.reindex(columns=[r for r in REGIME_ORDER if r in pivot.columns])

    # Add total portfolio row (avg R across all strategies per regime)
    portfolio_row = regime_trades.groupby('Regime')['R'].mean()
    pivot.loc['Total Portfolio'] = portfolio_row.reindex(pivot.columns, fill_value=0)

    def _r_color(val):
        if val > 0.5:
            return 'background-color: #1a7a1a; color: #c0f0c0'
        elif val > 0.3:
            return 'background-color: #2ca02c; color: #e0ffe0'
        elif val > 0.15:
            return 'background-color: #90d890; color: #1a5c1a'
        elif val >= 0:
            return 'background-color: #d4f5d4; color: #2a7a2a'
        elif val >= -0.1:
            return 'background-color: #f4a0a0; color: #7a1a1a'
        else:
            return 'background-color: #cc0000; color: #ffcccc'

    st.dataframe(
        pivot.style.format('{:.2f}R').map(_r_color),
        use_container_width=True,
    )

    with st.expander("Detailed Breakdown"):
        for regime in REGIME_ORDER:
            rdf = regime_trades[regime_trades['Regime'] == regime]
            if rdf.empty:
                continue

            st.markdown(f"**{regime}** ({len(rdf):,} trades)")

            detail = rdf.groupby('Strategy').agg(
                Trades=('Adj PnL', 'count'),
                Total_PnL=('Adj PnL', 'sum'),
                Win_Rate=('Adj PnL', lambda x: (x > 0).mean()),
                Avg_PnL=('Adj PnL', 'mean'),
            ).reset_index()

            # Profit factor
            def _pf(grp):
                wins = grp[grp > 0].sum()
                losses = abs(grp[grp < 0].sum())
                return wins / losses if losses > 0 else np.inf

            pf = rdf.groupby('Strategy')['Adj PnL'].apply(_pf).reset_index()
            pf.columns = ['Strategy', 'PF']
            detail = detail.merge(pf, on='Strategy')

            st.dataframe(
                detail.style.format({
                    'Total_PnL': '${:,.0f}',
                    'Win_Rate': '{:.1%}',
                    'Avg_PnL': '${:,.0f}',
                    'PF': '{:.2f}',
                    'Trades': '{:,.0f}',
                }),
                use_container_width=True,
                hide_index=True,
            )
else:
    st.info("All trades were skipped at current cutoff — no regime data to show.")

# ─── Section 5: Trade Log ───────────────────────────────────────────────────

st.subheader("Trade Log")

display_cols = [
    'Date', 'Exit Date', 'Strategy', 'Ticker', 'Action',
    'Fragility', 'Regime', 'Multiplier', 'Skipped',
    'Orig Shares', 'Adj Shares', 'Orig PnL', 'Adj PnL',
]
log_df = result[display_cols].copy()

st.dataframe(
    log_df.style.format({
        'Fragility': '{:.1f}',
        'Multiplier': '{:.2f}',
        'Orig Shares': '{:,.0f}',
        'Adj Shares': '{:,.0f}',
        'Orig PnL': '${:,.0f}',
        'Adj PnL': '${:,.0f}',
    }).map(
        lambda v: f'color: {REGIME_COLORS.get(v, "inherit")}',
        subset=['Regime'],
    ),
    use_container_width=True,
    hide_index=True,
    height=500,
)
