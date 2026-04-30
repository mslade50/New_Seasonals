"""
Exposure Backtester — buy-and-hold base portfolio with rule-based exposure tweaks.

Define a base allocation (e.g. 50% SPY, 50% QQQ), then layer rules that
scale exposure up or down when ALL of a rule's conditions fire (e.g.
"reduce QQQ to 50% when 21D return rank > 95 AND 5D ATR sznl < 25").
Rules compose multiplicatively across each other; conditions within a rule
are AND-joined. Slack goes to cash.

Companion to pages/backtester.py — same data layer (master_prices.parquet
+ atr_seasonal_ranks.parquet), but no per-trade signal logic. Just
continuous holdings with time-varying weights.
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
ATR_SZNL_PATH = "atr_seasonal_ranks.parquet"
SZNL_PATH = "sznl_ranks.csv"
FRAGILITY_PATH = "data/rd2_fragility.parquet"
ATR_SZNL_WINDOWS = [5, 10, 21, 63, 126, 252]
ATR_SZNL_COLS = [f"atr_sznl_{w}d" for w in ATR_SZNL_WINDOWS]
FRAGILITY_WINDOWS = [21, 63]
RANK_MIN_PERIODS = 252  # min history before percentile rank is meaningful

st.set_page_config(page_title="Exposure Backtester", layout="wide")

# -----------------------------------------------------------------------------
# DATA LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_atr_sznl_map():
    """{ticker: DataFrame indexed by date with columns atr_sznl_5d, _10d, ...}."""
    path = os.path.join(parent_dir, ATR_SZNL_PATH)
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_parquet(path)
        df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
        out = {}
        for tkr, grp in df.groupby('ticker'):
            out[str(tkr).upper()] = grp.set_index('Date')[ATR_SZNL_COLS].sort_index()
        return out
    except Exception as e:
        st.warning(f"ATR sznl parquet load failed: {e}")
        return {}


@st.cache_resource
def load_fragility_dials():
    """Load the daily fragility dial history from the Risk Dashboard.

    Returns a DataFrame indexed by date with columns:
      - '5d', '21d', '63d'           — raw fragility scores (0-100)
      - '21d_ma10', '63d_ma10'       — 10-day moving averages of the same
    Coverage starts 2016-05-02; pre-history dates evaluate False on any
    fragility-based condition.
    """
    path = os.path.join(parent_dir, FRAGILITY_PATH)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        st.warning(f"fragility parquet load failed: {e}")
        return None
    if df is None or df.empty:
        return None
    df.index = pd.to_datetime(df.index).normalize()
    try:
        df.index = df.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    df = df.sort_index()
    # Add 10d MA versions for the dials we expose to rule conditions.
    for w in FRAGILITY_WINDOWS:
        col = f"{w}d"
        if col in df.columns:
            df[f"{col}_ma10"] = df[col].rolling(10, min_periods=1).mean()
    return df


@st.cache_resource
def load_sznl_map():
    """{ticker: Series indexed by date with seasonal_rank}.
    Plain (non-ATR-normalized) seasonal rank from sznl_ranks.csv."""
    path = os.path.join(parent_dir, SZNL_PATH)
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        if 'Date' not in df.columns or 'ticker' not in df.columns:
            return {}
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()
        df = df.dropna(subset=['Date'])
        df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
        out = {}
        for tkr, grp in df.groupby('ticker'):
            out[tkr] = grp.set_index('Date')['seasonal_rank'].sort_index()
        return out
    except Exception as e:
        st.warning(f"sznl_ranks.csv load failed: {e}")
        return {}


@st.cache_data(ttl=3600)
def load_prices(tickers, start, end):
    """Load OHLCV via master_prices.parquet, falling back to yfinance for any
    ticker not in the master. Returns {ticker: DataFrame[OHLCV]}."""
    tickers = sorted(set(t.strip().upper() for t in tickers if t and t.strip()))
    out = {}
    if data_provider.has_master():
        out = data_provider.get_history(tickers, start=start, end=end)
    missing = [t for t in tickers if t not in out or out[t].empty]
    if missing:
        try:
            import yfinance as yf
            raw = yf.download(missing, start=start, end=end, auto_adjust=True, progress=False)
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    for t in missing:
                        try:
                            sub = raw.xs(t, level=1, axis=1).copy()
                            sub.columns = [str(c).capitalize() for c in sub.columns]
                            if 'Close' in sub.columns and not sub['Close'].dropna().empty:
                                out[t] = sub
                        except Exception:
                            pass
                else:
                    sub = raw.copy()
                    sub.columns = [str(c).capitalize() for c in sub.columns]
                    if missing and 'Close' in sub.columns:
                        out[missing[0]] = sub
        except Exception as e:
            st.warning(f"yfinance fallback failed for {missing}: {e}")
    return out


def compute_return_rank(close_series, window):
    """Expanding percentile rank of N-day returns (0-100).

    Identical recipe to indicators.py rank_ret_{window}d: 252-day min_periods
    expanding rank of pct_change(window, fill_method=None). pages/backtester.py
    uses the same series for its perf_filters, so a 21D return rank > 95 here
    matches a 21D rank filter > 95 there for the same ticker on the same date.

    Note: this is the *temporal* rank (vs the same ticker's history), not the
    cross-sectional rank built by build_xsec_rank_matrices in backtester.py.
    """
    ret = close_series.pct_change(window, fill_method=None)
    return ret.expanding(min_periods=RANK_MIN_PERIODS).rank(pct=True) * 100.0


# -----------------------------------------------------------------------------
# CONDITION EVALUATION
# -----------------------------------------------------------------------------
SIGNAL_TYPES = [
    "ATR Sznl Rank",
    "Return Rank",
    "Seasonal Rank",
    "Fragility Dial (Raw)",
    "Fragility Dial (10d MA)",
]
LOGIC_OPS = ["<", ">", "Between"]
# Signals that ignore indicator_ticker because they're market-wide
MARKET_WIDE_SIGNALS = {"Fragility Dial (Raw)", "Fragility Dial (10d MA)"}


def parse_portfolio_text(text):
    """Parse 'SPY:50, QQQ:50' into [(ticker, weight_pct), ...]."""
    if not text or not text.strip():
        return []
    pairs = []
    chunks = [c for c in text.replace('\n', ',').split(',') if c.strip()]
    for chunk in chunks:
        parts = chunk.replace(':', ' ').split()
        if len(parts) < 2:
            continue
        tkr = parts[0].strip().upper()
        try:
            w = float(parts[1])
        except ValueError:
            continue
        if tkr and w > 0:
            pairs.append((tkr, w))
    return pairs


def evaluate_condition_triggered(cond, indicator_series):
    """Boolean Series — True on dates where condition is satisfied."""
    op = cond['logic']
    if op == '<':
        return indicator_series < cond['thresh']
    elif op == '>':
        return indicator_series > cond['thresh']
    elif op == 'Between':
        return (indicator_series >= cond['thresh_min']) & (indicator_series <= cond['thresh_max'])
    return pd.Series(False, index=indicator_series.index)


def get_indicator_series(cond, target_ticker, prices_dict, atr_sznl_map,
                         sznl_map, fragility_df):
    """Resolve which indicator series this condition reads."""
    sig = cond['signal_type']
    win = int(cond['window'])

    # Market-wide signals (fragility) — indicator_ticker is ignored
    if sig in MARKET_WIDE_SIGNALS:
        if fragility_df is None or fragility_df.empty:
            return None, "fragility parquet not loaded"
        if sig == "Fragility Dial (Raw)":
            col = f"{win}d"
        else:
            col = f"{win}d_ma10"
        if col not in fragility_df.columns:
            return None, f"fragility column {col} not found"
        return fragility_df[col], None

    ind_tkr = cond.get('indicator_ticker', 'self')
    if ind_tkr == 'self' or not ind_tkr:
        ind_tkr = target_ticker
    ind_tkr = str(ind_tkr).upper()

    if sig == "ATR Sznl Rank":
        rank_df = atr_sznl_map.get(ind_tkr)
        if rank_df is None or rank_df.empty:
            return None, f"ATR sznl missing for {ind_tkr}"
        col = f"atr_sznl_{win}d"
        if col not in rank_df.columns:
            return None, f"window {win}D not in ATR sznl columns"
        return rank_df[col], None
    elif sig == "Return Rank":
        df = prices_dict.get(ind_tkr)
        if df is None or df.empty or 'Close' not in df.columns:
            return None, f"prices missing for {ind_tkr}"
        return compute_return_rank(df['Close'], win), None
    elif sig == "Seasonal Rank":
        s = sznl_map.get(ind_tkr)
        if s is None or s.empty:
            return None, f"sznl missing for {ind_tkr}"
        return s, None
    return None, f"unknown signal {sig}"


def cond_label(cond):
    sig = cond['signal_type']
    if sig in MARKET_WIDE_SIGNALS:
        prefix = f"{sig} {cond['window']}D"
    else:
        prefix = f"{cond.get('indicator_ticker', 'self')} {sig} {cond['window']}D"
    if cond['logic'] == 'Between':
        return f"{prefix} ∈ [{cond['thresh_min']:.0f},{cond['thresh_max']:.0f}]"
    return f"{prefix} {cond['logic']} {cond['thresh']:.0f}"


# -----------------------------------------------------------------------------
# ENGINE
# -----------------------------------------------------------------------------
def run_backtest(base_weights, rules, prices_dict, atr_sznl_map, sznl_map,
                 fragility_df, start_date, end_date, cash_apr=0.0,
                 starting_equity=100000.0):
    """Walk forward applying time-varying weights to base portfolio.

    rules: list of {target, scaler, conditions: [{indicator_ticker,
           signal_type, window, logic, thresh / thresh_min / thresh_max}, ...]}.
    Conditions within a rule are AND-joined. Rules compose multiplicatively.
    """
    if not base_weights:
        return None

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    asset_tickers = [t for t, _ in base_weights]

    closes = {}
    for t in asset_tickers:
        df = prices_dict.get(t)
        if df is None or df.empty or 'Close' not in df.columns:
            continue
        s = df['Close'].copy()
        s.index = pd.to_datetime(s.index).normalize()
        s = s[(s.index >= start_ts) & (s.index <= end_ts)].dropna()
        closes[t] = s
    if not closes:
        return None

    cal = sorted(set().union(*(s.index for s in closes.values())))
    cal = pd.DatetimeIndex(cal)
    if len(cal) < 2:
        return None

    close_df = pd.DataFrame({t: closes[t].reindex(cal).ffill() for t in closes.keys()})
    ret_df = close_df.pct_change().fillna(0.0)

    raw_w = np.array([w for _, w in base_weights], dtype=float)
    if raw_w.sum() <= 0:
        return None
    base_frac = pd.Series(raw_w / raw_w.sum(), index=[t for t, _ in base_weights])
    base_frac = base_frac[[t for t in base_frac.index if t in close_df.columns]]
    if base_frac.empty:
        return None
    base_frac = base_frac / base_frac.sum()

    asset_scaler = pd.DataFrame(1.0, index=cal, columns=base_frac.index)
    portfolio_scaler = pd.Series(1.0, index=cal)
    rule_diagnostics = []  # (rule_idx, target, n_triggered, [errors])

    for r_i, rule in enumerate(rules):
        target = rule.get('target', 'ALL')
        scaler_val = float(rule.get('scaler', 1.0))
        conditions = rule.get('conditions', [])
        if not conditions:
            rule_diagnostics.append((r_i, target, 0, ['no conditions']))
            continue

        if target != 'ALL' and target not in asset_scaler.columns:
            rule_diagnostics.append((r_i, target, 0, [f'target {target} not in portfolio']))
            continue

        target_for_self = target if target != 'ALL' else (
            rule.get('all_indicator_default') or base_frac.index[0]
        )
        triggered = pd.Series(True, index=cal)
        errors = []
        for c_i, cond in enumerate(conditions):
            ind_series, err = get_indicator_series(
                cond, target_for_self, prices_dict, atr_sznl_map, sznl_map,
                fragility_df,
            )
            if ind_series is None:
                errors.append(f"condition {c_i+1}: {err}")
                triggered = pd.Series(False, index=cal)
                break
            ind_series = ind_series.reindex(cal).ffill()
            cond_trig = evaluate_condition_triggered(cond, ind_series).fillna(False)
            triggered = triggered & cond_trig

        n_triggered = int(triggered.sum())
        rule_diagnostics.append((r_i, target, n_triggered, errors))

        if n_triggered == 0:
            continue
        mult = pd.Series(np.where(triggered, scaler_val, 1.0), index=cal)
        if target == 'ALL':
            portfolio_scaler *= mult
        else:
            asset_scaler[target] = asset_scaler[target] * mult

    eff_w = asset_scaler.multiply(base_frac, axis=1).multiply(portfolio_scaler, axis=0)
    eff_w = eff_w.fillna(0.0)
    cash_w = (1.0 - eff_w.sum(axis=1)).clip(lower=-2.0)

    eff_w_lagged = eff_w.shift(1).fillna(0.0)
    cash_w_lagged = cash_w.shift(1).fillna(1.0)
    daily_cash = (cash_apr / 100.0) / 252.0

    port_ret = (eff_w_lagged * ret_df).sum(axis=1) + cash_w_lagged * daily_cash
    equity = starting_equity * (1 + port_ret).cumprod()
    equity.iloc[0] = starting_equity

    bench_ret = (ret_df * base_frac).sum(axis=1)
    bench_eq = starting_equity * (1 + bench_ret).cumprod()
    bench_eq.iloc[0] = starting_equity

    return {
        'cal': cal,
        'equity': equity,
        'benchmark': bench_eq,
        'port_ret': port_ret,
        'bench_ret': bench_ret,
        'eff_weights': eff_w,
        'cash_weight': cash_w,
        'asset_scaler': asset_scaler,
        'portfolio_scaler': portfolio_scaler,
        'base_frac': base_frac,
        'close_df': close_df,
        'rule_diagnostics': rule_diagnostics,
    }


# -----------------------------------------------------------------------------
# STATS
# -----------------------------------------------------------------------------
def compute_stats(equity, daily_ret, name="Strategy"):
    eq = equity.dropna()
    if len(eq) < 2:
        return {'Name': name}
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
    cagr = (1 + total_ret) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    r = daily_ret.dropna()
    vol = float(r.std() * np.sqrt(252))
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    downside = r[r < 0]
    sortino = float(r.mean() / downside.std() * np.sqrt(252)) if not downside.empty and downside.std() > 0 else 0.0
    rolling_max = eq.cummax()
    dd = eq / rolling_max - 1
    max_dd = float(dd.min())
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0
    return {
        'Name': name,
        'CAGR': cagr,
        'Vol (Ann.)': vol,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Max DD': max_dd,
        'Calmar': calmar,
        'Total Return': total_ret,
        'Years': yrs,
    }


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("📊 Exposure Backtester")
st.caption(
    "Define a base portfolio, then layer rules that scale exposure up or down "
    "when ALL of a rule's conditions fire. Rules compose multiplicatively. "
    "Slack goes to cash."
)

# --- Sidebar ---
st.sidebar.header("Globals")
starting_equity = st.sidebar.number_input(
    "Starting Equity ($)", value=100000, step=10000, min_value=1000,
)
cash_apr = st.sidebar.number_input(
    "Cash APR (%)", value=0.0, step=0.25, min_value=-5.0, max_value=10.0,
    help="Annualized return on cash slack. 0 = no return on uninvested capital.",
)

# --- Section 1: Base Portfolio ---
st.subheader("1. Base Portfolio")
portfolio_text = st.text_area(
    "Tickers and weights (e.g. `SPY:50, QQQ:50` or `SPY 60, QQQ 30, GLD 10`)",
    value="SPY:50, QQQ:50",
    height=70,
    help="Weights are percentages; if they don't sum to 100 they'll be renormalized.",
)
base_weights = parse_portfolio_text(portfolio_text)
if base_weights:
    pw_df = pd.DataFrame(base_weights, columns=['Ticker', 'Weight'])
    total_w = pw_df['Weight'].sum()
    pw_df['Effective %'] = (pw_df['Weight'] / total_w * 100).round(1)
    st.dataframe(pw_df, hide_index=True, width='stretch')
    if abs(total_w - 100) > 0.01:
        st.caption(f"⚠️ Weights sum to {total_w:.1f}% — will be renormalized to 100%.")
else:
    st.warning("Add at least one ticker:weight pair.")

# --- Section 2: Date Range ---
st.subheader("2. Backtest Period")
c1, c2 = st.columns(2)
with c1:
    start_date = st.date_input(
        "Start", value=datetime.date(2010, 1, 1),
        min_value=datetime.date(1990, 1, 1),
    )
with c2:
    end_date = st.date_input(
        "End", value=datetime.date.today(),
        min_value=start_date,
    )
data_buffer_start = pd.Timestamp(start_date) - pd.Timedelta(days=400)
st.caption(
    f"Data buffer: 400 days before start (to {data_buffer_start.date()}) so "
    "expanding-rank indicators have history to anchor against."
)

# --- Section 3: Rules ---
st.subheader("3. Exposure Rules")
st.caption(
    "Each rule = target + scaler + ≥1 conditions. Conditions AND-join (all "
    "must fire for the rule to trigger). Rules compose across each other "
    "multiplicatively. `indicator_ticker = self` reads the target asset's own "
    "indicator; explicit symbols (e.g. `SPY`) gate on a market indicator."
)


def _new_condition():
    return {
        'indicator_ticker': 'self',
        'signal_type': 'ATR Sznl Rank',
        'window': 5,
        'logic': '<',
        'thresh': 30.0,
        'thresh_min': 0.0,
        'thresh_max': 100.0,
    }


def _new_rule():
    return {
        'target': 'ALL',
        'scaler': 0.5,
        'conditions': [_new_condition()],
    }


if 'exp_rules' not in st.session_state:
    # Seed: user's example — combo rule on QQQ
    st.session_state.exp_rules = [
        {
            'target': 'QQQ',
            'scaler': 0.5,
            'conditions': [
                {'indicator_ticker': 'self', 'signal_type': 'ATR Sznl Rank', 'window': 5,
                 'logic': '<', 'thresh': 25.0, 'thresh_min': 0.0, 'thresh_max': 100.0},
                {'indicator_ticker': 'self', 'signal_type': 'Return Rank', 'window': 21,
                 'logic': '>', 'thresh': 95.0, 'thresh_min': 0.0, 'thresh_max': 100.0},
            ],
        },
    ]

cc1, cc2 = st.columns([1, 5])
with cc1:
    if st.button("➕ Add Rule"):
        st.session_state.exp_rules.append(_new_rule())
        st.rerun()
with cc2:
    if st.button("🧹 Clear All Rules"):
        st.session_state.exp_rules = []
        st.rerun()

asset_options = ['ALL'] + [t for t, _ in base_weights]

rules = []
for i, r in enumerate(st.session_state.exp_rules):
    cond_strs = [cond_label(c) for c in r.get('conditions', [])]
    cond_summary = " AND ".join(cond_strs) if cond_strs else "(no conditions)"
    with st.expander(
        f"Rule {i+1}: {r['target']} × {r['scaler']:.2f} when [{cond_summary}]",
        expanded=False,
    ):
        # --- rule header: target + scaler + remove ---
        h1, h2, h3 = st.columns([1, 1, 1])
        with h1:
            tgt_default = r['target'] if r['target'] in asset_options else 'ALL'
            target = st.selectbox(
                "Apply to", asset_options,
                index=asset_options.index(tgt_default),
                key=f'rule_target_{i}',
            )
        with h2:
            scaler = st.number_input(
                "Scaler when triggered", value=float(r['scaler']),
                min_value=0.0, max_value=3.0, step=0.1, key=f'rule_scaler_{i}',
                help="1.0 = no change; 0.0 = move to cash; 0.5 = half exposure; 1.5 = 1.5× boost.",
            )
        with h3:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            if st.button("🗑️ Remove rule", key=f'rule_remove_{i}'):
                st.session_state.exp_rules.pop(i)
                st.rerun()

        st.markdown("**Conditions** — all must fire (AND)")

        new_conditions = []
        for c_i, cond in enumerate(r.get('conditions', [])):
            cc1_, cc2_, cc3_, cc4_, cc5_ = st.columns([1, 1, 1, 1, 1])
            with cc2_:
                sig = st.selectbox(
                    "Signal", SIGNAL_TYPES,
                    index=SIGNAL_TYPES.index(cond['signal_type']) if cond['signal_type'] in SIGNAL_TYPES else 0,
                    key=f'cond_sig_{i}_{c_i}',
                )
            with cc1_:
                if sig in MARKET_WIDE_SIGNALS:
                    st.text("(market-wide)")
                    ind_tkr = ''
                else:
                    ind_default = cond.get('indicator_ticker', 'self')
                    ind_tkr = st.text_input(
                        "Ind. ticker", value=ind_default,
                        key=f'cond_ind_{i}_{c_i}',
                        help="`self` = target asset; or explicit symbol like SPY",
                    )
            with cc3_:
                if sig == "ATR Sznl Rank":
                    win_idx = ATR_SZNL_WINDOWS.index(cond['window']) if cond['window'] in ATR_SZNL_WINDOWS else 0
                    window = st.selectbox(
                        "Window (days)", ATR_SZNL_WINDOWS,
                        index=win_idx,
                        key=f'cond_win_{i}_{c_i}',
                    )
                elif sig in MARKET_WIDE_SIGNALS:
                    win_idx = FRAGILITY_WINDOWS.index(cond['window']) if cond['window'] in FRAGILITY_WINDOWS else 0
                    window = st.selectbox(
                        "Dial (days)", FRAGILITY_WINDOWS,
                        index=win_idx,
                        key=f'cond_win_{i}_{c_i}',
                    )
                elif sig == "Seasonal Rank":
                    window = 0  # not used for plain seasonal_rank
                    st.text("daily")
                else:  # Return Rank
                    window = st.number_input(
                        "Window (days)", value=int(cond['window']),
                        min_value=1, max_value=252,
                        step=1, key=f'cond_win_{i}_{c_i}',
                    )
            with cc4_:
                logic = st.selectbox(
                    "Logic", LOGIC_OPS,
                    index=LOGIC_OPS.index(cond['logic']) if cond['logic'] in LOGIC_OPS else 0,
                    key=f'cond_logic_{i}_{c_i}',
                )
            with cc5_:
                if logic == 'Between':
                    sub1, sub2 = st.columns(2)
                    with sub1:
                        thresh_min = st.number_input(
                            "Min", value=float(cond['thresh_min']),
                            min_value=0.0, max_value=100.0, step=5.0,
                            key=f'cond_min_{i}_{c_i}',
                        )
                    with sub2:
                        thresh_max = st.number_input(
                            "Max", value=float(cond['thresh_max']),
                            min_value=0.0, max_value=100.0, step=5.0,
                            key=f'cond_max_{i}_{c_i}',
                        )
                    thresh = thresh_min
                else:
                    thresh = st.number_input(
                        "Threshold %ile", value=float(cond['thresh']),
                        min_value=0.0, max_value=100.0, step=5.0,
                        key=f'cond_thresh_{i}_{c_i}',
                    )
                    thresh_min, thresh_max = 0.0, 100.0

            rmc1, rmc2 = st.columns([1, 6])
            with rmc1:
                if st.button("🗑️ Remove cond", key=f'cond_remove_{i}_{c_i}'):
                    st.session_state.exp_rules[i]['conditions'].pop(c_i)
                    st.rerun()

            new_conditions.append({
                'indicator_ticker': ind_tkr.strip() or 'self',
                'signal_type': sig,
                'window': int(window),
                'logic': logic,
                'thresh': float(thresh),
                'thresh_min': float(thresh_min),
                'thresh_max': float(thresh_max),
            })

        if st.button("➕ Add condition", key=f'add_cond_{i}'):
            st.session_state.exp_rules[i]['conditions'].append(_new_condition())
            st.rerun()

        rules.append({
            'target': target,
            'scaler': float(scaler),
            'conditions': new_conditions,
        })

st.session_state.exp_rules = rules

# --- Section 4: Run ---
st.markdown("---")
if st.button("⚡ Run Backtest", type="primary"):
    if not base_weights:
        st.error("Define a base portfolio first.")
        st.stop()

    needed = {t for t, _ in base_weights}
    for r in rules:
        for c in r.get('conditions', []):
            ind = c.get('indicator_ticker', 'self')
            if ind and ind != 'self':
                needed.add(str(ind).upper())

    with st.spinner(f"Loading prices for {len(needed)} tickers..."):
        prices_dict = load_prices(
            list(needed),
            start=data_buffer_start.strftime('%Y-%m-%d'),
            end=(end_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d'),
        )
    missing = [t for t in needed if t not in prices_dict or prices_dict[t].empty]
    if missing:
        st.warning(f"No price data for: {missing}. They'll be skipped.")

    with st.spinner("Loading indicators..."):
        atr_sznl_map = load_atr_sznl_map()
        sznl_map = load_sznl_map()
        fragility_df = load_fragility_dials()
    if not atr_sznl_map:
        st.warning("ATR sznl parquet not found — `ATR Sznl Rank` conditions will fail.")
    if not sznl_map:
        st.info("sznl_ranks.csv not found — `Seasonal Rank` conditions will fail.")
    if fragility_df is None:
        st.info("Fragility parquet not found — `Fragility Dial` conditions will fail.")
    elif fragility_df.index.min() > pd.Timestamp(start_date):
        st.caption(
            f"⚠️ Fragility dial coverage starts {fragility_df.index.min().date()}; "
            "any fragility condition will evaluate False on dates before that."
        )

    with st.spinner("Running backtest..."):
        result = run_backtest(
            base_weights, rules, prices_dict, atr_sznl_map, sznl_map,
            fragility_df,
            start_date=start_date, end_date=end_date,
            cash_apr=cash_apr, starting_equity=starting_equity,
        )

    if result is None:
        st.error("Backtest produced no result — check tickers and date range.")
        st.stop()

    n_days = len(result['cal'])
    st.success(f"✅ Backtest complete: {n_days:,} trading days.")

    # --- PROMINENT rule diagnostics (top-of-results so you can't miss them) ---
    st.subheader("🎯 Rule Activations")
    diag_rows = []
    any_fired = False
    for r_i, target, n_triggered, errs in result['rule_diagnostics']:
        r = rules[r_i] if r_i < len(rules) else {}
        cond_strs = [cond_label(c) for c in r.get('conditions', [])]
        cond_text = " AND ".join(cond_strs) if cond_strs else "(no conditions)"
        diag_rows.append({
            'Rule #': r_i + 1,
            'Target': target,
            'Scaler': f"{r.get('scaler', 1.0):.2f}×",
            'Conditions (AND)': cond_text,
            'Days Fired': n_triggered,
            '% of Period': f"{(n_triggered / n_days * 100):.1f}%" if n_days else "0%",
            'Issues': "; ".join(errs) if errs else "",
        })
        if n_triggered > 0:
            any_fired = True
    st.dataframe(pd.DataFrame(diag_rows), hide_index=True, width='stretch')
    if not any_fired and rules:
        st.error(
            "⚠️ NONE of your rules fired during the backtest period. Strategy "
            "and benchmark will be identical. Check the 'Issues' column for "
            "missing indicators, or loosen the thresholds."
        )

    # --- Stats comparison ---
    st.subheader("📊 Performance Summary")
    strat_stats = compute_stats(result['equity'], result['port_ret'], "Strategy")
    bench_stats = compute_stats(result['benchmark'], result['bench_ret'], "Buy & Hold")
    stats_df = pd.DataFrame([strat_stats, bench_stats]).set_index('Name')
    fmt = {
        'CAGR': '{:.2%}', 'Vol (Ann.)': '{:.2%}', 'Sharpe': '{:.2f}',
        'Sortino': '{:.2f}', 'Max DD': '{:.2%}', 'Calmar': '{:.2f}',
        'Total Return': '{:.2%}', 'Years': '{:.1f}',
    }
    st.dataframe(stats_df.style.format(fmt), width='stretch')

    delta = result['equity'].iloc[-1] - result['benchmark'].iloc[-1]
    delta_pct = result['equity'].iloc[-1] / result['benchmark'].iloc[-1] - 1
    st.markdown(
        f"**End-of-period delta vs buy-and-hold: ${delta:,.0f} "
        f"({delta_pct:+.2%})**"
    )

    # --- Equity curves ---
    st.subheader("📈 Equity & Drawdown")
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.7, 0.3],
        subplot_titles=("Equity", "Drawdown"),
        vertical_spacing=0.08, shared_xaxes=True,
    )
    fig.add_trace(go.Scatter(
        x=result['equity'].index, y=result['equity'].values,
        name="Strategy", line=dict(color='#00CC66', width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=result['benchmark'].index, y=result['benchmark'].values,
        name="Buy & Hold", line=dict(color='#888888', width=1.5, dash='dot'),
    ), row=1, col=1)
    strat_dd = (result['equity'] / result['equity'].cummax() - 1) * 100
    bench_dd = (result['benchmark'] / result['benchmark'].cummax() - 1) * 100
    fig.add_trace(go.Scatter(
        x=strat_dd.index, y=strat_dd.values,
        name="Strategy DD", line=dict(color='#CC0033', width=1.5),
        fill='tozeroy', fillcolor='rgba(204,0,51,0.15)',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=bench_dd.index, y=bench_dd.values,
        name="B&H DD", line=dict(color='#666666', width=1, dash='dot'),
    ), row=2, col=1)
    fig.update_yaxes(title_text="Equity ($)", tickformat="$,.0f", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_layout(height=600, hovermode='x unified', margin=dict(t=40, b=40))
    st.plotly_chart(fig, width='stretch')

    # --- Exposure timeline ---
    st.subheader("🎚 Exposure Over Time")
    eff_w = result['eff_weights'].copy()
    cash_w = result['cash_weight'].copy()
    expo_fig = go.Figure()
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    for i_, col in enumerate(eff_w.columns):
        expo_fig.add_trace(go.Scatter(
            x=eff_w.index, y=eff_w[col] * 100,
            name=col, stackgroup='one',
            line=dict(width=0), fillcolor=palette[i_ % len(palette)],
        ))
    expo_fig.add_trace(go.Scatter(
        x=cash_w.index, y=cash_w.clip(lower=0) * 100,
        name='Cash', stackgroup='one',
        line=dict(width=0), fillcolor='rgba(180,180,180,0.6)',
    ))
    expo_fig.update_layout(
        height=350, yaxis_title="Allocation (%)",
        hovermode='x unified', margin=dict(t=20, b=40),
    )
    st.plotly_chart(expo_fig, width='stretch')
    avg_invested = float((1 - cash_w.clip(lower=0)).mean()) * 100
    days_at_full = int((cash_w <= 0.001).sum())
    st.caption(
        f"Avg invested: **{avg_invested:.1f}%** of capital · "
        f"Days at full base allocation: **{days_at_full}** of {len(cash_w)}."
    )

    # --- Annual returns ---
    st.subheader("📅 Annual Returns")
    yrly_strat = result['equity'].resample('YE').last().pct_change()
    yrly_bench = result['benchmark'].resample('YE').last().pct_change()
    if not result['equity'].empty:
        first_yr = result['equity'].index[0].year
        first_eq_strat = result['equity'].iloc[0]
        first_eq_bench = result['benchmark'].iloc[0]
        eoy_strat = result['equity'].loc[result['equity'].index.year == first_yr].iloc[-1]
        eoy_bench = result['benchmark'].loc[result['benchmark'].index.year == first_yr].iloc[-1]
        yrly_strat.iloc[0] = eoy_strat / first_eq_strat - 1
        yrly_bench.iloc[0] = eoy_bench / first_eq_bench - 1
    yr_df = pd.DataFrame({
        'Strategy': yrly_strat.values,
        'Buy & Hold': yrly_bench.values,
        'Excess': (yrly_strat - yrly_bench).values,
    }, index=[d.year for d in yrly_strat.index])
    st.dataframe(
        yr_df.style.format('{:.2%}').map(
            lambda v: 'color: #00CC66' if (isinstance(v, (int, float)) and v > 0) else 'color: #CC3333',
            subset=['Strategy', 'Buy & Hold', 'Excess'],
        ),
        width='stretch',
    )
