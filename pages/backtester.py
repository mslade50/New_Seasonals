import streamlit as st
import os
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import datetime
import random
import re
import time
import uuid
import json

from indicators import calculate_indicators, apply_first_instance_filter, get_sznl_val_series
from tr_vcr import log_returns as _tr_log_returns, rv_daily as _tr_rv_daily, rv_sampled as _tr_rv_sampled, vcr as _tr_vcr

try:
    from strategy_config import STRATEGY_BOOK as _STRATEGY_BOOK
    from strategy_config import CSV_UNIVERSE as _CSV_UNIVERSE, LIQUID_PLUS_COMMODITIES as _LIQUID_PLUS
except Exception:
    _STRATEGY_BOOK = []
    _CSV_UNIVERSE = []
    _LIQUID_PLUS = []

try:
    from overflow_universe import load_overflow_universe, load_overflow_universe_full
except Exception:
    def load_overflow_universe(fallback=None, **_kw):
        return list(fallback) if fallback is not None else []

    def load_overflow_universe_full(static_fallback=None, **_kw):
        return sorted(set(static_fallback or []))

# The legacy static overflow tier (CSV_UNIVERSE minus the liquid set). Unioned
# with the dynamic screen by load_overflow_universe_full to form the single
# comprehensive overflow universe used in backtests.
_OVERFLOW_STATIC_TIER = sorted(set(_CSV_UNIVERSE) - set(_LIQUID_PLUS))

# Params keys that always come from the UI widgets, never overridden by a
# loaded preset. Everything else in the preset's `settings` dict overrides
# the corresponding param value at Run Backtest time.
_USER_ADJUSTABLE_PARAM_KEYS = frozenset({
    'universe_tickers', 'backtest_start_date',
    'entry_type', 'entry_conf_bps',
    'stop_atr', 'tgt_atr', 'holding_days',
    'use_stop_loss', 'use_take_profit', 'time_exit_only',
    'use_trailing_stop', 'trail_atr', 'trail_anchor',
    'use_eod_dd_exit', 'eod_dd_atr', 'eod_dd_weekdays',
    'use_partial_exits', 'partial_target_fraction',
    'use_intraday',
    'slippage_bps',
    'use_max_daily_risk', 'max_daily_risk_pct',
    'max_one_pos', 'allow_same_day_reentry',
    'max_daily_entries', 'max_total_positions',
})

MARKET_TICKER = "^GSPC"
VIX_TICKER = "^VIX"

SECTOR_ETFS = ["IBB", "IHI", "ITA", "ITB", "IYR", "KRE", "OIH", "SMH", "VNQ",
    "XBI", "XHB", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP",
    "XLU", "XLV", "XLY", "XME", "XOP", "XRT"]
SPX=['^GSPC','SPY']
INDEX_ETFS = ["SPY", "QQQ", "IWM", "DIA", "SMH"]
INDICES_SPOT = ["^GSPC", "^NDX"]
# Mirrors strategy_config.LIQUID_PLUS_COMMODITIES' commodity additions —
# the OIH/XOP overlap with SECTOR_ETFS is intentional (commodity-proxy
# sector ETFs); the union below dedupes.
COMMODITY_ETFS = ["CEF", "GLD", "OIH", "SLV", "UNG", "USO", "UVXY", "XOP"]
INTERNATIONAL_ETFS = ["EWZ", "EWC", "ECH", "ECOL", "EWW", "ARGT", "EWQ", "EWG", "EWI", "EWU", "EWP", "EWK", "EWO", "EWN", "EWD", "EWL",
    "EWJ", "EWH", "MCHI", "INDA", "EWY", "EWT", "EWA", "EWS", "EWM", "THD", "EIDO", "VNM", "EPHE", "EZA", "TUR", "EGPT"]

# 3x Leveraged ETFs (most-liquid pick per underlying; dedupe SPXL/UPRO, TNA/URTY, etc.)
LEV3X_EQUITY_BULL_BROAD  = ["SPXL", "TQQQ", "UDOW", "TNA", "MIDU"]
LEV3X_EQUITY_BEAR_BROAD  = ["SPXS", "SQQQ", "SDOW", "TZA"]
LEV3X_EQUITY_BULL_SECTOR = ["SOXL", "FAS", "TECL", "LABU", "CURE", "ERX", "DPST",
                            "DRN", "NAIL", "RETL", "WEBL", "DFEN", "YINN", "BRZU", "EDC", "MEXX"]
LEV3X_EQUITY_BEAR_SECTOR = ["SOXS", "FAZ", "TECS", "LABD", "ERY", "DRV", "WEBS", "YANG", "EDZ"]
LEV3X_BOND_BULL          = ["TMF"]
LEV3X_BOND_BEAR          = ["TMV"]
LEV3X_COMMODITY_BULL     = ["NUGT", "JNUG", "GUSH"]
LEV3X_COMMODITY_BEAR     = ["DUST", "JDST", "DRIP"]

LEV3X_EQUITY_BULL_ALL = LEV3X_EQUITY_BULL_BROAD + LEV3X_EQUITY_BULL_SECTOR
LEV3X_EQUITY_BEAR_ALL = LEV3X_EQUITY_BEAR_BROAD + LEV3X_EQUITY_BEAR_SECTOR
LEV3X_EQUITY_ALL      = LEV3X_EQUITY_BULL_ALL + LEV3X_EQUITY_BEAR_ALL
LEV3X_EQUITY_BROAD    = LEV3X_EQUITY_BULL_BROAD + LEV3X_EQUITY_BEAR_BROAD
LEV3X_ALL             = (LEV3X_EQUITY_ALL + LEV3X_BOND_BULL + LEV3X_BOND_BEAR
                         + LEV3X_COMMODITY_BULL + LEV3X_COMMODITY_BEAR)

# Macro mix: commodities + sector ETFs + index ETFs (incl. ^GSPC/^NDX) +
# every 3x leveraged ETF the backtester knows about. Sorted+deduped so the
# selectbox option produces the same ordering on every rerun.
COMMODITY_SECTOR_INDEX_LEV3X = sorted(set(
    COMMODITY_ETFS + SECTOR_ETFS + INDEX_ETFS + INDICES_SPOT + LEV3X_ALL
))
CSV_PATH = "seasonal_ranks.csv"
ATR_SZNL_PATH = "atr_seasonal_ranks.parquet"
ATR_SZNL_WINDOWS = [5, 10, 21, 63, 126, 252]
ATR_SZNL_COLS = [f"atr_sznl_{w}d" for w in ATR_SZNL_WINDOWS]

@st.cache_resource
def load_seasonal_map():
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        return {}
    if df.empty: return {}
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce').dt.normalize().dt.tz_localize(None)
    df = df.dropna(subset=["Date"])
    output_map = {}
    for ticker, group in df.groupby("ticker"):
        output_map[str(ticker).upper()] = pd.Series(group.seasonal_rank.values, index=group.Date)
    return output_map


@st.cache_resource
def load_atr_seasonal_map():
    """Load ATR-normalized seasonal ranks. Returns {ticker: DataFrame with 6 rank columns}.

    Pulls atr_seasonal_ranks.parquet from R2 when missing/stale so a fresh
    machine (R2 creds in .env) gets the expanded ranks. NOTE: while the file is
    still git-tracked it checks out 'present + fresh', so this pull only fires
    once the file is git-removed (`git rm --cached` + .gitignore) or deleted
    locally. Until then a fresh checkout sees the committed (pre-overflow) ranks.
    """
    try:
        from overflow_universe import _maybe_pull_from_r2
        _maybe_pull_from_r2(ATR_SZNL_PATH, "atr_seasonal_ranks.parquet")
    except Exception:
        pass
    try:
        df = pd.read_parquet(ATR_SZNL_PATH)
    except Exception:
        return {}
    if df.empty:
        return {}
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    output = {}
    for ticker, group in df.groupby('ticker'):
        output[str(ticker).upper()] = group.set_index('Date')[ATR_SZNL_COLS].sort_index()
    return output


@st.cache_resource
def load_fragility_dials():
    """Load daily fragility dial history (5d/21d/63d scores) from the Risk Dashboard.

    Returns a DataFrame indexed by date with columns '5d', '21d', '63d' (0-100).
    Returns None if the parquet is missing or unreadable. Available 2016-04-25+
    for most backtests; trades before that date will fail any dial-based filter
    (NaN > threshold evaluates False).
    """
    path = "data/rd2_fragility.parquet"
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index).normalize()
    try:
        df.index = df.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return df.sort_index()


def compute_signed_earnings_offsets(df_dates, earnings_dates, holidays_d64):
    """For each date in df_dates, return the signed trading-day offset to the
    nearest earnings announcement.

    Convention: offset = signal_date - earnings_date in trading days.
        positive → signal is AFTER earnings (e.g. +2 = 2 trading days after)
        negative → signal is BEFORE earnings (e.g. -2 = 2 trading days before)
        0        → signal IS the earnings day
        NaN      → ticker has no earnings dates

    Vectorized: O(N log M + N) per ticker. Uses np.busday_count with the
    same US-federal-holiday calendar as order_staging.py.
    """
    if earnings_dates is None or len(earnings_dates) == 0:
        return pd.Series(np.nan, index=df_dates)
    d64 = pd.DatetimeIndex(df_dates).to_numpy().astype('datetime64[D]')
    e_sorted = np.sort(pd.DatetimeIndex(earnings_dates).to_numpy().astype('datetime64[D]'))
    pos = np.searchsorted(e_sorted, d64, side='right')

    # Past earnings: e_sorted[pos-1] when pos>0
    past_mask = pos > 0
    past_e = e_sorted[np.clip(pos - 1, 0, len(e_sorted) - 1)]
    past_off = np.where(
        past_mask,
        np.busday_count(past_e, d64, holidays=holidays_d64),
        2**31 - 1,  # sentinel: huge so it loses min-abs comparison
    ).astype(np.int64)

    # Future earnings: e_sorted[pos] when pos<len
    future_mask = pos < len(e_sorted)
    future_e = e_sorted[np.clip(pos, 0, len(e_sorted) - 1)]
    future_off = np.where(
        future_mask,
        -np.busday_count(d64, future_e, holidays=holidays_d64),
        -(2**31 - 1),
    ).astype(np.int64)

    # Pick the offset with smaller |value|
    use_past = np.abs(past_off) <= np.abs(future_off)
    nearest = np.where(use_past, past_off, future_off).astype(float)

    # NaN where neither past nor future earnings exist
    no_data = ~past_mask & ~future_mask
    nearest[no_data] = np.nan

    return pd.Series(nearest, index=df_dates)


def _load_earnings_frame():
    """Production earnings_calendar.parquet UNION the isolated overflow staging
    file (earnings_calendar_overflow.parquet), each pulled from R2 if missing/
    stale. Returns one concatenated DataFrame (possibly empty). The staging file
    carries the new overflow names' earnings so the daily CSV_UNIVERSE rebuild
    of production can't wipe them."""
    prod = "data/earnings_calendar.parquet"
    staging = "data/earnings_calendar_overflow.parquet"
    try:
        from earnings_filter import _refresh_from_r2_if_needed
        _refresh_from_r2_if_needed(prod)
    except Exception:
        pass
    try:
        from cache_io import download_to_local
        if (not os.path.exists(staging)) or (time.time() - os.path.getmtime(staging) > 18 * 3600):
            download_to_local("earnings_calendar_overflow.parquet", staging)
    except Exception:
        pass
    frames = []
    for p in (prod, staging):
        try:
            frames.append(pd.read_parquet(p))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


@st.cache_resource
def load_earnings_map():
    """Load earnings calendar (production + overflow staging) as
    {ticker: pd.DatetimeIndex of earnings dates}. Empty dict if unavailable —
    earnings filter then silently no-ops.
    """
    df = _load_earnings_frame()
    if df.empty or 'ticker' not in df.columns or 'date' not in df.columns:
        return {}
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    df = df.dropna(subset=['date', 'ticker'])
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    out = {}
    for tkr, grp in df.groupby('ticker'):
        out[tkr] = pd.DatetimeIndex(sorted(grp['date'].unique()))
    return out


_EARNINGS_METRIC_COLS = ['eps_surprise_pct', 'rev_surprise_pct', 'eps_yoy', 'rev_yoy']


@st.cache_resource
def load_earnings_metrics_map():
    """Load per-ticker DataFrame of earnings-quality metrics, indexed by date.

    Returns {ticker: DataFrame[eps_surprise_pct, rev_surprise_pct, eps_yoy,
    rev_yoy]}. Index is the earnings announcement date (normalized, tz-naive).

    Used by the engine to look up the most recent reported metrics at or
    before each bar via reindex(method='ffill'). Source columns are derived
    by scripts/build_earnings_calendar.py — older parquets without the
    derived columns yield an empty map (filter silently no-ops).
    """
    df = _load_earnings_frame()
    needed = ['ticker', 'date'] + _EARNINGS_METRIC_COLS
    if df.empty or not all(c in df.columns for c in needed):
        return {}
    df = df[needed].copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    df = df.dropna(subset=['date', 'ticker'])
    df = df.sort_values(['ticker', 'date'])
    # Dedup (ticker, date) so the per-ticker date index is unique — otherwise the
    # engine's reindex(method='ffill') raises "cannot reindex on an axis with
    # duplicate labels". Production already carries dup dates; the staging union
    # adds more. keep='last' = the most recently reported value for that date.
    df = df.drop_duplicates(subset=['ticker', 'date'], keep='last')
    out = {}
    for tkr, grp in df.groupby('ticker'):
        out[tkr] = grp.drop(columns=['ticker']).set_index('date')[_EARNINGS_METRIC_COLS]
    return out


@st.cache_resource
def load_grades_map_cached():
    """Streamlit-cached wrapper around analyst_grades.load_grades_map()."""
    try:
        from analyst_grades import load_grades_map
        return load_grades_map()
    except Exception:
        return {}


def clean_ticker_df(df):
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(1) or 'Adj Close' in df.columns.get_level_values(1):
             df.columns = df.columns.get_level_values(1)
        else:
             df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    if 'Close' not in df.columns and 'Adj close' in df.columns:
        df.rename(columns={'Adj close': 'Close'}, inplace=True)
    if 'Close' not in df.columns:
        return pd.DataFrame() 
    df = df.dropna(subset=['Close'])
    return df

def build_xsec_rank_matrices(data_dict, windows=[5, 10, 21]):
    """Build cross-sectional percentile rank matrices from downloaded data.

    For each return window, first computes each ticker's temporal percentile
    rank (expanding, min 252 days — same as indicators.py), then ranks those
    percentiles across tickers on each date. This normalizes for volatility:
    a stock at its 95th %ile of its own history is treated the same whether
    it's a low-vol utility or a high-vol biotech.

    Returns dict {window: DataFrame} where each DataFrame has Date index and
    ticker columns with values 0-100.
    """
    RANK_MIN_PERIODS = 252
    rank_dict = {}
    for ticker, df in data_dict.items():
        if 'Close' not in df.columns or len(df) < 50:
            continue
        for w in windows:
            ret = df['Close'].pct_change(w)
            # Temporal percentile: where is this return vs ticker's own history
            temporal_pctile = ret.expanding(min_periods=RANK_MIN_PERIODS).rank(pct=True) * 100.0
            rank_dict.setdefault(w, {})[ticker] = temporal_pctile

    result = {}
    for w in windows:
        if not rank_dict.get(w):
            continue
        mat = pd.DataFrame(rank_dict[w])
        # Rank the temporal percentiles across tickers on each date
        result[w] = mat.rank(axis=1, pct=True) * 100.0
    return result

def build_mtm_curves(trades_df, data_dict, starting_equity, risk_bps, mode='flat'):
    """Build daily MTM equity curves with intraday H/L envelopes.

    Chronologically walks trades in EntryDate order, sizing each at either a
    flat fraction of starting_equity or a dynamic fraction of running equity.
    For each day, computes the aggregate close-based equity and the best/worst
    intraday equity bounds using daily H/L of every open position.

    Returns a DataFrame indexed by Date with columns:
        Equity_Close, Equity_High, Equity_Low, InMarket (bool)
    """
    if trades_df.empty:
        return pd.DataFrame(columns=['Equity_Close', 'Equity_High', 'Equity_Low', 'InMarket'])

    trades = trades_df.copy()
    trades['EntryDate'] = pd.to_datetime(trades['EntryDate'])
    trades['ExitDate']  = pd.to_datetime(trades['ExitDate'])
    trades = trades[trades['Ticker'].isin(data_dict.keys())].sort_values('EntryDate').reset_index(drop=True)
    if trades.empty:
        return pd.DataFrame(columns=['Equity_Close', 'Equity_High', 'Equity_Low', 'InMarket'])

    # Pre-slice OHLC for each trade's holding period; override exit-day close with
    # the realized exit price so stop/target fills are honored.
    trade_prices = {}
    for idx, tr in trades.iterrows():
        df = data_dict[tr['Ticker']]
        mask = (df.index >= tr['EntryDate']) & (df.index <= tr['ExitDate'])
        sub = df.loc[mask, ['Open', 'High', 'Low', 'Close']].copy() if all(c in df.columns for c in ['Open','High','Low','Close']) else None
        if sub is None or sub.empty:
            trade_prices[idx] = None
            continue
        exit_dt = tr['ExitDate']
        if exit_dt in sub.index:
            sub.loc[exit_dt, 'Close'] = tr['Exit']
            if tr['Exit'] > sub.loc[exit_dt, 'High']: sub.loc[exit_dt, 'High'] = tr['Exit']
            if tr['Exit'] < sub.loc[exit_dt, 'Low']:  sub.loc[exit_dt, 'Low']  = tr['Exit']
        trade_prices[idx] = sub

    # Build calendar: ALL trading days in [min_entry, max_exit] from the full data_dict,
    # not just days spanned by positions. Without this, gap days (no active positions)
    # are absent from the index entirely — which silently forces Time-in-Market to 100%
    # and reduces the denominator for Sharpe/Parkinson vol.
    min_dt = trades['EntryDate'].min()
    max_dt = trades['ExitDate'].max()
    cal_set = set()
    for _df in data_dict.values():
        if _df is None or _df.empty:
            continue
        cal_set.update(_df.index[(_df.index >= min_dt) & (_df.index <= max_dt)])
    if not cal_set:
        return pd.DataFrame(columns=['Equity_Close', 'Equity_High', 'Equity_Low', 'InMarket'])
    all_dates = sorted(cal_set)

    entries_by_date = {}
    for idx, tr in trades.iterrows():
        entries_by_date.setdefault(tr['EntryDate'], []).append(idx)

    # Per-trade EffectiveBps overrides the global risk_bps when present.
    # Lets variable-sizing universes (liquid base vs overflow extras) flow
    # into MTM curves with the right per-trade $ risk.
    _has_per_trade_bps = 'EffectiveBps' in trades.columns
    flat_risk_dollars = starting_equity * risk_bps / 10000.0
    equity = float(starting_equity)

    eq_close, eq_high, eq_low, in_mkt = [], [], [], []
    active = {}  # idx -> {'shares','entry_price','direction','prev_close_val'}

    for date in all_dates:
        # 1. Open today's new trades, sized against equity at prior close
        for idx in entries_by_date.get(date, []):
            sub = trade_prices.get(idx)
            if sub is None or sub.empty:
                continue
            tr = trades.iloc[idx]
            tech_risk = tr['TechRisk']
            if not (tech_risk and tech_risk > 0):
                continue
            risk_scale = float(tr['RiskScale']) if 'RiskScale' in trades.columns and pd.notna(tr.get('RiskScale')) else 1.0
            if _has_per_trade_bps and pd.notna(tr.get('EffectiveBps')):
                _bps = float(tr['EffectiveBps'])
                _flat_d = starting_equity * _bps / 10000.0
                _dyn_d  = equity * _bps / 10000.0
                risk_dollars = (_flat_d if mode == 'flat' else _dyn_d) * risk_scale
            else:
                risk_dollars = (flat_risk_dollars if mode == 'flat' else equity * risk_bps / 10000.0) * risk_scale
            shares = risk_dollars / tech_risk
            if shares <= 0:
                continue
            active[idx] = {
                'shares': shares,
                'entry_price': tr['Entry'],
                'direction': tr['Direction'],
                'prev_close_val': 0.0,
            }

        # 2. Walk each active position and accumulate daily P&L deltas
        chg_close, chg_best, chg_worst = 0.0, 0.0, 0.0
        closing_today = []
        for idx, pos in active.items():
            sub = trade_prices[idx]
            if date not in sub.index:
                continue
            c = sub.at[date, 'Close']
            h = sub.at[date, 'High']
            l = sub.at[date, 'Low']
            if pos['direction'] == 'Long':
                val_c = (c - pos['entry_price']) * pos['shares']
                val_h = (h - pos['entry_price']) * pos['shares']
                val_l = (l - pos['entry_price']) * pos['shares']
            else:  # Short
                val_c = (pos['entry_price'] - c) * pos['shares']
                val_h = (pos['entry_price'] - l) * pos['shares']  # best case short = low print
                val_l = (pos['entry_price'] - h) * pos['shares']  # worst case short = high print
            prev = pos['prev_close_val']
            chg_close += (val_c - prev)
            chg_best  += (val_h - prev)
            chg_worst += (val_l - prev)
            pos['prev_close_val'] = val_c
            if date >= trades.iloc[idx]['ExitDate']:
                closing_today.append(idx)

        start_of_day = equity
        equity += chg_close
        eq_close.append(equity)
        eq_high.append(start_of_day + chg_best)
        eq_low.append(start_of_day + chg_worst)
        in_mkt.append(len(active) > 0)

        for idx in closing_today:
            del active[idx]

    return pd.DataFrame(
        {'Equity_Close': eq_close, 'Equity_High': eq_high, 'Equity_Low': eq_low, 'InMarket': in_mkt},
        index=pd.DatetimeIndex(all_dates, name='Date'),
    )


def compute_portfolio_stats(equity_df, starting_equity, risk_bps=None, trades_df=None):
    """Portfolio-level stats from an MTM equity curve with H/L envelope.

    Uses close-to-close daily returns for Sharpe/Sortino, Parkinson estimator
    on daily H/L for annualized vol (captures intraday swings), and
    peak-to-trough Close for max drawdown. Also reports MaxDD @ Lows which
    uses the low envelope (worst intraday mark) vs close-based running peak,
    with R-unit conversion and time-to-recover metrics.
    """
    empty = {'CAGR_Pct': 0, 'TotalReturn_Pct': 0, 'MaxDD_Pct': 0, 'MaxDD_Low_Pct': 0,
             'MaxDD_R': 0, 'MaxDD_Low_R': 0, 'Sharpe': 0, 'Sharpe_Active': 0,
             'Sortino': 0, 'Sortino_Active': 0, 'Calmar': 0, 'ParkinsonVol_Pct': 0,
             'TimeInMarket_Pct': 0, 'FinalEquity': starting_equity,
             'UnderwaterDays': 0, 'TradesDuringDD': None, 'DDStillOngoing': False,
             'PeakDate': None, 'TroughDate': None, 'RecoveryDate': None}
    if equity_df.empty or len(equity_df) < 2:
        return empty

    close = equity_df['Equity_Close']
    high = equity_df['Equity_High']
    low = equity_df['Equity_Low']

    years = max((equity_df.index[-1] - equity_df.index[0]).days / 365.25, 1e-9)
    total_ret = close.iloc[-1] / starting_equity
    cagr = (total_ret ** (1 / years) - 1) * 100 if total_ret > 0 else -100

    peak = close.cummax()
    dd_series = (close - peak) / peak
    max_dd = dd_series.min() * 100

    # Max DD @ Lows: intraday low vs close-based running peak. More conservative
    # (deeper) than C-C drawdown — reflects the worst mark you actually saw.
    dd_low_series = (low - peak) / peak
    max_dd_low = dd_low_series.min() * 100

    # R-unit conversion: DD_pct / risk_bps_as_pct = # of R. Equivalent to
    # dd_dollars (at starting-equity basis) / risk-per-trade.
    if risk_bps and risk_bps > 0:
        max_dd_R = abs(max_dd) * 100.0 / risk_bps
        max_dd_low_R = abs(max_dd_low) * 100.0 / risk_bps
    else:
        max_dd_R = 0.0
        max_dd_low_R = 0.0

    # Underwater duration + time-to-recover for the C-C max drawdown
    peak_date = trough_date = recovery_date = None
    underwater_days = 0
    trades_during_dd = None
    still_underwater = False
    if dd_series.min() < 0:
        trough_date = dd_series.idxmin()
        # Peak date = last date at-or-before trough where close equals its running peak
        _peak_mask = (equity_df.index <= trough_date) & (close == peak)
        peak_date = equity_df.index[_peak_mask][-1] if _peak_mask.any() else equity_df.index[0]
        peak_value = close.loc[peak_date]
        # Recovery = first date after trough where close reaches back to peak value
        _post = close.loc[trough_date:]
        _recov_mask = _post >= peak_value
        if _recov_mask.any():
            recovery_date = _post[_recov_mask].index[0]
            underwater_days = (recovery_date - peak_date).days
        else:
            # Still underwater at the end of the backtest
            still_underwater = True
            recovery_date = None
            underwater_days = (equity_df.index[-1] - peak_date).days

        if trades_df is not None and not trades_df.empty and 'EntryDate' in trades_df.columns:
            _tc = trades_df.copy()
            _ed = pd.to_datetime(_tc['EntryDate'], errors='coerce')
            end_cut = recovery_date if recovery_date is not None else equity_df.index[-1]
            trades_during_dd = int(((_ed >= peak_date) & (_ed <= end_cut)).sum())

    daily_ret = close.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    downside = daily_ret[daily_ret < 0]
    sortino = (daily_ret.mean() / downside.std() * np.sqrt(252)) if len(downside) > 1 and downside.std() > 0 else 0

    # Active-period Sharpe/Sortino: compute on in-market days only. More relevant
    # for ensemble-book sizing — answers "what's the quality of returns *when deployed*",
    # ignoring the cost of idle days (which other strategies in the book will fill).
    # Relationship: sharpe_calendar ≈ sharpe_active * sqrt(TIM).
    if 'InMarket' in equity_df.columns:
        # Align InMarket to daily_ret (both drop first NaN row of pct_change)
        active_mask = equity_df['InMarket'].iloc[1:].reset_index(drop=True)
        active_ret = daily_ret.reset_index(drop=True)[active_mask.values]
        if len(active_ret) > 1 and active_ret.std() > 0:
            sharpe_active = active_ret.mean() / active_ret.std() * np.sqrt(252)
            ds_active = active_ret[active_ret < 0]
            sortino_active = (active_ret.mean() / ds_active.std() * np.sqrt(252)) if len(ds_active) > 1 and ds_active.std() > 0 else 0
        else:
            sharpe_active, sortino_active = 0, 0
    else:
        sharpe_active, sortino_active = sharpe, sortino

    # Parkinson vol from aggregate daily H/L envelope of the equity curve itself
    valid = (high > 0) & (low > 0) & (high >= low)
    if valid.any():
        log_hl = np.log(high[valid] / low[valid])
        park_daily = np.sqrt((1.0 / (4 * np.log(2))) * (log_hl ** 2).mean())
        park_annual = park_daily * np.sqrt(252) * 100
    else:
        park_annual = 0

    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0
    tim_pct = equity_df['InMarket'].sum() / len(equity_df) * 100 if 'InMarket' in equity_df.columns else 0

    return {
        'CAGR_Pct': cagr,
        'TotalReturn_Pct': (total_ret - 1) * 100,
        'MaxDD_Pct': max_dd,
        'MaxDD_Low_Pct': max_dd_low,
        'MaxDD_R': max_dd_R,
        'MaxDD_Low_R': max_dd_low_R,
        'Sharpe': sharpe,
        'Sharpe_Active': sharpe_active,
        'Sortino': sortino,
        'Sortino_Active': sortino_active,
        'Calmar': calmar,
        'ParkinsonVol_Pct': park_annual,
        'TimeInMarket_Pct': tim_pct,
        'FinalEquity': close.iloc[-1],
        'UnderwaterDays': underwater_days,
        'TradesDuringDD': trades_during_dd,
        'DDStillOngoing': still_underwater,
        'PeakDate': peak_date,
        'TroughDate': trough_date,
        'RecoveryDate': recovery_date,
    }


def compute_trade_path_stats(trades_df, data_dict):
    """Compute per-trade MAE/MFE/Give-back/Capture using daily H/L.

    Entry-day handling is pessimistic: only adverse excursion counts
    (Long -> day's Low; Short -> day's High). Favorable side is zeroed
    because we don't know where in the bar the fill happened. For MOC
    entries (entry_price == entry-day close) the entry day is skipped
    entirely — the bar was complete before we got in. Middle days use
    full H/L both sides. Exit day uses full H/L (peak-after-exit is a
    minor overstatement flagged in the UI).

    Returns the trades_df with four new columns: MAE_R, MFE_R,
    GiveBack_R, Capture_Pct.
    """
    if trades_df.empty:
        return trades_df

    trades = trades_df.copy()
    # Dates may have been stringified already by upstream; normalize
    e_dt = pd.to_datetime(trades['EntryDate'])
    x_dt = pd.to_datetime(trades['ExitDate'])

    mae_R = np.full(len(trades), np.nan)
    mfe_R = np.full(len(trades), np.nan)

    for i, (_, tr) in enumerate(trades.iterrows()):
        ticker = tr['Ticker']
        if ticker not in data_dict:
            continue
        df = data_dict[ticker]
        if not all(c in df.columns for c in ['High', 'Low', 'Close']):
            continue
        entry_dt = e_dt.iloc[i]
        exit_dt  = x_dt.iloc[i]
        entry_px = float(tr['Entry'])
        tech_risk = float(tr['TechRisk']) if tr['TechRisk'] else 0.0
        if tech_risk <= 0:
            continue
        direction = tr['Direction']

        mask = (df.index >= entry_dt) & (df.index <= exit_dt)
        hold = df.loc[mask, ['High', 'Low', 'Close']]
        if hold.empty:
            continue

        worst_excursion = 0.0  # most adverse (negative)
        best_excursion  = 0.0  # most favorable (positive)
        for j, (dt, row) in enumerate(hold.iterrows()):
            h, l, c = float(row['High']), float(row['Low']), float(row['Close'])
            is_entry_day = (dt == entry_dt)
            is_moc_entry = is_entry_day and abs(entry_px - c) < 1e-9

            if is_moc_entry:
                continue  # bar completed before entry

            if direction == 'Long':
                adverse = l - entry_px   # negative when worse
                favor   = h - entry_px   # positive when better
            else:  # Short
                adverse = entry_px - h   # negative when worse (stock rallied)
                favor   = entry_px - l   # positive when better

            if is_entry_day:
                # Pessimistic: count adverse, zero favorable
                worst_excursion = min(worst_excursion, adverse)
            else:
                worst_excursion = min(worst_excursion, adverse)
                best_excursion  = max(best_excursion, favor)

        mae_R[i] = worst_excursion / tech_risk
        mfe_R[i] = best_excursion / tech_risk

    trades['MAE_R'] = mae_R
    trades['MFE_R'] = mfe_R
    trades['GiveBack_R'] = trades['MFE_R'] - trades['R']
    trades['Capture_Pct'] = np.where(
        trades['MFE_R'] > 0,
        trades['R'] / trades['MFE_R'] * 100,
        np.nan,
    )
    return trades


@st.cache_data(show_spinner=True)
def download_universe_data(tickers, fetch_start_date):
    if not tickers: return {} 
    clean_tickers = [str(t).strip().upper() for t in tickers if str(t).strip() != '']
    if not clean_tickers: return {}
    if isinstance(fetch_start_date, datetime.date):
        start_str = fetch_start_date.strftime('%Y-%m-%d')
    else:
        start_str = fetch_start_date 
    data_dict = {}
    CHUNK_SIZE = 50 
    total_tickers = len(clean_tickers)
    download_bar = st.progress(0)
    status_text = st.empty()
    for i in range(0, total_tickers, CHUNK_SIZE):
        chunk = clean_tickers[i:i + CHUNK_SIZE]
        status_text.text(f"Downloading batch {i} to {min(i+CHUNK_SIZE, total_tickers)} of {total_tickers}...")
        download_bar.progress(min((i + CHUNK_SIZE) / total_tickers, 1.0))
        try:
            df = yf.download(chunk, start=start_str, group_by='ticker', auto_adjust=True, progress=False, threads=True)
            if df.empty: continue
            if len(chunk) == 1:
                t = chunk[0]
                t_df = clean_ticker_df(df)
                if not t_df.empty:
                    t_df.index = t_df.index.normalize().tz_localize(None)
                    data_dict[t] = t_df
            else:
                for t in chunk:
                    try:
                        if t in df.columns.levels[0]:
                            t_df = df[t].copy()
                            t_df = clean_ticker_df(t_df)
                            if not t_df.empty:
                                t_df.index = t_df.index.normalize().tz_localize(None)
                                data_dict[t] = t_df
                    except: continue
            time.sleep(0.1)
        except Exception as e:
            if "rate limited" in str(e).lower(): break
            continue
    download_bar.empty()
    status_text.empty()
    return data_dict

def get_cycle_year(year):
    rem = year % 4
    if rem == 0: return "4. Election Year"
    if rem == 1: return "1. Post-Election"
    if rem == 2: return "2. Midterm Year"
    if rem == 3: return "3. Pre-Election"
    return "Unknown"

def get_age_bucket(years):
    if years < 3: return "< 3 Years"
    if years < 5: return "3-5 Years"
    if years < 10: return "5-10 Years"
    if years < 20: return "10-20 Years"
    return "> 20 Years"

def grade_strategy(pf, sqn, win_rate, total_trades):
    score = 0
    reasons = []
    if pf >= 2.0: score += 4
    elif pf >= 1.5: score += 3
    elif pf >= 1.2: score += 2
    elif pf >= 1.0: score += 1
    else: score -= 5 
    if sqn >= 3.0: score += 4
    elif sqn >= 2.0: score += 3
    elif sqn >= 1.5: score += 2
    elif sqn > 0: score += 1
    if total_trades < 30: score -= 2
    if score >= 7: return "A", "Excellent", reasons
    if score >= 5: return "B", "Good", reasons
    if score >= 3: return "C", "Marginal", reasons
    if score >= 0: return "D", "Poor", reasons
    return "F", "Uninvestable", reasons

def _infer_strategy_type(params):
    direction = params.get('trade_direction', 'Long')
    perf_filters = params.get('perf_filters', [])
    has_oversold = any(f['logic'] == '<' and f['thresh'] <= 33 for f in perf_filters)
    has_overbought = any(f['logic'] == '>' and f['thresh'] >= 67 for f in perf_filters)
    has_52w = params.get('use_52w', False)
    has_breakout = params.get('breakout_mode', 'None') != 'None'
    has_sznl = params.get('use_sznl', False) or params.get('use_market_sznl', False)
    
    if has_52w and params.get('52w_type') == 'New 52w High':
        strat_type = "Breakout"
        thesis = "Momentum continuation after new highs"
    elif has_breakout and "High" in params.get('breakout_mode', ''):
        strat_type = "Breakout"
        thesis = "Breakout above previous range"
    elif has_oversold and direction == 'Long':
        strat_type = "MeanReversion"
        thesis = "Oversold bounce setup"
    elif has_overbought and direction == 'Short':
        strat_type = "MeanReversion"
        thesis = "Overbought fade setup"
    elif has_sznl and not (has_oversold or has_overbought):
        strat_type = "Seasonal"
        thesis = "Seasonal tendency play"
    elif has_overbought and direction == 'Long':
        strat_type = "Momentum"
        thesis = "Momentum continuation in strong names"
    else:
        strat_type = "Custom"
        thesis = "[EDIT: Add thesis]"
    
    if has_sznl and strat_type != "Seasonal":
        sznl_logic = params.get('sznl_logic', '>')
        sznl_thresh = params.get('sznl_thresh', 50)
        if sznl_logic == '>' and sznl_thresh >= 65:
            thesis += " with strong seasonal tailwind"
        elif sznl_logic == '<' and sznl_thresh <= 33:
            thesis += " despite weak seasonality (contrarian)"
    
    trend = params.get('trend_filter', 'None')
    if "200 SMA" in trend and ">" in trend:
        thesis += " in uptrending names"
    elif "200 SMA" in trend and "<" in trend:
        thesis += " in downtrending names"
    
    return strat_type, thesis

def _generate_key_filters(params):
    filters = []
    direction = params.get('trade_direction', 'Long')
    
    for pf in params.get('perf_filters', []):
        consec_str = f" ({pf['consecutive']}d consecutive)" if pf.get('consecutive', 1) > 1 else ""
        if pf['logic'] == 'Between':
            filters.append(f"{pf['window']}D rank between {pf['thresh']:.0f}-{pf.get('thresh_max', 100):.0f}th %ile{consec_str}")
        elif pf['logic'] == 'Not Between':
            filters.append(f"{pf['window']}D rank NOT between {pf['thresh']:.0f}-{pf.get('thresh_max', 100):.0f}th %ile{consec_str}")
        else:
            filters.append(f"{pf['window']}D rank {pf['logic']} {pf['thresh']:.0f}th %ile{consec_str}")

    for paf in params.get('perf_atr_filters', []):
        consec_str = f" ({paf['consecutive']}d consecutive)" if paf.get('consecutive', 1) > 1 else ""
        if paf['logic'] == 'Between':
            filters.append(f"{paf['window']}D ATR rank between {paf['thresh']:.0f}-{paf.get('thresh_max', 100):.0f}th %ile{consec_str}")
        elif paf['logic'] == 'Not Between':
            filters.append(f"{paf['window']}D ATR rank NOT between {paf['thresh']:.0f}-{paf.get('thresh_max', 100):.0f}th %ile{consec_str}")
        else:
            filters.append(f"{paf['window']}D ATR rank {paf['logic']} {paf['thresh']:.0f}th %ile{consec_str}")

    for asf in params.get('atr_sznl_filters', []):
        consec_str = f" ({asf['consecutive']}d consecutive)" if asf.get('consecutive', 1) > 1 else ""
        if asf['logic'] == 'Between':
            filters.append(f"{asf['window']}D ATR seasonal rank between {asf['thresh']:.0f}-{asf.get('thresh_max', 100):.0f}th %ile{consec_str}")
        elif asf['logic'] == 'Not Between':
            filters.append(f"{asf['window']}D ATR seasonal rank NOT between {asf['thresh']:.0f}-{asf.get('thresh_max', 100):.0f}th %ile{consec_str}")
        else:
            filters.append(f"{asf['window']}D ATR seasonal rank {asf['logic']} {asf['thresh']:.0f}th %ile{consec_str}")
    
    for maf in params.get('ma_consec_filters', []):
        filters.append(f"Close {maf['logic'].lower()} {maf['length']} SMA ({maf['consec']}d consecutive)")
    
    if params.get('use_sznl'):
        filters.append(f"Ticker seasonal rank {params['sznl_logic']} {params['sznl_thresh']:.0f}")
    if params.get('use_market_sznl'):
        filters.append(f"Market seasonal {params['market_sznl_logic']} {params['market_sznl_thresh']:.0f}")
    
    if params.get('use_52w'):
        first_str = " (first in {0}d)".format(params.get('52w_lookback', 21)) if params.get('52w_first_instance') else ""
        _w52 = int(params.get('52w_window', 252))
        _hl = "High" if params.get('52w_type', 'New High') in ('New High', 'New 52w High') else "Low"
        filters.append(f"New {_w52}d {_hl}{first_str}")
    if params.get('exclude_52w_high'):
        filters.append("NOT at 52-week high")
    if params.get('use_ath'):
        filters.append(params.get('ath_type', 'Today is ATH'))
    if params.get('use_recent_ath'):
        prefix = "Has NOT made" if params.get('recent_ath_invert') else "Made"
        filters.append(f"{prefix} ATH in last {params['ath_lookback_days']} days")
    if params.get('use_recent_52w'):
        prefix = "Has NOT made" if params.get('recent_52w_invert') else "Made"
        filters.append(f"{prefix} 52w high in last {params['recent_52w_lookback']} days")
    if params.get('use_recent_52w_low'):
        prefix = "Has NOT made" if params.get('recent_52w_low_invert') else "Made"
        filters.append(f"{prefix} 52w low in last {params['recent_52w_low_lookback']} days")
    if params.get('breakout_mode', 'None') != 'None':
        filters.append(params['breakout_mode'])
    
    if params.get('require_close_gt_open'):
        filters.append("Close > Open (green candle)")
    
    if params.get('use_range_filter'):
        filters.append(f"Close in {params['range_min']}-{params['range_max']}% of daily range")
    
    if params.get('use_atr_ret_filter'):
        filters.append(f"Net change between {params['atr_ret_min']:.1f} and {params['atr_ret_max']:.1f} ATR")
    
    if params.get('use_range_atr_filter'):
        logic = params.get('range_atr_logic', 'Between')
        if logic == '>':
            filters.append(f"Today's range > {params['range_atr_min']:.1f} ATR")
        elif logic == '<':
            filters.append(f"Today's range < {params['range_atr_max']:.1f} ATR")
        else:
            filters.append(f"Today's range between {params['range_atr_min']:.1f}-{params['range_atr_max']:.1f} ATR")

    if params.get('use_open_gap_atr_filter'):
        logic = params.get('open_gap_atr_logic', 'Between')
        if logic == '>':
            filters.append(f"Open vs prev close > {params['open_gap_atr_min']:.1f} ATR")
        elif logic == '<':
            filters.append(f"Open vs prev close < {params['open_gap_atr_max']:.1f} ATR")
        else:
            filters.append(f"Open vs prev close between {params['open_gap_atr_min']:.1f}-{params['open_gap_atr_max']:.1f} ATR")
    day_label = {0: "Signal day", 1: "T-1", 2: "T-2", 3: "T-3", 4: "T-4", 5: "T-5"}
    for pa in params.get('price_action_filters', []):
        dl = day_label.get(pa.get('lag', 0), f"T-{pa.get('lag', 0)}")
        pa_type = pa['type']
        if pa_type == 'range_pct':
            filters.append(f"{dl}: Close in {pa['min']:.0f}-{pa['max']:.0f}% of range")
        elif pa_type == 'atr_ret':
            filters.append(f"{dl}: Net change {pa['min']:.1f} to {pa['max']:.1f} ATR")
        elif pa_type == 'range_atr':
            logic = pa.get('logic', 'Between')
            if logic == '>':
                filters.append(f"{dl}: Range > {pa['min']:.1f} ATR")
            elif logic == '<':
                filters.append(f"{dl}: Range < {pa['max']:.1f} ATR")
            else:
                filters.append(f"{dl}: Range {pa['min']:.1f}-{pa['max']:.1f} ATR")
        elif pa_type == 'close_gt_open':
            filters.append(f"{dl}: Close > Open (green)")
        elif pa_type == 'close_lt_open':
            filters.append(f"{dl}: Close < Open (red)")
        elif pa_type == 'close_gt_prev_high':
            filters.append(f"{dl}: Close > Prev High")
        elif pa_type == 'close_lt_prev_low':
            filters.append(f"{dl}: Close < Prev Low")
        elif pa_type == 'atr_move':
            n = int(pa.get('n_bars', 1))
            filters.append(f"{dl}: {n}-bar move {pa['min']:.1f} to {pa['max']:.1f} ATR")
        elif pa_type in ('trailing_low', 'trailing_high'):
            window = int(pa.get('window', 21))
            end_lag = int(pa.get('end_lag', pa.get('lag', 0)))
            side = 'low' if pa_type == 'trailing_low' else 'high'
            if end_lag > pa.get('lag', 0):
                el = day_label.get(end_lag, f"T-{end_lag}")
                filters.append(f"{dl}..{el}: prints {window}-bar {side}")
            else:
                filters.append(f"{dl}: prints {window}-bar {side}")
    if params.get('use_ma_dist_filter'):
        ma_type = params.get('dist_ma_type', 'SMA 200')
        logic = params.get('dist_logic', 'Between')
        if logic == "Greater Than (>)":
            filters.append(f"Distance from {ma_type} > {params['dist_min']:.1f} ATR")
        elif logic == "Less Than (<)":
            filters.append(f"Distance from {ma_type} < {params['dist_max']:.1f} ATR")
        else:
            filters.append(f"Distance from {ma_type} between {params['dist_min']:.1f}-{params['dist_max']:.1f} ATR")
    
    if params.get('use_vol'):
        if params.get('vol_logic', '>') == 'Between':
            filters.append(f"Volume between {params['vol_thresh']:.1f}x and {params.get('vol_thresh_max', 10.0):.1f}x 63-day avg")
        else:
            filters.append(f"Volume {params.get('vol_logic', '>')} {params['vol_thresh']:.1f}x 63-day avg")
    if params.get('use_vol_rank'):
        filters.append(f"10D vol rank {params['vol_rank_logic']} {params['vol_rank_thresh']:.0f}th %ile")
    if params.get('vol_gt_prev'):
        filters.append("Volume > previous day")
    
    if params.get('use_acc_count_filter'):
        filters.append(f"Acc days {params['acc_count_logic']} {params['acc_count_thresh']} in last {params['acc_count_window']}d")
    if params.get('use_dist_count_filter'):
        filters.append(f"Dist days {params['dist_count_logic']} {params['dist_count_thresh']} in last {params['dist_count_window']}d")
    
    if params.get('use_gap_filter'):
        filters.append(f"Gap count {params['gap_logic']} {params['gap_thresh']} in last {params['gap_lookback']}d")
    
    trend = params.get('trend_filter', 'None')
    if trend != 'None':
        filters.append(f"Trend: {trend}")
    
    if params.get('use_vix_filter'):
        filters.append(f"VIX between {params['vix_min']:.0f}-{params['vix_max']:.0f}")
    
    if params.get('use_ref_ticker_filter') and params.get('ref_filters'):
        ref_ticker = params.get('ref_ticker', 'IWM')
        for rf in params['ref_filters']:
            filters.append(f"{ref_ticker} {rf['window']}D rank {rf['logic']} {rf['thresh']:.0f}th %ile")

    if params.get('use_xsec_filter') and params.get('xsec_filters'):
        for xf in params['xsec_filters']:
            consec_str = f" ({xf['consecutive']}d consecutive)" if xf.get('consecutive', 1) > 1 else ""
            if xf['logic'] == 'Between':
                filters.append(f"XSec {xf['window']}D rank between {xf['thresh']:.0f}-{xf.get('thresh_max', 100):.0f}th %ile{consec_str}")
            elif xf['logic'] == 'Not Between':
                filters.append(f"XSec {xf['window']}D rank NOT between {xf['thresh']:.0f}-{xf.get('thresh_max', 100):.0f}th %ile{consec_str}")
            else:
                filters.append(f"XSec {xf['window']}D rank {xf['logic']} {xf['thresh']:.0f}th %ile{consec_str}")

    if params.get('use_weekly_ma_pullback'):
        filters.append(f"First touch of Weekly {params['wma_type']}{params['wma_period']} after {params['wma_min_ext_pct']:.0f}%+ extension (lookback {params['wma_lookback_months']}mo, {params['wma_touch_logic']})")

    if params.get('use_volret_delta'):
        filters.append(f"Vol/Ret delta ({params.get('vrd_method', 'Z-score diff')}, halflife {params.get('vrd_vol_halflife', 20)}d, ret {params.get('vrd_ret_horizon', 20)}d, ΔN {params.get('vrd_delta_n', 5)}d) rank in [{params.get('vrd_pctile_min', 70.0):.0f},{params.get('vrd_pctile_max', 90.0):.0f}] ({params.get('vrd_rank_window', 'Expanding')})")
    if params.get('use_tr_vcr_filter'):
        _tv_metric = params.get('tr_vcr_metric', 'Trend Ratio (TR)')
        _tv_win = params.get('tr_vcr_window', 20)
        _tv_sf = params.get('tr_vcr_sample_freq', 5)
        _tv_rw = params.get('tr_vcr_rank_window', 'Expanding')
        _tv_mode = params.get('tr_vcr_filter_mode', 'Percentile rank')
        _tv_consec = int(params.get('tr_vcr_min_consec', 1))
        _tv_first = params.get('tr_vcr_consec_first', False)
        _consec_str = ""
        if _tv_consec > 1:
            _consec_str = f", >={_tv_consec}d streak{' (first hit)' if _tv_first else ''}"
        if _tv_metric == 'Regime quadrant':
            _quads = ", ".join(params.get('tr_vcr_regime_quadrants', []) or []) or "(none)"
            filters.append(f"TR/VCR regime: {{{_quads}}} (win {_tv_win}d, sf {_tv_sf}d, {_tv_rw}{_consec_str})")
        elif _tv_mode == 'Raw value':
            _logic = params.get('tr_vcr_raw_logic', 'Between')
            _rmin = params.get('tr_vcr_raw_min', 1.0)
            _rmax = params.get('tr_vcr_raw_max', 5.0)
            if _logic == 'Between':
                filters.append(f"{_tv_metric} raw in [{_rmin:.2f},{_rmax:.2f}] (win {_tv_win}d, sf {_tv_sf}d{_consec_str})")
            else:
                filters.append(f"{_tv_metric} raw {_logic} {_rmin:.2f} (win {_tv_win}d, sf {_tv_sf}d{_consec_str})")
        else:
            filters.append(f"{_tv_metric} rank in [{params.get('tr_vcr_pctile_min', 70.0):.0f},{params.get('tr_vcr_pctile_max', 100.0):.0f}] (win {_tv_win}d, sf {_tv_sf}d, {_tv_rw}{_consec_str})")

    if params.get('use_dow_filter'):
        day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
        days_str = ", ".join([day_names.get(d, '?') for d in params.get('allowed_days', [])])
        filters.append(f"Entry days: {days_str}")

    if 'allowed_cycles' in params and len(params.get('allowed_cycles', [])) < 4:
        cycle_names = {0: 'Election', 1: 'Post-Election', 2: 'Midterm', 3: 'Pre-Election'}
        cycles_str = ", ".join([cycle_names.get(c, '?') for c in params['allowed_cycles']])
        filters.append(f"Cycle years: {cycles_str}")

    if params.get('excluded_years'):
        yrs = sorted(params['excluded_years'])
        filters.append(f"Excluded years: {', '.join(str(y) for y in yrs)}")
    
    if params.get('use_t1_open_filter'):
        for f in params.get('t1_open_filters', []):
            offset_str = f" {'+' if f['atr_offset'] >= 0 else ''}{f['atr_offset']} ATR" if f['atr_offset'] != 0 else ""
            filters.append(f"T+1 Open {f['logic']} {f['reference']}{offset_str}")
    
    return filters if filters else ["[EDIT: Add key filters]"]

def _generate_exit_summary(params):
    direction = params.get('trade_direction', 'Long')
    stop_atr = params.get('stop_atr', 2.0)
    tgt_atr = params.get('tgt_atr', 5.0)
    hold_days = params.get('holding_days', 10)
    use_stop = params.get('use_stop_loss', True)
    use_tgt = params.get('use_take_profit', True)
    use_trail = params.get('use_trailing_stop', False)
    trail_atr = params.get('trail_atr', 2.0)
    trail_anchor = params.get('trail_anchor', 'Peak High')

    if use_trail:
        side = "below" if direction == 'Long' else "above"
        anchor_desc = "running peak high" if trail_anchor == 'Peak High' else "running peak close"
        if direction == 'Short':
            anchor_desc = anchor_desc.replace("peak", "trough").replace("high", "low")
        return {
            "primary_exit": f"Trailing stop ({trail_atr:.2f} ATR off {anchor_desc}) or {hold_days}-day time stop",
            "stop_logic": f"Trail {trail_atr:.2f} ATR {side} {anchor_desc}, ATR frozen at signal-day value",
            "target_logic": "None (trail captures upside)",
            "notes": None
        }

    use_partial = params.get('use_partial_exits', False) and use_tgt
    tgt_frac = float(params.get('partial_target_fraction', 0.5)) if use_partial else 0.0

    if not use_stop and not use_tgt:
        primary = f"{hold_days}-day time stop"
    elif use_partial and use_stop:
        primary = f"{int(tgt_frac*100)}% at target, {int((1-tgt_frac)*100)}% at {hold_days}d time stop (or stop if first)"
    elif use_partial:
        primary = f"{int(tgt_frac*100)}% at target, {int((1-tgt_frac)*100)}% at {hold_days}d time stop"
    elif use_tgt and use_stop:
        primary = f"Target, Stop, or {hold_days}-day time stop"
    elif use_tgt:
        primary = f"Target or {hold_days}-day time stop"
    else:
        primary = f"Stop or {hold_days}-day time stop"
    
    if use_stop:
        if direction == 'Long':
            stop_logic = f"{stop_atr:.1f} ATR below entry"
        else:
            stop_logic = f"{stop_atr:.1f} ATR above entry"
    else:
        stop_logic = "None (time exit only)"
    
    if use_tgt:
        if direction == 'Long':
            tgt_logic = f"{tgt_atr:.1f} ATR above entry"
        else:
            tgt_logic = f"{tgt_atr:.1f} ATR below entry"
    else:
        tgt_logic = "None (time exit only)"
    
    return {
        "primary_exit": primary,
        "stop_logic": stop_logic,
        "target_logic": tgt_logic,
        "notes": None
    }

def _infer_timeframe(params):
    hold_days = params.get('holding_days', 10)
    entry_type = params.get('entry_type', '')
    
    if "Overnight" in entry_type or "Intraday" in entry_type or "Day Trade" in entry_type:
        return "Intraday"
    elif hold_days <= 2:
        return "Overnight"
    elif hold_days <= 10:
        return "Swing"
    else:
        return "Position"

def build_strategy_dict(params, tickers_to_run, pf, sqn, win_rate, expectancy_r):
    id_parts = []
    if params.get('perf_filters'):
        perf_str = "+".join([f"{f['window']}d {f['logic']} {f['thresh']:.0f}%ile" for f in params['perf_filters']])
        id_parts.append(perf_str)
    if params.get('perf_atr_filters'):
        patr_str = "+".join([f"{f['window']}d ATR {f['logic']} {f['thresh']:.0f}%ile" for f in params['perf_atr_filters']])
        id_parts.append(patr_str)
    if params.get('use_sznl'): id_parts.append(f"Sznl {params['sznl_logic']} {params['sznl_thresh']:.0f}")
    if params.get('use_acc_count_filter'): id_parts.append(f"{params['acc_count_thresh']} acc {params['acc_count_logic']} in {params['acc_count_window']}d")
    if params.get('use_dist_count_filter'): id_parts.append(f"{params['dist_count_thresh']} dist {params['dist_count_logic']} in {params['dist_count_window']}d")
    if params.get('use_ref_ticker_filter') and params.get('ref_filters'):
        ref_ticker = params.get('ref_ticker', 'IWM')
        ref_str = "+".join([f"{ref_ticker} {f['window']}d {f['logic']} {f['thresh']:.0f}%ile" for f in params['ref_filters']])
        id_parts.append(ref_str)
    if params.get('use_xsec_filter') and params.get('xsec_filters'):
        xsec_str = "+".join([f"XSec {f['window']}d {f['logic']} {f['thresh']:.0f}%ile" for f in params['xsec_filters']])
        id_parts.append(xsec_str)
    if params.get('use_t1_open_filter') and params.get('t1_open_filters'):
        for f in params['t1_open_filters']:
            t1_str = f"T+1 Open {f['logic']} {f['reference']}"
            if f['atr_offset'] != 0: t1_str += f" {'+' if f['atr_offset'] > 0 else ''}{f['atr_offset']} ATR"
            id_parts.append(t1_str)
    id_parts.append(f"Entry: {params['entry_type']}")
    id_parts.append(f"{params['holding_days']}d hold")
    strategy_id = ", ".join(id_parts) if id_parts else "Custom Strategy"
    
    grade, verdict, _ = grade_strategy(pf, sqn, win_rate, 100)
    strat_type, thesis = _infer_strategy_type(params)
    key_filters = _generate_key_filters(params)
    timeframe = _infer_timeframe(params)
    exit_summary = _generate_exit_summary(params)
    
    strategy = {
        "id": strategy_id,
        "name": "[EDIT: Strategy Name]",
        "setup": {
            "type": strat_type,
            "timeframe": timeframe,
            "thesis": thesis,
            "key_filters": key_filters
        },
        "exit_summary": exit_summary,
        "description": f"Backtest: {params['backtest_start_date']} to present. Tested on {len(tickers_to_run)} tickers.",
        "universe_tickers": "[EDIT: LIQUID_UNIVERSE or other]",
        "settings": {
            "trade_direction": params.get('trade_direction', 'Long'),
            "entry_type": params.get('entry_type', 'T+1 Open'),
            "max_one_pos": params.get('max_one_pos', True),
            "allow_same_day_reentry": params.get('allow_same_day_reentry', False),
            "max_daily_entries": params.get('max_daily_entries', 2),
            "max_total_positions": params.get('max_total_positions', 10),
            "entry_conf_bps": params.get('entry_conf_bps', 0),
            # Performance rank filters
            "perf_filters": params.get('perf_filters', []),
            "perf_atr_filters": params.get('perf_atr_filters', []),
            "perf_first_instance": params.get('perf_first_instance', False),
            "perf_lookback": params.get('perf_lookback', 21),
            # MA consecutive filters
            "ma_consec_filters": params.get('ma_consec_filters', []),
            # Seasonal
            "use_sznl": params.get('use_sznl', False), "sznl_logic": params.get('sznl_logic', '<'), "sznl_thresh": params.get('sznl_thresh', 33.0),
            "sznl_first_instance": params.get('sznl_first_instance', False), "sznl_lookback": params.get('sznl_lookback', 21),
            "use_market_sznl": params.get('use_market_sznl', False), "market_sznl_logic": params.get('market_sznl_logic', '<'), "market_sznl_thresh": params.get('market_sznl_thresh', 40.0),
            "market_ticker": "^GSPC",
            # 52-week / ATH
            "use_52w": params.get('use_52w', False), "52w_type": params.get('52w_type', 'New High'), "52w_first_instance": params.get('52w_first_instance', True),
            "52w_lookback": params.get('52w_lookback', 21), "52w_lag": params.get('52w_lag', 0),
            "52w_window": params.get('52w_window', 252),
            "exclude_52w_high": params.get('exclude_52w_high', False),
            "use_ath": params.get('use_ath', False), "ath_type": params.get('ath_type', 'Today is ATH'),
            "use_recent_ath": params.get('use_recent_ath', False), "recent_ath_invert": params.get('recent_ath_invert', False), "ath_lookback_days": params.get('ath_lookback_days', 21),
            "use_recent_52w": params.get('use_recent_52w', False), "recent_52w_invert": params.get('recent_52w_invert', False), "recent_52w_lookback": params.get('recent_52w_lookback', 21),
            "use_recent_52w_low": params.get('use_recent_52w_low', False), "recent_52w_low_invert": params.get('recent_52w_low_invert', False), "recent_52w_low_lookback": params.get('recent_52w_low_lookback', 21),
            # Price action
            "breakout_mode": params.get('breakout_mode', 'None'),
            "require_close_gt_open": params.get('require_close_gt_open', False),
            "use_range_filter": params.get('use_range_filter', False), "range_min": params.get('range_min', 0), "range_max": params.get('range_max', 100),
            "use_atr_ret_filter": params.get('use_atr_ret_filter', False), "atr_ret_min": params.get('atr_ret_min', 0.0), "atr_ret_max": params.get('atr_ret_max', 1.0),
            "use_range_atr_filter": params.get('use_range_atr_filter', False), "range_atr_logic": params.get('range_atr_logic', 'Between'), "range_atr_min": params.get('range_atr_min', 1.0), "range_atr_max": params.get('range_atr_max', 3.0),
            "use_open_gap_atr_filter": params.get('use_open_gap_atr_filter', False), "open_gap_atr_logic": params.get('open_gap_atr_logic', 'Between'), "open_gap_atr_min": params.get('open_gap_atr_min', 0.0), "open_gap_atr_max": params.get('open_gap_atr_max', 1.0),
            # Multi-day price action
            "price_action_filters": params.get('price_action_filters', []),
            # Distance from MA
            "use_ma_dist_filter": params.get('use_ma_dist_filter', False), "dist_ma_type": params.get('dist_ma_type', 'SMA 200'), "dist_logic": params.get('dist_logic', 'Between'), "dist_min": params.get('dist_min', 0.0), "dist_max": params.get('dist_max', 2.0),
            # Weekly MA Pullback
            "use_weekly_ma_pullback": params.get('use_weekly_ma_pullback', False), "wma_type": params.get('wma_type', 'EMA'), "wma_period": params.get('wma_period', 8),
            "wma_min_ext_pct": params.get('wma_min_ext_pct', 30.0), "wma_lookback_months": params.get('wma_lookback_months', 6), "wma_touch_logic": params.get('wma_touch_logic', 'Low <= MA'),
            # Vol/Return Delta Ratio
            "use_volret_delta": params.get('use_volret_delta', False), "vrd_method": params.get('vrd_method', 'Z-score diff'), "vrd_rank_window": params.get('vrd_rank_window', 'Expanding'),
            "vrd_vol_halflife": params.get('vrd_vol_halflife', 20), "vrd_ret_horizon": params.get('vrd_ret_horizon', 20), "vrd_delta_n": params.get('vrd_delta_n', 5),
            "vrd_min_periods": params.get('vrd_min_periods', 252), "vrd_pctile_min": params.get('vrd_pctile_min', 70.0), "vrd_pctile_max": params.get('vrd_pctile_max', 90.0),
            # Trend Ratio / VCR
            "use_tr_vcr_filter": params.get('use_tr_vcr_filter', False), "tr_vcr_metric": params.get('tr_vcr_metric', 'Trend Ratio (TR)'),
            "tr_vcr_window": params.get('tr_vcr_window', 20), "tr_vcr_sample_freq": params.get('tr_vcr_sample_freq', 5),
            "tr_vcr_min_periods": params.get('tr_vcr_min_periods', 252), "tr_vcr_rank_window": params.get('tr_vcr_rank_window', 'Expanding'),
            "tr_vcr_filter_mode": params.get('tr_vcr_filter_mode', 'Percentile rank'),
            "tr_vcr_pctile_min": params.get('tr_vcr_pctile_min', 70.0), "tr_vcr_pctile_max": params.get('tr_vcr_pctile_max', 100.0),
            "tr_vcr_raw_min": params.get('tr_vcr_raw_min', 1.0), "tr_vcr_raw_max": params.get('tr_vcr_raw_max', 5.0), "tr_vcr_raw_logic": params.get('tr_vcr_raw_logic', 'Between'),
            "tr_vcr_regime_quadrants": tuple(params.get('tr_vcr_regime_quadrants', ()) or ()),
            "tr_vcr_min_consec": params.get('tr_vcr_min_consec', 1), "tr_vcr_consec_first": params.get('tr_vcr_consec_first', False),
            # Volume
            "vol_gt_prev": params.get('vol_gt_prev', False),
            "use_vol": params.get('use_vol', False), "vol_logic": params.get('vol_logic', '>'), "vol_thresh": params.get('vol_thresh', 1.5), "vol_thresh_max": params.get('vol_thresh_max', 10.0),
            "use_vol_rank": params.get('use_vol_rank', False), "vol_rank_logic": params.get('vol_rank_logic', '<'), "vol_rank_thresh": params.get('vol_rank_thresh', 15.0),
            # Acc/Dist counts
            "use_acc_count_filter": params.get('use_acc_count_filter', False), "acc_count_window": params.get('acc_count_window', 21), "acc_count_logic": params.get('acc_count_logic', '>'), "acc_count_thresh": params.get('acc_count_thresh', 3),
            "use_dist_count_filter": params.get('use_dist_count_filter', False), "dist_count_window": params.get('dist_count_window', 21), "dist_count_logic": params.get('dist_count_logic', '>'), "dist_count_thresh": params.get('dist_count_thresh', 3),
            # Gap filter
            "use_gap_filter": params.get('use_gap_filter', False), "gap_lookback": params.get('gap_lookback', 21), "gap_logic": params.get('gap_logic', '>'), "gap_thresh": params.get('gap_thresh', 3),
            # Trend & regime
            "trend_filter": params.get('trend_filter', 'None'),
            "use_vix_filter": params.get('use_vix_filter', False), "vix_min": params.get('vix_min', 0.0), "vix_max": params.get('vix_max', 20.0),
            # Liquidity & data
            "min_price": params.get('min_price', 10.0), "min_vol": params.get('min_vol', 100000),
            "min_age": params.get('min_age', 0.25), "max_age": params.get('max_age', 100.0),
            "min_atr_pct": params.get('min_atr_pct', 0.0), "max_atr_pct": params.get('max_atr_pct', 10.0),
            # Time & cycle
            "use_dow_filter": params.get('use_dow_filter', False), "allowed_days": params.get('allowed_days', [0, 1, 2, 3, 4]),
            "allowed_cycles": params.get('allowed_cycles', [0, 1, 2, 3]),
            "excluded_years": params.get('excluded_years', []),
            # Reference ticker
            "use_ref_ticker_filter": params.get('use_ref_ticker_filter', False), "ref_ticker": params.get('ref_ticker', 'IWM'), "ref_filters": params.get('ref_filters', []),
            # T+1 open filter
            "use_t1_open_filter": params.get('use_t1_open_filter', False), "t1_open_filters": params.get('t1_open_filters', []),
            # Cross-sectional rank
            "use_xsec_filter": params.get('use_xsec_filter', False), "xsec_filters": params.get('xsec_filters', []),
            # ATR seasonal ranks
            "atr_sznl_filters": params.get('atr_sznl_filters', [])
        },
        "execution": {
            "risk_bps": 35,
            "risk_per_trade": "[EDIT: calculated from account size]",
            "slippage_bps": params.get('slippage_bps', 5),
            "stop_atr": params.get('stop_atr', 2.0),
            "tgt_atr": params.get('tgt_atr', 5.0),
            "hold_days": params.get('holding_days', 10),
            "use_stop_loss": params.get('use_stop_loss', True),
            "use_take_profit": params.get('use_take_profit', True),
            "use_trailing_stop": params.get('use_trailing_stop', False),
            "trail_atr": params.get('trail_atr', 2.0),
            "trail_anchor": params.get('trail_anchor', 'Peak High')
        },
        "stats": {
            "grade": f"{grade} ({verdict})",
            "win_rate": f"{win_rate:.1f}%",
            "expectancy": f"{expectancy_r:.2f}r",
            "profit_factor": f"{pf:.2f}"
        }
    }
    return strategy
    
def run_engine(universe_dict, params, sznl_map, market_series=None, vix_series=None, market_sznl_series=None, ref_ticker_ranks=None, xsec_rank_matrices=None, atr_sznl_map=None, fragility_df=None, market_sma_not_declining_series=None):
    all_potential_trades = []
    total = len(universe_dict)
    bt_start_ts = pd.to_datetime(params['backtest_start_date'])
    direction = params.get('trade_direction', 'Long')
    max_one_pos_per_ticker = params.get('max_one_pos', True)
    slippage_bps = params.get('slippage_bps', 5)
    
    # --- ENTRY MODES ---
    entry_mode = params['entry_type']
    is_pullback = "Pullback" in entry_mode
    is_limit_pers_025 = "Limit Order -0.25 ATR" in entry_mode
    is_limit_pers_05 = "Limit Order -0.5 ATR" in entry_mode
    is_limit_pers_10 = "Limit Order -1 ATR" in entry_mode
    is_limit_atr = entry_mode == "Limit (Close -0.5 ATR)"
    is_limit_prev = entry_mode == "Limit (Prev Close)"
    is_limit_open_atr_025 = entry_mode == "Limit (Open +/- 0.25 ATR)"
    is_limit_open_atr = entry_mode == "Limit (Open +/- 0.5 ATR)"
    is_limit_open_atr_075 = entry_mode == "Limit (Open +/- 0.75 ATR)"
    is_limit_open_atr_100 = entry_mode == "Limit (Open +/- 1 ATR)"
    is_limit_open_atr_gtc = entry_mode == "Limit (Open +/- 0.5 ATR) GTC"
    is_day_trade_limit = entry_mode in (
        "Day Trade (Limit Open +/- 0.5 ATR, Exit Close)",
        "Day Trade (Limit Open +/- 0.75 ATR, Exit Close)",
        "Day Trade (Limit Open +/- 1 ATR, Exit Close)",
    )
    # Pull the ATR offset out of the label so the entry block stays generic.
    if is_day_trade_limit:
        if "0.75 ATR" in entry_mode:
            day_trade_atr_mult = 0.75
        elif "1 ATR" in entry_mode:
            day_trade_atr_mult = 1.0
        else:
            day_trade_atr_mult = 0.5
    else:
        day_trade_atr_mult = 0.5

    # Intraday mode toggle — only meaningful for Day Trade entries. When on,
    # 15min bars from intraday_data drive the entry-bar fill and bar-by-bar
    # stop/target walk on the entry day.
    intraday_mode = bool(is_day_trade_limit and params.get('use_intraday', False))
    intraday_loader = None
    intraday_universe = set()
    if intraday_mode:
        try:
            import intraday_data as _intraday_loader_mod
            intraday_loader = _intraday_loader_mod
            intraday_universe = intraday_loader.available_tickers()
        except Exception:
            intraday_mode = False
    is_limit_pivot = entry_mode == "Limit (Untested Pivot)"
    is_gap_up = "Gap Up Only" in entry_mode
    is_overnight = "Overnight" in entry_mode
    is_intraday = "Intraday" in entry_mode
    is_cond_close_lower = "T+1 Close if < Signal Close" in entry_mode
    # Band-conditional T+1 close entry: "+X to +Y ATR" parsed from entry_mode.
    # Adds new modes (e.g. +0.5 to +1, +0.75 to +1) without separate flags.
    _band_match = re.search(r'\+(\d+(?:\.\d+)?) to \+(\d+(?:\.\d+)?) ATR', entry_mode)
    is_cond_close_higher_band = _band_match is not None
    _band_lower_mult = float(_band_match.group(1)) if is_cond_close_higher_band else 0.0
    _band_upper_mult = float(_band_match.group(2)) if is_cond_close_higher_band else 0.0
    is_cond_close_higher = ("T+1 Close if > Signal Close" in entry_mode) and not is_cond_close_higher_band
    
    pullback_col = None
    if "10 SMA" in entry_mode: pullback_col = "SMA10"
    elif "21 EMA" in entry_mode: pullback_col = "EMA21"
    pullback_use_level = "Level" in entry_mode
    
    gap_window = params.get('gap_lookback', 21)
    total_signals_generated = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    req_custom_mas = list(set([f['length'] for f in params.get('ma_consec_filters', [])]))
    acc_win = params['acc_count_window'] if params.get('use_acc_count_filter', False) else None
    dist_win = params['dist_count_window'] if params.get('use_dist_count_filter', False) else None
    use_t1_open_filter = params.get('use_t1_open_filter', False)
    t1_open_filters = params.get('t1_open_filters', [])

    # Weekly MA Pullback configs
    weekly_ma_configs = None
    if params.get('use_weekly_ma_pullback', False):
        weekly_ma_configs = [{'type': params['wma_type'], 'period': params['wma_period']}]

    # Reference Ticker Filter params
    use_ref_ticker_filter = params.get('use_ref_ticker_filter', False)
    ref_filters = params.get('ref_filters', [])

    # Cross-Sectional Rank Filter params
    use_xsec_filter = params.get('use_xsec_filter', False)
    xsec_filters = params.get('xsec_filters', [])

    for i, (ticker, df_raw) in enumerate(universe_dict.items()):
        status_text.text(f"Scanning signals for {ticker}...")
        progress_bar.progress((i+1)/total)

        if ticker == MARKET_TICKER and MARKET_TICKER not in params.get('universe_tickers', []): continue
        if ticker == VIX_TICKER: continue
        if len(df_raw) < 100: continue

        ticker_last_exit = pd.Timestamp.min

        # Intraday prefetch: pull the full 15min parquet for this ticker into
        # an in-memory cache once, so each per-signal-day lookup below is an
        # O(1) dict hit instead of re-reading the parquet (~200x speedup).
        # Single-slot cache: clearing first keeps memory bounded to one ticker.
        if intraday_mode and ticker in intraday_universe:
            try:
                intraday_loader.clear_bars_cache()
                intraday_loader.prefetch_ticker(ticker)
            except Exception:
                pass

        try:
            df = calculate_indicators(df_raw, sznl_map, ticker, market_series, vix_series, market_sznl_series, gap_window, req_custom_mas, acc_win, dist_win, ref_ticker_ranks, weekly_ma_configs, xsec_rank_matrices)
            if market_sma_not_declining_series is not None:
                df['Market_SMA200_Not_Declining'] = market_sma_not_declining_series.reindex(df.index, method='ffill').fillna(False)
            df = df[df.index >= bt_start_ts]
            if df.empty: continue

            # Merge ATR seasonal ranks (if available) — 6 rank columns joined by date
            if atr_sznl_map and ticker in atr_sznl_map:
                atr_ranks = atr_sznl_map[ticker]
                df_dates = df.index.normalize()
                for col in ATR_SZNL_COLS:
                    if col in atr_ranks.columns:
                        df[col] = atr_ranks[col].reindex(df_dates).values
            
            # --- SIGNAL GENERATION ---
            conditions = []
            trend_opt = params.get('trend_filter', 'None')
            
            if trend_opt == "Price > 200 SMA": conditions.append(df['Close'] > df['SMA200'])
            elif trend_opt == "Price > Rising 200 SMA": conditions.append((df['Close'] > df['SMA200']) & (df['SMA200'] > df['SMA200'].shift(1)))
            elif trend_opt == "Not Below Declining 200 SMA": conditions.append(~((df['Close'] < df['SMA200']) & (df['SMA200'] < df['SMA200'].shift(1))))
            elif trend_opt == "Price < 200 SMA": conditions.append(df['Close'] < df['SMA200'])
            elif trend_opt == "Price < Falling 200 SMA": conditions.append((df['Close'] < df['SMA200']) & (df['SMA200'] < df['SMA200'].shift(1)))
            elif "Market" in trend_opt and 'Market_Above_SMA200' in df.columns:
                if trend_opt == "Market > 200 SMA": conditions.append(df['Market_Above_SMA200'])
                elif trend_opt == "Market < 200 SMA": conditions.append(~df['Market_Above_SMA200'])
                elif trend_opt == "Market Not Declining 200 SMA" and 'Market_SMA200_Not_Declining' in df.columns:
                    conditions.append(df['Market_SMA200_Not_Declining'])
                
            _max_vol = params.get('max_vol', 0) or 0
            _vol_cond = (df['vol_ma'] >= params['min_vol'])
            if _max_vol > 0:
                _vol_cond = _vol_cond & (df['vol_ma'] <= _max_vol)
            conditions.append((df['Close'] >= params['min_price']) & _vol_cond & (df['age_years'] >= params['min_age']) & (df['age_years'] <= params['max_age']) & (df['ATR_Pct'] >= params['min_atr_pct']) & (df['ATR_Pct'] <= params['max_atr_pct']))
            
            if params.get('require_close_gt_open', False): conditions.append(df['Close'] > df['Open'])
            
            bk_mode = params.get('breakout_mode', 'None')
            if bk_mode == "Close > Prev Day High": conditions.append(df['Close'] > df['High'].shift(1))
            elif bk_mode == "Close < Prev Day Low": conditions.append(df['Close'] < df['Low'].shift(1))
            
            if params.get('use_range_filter', False): conditions.append((df['RangePct'] * 100 >= params['range_min']) & (df['RangePct'] * 100 <= params['range_max']))
            
            if params.get('use_atr_ret_filter', False):
                conditions.append((df['today_return_atr'] >= params['atr_ret_min']) & (df['today_return_atr'] <= params['atr_ret_max']))
            if params.get('use_range_atr_filter', False):
                range_in_atr = (df['High'] - df['Low']) / df['ATR']
                if params['range_atr_logic'] == '>':
                    conditions.append(range_in_atr > params['range_atr_min'])
                elif params['range_atr_logic'] == '<':
                    conditions.append(range_in_atr < params['range_atr_max'])
                elif params['range_atr_logic'] == 'Between':
                    conditions.append((range_in_atr >= params['range_atr_min']) & (range_in_atr <= params['range_atr_max']))
            if params.get('use_open_gap_atr_filter', False):
                open_gap_atr = (df['Open'] - df['Close'].shift(1)) / df['ATR']
                if params['open_gap_atr_logic'] == '>':
                    conditions.append(open_gap_atr > params['open_gap_atr_min'])
                elif params['open_gap_atr_logic'] == '<':
                    conditions.append(open_gap_atr < params['open_gap_atr_max'])
                elif params['open_gap_atr_logic'] == 'Between':
                    conditions.append((open_gap_atr >= params['open_gap_atr_min']) & (open_gap_atr <= params['open_gap_atr_max']))
            # --- MULTI-DAY PRICE ACTION FILTERS ---
            for pa in params.get('price_action_filters', []):
                pa_lag = pa.get('lag', 0)
                pa_type = pa['type']
                if pa_type == 'range_pct':
                    series = df['RangePct'].shift(pa_lag) * 100
                    conditions.append((series >= pa['min']) & (series <= pa['max']))
                elif pa_type == 'atr_ret':
                    series = df['today_return_atr'].shift(pa_lag)
                    conditions.append((series >= pa['min']) & (series <= pa['max']))
                elif pa_type == 'range_atr':
                    series = ((df['High'] - df['Low']) / df['ATR']).shift(pa_lag)
                    pa_logic = pa.get('logic', 'Between')
                    if pa_logic == '>':
                        conditions.append(series > pa['min'])
                    elif pa_logic == '<':
                        conditions.append(series < pa['max'])
                    else:
                        conditions.append((series >= pa['min']) & (series <= pa['max']))
                elif pa_type == 'close_gt_open':
                    conditions.append(df['Close'].shift(pa_lag) > df['Open'].shift(pa_lag))
                elif pa_type == 'close_lt_open':
                    conditions.append(df['Close'].shift(pa_lag) < df['Open'].shift(pa_lag))
                elif pa_type == 'close_gt_prev_high':
                    conditions.append(df['Close'].shift(pa_lag) > df['High'].shift(pa_lag + 1))
                elif pa_type == 'close_lt_prev_low':
                    conditions.append(df['Close'].shift(pa_lag) < df['Low'].shift(pa_lag + 1))
                elif pa_type == 'atr_move':
                    n_bars = int(pa.get('n_bars', 1))
                    series = (df['Close'].shift(pa_lag) - df['Close'].shift(pa_lag + n_bars)) / df['ATR'].shift(pa_lag)
                    conditions.append((series >= pa['min']) & (series <= pa['max']))
                elif pa_type in ('trailing_low', 'trailing_high'):
                    window = int(pa.get('window', 21))
                    end_lag = int(pa.get('end_lag', pa_lag))
                    if end_lag < pa_lag:
                        end_lag = pa_lag
                    if pa_type == 'trailing_low':
                        rolling = df['Low'].rolling(window).min()
                        match = lambda L: df['Low'].shift(L) <= rolling.shift(L)
                    else:
                        rolling = df['High'].rolling(window).max()
                        match = lambda L: df['High'].shift(L) >= rolling.shift(L)
                    combined = match(pa_lag)
                    for L in range(pa_lag + 1, end_lag + 1):
                        combined = combined | match(L)
                    conditions.append(combined)
            if params.get('use_dow_filter', False): conditions.append(df['DayOfWeekVal'].isin(params['allowed_days']))

            if params.get('use_month_filter', False):
                allowed_months = params.get('allowed_months', list(range(1, 13)))
                conditions.append(pd.Series(df.index.month, index=df.index).isin(allowed_months))

            if 'allowed_cycles' in params and len(params['allowed_cycles']) < 4:
                year_rems = df.index.year % 4
                conditions.append(pd.Series(year_rems, index=df.index).isin(params['allowed_cycles']))

            _excluded_years = params.get('excluded_years', [])
            if _excluded_years:
                conditions.append(~pd.Series(df.index.year, index=df.index).isin(_excluded_years))
                
            if params.get('use_gap_filter', False):
                if params['gap_logic'] == ">": conditions.append(df['GapCount'] > params['gap_thresh'])
                elif params['gap_logic'] == "<": conditions.append(df['GapCount'] < params['gap_thresh'])
                elif params['gap_logic'] == "=": conditions.append(df['GapCount'] == params['gap_thresh'])

            # Earnings proximity filter — comparison operator on the SIGNED
            # trading-day offset to the nearest earnings announcement.
            # Convention: offset = signal_date - earnings_date (trading days).
            #   negative → signal is before earnings; positive → after; 0 → day-of.
            # NaN (ticker has no earnings) is treated as False for positive
            # operators (=, <, >, Between) and True for Not Between, mirroring
            # the prior Within/Outside semantics.
            if params.get('use_earnings_filter', False):
                _e_map = params.get('earnings_map') or {}
                _e_dates = _e_map.get(ticker.upper()) if _e_map else None
                _holidays = params.get('_earnings_holidays')
                if _holidays is None:
                    from pandas.tseries.holiday import USFederalHolidayCalendar
                    _hcal = USFederalHolidayCalendar()
                    _holidays = _hcal.holidays(start='1990-01-01', end='2035-01-01').to_numpy().astype('datetime64[D]')
                    params['_earnings_holidays'] = _holidays
                _nearest = compute_signed_earnings_offsets(df.index, _e_dates, _holidays)
                _op = params.get('earnings_logic', 'Between')
                if _op == '=':
                    _v = int(params.get('earnings_value', 0))
                    cond = (_nearest == _v)
                elif _op == '<':
                    _v = int(params.get('earnings_value', 0))
                    cond = (_nearest < _v)
                elif _op == '>':
                    _v = int(params.get('earnings_value', 0))
                    cond = (_nearest > _v)
                elif _op == 'Between':
                    _lo = int(params.get('earnings_min', 0))
                    _hi = int(params.get('earnings_max', 0))
                    if _lo > _hi:
                        _lo, _hi = _hi, _lo
                    cond = (_nearest >= _lo) & (_nearest <= _hi)
                elif _op == 'Not Between':
                    _lo = int(params.get('earnings_min', 0))
                    _hi = int(params.get('earnings_max', 0))
                    if _lo > _hi:
                        _lo, _hi = _hi, _lo
                    cond = (_nearest < _lo) | (_nearest > _hi) | _nearest.isna()
                else:
                    cond = pd.Series(True, index=df.index)
                # NaN safety: positive ops treat NaN as False (signal excluded);
                # Not Between explicitly OR'd NaN above so leave alone.
                if _op != 'Not Between':
                    cond = cond.fillna(False)
                conditions.append(cond)

            # Earnings-quality filters — eps_surprise_pct, rev_surprise_pct,
            # eps_yoy, rev_yoy from the LAST reported earnings at or before
            # each bar. We reindex the per-ticker metrics frame onto df.index
            # with method='ffill' so each bar inherits the most recent
            # announcement's metrics. NaN before the first earnings (or for
            # tickers with no coverage) -> filter excludes the signal.
            _quality_active = any(
                params.get(k, False) for k in (
                    'use_eps_surp_filter', 'use_rev_surp_filter',
                    'use_eps_yoy_filter',  'use_rev_yoy_filter',
                )
            )
            if _quality_active:
                _met_map = params.get('earnings_metrics_map') or {}
                _met_df = _met_map.get(ticker.upper())
                if _met_df is not None and not _met_df.empty:
                    _bar_dates = pd.DatetimeIndex(df.index).normalize()
                    try:
                        _bar_dates = _bar_dates.tz_localize(None)
                    except (TypeError, AttributeError):
                        pass
                    _met_aligned = _met_df.reindex(_bar_dates, method='ffill')
                    _met_aligned.index = df.index
                else:
                    _met_aligned = pd.DataFrame(
                        np.nan, index=df.index,
                        columns=['eps_surprise_pct', 'rev_surprise_pct', 'eps_yoy', 'rev_yoy'],
                    )

                def _apply_quality(use_key, logic_key, lo_key, hi_key, col):
                    if not params.get(use_key, False):
                        return
                    _series = _met_aligned[col]
                    _op_q = params.get(logic_key, '>')
                    _lo_q = float(params.get(lo_key, 0.0))
                    if _op_q == '>':
                        _c = _series > _lo_q
                    elif _op_q == '<':
                        _c = _series < _lo_q
                    elif _op_q == 'Between':
                        _hi_q = float(params.get(hi_key, _lo_q))
                        if _lo_q > _hi_q:
                            _lo_q, _hi_q = _hi_q, _lo_q
                        _c = (_series >= _lo_q) & (_series <= _hi_q)
                    else:
                        _c = pd.Series(True, index=df.index)
                    conditions.append(_c.fillna(False))

                _apply_quality('use_eps_surp_filter', 'eps_surp_logic',
                               'eps_surp_min', 'eps_surp_max', 'eps_surprise_pct')
                _apply_quality('use_rev_surp_filter', 'rev_surp_logic',
                               'rev_surp_min', 'rev_surp_max', 'rev_surprise_pct')
                _apply_quality('use_eps_yoy_filter', 'eps_yoy_logic',
                               'eps_yoy_min', 'eps_yoy_max', 'eps_yoy')
                _apply_quality('use_rev_yoy_filter', 'rev_yoy_logic',
                               'rev_yoy_min', 'rev_yoy_max', 'rev_yoy')

            # Analyst grades filter — trailing net upgrades (upgrades minus
            # downgrades) over a calendar-day window ending on each bar.
            # Tickers without coverage produce all-zero counts, so ">= 1"
            # naturally excludes them.
            if params.get('use_grades_filter', False):
                _g_map = params.get('grades_map') or {}
                _g_df = _g_map.get(ticker.upper())
                try:
                    from analyst_grades import trailing_counts as _trailing_counts
                except ImportError:
                    _trailing_counts = None
                if _trailing_counts is not None:
                    _window = int(params.get('grades_window_days', 30))
                    _counts = _trailing_counts(_g_df, df.index, _window)
                    _net = _counts['net'].reindex(df.index)
                    _op_g = params.get('grades_logic', '>=')
                    _t_g = int(params.get('grades_thresh', 1))
                    if _op_g == '>=':
                        _c = _net >= _t_g
                    elif _op_g == '<=':
                        _c = _net <= _t_g
                    elif _op_g == '>':
                        _c = _net > _t_g
                    elif _op_g == '<':
                        _c = _net < _t_g
                    elif _op_g == '=':
                        _c = _net == _t_g
                    else:
                        _c = pd.Series(True, index=df.index)
                    conditions.append(_c.fillna(False))

            if params.get('use_acc_count_filter', False):
                v2_prefix = "AccCount_v2_" if params.get('use_acc_dist_v2', False) else "AccCount_"
                col = f"{v2_prefix}{params['acc_count_window']}"
                if col in df.columns:
                    r_acc = df[col]
                    if params['acc_count_logic'] == ">": conditions.append(r_acc > params['acc_count_thresh'])
                    elif params['acc_count_logic'] == "<": conditions.append(r_acc < params['acc_count_thresh'])
                    elif params['acc_count_logic'] == "=": conditions.append(r_acc == params['acc_count_thresh'])

            if params.get('use_dist_count_filter', False):
                v2_prefix = "DistCount_v2_" if params.get('use_acc_dist_v2', False) else "DistCount_"
                col = f"{v2_prefix}{params['dist_count_window']}"
                if col in df.columns:
                    r_dist = df[col]
                    if params['dist_count_logic'] == ">": conditions.append(r_dist > params['dist_count_thresh'])
                    elif params['dist_count_logic'] == "<": conditions.append(r_dist < params['dist_count_thresh'])
                    elif params['dist_count_logic'] == "=": conditions.append(r_dist == params['dist_count_thresh'])
            
            for pf in params.get('perf_filters', []):
                col = f"rank_ret_{pf['window']}d"
                if pf['logic'] == '<': c_f = (df[col] < pf['thresh'])
                elif pf['logic'] == '>': c_f = (df[col] > pf['thresh'])
                elif pf['logic'] == 'Between': c_f = (df[col] >= pf['thresh']) & (df[col] <= pf.get('thresh_max', 100.0))
                elif pf['logic'] == 'Not Between': c_f = (df[col] < pf['thresh']) | (df[col] > pf.get('thresh_max', 100.0))
                else: continue
                if pf['consecutive'] > 1: c_f = c_f.rolling(pf['consecutive']).sum() == pf['consecutive']
                conditions.append(c_f)

            for paf in params.get('perf_atr_filters', []):
                col = f"rank_ret_atr_{paf['window']}d"
                if col not in df.columns: continue
                if paf['logic'] == '<': c_f = (df[col] < paf['thresh'])
                elif paf['logic'] == '>': c_f = (df[col] > paf['thresh'])
                elif paf['logic'] == 'Between': c_f = (df[col] >= paf['thresh']) & (df[col] <= paf.get('thresh_max', 100.0))
                elif paf['logic'] == 'Not Between': c_f = (df[col] < paf['thresh']) | (df[col] > paf.get('thresh_max', 100.0))
                else: continue
                if paf.get('consecutive', 1) > 1: c_f = c_f.rolling(paf['consecutive']).sum() == paf['consecutive']
                conditions.append(c_f)

            for asf in params.get('atr_sznl_filters', []):
                col = f"atr_sznl_{asf['window']}d"
                if col not in df.columns: continue
                if asf['logic'] == '<': c_f = (df[col] < asf['thresh'])
                elif asf['logic'] == '>': c_f = (df[col] > asf['thresh'])
                elif asf['logic'] == 'Between': c_f = (df[col] >= asf['thresh']) & (df[col] <= asf.get('thresh_max', 100.0))
                elif asf['logic'] == 'Not Between': c_f = (df[col] < asf['thresh']) | (df[col] > asf.get('thresh_max', 100.0))
                else: continue
                if asf.get('consecutive', 1) > 1: c_f = c_f.rolling(asf['consecutive']).sum() == asf['consecutive']
                conditions.append(c_f)

            # --- CROSS-SECTIONAL RANK FILTER ---
            if use_xsec_filter and xsec_filters:
                for xf in xsec_filters:
                    col = f"xsec_rank_ret_{xf['window']}d"
                    if col in df.columns:
                        if xf['logic'] == '<': c_f = (df[col] < xf['thresh'])
                        elif xf['logic'] == '>': c_f = (df[col] > xf['thresh'])
                        elif xf['logic'] == 'Between': c_f = (df[col] >= xf['thresh']) & (df[col] <= xf.get('thresh_max', 100.0))
                        elif xf['logic'] == 'Not Between': c_f = (df[col] < xf['thresh']) | (df[col] > xf.get('thresh_max', 100.0))
                        else: continue
                        if xf.get('consecutive', 1) > 1: c_f = c_f.rolling(xf['consecutive']).sum() == xf['consecutive']
                        conditions.append(c_f)

            # --- OR FILTER GROUPS (at least one condition in each group must be true) ---
            for group in params.get('or_filter_groups', []):
                group_conds = []
                for cond in group:
                    ctype = cond.get('type', 'perf')
                    window = cond['window']
                    if ctype == 'perf':
                        col = f"rank_ret_{window}d"
                    elif ctype == 'xsec':
                        col = f"xsec_rank_ret_{window}d"
                    else:
                        continue
                    if col not in df.columns:
                        continue
                    if cond['logic'] == '<':
                        group_conds.append(df[col] < cond['thresh'])
                    elif cond['logic'] == '>':
                        group_conds.append(df[col] > cond['thresh'])
                if group_conds:
                    or_result = group_conds[0]
                    for gc in group_conds[1:]:
                        or_result = or_result | gc
                    conditions.append(or_result)

            for f in params.get('ma_consec_filters', []):
                col = f"SMA{f['length']}"
                mask = (df['Close'] > df[col]) if f['logic'] == 'Above' else (df['Close'] < df[col])
                if f['consec'] > 1: mask = mask.rolling(f['consec']).sum() == f['consec']
                conditions.append(mask)

            if params.get('use_sznl', False):
                sznl_cond = (df['Sznl'] < params['sznl_thresh']) if params['sznl_logic'] == '<' else (df['Sznl'] > params['sznl_thresh'])
                if params.get('sznl_first_instance', False): sznl_cond = apply_first_instance_filter(sznl_cond, params.get('sznl_lookback', 21))
                conditions.append(sznl_cond)

            if params.get('use_market_sznl', False) and 'Market_Sznl' in df.columns:
                market_sznl_cond = (df['Market_Sznl'] < params['market_sznl_thresh']) if params['market_sznl_logic'] == '<' else (df['Market_Sznl'] > params['market_sznl_thresh'])
                conditions.append(market_sznl_cond)
                
            if params.get('use_52w', False):
                # Window now configurable (default 252 = 52w). Old strategies
                # carrying '52w_type': 'New 52w High'/'New 52w Low' continue
                # to work — the type-string check accepts both legacy and new
                # forms ('New High' / 'New Low'). At window=252 the result is
                # identical to df['is_52w_high'] / df['is_52w_low'], so behavior
                # of legacy configs is unchanged.
                _w52 = int(params.get('52w_window', 252))
                _t52 = params.get('52w_type', 'New High')
                _is_high = _t52 in ('New High', 'New 52w High')
                if _is_high:
                    _rolling_max = df['High'].shift(1).rolling(_w52).max()
                    c_52_raw = df['High'] > _rolling_max
                else:
                    _rolling_min = df['Low'].shift(1).rolling(_w52).min()
                    c_52_raw = df['Low'] < _rolling_min
                lag_days = params.get('52w_lag', 0)
                if lag_days > 0: c_52_raw = c_52_raw.shift(lag_days).fillna(False)
                if params.get('52w_first_instance', False): c_52 = apply_first_instance_filter(c_52_raw, params.get('52w_lookback', 21))
                else: c_52 = c_52_raw
                conditions.append(c_52)
                
            if params.get('exclude_52w_high', False): 
                conditions.append(~df['is_52w_high'])
                
            if params.get('use_ath', False):
                if params['ath_type'] == 'Today is ATH':
                    conditions.append(df['is_ath'])
                else:  # Today is NOT ATH
                    conditions.append(~df['is_ath'])
            if params.get('use_recent_ath', False):
                ath_lookback = params.get('ath_lookback_days', 21)
                recent_ath_mask = df['is_ath'].rolling(window=ath_lookback, min_periods=1).max().astype(bool)
                if params.get('recent_ath_invert', False):
                    conditions.append(~recent_ath_mask)
                else:
                    conditions.append(recent_ath_mask)
            if params.get('use_recent_52w', False):
                r52w_lookback = params.get('recent_52w_lookback', 21)
                recent_52w_mask = df['is_52w_high'].rolling(window=r52w_lookback, min_periods=1).max().astype(bool)
                if params.get('recent_52w_invert', False):
                    conditions.append(~recent_52w_mask)
                else:
                    conditions.append(recent_52w_mask)
            if params.get('use_recent_52w_low', False):
                r52w_low_lookback = params.get('recent_52w_low_lookback', 21)
                recent_52w_low_mask = df['is_52w_low'].rolling(window=r52w_low_lookback, min_periods=1).max().astype(bool)
                if params.get('recent_52w_low_invert', False):
                    conditions.append(~recent_52w_low_mask)
                else:
                    conditions.append(recent_52w_low_mask)
            if params.get('vol_gt_prev', False): 
                conditions.append(df['Volume'] > df['Volume'].shift(1))
            if params.get('use_vol', False):
                _vl = params.get('vol_logic', '>')
                if _vl == '<':
                    conditions.append(df['vol_ratio'] < params['vol_thresh'])
                elif _vl == 'Between':
                    conditions.append((df['vol_ratio'] >= params['vol_thresh']) & (df['vol_ratio'] <= params.get('vol_thresh_max', 10.0)))
                else:
                    conditions.append(df['vol_ratio'] > params['vol_thresh'])
            
            if params.get('use_vol_rank', False):
                vr_col = 'vol_ratio_10d_rank'
                if params['vol_rank_logic'] == ">": conditions.append(df[vr_col] > params['vol_rank_thresh'])
                elif params['vol_rank_logic'] == "<": conditions.append(df[vr_col] < params['vol_rank_thresh'])

            if params.get('use_ma_dist_filter', False):
                ma_col_map = {"SMA 10": "SMA10", "SMA 20": "SMA20", "SMA 50": "SMA50", "SMA 100": "SMA100", "SMA 200": "SMA200", "EMA 8": "EMA8", "EMA 11": "EMA11", "EMA 21": "EMA21","52-Week High": "High_52w", "All-Time High": "ATH_Level"}
                ma_target = ma_col_map.get(params['dist_ma_type'])
                if ma_target and ma_target in df.columns:
                    # Percent-space distance: distance from MA expressed as a
                    # percentage of the MA itself, divided by ATR as a percent
                    # of price. Differs from (Close - MA) / ATR because the
                    # MA-distance denominator is the MA, not the price.
                    atr_pct = df['ATR'] / df['Close']
                    ma_pct = (df['Close'] - df[ma_target]) / df[ma_target]
                    dist_val = ma_pct / atr_pct
                    if params['dist_logic'] == "Greater Than (>)": conditions.append(dist_val > params['dist_min'])
                    elif params['dist_logic'] == "Less Than (<)": conditions.append(dist_val < params['dist_max'])
                    elif params['dist_logic'] == "Between": conditions.append((dist_val >= params['dist_min']) & (dist_val <= params['dist_max']))
                    
            if params.get('use_vix_filter', False) and 'VIX_Value' in df.columns:
                vix_val = df['VIX_Value']
                conditions.append((vix_val >= params['vix_min']) & (vix_val <= params['vix_max']))

            # --- WEEKLY MA PULLBACK FILTER (first touch after extension) ---
            if params.get('use_weekly_ma_pullback', False):
                wma_col = f"Weekly_{params['wma_type']}{params['wma_period']}"
                if wma_col in df.columns:
                    lookback_days = params['wma_lookback_months'] * 21
                    pct_above = (df['High'] / df[wma_col] - 1) * 100

                    # Staleness guard: extension must have occurred within lookback window
                    max_ext = pct_above.rolling(lookback_days, min_periods=21).max()
                    recently_extended = max_ext >= params['wma_min_ext_pct']

                    # Touch condition
                    if params.get('wma_touch_logic', 'Low <= MA') == 'Low <= MA':
                        is_touch = df['Low'] <= df[wma_col]
                    else:
                        is_touch = df['Close'] <= df[wma_col]

                    # First-touch-after-extension: only fire on the FIRST day
                    # the stock crosses below the MA since it was extended.
                    # Uses cumsum trick: extension days increment a counter;
                    # a touch is "new" only if that counter advanced since the
                    # previous touch.
                    is_extended = pct_above >= params['wma_min_ext_pct']
                    ext_cumsum = is_extended.astype(int).cumsum()
                    touch_ext_count = ext_cumsum.where(is_touch)
                    prev_touch_ext = touch_ext_count.ffill().shift(1).fillna(0)
                    first_touch = is_touch & (ext_cumsum > prev_touch_ext) & recently_extended

                    conditions.append(first_touch)

            # --- VOL/RETURN DELTA RATIO FILTER ---
            # "Is vol expanding more abnormally than return is changing, relative
            # to each series' own history?" Three metric variants; ranked vs own
            # history (expanding or rolling 5y); rank shifted by 1 to avoid
            # look-ahead. Fires when rank lands in the configured pctile band.
            if params.get('use_volret_delta', False):
                halflife = int(params.get('vrd_vol_halflife', 20))
                ret_horizon = int(params.get('vrd_ret_horizon', 20))
                delta_n = int(params.get('vrd_delta_n', 5))
                method = params.get('vrd_method', 'Z-score diff')
                rank_window = params.get('vrd_rank_window', 'Expanding')
                p_min = params.get('vrd_pctile_min', 70.0) / 100.0
                p_max = params.get('vrd_pctile_max', 90.0) / 100.0
                min_periods = int(params.get('vrd_min_periods', 252))

                rets = df['Close'].pct_change()
                vol_t = rets.ewm(halflife=halflife, min_periods=halflife).std() * np.sqrt(252)
                ret_t = df['Close'].pct_change(ret_horizon)
                d_vol = vol_t.diff(delta_n)
                d_ret = ret_t.diff(delta_n)

                if method == 'd_vol / |d_ret|':
                    abs_dret = d_ret.abs()
                    eps = abs_dret.rolling(252, min_periods=21).mean() * 0.01
                    safe_dret = abs_dret.where(abs_dret > eps, eps).replace(0, np.nan)
                    metric = d_vol / safe_dret
                elif method == 'Pct-change diff':
                    vol_ref = vol_t.shift(delta_n)
                    ret_ref = ret_t.shift(delta_n)
                    vol_ref = vol_ref.where(vol_ref.abs() > 1e-8)
                    ret_ref = ret_ref.where(ret_ref.abs() > 1e-6)
                    metric = (d_vol / vol_ref) - (d_ret / ret_ref)
                elif method == 'Vol delta only':
                    metric = d_vol
                else:  # Z-score diff (default, most stable)
                    z_win = 252
                    vol_z = (d_vol - d_vol.rolling(z_win).mean()) / d_vol.rolling(z_win).std()
                    ret_z = (d_ret - d_ret.rolling(z_win).mean()) / d_ret.rolling(z_win).std()
                    metric = vol_z - ret_z

                if rank_window == 'Rolling 5y':
                    metric_rank = metric.rolling(252 * 5, min_periods=min_periods).rank(pct=True)
                else:
                    metric_rank = metric.expanding(min_periods=min_periods).rank(pct=True)

                metric_rank = metric_rank.shift(1)
                in_band = (metric_rank > p_min) & (metric_rank < p_max)
                conditions.append(in_band.fillna(False))

            # --- TREND RATIO / VCR FILTER (Abdelmessih) ---
            # TR = weekly-sampled RV / daily-sampled RV; VCR = max(r^2)/sum(r^2)
            # over a rolling window. Median-split (or pctile-band) classifier
            # of trend vs spike. All series shift(1) before evaluation to use
            # only data through prior close.
            if params.get('use_tr_vcr_filter', False):
                metric_name = params.get('tr_vcr_metric', 'Trend Ratio (TR)')
                window_tv = int(params.get('tr_vcr_window', 20))
                sample_freq_tv = int(params.get('tr_vcr_sample_freq', 5))
                rank_window_tv = params.get('tr_vcr_rank_window', 'Expanding')
                min_periods_tv = int(params.get('tr_vcr_min_periods', 252))
                filter_mode_tv = params.get('tr_vcr_filter_mode', 'Percentile rank')
                p_min_tv = params.get('tr_vcr_pctile_min', 70.0) / 100.0
                p_max_tv = params.get('tr_vcr_pctile_max', 100.0) / 100.0
                raw_min_tv = float(params.get('tr_vcr_raw_min', 1.0))
                raw_max_tv = float(params.get('tr_vcr_raw_max', 5.0))
                raw_logic_tv = params.get('tr_vcr_raw_logic', 'Between')
                quadrants_tv = list(params.get('tr_vcr_regime_quadrants', []) or [])

                close_tv = df['Close']
                r_tv = _tr_log_returns(close_tv)
                vcr_series = _tr_vcr(r_tv, window_tv)
                tr_series = None
                if metric_name in ('Trend Ratio (TR)', 'Regime quadrant'):
                    if window_tv % sample_freq_tv == 0:
                        rv_d = _tr_rv_daily(r_tv, window_tv)
                        rv_w = _tr_rv_sampled(close_tv, window_tv, sample_freq_tv)
                        tr_series = rv_w / rv_d
                    else:
                        tr_series = pd.Series(np.nan, index=df.index)

                def _rank(s):
                    if rank_window_tv == 'Rolling 5y':
                        return s.rolling(252 * 5, min_periods=min_periods_tv).rank(pct=True)
                    return s.expanding(min_periods=min_periods_tv).rank(pct=True)

                def _raw_cond(s):
                    s_lag = s.shift(1)
                    if   raw_logic_tv == '>':  return s_lag > raw_min_tv
                    elif raw_logic_tv == '>=': return s_lag >= raw_min_tv
                    elif raw_logic_tv == '<':  return s_lag < raw_min_tv
                    elif raw_logic_tv == '<=': return s_lag <= raw_min_tv
                    else:                      return (s_lag >= raw_min_tv) & (s_lag <= raw_max_tv)

                if metric_name == 'Regime quadrant':
                    # Always median-split rank, irrespective of filter mode.
                    tr_rk = _rank(tr_series).shift(1)
                    vcr_rk = _rank(vcr_series).shift(1)
                    high_tr_rk = tr_rk > 0.5
                    high_vcr_rk = vcr_rk > 0.5
                    cond_tv = pd.Series(False, index=df.index)
                    if 'grinding_trend' in quadrants_tv: cond_tv = cond_tv | (high_tr_rk & ~high_vcr_rk)
                    if 'spike_trend' in quadrants_tv:    cond_tv = cond_tv | (high_tr_rk & high_vcr_rk)
                    if 'choppy_grind' in quadrants_tv:   cond_tv = cond_tv | (~high_tr_rk & ~high_vcr_rk)
                    if 'spike_revert' in quadrants_tv:   cond_tv = cond_tv | (~high_tr_rk & high_vcr_rk)
                else:
                    target = vcr_series if metric_name == 'Variance Contribution (VCR)' else tr_series
                    if filter_mode_tv == 'Raw value':
                        cond_tv = _raw_cond(target)
                    else:
                        rk = _rank(target).shift(1)
                        cond_tv = (rk > p_min_tv) & (rk < p_max_tv)

                cond_tv = cond_tv.fillna(False).astype(bool)
                min_consec_tv = int(params.get('tr_vcr_min_consec', 1))
                if min_consec_tv > 1:
                    streak_ok = cond_tv.rolling(min_consec_tv, min_periods=min_consec_tv).sum() >= min_consec_tv
                    streak_ok = streak_ok.fillna(False).astype(bool)
                    if params.get('tr_vcr_consec_first', False):
                        # Fire only on the first bar the streak threshold is hit
                        streak_ok = streak_ok & (~streak_ok.shift(1).fillna(False).astype(bool))
                    cond_tv = streak_ok

                conditions.append(cond_tv.fillna(False))

            # --- REFERENCE TICKER FILTER ---
            if use_ref_ticker_filter and ref_filters:
                for rf in ref_filters:
                    col = f"Ref_rank_ret_{rf['window']}d"
                    if col in df.columns:
                        if rf['logic'] == '<':
                            conditions.append(df[col] < rf['thresh'])
                        elif rf['logic'] == '>':
                            conditions.append(df[col] > rf['thresh'])
                
            if use_t1_open_filter and t1_open_filters:
                for t1_filter in t1_open_filters:
                    ref_col = t1_filter['reference']
                    atr_offset = t1_filter['atr_offset']
                    logic = t1_filter['logic']
                    threshold = df[ref_col] + (atr_offset * df['ATR'])
                    if logic == '>': t1_cond = df['NextOpen'] > threshold
                    elif logic == '>=': t1_cond = df['NextOpen'] >= threshold
                    elif logic == '<': t1_cond = df['NextOpen'] < threshold
                    elif logic == '<=': t1_cond = df['NextOpen'] <= threshold
                    else: continue
                    conditions.append(t1_cond)

            # Risk dial filters — apply fragility score thresholds on signal date
            dial_filters = params.get('dial_filters', [])
            if dial_filters and fragility_df is not None and not fragility_df.empty:
                for df_filter in dial_filters:
                    dial_col = df_filter.get('dial')
                    if dial_col not in fragility_df.columns:
                        continue
                    win = max(1, int(df_filter.get('window', 1)))
                    dial_series = fragility_df[dial_col]
                    if win > 1:
                        dial_series = dial_series.rolling(win, min_periods=win).mean()
                    dial_on_date = dial_series.reindex(df.index, method='ffill')
                    thresh = float(df_filter.get('thresh', 0))
                    logic = df_filter.get('logic', '>')
                    vals = dial_on_date.values
                    if logic == '>':    cond = (vals > thresh)
                    elif logic == '<':  cond = (vals < thresh)
                    elif logic == '>=': cond = (vals >= thresh)
                    elif logic == '<=': cond = (vals <= thresh)
                    else: continue
                    # NaN dial (pre-2016 or missing) → pass through, treat as if
                    # the dial filter doesn't apply for those trades. Lets the
                    # full historical sample contribute to backtest stats while
                    # still gating modern-era trades on the dial criterion.
                    cond = np.where(np.isnan(vals), True, cond)
                    conditions.append(pd.Series(cond, index=df.index))

            if not conditions: continue

            final_signal = conditions[0]
            for c in conditions[1:]: final_signal = final_signal & c
            
            if params.get('perf_first_instance', False) and len(params.get('perf_filters', [])) > 0:
                final_signal = apply_first_instance_filter(final_signal, params.get('perf_lookback', 21))
                
            signal_dates = df.index[final_signal]
            total_signals_generated += len(signal_dates)
            
            # --- EXECUTION LOGIC ---
            for signal_date in signal_dates:
                if max_one_pos_per_ticker and signal_date <= ticker_last_exit: continue
                
                sig_idx = df.index.get_loc(signal_date)
                if sig_idx + 1 >= len(df): continue
                
                found_entry, actual_entry_idx, actual_entry_price = False, -1, 0.0
                
                # ========== ENTRY LOGIC SECTION ==========
                if is_overnight:
                    found_entry, actual_entry_idx, actual_entry_price = True, sig_idx, df['Close'].iloc[sig_idx]
                elif is_intraday:
                    found_entry, actual_entry_idx, actual_entry_price = True, sig_idx + 1, df['Open'].iloc[sig_idx + 1]
                elif is_pullback and pullback_col:
                    for wait_i in range(1, params['holding_days'] + 1):
                        curr_idx = sig_idx + wait_i
                        if curr_idx >= len(df): break
                        if df.iloc[curr_idx]['Low'] <= df.iloc[curr_idx][pullback_col]:
                            found_entry, actual_entry_idx = True, curr_idx
                            actual_entry_price = df.iloc[curr_idx][pullback_col] if pullback_use_level else df.iloc[curr_idx]['Close']
                            break
                elif is_gap_up:
                    if df['Open'].iloc[sig_idx + 1] > df['High'].iloc[sig_idx]:
                        found_entry, actual_entry_idx, actual_entry_price = True, sig_idx + 1, df['Open'].iloc[sig_idx + 1]
                elif is_limit_pers_025 or is_limit_pers_05 or is_limit_pers_10:
                    atr_mult = 0.25 if is_limit_pers_025 else (0.5 if is_limit_pers_05 else 1.0)
                    sig_close, sig_atr = df['Close'].iloc[sig_idx], df['ATR'].iloc[sig_idx]
                    limit_price = (sig_close - (sig_atr * atr_mult)) if direction == 'Long' else (sig_close + (sig_atr * atr_mult))
                    for wait_i in range(1, params['holding_days'] + 1):
                        curr_idx = sig_idx + wait_i
                        if curr_idx >= len(df): break
                        day_low, day_high, day_open = df['Low'].iloc[curr_idx], df['High'].iloc[curr_idx], df['Open'].iloc[curr_idx]
                        filled, fill_px = False, limit_price
                        if direction == 'Long':
                            if day_low <= limit_price: filled = True; fill_px = day_open if day_open < limit_price else limit_price
                        else:
                            if day_high >= limit_price: filled = True; fill_px = day_open if day_open > limit_price else limit_price
                        if filled: found_entry, actual_entry_idx, actual_entry_price = True, curr_idx, fill_px; break
                elif is_limit_atr:
                    sig_close, sig_atr = df['Close'].iloc[sig_idx], df['ATR'].iloc[sig_idx]
                    limit_price = (sig_close - (sig_atr * 0.5)) if direction == 'Long' else (sig_close + (sig_atr * 0.5))
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        day_low, day_high, day_open = df['Low'].iloc[next_idx], df['High'].iloc[next_idx], df['Open'].iloc[next_idx]
                        if direction == 'Long' and day_low <= limit_price:
                            found_entry, actual_entry_idx, actual_entry_price = True, next_idx, min(limit_price, day_open) if day_open < limit_price else limit_price
                        elif direction == 'Short' and day_high >= limit_price:
                            found_entry, actual_entry_idx, actual_entry_price = True, next_idx, max(limit_price, day_open) if day_open > limit_price else limit_price
                elif is_limit_prev:
                    limit_price = df['Close'].iloc[sig_idx]
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        day_low, day_high, day_open = df['Low'].iloc[next_idx], df['High'].iloc[next_idx], df['Open'].iloc[next_idx]
                        if direction == 'Long' and day_low <= limit_price:
                            found_entry, actual_entry_idx, actual_entry_price = True, next_idx, min(limit_price, day_open) if day_open < limit_price else limit_price
                        elif direction == 'Short' and day_high >= limit_price:
                            found_entry, actual_entry_idx, actual_entry_price = True, next_idx, max(limit_price, day_open) if day_open > limit_price else limit_price
                elif is_limit_open_atr_025:
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        sig_atr, day_open = df['ATR'].iloc[sig_idx], df['Open'].iloc[next_idx]
                        day_low, day_high = df['Low'].iloc[next_idx], df['High'].iloc[next_idx]
                        limit_price = (day_open - (sig_atr * 0.25)) if direction == 'Long' else (day_open + (sig_atr * 0.25))
                        if direction == 'Long' and day_low <= limit_price: found_entry, actual_entry_idx, actual_entry_price = True, next_idx, limit_price
                        elif direction == 'Short' and day_high >= limit_price: found_entry, actual_entry_idx, actual_entry_price = True, next_idx, limit_price
                elif is_limit_open_atr:
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        sig_atr, day_open = df['ATR'].iloc[sig_idx], df['Open'].iloc[next_idx]
                        day_low, day_high = df['Low'].iloc[next_idx], df['High'].iloc[next_idx]
                        limit_price = (day_open - (sig_atr * 0.5)) if direction == 'Long' else (day_open + (sig_atr * 0.5))
                        if direction == 'Long' and day_low <= limit_price: found_entry, actual_entry_idx, actual_entry_price = True, next_idx, limit_price
                        elif direction == 'Short' and day_high >= limit_price: found_entry, actual_entry_idx, actual_entry_price = True, next_idx, limit_price
                elif is_limit_open_atr_075:
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        sig_atr, day_open = df['ATR'].iloc[sig_idx], df['Open'].iloc[next_idx]
                        day_low, day_high = df['Low'].iloc[next_idx], df['High'].iloc[next_idx]
                        limit_price = (day_open - (sig_atr * 0.75)) if direction == 'Long' else (day_open + (sig_atr * 0.75))
                        if direction == 'Long' and day_low <= limit_price: found_entry, actual_entry_idx, actual_entry_price = True, next_idx, limit_price
                        elif direction == 'Short' and day_high >= limit_price: found_entry, actual_entry_idx, actual_entry_price = True, next_idx, limit_price
                elif is_limit_open_atr_100:
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        sig_atr, day_open = df['ATR'].iloc[sig_idx], df['Open'].iloc[next_idx]
                        day_low, day_high = df['Low'].iloc[next_idx], df['High'].iloc[next_idx]
                        limit_price = (day_open - sig_atr) if direction == 'Long' else (day_open + sig_atr)
                        if direction == 'Long' and day_low <= limit_price: found_entry, actual_entry_idx, actual_entry_price = True, next_idx, limit_price
                        elif direction == 'Short' and day_high >= limit_price: found_entry, actual_entry_idx, actual_entry_price = True, next_idx, limit_price
                elif is_limit_open_atr_gtc:
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        sig_atr = df['ATR'].iloc[sig_idx]
                        base_price = df['Open'].iloc[next_idx]
                        limit_price = (base_price - (sig_atr * 0.5)) if direction == 'Long' else (base_price + (sig_atr * 0.5))
                        for wait_i in range(1, params['holding_days'] + 1):
                            curr_idx = sig_idx + wait_i
                            if curr_idx >= len(df): break
                            day_low, day_high, day_open = df['Low'].iloc[curr_idx], df['High'].iloc[curr_idx], df['Open'].iloc[curr_idx]
                            filled, fill_px = False, limit_price
                            if direction == 'Long':
                                if day_open < limit_price: filled, fill_px = True, day_open
                                elif day_low <= limit_price: filled, fill_px = True, limit_price
                            else:
                                if day_open > limit_price: filled, fill_px = True, day_open
                                elif day_high >= limit_price: filled, fill_px = True, limit_price
                            if filled: found_entry, actual_entry_idx, actual_entry_price = True, curr_idx, fill_px; break
                elif is_day_trade_limit:
                    # Reset per-signal intraday context; the exit block reads these.
                    _id_bars = None
                    _id_entry_pos = None
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        sig_atr = df['ATR'].iloc[sig_idx]
                        if intraday_mode:
                            if ticker not in intraday_universe:
                                continue
                            next_date = df.index[next_idx]
                            try:
                                _id_bars = intraday_loader.get_intraday_for_date(ticker, next_date)
                            except Exception:
                                _id_bars = None
                            if _id_bars is None or _id_bars.empty:
                                continue
                            day_open = float(_id_bars['open'].iloc[0])
                            limit_price = (day_open - (sig_atr * day_trade_atr_mult)) if direction == 'Long' else (day_open + (sig_atr * day_trade_atr_mult))
                            for _bp in range(len(_id_bars)):
                                bar = _id_bars.iloc[_bp]
                                bar_open, bar_low, bar_high = float(bar['open']), float(bar['low']), float(bar['high'])
                                if direction == 'Long' and bar_low <= limit_price:
                                    fill_px = bar_open if bar_open <= limit_price else limit_price
                                    found_entry, actual_entry_idx, actual_entry_price = True, next_idx, fill_px
                                    _id_entry_pos = _bp; break
                                if direction == 'Short' and bar_high >= limit_price:
                                    fill_px = bar_open if bar_open >= limit_price else limit_price
                                    found_entry, actual_entry_idx, actual_entry_price = True, next_idx, fill_px
                                    _id_entry_pos = _bp; break
                        else:
                            day_open = df['Open'].iloc[next_idx]
                            day_low, day_high = df['Low'].iloc[next_idx], df['High'].iloc[next_idx]
                            limit_price = (day_open - (sig_atr * day_trade_atr_mult)) if direction == 'Long' else (day_open + (sig_atr * day_trade_atr_mult))
                            if direction == 'Long' and day_low <= limit_price:
                                found_entry, actual_entry_idx, actual_entry_price = True, next_idx, limit_price
                            elif direction == 'Short' and day_high >= limit_price:
                                found_entry, actual_entry_idx, actual_entry_price = True, next_idx, limit_price
                elif is_limit_pivot:
                    sig_close = df['Close'].iloc[sig_idx]
                    sig_atr = df['ATR'].iloc[sig_idx]
                    if direction == 'Long' and 'LastPivotLow' in df.columns:
                        limit_price = df['LastPivotLow'].iloc[sig_idx]
                        # Pivot must be below close and at least 0.5 ATR away
                        if pd.notna(limit_price) and limit_price < sig_close and (sig_close - limit_price) >= 0.5 * sig_atr:
                            for wait_i in range(1, params['holding_days'] + 1):
                                curr_idx = sig_idx + wait_i
                                if curr_idx >= len(df): break
                                day_low, day_open = df['Low'].iloc[curr_idx], df['Open'].iloc[curr_idx]
                                if day_open <= limit_price:
                                    found_entry, actual_entry_idx, actual_entry_price = True, curr_idx, day_open
                                    break
                                elif day_low <= limit_price:
                                    found_entry, actual_entry_idx, actual_entry_price = True, curr_idx, limit_price
                                    break
                    elif direction == 'Short' and 'LastPivotHigh' in df.columns:
                        limit_price = df['LastPivotHigh'].iloc[sig_idx]
                        # Pivot must be above close and at least 0.5 ATR away
                        if pd.notna(limit_price) and limit_price > sig_close and (limit_price - sig_close) >= 0.5 * sig_atr:
                            for wait_i in range(1, params['holding_days'] + 1):
                                curr_idx = sig_idx + wait_i
                                if curr_idx >= len(df): break
                                day_high, day_open = df['High'].iloc[curr_idx], df['Open'].iloc[curr_idx]
                                if day_open >= limit_price:
                                    found_entry, actual_entry_idx, actual_entry_price = True, curr_idx, day_open
                                    break
                                elif day_high >= limit_price:
                                    found_entry, actual_entry_idx, actual_entry_price = True, curr_idx, limit_price
                                    break
                elif is_cond_close_lower:
                    atr_mult = 0.5 if "-0.5 ATR" in entry_mode else (1.0 if "-1 ATR" in entry_mode else 0.0)
                    sig_val, sig_atr = df['Close'].iloc[sig_idx], df['ATR'].iloc[sig_idx]
                    limit_price = sig_val - (atr_mult * sig_atr)
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        t1_close = df['Close'].iloc[next_idx]
                        if t1_close < limit_price: found_entry, actual_entry_idx, actual_entry_price = True, next_idx, t1_close
                elif is_cond_close_higher_band:
                    # Band-conditional: T+1 close must be inside [+lower, +upper] ATR
                    # above signal close. Captures "moderate continuation" without
                    # chasing the parabolic gap. Bounds parsed from entry_mode name.
                    sig_val, sig_atr = df['Close'].iloc[sig_idx], df['ATR'].iloc[sig_idx]
                    lower_bound = sig_val + (_band_lower_mult * sig_atr)
                    upper_bound = sig_val + (_band_upper_mult * sig_atr)
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        t1_close = df['Close'].iloc[next_idx]
                        if lower_bound <= t1_close <= upper_bound:
                            found_entry, actual_entry_idx, actual_entry_price = True, next_idx, t1_close
                elif is_cond_close_higher:
                    atr_mult = 0.5 if "+0.5 ATR" in entry_mode else (1.0 if "+1 ATR" in entry_mode else 0.0)
                    sig_val, sig_atr = df['Close'].iloc[sig_idx], df['ATR'].iloc[sig_idx]
                    limit_price = sig_val + (atr_mult * sig_atr)
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        t1_close = df['Close'].iloc[next_idx]
                        if t1_close > limit_price: found_entry, actual_entry_idx, actual_entry_price = True, next_idx, t1_close
                else:
                    found_entry = True
                    actual_entry_idx = sig_idx if entry_mode == 'Signal Close' else sig_idx + 1
                    if actual_entry_idx >= len(df): found_entry = False
                    else: actual_entry_price = df['Close'].iloc[actual_entry_idx] if 'Close' in entry_mode else df['Open'].iloc[actual_entry_idx]

                if not found_entry: continue

                atr = df['ATR'].iloc[actual_entry_idx]
                if np.isnan(atr) or atr == 0: continue
                
                # ========== EXIT LOGIC SECTION ==========
                if is_overnight:
                    exit_idx = sig_idx + 1
                    if exit_idx >= len(df): continue
                    exit_date = df.index[exit_idx]
                    exit_price = df['Open'].iloc[exit_idx]
                    exit_type = "Time"
                elif is_intraday:
                    exit_idx = actual_entry_idx
                    exit_date = df.index[exit_idx]
                    exit_price = df['Close'].iloc[exit_idx]
                    exit_type = "Time"
                elif is_day_trade_limit:
                    exit_idx = actual_entry_idx
                    exit_date = df.index[exit_idx]
                    stop_price = actual_entry_price - (atr * params['stop_atr']) if direction == 'Long' else actual_entry_price + (atr * params['stop_atr'])
                    tgt_price = actual_entry_price + (atr * params['tgt_atr']) if direction == 'Long' else actual_entry_price - (atr * params['tgt_atr'])
                    if intraday_mode and _id_bars is not None and _id_entry_pos is not None:
                        # Entry-bar handling: the trigger bar fired BECAUSE
                        # price moved in the entry direction (long limit pullback
                        # → bar dipped; short limit rally → bar spiked). Post-fill
                        # continuation of that move is realistic, so the entry
                        # bar IS eligible for a stop hit — but NOT for a target
                        # hit (that would require an immediate reversal inside
                        # the same 15min bar, indistinguishable from noise without
                        # tick data).
                        exit_price, exit_type = None, None
                        _entry_bar = _id_bars.iloc[_id_entry_pos]
                        _eb_low, _eb_high = float(_entry_bar['low']), float(_entry_bar['high'])
                        if direction == 'Long':
                            if params['use_stop_loss'] and _eb_low <= stop_price:
                                exit_price, exit_type = stop_price, "Stop"
                        else:
                            if params['use_stop_loss'] and _eb_high >= stop_price:
                                exit_price, exit_type = stop_price, "Stop"

                        # Post-entry walk: 15min bars from the bar AFTER entry
                        # forward. Stop-first within a bar (conservative on
                        # collisions). EOD = last bar's close.
                        if exit_price is None:
                            for _bp in range(_id_entry_pos + 1, len(_id_bars)):
                                bar = _id_bars.iloc[_bp]
                                bar_low, bar_high = float(bar['low']), float(bar['high'])
                                if direction == 'Long':
                                    if params['use_stop_loss'] and bar_low <= stop_price:
                                        exit_price, exit_type = stop_price, "Stop"; break
                                    if params['use_take_profit'] and bar_high >= tgt_price:
                                        exit_price, exit_type = tgt_price, "Target"; break
                                else:
                                    if params['use_stop_loss'] and bar_high >= stop_price:
                                        exit_price, exit_type = stop_price, "Stop"; break
                                    if params['use_take_profit'] and bar_low <= tgt_price:
                                        exit_price, exit_type = tgt_price, "Target"; break
                        if exit_price is None:
                            exit_price = float(_id_bars['close'].iloc[-1])
                            exit_type = "Time (EOD)"
                    else:
                        exit_price = df['Close'].iloc[exit_idx]
                        exit_type = "Time (EOD)"
                        day_low, day_high = df['Low'].iloc[exit_idx], df['High'].iloc[exit_idx]
                        # Stop-first convention (conservative): if the day's range
                        # swept through both stop AND target, assume stop fired
                        # first. Matches the intraday exit path (line ~2245) and
                        # the multi-day-hold exit path (line ~2349). Without
                        # sub-day data we can't know the actual sequence.
                        if direction == 'Long':
                            if params['use_stop_loss'] and day_low <= stop_price: exit_price, exit_type = stop_price, "Stop"
                            elif params['use_take_profit'] and day_high >= tgt_price: exit_price, exit_type = tgt_price, "Target"
                        else:
                            if params['use_stop_loss'] and day_high >= stop_price: exit_price, exit_type = stop_price, "Stop"
                            elif params['use_take_profit'] and day_low <= tgt_price: exit_price, exit_type = tgt_price, "Target"
                else:
                    fixed_exit_idx = min(actual_entry_idx + params['holding_days'], len(df) - 1)
                    future = df.iloc[actual_entry_idx + 1 : fixed_exit_idx + 1]
                    if future.empty: continue

                    # End-of-entry-day drawdown stop: exit at signal/entry-day close if the
                    # trade is underwater by more than the configured ATR threshold. Naturally
                    # a no-op for entry types where actual_entry_price == entry-day close
                    # (Signal Close, T+1 Close, persistent limits filling at close).
                    if params.get('use_eod_dd_exit', False):
                        entry_date = df.index[actual_entry_idx]
                        allowed_eod_dow = params.get('eod_dd_weekdays', [0, 1, 2, 3, 4])
                        eod_dow_ok = (not allowed_eod_dow) or (entry_date.weekday() in allowed_eod_dow)
                        entry_close = df['Close'].iloc[actual_entry_idx]
                        if direction == 'Long':
                            dd_atr = (actual_entry_price - entry_close) / atr
                        else:
                            dd_atr = (entry_close - actual_entry_price) / atr
                        if eod_dow_ok and dd_atr > params.get('eod_dd_atr', 1.0):
                            exit_price = entry_close
                            exit_date = df.index[actual_entry_idx]
                            exit_type = "EOD-DD"
                            ticker_last_exit = exit_date
                            slip = slippage_bps / 10000.0
                            tech_risk = atr * params['stop_atr']
                            if tech_risk <= 0: tech_risk = 0.001
                            pnl = (exit_price*(1-slip) - actual_entry_price*(1+slip)) if direction == 'Long' else (actual_entry_price*(1-slip) - exit_price*(1+slip))
                            all_potential_trades.append({"Ticker": ticker, "SignalDate": signal_date, "EntryDate": df.index[actual_entry_idx], "Direction": direction, "Entry": actual_entry_price, "Exit": exit_price, "ExitDate": exit_date, "Type": exit_type, "R": pnl / tech_risk, "TechRisk": tech_risk, "Age": df['age_years'].iloc[sig_idx], "AvgVol": df['vol_ma'].iloc[sig_idx], "Status": "Valid Signal", "Reason": "Executed"})
                            continue

                    if params.get('use_trailing_stop', False):
                        trail_dist = atr * params.get('trail_atr', 2.0)
                        anchor_mode = params.get('trail_anchor', 'Peak High')
                        if direction == 'Long':
                            anchor = actual_entry_price
                            trail_stop = actual_entry_price - trail_dist
                        else:
                            anchor = actual_entry_price
                            trail_stop = actual_entry_price + trail_dist
                        exit_price, exit_type, exit_date = actual_entry_price, "Hold", None
                        for f_date, f_row in future.iterrows():
                            if direction == 'Long':
                                if f_row['Low'] <= trail_stop:
                                    actual_stop_px = min(f_row['Open'], trail_stop) if f_row['Open'] <= trail_stop else trail_stop
                                    exit_price, exit_type, exit_date = actual_stop_px, "Trail", f_date
                                    break
                                new_anchor = f_row['High'] if anchor_mode == 'Peak High' else f_row['Close']
                                if new_anchor > anchor:
                                    anchor = new_anchor
                                    trail_stop = max(trail_stop, anchor - trail_dist)
                            else:
                                if f_row['High'] >= trail_stop:
                                    actual_stop_px = max(f_row['Open'], trail_stop) if f_row['Open'] >= trail_stop else trail_stop
                                    exit_price, exit_type, exit_date = actual_stop_px, "Trail", f_date
                                    break
                                new_anchor = f_row['Low'] if anchor_mode == 'Peak High' else f_row['Close']
                                if new_anchor < anchor:
                                    anchor = new_anchor
                                    trail_stop = min(trail_stop, anchor + trail_dist)
                        if exit_type == "Hold":
                            exit_price = future['Close'].iloc[-1]
                            exit_date = future.index[-1]
                            exit_type = "Time"

                        ticker_last_exit = exit_date
                        slip = slippage_bps / 10000.0
                        tech_risk = atr * params.get('trail_atr', 2.0)
                        if tech_risk <= 0: tech_risk = 0.001
                        pnl = (exit_price*(1-slip) - actual_entry_price*(1+slip)) if direction == 'Long' else (actual_entry_price*(1-slip) - exit_price*(1+slip))
                        all_potential_trades.append({"Ticker": ticker, "SignalDate": signal_date, "EntryDate": df.index[actual_entry_idx], "Direction": direction, "Entry": actual_entry_price, "Exit": exit_price, "ExitDate": exit_date, "Type": exit_type, "R": pnl / tech_risk, "TechRisk": tech_risk, "Age": df['age_years'].iloc[sig_idx], "AvgVol": df['vol_ma'].iloc[sig_idx], "Status": "Valid Signal", "Reason": "Executed"})
                        continue

                    stop_price = actual_entry_price - (atr * params['stop_atr']) if direction == 'Long' else actual_entry_price + (atr * params['stop_atr'])
                    tgt_price = actual_entry_price + (atr * params['tgt_atr']) if direction == 'Long' else actual_entry_price - (atr * params['tgt_atr'])

                    use_partial = params.get('use_partial_exits', False) and params.get('use_take_profit', False)
                    tgt_frac = float(params.get('partial_target_fraction', 0.5)) if use_partial else 0.0

                    exit_price, exit_type, exit_date = actual_entry_price, "Hold", None
                    target_filled = False
                    target_fill_px = None

                    for f_date, f_row in future.iterrows():
                        if direction == 'Long':
                            # Stop check — closes remaining fraction (or full pos if no partial target hit yet)
                            if params['use_stop_loss'] and f_row['Low'] <= stop_price:
                                actual_stop_px = min(f_row['Open'], stop_price) if f_row['Open'] <= stop_price else stop_price
                                if use_partial and target_filled:
                                    exit_price = tgt_frac * target_fill_px + (1 - tgt_frac) * actual_stop_px
                                    exit_type, exit_date = "Target→Stop", f_date
                                else:
                                    exit_price, exit_type, exit_date = actual_stop_px, "Stop", f_date
                                break
                            # Target check
                            if params['use_take_profit'] and f_row['High'] >= tgt_price:
                                actual_tgt_px = max(f_row['Open'], tgt_price) if f_row['Open'] >= tgt_price else tgt_price
                                if not use_partial:
                                    exit_price, exit_type, exit_date = actual_tgt_px, "Target", f_date
                                    break
                                elif not target_filled:
                                    # Book the target leg; remaining fraction keeps walking forward
                                    target_filled = True
                                    target_fill_px = actual_tgt_px
                        else:
                            # Stop check
                            if params['use_stop_loss'] and f_row['High'] >= stop_price:
                                actual_stop_px = max(f_row['Open'], stop_price) if f_row['Open'] >= stop_price else stop_price
                                if use_partial and target_filled:
                                    exit_price = tgt_frac * target_fill_px + (1 - tgt_frac) * actual_stop_px
                                    exit_type, exit_date = "Target→Stop", f_date
                                else:
                                    exit_price, exit_type, exit_date = actual_stop_px, "Stop", f_date
                                break
                            # Target check
                            if params['use_take_profit'] and f_row['Low'] <= tgt_price:
                                actual_tgt_px = min(f_row['Open'], tgt_price) if f_row['Open'] <= tgt_price else tgt_price
                                if not use_partial:
                                    exit_price, exit_type, exit_date = actual_tgt_px, "Target", f_date
                                    break
                                elif not target_filled:
                                    target_filled = True
                                    target_fill_px = actual_tgt_px

                    # No stop fired — if partial target was booked, blend with time-stop close;
                    # otherwise close full position at time stop.
                    if exit_type == "Hold":
                        time_exit_px = future['Close'].iloc[-1]
                        time_exit_date = future.index[-1]
                        if use_partial and target_filled:
                            exit_price = tgt_frac * target_fill_px + (1 - tgt_frac) * time_exit_px
                            exit_type, exit_date = "Target→Time", time_exit_date
                        else:
                            exit_price, exit_date, exit_type = time_exit_px, time_exit_date, "Time"
                    
                ticker_last_exit = exit_date
                slip = slippage_bps / 10000.0
                tech_risk = atr * params['stop_atr']
                if tech_risk <= 0: tech_risk = 0.001
                pnl = (exit_price*(1-slip) - actual_entry_price*(1+slip)) if direction == 'Long' else (actual_entry_price*(1-slip) - exit_price*(1+slip))
                
                all_potential_trades.append({"Ticker": ticker, "SignalDate": signal_date, "EntryDate": df.index[actual_entry_idx], "Direction": direction, "Entry": actual_entry_price, "Exit": exit_price, "ExitDate": exit_date, "Type": exit_type, "R": pnl / tech_risk, "TechRisk": tech_risk, "Age": df['age_years'].iloc[sig_idx], "AvgVol": df['vol_ma'].iloc[sig_idx], "Status": "Valid Signal", "Reason": "Executed"})
        except Exception: continue

    progress_bar.empty()
    status_text.empty()
    
    if not all_potential_trades: return pd.DataFrame(), pd.DataFrame(), 0
    
    pot_df = pd.DataFrame(all_potential_trades).sort_values(by=["EntryDate", "Ticker"])
    final_log, rejected_log, active_pos, daily_count = [], [], [], {}
    max_total, max_daily = params.get('max_total_positions', 100), params.get('max_daily_entries', 100)
    
    for _, trade in pot_df.iterrows():
        active_pos = [t for t in active_pos if t['ExitDate'] > trade['EntryDate']]
        today_num = daily_count.get(trade['EntryDate'], 0)
        if len(active_pos) >= max_total or today_num >= max_daily:
            trade_copy = trade.copy()
            trade_copy['Status'], trade_copy['Reason'] = "Portfolio Rejected", "Constraints"
            rejected_log.append(trade_copy)
        else:
            final_log.append(trade)
            active_pos.append(trade)
            daily_count[trade['EntryDate']] = today_num + 1
            
    return pd.DataFrame(final_log), pd.DataFrame(rejected_log), total_signals_generated
    
def main():
    st.set_page_config(layout="wide", page_title="Quantitative Backtester")
    st.title("Quantitative Strategy Backtester")

    # Strategy preset loader. When active, the preset's `settings` dict
    # overrides every params field NOT in _USER_ADJUSTABLE_PARAM_KEYS at
    # Run Backtest time. The filter widgets below are visually ignored
    # while a preset is loaded — the banner makes that explicit.
    _active_preset = st.session_state.get('_active_preset')
    with st.expander(
        f"Load Strategy Preset  [{_active_preset['name']}]" if _active_preset
        else "Load Strategy Preset",
        expanded=False,
    ):
        if not _STRATEGY_BOOK:
            st.warning("strategy_config.STRATEGY_BOOK not importable — preset loader disabled.")
        else:
            _preset_names = ['(none)'] + [s['name'] for s in _STRATEGY_BOOK]
            _default_idx = 0
            if _active_preset and _active_preset['name'] in _preset_names:
                _default_idx = _preset_names.index(_active_preset['name'])
            _pick = st.selectbox(
                "Strategy",
                _preset_names,
                index=_default_idx,
                key='_preset_pick',
                help=(
                    "Loading a preset overrides every filter/condition at Run Backtest time. "
                    "Universe, dates, entry/exit, and risk inputs remain user-adjustable."
                ),
            )
            pc1, pc2, _ = st.columns([1, 1, 4])
            with pc1:
                if st.button("Load Preset", disabled=(_pick == '(none)')):
                    chosen = next((s for s in _STRATEGY_BOOK if s['name'] == _pick), None)
                    if chosen:
                        st.session_state['_active_preset'] = {
                            'name': chosen['name'],
                            'settings': dict(chosen.get('settings', {})),
                        }
                        st.rerun()
            with pc2:
                if st.button("Clear Preset", disabled=(_active_preset is None)):
                    st.session_state.pop('_active_preset', None)
                    st.rerun()
            if _active_preset:
                st.success(
                    f"Active preset: **{_active_preset['name']}** — filter widgets below "
                    "are overridden at Run Backtest time. Universe / dates / entry / "
                    "exit / risk inputs still apply."
                )

    st.markdown("---")
    st.subheader("1. Universe & Data")
    col_u1, col_u2, col_u3 = st.columns([1, 1, 2])
    sample_pct = 100; use_full_history = False
    with col_u1: univ_choice = st.selectbox("Choose Universe", ["All CSV Tickers", "All CSV + Overflow Extras", "All CSV + Overflow Extras (no ^ indices)", "Overflow (dynamic)", "OOS", "Sector ETFs","SPX", "Indices", "Indices (Spot — ^GSPC/^NDX)", "International ETFs", "Sector + Index ETFs", "Commodity + Sector + Index + 3x Lev", "All CSV (Equities Only)", "3x Leveraged (All)", "3x Leveraged Equities", "3x Leveraged Equities (Bull)", "3x Leveraged Equities (Bear)", "3x Leveraged Equities (Broad Only)", "Custom (Comma-Separated)", "Custom (Upload CSV)"])
    with col_u2:
        default_start = datetime.date(2000, 1, 1)
        start_date = st.date_input("Backtest Start Date", value=default_start, min_value=datetime.date(1950, 1, 1), max_value=datetime.date.today())
        end_date = st.date_input("Backtest End Date", value=datetime.date.today(), min_value=start_date, max_value=datetime.date.today(),
                                 help="Truncate the backtest at this date. Useful for out-of-sample testing — set start to e.g. 2000 and end to 2020 to leave 2020-present untouched as a holdout.")
    custom_tickers = []
    extras_tickers = []
    if univ_choice == "Custom (Upload CSV)":
        with col_u3:
            sample_pct = st.slider("Random Sample %", 1, 100, 100)
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file:
                try:
                    c_df = pd.read_csv(uploaded_file)
                    if "Ticker" in c_df.columns:
                        c_df["Ticker"] = c_df["Ticker"].astype(str).str.strip().str.upper()
                        c_df = c_df[~c_df["Ticker"].isin(["NAN", "NONE", "NULL", ""])]
                        custom_tickers = c_df["Ticker"].unique().tolist()
                        if len(custom_tickers) > 0: st.success(f"Loaded {len(custom_tickers)} valid tickers.")
                except: st.error("Invalid CSV.")
    elif univ_choice == "Custom (Comma-Separated)":
        with col_u3:
            ticker_text = st.text_area(
                "Tickers (comma, space, or newline separated)",
                value="",
                height=80,
                key="custom_csl_tickers",
                placeholder="AAPL, MSFT, NVDA, GOOGL",
                help="Paste any ticker list. Separators: commas, spaces, tabs, or newlines.",
            )
            if ticker_text.strip():
                raw = re.split(r"[,\s]+", ticker_text)
                seen = set()
                custom_tickers = []
                for t in raw:
                    t = t.strip().upper()
                    if t and t not in {"NAN", "NONE", "NULL"} and t not in seen:
                        seen.add(t)
                        custom_tickers.append(t)
                if custom_tickers:
                    st.success(f"Parsed {len(custom_tickers)} unique tickers.")
    elif univ_choice in ("All CSV + Overflow Extras", "All CSV + Overflow Extras (no ^ indices)"):
        with col_u3:
            ex_c1, ex_c2 = st.columns([2, 1])
            with ex_c1:
                n_extras = st.number_input(
                    "# Random extras from overflow pool", min_value=0, max_value=2000,
                    value=300, step=50, key="n_overflow_extras",
                    help="Randomly drawn from sznl_ranks.csv (overflow universe), excluding tickers already in the All CSV base. Sample persists across backtests in this session so the cache stays warm — click Reshuffle to regenerate.",
                )
            with ex_c2:
                reshuffle = st.button("🎲 Reshuffle", help="Generate a new random sample from the overflow pool")

            # Build / refresh the sample. Cached in session_state so repeated backtests
            # hit the same ticker list (and thus the same @st.cache_data key).
            _regen = (
                reshuffle
                or 'overflow_extras_sample' not in st.session_state
                or st.session_state.get('_overflow_n') != n_extras
            )
            if _regen:
                try:
                    _of_df = pd.read_csv("sznl_ranks.csv")
                    _of_pool = [str(t).strip().upper() for t in _of_df['ticker'].dropna().unique().tolist()]
                    _base_keys = set(load_seasonal_map().keys())
                    _of_pool = [t for t in _of_pool if t and t not in _base_keys and t not in {"NAN", "NONE", "NULL"}]
                    if n_extras == 0 or n_extras >= len(_of_pool):
                        st.session_state.overflow_extras_sample = sorted(_of_pool)
                    else:
                        st.session_state.overflow_extras_sample = sorted(random.sample(_of_pool, n_extras))
                    st.session_state._overflow_n = n_extras
                except Exception as e:
                    st.error(f"Couldn't read sznl_ranks.csv: {e}")
                    st.session_state.overflow_extras_sample = []
            extras_tickers = st.session_state.get('overflow_extras_sample', [])
            if extras_tickers:
                st.success(f"Sample: {len(extras_tickers)} extras (pinned for this session — reshuffle resets)")
            run_scope = st.radio(
                "Run scope (subset — doesn't affect cache)",
                ["All (base + extras)", "Base only (All CSV)", "Extras only (overflow)"],
                horizontal=True,
                key="overflow_run_scope",
                help="Both base and extras are still downloaded so their caches stay warm. Scope only filters which tickers the strategy is tested on.",
            )
    st.write("")
    use_full_history = st.checkbox("Download Full History (1950+) for Accurate 'Age'", value=False)
    use_master_parquet = st.checkbox(
        "📦 Use master parquet (opt-in) — read OHLCV from data/master_prices.parquet instead of yfinance",
        value=False,
        help=(
            "Off by default: legacy yfinance behavior is preserved exactly. "
            "When checked, reads from data/master_prices.parquet (built via "
            "scripts/build_master_prices.py, refreshed via scripts/update_master_prices.py). "
            "Falls back to yfinance per-ticker if the master file is missing."
        ),
    )
    include_dynamic_overflow = st.checkbox(
        "🌊 Also run the comprehensive Overflow universe alongside the selected universe",
        value=False,
        help=(
            "Unions the comprehensive overflow universe (dynamic screen ∪ legacy static tier, "
            "~1,350 names) on top of whatever 'Choose Universe' is set to, so an older list can be "
            "backtested together with the overflow names in one run. The overflow names are always "
            "priced from master_prices + overflow_prices (some live only in overflow_prices); the "
            "base universe keeps its normal source. No effect when the universe is already "
            "'Overflow (dynamic)'."
        ),
    )
    st.markdown("---")
    st.subheader("2. Execution & Risk")
    r_c1, r_c2, r_c3 = st.columns(3)
    with r_c1: trade_direction = st.selectbox("Trade Direction", ["Long", "Short"])
    with r_c2:
        exit_mode = st.selectbox("Exit Mode", ["Time Only (Hold)", "Standard (Stop & Target)", "No Stop (Target + Time)", "Trailing Stop (ATR)"])
        use_trailing_stop = (exit_mode == "Trailing Stop (ATR)")
        use_stop_loss = (exit_mode == "Standard (Stop & Target)")
        use_take_profit = (exit_mode in ("Standard (Stop & Target)", "No Stop (Target + Time)"))
        time_exit_only = (exit_mode == "Time Only (Hold)")
    with r_c3: max_one_pos = st.checkbox("Max 1 Position/Ticker", value=True)
    p_c1, p_c2, p_c3 = st.columns(3)
    with p_c1: max_daily_entries = st.number_input("Max New Trades Per Day", 1, 100, 20)
    with p_c2: max_total_positions = st.number_input("Max Total Positions", 1, 200, 99)
    with p_c3: slippage_bps = st.number_input("Slippage (bps)", value=2)
    c_re, c_conf = st.columns(2)
    with c_re: allow_same_day_reentry = st.checkbox("Allow Same-Day Re-entry", value=False)
    with c_conf: entry_conf_bps = st.number_input("Entry Confirmation (bps)", value=0)
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: 
        entry_type = st.selectbox("Entry Price", [
            "Limit (Open +/- 0.25 ATR)",
            "Limit (Open +/- 0.5 ATR)",
            "Limit (Open +/- 0.75 ATR)",
            "Limit (Open +/- 1 ATR)",
            "Signal Close", "T+1 Open", "T+1 Close",
            "Overnight (Buy Close, Sell T+1 Open)", "Intraday (Buy Open, Sell Close)",
            "Day Trade (Limit Open +/- 0.5 ATR, Exit Close)",
            "Day Trade (Limit Open +/- 0.75 ATR, Exit Close)",
            "Day Trade (Limit Open +/- 1 ATR, Exit Close)",
            "Gap Up Only (Open > Prev High)",
            "Limit Order -0.25 ATR (Persistent)", "Limit Order -0.5 ATR (Persistent)", "Limit Order -1 ATR (Persistent)",
            "Limit (Close -0.5 ATR)", "Limit (Prev Close)",
            "Limit (Open +/- 0.5 ATR) GTC",
            "Limit (Untested Pivot)", 
            "Pullback 10 SMA (Entry: Close)", "Pullback 10 SMA (Entry: Level)", 
            "Pullback 21 EMA (Entry: Close)", "Pullback 21 EMA (Entry: Level)", 
            "T+1 Close if < Signal Close", "T+1 Close if < Signal Close -0.5 ATR",
            "T+1 Close if < Signal Close -1 ATR", "T+1 Close if > Signal Close",
            "T+1 Close if > Signal Close +0.5 ATR", "T+1 Close if > Signal Close +1 ATR",
            "T+1 Close if > Signal Close +0.5 to +1 ATR",
            "T+1 Close if > Signal Close +0.75 to +1 ATR"
        ])
        use_ma_entry_filter = st.checkbox("Filter: Close > MA - 0.25*ATR", value=False) if "Pullback" in entry_type else False
        # Intraday Day Trade option — only meaningful for Day Trade entry modes.
        # When enabled, uses 15min bars from data/intraday/ (R2 cache) to find
        # the actual fill bar/price and walk stops/targets bar-by-bar instead of
        # approximating with daily H/L. Available for ~31 ETFs and a few liquids.
        is_day_trade_entry = "Day Trade" in entry_type
        try:
            import intraday_data as _id_loader
            _intraday_universe = _id_loader.available_tickers()
        except Exception:
            _intraday_universe = set()
        use_intraday = st.checkbox(
            f"Use Intraday 15min for Day Trade ({len(_intraday_universe)} tickers available)",
            value=False, disabled=not is_day_trade_entry,
            help="When entry is Day Trade, walks 15min bars to find precise fill, "
                 "then bar-by-bar stop/target. Tickers without intraday data are skipped silently.",
        ) if is_day_trade_entry else False
    with c2: stop_atr = st.number_input("Stop Loss (ATR)", value=1.0, step=0.1, disabled=not use_stop_loss)
    with c3: tgt_atr = st.number_input("Target (ATR)", value=8.0, step=0.1, disabled=not use_take_profit)
    if use_trailing_stop:
        ts_c1, ts_c2 = st.columns([1, 2])
        with ts_c1:
            trail_atr = st.number_input(
                "Trail Distance (ATR)", value=2.0, step=0.1, min_value=0.1,
                help="Trailing stop distance in ATR units. ATR is frozen at signal-day value.",
            )
        with ts_c2:
            trail_anchor = st.selectbox(
                "Trail Anchor",
                ["Peak High", "Close"],
                help=(
                    "Peak High: stop ratchets to running max bar high - N*ATR (Long) / "
                    "running min low + N*ATR (Short). Close: anchored to running max/min close instead — "
                    "smoother, lets pullback wicks ride."
                ),
            )
    else:
        trail_atr = 2.0
        trail_anchor = "Peak High"
    with c4: hold_days = st.number_input("Max Holding Days", min_value=1, value=10, step=1)
    with c5:
        starting_portfolio = st.number_input("Starting Portfolio ($)", value=100000, step=1000, min_value=100)
        risk_bps_input = st.number_input("Risk per Trade (bps)", value=25, step=5, min_value=1, max_value=500,
                                         help="Basis points of starting portfolio risked per trade. 25 bps on $100k = $250/trade.")
        _is_overflow_universe = univ_choice in (
            "All CSV + Overflow Extras",
            "All CSV + Overflow Extras (no ^ indices)",
        )
        overflow_bps_input = st.number_input(
            "Overflow Ticker Risk (bps)",
            value=int(risk_bps_input), step=5, min_value=1, max_value=500,
            disabled=not _is_overflow_universe,
            help=(
                "Per-trade risk (bps) applied to overflow extras tickers — "
                "i.e. tickers in the random sample drawn from sznl_ranks that "
                "are NOT in the All CSV base. Liquid base tickers still size "
                "off 'Risk per Trade'. Only relevant when 'All CSV + Overflow "
                "Extras' is selected. Set equal to primary risk for uniform sizing."
            ),
        )
        use_max_daily_risk = st.checkbox(
            "Cap Total Daily Risk", value=False,
            help="If a day's signals would risk more than the cap in aggregate, scale every trade that day down pro-rata so total risk = cap. No effect on days under the cap.",
        )
        max_daily_risk_pct = st.number_input(
            "Max Daily Risk (% of equity)", value=3.0, step=0.25, min_value=0.05, max_value=100.0,
            disabled=not use_max_daily_risk,
            help="E.g. 3.0%. With 25bps/trade, 12 trades fit under 3%; a 25-signal day would scale each down by 0.48x.",
        )
    risk_per_trade = starting_portfolio * risk_bps_input / 10000.0

    # --- Partial exits: scale out at target, let remainder ride ---
    with st.expander("Partial Exits (scale out at target)", expanded=False):
        st.markdown(
            "Take profit on a fraction of the position when the target prints, "
            "then let the remainder ride to the time stop. If the stop triggers "
            "before the remaining fraction exits, the remainder exits at the stop."
        )
        pe_c1, pe_c2 = st.columns([1, 2])
        with pe_c1:
            use_partial_exits = st.checkbox(
                "Enable partial profit-taking",
                value=False,
                disabled=not use_take_profit,
            )
            if use_partial_exits and not use_take_profit:
                st.caption("⚠ Requires a profit target (change Exit Mode).")
        with pe_c2:
            partial_target_fraction = st.slider(
                "Fraction exiting at target",
                min_value=0.10, max_value=0.90, value=0.50, step=0.05,
                disabled=not (use_partial_exits and use_take_profit),
                help="The remaining fraction exits at the time stop or, if it fires first, the stop.",
            )

    # --- End-of-entry-day drawdown stop ---
    with st.expander("End-of-Entry-Day Drawdown Exit", expanded=False):
        st.markdown(
            "Cuts a position at the close of its entry day if it's already underwater "
            "by more than the configured ATR threshold (signal-day ATR). Useful for "
            "killing trades that immediately go offsides. No effect on entry types that "
            "fill at the close (Signal Close, T+1 Close, persistent limits) — drawdown "
            "is zero by construction."
        )
        eod_c1, eod_c2, eod_c3 = st.columns([1, 1, 2])
        with eod_c1:
            use_eod_dd_exit = st.checkbox("Enable EOD drawdown exit", value=False)
        with eod_c2:
            eod_dd_atr = st.number_input(
                "Min Loss to Exit (ATR)",
                value=1.0, step=0.1, min_value=0.1,
                disabled=not use_eod_dd_exit,
                help="If entry-day close shows a loss greater than this many ATRs, exit at that close. Tagged 'EOD-DD' in the trade log.",
            )
        with eod_c3:
            _eod_dow_options = [("Mon", 0), ("Tue", 1), ("Wed", 2), ("Thu", 3), ("Fri", 4)]
            _eod_dow_labels = [lbl for lbl, _ in _eod_dow_options]
            _eod_label_to_val = dict(_eod_dow_options)
            eod_dd_dow_selected = st.multiselect(
                "Apply on weekdays",
                options=_eod_dow_labels,
                default=_eod_dow_labels,
                disabled=not use_eod_dd_exit,
                help="Restrict EOD-DD exit to signals whose entry day falls on these weekdays. Default = all weekdays (no restriction).",
            )
            eod_dd_weekdays = sorted({_eod_label_to_val[lbl] for lbl in eod_dd_dow_selected})
    st.markdown("---")
    st.subheader("3. Signal Criteria")
    with st.expander("Liquidity & Data History Filters", expanded=True):
        l1, l2, l3, l4, l5, l6, l7 = st.columns(7)
        with l1: min_price = st.number_input("Min Price ($)", value=10.0, step=1.0)
        with l2: min_vol = st.number_input("Min Avg Volume", value=100000, step=50000)
        with l3: max_vol = st.number_input("Max Avg Volume", value=0, step=1_000_000,
                                            help="0 = no upper bound. Useful for capping mega-cap names that dominate liquid screens.")
        with l4: min_age = st.number_input("Min True Age (Yrs)", value=0.25, step=0.25)
        with l5: max_age = st.number_input("Max True Age (Yrs)", value=100.0, step=1.0)
        with l6: min_atr_pct = st.number_input("Min ATR %", value=0.2, step=0.1)
        with l7: max_atr_pct = st.number_input("Max ATR %", value=10.0, step=0.1)
    with st.expander("T+1 Open Filter (Next Day Open vs Today's OHLC)", expanded=False):
        st.markdown("**Filter signals based on how the next day opens relative to today's price action.**")
        use_t1_open_filter = st.checkbox("Enable T+1 Open Filter", value=False)
        t1_open_filters = []
        if use_t1_open_filter:
            st.markdown("---\n**Configure up to 3 T+1 Open conditions (AND logic)**")
            t1_c1, t1_c2, t1_c3 = st.columns(3)
            with t1_c1:
                st.markdown("**Condition 1**")
                use_t1_1 = st.checkbox("Enable", key="use_t1_1", value=True)
                t1_logic_1 = st.selectbox("T+1 Open is", [">", ">=", "<", "<="], key="t1_logic_1", disabled=not use_t1_1)
                t1_ref_1 = st.selectbox("Today's", ["High", "Low", "Open", "Close"], key="t1_ref_1", disabled=not use_t1_1)
                t1_atr_1 = st.number_input("ATR Offset (+/-)", value=0.5, step=0.25, min_value=-5.0, max_value=5.0, key="t1_atr_1", disabled=not use_t1_1)
                if use_t1_1: t1_open_filters.append({'logic': t1_logic_1, 'reference': t1_ref_1, 'atr_offset': t1_atr_1})
            with t1_c2:
                st.markdown("**Condition 2**")
                use_t1_2 = st.checkbox("Enable", key="use_t1_2", value=False)
                t1_logic_2 = st.selectbox("T+1 Open is", [">", ">=", "<", "<="], key="t1_logic_2", disabled=not use_t1_2)
                t1_ref_2 = st.selectbox("Today's", ["High", "Low", "Open", "Close"], key="t1_ref_2", index=1, disabled=not use_t1_2)
                t1_atr_2 = st.number_input("ATR Offset (+/-)", value=0.0, step=0.25, min_value=-5.0, max_value=5.0, key="t1_atr_2", disabled=not use_t1_2)
                if use_t1_2: t1_open_filters.append({'logic': t1_logic_2, 'reference': t1_ref_2, 'atr_offset': t1_atr_2})
            with t1_c3:
                st.markdown("**Condition 3**")
                use_t1_3 = st.checkbox("Enable", key="use_t1_3", value=False)
                t1_logic_3 = st.selectbox("T+1 Open is", [">", ">=", "<", "<="], key="t1_logic_3", disabled=not use_t1_3)
                t1_ref_3 = st.selectbox("Today's", ["High", "Low", "Open", "Close"], key="t1_ref_3", index=3, disabled=not use_t1_3)
                t1_atr_3 = st.number_input("ATR Offset (+/-)", value=0.0, step=0.25, min_value=-5.0, max_value=5.0, key="t1_atr_3", disabled=not use_t1_3)
                if use_t1_3: t1_open_filters.append({'logic': t1_logic_3, 'reference': t1_ref_3, 'atr_offset': t1_atr_3})
    with st.expander("Accumulation/Distribution Counts", expanded=False):
        st.markdown("**Filters are additive (AND logic).**")
        acc_dist_version = st.radio("Signal Version", ["v1 (Vol > Prev & > Avg)", "v2 (Vol > 1.25x Prev & > Avg)"], horizontal=True, key="acc_dist_ver")
        use_acc_dist_v2 = acc_dist_version.startswith("v2")
        c_acc, c_dist = st.columns(2)
        with c_acc:
            st.markdown("#### Accumulation Filter")
            use_acc_count_filter = st.checkbox("Enable Acc Count", value=False)
            acc_count_window = st.selectbox("Acc Window", [5, 10, 21, 42], index=2, disabled=not use_acc_count_filter)
            acc_count_logic = st.selectbox("Acc Logic", [">", "<", "="], disabled=not use_acc_count_filter)
            acc_count_thresh = st.number_input("Acc Threshold", 0, 42, 3, disabled=not use_acc_count_filter)
        with c_dist:
            st.markdown("#### Distribution Filter")
            use_dist_count_filter = st.checkbox("Enable Dist Count", value=False)
            dist_count_window = st.selectbox("Dist Window", [5, 10, 21, 42], index=2, disabled=not use_dist_count_filter)
            dist_count_logic = st.selectbox("Dist Logic", [">", "<", "="], disabled=not use_dist_count_filter)
            dist_count_thresh = st.number_input("Dist Threshold", 0, 42, 3, disabled=not use_dist_count_filter)
    with st.expander("Gap Filter", expanded=False):
        use_gap_filter = st.checkbox("Enable Open Gap Filter", value=False)
        g1, g2, g3 = st.columns(3)
        with g1: gap_lookback = st.number_input("Lookback Window (Days)", 1, 252, 21, disabled=not use_gap_filter)
        with g2: gap_logic = st.selectbox("Gap Count Logic", [">", "<", "="], disabled=not use_gap_filter)
        with g3: gap_thresh = st.number_input("Count Threshold", 0, 50, 3, disabled=not use_gap_filter)
    with st.expander("Earnings Proximity Filter", expanded=False):
        st.caption(
            "Filter signals by their **trading-day offset** to the nearest earnings "
            "announcement. Convention: `offset = signal_date - earnings_date` "
            "in trading days. **Negative = before earnings, 0 = day-of, "
            "positive = after.** Operator behaves like the perf-rank filters: "
            "`= -2` keeps only T-2 entries, `> 2` keeps signals more than 2 "
            "trading days past earnings, `Between -2 and 2` keeps the full "
            "±2 window, `Not Between 0 and 2` skips the 'today + last 2 days' "
            "blackout. Source: data/earnings_calendar.parquet (FMP)."
        )
        use_earnings_filter = st.checkbox("Enable Earnings Proximity Filter", value=False)
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            earnings_logic = st.selectbox(
                "Logic", ["=", "<", ">", "Between", "Not Between"],
                index=3,
                disabled=not use_earnings_filter,
                key="earnings_logic_op",
            )
        if earnings_logic in ("Between", "Not Between"):
            with ec2:
                earnings_min = st.number_input(
                    "Min Offset (trading days)", -252, 252, -2, step=1,
                    disabled=not use_earnings_filter,
                    help="Negative = before earnings.",
                )
            with ec3:
                earnings_max = st.number_input(
                    "Max Offset (trading days)", -252, 252, 2, step=1,
                    disabled=not use_earnings_filter,
                    help="Positive = after earnings.",
                )
            earnings_value = 0  # unused
        else:
            with ec2:
                earnings_value = st.number_input(
                    "Offset (trading days)", -252, 252, 0, step=1,
                    disabled=not use_earnings_filter,
                    help="Negative = before earnings, 0 = day-of, positive = after.",
                )
            earnings_min, earnings_max = 0, 0  # unused
    with st.expander("Earnings Quality + Analyst Grades Filters", expanded=False):
        st.caption(
            "Filter signals by the **last reported** earnings (most recent announcement "
            "at or before the signal date) and by trailing analyst grade activity. All "
            "default off. Sources: `data/earnings_calendar.parquet` (derived "
            "`eps_surprise_pct`, `rev_surprise_pct`, `eps_yoy`, `rev_yoy`) and "
            "`data/analyst_grades.parquet` (FMP `/stable/grades`). Thresholds are "
            "fractions: `0.05` = +5%."
        )

        def _quality_filter_row(label, key_prefix, default_min=0.0, default_max=1.0,
                                min_bound=-2.0, max_bound=10.0, step=0.01):
            use = st.checkbox(f"Filter by Last {label}", value=False, key=f"{key_prefix}_chk")
            c1, c2, c3 = st.columns(3)
            with c1:
                logic = st.selectbox(
                    "Logic", [">", "<", "Between"], index=0,
                    key=f"{key_prefix}_logic", disabled=not use,
                )
            with c2:
                lo = st.number_input(
                    "Threshold / Min", min_bound, max_bound, default_min, step=step,
                    key=f"{key_prefix}_lo", disabled=not use,
                )
            with c3:
                hi = st.number_input(
                    "Max (Between only)", min_bound, max_bound, default_max, step=step,
                    key=f"{key_prefix}_hi",
                    disabled=not use or logic != "Between",
                )
            return use, logic, lo, hi

        use_eps_surp_filter, eps_surp_logic, eps_surp_min, eps_surp_max = \
            _quality_filter_row("EPS Surprise %", "eps_surp", 0.05, 1.0)
        use_rev_surp_filter, rev_surp_logic, rev_surp_min, rev_surp_max = \
            _quality_filter_row("Revenue Surprise %", "rev_surp", 0.02, 0.50)
        use_eps_yoy_filter, eps_yoy_logic, eps_yoy_min, eps_yoy_max = \
            _quality_filter_row("EPS YoY Growth", "eps_yoy", 0.10, 5.0)
        use_rev_yoy_filter, rev_yoy_logic, rev_yoy_min, rev_yoy_max = \
            _quality_filter_row("Revenue YoY Growth", "rev_yoy", 0.10, 5.0)

        st.markdown("---")
        st.caption(
            "**Trailing Analyst Net Upgrades** — counts upgrades minus downgrades "
            "in the lookback window ending on the signal date (calendar days). "
            "Maintains are ignored. Tickers without analyst coverage (FX, futures, "
            "crypto) silently fail this filter when enabled."
        )
        use_grades_filter = st.checkbox(
            "Filter by Trailing Analyst Net Upgrades", value=False, key="grades_chk",
        )
        gr1, gr2, gr3 = st.columns(3)
        with gr1:
            grades_window_days = st.number_input(
                "Lookback (calendar days)", 7, 365, 30, step=1,
                key="grades_window", disabled=not use_grades_filter,
            )
        with gr2:
            grades_logic = st.selectbox(
                "Logic", [">=", "<=", ">", "<", "="], index=0,
                key="grades_logic_op", disabled=not use_grades_filter,
            )
        with gr3:
            grades_thresh = st.number_input(
                "Net Upgrades Threshold", -20, 20, 1, step=1,
                key="grades_thresh", disabled=not use_grades_filter,
                help="net = upgrades - downgrades over the lookback.",
            )
    with st.expander("Distance from MA Filter", expanded=False):
        use_ma_dist_filter = st.checkbox("Enable Distance Filter", value=False)
        d1, d2, d3, d4 = st.columns(4)
        with d1: dist_ma_type = st.selectbox("Select MA", ["SMA 10", "SMA 20", "SMA 50", "SMA 100", "SMA 200", "EMA 8", "EMA 11", "EMA 21","52-Week High", "All-Time High"], disabled=not use_ma_dist_filter)
        with d2: dist_logic = st.selectbox("Logic", ["Greater Than (>)", "Less Than (<)", "Between"], disabled=not use_ma_dist_filter)
        with d3: dist_min = st.number_input("Min ATR Dist", -50.0, 50.0, 0.0, step=0.5, disabled=not use_ma_dist_filter)
        with d4: dist_max = st.number_input("Max ATR Dist", -50.0, 50.0, 2.0, step=0.5, disabled=not use_ma_dist_filter)
    with st.expander("Weekly MA Pullback", expanded=False):
        use_weekly_ma_pullback = st.checkbox("Enable Weekly MA Pullback Filter", value=False)
        st.caption("Buy pullbacks to a weekly MA after the stock was significantly extended above it.")
        wma_c1, wma_c2, wma_c3, wma_c4 = st.columns(4)
        with wma_c1: wma_type = st.selectbox("MA Type", ["EMA", "SMA"], key="wma_type", disabled=not use_weekly_ma_pullback)
        with wma_c2: wma_period = st.number_input("MA Period", 2, 100, 8, key="wma_period", disabled=not use_weekly_ma_pullback)
        with wma_c3: wma_min_ext_pct = st.number_input("Min Extension %", 1.0, 200.0, 30.0, step=5.0, key="wma_min_ext", disabled=not use_weekly_ma_pullback, help="Stock must have been at least this % above the weekly MA at some point in the lookback window")
        with wma_c4: wma_lookback_months = st.number_input("Lookback Months", 1, 24, 6, key="wma_lookback_mo", disabled=not use_weekly_ma_pullback, help="How far back to check for the extension")
        wma_touch_logic = st.selectbox("Touch Logic", ["Low <= MA", "Close <= MA"], key="wma_touch", disabled=not use_weekly_ma_pullback, help="How to define 'touching' the weekly MA on the signal day")
    with st.expander("Vol/Return Delta Ratio", expanded=False):
        use_volret_delta = st.checkbox("Enable Vol/Return Delta Ratio Filter", value=False, key="vrd_enable")
        st.caption("Fires when vol is expanding abnormally vs return change, ranked vs own history. Pct band selects which slice of the rank distribution counts as a signal.")
        vrd_c1, vrd_c2, vrd_c3, vrd_c4 = st.columns(4)
        with vrd_c1: vrd_vol_halflife = st.number_input("Vol EWM Halflife (d)", 2, 252, 20, key="vrd_hl", disabled=not use_volret_delta, help="Halflife for the EWM stdev of daily returns (annualized)")
        with vrd_c2: vrd_ret_horizon = st.number_input("Return Horizon (d)", 1, 252, 20, key="vrd_rh", disabled=not use_volret_delta, help="Window for pct_change return")
        with vrd_c3: vrd_delta_n = st.number_input("Delta Lookback N (d)", 1, 60, 5, key="vrd_dn", disabled=not use_volret_delta, help="N for d_vol = vol_t.diff(N) and d_ret = ret_t.diff(N)")
        with vrd_c4: vrd_min_periods = st.number_input("Rank Min Periods", 50, 2520, 252, step=21, key="vrd_mp", disabled=not use_volret_delta, help="Burn-in before rank/pctile is computed (~1y default)")
        vrd_c5, vrd_c6, vrd_c7, vrd_c8 = st.columns(4)
        with vrd_c5: vrd_method = st.selectbox("Metric", ["Z-score diff", "d_vol / |d_ret|", "Pct-change diff", "Vol delta only"], key="vrd_method", disabled=not use_volret_delta, help="Z-score diff (default): z(d_vol)-z(d_ret), most stable. d_vol/|d_ret|: classic ratio with sign preserved. Pct-change diff: (d_vol/vol)-(d_ret/ret), dimensionless. Vol delta only: rank d_vol against own history, ignore returns — high band = expanding vol, low band = contracting vol")
        with vrd_c6: vrd_rank_window = st.selectbox("Rank Window", ["Expanding", "Rolling 5y"], key="vrd_rw", disabled=not use_volret_delta, help="Expanding uses full history; Rolling 5y is more responsive to regime shifts")
        with vrd_c7: vrd_pctile_min = st.number_input("Pctile Min", 0.0, 100.0, 70.0, step=5.0, key="vrd_pmin", disabled=not use_volret_delta)
        with vrd_c8: vrd_pctile_max = st.number_input("Pctile Max", 0.0, 100.0, 90.0, step=5.0, key="vrd_pmax", disabled=not use_volret_delta)
    with st.expander("Trend Ratio / VCR (Abdelmessih)", expanded=False):
        use_tr_vcr_filter = st.checkbox("Enable TR/VCR Filter", value=False, key="trvcr_enable")
        st.caption("TR = weekly-RV / daily-RV (>1 = trending). VCR = max(r^2)/sum(r^2) (high = single-day spike). Filter by raw value (asset-natural levels) or by percentile rank (vs own history). Regime quadrant uses median-split rank regardless of mode.")
        tv_c1, tv_c2, tv_c3, tv_c4 = st.columns(4)
        with tv_c1: tr_vcr_metric = st.selectbox("Metric", ["Trend Ratio (TR)", "Variance Contribution (VCR)", "Regime quadrant"], key="trvcr_metric", disabled=not use_tr_vcr_filter)
        with tv_c2: tr_vcr_window = st.number_input("Window (d)", 5, 252, 20, step=1, key="trvcr_win", disabled=not use_tr_vcr_filter, help="Rolling window for the RV/VCR estimator (Kris uses 20d).")
        with tv_c3: tr_vcr_sample_freq = st.number_input("Sample Freq (d)", 1, 21, 5, step=1, key="trvcr_sf", disabled=not use_tr_vcr_filter, help="Block size for weekly-sampled RV (TR/regime only). Window must be divisible by this.")
        with tv_c4: tr_vcr_filter_mode = st.selectbox("Filter Mode", ["Percentile rank", "Raw value"], key="trvcr_mode", disabled=not use_tr_vcr_filter, help="Percentile: rank vs own history (asset-relative). Raw: gate on the natural value (TR ~1 = noise; VCR in [1/n, 1]). Regime quadrant ignores this — always uses median-split rank.")

        _is_regime = (tr_vcr_metric == "Regime quadrant")
        _is_raw = (tr_vcr_filter_mode == "Raw value") and (not _is_regime)
        _is_pctile = (tr_vcr_filter_mode == "Percentile rank") and (not _is_regime)

        tv_c5, tv_c6, tv_c7, tv_c8 = st.columns(4)
        with tv_c5: tr_vcr_rank_window = st.selectbox("Rank Window", ["Expanding", "Rolling 5y"], key="trvcr_rw", disabled=(not use_tr_vcr_filter) or _is_raw, help="Used by Percentile mode and by Regime quadrant medians.")
        with tv_c6: tr_vcr_min_periods = st.number_input("Rank Min Periods", 50, 2520, 252, step=21, key="trvcr_mp", disabled=(not use_tr_vcr_filter) or _is_raw, help="Burn-in before rank is computed (~1y default).")
        with tv_c7: tr_vcr_pctile_min = st.number_input("Pctile Min", 0.0, 100.0, 70.0, step=5.0, key="trvcr_pmin", disabled=not _is_pctile)
        with tv_c8: tr_vcr_pctile_max = st.number_input("Pctile Max", 0.0, 100.0, 100.0, step=5.0, key="trvcr_pmax", disabled=not _is_pctile)

        tv_c9, tv_c10, tv_c11, tv_c12 = st.columns(4)
        # Defaults pick a "trending" cut for TR and pass-through bounds for VCR.
        _is_vcr = (tr_vcr_metric == "Variance Contribution (VCR)")
        _raw_default_min = 0.0 if _is_vcr else 1.0
        _raw_default_max = 1.0 if _is_vcr else 5.0
        _raw_upper_cap = 1.0 if _is_vcr else 10.0
        with tv_c9:  tr_vcr_raw_min = st.number_input("Raw Min", 0.0, _raw_upper_cap, _raw_default_min, step=0.05, key="trvcr_rmin", disabled=not _is_raw, help="TR ~1.0 separates noise from trend; VCR floor = 1/n_blocks where n_blocks = window/sample_freq (e.g. 0.25 for window=20/sf=5).")
        with tv_c10: tr_vcr_raw_max = st.number_input("Raw Max", 0.0, _raw_upper_cap, _raw_default_max, step=0.05, key="trvcr_rmax", disabled=not _is_raw)
        with tv_c11: tr_vcr_raw_logic = st.selectbox("Raw Logic", ["Between", ">", ">=", "<", "<="], key="trvcr_rlogic", disabled=not _is_raw, help="Between uses [Raw Min, Raw Max] inclusive. Comparison uses Raw Min as the threshold.")
        with tv_c12:
            tr_vcr_regime_quadrants = st.multiselect("Quadrants", ["grinding_trend", "spike_trend", "choppy_grind", "spike_revert"], default=["grinding_trend"], key="trvcr_quad", disabled=(not use_tr_vcr_filter) or (not _is_regime), help="Median-split (rank) of TR and VCR; pick the quadrant(s) you want to allow.")
        tv_c13, tv_c14, _tv_pad1, _tv_pad2 = st.columns(4)
        with tv_c13: tr_vcr_min_consec = st.number_input("Min Consec Days", 1, 252, 1, step=1, key="trvcr_consec", disabled=not use_tr_vcr_filter, help="Require the regime/condition to have held for at least N consecutive days ending at the prior close. 1 = no streak requirement; e.g. 30 = grinding_trend has held for at least 30 trading days.")
        with tv_c14: tr_vcr_consec_first = st.checkbox("First instance only", value=False, key="trvcr_first", disabled=(not use_tr_vcr_filter) or (tr_vcr_min_consec <= 1), help="When checked, fires only on the first day the streak length is reached, not every day the streak persists. Useful for stored-energy / unwind setups so you don't re-enter the same regime episode.")
        if use_tr_vcr_filter and tr_vcr_window % tr_vcr_sample_freq != 0 and tr_vcr_metric != "Variance Contribution (VCR)":
            st.warning(f"Window ({tr_vcr_window}) is not divisible by Sample Freq ({tr_vcr_sample_freq}); TR computation will be skipped on signal evaluation.")
    with st.expander("Price Action", expanded=False):
        pa1, pa2 = st.columns(2)
        with pa1: 
            req_green_candle = st.checkbox("Require Close > Open", value=False)
            breakout_mode = st.selectbox("Close vs Prev Range", ["None", "Close > Prev Day High", "Close < Prev Day Low"])
        with pa2:
            use_range_filter = st.checkbox("Filter by Daily Candle Range %", value=False)
            r1, r2 = st.columns(2)
            with r1: range_min = st.number_input("Min % (0=Low)", 0, 100, 0, disabled=not use_range_filter)
            with r2: range_max = st.number_input("Max % (100=High)", 0, 100, 100, disabled=not use_range_filter)
        st.markdown("---")
        use_atr_ret_filter = st.checkbox("Filter by Today's Net Change (in ATR units)", value=False)
        st.caption("Calculates (Close - Prev Close) / ATR.")
        ar1, ar2 = st.columns(2)
        with ar1: atr_ret_min = st.number_input("Min Return (ATR)", -10.0, 10.0, 0.0, step=0.1, disabled=not use_atr_ret_filter)
        with ar2: atr_ret_max = st.number_input("Max Return (ATR)", -10.0, 10.0, 1.0, step=0.1, disabled=not use_atr_ret_filter)
        st.markdown("---")
        use_range_atr_filter = st.checkbox("Filter by Today's Range (in ATR units)", value=False)
        st.caption("Calculates (High - Low) / ATR.")
        ra1, ra2, ra3 = st.columns(3)
        with ra1: range_atr_logic = st.selectbox("Logic", [">", "<", "Between"], key="range_atr_logic", disabled=not use_range_atr_filter)
        with ra2: range_atr_min = st.number_input("Min Range (ATR)", 0.0, 20.0, 1.0, step=0.1, key="range_atr_min", disabled=not use_range_atr_filter)
        with ra3: range_atr_max = st.number_input("Max Range (ATR)", 0.0, 20.0, 3.0, step=0.1, key="range_atr_max", disabled=not use_range_atr_filter)
        st.markdown("---")
        use_open_gap_atr_filter = st.checkbox("Filter by Today's Open vs Prev Close (in ATR units)", value=False)
        st.caption("Calculates (Today's Open - Prev Close) / ATR. Overnight gap in ATR units.")
        og1, og2, og3 = st.columns(3)
        with og1: open_gap_atr_logic = st.selectbox("Logic", [">", "<", "Between"], key="open_gap_atr_logic", disabled=not use_open_gap_atr_filter)
        with og2: open_gap_atr_min = st.number_input("Min Gap (ATR)", -10.0, 10.0, 0.0, step=0.1, key="open_gap_atr_min", disabled=not use_open_gap_atr_filter)
        with og3: open_gap_atr_max = st.number_input("Max Gap (ATR)", -10.0, 10.0, 1.0, step=0.1, key="open_gap_atr_max", disabled=not use_open_gap_atr_filter)
        st.markdown("---")
        st.markdown("#### Multi-Day Price Action Conditions")
        st.caption("Add conditions on prior days' candles. Lag 0 = signal day, 1 = day before signal, etc. All conditions are AND logic.")
        num_pa_conditions = st.number_input("Number of conditions", 0, 6, 0, key="num_pa_cond")
        price_action_filters = []
        for pa_i in range(num_pa_conditions):
            st.markdown(f"**Condition {pa_i + 1}**")
            pa_cols = st.columns([2, 1, 1, 1, 1])
            with pa_cols[0]:
                pa_type = st.selectbox("Type", [
                    "range_pct", "atr_ret", "range_atr",
                    "close_gt_open", "close_lt_open",
                    "close_gt_prev_high", "close_lt_prev_low",
                    "atr_move", "trailing_low", "trailing_high",
                ], key=f"pa_type_{pa_i}",
                help=(
                    "atr_move: cumulative N-bar return in ATR units, anchored at lag.\n"
                    "trailing_low/high: bar at lag (or anywhere in [lag, end_lag]) "
                    "prints the trailing N-bar extreme."
                ))
            with pa_cols[1]:
                pa_lag = st.number_input("Day Offset", 0, 5, 0, key=f"pa_lag_{pa_i}")
            pa_dict = {"type": pa_type, "lag": pa_lag}
            if pa_type in ("range_pct", "atr_ret", "range_atr"):
                with pa_cols[2]:
                    pa_dict["min"] = st.number_input("Min", -10.0, 100.0, 0.0, step=0.1, key=f"pa_min_{pa_i}")
                with pa_cols[3]:
                    pa_dict["max"] = st.number_input("Max", -10.0, 100.0, 100.0, step=0.1, key=f"pa_max_{pa_i}")
                if pa_type == "range_atr":
                    with pa_cols[4]:
                        pa_dict["logic"] = st.selectbox("Logic", [">", "<", "Between"], key=f"pa_ra_logic_{pa_i}")
            elif pa_type == "atr_move":
                with pa_cols[2]:
                    pa_dict["n_bars"] = st.number_input("Over N Bars", 1, 252, 21, key=f"pa_atrmove_n_{pa_i}",
                        help="(close[lag] - close[lag+N]) / atr[lag]. N=21 over a 21-bar window.")
                with pa_cols[3]:
                    pa_dict["min"] = st.number_input("Min ATR", -50.0, 50.0, -10.0, step=0.5, key=f"pa_atrmove_min_{pa_i}")
                with pa_cols[4]:
                    pa_dict["max"] = st.number_input("Max ATR", -50.0, 50.0, 10.0, step=0.5, key=f"pa_atrmove_max_{pa_i}")
            elif pa_type in ("trailing_low", "trailing_high"):
                with pa_cols[2]:
                    pa_dict["window"] = st.number_input("Window (bars)", 2, 252, 21, key=f"pa_te_w_{pa_i}",
                        help="Trailing window for the rolling low/high.")
                with pa_cols[3]:
                    pa_dict["end_lag"] = st.number_input("End Lag (incl.)", 0, 5, pa_lag, key=f"pa_te_endlag_{pa_i}",
                        help="If > 'Day Offset', the filter passes if ANY bar from lag to end_lag prints the trailing extreme.")
            price_action_filters.append(pa_dict)
    with st.expander("Time & Cycle Filters", expanded=False):
        t_c1, t_c2 = st.columns(2)
        with t_c1:
            use_dow_filter = st.checkbox("Enable Day of Week Filter", value=False)
            c_mon, c_tue, c_wed, c_thu, c_fri = st.columns(5)
            valid_days = []
            with c_mon: 
                if st.checkbox("Mon", value=True, disabled=not use_dow_filter): valid_days.append(0)
            with c_tue: 
                if st.checkbox("Tue", value=True, disabled=not use_dow_filter): valid_days.append(1)
            with c_wed: 
                if st.checkbox("Wed", value=True, disabled=not use_dow_filter): valid_days.append(2)
            with c_thu: 
                if st.checkbox("Thu", value=True, disabled=not use_dow_filter): valid_days.append(3)
            with c_fri: 
                if st.checkbox("Fri", value=True, disabled=not use_dow_filter): valid_days.append(4)
        with t_c2:
            cycle_options = {"1. Post-Election": 1, "2. Midterm Year": 2, "3. Pre-Election": 3, "4. Election Year": 0}
            sel_cycles = st.multiselect("Include Years:", options=list(cycle_options.keys()), default=list(cycle_options.keys()))
            allowed_cycles = [cycle_options[x] for x in sel_cycles]
            _year_options = list(range(2000, datetime.datetime.now().year + 1))
            excluded_years = st.multiselect(
                "Exclude Years (drop trades entered in these years):",
                options=_year_options, default=[],
                help="Signals that would enter in these years are skipped. Useful for seeing how the strategy looks ex-2008, ex-2020, etc."
            )
    with st.expander("Trend Filter", expanded=False):
        trend_filter = st.selectbox("Trend Condition", ["None", "Price > 200 SMA", "Not Below Declining 200 SMA", "Price > Rising 200 SMA", "Market > 200 SMA", "Market Not Declining 200 SMA", "Price < 200 SMA", "Price < Falling 200 SMA", "Market < 200 SMA"])
    with st.expander("Performance Percentile Rank", expanded=False):
        col_p_config, col_p_seq = st.columns([3, 1])
        perf_filters = []
        with col_p_config:
            c2d, c5d, c10d, c21d, c126d, c252d = st.columns(6)
            with c2d:
                use_2d = st.checkbox("Enable 2D Rank")
                logic_2d = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="l2d", disabled=not use_2d)
                l_2d_txt = "Min %ile" if logic_2d in ("Between", "Not Between") else "Threshold"
                thresh_2d = st.number_input(l_2d_txt, 0.0, 100.0, 85.0, key="t2d", disabled=not use_2d)
                thresh_2d_max = 100.0
                if logic_2d in ("Between", "Not Between"): thresh_2d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="t2d_max")
                consec_2d = st.number_input("Consec Days", 1, 10, 1, key="c2d_days", disabled=not use_2d)
                if use_2d: perf_filters.append({'window': 2, 'logic': logic_2d, 'thresh': thresh_2d, 'thresh_max': thresh_2d_max, 'consecutive': consec_2d})
            with c5d:
                use_5d = st.checkbox("Enable 5D Rank")
                logic_5d = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="l5d", disabled=not use_5d)
                l_5d_txt = "Min %ile" if logic_5d in ("Between", "Not Between") else "Threshold"
                thresh_5d = st.number_input(l_5d_txt, 0.0, 100.0, 85.0, key="t5d", disabled=not use_5d)
                thresh_5d_max = 100.0
                if logic_5d in ("Between", "Not Between"): thresh_5d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="t5d_max")
                consec_5d = st.number_input("Consec Days", 1, 10, 1, key="c5d_days", disabled=not use_5d)
                if use_5d: perf_filters.append({'window': 5, 'logic': logic_5d, 'thresh': thresh_5d, 'thresh_max': thresh_5d_max, 'consecutive': consec_5d})
            with c10d:
                use_10d = st.checkbox("Enable 10D Rank")
                logic_10d = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="l10d", disabled=not use_10d)
                l_10d_txt = "Min %ile" if logic_10d in ("Between", "Not Between") else "Threshold"
                thresh_10d = st.number_input(l_10d_txt, 0.0, 100.0, 85.0, key="t10d", disabled=not use_10d)
                thresh_10d_max = 100.0
                if logic_10d in ("Between", "Not Between"): thresh_10d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="t10d_max")
                consec_10d = st.number_input("Consec Days", 1, 10, 1, key="c10d_days", disabled=not use_10d)
                if use_10d: perf_filters.append({'window': 10, 'logic': logic_10d, 'thresh': thresh_10d, 'thresh_max': thresh_10d_max, 'consecutive': consec_10d})
            with c21d:
                use_21d = st.checkbox("Enable 21D Rank")
                logic_21d = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="l21d", disabled=not use_21d)
                l_21d_txt = "Min %ile" if logic_21d in ("Between", "Not Between") else "Threshold"
                thresh_21d = st.number_input(l_21d_txt, 0.0, 100.0, 85.0, key="t21d", disabled=not use_21d)
                thresh_21d_max = 100.0
                if logic_21d in ("Between", "Not Between"): thresh_21d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="t21d_max")
                consec_21d = st.number_input("Consec Days", 1, 10, 1, key="c21d_days", disabled=not use_21d)
                if use_21d: perf_filters.append({'window': 21, 'logic': logic_21d, 'thresh': thresh_21d, 'thresh_max': thresh_21d_max, 'consecutive': consec_21d})
            with c126d:
                use_126d = st.checkbox("Enable 126D Rank")
                logic_126d = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="l126d", disabled=not use_126d)
                l_126d_txt = "Min %ile" if logic_126d in ("Between", "Not Between") else "Threshold"
                thresh_126d = st.number_input(l_126d_txt, 0.0, 100.0, 85.0, key="t126d", disabled=not use_126d)
                thresh_126d_max = 100.0
                if logic_126d in ("Between", "Not Between"): thresh_126d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="t126d_max")
                consec_126d = st.number_input("Consec Days", 1, 10, 1, key="c126d_days", disabled=not use_126d)
                if use_126d: perf_filters.append({'window': 126, 'logic': logic_126d, 'thresh': thresh_126d, 'thresh_max': thresh_126d_max, 'consecutive': consec_126d})
            with c252d:
                use_252d = st.checkbox("Enable 252D Rank")
                logic_252d = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="l252d", disabled=not use_252d)
                l_252d_txt = "Min %ile" if logic_252d in ("Between", "Not Between") else "Threshold"
                thresh_252d = st.number_input(l_252d_txt, 0.0, 100.0, 85.0, key="t252d", disabled=not use_252d)
                thresh_252d_max = 100.0
                if logic_252d in ("Between", "Not Between"): thresh_252d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="t252d_max")
                consec_252d = st.number_input("Consec Days", 1, 10, 1, key="c252d_days", disabled=not use_252d)
                if use_252d: perf_filters.append({'window': 252, 'logic': logic_252d, 'thresh': thresh_252d, 'thresh_max': thresh_252d_max, 'consecutive': consec_252d})
        with col_p_seq:
            perf_first = st.checkbox("First Instance", value=False)
            perf_lookback = st.number_input("Lookback (Days)", 1, 100, 21, disabled=not perf_first)

    perf_atr_filters = []
    with st.expander("ATR-Normalized Performance Percentile Rank", expanded=False):
        st.markdown("**Same as Performance Percentile Rank, but N-day moves are measured in ATR units before ranking.** Better for comparing moves across vol regimes within the same ticker.")
        _atr_windows = [2, 5, 10, 21, 126, 252]
        _atr_cols = st.columns(6)
        for _col, _w in zip(_atr_cols, _atr_windows):
            with _col:
                _use = st.checkbox(f"Enable {_w}D ATR Rank", key=f"use_atrp_{_w}")
                _logic = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key=f"atrp_l_{_w}", disabled=not _use)
                _lbl = "Min %ile" if _logic in ("Between", "Not Between") else "Threshold"
                _thresh = st.number_input(_lbl, 0.0, 100.0, 85.0, key=f"atrp_t_{_w}", disabled=not _use)
                _thresh_max = 100.0
                if _logic in ("Between", "Not Between"):
                    _thresh_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key=f"atrp_tmax_{_w}", disabled=not _use)
                _consec = st.number_input("Consec Days", 1, 10, 1, key=f"atrp_c_{_w}", disabled=not _use)
                if _use:
                    perf_atr_filters.append({
                        'window': _w, 'logic': _logic, 'thresh': _thresh,
                        'thresh_max': _thresh_max, 'consecutive': _consec
                    })

    atr_sznl_filters = []
    with st.expander("ATR Seasonal Rank Filter", expanded=False):
        st.markdown("**Rank historical ATR-normalized forward returns per day-of-year.** Walk-forward safe. 100 = best seasonal window, 0 = worst. Requires `atr_seasonal_ranks.parquet`.")
        asz_cols = st.columns(6)
        _asz_windows = [5, 10, 21, 63, 126, 252]
        for _col, _w in zip(asz_cols, _asz_windows):
            with _col:
                _use = st.checkbox(f"Enable {_w}D", key=f"use_asz_{_w}")
                _logic = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key=f"asz_l_{_w}", disabled=not _use)
                _lbl = "Min %ile" if _logic in ("Between", "Not Between") else "Threshold"
                _thresh = st.number_input(_lbl, 0.0, 100.0, 80.0, key=f"asz_t_{_w}", disabled=not _use)
                _thresh_max = 100.0
                if _logic in ("Between", "Not Between"):
                    _thresh_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key=f"asz_tmax_{_w}", disabled=not _use)
                _consec = st.number_input("Consec Days", 1, 10, 1, key=f"asz_c_{_w}", disabled=not _use)
                if _use:
                    atr_sznl_filters.append({
                        'window': _w, 'logic': _logic, 'thresh': _thresh,
                        'thresh_max': _thresh_max, 'consecutive': _consec
                    })

    xsec_filters = []
    with st.expander("Cross-Sectional Rank (vs Universe Peers)", expanded=False):
        st.markdown("**Rank this ticker's return percentile vs all other tickers in the universe on each date.** Normalized for volatility. 100 = most overbought vs peers, 0 = most oversold.")
        use_xsec_filter = st.checkbox("Enable Cross-Sectional Rank Filter", value=False)
        xc5d, xc10d, xc21d, xc126d, xc252d = st.columns(5)
        with xc5d:
            st.markdown("**5-Day Rank**")
            use_xsec_5d = st.checkbox("Enable", key="use_xsec_5d", value=False, disabled=not use_xsec_filter)
            xsec_5d_logic = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="xsec_l5d", disabled=not (use_xsec_filter and use_xsec_5d))
            xsec_5d_lbl = "Min %ile" if xsec_5d_logic in ("Between", "Not Between") else "Threshold"
            xsec_5d_thresh = st.number_input(xsec_5d_lbl, 0.0, 100.0, 85.0, key="xsec_t5d", disabled=not (use_xsec_filter and use_xsec_5d))
            xsec_5d_max = 100.0
            if xsec_5d_logic in ("Between", "Not Between"): xsec_5d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="xsec_t5d_max")
            xsec_5d_consec = st.number_input("Consec Days", 1, 10, 1, key="xsec_c5d", disabled=not (use_xsec_filter and use_xsec_5d))
            if use_xsec_5d: xsec_filters.append({'window': 5, 'logic': xsec_5d_logic, 'thresh': xsec_5d_thresh, 'thresh_max': xsec_5d_max, 'consecutive': xsec_5d_consec})
        with xc10d:
            st.markdown("**10-Day Rank**")
            use_xsec_10d = st.checkbox("Enable", key="use_xsec_10d", value=False, disabled=not use_xsec_filter)
            xsec_10d_logic = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="xsec_l10d", disabled=not (use_xsec_filter and use_xsec_10d))
            xsec_10d_lbl = "Min %ile" if xsec_10d_logic in ("Between", "Not Between") else "Threshold"
            xsec_10d_thresh = st.number_input(xsec_10d_lbl, 0.0, 100.0, 85.0, key="xsec_t10d", disabled=not (use_xsec_filter and use_xsec_10d))
            xsec_10d_max = 100.0
            if xsec_10d_logic in ("Between", "Not Between"): xsec_10d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="xsec_t10d_max")
            xsec_10d_consec = st.number_input("Consec Days", 1, 10, 1, key="xsec_c10d", disabled=not (use_xsec_filter and use_xsec_10d))
            if use_xsec_10d: xsec_filters.append({'window': 10, 'logic': xsec_10d_logic, 'thresh': xsec_10d_thresh, 'thresh_max': xsec_10d_max, 'consecutive': xsec_10d_consec})
        with xc21d:
            st.markdown("**21-Day Rank**")
            use_xsec_21d = st.checkbox("Enable", key="use_xsec_21d", value=False, disabled=not use_xsec_filter)
            xsec_21d_logic = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="xsec_l21d", disabled=not (use_xsec_filter and use_xsec_21d))
            xsec_21d_lbl = "Min %ile" if xsec_21d_logic in ("Between", "Not Between") else "Threshold"
            xsec_21d_thresh = st.number_input(xsec_21d_lbl, 0.0, 100.0, 85.0, key="xsec_t21d", disabled=not (use_xsec_filter and use_xsec_21d))
            xsec_21d_max = 100.0
            if xsec_21d_logic in ("Between", "Not Between"): xsec_21d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="xsec_t21d_max")
            xsec_21d_consec = st.number_input("Consec Days", 1, 10, 1, key="xsec_c21d", disabled=not (use_xsec_filter and use_xsec_21d))
            if use_xsec_21d: xsec_filters.append({'window': 21, 'logic': xsec_21d_logic, 'thresh': xsec_21d_thresh, 'thresh_max': xsec_21d_max, 'consecutive': xsec_21d_consec})
        with xc126d:
            st.markdown("**126-Day Rank**")
            use_xsec_126d = st.checkbox("Enable", key="use_xsec_126d", value=False, disabled=not use_xsec_filter)
            xsec_126d_logic = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="xsec_l126d", disabled=not (use_xsec_filter and use_xsec_126d))
            xsec_126d_lbl = "Min %ile" if xsec_126d_logic in ("Between", "Not Between") else "Threshold"
            xsec_126d_thresh = st.number_input(xsec_126d_lbl, 0.0, 100.0, 85.0, key="xsec_t126d", disabled=not (use_xsec_filter and use_xsec_126d))
            xsec_126d_max = 100.0
            if xsec_126d_logic in ("Between", "Not Between"): xsec_126d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="xsec_t126d_max")
            xsec_126d_consec = st.number_input("Consec Days", 1, 10, 1, key="xsec_c126d", disabled=not (use_xsec_filter and use_xsec_126d))
            if use_xsec_126d: xsec_filters.append({'window': 126, 'logic': xsec_126d_logic, 'thresh': xsec_126d_thresh, 'thresh_max': xsec_126d_max, 'consecutive': xsec_126d_consec})
        with xc252d:
            st.markdown("**252-Day Rank**")
            use_xsec_252d = st.checkbox("Enable", key="use_xsec_252d", value=False, disabled=not use_xsec_filter)
            xsec_252d_logic = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="xsec_l252d", disabled=not (use_xsec_filter and use_xsec_252d))
            xsec_252d_lbl = "Min %ile" if xsec_252d_logic in ("Between", "Not Between") else "Threshold"
            xsec_252d_thresh = st.number_input(xsec_252d_lbl, 0.0, 100.0, 85.0, key="xsec_t252d", disabled=not (use_xsec_filter and use_xsec_252d))
            xsec_252d_max = 100.0
            if xsec_252d_logic in ("Between", "Not Between"): xsec_252d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="xsec_t252d_max")
            xsec_252d_consec = st.number_input("Consec Days", 1, 10, 1, key="xsec_c252d", disabled=not (use_xsec_filter and use_xsec_252d))
            if use_xsec_252d: xsec_filters.append({'window': 252, 'logic': xsec_252d_logic, 'thresh': xsec_252d_thresh, 'thresh_max': xsec_252d_max, 'consecutive': xsec_252d_consec})
    ma_consec_filters = []
    with st.expander("Consecutive Closes vs SMA", expanded=False):
        c_ma1, c_ma2, c_ma3 = st.columns(3)
        with c_ma1:
            use_ma1 = st.checkbox("Enable MA 1", key="use_ma1")
            len_ma1 = st.number_input("Length", 2, 500, 10, key="len_ma1", disabled=not use_ma1)
            logic_ma1 = st.selectbox("Close vs MA", ["Above", "Below"], key="logic_ma1", disabled=not use_ma1)
            consec_ma1 = st.number_input("Consecutive Days", 1, 50, 1, key="consec_ma1", disabled=not use_ma1)
            if use_ma1: ma_consec_filters.append({'length': len_ma1, 'logic': logic_ma1, 'consec': consec_ma1})
        with c_ma2:
            use_ma2 = st.checkbox("Enable MA 2", key="use_ma2")
            len_ma2 = st.number_input("Length", 2, 500, 20, key="len_ma2", disabled=not use_ma2)
            logic_ma2 = st.selectbox("Close vs MA", ["Above", "Below"], key="logic_ma2", disabled=not use_ma2)
            consec_ma2 = st.number_input("Consecutive Days", 1, 50, 1, key="consec_ma2", disabled=not use_ma2)
            if use_ma2: ma_consec_filters.append({'length': len_ma2, 'logic': logic_ma2, 'consec': consec_ma2})
        with c_ma3:
            use_ma3 = st.checkbox("Enable MA 3", key="use_ma3")
            len_ma3 = st.number_input("Length", 2, 500, 50, key="len_ma3", disabled=not use_ma3)
            logic_ma3 = st.selectbox("Close vs MA", ["Above", "Below"], key="logic_ma3", disabled=not use_ma3)
            consec_ma3 = st.number_input("Consecutive Days", 1, 50, 1, key="consec_ma3", disabled=not use_ma3)
            if use_ma3: ma_consec_filters.append({'length': len_ma3, 'logic': logic_ma3, 'consec': consec_ma3})
    with st.expander("Seasonal Rank", expanded=False):
        use_sznl = st.checkbox("Enable Ticker Seasonal Filter", value=False)
        s1, s2, s3, s4 = st.columns(4)
        with s1: sznl_logic = st.selectbox("Logic", ["<", ">"], key="sl", disabled=not use_sznl)
        with s2: sznl_thresh = st.number_input("Threshold", 0.0, 100.0, 15.0, key="st", disabled=not use_sznl)
        with s3: sznl_first = st.checkbox("First Instance Only", value=False, key="sf", disabled=not use_sznl)
        with s4: sznl_lookback = st.number_input("Instance Lookback (Days)", 1, 100, 21, key="slb", disabled=not use_sznl)
        st.markdown("---")
        use_market_sznl = st.checkbox("Enable Market Seasonal Filter", value=False)
        spy1, spy2 = st.columns(2)
        with spy1: market_sznl_logic = st.selectbox("Market Logic", ["<", ">"], key="spy_sl", disabled=not use_market_sznl)
        with spy2: market_sznl_thresh = st.number_input("Market Threshold", 0.0, 100.0, 15.0, key="spy_st", disabled=not use_market_sznl)
    with st.expander("Rolling High/Low (N-day)", expanded=False):
        use_52w = st.checkbox("Enable Rolling High/Low Filter", value=False)
        h1, h2, h3, h4, h5 = st.columns(5)
        with h1: type_52w = st.selectbox("Condition", ["New High", "New Low"], disabled=not use_52w)
        with h2: window_52w = st.number_input(
            "Window (days)", 5, 1260, 252,
            help="Rolling lookback for the high/low. 252 = 52w (default), 63 = ~3mo, 21 = ~1mo, 1260 = 5y.",
            disabled=not use_52w,
        )
        with h3: first_52w = st.checkbox("First Instance Only", value=False, key="hf", disabled=not use_52w)
        with h4: lookback_52w = st.number_input("Instance Lookback (Days)", 1, 252, 21, key="hlb", disabled=not use_52w)
        with h5: lag_52w = st.number_input("Lag (Days)", 0, 10, 0, disabled=not use_52w)
        exclude_52w_high = st.checkbox("Exclude if Today IS a 52w High", value=False)
        recent_52w_mode = st.selectbox("Trailing 52w High Filter", ["Disabled", "Made 52w High in Last N Days", "Has NOT Made 52w High in Last N Days"])
        use_recent_52w = recent_52w_mode != "Disabled"
        recent_52w_invert = recent_52w_mode == "Has NOT Made 52w High in Last N Days"
        recent_52w_lookback = st.number_input("52w High Lookback (Days)", 1, 252, 21, disabled=not use_recent_52w)
        recent_52w_low_mode = st.selectbox("Trailing 52w Low Filter", ["Disabled", "Made 52w Low in Last N Days", "Has NOT Made 52w Low in Last N Days"], key="recent_52w_low_mode")
        use_recent_52w_low = recent_52w_low_mode != "Disabled"
        recent_52w_low_invert = recent_52w_low_mode == "Has NOT Made 52w Low in Last N Days"
        recent_52w_low_lookback = st.number_input("52w Low Lookback (Days)", 1, 252, 21, disabled=not use_recent_52w_low, key="recent_52w_low_lb")
        st.markdown("---")
        use_ath = st.checkbox("Enable All-Time High Filter", value=False)
        ath_type = st.selectbox("ATH Condition", ["Today is ATH", "Today is NOT ATH"], disabled=not use_ath)
        recent_ath_mode = st.selectbox("Trailing ATH Filter", ["Disabled", "Made ATH in Last N Days", "Has NOT Made ATH in Last N Days"])
        use_recent_ath = recent_ath_mode != "Disabled"
        recent_ath_invert = recent_ath_mode == "Has NOT Made ATH in Last N Days"
        ath_lookback_days = st.number_input("ATH Lookback (Days)", 1, 252, 21, disabled=not use_recent_ath)
    with st.expander("Market Regime (VIX)", expanded=False):
        use_vix_filter = st.checkbox(f"Enable {VIX_TICKER} Filter", value=False)
        v1, v2 = st.columns(2)
        with v1: vix_min = st.number_input("Min VIX Value", 0.0, 200.0, 0.0, disabled=not use_vix_filter)
        with v2: vix_max = st.number_input("Max VIX Value", 0.0, 200.0, 20.0, disabled=not use_vix_filter)

    with st.expander("Risk Dial Filters (Fragility Score, 2016+)", expanded=False):
        _frag_df = load_fragility_dials()
        if _frag_df is None or _frag_df.empty:
            st.warning("data/rd2_fragility.parquet not found — run the Risk Dashboard once to generate it.")
            dial_filters = []
        else:
            st.caption(
                f"Fragility dial (0-100) from Risk Dashboard. History: {_frag_df.index.min().strftime('%Y-%m-%d')} to "
                f"{_frag_df.index.max().strftime('%Y-%m-%d')}. Pre-history trades fail any enabled dial filter."
            )
            dial_filters = []
            for i in range(1, 4):
                fc1, fc2, fc3, fc4, fc5 = st.columns([1, 1, 1.1, 0.9, 1])
                with fc1:
                    _use = st.checkbox(f"Filter {i}", key=f"dial_use_{i}")
                with fc2:
                    _dial = st.selectbox(
                        "Dial", ["5d", "21d", "63d"], key=f"dial_col_{i}",
                        index=(i - 1) if i <= 3 else 0, disabled=not _use,
                    )
                with fc3:
                    _win = st.number_input(
                        "Rolling avg (days)", 1, 63, 10, step=1,
                        key=f"dial_win_{i}", disabled=not _use,
                        help="1 = raw dial value; higher = smoothed over N days",
                    )
                with fc4:
                    _log = st.selectbox(
                        "Logic", [">", "<", ">=", "<="], key=f"dial_log_{i}", disabled=not _use,
                    )
                with fc5:
                    _thr = st.number_input(
                        "Threshold", 0.0, 100.0, 40.0 if i == 1 else 30.0 if i == 2 else 20.0,
                        step=5.0, key=f"dial_thr_{i}", disabled=not _use,
                    )
                if _use:
                    dial_filters.append({
                        'dial': _dial, 'window': int(_win),
                        'logic': _log, 'thresh': float(_thr),
                    })
            if dial_filters:
                st.success(
                    "Active dial filters: "
                    + "; ".join(f"{f['dial']} dial ({f['window']}d avg) {f['logic']} {f['thresh']:.0f}" for f in dial_filters)
                )
    with st.expander("Volume Filters", expanded=False):
        use_vol_gt_prev = st.checkbox("Require Volume > Prev Day Volume", value=False)
        c1, c2 = st.columns(2)
        with c1:
            use_vol = st.checkbox("Enable Spike Filter", value=False)
            vs_col1, vs_col2 = st.columns(2)
            with vs_col1: vol_logic = st.selectbox("Logic", [">", "<", "Between"], key="vs_logic", disabled=not use_vol)
            vs_lbl = "Min Multiple" if vol_logic == "Between" else "Vol Multiple (vs 63d Avg)"
            with vs_col2: vol_thresh = st.number_input(vs_lbl, 0.1, 10.0, 1.5, disabled=not use_vol)
            vol_thresh_max = 10.0
            if vol_logic == "Between":
                vol_thresh_max = st.number_input("Max Multiple", 0.1, 10.0, 3.0, key="vs_tmax", disabled=not use_vol)
        with c2:
            use_vol_rank = st.checkbox("Enable Regime Filter", value=False)
            v_col1, v_col2 = st.columns(2)
            with v_col1: vol_rank_logic = st.selectbox("Logic", ["<", ">"], key="vrl", disabled=not use_vol_rank)
            with v_col2: vol_rank_thresh = st.number_input("Percentile (0-100)", 0.0, 100.0, 50.0, key="vrt", disabled=not use_vol_rank)
    # =====================================================
    # NEW: REFERENCE TICKER FILTER
    # =====================================================
    with st.expander("Reference Ticker Filter (e.g., IWM, SPY regime)", expanded=False):
        st.markdown("**Filter signals based on another ticker's performance rank.** Useful for regime filtering (e.g., only take longs when IWM is oversold).")
        use_ref_ticker_filter = st.checkbox("Enable Reference Ticker Filter", value=False)
        ref_ticker_input = st.text_input("Reference Ticker", value="IWM", disabled=not use_ref_ticker_filter).strip().upper()
        st.markdown("---")
        ref_filters = []
        ref_c1, ref_c2, ref_c3 = st.columns(3)
        with ref_c1:
            st.markdown("**5-Day Rank**")
            use_ref_5d = st.checkbox("Enable", key="use_ref_5d", value=False, disabled=not use_ref_ticker_filter)
            ref_5d_logic = st.selectbox("Logic", ["<", ">"], key="ref_5d_logic", disabled=not (use_ref_ticker_filter and use_ref_5d))
            ref_5d_thresh = st.number_input("Threshold %ile", 0.0, 100.0, 33.0, key="ref_5d_thresh", disabled=not (use_ref_ticker_filter and use_ref_5d))
            if use_ref_5d: ref_filters.append({'window': 5, 'logic': ref_5d_logic, 'thresh': ref_5d_thresh})
        with ref_c2:
            st.markdown("**10-Day Rank**")
            use_ref_10d = st.checkbox("Enable", key="use_ref_10d", value=False, disabled=not use_ref_ticker_filter)
            ref_10d_logic = st.selectbox("Logic", ["<", ">"], key="ref_10d_logic", disabled=not (use_ref_ticker_filter and use_ref_10d))
            ref_10d_thresh = st.number_input("Threshold %ile", 0.0, 100.0, 33.0, key="ref_10d_thresh", disabled=not (use_ref_ticker_filter and use_ref_10d))
            if use_ref_10d: ref_filters.append({'window': 10, 'logic': ref_10d_logic, 'thresh': ref_10d_thresh})
        with ref_c3:
            st.markdown("**21-Day Rank**")
            use_ref_21d = st.checkbox("Enable", key="use_ref_21d", value=False, disabled=not use_ref_ticker_filter)
            ref_21d_logic = st.selectbox("Logic", ["<", ">"], key="ref_21d_logic", disabled=not (use_ref_ticker_filter and use_ref_21d))
            ref_21d_thresh = st.number_input("Threshold %ile", 0.0, 100.0, 33.0, key="ref_21d_thresh", disabled=not (use_ref_ticker_filter and use_ref_21d))
            if use_ref_21d: ref_filters.append({'window': 21, 'logic': ref_21d_logic, 'thresh': ref_21d_thresh})
    st.markdown("---")
    if st.button("Run Backtest", type="primary", use_container_width=True):
        tickers_to_run = []
        sznl_map = load_seasonal_map()

        def _fetch_prices(tickers, start):
            """Default behavior: yfinance via download_universe_data (legacy).
            When use_master_parquet is on — OR the dynamic Overflow universe is
            selected — AND the master file exists, read from the parquet caches
            via data_provider instead. The Overflow universe also unions the
            isolated overflow_prices.parquet staging cache (include_overflow).
            Returns dict {ticker: DataFrame} matching the legacy shape."""
            _is_overflow = univ_choice in ("Overflow (dynamic)", "OOS")
            if use_master_parquet or _is_overflow:
                try:
                    import data_provider
                    if data_provider.has_master():
                        return data_provider.get_history(tickers, start=start, include_overflow=_is_overflow)
                    r2_err = data_provider.last_r2_error()
                    warn = "Master parquet not found at data/master_prices.parquet"
                    if r2_err:
                        warn += f" (R2 refresh failed: {r2_err})"
                    warn += " — falling back to yfinance for this run."
                    st.warning(warn)
                except Exception as _e:
                    st.warning(f"data_provider unavailable ({_e}); using yfinance.")
            return download_universe_data(tickers, start)

        def _fetch_overflow(tickers, start):
            """Always price dynamic-overflow names from master_prices ∪ overflow_prices
            (include_overflow=True), regardless of the master-parquet toggle — ~481 of
            them live only in overflow_prices. Falls back to yfinance if the master
            file/R2 is unavailable."""
            try:
                import data_provider
                if data_provider.has_master():
                    return data_provider.get_history(tickers, start=start, include_overflow=True)
                st.warning("Master parquet unavailable for dynamic-overflow names — using yfinance.")
            except Exception as _e:
                st.warning(f"data_provider unavailable ({_e}); using yfinance for dynamic names.")
            return download_universe_data(tickers, start)

        if univ_choice == "Sector ETFs": tickers_to_run = SECTOR_ETFS
        elif univ_choice == "Indices": tickers_to_run = INDEX_ETFS
        elif univ_choice == "Indices (Spot — ^GSPC/^NDX)": tickers_to_run = INDICES_SPOT
        elif univ_choice == "SPX": tickers_to_run = SPX
        elif univ_choice == "International ETFs": tickers_to_run = INTERNATIONAL_ETFS
        elif univ_choice == "Sector + Index ETFs": tickers_to_run = list(set(SECTOR_ETFS + INDEX_ETFS))
        elif univ_choice == "Commodity + Sector + Index + 3x Lev": tickers_to_run = COMMODITY_SECTOR_INDEX_LEV3X
        elif univ_choice == "All CSV Tickers": tickers_to_run = [t for t in list(sznl_map.keys())]
        elif univ_choice in ("All CSV + Overflow Extras", "All CSV + Overflow Extras (no ^ indices)"):
            _drop_caret = univ_choice == "All CSV + Overflow Extras (no ^ indices)"
            _base = [t for t in list(sznl_map.keys())]
            if _drop_caret:
                _base = [t for t in _base if "^" not in t]
            _base_set = set(_base)
            _new_extras = [t for t in extras_tickers if t not in _base_set]
            if _drop_caret:
                _new_extras = [t for t in _new_extras if "^" not in t]
            _scope = st.session_state.get('overflow_run_scope', 'All (base + extras)')
            if _scope == "Base only (All CSV)":
                tickers_to_run = list(_base)
            elif _scope == "Extras only (overflow)":
                tickers_to_run = list(_new_extras)
            else:
                tickers_to_run = _base + _new_extras
            if extras_tickers:
                _suffix = " — ^ indices excluded" if _drop_caret else ""
                st.info(f"Universe cached: {len(_base)} base + {len(_new_extras)} extras{_suffix}. **Running on {len(tickers_to_run)}** ({_scope}).")
        elif univ_choice == "Overflow (dynamic)":
            # Comprehensive overflow universe = dynamic screen ∪ legacy static tier
            # (CSV_UNIVERSE − liquid). respect_active=False so backtests read the full
            # union regardless of the live activation gate.
            tickers_to_run = load_overflow_universe_full(
                static_fallback=_OVERFLOW_STATIC_TIER, respect_active=False
            )
            st.info(f"🌊 Overflow (comprehensive): **{len(tickers_to_run)}** names "
                    "(dynamic screen ∪ legacy static tier). "
                    "Prices read from master_prices ∪ overflow_prices; earnings from "
                    "production ∪ overflow staging; ATR-seasonal ranks from atr_seasonal_ranks.parquet. "
                    "Caveats: dynamic membership is today's screen (survivorship); names not in the "
                    "seasonal map get a neutral Sznl=50, so seasonal-rank filters are degraded on those.")
        elif univ_choice == "OOS":
            # Out-of-sample: the comprehensive overflow names that are NOT in the
            # legacy static tier (CSV_UNIVERSE − liquid) — i.e. the brand-new names
            # the dynamic screen added. A holdout set never present in the old universe.
            _comp = load_overflow_universe_full(static_fallback=_OVERFLOW_STATIC_TIER, respect_active=False)
            _static_norm = {str(t).upper().replace('.', '-') for t in _OVERFLOW_STATIC_TIER}
            tickers_to_run = [t for t in _comp if t not in _static_norm]
            st.info(f"🧪 OOS: **{len(tickers_to_run)}** out-of-sample names "
                    "(dynamic overflow screen minus the legacy static tier — never in the old universe). "
                    "Prices from master_prices ∪ overflow_prices; earnings from production ∪ overflow staging. "
                    "Caveats: survivorship (today's screen across history); recent IPOs may lack ATR-seasonal "
                    "ranks (fail-closed on those filters) and seasonal-map coverage (neutral Sznl=50).")
        elif univ_choice == "All CSV (Equities Only)": tickers_to_run = [t for t in list(sznl_map.keys()) if t not in ["BTC-USD", "ETH-USD", "SLV", "GLD", "USO", "UVXY", "CEF", "UNG", "XOP"] + SECTOR_ETFS + INDEX_ETFS + INTERNATIONAL_ETFS + SPX]
        elif univ_choice == "3x Leveraged (All)": tickers_to_run = LEV3X_ALL
        elif univ_choice == "3x Leveraged Equities": tickers_to_run = LEV3X_EQUITY_ALL
        elif univ_choice == "3x Leveraged Equities (Bull)": tickers_to_run = LEV3X_EQUITY_BULL_ALL
        elif univ_choice == "3x Leveraged Equities (Bear)": tickers_to_run = LEV3X_EQUITY_BEAR_ALL
        elif univ_choice == "3x Leveraged Equities (Broad Only)": tickers_to_run = LEV3X_EQUITY_BROAD
        elif univ_choice == "Custom (Upload CSV)": tickers_to_run = custom_tickers
        elif univ_choice == "Custom (Comma-Separated)": tickers_to_run = custom_tickers
        if tickers_to_run and sample_pct < 100:
            count = max(1, int(len(tickers_to_run) * (sample_pct / 100)))
            tickers_to_run = random.sample(tickers_to_run, count)
            st.info(f"Randomly selected {len(tickers_to_run)} tickers.")
        # Optional union: graft the dynamic overflow screen on top of the selected
        # universe so older lists / older overflow extras can be backtested alongside
        # the new dynamic overflow in a single run. These names are fetched separately
        # via _fetch_overflow (master ∪ overflow), so the base universe's data source
        # is untouched. No-op when the universe is already "Overflow (dynamic)".
        _dyn_overflow_add = []
        if include_dynamic_overflow and univ_choice != "Overflow (dynamic)":
            _dyn = load_overflow_universe_full(
                static_fallback=_OVERFLOW_STATIC_TIER, respect_active=False
            )
            _seen = {str(t).upper() for t in tickers_to_run}
            _dyn_overflow_add = [t for t in _dyn if str(t).upper() not in _seen]
            tickers_to_run = list(tickers_to_run) + _dyn_overflow_add
            st.info(f"🌊 Unioned Overflow (comprehensive): +{len(_dyn_overflow_add)} new names "
                    f"on top of {univ_choice} (**{len(tickers_to_run)}** total to run).")
        if not tickers_to_run: st.error("No tickers found."); return
        fetch_start = "1950-01-01" if use_full_history else start_date - datetime.timedelta(days=365)
        # Split download: in hybrid mode, fetch base and extras separately so the 1k-ticker
        # base keeps its cache key across different extras uploads (only the delta re-fetches).
        _dyn_add_set = set(_dyn_overflow_add)
        if univ_choice in ("All CSV + Overflow Extras", "All CSV + Overflow Extras (no ^ indices)") and extras_tickers:
            st.info(f"Loading base ({len(_base)}) + extras ({len(_new_extras)} new)...")
            base_data = _fetch_prices(_base, fetch_start)
            extras_data = _fetch_prices(_new_extras, fetch_start) if _new_extras else {}
            _full_data = {**base_data, **extras_data}
            _run_set = set(tickers_to_run)
            data_dict = {t: df for t, df in _full_data.items() if t in _run_set}
        else:
            _base_only = [t for t in tickers_to_run if t not in _dyn_add_set] if _dyn_add_set else tickers_to_run
            st.info(f"Loading data ({len(_base_only)} tickers)...")
            data_dict = _fetch_prices(_base_only, fetch_start)
        # Merge the dynamic-overflow names (master ∪ overflow), priced independently
        # of the base universe's source.
        if _dyn_overflow_add:
            st.info(f"Loading {len(_dyn_overflow_add)} dynamic-overflow names (master ∪ overflow)...")
            data_dict.update(_fetch_overflow(_dyn_overflow_add, fetch_start))
        if not data_dict: return

        # Truncate to end_date — drops any data after the chosen end so signals
        # can't fire and trades can't extend past the backtest window. Useful
        # for out-of-sample testing.
        if end_date < datetime.date.today():
            _end_ts = pd.Timestamp(end_date)
            _truncated = 0
            for _t, _df in list(data_dict.items()):
                if _df is None or _df.empty:
                    continue
                _sliced = _df[_df.index <= _end_ts]
                if len(_sliced) < len(_df):
                    _truncated += 1
                if _sliced.empty:
                    del data_dict[_t]
                else:
                    data_dict[_t] = _sliced
            st.info(f"Truncated {_truncated} ticker(s) to end date {end_date}")
        market_series, market_sznl_series = None, None
        market_sma_not_declining_series = None
        need_market_data = ("Market" in trend_filter) or use_market_sznl
        if need_market_data:
            market_df = data_dict.get(MARKET_TICKER)
            if market_df is None:
                st.info(f"Fetching {MARKET_TICKER} data...")
                market_dict_temp = _fetch_prices([MARKET_TICKER], fetch_start)
                market_df = market_dict_temp.get(MARKET_TICKER, None)
            if market_df is not None and not market_df.empty:
                if market_df.index.tz is not None: market_df.index = market_df.index.tz_localize(None)
                market_df.index = market_df.index.normalize()
                market_df['SMA200'] = market_df['Close'].rolling(200).mean()
                market_series = market_df['Close'] > market_df['SMA200']
                market_sma_not_declining_series = market_df['SMA200'] >= market_df['SMA200'].shift(1)
                if use_market_sznl: market_sznl_series = get_sznl_val_series(MARKET_TICKER, market_df.index, sznl_map)
        vix_series = None
        if use_vix_filter:
            vix_df = data_dict.get(VIX_TICKER)
            if vix_df is None:
                st.info(f"Fetching {VIX_TICKER} data...")
                vix_dict_temp = _fetch_prices([VIX_TICKER], fetch_start)
                vix_df = vix_dict_temp.get(VIX_TICKER, None)
            if vix_df is not None and not vix_df.empty:
                if vix_df.index.tz is not None: vix_df.index = vix_df.index.tz_localize(None)
                vix_df.index = vix_df.index.normalize()
                vix_series = vix_df['Close']
        
        # =====================================================
        # NEW: PREPARE REFERENCE TICKER DATA
        # =====================================================
        ref_ticker_ranks = None
        if use_ref_ticker_filter and ref_filters and ref_ticker_input:
            st.info(f"Preparing reference ticker data for {ref_ticker_input}...")
            ref_ticker_clean = ref_ticker_input.replace('.', '-')
            ref_df = data_dict.get(ref_ticker_clean)
            if ref_df is None:
                ref_dict_temp = _fetch_prices([ref_ticker_clean], fetch_start)
                ref_df = ref_dict_temp.get(ref_ticker_clean, None)
            if ref_df is not None and not ref_df.empty:
                # Calculate indicators for the reference ticker
                ref_df_calc = calculate_indicators(ref_df.copy(), sznl_map, ref_ticker_clean, market_series, vix_series, market_sznl_series)
                # Extract the rank series we need
                ref_ticker_ranks = {}
                for rf in ref_filters:
                    window = rf['window']
                    col = f'rank_ret_{window}d'
                    if col in ref_df_calc.columns:
                        ref_ticker_ranks[window] = ref_df_calc[col]
                st.success(f"✓ Reference ticker {ref_ticker_input} loaded with {len(ref_df_calc)} rows")
            else:
                st.warning(f"Could not load data for reference ticker {ref_ticker_input}. Filter will be skipped.")
        
        params = {
            'use_max_daily_risk': use_max_daily_risk, 'max_daily_risk_pct': max_daily_risk_pct,
            'backtest_start_date': start_date, 'trade_direction': trade_direction, 'max_one_pos': max_one_pos, 'allow_same_day_reentry': allow_same_day_reentry,
            'max_daily_entries': max_daily_entries, 'max_total_positions': max_total_positions, 'use_stop_loss': use_stop_loss, 'use_take_profit': use_take_profit, 'time_exit_only': time_exit_only,
            'use_partial_exits': use_partial_exits, 'partial_target_fraction': partial_target_fraction,
            'use_eod_dd_exit': use_eod_dd_exit, 'eod_dd_atr': eod_dd_atr, 'eod_dd_weekdays': eod_dd_weekdays,
            'stop_atr': stop_atr, 'tgt_atr': tgt_atr, 'holding_days': hold_days, 'entry_type': entry_type, 'use_ma_entry_filter': use_ma_entry_filter, 'require_close_gt_open': req_green_candle,
            'use_intraday': use_intraday,
            'use_trailing_stop': use_trailing_stop, 'trail_atr': trail_atr, 'trail_anchor': trail_anchor,
            'breakout_mode': breakout_mode, 'use_range_filter': use_range_filter, 'range_min': range_min, 'range_max': range_max, 'use_dow_filter': use_dow_filter, 'allowed_days': valid_days,
            'allowed_cycles': allowed_cycles, 'excluded_years': excluded_years, 'min_price': min_price, 'min_vol': min_vol, 'max_vol': max_vol, 'min_age': min_age, 'max_age': max_age, 'min_atr_pct': min_atr_pct, 'max_atr_pct': max_atr_pct,
            'trend_filter': trend_filter, 'universe_tickers': tickers_to_run, 'slippage_bps': slippage_bps, 'entry_conf_bps': entry_conf_bps, 'perf_filters': perf_filters, 'perf_atr_filters': perf_atr_filters, 'perf_first_instance': perf_first,
            'use_atr_ret_filter': use_atr_ret_filter, 'atr_ret_min': atr_ret_min, 'atr_ret_max': atr_ret_max,
            'use_range_atr_filter': use_range_atr_filter, 'range_atr_logic': range_atr_logic, 'range_atr_min': range_atr_min, 'range_atr_max': range_atr_max,
            'use_open_gap_atr_filter': use_open_gap_atr_filter, 'open_gap_atr_logic': open_gap_atr_logic, 'open_gap_atr_min': open_gap_atr_min, 'open_gap_atr_max': open_gap_atr_max,
            'price_action_filters': price_action_filters,
            'perf_lookback': perf_lookback, 'ma_consec_filters': ma_consec_filters, 'use_sznl': use_sznl, 'sznl_logic': sznl_logic, 'sznl_thresh': sznl_thresh, 'sznl_first_instance': sznl_first,
            'sznl_lookback': sznl_lookback, 'use_market_sznl': use_market_sznl, 'market_sznl_logic': market_sznl_logic, 'market_sznl_thresh': market_sznl_thresh, 'use_52w': use_52w, '52w_type': type_52w,
            'use_ath': use_ath, 'ath_type': ath_type,
            '52w_first_instance': first_52w, '52w_lookback': lookback_52w, '52w_lag': lag_52w, '52w_window': window_52w, 'exclude_52w_high': exclude_52w_high, 'use_vix_filter': use_vix_filter, 'vix_min': vix_min, 'vix_max': vix_max,
            'use_recent_52w': use_recent_52w, 'recent_52w_invert': recent_52w_invert, 'recent_52w_lookback': recent_52w_lookback,
            'use_recent_52w_low': use_recent_52w_low, 'recent_52w_low_invert': recent_52w_low_invert, 'recent_52w_low_lookback': recent_52w_low_lookback,
            'vol_gt_prev': use_vol_gt_prev, 'use_vol': use_vol, 'vol_logic': vol_logic, 'vol_thresh': vol_thresh, 'vol_thresh_max': vol_thresh_max, 'use_vol_rank': use_vol_rank, 'vol_rank_logic': vol_rank_logic, 'vol_rank_thresh': vol_rank_thresh,
            'use_ma_dist_filter': use_ma_dist_filter, 'dist_ma_type': dist_ma_type, 'dist_logic': dist_logic, 'dist_min': dist_min, 'dist_max': dist_max,
            'use_weekly_ma_pullback': use_weekly_ma_pullback, 'wma_type': wma_type, 'wma_period': wma_period, 'wma_min_ext_pct': wma_min_ext_pct, 'wma_lookback_months': wma_lookback_months, 'wma_touch_logic': wma_touch_logic,
            'use_volret_delta': use_volret_delta, 'vrd_method': vrd_method, 'vrd_rank_window': vrd_rank_window, 'vrd_vol_halflife': vrd_vol_halflife, 'vrd_ret_horizon': vrd_ret_horizon, 'vrd_delta_n': vrd_delta_n, 'vrd_min_periods': vrd_min_periods, 'vrd_pctile_min': vrd_pctile_min, 'vrd_pctile_max': vrd_pctile_max,
            'use_tr_vcr_filter': use_tr_vcr_filter, 'tr_vcr_metric': tr_vcr_metric, 'tr_vcr_window': tr_vcr_window, 'tr_vcr_sample_freq': tr_vcr_sample_freq, 'tr_vcr_min_periods': tr_vcr_min_periods, 'tr_vcr_rank_window': tr_vcr_rank_window, 'tr_vcr_filter_mode': tr_vcr_filter_mode, 'tr_vcr_pctile_min': tr_vcr_pctile_min, 'tr_vcr_pctile_max': tr_vcr_pctile_max, 'tr_vcr_raw_min': tr_vcr_raw_min, 'tr_vcr_raw_max': tr_vcr_raw_max, 'tr_vcr_raw_logic': tr_vcr_raw_logic, 'tr_vcr_regime_quadrants': tr_vcr_regime_quadrants, 'tr_vcr_min_consec': tr_vcr_min_consec, 'tr_vcr_consec_first': tr_vcr_consec_first,
            'use_gap_filter': use_gap_filter, 'gap_lookback': gap_lookback, 'gap_logic': gap_logic, 'gap_thresh': gap_thresh,
            'use_earnings_filter': use_earnings_filter, 'earnings_logic': earnings_logic,
            'earnings_value': earnings_value, 'earnings_min': earnings_min, 'earnings_max': earnings_max,
            'use_eps_surp_filter': use_eps_surp_filter, 'eps_surp_logic': eps_surp_logic,
            'eps_surp_min': eps_surp_min, 'eps_surp_max': eps_surp_max,
            'use_rev_surp_filter': use_rev_surp_filter, 'rev_surp_logic': rev_surp_logic,
            'rev_surp_min': rev_surp_min, 'rev_surp_max': rev_surp_max,
            'use_eps_yoy_filter': use_eps_yoy_filter, 'eps_yoy_logic': eps_yoy_logic,
            'eps_yoy_min': eps_yoy_min, 'eps_yoy_max': eps_yoy_max,
            'use_rev_yoy_filter': use_rev_yoy_filter, 'rev_yoy_logic': rev_yoy_logic,
            'rev_yoy_min': rev_yoy_min, 'rev_yoy_max': rev_yoy_max,
            'use_grades_filter': use_grades_filter, 'grades_window_days': grades_window_days,
            'grades_logic': grades_logic, 'grades_thresh': grades_thresh,
            'use_acc_count_filter': use_acc_count_filter, 'acc_count_window': acc_count_window, 'acc_count_logic': acc_count_logic, 'acc_count_thresh': acc_count_thresh,
            'use_dist_count_filter': use_dist_count_filter, 'dist_count_window': dist_count_window, 'dist_count_logic': dist_count_logic, 'dist_count_thresh': dist_count_thresh,
            'use_acc_dist_v2': use_acc_dist_v2,
            'use_t1_open_filter': use_t1_open_filter, 't1_open_filters': t1_open_filters,
            'use_recent_ath': use_recent_ath, 'recent_ath_invert': recent_ath_invert, 'ath_lookback_days': ath_lookback_days,
            'use_ref_ticker_filter': use_ref_ticker_filter, 'ref_ticker': ref_ticker_input, 'ref_filters': ref_filters,
            'use_xsec_filter': use_xsec_filter, 'xsec_filters': xsec_filters,
            'atr_sznl_filters': atr_sznl_filters,
            'dial_filters': dial_filters
        }

        _active_preset = st.session_state.get('_active_preset')
        if _active_preset:
            _ps = _active_preset.get('settings', {})
            _injected = []
            for _k, _v in _ps.items():
                if _k in _USER_ADJUSTABLE_PARAM_KEYS:
                    continue
                params[_k] = _v
                _injected.append(_k)
            st.info(
                f"Preset '{_active_preset['name']}' applied — overrode "
                f"{len(_injected)} filter/condition fields from STRATEGY_BOOK. "
                "Universe / dates / entry / exit / risk inputs from the UI above were kept."
            )
            # Re-derive the exit-mode booleans inside run_engine's view by also
            # honoring the preset's xsec_filters for the cross-sectional rank
            # builder a few lines below (otherwise xsec ranks are skipped when
            # the preset turns the filter on but the UI checkbox is off).
            use_xsec_filter = bool(params.get('use_xsec_filter'))
            xsec_filters = params.get('xsec_filters', []) or []
            atr_sznl_filters = params.get('atr_sznl_filters', []) or []

        # Build cross-sectional rank matrices if enabled
        xsec_rank_matrices = None
        if use_xsec_filter and xsec_filters:
            xsec_windows = list(set(f['window'] for f in xsec_filters))
            st.info(f"Computing cross-sectional ranks ({len(data_dict)} tickers, windows: {xsec_windows})...")
            xsec_rank_matrices = build_xsec_rank_matrices(data_dict, xsec_windows)
            st.success(f"Cross-sectional ranks computed.")

        atr_sznl_map = load_atr_seasonal_map() if atr_sznl_filters else None
        if atr_sznl_filters and not atr_sznl_map:
            st.warning("ATR Seasonal Rank filter is enabled but atr_seasonal_ranks.parquet could not be loaded. Filter will be skipped.")

        fragility_df = load_fragility_dials() if dial_filters else None

        # Earnings calendar — load once and stash in params so the engine can
        # look up per-ticker dates inside its loop without re-reading parquet.
        if use_earnings_filter:
            earnings_map = load_earnings_map()
            if not earnings_map:
                st.warning(
                    "Earnings filter enabled but data/earnings_calendar.parquet "
                    "missing — filter will silently no-op. Run "
                    "`python scripts/build_earnings_calendar.py` to backfill."
                )
            else:
                _n_tkrs = len(earnings_map)
                _n_rows = sum(len(v) for v in earnings_map.values())
                if earnings_logic in ("Between", "Not Between"):
                    _cond_str = f"{earnings_logic} {earnings_min} and {earnings_max}"
                else:
                    _cond_str = f"offset {earnings_logic} {earnings_value}"
                st.info(
                    f"📅 Earnings filter active ({_cond_str} trading days from "
                    f"earnings): {_n_tkrs} tickers, {_n_rows:,} earnings dates loaded."
                )
            params['earnings_map'] = earnings_map

        # Earnings-quality (beat-size + YoY) filters — same parquet, different
        # cached view that exposes the derived metric columns indexed by date.
        _need_metrics = any(
            params.get(k, False) for k in (
                'use_eps_surp_filter', 'use_rev_surp_filter',
                'use_eps_yoy_filter',  'use_rev_yoy_filter',
            )
        )
        if _need_metrics:
            metrics_map = load_earnings_metrics_map()
            if not metrics_map:
                st.warning(
                    "Earnings-quality filter enabled but derived columns not "
                    "found in data/earnings_calendar.parquet — filter will "
                    "silently no-op. Run `python scripts/build_earnings_calendar.py "
                    "--derive-only` to backfill the derived columns."
                )
            else:
                _active = [
                    label for k, label in [
                        ('use_eps_surp_filter', 'EPS surprise'),
                        ('use_rev_surp_filter', 'Rev surprise'),
                        ('use_eps_yoy_filter',  'EPS YoY'),
                        ('use_rev_yoy_filter',  'Rev YoY'),
                    ] if params.get(k, False)
                ]
                st.info(f"📈 Earnings-quality filters active ({', '.join(_active)}): "
                        f"{len(metrics_map)} tickers loaded.")
            params['earnings_metrics_map'] = metrics_map

        if params.get('use_grades_filter', False):
            grades_map = load_grades_map_cached()
            if not grades_map:
                st.warning(
                    "Analyst grades filter enabled but data/analyst_grades.parquet "
                    "is missing — filter will silently no-op. Run "
                    "`python scripts/build_analyst_grades.py` to backfill."
                )
            else:
                _n_evt = sum(len(v) for v in grades_map.values())
                st.info(
                    f"🏷️ Analyst grades filter active ({grades_logic} {grades_thresh} "
                    f"net upgrades over {grades_window_days}d): "
                    f"{len(grades_map)} tickers, {_n_evt:,} events loaded."
                )
            params['grades_map'] = grades_map

        trades_df, rejected_df, total_signals = run_engine(data_dict, params, sznl_map, market_series, vix_series, market_sznl_series, ref_ticker_ranks, xsec_rank_matrices, atr_sznl_map, fragility_df=fragility_df, market_sma_not_declining_series=market_sma_not_declining_series)
        if trades_df.empty: st.warning("No executed signals.")
        if not trades_df.empty:
            trades_df = trades_df.sort_values("ExitDate")

            # Per-trade EffectiveBps: overflow-extras tickers get overflow_bps_input,
            # everyone else gets primary risk_bps_input. Identical to primary
            # when 'All CSV + Overflow Extras*' is not the universe (overflow set empty).
            try:
                _overflow_ticker_set = set(_new_extras) if _is_overflow_universe else set()
            except NameError:
                _overflow_ticker_set = set()
            trades_df['IsOverflow'] = trades_df['Ticker'].isin(_overflow_ticker_set)
            trades_df['EffectiveBps'] = np.where(
                trades_df['IsOverflow'], int(overflow_bps_input), int(risk_bps_input)
            )
            _split_used = trades_df['IsOverflow'].any() and overflow_bps_input != risk_bps_input
            if _split_used:
                _n_of = int(trades_df['IsOverflow'].sum())
                st.info(
                    f"📐 Variable sizing active: {len(trades_df) - _n_of} liquid trades @ {risk_bps_input} bps · "
                    f"{_n_of} overflow trades @ {overflow_bps_input} bps."
                )

            # Per-trade RiskScale: pro-rata down on days where aggregate raw risk
            # would exceed the daily cap. Uses each trade's EffectiveBps so a
            # day mixing 30/10 bps trades caps correctly. Always present (1.0 when off).
            if use_max_daily_risk and risk_bps_input > 0:
                _entry_dt = pd.to_datetime(trades_df['EntryDate'])
                # Per-trade %: EffectiveBps / 100 (bps → %)
                _per_trade_pct = trades_df['EffectiveBps'].astype(float) / 100.0
                _per_trade_pct.index = _entry_dt
                day_raw_pct = _per_trade_pct.groupby(_per_trade_pct.index).sum()
                day_scale = (max_daily_risk_pct / day_raw_pct).clip(upper=1.0)
                trades_df['RiskScale'] = _entry_dt.map(day_scale).astype(float)
                _capped_days = int((day_scale < 1.0).sum())
                _capped_trades = int((trades_df['RiskScale'] < 1.0).sum())
                if _capped_days > 0:
                    _avg_scale = float(trades_df.loc[trades_df['RiskScale'] < 1.0, 'RiskScale'].mean())
                    st.info(f"Daily risk cap engaged on **{_capped_days}** day(s), scaling **{_capped_trades}** trades (avg scale {_avg_scale:.2f}x).")
            else:
                trades_df['RiskScale'] = 1.0

            # Per-trade risk dollars use EffectiveBps (split-aware).
            trades_df['_RiskPerTradeDollar'] = starting_portfolio * trades_df['EffectiveBps'] / 10000.0
            trades_df['PnL_Dollar'] = trades_df['R'] * trades_df['_RiskPerTradeDollar'] * trades_df['RiskScale']
            trades_df = trades_df.drop(columns=['_RiskPerTradeDollar'])
            trades_df['CumPnL'] = trades_df['PnL_Dollar'].cumsum()
            # Build MTM curves (flat + dynamic) before dates get stringified.
            # build_mtm_curves picks up trades_df['EffectiveBps'] when present,
            # so per-trade sizing flows into MTM equity automatically.
            mtm_flat = build_mtm_curves(trades_df, data_dict, starting_portfolio, risk_bps_input, mode='flat')
            mtm_dyn  = build_mtm_curves(trades_df, data_dict, starting_portfolio, risk_bps_input, mode='dynamic')
            portfolio_stats = compute_portfolio_stats(mtm_dyn, starting_portfolio, risk_bps=risk_bps_input, trades_df=trades_df)
            trades_df = compute_trade_path_stats(trades_df, data_dict)
            trades_df['SignalDate'] = pd.to_datetime(trades_df['SignalDate'])
            trades_df['EntryDate'] = pd.to_datetime(trades_df['EntryDate'])
            trades_df['DayOfWeek'] = trades_df['EntryDate'].dt.day_name()
            trades_df['Year'] = trades_df['SignalDate'].dt.year
            trades_df['Month'] = trades_df['SignalDate'].dt.strftime('%b')
            trades_df['SignalDate'] = trades_df['SignalDate'].dt.strftime('%Y-%m-%d')
            trades_df['EntryDate'] = trades_df['EntryDate'].dt.strftime('%Y-%m-%d')
            trades_df['ExitDate'] = pd.to_datetime(trades_df['ExitDate']).dt.strftime('%Y-%m-%d')
            trades_df['CyclePhase'] = trades_df['Year'].apply(get_cycle_year)
            trades_df['AgeBucket'] = trades_df['Age'].apply(get_age_bucket)
            if len(trades_df) >= 10:
                try: trades_df['VolDecile'] = pd.qcut(trades_df['AvgVol'], 10, labels=False, duplicates='drop') + 1
                except: trades_df['VolDecile'] = 1
            else: trades_df['VolDecile'] = 1
            wins = trades_df[trades_df['R'] > 0]
            losses = trades_df[trades_df['R'] <= 0]
            win_rate = len(wins) / len(trades_df) * 100
            pf = wins['PnL_Dollar'].sum() / abs(losses['PnL_Dollar'].sum()) if not losses.empty else 999
            r_series = trades_df['R']
            sqn = np.sqrt(len(trades_df)) * (r_series.mean() / r_series.std()) if len(trades_df) > 1 else 0
            expectancy_r = r_series.mean()
        else:
            win_rate, pf, sqn, expectancy_r = 0, 0, 0, 0
        fill_rate = (len(trades_df) / total_signals * 100) if total_signals > 0 else 0
        grade, verdict, notes = grade_strategy(pf, sqn, win_rate, len(trades_df))
        st.success("Backtest Complete!")
        st.markdown(f"""
        <div style="background-color: #0e1117; padding: 20px; border-radius: 10px; border: 1px solid #444;">
            <h2 style="margin-top:0; color: #ffffff;">Strategy Grade: <span style="color: {'#00ff00' if grade in ['A','B'] else '#ffaa00' if grade=='C' else '#ff0000'};">{grade}</span> ({verdict})</h2>
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                <div><h3>Profit Factor: {pf:.2f}</h3></div>
                <div><h3>SQN: {sqn:.2f}</h3></div>
                <div><h3>Win Rate: {win_rate:.1f}%</h3></div>
                <div><h3>Expectancy: ${trades_df['PnL_Dollar'].mean() if not trades_df.empty else 0:.2f}</h3></div>
            </div>
            <div style="margin-top: 10px; color: #aaa; font-size: 14px;">
               Fill Rate: {fill_rate:.1f}% ({len(trades_df)} executed vs {total_signals} raw signals) | Rejection/Missed: {len(rejected_df)} trades
            </div>
        </div>
        """, unsafe_allow_html=True)
        if notes: st.warning("Notes: " + ", ".join(notes))
        if not trades_df.empty:
            strategy_dict = build_strategy_dict(params, tickers_to_run, pf, sqn, win_rate, expectancy_r)
            with st.expander("Export Strategy Configuration", expanded=False):
                st.markdown("**Copy this dictionary to `strategy_config.py`:**")
                import pprint
                py_code = pprint.pformat(strategy_dict, width=120, sort_dicts=False)
                st.code(py_code, language="python")
                st.download_button("Download as Python", py_code, file_name="strategy_export.py", mime="text/x-python")
        if not trades_df.empty:
            def _mtm_fig(df, title):
                d = df.reset_index()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=d['Date'], y=d['Equity_High'], mode='lines',
                                         line=dict(width=0), hoverinfo='skip', showlegend=False))
                fig.add_trace(go.Scatter(x=d['Date'], y=d['Equity_Low'], mode='lines',
                                         line=dict(width=0), fill='tonexty',
                                         fillcolor='rgba(100,149,237,0.18)',
                                         name='Intraday H/L band', hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=d['Date'], y=d['Equity_Close'], mode='lines',
                                         line=dict(width=1.8, color='rgb(100,149,237)'),
                                         name='Close MTM'))
                fig.update_layout(title=title, yaxis_title='Equity ($)', xaxis_title='Date',
                                  hovermode='x unified', height=420, margin=dict(t=50, b=40, l=40, r=20))
                return fig

            def _edge_over_time_fig(trades, window=None):
                """Rolling mean R + std R — direct view of edge vs variance.

                Two lines, same R-unit scale. The gap between mean R and
                std R is the quality (implied per-trade Sharpe = mean / std);
                each line read separately tells you whether changes are
                driven by edge or by regime variance.

                Window size adapts to total trade count: roughly 10% of
                trades, clamped to [20, 100], so the chart shows ~10 buckets
                across the backtest regardless of strategy frequency.

                Read it as:
                  Mean rising / std flat       -> edge improving cleanly
                  Mean flat / std rising       -> same edge, more variance
                                                  required to capture it
                  Mean falling / std rising    -> unfavorable regime
                  Both falling together        -> trade-size compression
                """
                if trades is None or trades.empty or 'R' not in trades.columns:
                    return None
                tr = trades.copy()
                tr['EntryDate'] = pd.to_datetime(tr['EntryDate'])
                tr = tr.sort_values('EntryDate').reset_index(drop=True)
                if len(tr) < 20:
                    return None
                if window is None:
                    window = max(20, min(100, len(tr) // 10))
                rolling_mean = tr['R'].rolling(window, min_periods=window).mean()
                rolling_std = tr['R'].rolling(window, min_periods=window).std()
                avg_r = float(tr['R'].mean())
                avg_std = float(tr['R'].std())

                fig = go.Figure()
                # Std R band: lighter, on top — represents the "variance budget"
                fig.add_trace(go.Scatter(
                    x=tr['EntryDate'], y=rolling_std,
                    mode='lines', line=dict(width=1.4, color='rgba(200,200,200,0.85)', dash='dot'),
                    name=f'Std R ({window}-trade)',
                    hovertemplate='%{x|%Y-%m-%d}: %{y:.2f}r<extra></extra>',
                ))
                # Mean R: solid bright orange — the edge
                fig.add_trace(go.Scatter(
                    x=tr['EntryDate'], y=rolling_mean,
                    mode='lines', line=dict(width=1.8, color='rgb(255,176,0)'),
                    name=f'Mean R ({window}-trade)',
                    hovertemplate='%{x|%Y-%m-%d}: %{y:.3f}r<extra></extra>',
                ))
                fig.add_hline(y=avg_r, line_dash='dash', line_color='rgba(255,176,0,0.5)',
                              line_width=1, annotation_text=f'All-trade mean: {avg_r:.2f}r',
                              annotation_position='top right', annotation_font_size=10)
                fig.add_hline(y=avg_std, line_dash='dash', line_color='rgba(200,200,200,0.5)',
                              line_width=1, annotation_text=f'All-trade std: {avg_std:.2f}r',
                              annotation_position='bottom right', annotation_font_size=10)
                fig.add_hline(y=0, line_dash='dot', line_color='#888', line_width=1)
                fig.update_layout(
                    title=(f'Edge vs Variance Over Time — Rolling {window}-Trade '
                           f'(N={len(tr)}, mean {avg_r:.2f}r / std {avg_std:.2f}r)'),
                    xaxis_title='Entry Date',
                    yaxis=dict(title='R-multiples (rolling)'),
                    hovermode='x unified', height=260, margin=dict(t=40, b=30, l=40, r=40),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                )
                return fig

            def _trade_mae_scatter_fig(trades, stop_atr_val):
                """Scatter of every trade's max intra-trade drawdown (MAE) in
                ATR-multiples vs exit date. Mirrors `_trade_scatter_fig` styling
                but plots MAE_R * stop_atr — the worst unrealized loss the trade
                ever showed before closing, normalized by ATR.
                """
                if trades is None or trades.empty or 'MAE_R' not in trades.columns:
                    return None
                df_s = trades.copy()
                df_s['ExitDate'] = pd.to_datetime(df_s['ExitDate'])
                stop_atr_mult = float(stop_atr_val) if stop_atr_val else 1.0
                df_s['MAE_ATR'] = df_s['MAE_R'] * stop_atr_mult
                df_s = df_s.dropna(subset=['MAE_ATR'])
                if df_s.empty:
                    return None

                df_s['ATR_Multiple'] = df_s['R'] * stop_atr_mult

                mean_v = float(df_s['MAE_ATR'].mean())
                std_v = float(df_s['MAE_ATR'].std())
                med_v = float(df_s['MAE_ATR'].median())
                colour_cap = max(float(df_s['ATR_Multiple'].abs().quantile(0.95)), 1e-6)

                _abs_mae = df_s['MAE_ATR'].abs()
                _abs_cap = max(float(_abs_mae.quantile(0.95)), 1e-6)
                marker_size = (5.0 + 11.0 * (_abs_mae / _abs_cap).clip(0, 1)).tolist()

                hover_text = df_s.apply(
                    lambda r: (
                        f"<b>{r['Ticker']}</b> ({r.get('Direction', '')})<br>"
                        f"Entry {pd.to_datetime(r['EntryDate']):%Y-%m-%d} @ {r['Entry']:.2f}<br>"
                        f"Exit  {pd.to_datetime(r['ExitDate']):%Y-%m-%d} @ {r['Exit']:.2f}<br>"
                        f"MAE {r['MAE_R']:+.2f}R | ATR-mult {r['MAE_ATR']:+.2f}<br>"
                        f"Realized R {r['R']:+.2f} | ATR-mult {r['ATR_Multiple']:+.2f}<br>"
                        f"Exit type: {r.get('Type', '')}"
                    ),
                    axis=1,
                )

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_s['ExitDate'], y=df_s['MAE_ATR'],
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=df_s['ATR_Multiple'],
                        colorscale='RdYlGn',
                        cmin=-colour_cap, cmax=colour_cap, cmid=0,
                        line=dict(width=0.4, color='rgba(0,0,0,0.35)'),
                        showscale=False,
                    ),
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=False,
                ))
                fig.add_hline(y=0, line_dash='dot', line_color='#888', line_width=1)
                fig.add_hline(y=mean_v, line_dash='dash', line_color='rgba(255,176,0,0.7)',
                              line_width=1, annotation_text=f'Mean {mean_v:+.2f}',
                              annotation_position='top right', annotation_font_size=10)
                fig.add_hline(y=mean_v + std_v, line_dash='dot',
                              line_color='rgba(200,200,200,0.45)', line_width=1)
                fig.add_hline(y=mean_v - std_v, line_dash='dot',
                              line_color='rgba(200,200,200,0.45)', line_width=1)
                title = (
                    f'Per-Trade Max Intra-Trade Drawdown (ATR-multiples) — '
                    f'N={len(df_s)}, mean {mean_v:+.2f}, median {med_v:+.2f}, '
                    f'std {std_v:.2f} (stop_atr={stop_atr_mult:g})'
                )
                fig.update_layout(
                    title=title,
                    xaxis_title='Exit Date',
                    yaxis=dict(title='MAE (ATR-multiples)', zeroline=False),
                    height=320, margin=dict(t=40, b=30, l=40, r=40),
                    hovermode='closest',
                )
                return fig

            def _trade_scatter_fig(trades, stop_atr_val):
                """Scatter of every trade's return in ATR-multiples vs exit date.

                ATR-multiple = R * stop_atr (R is PnL per unit risk, where risk
                is stop_atr * ATR_at_signal, so R * stop_atr collapses back to
                PnL_per_share / ATR_at_signal). The y-axis shows raw
                ATR-normalized returns; markers are colored by sign and sized
                by absolute return so big winners/losers pop visually.
                """
                if trades is None or trades.empty or 'R' not in trades.columns:
                    return None
                df_s = trades.copy()
                df_s['ExitDate'] = pd.to_datetime(df_s['ExitDate'])
                stop_atr_mult = float(stop_atr_val) if stop_atr_val else 1.0
                df_s['ATR_Multiple'] = df_s['R'] * stop_atr_mult
                df_s = df_s.dropna(subset=['ATR_Multiple'])
                if df_s.empty:
                    return None

                mean_v = float(df_s['ATR_Multiple'].mean())
                std_v = float(df_s['ATR_Multiple'].std())
                med_v = float(df_s['ATR_Multiple'].median())
                # Symmetric color range capped at the 95th absolute percentile
                # so a single outlier doesn't wash out the rest of the cloud.
                colour_cap = max(float(df_s['ATR_Multiple'].abs().quantile(0.95)), 1e-6)

                # Marker size scaled by abs return, clipped so outliers don't
                # become giant blobs. 5 = minimum legible, 16 = max.
                _abs_atr = df_s['ATR_Multiple'].abs()
                _abs_cap = max(float(_abs_atr.quantile(0.95)), 1e-6)
                marker_size = (5.0 + 11.0 * (_abs_atr / _abs_cap).clip(0, 1)).tolist()

                hover_text = df_s.apply(
                    lambda r: (
                        f"<b>{r['Ticker']}</b> ({r.get('Direction', '')})<br>"
                        f"Entry {pd.to_datetime(r['EntryDate']):%Y-%m-%d} @ {r['Entry']:.2f}<br>"
                        f"Exit  {pd.to_datetime(r['ExitDate']):%Y-%m-%d} @ {r['Exit']:.2f}<br>"
                        f"R {r['R']:+.2f} | ATR-mult {r['ATR_Multiple']:+.2f}<br>"
                        f"Exit type: {r.get('Type', '')}"
                    ),
                    axis=1,
                )

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_s['ExitDate'], y=df_s['ATR_Multiple'],
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=df_s['ATR_Multiple'],
                        colorscale='RdYlGn',
                        cmin=-colour_cap, cmax=colour_cap, cmid=0,
                        line=dict(width=0.4, color='rgba(0,0,0,0.35)'),
                        showscale=False,
                    ),
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=False,
                ))
                fig.add_hline(y=0, line_dash='dot', line_color='#888', line_width=1)
                fig.add_hline(y=mean_v, line_dash='dash', line_color='rgba(255,176,0,0.7)',
                              line_width=1, annotation_text=f'Mean {mean_v:+.2f}',
                              annotation_position='top right', annotation_font_size=10)
                fig.add_hline(y=mean_v + std_v, line_dash='dot',
                              line_color='rgba(200,200,200,0.45)', line_width=1)
                fig.add_hline(y=mean_v - std_v, line_dash='dot',
                              line_color='rgba(200,200,200,0.45)', line_width=1)
                title = (
                    f'Per-Trade Return Distribution (ATR-multiples) — '
                    f'N={len(df_s)}, mean {mean_v:+.2f}, median {med_v:+.2f}, '
                    f'std {std_v:.2f} (stop_atr={stop_atr_mult:g})'
                )
                fig.update_layout(
                    title=title,
                    xaxis_title='Exit Date',
                    yaxis=dict(title='Return (ATR-multiples)', zeroline=False),
                    height=320, margin=dict(t=40, b=30, l=40, r=40),
                    hovermode='closest',
                )
                return fig

            # ========== SECTION 1: FLAT STAKING ==========
            st.subheader(f"Flat Staking — ${risk_per_trade:,.0f} per trade ({risk_bps_input} bps of ${starting_portfolio:,})")
            if not mtm_flat.empty:
                st.plotly_chart(_mtm_fig(mtm_flat, "Mark-to-Market Equity — Flat Risk"), use_container_width=True)

            st.subheader("Performance Breakdowns (Trade-level, Flat Risk)")
            y1, y2 = st.columns(2)
            y1.plotly_chart(px.bar(trades_df.groupby('Year')['PnL_Dollar'].sum().reset_index(), x='Year', y='PnL_Dollar', title="Net Profit ($) by Year", text_auto='.2s'), use_container_width=True)
            trades_per_year = trades_df.groupby('Year').size().reset_index(name='Count')
            y2.plotly_chart(px.bar(trades_per_year, x='Year', y='Count', title="Total Trades by Year", text_auto=True), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.plotly_chart(px.bar(trades_df.groupby('CyclePhase')['PnL_Dollar'].sum().reset_index().sort_values('CyclePhase'), x='CyclePhase', y='PnL_Dollar', title="PnL by Cycle", text_auto='.2s'), use_container_width=True)
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_pnl = trades_df.groupby("Month")["PnL_Dollar"].sum().reindex(month_order).reset_index()
            c2.plotly_chart(px.bar(monthly_pnl, x="Month", y="PnL_Dollar", title="PnL by Month", text_auto='.2s'), use_container_width=True)
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            dow_pnl = trades_df.groupby("DayOfWeek")["PnL_Dollar"].sum().reindex(dow_order).reset_index()
            c3.plotly_chart(px.bar(dow_pnl, x="DayOfWeek", y="PnL_Dollar", title="PnL by Entry Day", text_auto='.2s'), use_container_width=True)
            ticker_pnl = trades_df.groupby("Ticker")["PnL_Dollar"].sum().reset_index()
            ticker_pnl = ticker_pnl.sort_values("PnL_Dollar", ascending=False).head(75)
            st.plotly_chart(px.bar(ticker_pnl, x="Ticker", y="PnL_Dollar", title="Cumulative PnL by Ticker (Top 75)", text_auto='.2s'), use_container_width=True)

            # ---- Exit-Reason Breakdown ----
            # What's actually closing the trades — stop, target, time, trailing,
            # EOD-DD, etc. Surfaces which exit mechanism is load-bearing for the
            # strategy's edge and which is mostly defensive.
            if 'Type' in trades_df.columns and trades_df['Type'].notna().any():
                _exit = trades_df.dropna(subset=['Type']).copy()
                if 'EntryDate' in _exit.columns and 'ExitDate' in _exit.columns:
                    _hold = (
                        pd.to_datetime(_exit['ExitDate']) - pd.to_datetime(_exit['EntryDate'])
                    ).dt.days
                else:
                    _hold = pd.Series([float('nan')] * len(_exit), index=_exit.index)
                _exit['_HoldDays'] = _hold

                _grp = _exit.groupby('Type', dropna=False).agg(
                    Trades=('R', 'size'),
                    WinRate=('R', lambda s: (s > 0).mean() * 100),
                    AvgR=('R', 'mean'),
                    MedianR=('R', 'median'),
                    TotalR=('R', 'sum'),
                    AvgHoldDays=('_HoldDays', 'mean'),
                    PnL_Dollar=('PnL_Dollar', 'sum'),
                ).reset_index()
                _grp['PctTrades'] = _grp['Trades'] / _grp['Trades'].sum() * 100
                _grp['PctTotalR'] = _grp['TotalR'] / _grp['TotalR'].sum() * 100 if _grp['TotalR'].sum() != 0 else 0.0
                _grp = _grp.sort_values('TotalR', ascending=False).reset_index(drop=True)

                st.markdown("---")
                st.subheader("Exit-Reason Breakdown")
                ec1, ec2 = st.columns([3, 2])
                with ec1:
                    _disp = _grp[['Type', 'Trades', 'PctTrades', 'WinRate', 'AvgR', 'MedianR', 'TotalR', 'PctTotalR', 'AvgHoldDays', 'PnL_Dollar']].copy()
                    _disp.columns = ['Exit Type', 'N', '% of N', 'Win %', 'Avg R', 'Median R', 'Total R', '% Total R', 'Avg Hold (d)', 'PnL $']
                    st.dataframe(
                        _disp.style.format({
                            'N': '{:,.0f}',
                            '% of N': '{:.1f}%',
                            'Win %': '{:.1f}%',
                            'Avg R': '{:+.2f}',
                            'Median R': '{:+.2f}',
                            'Total R': '{:+.1f}',
                            '% Total R': '{:+.1f}%',
                            'Avg Hold (d)': '{:.1f}',
                            'PnL $': '${:,.0f}',
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
                with ec2:
                    _bar = _grp.copy()
                    _bar['_color'] = ['#6bcf7f' if v >= 0 else '#ff6b6b' for v in _bar['TotalR']]
                    _fig_ex = go.Figure(go.Bar(
                        x=_bar['TotalR'],
                        y=_bar['Type'],
                        orientation='h',
                        marker_color=_bar['_color'],
                        text=[f"{n:,.0f} trades" for n in _bar['Trades']],
                        textposition='auto',
                    ))
                    _fig_ex.update_layout(
                        title='Total R Contribution by Exit Type',
                        xaxis_title='Total R',
                        yaxis=dict(autorange='reversed'),
                        height=max(220, 40 * len(_bar) + 80),
                        margin=dict(t=40, b=30, l=10, r=10),
                    )
                    st.plotly_chart(_fig_ex, use_container_width=True)
                st.caption(
                    "Where the edge actually comes from. Positive Total R from `Target` "
                    "means the target is doing the work; positive `Time` means winners "
                    "are slow grinders; large negative `Stop` is the defensive cost. "
                    "Avg Hold (d) flags exits firing earlier or later than intended."
                )

            # ========== SECTION 2: DYNAMIC SIZING (COMPOUNDING) ==========
            st.markdown("---")
            st.subheader(f"Dynamic Sizing — {risk_bps_input} bps of running equity (starting ${starting_portfolio:,})")
            if not mtm_dyn.empty:
                st.plotly_chart(_mtm_fig(mtm_dyn, "Mark-to-Market Equity — Dynamic Risk (Compounded)"), use_container_width=True)
                _scatter_fig = _trade_scatter_fig(trades_df, params.get('stop_atr', 1.0))
                if _scatter_fig is not None:
                    st.plotly_chart(_scatter_fig, use_container_width=True)
                _mae_scatter_fig = _trade_mae_scatter_fig(trades_df, params.get('stop_atr', 1.0))
                if _mae_scatter_fig is not None:
                    st.plotly_chart(_mae_scatter_fig, use_container_width=True)
                _edge_fig = _edge_over_time_fig(trades_df)
                if _edge_fig is not None:
                    st.plotly_chart(_edge_fig, use_container_width=True)

                ps = portfolio_stats
                dd_color = '#00ff00' if ps['MaxDD_Pct'] > -15 else '#ffaa00' if ps['MaxDD_Pct'] > -30 else '#ff0000'
                dd_low_color = '#00ff00' if ps['MaxDD_Low_Pct'] > -20 else '#ffaa00' if ps['MaxDD_Low_Pct'] > -35 else '#ff0000'
                sh_color = '#00ff00' if ps['Sharpe'] >= 1.0 else '#ffaa00' if ps['Sharpe'] >= 0.5 else '#ff0000'
                sha_color = '#00ff00' if ps['Sharpe_Active'] >= 1.0 else '#ffaa00' if ps['Sharpe_Active'] >= 0.5 else '#ff0000'

                # Build underwater / recovery display strings
                _pd_str = ps['PeakDate'].strftime('%Y-%m-%d') if ps.get('PeakDate') is not None else '—'
                _td_str = ps['TroughDate'].strftime('%Y-%m-%d') if ps.get('TroughDate') is not None else '—'
                if ps.get('DDStillOngoing'):
                    _rd_str = f"<span style='color:#ffaa00;'>still underwater</span>"
                elif ps.get('RecoveryDate') is not None:
                    _rd_str = ps['RecoveryDate'].strftime('%Y-%m-%d')
                else:
                    _rd_str = '—'
                _uw_days = ps.get('UnderwaterDays', 0)
                _uw_trades = ps.get('TradesDuringDD')
                _uw_trades_str = f"{_uw_trades:,}" if _uw_trades is not None else '—'

                st.markdown(f"""
                <div style="background-color: #0e1117; padding: 20px; border-radius: 10px; border: 1px solid #444; margin-top: 10px;">
                    <h3 style="margin-top:0; color:#ffffff;">Portfolio Stats (Dynamic, MTM)</h3>
                    <div style="display: flex; flex-wrap: wrap; gap: 24px;">
                        <div><strong>Final Equity:</strong> ${ps['FinalEquity']:,.0f}</div>
                        <div><strong>Total Return:</strong> {ps['TotalReturn_Pct']:.1f}%</div>
                        <div><strong>CAGR:</strong> {ps['CAGR_Pct']:.2f}%</div>
                        <div><strong>Calmar:</strong> {ps['Calmar']:.2f}</div>
                        <div><strong>Parkinson Vol (ann.):</strong> {ps['ParkinsonVol_Pct']:.2f}%</div>
                        <div><strong>Time in Market:</strong> {ps['TimeInMarket_Pct']:.1f}%</div>
                    </div>
                    <div style="margin-top: 14px; padding-top: 12px; border-top: 1px dashed #333; display: flex; flex-wrap: wrap; gap: 24px;">
                        <div>
                            <div style="color:#aaa; font-size:11px;">CALENDAR (standalone view)</div>
                            <div><strong style="color:{sh_color};">Sharpe:</strong> {ps['Sharpe']:.2f} &nbsp; <strong>Sortino:</strong> {ps['Sortino']:.2f}</div>
                        </div>
                        <div>
                            <div style="color:#aaa; font-size:11px;">ACTIVE-PERIOD (ensemble-relevant)</div>
                            <div><strong style="color:{sha_color};">Sharpe:</strong> {ps['Sharpe_Active']:.2f} &nbsp; <strong>Sortino:</strong> {ps['Sortino_Active']:.2f}</div>
                        </div>
                    </div>
                    <div style="margin-top: 14px; padding-top: 12px; border-top: 1px dashed #333; display: flex; flex-wrap: wrap; gap: 24px;">
                        <div>
                            <div style="color:#aaa; font-size:11px;">MAX DD (close-to-close)</div>
                            <div><strong style="color:{dd_color};">{ps['MaxDD_Pct']:.2f}%</strong> &nbsp; / &nbsp; <strong>{ps['MaxDD_R']:.1f} R</strong></div>
                        </div>
                        <div>
                            <div style="color:#aaa; font-size:11px;">MAX DD @ LOWS (worst intraday mark)</div>
                            <div><strong style="color:{dd_low_color};">{ps['MaxDD_Low_Pct']:.2f}%</strong> &nbsp; / &nbsp; <strong>{ps['MaxDD_Low_R']:.1f} R</strong></div>
                        </div>
                        <div>
                            <div style="color:#aaa; font-size:11px;">UNDERWATER</div>
                            <div><strong>{_uw_days:,}</strong> calendar days &nbsp; / &nbsp; <strong>{_uw_trades_str}</strong> trades</div>
                        </div>
                        <div>
                            <div style="color:#aaa; font-size:11px;">PEAK → TROUGH → RECOVERY</div>
                            <div style="font-size:13px;">{_pd_str} → {_td_str} → {_rd_str}</div>
                        </div>
                    </div>
                    <div style="margin-top: 10px; color:#aaa; font-size: 13px;">
                        Calendar Sharpe/Sortino use all trading days (idle days = 0 return). Active-period uses only days with an open position — more relevant when this strategy is one block in a multi-strategy book. Relationship: Sharpe_calendar ≈ Sharpe_active × √(TIM). Max DD @ Lows uses the intraday low envelope vs close-based running peak — reflects the worst mark you actually felt. R-units use starting-equity basis: DD_R = |DD_pct| / risk_bps × 100 (so 4% DD at 25 bps = 16 R). Underwater = days & trades between the pre-trough peak and the eventual recovery to new highs.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ---- Bootstrap Confidence Intervals (trade-level) ----
            # Resample the per-trade R series with replacement N times to build
            # a distribution of each headline stat. The 5th/95th percentile band
            # tells you the range of "true" values the historical sample is
            # consistent with — wide bands = small sample / not yet trustworthy.
            _R_vals = trades_df['R'].dropna().to_numpy(dtype=float) if 'R' in trades_df.columns else np.array([])
            if len(_R_vals) >= 30:
                st.markdown("---")
                st.subheader("Bootstrap Confidence Intervals (1000 resamples)")

                _N_BOOT = 1000
                _rng = np.random.default_rng(42)
                _n = len(_R_vals)
                _idx = _rng.integers(0, _n, size=(_N_BOOT, _n))
                _samples = _R_vals[_idx]  # (B, n)

                _wr_b = (_samples > 0).mean(axis=1) * 100.0
                _avgR_b = _samples.mean(axis=1)
                _totR_b = _samples.sum(axis=1)
                _std_b = _samples.std(axis=1, ddof=1)
                _sharpe_b = np.where(_std_b > 0, _avgR_b / _std_b, np.nan)
                _gains_b = np.where(_samples > 0, _samples, 0).sum(axis=1)
                _losses_b = np.where(_samples < 0, _samples, 0).sum(axis=1)
                _pf_b = np.where(_losses_b < 0, _gains_b / -_losses_b, np.nan)
                # Random-order MaxDD: resampled cumulative R curve drawdown.
                _cum_b = np.cumsum(_samples, axis=1)
                _running_max = np.maximum.accumulate(_cum_b, axis=1)
                _maxDD_b = (_cum_b - _running_max).min(axis=1)

                _pt_wr = (_R_vals > 0).mean() * 100.0
                _pt_avgR = _R_vals.mean()
                _pt_totR = _R_vals.sum()
                _pt_std = _R_vals.std(ddof=1)
                _pt_sharpe = _pt_avgR / _pt_std if _pt_std > 0 else float('nan')
                _pt_gains = _R_vals[_R_vals > 0].sum()
                _pt_losses = _R_vals[_R_vals < 0].sum()
                _pt_pf = (_pt_gains / -_pt_losses) if _pt_losses < 0 else float('nan')
                # Historical-order MaxDD (matches the actual equity path users felt)
                _cum_hist = np.cumsum(_R_vals)
                _pt_maxDD = (_cum_hist - np.maximum.accumulate(_cum_hist)).min()

                _ci_rows = [
                    ('Win Rate (%)',     _pt_wr,     _wr_b,     '{:.1f}%',   '{:.1f}%'),
                    ('Avg R per Trade',  _pt_avgR,   _avgR_b,   '{:+.3f}',   '{:+.3f}'),
                    ('Sharpe per Trade', _pt_sharpe, _sharpe_b, '{:.3f}',    '{:.3f}'),
                    ('Profit Factor',    _pt_pf,     _pf_b,     '{:.2f}',    '{:.2f}'),
                    ('Total R',          _pt_totR,   _totR_b,   '{:+.1f}',   '{:+.1f}'),
                    ('Max DD (R, random order)', _pt_maxDD, _maxDD_b, '{:+.1f}', '{:+.1f}'),
                ]

                _ci_table = []
                for _name, _point, _arr, _pfmt, _bfmt in _ci_rows:
                    _arr = _arr[~np.isnan(_arr)]
                    if len(_arr) == 0:
                        continue
                    _p05, _p50, _p95 = np.percentile(_arr, [5, 50, 95])
                    _row = {
                        'Stat': _name,
                        'Point': _pfmt.format(_point) if not np.isnan(_point) else '—',
                        'p5':    _bfmt.format(_p05),
                        'Median': _bfmt.format(_p50),
                        'p95':   _bfmt.format(_p95),
                        '90% CI Width': _bfmt.format(_p95 - _p05),
                    }
                    _ci_table.append(_row)

                bc1, bc2 = st.columns([3, 2])
                with bc1:
                    st.dataframe(pd.DataFrame(_ci_table), use_container_width=True, hide_index=True)
                with bc2:
                    _sharpe_arr_clean = _sharpe_b[~np.isnan(_sharpe_b)]
                    if len(_sharpe_arr_clean) > 0:
                        _sp_p05, _sp_p95 = np.percentile(_sharpe_arr_clean, [5, 95])
                        _fig_hist = go.Figure(go.Histogram(
                            x=_sharpe_arr_clean,
                            nbinsx=40,
                            marker_color='#6bcf7f' if _pt_sharpe > 0 else '#ff6b6b',
                            opacity=0.75,
                        ))
                        _fig_hist.add_vline(x=_pt_sharpe, line_color='#ffaa00', line_width=2,
                                            annotation_text=f"Point {_pt_sharpe:.2f}", annotation_position='top')
                        _fig_hist.add_vline(x=_sp_p05, line_dash='dot', line_color='#888',
                                            annotation_text=f"p5 {_sp_p05:.2f}", annotation_position='bottom left')
                        _fig_hist.add_vline(x=_sp_p95, line_dash='dot', line_color='#888',
                                            annotation_text=f"p95 {_sp_p95:.2f}", annotation_position='bottom right')
                        _fig_hist.add_vline(x=0, line_color='#444', line_width=1)
                        _fig_hist.update_layout(
                            title=f'Bootstrap distribution: Sharpe per Trade (N={_n:,} trades)',
                            xaxis_title='Sharpe per Trade',
                            yaxis_title='Count',
                            height=320, margin=dict(t=40, b=30, l=40, r=10),
                            showlegend=False,
                        )
                        st.plotly_chart(_fig_hist, use_container_width=True)
                st.caption(
                    "Trade-level bootstrap with replacement (1000 resamples, IID assumption). "
                    "Point = actual sample statistic; p5/p95 = 90% confidence band on the 'true' "
                    "value the sample is consistent with. Tight band = enough trades to trust the "
                    "edge. Wide band crossing zero = sample too small or noisy to distinguish from "
                    "no-edge. Max DD here uses random-order resamples and is generally MORE optimistic "
                    "than the historical MTM Max DD above (which preserves serial dependence and "
                    "regime clustering)."
                )

            # ========== SECTION 3: INTRA-TRADE PATH ANALYSIS ==========
            if 'MAE_R' in trades_df.columns and trades_df['MAE_R'].notna().any():
                st.markdown("---")
                st.subheader("Intra-Trade Path (MAE / MFE / Give-back)")

                path = trades_df.dropna(subset=['MAE_R', 'MFE_R']).copy()
                wins_p = path[path['R'] > 0]
                losses_p = path[path['R'] <= 0]

                def _avg(s): return s.mean() if len(s) else 0.0

                avg_mae_all = _avg(path['MAE_R'])
                avg_mae_win = _avg(wins_p['MAE_R'])
                avg_mae_los = _avg(losses_p['MAE_R'])
                avg_mfe_all = _avg(path['MFE_R'])
                avg_mfe_win = _avg(wins_p['MFE_R'])
                avg_mfe_los = _avg(losses_p['MFE_R'])
                avg_gb_win  = _avg(wins_p['GiveBack_R']) if len(wins_p) else 0.0
                # Capture efficiency: total R on winners / total MFE on winners
                mfe_sum = wins_p['MFE_R'].sum()
                capture_eff = (wins_p['R'].sum() / mfe_sum * 100) if mfe_sum > 0 else 0.0

                st.markdown(f"""
                <div style="background-color: #0e1117; padding: 18px; border-radius: 10px; border: 1px solid #444; margin-top: 6px;">
                    <div style="display: flex; flex-wrap: wrap; gap: 28px;">
                        <div>
                            <div style="color:#aaa; font-size:12px;">AVG MAE (R)</div>
                            <div style="color:#ff6b6b;"><strong>All:</strong> {avg_mae_all:.2f} &nbsp; <strong>Win:</strong> {avg_mae_win:.2f} &nbsp; <strong>Loss:</strong> {avg_mae_los:.2f}</div>
                        </div>
                        <div>
                            <div style="color:#aaa; font-size:12px;">AVG MFE (R)</div>
                            <div style="color:#6bcf7f;"><strong>All:</strong> {avg_mfe_all:.2f} &nbsp; <strong>Win:</strong> {avg_mfe_win:.2f} &nbsp; <strong>Loss:</strong> {avg_mfe_los:.2f}</div>
                        </div>
                        <div>
                            <div style="color:#aaa; font-size:12px;">GIVE-BACK (winners, MFE − R)</div>
                            <div><strong>{avg_gb_win:.2f} R</strong> avg per winner</div>
                        </div>
                        <div>
                            <div style="color:#aaa; font-size:12px;">CAPTURE EFFICIENCY (winners)</div>
                            <div><strong>{capture_eff:.1f}%</strong> of MFE locked in</div>
                        </div>
                        <div>
                            <div style="color:#aaa; font-size:12px;">TRADES ANALYZED</div>
                            <div><strong>{len(path):,}</strong> ({len(wins_p):,} W / {len(losses_p):,} L)</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Build binned bar charts
                mae_edges = [-float('inf'), -2.0, -1.0, -0.5, -0.25, 0.0001]
                mae_labels = ['≤ -2R', '-2 to -1R', '-1 to -0.5R', '-0.5 to -0.25R', '0 to -0.25R']
                mfe_edges = [-0.0001, 0.25, 0.5, 1.0, 2.0, float('inf')]
                mfe_labels = ['0 to 0.25R', '0.25 to 0.5R', '0.5 to 1R', '1 to 2R', '≥ 2R']

                path['MAE_Bucket'] = pd.cut(path['MAE_R'], bins=mae_edges, labels=mae_labels, include_lowest=True)
                path['MFE_Bucket'] = pd.cut(path['MFE_R'], bins=mfe_edges, labels=mfe_labels, include_lowest=True)

                mae_group = path.groupby('MAE_Bucket', observed=True).agg(
                    avg_R=('R', 'mean'),
                    win_rate=('R', lambda s: (s > 0).mean() * 100),
                    count=('R', 'size'),
                ).reindex(mae_labels).fillna(0).reset_index()
                mfe_group = path.groupby('MFE_Bucket', observed=True).agg(
                    avg_R=('R', 'mean'),
                    win_rate=('R', lambda s: (s > 0).mean() * 100),
                    count=('R', 'size'),
                ).reindex(mfe_labels).fillna(0).reset_index()

                def _dual_axis_bar(df, x_col, title, x_title):
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df[x_col], y=df['avg_R'], name='Avg Final R',
                        marker_color=['#ff6b6b' if v < 0 else '#6bcf7f' for v in df['avg_R']],
                        text=[f"n={int(c)}" for c in df['count']], textposition='outside',
                    ))
                    fig.add_trace(go.Scatter(
                        x=df[x_col], y=df['win_rate'], name='Win Rate (%)',
                        mode='lines+markers', yaxis='y2',
                        line=dict(color='rgba(200,200,200,0.9)', width=2, dash='dot'),
                        marker=dict(size=8),
                    ))
                    fig.update_layout(
                        title=title, xaxis_title=x_title,
                        yaxis=dict(title='Avg Final R', zeroline=True, zerolinecolor='rgba(255,255,255,0.3)'),
                        yaxis2=dict(title='Win Rate (%)', overlaying='y', side='right', range=[0, 100]),
                        height=380, margin=dict(t=60, b=60, l=50, r=50),
                        legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'),
                    )
                    return fig

                colA, colB = st.columns(2)
                colA.plotly_chart(
                    _dual_axis_bar(mae_group, 'MAE_Bucket',
                                   "Did trades that suffered more still tend to win?",
                                   'MAE bucket (worst intra-trade drawdown, R)'),
                    use_container_width=True,
                )
                colB.plotly_chart(
                    _dual_axis_bar(mfe_group, 'MFE_Bucket',
                                   "Did trades that ran further close strong, or fade?",
                                   'MFE bucket (best intra-trade run, R)'),
                    use_container_width=True,
                )

                st.caption(
                    "MAE/MFE computed from daily H/L. Entry day is pessimistic — only adverse "
                    "excursion counts (we don't know where in the bar our fill happened); MOC entries "
                    "skip entry day entirely. Exit day uses full H/L, which can slightly overstate MFE "
                    "if the peak printed after the exit fill. Intraday ordering within a bar is unknown, "
                    "so a trade that got stopped intraday and recovered to close profitable cannot be "
                    "detected at this resolution."
                )
        st.subheader("Trade Logs")
        tab1, tab2 = st.tabs(["Executed Trades", "Missed Trades"])
        with tab1:
            if not trades_df.empty:
                st.dataframe(trades_df.style.format({"Entry": "{:.2f}", "Exit": "{:.2f}", "R": "{:.2f}", "PnL_Dollar": "${:,.2f}", "Age": "{:.1f}y", "AvgVol": "{:,.0f}"}), use_container_width=True)
            else: st.info("No Executed Trades.")
        with tab2:
            if not rejected_df.empty:
                if 'PnL_Dollar' not in rejected_df.columns: rejected_df['PnL_Dollar'] = rejected_df['R'] * risk_per_trade
                st.dataframe(rejected_df.style.format({"Entry": "{:.2f}", "Exit": "{:.2f}", "R": "{:.2f}", "PnL_Dollar": "${:,.2f}", "Age": "{:.1f}y", "AvgVol": "{:,.0f}"}), use_container_width=True)
            else: st.info("No signals were rejected.")

if __name__ == "__main__":
    main()
