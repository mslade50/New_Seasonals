"""
Walk-Forward Validation & Strategy Decay Analysis
===================================================
Anchored walk-forward: fixed parameters, non-overlapping time folds.
Tests whether your edge is consistent across market regimes — NOT a
parameter optimization tool.

Commission Model: IBKR Pro Fixed ($1.00/order min, $0.005/share, +regulatory)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import time
import copy
import sys
import os

# ---------------------------------------------------------------------------
# IMPORT STRATEGY BOOK
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from strategy_config import _STRATEGY_BOOK_RAW, ACCOUNT_VALUE
except ImportError:
    st.error("❌ Could not import from strategy_config.py")
    _STRATEGY_BOOK_RAW = []
    ACCOUNT_VALUE = 750000

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
PRIMARY_SZNL_PATH = os.path.join(parent_dir, "sznl_ranks.csv")
BACKUP_SZNL_PATH = os.path.join(parent_dir, "seasonal_ranks.csv")

# Default 5 folds spanning full history
DEFAULT_FOLDS = [
    ("2000-01-01", "2004-12-31", "2000–2004"),
    ("2005-01-01", "2009-12-31", "2005–2009"),
    ("2010-01-01", "2014-12-31", "2010–2014"),
    ("2015-01-01", "2019-12-31", "2015–2019"),
    ("2020-01-01", "2024-12-31", "2020–2024"),
]


# ============================================================================
# COMMISSION MODEL — IBKR Pro Fixed
# ============================================================================

def calculate_commission_rt(shares: int) -> float:
    """
    Round-trip commission for IBKR Pro Fixed pricing.
    
    Per side:
      - Base: max($1.00, shares × $0.005)
      - Regulatory (SEC + FINRA + exchange): ~$0.003/share
    
    Returns total round-trip cost in dollars.
    """
    per_side_base = max(1.00, shares * 0.005)
    per_side_reg = shares * 0.003
    per_side = per_side_base + per_side_reg
    return per_side * 2  # entry + exit


def commission_as_r_fraction(shares: int, risk_dollars: float) -> float:
    """Commission as a fraction of risk (for R-multiple adjustment)."""
    if risk_dollars <= 0:
        return 0.0
    return calculate_commission_rt(shares) / risk_dollars


# ============================================================================
# DATA LOADING (mirrors strat_backtester patterns)
# ============================================================================

@st.cache_resource
def load_seasonal_map():
    """Load seasonal rank data from CSV files."""
    def _load(path):
        try:
            df = pd.read_csv(path)
            if 'ticker' not in df.columns or 'Date' not in df.columns:
                return pd.DataFrame()
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            df = df.dropna(subset=["Date", "ticker"])
            df["Date"] = df["Date"].dt.normalize()
            df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
            return df
        except Exception:
            return pd.DataFrame()

    df1 = _load(PRIMARY_SZNL_PATH)
    df2 = _load(BACKUP_SZNL_PATH)

    if df1.empty and df2.empty:
        return {}
    final = pd.concat([df1, df2], axis=0).drop_duplicates(subset=['ticker', 'Date'], keep='first')
    final = final.sort_values("Date")

    output = {}
    for ticker, group in final.groupby("ticker"):
        output[ticker] = group.set_index("Date")["seasonal_rank"]
    return output


def get_sznl_val_series(ticker, dates, sznl_map):
    ticker = ticker.upper()
    t_series = sznl_map.get(ticker)
    if t_series is None and ticker == "^GSPC":
        t_series = sznl_map.get("SPY")
    if t_series is None:
        return pd.Series(50.0, index=dates)
    return dates.map(t_series).fillna(50.0)


@st.cache_data(show_spinner=False)
def download_historical_data(tickers, start_date="1999-01-01"):
    """Download price data for a list of tickers, batched."""
    if not tickers:
        return {}
    clean_tickers = list(set([str(t).strip().upper().replace('.', '-') for t in tickers]))
    data_dict = {}
    CHUNK = 50

    for i in range(0, len(clean_tickers), CHUNK):
        chunk = clean_tickers[i:i + CHUNK]
        try:
            df = yf.download(chunk, start=start_date, group_by='ticker',
                             auto_adjust=True, progress=False, threads=True)
            if df.empty:
                continue
            if len(chunk) == 1:
                t = chunk[0]
                if 'Close' in df.columns:
                    df.index = df.index.tz_localize(None)
                    data_dict[t] = df
            else:
                avail = df.columns.levels[0]
                for t in avail:
                    try:
                        tdf = df[t].copy()
                        if tdf.empty or 'Close' not in tdf.columns:
                            continue
                        tdf.index = tdf.index.tz_localize(None)
                        data_dict[t] = tdf
                    except Exception:
                        continue
            time.sleep(0.25)
        except Exception as e:
            st.warning(f"Batch download error: {e}")
    return data_dict


# ============================================================================
# INDICATOR ENGINE (mirrors strat_backtester / daily_scan)
# ============================================================================

def calculate_indicators(df, sznl_map, ticker, market_series=None, vix_series=None,
                         gap_window=21, acc_window=None, dist_window=None,
                         custom_sma_lengths=None):
    """
    Vectorised indicator computation. Kept in sync with strat_backtester.py.
    """
    df = df.copy()

    # --- Flatten MultiIndex (yfinance trap) ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    n = len(df)

    # --- Moving Averages ---
    df['SMA10'] = close.rolling(10).mean()
    df['SMA20'] = close.rolling(20).mean()
    df['SMA50'] = close.rolling(50).mean()
    df['SMA100'] = close.rolling(100).mean()
    df['SMA200'] = close.rolling(200).mean()

    if custom_sma_lengths:
        for length in custom_sma_lengths:
            col = f"SMA{length}"
            if col not in df.columns:
                df[col] = close.rolling(length).mean()

    df['EMA8'] = close.ewm(span=8, adjust=False).mean()
    df['EMA11'] = close.ewm(span=11, adjust=False).mean()
    df['EMA21'] = close.ewm(span=21, adjust=False).mean()

    # --- Performance Ranks (expanding — intentional, see design notes) ---
    for window in [2, 5, 10, 21]:
        df[f'ret_{window}d'] = close.pct_change(window, fill_method=None)
        df[f'rank_ret_{window}d'] = df[f'ret_{window}d'].expanding(min_periods=252).rank(pct=True) * 100.0

    # --- ATR (14-period) ---
    hl = high - low
    hc = np.abs(high - close.shift())
    lc = np.abs(low - close.shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['ATR_Pct'] = (df['ATR'] / close) * 100
    df['Change_in_ATR'] = (close - close.shift(1)) / df['ATR']

    # --- Volume metrics ---
    vol_ma = volume.rolling(63).mean()
    df['vol_ratio'] = volume / vol_ma
    df['vol_ma'] = vol_ma

    vol_spike = (volume > vol_ma) & (volume > volume.shift(1))
    df['Vol_Spike'] = vol_spike

    green = close > df['Open']
    red = close < df['Open']

    acc_win = acc_window or 21
    dist_win = dist_window or 21
    df['AccCount_21'] = (vol_spike & green).astype(int).rolling(acc_win).sum()
    df['DistCount_21'] = (vol_spike & red).astype(int).rolling(dist_win).sum()

    vol_ma_10 = volume.rolling(10).mean()
    df['vol_ratio_10d'] = vol_ma_10 / vol_ma
    df['vol_ratio_10d_rank'] = df['vol_ratio_10d'].expanding(min_periods=50).rank(pct=True) * 100.0

    # --- Range % ---
    denom = high - low
    df['RangePct'] = np.where(denom == 0, 0.5, (close - low) / denom)

    # --- Gap Count ---
    is_gap = (low > high.shift(1)).astype(int)
    df['GapCount'] = is_gap.rolling(gap_window).sum()

    # --- Seasonality ---
    df['Sznl'] = get_sznl_val_series(ticker, df.index, sznl_map)
    df['Mkt_Sznl_Ref'] = get_sznl_val_series("^GSPC", df.index, sznl_map)

    # --- Age ---
    if not df.empty:
        df['age_years'] = (df.index - df.index[0]).days / 365.25
    else:
        df['age_years'] = 0.0

    # --- Market Regime ---
    if market_series is not None:
        df['Market_Above_SMA200'] = market_series.reindex(df.index, method='ffill').fillna(False)

    if vix_series is not None:
        df['VIX_Value'] = vix_series.reindex(df.index, method='ffill').fillna(0)
    else:
        df['VIX_Value'] = 0.0

    # --- 52w High/Low ---
    roll_high = high.shift(1).rolling(252).max()
    roll_low = low.shift(1).rolling(252).min()
    df['is_52w_high'] = high > roll_high
    df['is_52w_low'] = low < roll_low

    # --- Day of week ---
    df['DayOfWeekVal'] = df.index.dayofweek

    # --- Pivot Points (for limit entries) ---
    piv = 20
    df['PrevHigh'] = high.shift(1)
    df['PrevLow'] = low.shift(1)

    return df


# ============================================================================
# SIGNAL GENERATION (mirrors strat_backtester.get_historical_mask)
# ============================================================================

def get_historical_mask(df, params, sznl_map, ticker_name="UNK"):
    """
    Build a boolean mask of signal dates for a single ticker/strategy pair.
    Replicates strat_backtester.get_historical_mask logic.
    """
    n = len(df)
    conditions = []
    direction = params.get('trade_direction', 'Long')

    # --- Trend Filter ---
    trend = params.get('trend_filter', 'None')
    if trend == "Price > 200 SMA":
        conditions.append(df['Close'].values > df['SMA200'].values)
    elif trend == "Price > Rising 200 SMA":
        conditions.append((df['Close'].values > df['SMA200'].values) &
                          (df['SMA200'].values > df['SMA200'].shift(1).values))
    elif trend == "Not Below Declining 200 SMA":
        conditions.append(~((df['Close'].values < df['SMA200'].values) &
                            (df['SMA200'].values < df['SMA200'].shift(1).values)))
    elif trend == "Price < 200 SMA":
        conditions.append(df['Close'].values < df['SMA200'].values)
    elif trend == "Price < Falling 200 SMA":
        conditions.append((df['Close'].values < df['SMA200'].values) &
                          (df['SMA200'].values < df['SMA200'].shift(1).values))
    elif "Market" in trend and 'Market_Above_SMA200' in df.columns:
        if trend == "Market > 200 SMA":
            conditions.append(df['Market_Above_SMA200'].values)
        elif trend == "Market < 200 SMA":
            conditions.append(~df['Market_Above_SMA200'].values)

    # --- Liquidity / Age / ATR gates ---
    conditions.append(
        (df['Close'].values >= params.get('min_price', 0)) &
        (df['vol_ma'].values >= params.get('min_vol', 0)) &
        (df['age_years'].values >= params.get('min_age', 0)) &
        (df['age_years'].values <= params.get('max_age', 100)) &
        (df['ATR_Pct'].values >= params.get('min_atr_pct', 0)) &
        (df['ATR_Pct'].values <= params.get('max_atr_pct', 100))
    )

    # --- Breakout Mode ---
    bk = params.get('breakout_mode', 'None')
    if bk == "Close > Prev Day High":
        conditions.append(df['Close'].values > df['High'].shift(1).values)
    elif bk == "Close < Prev Day Low":
        conditions.append(df['Close'].values < df['Low'].shift(1).values)

    # --- Range Filter ---
    if params.get('use_range_filter', False):
        rng = df['RangePct'].values * 100
        conditions.append((rng >= params.get('range_min', 0)) &
                          (rng <= params.get('range_max', 100)))

    # --- Day of Week ---
    if params.get('use_dow_filter', False):
        allowed = params.get('allowed_days', [0, 1, 2, 3, 4])
        conditions.append(np.isin(df['DayOfWeekVal'].values, allowed))

    # --- Cycle Year ---
    if 'allowed_cycles' in params:
        ac = params['allowed_cycles']
        if ac and len(ac) < 4:
            years = df.index.year
            cycle_rem = years % 4
            conditions.append(np.isin(cycle_rem, ac))

    # --- VIX Filter ---
    if params.get('use_vix_filter', False):
        vix = df['VIX_Value'].values
        conditions.append((vix >= params.get('vix_min', 0)) &
                          (vix <= params.get('vix_max', 100)))

    # --- Volume Filters ---
    if params.get('use_vol', False):
        conditions.append(df['vol_ratio'].values > params.get('vol_thresh', 1.5))

    if params.get('use_vol_rank', False):
        vr = df['vol_ratio_10d_rank'].values
        if params['vol_rank_logic'] == '<':
            conditions.append(vr < params['vol_rank_thresh'])
        else:
            conditions.append(vr > params['vol_rank_thresh'])

    # --- Performance Filters ---
    for pf in params.get('perf_filters', []):
        col = f"rank_ret_{pf['window']}d"
        vals = df[col].values
        if pf['logic'] == '<':
            cond = vals < pf['thresh']
        else:
            cond = vals > pf['thresh']
        consec = pf.get('consecutive', 1)
        if consec > 1:
            cond = pd.Series(cond).rolling(consec).sum().values == consec
        conditions.append(cond)

    # NOTE: perf_first_instance for list-based perf_filters is handled in daily_scan.py
    # but NOT in strat_backtester.py's get_historical_mask. We match strat_backtester
    # behavior here for consistency. The legacy use_perf_rank path (below) does handle it.

    # --- Legacy Perf Rank (single filter with first_instance support) ---
    if params.get('use_perf_rank', False):
        col = f"rank_ret_{params['perf_window']}d"
        vals = df[col].values
        if params['perf_logic'] == '<':
            raw = vals < params['perf_thresh']
        else:
            raw = vals > params['perf_thresh']
        consec = params.get('perf_consecutive', 1)
        if consec > 1:
            persist = pd.Series(raw).rolling(consec).sum().values == consec
        else:
            persist = raw
        if params.get('perf_first_instance', False):
            lookback = params.get('perf_lookback', 21)
            ps = pd.Series(persist, index=df.index)
            prev = ps.shift(1).rolling(lookback).sum()
            persist = persist & (prev.values == 0)
        conditions.append(persist)

    # --- MA Consecutive Filters ---
    for f in params.get('ma_consec_filters', []):
        col = f"SMA{f['length']}"
        if col in df.columns:
            if f['logic'] == 'Above':
                cond = df['Close'].values > df[col].values
            else:
                cond = df['Close'].values < df[col].values
            if f['consec'] > 1:
                cond = pd.Series(cond).rolling(f['consec']).sum().values == f['consec']
            conditions.append(cond)

    # --- Seasonality ---
    if params.get('use_sznl', False):
        sznl = df['Sznl'].values
        if params['sznl_logic'] == '<':
            cond_s = sznl < params['sznl_thresh']
        else:
            cond_s = sznl > params['sznl_thresh']
        if params.get('sznl_first_instance', False):
            lb = params.get('sznl_lookback', 21)
            s = pd.Series(cond_s, index=df.index)
            prev = s.shift(1).rolling(lb).sum()
            cond_s = cond_s & (prev.values == 0)
        conditions.append(cond_s)

    if params.get('use_market_sznl', False):
        ms = df['Mkt_Sznl_Ref'].values
        if params['market_sznl_logic'] == '<':
            conditions.append(ms < params['market_sznl_thresh'])
        else:
            conditions.append(ms > params['market_sznl_thresh'])

    # --- 52w Filters ---
    if params.get('use_52w', False):
        if params['52w_type'] == 'New 52w High':
            c52 = df['is_52w_high'].values
        else:
            c52 = df['is_52w_low'].values
        if params.get('52w_first_instance', True):
            lb = params.get('52w_lookback', 21)
            s = pd.Series(c52, index=df.index)
            prev = s.shift(1).rolling(lb).sum()
            c52 = c52 & (prev.values == 0)
        conditions.append(c52)

    if params.get('exclude_52w_high', False):
        conditions.append(~df['is_52w_high'].values)

    # --- Accumulation / Distribution Count ---
    if params.get('use_acc_count_filter', False):
        acc = df['AccCount_21'].values
        t = params['acc_count_thresh']
        if params['acc_count_logic'] == '>':
            conditions.append(acc > t)
        else:
            conditions.append(acc < t)

    if params.get('use_dist_count_filter', False):
        dist = df['DistCount_21'].values
        t = params['dist_count_thresh']
        if params['dist_count_logic'] == '>':
            conditions.append(dist > t)
        else:
            conditions.append(dist < t)

    # --- Gap Count ---
    if params.get('use_gap_filter', False):
        gc = df['GapCount'].values
        t = params['gap_thresh']
        if params['gap_logic'] == '>':
            conditions.append(gc > t)
        else:
            conditions.append(gc < t)

    # --- Combine ---
    if conditions:
        combined = np.ones(n, dtype=bool)
        for c in conditions:
            arr = np.asarray(c)
            arr = np.where(np.isnan(arr.astype(float)), False, arr)
            combined = combined & arr.astype(bool)
        return pd.Series(combined, index=df.index)
    else:
        return pd.Series(True, index=df.index)


# ============================================================================
# TRADE SIMULATOR (simplified for walk-forward: fixed sizing, R-multiples)
# ============================================================================

def simulate_trades(df, signal_mask, settings, execution, account_value):
    """
    Simulate trades for a single ticker against a signal mask.
    
    Uses FIXED position sizing (not MTM compounding) for clean fold comparison.
    Returns a list of trade dicts with R-multiples and commission-adjusted PnL.
    """
    direction = settings.get('trade_direction', 'Long')
    entry_type = settings.get('entry_type', 'Signal Close')
    hold_days = execution.get('hold_days', 5)
    stop_atr = execution.get('stop_atr', 2.0)
    tgt_atr = execution.get('tgt_atr', 5.0)
    use_stop = execution.get('use_stop_loss', True)
    use_target = execution.get('use_take_profit', True)
    slippage_bps = execution.get('slippage_bps', 2)
    risk_bps = execution.get('risk_bps', 35)
    max_one_pos = settings.get('max_one_pos', True)

    # Fixed dollar risk for this strategy
    risk_dollars = account_value * risk_bps / 10000

    signal_dates = signal_mask[signal_mask].index
    trades = []
    last_exit = pd.Timestamp.min

    # Determine entry mode flags
    et_upper = entry_type.upper()
    is_signal_close = 'SIGNAL CLOSE' in et_upper
    is_t1_open = entry_type == 'T+1 Open'
    is_t1_close = entry_type == 'T+1 Close'
    is_limit_open_gtc = 'GTC' in et_upper and 'OPEN' in et_upper
    is_limit_pers = 'LIMIT ORDER' in et_upper and 'PERSISTENT' in et_upper
    is_limit_open_atr = 'LIMIT' in et_upper and 'OPEN' in et_upper and 'ATR' in et_upper and not is_limit_open_gtc

    for sig_date in signal_dates:
        if max_one_pos and sig_date <= last_exit:
            continue

        sig_idx = df.index.get_loc(sig_date)

        # ---- ENTRY LOGIC ----
        entry_price = None
        entry_idx = None
        entry_date = None

        if is_signal_close:
            entry_price = df['Close'].iloc[sig_idx]
            entry_idx = sig_idx
            entry_date = sig_date
        elif is_t1_open:
            if sig_idx + 1 >= len(df):
                continue
            entry_price = df['Open'].iloc[sig_idx + 1]
            entry_idx = sig_idx + 1
            entry_date = df.index[sig_idx + 1]
        elif is_t1_close:
            if sig_idx + 1 >= len(df):
                continue
            entry_price = df['Close'].iloc[sig_idx + 1]
            entry_idx = sig_idx + 1
            entry_date = df.index[sig_idx + 1]
        elif is_limit_open_gtc:
            # GTC limit: anchored to T+1 open, persists across hold window
            if sig_idx + 1 >= len(df):
                continue
            sig_atr = df['ATR'].iloc[sig_idx]
            base = df['Open'].iloc[sig_idx + 1]
            limit_px = (base - sig_atr * 0.5) if direction == 'Long' else (base + sig_atr * 0.5)
            for wi in range(1, hold_days + 1):
                ci = sig_idx + wi
                if ci >= len(df):
                    break
                day_low, day_high, day_open = df['Low'].iloc[ci], df['High'].iloc[ci], df['Open'].iloc[ci]
                if direction == 'Long':
                    if day_open < limit_px:
                        entry_price, entry_idx, entry_date = day_open, ci, df.index[ci]
                        break
                    elif day_low <= limit_px:
                        entry_price, entry_idx, entry_date = limit_px, ci, df.index[ci]
                        break
                else:
                    if day_open > limit_px:
                        entry_price, entry_idx, entry_date = day_open, ci, df.index[ci]
                        break
                    elif day_high >= limit_px:
                        entry_price, entry_idx, entry_date = limit_px, ci, df.index[ci]
                        break
        elif is_limit_pers:
            # Persistent limit anchored to signal close
            atr_mult = 1.0 if '1 ATR' in entry_type else 0.5
            sig_close = df['Close'].iloc[sig_idx]
            sig_atr = df['ATR'].iloc[sig_idx]
            limit_px = (sig_close - sig_atr * atr_mult) if direction == 'Long' else (sig_close + sig_atr * atr_mult)
            for wi in range(1, hold_days + 1):
                ci = sig_idx + wi
                if ci >= len(df):
                    break
                day_low, day_high, day_open = df['Low'].iloc[ci], df['High'].iloc[ci], df['Open'].iloc[ci]
                if direction == 'Long' and day_low <= limit_px:
                    entry_price = day_open if day_open < limit_px else limit_px
                    entry_idx, entry_date = ci, df.index[ci]
                    break
                elif direction == 'Short' and day_high >= limit_px:
                    entry_price = day_open if day_open > limit_px else limit_px
                    entry_idx, entry_date = ci, df.index[ci]
                    break
        elif is_limit_open_atr:
            # One-day limit at open +/- 0.5 ATR
            if sig_idx + 1 >= len(df):
                continue
            sig_atr = df['ATR'].iloc[sig_idx]
            ni = sig_idx + 1
            day_open = df['Open'].iloc[ni]
            limit_px = (day_open - sig_atr * 0.5) if direction == 'Long' else (day_open + sig_atr * 0.5)
            if direction == 'Long' and df['Low'].iloc[ni] <= limit_px:
                entry_price = min(limit_px, day_open) if day_open < limit_px else limit_px
                entry_idx, entry_date = ni, df.index[ni]
            elif direction == 'Short' and df['High'].iloc[ni] >= limit_px:
                entry_price = max(limit_px, day_open) if day_open > limit_px else limit_px
                entry_idx, entry_date = ni, df.index[ni]
        else:
            # Default: T+1 Open
            if sig_idx + 1 >= len(df):
                continue
            entry_price = df['Open'].iloc[sig_idx + 1]
            entry_idx = sig_idx + 1
            entry_date = df.index[sig_idx + 1]

        if entry_price is None or entry_idx is None:
            continue

        atr = df['ATR'].iloc[entry_idx]
        if pd.isna(atr) or atr <= 0:
            continue

        # ---- STOP / TARGET ----
        if direction == 'Long':
            stop_px = entry_price - (atr * stop_atr)
            tgt_px = entry_price + (atr * tgt_atr)
        else:
            stop_px = entry_price + (atr * stop_atr)
            tgt_px = entry_price - (atr * tgt_atr)

        # ---- EXIT SIMULATION ----
        max_exit_idx = min(entry_idx + hold_days, len(df) - 1)
        exit_price = df['Close'].iloc[max_exit_idx]
        exit_date = df.index[max_exit_idx]
        exit_type = "Time"

        if use_stop or use_target:
            for ci in range(entry_idx + 1, max_exit_idx + 1):
                row = df.iloc[ci]
                if direction == 'Long':
                    if use_stop and row['Low'] <= stop_px:
                        # Gap-through: use min(stop, open) for realism
                        exit_price = min(stop_px, row['Open']) if row['Open'] < stop_px else stop_px
                        exit_date, exit_type = row.name, "Stop"
                        break
                    if use_target and row['High'] >= tgt_px:
                        exit_price, exit_date, exit_type = tgt_px, row.name, "Target"
                        break
                else:
                    if use_stop and row['High'] >= stop_px:
                        exit_price = max(stop_px, row['Open']) if row['Open'] > stop_px else stop_px
                        exit_date, exit_type = row.name, "Stop"
                        break
                    if use_target and row['Low'] <= tgt_px:
                        exit_price, exit_date, exit_type = tgt_px, row.name, "Target"
                        break

        last_exit = exit_date

        # ---- PnL CALCULATION ----
        slip = slippage_bps / 10000.0
        if direction == 'Long':
            raw_pnl_per_share = exit_price * (1 - slip) - entry_price * (1 + slip)
        else:
            raw_pnl_per_share = entry_price * (1 - slip) - exit_price * (1 + slip)

        tech_risk = atr * stop_atr
        if tech_risk <= 0:
            tech_risk = 0.001

        # R-multiple (before commissions)
        r_raw = raw_pnl_per_share / tech_risk

        # Shares and commission
        shares = int(risk_dollars / tech_risk) if tech_risk > 0 else 0
        if shares == 0:
            continue
        commission_rt = calculate_commission_rt(shares)
        commission_r = commission_rt / risk_dollars if risk_dollars > 0 else 0

        # R-multiple after commissions
        r_net = r_raw - commission_r

        # Dollar PnL
        pnl_gross = raw_pnl_per_share * shares
        pnl_net = pnl_gross - commission_rt

        trades.append({
            'SignalDate': sig_date,
            'EntryDate': entry_date,
            'ExitDate': exit_date,
            'ExitType': exit_type,
            'Entry': entry_price,
            'Exit': exit_price,
            'ATR': atr,
            'Shares': shares,
            'R_Gross': r_raw,
            'R_Net': r_net,
            'PnL_Gross': pnl_gross,
            'PnL_Net': pnl_net,
            'Commission': commission_rt,
            'Risk_Dollars': risk_dollars,
        })

    return trades


# ============================================================================
# WALK-FORWARD ENGINE
# ============================================================================

def run_strategy_full(strat, master_dict, sznl_map, vix_series, market_series, account_value):
    """
    Run a single strategy across ALL available data.
    Returns a DataFrame of all trades.
    """
    settings = strat['settings']
    execution = strat['execution']
    all_trades = []

    # Determine which custom MAs are needed
    req_mas = list(set([f['length'] for f in settings.get('ma_consec_filters', [])]))
    acc_win = settings.get('acc_count_window') if settings.get('use_acc_count_filter') else None
    dist_win = settings.get('dist_count_window') if settings.get('use_dist_count_filter') else None
    gap_win = settings.get('gap_lookback', 21)

    for ticker in strat['universe_tickers']:
        t_clean = ticker.replace('.', '-')
        df = master_dict.get(t_clean)
        if df is None or len(df) < 300:
            continue

        try:
            calc_df = calculate_indicators(
                df, sznl_map, t_clean, market_series, vix_series,
                gap_window=gap_win, acc_window=acc_win, dist_window=dist_win,
                custom_sma_lengths=req_mas
            )
            mask = get_historical_mask(calc_df, settings, sznl_map, t_clean)
            if not mask.any():
                continue

            ticker_trades = simulate_trades(calc_df, mask, settings, execution, account_value)
            for t in ticker_trades:
                t['Ticker'] = ticker
            all_trades.extend(ticker_trades)
        except Exception:
            continue

    if not all_trades:
        return pd.DataFrame()

    return pd.DataFrame(all_trades).sort_values('ExitDate').reset_index(drop=True)


def split_into_folds(trades_df, folds):
    """Split a trades DataFrame into non-overlapping time folds based on SignalDate."""
    fold_results = []
    for start, end, label in folds:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        fold_df = trades_df[(trades_df['SignalDate'] >= start_ts) &
                            (trades_df['SignalDate'] <= end_ts)].copy()
        fold_results.append((label, fold_df))
    return fold_results


def compute_fold_metrics(fold_df):
    """Compute key metrics for a single fold."""
    if fold_df.empty:
        return {
            'Trades': 0, 'Win Rate': 0, 'PF_Gross': 0, 'PF_Net': 0,
            'Avg_R_Gross': 0, 'Avg_R_Net': 0, 'Total_PnL_Gross': 0,
            'Total_PnL_Net': 0, 'Total_Commission': 0, 'Max_Consecutive_Loss': 0,
            'Worst_Trade_R': 0, 'Best_Trade_R': 0, 'SQN_Net': 0,
            'Expectancy_R_Net': 0,
        }

    n = len(fold_df)
    wins_gross = fold_df[fold_df['R_Gross'] > 0]
    losses_gross = fold_df[fold_df['R_Gross'] <= 0]
    wins_net = fold_df[fold_df['R_Net'] > 0]
    losses_net = fold_df[fold_df['R_Net'] <= 0]

    # Profit Factor
    pf_gross = (wins_gross['PnL_Gross'].sum() / abs(losses_gross['PnL_Gross'].sum())
                if not losses_gross.empty and losses_gross['PnL_Gross'].sum() != 0 else 999)
    pf_net = (wins_net['PnL_Net'].sum() / abs(losses_net['PnL_Net'].sum())
              if not losses_net.empty and losses_net['PnL_Net'].sum() != 0 else 999)

    # SQN on net R
    r_net = fold_df['R_Net']
    sqn_net = np.sqrt(n) * (r_net.mean() / r_net.std()) if n > 1 and r_net.std() > 0 else 0

    # Max consecutive losses
    is_loss = (fold_df['R_Net'] <= 0).astype(int)
    max_consec = 0
    current = 0
    for val in is_loss:
        current = current + 1 if val else 0
        max_consec = max(max_consec, current)

    return {
        'Trades': n,
        'Win Rate': len(wins_net) / n * 100,
        'PF_Gross': min(pf_gross, 99),
        'PF_Net': min(pf_net, 99),
        'Avg_R_Gross': fold_df['R_Gross'].mean(),
        'Avg_R_Net': fold_df['R_Net'].mean(),
        'Total_PnL_Gross': fold_df['PnL_Gross'].sum(),
        'Total_PnL_Net': fold_df['PnL_Net'].sum(),
        'Total_Commission': fold_df['Commission'].sum(),
        'Max_Consecutive_Loss': max_consec,
        'Worst_Trade_R': fold_df['R_Net'].min(),
        'Best_Trade_R': fold_df['R_Net'].max(),
        'SQN_Net': sqn_net,
        'Expectancy_R_Net': fold_df['R_Net'].mean(),
    }


def evaluate_pass_fail(fold_metrics_list, folds_labels):
    """
    Apply pass/fail criteria:
    1. Profitable (PF_Net > 1.0) in >= 4 of 5 folds
    2. Worst fold PF_Net >= 1.2
    3. No single fold > 50% of total PnL
    4. Overall PF_Net >= 1.5
    
    Returns (pass: bool, reasons: list[str], warnings: list[str])
    """
    reasons = []
    warnings = []

    # Filter to folds with trades
    active_folds = [(label, m) for label, m in zip(folds_labels, fold_metrics_list)
                    if m['Trades'] >= 5]  # Need minimum 5 trades to count

    if len(active_folds) < 3:
        return False, ["Insufficient folds with >= 5 trades"], []

    # Criterion 1: Profitable in >= 80% of active folds
    profitable_count = sum(1 for _, m in active_folds if m['PF_Net'] > 1.0)
    required = max(3, int(len(active_folds) * 0.8))
    c1_pass = profitable_count >= required
    if not c1_pass:
        reasons.append(f"Profitable in only {profitable_count}/{len(active_folds)} folds (need {required})")

    # Criterion 2: Worst fold PF >= 1.2 (among folds with 10+ trades)
    substantive = [m for _, m in active_folds if m['Trades'] >= 10]
    if substantive:
        worst_pf = min(m['PF_Net'] for m in substantive)
        c2_pass = worst_pf >= 1.2
        if not c2_pass:
            reasons.append(f"Worst fold PF = {worst_pf:.2f} (need >= 1.2)")
    else:
        c2_pass = True
        warnings.append("No folds with 10+ trades for worst-fold check")

    # Criterion 3: No single fold > 50% of total PnL
    total_pnl = sum(m['Total_PnL_Net'] for _, m in active_folds)
    c3_pass = True
    if total_pnl > 0:
        for label, m in active_folds:
            pct = m['Total_PnL_Net'] / total_pnl if total_pnl != 0 else 0
            if pct > 0.50:
                c3_pass = False
                reasons.append(f"Fold '{label}' contributes {pct:.0%} of total PnL (max 50%)")
                break

    # Criterion 4: Overall PF_Net >= 1.5
    all_trades_pnl_win = sum(m['Total_PnL_Net'] for _, m in active_folds if m['Total_PnL_Net'] > 0)
    all_trades_pnl_loss = sum(abs(m['Total_PnL_Net']) for _, m in active_folds if m['Total_PnL_Net'] <= 0)
    overall_pf = all_trades_pnl_win / all_trades_pnl_loss if all_trades_pnl_loss > 0 else 999
    c4_pass = overall_pf >= 1.5
    if not c4_pass:
        reasons.append(f"Overall PF = {overall_pf:.2f} (need >= 1.5)")

    # Warnings (not fail conditions)
    for label, m in active_folds:
        if m['Max_Consecutive_Loss'] >= 10:
            warnings.append(f"Fold '{label}': {m['Max_Consecutive_Loss']} consecutive losses")

    passed = c1_pass and c2_pass and c3_pass and c4_pass
    return passed, reasons, warnings


def compute_rolling_pf(trades_df, window=60):
    """Compute rolling Profit Factor over a trade-count window."""
    if len(trades_df) < window:
        return pd.DataFrame()

    trades_sorted = trades_df.sort_values('ExitDate').reset_index(drop=True)
    results = []

    for i in range(window, len(trades_sorted) + 1):
        chunk = trades_sorted.iloc[i - window:i]
        wins = chunk[chunk['PnL_Net'] > 0]['PnL_Net'].sum()
        losses = abs(chunk[chunk['PnL_Net'] <= 0]['PnL_Net'].sum())
        pf = wins / losses if losses > 0 else 10.0
        results.append({
            'TradeNum': i,
            'Date': chunk['ExitDate'].iloc[-1],
            'Rolling_PF': min(pf, 10.0),
            'Rolling_WR': (chunk['R_Net'] > 0).mean() * 100,
        })

    return pd.DataFrame(results)


def compute_annual_metrics(trades_df):
    """Compute metrics broken down by calendar year."""
    if trades_df.empty:
        return pd.DataFrame()
    
    trades_df = trades_df.copy()
    trades_df['Year'] = trades_df['SignalDate'].dt.year
    
    results = []
    for year, ydf in trades_df.groupby('Year'):
        m = compute_fold_metrics(ydf)
        m['Year'] = year
        results.append(m)
    
    return pd.DataFrame(results).sort_values('Year')


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(layout="wide", page_title="Walk-Forward Validation")
    st.title("🔬 Walk-Forward Validation & Decay Analysis")
    st.markdown("""
    **Anchored walk-forward**: your current parameters are tested unchanged across non-overlapping 
    time periods. This measures whether the edge is *consistent* — not whether it can be *optimized*.
    
    **Commission model**: IBKR Pro Fixed ($1.00/order min + regulatory fees).
    """)
    st.markdown("---")

    if not _STRATEGY_BOOK_RAW:
        st.error("No strategies found in strategy_config.py")
        return

    # --- Sidebar ---
    st.sidebar.header("⚙️ Settings")

    if st.sidebar.button("🔴 Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        if 'wf_data' in st.session_state:
            del st.session_state['wf_data']
        st.rerun()

    strat_names = [s['name'] for s in _STRATEGY_BOOK_RAW]
    selected = st.sidebar.multiselect("Strategies to Validate", strat_names, default=strat_names)

    account_val = st.sidebar.number_input("Account Value ($)", value=ACCOUNT_VALUE, step=10000)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Fold Configuration")
    fold_mode = st.sidebar.radio("Fold Mode", ["Default (5-year blocks)", "Custom"])

    if fold_mode == "Custom":
        n_folds = st.sidebar.slider("Number of Folds", 3, 8, 5)
        fold_years = st.sidebar.slider("Years per Fold", 2, 8, 5)
        # Build custom folds working backward from 2024
        custom_folds = []
        end_year = 2024
        for i in range(n_folds):
            fy_end = end_year - (i * fold_years)
            fy_start = fy_end - fold_years + 1
            if fy_start < 1995:
                break
            custom_folds.append((
                f"{fy_start}-01-01",
                f"{fy_end}-12-31",
                f"{fy_start}–{fy_end}"
            ))
        folds = list(reversed(custom_folds))
    else:
        folds = DEFAULT_FOLDS

    st.sidebar.markdown("---")
    st.sidebar.subheader("Pass/Fail Thresholds")
    st.sidebar.caption("Defaults from panel recommendation")
    min_fold_pf = st.sidebar.number_input("Min Worst-Fold PF", value=1.2, step=0.1)
    max_single_fold_pct = st.sidebar.number_input("Max Single Fold PnL %", value=50, step=5)
    rolling_window = st.sidebar.number_input("Rolling PF Window (trades)", value=60, step=10, min_value=20)

    run_btn = st.sidebar.button("⚡ Run Walk-Forward", type="primary")

    # --- Main Area ---
    if not run_btn:
        st.info("👈 Select strategies and click **Run Walk-Forward** to begin.")

        # Show fold structure preview
        st.subheader("📅 Fold Structure")
        fold_preview = pd.DataFrame(folds, columns=['Start', 'End', 'Label'])
        st.dataframe(fold_preview, hide_index=True)

        st.subheader("📊 Commission Impact Preview")
        st.caption("Estimated per-trade commission cost for each strategy")
        preview_data = []
        for s in _STRATEGY_BOOK_RAW:
            if s['name'] not in selected:
                continue
            risk_d = account_val * s['execution']['risk_bps'] / 10000
            # Estimate typical shares (assume ATR ~$2 for large caps, stop_atr from config)
            est_atr = 2.0
            est_tech_risk = est_atr * s['execution']['stop_atr']
            est_shares = int(risk_d / est_tech_risk) if est_tech_risk > 0 else 100
            comm = calculate_commission_rt(est_shares)
            preview_data.append({
                'Strategy': s['name'],
                'Risk ($)': f"${risk_d:,.0f}",
                'Est. Shares': est_shares,
                'Commission RT': f"${comm:.2f}",
                'Comm % of Risk': f"{comm / risk_d * 100:.2f}%" if risk_d > 0 else "N/A"
            })
        if preview_data:
            st.dataframe(pd.DataFrame(preview_data), hide_index=True)
        return

    # ---- RUN WALK-FORWARD ----
    strategies = [copy.deepcopy(s) for s in _STRATEGY_BOOK_RAW if s['name'] in selected]

    if not strategies:
        st.warning("No strategies selected.")
        return

    # 1. Gather all tickers
    all_tickers = set()
    for s in strategies:
        all_tickers.update(s['universe_tickers'])
        settings = s['settings']
        if settings.get('use_market_sznl'):
            all_tickers.add(settings.get('market_ticker', '^GSPC'))
        if "Market" in settings.get('trend_filter', ''):
            all_tickers.add(settings.get('market_ticker', 'SPY'))
        if settings.get('use_vix_filter'):
            all_tickers.add('^VIX')
    all_tickers.add('SPY')
    all_tickers.add('^VIX')

    # 2. Download data
    st.write(f"📥 **Phase 1:** Downloading data for {len(all_tickers)} tickers...")
    if 'wf_data' not in st.session_state:
        st.session_state['wf_data'] = {}
    existing = set(st.session_state['wf_data'].keys())
    clean_list = [t.replace('.', '-') for t in all_tickers]
    missing = list(set(clean_list) - existing)
    if missing:
        new_data = download_historical_data(missing, start_date="1999-01-01")
        st.session_state['wf_data'].update(new_data)
    master_dict = st.session_state['wf_data']
    st.write(f"   ✅ {len(master_dict)} tickers available")

    # 3. Prepare market/VIX series
    sznl_map = load_seasonal_map()

    spy_df = master_dict.get('SPY')
    market_series = None
    if spy_df is not None:
        temp = spy_df.copy()
        if isinstance(temp.columns, pd.MultiIndex):
            temp.columns = temp.columns.get_level_values(0)
        temp.columns = [c.capitalize() for c in temp.columns]
        if temp.index.tz is not None:
            temp.index = temp.index.tz_localize(None)
        temp['SMA200'] = temp['Close'].rolling(200).mean()
        market_series = temp['Close'] > temp['SMA200']

    vix_df = master_dict.get('^VIX')
    vix_series = None
    if vix_df is not None:
        temp_v = vix_df.copy()
        if isinstance(temp_v.columns, pd.MultiIndex):
            temp_v.columns = temp_v.columns.get_level_values(0)
        temp_v.columns = [c.capitalize() for c in temp_v.columns]
        if temp_v.index.tz is not None:
            temp_v.index = temp_v.index.tz_localize(None)
        vix_series = temp_v['Close']

    # 4. Run each strategy
    all_results = {}
    progress = st.progress(0)
    for idx, strat in enumerate(strategies):
        st.write(f"🔍 **Phase 2:** Running {strat['name']}...")
        t0 = time.time()
        trades_df = run_strategy_full(strat, master_dict, sznl_map, vix_series, market_series, account_val)
        elapsed = time.time() - t0
        n_trades = len(trades_df)
        st.write(f"   → {n_trades} trades in {elapsed:.1f}s")
        all_results[strat['name']] = trades_df
        progress.progress((idx + 1) / len(strategies))

    progress.empty()
    st.success("✅ Walk-forward analysis complete!")
    st.markdown("---")

    # ====================================================================
    # RESULTS DISPLAY
    # ====================================================================

    # --- SUMMARY SCORECARD ---
    st.header("📊 Strategy Scorecard")

    scorecard_data = []
    for strat in strategies:
        name = strat['name']
        tdf = all_results[name]
        if tdf.empty:
            scorecard_data.append({
                'Strategy': name, 'Total Trades': 0, 'Pass': '❌',
                'Overall PF (Net)': 0, 'Fail Reasons': 'No trades generated'
            })
            continue

        fold_splits = split_into_folds(tdf, folds)
        fold_labels = [f[0] for f in fold_splits]
        fold_metrics = [compute_fold_metrics(f[1]) for f in fold_splits]

        passed, reasons, warnings = evaluate_pass_fail(fold_metrics, fold_labels)

        wins_net = tdf[tdf['PnL_Net'] > 0]['PnL_Net'].sum()
        losses_net = abs(tdf[tdf['PnL_Net'] <= 0]['PnL_Net'].sum())
        overall_pf = wins_net / losses_net if losses_net > 0 else 999

        scorecard_data.append({
            'Strategy': name,
            'Total Trades': len(tdf),
            'Pass': '✅ PASS' if passed else '⚠️ FAIL',
            'Overall PF (Net)': f"{overall_pf:.2f}",
            'Win Rate': f"{(tdf['R_Net'] > 0).mean() * 100:.1f}%",
            'Avg R (Net)': f"{tdf['R_Net'].mean():.3f}",
            'Total Commission': f"${tdf['Commission'].sum():,.0f}",
            'Commission % of Gross': f"{tdf['Commission'].sum() / max(tdf['PnL_Gross'].abs().sum(), 1) * 100:.1f}%",
            'Fail Reasons': '; '.join(reasons) if reasons else '—',
        })

    sc_df = pd.DataFrame(scorecard_data)
    st.dataframe(sc_df, hide_index=True, use_container_width=True)

    # --- PER-STRATEGY DETAIL ---
    for strat in strategies:
        name = strat['name']
        tdf = all_results[name]
        if tdf.empty:
            continue

        st.markdown("---")
        st.header(f"📈 {name}")

        # Fold Metrics Table
        fold_splits = split_into_folds(tdf, folds)
        fold_labels = [f[0] for f in fold_splits]
        fold_metrics = [compute_fold_metrics(f[1]) for f in fold_splits]
        passed, reasons, warnings = evaluate_pass_fail(fold_metrics, fold_labels)

        # Show pass/fail prominently
        if passed:
            st.success(f"✅ **PASS** — Edge is consistent across {len([m for m in fold_metrics if m['Trades'] >= 5])} active folds")
        else:
            st.error(f"⚠️ **FAIL** — {'; '.join(reasons)}")
        if warnings:
            for w in warnings:
                st.warning(f"⚠️ {w}")

        # Fold comparison table
        st.subheader("📅 Fold Breakdown")
        fold_table = []
        for label, metrics in zip(fold_labels, fold_metrics):
            fold_table.append({
                'Fold': label,
                'Trades': metrics['Trades'],
                'Win Rate': f"{metrics['Win Rate']:.1f}%",
                'PF (Gross)': f"{metrics['PF_Gross']:.2f}",
                'PF (Net)': f"{metrics['PF_Net']:.2f}",
                'Avg R (Net)': f"{metrics['Avg_R_Net']:.3f}",
                'SQN': f"{metrics['SQN_Net']:.2f}",
                'PnL (Gross)': f"${metrics['Total_PnL_Gross']:,.0f}",
                'PnL (Net)': f"${metrics['Total_PnL_Net']:,.0f}",
                'Commission': f"${metrics['Total_Commission']:,.0f}",
                'Max Consec Loss': metrics['Max_Consecutive_Loss'],
            })
        st.dataframe(pd.DataFrame(fold_table), hide_index=True, use_container_width=True)

        # PnL Concentration check
        total_pnl = sum(m['Total_PnL_Net'] for m in fold_metrics)
        if total_pnl > 0:
            conc_data = []
            for label, m in zip(fold_labels, fold_metrics):
                pct = m['Total_PnL_Net'] / total_pnl * 100 if total_pnl != 0 else 0
                conc_data.append({'Fold': label, 'PnL Contribution': pct})
            fig_conc = px.bar(
                pd.DataFrame(conc_data), x='Fold', y='PnL Contribution',
                title="PnL Concentration by Fold (no single fold should dominate >50%)",
                text_auto='.1f'
            )
            fig_conc.add_hline(y=50, line_dash="dash", line_color="red",
                               annotation_text="50% threshold")
            fig_conc.update_layout(height=300, yaxis_title="% of Total PnL")
            st.plotly_chart(fig_conc, use_container_width=True)

        # Rolling PF Decay Chart
        st.subheader("📉 Rolling Profit Factor (Decay Detection)")
        rpf = compute_rolling_pf(tdf, window=rolling_window)
        if not rpf.empty:
            fig_rpf = make_subplots(specs=[[{"secondary_y": True}]])
            fig_rpf.add_trace(
                go.Scatter(x=rpf['Date'], y=rpf['Rolling_PF'], mode='lines',
                           name=f'Rolling PF ({rolling_window} trades)',
                           line=dict(color='#00CC00', width=2)),
                secondary_y=False
            )
            fig_rpf.add_trace(
                go.Scatter(x=rpf['Date'], y=rpf['Rolling_WR'], mode='lines',
                           name='Rolling Win Rate %',
                           line=dict(color='#0066CC', width=1, dash='dot')),
                secondary_y=True
            )
            fig_rpf.add_hline(y=1.0, line_dash="dash", line_color="red",
                               annotation_text="Break-even")
            # Linear regression for decay detection
            if len(rpf) > 10:
                x_vals = np.arange(len(rpf))
                slope, intercept = np.polyfit(x_vals, rpf['Rolling_PF'].values, 1)
                trend_line = slope * x_vals + intercept
                fig_rpf.add_trace(
                    go.Scatter(x=rpf['Date'], y=trend_line, mode='lines',
                               name=f'Trend (slope: {slope:.4f}/trade)',
                               line=dict(color='orange', width=1, dash='dash')),
                    secondary_y=False
                )
                if slope < -0.001:
                    st.warning(f"⚠️ **Decay detected**: Rolling PF trend slope = {slope:.5f}/trade. "
                               f"The edge may be eroding over time.")
                elif slope > 0.001:
                    st.info(f"📈 **Strengthening**: Rolling PF trend slope = {slope:.5f}/trade.")

            fig_rpf.update_layout(
                height=400, title=f"Rolling {rolling_window}-Trade Metrics",
                hovermode="x unified"
            )
            fig_rpf.update_yaxes(title_text="Profit Factor", secondary_y=False)
            fig_rpf.update_yaxes(title_text="Win Rate %", secondary_y=True)
            st.plotly_chart(fig_rpf, use_container_width=True)
        else:
            st.info(f"Need at least {rolling_window} trades for rolling analysis.")

        # Annual Breakdown
        st.subheader("📆 Annual Performance (Net of Commissions)")
        annual = compute_annual_metrics(tdf)
        if not annual.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig_yr = px.bar(annual, x='Year', y='Total_PnL_Net',
                                title="Net PnL by Year", text_auto='$.0s',
                                color='Total_PnL_Net',
                                color_continuous_scale=['#CC0000', '#CCCCCC', '#00CC00'],
                                color_continuous_midpoint=0)
                fig_yr.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_yr, use_container_width=True)

            with col2:
                fig_wr = px.bar(annual, x='Year', y='Win Rate',
                                title="Win Rate by Year", text_auto='.1f')
                fig_wr.add_hline(y=50, line_dash="dash", line_color="gray")
                fig_wr.update_layout(height=350)
                st.plotly_chart(fig_wr, use_container_width=True)

            # Year-over-year table
            display_annual = annual[['Year', 'Trades', 'Win Rate', 'PF_Net',
                                      'Avg_R_Net', 'Total_PnL_Net', 'Total_Commission']].copy()
            display_annual.columns = ['Year', 'Trades', 'Win Rate %', 'PF (Net)',
                                       'Avg R (Net)', 'PnL (Net $)', 'Commission $']
            st.dataframe(display_annual.style.format({
                'Win Rate %': '{:.1f}',
                'PF (Net)': '{:.2f}',
                'Avg R (Net)': '{:.3f}',
                'PnL (Net $)': '${:,.0f}',
                'Commission $': '${:,.0f}',
            }), hide_index=True, use_container_width=True)

        # Cumulative Equity Curve (net of commissions)
        st.subheader("💰 Cumulative PnL (Net)")
        cum_df = tdf.sort_values('ExitDate').copy()
        cum_df['CumPnL_Net'] = cum_df['PnL_Net'].cumsum()
        cum_df['CumPnL_Gross'] = cum_df['PnL_Gross'].cumsum()

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=cum_df['ExitDate'], y=cum_df['CumPnL_Gross'],
            mode='lines', name='Gross', line=dict(color='#AAAAAA', width=1, dash='dot')
        ))
        fig_eq.add_trace(go.Scatter(
            x=cum_df['ExitDate'], y=cum_df['CumPnL_Net'],
            mode='lines', name='Net (after commissions)',
            line=dict(color='#00CC00', width=2)
        ))
        fig_eq.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_eq.update_layout(height=400, title="Cumulative PnL", hovermode="x unified",
                             yaxis_tickformat="$,.0f")
        st.plotly_chart(fig_eq, use_container_width=True)

        # Commission Summary
        with st.expander("💳 Commission Impact Detail"):
            total_gross = tdf['PnL_Gross'].sum()
            total_comm = tdf['Commission'].sum()
            total_net = tdf['PnL_Net'].sum()
            avg_comm = tdf['Commission'].mean()
            avg_shares = tdf['Shares'].mean()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Gross PnL", f"${total_gross:,.0f}")
            c2.metric("Total Commissions", f"${total_comm:,.0f}")
            c3.metric("Total Net PnL", f"${total_net:,.0f}",
                       delta=f"-{total_comm / max(total_gross, 1) * 100:.1f}% drag")
            c4.metric("Avg Commission/Trade", f"${avg_comm:.2f}",
                       help=f"Avg shares: {avg_shares:.0f}")


if __name__ == "__main__":
    main()
