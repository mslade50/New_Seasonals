import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import time
from pandas.tseries.offsets import BusinessDay
import plotly.graph_objects as go
import sys
import os

# -----------------------------------------------------------------------------
# IMPORT STRATEGY BOOK FROM ROOT
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from strategy_config import _STRATEGY_BOOK_RAW, ACCOUNT_VALUE
except ImportError:
    st.error("Could not find strategy_config.py in the root directory.")
    _STRATEGY_BOOK_RAW = []
    ACCOUNT_VALUE = 150000

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
PRIMARY_SZNL_PATH = "sznl_ranks.csv"      
BACKUP_SZNL_PATH = "seasonal_ranks.csv"   

@st.cache_resource 
def load_seasonal_map():
    def load_raw_csv(path):
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

    df_primary = load_raw_csv(PRIMARY_SZNL_PATH)
    df_backup = load_raw_csv(BACKUP_SZNL_PATH)

    if df_primary.empty and df_backup.empty:
        return {}
    elif df_primary.empty:
        final_df = df_backup
    elif df_backup.empty:
        final_df = df_primary
    else:
        final_df = pd.concat([df_primary, df_backup], axis=0)
        final_df = final_df.drop_duplicates(subset=['ticker', 'Date'], keep='first')

    output_map = {}
    final_df = final_df.sort_values(by="Date")
    for ticker, group in final_df.groupby("ticker"):
        series = group.set_index("Date")["seasonal_rank"]
        output_map[ticker] = series
        
    return output_map

def get_sznl_val_series(ticker, dates, sznl_map):
    ticker = ticker.upper()
    t_series = sznl_map.get(ticker)
    if t_series is None and ticker == "^GSPC":
        t_series = sznl_map.get("SPY")
    if t_series is None:
        return pd.Series(50.0, index=dates)
    return dates.map(t_series).fillna(50.0)

# -----------------------------------------------------------------------------
# HELPER: BATCH DOWNLOADER
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def download_historical_data(tickers, start_date="2000-01-01"):
    if not tickers: return {}
    clean_tickers = list(set([str(t).strip().upper().replace('.', '-') for t in tickers]))
    data_dict = {}
    CHUNK_SIZE = 50 
    total = len(clean_tickers)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, total, CHUNK_SIZE):
        chunk = clean_tickers[i : i + CHUNK_SIZE]
        current_progress = min((i + CHUNK_SIZE) / total, 1.0)
        status_text.text(f"📥 Downloading batch {i+1}-{min(i+CHUNK_SIZE, total)} of {total}...")
        progress_bar.progress(current_progress)
        try:
            df = yf.download(chunk, start=start_date, group_by='ticker', auto_adjust=False, progress=False, threads=True)
            if df.empty: continue
            if len(chunk) == 1:
                ticker = chunk[0]
                if 'Close' in df.columns:
                    df.index = df.index.tz_localize(None)
                    data_dict[ticker] = df
            else:
                available_tickers = df.columns.levels[0]
                for t in available_tickers:
                    try:
                        t_df = df[t].copy()
                        if t_df.empty or 'Close' not in t_df.columns: continue
                        t_df.index = t_df.index.tz_localize(None)
                        data_dict[t] = t_df
                    except Exception:
                        continue
            time.sleep(0.25)
        except Exception:
            continue

    progress_bar.empty()
    status_text.empty()
    return data_dict


def calculate_indicators(df, sznl_map, ticker, market_series=None, vix_series=None, gap_window=21, custom_sma_lengths=None, acc_window=None, dist_window=None):
    df = df.copy()
    df.sort_index(inplace=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    
    # --- Standard SMAs ---
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA100'] = df['Close'].rolling(100).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    if custom_sma_lengths:
        for length in custom_sma_lengths:
            col_name = f"SMA{length}"
            if col_name not in df.columns:
                df[col_name] = df['Close'].rolling(length).mean()

    # --- EMAs ---
    df['EMA8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA11'] = df['Close'].ewm(span=11, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # --- Perf Ranks ---
    for window in [5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window)
        df[f'rank_ret_{window}d'] = df[f'ret_{window}d'].expanding(min_periods=252).rank(pct=True) * 100.0
    
    # --- ATR ---
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
    
    # --- SEASONALITY ---
    df['Sznl'] = get_sznl_val_series(ticker, df.index, sznl_map)
    df['Mkt_Sznl_Ref'] = get_sznl_val_series("^GSPC", df.index, sznl_map)
    
    # --- 52w High/Low ---
    rolling_high = df['High'].shift(1).rolling(252).max()
    rolling_low = df['Low'].shift(1).rolling(252).min()
    df['is_52w_high'] = df['High'] > rolling_high
    df['is_52w_low'] = df['Low'] < rolling_low
    
    # --- Volume & Vol Ratio ---
    vol_ma = df['Volume'].rolling(63).mean()
    df['vol_ma'] = vol_ma
    df['vol_ratio'] = df['Volume'] / vol_ma
    
    vol_ma_10 = df['Volume'].rolling(10).mean()
    df['vol_ratio_10d'] = vol_ma_10 / vol_ma
    df['vol_ratio_10d_rank'] = df['vol_ratio_10d'].expanding(min_periods=252).rank(pct=True) * 100.0
    
    # --- Acc/Dist Flags ---
    vol_gt_prev = df['Volume'] > df['Volume'].shift(1)
    vol_gt_ma = df['Volume'] > df['vol_ma']
    is_green = df['Close'] > df['Open']
    is_red = df['Close'] < df['Open']

    df['is_acc_day'] = (is_green & vol_gt_prev & vol_gt_ma).astype(int)
    df['is_dist_day'] = (is_red & vol_gt_prev & vol_gt_ma).astype(int)
    
    if acc_window:
        df[f'AccCount_{acc_window}'] = df['is_acc_day'].rolling(acc_window).sum()
    if dist_window:
        df[f'DistCount_{dist_window}'] = df['is_dist_day'].rolling(dist_window).sum()

    # --- Age ---
    if not df.empty:
        start_ts = df.index[0]
        df['age_years'] = (df.index - start_ts).days / 365.25
    else:
        df['age_years'] = 0.0
        
    # --- External Series ---
    if market_series is not None:
        df['Market_Above_SMA200'] = market_series.reindex(df.index, method='ffill').fillna(False)

    if vix_series is not None:
        df['VIX_Value'] = vix_series.reindex(df.index, method='ffill').fillna(0)
    else:
        df['VIX_Value'] = 0.0

    # --- Candle Range % Location ---
    denom = (df['High'] - df['Low'])
    df['RangePct'] = np.where(denom == 0, 0.5, (df['Close'] - df['Low']) / denom)

    # --- Day of Week & Gaps ---
    df['DayOfWeekVal'] = df.index.dayofweek
    is_open_gap = (df['Low'] > df['High'].shift(1)).astype(int)
    df['GapCount'] = is_open_gap.rolling(gap_window).sum()

    # --- Pivots ---
    piv_len = 20 
    roll_max = df['High'].rolling(window=piv_len*2+1, center=True).max()
    df['is_pivot_high'] = (df['High'] == roll_max)
    roll_min = df['Low'].rolling(window=piv_len*2+1, center=True).min()
    df['is_pivot_low'] = (df['Low'] == roll_min)

    df['LastPivotHigh'] = np.where(df['is_pivot_high'], df['High'], np.nan)
    df['LastPivotHigh'] = df['LastPivotHigh'].shift(piv_len).ffill()
    df['LastPivotLow'] = np.where(df['is_pivot_low'], df['Low'], np.nan)
    df['LastPivotLow'] = df['LastPivotLow'].shift(piv_len).ffill()
    
    df['PrevHigh'] = df['High'].shift(1)
    df['PrevLow'] = df['Low'].shift(1)

    return df


def get_historical_mask(df, params, sznl_map, ticker_name="UNK"):
    mask = pd.Series(True, index=df.index)
    
    trend_opt = params.get('trend_filter', 'None')
    if trend_opt == "Price > 200 SMA":
        mask &= (df['Close'] > df['SMA200'])
    elif trend_opt == "Price > Rising 200 SMA":
        mask &= (df['Close'] > df['SMA200']) & (df['SMA200'] > df['SMA200'].shift(1))
    elif trend_opt == "Not Below Declining 200 SMA":
        mask &= ~((df['Close'] < df['SMA200']) & (df['SMA200'] < df['SMA200'].shift(1)))
    elif trend_opt == "Price < 200 SMA":
        mask &= (df['Close'] < df['SMA200'])
    elif trend_opt == "Price < Falling 200 SMA":
        mask &= (df['Close'] < df['SMA200']) & (df['SMA200'] < df['SMA200'].shift(1))
    elif "Market" in trend_opt and 'Market_Above_SMA200' in df.columns:
        if trend_opt == "Market > 200 SMA": mask &= df['Market_Above_SMA200']
        elif trend_opt == "Market < 200 SMA": mask &= ~df['Market_Above_SMA200']

    mask &= (df['Close'] >= params.get('min_price', 0))
    mask &= (df['vol_ma'] >= params.get('min_vol', 0))
    mask &= (df['age_years'] >= params.get('min_age', 0))
    mask &= (df['age_years'] <= params.get('max_age', 100))
    
    if 'ATR_Pct' in df.columns:
        mask &= (df['ATR_Pct'] >= params.get('min_atr_pct', 0)) & (df['ATR_Pct'] <= params.get('max_atr_pct', 1000))
        
    if params.get('require_close_gt_open', False):
        mask &= (df['Close'] > df['Open'])

    bk_mode = params.get('breakout_mode', 'None')
    if bk_mode == "Close > Prev Day High":
        mask &= (df['Close'] > df['PrevHigh'])
    elif bk_mode == "Close < Prev Day Low":
        mask &= (df['Close'] < df['PrevLow'])

    if params.get('use_range_filter', False):
        rn_val = df['RangePct'] * 100
        mask &= (rn_val >= params.get('range_min', 0)) & (rn_val <= params.get('range_max', 100))

    if params.get('use_dow_filter', False):
        allowed = params.get('allowed_days', [])
        if allowed:
            mask &= df['DayOfWeekVal'].isin(allowed)

    if 'allowed_cycles' in params:
        allowed = params['allowed_cycles']
        if allowed and len(allowed) < 4:
            year_rems = df.index.year % 4
            mask &= pd.Series(year_rems, index=df.index).isin(allowed)

    if params.get('use_gap_filter', False):
        g_val = df['GapCount']
        g_thresh = params.get('gap_thresh', 0)
        g_logic = params.get('gap_logic', '>')
        if g_logic == ">": mask &= (g_val > g_thresh)
        elif g_logic == "<": mask &= (g_val < g_thresh)
        elif g_logic == "=": mask &= (g_val == g_thresh)

    if params.get('use_acc_count_filter', False):
        col = f"AccCount_{params['acc_count_window']}"
        if col in df.columns:
            if params['acc_count_logic'] == ">": mask &= (df[col] > params['acc_count_thresh'])
            elif params['acc_count_logic'] == "<": mask &= (df[col] < params['acc_count_thresh'])
            elif params['acc_count_logic'] == "=": mask &= (df[col] == params['acc_count_thresh'])

    if params.get('use_dist_count_filter', False):
        col = f"DistCount_{params['dist_count_window']}"
        if col in df.columns:
            if params['dist_count_logic'] == ">": mask &= (df[col] > params['dist_count_thresh'])
            elif params['dist_count_logic'] == "<": mask &= (df[col] < params['dist_count_thresh'])
            elif params['dist_count_logic'] == "=": mask &= (df[col] == params['dist_count_thresh'])

    for pf in params.get('perf_filters', []):
        col = f"rank_ret_{pf['window']}d"
        cond = (df[col] < pf['thresh']) if pf['logic'] == '<' else (df[col] > pf['thresh'])
        if pf['consecutive'] > 1: cond = cond.rolling(pf['consecutive']).sum() == pf['consecutive']
        mask &= cond

    if params.get('use_perf_rank', False):
        col = f"rank_ret_{params['perf_window']}d"
        if params['perf_logic'] == '<': 
            raw = df[col] < params['perf_thresh']
        else: 
            raw = df[col] > params['perf_thresh']
        
        consec = params.get('perf_consecutive', 1)
        if consec > 1: 
            persist = (raw.rolling(consec).sum() == consec)
        else: 
            persist = raw
        
        mask &= persist
        
        if params.get('perf_first_instance', False):
            lookback = params.get('perf_lookback', 21)
            prev_inst = mask.shift(1).rolling(lookback).sum()
            mask &= (prev_inst == 0)

    for f in params.get('ma_consec_filters', []):
        col = f"SMA{f['length']}"
        if col in df.columns:
            cond = (df['Close'] > df[col]) if f['logic'] == 'Above' else (df['Close'] < df[col])
            if f['consec'] > 1: cond = cond.rolling(f['consec']).sum() == f['consec']
            mask &= cond

    if params.get('use_sznl', False):
        cond_s = (df['Sznl'] < params['sznl_thresh']) if params['sznl_logic'] == '<' else (df['Sznl'] > params['sznl_thresh'])
        if params.get('sznl_first_instance', False):
            lookback = params.get('sznl_lookback', 21)
            prev = cond_s.shift(1).rolling(lookback).max().fillna(0)
            cond_s &= (prev == 0)
        mask &= cond_s

    if params.get('use_market_sznl', False):
        cond_ms = (df['Mkt_Sznl_Ref'] < params['market_sznl_thresh']) if params['market_sznl_logic'] == '<' else (df['Mkt_Sznl_Ref'] > params['market_sznl_thresh'])
        mask &= cond_ms

    if params.get('use_52w', False):
        cond_52 = df['is_52w_high'] if params['52w_type'] == 'New 52w High' else df['is_52w_low']
        
        if params.get('52w_lag', 0) > 0:
            cond_52 = cond_52.shift(params['52w_lag']).fillna(False)

        if params.get('52w_first_instance', False):
            lookback = params.get('52w_lookback', 21)
            prev = cond_52.shift(1).rolling(lookback).max().fillna(0)
            cond_52 &= (prev == 0)
        
        mask &= cond_52

    if params.get('exclude_52w_high', False):
        mask &= (~df['is_52w_high'])

    if params.get('vol_gt_prev', False):
        mask &= (df['Volume'] > df['Volume'].shift(1))
    
    if params.get('use_vol', False):
        mask &= (df['vol_ratio'] > params['vol_thresh'])

    if params.get('use_vol_rank', False):
        col = 'vol_ratio_10d_rank'
        if params['vol_rank_logic'] == ">": mask &= (df[col] > params['vol_rank_thresh'])
        elif params['vol_rank_logic'] == "<": mask &= (df[col] < params['vol_rank_thresh'])

    if params.get('use_ma_dist_filter', False):
        ma_col_map = {"SMA 10": "SMA10", "SMA 20": "SMA20", "SMA 50": "SMA50", "SMA 100": "SMA100", "SMA 200": "SMA200", 
                      "EMA 8": "EMA8", "EMA 11": "EMA11", "EMA 21": "EMA21"}
        ma_target = ma_col_map.get(params['dist_ma_type'])
        if ma_target and ma_target in df.columns:
            dist_val = (df['Close'] - df[ma_target]) / df['ATR']
            if params['dist_logic'] == "Greater Than (>)": mask &= (dist_val > params['dist_min'])
            elif params['dist_logic'] == "Less Than (<)": mask &= (dist_val < params['dist_max'])
            elif params['dist_logic'] == "Between": mask &= (dist_val >= params['dist_min']) & (dist_val <= params['dist_max'])

    if params.get('use_vix_filter', False) and 'VIX_Value' in df.columns:
        mask &= (df['VIX_Value'] >= params.get('vix_min', 0)) & (df['VIX_Value'] <= params.get('vix_max', 100))

    if params.get('use_ma_touch', False):
        ma_col_map = {"SMA 10": "SMA10", "SMA 20": "SMA20", "SMA 50": "SMA50", "SMA 100": "SMA100", "SMA 200": "SMA200", 
                      "EMA 8": "EMA8", "EMA 11": "EMA11", "EMA 21": "EMA21"}
        ma_target = ma_col_map.get(params.get('ma_touch_type'))
        
        if ma_target and ma_target in df.columns:
            ma_series = df[ma_target]
            direction = params.get('trade_direction', 'Long')
            slope_lookback = params.get('ma_slope_days', 20)
            untested_lookback = params.get('ma_untested_days', 50)

            if direction == 'Long': is_slope_ok = (ma_series > ma_series.shift(1))
            else: is_slope_ok = (ma_series < ma_series.shift(1))
            
            if slope_lookback > 1:
                is_slope_ok = (is_slope_ok.rolling(slope_lookback).sum() == slope_lookback)
            
            if untested_lookback > 0:
                if direction == 'Long':
                    was_untested = (df['Low'] > ma_series).shift(1).rolling(untested_lookback).min() == 1.0
                else:
                    was_untested = (df['High'] < ma_series).shift(1).rolling(untested_lookback).min() == 1.0
                was_untested = was_untested.fillna(False)
            else:
                was_untested = True

            if direction == 'Long': touched_today = (df['Low'] <= ma_series)
            else: touched_today = (df['High'] >= ma_series)

            mask &= (is_slope_ok & was_untested & touched_today)

    return mask.fillna(False)


# -----------------------------------------------------------------------------
# OPTIMIZED: PRE-COMPUTE ALL INDICATOR DATAFRAMES ONCE
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def precompute_all_indicators(_master_dict, _strategies, _sznl_map, _vix_series):
    """
    Pre-compute indicators for all tickers once, then cache.
    Returns dict: ticker -> processed DataFrame
    """
    processed = {}
    
    # Build market series (SPY > 200 SMA)
    spy_df = _master_dict.get('SPY')
    market_series = None
    if spy_df is not None:
        temp = spy_df.copy()
        if isinstance(temp.columns, pd.MultiIndex):
            temp.columns = [c[0] if isinstance(c, tuple) else c for c in temp.columns]
        temp.columns = [c.capitalize() for c in temp.columns]
        temp['SMA200'] = temp['Close'].rolling(200).mean()
        market_series = temp['Close'] > temp['SMA200']
    
    # Collect all unique tickers and their required params
    ticker_params = {}
    for strat in _strategies:
        settings = strat['settings']
        gap_win = settings.get('gap_lookback', 21)
        acc_win = settings.get('acc_count_window') if settings.get('use_acc_count_filter') else None
        dist_win = settings.get('dist_count_window') if settings.get('use_dist_count_filter') else None
        req_custom_mas = list(set([f['length'] for f in settings.get('ma_consec_filters', [])]))
        
        for ticker in strat['universe_tickers']:
            t_clean = ticker.replace('.', '-')
            if t_clean not in ticker_params:
                ticker_params[t_clean] = {'gap': gap_win, 'acc': acc_win, 'dist': dist_win, 'mas': set(req_custom_mas)}
            else:
                # Merge params (use max gap, union of MAs, etc.)
                ticker_params[t_clean]['gap'] = max(ticker_params[t_clean]['gap'], gap_win)
                if acc_win: ticker_params[t_clean]['acc'] = acc_win
                if dist_win: ticker_params[t_clean]['dist'] = dist_win
                ticker_params[t_clean]['mas'].update(req_custom_mas)
    
    # Process each ticker once
    for t_clean, params in ticker_params.items():
        df = _master_dict.get(t_clean)
        if df is None or len(df) < 200:
            continue
        try:
            processed[t_clean] = calculate_indicators(
                df, _sznl_map, t_clean, market_series, _vix_series,
                gap_window=params['gap'],
                acc_window=params['acc'],
                dist_window=params['dist'],
                custom_sma_lengths=list(params['mas'])
            )
        except Exception:
            continue
    
    return processed


# -----------------------------------------------------------------------------
# OPTIMIZED: GENERATE CANDIDATES AS LIGHTWEIGHT TUPLES
# -----------------------------------------------------------------------------
def generate_candidates_fast(processed_dict, strategies, sznl_map, user_start_date):
    """
    Generate candidate signals as lightweight tuples (not storing full DataFrames).
    Returns list of tuples: (signal_date_ts, ticker, t_clean, strat_idx, signal_idx)
    Plus a lookup dict for signal row data.
    """
    candidates = []
    signal_data = {}  # (t_clean, signal_idx) -> row data dict
    cutoff_ts = pd.Timestamp(user_start_date)
    
    for strat_idx, strat in enumerate(strategies):
        settings = strat['settings']
        
        for ticker in strat['universe_tickers']:
            t_clean = ticker.replace('.', '-')
            df = processed_dict.get(t_clean)
            if df is None:
                continue
            
            try:
                mask = get_historical_mask(df, settings, sznl_map, ticker)
                mask = mask[mask.index >= cutoff_ts]
                if not mask.any():
                    continue
                
                true_indices = mask[mask].index
                
                for signal_date in true_indices:
                    signal_idx = df.index.get_loc(signal_date)
                    
                    # Store minimal data needed for sizing
                    key = (t_clean, signal_idx)
                    if key not in signal_data:
                        row = df.iloc[signal_idx]
                        signal_data[key] = {
                            'atr': row['ATR'],
                            'close': row['Close'],
                            'open': row['Open'],
                            'high': row['High'],
                            'low': row['Low'],
                            'vol_ratio': row.get('vol_ratio', 0),
                            'sznl': row.get('Sznl', 50),
                            'range_pct': row['RangePct'] * 100
                        }
                    
                    # Lightweight candidate tuple
                    candidates.append((
                        signal_date.value,  # int64 timestamp for fast sorting
                        ticker,
                        t_clean,
                        strat_idx,
                        signal_idx
                    ))
            except Exception:
                continue
    
    return candidates, signal_data


# -----------------------------------------------------------------------------
# OPTIMIZED: PROCESS SIGNALS WITH NUMPY ARRAYS WHERE POSSIBLE
# -----------------------------------------------------------------------------
def process_signals_fast(candidates, signal_data, processed_dict, strategies, starting_equity):
    """
    Process candidates chronologically with dynamic sizing.
    Optimized for speed using direct array access.
    """
    if not candidates:
        return pd.DataFrame()
    
    # Sort by timestamp (first element of tuple)
    candidates.sort(key=lambda x: x[0])
    
    current_equity = starting_equity
    realized_pnl = 0.0
    position_last_exit = {}  # (strat_name, ticker) -> exit_date_ts
    
    results = []
    
    for cand in candidates:
        signal_ts, ticker, t_clean, strat_idx, signal_idx = cand
        signal_date = pd.Timestamp(signal_ts)
        
        strat = strategies[strat_idx]
        settings = strat['settings']
        strat_name = strat['name']
        
        pos_key = (strat_name, ticker)
        
        # Skip if last exit hasn't occurred
        last_exit_ts = position_last_exit.get(pos_key)
        if last_exit_ts is not None and signal_ts <= last_exit_ts:
            continue
        
        df = processed_dict[t_clean]
        row_data = signal_data[(t_clean, signal_idx)]
        
        # Get entry details
        entry_type = settings.get('entry_type', 'Signal Close')
        hold_days = strat['execution']['hold_days']
        atr = row_data['atr']
        
        if signal_idx + 1 >= len(df):
            continue
        
        # Direct iloc access is faster than loc
        entry_row = df.iloc[signal_idx + 1]
        
        valid_entry = True
        entry_price = None
        entry_date = None
        
        if entry_type == 'T+1 Close':
            entry_price = entry_row['Close']
            entry_date = entry_row.name
        elif entry_type == 'T+1 Open':
            entry_price = entry_row['Open']
            entry_date = entry_row.name
        elif entry_type == 'T+1 Close if < Signal Close':
            if entry_row['Close'] < row_data['close']:
                entry_price = entry_row['Close']
                entry_date = entry_row.name
            else:
                valid_entry = False
        elif entry_type == "Limit (Open +/- 0.5 ATR)":
            limit_offset = 0.5 * atr
            if settings['trade_direction'] == 'Short':
                limit_price = entry_row['Open'] + limit_offset
                if entry_row['High'] >= limit_price:
                    entry_price = limit_price
                else:
                    valid_entry = False
            else:
                limit_price = entry_row['Open'] - limit_offset
                if entry_row['Low'] <= limit_price:
                    entry_price = limit_price
                else:
                    valid_entry = False
            entry_date = entry_row.name
        elif "Persistent" in entry_type:
            limit_offset = 0.5 * atr
            limit_base = row_data['close']
            found_fill = False
            search_end = min(signal_idx + 1 + hold_days, len(df))
            
            for i in range(signal_idx + 1, search_end):
                check_row = df.iloc[i]
                if settings['trade_direction'] == 'Short':
                    limit_price = limit_base + limit_offset
                    if check_row['High'] >= limit_price:
                        entry_price = limit_price
                        entry_date = check_row.name
                        hold_days = max(1, strat['execution']['hold_days'] - (i - signal_idx))
                        found_fill = True
                        break
                else:
                    limit_price = limit_base - limit_offset
                    if check_row['Low'] <= limit_price:
                        entry_price = limit_price
                        entry_date = check_row.name
                        hold_days = max(1, strat['execution']['hold_days'] - (i - signal_idx))
                        found_fill = True
                        break
            valid_entry = found_fill
        else:
            entry_price = row_data['close']
            entry_date = signal_date
        
        if not valid_entry or entry_price is None:
            continue
        
        # Dynamic position sizing
        risk_bps = strat['execution']['risk_bps']
        base_risk = current_equity * risk_bps / 10000
        
        # Strategy-specific adjustments
        if strat_name == "Overbot Vol Spike":
            vol_ratio = row_data['vol_ratio']
            if vol_ratio > 2.0:
                base_risk = current_equity * 45 / 10000
            elif vol_ratio > 1.5:
                base_risk = current_equity * 35 / 10000
        
        if strat_name == "Weak Close Decent Sznls":
            sznl_val = row_data['sznl']
            if sznl_val >= 65:
                base_risk *= 1.5
            elif sznl_val >= 33:
                base_risk *= 0.66 if sznl_val < 50 else 1.0
        
        direction = settings.get('trade_direction', 'Long')
        stop_atr = strat['execution']['stop_atr']
        
        if direction == 'Long':
            dist = atr * stop_atr
            action = "BUY"
        else:
            dist = atr * stop_atr
            action = "SELL SHORT"
        
        shares = int(base_risk / dist) if dist > 0 else 0
        if shares == 0:
            continue
        
        # Calculate exit
        entry_idx = df.index.get_loc(entry_date)
        exit_idx = min(entry_idx + hold_days, len(df) - 1)
        exit_row = df.iloc[exit_idx]
        exit_price = exit_row['Close']
        exit_date = exit_row.name
        
        if action == "BUY":
            pnl = (exit_price - entry_price) * shares
        else:
            pnl = (entry_price - exit_price) * shares
        
        # Time stop - calculate the intended exit date (may be in future)
        target_ts_idx = entry_idx + hold_days
        if target_ts_idx < len(df):
            time_stop_date = df.index[target_ts_idx]
        else:
            # Position hasn't reached time stop yet - calculate future date
            time_stop_date = entry_date + BusinessDay(hold_days)
        
        # Update state
        equity_at_signal = current_equity
        realized_pnl += pnl
        current_equity = starting_equity + realized_pnl
        position_last_exit[pos_key] = exit_date.value
        
        results.append({
            "Date": signal_date,
            "Entry Date": entry_date,
            "Exit Date": exit_date,
            "Time Stop": time_stop_date,
            "Strategy": strat_name,
            "Ticker": ticker,
            "Action": action,
            "Entry Criteria": entry_type,
            "Price": entry_price,
            "Shares": shares,
            "PnL": pnl,
            "ATR": atr,
            "Range %": row_data['range_pct'],
            "Equity at Signal": equity_at_signal,
            "Risk $": base_risk,
            "Risk bps": risk_bps
        })
    
    if not results:
        return pd.DataFrame()
    
    sig_df = pd.DataFrame(results)
    
    # Ensure datetime columns are proper Timestamps
    sig_df['Date'] = pd.to_datetime(sig_df['Date'])
    sig_df['Entry Date'] = pd.to_datetime(sig_df['Entry Date'])
    sig_df['Exit Date'] = pd.to_datetime(sig_df['Exit Date'])
    sig_df['Time Stop'] = pd.to_datetime(sig_df['Time Stop'])
    
    return sig_df.sort_values(by="Exit Date")


# -----------------------------------------------------------------------------
# MARK-TO-MARKET & STATS (unchanged but streamlined)
# -----------------------------------------------------------------------------
def get_daily_mtm_series(sig_df, master_dict):
    if sig_df.empty:
        return pd.Series(dtype=float)
    
    min_date = sig_df['Date'].min()
    max_date = max(sig_df['Exit Date'].max(), pd.Timestamp.today())
    all_dates = pd.date_range(start=min_date, end=max_date, freq='B')
    daily_pnl = pd.Series(0.0, index=all_dates)
    
    for _, trade in sig_df.iterrows():
        ticker = trade['Ticker'].replace('.', '-')
        action = trade['Action']
        shares = trade['Shares']
        entry_date = trade['Date']
        exit_date = trade['Exit Date']
        entry_price = trade['Price']
        
        t_df = master_dict.get(ticker)
        if t_df is None or t_df.empty:
            if exit_date in daily_pnl.index:
                daily_pnl[exit_date] += trade['PnL']
            continue
        
        # Normalize columns
        if isinstance(t_df.columns, pd.MultiIndex):
            t_df.columns = [c[0] if isinstance(c, tuple) else c for c in t_df.columns]
        t_df.columns = [c.capitalize() for c in t_df.columns]
        
        trade_dates = all_dates[(all_dates >= entry_date) & (all_dates <= exit_date)]
        if len(trade_dates) == 0:
            continue
            
        closes = t_df['Close'].reindex(trade_dates).ffill()
        if closes.empty:
            continue
        
        # First day PnL
        first_date = trade_dates[0]
        if first_date in closes.index:
            if action == "BUY":
                daily_pnl[first_date] += (closes[first_date] - entry_price) * shares
            else:
                daily_pnl[first_date] += (entry_price - closes[first_date]) * shares
        
        # Subsequent days
        if len(trade_dates) > 1:
            diffs = closes.diff().dropna()
            if action == "SELL SHORT":
                diffs = -diffs
            for d, val in (diffs * shares).items():
                if d in daily_pnl.index:
                    daily_pnl[d] += val
    
    return daily_pnl


def calculate_mark_to_market_curve(sig_df, master_dict):
    daily_pnl = get_daily_mtm_series(sig_df, master_dict)
    if daily_pnl.empty:
        return pd.DataFrame(columns=['Equity'])
    return daily_pnl.cumsum().to_frame(name='Equity')


def calculate_daily_exposure(sig_df, starting_equity=None):
    """
    Calculate daily exposure. If starting_equity is provided, returns as % of equity.
    """
    if sig_df.empty:
        return pd.DataFrame()
    min_date = sig_df['Date'].min()
    max_date = sig_df['Exit Date'].max()
    all_dates = pd.date_range(start=min_date, end=max_date)
    exposure_df = pd.DataFrame(0.0, index=all_dates, columns=['Long Exposure', 'Short Exposure'])
    
    for _, row in sig_df.iterrows():
        trade_dates = pd.date_range(start=row['Date'], end=row['Exit Date'])
        dollar_val = row['Price'] * row['Shares']
        col = 'Long Exposure' if row['Action'] == 'BUY' else 'Short Exposure'
        exposure_df.loc[exposure_df.index.isin(trade_dates), col] += dollar_val
    
    exposure_df['Net Exposure'] = exposure_df['Long Exposure'] - exposure_df['Short Exposure']
    exposure_df['Gross Exposure'] = exposure_df['Long Exposure'] + exposure_df['Short Exposure']
    
    # Convert to percentage of equity if starting_equity provided
    if starting_equity is not None:
        # Build running equity series based on realized PnL
        # For simplicity, use starting equity (could enhance to use MTM equity)
        equity_series = pd.Series(starting_equity, index=all_dates)
        
        # Calculate cumulative realized PnL up to each date
        for date in all_dates:
            closed_trades = sig_df[sig_df['Exit Date'] <= date]
            realized_pnl = closed_trades['PnL'].sum()
            equity_series[date] = starting_equity + realized_pnl
        
        # Convert to percentages
        exposure_df['Long Exposure %'] = (exposure_df['Long Exposure'] / equity_series) * 100
        exposure_df['Short Exposure %'] = (exposure_df['Short Exposure'] / equity_series) * 100
        exposure_df['Net Exposure %'] = (exposure_df['Net Exposure'] / equity_series) * 100
        exposure_df['Gross Exposure %'] = (exposure_df['Gross Exposure'] / equity_series) * 100
        
        # Return only percentage columns
        return exposure_df[['Long Exposure %', 'Short Exposure %', 'Net Exposure %', 'Gross Exposure %']]
    
    return exposure_df


def calculate_annual_stats(daily_pnl_series, starting_equity):
    if daily_pnl_series.empty:
        return pd.DataFrame()
    
    equity_series = starting_equity + daily_pnl_series.cumsum()
    daily_rets = equity_series.pct_change().fillna(0)
    
    yearly_stats = []
    for year_date, rets in daily_rets.resample('YE'):
        year = year_date.year
        if rets.empty:
            continue
        
        year_mask = equity_series.index.year == year
        year_eq = equity_series[year_mask]
        if len(year_eq) < 2:
            continue
            
        year_start = year_eq.iloc[0]
        year_end = year_eq.iloc[-1]
        
        total_ret_pct = (year_end - year_start) / year_start if year_start != 0 else 0
        total_ret_dollar = year_end - year_start
        
        std_dev = rets.std() * np.sqrt(252)
        mean_ret = rets.mean() * 252
        sharpe = mean_ret / std_dev if std_dev != 0 else 0
        
        neg_rets = rets[rets < 0]
        downside_std = np.sqrt((neg_rets**2).mean()) * np.sqrt(252) if len(neg_rets) > 0 else 0
        sortino = mean_ret / downside_std if downside_std != 0 else 0
        
        running_max = year_eq.expanding().max()
        drawdown = (year_eq - running_max) / running_max
        max_dd = drawdown.min()
        
        yearly_stats.append({
            "Year": year,
            "Total Return ($)": total_ret_dollar,
            "Total Return (%)": total_ret_pct,
            "Max Drawdown": max_dd,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Std Dev": std_dev
        })
    
    return pd.DataFrame(yearly_stats)


def calculate_performance_stats(sig_df, master_dict, starting_equity):
    stats = []
    
    def get_metrics(df, name, calc_ratios=True):
        if df.empty:
            return None
        
        count = len(df)
        total_pnl = df['PnL'].sum()
        gross_profit = df[df['PnL'] > 0]['PnL'].sum()
        gross_loss = abs(df[df['PnL'] < 0]['PnL'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan
        avg_pnl = df['PnL'].mean()
        std_pnl = df['PnL'].std()
        sqn = (avg_pnl / std_pnl * np.sqrt(count)) if std_pnl != 0 else 0
        
        sharpe, sortino = np.nan, np.nan
        if calc_ratios:
            daily_mtm = get_daily_mtm_series(df, master_dict)
            if not daily_mtm.empty:
                equity = starting_equity + daily_mtm.cumsum()
                rets = equity.pct_change().fillna(0)
                mean_ret = rets.mean() * 252
                std_dev = rets.std() * np.sqrt(252)
                sharpe = mean_ret / std_dev if std_dev != 0 else 0
                neg = rets[rets < 0]
                down_std = np.sqrt((neg**2).mean()) * np.sqrt(252) if len(neg) > 0 else 0
                sortino = mean_ret / down_std if down_std != 0 else 0
        
        return {
            "Strategy": name, "Trades": count, "Total PnL": total_pnl,
            "Sharpe": sharpe, "Sortino": sortino,
            "Profit Factor": profit_factor, "SQN": sqn
        }
    
    for strat in sig_df['Strategy'].unique():
        m = get_metrics(sig_df[sig_df['Strategy'] == strat], strat, calc_ratios=False)
        if m:
            stats.append(m)
    
    total_m = get_metrics(sig_df, "TOTAL PORTFOLIO", calc_ratios=True)
    if total_m:
        stats.append(total_m)
    
    return pd.DataFrame(stats)


def analyze_signal_density(sig_df, window_days=0):
    """
    Analyze performance based on signal clustering/isolation.
    
    Args:
        sig_df: DataFrame of executed trades
        window_days: Days around signal to count neighbors (0 = same day only)
    
    Returns:
        DataFrame with density analysis by bucket
    """
    if sig_df.empty:
        return pd.DataFrame(), sig_df
    
    df = sig_df.copy()
    
    # Count signals per day
    signal_counts = df.groupby('Date').size().to_dict()
    
    # For each trade, get the signal count on that day
    df['Signals Same Day'] = df['Date'].map(signal_counts)
    
    # If window > 0, count signals in surrounding days too
    if window_days > 0:
        def count_nearby_signals(date):
            nearby = df[(df['Date'] >= date - pd.Timedelta(days=window_days)) & 
                       (df['Date'] <= date + pd.Timedelta(days=window_days))]
            return len(nearby)
        df['Signals in Window'] = df['Date'].apply(count_nearby_signals)
        density_col = 'Signals in Window'
    else:
        density_col = 'Signals Same Day'
    
    # Create density buckets
    def get_density_bucket(count):
        if count == 1:
            return '1 (Isolated)'
        elif count <= 3:
            return '2-3 (Low)'
        elif count <= 6:
            return '4-6 (Moderate)'
        elif count <= 10:
            return '7-10 (High)'
        else:
            return '11+ (Clustered)'
    
    df['Density Bucket'] = df[density_col].apply(get_density_bucket)
    
    # Calculate R-multiple (PnL / Risk)
    df['R-Multiple'] = df['PnL'] / df['Risk $']
    
    # Aggregate stats by density bucket
    results = []
    bucket_order = ['1 (Isolated)', '2-3 (Low)', '4-6 (Moderate)', '7-10 (High)', '11+ (Clustered)']
    
    for bucket in bucket_order:
        bucket_df = df[df['Density Bucket'] == bucket]
        if bucket_df.empty:
            continue
        
        n_trades = len(bucket_df)
        total_pnl = bucket_df['PnL'].sum()
        total_risk = bucket_df['Risk $'].sum()
        
        winners = bucket_df[bucket_df['PnL'] > 0]
        losers = bucket_df[bucket_df['PnL'] <= 0]
        
        win_rate = len(winners) / n_trades if n_trades > 0 else 0
        avg_r = bucket_df['R-Multiple'].mean()
        
        avg_win_r = winners['R-Multiple'].mean() if len(winners) > 0 else 0
        avg_loss_r = losers['R-Multiple'].mean() if len(losers) > 0 else 0
        
        pnl_per_risk = total_pnl / total_risk if total_risk > 0 else 0
        
        # Profit factor
        gross_profit = winners['PnL'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['PnL'].sum()) if len(losers) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
        
        results.append({
            'Density': bucket,
            'Trades': n_trades,
            '% of Trades': n_trades / len(df),
            'Win Rate': win_rate,
            'Avg R': avg_r,
            'Avg Win R': avg_win_r,
            'Avg Loss R': avg_loss_r,
            'PnL/$ Risk': pnl_per_risk,
            'Profit Factor': profit_factor,
            'Total PnL': total_pnl,
            '% of PnL': total_pnl / df['PnL'].sum() if df['PnL'].sum() != 0 else 0
        })
    
    return pd.DataFrame(results), df


def analyze_density_by_strategy(sig_df):
    """
    Break down signal density performance by strategy.
    """
    if sig_df.empty:
        return pd.DataFrame()
    
    df = sig_df.copy()
    
    # Add signal counts
    signal_counts = df.groupby('Date').size().to_dict()
    df['Signals Same Day'] = df['Date'].map(signal_counts)
    df['R-Multiple'] = df['PnL'] / df['Risk $']
    
    # Simple split: isolated (1-3) vs clustered (4+)
    df['Is Isolated'] = df['Signals Same Day'] <= 3
    
    results = []
    
    for strat in df['Strategy'].unique():
        strat_df = df[df['Strategy'] == strat]
        
        isolated = strat_df[strat_df['Is Isolated']]
        clustered = strat_df[~strat_df['Is Isolated']]
        
        iso_r = isolated['R-Multiple'].mean() if len(isolated) > 0 else np.nan
        clu_r = clustered['R-Multiple'].mean() if len(clustered) > 0 else np.nan
        
        iso_wr = (isolated['PnL'] > 0).mean() if len(isolated) > 0 else np.nan
        clu_wr = (clustered['PnL'] > 0).mean() if len(clustered) > 0 else np.nan
        
        # Calculate the "isolation edge" - how much better are isolated signals?
        if not np.isnan(iso_r) and not np.isnan(clu_r) and clu_r != 0:
            edge = (iso_r - clu_r) / abs(clu_r) if clu_r != 0 else np.nan
        else:
            edge = np.nan
        
        results.append({
            'Strategy': strat,
            'Isolated Trades': len(isolated),
            'Clustered Trades': len(clustered),
            'Isolated Avg R': iso_r,
            'Clustered Avg R': clu_r,
            'Isolated WR': iso_wr,
            'Clustered WR': clu_wr,
            'Isolation Edge': edge,
            'Size Up Isolated?': 'Yes ✓' if (not np.isnan(edge) and edge > 0.15) else 'No'
        })
    
    return pd.DataFrame(results).sort_values('Isolation Edge', ascending=False)


def calculate_optimal_density_sizing(sig_df):
    """
    Calculate suggested sizing multipliers based on signal density.
    Returns a dict of density bucket -> multiplier.
    """
    density_df, _ = analyze_signal_density(sig_df)
    
    if density_df.empty:
        return {}
    
    # Use PnL/$ Risk as the efficiency metric
    # Normalize so average multiplier = 1.0
    avg_efficiency = density_df['PnL/$ Risk'].mean()
    
    multipliers = {}
    for _, row in density_df.iterrows():
        bucket = row['Density']
        efficiency = row['PnL/$ Risk']
        
        if avg_efficiency > 0:
            raw_mult = efficiency / avg_efficiency
            # Clamp between 0.5x and 2.0x
            multipliers[bucket] = max(0.5, min(2.0, raw_mult))
        else:
            multipliers[bucket] = 1.0
    
    return multipliers


def calculate_capital_efficiency(sig_df, strategies):
    """
    Calculate capital efficiency metrics for each strategy.
    Accounts for hold duration to measure time-adjusted returns.
    """
    if sig_df.empty:
        return pd.DataFrame()
    
    # Build lookup for strategy settings
    strat_settings = {s['name']: s for s in strategies}
    
    results = []
    
    for strat_name in sig_df['Strategy'].unique():
        strat_df = sig_df[sig_df['Strategy'] == strat_name].copy()
        
        if strat_df.empty:
            continue
        
        # Basic stats
        n_trades = len(strat_df)
        total_pnl = strat_df['PnL'].sum()
        total_risk = strat_df['Risk $'].sum()
        
        # Win rate and avg win/loss
        winners = strat_df[strat_df['PnL'] > 0]
        losers = strat_df[strat_df['PnL'] <= 0]
        win_rate = len(winners) / n_trades if n_trades > 0 else 0
        avg_win = winners['PnL'].mean() if len(winners) > 0 else 0
        avg_loss = abs(losers['PnL'].mean()) if len(losers) > 0 else 0
        
        # Calculate actual hold duration
        strat_df['Hold Days'] = (strat_df['Exit Date'] - strat_df['Entry Date']).dt.days
        avg_hold_days = strat_df['Hold Days'].mean()
        
        # Get configured hold days from strategy
        config_hold_days = strat_settings.get(strat_name, {}).get('execution', {}).get('hold_days', avg_hold_days)
        
        # Current risk_bps
        current_bps = strat_settings.get(strat_name, {}).get('execution', {}).get('risk_bps', 0)
        
        # Core efficiency metrics
        pnl_per_risk = total_pnl / total_risk if total_risk > 0 else 0
        
        # Time-adjusted: how much return per dollar risked per day
        pnl_per_risk_per_day = pnl_per_risk / avg_hold_days if avg_hold_days > 0 else 0
        
        # Annualized return on risk (assuming capital can be redeployed)
        capital_turns_per_year = 252 / avg_hold_days if avg_hold_days > 0 else 0
        annualized_ror = pnl_per_risk * capital_turns_per_year
        
        # Signals per year (approximate)
        date_range = (strat_df['Date'].max() - strat_df['Date'].min()).days
        years = date_range / 365.25 if date_range > 0 else 1
        signals_per_year = n_trades / years if years > 0 else n_trades
        
        # Risk contribution (% of total risk this strategy consumed)
        total_portfolio_risk = sig_df['Risk $'].sum()
        risk_contribution = total_risk / total_portfolio_risk if total_portfolio_risk > 0 else 0
        
        # PnL contribution
        total_portfolio_pnl = sig_df['PnL'].sum()
        pnl_contribution = total_pnl / total_portfolio_pnl if total_portfolio_pnl > 0 else 0
        
        # Efficiency ratio: are you getting your fair share of PnL for the risk taken?
        efficiency_ratio = pnl_contribution / risk_contribution if risk_contribution > 0 else 0
        
        results.append({
            'Strategy': strat_name,
            'Trades': n_trades,
            'Signals/Yr': signals_per_year,
            'Avg Days': avg_hold_days,
            'Win Rate': win_rate,
            'Total $ Risked': total_risk,
            'Total PnL': total_pnl,
            'PnL/$ Risk': pnl_per_risk,
            'Ann. Turns': capital_turns_per_year,
            'Ann. RoR': annualized_ror,
            '% of Risk': risk_contribution,
            '% of PnL': pnl_contribution,
            'Efficiency': efficiency_ratio,
            'Current Bps': current_bps,
        })
    
    df = pd.DataFrame(results)
    
    if df.empty:
        return df
    
    # Calculate suggested bps based on annualized RoR
    # Normalize so total suggested bps equals current total bps
    total_current_bps = df['Current Bps'].sum()
    
    # Weight by annualized RoR (higher = more allocation)
    df['RoR Weight'] = df['Ann. RoR'] / df['Ann. RoR'].sum() if df['Ann. RoR'].sum() > 0 else 1 / len(df)
    df['Suggested Bps'] = (df['RoR Weight'] * total_current_bps).round(0).astype(int)
    
    # Ensure minimum of 10 bps for any strategy
    df['Suggested Bps'] = df['Suggested Bps'].clip(lower=10)
    
    # Calculate the change
    df['Bps Δ'] = df['Suggested Bps'] - df['Current Bps']
    
    # Sort by annualized RoR descending
    df = df.sort_values('Ann. RoR', ascending=False)
    
    return df


# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Strategy Backtest Lab v3")
    st.sidebar.header("⚙️ Backtest Settings")

    if st.sidebar.button("🔴 Force Clear Cache & Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        if 'backtest_data' in st.session_state:
            del st.session_state['backtest_data']
        st.rerun()
    
    current_year = datetime.date.today().year
    selected_year = st.sidebar.slider("Select Start Year", 2000, current_year, current_year - 2)
    default_date = datetime.date(selected_year, 1, 1)
    
    with st.sidebar.form("backtest_form"):
        user_start_date = st.date_input("Backtest Start Date", value=default_date, min_value=datetime.date(2000, 1, 1))
        starting_equity = st.number_input("Starting Equity ($)", value=150000, step=10000)
        st.caption(f"Data buffer: 365 days prior to {user_start_date}.")
        st.markdown("---")
        st.markdown("**🔄 Dynamic Position Sizing**")
        st.caption("Sizes scale with equity (bps of current value).")
        run_btn = st.form_submit_button("⚡ Run Backtest")

    st.title("⚡ Strategy Backtest Lab v3")
    st.markdown(f"**Start:** {user_start_date} | **Equity:** ${starting_equity:,.0f}")
    st.info("💡 **v3:** Optimized for speed. Position sizes scale dynamically with equity.")
    st.markdown("---")

    if run_btn:
        sznl_map = load_seasonal_map()
        if 'backtest_data' not in st.session_state:
            st.session_state['backtest_data'] = {}

        import copy
        strategies = [copy.deepcopy(s) for s in _STRATEGY_BOOK_RAW]

        # Collect tickers
        long_term_tickers = set()
        for strat in strategies:
            long_term_tickers.update(strat['universe_tickers'])
            s = strat['settings']
            if s.get('use_market_sznl'):
                long_term_tickers.add(s.get('market_ticker', '^GSPC'))
            if "Market" in s.get('trend_filter', ''):
                long_term_tickers.add(s.get('market_ticker', 'SPY'))
            if s.get('use_vix_filter'):
                long_term_tickers.add('^VIX')
        long_term_tickers.add('SPY')  # Always need for market filter

        long_term_list = [t.replace('.', '-') for t in long_term_tickers]
        existing = set(st.session_state['backtest_data'].keys())
        missing = list(set(long_term_list) - existing)
        
        if missing:
            st.write(f"📥 Downloading {len(missing)} tickers...")
            data = download_historical_data(missing, start_date="2000-01-01")
            st.session_state['backtest_data'].update(data)
            st.success("✅ Download complete.")

        master_dict = st.session_state['backtest_data']
        
        # VIX series
        vix_df = master_dict.get('^VIX')
        vix_series = None
        if vix_df is not None and not vix_df.empty:
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = [c[0] if isinstance(c, tuple) else c for c in vix_df.columns]
            vix_df.columns = [c.capitalize() for c in vix_df.columns]
            vix_series = vix_df['Close']

        # PHASE 1: Pre-compute indicators (cached)
        st.write("📊 **Phase 1:** Computing indicators...")
        t0 = time.time()
        processed_dict = precompute_all_indicators(master_dict, strategies, sznl_map, vix_series)
        st.write(f"   Processed {len(processed_dict)} tickers in {time.time()-t0:.1f}s")

        # PHASE 2: Generate candidates
        st.write("🔍 **Phase 2:** Finding signals...")
        t0 = time.time()
        candidates, signal_data = generate_candidates_fast(processed_dict, strategies, sznl_map, user_start_date)
        st.write(f"   Found {len(candidates):,} candidates in {time.time()-t0:.1f}s")

        # PHASE 3: Process chronologically
        st.write("📈 **Phase 3:** Processing with dynamic sizing...")
        t0 = time.time()
        sig_df = process_signals_fast(candidates, signal_data, processed_dict, strategies, starting_equity)
        st.write(f"   Executed {len(sig_df):,} trades in {time.time()-t0:.1f}s")

        if not sig_df.empty:
            st.success(f"✅ Backtest complete: {len(sig_df):,} trades")
            
            # Current positions
            today = pd.Timestamp(datetime.date.today())
            open_mask = sig_df['Time Stop'] >= today
            open_df = sig_df[open_mask].copy()

            if not open_df.empty:
                current_prices, open_pnls, current_values = [], [], []
                for _, row in open_df.iterrows():
                    t_df = master_dict.get(row['Ticker'].replace('.', '-'))
                    if t_df is not None and not t_df.empty:
                        if isinstance(t_df.columns, pd.MultiIndex):
                            t_df.columns = [c[0] if isinstance(c, tuple) else c for c in t_df.columns]
                        t_df.columns = [c.capitalize() for c in t_df.columns]
                        last_close = t_df['Close'].iloc[-1]
                    else:
                        last_close = row['Price']
                    
                    if row['Action'] == 'BUY':
                        pnl = (last_close - row['Price']) * row['Shares']
                    else:
                        pnl = (row['Price'] - last_close) * row['Shares']
                    
                    current_prices.append(last_close)
                    open_pnls.append(pnl)
                    current_values.append(last_close * row['Shares'])

                open_df['Current Price'] = current_prices
                open_df['Open PnL'] = open_pnls
                open_df['Mkt Value'] = current_values
                
                total_long = open_df[open_df['Action'] == 'BUY']['Mkt Value'].sum()
                total_short = open_df[open_df['Action'] == 'SELL SHORT']['Mkt Value'].sum()
                net_exposure = total_long - total_short
                total_open_pnl = open_df['Open PnL'].sum()
                
                st.divider()
                st.subheader("💼 Current Exposure (Active Positions)")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("# Positions", len(open_df))
                m2.metric("Total Long", f"${total_long:,.0f}")
                m3.metric("Total Short", f"${total_short:,.0f}")
                m4.metric("Net Exposure", f"${net_exposure:,.0f}")
                m5.metric("Total Open PnL", f"${total_open_pnl:,.2f}", delta_color="normal", delta=f"{total_open_pnl:,.2f}")
                
                st.dataframe(open_df.style.format({
                    "Date": "{:%Y-%m-%d}", "Entry Date": "{:%Y-%m-%d}", "Time Stop": "{:%Y-%m-%d}",
                    "Price": "${:.2f}", "Current Price": "${:.2f}", "Open PnL": "${:,.2f}",
                    "Range %": "{:.1f}%", "Equity at Signal": "${:,.0f}", "Risk $": "${:,.0f}"
                }), use_container_width=True)
            else:
                st.divider()
                st.info("No active positions (Time Stop >= Today).")

            st.divider()
            
            # Position sizing analysis
            st.subheader("📐 Dynamic Sizing Analysis")
            cols = st.columns(3)
            cols[0].metric("Avg Risk/Trade", f"${sig_df['Risk $'].mean():,.0f}")
            cols[1].metric("Risk Range", f"${sig_df['Risk $'].min():,.0f} - ${sig_df['Risk $'].max():,.0f}")
            final_eq = starting_equity + sig_df['PnL'].sum()
            cols[2].metric("Final Equity", f"${final_eq:,.0f}", delta=f"{(final_eq/starting_equity-1)*100:.1f}%")
            
            # Annual stats
            st.subheader("📅 Annual Performance")
            port_daily_pnl = get_daily_mtm_series(sig_df, master_dict)
            annual_df = calculate_annual_stats(port_daily_pnl, starting_equity)
            if not annual_df.empty:
                st.dataframe(annual_df.style.format({
                    "Total Return ($)": "${:,.0f}", "Total Return (%)": "{:.1%}", "Max Drawdown": "{:.1%}",
                    "Sharpe Ratio": "{:.2f}", "Sortino Ratio": "{:.2f}", "Std Dev": "{:.1%}"
                }), use_container_width=True)

            # Strategy metrics
            st.subheader("📊 Strategy Metrics")
            stats_df = calculate_performance_stats(sig_df, master_dict, starting_equity)
            st.dataframe(stats_df.style.format({
                "Total PnL": "${:,.0f}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
                "Profit Factor": "{:.2f}", "SQN": "{:.2f}"
            }), use_container_width=True)

            # Capital Efficiency Analysis
            st.subheader("💰 Capital Efficiency & Sizing Analysis")
            st.caption("Accounts for hold duration - shorter trades can redeploy capital more often, making them more efficient per unit of time.")
            
            efficiency_df = calculate_capital_efficiency(sig_df, strategies)
            
            if not efficiency_df.empty:
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                most_efficient = efficiency_df.iloc[0]['Strategy']
                best_ror = efficiency_df.iloc[0]['Ann. RoR']
                col1.metric("Most Efficient Strategy", most_efficient, delta=f"{best_ror:.1%} Ann. RoR")
                
                under_allocated = efficiency_df[efficiency_df['Efficiency'] > 1.2]
                over_allocated = efficiency_df[efficiency_df['Efficiency'] < 0.8]
                col2.metric("Under-allocated", f"{len(under_allocated)} strategies", help="Efficiency > 1.2x: earning more than their share of risk")
                col3.metric("Over-allocated", f"{len(over_allocated)} strategies", help="Efficiency < 0.8x: earning less than their share of risk")
                
                # Main efficiency table
                display_cols = ['Strategy', 'Trades', 'Signals/Yr', 'Avg Days', 'Win Rate', 
                               'PnL/$ Risk', 'Ann. RoR', '% of Risk', '% of PnL', 'Efficiency',
                               'Current Bps', 'Suggested Bps', 'Bps Δ']
                
                # Color code the efficiency and Bps change
                def highlight_efficiency(val):
                    if val > 1.2:
                        return 'background-color: #1a472a; color: white'  # Dark green - under-allocated
                    elif val < 0.8:
                        return 'background-color: #4a1a1a; color: white'  # Dark red - over-allocated
                    else:
                        return ''
                
                def highlight_bps_change(val):
                    if val > 10:
                        return 'background-color: #1a472a; color: white'  # Increase
                    elif val < -10:
                        return 'background-color: #4a1a1a; color: white'  # Decrease
                    else:
                        return ''
                
                styled_df = efficiency_df[display_cols].style.format({
                    'Signals/Yr': '{:.1f}',
                    'Avg Days': '{:.1f}',
                    'Win Rate': '{:.1%}',
                    'PnL/$ Risk': '{:.2f}',
                    'Ann. RoR': '{:.1%}',
                    '% of Risk': '{:.1%}',
                    '% of PnL': '{:.1%}',
                    'Efficiency': '{:.2f}x',
                    'Current Bps': '{:.0f}',
                    'Suggested Bps': '{:.0f}',
                    'Bps Δ': '{:+.0f}'
                }).applymap(highlight_efficiency, subset=['Efficiency']
                ).applymap(highlight_bps_change, subset=['Bps Δ'])
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Explanation
                with st.expander("📖 How to interpret this table"):
                    st.markdown("""
                    **Key Columns:**
                    - **Avg Days**: Average hold duration per trade
                    - **PnL/$ Risk**: Raw return per dollar risked (ignores time)
                    - **Ann. RoR**: Annualized Return on Risk = PnL/$ Risk × (252 / Avg Days). This accounts for capital turnover.
                    - **% of Risk**: What portion of your total risk budget this strategy consumes
                    - **% of PnL**: What portion of total profits this strategy generates
                    - **Efficiency**: % of PnL ÷ % of Risk. 
                        - **> 1.0** = Under-allocated (earning more than its share)
                        - **< 1.0** = Over-allocated (earning less than its share)
                    - **Suggested Bps**: Recommended sizing based on Annualized RoR (re-normalizes total bps budget)
                    - **Bps Δ**: Change from current sizing
                    
                    **Strategy:**
                    - Increase allocation to high Ann. RoR strategies (short hold, good returns)
                    - Decrease allocation to low Ann. RoR strategies (long hold, mediocre returns)
                    - Strategies with Efficiency > 1.2x are your workhorses - they're under-sized
                    - Strategies with Efficiency < 0.8x are dragging on performance - consider reducing
                    """)
                
                # Visual: Ann. RoR bar chart
                st.markdown("**Annualized Return on Risk by Strategy**")
                chart_df = efficiency_df[['Strategy', 'Ann. RoR']].set_index('Strategy')
                st.bar_chart(chart_df)

            # Signal Density Analysis
            st.subheader("🎯 Signal Density Analysis")
            st.caption("Do isolated signals perform better than clustered ones? Should you size up when fewer signals fire?")
            
            density_df, sig_df_with_density = analyze_signal_density(sig_df, window_days=0)
            
            if not density_df.empty:
                # Summary insight
                isolated_row = density_df[density_df['Density'] == '1 (Isolated)']
                clustered_row = density_df[density_df['Density'] == '11+ (Clustered)']
                
                if not isolated_row.empty and not clustered_row.empty:
                    iso_r = isolated_row['Avg R'].values[0]
                    clu_r = clustered_row['Avg R'].values[0]
                    if clu_r != 0:
                        edge_pct = ((iso_r - clu_r) / abs(clu_r)) * 100
                        if edge_pct > 0:
                            st.success(f"✅ Isolated signals outperform clustered by **{edge_pct:.1f}%** on average R-multiple")
                        else:
                            st.warning(f"⚠️ Clustered signals actually outperform isolated by **{abs(edge_pct):.1f}%** - sizing up isolated may not help")
                
                # Main density table
                st.markdown("**Performance by Signal Density (Same Day)**")
                
                def highlight_avg_r(val):
                    if val > 0.3:
                        return 'background-color: #1a472a; color: white'
                    elif val < 0.1:
                        return 'background-color: #4a1a1a; color: white'
                    return ''
                
                display_cols = ['Density', 'Trades', '% of Trades', 'Win Rate', 'Avg R', 
                               'Avg Win R', 'Avg Loss R', 'PnL/$ Risk', 'Profit Factor', 'Total PnL']
                
                styled_density = density_df[display_cols].style.format({
                    '% of Trades': '{:.1%}',
                    'Win Rate': '{:.1%}',
                    'Avg R': '{:.2f}',
                    'Avg Win R': '{:.2f}',
                    'Avg Loss R': '{:.2f}',
                    'PnL/$ Risk': '{:.2f}',
                    'Profit Factor': '{:.2f}',
                    'Total PnL': '${:,.0f}'
                }).applymap(highlight_avg_r, subset=['Avg R'])
                
                st.dataframe(styled_density, use_container_width=True)
                
                # Strategy-level breakdown
                with st.expander("📊 Density Analysis by Strategy"):
                    strat_density_df = analyze_density_by_strategy(sig_df)
                    if not strat_density_df.empty:
                        st.dataframe(strat_density_df.style.format({
                            'Isolated Avg R': '{:.2f}',
                            'Clustered Avg R': '{:.2f}',
                            'Isolated WR': '{:.1%}',
                            'Clustered WR': '{:.1%}',
                            'Isolation Edge': '{:.1%}'
                        }), use_container_width=True)
                        
                        st.markdown("""
                        **Isolation Edge** = (Isolated Avg R - Clustered Avg R) / |Clustered Avg R|
                        - Positive = isolated signals are better → size them up
                        - Negative = clustered signals are better → don't penalize clustering
                        - **Size Up Isolated?** = Yes if edge > 15%
                        """)
                
                # Suggested multipliers
                with st.expander("🔧 Suggested Density-Based Sizing Multipliers"):
                    multipliers = calculate_optimal_density_sizing(sig_df)
                    
                    if multipliers:
                        mult_df = pd.DataFrame([
                            {'Density': k, 'Multiplier': v, 'Effect': f"{v:.2f}x base risk"}
                            for k, v in multipliers.items()
                        ])
                        st.dataframe(mult_df, use_container_width=True)
                        
                        st.markdown("""
                        **How to use:**
                        - When signal fires, count how many other signals are firing that day
                        - Look up the multiplier for that density bucket
                        - Multiply your base `risk_bps` by the multiplier
                        
                        **Example:** If "1 (Isolated)" has 1.5x multiplier and base risk is 30 bps,
                        use 45 bps when only 1 signal fires that day.
                        """)
                        
                        st.info("💡 To implement this, you'd add logic in `process_signals_fast` to count same-day signals and adjust `base_risk` accordingly.")

            # Charts
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Portfolio PnL (MTM)")
                df_eq = calculate_mark_to_market_curve(sig_df, master_dict)
                if not df_eq.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq['Equity'], mode='lines', name='PnL', line=dict(color='#00FF00', width=2)))
                    fig.update_layout(height=350, margin=dict(l=10,r=10,t=30,b=10))
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📉 PnL by Strategy")
                strat_pnl = sig_df.pivot_table(index='Exit Date', columns='Strategy', values='PnL', aggfunc='sum').fillna(0)
                st.line_chart(strat_pnl.cumsum())

            # Exposure
            st.subheader("⚖️ Exposure Over Time (% of Equity)")
            exposure_df = calculate_daily_exposure(sig_df, starting_equity=starting_equity)
            if not exposure_df.empty:
                st.line_chart(exposure_df)
                
                # Summary stats
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Gross Exposure", f"{exposure_df['Gross Exposure %'].mean():.1f}%")
                col2.metric("Max Gross Exposure", f"{exposure_df['Gross Exposure %'].max():.1f}%")
                col3.metric("Avg Net Exposure", f"{exposure_df['Net Exposure %'].mean():.1f}%")
                col4.metric("Max Net Exposure", f"{exposure_df['Net Exposure %'].max():.1f}%")

            # Trade log
            st.subheader("📜 Trade Log")
            display_cols = ["Date", "Entry Date", "Exit Date", "Strategy", "Ticker", "Action",
                          "Price", "Shares", "PnL", "Equity at Signal", "Risk $"]
            st.dataframe(sig_df[display_cols].sort_values("Date", ascending=False).style.format({
                "Price": "${:.2f}", "PnL": "${:,.0f}", "Date": "{:%Y-%m-%d}",
                "Entry Date": "{:%Y-%m-%d}", "Exit Date": "{:%Y-%m-%d}",
                "Equity at Signal": "${:,.0f}", "Risk $": "${:,.0f}"
            }), use_container_width=True, height=400)
        else:
            st.warning(f"No signals found starting from {user_start_date}.")
    else:
        st.info("👈 Configure settings and click 'Run Backtest' to begin.")


if __name__ == "__main__":
    main()
