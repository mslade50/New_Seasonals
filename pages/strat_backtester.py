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
    
    # Handle MultiIndex and Column Names
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
    
    # --- Dynamic SMAs ---
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
    mkt_ticker_ref = "^GSPC" 
    df['Mkt_Sznl_Ref'] = get_sznl_val_series(mkt_ticker_ref, df.index, sznl_map)
    
    # --- 52w High/Low ---
    rolling_high = df['High'].shift(1).rolling(252).max()
    rolling_low = df['Low'].shift(1).rolling(252).min()
    df['is_52w_high'] = df['High'] > rolling_high
    df['is_52w_low'] = df['Low'] < rolling_low
    
    # --- Volume & Vol Ratio ---
    vol_ma = df['Volume'].rolling(63).mean()
    df['vol_ma'] = vol_ma
    df['vol_ratio'] = df['Volume'] / vol_ma
    
    # 10d Relative Volume Percentile
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
        
    # --- External Series (Market/VIX) ---
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
    # Dynamic Gap Window
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
    
    # --- Helper for Breakout Mode ---
    df['PrevHigh'] = df['High'].shift(1)
    df['PrevLow'] = df['Low'].shift(1)

    return df

def get_historical_mask(df, params, sznl_map, ticker_name="UNK"):
    mask = pd.Series(True, index=df.index)
    
    # --- 1. TREND FILTER ---
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

    # --- 2. LIQUIDITY & GATES ---
    mask &= (df['Close'] >= params.get('min_price', 0))
    mask &= (df['vol_ma'] >= params.get('min_vol', 0))
    mask &= (df['age_years'] >= params.get('min_age', 0))
    mask &= (df['age_years'] <= params.get('max_age', 100))
    
    if 'ATR_Pct' in df.columns:
        mask &= (df['ATR_Pct'] >= params.get('min_atr_pct', 0)) & (df['ATR_Pct'] <= params.get('max_atr_pct', 1000))
        
    if params.get('require_close_gt_open', False):
        mask &= (df['Close'] > df['Open'])

    # --- 3. BREAKOUT MODE ---
    bk_mode = params.get('breakout_mode', 'None')
    if bk_mode == "Close > Prev Day High":
        mask &= (df['Close'] > df['PrevHigh'])
    elif bk_mode == "Close < Prev Day Low":
        mask &= (df['Close'] < df['PrevLow'])

    # --- 4. RANGE % ---
    if params.get('use_range_filter', False):
        rn_val = df['RangePct'] * 100
        mask &= (rn_val >= params.get('range_min', 0)) & (rn_val <= params.get('range_max', 100))

    # --- 5. DAY OF WEEK ---
    if params.get('use_dow_filter', False):
        allowed = params.get('allowed_days', [])
        if allowed:
            mask &= df['DayOfWeekVal'].isin(allowed)

    # --- 6. CYCLE ---
    if 'allowed_cycles' in params:
        allowed = params['allowed_cycles']
        if allowed and len(allowed) < 4:
            year_rems = df.index.year % 4
            mask &= pd.Series(year_rems, index=df.index).isin(allowed)

    # --- 7. GAP FILTER ---
    if params.get('use_gap_filter', False):
        g_val = df['GapCount']
        g_thresh = params.get('gap_thresh', 0)
        g_logic = params.get('gap_logic', '>')
        if g_logic == ">": mask &= (g_val > g_thresh)
        elif g_logic == "<": mask &= (g_val < g_thresh)
        elif g_logic == "=": mask &= (g_val == g_thresh)

    # --- 8. ACC/DIST COUNTS ---
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

    # --- 9. PERFORMANCE RANK ---
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

    # --- 10. MA CONSECUTIVE ---
    for f in params.get('ma_consec_filters', []):
        col = f"SMA{f['length']}"
        if col in df.columns:
            cond = (df['Close'] > df[col]) if f['logic'] == 'Above' else (df['Close'] < df[col])
            if f['consec'] > 1: cond = cond.rolling(f['consec']).sum() == f['consec']
            mask &= cond

    # --- 11. TICKER SEASONALITY ---
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

    # --- 12. 52-WEEK HIGH/LOW ---
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

    # --- 13. VOLUME FILTERS ---
    if params.get('vol_gt_prev', False):
        mask &= (df['Volume'] > df['Volume'].shift(1))
    
    if params.get('use_vol', False):
        mask &= (df['vol_ratio'] > params['vol_thresh'])

    if params.get('use_vol_rank', False):
        col = 'vol_ratio_10d_rank'
        if params['vol_rank_logic'] == ">": mask &= (df[col] > params['vol_rank_thresh'])
        elif params['vol_rank_logic'] == "<": mask &= (df[col] < params['vol_rank_thresh'])

    # --- 14. MA DISTANCE ---
    if params.get('use_ma_dist_filter', False):
        ma_col_map = {"SMA 10": "SMA10", "SMA 20": "SMA20", "SMA 50": "SMA50", "SMA 100": "SMA100", "SMA 200": "SMA200", 
                      "EMA 8": "EMA8", "EMA 11": "EMA11", "EMA 21": "EMA21"}
        ma_target = ma_col_map.get(params['dist_ma_type'])
        if ma_target and ma_target in df.columns:
            dist_val = (df['Close'] - df[ma_target]) / df['ATR']
            if params['dist_logic'] == "Greater Than (>)": mask &= (dist_val > params['dist_min'])
            elif params['dist_logic'] == "Less Than (<)": mask &= (dist_val < params['dist_max'])
            elif params['dist_logic'] == "Between": mask &= (dist_val >= params['dist_min']) & (dist_val <= params['dist_max'])

    # --- 15. VIX REGIME ---
    if params.get('use_vix_filter', False) and 'VIX_Value' in df.columns:
        mask &= (df['VIX_Value'] >= params.get('vix_min', 0)) & (df['VIX_Value'] <= params.get('vix_max', 100))

    # --- 16. MA TOUCH LOGIC ---
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
# PHASE 1: GENERATE ALL CANDIDATE SIGNALS (without sizing)
# -----------------------------------------------------------------------------
def generate_candidate_signals(master_dict, strategies, sznl_map, user_start_date, vix_series=None):
    """
    Generate all candidate signal dates across all strategies/tickers.
    Returns a list of dicts with signal metadata (no position sizing yet).
    """
    candidates = []
    
    for strat in strategies:
        settings = strat['settings']
        
        # Get market series for this strategy
        strat_mkt_ticker = settings.get('market_ticker', 'SPY')
        mkt_df = master_dict.get(strat_mkt_ticker)
        if mkt_df is None: 
            mkt_df = master_dict.get('SPY')
        
        market_series = None
        if mkt_df is not None:
            temp_mkt = mkt_df.copy()
            temp_mkt['SMA200'] = temp_mkt['Close'].rolling(200).mean()
            market_series = temp_mkt['Close'] > temp_mkt['SMA200']

        # Extract strategy-specific parameters
        gap_win = settings.get('gap_lookback', 21)
        acc_win = settings.get('acc_count_window') if settings.get('use_acc_count_filter') else None
        dist_win = settings.get('dist_count_window') if settings.get('use_dist_count_filter') else None
        req_custom_mas = list(set([f['length'] for f in settings.get('ma_consec_filters', [])]))
        
        if settings.get('use_ma_touch'):
            ma_type = settings.get('ma_touch_type', '')
            if 'SMA' in ma_type and ma_type not in ["SMA 10", "SMA 20", "SMA 50", "SMA 100", "SMA 200"]:
                try:
                    val = int(ma_type.replace("SMA", "").strip())
                    req_custom_mas.append(val)
                except: pass

        for ticker in strat['universe_tickers']:
            t_clean = ticker.replace('.', '-')
            df = master_dict.get(t_clean)
            if df is None or len(df) < 200: 
                continue
            
            try:
                df = calculate_indicators(
                    df, sznl_map, t_clean, market_series, vix_series,
                    gap_window=gap_win,
                    acc_window=acc_win,
                    dist_window=dist_win,
                    custom_sma_lengths=req_custom_mas
                )
                
                mask = get_historical_mask(df, settings, sznl_map, ticker)
                cutoff_ts = pd.Timestamp(user_start_date)
                mask = mask[mask.index >= cutoff_ts]
                if not mask.any(): 
                    continue

                true_dates = mask[mask].index
                
                for signal_date in true_dates:
                    row = df.loc[signal_date]
                    
                    # Store candidate with all info needed for later sizing
                    candidates.append({
                        'signal_date': signal_date,
                        'ticker': ticker,
                        't_clean': t_clean,
                        'strategy_name': strat['name'],
                        'strategy': strat,
                        'settings': settings,
                        'signal_row': row,
                        'df': df,  # Reference to full dataframe
                        'atr': row['ATR'],
                        'signal_close': row['Close'],
                        'vol_ratio': row.get('vol_ratio', 0),
                        'sznl': row.get('Sznl', 50),
                        'range_pct': row['RangePct'] * 100
                    })
            except Exception as e:
                continue
    
    return candidates


# -----------------------------------------------------------------------------
# PHASE 2: PROCESS SIGNALS CHRONOLOGICALLY WITH DYNAMIC SIZING
# -----------------------------------------------------------------------------
def process_signals_chronologically(candidates, master_dict, starting_equity):
    """
    Process all candidate signals in chronological order.
    Size each trade based on current equity at time of signal.
    Track open positions and running equity.
    """
    if not candidates:
        return pd.DataFrame()
    
    # Sort candidates by signal date
    candidates = sorted(candidates, key=lambda x: x['signal_date'])
    
    # State tracking
    current_equity = starting_equity
    realized_pnl = 0.0
    open_positions = {}  # key: (strategy_name, ticker), value: position dict
    position_last_exit = {}  # key: (strategy_name, ticker), value: last exit date
    
    executed_trades = []
    
    for cand in candidates:
        signal_date = cand['signal_date']
        ticker = cand['ticker']
        t_clean = cand['t_clean']
        strat = cand['strategy']
        settings = cand['settings']
        df = cand['df']
        
        # --- Check if we can take this position ---
        pos_key = (strat['name'], ticker)
        
        # Skip if already in position for this strategy/ticker
        if pos_key in open_positions:
            continue
        
        # Skip if last exit hasn't occurred yet (no re-entry until position closed)
        last_exit = position_last_exit.get(pos_key)
        if last_exit is not None and signal_date <= last_exit:
            continue
        
        # --- Determine entry details ---
        entry_type = settings.get('entry_type', 'Signal Close')
        entry_idx = df.index.get_loc(signal_date)
        
        if entry_idx + 1 >= len(df):
            continue
            
        entry_row = df.iloc[entry_idx + 1]
        signal_row = cand['signal_row']
        atr = cand['atr']
        hold_days = strat['execution']['hold_days']
        
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
            if entry_row['Close'] < signal_row['Close']:
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
            limit_base = signal_row['Close']
            found_fill = False
            max_days = hold_days
            search_end = min(entry_idx + 1 + max_days, len(df))
            
            for i in range(entry_idx + 1, search_end):
                check_row = df.iloc[i]
                if settings['trade_direction'] == 'Short':
                    limit_price = limit_base + limit_offset
                    if check_row['High'] >= limit_price:
                        entry_price = limit_price
                        entry_date = check_row.name
                        days_elapsed = i - entry_idx
                        hold_days = max(1, strat['execution']['hold_days'] - days_elapsed)
                        found_fill = True
                        break
                else:
                    limit_price = limit_base - limit_offset
                    if check_row['Low'] <= limit_price:
                        entry_price = limit_price
                        entry_date = check_row.name
                        days_elapsed = i - entry_idx
                        hold_days = max(1, strat['execution']['hold_days'] - days_elapsed)
                        found_fill = True
                        break
            valid_entry = found_fill
        else:
            entry_price = signal_row['Close']
            entry_date = signal_date
        
        if not valid_entry or entry_price is None:
            continue
        
        # --- DYNAMIC POSITION SIZING based on current equity ---
        risk_bps = strat['execution']['risk_bps']
        base_risk = current_equity * risk_bps / 10000
        
        # Strategy-specific risk adjustments
        if strat['name'] == "Overbot Vol Spike":
            vol_ratio = cand['vol_ratio']
            if vol_ratio > 2.0:
                base_risk = current_equity * 45 / 10000  # ~675 at 150k
            elif vol_ratio > 1.5:
                base_risk = current_equity * 35 / 10000  # ~525 at 150k
        
        if strat['name'] == "Weak Close Decent Sznls":
            sznl_val = cand['sznl']
            if sznl_val >= 65:
                base_risk = base_risk * 1.5
            elif sznl_val >= 50:
                base_risk = base_risk * 1.0
            elif sznl_val >= 33:
                base_risk = base_risk * 0.66
        
        # Calculate position size
        direction = settings.get('trade_direction', 'Long')
        stop_atr = strat['execution']['stop_atr']
        
        if direction == 'Long':
            stop_price = entry_price - (atr * stop_atr)
            dist = entry_price - stop_price
            action = "BUY"
        else:
            stop_price = entry_price + (atr * stop_atr)
            dist = stop_price - entry_price
            action = "SELL SHORT"
        
        shares = int(base_risk / dist) if dist > 0 else 0
        
        if shares == 0:
            continue
        
        # --- Calculate exit ---
        start_idx = df.index.searchsorted(entry_date)
        if start_idx >= len(df) - 1:
            continue
            
        window = df.iloc[start_idx + 1: start_idx + 1 + hold_days].copy()
        if window.empty:
            continue
            
        exit_row = window.iloc[-1]
        exit_price = exit_row['Close']
        exit_date = exit_row.name
        
        if action == "BUY":
            pnl = (exit_price - entry_price) * shares
        else:
            pnl = (entry_price - exit_price) * shares
        
        # Calculate time stop date
        try:
            e_idx = df.index.get_loc(entry_date)
            ts_idx = e_idx + hold_days
            time_stop_date = df.index[ts_idx] if ts_idx < len(df) else entry_date + BusinessDay(hold_days)
        except:
            time_stop_date = pd.NaT
        
        # --- Update state ---
        realized_pnl += pnl
        current_equity = starting_equity + realized_pnl
        position_last_exit[pos_key] = exit_date
        
        # Record trade
        executed_trades.append({
            "Date": signal_date.date(),
            "Entry Date": entry_date.date() if hasattr(entry_date, 'date') else entry_date,
            "Exit Date": exit_date.date() if hasattr(exit_date, 'date') else exit_date,
            "Time Stop": time_stop_date,
            "Strategy": strat['name'],
            "Ticker": ticker,
            "Action": action,
            "Entry Criteria": entry_type,
            "Price": entry_price,
            "Shares": shares,
            "PnL": pnl,
            "ATR": atr,
            "Range %": cand['range_pct'],
            "Equity at Signal": current_equity - pnl,  # Equity BEFORE this trade's PnL
            "Risk $": base_risk,
            "Risk bps": risk_bps
        })
    
    if not executed_trades:
        return pd.DataFrame()
    
    sig_df = pd.DataFrame(executed_trades)
    sig_df['Date'] = pd.to_datetime(sig_df['Date'])
    sig_df['Entry Date'] = pd.to_datetime(sig_df['Entry Date'])
    sig_df['Exit Date'] = pd.to_datetime(sig_df['Exit Date'])
    sig_df['Time Stop'] = pd.to_datetime(sig_df['Time Stop'])
    
    return sig_df.sort_values(by="Exit Date")


# -----------------------------------------------------------------------------
# CORE LOGIC: DAILY MARK-TO-MARKET PNL
# -----------------------------------------------------------------------------
def get_daily_mtm_series(sig_df, master_dict):
    if sig_df.empty: return pd.Series(dtype=float)
    
    min_date = sig_df['Date'].min()
    max_date = max(sig_df['Exit Date'].max(), pd.Timestamp.today())
    all_dates = pd.date_range(start=min_date, end=max_date, freq='B') 
    
    daily_pnl = pd.Series(0.0, index=all_dates)
    
    for idx, trade in sig_df.iterrows():
        ticker = trade['Ticker'].replace('.', '-')
        action = trade['Action']
        shares = trade['Shares']
        entry_date = trade['Date']
        exit_date = trade['Exit Date']
        entry_price = trade['Price']
        
        t_df = master_dict.get(ticker)
        if t_df is None or t_df.empty:
            if exit_date in daily_pnl.index: daily_pnl[exit_date] += trade['PnL']
            continue
            
        trade_dates = all_dates[(all_dates >= entry_date) & (all_dates <= exit_date)]
        closes = t_df['Close'].reindex(trade_dates).ffill()
        if closes.empty: continue
        
        current_pnl = pd.Series(0.0, index=trade_dates)
        first_date = trade_dates[0]
        if first_date in closes.index:
            if action == "BUY": current_pnl[first_date] = (closes[first_date] - entry_price) * shares
            else: current_pnl[first_date] = (entry_price - closes[first_date]) * shares
                
        if len(trade_dates) > 1:
            diffs = closes.diff().dropna()
            if action == "SELL SHORT": diffs = -diffs
            subsequent_pnl = diffs * shares
            for d, val in subsequent_pnl.items():
                if d in current_pnl.index: current_pnl[d] = subsequent_pnl[d]
        
        for d, val in current_pnl.items():
            if d in daily_pnl.index: daily_pnl[d] += val
                
    return daily_pnl

def calculate_mark_to_market_curve(sig_df, master_dict):
    daily_pnl = get_daily_mtm_series(sig_df, master_dict)
    if daily_pnl.empty: return pd.DataFrame(columns=['Equity'])
    return daily_pnl.cumsum().to_frame(name='Equity')

def calculate_daily_exposure(sig_df):
    if sig_df.empty: return pd.DataFrame()
    min_date = sig_df['Date'].min()
    max_date = sig_df['Exit Date'].max()
    all_dates = pd.date_range(start=min_date, end=max_date)
    exposure_df = pd.DataFrame(0.0, index=all_dates, columns=['Long Exposure ($)', 'Short Exposure ($)'])
    for idx, row in sig_df.iterrows():
        trade_dates = pd.date_range(start=row['Date'], end=row['Exit Date'])
        dollar_val = row['Price'] * row['Shares']
        if row['Action'] == 'BUY':
            exposure_df.loc[exposure_df.index.isin(trade_dates), 'Long Exposure ($)'] += dollar_val
        elif row['Action'] == 'SELL SHORT':
            exposure_df.loc[exposure_df.index.isin(trade_dates), 'Short Exposure ($)'] += dollar_val
    exposure_df['Net Exposure ($)'] = exposure_df['Long Exposure ($)'] - exposure_df['Short Exposure ($)']
    return exposure_df


# -----------------------------------------------------------------------------
# ANNUAL STATS & METRICS
# -----------------------------------------------------------------------------
def calculate_annual_stats(daily_pnl_series, starting_equity):
    if daily_pnl_series.empty: return pd.DataFrame()
    
    equity_series = starting_equity + daily_pnl_series.cumsum()
    daily_rets = equity_series.pct_change().fillna(0)
    
    yearly_stats = []
    years = daily_rets.resample('YE')
    
    for year_date, rets in years:
        year = year_date.year
        if rets.empty: continue
        
        year_start_eq = equity_series.loc[:year_date].iloc[-len(rets)-1] if len(equity_series.loc[:year_date]) > len(rets) else starting_equity
        year_end_eq = equity_series.loc[:year_date].iloc[-1]
        
        total_ret_pct = (year_end_eq - year_start_eq) / year_start_eq
        total_ret_dollar = year_end_eq - year_start_eq
        
        std_dev = rets.std() * np.sqrt(252)
        mean_ret = rets.mean() * 252
        sharpe = mean_ret / std_dev if std_dev != 0 else 0
        
        neg_rets = rets[rets < 0]
        downside_std = np.sqrt((neg_rets**2).mean()) * np.sqrt(252)
        sortino = mean_ret / downside_std if downside_std != 0 else 0
        
        running_max = equity_series.loc[:year_date].expanding().max()
        drawdown = (equity_series.loc[:year_date] - running_max) / running_max
        max_dd = drawdown.loc[str(year)].min()
        
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
    
    def get_metrics(df, name, master_dict, equity_base, calculate_ratios=True):
        if df.empty: return None
        
        count = len(df)
        total_pnl = df['PnL'].sum()
        gross_profit = df[df['PnL'] > 0]['PnL'].sum()
        gross_loss = abs(df[df['PnL'] < 0]['PnL'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan
        avg_pnl = df['PnL'].mean()
        std_pnl = df['PnL'].std()
        sqn = (avg_pnl / std_pnl * np.sqrt(count)) if std_pnl != 0 else 0
        
        sharpe = np.nan
        sortino = np.nan

        if calculate_ratios:
            daily_mtm_pnl = get_daily_mtm_series(df, master_dict)
            if not daily_mtm_pnl.empty:
                curr_equity = equity_base + daily_mtm_pnl.cumsum()
                daily_rets = curr_equity.pct_change().fillna(0)
                
                mean_ret = daily_rets.mean() * 252
                std_dev = daily_rets.std() * np.sqrt(252)
                sharpe = mean_ret / std_dev if std_dev != 0 else 0
                
                neg_rets = daily_rets[daily_rets < 0]
                downside_std = np.sqrt((neg_rets**2).mean()) * np.sqrt(252)
                sortino = mean_ret / downside_std if downside_std != 0 else 0

        return {
            "Strategy": name, "Trades": count, "Total PnL": total_pnl,
            "Sharpe": sharpe, "Sortino": sortino,
            "Profit Factor": profit_factor, "SQN": sqn
        }
        
    strategies = sig_df['Strategy'].unique()
    for strat in strategies:
        strat_df = sig_df[sig_df['Strategy'] == strat]
        m = get_metrics(strat_df, strat, master_dict, starting_equity, calculate_ratios=False)
        if m: stats.append(m)
        
    total_m = get_metrics(sig_df, "TOTAL PORTFOLIO", master_dict, starting_equity, calculate_ratios=True)
    if total_m: stats.append(total_m)
    
    return pd.DataFrame(stats)


# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Strategy Backtest Lab v2")
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
        st.caption(f"Data download buffer: 365 days prior to {user_start_date}.")
        st.markdown("---")
        st.markdown("**🔄 Dynamic Position Sizing**")
        st.caption("Position sizes now scale with equity. Risk is calculated as basis points of current account value at time of each signal.")
        run_btn = st.form_submit_button("⚡ Run Backtest")

    st.title("⚡ Strategy Backtest Lab v2")
    st.markdown("### Dynamic Equity-Based Position Sizing")
    st.markdown(f"**Selected Start Date:** {user_start_date} | **Starting Equity:** ${starting_equity:,.0f}")
    st.info("💡 **New in v2:** Position sizes now scale dynamically with your equity. Each trade is sized based on your account value at the time of the signal, not a fixed dollar amount.")
    st.markdown("---")

    if run_btn:
        sznl_map = load_seasonal_map()
        if 'backtest_data' not in st.session_state:
            st.session_state['backtest_data'] = {}

        # Build strategy book (without fixed dollar amounts - we'll calculate dynamically)
        strategies = []
        for strat in _STRATEGY_BOOK_RAW:
            import copy
            s = copy.deepcopy(strat)
            # Don't pre-calculate risk_per_trade - we'll do it dynamically
            strategies.append(s)

        long_term_tickers = set()
        for strat in strategies:
            for t in strat['universe_tickers']:
                long_term_tickers.add(t)
            s = strat['settings']
            if s.get('use_market_sznl'): long_term_tickers.add(s.get('market_ticker', '^GSPC'))
            if "Market" in s.get('trend_filter', ''): long_term_tickers.add(s.get('market_ticker', 'SPY'))
            if "SPY" in s.get('trend_filter', ''): long_term_tickers.add("SPY")
            if s.get('use_vix_filter'): long_term_tickers.add('^VIX')

        long_term_list = [t.replace('.', '-') for t in long_term_tickers]
        existing_keys = set(st.session_state['backtest_data'].keys())
        missing_long = list(set(long_term_list) - existing_keys)
        
        if missing_long:
            st.write(f"📥 Downloading **Deep History (from 2000)** for {len(missing_long)} tickers...")
            data_long = download_historical_data(missing_long, start_date="2000-01-01")
            st.session_state['backtest_data'].update(data_long)
            st.success("✅ Download Batch Complete.")

        master_dict = st.session_state['backtest_data']
        downloaded_keys = set(master_dict.keys())
        requested_set = set(long_term_list)
        failed_tickers = requested_set - downloaded_keys
        
        if failed_tickers:
            st.error(f"⚠️ {len(failed_tickers)} Tickers failed to download.")
            with st.expander("View Failed Tickers"):
                st.write(list(failed_tickers))

        # Prepare VIX series
        vix_df = master_dict.get('^VIX')
        vix_series = vix_df['Close'] if vix_df is not None and not vix_df.empty else None

        # --- PHASE 1: Generate all candidate signals ---
        st.write("📊 **Phase 1:** Generating candidate signals across all strategies...")
        candidates = generate_candidate_signals(master_dict, strategies, sznl_map, user_start_date, vix_series)
        st.write(f"   Found **{len(candidates):,}** candidate signals")

        # --- PHASE 2: Process chronologically with dynamic sizing ---
        st.write("📈 **Phase 2:** Processing signals chronologically with dynamic position sizing...")
        sig_df = process_signals_chronologically(candidates, master_dict, starting_equity)
        
        if not sig_df.empty:
            st.success(f"✅ Executed **{len(sig_df):,}** trades")
            
            today = pd.Timestamp(datetime.date.today())
            open_mask = sig_df['Time Stop'] >= today
            open_df = sig_df[open_mask].copy()

            if not open_df.empty:
                current_prices, open_pnls, current_values = [], [], []
                for idx, row in open_df.iterrows():
                    ticker = row['Ticker']
                    t_df = master_dict.get(ticker.replace('.', '-'))
                    last_close = t_df.iloc[-1]['Close'] if t_df is not None and not t_df.empty else row['Price']
                    if row['Action'] == 'BUY':
                        pnl = (last_close - row['Price']) * row['Shares']
                        val = last_close * row['Shares']
                    else:
                        pnl = (row['Price'] - last_close) * row['Shares']
                        val = last_close * row['Shares']
                    current_prices.append(last_close)
                    open_pnls.append(pnl)
                    current_values.append(val)

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
                st.info("No active positions (Time Stop >= Today).")

            st.divider()
            
            # --- Position Sizing Analysis ---
            st.subheader("📐 Dynamic Position Sizing Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_risk = sig_df['Risk $'].mean()
                st.metric("Avg Risk per Trade", f"${avg_risk:,.0f}")
            with col2:
                min_risk = sig_df['Risk $'].min()
                max_risk = sig_df['Risk $'].max()
                st.metric("Risk Range", f"${min_risk:,.0f} - ${max_risk:,.0f}")
            with col3:
                final_equity = starting_equity + sig_df['PnL'].sum()
                equity_growth = (final_equity / starting_equity - 1) * 100
                st.metric("Final Equity", f"${final_equity:,.0f}", delta=f"{equity_growth:.1f}%")
            
            # Show equity at signal over time
            fig_sizing = go.Figure()
            fig_sizing.add_trace(go.Scatter(
                x=sig_df['Date'], 
                y=sig_df['Equity at Signal'],
                mode='markers+lines',
                name='Equity at Signal',
                line=dict(color='#00FF00', width=1),
                marker=dict(size=4)
            ))
            fig_sizing.add_trace(go.Scatter(
                x=sig_df['Date'], 
                y=sig_df['Risk $'],
                mode='markers',
                name='Risk $ per Trade',
                yaxis='y2',
                marker=dict(size=4, color='orange')
            ))
            fig_sizing.update_layout(
                title="Equity & Position Risk Over Time",
                yaxis=dict(title="Equity ($)", side='left'),
                yaxis2=dict(title="Risk per Trade ($)", side='right', overlaying='y'),
                height=350,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_sizing, use_container_width=True)
            
            # --- Annual Breakdown ---
            st.subheader("📅 Annual Performance Breakdown")
            port_daily_pnl = get_daily_mtm_series(sig_df, master_dict)
            annual_df = calculate_annual_stats(port_daily_pnl, starting_equity=starting_equity)
            if not annual_df.empty:
                st.dataframe(annual_df.style.format({
                    "Total Return ($)": "${:,.2f}", "Total Return (%)": "{:.2%}", "Max Drawdown": "{:.2%}",
                    "Sharpe Ratio": "{:.2f}", "Sortino Ratio": "{:.2f}", "Std Dev": "{:.2%}"
                }), use_container_width=True)

            st.subheader("📊 Strategy Performance Metrics")
            stats_df = calculate_performance_stats(sig_df, master_dict, starting_equity)
            st.dataframe(stats_df.style.format({
                "Total PnL": "${:,.2f}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Profit Factor": "{:.2f}", "SQN": "{:.2f}"
            }), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Total Portfolio PnL (Mark-to-Market)")
                df_eq = calculate_mark_to_market_curve(sig_df, master_dict)
                if not df_eq.empty:
                    df_eq['SMA20'] = df_eq['Equity'].rolling(window=20).mean()
                    df_eq['StdDev'] = df_eq['Equity'].rolling(window=20).std()
                    df_eq['Upper'] = df_eq['SMA20'] + (2 * df_eq['StdDev'])
                    df_eq['Lower'] = df_eq['SMA20'] - (2 * df_eq['StdDev'])
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq['Upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                    fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq['Lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 255, 255, 0.1)', name='Bollinger Band (20, 2)', hoverinfo='skip'))
                    fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq['SMA20'], mode='lines', name='SMA 20', line=dict(color='orange', width=1, dash='dot')))
                    fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq['Equity'], mode='lines', name='Total PnL', line=dict(color='#00FF00', width=2)))
                    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified", xaxis_title="Date", yaxis_title="Cumulative PnL ($)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No trades to plot.")
            with col2:
                st.subheader("📉 Cumulative Realized PnL by Strategy")
                strat_pnl = sig_df.pivot_table(index='Exit Date', columns='Strategy', values='PnL', aggfunc='sum').fillna(0)
                st.line_chart(strat_pnl.cumsum())

            st.subheader("⚖️ Portfolio Exposure Over Time")
            exposure_df = calculate_daily_exposure(sig_df)
            if not exposure_df.empty:
                st.line_chart(exposure_df)

            st.subheader("📜 Historical Signal Log")
            display_cols = ["Date", "Entry Date", "Exit Date", "Time Stop", "Strategy", "Ticker", 
                          "Action", "Entry Criteria", "Price", "Shares", "PnL", "ATR", "Range %",
                          "Equity at Signal", "Risk $", "Risk bps"]
            st.dataframe(sig_df[display_cols].sort_values(by="Date", ascending=False).style.format({
                "Price": "${:.2f}", "PnL": "${:.2f}", "Date": "{:%Y-%m-%d}", "Entry Date": "{:%Y-%m-%d}", 
                "Exit Date": "{:%Y-%m-%d}", "Time Stop": "{:%Y-%m-%d}", "Range %": "{:.1f}%",
                "Equity at Signal": "${:,.0f}", "Risk $": "${:,.0f}"
            }), use_container_width=True, height=400)
        else:
            st.warning(f"No signals found in the backtest period starting from {user_start_date}.")

    else:
        st.info("👈 Please select a start year/date and click 'Run Backtest' in the sidebar to begin.")

if __name__ == "__main__":
    main()
