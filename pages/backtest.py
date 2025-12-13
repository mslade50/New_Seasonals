import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import datetime
import random
import time
import uuid

# -----------------------------------------------------------------------------
# CONFIG / CONSTANTS
# -----------------------------------------------------------------------------
MARKET_TICKER = "^GSPC" 

SECTOR_ETFS = [
    "IBB", "IHI", "ITA", "ITB", "IYR", "KRE", "OIH", "SMH", "VNQ",
    "XBI", "XHB", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP",
    "XLU", "XLV", "XLY", "XME", "XOP", "XRT", "GLD", "CEF", "SLV",
]

INDEX_ETFS = ["SPY", "QQQ", "IWM", "DIA", "SMH"]

INTERNATIONAL_ETFS = [
    "EWZ", "EWC", "ECH", "ECOL", "EWW", "ARGT",
    "EWQ", "EWG", "EWI", "EWU", "EWP", "EWK", "EWO", "EWN", "EWD", "EWL",
    "EWJ", "EWH", "MCHI", "INDA", "EWY", "EWT", "EWA", "EWS", "EWM", "THD", "EIDO", "VNM", "EPHE",
    "EZA", "TUR", "EGPT"
]

CSV_PATH = "seasonal_ranks.csv" 

# -----------------------------------------------------------------------------
# DATA UTILS
# -----------------------------------------------------------------------------

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
        output_map[str(ticker).upper()] = pd.Series(
            group.seasonal_rank.values, index=group.Date
        ).to_dict()
    return output_map

def get_sznl_val_series(ticker, dates, sznl_map):
    ticker = ticker.upper()
    t_map = sznl_map.get(ticker, {})
    
    if not t_map and ticker == "^GSPC":
        t_map = sznl_map.get("SPY", {})

    if not t_map:
        return pd.Series(50.0, index=dates)
        
    return dates.map(t_map).fillna(50.0)

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
                t_df = df
                t_df = clean_ticker_df(t_df)
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
            err_msg = str(e).lower()
            if "rate limited" in err_msg or "too many requests" in err_msg:
                st.warning(f"⚠️ Rate limit hit. Stopping download.")
                break
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

# -----------------------------------------------------------------------------
# ANALYSIS ENGINE
# -----------------------------------------------------------------------------

def calculate_indicators(df, sznl_map, ticker, market_series=None, gap_window=21):
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    
    # SMAs
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA100'] = df['Close'].rolling(100).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    # EMAs
    df['EMA8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA11'] = df['Close'].ewm(span=11, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # Perf Ranks
    for window in [5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window)
        df[f'rank_ret_{window}d'] = df[f'ret_{window}d'].expanding(min_periods=252).rank(pct=True) * 100.0
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    # ATR % Calculation
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
    
    # Seasonality
    df['Sznl'] = get_sznl_val_series(ticker, df.index, sznl_map)
    
    # 52w High/Low
    rolling_high = df['High'].shift(1).rolling(252).max()
    rolling_low = df['Low'].shift(1).rolling(252).min()
    df['is_52w_high'] = df['High'] > rolling_high
    df['is_52w_low'] = df['Low'] < rolling_low
    
    # Volume & Vol Ratio
    vol_ma = df['Volume'].rolling(63).mean()
    df['vol_ma'] = vol_ma
    df['vol_ratio'] = df['Volume'] / vol_ma
    
    # 10d Relative Volume Percentile
    vol_ma_10 = df['Volume'].rolling(10).mean()
    df['vol_ratio_10d'] = vol_ma_10 / vol_ma
    df['vol_ratio_10d_rank'] = df['vol_ratio_10d'].expanding(min_periods=252).rank(pct=True) * 100.0
    
    # --- ACCUMULATION / DISTRIBUTION FLAGS ---
    vol_gt_prev = df['Volume'] > df['Volume'].shift(1)
    vol_gt_ma = df['Volume'] > df['vol_ma']
    is_green = df['Close'] > df['Open']
    is_red = df['Close'] < df['Open']

    df['is_acc_day'] = (is_green & vol_gt_prev & vol_gt_ma).astype(int)
    df['is_dist_day'] = (is_red & vol_gt_prev & vol_gt_ma).astype(int)
    
    # Age
    if not df.empty:
        start_ts = df.index[0]
        df['age_years'] = (df.index - start_ts).days / 365.25
    else:
        df['age_years'] = 0.0
        
    # Market Regime Mapping
    if market_series is not None:
        df['Market_Above_SMA200'] = market_series.reindex(df.index, method='ffill').fillna(False)

    # Candle Range % Location
    denom = (df['High'] - df['Low'])
    df['RangePct'] = np.where(denom == 0, 0.5, (df['Close'] - df['Low']) / denom)

    # Day of Week
    df['DayOfWeekVal'] = df.index.dayofweek
    
    # Open Gap Count
    is_open_gap = (df['Low'] > df['High'].shift(1)).astype(int)
    df['GapCount'] = is_open_gap.rolling(gap_window).sum()

    # --- NEW: PIVOT POINT CALCULATION (Forward Filled) ---
    piv_len = 20 # 5 bars left, 5 bars right
    # Identify Pivot Highs/Lows using centered window
    roll_max = df['High'].rolling(window=piv_len*2+1, center=True).max()
    df['is_pivot_high'] = (df['High'] == roll_max)
    
    roll_min = df['Low'].rolling(window=piv_len*2+1, center=True).min()
    df['is_pivot_low'] = (df['Low'] == roll_min)

    # Forward Fill: shift by piv_len because we confirm it 'piv_len' days later
    df['LastPivotHigh'] = np.where(df['is_pivot_high'], df['High'], np.nan)
    df['LastPivotHigh'] = df['LastPivotHigh'].shift(piv_len).ffill()
    
    df['LastPivotLow'] = np.where(df['is_pivot_low'], df['Low'], np.nan)
    df['LastPivotLow'] = df['LastPivotLow'].shift(piv_len).ffill()

    return df

def run_engine(universe_dict, params, sznl_map, market_series=None):
    all_potential_trades = []
    
    total = len(universe_dict)
    bt_start_ts = pd.to_datetime(params['backtest_start_date'])
    direction = params.get('trade_direction', 'Long')
    max_one_pos_per_ticker = params.get('max_one_pos', True)
    allow_same_day_reentry = params.get('allow_same_day_reentry', False)
    slippage_bps = params.get('slippage_bps', 5)
    entry_conf_bps = params.get('entry_conf_bps', 0)
    max_atr_pct = params.get('max_atr_pct', 1000.0)
    
    # Entry Logic flags
    entry_mode = params['entry_type']
    is_pullback = "Pullback" in entry_mode
    is_limit_atr = "Limit (Close -0.5 ATR)" in entry_mode
    is_limit_prev = "Limit (Prev Close)" in entry_mode
    is_limit_open_atr = "Limit (Open +/- 0.5 ATR)" in entry_mode 
    is_limit_pivot = "Limit (Untested Pivot)" in entry_mode
    is_limit_entry = is_limit_atr or is_limit_prev or is_limit_open_atr or is_limit_pivot
    is_gap_up = "Gap Up Only" in entry_mode

    use_ma_filter = params.get('use_ma_entry_filter', False)
    pullback_col = None
    if "10 SMA" in entry_mode: pullback_col = "SMA10"
    elif "21 EMA" in entry_mode: pullback_col = "EMA21"
    pullback_use_level = "Level" in entry_mode
    
    gap_window = params.get('gap_lookback', 21)

    total_signals_generated = 0

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (ticker, df_raw) in enumerate(universe_dict.items()):
        status_text.text(f"Scanning signals for {ticker}...")
        progress_bar.progress((i+1)/total)
        
        if ticker == MARKET_TICKER and MARKET_TICKER not in params.get('universe_tickers', []):
            continue

        if len(df_raw) < 100: continue
        
        ticker_last_exit = pd.Timestamp.min
        
        try:
            df = calculate_indicators(df_raw, sznl_map, ticker, market_series, gap_window=gap_window)
            
            df = df[df.index >= bt_start_ts]
            if df.empty: continue
            
            conditions = []
            
            # --- TREND FILTER ---
            trend_opt = params.get('trend_filter', 'None')
            if trend_opt == "Price > 200 SMA":
                conditions.append(df['Close'] > df['SMA200'])
            elif trend_opt == "Price > Rising 200 SMA":
                conditions.append((df['Close'] > df['SMA200']) & (df['SMA200'] > df['SMA200'].shift(1)))
            elif trend_opt == "Price < 200 SMA":
                conditions.append(df['Close'] < df['SMA200'])
            elif trend_opt == "Price < Falling 200 SMA":
                conditions.append((df['Close'] < df['SMA200']) & (df['SMA200'] < df['SMA200'].shift(1)))
            elif "Market" in trend_opt and 'Market_Above_SMA200' in df.columns:
                if trend_opt == "Market > 200 SMA":
                    conditions.append(df['Market_Above_SMA200'])
                elif trend_opt == "Market < 200 SMA":
                    conditions.append(~df['Market_Above_SMA200']) 
            
            # --- LIQUIDITY GATES ---
            curr_age = df['age_years'].fillna(0)
            curr_vol = df['vol_ma'].fillna(0)
            curr_close = df['Close'].fillna(0)
            curr_atr_pct = df['ATR_Pct'].fillna(0) 
            
            gate = (curr_close >= params['min_price']) & \
                   (curr_vol >= params['min_vol']) & \
                   (curr_age >= params['min_age']) & \
                   (curr_age <= params['max_age']) & \
                   (curr_atr_pct >= params['min_atr_pct'])& \
                   (curr_atr_pct <= max_atr_pct)
            conditions.append(gate)

            # --- PRICE ACTION ---
            if params.get('require_close_gt_open', False):
                conditions.append(df['Close'] > df['Open'])
            
            # --- NEW: BREAKOUT FILTER ---
            bk_mode = params.get('breakout_mode', 'None')
            if bk_mode == "Close > Prev Day High":
                conditions.append(df['Close'] > df['High'].shift(1))
            elif bk_mode == "Close < Prev Day Low":
                conditions.append(df['Close'] < df['Low'].shift(1))
            
            if params.get('use_range_filter', False):
                r_min = params.get('range_min', 0.0)
                r_max = params.get('range_max', 100.0)
                conditions.append((df['RangePct'] * 100 >= r_min) & (df['RangePct'] * 100 <= r_max))
            
            if params.get('use_dow_filter', False):
                allowed_days = params.get('allowed_days', [])
                if allowed_days:
                    conditions.append(df['DayOfWeekVal'].isin(allowed_days))
            
            # --- GAP FILTER ---
            if params.get('use_gap_filter', False):
                g_logic = params['gap_logic']
                g_thresh = params['gap_thresh']
                if g_logic == ">": conditions.append(df['GapCount'] > g_thresh)
                elif g_logic == "<": conditions.append(df['GapCount'] < g_thresh)
                elif g_logic == "=": conditions.append(df['GapCount'] == g_thresh)

            # --- ACCUMULATION FILTER ---
            if params.get('use_acc_count_filter', False):
                win = params['acc_count_window']
                logic = params['acc_count_logic']
                thresh = params['acc_count_thresh']
                
                rolling_acc = df['is_acc_day'].rolling(win).sum()
                
                if logic == ">": conditions.append(rolling_acc > thresh)
                elif logic == "<": conditions.append(rolling_acc < thresh)
                elif logic == "=": conditions.append(rolling_acc == thresh)

            # --- DISTRIBUTION FILTER ---
            if params.get('use_dist_count_filter', False):
                win = params['dist_count_window']
                logic = params['dist_count_logic']
                thresh = params['dist_count_thresh']
                
                rolling_dist = df['is_dist_day'].rolling(win).sum()
                
                if logic == ">": conditions.append(rolling_dist > thresh)
                elif logic == "<": conditions.append(rolling_dist < thresh)
                elif logic == "=": conditions.append(rolling_dist == thresh)

            # --- MULTI-PERFORMANCE FILTER ---
            perf_filters = params.get('perf_filters', [])
            if perf_filters:
                combined_perf_raw = pd.Series(True, index=df.index)
                for pf in perf_filters:
                    col = f"rank_ret_{pf['window']}d"
                    consec_days = pf['consecutive'] 
                    if pf['logic'] == '<': cond_f = df[col] < pf['thresh']
                    else: cond_f = df[col] > pf['thresh']
                    if consec_days > 1: cond_f = cond_f.rolling(consec_days).sum() == consec_days
                    combined_perf_raw = combined_perf_raw & cond_f
                
                final_perf_cond = combined_perf_raw
                if params.get('perf_first_instance', False):
                    lookback = params.get('perf_lookback', 21)
                    prev_instances = final_perf_cond.shift(1).rolling(lookback).sum()
                    final_perf_cond = final_perf_cond & (prev_instances == 0)
                conditions.append(final_perf_cond)

            # --- DISTANCE FROM MA FILTER ---
            if params.get('use_ma_dist_filter', False):
                ma_choice = params['dist_ma_type']
                ma_col_map = {
                    "SMA 10": "SMA10", "SMA 20": "SMA20", "SMA 50": "SMA50", 
                    "SMA 100": "SMA100", "SMA 200": "SMA200", 
                    "EMA 8": "EMA8", "EMA 11": "EMA11", "EMA 21": "EMA21"
                }
                ma_col = ma_col_map.get(ma_choice, "SMA200")
                
                dist_atr_units = (df['Close'] - df[ma_col]) / df['ATR']
                
                logic = params['dist_logic']
                d_min = params['dist_min']
                d_max = params['dist_max']
                
                if logic == "Greater Than (>)":
                    conditions.append(dist_atr_units > d_min)
                elif logic == "Less Than (<)":
                    conditions.append(dist_atr_units < d_max)
                elif logic == "Between":
                    conditions.append((dist_atr_units >= d_min) & (dist_atr_units <= d_max))

            # --- SEASONALITY ---
            if params['use_sznl']:
                if params['sznl_logic'] == '<': cond = df['Sznl'] < params['sznl_thresh']
                else: cond = df['Sznl'] > params['sznl_thresh']
                if params['sznl_first_instance']:
                    prev = cond.shift(1).rolling(params['sznl_lookback']).sum()
                    cond = cond & (prev == 0)
                conditions.append(cond)
            
            if params.get('use_market_sznl', False):
                market_ranks = get_sznl_val_series(MARKET_TICKER, df.index, sznl_map)
                if params['market_sznl_logic'] == '<': cond = market_ranks < params['market_sznl_thresh']
                else: cond = market_ranks > params['market_sznl_thresh']
                conditions.append(cond)
            
            # --- 52W HIGHS/LOWS with LAG ---
            if params['use_52w']:
                if params['52w_type'] == 'New 52w High': cond = df['is_52w_high']
                else: cond = df['is_52w_low']
                
                # --- LAG LOGIC ---
                lag = params.get('52w_lag', 0)
                if lag > 0:
                    cond = cond.shift(lag).fillna(False)
                # -----------------

                if params['52w_first_instance']:
                    prev = cond.shift(1).rolling(params['52w_lookback']).sum()
                    cond = cond & (prev == 0)
                conditions.append(cond)
            
            # --- VOL FILTERS ---
            if params['use_vol']:
                cond = df['vol_ratio'] > params['vol_thresh']
                conditions.append(cond)

            if params['use_vol_rank']:
                if params['vol_rank_logic'] == '<': cond = df['vol_ratio_10d_rank'] < params['vol_rank_thresh']
                else: cond = df['vol_ratio_10d_rank'] > params['vol_rank_thresh']
                conditions.append(cond)

            if not conditions: continue
            
            final_signal = conditions[0]
            for c in conditions[1:]: final_signal = final_signal & c
            
            signal_dates = df.index[final_signal]
            total_signals_generated += len(signal_dates)
            
            # --- TRADE EXECUTION SIMULATION ---
            for signal_date in signal_dates:
                
                if max_one_pos_per_ticker:
                    if allow_same_day_reentry:
                        if signal_date < ticker_last_exit: continue
                    else:
                        if signal_date <= ticker_last_exit: continue
                
                sig_idx = df.index.get_loc(signal_date)
                fixed_exit_idx = sig_idx + params['holding_days']
                if fixed_exit_idx >= len(df): continue

                found_entry = False
                actual_entry_idx = -1
                actual_entry_price = 0.0
                entry_failure_reason = "" # Capture why entry failed
                
                # PATH A: PULLBACK
                if is_pullback and pullback_col:
                    for wait_i in range(1, params['holding_days'] + 1):
                        curr_check_idx = sig_idx + wait_i
                        if curr_check_idx >= len(df): break
                        
                        row = df.iloc[curr_check_idx]
                        ma_val = row[pullback_col]
                        if np.isnan(ma_val): continue

                        if row['Low'] <= ma_val:
                            if use_ma_filter:
                                cutoff = ma_val - (0.25 * row['ATR'])
                                if row['Close'] < cutoff:
                                    entry_failure_reason = "MA Filter (Close < MA-0.25ATR)"
                                    continue 
                            
                            found_entry = True
                            actual_entry_idx = curr_check_idx
                            if pullback_use_level: actual_entry_price = ma_val
                            else: actual_entry_price = row['Close']
                            break
                        else:
                            entry_failure_reason = "Price did not touch MA"

                # PATH B: LIMIT ENTRIES (INCLUDING PIVOT)
                elif is_limit_entry:
                    sig_row = df.iloc[sig_idx]
                    sig_close = sig_row['Close']
                    sig_atr = sig_row['ATR']
                    
                    limit_price = 0.0
                    
                    # 1. Untested Pivot Logic
                    if is_limit_pivot:
                        if direction == 'Long':
                            limit_price = sig_row.get('LastPivotLow', 0.0)
                            # Logic: If signal close is ALREADY below pivot, it's broken or we missed it.
                            if sig_close < limit_price: limit_price = 0.0
                        else:
                            limit_price = sig_row.get('LastPivotHigh', 0.0)
                            # Logic: If signal close is ALREADY above pivot, broken.
                            if sig_close > limit_price: limit_price = 0.0
                    
                    # 2. Standard Limit Logic
                    elif not np.isnan(sig_atr):
                        if is_limit_atr:
                            if direction == 'Long': limit_price = sig_close - (0.5 * sig_atr)
                            else: limit_price = sig_close + (0.5 * sig_atr)
                        elif is_limit_prev: 
                            limit_price = sig_close
                    
                    # Loop for X days to see if limit is hit
                    for wait_i in range(1, 4):
                        curr_check_idx = sig_idx + wait_i
                        if curr_check_idx >= len(df): break
                        
                        row = df.iloc[curr_check_idx]
                        
                        # Dynamic Limit (ATR relative to open)
                        if is_limit_open_atr and not np.isnan(sig_atr):
                            if direction == 'Long':
                                limit_price = row['Open'] - (0.5 * sig_atr)
                            else:
                                limit_price = row['Open'] + (0.5 * sig_atr)

                        if limit_price == 0.0 or np.isnan(limit_price): break 

                        is_filled = False
                        if direction == 'Long':
                            # Buy Limit: Did Price drop to Limit?
                            if row['Low'] <= limit_price:
                                is_filled = True
                                actual_entry_price = limit_price 
                                if row['Open'] < limit_price: actual_entry_price = row['Open'] # Gap fill
                        else:
                            # Sell Limit: Did Price rise to Limit?
                            if row['High'] >= limit_price:
                                is_filled = True
                                actual_entry_price = limit_price
                                if row['Open'] > limit_price: actual_entry_price = row['Open'] # Gap fill

                        if is_filled:
                            found_entry = True
                            actual_entry_idx = curr_check_idx
                            break
                        else:
                            entry_failure_reason = "Limit Price Not Reached"

                # PATH C: GAP UP ONLY
                elif is_gap_up:
                    check_idx = sig_idx + 1
                    if check_idx < len(df):
                        sig_high = df['High'].iloc[sig_idx]
                        next_open = df['Open'].iloc[check_idx]
                        
                        if next_open > sig_high:
                            found_entry = True
                            actual_entry_idx = check_idx
                            actual_entry_price = next_open
                        else:
                            entry_failure_reason = f"No Gap Up (Open {next_open:.2f} <= High {sig_high:.2f})"

                # PATH D: IMMEDIATE
                else:
                    found_entry = True
                    if entry_mode == 'Signal Close':
                        actual_entry_idx = sig_idx
                        actual_entry_price = df['Close'].iloc[sig_idx]
                    elif entry_mode == 'T+1 Open':
                        actual_entry_idx = sig_idx + 1
                        actual_entry_price = df['Open'].iloc[sig_idx + 1]
                    else: # T+1 Close
                        actual_entry_idx = sig_idx + 1
                        actual_entry_price = df['Close'].iloc[sig_idx + 1]

                # --- ENTRY CONFIRMATION CHECK ---
                if found_entry and entry_conf_bps > 0:
                    entry_candle = df.iloc[actual_entry_idx]
                    threshold_price = entry_candle['Open'] * (1 + entry_conf_bps/10000.0)
                    if entry_candle['High'] < threshold_price:
                        found_entry = False
                        entry_failure_reason = f"No Confirmation (High < Open+{entry_conf_bps}bps)"

                # --- IF ENTRY FAILED, LOG IT ---
                if not found_entry:
                    theo_entry_idx = min(sig_idx + 1, len(df)-1)
                    theo_entry_price = df['Open'].iloc[theo_entry_idx]
                    theo_exit_idx = min(theo_entry_idx + params['holding_days'], len(df)-1)
                    theo_exit_price = df['Close'].iloc[theo_exit_idx]
                    
                    if direction == 'Long': theo_pnl = theo_exit_price - theo_entry_price
                    else: theo_pnl = theo_entry_price - theo_exit_price
                    
                    sig_atr = df['ATR'].iloc[sig_idx] if not np.isnan(df['ATR'].iloc[sig_idx]) else 1.0
                    theo_r = theo_pnl / (sig_atr * 2)

                    all_potential_trades.append({
                        "Ticker": ticker,
                        "SignalDate": signal_date, 
                        "EntryDate": df.index[theo_entry_idx],
                        "Direction": direction,
                        "Entry": 0.0,
                        "Exit": 0.0,
                        "ExitDate": df.index[theo_exit_idx],
                        "Type": "Missed",
                        "R": theo_r, 
                        "Age": df['age_years'].iloc[sig_idx],
                        "AvgVol": df['vol_ma'].iloc[sig_idx],
                        "Status": "Entry Cond Failed",
                        "Reason": entry_failure_reason
                    })
                    continue

                # --- IF ENTRY SUCCEEDED, SIMULATE TRADE ---
                if found_entry and actual_entry_idx < len(df):
                    start_exit_scan = actual_entry_idx + 1
                    if start_exit_scan > fixed_exit_idx: continue 
                    future = df.iloc[start_exit_scan : fixed_exit_idx + 1]
                    if future.empty: continue
                    
                    atr = df['ATR'].iloc[actual_entry_idx]
                    if np.isnan(atr) or atr == 0: 
                        atr = df['ATR'].iloc[actual_entry_idx-1] if actual_entry_idx > 0 else 0
                        if atr == 0: continue

                    # Calculate Levels
                    if direction == 'Long':
                        stop_price = actual_entry_price - (atr * params['stop_atr'])
                        tgt_price = actual_entry_price + (atr * params['tgt_atr'])
                    else:
                        stop_price = actual_entry_price + (atr * params['stop_atr'])
                        tgt_price = actual_entry_price - (atr * params['tgt_atr'])

                    exit_price = actual_entry_price
                    exit_type = "Hold"
                    exit_date = None
                    
                    # --- EXIT LOGIC ---
                    use_stop = params.get('use_stop_loss', True)
                    use_target = params.get('use_take_profit', True)

                    for f_date, f_row in future.iterrows():
                        triggered = False
                        
                        # 1. CHECK STOP LOSS (If Enabled)
                        if use_stop:
                            if direction == 'Long':
                                if f_row['Low'] <= stop_price:
                                    exit_price = f_row['Open'] if f_row['Open'] < stop_price else stop_price
                                    exit_type = "Stop"
                                    exit_date = f_date
                                    triggered = True
                            else: # Short
                                if f_row['High'] >= stop_price:
                                    exit_price = f_row['Open'] if f_row['Open'] > stop_price else stop_price
                                    exit_type = "Stop"
                                    exit_date = f_date
                                    triggered = True
                        
                        if triggered: break

                        # 2. CHECK PROFIT TARGET (If Enabled)
                        if use_target:
                            if direction == 'Long':
                                if f_row['High'] >= tgt_price:
                                    exit_price = f_row['Open'] if f_row['Open'] > tgt_price else tgt_price
                                    exit_type = "Target"
                                    exit_date = f_date
                                    triggered = True
                            else: # Short
                                if f_row['Low'] <= tgt_price:
                                    exit_price = f_row['Open'] if f_row['Open'] < tgt_price else tgt_price
                                    exit_type = "Target"
                                    exit_date = f_date
                                    triggered = True
                        
                        if triggered: break
                    
                    if exit_type == "Hold":
                        exit_price = future['Close'].iloc[-1]
                        exit_date = future.index[-1]
                        exit_type = "Time"
                    
                    ticker_last_exit = exit_date

                    slip_pct = slippage_bps / 10000.0
                    if direction == 'Long':
                        adj_entry = actual_entry_price * (1 + slip_pct)
                        adj_exit = exit_price * (1 - slip_pct)
                        tech_risk = actual_entry_price - stop_price
                        pnl = adj_exit - adj_entry
                    else:
                        adj_entry = actual_entry_price * (1 - slip_pct)
                        adj_exit = exit_price * (1 + slip_pct)
                        tech_risk = stop_price - actual_entry_price
                        pnl = adj_entry - adj_exit
                        
                    if tech_risk <= 0: tech_risk = 0.001
                    r = pnl / tech_risk
                    
                    all_potential_trades.append({
                        "Ticker": ticker,
                        "SignalDate": signal_date, 
                        "EntryDate": df.index[actual_entry_idx], 
                        "Direction": direction,
                        "Entry": actual_entry_price,
                        "Exit": exit_price,
                        "ExitDate": exit_date,
                        "Type": exit_type,
                        "R": r,
                        "Age": df['age_years'].iloc[sig_idx],
                        "AvgVol": df['vol_ma'].iloc[sig_idx],
                        "Status": "Valid Signal",
                        "Reason": "Executed"
                    })
        except: continue
    
    progress_bar.empty()
    status_text.empty()

    if not all_potential_trades:
        return pd.DataFrame(), pd.DataFrame(), 0

    st.info(f"Processing Constraints on {len(all_potential_trades)} signals (Executed + Failed Entries)...")
    
    potential_df = pd.DataFrame(all_potential_trades)
    potential_df = potential_df.sort_values(by=["EntryDate", "Ticker"])
    
    final_trades_log = []
    rejected_trades_log = []
    
    active_positions = [] 
    daily_entries = {} 
    
    max_daily = params.get('max_daily_entries', 100)
    max_total = params.get('max_total_positions', 100)

    for idx, trade in potential_df.iterrows():
        # If entry logic failed, it goes straight to rejected, bypassing portfolio constraints
        if trade['Status'] == "Entry Cond Failed":
            rejected_trades_log.append(trade)
            continue

        # Check Portfolio Constraints for Valid Signals
        entry_date = trade['EntryDate']
        active_positions = [t for t in active_positions if t['ExitDate'] > entry_date]
        
        is_rejected = False
        rejection_reason = ""

        if len(active_positions) >= max_total:
            is_rejected = True
            rejection_reason = "Max Total Pos"
        
        today_count = daily_entries.get(entry_date, 0)
        if not is_rejected and today_count >= max_daily:
            is_rejected = True
            rejection_reason = "Max Daily Entries"
            
        if is_rejected:
            trade['Status'] = "Portfolio Rejected"
            trade['Reason'] = rejection_reason
            rejected_trades_log.append(trade)
        else:
            final_trades_log.append(trade)
            active_positions.append(trade)
            daily_entries[entry_date] = today_count + 1

    return pd.DataFrame(final_trades_log), pd.DataFrame(rejected_trades_log), total_signals_generated

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

def main():
    st.set_page_config(layout="wide", page_title="Quantitative Backtester")
    st.title("Quantitative Strategy Backtester")
    st.markdown("---")

    st.subheader("1. Universe & Data")
    col_u1, col_u2, col_u3 = st.columns([1, 1, 2])
    sample_pct = 100 
    use_full_history = False
    with col_u1:
        univ_choice = st.selectbox("Choose Universe", 
            ["Sector ETFs", "Indices", "International ETFs", "Sector + Index ETFs", "All CSV Tickers", "Custom (Upload CSV)"])
    with col_u2:
        default_start = datetime.date(2000, 1, 1)
        start_date = st.date_input("Backtest Start Date", value=default_start, min_value=datetime.date(1950, 1, 1), max_value=datetime.date.today())
    custom_tickers = []
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
    st.write("")
    use_full_history = st.checkbox("⚠️ Download Full History (1950+) for Accurate 'Age'", value=False)

    st.markdown("---")
    st.subheader("2. Execution & Risk")
    r_c1, r_c2, r_c3 = st.columns(3)
    with r_c1: trade_direction = st.selectbox("Trade Direction", ["Long", "Short"])
    
    with r_c2: 
        exit_mode = st.selectbox(
            "Exit Mode", 
            ["Standard (Stop & Target)", "No Stop (Target + Time)", "Time Only (Hold)"],
            help="'No Stop' keeps the profit target active but ignores the stop loss."
        )
        use_stop_loss = (exit_mode == "Standard (Stop & Target)")
        use_take_profit = (exit_mode != "Time Only (Hold)")
        time_exit_only = (exit_mode == "Time Only (Hold)")

    with r_c3: max_one_pos = st.checkbox("Max 1 Position/Ticker", value=True)
    
    p_c1, p_c2, p_c3 = st.columns(3)
    with p_c1: max_daily_entries = st.number_input("Max New Trades Per Day", 1, 100, 2)
    with p_c2: max_total_positions = st.number_input("Max Total Positions", 1, 200, 10)
    with p_c3: slippage_bps = st.number_input("Slippage (bps)", value=5)
    
    st.write("")
    c_re, c_conf = st.columns(2)
    with c_re: allow_same_day_reentry = st.checkbox("Allow Same-Day Re-entry", value=False)
    with c_conf: entry_conf_bps = st.number_input("Entry Confirmation (bps)", value=0)

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: 
        entry_type = st.selectbox("Entry Price", [
            "Signal Close", "T+1 Open", "T+1 Close",
            "Gap Up Only (Open > Prev High)", 
            "Limit (Close -0.5 ATR)", "Limit (Prev Close)", 
            "Limit (Open +/- 0.5 ATR)", 
            "Limit (Untested Pivot)", 
            "Pullback 10 SMA (Entry: Close)", "Pullback 10 SMA (Entry: Level)",
            "Pullback 21 EMA (Entry: Close)", "Pullback 21 EMA (Entry: Level)"
        ])
        if "Pullback" in entry_type: use_ma_entry_filter = st.checkbox("Filter: Close > MA - 0.25*ATR", value=False)
        else: use_ma_entry_filter = False
        
    with c2: 
        stop_atr = st.number_input("Stop Loss (ATR)", value=3.0, step=0.1, disabled=not use_stop_loss)
    with c3: 
        tgt_atr = st.number_input("Target (ATR)", value=8.0, step=0.1, disabled=not use_take_profit)
    with c4: hold_days = st.number_input("Max Holding Days", value=10, step=1)
    with c5: risk_per_trade = st.number_input("Risk Amount ($)", value=1000, step=100)

    st.markdown("---")
    st.subheader("3. Signal Criteria")
    with st.expander("Liquidity & Data History Filters", expanded=True):
        l1, l2, l3, l4, l5, l6 = st.columns(6) 
        with l1: min_price = st.number_input("Min Price ($)", value=10.0, step=1.0)
        with l2: min_vol = st.number_input("Min Avg Volume", value=100000, step=50000)
        with l3: min_age = st.number_input("Min True Age (Yrs)", value=0.25, step=0.25)
        with l4: max_age = st.number_input("Max True Age (Yrs)", value=100.0, step=1.0)
        with l5: min_atr_pct = st.number_input("Min ATR %", value=2.5, step=0.1)
        with l6: max_atr_pct = st.number_input("Max ATR %", value=10.0, step=0.1) 
    
    with st.expander("Accumulation/Distribution Counts (Independent)", expanded=False):
        st.markdown("**Filters are additive (AND logic).**")
        st.markdown("• **Acc:** Close > Open, Vol > Prev, Vol > 63d MA.\n• **Dist:** Close < Open, Vol > Prev, Vol > 63d MA.")
        
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

    with st.expander("Gap Filter (Momentum/Exhaustion)", expanded=False):
        use_gap_filter = st.checkbox("Enable Open Gap Filter", value=False, help="Count days where Low > Prev High (Unfilled Gap Up) in the window.")
        g1, g2, g3 = st.columns(3)
        with g1: gap_lookback = st.number_input("Lookback Window (Days)", 1, 252, 21, disabled=not use_gap_filter)
        with g2: gap_logic = st.selectbox("Gap Count Logic", [">", "<", "="], disabled=not use_gap_filter)
        with g3: gap_thresh = st.number_input("Count Threshold", 0, 50, 3, disabled=not use_gap_filter)

    with st.expander("Distance from MA Filter (ATR Units)", expanded=False):
        st.caption("Calculation: (Close - Moving Average) / 14-Period ATR")
        use_ma_dist_filter = st.checkbox("Enable Distance Filter", value=False)
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            dist_ma_type = st.selectbox("Select Moving Average", 
                ["SMA 10", "SMA 20", "SMA 50", "SMA 100", "SMA 200", "EMA 8", "EMA 11", "EMA 21"],
                disabled=not use_ma_dist_filter
            )
        with d2:
            dist_logic = st.selectbox("Logic", ["Greater Than (>)", "Less Than (<)", "Between"], disabled=not use_ma_dist_filter)
        with d3:
            dist_min = st.number_input("Min ATR Dist (or > threshold)", -50.0, 50.0, 0.0, step=0.5, disabled=not use_ma_dist_filter)
        with d4:
            dist_max = st.number_input("Max ATR Dist (or < threshold)", -50.0, 50.0, 2.0, step=0.5, disabled=not use_ma_dist_filter)

    with st.expander("Price Action", expanded=False):
        pa1, pa2 = st.columns(2)
        with pa1: 
            req_green_candle = st.checkbox("Require Close > Open (Green Candle)", value=False)
            
            # --- NEW: BREAKOUT FILTER ---
            st.markdown("**Daily Breakout**")
            breakout_mode = st.selectbox("Close vs Prev Range", 
                ["None", "Close > Prev Day High", "Close < Prev Day Low"],
                help="Requires today's close to be outside yesterday's range."
            )
            # ----------------------------

        with pa2:
            st.markdown("**Candle Range Location %**")
            use_range_filter = st.checkbox("Filter by Range %", value=False)
            r1, r2 = st.columns(2)
            with r1: range_min = st.number_input("Min % (0=Low)", 0, 100, 0, disabled=not use_range_filter)
            with r2: range_max = st.number_input("Max % (100=High)", 0, 100, 100, disabled=not use_range_filter)

    with st.expander("Day of Week Filter", expanded=False):
        use_dow_filter = st.checkbox("Enable Day of Week Filter", value=False)
        c_mon, c_tue, c_wed, c_thu, c_fri = st.columns(5)
        valid_days = []
        with c_mon: 
            if st.checkbox("Monday", value=True, disabled=not use_dow_filter): valid_days.append(0)
        with c_tue: 
            if st.checkbox("Tuesday", value=True, disabled=not use_dow_filter): valid_days.append(1)
        with c_wed: 
            if st.checkbox("Wednesday", value=True, disabled=not use_dow_filter): valid_days.append(2)
        with c_thu: 
            if st.checkbox("Thursday", value=True, disabled=not use_dow_filter): valid_days.append(3)
        with c_fri: 
            if st.checkbox("Friday", value=True, disabled=not use_dow_filter): valid_days.append(4)
    with st.expander("Trend Filter", expanded=False):
        t1, _ = st.columns([1, 3])
        with t1:
            trend_filter = st.selectbox("Trend Condition", 
                ["None", "Price > 200 SMA", "Price > Rising 200 SMA", "Market > 200 SMA",
                 "Price < 200 SMA", "Price < Falling 200 SMA", "Market < 200 SMA"],
                help=f"Requires 200 days of data. 'Market' filters check the regime of {MARKET_TICKER}.")
    with st.expander("Performance Percentile Rank (Granular Multi-Filter)", expanded=False):
        st.write("Enable multiple windows. Set 'Consec Days' for **each** timeframe individually.")
        col_p_config, col_p_seq = st.columns([3, 1])
        perf_filters = []
        with col_p_config:
            c5d, c10d, c21d = st.columns(3)
            with c5d:
                use_5d = st.checkbox("Enable 5D Rank")
                logic_5d = st.selectbox("Logic", [">", "<"], key="l5d", disabled=not use_5d)
                thresh_5d = st.number_input("Threshold", 0.0, 100.0, 80.0, key="t5d", disabled=not use_5d)
                consec_5d = st.number_input("Consec Days", 1, 10, 1, key="c5d_days", disabled=not use_5d)
                if use_5d: perf_filters.append({'window': 5, 'logic': logic_5d, 'thresh': thresh_5d, 'consecutive': consec_5d})
            with c10d:
                use_10d = st.checkbox("Enable 10D Rank")
                logic_10d = st.selectbox("Logic", [">", "<"], key="l10d", disabled=not use_10d)
                thresh_10d = st.number_input("Threshold", 0.0, 100.0, 80.0, key="t10d", disabled=not use_10d)
                consec_10d = st.number_input("Consec Days", 1, 10, 1, key="c10d_days", disabled=not use_10d)
                if use_10d: perf_filters.append({'window': 10, 'logic': logic_10d, 'thresh': thresh_10d, 'consecutive': consec_10d})
            with c21d:
                use_21d = st.checkbox("Enable 21D Rank")
                logic_21d = st.selectbox("Logic", [">", "<"], key="l21d", disabled=not use_21d)
                thresh_21d = st.number_input("Threshold", 0.0, 100.0, 80.0, key="t21d", disabled=not use_21d)
                consec_21d = st.number_input("Consec Days", 1, 10, 1, key="c21d_days", disabled=not use_21d)
                if use_21d: perf_filters.append({'window': 21, 'logic': logic_21d, 'thresh': thresh_21d, 'consecutive': consec_21d})
        with col_p_seq:
            st.markdown("**Combined Signal**")
            st.caption("After individual filters pass, alert on:")
            perf_first = st.checkbox("First Instance", value=True)
            perf_lookback = st.number_input("Lookback (Days)", 1, 100, 21, disabled=not perf_first)
    with st.expander("Seasonal Rank", expanded=False):
        st.markdown("**Ticker Specific Seasonality**")
        use_sznl = st.checkbox("Enable Ticker Seasonal Filter", value=False)
        s1, s2, s3, s4 = st.columns(4)
        with s1: sznl_logic = st.selectbox("Logic", ["<", ">"], key="sl", disabled=not use_sznl)
        with s2: sznl_thresh = st.number_input("Threshold", 0.0, 100.0, 15.0, key="st", disabled=not use_sznl)
        with s3: sznl_first = st.checkbox("First Instance Only", value=True, key="sf", disabled=not use_sznl)
        with s4: sznl_lookback = st.number_input("Instance Lookback (Days)", 1, 100, 21, key="slb", disabled=not use_sznl)
        st.markdown("---")
        st.markdown(f"**Market ({MARKET_TICKER}) Seasonality**")
        use_market_sznl = st.checkbox("Enable Market Seasonal Filter", value=False)
        spy1, spy2 = st.columns(2)
        with spy1: market_sznl_logic = st.selectbox("Market Logic", ["<", ">"], key="spy_sl", disabled=not use_market_sznl)
        with spy2: market_sznl_thresh = st.number_input("Market Threshold", 0.0, 100.0, 15.0, key="spy_st", disabled=not use_market_sznl)
    
    with st.expander("52-Week High/Low", expanded=False):
        use_52w = st.checkbox("Enable 52w High/Low Filter", value=False)
        h1, h2, h3, h4 = st.columns(4) 
        with h1: type_52w = st.selectbox("Condition", ["New 52w High", "New 52w Low"], disabled=not use_52w)
        with h2: first_52w = st.checkbox("First Instance Only", value=True, key="hf", disabled=not use_52w)
        with h3: lookback_52w = st.number_input("Instance Lookback (Days)", 1, 252, 21, key="hlb", disabled=not use_52w)
        # --- NEW: LAG INPUT ---
        with h4: 
            lag_52w = st.number_input("Lag (Days)", 0, 10, 0, disabled=not use_52w, 
                                      help="0 = Today, 1 = Yesterday (e.g., Signal if YESTERDAY was a 52w High)")
        # ----------------------

    with st.expander("Volume Filters (Spike & Regime)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Volume Spike** (Raw Ratio)")
            use_vol = st.checkbox("Enable Spike Filter", value=False)
            vol_thresh = st.number_input("Vol Multiple (> X * 63d Avg)", 1.0, 10.0, 1.5, disabled=not use_vol)
        with c2:
            st.markdown("**Volume Regime** (10d Rel Vol Rank)")
            use_vol_rank = st.checkbox("Enable Regime Filter", value=False)
            v_col1, v_col2 = st.columns(2)
            with v_col1: vol_rank_logic = st.selectbox("Logic", ["<", ">"], key="vrl", disabled=not use_vol_rank)
            with v_col2: vol_rank_thresh = st.number_input("Percentile (0-100)", 0.0, 100.0, 50.0, key="vrt", disabled=not use_vol_rank)

    st.markdown("---")
    
    if st.button("Run Backtest", type="primary", use_container_width=True):
        tickers_to_run = []
        sznl_map = load_seasonal_map()
        if univ_choice == "Sector ETFs": tickers_to_run = SECTOR_ETFS
        elif univ_choice == "Indices": tickers_to_run = INDEX_ETFS
        elif univ_choice == "International ETFs": tickers_to_run = INTERNATIONAL_ETFS
        elif univ_choice == "Sector + Index ETFs": tickers_to_run = list(set(SECTOR_ETFS + INDEX_ETFS))
        elif univ_choice == "All CSV Tickers": tickers_to_run = [t for t in list(sznl_map.keys()) if t not in ["BTC-USD", "ETH-USD"]]
        elif univ_choice == "Custom (Upload CSV)": tickers_to_run = custom_tickers
        if tickers_to_run and sample_pct < 100:
            count = max(1, int(len(tickers_to_run) * (sample_pct / 100)))
            tickers_to_run = random.sample(tickers_to_run, count)
            st.info(f"Randomly selected {len(tickers_to_run)} tickers for this run.")
        if not tickers_to_run:
            st.error("No tickers found.")
            return
        if use_full_history: fetch_start = "1950-01-01"
        else: fetch_start = start_date - datetime.timedelta(days=365)

        st.info(f"Downloading data ({len(tickers_to_run)} tickers)...")
        data_dict = download_universe_data(tickers_to_run, fetch_start)
        if not data_dict: return
        
        market_series = None
        need_market_data = ("Market" in trend_filter) or use_market_sznl
        if need_market_data:
            if MARKET_TICKER in data_dict: market_df = data_dict[MARKET_TICKER]
            else:
                st.info(f"Fetching {MARKET_TICKER} data for regime/seasonal filter...")
                market_dict_temp = download_universe_data([MARKET_TICKER], fetch_start)
                market_df = market_dict_temp.get(MARKET_TICKER, None)
            if market_df is not None and not market_df.empty:
                if market_df.index.tz is not None: market_df.index = market_df.index.tz_localize(None)
                market_df.index = market_df.index.normalize()
                market_df['SMA200'] = market_df['Close'].rolling(200).mean()
                market_series = market_df['Close'] > market_df['SMA200']
            else: st.warning(f"⚠️ {MARKET_TICKER} data unavailable. Market filters ignored.")

        params = {
            'backtest_start_date': start_date, 'trade_direction': trade_direction,
            'max_one_pos': max_one_pos, 'allow_same_day_reentry': allow_same_day_reentry,
            'use_stop_loss': use_stop_loss,     
            'use_take_profit': use_take_profit, 
            'time_exit_only': time_exit_only,   
            'stop_atr': stop_atr, 'tgt_atr': tgt_atr, 'holding_days': hold_days,
            'entry_type': entry_type, 'use_ma_entry_filter': use_ma_entry_filter, 'require_close_gt_open': req_green_candle,
            'breakout_mode': breakout_mode, 
            'use_range_filter': use_range_filter, 'range_min': range_min, 'range_max': range_max,
            'use_dow_filter': use_dow_filter, 'allowed_days': valid_days,
            'min_price': min_price, 'min_vol': min_vol, 'min_age': min_age, 'max_age': max_age, 'min_atr_pct': min_atr_pct,'max_atr_pct': max_atr_pct,
            'trend_filter': trend_filter, 'universe_tickers': tickers_to_run, 'slippage_bps': slippage_bps,
            'entry_conf_bps': entry_conf_bps,
            'perf_filters': perf_filters, 'perf_first_instance': perf_first, 'perf_lookback': perf_lookback, 
            'use_sznl': use_sznl, 'sznl_logic': sznl_logic, 'sznl_thresh': sznl_thresh, 
            'sznl_first_instance': sznl_first, 'sznl_lookback': sznl_lookback,
            'use_market_sznl': use_market_sznl, 'market_sznl_logic': market_sznl_logic, 'market_sznl_thresh': market_sznl_thresh,
            'use_52w': use_52w, '52w_type': type_52w, '52w_first_instance': first_52w, '52w_lookback': lookback_52w, '52w_lag': lag_52w, 
            'use_vol': use_vol, 'vol_thresh': vol_thresh,
            'use_vol_rank': use_vol_rank, 'vol_rank_logic': vol_rank_logic, 'vol_rank_thresh': vol_rank_thresh,
            'use_ma_dist_filter': use_ma_dist_filter, 'dist_ma_type': dist_ma_type, 
            'dist_logic': dist_logic, 'dist_min': dist_min, 'dist_max': dist_max,
            'use_gap_filter': use_gap_filter, 'gap_lookback': gap_lookback, 
            'gap_logic': gap_logic, 'gap_thresh': gap_thresh,
            'use_acc_count_filter': use_acc_count_filter, 'acc_count_window': acc_count_window, 
            'acc_count_logic': acc_count_logic, 'acc_count_thresh': acc_count_thresh,
            'use_dist_count_filter': use_dist_count_filter, 'dist_count_window': dist_count_window, 
            'dist_count_logic': dist_count_logic, 'dist_count_thresh': dist_count_thresh
        }
        
        trades_df, rejected_df, total_signals = run_engine(data_dict, params, sznl_map, market_series)
        
        if trades_df.empty: st.warning("No executed signals (Check constraints or signal logic).")
        
        if not trades_df.empty:
            trades_df = trades_df.sort_values("ExitDate")
            trades_df['PnL_Dollar'] = trades_df['R'] * risk_per_trade
            trades_df['CumPnL'] = trades_df['PnL_Dollar'].cumsum()
            trades_df['SignalDate'] = pd.to_datetime(trades_df['SignalDate'])
            trades_df['EntryDate'] = pd.to_datetime(trades_df['EntryDate'])
            trades_df['DayOfWeek'] = trades_df['EntryDate'].dt.day_name()
            trades_df['Year'] = trades_df['SignalDate'].dt.year
            trades_df['Month'] = trades_df['SignalDate'].dt.strftime('%b')
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
        else:
            win_rate = 0; pf = 0; sqn = 0
            
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
            fig = px.line(trades_df, x="ExitDate", y="CumPnL", title=f"Actual Portfolio Equity (Risk: ${risk_per_trade}/trade)", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Performance Breakdowns")
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
            ticker_pnl = ticker_pnl.sort_values("PnL_Dollar", ascending=False).head(275)
            st.plotly_chart(px.bar(ticker_pnl, x="Ticker", y="PnL_Dollar", title="Cumulative PnL by Ticker (Top 75)", text_auto='.2s'), use_container_width=True)
        
        st.subheader("Trade Logs")
        tab1, tab2 = st.tabs(["Executed Trades", "Missed Trades (Failed Entry or Constraints)"])
        with tab1:
            if not trades_df.empty:
                st.dataframe(trades_df.style.format({
                    "Entry": "{:.2f}", "Exit": "{:.2f}", "R": "{:.2f}", "PnL_Dollar": "${:,.2f}",
                    "Age": "{:.1f}y", "AvgVol": "{:,.0f}"
                }), use_container_width=True)
            else: st.info("No Executed Trades.")
        with tab2:
            if not rejected_df.empty:
                st.markdown("**Includes both Portfolio Constraints (Max Positions) AND Technical Entry Failures (e.g. No Gap Up).**")
                # Calc PnL for rejected (Assuming Open Entry)
                if 'PnL_Dollar' not in rejected_df.columns:
                    rejected_df['PnL_Dollar'] = rejected_df['R'] * risk_per_trade
                st.dataframe(rejected_df.style.format({
                    "Entry": "{:.2f}", "Exit": "{:.2f}", "R": "{:.2f}", "PnL_Dollar": "${:,.2f}",
                    "Age": "{:.1f}y", "AvgVol": "{:,.0f}"
                }), use_container_width=True)
            else: st.info("No signals were rejected.")

        if not trades_df.empty or not rejected_df.empty:
            st.markdown("---")
            st.subheader("Configuration & Results (Copy Code)")
            st.info("Copy the dictionary below and paste it into your `STRATEGY_BOOK` list in the Screener.")
            dict_str = f"""{{
    "id": "STRAT_{int(time.time())}",
    "name": "Generated Strategy ({grade})",
    "description": "Start: {start_date}. Universe: {univ_choice}. Dir: {trade_direction}. Filter: {trend_filter}. PF: {pf:.2f}. SQN: {sqn:.2f}.",
    "universe_tickers": {tickers_to_run}, 
    "settings": {{
        "trade_direction": "{trade_direction}",
        "entry_type": "{entry_type}",
        "max_one_pos": {max_one_pos},
        "allow_same_day_reentry": {allow_same_day_reentry},
        "max_daily_entries": {max_daily_entries},
        "max_total_positions": {max_total_positions},
        "perf_filters": {perf_filters},
        "perf_first_instance": {perf_first}, "perf_lookback": {perf_lookback},
        "use_sznl": {use_sznl}, "sznl_logic": "{sznl_logic}", "sznl_thresh": {sznl_thresh}, "sznl_first_instance": {sznl_first}, "sznl_lookback": {sznl_lookback},
        "use_market_sznl": {use_market_sznl}, "market_sznl_logic": "{market_sznl_logic}", "market_sznl_thresh": {market_sznl_thresh},
        "market_ticker": "{MARKET_TICKER}",
        "use_52w": {use_52w}, "52w_type": "{type_52w}", "52w_first_instance": {first_52w}, "52w_lookback": {lookback_52w}, "52w_lag": {lag_52w},
        "breakout_mode": "{breakout_mode}",
        "use_vol": {use_vol}, "vol_thresh": {vol_thresh},
        "use_vol_rank": {use_vol_rank}, "vol_rank_logic": "{vol_rank_logic}", "vol_rank_thresh": {vol_rank_thresh},
        "trend_filter": "{trend_filter}",
        "min_price": {min_price}, "min_vol": {min_vol},
        "min_age": {min_age}, "max_age": {max_age},
        "min_atr_pct": {min_atr_pct},"max_atr_pct": {max_atr_pct},
        "entry_conf_bps": {entry_conf_bps},
        "use_ma_dist_filter": {use_ma_dist_filter}, "dist_ma_type": "{dist_ma_type}", 
        "dist_logic": "{dist_logic}", "dist_min": {dist_min}, "dist_max": {dist_max},
        "use_gap_filter": {use_gap_filter}, "gap_lookback": {gap_lookback}, 
        "gap_logic": "{gap_logic}", "gap_thresh": {gap_thresh},
        "use_acc_count_filter": {use_acc_count_filter}, "acc_count_window": {acc_count_window}, "acc_count_logic": "{acc_count_logic}", "acc_count_thresh": {acc_count_thresh},
        "use_dist_count_filter": {use_dist_count_filter}, "dist_count_window": {dist_count_window}, "dist_count_logic": "{dist_count_logic}", "dist_count_thresh": {dist_count_thresh}
    }},
    "execution": {{
        "risk_per_trade": {risk_per_trade},
        "slippage_bps": {slippage_bps},
        "stop_atr": {stop_atr},
        "tgt_atr": {tgt_atr},
        "hold_days": {hold_days}
    }},
    "stats": {{
        "grade": "{grade} ({verdict})",
        "win_rate": "{win_rate:.1f}%",
        "expectancy": "${trades_df['PnL_Dollar'].mean() if not trades_df.empty else 0:.2f}",
        "profit_factor": "{pf:.2f}"
    }}
}},"""
            st.code(dict_str, language="python")

if __name__ == "__main__":
    main()
