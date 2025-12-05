import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import datetime
import random
import time
import uuid

# -----------------------------------------------------------------------------
# CONFIG / CONSTANTS
# -----------------------------------------------------------------------------
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
    """
    Loads the seasonal ranks CSV and creates a mapping of Ticker -> {Date -> Rank}.
    """
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        return {}

    if df.empty: return {}

    df["Date"] = pd.to_datetime(df["Date"], errors='coerce').dt.normalize().dt.tz_localize(None)
    df = df.dropna(subset=["Date"])
    
    output_map = {}
    for ticker, group in df.groupby("ticker"):
        output_map[ticker] = pd.Series(
            group.seasonal_rank.values, index=group.Date
        ).to_dict()
    return output_map

def get_sznl_val_series(ticker, dates, sznl_map):
    t_map = sznl_map.get(ticker, {})
    if not t_map:
        return pd.Series(50.0, index=dates)
    return dates.map(t_map).fillna(50.0)

def clean_ticker_df(df):
    if df.empty: return df
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    
    if 'Close' not in df.columns and 'Adj close' in df.columns:
        df.rename(columns={'Adj close': 'Close'}, inplace=True)
        
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
# ANALYSIS ENGINE (UPDATED FOR MULTI-PERF FILTERS)
# -----------------------------------------------------------------------------

def calculate_indicators(df, sznl_map, ticker, spy_series=None):
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    
    # Moving Averages
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # Perf Ranks (Always calculate 5, 10, 21 so they are available)
    for window in [5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window)
        df[f'rank_ret_{window}d'] = df[f'ret_{window}d'].expanding(min_periods=252).rank(pct=True) * 100.0
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    
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
    
    # Age
    if not df.empty:
        start_ts = df.index[0]
        df['age_years'] = (df.index - start_ts).days / 365.25
    else:
        df['age_years'] = 0.0
        
    # SPY Regime Mapping
    if spy_series is not None:
        df['SPY_Above_SMA200'] = spy_series.reindex(df.index, method='ffill').fillna(False)

    # Candle Range % Location
    denom = (df['High'] - df['Low'])
    df['RangePct'] = np.where(denom == 0, 0.5, (df['Close'] - df['Low']) / denom)

    # Day of Week
    df['DayOfWeekVal'] = df.index.dayofweek

    return df

def run_engine(universe_dict, params, sznl_map, spy_series=None):
    all_potential_trades = []
    
    total = len(universe_dict)
    bt_start_ts = pd.to_datetime(params['backtest_start_date'])
    direction = params.get('trade_direction', 'Long')
    max_one_pos_per_ticker = params.get('max_one_pos', True)
    
    # Entry Logic flags
    entry_mode = params['entry_type']
    is_pullback = "Pullback" in entry_mode
    is_limit_atr = "Limit (Close -0.5 ATR)" in entry_mode
    is_limit_prev = "Limit (Prev Close)" in entry_mode
    is_limit_entry = is_limit_atr or is_limit_prev

    use_ma_filter = params.get('use_ma_entry_filter', False)
    pullback_col = None
    if "10 SMA" in entry_mode: pullback_col = "SMA10"
    elif "21 EMA" in entry_mode: pullback_col = "EMA21"
    pullback_use_level = "Level" in entry_mode

    total_signals_generated = 0

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (ticker, df_raw) in enumerate(universe_dict.items()):
        status_text.text(f"Scanning signals for {ticker}...")
        progress_bar.progress((i+1)/total)
        
        if len(df_raw) < 100: continue
        
        ticker_last_exit = pd.Timestamp.min
        
        try:
            df = calculate_indicators(df_raw, sznl_map, ticker, spy_series)
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
            elif "SPY" in trend_opt and 'SPY_Above_SMA200' in df.columns:
                if trend_opt == "SPY > 200 SMA":
                    conditions.append(df['SPY_Above_SMA200'])
                elif trend_opt == "SPY < 200 SMA":
                    conditions.append(~df['SPY_Above_SMA200']) 
            
            # --- LIQUIDITY GATES ---
            curr_age = df['age_years'].fillna(0)
            curr_vol = df['vol_ma'].fillna(0)
            curr_close = df['Close'].fillna(0)
            gate = (curr_close >= params['min_price']) & \
                   (curr_vol >= params['min_vol']) & \
                   (curr_age >= params['min_age']) & \
                   (curr_age <= params['max_age'])
            conditions.append(gate)

            # --- PRICE ACTION ---
            if params.get('require_close_gt_open', False):
                conditions.append(df['Close'] > df['Open'])
            
            if params.get('use_range_filter', False):
                r_min = params.get('range_min', 0.0)
                r_max = params.get('range_max', 100.0)
                conditions.append((df['RangePct'] * 100 >= r_min) & (df['RangePct'] * 100 <= r_max))
            
            if params.get('use_dow_filter', False):
                allowed_days = params.get('allowed_days', [])
                if allowed_days:
                    conditions.append(df['DayOfWeekVal'].isin(allowed_days))

            # --- MULTI-PERFORMANCE FILTER (UPDATED) ---
            perf_filters = params.get('perf_filters', [])
            if perf_filters:
                # 1. Combine all active filters into one raw boolean series (AND logic)
                combined_perf_raw = pd.Series(True, index=df.index)
                
                for pf in perf_filters:
                    col = f"rank_ret_{pf['window']}d"
                    if pf['logic'] == '<': 
                        cond_f = df[col] < pf['thresh']
                    else: 
                        cond_f = df[col] > pf['thresh']
                    combined_perf_raw = combined_perf_raw & cond_f
                
                # 2. Apply Sequence Logic (Consecutive & First Instance) to the Combined Signal
                final_perf_cond = combined_perf_raw
                
                # Consecutive Days
                consec_days = params.get('perf_consecutive', 1)
                if consec_days > 1:
                    final_perf_cond = final_perf_cond.rolling(consec_days).sum() == consec_days
                
                # First Instance Lookback
                if params.get('perf_first_instance', False):
                    lookback = params.get('perf_lookback', 21)
                    prev_instances = final_perf_cond.shift(1).rolling(lookback).sum()
                    final_perf_cond = final_perf_cond & (prev_instances == 0)
                
                conditions.append(final_perf_cond)

            # --- SEASONALITY ---
            if params['use_sznl']:
                if params['sznl_logic'] == '<': cond = df['Sznl'] < params['sznl_thresh']
                else: cond = df['Sznl'] > params['sznl_thresh']
                if params['sznl_first_instance']:
                    prev = cond.shift(1).rolling(params['sznl_lookback']).sum()
                    cond = cond & (prev == 0)
                conditions.append(cond)
            
            # --- 52W HIGHS ---
            if params['use_52w']:
                if params['52w_type'] == 'New 52w High': cond = df['is_52w_high']
                else: cond = df['is_52w_low']
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
            
            # --- TRADE EXECUTION SIMULATION (Per Ticker) ---
            for signal_date in signal_dates:
                if max_one_pos_per_ticker and signal_date <= ticker_last_exit: continue
                
                sig_idx = df.index.get_loc(signal_date)
                fixed_exit_idx = sig_idx + params['holding_days']
                if fixed_exit_idx >= len(df): continue

                found_entry = False
                actual_entry_idx = -1
                actual_entry_price = 0.0
                
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
                                    continue 
                            
                            found_entry = True
                            actual_entry_idx = curr_check_idx
                            if pullback_use_level:
                                actual_entry_price = ma_val
                            else:
                                actual_entry_price = row['Close']
                            break

                # PATH B: LIMIT ENTRIES
                elif is_limit_entry:
                    sig_row = df.iloc[sig_idx]
                    sig_close = sig_row['Close']
                    sig_atr = sig_row['ATR']
                    
                    if np.isnan(sig_atr): continue

                    if is_limit_atr:
                        if direction == 'Long':
                            limit_price = sig_close - (0.5 * sig_atr)
                        else:
                            limit_price = sig_close + (0.5 * sig_atr)
                    else: 
                        limit_price = sig_close

                    for wait_i in range(1, 4):
                        curr_check_idx = sig_idx + wait_i
                        if curr_check_idx >= len(df): break
                        
                        row = df.iloc[curr_check_idx]
                        is_filled = False
                        if direction == 'Long':
                            if row['Low'] <= limit_price:
                                is_filled = True
                                actual_entry_price = limit_price 
                        else:
                            if row['High'] >= limit_price:
                                is_filled = True
                                actual_entry_price = limit_price

                        if is_filled:
                            found_entry = True
                            actual_entry_idx = curr_check_idx
                            break

                # PATH C: IMMEDIATE
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

                if found_entry and actual_entry_idx < len(df):
                    
                    start_exit_scan = actual_entry_idx + 1
                    if start_exit_scan > fixed_exit_idx: continue 
                    
                    future = df.iloc[start_exit_scan : fixed_exit_idx + 1]
                    if future.empty: continue
                    
                    atr = df['ATR'].iloc[actual_entry_idx]
                    if np.isnan(atr) or atr == 0: 
                        atr = df['ATR'].iloc[actual_entry_idx-1] if actual_entry_idx > 0 else 0
                        if atr == 0: continue

                    if direction == 'Long':
                        stop_price = actual_entry_price - (atr * params['stop_atr'])
                        tgt_price = actual_entry_price + (atr * params['tgt_atr'])
                    else:
                        stop_price = actual_entry_price + (atr * params['stop_atr'])
                        tgt_price = actual_entry_price - (atr * params['tgt_atr'])

                    exit_price = actual_entry_price
                    exit_type = "Hold"
                    exit_date = None
                    
                    if not params['time_exit_only']:
                        for f_date, f_row in future.iterrows():
                            if direction == 'Long':
                                if f_row['Low'] <= stop_price:
                                    exit_price = f_row['Open'] if f_row['Open'] < stop_price else stop_price
                                    exit_type = "Stop"
                                    exit_date = f_date
                                    break
                                if f_row['High'] >= tgt_price:
                                    exit_price = f_row['Open'] if f_row['Open'] > tgt_price else tgt_price
                                    exit_type = "Target"
                                    exit_date = f_date
                                    break
                            else: # Short
                                if f_row['High'] >= stop_price:
                                    exit_price = f_row['Open'] if f_row['Open'] > stop_price else stop_price
                                    exit_type = "Stop"
                                    exit_date = f_date
                                    break
                                if f_row['Low'] <= tgt_price:
                                    exit_price = f_row['Open'] if f_row['Open'] < tgt_price else tgt_price
                                    exit_type = "Target"
                                    exit_date = f_date
                                    break
                    
                    if exit_type == "Hold":
                        exit_price = future['Close'].iloc[-1]
                        exit_date = future.index[-1]
                        exit_type = "Time"
                    
                    ticker_last_exit = exit_date

                    if direction == 'Long':
                        risk_unit = actual_entry_price - stop_price
                        pnl = exit_price - actual_entry_price
                    else:
                        risk_unit = stop_price - actual_entry_price
                        pnl = actual_entry_price - exit_price
                        
                    if risk_unit <= 0: risk_unit = 0.001
                    r = pnl / risk_unit
                    
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
                        "AvgVol": df['vol_ma'].iloc[sig_idx]
                    })
        except: continue
    
    progress_bar.empty()
    status_text.empty()

    if not all_potential_trades:
        return pd.DataFrame(), 0

    st.info(f"Processing Portfolio Constraints on {len(all_potential_trades)} potential signals...")
    
    potential_df = pd.DataFrame(all_potential_trades)
    potential_df = potential_df.sort_values(by=["EntryDate", "Ticker"])
    
    final_trades_log = []
    active_positions = [] 
    daily_entries = {} 
    
    max_daily = params.get('max_daily_entries', 100)
    max_total = params.get('max_total_positions', 100)

    for idx, trade in potential_df.iterrows():
        entry_date = trade['EntryDate']
        
        active_positions = [t for t in active_positions if t['ExitDate'] > entry_date]
        
        if len(active_positions) >= max_total:
            continue
            
        today_count = daily_entries.get(entry_date, 0)
        if today_count >= max_daily:
            continue
            
        final_trades_log.append(trade)
        active_positions.append(trade)
        daily_entries[entry_date] = today_count + 1

    return pd.DataFrame(final_trades_log), total_signals_generated

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

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="Quantitative Backtester")
    st.title("Quantitative Strategy Backtester")
    st.markdown("---")

    # 1. UNIVERSE
    st.subheader("1. Universe & Data")
    col_u1, col_u2, col_u3 = st.columns([1, 1, 2])
    
    sample_pct = 100 
    use_full_history = False
    
    with col_u1:
        univ_choice = st.selectbox("Choose Universe", 
            ["Sector ETFs", "Indices", "International ETFs", "Sector + Index ETFs", "All CSV Tickers", "Custom (Upload CSV)"])
    with col_u2:
        default_start = datetime.date(2000, 1, 1)
        start_date = st.date_input("Backtest Start Date", 
                                   value=default_start, 
                                   min_value=datetime.date(1950, 1, 1),
                                   max_value=datetime.date.today())
    
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
    with r_c1:
        trade_direction = st.selectbox("Trade Direction", ["Long", "Short"])
    with r_c2:
        time_exit_only = st.checkbox("Time Exit Only (Disable Stop/Target)")
    with r_c3:
        max_one_pos = st.checkbox("Max 1 Position/Ticker", value=True, 
            help="If checked, allows only one open trade at a time per ticker.")
    
    p_c1, p_c2 = st.columns(2)
    with p_c1:
        max_daily_entries = st.number_input("Max New Trades Per Day", min_value=1, max_value=100, value=2, step=1)
    with p_c2:
        max_total_positions = st.number_input("Max Total Positions Held", min_value=1, max_value=200, value=10, step=1)
    
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: 
        entry_type = st.selectbox("Entry Price", [
            "Signal Close", 
            "T+1 Open", 
            "T+1 Close",
            "Limit (Close -0.5 ATR)",
            "Limit (Prev Close)", 
            "Pullback 10 SMA (Entry: Close)",
            "Pullback 10 SMA (Entry: Level)",
            "Pullback 21 EMA (Entry: Close)",
            "Pullback 21 EMA (Entry: Level)"
        ])
        if "Pullback" in entry_type:
            use_ma_entry_filter = st.checkbox("Filter: Close > MA - 0.25*ATR", value=False)
        else:
            use_ma_entry_filter = False

    with c2: stop_atr = st.number_input("Stop Loss (ATR)", value=3.0, step=0.1)
    with c3: tgt_atr = st.number_input("Target (ATR)", value=8.0, step=0.1, disabled=time_exit_only)
    with c4: hold_days = st.number_input("Max Holding Days", value=10, step=1)
    with c5: risk_per_trade = st.number_input("Risk Amount ($)", value=1000, step=100)

    st.markdown("---")
    st.subheader("3. Signal Criteria")

    # A. LIQUIDITY
    with st.expander("Liquidity & Data History Filters", expanded=True):
        l1, l2, l3, l4 = st.columns(4)
        with l1: min_price = st.number_input("Min Price ($)", value=10.0, step=1.0)
        with l2: min_vol = st.number_input("Min Avg Volume", value=100000, step=50000)
        with l3: min_age = st.number_input("Min True Age (Yrs)", value=0.25, step=0.25)
        with l4: max_age = st.number_input("Max True Age (Yrs)", value=100.0, step=1.0)
    
    with st.expander("Price Action", expanded=True):
        pa1, pa2 = st.columns(2)
        with pa1:
            req_green_candle = st.checkbox("Require Close > Open (Green Candle)", value=False)
        with pa2:
            st.markdown("**Candle Range Location %**")
            use_range_filter = st.checkbox("Filter by Range %", value=False)
            r1, r2 = st.columns(2)
            with r1: range_min = st.number_input("Min % (0=Low)", 0, 100, 0, disabled=not use_range_filter)
            with r2: range_max = st.number_input("Max % (100=High)", 0, 100, 100, disabled=not use_range_filter)

    with st.expander("Day of Week Filter", expanded=False):
        use_dow_filter = st.checkbox("Enable Day of Week Filter", value=False)
        st.caption("Select valid days for a signal:")
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
        
    # B. TREND
    with st.expander("Trend Filter", expanded=True):
        t1, _ = st.columns([1, 3])
        with t1:
            trend_filter = st.selectbox("Trend Condition", 
                ["None", "Price > 200 SMA", "Price > Rising 200 SMA", "SPY > 200 SMA",
                 "Price < 200 SMA", "Price < Falling 200 SMA", "SPY < 200 SMA"],
                help="Requires 200 days of data. 'SPY' filters check the broad market regime.")

    # C. STRATEGY FILTERS
    # --- UPDATED SECTION FOR MULTIPLE PERF FILTERS ---
    with st.expander("Performance Percentile Rank (Multi-Filter)", expanded=False):
        st.write("Enable multiple windows to filter for overlap (e.g. 5d AND 21d).")
        
        col_p_config, col_p_seq = st.columns([3, 1])
        
        perf_filters = []
        
        with col_p_config:
            c5d, c10d, c21d = st.columns(3)
            with c5d:
                use_5d = st.checkbox("Enable 5D Rank")
                logic_5d = st.selectbox("Logic", [">", "<"], key="l5d", disabled=not use_5d)
                thresh_5d = st.number_input("Threshold", 0.0, 100.0, 80.0, key="t5d", disabled=not use_5d)
                if use_5d: perf_filters.append({'window': 5, 'logic': logic_5d, 'thresh': thresh_5d})
            
            with c10d:
                use_10d = st.checkbox("Enable 10D Rank")
                logic_10d = st.selectbox("Logic", [">", "<"], key="l10d", disabled=not use_10d)
                thresh_10d = st.number_input("Threshold", 0.0, 100.0, 80.0, key="t10d", disabled=not use_10d)
                if use_10d: perf_filters.append({'window': 10, 'logic': logic_10d, 'thresh': thresh_10d})

            with c21d:
                use_21d = st.checkbox("Enable 21D Rank")
                logic_21d = st.selectbox("Logic", [">", "<"], key="l21d", disabled=not use_21d)
                thresh_21d = st.number_input("Threshold", 0.0, 100.0, 80.0, key="t21d", disabled=not use_21d)
                if use_21d: perf_filters.append({'window': 21, 'logic': logic_21d, 'thresh': thresh_21d})

        with col_p_seq:
            st.markdown("**Sequence**")
            st.caption("Applied to Combined Signal")
            perf_consecutive = st.number_input("Min Consec Days", 1, 20, 1)
            perf_first = st.checkbox("First Instance", value=True)
            perf_lookback = st.number_input("Lookback (Days)", 1, 100, 21, disabled=not perf_first)

    with st.expander("Seasonal Rank", expanded=False):
        use_sznl = st.checkbox("Enable Seasonal Filter", value=False)
        s1, s2, s3, s4 = st.columns(4)
        with s1: sznl_logic = st.selectbox("Seasonality", ["<", ">"], key="sl", disabled=not use_sznl)
        with s2: sznl_thresh = st.number_input("Seasonal Rank Threshold", 0.0, 100.0, 15.0, key="st", disabled=not use_sznl)
        with s3: sznl_first = st.checkbox("First Instance Only", value=True, key="sf", disabled=not use_sznl)
        with s4: sznl_lookback = st.number_input("Instance Lookback (Days)", 1, 100, 21, key="slb", disabled=not use_sznl)

    with st.expander("52-Week High/Low", expanded=False):
        use_52w = st.checkbox("Enable 52w High/Low Filter", value=False)
        h1, h2, h3 = st.columns(3)
        with h1: type_52w = st.selectbox("Condition", ["New 52w High", "New 52w Low"], disabled=not use_52w)
        with h2: first_52w = st.checkbox("First Instance Only", value=True, key="hf", disabled=not use_52w)
        with h3: lookback_52w = st.number_input("Instance Lookback (Days)", 1, 252, 21, key="hlb", disabled=not use_52w)

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

        if use_full_history:
             fetch_start = "1950-01-01"
        else:
             fetch_start = start_date - datetime.timedelta(days=365)

        st.info(f"Downloading data ({len(tickers_to_run)} tickers)...")
        
        data_dict = download_universe_data(tickers_to_run, fetch_start)
        if not data_dict: return
        
        # --- SPY HANDLING ---
        spy_series = None
        if "SPY" in trend_filter:
            if "SPY" in data_dict:
                spy_df = data_dict["SPY"]
            else:
                st.info("Fetching SPY data for regime filter...")
                spy_dict_temp = download_universe_data(["SPY"], fetch_start)
                spy_df = spy_dict_temp.get("SPY", None)

            if spy_df is not None and not spy_df.empty:
                if spy_df.index.tz is not None:
                      spy_df.index = spy_df.index.tz_localize(None)
                spy_df.index = spy_df.index.normalize()
                
                spy_df['SMA200'] = spy_df['Close'].rolling(200).mean()
                spy_series = spy_df['Close'] > spy_df['SMA200']
            else:
                st.warning("⚠️ SPY data unavailable. Regime filter ignored.")

        params = {
            'backtest_start_date': start_date,
            'trade_direction': trade_direction,
            'max_one_pos': max_one_pos,
            'time_exit_only': time_exit_only,
            'stop_atr': stop_atr, 'tgt_atr': tgt_atr, 'holding_days': hold_days, 'entry_type': entry_type,
            'use_ma_entry_filter': use_ma_entry_filter,
            'require_close_gt_open': req_green_candle,
            'use_range_filter': use_range_filter, 'range_min': range_min, 'range_max': range_max,
            'use_dow_filter': use_dow_filter, 'allowed_days': valid_days,
            'min_price': min_price, 'min_vol': min_vol, 'min_age': min_age, 'max_age': max_age,
            'trend_filter': trend_filter,
            # Updated Params for Multi-Perf
            'perf_filters': perf_filters,
            'perf_first_instance': perf_first, 'perf_lookback': perf_lookback, 'perf_consecutive': perf_consecutive,
            # End Updated Params
            'use_sznl': use_sznl, 'sznl_logic': sznl_logic, 'sznl_thresh': sznl_thresh, 
            'sznl_first_instance': sznl_first, 'sznl_lookback': sznl_lookback,
            'use_52w': use_52w, '52w_type': type_52w, '52w_first_instance': first_52w, '52w_lookback': lookback_52w,
            'use_vol': use_vol, 'vol_thresh': vol_thresh,
            'use_vol_rank': use_vol_rank, 'vol_rank_logic': vol_rank_logic, 'vol_rank_thresh': vol_rank_thresh
        }
        
        trades_df, total_signals = run_engine(data_dict, params, sznl_map, spy_series)
        
        if trades_df.empty:
            st.warning("No signals generated.")
            if total_signals > 0:
                st.warning(f"Note: {total_signals} raw signals were found, but none triggered an entry (Check Fill Logic).")
            return

        trades_df = trades_df.sort_values("ExitDate")
        trades_df['PnL_Dollar'] = trades_df['R'] * risk_per_trade
        trades_df['CumPnL'] = trades_df['PnL_Dollar'].cumsum()
        trades_df['SignalDate'] = pd.to_datetime(trades_df['SignalDate'])
        trades_df['Year'] = trades_df['SignalDate'].dt.year
        trades_df['Month'] = trades_df['SignalDate'].dt.strftime('%b')
        trades_df['MonthNum'] = trades_df['SignalDate'].dt.month
        trades_df['DayOfWeek'] = trades_df['SignalDate'].dt.day_name()
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
                <div><h3>Expectancy: ${trades_df['PnL_Dollar'].mean():.2f}</h3></div>
            </div>
            <div style="margin-top: 10px; color: #aaa; font-size: 14px;">
               Fill Rate: {fill_rate:.1f}% ({len(trades_df)} trades from {total_signals} signals)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if notes: st.warning("Notes: " + ", ".join(notes))

        fig = px.line(trades_df, x="ExitDate", y="CumPnL", title=f"Cumulative Equity (Risk: ${risk_per_trade}/trade)", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Performance Breakdowns")
        
        # 1. Year and Count
        y1, y2 = st.columns(2)
        y1.plotly_chart(px.bar(trades_df.groupby('Year')['PnL_Dollar'].sum().reset_index(), x='Year', y='PnL_Dollar', title="Net Profit ($) by Year", text_auto='.2s'), use_container_width=True)
        
        trades_per_year = trades_df.groupby('Year').size().reset_index(name='Count')
        y2.plotly_chart(px.bar(trades_per_year, x='Year', y='Count', title="Total Trades by Year", text_auto=True), use_container_width=True)
        
        # 2. Cycle and Month
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(trades_df.groupby('CyclePhase')['PnL_Dollar'].sum().reset_index().sort_values('CyclePhase'), x='CyclePhase', y='PnL_Dollar', title="PnL by Cycle", text_auto='.2s'), use_container_width=True)
        
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_pnl = trades_df.groupby("Month")["PnL_Dollar"].sum().reindex(month_order).reset_index()
        c2.plotly_chart(px.bar(monthly_pnl, x="Month", y="PnL_Dollar", title="PnL by Month (Seasonality)", text_auto='.2s'), use_container_width=True)
        
        # 3. Ticker
        ticker_pnl = trades_df.groupby("Ticker")["PnL_Dollar"].sum().reset_index()
        ticker_pnl = ticker_pnl.sort_values("PnL_Dollar", ascending=False).head(75)
        st.plotly_chart(px.bar(ticker_pnl, x="Ticker", y="PnL_Dollar", title="Cumulative PnL by Ticker (Top 75)", text_auto='.2s'), use_container_width=True)
        
        st.subheader("Trade Log")
        st.dataframe(trades_df.style.format({
            "Entry": "{:.2f}", "Exit": "{:.2f}", "R": "{:.2f}", "PnL_Dollar": "${:,.2f}",
            "Age": "{:.1f}y", "AvgVol": "{:,.0f}"
        }), use_container_width=True)

        # --- COPYABLE DICTIONARY OUTPUT ---
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
        "max_daily_entries": {max_daily_entries},
        "max_total_positions": {max_total_positions},
        "perf_filters": {perf_filters},
        "perf_first_instance": {perf_first}, "perf_lookback": {perf_lookback}, "perf_consecutive": {perf_consecutive},
        "use_sznl": {use_sznl}, "sznl_logic": "{sznl_logic}", "sznl_thresh": {sznl_thresh}, "sznl_first_instance": {sznl_first}, "sznl_lookback": {sznl_lookback},
        "use_52w": {use_52w}, "52w_type": "{type_52w}", "52w_first_instance": {first_52w}, "52w_lookback": {lookback_52w},
        "use_vol": {use_vol}, "vol_thresh": {vol_thresh},
        "use_vol_rank": {use_vol_rank}, "vol_rank_logic": "{vol_rank_logic}", "vol_rank_thresh": {vol_rank_thresh},
        "trend_filter": "{trend_filter}",
        "min_price": {min_price}, "min_vol": {min_vol},
        "min_age": {min_age}, "max_age": {max_age}
    }},
    "execution": {{
        "risk_per_trade": {risk_per_trade},
        "stop_atr": {stop_atr},
        "tgt_atr": {tgt_atr},
        "hold_days": {hold_days}
    }},
    "stats": {{
        "grade": "{grade} ({verdict})",
        "win_rate": "{win_rate:.1f}%",
        "expectancy": "${trades_df['PnL_Dollar'].mean():.2f}",
        "profit_factor": "{pf:.2f}"
    }}
}},"""
        st.code(dict_str, language="python")
if __name__ == "__main__":
    main()
