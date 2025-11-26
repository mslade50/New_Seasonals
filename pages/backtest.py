import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import datetime
import random
import time
import uuid
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

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

@st.cache_data(show_spinner=False)
def load_seasonal_map():
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        return {}

    if df.empty: return {}

    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])
    df["MD"] = df["Date"].apply(lambda x: (x.month, x.day))
    
    output_map = {}
    for ticker, group in df.groupby("ticker"):
        output_map[ticker] = pd.Series(
            group.seasonal_rank.values, index=group.MD
        ).to_dict()
    return output_map

# --- DYNAMIC SEASONALITY LOGIC (WALK-FORWARD) ---
def generate_dynamic_seasonal_profile(df, cutoff_date, target_year):
    """
    Calculates seasonal profile for a SPECIFIC target_year using only data 
    available prior to cutoff_date.
    """
    # 1. Filter History (Strictly Past Data)
    work_df = df[df.index < cutoff_date].copy()
    
    # Need enough history to form a valid opinion (at least 2 years)
    if len(work_df) < 500: 
        return {}

    if 'LogRet' not in work_df.columns:
        work_df['LogRet'] = np.log(work_df['Close'] / work_df['Close'].shift(1)) * 100.0
    
    work_df['Year'] = work_df.index.year
    work_df['MD'] = work_df.index.map(lambda x: (x.month, x.day))
    
    # 2. Determine Cycle Phase for the TARGET YEAR
    # (e.g., if forecasting 2012, we treat it as an Election Year phase)
    cycle_remainder = target_year % 4
    
    # 3. Calculate Forward Returns
    windows = [5, 10, 21]
    fwd_cols = []
    for w in windows:
        col_name = f'Fwd_{w}d'
        work_df[col_name] = work_df['LogRet'].shift(-w).rolling(w).sum()
        fwd_cols.append(col_name)

    # 4. Group by MD for ALL years available
    stats_all = work_df.groupby('MD')[fwd_cols].mean()
    
    # 5. Group by MD for same CYCLE years available
    cycle_df = work_df[work_df['Year'] % 4 == cycle_remainder]
    
    # Fallback: If we don't have enough cycle-specific years in history yet,
    # rely more heavily on the 'All' stats.
    if len(cycle_df) < 250: 
        stats_cycle = stats_all.copy()
    else:
        stats_cycle = cycle_df.groupby('MD')[fwd_cols].mean()
    
    # Align
    stats_cycle = stats_cycle.reindex(stats_all.index).fillna(method='ffill').fillna(method='bfill')
    
    # Rank (0-100)
    rnk_all = stats_all.rank(pct=True) * 100.0
    rnk_cycle = stats_cycle.rank(pct=True) * 100.0
    
    # Average Ranks across windows
    avg_rank_all = rnk_all.mean(axis=1)
    avg_rank_cycle = rnk_cycle.mean(axis=1)
    
    # Weighted Blend
    final_rank = (avg_rank_all + 3 * avg_rank_cycle) / 4.0
    
    # Smoothing
    final_rank_smooth = final_rank.rolling(window=5, center=True, min_periods=1).mean()
    
    return final_rank_smooth.to_dict()

def get_sznl_val_series(ticker, dates, sznl_map, df_hist=None):
    # 1. Static CSV Check
    # If the ticker is in the CSV, we use that (assuming user wants the specific static file data)
    t_map = sznl_map.get(ticker, {})
    if t_map:
        mds = dates.map(lambda x: (x.month, x.day))
        return mds.map(t_map).fillna(50.0)

    # 2. Dynamic Walk-Forward Generation
    if df_hist is not None and not df_hist.empty:
        # We need to construct the series year by year to simulate "learning"
        # Create a Series with the same index as dates, default to 50
        sznl_series = pd.Series(50.0, index=dates)
        
        # Identify years in the requested range
        unique_years = dates.year.unique()
        
        # Iterate through years to build the 'Expanding Window'
        for yr in unique_years:
            # We "stand" on Jan 1st of 'yr'
            cutoff = pd.Timestamp(yr, 1, 1)
            
            # Generate profile using only history before this year
            yearly_profile = generate_dynamic_seasonal_profile(df_hist, cutoff, yr)
            
            if yearly_profile:
                # Apply this profile to the dates within this specific year
                mask = (dates.year == yr)
                # Map (Month, Day) to the profile
                year_mds = dates[mask].map(lambda x: (x.month, x.day))
                sznl_series.loc[mask] = year_mds.map(yearly_profile).fillna(50.0)
                
        return sznl_series

    return pd.Series(50.0, index=dates)

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
                t_df = df[t].copy() if isinstance(df.columns, pd.MultiIndex) and t in df.columns.levels[0] else df.copy()
                t_df = clean_ticker_df(t_df)
                if not t_df.empty: data_dict[t] = t_df
            else:
                if not isinstance(df.columns, pd.MultiIndex): continue 
                for t in chunk:
                    try:
                        if t in df.columns.levels[0]:
                            t_df = df[t].copy()
                            t_df = clean_ticker_df(t_df)
                            if not t_df.empty: data_dict[t] = t_df
                    except: continue
            time.sleep(0.1)
        except Exception: continue
    
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

def calculate_indicators(df, sznl_map, ticker, spy_series=None):
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1)) * 100.0
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    for window in [5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window)
        df[f'rank_ret_{window}d'] = df[f'ret_{window}d'].expanding(min_periods=252).rank(pct=True) * 100.0
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    
    # Seasonality (Walk-Forward is handled inside get_sznl_val_series)
    df['Sznl'] = get_sznl_val_series(ticker, df.index, sznl_map, df)
    
    rolling_high = df['High'].shift(1).rolling(252).max()
    rolling_low = df['Low'].shift(1).rolling(252).min()
    df['is_52w_high'] = df['High'] > rolling_high
    df['is_52w_low'] = df['Low'] < rolling_low
    
    vol_ma = df['Volume'].rolling(63).mean()
    df['vol_ma'] = vol_ma
    df['vol_ratio'] = df['Volume'] / vol_ma
    
    vol_ma_10 = df['Volume'].rolling(10).mean()
    df['vol_ratio_10d'] = vol_ma_10 / vol_ma
    df['vol_ratio_10d_rank'] = df['vol_ratio_10d'].expanding(min_periods=252).rank(pct=True) * 100.0
    
    if not df.empty:
        start_ts = df.index[0]
        df['age_years'] = (df.index - start_ts).days / 365.25
    else:
        df['age_years'] = 0.0
        
    if spy_series is not None:
        df['SPY_Above_SMA200'] = spy_series.reindex(df.index, method='ffill').fillna(False)

    return df

def run_engine(universe_dict, params, sznl_map, spy_series=None):
    trades = []
    total = len(universe_dict)
    bt_start_ts = pd.to_datetime(params['backtest_start_date'])
    
    direction = params.get('trade_direction', 'Long')
    max_one_pos = params.get('max_one_pos', False)
    entry_mode = params['entry_type']
    is_pullback = "Pullback" in entry_mode
    use_ma_filter = params.get('use_ma_entry_filter', False)
    
    pullback_col = None
    if "10 SMA" in entry_mode: pullback_col = "SMA10"
    elif "21 EMA" in entry_mode: pullback_col = "EMA21"
    pullback_use_level = "Level" in entry_mode

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (ticker, df_raw) in enumerate(universe_dict.items()):
        status_text.text(f"Processing {ticker}...")
        progress_bar.progress((i+1)/total)
        
        if len(df_raw) < 100: continue
        last_exit_date = pd.Timestamp.min
        
        try:
            # Pass ticker's full history for walk-forward calculation
            df = calculate_indicators(df_raw, sznl_map, ticker, spy_series)
            
            # Filter for backtest window
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
                if trend_opt == "SPY > 200 SMA": conditions.append(df['SPY_Above_SMA200'])
                elif trend_opt == "SPY < 200 SMA": conditions.append(~df['SPY_Above_SMA200']) 
            
            # --- LIQUIDITY GATES ---
            gate = (df['Close'] >= params['min_price']) & \
                   (df['vol_ma'] >= params['min_vol']) & \
                   (df['age_years'] >= params['min_age']) & \
                   (df['age_years'] <= params['max_age'])
            conditions.append(gate)

            if params.get('require_close_gt_open', False):
                conditions.append(df['Close'] > df['Open'])

            # --- STRATEGY SIGNALS ---
            if params['use_perf_rank']:
                col = f"rank_ret_{params['perf_window']}d"
                raw_cond = df[col] < params['perf_thresh'] if params['perf_logic'] == '<' else df[col] > params['perf_thresh']
                cond = raw_cond.rolling(params['perf_consecutive']).sum() == params['perf_consecutive'] if params['perf_consecutive'] > 1 else raw_cond
                if params['perf_first_instance']:
                    prev = cond.shift(1).rolling(params['perf_lookback']).sum()
                    cond = cond & (prev == 0)
                conditions.append(cond)

            if params['use_sznl']:
                cond = df['Sznl'] < params['sznl_thresh'] if params['sznl_logic'] == '<' else df['Sznl'] > params['sznl_thresh']
                if params['sznl_first_instance']:
                    prev = cond.shift(1).rolling(params['sznl_lookback']).sum()
                    cond = cond & (prev == 0)
                conditions.append(cond)
            
            if params['use_52w']:
                cond = df['is_52w_high'] if params['52w_type'] == 'New 52w High' else df['is_52w_low']
                if params['52w_first_instance']:
                    prev = cond.shift(1).rolling(params['52w_lookback']).sum()
                    cond = cond & (prev == 0)
                conditions.append(cond)
            
            if params['use_vol']:
                conditions.append(df['vol_ratio'] > params['vol_thresh'])

            if params['use_vol_rank']:
                cond = df['vol_ratio_10d_rank'] < params['vol_rank_thresh'] if params['vol_rank_logic'] == '<' else df['vol_ratio_10d_rank'] > params['vol_rank_thresh']
                conditions.append(cond)

            if not conditions: continue
            
            final_signal = conditions[0]
            for c in conditions[1:]: final_signal = final_signal & c
            signal_dates = df.index[final_signal]
            
            # --- TRADE EXECUTION ---
            for signal_date in signal_dates:
                if max_one_pos and signal_date <= last_exit_date: continue
                sig_idx = df.index.get_loc(signal_date)
                fixed_exit_idx = sig_idx + params['holding_days']
                if fixed_exit_idx >= len(df): continue

                found_entry = False
                actual_entry_idx = -1
                actual_entry_price = 0.0
                
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
                                if row['Close'] < cutoff: continue 
                            found_entry = True
                            actual_entry_idx = curr_check_idx
                            actual_entry_price = ma_val if pullback_use_level else row['Close']
                            break
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
                                    exit_type = "Stop"; exit_date = f_date; break
                                if f_row['High'] >= tgt_price:
                                    exit_price = f_row['Open'] if f_row['Open'] > tgt_price else tgt_price
                                    exit_type = "Target"; exit_date = f_date; break
                            else: # Short
                                if f_row['High'] >= stop_price:
                                    exit_price = f_row['Open'] if f_row['Open'] > stop_price else stop_price
                                    exit_type = "Stop"; exit_date = f_date; break
                                if f_row['Low'] <= tgt_price:
                                    exit_price = f_row['Open'] if f_row['Open'] < tgt_price else tgt_price
                                    exit_type = "Target"; exit_date = f_date; break
                    
                    if exit_type == "Hold":
                        exit_price = future['Close'].iloc[-1]
                        exit_date = future.index[-1]
                        exit_type = "Time"
                    
                    last_exit_date = exit_date
                    risk_unit = actual_entry_price - stop_price if direction == 'Long' else stop_price - actual_entry_price
                    pnl = exit_price - actual_entry_price if direction == 'Long' else actual_entry_price - exit_price
                    if risk_unit <= 0: risk_unit = 0.001
                    r = pnl / risk_unit
                    
                    trades.append({
                        "Ticker": ticker, "SignalDate": signal_date, "Direction": direction,
                        "Entry": actual_entry_price, "Exit": exit_price, "ExitDate": exit_date,
                        "Type": exit_type, "R": r, "Age": df['age_years'].iloc[sig_idx], "AvgVol": df['vol_ma'].iloc[sig_idx]
                    })
        except: continue
        
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(trades)

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
    with r_c1:
        trade_direction = st.selectbox("Trade Direction", ["Long", "Short"])
    with r_c2:
        time_exit_only = st.checkbox("Time Exit Only (Disable Stop/Target)")
    with r_c3:
        max_one_pos = st.checkbox("Max 1 Position/Ticker", value=True, help="If checked, allows only one open trade at a time per ticker.")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: 
        entry_type = st.selectbox("Entry Price", [
            "Signal Close", "T+1 Open", "T+1 Close",
            "Pullback 10 SMA (Entry: Close)", "Pullback 10 SMA (Entry: Level)",
            "Pullback 21 EMA (Entry: Close)", "Pullback 21 EMA (Entry: Level)"
        ])
        use_ma_entry_filter = st.checkbox("Filter: Close > MA - 0.25*ATR", value=False) if "Pullback" in entry_type else False

    with c2: stop_atr = st.number_input("Stop Loss (ATR)", value=3.0, step=0.1)
    with c3: tgt_atr = st.number_input("Target (ATR)", value=8.0, step=0.1, disabled=time_exit_only)
    with c4: hold_days = st.number_input("Max Holding Days", value=10, step=1)
    with c5: risk_per_trade = st.number_input("Risk Amount ($)", value=1000, step=100)

    st.markdown("---")
    st.subheader("3. Signal Criteria")

    with st.expander("Liquidity & Data History Filters", expanded=True):
        l1, l2, l3, l4 = st.columns(4)
        with l1: min_price = st.number_input("Min Price ($)", value=10.0, step=1.0)
        with l2: min_vol = st.number_input("Min Avg Volume", value=100000, step=50000)
        with l3: min_age = st.number_input("Min True Age (Yrs)", value=0.25, step=0.25)
        with l4: max_age = st.number_input("Max True Age (Yrs)", value=100.0, step=1.0)
    
    with st.expander("Price Action", expanded=True):
        req_green_candle = st.checkbox("Require Close > Open (Green Candle)", value=False)
        
    with st.expander("Trend Filter", expanded=True):
        t1, _ = st.columns([1, 3])
        with t1:
            trend_filter = st.selectbox("Trend Condition", 
                ["None", "Price > 200 SMA", "Price > Rising 200 SMA", "SPY > 200 SMA",
                 "Price < 200 SMA", "Price < Falling 200 SMA", "SPY < 200 SMA"],
                help="Requires 200 days of data. 'SPY' filters check the broad market regime.")

    with st.expander("Performance Percentile Rank", expanded=False):
        use_perf = st.checkbox("Enable Performance Filter", value=False)
        p1, p2, p3, p4, p5 = st.columns(5)
        with p1: perf_window = st.selectbox("Window", [5, 10, 21], disabled=not use_perf)
        with p2: perf_logic = st.selectbox("Logic", ["<", ">"], disabled=not use_perf)
        with p3: perf_thresh = st.number_input("Threshold (%)", 0.0, 100.0, 15.0, disabled=not use_perf)
        with p4: 
            perf_first = st.checkbox("First Instance Only", value=True, disabled=not use_perf)
            perf_consecutive = st.number_input("Min Consecutive Days", 1, 20, 1, disabled=not use_perf)
        with p5: perf_lookback = st.number_input("Instance Lookback (Days)", 1, 100, 21, disabled=not use_perf)

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
        
        spy_series = None
        if "SPY" in trend_filter:
            if "SPY" in data_dict: spy_df = data_dict["SPY"]
            else:
                st.info("Fetching SPY data for regime filter...")
                spy_dict_temp = download_universe_data(["SPY"], fetch_start)
                spy_df = spy_dict_temp.get("SPY", None)
            if spy_df is not None and not spy_df.empty:
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
            'min_price': min_price, 'min_vol': min_vol, 'min_age': min_age, 'max_age': max_age,
            'trend_filter': trend_filter,
            'use_perf_rank': use_perf, 'perf_window': perf_window, 'perf_logic': perf_logic, 
            'perf_thresh': perf_thresh, 'perf_first_instance': perf_first, 'perf_lookback': perf_lookback,
            'perf_consecutive': perf_consecutive,
            'use_sznl': use_sznl, 'sznl_logic': sznl_logic, 'sznl_thresh': sznl_thresh, 
            'sznl_first_instance': sznl_first, 'sznl_lookback': sznl_lookback,
            'use_52w': use_52w, '52w_type': type_52w, '52w_first_instance': first_52w, '52w_lookback': lookback_52w,
            'use_vol': use_vol, 'vol_thresh': vol_thresh,
            'use_vol_rank': use_vol_rank, 'vol_rank_logic': vol_rank_logic, 'vol_rank_thresh': vol_rank_thresh
        }
        
        trades_df = run_engine(data_dict, params, sznl_map, spy_series)
        if trades_df.empty:
            st.warning("No signals generated.")
            return

        trades_df = trades_df.sort_values("ExitDate")
        trades_df['PnL_Dollar'] = trades_df['R'] * risk_per_trade
        trades_df['CumPnL'] = trades_df['PnL_Dollar'].cumsum()
        trades_df['SignalDate'] = pd.to_datetime(trades_df['SignalDate'])
        trades_df['Year'] = trades_df['SignalDate'].dt.year
        trades_df['Month'] = trades_df['SignalDate'].dt.strftime('%b')
        trades_df['MonthNum'] = trades_df['SignalDate'].dt.month
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
        </div>
        """, unsafe_allow_html=True)
        
        if notes: st.warning("Notes: " + ", ".join(notes))

        fig = px.line(trades_df, x="ExitDate", y="CumPnL", title=f"Cumulative Equity (Risk: ${risk_per_trade}/trade)", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Performance Breakdowns")
        b1, b2 = st.columns(2)
        b1.plotly_chart(px.bar(trades_df.groupby('Year')['PnL_Dollar'].sum().reset_index(), x='Year', y='PnL_Dollar', title="PnL by Year", text_auto='.2s'), use_container_width=True)
        b2.plotly_chart(px.bar(trades_df.groupby('CyclePhase')['PnL_Dollar'].sum().reset_index().sort_values('CyclePhase'), x='CyclePhase', y='PnL_Dollar', title="PnL by Cycle", text_auto='.2s'), use_container_width=True)
        
        b3, b4 = st.columns(2)
        ticker_pnl = trades_df.groupby("Ticker")["PnL_Dollar"].sum().reset_index().sort_values("PnL_Dollar", ascending=False).head(75)
        b3.plotly_chart(px.bar(ticker_pnl, x="Ticker", y="PnL_Dollar", title="Cumulative PnL by Ticker (Top 75)", text_auto='.2s'), use_container_width=True)
        
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_pnl = trades_df.groupby("Month")["PnL_Dollar"].sum().reindex(month_order).reset_index()
        b4.plotly_chart(px.bar(monthly_pnl, x="Month", y="PnL_Dollar", title="Cumulative PnL by Month (Seasonality)", text_auto='.2s'), use_container_width=True)

        st.subheader("Trade Log")
        st.dataframe(trades_df.style.format({
            "Entry": "{:.2f}", "Exit": "{:.2f}", "R": "{:.2f}", "PnL_Dollar": "${:,.2f}",
            "Age": "{:.1f}y", "AvgVol": "{:,.0f}"
        }), use_container_width=True)

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
        "use_perf_rank": {use_perf}, "perf_window": {perf_window}, "perf_logic": "{perf_logic}", "perf_thresh": {perf_thresh},
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
