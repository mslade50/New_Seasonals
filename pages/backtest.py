import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import datetime
import random
import time

# -----------------------------------------------------------------------------
# CONFIG / CONSTANTS
# -----------------------------------------------------------------------------
SECTOR_ETFS = [
    "IBB", "IHI", "ITA", "ITB", "IYR", "KRE", "OIH", "SMH", "VNQ",
    "XBI", "XHB", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP",
    "XLU", "XLV", "XLY", "XME", "XOP", "XRT", "GLD", "CEF", "SLV",
]
INDEX_ETFS = ["SPY", "QQQ", "IWM", "DIA", "SMH"]
CSV_PATH = "seasonal_ranks.csv"

# -----------------------------------------------------------------------------
# DATA UTILS
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_seasonal_map():
    """Loads CSV for seasonality lookups."""
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

def get_sznl_val_series(ticker, dates, sznl_map):
    """Vectorized seasonal lookup."""
    t_map = sznl_map.get(ticker, {})
    if not t_map:
        return pd.Series(50.0, index=dates)
    
    mds = dates.map(lambda x: (x.month, x.day))
    return mds.map(t_map).fillna(50.0)

@st.cache_data(show_spinner=True)
def download_universe_data(tickers, fetch_start_date):
    """
    Downloads data in chunks using a specific Fetch Date.
    If user wants 'True Age', fetch_start_date will be 1950.
    If user wants 'Speed', fetch_start_date will be their Backtest Start.
    """
    if not tickers:
        return {} 
    
    # Sanitize
    clean_tickers = [str(t).strip().upper() for t in tickers if str(t).strip() != '']
    if not clean_tickers:
        return {}

    # Format Date
    if isinstance(fetch_start_date, datetime.date):
        start_str = fetch_start_date.strftime('%Y-%m-%d')
    else:
        start_str = fetch_start_date # Assume string if passed as string

    data_dict = {}
    
    # Chunking Config
    CHUNK_SIZE = 50 
    total_tickers = len(clean_tickers)
    
    # Progress bar for downloads
    download_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, total_tickers, CHUNK_SIZE):
        chunk = clean_tickers[i:i + CHUNK_SIZE]
        
        status_text.text(f"Downloading batch {i} to {min(i+CHUNK_SIZE, total_tickers)} of {total_tickers}...")
        download_bar.progress(min((i + CHUNK_SIZE) / total_tickers, 1.0))
        
        try:
            # threads=True helps, but we add a tiny sleep to be nice to API
            df = yf.download(chunk, start=start_str, group_by='ticker', auto_adjust=True, progress=False, threads=True)
            
            if df.empty:
                continue

            # PROCESS CHUNK
            if len(chunk) == 1:
                t = chunk[0]
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
                if not df.empty:
                    data_dict[t] = df
            else:
                # Multi-ticker batch
                for t in chunk:
                    try:
                        if t not in df.columns.levels[0]:
                            continue
                        
                        t_df = df[t].copy()
                        col_to_check = 'Close' if 'Close' in t_df.columns else 'close'
                        if col_to_check in t_df.columns:
                            t_df = t_df.dropna(subset=[col_to_check])
                        
                        if not t_df.empty:
                            data_dict[t] = t_df
                    except:
                        continue
            
            time.sleep(0.1)

        except Exception as e:
            err_msg = str(e).lower()
            if "rate limited" in err_msg or "too many requests" in err_msg:
                st.warning(f"⚠️ Rate limit hit at ticker #{i}. Stopping download. Running backtest on the {len(data_dict)} tickers collected so far.")
                break
            else:
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
# BACKTEST LOGIC
# -----------------------------------------------------------------------------

def calculate_indicators(df, sznl_map, ticker):
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    
    # Fix Timezones
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # 1. Returns & Expanding Ranks
    for window in [5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window)
        df[f'rank_ret_{window}d'] = df[f'ret_{window}d'].expanding(min_periods=252).rank(pct=True) * 100.0
        
    # 2. ATR(14)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # 3. Seasonality
    df['Sznl'] = get_sznl_val_series(ticker, df.index, sznl_map)
    
    # 4. 52w High / Low
    rolling_high = df['High'].shift(1).rolling(252).max()
    rolling_low = df['Low'].shift(1).rolling(252).min()
    df['is_52w_high'] = df['High'] > rolling_high
    df['is_52w_low'] = df['Low'] < rolling_low
    
    # 5. Volume (63d MA)
    vol_ma = df['Volume'].rolling(63).mean()
    df['vol_ma'] = vol_ma
    df['vol_ratio'] = df['Volume'] / vol_ma
    
    # 6. True Age Calculation
    # Uses the very first index in the downloaded data (which might be 1980)
    if not df.empty:
        start_ts = df.index[0]
        df['age_years'] = (df.index - start_ts).days / 365.25
    else:
        df['age_years'] = 0.0
    
    return df

def run_engine(universe_dict, params, sznl_map):
    trades = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(universe_dict)
    
    # Convert user backtest start date to timestamp for slicing
    bt_start_ts = pd.to_datetime(params['backtest_start_date'])

    if total == 0:
        return pd.DataFrame()

    for i, (ticker, df_raw) in enumerate(universe_dict.items()):
        status_text.text(f"Processing {ticker}...")
        progress_bar.progress((i+1)/total)
        
        if len(df_raw) < 100: continue
        
        try:
            # 1. Calculate Indicators on FULL History (for accurate Age/MA)
            df = calculate_indicators(df_raw, sznl_map, ticker)
            
            # 2. SLICE to Backtest Period
            # We only trade AFTER the user's selected start date
            df = df[df.index >= bt_start_ts]
            
            if df.empty: continue
            
            # --- SIGNAL LOGIC ---
            conditions = []
            
            # Basic Filters
            current_age = df['age_years'].fillna(0)
            current_vol = df['vol_ma'].fillna(0)
            current_close = df['Close'].fillna(0)
            
            gate_cond = (current_close >= params['min_price']) & \
                        (current_vol >= params['min_vol']) & \
                        (current_age >= params['min_age']) & \
                        (current_age <= params['max_age'])
                        
            conditions.append(gate_cond)
            
            # Perf Rank
            if params['use_perf_rank']:
                col = f"rank_ret_{params['perf_window']}d"
                if params['perf_logic'] == '<':
                    cond = df[col] < params['perf_thresh']
                else:
                    cond = df[col] > params['perf_thresh']
                if params['perf_first_instance']:
                    prev = cond.shift(1).rolling(params['perf_lookback']).sum()
                    cond = cond & (prev == 0)
                conditions.append(cond)

            # Seasonal
            if params['use_sznl']:
                if params['sznl_logic'] == '<':
                    cond = df['Sznl'] < params['sznl_thresh']
                else:
                    cond = df['Sznl'] > params['sznl_thresh']
                if params['sznl_first_instance']:
                    prev = cond.shift(1).rolling(params['sznl_lookback']).sum()
                    cond = cond & (prev == 0)
                conditions.append(cond)
                
            # 52w High/Low
            if params['use_52w']:
                if params['52w_type'] == 'New 52w High':
                    cond = df['is_52w_high']
                else:
                    cond = df['is_52w_low']
                if params['52w_first_instance']:
                    prev = cond.shift(1).rolling(params['52w_lookback']).sum()
                    cond = cond & (prev == 0)
                conditions.append(cond)
                
            # Volume Spike
            if params['use_vol']:
                cond = df['vol_ratio'] > params['vol_thresh']
                conditions.append(cond)
                
            if not conditions: continue
                
            # Combine conditions
            final_signal = conditions[0]
            for c in conditions[1:]:
                final_signal = final_signal & c
                
            signal_dates = df.index[final_signal]
            
            # --- TRADE MANAGEMENT ---
            for signal_date in signal_dates:
                try:
                    sig_idx = df.index.get_loc(signal_date)
                    
                    if sig_idx + params['holding_days'] + 2 >= len(df): continue
                    
                    atr = df['ATR'].iloc[sig_idx]
                    
                    # Snapshots
                    snap_age = df['age_years'].iloc[sig_idx]
                    snap_vol = df['vol_ma'].iloc[sig_idx]

                    if np.isnan(atr) or atr == 0: continue
                    
                    # Entry
                    if params['entry_type'] == 'Signal Close':
                        entry_price = df['Close'].iloc[sig_idx]
                        start_idx = sig_idx + 1
                    elif params['entry_type'] == 'T+1 Open':
                        entry_price = df['Open'].iloc[sig_idx + 1]
                        start_idx = sig_idx + 1
                    else: # T+1 Close
                        entry_price = df['Close'].iloc[sig_idx + 1]
                        start_idx = sig_idx + 2
                    
                    # Risk Management 
                    stop_price = entry_price - (atr * params['stop_atr'])
                    tgt_price = entry_price + (atr * params['tgt_atr'])
                    
                    exit_price = entry_price
                    exit_type = "Hold"
                    exit_date = None
                    
                    future = df.iloc[start_idx : start_idx + params['holding_days']]
                    
                    # Exit Logic
                    if not params['time_exit_only']:
                        for f_date, f_row in future.iterrows():
                            # Hit Stop?
                            if f_row['Low'] <= stop_price:
                                exit_price = f_row['Open'] if f_row['Open'] < stop_price else stop_price
                                exit_type = "Stop"
                                exit_date = f_date
                                break
                            # Hit Target?
                            if f_row['High'] >= tgt_price:
                                exit_price = f_row['Open'] if f_row['Open'] > tgt_price else tgt_price
                                exit_type = "Target"
                                exit_date = f_date
                                break
                    
                    if exit_type == "Hold":
                        exit_price = future['Close'].iloc[-1]
                        exit_date = future.index[-1]
                        exit_type = "Time"
                    
                    # R Calculation
                    risk_unit = entry_price - stop_price
                    if risk_unit <= 0: risk_unit = 0.001 
                    pnl = exit_price - entry_price
                    r_realized = pnl / risk_unit
                    
                    trades.append({
                        "Ticker": ticker,
                        "SignalDate": signal_date,
                        "Entry": entry_price,
                        "Exit": exit_price,
                        "ExitDate": exit_date,
                        "Type": exit_type,
                        "R": r_realized,
                        "Age": snap_age,
                        "AvgVol": snap_vol
                    })
                except: continue
        except:
            continue

    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(trades)

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="Quantitative Backtester")
    st.title("Quantitative Strategy Backtester")
    st.markdown("---")

    # --------------------------------
    # 1. UNIVERSE & DATA
    # --------------------------------
    st.subheader("1. Universe & Data")
    col_u1, col_u2, col_u3 = st.columns([1, 1, 2])
    
    sample_pct = 100 # Default
    use_full_history = False
    
    with col_u1:
        univ_choice = st.selectbox("Choose Universe", 
            ["Sector ETFs", "Sector + Index ETFs", "All CSV Tickers", "Custom (Upload CSV)"])
            
    with col_u2:
        default_start = datetime.date.today() - datetime.timedelta(days=365*5)
        start_date = st.date_input("Backtest Start Date", value=default_start)
        
        # NEW: Option for Accurate Age
        use_full_history = st.checkbox("Download Full History (Accurate Age)", value=False,
            help="If checked, downloads data from 1950 to calculate 'True Age', then cuts to your Backtest Start Date. Slower but more accurate.")
    
    custom_tickers = []
    if univ_choice == "Custom (Upload CSV)":
        with col_u3:
            sample_pct = st.slider("Random Sample % (Run on subset)", 1, 100, 100, 
                                  help="Randomly selects this % of tickers from the file.")
            
            uploaded_file = st.file_uploader("Upload CSV (Must have 'Ticker' header)", type=["csv"])
            if uploaded_file:
                try:
                    c_df = pd.read_csv(uploaded_file)
                    if "Ticker" in c_df.columns:
                        # --- SANITIZE INPUT ---
                        c_df["Ticker"] = c_df["Ticker"].astype(str).str.strip().str.upper()
                        c_df = c_df[~c_df["Ticker"].isin(["NAN", "NONE", "NULL", ""])]
                        
                        custom_tickers = c_df["Ticker"].unique().tolist()
                        
                        if len(custom_tickers) > 0:
                            st.success(f"Loaded {len(custom_tickers)} valid tickers. Will run on {int(len(custom_tickers)*(sample_pct/100))} randomly.")
                        else:
                            st.error("CSV found but contained no valid tickers after cleaning.")
                    else:
                        st.error("CSV missing 'Ticker' column.")
                except Exception as e:
                    st.error(f"Invalid CSV: {e}")

    st.markdown("---")

    # --------------------------------
    # 2. EXECUTION & RISK
    # --------------------------------
    st.subheader("2. Execution & Risk Management")
    
    time_exit_only = st.checkbox("Time Exit Only (Disable Stop/Target)", 
                                 help="If checked, trades only exit after 'Max Holding Days'.")

    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        entry_type = st.selectbox("Entry Price", ["Signal Close", "T+1 Open", "T+1 Close"])
    with c2:
        stop_atr = st.number_input("Stop Loss (ATR Multiple)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    with c3:
        tgt_atr = st.number_input("Target (ATR Multiple)", min_value=0.1, max_value=20.0, value=4.0, step=0.1, disabled=time_exit_only)
    with c4:
        hold_days = st.number_input("Max Holding Days", min_value=1, max_value=365, value=10, step=1)
    with c5:
        risk_per_trade = st.number_input("Risk Amount ($)", min_value=1, value=1000, step=100)

    st.markdown("---")

    # --------------------------------
    # 3. SIGNAL CRITERIA
    # --------------------------------
    st.subheader("3. Signal Criteria")

    # A. Liquidity & Age Filters
    with st.expander("Liquidity & Data History Filters (Defaults = Disabled)", expanded=True):
        l1, l2, l3, l4 = st.columns(4)
        with l1:
            min_price = st.number_input("Min Price ($)", value=0.0, step=1.0, help="Set to 0 to disable.")
        with l2:
            min_vol = st.number_input("Min Avg Volume", value=0, step=50000, help="Set to 0 to disable.")
        with l3:
            min_age = st.number_input("Min True Age (Yrs)", value=0.0, step=0.5, help="Uses Full History if enabled above. Otherwise uses Data Start Date.")
        with l4:
            max_age = st.number_input("Max True Age (Yrs)", value=100.0, step=1.0)

    # B. Performance Rank
    with st.expander("Performance Percentile Rank", expanded=False):
        use_perf = st.checkbox("Enable Performance Filter", value=True)
        p1, p2, p3, p4, p5 = st.columns(5)
        with p1: perf_window = st.selectbox("Window", [5, 10, 21], disabled=not use_perf)
        with p2: perf_logic = st.selectbox("Logic", ["<", ">"], disabled=not use_perf)
        with p3: perf_thresh = st.number_input("Threshold (%)", 0.0, 100.0, 15.0, 0.5, disabled=not use_perf)
        with p4: perf_first = st.checkbox("First Instance Only", value=True, disabled=not use_perf)
        with p5: perf_lookback = st.number_input("Instance Lookback (Days)", 1, 100, 21, disabled=not use_perf)

    # C. Seasonal Rank
    with st.expander("Seasonal Rank", expanded=False):
        use_sznl = st.checkbox("Enable Seasonal Filter", value=False)
        s1, s2, s3, s4 = st.columns(4)
        with s1: sznl_logic = st.selectbox("Seasonality", ["<", ">"], key="sl", disabled=not use_sznl)
        with s2: sznl_thresh = st.number_input("Seasonal Rank Threshold", 0.0, 100.0, 15.0, 0.5, key="st", disabled=not use_sznl)
        with s3: sznl_first = st.checkbox("First Instance Only", value=True, key="sf", disabled=not use_sznl)
        with s4: sznl_lookback = st.number_input("Instance Lookback (Days)", 1, 100, 21, key="slb", disabled=not use_sznl)

    # D. 52-Week
    with st.expander("52-Week High/Low", expanded=False):
        use_52w = st.checkbox("Enable 52w High/Low Filter", value=False)
        h1, h2, h3 = st.columns(3)
        with h1: type_52w = st.selectbox("Condition", ["New 52w High", "New 52w Low"], disabled=not use_52w)
        with h2: first_52w = st.checkbox("First Instance Only", value=True, key="hf", disabled=not use_52w)
        with h3: lookback_52w = st.number_input("Instance Lookback (Days)", 1, 252, 21, key="hlb", disabled=not use_52w)

    # E. Volume Spike
    with st.expander("Volume Spike", expanded=False):
        use_vol = st.checkbox("Enable Volume Spike Filter", value=False)
        v1, _ = st.columns([1, 3])
        with v1: vol_thresh = st.number_input("Vol Multiple (> X * 63d Avg)", 1.0, 10.0, 1.5, 0.1, disabled=not use_vol)

    st.markdown("---")
    
    # --------------------------------
    # 4. EXECUTION
    # --------------------------------
    if st.button("Run Backtest", type="primary", use_container_width=True):
        
        tickers_to_run = []
        sznl_map = load_seasonal_map()
        
        if univ_choice == "Sector ETFs":
            tickers_to_run = SECTOR_ETFS
        elif univ_choice == "Sector + Index ETFs":
            tickers_to_run = list(set(SECTOR_ETFS + INDEX_ETFS))
        elif univ_choice == "All CSV Tickers":
            raw_keys = list(sznl_map.keys())
            exclude_list = ["BTC-USD", "ETH-USD"]
            tickers_to_run = [t for t in raw_keys if t not in exclude_list]
        elif univ_choice == "Custom (Upload CSV)":
            tickers_to_run = custom_tickers
            if tickers_to_run and sample_pct < 100:
                count = max(1, int(len(tickers_to_run) * (sample_pct / 100)))
                tickers_to_run = random.sample(tickers_to_run, count)
                st.info(f"Randomly selected {len(tickers_to_run)} tickers for this run.")
            
        if not tickers_to_run:
            st.error("No tickers found.")
            return
        
        # DETERMINE DOWNLOAD START DATE
        # If Full History is checked, we grab from 1950.
        # Otherwise we grab from the user's Backtest Start Date.
        fetch_start = "1950-01-01" if use_full_history else start_date
        
        msg = "Downloading FULL history (1950+) for Accurate Age..." if use_full_history else f"Downloading data from {start_date}..."
        st.info(f"{msg} ({len(tickers_to_run)} tickers)")
        
        data_dict = download_universe_data(tickers_to_run, fetch_start)
        
        if not data_dict:
            st.error("Failed to download data, or no valid data found for these tickers.")
            return
            
        params = {
            'backtest_start_date': start_date, # Pass this so we know where to SLICE
            'time_exit_only': time_exit_only,
            'stop_atr': stop_atr, 'tgt_atr': tgt_atr, 'holding_days': hold_days, 'entry_type': entry_type,
            'min_price': min_price, 'min_vol': min_vol, 'min_age': min_age, 'max_age': max_age,
            'use_perf_rank': use_perf, 'perf_window': perf_window, 'perf_logic': perf_logic, 
            'perf_thresh': perf_thresh, 'perf_first_instance': perf_first, 'perf_lookback': perf_lookback,
            'use_sznl': use_sznl, 'sznl_logic': sznl_logic, 'sznl_thresh': sznl_thresh, 
            'sznl_first_instance': sznl_first, 'sznl_lookback': sznl_lookback,
            'use_52w': use_52w, '52w_type': type_52w, '52w_first_instance': first_52w, '52w_lookback': lookback_52w,
            'use_vol': use_vol, 'vol_thresh': vol_thresh
        }
        
        trades_df = run_engine(data_dict, params, sznl_map)
        
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
        trades_df['DayOfWeek'] = trades_df['SignalDate'].dt.day_name()
        trades_df['CyclePhase'] = trades_df['Year'].apply(get_cycle_year)
        
        # --- NEW BUCKETS ---
        trades_df['AgeBucket'] = trades_df['Age'].apply(get_age_bucket)
        age_order = ["< 3 Years", "3-5 Years", "5-10 Years", "10-20 Years", "> 20 Years"]
        
        if len(trades_df) >= 10:
            try:
                trades_df['VolDecile'] = pd.qcut(trades_df['AvgVol'], 10, labels=False, duplicates='drop') + 1
            except:
                trades_df['VolDecile'] = 1
        else:
            trades_df['VolDecile'] = 1

        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

        # Stats
        wins = trades_df[trades_df['R'] > 0]
        losses = trades_df[trades_df['R'] <= 0]
        
        win_rate = len(wins) / len(trades_df) * 100
        avg_pnl = trades_df['PnL_Dollar'].mean()
        
        avg_win_val = wins['PnL_Dollar'].mean() if not wins.empty else 0
        avg_loss_val = losses['PnL_Dollar'].mean() if not losses.empty else 0
        
        cum_pnl = trades_df['CumPnL'].values
        running_max = np.maximum.accumulate(cum_pnl)
        dd = running_max - cum_pnl
        max_dd = dd.max()
        
        st.success("Backtest Complete!")
        
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Trades", len(trades_df))
        k2.metric("Win Rate", f"{win_rate:.1f}%")
        k3.metric("Avg PnL", f"${avg_pnl:,.0f}")
        k4.metric("Avg Win", f"${avg_win_val:,.0f}")
        k5.metric("Avg Loss", f"${avg_loss_val:,.0f}")
        
        fig = px.line(trades_df, x="ExitDate", y="CumPnL", title=f"Cumulative Equity (Risk: ${risk_per_trade}/trade)", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # --- BREAKDOWN CHARTS ---
        st.subheader("Performance Breakdowns")
        
        b1, b2 = st.columns(2)
        df_year = trades_df.groupby('Year')['PnL_Dollar'].sum().reset_index()
        fig_year = px.bar(df_year, x='Year', y='PnL_Dollar', title="PnL by Year (Trade Start)", text_auto='.2s')
        b1.plotly_chart(fig_year, use_container_width=True)
        
        df_cycle = trades_df.groupby('CyclePhase')['PnL_Dollar'].sum().reset_index().sort_values('CyclePhase')
        fig_cycle = px.bar(df_cycle, x='CyclePhase', y='PnL_Dollar', title="PnL by Election Cycle", text_auto='.2s')
        b2.plotly_chart(fig_cycle, use_container_width=True)
        
        b3, b4 = st.columns(2)
        df_month = trades_df.groupby('Month')['PnL_Dollar'].sum().reindex(month_order).reset_index()
        fig_month = px.bar(df_month, x='Month', y='PnL_Dollar', title="PnL by Month (Seasonality)", text_auto='.2s')
        b3.plotly_chart(fig_month, use_container_width=True)
        
        df_day = trades_df.groupby('DayOfWeek')['PnL_Dollar'].sum().reindex(day_order).reset_index()
        fig_day = px.bar(df_day, x='DayOfWeek', y='PnL_Dollar', title="PnL by Day of Week", text_auto='.2s')
        b4.plotly_chart(fig_day, use_container_width=True)
        
        # --- NEW ROW: Age & Volume ---
        b5, b6 = st.columns(2)
        
        df_age = trades_df.groupby('AgeBucket')['PnL_Dollar'].sum().reindex(age_order).reset_index()
        fig_age = px.bar(df_age, x='AgeBucket', y='PnL_Dollar', title="PnL by History Length", text_auto='.2s')
        b5.plotly_chart(fig_age, use_container_width=True)
        
        df_vol = trades_df.groupby('VolDecile')['PnL_Dollar'].sum().reset_index()
        fig_vol = px.bar(df_vol, x='VolDecile', y='PnL_Dollar', title="PnL by Avg Volume Decile (1=Low, 10=High)", text_auto='.2s')
        fig_vol.update_xaxes(tickmode='linear', dtick=1)
        b6.plotly_chart(fig_vol, use_container_width=True)

        st.subheader("Trade Log")
        st.dataframe(trades_df.style.format({
            "Entry": "{:.2f}", "Exit": "{:.2f}", "R": "{:.2f}", "PnL_Dollar": "${:,.2f}",
            "Age": "{:.1f}y", "AvgVol": "{:,.0f}"
        }), use_container_width=True)

        st.markdown("---")
        st.subheader("Configuration & Results Report")
        
        history_mode = "Full History (1950+)" if use_full_history else f"Since {start_date}"
        
        report_text = f"""
        --- BACKTEST CONFIGURATION ---
        Universe: {univ_choice} (Sample: {sample_pct}%)
        Data Source: {history_mode}
        Backtest Start: {start_date}
        Risk Per Trade: ${risk_per_trade}
        
        -- LIQUIDITY FILTERS --
        Min Price: ${min_price}
        Min Avg Vol: {min_vol}
        Min History: {min_age} years
        
        -- EXECUTION --
        Time Exit Only: {time_exit_only}
        Entry: {entry_type}
        Stop Loss: {stop_atr} ATR
        Target: {tgt_atr} ATR
        Max Hold: {hold_days} Days
        
        -- SIGNAL FILTERS --
        Performance Rank: {use_perf} ({perf_window}d {perf_logic} {perf_thresh}%)
        Seasonal Rank: {use_sznl} ({sznl_logic} {sznl_thresh})
        52-Week: {use_52w} ({type_52w})
        Volume Spike: {use_vol} (> {vol_thresh}x)
        
        --- RESULTS ---
        Trades: {len(trades_df)}
        Win Rate: {win_rate:.1f}%
        Total PnL: ${trades_df['PnL_Dollar'].sum():,.2f}
        Avg Trade: ${avg_pnl:,.2f}
        Avg Win: ${avg_win_val:,.2f}
        Avg Loss: ${avg_loss_val:,.2f}
        """
        
        st.text_area("Copy this summary:", value=report_text, height=300)

if __name__ == "__main__":
    main()
