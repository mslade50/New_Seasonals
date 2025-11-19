import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import datetime

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
        return pd.Series(np.nan, index=dates)
    mds = dates.map(lambda x: (x.month, x.day))
    return mds.map(t_map)

@st.cache_data(show_spinner=True)
def download_universe_data(tickers):
    """Downloads OHLCV data for a list of tickers."""
    if not tickers:
        return pd.DataFrame()
    
    # Download in one batch
    df = yf.download(tickers, period="5y", group_by='ticker', auto_adjust=True, progress=False)
    
    data_dict = {}
    if len(tickers) == 1:
        t = tickers[0]
        # Clean columns if single ticker
        df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
        if not df.empty:
            data_dict[t] = df
    else:
        for t in tickers:
            try:
                t_df = df[t].copy()
                if t_df.dropna(how='all').empty: continue
                data_dict[t] = t_df
            except KeyError:
                continue
                
    return data_dict

# -----------------------------------------------------------------------------
# BACKTEST LOGIC
# -----------------------------------------------------------------------------

def calculate_indicators(df, sznl_map, ticker):
    """Calculates indicators including expanding ranks to avoid lookahead bias."""
    df = df.copy()
    
    # 1. Returns & Expanding Ranks
    for window in [5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window)
        # Expanding rank (0.0 to 100.0)
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
    
    # 4. 52w High / Low (Shifted 1 day to use yesterday's high/low reference)
    rolling_high = df['High'].shift(1).rolling(252).max()
    rolling_low = df['Low'].shift(1).rolling(252).min()
    df['is_52w_high'] = df['High'] > rolling_high
    df['is_52w_low'] = df['Low'] < rolling_low
    
    # 5. Volume
    vol_ma = df['Volume'].rolling(63).mean()
    df['vol_ratio'] = df['Volume'] / vol_ma
    
    return df

def run_engine(universe_dict, params, sznl_map):
    trades = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(universe_dict)
    
    for i, (ticker, df_raw) in enumerate(universe_dict.items()):
        status_text.text(f"Processing {ticker}...")
        progress_bar.progress((i+1)/total)
        
        if len(df_raw) < 260: continue
        
        df = calculate_indicators(df_raw, sznl_map, ticker)
        df = df.dropna()
        
        # --- SIGNAL LOGIC ---
        conditions = []
        
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
            
        # Volume
        if params['use_vol']:
            cond = df['vol_ratio'] > params['vol_thresh']
            conditions.append(cond)
            
        if not conditions: continue
            
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
                if np.isnan(atr) or atr == 0: continue
                
                # Determine Entry
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
                
                # Outcome Loop
                exit_price = entry_price
                exit_type = "Hold"
                exit_date = None
                
                future = df.iloc[start_idx : start_idx + params['holding_days']]
                
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
                    "R": r_realized
                })
            except: continue

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
    # 1. UNIVERSE SELECTION
    # --------------------------------
    st.subheader("1. Universe Selection")
    col_u1, col_u2 = st.columns([1, 2])
    
    with col_u1:
        univ_choice = st.selectbox("Choose Universe", 
            ["Sector ETFs", "Sector + Index ETFs", "All CSV Tickers", "Custom (Upload CSV)"])
    
    custom_tickers = []
    if univ_choice == "Custom (Upload CSV)":
        with col_u2:
            uploaded_file = st.file_uploader("Upload CSV (Must have 'Ticker' header)", type=["csv"])
            if uploaded_file:
                try:
                    c_df = pd.read_csv(uploaded_file)
                    if "Ticker" in c_df.columns:
                        custom_tickers = c_df["Ticker"].unique().tolist()
                        st.success(f"Loaded {len(custom_tickers)} tickers.")
                    else:
                        st.error("CSV missing 'Ticker' column.")
                except:
                    st.error("Invalid CSV.")

    st.markdown("---")

    # --------------------------------
    # 2. TRADE MANAGEMENT
    # --------------------------------
    st.subheader("2. Execution & Risk Management")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        entry_type = st.selectbox("Entry Price", ["Signal Close", "T+1 Open", "T+1 Close"])
    with c2:
        stop_atr = st.number_input("Stop Loss (ATR Multiple)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    with c3:
        tgt_atr = st.number_input("Target (ATR Multiple)", min_value=0.1, max_value=20.0, value=4.0, step=0.1)
    with c4:
        hold_days = st.number_input("Max Holding Days", min_value=1, max_value=365, value=10, step=1)

    st.markdown("---")

    # --------------------------------
    # 3. SIGNAL CRITERIA
    # --------------------------------
    st.subheader("3. Signal Criteria")

    # A. Performance Rank
    with st.expander("Performance Percentile Rank", expanded=True):
        use_perf = st.checkbox("Enable Performance Filter", value=True)
        p1, p2, p3, p4, p5 = st.columns(5)
        with p1:
            perf_window = st.selectbox("Window", [5, 10, 21], disabled=not use_perf)
        with p2:
            perf_logic = st.selectbox("Logic", ["<", ">"], disabled=not use_perf)
        with p3:
            perf_thresh = st.number_input("Threshold (%)", 0.0, 100.0, 15.0, 0.5, disabled=not use_perf)
        with p4:
            perf_first = st.checkbox("First Instance Only", value=True, disabled=not use_perf)
        with p5:
            perf_lookback = st.number_input("Instance Lookback (Days)", 1, 100, 21, disabled=not use_perf)

    # B. Seasonal Rank
    with st.expander("Seasonal Rank", expanded=True):
        use_sznl = st.checkbox("Enable Seasonal Filter", value=False)
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            sznl_logic = st.selectbox("Seasonality", ["<", ">"], key="sl", disabled=not use_sznl)
        with s2:
            sznl_thresh = st.number_input("Seasonal Rank Threshold", 0.0, 100.0, 15.0, 0.5, key="st", disabled=not use_sznl)
        with s3:
            sznl_first = st.checkbox("First Instance Only", value=True, key="sf", disabled=not use_sznl)
        with s4:
            sznl_lookback = st.number_input("Instance Lookback (Days)", 1, 100, 21, key="slb", disabled=not use_sznl)

    # C. 52-Week Extremes
    with st.expander("52-Week High/Low", expanded=False):
        use_52w = st.checkbox("Enable 52w High/Low Filter", value=False)
        h1, h2, h3 = st.columns(3)
        with h1:
            type_52w = st.selectbox("Condition", ["New 52w High", "New 52w Low"], disabled=not use_52w)
        with h2:
            first_52w = st.checkbox("First Instance Only", value=True, key="hf", disabled=not use_52w)
        with h3:
            lookback_52w = st.number_input("Instance Lookback (Days)", 1, 252, 21, key="hlb", disabled=not use_52w)

    # D. Volume Spike
    with st.expander("Volume Threshold", expanded=False):
        use_vol = st.checkbox("Enable Volume Filter", value=False)
        v1, _ = st.columns([1, 3])
        with v1:
            vol_thresh = st.number_input("Vol Multiple (> X * 63d Avg)", 1.0, 10.0, 1.5, 0.1, disabled=not use_vol)

    st.markdown("---")
    
    # --------------------------------
    # 4. EXECUTION
    # --------------------------------
    if st.button("Run Backtest", type="primary", use_container_width=True):
        
        # 1. Universe Resolution
        tickers_to_run = []
        sznl_map = load_seasonal_map()
        
        if univ_choice == "Sector ETFs":
            tickers_to_run = SECTOR_ETFS
        elif univ_choice == "Sector + Index ETFs":
            tickers_to_run = list(set(SECTOR_ETFS + INDEX_ETFS))
        elif univ_choice == "All CSV Tickers":
            tickers_to_run = list(sznl_map.keys())
        elif univ_choice == "Custom (Upload CSV)":
            tickers_to_run = custom_tickers
            
        if not tickers_to_run:
            st.error("No tickers found. Please check your Universe selection.")
            return
            
        # 2. Download Data
        st.info(f"Fetching data for {len(tickers_to_run)} tickers...")
        data_dict = download_universe_data(tickers_to_run)
        
        if not data_dict:
            st.error("Failed to download data.")
            return
            
        # 3. Pack Params
        params = {
            'stop_atr': stop_atr, 'tgt_atr': tgt_atr, 'holding_days': hold_days, 'entry_type': entry_type,
            'use_perf_rank': use_perf, 'perf_window': perf_window, 'perf_logic': perf_logic, 
            'perf_thresh': perf_thresh, 'perf_first_instance': perf_first, 'perf_lookback': perf_lookback,
            'use_sznl': use_sznl, 'sznl_logic': sznl_logic, 'sznl_thresh': sznl_thresh, 
            'sznl_first_instance': sznl_first, 'sznl_lookback': sznl_lookback,
            'use_52w': use_52w, '52w_type': type_52w, '52w_first_instance': first_52w, '52w_lookback': lookback_52w,
            'use_vol': use_vol, 'vol_thresh': vol_thresh
        }
        
        # 4. Run Engine
        trades_df = run_engine(data_dict, params, sznl_map)
        
        if trades_df.empty:
            st.warning("Backtest complete. No signals generated.")
            return
            
        # 5. Results
        trades_df = trades_df.sort_values("ExitDate")
        trades_df['CumR'] = trades_df['R'].cumsum()
        
        # Stats
        wins = trades_df[trades_df['R'] > 0]
        win_rate = len(wins) / len(trades_df) * 100
        avg_r = trades_df['R'].mean()
        
        # Drawdown
        cum_r = trades_df['CumR'].values
        running_max = np.maximum.accumulate(cum_r)
        dd = running_max - cum_r
        max_dd = dd.max()
        
        st.success("Backtest Complete!")
        
        # KPI Row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Trades", len(trades_df))
        k2.metric("Win Rate", f"{win_rate:.1f}%")
        k3.metric("Avg R / Trade", f"{avg_r:.2f}R")
        k4.metric("Max Drawdown", f"{max_dd:.2f}R")
        
        # Equity Curve
        fig = px.line(trades_df, x="ExitDate", y="CumR", title="Cumulative Equity (R-Multiples)", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data Table
        st.subheader("Trade Log")
        st.dataframe(trades_df.style.format({
            "Entry": "{:.2f}", "Exit": "{:.2f}", "R": "{:.2f}"
        }), use_container_width=True)

if __name__ == "__main__":
    main()
