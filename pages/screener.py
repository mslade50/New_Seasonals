import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from pandas.tseries.offsets import BusinessDay

# -----------------------------------------------------------------------------
# 1. THE STRATEGY BOOK (STATIC BATCH)
# -----------------------------------------------------------------------------
STRATEGY_BOOK = [
    # STRATEGY 1: OVERSOLD INDICES
    {
        "id": "IND_OS_SZNL",
        "name": "Oversold Indices + Bullish Seasonality",
        "description": "Buying major indices when short-term momentum is washed out but seasonal tailwinds are strong.",
        "universe_tickers": ["SPY", "QQQ", "IWM", "DIA", "SMH"], 
        "settings": {
            "trade_direction": "Long",
            "max_one_pos": False,
            "use_perf_rank": True, "perf_window": 5, "perf_logic": "<", "perf_thresh": 15.0,
            "perf_first_instance": True, "perf_lookback": 21, "perf_consecutive": 1,
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 80.0, "sznl_first_instance": True, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": False, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0
        },
        "execution": {
            "risk_per_trade": 1000,
            "stop_atr": 3.0,
            "tgt_atr": 8.0,
            "hold_days": 10
        },
        "stats": {
            "grade": "B (Good)",
            "win_rate": "88.2%",
            "expectancy": "$995.68",
            "profit_factor": "8.38"
        }
    },
    # STRATEGY 2: LARGE CAP MEAN REVERSION (A)
    {
        "id": "STRAT_1763673538",
        "name": "Large Cap Mean Reversion (A)",
        "description": "Universe: Liquid Large Caps. Sznl >80, 5d perf < 15. Filter: SPY > 200 SMA.",
        "universe_tickers": ['AAPL', 'AMGN', 'AMZN', 'AVGO', 'AXP', 'BA', 'CAT', 'CEF', 'CRM', 'CSCO', 'CVX', 'DIA', 'DIS', 'GLD', 'GOOG', 'GS', 'HD', 'HON', 'IBB', 'IBM', 'IHI', 'INTC', 'ITA', 'ITB', 'IWM', 'IYR', 'JNJ', 'JPM', 'KO', 'KRE', 'MCD', 'META', 'MMM', 'MRK', 'MSFT', 'NKE', 'NVDA', 'OIH', 'PG', 'QQQ', 'SLV', 'SMH', 'SPY', 'TRV', 'UNG', 'UNH', 'UVXY', 'V', 'VNQ', 'VZ', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOP', 'XRT'], 
        "settings": {
            "trade_direction": "Long",
            "max_one_pos": True,
            "use_perf_rank": True, "perf_window": 5, "perf_logic": "<", "perf_thresh": 15.0,
            "perf_first_instance": True, "perf_lookback": 3, "perf_consecutive": 1,
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 80.0, "sznl_first_instance": True, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": False, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "SPY > 200 SMA",
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0
        },
        "execution": {
            "risk_per_trade": 1000,
            "stop_atr": 2.0,
            "tgt_atr": 8.0,
            "hold_days": 21
        },
        "stats": {
            "grade": "A (Excellent)",
            "win_rate": "63.9%",
            "expectancy": "$646.42",
            "profit_factor": "2.85"
        }
    },
    # STRATEGY 3: LARGE CAP MEAN REVERSION (B) - HIGHER PRECISION
    {
        "id": "STRAT_1763756935",
        "name": "Large Cap Mean Reversion (B)",
        "description": "Universe: Liquid Large Caps. 21d perf < 15 (5 consec days). Low Rel Vol (<15 rank). No Trend Filter.",
        "universe_tickers": ['AAPL', 'AMGN', 'AMZN', 'AVGO', 'AXP', 'BA', 'CAT', 'CEF', 'CRM', 'CSCO', 'CVX', 'DIA', 'DIS', 'GLD', 'GOOG', 'GS', 'HD', 'HON', 'IBB', 'IBM', 'IHI', 'INTC', 'ITA', 'ITB', 'IWM', 'IYR', 'JNJ', 'JPM', 'KO', 'KRE', 'MCD', 'META', 'MMM', 'MRK', 'MSFT', 'NKE', 'NVDA', 'OIH', 'PG', 'QQQ', 'SLV', 'SMH', 'SPY', 'TRV', 'UNG', 'UNH', 'UVXY', 'V', 'VNQ', 'VZ', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOP', 'XRT'], 
        "settings": {
            "trade_direction": "Long",
            "max_one_pos": True,
            "use_perf_rank": True, "perf_window": 21, "perf_logic": "<", "perf_thresh": 15.0,
            "perf_first_instance": True, "perf_lookback": 21, "perf_consecutive": 5,
            "use_sznl": False, "sznl_logic": "<", "sznl_thresh": 15.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": False, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": True, "vol_rank_logic": "<", "vol_rank_thresh": 15.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0
        },
        "execution": {
            "risk_per_trade": 1000,
            "stop_atr": 2.0,
            "tgt_atr": 8.0,
            "hold_days": 21
        },
        "stats": {
            "grade": "A (Excellent)",
            "win_rate": "73.1%",
            "expectancy": "$726.57",
            "profit_factor": "3.71"
        }
    }
]

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
CSV_PATH = "seasonal_ranks.csv"

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

def get_sznl_val_series(ticker, dates, sznl_map):
    t_map = sznl_map.get(ticker, {})
    if not t_map: return pd.Series(50.0, index=dates)
    mds = dates.map(lambda x: (x.month, x.day))
    return mds.map(t_map).fillna(50.0)

# -----------------------------------------------------------------------------
# ENGINE
# -----------------------------------------------------------------------------

def calculate_indicators(df, sznl_map, ticker, spy_series=None):
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    
    # SMA 200 (Trend)
    df['SMA200'] = df['Close'].rolling(200).mean()

    # Perf Ranks
    for window in [5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window)
        # Min periods 50 ensures we get a rank if we downloaded 400 days
        df[f'rank_ret_{window}d'] = df[f'ret_{window}d'].expanding(min_periods=50).rank(pct=True) * 100.0
        
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
    
    # Volume
    vol_ma = df['Volume'].rolling(63).mean()
    df['vol_ratio'] = df['Volume'] / vol_ma
    df['vol_ma'] = vol_ma
    
    # Volume Rank (10d Relative)
    vol_ma_10 = df['Volume'].rolling(10).mean()
    df['vol_ratio_10d'] = vol_ma_10 / vol_ma
    df['vol_ratio_10d_rank'] = df['vol_ratio_10d'].expanding(min_periods=50).rank(pct=True) * 100.0

    # Age
    if not df.empty:
        start_ts = df.index[0]
        df['age_years'] = (df.index - start_ts).days / 365.25
    else:
        df['age_years'] = 0.0

    # SPY Regime
    if spy_series is not None:
        df['SPY_Above_SMA200'] = spy_series.reindex(df.index, method='ffill').fillna(False)
        
    return df

def check_signal(df, params):
    """
    Checks conditions on the DataFrame. 
    Returns True/False for the LAST ROW only.
    """
    # We calculate conditions on the whole DF to handle 'consecutive' logic
    last_idx = df.index[-1]
    last_row = df.iloc[-1]
    
    # 1. Liquidity Gates (Check last row only)
    if last_row['Close'] < params.get('min_price', 0): return False
    if last_row['vol_ma'] < params.get('min_vol', 0): return False
    if last_row['age_years'] < params.get('min_age', 0): return False
    if last_row['age_years'] > params.get('max_age', 100): return False

    # 2. Trend Filter
    trend_opt = params.get('trend_filter', 'None')
    # Long Logic
    if trend_opt == "Price > 200 SMA":
        if not (last_row['Close'] > last_row['SMA200']): return False
    elif trend_opt == "Price > Rising 200 SMA":
        prev_row = df.iloc[-2]
        if not ((last_row['Close'] > last_row['SMA200']) and (last_row['SMA200'] > prev_row['SMA200'])): return False
    elif trend_opt == "SPY > 200 SMA":
        if 'SPY_Above_SMA200' in df.columns and not last_row['SPY_Above_SMA200']: return False
    # Short Logic
    elif trend_opt == "Price < 200 SMA":
        if not (last_row['Close'] < last_row['SMA200']): return False
    elif trend_opt == "Price < Falling 200 SMA":
        prev_row = df.iloc[-2]
        if not ((last_row['Close'] < last_row['SMA200']) and (last_row['SMA200'] < prev_row['SMA200'])): return False
    elif trend_opt == "SPY < 200 SMA":
        if 'SPY_Above_SMA200' in df.columns and last_row['SPY_Above_SMA200']: return False

    # 3. Perf Rank (with Consecutive logic)
    if params['use_perf_rank']:
        col = f"rank_ret_{params['perf_window']}d"
        # Calc raw condition for whole column
        if params['perf_logic'] == '<': 
            raw_cond = df[col] < params['perf_thresh']
        else: 
            raw_cond = df[col] > params['perf_thresh']
            
        consec = params.get('perf_consecutive', 1)
        if consec > 1:
            # Rolling sum of Trues must equal window size
            persist_cond = raw_cond.rolling(consec).sum() == consec
            if not persist_cond.iloc[-1]: return False
        else:
            if not raw_cond.iloc[-1]: return False

    # 4. Seasonal
    if params['use_sznl']:
        if params['sznl_logic'] == '<':
            if not (last_row['Sznl'] < params['sznl_thresh']): return False
        else:
            if not (last_row['Sznl'] > params['sznl_thresh']): return False

    # 5. 52w
    if params['use_52w']:
        if params['52w_type'] == 'New 52w High':
            if not last_row['is_52w_high']: return False
        else:
            if not last_row['is_52w_low']: return False

    # 6. Volume Spike
    if params['use_vol']:
        if not (last_row['vol_ratio'] > params['vol_thresh']): return False

    # 7. Volume Rank
    if params.get('use_vol_rank'):
        val = last_row['vol_ratio_10d_rank']
        if params['vol_rank_logic'] == '<':
            if not (val < params['vol_rank_thresh']): return False
        else:
            if not (val > params['vol_rank_thresh']): return False
        
    return True

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

# ... (Imports and helper functions remain the same) ...

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
        start_date = st.date_input("Backtest Start Date", value=default_start)
    
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
            help="If checked, allows only one open trade at a time per ticker (prevents pyramiding).")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: entry_type = st.selectbox("Entry Price", ["Signal Close", "T+1 Open", "T+1 Close"])
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
        
    # B. TREND
    with st.expander("Trend Filter", expanded=True):
        t1, _ = st.columns([1, 3])
        with t1:
            trend_filter = st.selectbox("Trend Condition", 
                ["None", "Price > 200 SMA", "Price > Rising 200 SMA", "SPY > 200 SMA",
                 "Price < 200 SMA", "Price < Falling 200 SMA", "SPY < 200 SMA"],
                help="Requires 200 days of data. 'SPY' filters check the broad market regime.")

    # C. STRATEGY FILTERS
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
        
        # Apply sampling
        if tickers_to_run and sample_pct < 100:
            count = max(1, int(len(tickers_to_run) * (sample_pct / 100)))
            tickers_to_run = random.sample(tickers_to_run, count)
            st.info(f"Randomly selected {len(tickers_to_run)} tickers for this run.")
            
        if not tickers_to_run:
            st.error("No tickers found.")
            return

        fetch_start = "1950-01-01" if use_full_history else start_date
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
        
        # 1. Year and Cycle
        b1, b2 = st.columns(2)
        b1.plotly_chart(px.bar(trades_df.groupby('Year')['PnL_Dollar'].sum().reset_index(), x='Year', y='PnL_Dollar', title="PnL by Year", text_auto='.2s'), use_container_width=True)
        b2.plotly_chart(px.bar(trades_df.groupby('CyclePhase')['PnL_Dollar'].sum().reset_index().sort_values('CyclePhase'), x='CyclePhase', y='PnL_Dollar', title="PnL by Cycle", text_auto='.2s'), use_container_width=True)
        
        # 2. Ticker and Month Seasonality (Fixed)
        b3, b4 = st.columns(2)
        
        # Top 75 Tickers
        ticker_pnl = trades_df.groupby("Ticker")["PnL_Dollar"].sum().reset_index()
        ticker_pnl = ticker_pnl.sort_values("PnL_Dollar", ascending=False).head(75)
        b3.plotly_chart(px.bar(ticker_pnl, x="Ticker", y="PnL_Dollar", title="Cumulative PnL by Ticker (Top 75)", text_auto='.2s'), use_container_width=True)
        
        # Monthly Seasonality
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_pnl = trades_df.groupby("Month")["PnL_Dollar"].sum().reindex(month_order).reset_index()
        b4.plotly_chart(px.bar(monthly_pnl, x="Month", y="PnL_Dollar", title="Cumulative PnL by Month (Seasonality)", text_auto='.2s'), use_container_width=True)

        st.subheader("Trade Log")
        st.dataframe(trades_df.style.format({
            "Entry": "{:.2f}", "Exit": "{:.2f}", "R": "{:.2f}", "PnL_Dollar": "${:,.2f}",
            "Age": "{:.1f}y", "AvgVol": "{:,.0f}"
        }), use_container_width=True)

        # --- COPYABLE DICTIONARY OUTPUT ---
        st.markdown("---")
        st.subheader("Configuration & Results (Copy Code)")
        st.info("Copy the dictionary below and paste it into your `STRATEGY_BOOK` list in the Screener.")

        # Updated to match exact requested format
        dict_str = f"""{{
    "id": "STRAT_{int(time.time())}",
    "name": "Generated Strategy ({grade})",
    "description": "Universe: {univ_choice}. Dir: {trade_direction}. Filter: {trend_filter}. PF: {pf:.2f}. SQN: {sqn:.2f}.",
    "universe_tickers": {tickers_to_run}, 
    "settings": {{
        "trade_direction": "{trade_direction}",
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
