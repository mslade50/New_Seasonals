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
    "XLU", "XLV", "XLY", "XME", "XOP", "XRT", "GLD", "CEF", "SLV", "BTC-USD",
    "ETH-USD", "UNG", "UVXY",
]
INDEX_ETFS = ["SPY", "QQQ", "IWM", "DIA", "SMH"]
CORE_TICKERS = ["SPY", "QQQ", "IWM", "SMH", "DIA"]
CSV_PATH = "seasonal_ranks.csv"

# -----------------------------------------------------------------------------
# UTILS & CACHING
# -----------------------------------------------------------------------------

def clear_all_caches():
    st.cache_data.clear()

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
    """Vectorized seasonal lookup for backtesting."""
    t_map = sznl_map.get(ticker, {})
    if not t_map:
        return pd.Series(np.nan, index=dates)
    
    # Create a series of (Month, Day) tuples
    mds = dates.map(lambda x: (x.month, x.day))
    return mds.map(t_map)

@st.cache_data(show_spinner=True)
def download_universe_data(tickers):
    """Downloads OHLCV data for a list of tickers."""
    if not tickers:
        return pd.DataFrame()
    
    # Download in one batch for speed
    df = yf.download(tickers, period="5y", group_by='ticker', auto_adjust=True, progress=False)
    
    # Reformat to a dictionary of DataFrames for easier per-ticker processing
    data_dict = {}
    
    if len(tickers) == 1:
        # Handle single ticker case from yfinance
        t = tickers[0]
        df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
        if not df.empty:
            data_dict[t] = df
    else:
        for t in tickers:
            try:
                # yf.download with group_by='ticker' returns MultiIndex columns
                t_df = df[t].copy()
                # Check if empty (all NaNs)
                if t_df.dropna(how='all').empty:
                    continue
                data_dict[t] = t_df
            except KeyError:
                continue
                
    return data_dict

# -----------------------------------------------------------------------------
# DASHBOARD LOGIC (Original Code)
# -----------------------------------------------------------------------------

def percentile_rank(series: pd.Series, value) -> float:
    s = series.dropna().values
    if s.size == 0: return np.nan
    if isinstance(value, (pd.Series, np.ndarray, list, tuple)):
        arr = np.asarray(value).ravel()
        v = float(arr[-1]) if arr.size > 0 else np.nan
    else:
        try: v = float(value)
        except: return np.nan
    if np.isnan(v): return np.nan
    return float((s <= v).sum() / s.size * 100.0)

@st.cache_data(show_spinner=True)
def load_sector_metrics(tickers):
    sznl_map = load_seasonal_map()
    today = datetime.datetime.now()
    rows = []

    for t in tickers:
        try:
            df = yf.download(t, period="2y", interval="1d", auto_adjust=True, progress=False)
        except: continue
        if df.empty: continue
        
        col_name = "Adj Close" if "Adj Close" in df.columns else "Close"
        close = df[col_name].dropna()
        if close.empty: continue

        ma5, ma20, ma50, ma200 = [close.rolling(w).mean() for w in [5, 20, 50, 200]]
        dist5, dist20, dist50, dist200 = [(close - ma) / ma * 100.0 for ma in [ma5, ma20, ma50, ma200]]

        vals = [d.dropna().iloc[-1] if not d.dropna().empty else np.nan 
                for d in [dist5, dist20, dist50, dist200]]
        
        ranks = [percentile_rank(d, v) for d, v in zip([dist5, dist20, dist50, dist200], vals)]
        
        sznl_val = get_sznl_val_series(t, pd.Series([today]), sznl_map).iloc[0] if t in sznl_map else np.nan

        rows.append({
            "Ticker": t,
            "Price": float(close.iloc[-1]),
            "Sznl": sznl_val,
            "PctRank5": ranks[0],
            "PctRank20": ranks[1],
            "PctRank50": ranks[2],
            "PctRank200": ranks[3],
        })

    if not rows: return pd.DataFrame()
    df_out = pd.DataFrame(rows)
    return df_out

@st.cache_data(show_spinner=True)
def load_core_distance_frame():
    sznl_map = load_seasonal_map()
    all_feats = []
    close_series_map = {} 

    for t in CORE_TICKERS:
        try:
            df = yf.download(t, period="max", interval="1d", auto_adjust=True, progress=False)
        except: continue
        
        col_name = "Adj Close" if "Adj Close" in df.columns else "Close"
        close = df[col_name].dropna()
        if close.empty: continue
        close_series_map[t] = close.copy() 

        dists = pd.DataFrame(index=close.index)
        for w in [5, 20, 50, 200]:
            ma = close.rolling(w).mean()
            dists[f"raw_dist{w}"] = (close - ma) / ma * 100.0
            
        ranked_dists = dists.rank(pct=True)
        
        feats = pd.DataFrame(index=close.index)
        for w in [5, 20, 50, 200]:
            feats[f"{t}_dist{w}"] = ranked_dists[f"raw_dist{w}"]

        t_map = sznl_map.get(t, {})
        feats[f"{t}_sznl"] = [t_map.get((m, d), np.nan) for m, d in zip(feats.index.month, feats.index.day)]
        all_feats.append(feats)

    if not all_feats: return pd.DataFrame()

    core_df = all_feats[0]
    for feats in all_feats[1:]:
        core_df = core_df.join(feats, how="inner")

    core_df = core_df.dropna().sort_index()
    horizons = [2, 5, 10, 21, 63]
    for t in CORE_TICKERS:
        close = close_series_map.get(t)
        if close is not None:
            close = close.reindex(core_df.index)
            for h in horizons:
                core_df[f"{t}_fwd_{h}d"] = close.shift(-h) / close - 1.0

    return core_df

def compute_distance_matches(core_df):
    if core_df.empty: return pd.DataFrame(), pd.DataFrame()
    df = core_df.copy().sort_index()
    feature_cols = [c for c in df.columns if "dist" in c or "sznl" in c]
    sznl_cols = [c for c in feature_cols if "sznl" in c]
    df = df.dropna(subset=feature_cols)
    
    if len(df) < 200: return pd.DataFrame(), pd.DataFrame()

    X_full = df[feature_cols].astype(float).values
    sznl_indices = [df[feature_cols].columns.get_loc(c) for c in sznl_cols]
    X_scaled = X_full.copy()
    X_scaled[:, sznl_indices] /= 100.0 

    target_vector = X_scaled[-1]
    history_matrix = X_scaled[:-63] # Exclude recent history
    hist_index = df.index[:-63]
    
    # Filter logical dates
    valid_mask = (hist_index.year > 1997)
    if not valid_mask.any(): return pd.DataFrame(), pd.DataFrame()

    history_matrix = history_matrix[valid_mask]
    hist_index = hist_index[valid_mask]

    diffs = history_matrix - target_vector
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    
    results = df.loc[hist_index].copy()
    results["distance"] = dists
    results = results.sort_values("distance")

    selected_dates = []
    for dt in results.index:
        if all(abs((dt - prev).days) > 21 for prev in selected_dates):
            selected_dates.append(dt)
            if len(selected_dates) >= 20: break

    if not selected_dates: return pd.DataFrame(), pd.DataFrame()

    matches = results.loc[selected_dates].copy().reset_index().rename(columns={"index": "Date"})
    
    # Contribution calculation
    locs = [hist_index.get_loc(d) for d in selected_dates]
    selected_vectors = history_matrix[locs]
    contrib_df = pd.DataFrame((selected_vectors - target_vector)**2, index=selected_dates, columns=feature_cols)
    
    return matches, contrib_df

# -----------------------------------------------------------------------------
# BACKTESTER LOGIC
# -----------------------------------------------------------------------------

def calculate_backtest_indicators(df, sznl_map, ticker):
    """
    Calculates all necessary indicators for the backtest signals.
    Uses EXPANDING rank to prevent look-ahead bias.
    """
    df = df.copy()
    
    # 1. Performance Returns
    for window in [5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window)
        
    # 2. Expanding Percentile Ranks of Returns (Point-in-time)
    # We use a min_periods to let data stabilize
    for window in [5, 10, 21]:
        col = f'ret_{window}d'
        # Expanding rank (0.0 to 1.0) -> * 100
        df[f'rank_ret_{window}d'] = df[col].expanding(min_periods=252).rank(pct=True) * 100.0
        
    # 3. ATR(14)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # 4. Seasonality
    df['Sznl'] = get_sznl_val_series(ticker, df.index, sznl_map)
    
    # 5. 52w High / Low
    # Shifted by 1 because we want to know if TODAY is a new high relative to PREVIOUS year
    rolling_high = df['High'].shift(1).rolling(252).max()
    rolling_low = df['Low'].shift(1).rolling(252).min()
    df['is_52w_high'] = df['High'] > rolling_high
    df['is_52w_low'] = df['Low'] < rolling_low
    
    # 6. Volume Threshold
    # Vol > X * 63d Avg
    vol_ma = df['Volume'].rolling(63).mean()
    df['vol_ratio'] = df['Volume'] / vol_ma
    
    return df

def run_backtest_engine(universe_dict, params, sznl_map):
    """
    Iterates through tickers, generates signals, and simulates trades.
    """
    trades = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_tickers = len(universe_dict)
    
    for i, (ticker, df_raw) in enumerate(universe_dict.items()):
        status_text.text(f"Backtesting {ticker}...")
        progress_bar.progress((i+1)/total_tickers)
        
        if len(df_raw) < 260: continue # Skip short history
        
        # Calc Indicators
        df = calculate_backtest_indicators(df_raw, sznl_map, ticker)
        df = df.dropna()
        
        # --- SIGNAL GENERATION ---
        conditions = []
        
        # Perf Rank Condition
        if params['use_perf_rank']:
            col = f"rank_ret_{params['perf_window']}d"
            if params['perf_logic'] == '<':
                cond = df[col] < params['perf_thresh']
            else:
                cond = df[col] > params['perf_thresh']
                
            if params['perf_first_instance']:
                # Check if condition was NOT met in previous X days
                # rolling sum of boolean map
                prev_instances = cond.shift(1).rolling(params['perf_lookback']).sum()
                cond = cond & (prev_instances == 0)
            
            conditions.append(cond)

        # Seasonal Condition
        if params['use_sznl']:
            if params['sznl_logic'] == '<':
                cond = df['Sznl'] < params['sznl_thresh']
            else:
                cond = df['Sznl'] > params['sznl_thresh']
                
            if params['sznl_first_instance']:
                prev_instances = cond.shift(1).rolling(params['sznl_lookback']).sum()
                cond = cond & (prev_instances == 0)
            
            conditions.append(cond)
            
        # 52w High/Low
        if params['use_52w']:
            if params['52w_type'] == 'High':
                cond = df['is_52w_high']
            else:
                cond = df['is_52w_low']
            
            if params['52w_first_instance']:
                prev_instances = cond.shift(1).rolling(params['52w_lookback']).sum()
                cond = cond & (prev_instances == 0)
            
            conditions.append(cond)
            
        # Volume
        if params['use_vol']:
            cond = df['vol_ratio'] > params['vol_thresh']
            conditions.append(cond)
            
        # COMBINE SIGNALS
        if not conditions:
            continue
            
        final_signal = conditions[0]
        for c in conditions[1:]:
            final_signal = final_signal & c
            
        signal_dates = df.index[final_signal]
        
        # --- TRADE SIMULATION (Event Loop per Ticker) ---
        # To properly handle Stops/Targets, we must look forward from signal
        
        for signal_date in signal_dates:
            try:
                # Data at signal
                sig_idx = df.index.get_loc(signal_date)
                # Need enough future data
                if sig_idx + params['holding_days'] + 2 >= len(df): continue
                
                atr = df['ATR'].iloc[sig_idx]
                if np.isnan(atr) or atr == 0: continue
                
                # Entry Logic
                entry_price = 0.0
                if params['entry_type'] == 'Signal Close':
                    entry_price = df['Close'].iloc[sig_idx]
                    start_idx = sig_idx + 1 # Check stops starting next day
                elif params['entry_type'] == 'T+1 Open':
                    entry_price = df['Open'].iloc[sig_idx + 1]
                    start_idx = sig_idx + 1 # Check stops T+1 Intraday
                elif params['entry_type'] == 'T+1 Close':
                    entry_price = df['Close'].iloc[sig_idx + 1]
                    start_idx = sig_idx + 2
                
                # Stop / Target Levels
                # Assuming Long logic (Target > Entry). If Strategy is shorting, need to flip.
                # Implicitly assuming Long for this code based on "Equity Curve" request.
                stop_dist = atr * params['stop_atr']
                tgt_dist = atr * params['tgt_atr']
                
                stop_price = entry_price - stop_dist
                tgt_price = entry_price + tgt_dist
                
                # Scan forward for outcome
                exit_price = entry_price # default
                r_realized = 0.0
                exit_date = None
                exit_type = "Hold"
                
                # Slice future window
                future_window = df.iloc[start_idx : start_idx + params['holding_days']]
                
                for f_date, f_row in future_window.iterrows():
                    # Check Low against Stop
                    if f_row['Low'] <= stop_price:
                        # Assuming we get stopped at price (slippage ignored)
                        # Check if Open was below stop (gap down)
                        if f_row['Open'] < stop_price:
                            exit_price = f_row['Open']
                        else:
                            exit_price = stop_price
                        exit_type = "Stop"
                        exit_date = f_date
                        break
                    
                    # Check High against Target
                    if f_row['High'] >= tgt_price:
                        if f_row['Open'] > tgt_price:
                            exit_price = f_row['Open']
                        else:
                            exit_price = tgt_price
                        exit_type = "Target"
                        exit_date = f_date
                        break
                
                # If loop finishes without break, it's a Time Exit
                if exit_type == "Hold":
                    exit_price = future_window['Close'].iloc[-1]
                    exit_date = future_window.index[-1]
                    exit_type = "Time"
                
                # Calculate R
                # Risk = Entry - Stop Price (Expected)
                risk_unit = entry_price - stop_price
                if risk_unit <= 0: risk_unit = 0.01 # prevent div/0
                
                raw_pnl = exit_price - entry_price
                r_realized = raw_pnl / risk_unit
                
                trades.append({
                    "Ticker": ticker,
                    "SignalDate": signal_date,
                    "EntryPrice": entry_price,
                    "ExitDate": exit_date,
                    "ExitPrice": exit_price,
                    "ExitType": exit_type,
                    "R": r_realized
                })
                
            except Exception as e:
                continue

    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(trades)

# -----------------------------------------------------------------------------
# PAGES
# -----------------------------------------------------------------------------

def page_dashboard():
    st.title("Sector ETF Trend Dashboard")
    if st.button("Refresh Data"):
        clear_all_caches()
        st.rerun()
    
    # ... (Rest of the original dashboard code here, condensed for brevity) ...
    # Re-using logic from previous prompt for display
    
    with st.spinner("Loading Metrics..."):
        df_sectors = load_sector_metrics(SECTOR_ETFS)
    
    if not df_sectors.empty:
        st.subheader("Sector Metrics")
        st.dataframe(df_sectors.style.format("{:.2f}"), use_container_width=True)
        
    st.subheader("Historical Pattern Matcher")
    with st.spinner("Computing..."):
        core_df = load_core_distance_frame()
        matches, contrib = compute_distance_matches(core_df)
        
    if not matches.empty:
        st.write("Top Matches based on Euclidean Distance of Ranks")
        st.dataframe(matches, use_container_width=True)
        if not contrib.empty:
            st.write("Factor Contribution (Squared Error)")
            st.bar_chart(contrib)

def page_backtester():
    st.title("Strategy Backtester")
    
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("1. Universe")
    univ_choice = st.sidebar.selectbox("Select Universe", 
                                       ["Sector ETFs", "Sector + Index", "All CSV Tickers", "Custom (Upload CSV)"])
    
    custom_tickers = []
    if univ_choice == "Custom (Upload CSV)":
        uploaded_file = st.sidebar.file_uploader("Upload CSV (Header: 'Ticker')", type=["csv"])
        if uploaded_file:
            try:
                c_df = pd.read_csv(uploaded_file)
                if "Ticker" in c_df.columns:
                    custom_tickers = c_df["Ticker"].unique().tolist()
                    st.sidebar.success(f"Loaded {len(custom_tickers)} tickers.")
                else:
                    st.sidebar.error("CSV must have a 'Ticker' column.")
            except:
                st.sidebar.error("Invalid CSV.")
    
    st.sidebar.header("2. Execution Params")
    stop_atr = st.sidebar.number_input("Stop Loss (ATR Multiplier)", 0.5, 10.0, 2.0, 0.1)
    tgt_atr = st.sidebar.number_input("Target (ATR Multiplier)", 0.5, 20.0, 4.0, 0.1)
    hold_days = st.sidebar.number_input("Holding Period (Days)", 1, 252, 10)
    entry_type = st.sidebar.selectbox("Entry Execution", ["Signal Close", "T+1 Open", "T+1 Close"])
    
    st.sidebar.header("3. Signal Logic")
    
    # Perf Rank
    use_perf = st.sidebar.checkbox("Filter: Perf Percentile Rank", value=True)
    perf_params = {}
    if use_perf:
        perf_params['window'] = st.sidebar.selectbox("Perf Window", [5, 10, 21], index=0)
        perf_params['logic'] = st.sidebar.selectbox("Logic", ["<", ">"], index=0)
        perf_params['thresh'] = st.sidebar.slider("Threshold", 0, 100, 15)
        perf_params['first'] = st.sidebar.checkbox("First Instance (Perf)", value=True)
        perf_params['lookback'] = st.sidebar.number_input("Lookback (Perf)", 5, 100, 21)
    
    # Seasonal
    use_sznl = st.sidebar.checkbox("Filter: Seasonal Rank")
    sznl_params = {}
    if use_sznl:
        sznl_params['logic'] = st.sidebar.selectbox("Sznl Logic", ["<", ">"], key='sz_l')
        sznl_params['thresh'] = st.sidebar.slider("Sznl Threshold", 0, 100, 15, key='sz_t')
        sznl_params['first'] = st.sidebar.checkbox("First Instance (Sznl)", value=True)
        sznl_params['lookback'] = st.sidebar.number_input("Lookback (Sznl)", 5, 100, 21)

    # 52w
    use_52w = st.sidebar.checkbox("Filter: 52w High/Low")
    highlow_params = {}
    if use_52w:
        highlow_params['type'] = st.sidebar.selectbox("Type", ["High", "Low"])
        highlow_params['first'] = st.sidebar.checkbox("First Instance (52w)", value=True)
        highlow_params['lookback'] = st.sidebar.number_input("Lookback (52w)", 5, 100, 21)
        
    # Volume
    use_vol = st.sidebar.checkbox("Filter: Volume Spike")
    vol_params = {}
    if use_vol:
        vol_params['thresh'] = st.sidebar.number_input("Vol > X * 63d Avg", 1.0, 10.0, 1.5, 0.1)
    
    # --- EXECUTION ---
    if st.button("Run Backtest"):
        # 1. Resolve Universe
        tickers_to_run = []
        sznl_map = load_seasonal_map()
        
        if univ_choice == "Sector ETFs":
            tickers_to_run = SECTOR_ETFS
        elif univ_choice == "Sector + Index":
            tickers_to_run = list(set(SECTOR_ETFS + INDEX_ETFS))
        elif univ_choice == "All CSV Tickers":
            tickers_to_run = list(sznl_map.keys())
            # Filter out cryptos or weird tickers if necessary, keeping simple for now
        elif univ_choice == "Custom (Upload CSV)":
            tickers_to_run = custom_tickers
            
        if not tickers_to_run:
            st.error("No tickers selected.")
            return

        st.write(f"**Universe Size:** {len(tickers_to_run)} tickers")
        
        with st.spinner("Downloading Data... (This may take a moment)"):
            data_dict = download_universe_data(tickers_to_run)
        
        if not data_dict:
            st.error("Failed to download data.")
            return
            
        # 2. Pack Params
        b_params = {
            'stop_atr': stop_atr, 'tgt_atr': tgt_atr, 
            'holding_days': hold_days, 'entry_type': entry_type,
            
            'use_perf_rank': use_perf, 
            'perf_window': perf_params.get('window'), 'perf_logic': perf_params.get('logic'), 
            'perf_thresh': perf_params.get('thresh'), 'perf_first_instance': perf_params.get('first'),
            'perf_lookback': perf_params.get('lookback'),
            
            'use_sznl': use_sznl,
            'sznl_logic': sznl_params.get('logic'), 'sznl_thresh': sznl_params.get('thresh'),
            'sznl_first_instance': sznl_params.get('first'), 'sznl_lookback': sznl_params.get('lookback'),
            
            'use_52w': use_52w,
            '52w_type': highlow_params.get('type'), '52w_first_instance': highlow_params.get('first'),
            '52w_lookback': highlow_params.get('lookback'),
            
            'use_vol': use_vol, 'vol_thresh': vol_params.get('thresh')
        }
        
        # 3. Run
        trades_df = run_backtest_engine(data_dict, b_params, sznl_map)
        
        if trades_df.empty:
            st.warning("No trades generated with these settings.")
            return
            
        # 4. Results & Viz
        trades_df = trades_df.sort_values("ExitDate")
        trades_df['CumR'] = trades_df['R'].cumsum()
        
        # Stats
        total_trades = len(trades_df)
        win_trades = trades_df[trades_df['R'] > 0]
        loss_trades = trades_df[trades_df['R'] <= 0]
        win_rate = len(win_trades) / total_trades * 100
        avg_r = trades_df['R'].mean()
        expectancy = avg_r # Simplified R expectancy
        
        # Max Drawdown on R curve
        cum_r = trades_df['CumR'].values
        running_max = np.maximum.accumulate(cum_r)
        drawdown = running_max - cum_r
        max_dd = drawdown.max()
        
        st.subheader("Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", total_trades)
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Avg R / Trade", f"{avg_r:.2f}R")
        col4.metric("Max Drawdown", f"{max_dd:.2f}R")
        
        st.subheader("Cumulative Equity Curve (R-Multiples)")
        fig = px.line(trades_df, x="ExitDate", y="CumR", title="Equity Curve")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Trade List")
        st.dataframe(trades_df.style.format({
            "EntryPrice": "{:.2f}", "ExitPrice": "{:.2f}", "R": "{:.2f}"
        }), use_container_width=True)


# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Backtester"])
    
    if page == "Dashboard":
        page_dashboard()
    else:
        page_backtester()

if __name__ == "__main__":
    main()
