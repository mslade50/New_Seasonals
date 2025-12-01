import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import datetime
import itertools

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CSV_PATH = "seasonal_ranks.csv"
MARKET_METRICS_PATH = "market_metrics_full_export.csv"

# SCANNER SETTINGS
NEIGHBORS_K = 50           # Number of similar historical dates to find
FWD_WINDOWS = [5, 10, 21]  # Target Horizons (Days)
MIN_HISTORY_YRS = 10       # Minimum history required to trust the scan

# UNIVERSE
SELECTED_UNIVERSE = [
    'SPY', 'QQQ', 'IWM', 'DIA','SMH',
    'GLD', 'SLV', 'UNG', 'OIH', 'CEF','FCX',
    'UVXY', '^VIX',
    'XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLU', 'XLB',
    'IBB', 'XBI', 'KRE', 'XRT', 'XHB', 'XOP', 'XME',
    'VNQ', 'IYR', 'ITA', 'ITB', 'IHI', 
    'AAPL', 'MSFT', 'NVDA', 'GOOG', 'AMZN', 'META', 'AMD', 'AVGO', 
    'ORCL', 'QCOM', 'NFLX', 'TSLA',
    'JPM', 'BAC', 'GS', 'MS', 'C',
    'COST', 'HD', 'NKE', 'SBUX', 'DIS', 'FDX', 'LUV', 'V', 'AXP',
    'LLY', 'UNH', 'JNJ', 'PFE', 'MRK',
    'CAT', 'UNP', 'XOM', 'CVX','DE',
    'PG', 'KO','WMT','MCD', 'PEP',
    'CRM', 'NOW',
    'RTX', 'LMT','BA', 'GE', 
]

# SECTOR DEFINITIONS
SECTOR_DEFINITIONS = {
    "Indices": ['SPY', 'QQQ', 'IWM', 'DIA'],
    "Semis (Breakout)": ['SMH', 'NVDA', 'AMD', 'AVGO', 'QCOM'], 
    "Mega Cap Tech": ['AAPL', 'MSFT', 'NVDA', 'GOOG', 'AMZN', 'META', 'AMD', 'AVGO', 'ORCL', 'QCOM', 'NFLX', 'TSLA'],
    "PMs": ['GLD', 'SLV','CEF'],
    "Commodities": ['UNG', 'OIH', 'CEF', 'FCX','USO','XME'],
    "Vol": ['UVXY', '^VIX'],
    "Sector ETFs": ['XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLU', 'XLB', 'IBB', 'XBI', 'KRE', 'XRT', 'XHB', 'VNQ', 'IYR', 'ITA', 'ITB', 'IHI'],
    "Banks": ['JPM', 'BAC', 'GS', 'MS', 'C'],
    "Consumer & Payments": ['COST', 'HD', 'NKE', 'SBUX', 'DIS', 'FDX', 'LUV', 'V', 'AXP'],
    "Healthcare": ['LLY', 'UNH', 'JNJ', 'PFE', 'MRK'],
    "Industrials & Energy": ['CAT', 'UNP', 'XOM', 'CVX', 'DE','XOP'],
    "Staples": ['PG', 'KO', 'WMT', 'MCD', 'PEP'],
    "Software": ['CRM', 'NOW'],
    "Defense": ['RTX', 'LMT', 'BA', 'GE'],
    "FX": ['USDJPY=X','DX-Y.NYB','USDEUR=X', 'USDCHF=X', 'USDAUD=X', 'USDMXN=X'],
    "Crypto": ['BTC-USD', 'ETH-USD'],
}

# -----------------------------------------------------------------------------
# DATA ENGINE
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
        output_map[ticker] = pd.Series(group.seasonal_rank.values, index=group.MD).to_dict()
    return output_map

@st.cache_data(show_spinner=False)
def load_market_metrics():
    try:
        df = pd.read_csv(MARKET_METRICS_PATH)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.set_index('date').sort_index()
        if 'net_new_highs' in df.columns:
             df['Mkt_Total_NH_5d_Rank'] = df['net_new_highs'].rolling(5).mean().expanding().rank(pct=True) * 100
             df['Mkt_Total_NH_21d_Rank'] = df['net_new_highs'].rolling(21).mean().expanding().rank(pct=True) * 100
             return df[['Mkt_Total_NH_5d_Rank', 'Mkt_Total_NH_21d_Rank']]
        return pd.DataFrame()
    except:
        return pd.DataFrame() 

def get_sznl_val_series(ticker, dates, sznl_map):
    t_map = sznl_map.get(ticker, {})
    if not t_map:
        return pd.Series(50.0, index=dates)
    mds = dates.map(lambda x: (x.month, x.day))
    return mds.map(t_map).fillna(50.0)

@st.cache_data(ttl=3600, show_spinner=True)
def get_batch_data(ticker_list):
    tickers = list(set([t.strip().upper() for t in ticker_list]))
    if not tickers: return {}, ""
    data_dict = {}
    progress_bar = st.progress(0)
    for i, t in enumerate(tickers):
        try:
            # Use 25y period to allow plenty of history for the start_date filter
            df = yf.download(t, period="25y", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                if t in df.columns.levels[0]:
                    df = df[t]
                else:
                    df.columns = [c[0] for c in df.columns]
            df.index = pd.to_datetime(df.index)
            if len(df) > 500: 
                data_dict[t] = df
        except:
            pass
        progress_bar.progress((i + 1) / len(tickers))
    progress_bar.empty()
    
    # Format timestamp exactly like heatmap: HH:MM AM/PM EST
    timestamp = pd.Timestamp.now(tz='US/Eastern').strftime("%I:%M %p EST")
    return data_dict, timestamp

# -----------------------------------------------------------------------------
# FEATURE ENGINEERING (EXACT HEATMAP MATCH)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def calculate_features_and_rank(df, sznl_map, ticker, market_metrics=None):
    df = df.copy()
    
    # 1. Base Calcs
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. Predictor Variables (Raw)
    for w in [5, 10, 21, 252]:
        df[f'Ret_{w}d'] = df['Close'].pct_change(w)

    for w in [21, 63]:
        df[f'RealVol_{w}d'] = df['LogRet'].rolling(w).std() * np.sqrt(252) * 100.0
    
    for w in [10, 21]:
        df[f'VolRatio_{w}d'] = df['Volume'].rolling(w).mean() / df['Volume'].rolling(63).mean()

    # Seasonality
    df['Seasonal'] = get_sznl_val_series(ticker, df.index, sznl_map)

    # 3. Market Metrics
    if market_metrics is not None and not market_metrics.empty:
        df = df.join(market_metrics, how='left')
        df.update(df.filter(regex='^Mkt_').ffill(limit=3))

    # 4. Targets: Calculate ALL windows
    for w in FWD_WINDOWS:
        df[f'FwdRet_{w}d'] = (df['Close'].shift(-w) / df['Close'] - 1.0) * 100.0

    # 5. Rank Transformation
    vars_to_rank = [
        'Ret_5d', 'Ret_10d', 'Ret_21d', 'Ret_252d', 
        'RealVol_21d', 'RealVol_63d', 
        'VolRatio_10d', 'VolRatio_21d'
    ]
    
    rank_cols = []
    df['Seasonal_Rank'] = df['Seasonal']
    rank_cols.append('Seasonal_Rank')

    if 'Mkt_Total_NH_5d_Rank' in df.columns:
        rank_cols.extend(['Mkt_Total_NH_5d_Rank', 'Mkt_Total_NH_21d_Rank'])

    for v in vars_to_rank:
        if v in df.columns:
            col_name = v + '_Rank'
            df[col_name] = df[v].expanding(min_periods=252).rank(pct=True) * 100.0
            rank_cols.append(col_name)

    # Drop NA based on FEATURES, not targets
    df = df.dropna(subset=rank_cols)
    
    return df, rank_cols

def run_euclidean_scanner(data_dict, sznl_map, market_metrics, start_date):
    results = []
    
    # Convert start_date to pandas timestamp
    start_ts = pd.to_datetime(start_date)

    for ticker, df in data_dict.items():
        processed_df, rank_cols = calculate_features_and_rank(df, sznl_map, ticker, market_metrics)
        
        # Ensure we have data
        if processed_df.empty or len(processed_df) < 252: continue
        
        # Localize start_ts if needed
        if processed_df.index.tz is not None and start_ts.tz is None:
            start_ts = start_ts.tz_localize(processed_df.index.tz)

        # 1. Get Today's Feature Vector (Last available row)
        current_row = processed_df.iloc[-1]
        target_vec = current_row[rank_cols].astype(float).values
        
        # 2. Get History Pool (Exact Heatmap Match + Start Date Filter)
        # We take everything up until yesterday (iloc[:-1])
        # THEN we filter by start_date
        history_pool = processed_df.iloc[:-1].copy()
        history_pool = history_pool[history_pool.index >= start_ts]

        # Need enough history to find K neighbors
        if len(history_pool) < NEIGHBORS_K: continue

        feature_matrix = history_pool[rank_cols].astype(float).values
        
        # 3. Calculate Euclidean Distance
        diff = feature_matrix - target_vec
        dist_sq = np.sum(diff**2, axis=1)
        history_pool['Euclidean_Dist'] = np.sqrt(dist_sq)
        
        # 4. Select Neighbors
        nearest = history_pool.nsmallest(NEIGHBORS_K, 'Euclidean_Dist')
        
        # 5. Calculate Forecast Stats for ALL Windows
        row_data = {
            "Ticker": ticker,
            "Price": current_row['Close'],
            "Last_Date": processed_df.index[-1]
        }
        
        for w in FWD_WINDOWS:
            target_col = f'FwdRet_{w}d'
            
            # Use dropna() to handle recent neighbors that haven't realized returns yet
            outcomes = nearest[target_col].dropna()
            
            if outcomes.empty:
                row_data[f"Exp_Ret_{w}d"] = np.nan
                row_data[f"Alpha_{w}d"] = np.nan
                row_data[f"Win_Rate_{w}d"] = np.nan
            else:
                mean_ret = outcomes.mean()
                
                # Baseline is strictly from the filtered history pool to be fair
                # or global processed_df? Usually global processed_df is the better baseline reference.
                baseline = processed_df[target_col].mean()
                
                alpha = mean_ret - baseline
                win_rate = (outcomes > 0).mean() * 100
                
                row_data[f"Exp_Ret_{w}d"] = mean_ret
                row_data[f"Alpha_{w}d"] = alpha
                row_data[f"Win_Rate_{w}d"] = win_rate
                
                if w == 5:
                    std_dev = processed_df[target_col].std()
                    n_valid = len(outcomes)
                    row_data["Alpha_Z"] = alpha / (std_dev / np.sqrt(n_valid)) if std_dev > 0 and n_valid > 0 else 0

        results.append(row_data)

    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Euclidean Alpha Scanner")
    
    st.title("🧪 Euclidean Similarity Scanner (Multi-Horizon)")
    st.markdown("""
    Projects **5, 10, and 21-Day Returns** by identifying the **50 most mathematically similar historical days**.
    """)
    
    sznl_map = load_seasonal_map()
    mkt_metrics = load_market_metrics()
    active_tickers_str = ", ".join(SELECTED_UNIVERSE)

    # --- 1. INITIALIZE SESSION STATE ---
    if 'scan_results' not in st.session_state:
        st.session_state['scan_results'] = None
    if 'raw_data' not in st.session_state:
        st.session_state['raw_data'] = None
    if 'last_update' not in st.session_state:
        st.session_state['last_update'] = None

    # --- 2. SIDEBAR / SETTINGS ---
    with st.expander("⚙️ Universe & Settings", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"Scanning {len(SELECTED_UNIVERSE)} Tickers")
            st.text_area("Active Universe", value=active_tickers_str, height=70, disabled=True)
            
            # START DATE INPUT
            start_date_input = st.date_input("Match History Start Date", value=datetime.date(2000, 1, 1))
            
        with col2:
            st.metric("Neighbors (K)", NEIGHBORS_K)
            if st.button("Run Euclidean Scan", type="primary", use_container_width=True):
                with st.spinner(f"Fetching Data & Calculating Euclidean Distances..."):
                    
                    data_dict, fetch_time = get_batch_data(SELECTED_UNIVERSE)
                    
                    if not data_dict:
                        st.error("No valid data found.")
                    else:
                        df_res = run_euclidean_scanner(data_dict, sznl_map, mkt_metrics, start_date_input)
                        
                        if df_res.empty:
                            st.warning("No results generated.")
                        else:
                            st.session_state['raw_data'] = data_dict
                            st.session_state['scan_results'] = df_res.sort_values("Alpha_5d", ascending=False)
                            st.session_state['last_update'] = fetch_time
                            st.success(f"Scan Complete! Processed {len(df_res)} tickers.")

    # DISPLAY TIMESTAMP
    if st.session_state['last_update']:
        st.info(f"✅ Data Updated: {st.session_state['last_update']}")

    st.divider()

    # --- 3. RENDER RESULTS (FROM STATE) ---
    if st.session_state['scan_results'] is not None:
        
        results_df = st.session_state['scan_results']
        raw_data = st.session_state['raw_data']
        
        # A. Summary Tables (Focus on 5d Alpha for sorting, but show all alphas)
        display_cols = ['Ticker', 'Alpha_5d', 'Alpha_10d', 'Alpha_21d', 'Exp_Ret_5d', 'Win_Rate_5d', 'Alpha_Z']
        
        top_bulls = results_df.sort_values("Alpha_5d", ascending=False).head(10)
        top_bears = results_df.sort_values("Alpha_5d", ascending=True).head(10)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🚀 Top Bullish Setups (5d Sorted)")
            st.dataframe(
                top_bulls[display_cols].style.format({
                    'Alpha_5d': "+{:.2f}%", 'Alpha_10d': "+{:.2f}%", 'Alpha_21d': "+{:.2f}%",
                    'Exp_Ret_5d': "{:.2f}%", 'Win_Rate_5d': "{:.1f}%", 'Alpha_Z': "{:.2f}"
                }).background_gradient(subset=['Alpha_5d', 'Alpha_10d', 'Alpha_21d'], cmap='Greens'),
                use_container_width=True, hide_index=True
            )
            
        with c2:
            st.subheader("🐻 Top Bearish Setups (5d Sorted)")
            st.dataframe(
                top_bears[display_cols].style.format({
                    'Alpha_5d': "{:.2f}%", 'Alpha_10d': "{:.2f}%", 'Alpha_21d': "{:.2f}%",
                    'Exp_Ret_5d': "{:.2f}%", 'Win_Rate_5d': "{:.1f}%", 'Alpha_Z': "{:.2f}"
                }).background_gradient(subset=['Alpha_5d', 'Alpha_10d', 'Alpha_21d'], cmap='Reds_r'),
                use_container_width=True, hide_index=True
            )
        
        st.divider()
        
        # B. Deep Dive Section
        st.subheader("🔎 Signal Inspector")
        
        col_sel, col_viz = st.columns([1, 3])
        
        with col_sel:
            selected_ticker = st.selectbox("Select Ticker to Inspect", results_df['Ticker'].tolist())
            inspect_window = st.selectbox("Forecast Horizon", [5, 10, 21], index=0)
            
            sel_row = results_df[results_df['Ticker'] == selected_ticker].iloc[0]
            
            alpha_val = sel_row[f'Alpha_{inspect_window}d']
            exp_val = sel_row[f'Exp_Ret_{inspect_window}d']
            win_val = sel_row[f'Win_Rate_{inspect_window}d']
            
            st.metric(f"Projected {inspect_window}D Return", f"{exp_val:.2f}%", delta=f"{alpha_val:.2f}% vs Baseline")
            st.metric("Win Rate", f"{win_val:.1f}%")
            if inspect_window == 5:
                st.metric("Z-Score (5d)", f"{sel_row['Alpha_Z']:.2f}")

        with col_viz:
            df_viz, rank_cols = calculate_features_and_rank(raw_data[selected_ticker], sznl_map, selected_ticker, mkt_metrics)
            
            # Re-Find Neighbors using same start_date logic (though visually we can show them all)
            current_vec = df_viz.iloc[-1][rank_cols].astype(float).values
            hist_viz = df_viz.iloc[:-1].copy()
            # Note: For visualization, we could show full history, but best to stick to user filter
            # We don't have the user filter in session state explicitly as a var, 
            # but usually it's fine to show all history for context or just filter implicitly.
            # Let's just recalculate on full history for visual context, OR use the filter?
            # Better to use the filter to match the numbers shown on left.
            
            # Simple workaround: The user input 'start_date_input' is available in main scope.
            # Convert to TS
            viz_start_ts = pd.to_datetime(start_date_input)
            if hist_viz.index.tz is not None and viz_start_ts.tz is None:
                viz_start_ts = viz_start_ts.tz_localize(hist_viz.index.tz)
            
            hist_viz = hist_viz[hist_viz.index >= viz_start_ts]

            dists = np.sqrt(np.sum((hist_viz[rank_cols].values - current_vec)**2, axis=1))
            hist_viz['Euclidean_Dist'] = dists
            neighbors = hist_viz.nsmallest(NEIGHBORS_K, 'Euclidean_Dist')
            
            # Draw Chart
            fig = go.Figure()
            target_col = f'FwdRet_{inspect_window}d'
            
            # Filter NaNs for plotting histogram
            plot_data = neighbors[target_col].dropna()
            
            fig.add_trace(go.Histogram(
                x=plot_data, nbinsx=25,
                name='Neighbor Outcomes', marker_color='royalblue', opacity=0.7
            ))
            
            if not plot_data.empty:
                mean_val = plot_data.mean()
                fig.add_vline(x=mean_val, line_dash="dash", line_color="orange", annotation_text=f"Mean: {mean_val:.2f}%")
            
            fig.add_vline(x=0, line_color="white", line_width=1)
            
            fig.update_layout(
                title=f"Distribution of {inspect_window}D Returns for {selected_ticker}'s Matches (Post-{start_date_input})",
                xaxis_title=f"{inspect_window}D Forward Return (%)",
                yaxis_title="Count",
                template="plotly_dark", height=400, bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("See Neighbor Details"):
                st.dataframe(
                    neighbors[['Close', 'Euclidean_Dist', target_col]].sort_values('Euclidean_Dist').style.format({
                        'Close': "{:.2f}", 'Euclidean_Dist': "{:.2f}", target_col: "{:+.2f}%"
                    }), use_container_width=True
                )

        st.divider()

        # --- C. SECTOR AGGREGATES (MULTI-HORIZON) ---
        st.subheader("📊 Sector & Segment Aggregates (Curve)")
        st.caption("Average Expected Return across time horizons.")

        sector_stats = []
        for sector, tickers in SECTOR_DEFINITIONS.items():
            matches = results_df[results_df['Ticker'].isin(tickers)]
            
            if not matches.empty:
                sector_stats.append({
                    "Segment": sector,
                    "Exp Ret 5d": matches['Exp_Ret_5d'].mean(),
                    "Exp Ret 10d": matches['Exp_Ret_10d'].mean(),
                    "Exp Ret 21d": matches['Exp_Ret_21d'].mean(),
                    "Alpha 5d": matches['Alpha_5d'].mean(),
                    "Count": len(matches)
                })
        
        if sector_stats:
            sector_df = pd.DataFrame(sector_stats).sort_values("Exp Ret 5d", ascending=False)
            
            st.dataframe(
                sector_df.style.format({
                    "Exp Ret 5d": "{:.2f}%", "Exp Ret 10d": "{:.2f}%", "Exp Ret 21d": "{:.2f}%",
                    "Alpha 5d": "{:+.2f}%"
                }).background_gradient(subset=['Exp Ret 5d', 'Exp Ret 10d', 'Exp Ret 21d'], cmap="RdBu", vmin=-1.0, vmax=1.0),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No sector data available.")

if __name__ == "__main__":
    main()
