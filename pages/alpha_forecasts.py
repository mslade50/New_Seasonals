import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import datetime

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CSV_PATH = "seasonal_ranks.csv"
MARKET_METRICS_PATH = "market_metrics_full_export.csv" # Optional

# SCANNER SETTINGS
NEIGHBORS_K = 50           # Number of similar historical dates to find
FWD_WINDOW = 5             # Target Horizon (Days)
MIN_HISTORY_YRS = 10       # Minimum history required to trust the scan

# PARED DOWN UNIVERSE
SELECTED_UNIVERSE = [
    # --- INDICES & COMMODITIES ---
    'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'SLV', 'UNG', 'UVXY',
    # --- SECTOR & INDUSTRY ETFs ---
    'XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLU', 'XLB',
    'SMH', 'IBB', 'XBI', 'KRE', 'XRT', 'XHB', 'XOP', 'XME',
    'VNQ', 'IYR', 'ITA', 'ITB', 'IHI', 'OIH', 'CEF',
    # --- MEGA CAP TECH & SEMIS ---
    'AAPL', 'MSFT', 'NVDA', 'GOOG', 'AMZN', 'META', 'AMD', 'AVGO', 
    'ORCL', 'QCOM', 'TXN', 'ADBE',
    # --- FINANCE ---
    'JPM', 'BAC', 'GS', 'MS', 'V', 'AXP',
    # --- CONSUMER ---
    'WMT', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'PG', 'KO','FCX',
    # --- HEALTHCARE ---
    'LLY', 'UNH', 'JNJ', 'PFE', 'MRK',
    # --- INDUSTRIAL & ENERGY ---
    'CAT', 'BA', 'GE', 'UNP', 'XOM', 'CVX',
    # --- OTHERS ---
    'DIS', 'NFLX', 'CRM', 'TSLA', 'RTX', 'LMT','LUV','DE','FDX' 
]

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
        output_map[ticker] = pd.Series(
            group.seasonal_rank.values, index=group.MD
        ).to_dict()
    return output_map

@st.cache_data(show_spinner=False)
def load_market_metrics():
    """Loads external market breadth metrics if available."""
    try:
        df = pd.read_csv(MARKET_METRICS_PATH)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.set_index('date').sort_index()
        # Create a mock rank for Total Net Highs if columns exist
        if 'net_new_highs' in df.columns:
             # Simple rolling rank logic just to have the feature
             df['Mkt_Total_NH_5d_Rank'] = df['net_new_highs'].rolling(5).mean().expanding().rank(pct=True) * 100
             df['Mkt_Total_NH_21d_Rank'] = df['net_new_highs'].rolling(21).mean().expanding().rank(pct=True) * 100
             return df[['Mkt_Total_NH_5d_Rank', 'Mkt_Total_NH_21d_Rank']]
        return pd.DataFrame()
    except:
        return pd.DataFrame() # Fail silently if file missing

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
            # Fetch max to ensure we have enough history for Euclidean matching
            df = yf.download(t, period="25y", progress=False, auto_adjust=True)
            
            if isinstance(df.columns, pd.MultiIndex):
                if t in df.columns.levels[0]:
                    df = df[t]
                else:
                    df.columns = [c[0] for c in df.columns]
            
            df.index = pd.to_datetime(df.index)
            if len(df) > 500: # Minimum requirement
                data_dict[t] = df
        except:
            pass
        progress_bar.progress((i + 1) / len(tickers))
    
    progress_bar.empty()
    timestamp = pd.Timestamp.now(tz='US/Eastern').strftime("%Y-%m-%d %I:%M %p %Z")
    
    return data_dict, timestamp

# -----------------------------------------------------------------------------
# EUCLIDEAN ENGINE
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

    # 3. Market Metrics Join (if available)
    if market_metrics is not None and not market_metrics.empty:
        df = df.join(market_metrics, how='left')
        df.update(df.filter(regex='^Mkt_').ffill(limit=3))

    # 4. Target: 5D Forward Return
    df[f'FwdRet_{FWD_WINDOW}d'] = (df['Close'].shift(-FWD_WINDOW) / df['Close'] - 1.0) * 100.0

    # 5. Rank Transformation (0-100)
    # We rank everything to normalize for Euclidean Distance
    vars_to_rank = [
        'Ret_5d', 'Ret_10d', 'Ret_21d', 'Ret_252d', 
        'RealVol_21d', 'RealVol_63d', 
        'VolRatio_10d', 'VolRatio_21d'
    ]
    
    rank_cols = []
    
    # Rank Seasonality (It's already 0-100, but ensuring float)
    df['Seasonal_Rank'] = df['Seasonal']
    rank_cols.append('Seasonal_Rank')

    # Rank Market Metrics if they exist
    if 'Mkt_Total_NH_5d_Rank' in df.columns:
        rank_cols.extend(['Mkt_Total_NH_5d_Rank', 'Mkt_Total_NH_21d_Rank'])

    # Rank Technicals
    for v in vars_to_rank:
        if v in df.columns:
            col_name = v + '_Rank'
            df[col_name] = df[v].expanding(min_periods=252).rank(pct=True) * 100.0
            rank_cols.append(col_name)

    # Clean data
    df = df.dropna(subset=rank_cols + [f'FwdRet_{FWD_WINDOW}d'])
    
    return df, rank_cols

def run_euclidean_scanner(data_dict, sznl_map, market_metrics):
    results = []
    
    for ticker, df in data_dict.items():
        processed_df, rank_cols = calculate_features_and_rank(df, sznl_map, ticker, market_metrics)
        
        if processed_df.empty or len(processed_df) < 252: continue
        
        # --- THE EUCLIDEAN MATCHING LOGIC ---
        
        # 1. Get Today's Feature Vector
        current_row = processed_df.iloc[-1]
        target_vec = current_row[rank_cols].astype(float).values
        
        # 2. Get History (Excluding today)
        # Important: exclude overlapping forward windows if backtesting, 
        # but here we just exclude the very last row to find neighbors in the past
        history = processed_df.iloc[:-FWD_WINDOW].copy() 
        
        if len(history) < NEIGHBORS_K: continue

        feature_matrix = history[rank_cols].astype(float).values
        
        # 3. Calculate Euclidean Distance
        # dist = sqrt(sum((x - y)^2))
        diff = feature_matrix - target_vec
        dist_sq = np.sum(diff**2, axis=1)
        history['Euclidean_Dist'] = np.sqrt(dist_sq)
        
        # 4. Select Neighbors
        nearest = history.nsmallest(NEIGHBORS_K, 'Euclidean_Dist')
        
        # 5. Calculate Forecast Stats
        outcomes = nearest[f'FwdRet_{FWD_WINDOW}d']
        
        mean_ret = outcomes.mean()
        median_ret = outcomes.median()
        win_rate = (outcomes > 0).mean() * 100
        
        # Baseline (Full history mean)
        baseline = processed_df[f'FwdRet_{FWD_WINDOW}d'].mean()
        alpha = mean_ret - baseline
        
        # Volatility of the neighbors (Uncertainty)
        sigma_forecast = outcomes.std()
        
        # Z-Score of the Alpha (Is this signal significant?)
        alpha_z = alpha / (processed_df[f'FwdRet_{FWD_WINDOW}d'].std() / np.sqrt(NEIGHBORS_K))

        results.append({
            "Ticker": ticker,
            "Price": current_row['Close'],
            "Euclidean_Alpha": alpha,
            "Exp_Return": mean_ret,
            "Baseline": baseline,
            "Win_Rate": win_rate,
            "Alpha_Z": alpha_z,
            "Sigma": sigma_forecast,
            "Last_Date": processed_df.index[-1]
        })

    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# UI (UPDATED WITH SESSION STATE CACHING)
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Euclidean Alpha Scanner")
    
    st.title("🧪 Euclidean Similarity Scanner (5D Horizon)")
    st.markdown("""
    This tool projects **5-Day Returns** by finding the **50 most mathematically similar historical days** for each ticker based on a multi-factor vector (Seasonality, Momentum, Volatility, Volume).
    """)
    
    # Load static maps (fast)
    sznl_map = load_seasonal_map()
    mkt_metrics = load_market_metrics()
    active_tickers_str = ", ".join(SELECTED_UNIVERSE)

    # --- 1. INITIALIZE SESSION STATE ---
    if 'scan_results' not in st.session_state:
        st.session_state['scan_results'] = None
    if 'raw_data' not in st.session_state:
        st.session_state['raw_data'] = None

    # --- 2. SIDEBAR / SETTINGS ---
    with st.expander("⚙️ Universe & Settings", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"Scanning {len(SELECTED_UNIVERSE)} Tickers")
            st.text_area("Active Universe", value=active_tickers_str, height=60, disabled=True)
        with col2:
            st.metric("Neighbors (K)", NEIGHBORS_K)
            # This button triggers the heavy lift
            if st.button("Run Euclidean Scan", type="primary", use_container_width=True):
                with st.spinner(f"Fetching Data & Calculating Euclidean Distances..."):
                    
                    # A. Fetch Data
                    data_dict, fetch_time = get_batch_data(SELECTED_UNIVERSE)
                    
                    if not data_dict:
                        st.error("No valid data found.")
                    else:
                        # B. Run Scanner
                        df_res = run_euclidean_scanner(data_dict, sznl_map, mkt_metrics)
                        
                        if df_res.empty:
                            st.warning("No results generated.")
                        else:
                            # C. SAVE TO SESSION STATE
                            st.session_state['raw_data'] = data_dict
                            st.session_state['scan_results'] = df_res.sort_values("Euclidean_Alpha", ascending=False)
                            st.success(f"Scan Complete! Processed {len(df_res)} tickers.")

    st.divider()

    # --- 3. RENDER RESULTS (FROM STATE) ---
    # Only render if we have data in the state
    if st.session_state['scan_results'] is not None:
        
        results_df = st.session_state['scan_results']
        raw_data = st.session_state['raw_data']
        
        # A. Summary Tables
        top_bulls = results_df.head(10)
        top_bears = results_df.tail(10).sort_values("Euclidean_Alpha", ascending=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🚀 Top Bullish Setups")
            st.dataframe(
                top_bulls[['Ticker', 'Euclidean_Alpha', 'Exp_Return', 'Win_Rate', 'Alpha_Z']].style.format({
                    'Euclidean_Alpha': "+{:.2f}%",
                    'Exp_Return': "{:.2f}%",
                    'Win_Rate': "{:.1f}%",
                    'Alpha_Z': "{:.2f}"
                }).background_gradient(subset=['Euclidean_Alpha'], cmap='Greens'),
                use_container_width=True, hide_index=True
            )
            
        with c2:
            st.subheader("🐻 Top Bearish Setups")
            st.dataframe(
                top_bears[['Ticker', 'Euclidean_Alpha', 'Exp_Return', 'Win_Rate', 'Alpha_Z']].style.format({
                    'Euclidean_Alpha': "{:.2f}%",
                    'Exp_Return': "{:.2f}%",
                    'Win_Rate': "{:.1f}%",
                    'Alpha_Z': "{:.2f}"
                }).background_gradient(subset=['Euclidean_Alpha'], cmap='Reds_r'),
                use_container_width=True, hide_index=True
            )
        
        st.divider()
        
        # B. Deep Dive Section (This is what you want to be interactive)
        st.subheader("🔎 Signal Inspector")
        
        col_sel, col_viz = st.columns([1, 3])
        
        with col_sel:
            # changing this dropdown will rerun the script, 
            # but because 'scan_results' is in session_state, we skip the heavy math above
            selected_ticker = st.selectbox("Select Ticker to Inspect", results_df['Ticker'].tolist())
            
            sel_row = results_df[results_df['Ticker'] == selected_ticker].iloc[0]
            st.metric("Projected 5D Return", f"{sel_row['Exp_Return']:.2f}%", delta=f"{sel_row['Euclidean_Alpha']:.2f}% vs Baseline")
            st.metric("Win Rate", f"{sel_row['Win_Rate']:.1f}%")
            st.metric("Z-Score", f"{sel_row['Alpha_Z']:.2f}")

        with col_viz:
            # We recalculate neighbors for just ONE ticker here to keep memory light, 
            # but we use the cached 'raw_data' so no download happens.
            df_viz, rank_cols = calculate_features_and_rank(raw_data[selected_ticker], sznl_map, selected_ticker, mkt_metrics)
            
            # Get Neighbors
            current_vec = df_viz.iloc[-1][rank_cols].astype(float).values
            hist_viz = df_viz.iloc[:-FWD_WINDOW].copy()
            dists = np.sqrt(np.sum((hist_viz[rank_cols].values - current_vec)**2, axis=1))
            hist_viz['Euclidean_Dist'] = dists
            
            neighbors = hist_viz.nsmallest(NEIGHBORS_K, 'Euclidean_Dist')
            
            # Draw Chart
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=neighbors[f'FwdRet_{FWD_WINDOW}d'],
                nbinsx=20,
                name='Neighbor Outcomes',
                marker_color='royalblue',
                opacity=0.7
            ))
            
            mean_val = neighbors[f'FwdRet_{FWD_WINDOW}d'].mean()
            fig.add_vline(x=mean_val, line_dash="dash", line_color="orange", annotation_text=f"Mean: {mean_val:.2f}%")
            fig.add_vline(x=0, line_color="white", line_width=1)
            
            fig.update_layout(
                title=f"Distribution of 5D Returns for {selected_ticker}'s Top {NEIGHBORS_K} Matches",
                xaxis_title="5D Forward Return (%)",
                yaxis_title="Count",
                template="plotly_dark",
                height=400,
                bargap=0.1
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("See Neighbor Details"):
                st.dataframe(
                    neighbors[['Close', 'Euclidean_Dist', f'FwdRet_{FWD_WINDOW}d']].sort_values('Euclidean_Dist').style.format({
                        'Close': "{:.2f}",
                        'Euclidean_Dist': "{:.2f}",
                        f'FwdRet_{FWD_WINDOW}d': "{:+.2f}%"
                    }),
                    use_container_width=True
                )

if __name__ == "__main__":
    main()
