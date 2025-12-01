import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import datetime
import itertools

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CSV_PATH = "seasonal_ranks.csv"

# --- MODIFIED: ONLY PREDICT 5D RETURNS ---
FWD_WINDOWS = [5] 
RECENT_WINDOW_YEARS = 5  # The "Regime Check" Window

# PARED DOWN UNIVERSE (~75 Tickers)
SELECTED_UNIVERSE = [
    # --- INDICES & COMMODITIES (8) ---
    'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'SLV', 'UNG', 'UVXY',
    
    # --- SECTOR & INDUSTRY ETFs (24) ---
    'XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLU', 'XLB',
    'SMH', 'IBB', 'XBI', 'KRE', 'XRT', 'XHB', 'XOP', 'XME',
    'VNQ', 'IYR', 'ITA', 'ITB', 'IHI', 'OIH', 'CEF',

    # --- MEGA CAP TECH & SEMIS (12) ---
    'AAPL', 'MSFT', 'NVDA', 'GOOG', 'AMZN', 'META', 'AMD', 'AVGO', 
    'ORCL', 'QCOM', 'TXN', 'ADBE',

    # --- FINANCE (6) ---
    'JPM', 'BAC', 'GS', 'MS', 'V', 'AXP',

    # --- CONSUMER (8) ---
    'WMT', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'PG', 'KO','FCX',

    # --- HEALTHCARE (5) ---
    'LLY', 'UNH', 'JNJ', 'PFE', 'MRK',

    # --- INDUSTRIAL & ENERGY (6) ---
    'CAT', 'BA', 'GE', 'UNP', 'XOM', 'CVX',

    # --- OTHERS (6) ---
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
        return {}, []

    if df.empty: return {}, []

    # Get unique tickers available in the CSV
    ticker_list = df['ticker'].unique().tolist()

    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])
    df["MD"] = df["Date"].apply(lambda x: (x.month, x.day))
    
    output_map = {}
    for ticker, group in df.groupby("ticker"):
        output_map[ticker] = pd.Series(
            group.seasonal_rank.values, index=group.MD
        ).to_dict()
    return output_map, ticker_list

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
            # Fetch 25 years to establish long-term baseline
            df = yf.download(t, period="25y", progress=False, auto_adjust=True)
            
            if isinstance(df.columns, pd.MultiIndex):
                if t in df.columns.levels[0]:
                    df = df[t]
                else:
                    df.columns = [c[0] for c in df.columns]
            
            df.index = pd.to_datetime(df.index)
            
            start_date = pd.Timestamp.now() - pd.DateOffset(years=25)
            df = df[df.index >= start_date]
            
            if len(df) > 252:
                data_dict[t] = df
        except:
            pass
        progress_bar.progress((i + 1) / len(tickers))
    
    progress_bar.empty()
    timestamp = pd.Timestamp.now(tz='US/Eastern').strftime("%Y-%m-%d %I:%M %p %Z")
    
    return data_dict, timestamp

# -----------------------------------------------------------------------------
# FEATURE ENGINEERING (MODIFIED)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def calculate_internal_features(df, sznl_map, ticker):
    df = df.copy()
    
    # Log Returns & Volatility
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_Daily'] = df['LogRet'].rolling(21).std()

    # --- MODIFIED: Momentum Windows ---
    # Removed 126d
    for w in [5, 10, 21, 63, 252]:
        df[f'Ret_{w}d'] = df['Close'].pct_change(w)

    # --- MODIFIED: Realized Volatility ---
    # Removed 63d, kept only 21d
    for w in [21]:
        df[f'RealVol_{w}d'] = df['LogRet'].rolling(w).std() * np.sqrt(252) * 100.0
    
    # --- REMOVED: Volatility Regimes (VolChange 5d/21d) ---
    # The previous block calculating VolChange_5d and VolChange_21d was deleted here.

    # --- MODIFIED: Relative Volume ---
    # Removed 5d, kept only 21d
    for w in [21]:
        df[f'VolRatio_{w}d'] = df['Volume'].rolling(w).mean() / df['Volume'].rolling(63).mean()

    # Seasonality Injection
    df['Seasonal'] = get_sznl_val_series(ticker, df.index, sznl_map)

    # --- MODIFIED: Ranking List ---
    # Updated to reflect removed features
    vars_to_rank = [
        'Seasonal',
        'Ret_5d', 'Ret_10d', 'Ret_21d', 'Ret_63d', 'Ret_252d',
        'RealVol_21d', 
        # 'RealVol_63d' -> REMOVED
        # 'VolChange_5d', 'VolChange_21d' -> REMOVED
        # 'VolRatio_5d' -> REMOVED
        'VolRatio_21d'
    ]
    
    rank_cols = []
    for v in vars_to_rank:
        rank_col = v + '_Rank'
        if v == 'Seasonal':
            df[rank_col] = df[v]
        else:
            df[rank_col] = df[v].expanding(min_periods=252).rank(pct=True) * 100.0
        rank_cols.append(rank_col)

    # Forward Returns (Targets)
    for w in FWD_WINDOWS:
        col_raw = f'FwdRet_{w}d_Raw'
        col_sigma = f'FwdRet_{w}d_Sigma'
        
        # Raw Return
        df[col_raw] = (df['Close'].shift(-w) / df['Close']) - 1.0
        
        # Sigma-adjusted Return
        expected_move = df['Vol_Daily'] * np.sqrt(w)
        df[col_sigma] = df[col_raw] / expected_move.replace(0, np.nan)

    # Drop rows where we can't calculate ranks or targets
    df = df.dropna(subset=['Vol_Daily'] + rank_cols)
    return df, rank_cols

# -----------------------------------------------------------------------------
# ALGORITHM: REGIME-AWARE SCANNER
# -----------------------------------------------------------------------------
def run_internal_scanner(data_dict, sznl_map):
    results = []
    
    for ticker, df in data_dict.items():
        processed_df, rank_cols = calculate_internal_features(df, sznl_map, ticker)
        if processed_df.empty: continue
        
        # --- BASELINE CALCULATION (Full vs Recent) ---
        cutoff_date = processed_df.index[-1] - pd.DateOffset(years=RECENT_WINDOW_YEARS)
        
        full_df = processed_df
        recent_df = processed_df[processed_df.index >= cutoff_date]
        
        baselines_full = {}
        baselines_recent = {}
        
        for w in FWD_WINDOWS:
            col = f'FwdRet_{w}d_Sigma'
            baselines_full[w]   = full_df[col].mean()
            baselines_recent[w] = recent_df[col].mean() if len(recent_df) > 50 else baselines_full[w]
        
        # Get Current State
        current_state = processed_df.iloc[-1]
        
        feature_pairs = list(itertools.combinations(rank_cols, 2))
        
        for x_col, y_col in feature_pairs:
            
            curr_x = current_state[x_col]
            curr_y = current_state[y_col]
            
            # Discretize Current State
            x_tail = "UPPER" if curr_x > 75 else ("LOWER" if curr_x < 25 else "MID")
            y_tail = "UPPER" if curr_y > 75 else ("LOWER" if curr_y < 25 else "MID")
            
            if x_tail == "MID" or y_tail == "MID": continue
            
            # Create Masks
            if x_tail == "UPPER": mask_x = processed_df[x_col] >= 75
            else:                 mask_x = processed_df[x_col] <= 25
                
            if y_tail == "UPPER": mask_y = processed_df[y_col] >= 75
            else:                 mask_y = processed_df[y_col] <= 25
            
            valid_mask = mask_x & mask_y
            
            mask_full = valid_mask
            mask_recent = valid_mask & (processed_df.index >= cutoff_date)
            
            match_count_full = mask_full.sum()
            match_count_recent = mask_recent.sum()
            
            if match_count_full < 10: continue
            
            # --- CALCULATE ALPHA FOR BOTH REGIMES ---
            for fwd_w in FWD_WINDOWS:
                col_sigma = f'FwdRet_{fwd_w}d_Sigma'
                col_raw   = f'FwdRet_{fwd_w}d_Raw'
                
                # FULL STATS
                outcomes_sigma = processed_df.loc[mask_full, col_sigma]
                outcomes_raw   = processed_df.loc[mask_full, col_raw]
                
                real_count = outcomes_sigma.count()
                if real_count < 10: continue
                
                avg_sigma_full = outcomes_sigma.mean()
                alpha_full = avg_sigma_full - baselines_full[fwd_w]
                
                win_rate = (outcomes_raw > 0).sum() / real_count * 100
                exp_return = outcomes_raw.mean() * 100
                
                # RECENT STATS
                if match_count_recent < 3:
                    alpha_recent = np.nan
                    status = "👻 Ghost"
                else:
                    outcomes_sigma_recent = processed_df.loc[mask_recent, col_sigma]
                    avg_sigma_recent = outcomes_sigma_recent.mean()
                    alpha_recent = avg_sigma_recent - baselines_recent[fwd_w]
                    
                    if alpha_full > 0:
                        if alpha_recent < 0: status = "⚠️ Decaying"
                        elif alpha_recent > (alpha_full * 1.2): status = "🚀 Accelerating"
                        else: status = "✅ Stable"
                    else:
                        if alpha_recent > 0: status = "⚠️ Decaying"
                        elif alpha_recent < (alpha_full * 1.2): status = "🚀 Accelerating"
                        else: status = "✅ Stable"

                results.append({
                    "Ticker": ticker,
                    "Feature_X": x_col.replace("_Rank", ""),
                    "Feature_Y": y_col.replace("_Rank", ""),
                    "Current_Setup": f"X:{int(curr_x)} | Y:{int(curr_y)}",
                    "Fwd_Horizon": f"{fwd_w}d",
                    "History": real_count,
                    "Recent_Count": match_count_recent,
                    "Win_Rate": win_rate,
                    "Full_Excess": alpha_full,
                    "Recent_Excess": alpha_recent,
                    "Status": status,
                    "Exp_Return": exp_return
                })
                
    return pd.DataFrame(results)

def generate_ensemble(df_results, alpha_threshold=0.25):
    if df_results.empty: return pd.DataFrame()
    
    valid_status = ["✅ Stable", "🚀 Accelerating"]
    active_signals = df_results[df_results['Status'].isin(valid_status)].copy()
    
    if active_signals.empty: return pd.DataFrame()

    bull_mask = active_signals['Full_Excess'] > alpha_threshold
    bear_mask = (active_signals['Full_Excess'] < -alpha_threshold) & (active_signals['Exp_Return'] < 0)
    
    valid_signals = active_signals[bull_mask | bear_mask].copy()
    if valid_signals.empty: return pd.DataFrame()
    
    ensemble = valid_signals.groupby(['Ticker', 'Fwd_Horizon']).agg({
        'Full_Excess': 'sum',
        'Recent_Excess': 'mean',
        'Win_Rate': 'mean',
        'Exp_Return': 'mean',
        'Feature_X': 'count'
    }).reset_index()
    
    ensemble.rename(columns={
        'Full_Excess': 'Conviction_Score',
        'Feature_X': 'Signal_Breadth'
    }, inplace=True)
    
    ensemble['Abs_Score'] = ensemble['Conviction_Score'].abs()
    ensemble = ensemble.sort_values('Abs_Score', ascending=False).drop(columns=['Abs_Score'])
    
    return ensemble

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Regime-Aware Alpha")
    st.title("🔮 Regime-Aware Alpha Forecasts (5D Focused)")
    
    st.markdown("""
    **The "Stability" Test:**
    We compare the signal's performance over the **Last 5 Years** vs. **25 Years**.
    
    * ✅ **Stable:** Edge is consistent.
    * 🚀 **Accelerating:** Edge is getting stronger recently.
    * ⚠️ **Decaying:** Edge has disappeared in recent years.
    * 👻 **Ghost:** Signal hasn't triggered in the last 5 years.
    """)
    
    sznl_map, csv_tickers_full = load_seasonal_map()
    
    active_tickers = SELECTED_UNIVERSE
    active_tickers_str = ", ".join(active_tickers)

    with st.expander("⚙️ Forecast Settings", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"Scanning {len(active_tickers)} Tickers")
            st.text_area("Active Universe (Read-Only)", value=active_tickers_str, height=70, disabled=True)
        with col2:
            st.info("Filtering: |Excess| > 0.25σ") 
            run_btn = st.button("Run Regime Scan", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"Downloading data & Running Regime Checks..."):
            
            raw_data, fetch_time = get_batch_data(active_tickers)
            if not raw_data:
                st.error("No valid data found.")
                return
            st.success(f"✅ Data Refreshed: {fetch_time}")
            
            df_results = run_internal_scanner(raw_data, sznl_map)
            
            if df_results.empty:
                st.warning("No high-conviction internal setups found.")
                return
            
            # --- GENERATE ENSEMBLE ---
            ensemble_df = generate_ensemble(df_results, alpha_threshold=0.25)
            
            top_bulls = ensemble_df[ensemble_df['Conviction_Score'] > 0].head(10)
            top_bears = ensemble_df[ensemble_df['Conviction_Score'] < 0].head(10)
            
            # --- DISPLAY ENSEMBLE ---
            st.header("🏆 Top Validated Forecasts (Stable/Accelerating Only)")
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("🚀 Bullish Forecasts")
                if not top_bulls.empty:
                    st.dataframe(
                        top_bulls.style.format({
                            'Conviction_Score': "{:.2f}",
                            'Recent_Excess': "+{:.2f}σ",
                            'Win_Rate': "{:.1f}%",
                            'Exp_Return': "+{:.2f}%",
                        }).background_gradient(subset=['Conviction_Score'], cmap='Greens'),
                        use_container_width=True
                    )
                else:
                    st.info("No strong, stable bullish forecasts.")
                    
            with c2:
                st.subheader("🐻 Bearish Forecasts")
                if not top_bears.empty:
                    st.dataframe(
                        top_bears.style.format({
                            'Conviction_Score': "{:.2f}",
                            'Recent_Excess': "{:.2f}σ",
                            'Win_Rate': "{:.1f}%",
                            'Exp_Return': "{:.2f}%",
                        }).background_gradient(subset=['Conviction_Score'], cmap='Reds_r'),
                        use_container_width=True
                    )
                else:
                    st.info("No strong, stable bearish forecasts.")

            st.divider()
            
            # --- DISPLAY DETAILED TABLE ---
            st.subheader("🔎 Detailed Signal Regime Analysis")
            
            ALPHA_THRESHOLD = 0.25
            
            bullish_details = df_results[df_results['Full_Excess'] > ALPHA_THRESHOLD].sort_values(by="Full_Excess", ascending=False)
            
            bearish_details = df_results[
                (df_results['Full_Excess'] < -ALPHA_THRESHOLD) & 
                (df_results['Exp_Return'] < 0)
            ].sort_values(by="Full_Excess", ascending=True)

            tab1, tab2 = st.tabs(["Bullish Signals", "Bearish Signals"])
            
            cols = ["Ticker", "Fwd_Horizon", "Status", "Full_Excess", "Recent_Excess", "Feature_X", "Feature_Y", "History", "Recent_Count"]

            with tab1:
                st.dataframe(
                    bullish_details.head(100).style.format({
                        "Full_Excess": "+{:.2f}σ", "Recent_Excess": "{:.2f}σ"
                    }).map(lambda x: "color: red" if "Decaying" in str(x) else ("color: green" if "Accelerating" in str(x) else ""), subset=["Status"]),
                    use_container_width=True,
                    column_order=cols
                )
                
            with tab2:
                st.dataframe(
                    bearish_details.head(100).style.format({
                        "Full_Excess": "{:.2f}σ", "Recent_Excess": "{:.2f}σ"
                    }).map(lambda x: "color: red" if "Decaying" in str(x) else ("color: green" if "Accelerating" in str(x) else ""), subset=["Status"]),
                    use_container_width=True,
                    column_order=cols
                )
            
            # --- SCATTER SUMMARY ---
            st.divider()
            fig = px.scatter(
                df_results, 
                x="Full_Excess", 
                y="Recent_Excess", 
                hover_data=["Ticker", "Status", "Feature_X", "Feature_Y"],
                color="Status", 
                color_discrete_map={
                    "✅ Stable": "blue",
                    "🚀 Accelerating": "green",
                    "⚠️ Decaying": "red",
                    "👻 Ghost": "gray"
                },
                title="Regime Stability Map: Historic Alpha vs Recent Alpha"
            )
            fig.add_shape(type="line", x0=-2, y0=-2, x1=2, y1=2, line=dict(color="gray", dash="dash"))
            
            fig.update_layout(xaxis_title="Full History Alpha (25y)", yaxis_title="Recent Alpha (5y)")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
