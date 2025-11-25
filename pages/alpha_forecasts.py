import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import plotly.express as px
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import itertools

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CSV_PATH = "seasonal_ranks.csv"
FWD_WINDOWS = [5, 10, 21] # We focus on short-to-medium term alpha

# Fallback tickers if CSV is missing or empty
FALLBACK_TICKERS = ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "AMD", "MSFT", "GOOG", "AMZN", "META", "NFLX"]

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

    # Extract unique tickers from the CSV
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
    """
    Downloads MAX history for internal scanning.
    """
    tickers = list(set([t.strip().upper() for t in ticker_list]))
    
    if not tickers:
        return {}, ""
    
    data_dict = {}
    
    progress_bar = st.progress(0)
    for i, t in enumerate(tickers):
        try:
            df = yf.download(t, period="max", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            
            # Filter to last 25 years
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
# FEATURE ENGINEERING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def calculate_internal_features(df, sznl_map, ticker):
    """
    Calculates the full suite of internal ranks + Forward Sigmas.
    """
    df = df.copy()
    
    # 1. Base Volatility (For Sigma Norm)
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_Daily'] = df['LogRet'].rolling(21).std()

    # 2. Variable Calculation (Same as Heatmap Tool)
    
    # Returns
    for w in [5, 10, 21, 63, 126, 252]:
        df[f'Ret_{w}d'] = df['Close'].pct_change(w)

    # Realized Vol
    for w in [21, 63]:
        df[f'RealVol_{w}d'] = df['LogRet'].rolling(w).std() * np.sqrt(252) * 100.0
    
    # Vol Change
    df['VolChange_5d']  = df['RealVol_21d'] - df['RealVol_63d'].shift(5) # Approx
    df['VolChange_21d'] = df['RealVol_21d'] - df['RealVol_63d']

    # Volume Ratios
    for w in [5, 21]:
        df[f'VolRatio_{w}d'] = df['Volume'].rolling(w).mean() / df['Volume'].rolling(63).mean()

    # Seasonality
    df['Seasonal'] = get_sznl_val_series(ticker, df.index, sznl_map)

    # 3. RANK TRANSFORMATION
    # These are the "X" and "Y" variables we will scan
    vars_to_rank = [
        'Seasonal',
        'Ret_5d', 'Ret_10d', 'Ret_21d', 'Ret_63d', 'Ret_252d',
        'RealVol_21d', 'RealVol_63d', 
        'VolChange_5d', 'VolChange_21d',
        'VolRatio_5d', 'VolRatio_21d'
    ]
    
    rank_cols = []
    for v in vars_to_rank:
        rank_col = v + '_Rank'
        # Seasonality is already 0-100, others need ranking
        if v == 'Seasonal':
            df[rank_col] = df[v] # Already ranked 0-100
        else:
            df[rank_col] = df[v].expanding(min_periods=252).rank(pct=True) * 100.0
        rank_cols.append(rank_col)

    # 4. FORWARD OUTCOMES (Z-Axis)
    # Calculate Sigma Returns for 5, 10, 21 days
    for w in FWD_WINDOWS:
        col_raw = f'FwdRet_{w}d_Raw'
        col_sigma = f'FwdRet_{w}d_Sigma'
        
        df[col_raw] = (df['Close'].shift(-w) / df['Close']) - 1.0
        expected_move = df['Vol_Daily'] * np.sqrt(w)
        df[col_sigma] = df[col_raw] / expected_move

    df = df.dropna(subset=['Vol_Daily'] + rank_cols)
    
    return df, rank_cols

# -----------------------------------------------------------------------------
# ALGORITHM: INTERNAL FRACTAL SCANNER
# -----------------------------------------------------------------------------
def run_internal_scanner(data_dict, sznl_map):
    results = []
    
    # We iterate through every ticker
    for ticker, df in data_dict.items():
        
        # 1. Calc Features for this ticker
        processed_df, rank_cols = calculate_internal_features(df, sznl_map, ticker)
        if processed_df.empty: continue
        
        # 2. Pre-calculate Unconditional Baselines (for Alpha)
        baselines = {}
        for w in FWD_WINDOWS:
            baselines[w] = processed_df[f'FwdRet_{w}d_Sigma'].mean()
        
        # 3. Get Current Ranks (The "Now")
        current_state = processed_df.iloc[-1]
        
        # 4. Generate All Combinations of Feature Pairs (X, Y)
        # itertools.combinations(rank_cols, 2) gives all unique pairs
        feature_pairs = list(itertools.combinations(rank_cols, 2))
        
        for x_col, y_col in feature_pairs:
            
            # --- FILTER 1: EXTREMITY CHECK (Double Tail) ---
            curr_x = current_state[x_col]
            curr_y = current_state[y_col]
            
            # Tail Logic (Same as Cross Asset: 25/75)
            x_tail = "UPPER" if curr_x > 75 else ("LOWER" if curr_x < 25 else "MID")
            y_tail = "UPPER" if curr_y > 75 else ("LOWER" if curr_y < 25 else "MID")
            
            # Optimization: If either is MID, skip immediately
            if x_tail == "MID" or y_tail == "MID": continue
            
            # --- FILTER 2: HISTORICAL MATCHING ---
            # Create masks based on tail direction
            if x_tail == "UPPER": mask_x = processed_df[x_col] >= curr_x
            else:                 mask_x = processed_df[x_col] <= curr_x
                
            if y_tail == "UPPER": mask_y = processed_df[y_col] >= curr_y
            else:                 mask_y = processed_df[y_col] <= curr_y
            
            valid_mask = mask_x & mask_y
            match_count = valid_mask.sum()
            
            if match_count < 10: continue
            
            # --- FILTER 3: CHECK FORWARD OUTCOMES ---
            for fwd_w in FWD_WINDOWS:
                
                # Get outcomes for the matched dates
                outcomes_sigma = processed_df.loc[valid_mask, f'FwdRet_{fwd_w}d_Sigma']
                outcomes_raw   = processed_df.loc[valid_mask, f'FwdRet_{fwd_w}d_Raw']
                
                real_count = outcomes_sigma.count() # Handle potential NaNs at end of df
                if real_count < 10: continue
                
                cond_avg_sigma = outcomes_sigma.mean()
                
                # ALPHA = Conditional - Unconditional
                alpha_sigma = cond_avg_sigma - baselines[fwd_w]
                
                avg_raw_pct = outcomes_raw.mean() * 100
                win_rate    = (outcomes_raw > 0).sum() / real_count * 100
                
                results.append({
                    "Ticker": ticker,
                    "Feature_X": x_col.replace("_Rank", ""),
                    "Feature_Y": y_col.replace("_Rank", ""),
                    "Current_Setup": f"X:{int(curr_x)} | Y:{int(curr_y)}",
                    "Fwd_Horizon": f"{fwd_w}d",
                    "History": real_count,
                    "Win_Rate": win_rate,
                    "Excess_Sigma": alpha_sigma,
                    "Exp_Return": avg_raw_pct
                })
                
    return pd.DataFrame(results)

def generate_ensemble(df_results, alpha_threshold=0.25):
    """
    Aggregates signals for the final dashboard.
    """
    if df_results.empty: return pd.DataFrame()
    
    # 1. Bullish: Excess > Threshold
    bull_mask = df_results['Excess_Sigma'] > alpha_threshold
    
    # 2. Bearish: Excess < -Threshold AND Negative Returns (Safety)
    bear_mask = (df_results['Excess_Sigma'] < -alpha_threshold) & (df_results['Exp_Return'] < 0)
    
    valid_signals = df_results[bull_mask | bear_mask].copy()
    if valid_signals.empty: return pd.DataFrame()
    
    # 3. Aggregate
    ensemble = valid_signals.groupby(['Ticker', 'Fwd_Horizon']).agg({
        'Excess_Sigma': 'sum',
        'Win_Rate': 'mean',
        'Exp_Return': 'mean',
        'Feature_X': 'count' # Just counting rows gives breadth of combinations
    }).reset_index()
    
    ensemble.rename(columns={
        'Excess_Sigma': 'Conviction_Score',
        'Feature_X': 'Signal_Breadth'
    }, inplace=True)
    
    ensemble['Abs_Score'] = ensemble['Conviction_Score'].abs()
    ensemble = ensemble.sort_values('Abs_Score', ascending=False).drop(columns=['Abs_Score'])
    
    return ensemble

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Internal Alpha Forecasts")
    st.title("🔮 Internal Alpha Forecasts")
    
    st.markdown("""
    **The "Self-Scanner":**
    Instead of checking correlations between *different* assets, this tool scans **internal relationships** within each stock.
    
    * **Logic:** Checks combinations of **Seasonality, Trailing Returns, Volatility, and Volume**.
    * **Trigger:** Flags setups where **two internal metrics are simultaneously extreme** (Double Tail).
    * **Alpha:** Calculates if the stock behaves differently (Excess Sigma) in this specific state compared to its baseline.
    """)
    
    # Load tickers from CSV
    sznl_map, csv_tickers = load_seasonal_map()
    
    # Use CSV tickers if available, otherwise fallback
    active_tickers = csv_tickers if csv_tickers else FALLBACK_TICKERS
    active_tickers_str = ", ".join(active_tickers)

    with st.expander("⚙️ Forecast Settings", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"Scanning {len(active_tickers)} Tickers found in seasonal_ranks.csv")
            st.text_area("Active Universe (Read-Only)", value=active_tickers_str, height=70, disabled=True)
        with col2:
            st.info("Filtering: |Excess| > 0.25σ") 
            run_btn = st.button("Run Alpha Forecast", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"Downloading data & Running internal combinations for {len(active_tickers)} tickers..."):
            
            # A. Process Data
            raw_data, fetch_time = get_batch_data(active_tickers)
            if not raw_data:
                st.error("No valid data found.")
                return
            st.success(f"✅ Data Refreshed: {fetch_time}")
            
            # B. Run Algorithm
            df_results = run_internal_scanner(raw_data, sznl_map)
            
            if df_results.empty:
                st.warning("No high-conviction internal setups found.")
                return
            
            # --- GENERATE ENSEMBLE ---
            ensemble_df = generate_ensemble(df_results, alpha_threshold=0.25)
            
            # Top Lists
            top_bulls = ensemble_df[ensemble_df['Conviction_Score'] > 0].head(10)
            top_bears = ensemble_df[ensemble_df['Conviction_Score'] < 0].head(10)
            
            # --- DISPLAY ENSEMBLE ---
            st.header("🏆 Top High-Conviction Forecasts")
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("🚀 Bullish Forecasts")
                if not top_bulls.empty:
                    st.dataframe(
                        top_bulls.style.format({
                            'Conviction_Score': "{:.2f}",
                            'Win_Rate': "{:.1f}%",
                            'Exp_Return': "+{:.2f}%",
                        }).background_gradient(subset=['Conviction_Score'], cmap='Greens'),
                        use_container_width=True
                    )
                else:
                    st.info("No strong bullish forecasts.")
                    
            with c2:
                st.subheader("🐻 Bearish Forecasts")
                st.caption("Safety: Only Includes Negative Expectancy")
                if not top_bears.empty:
                    st.dataframe(
                        top_bears.style.format({
                            'Conviction_Score': "{:.2f}",
                            'Win_Rate': "{:.1f}%",
                            'Exp_Return': "{:.2f}%",
                        }).background_gradient(subset=['Conviction_Score'], cmap='Reds_r'),
                        use_container_width=True
                    )
                else:
                    st.info("No strong bearish forecasts.")

            st.divider()
            
            # --- DISPLAY DETAILED TABLE ---
            st.subheader("🔎 Detailed Signal Breakdown")
            
            ALPHA_THRESHOLD = 0.25
            
            bullish_details = df_results[df_results['Excess_Sigma'] > ALPHA_THRESHOLD].sort_values(by="Excess_Sigma", ascending=False)
            
            # Bearish with Safety Filter
            bearish_details = df_results[
                (df_results['Excess_Sigma'] < -ALPHA_THRESHOLD) & 
                (df_results['Exp_Return'] < 0)
            ].sort_values(by="Excess_Sigma", ascending=True)

            tab1, tab2 = st.tabs(["Bullish Details", "Bearish Details"])
            
            with tab1:
                st.dataframe(
                    bullish_details.head(100).style.format({
                        "Win_Rate": "{:.1f}%", "Excess_Sigma": "+{:.2f}σ", "Exp_Return": "+{:.2f}%"
                    }).background_gradient(subset=["Excess_Sigma"], cmap="Greens", vmin=0, vmax=1.0),
                    use_container_width=True,
                    column_order=["Ticker", "Fwd_Horizon", "Feature_X", "Feature_Y", "Excess_Sigma", "Exp_Return", "Win_Rate", "History", "Current_Setup"]
                )
                
            with tab2:
                st.dataframe(
                    bearish_details.head(100).style.format({
                        "Win_Rate": "{:.1f}%", "Excess_Sigma": "{:.2f}σ", "Exp_Return": "{:.2f}%"
                    }).background_gradient(subset=["Excess_Sigma"], cmap="Reds_r", vmin=-1.0, vmax=0),
                    use_container_width=True,
                    column_order=["Ticker", "Fwd_Horizon", "Feature_X", "Feature_Y", "Excess_Sigma", "Exp_Return", "Win_Rate", "History", "Current_Setup"]
                )
            
            # --- SCATTER SUMMARY ---
            st.divider()
            fig = px.scatter(
                df_results, 
                x="Win_Rate", 
                y="Exp_Return", 
                hover_data=["Ticker", "Feature_X", "Feature_Y", "Current_Setup", "Excess_Sigma", "History"],
                color="Excess_Sigma", 
                color_continuous_scale="RdBu",
                title="Forecast Map: Win Rate vs Expected Return %"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
