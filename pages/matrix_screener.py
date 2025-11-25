import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import plotly.express as px

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DEFAULT_TICKERS = """
SPY, QQQ, IWM, DIA, TLT, GLD, USO, UUP, HYG, XLF, XLE, XLK, XBI, SMH, ARKK, BTC-USD,
JPM, AAPL, GOOG, XOM, NVDA, TSLA, KO, UVXY, XLP, XLV, XLU, UNG, MSFT, WMT, AMD, ITA, SLV, CEF
"""

# The Matrix Dimensions
TRAIL_WINDOWS = [2, 5, 10, 21, 63]
FWD_WINDOWS   = [2, 5, 10, 21, 63]

# -----------------------------------------------------------------------------
# DATA ENGINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=True)
def get_batch_data(ticker_list):
    """
    Downloads MAX history (last 25y).
    Returns: (data_dict, timestamp_string)
    """
    tickers = [t.strip().upper() for t in ticker_list.replace("\n", "").split(",") if t.strip()]
    tickers = list(set(tickers)) 
    
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

@st.cache_data(show_spinner=False)
def calculate_features(data_dict):
    """
    Calculates ALL Trailing Ranks and ALL Forward Sigmas.
    """
    processed = {}
    
    for ticker, df in data_dict.items():
        df = df.copy()
        
        # 1. Base Volatility (21d Annualized) - The "Standard Ruler"
        df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_Daily'] = df['LogRet'].rolling(21).std()
        
        # 2. Calculate Trailing Ranks (The Setup)
        for w in TRAIL_WINDOWS:
            # Trailing Return
            col_ret = f'Ret_{w}d'
            df[col_ret] = df['Close'].pct_change(w)
            
            # Rank (Expanding Window)
            col_rank = f'Rank_{w}d'
            df[col_rank] = df[col_ret].expanding(min_periods=252).rank(pct=True) * 100

        # 3. Calculate Forward Outcomes (The Payoff)
        for w in FWD_WINDOWS:
            # Raw Forward Return
            # We look W days into the future
            col_fwd_raw = f'FwdRet_{w}d_Raw'
            df[col_fwd_raw] = (df['Close'].shift(-w) / df['Close']) - 1.0
            
            # Vol-Normalized Return (Sigma)
            # Expected Move = DailyVol * sqrt(Days)
            col_fwd_sigma = f'FwdRet_{w}d_Sigma'
            expected_move = df['Vol_Daily'] * np.sqrt(w)
            df[col_fwd_sigma] = df[col_fwd_raw] / expected_move

        # Cleanup: Keep only what we need to save memory
        cols_to_keep = ['Close', 'Vol_Daily'] + \
                       [f'Rank_{w}d' for w in TRAIL_WINDOWS] + \
                       [f'FwdRet_{w}d_Raw' for w in FWD_WINDOWS] + \
                       [f'FwdRet_{w}d_Sigma' for w in FWD_WINDOWS]
                       
        df = df.dropna(subset=['Vol_Daily']) # Ensure vol exists
        processed[ticker] = df[cols_to_keep]
        
    return processed

def run_scanner(processed_data):
    """
    Iterates through Pairs -> Trailing Windows -> Forward Windows.
    Applies Hierarchical Logic & Cumulative Tails.
    """
    results = []
    tickers = list(processed_data.keys())
    
    # Pre-fetch current ranks for all windows to avoid loc lookups inside loops
    current_ranks = {}
    for t in tickers:
        current_ranks[t] = {}
        try:
            last_row = processed_data[t].iloc[-1]
            for w in TRAIL_WINDOWS:
                current_ranks[t][w] = last_row[f'Rank_{w}d']
        except:
            pass

    # --- PAIR LOOP ---
    for target in tickers:
        t_df = processed_data[target]

        for signal in tickers:
            if target == signal: continue 
            
            # --- HIERARCHICAL LOGIC ---
            allowed_signals = ["SPY", "TLT"]
            if target == "SPY": allowed_signals = ["TLT", "USO", "UUP", "GLD"]
            elif target == "XOM": allowed_signals = ["SPY", "TLT", "USO"]
            elif target == "JPM": allowed_signals = ["SPY", "TLT", "XLF"]
            elif target in ["NVDA", "AMD"]: allowed_signals = ["SPY", "TLT", "SMH"]
            elif target == "SLV": allowed_signals = ["SPY", "TLT", "GLD"]
            elif target == "GLD": allowed_signals = ["SPY", "TLT", "SLV"]
            elif target == "CEF": allowed_signals = ["SPY", "TLT", "GLD", "SLV"]
            elif target == "UNG": allowed_signals = ["SPY", "TLT", "USO"]
            elif target == "USO": allowed_signals = ["SPY", "TLT", "UNG"]
            
            if signal not in allowed_signals: continue
            
            s_df = processed_data[signal]

            # --- TRAILING WINDOW LOOP (The Setup) ---
            for trail_w in TRAIL_WINDOWS:
                curr_t_rank = current_ranks.get(target, {}).get(trail_w, np.nan)
                curr_s_rank = current_ranks.get(signal, {}).get(trail_w, np.nan)
                
                if np.isnan(curr_t_rank) or np.isnan(curr_s_rank): continue
                
                # Check Extremity (Double Tail) - Slightly relaxed for multi-timeframe detection
                t_tail = "UPPER" if curr_t_rank > 75 else ("LOWER" if curr_t_rank < 25 else "MID")
                s_tail = "UPPER" if curr_s_rank > 75 else ("LOWER" if curr_s_rank < 25 else "MID")
                
                if t_tail == "MID" or s_tail == "MID": continue

                # Prepare History for this Window
                t_rank_col = f'Rank_{trail_w}d'
                s_rank_col = f'Rank_{trail_w}d'
                
                # We align indices first to find matching dates
                # Note: We don't join forward columns yet to save memory/speed
                common_idx = t_df.index.intersection(s_df.index)
                
                # Get historical ranks
                hist_t_ranks = t_df.loc[common_idx, t_rank_col]
                hist_s_ranks = s_df.loc[common_idx, s_rank_col]
                
                # Create Masks for Cumulative Tails
                if t_tail == "UPPER": mask_t = hist_t_ranks >= curr_t_rank
                else:                 mask_t = hist_t_ranks <= curr_t_rank
                    
                if s_tail == "UPPER": mask_s = hist_s_ranks >= curr_s_rank
                else:                 mask_s = hist_s_ranks <= curr_s_rank
                
                # The Valid Historical Dates
                valid_dates_mask = mask_t & mask_s
                valid_count = valid_dates_mask.sum()
                
                if valid_count < 10: continue
                
                # --- FORWARD WINDOW LOOP (The Payoff) ---
                # Now we check all future outcomes for these valid dates
                for fwd_w in FWD_WINDOWS:
                    # Get the outcomes for the Target only (Signal outcomes don't matter, only its setup)
                    outcome_sigma_col = f'FwdRet_{fwd_w}d_Sigma'
                    outcome_raw_col   = f'FwdRet_{fwd_w}d_Raw'
                    
                    # Extract values using the boolean mask
                    # Shift logic is handled in pre-calc, so we just grab the row at the date of the signal
                    outcomes_sigma = t_df.loc[common_idx][valid_dates_mask][outcome_sigma_col].dropna()
                    outcomes_raw   = t_df.loc[common_idx][valid_dates_mask][outcome_raw_col].dropna()
                    
                    real_count = len(outcomes_sigma)
                    if real_count < 10: continue

                    avg_sigma = outcomes_sigma.mean()
                    avg_raw   = outcomes_raw.mean() * 100
                    win_rate  = (outcomes_raw > 0).sum() / real_count * 100
                    
                    results.append({
                        "Target": target,
                        "Signal": signal,
                        "Setup_Window": f"{trail_w}d",
                        "Fwd_Window": f"{fwd_w}d",
                        "Setup_Ranks": f"T:{int(curr_t_rank)}|S:{int(curr_s_rank)}",
                        "History": real_count,
                        "Win_Rate": win_rate,
                        "Exp_Sigma": avg_sigma,
                        "Exp_Return": avg_raw,
                    })

    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Multi-Fractal Matrix")
    st.title("⚡ Multi-Timeframe Matrix")
    
    st.markdown("""
    **Scanning 25 Combinations per Pair:**
    * **Trailing (Setup):** 2d, 5d, 10d, 21d, 63d Ranks
    * **Forward (Payoff):** 2d, 5d, 10d, 21d, 63d Returns
    * **Logic:** Hierarchical Relationships + Cumulative Tail Analysis.
    """)
    
    # 1. INPUTS
    with st.expander("⚙️ Screener Settings", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker_input = st.text_area("Universe", value=DEFAULT_TICKERS, height=100)
        with col2:
            st.info("Filtering: |Sigma| > 0.35σ") # Bumped slightly for multi-window noise
            run_btn = st.button("Run Multi-Fractal Scan", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Calculating 25 Timeframe Combinations per pair..."):
            # A. Process Data
            raw_data, fetch_time = get_batch_data(ticker_input)
            if not raw_data:
                st.error("No valid data found.")
                return
            st.success(f"✅ Data Refreshed: {fetch_time}")
            
            processed = calculate_features(raw_data)
            
            # B. Run Algorithm
            df_results = run_scanner(processed)
            
            if df_results.empty:
                st.warning("No opportunities found matching criteria.")
                return
            
            # C. Formatting & Display
            st.divider()
            
            # Filter Threshold (Stricter because we have 25x more data points)
            SIGMA_THRESHOLD = 0.35
            
            bullish = df_results[df_results['Exp_Sigma'] > SIGMA_THRESHOLD].sort_values(by="Exp_Sigma", ascending=False)
            bearish = df_results[df_results['Exp_Sigma'] < -SIGMA_THRESHOLD].sort_values(by="Exp_Sigma", ascending=True)

            # --- DISPLAY BULLISH ---
            st.subheader(f"🟢 Top Bullish Fractals (> +{SIGMA_THRESHOLD}σ)")
            if not bullish.empty:
                st.dataframe(
                    bullish.head(50).style.format({
                        "Win_Rate": "{:.1f}%",
                        "Exp_Sigma": "+{:.2f}σ",
                        "Exp_Return": "+{:.2f}%",
                    }).background_gradient(subset=["Exp_Sigma"], cmap="Greens", vmin=0, vmax=1.5),
                    use_container_width=True,
                    column_order=["Target", "Signal", "Setup_Window", "Fwd_Window", "Exp_Return", "Exp_Sigma", "Win_Rate", "History", "Setup_Ranks"]
                )
            else:
                st.info(f"No signals found > {SIGMA_THRESHOLD}σ")

            # --- DISPLAY BEARISH ---
            st.subheader(f"🔴 Top Bearish Fractals (< -{SIGMA_THRESHOLD}σ)")
            if not bearish.empty:
                st.dataframe(
                    bearish.head(50).style.format({
                        "Win_Rate": "{:.1f}%",
                        "Exp_Sigma": "{:.2f}σ",
                        "Exp_Return": "{:.2f}%",
                    }).background_gradient(subset=["Exp_Sigma"], cmap="Reds_r", vmin=-1.5, vmax=0),
                    use_container_width=True,
                    column_order=["Target", "Signal", "Setup_Window", "Fwd_Window", "Exp_Return", "Exp_Sigma", "Win_Rate", "History", "Setup_Ranks"]
                )
            else:
                st.info(f"No signals found < -{SIGMA_THRESHOLD}σ")
            
            # --- SCATTER SUMMARY ---
            st.divider()
            st.subheader("Map of Timeframe Opportunities")
            
            fig = px.scatter(
                df_results, 
                x="Win_Rate", 
                y="Exp_Return", 
                # Added Tooltip info for Setup/Fwd Windows
                hover_data=["Target", "Signal", "Setup_Window", "Fwd_Window", "Exp_Sigma", "History", "Setup_Ranks"],
                color="Exp_Sigma", 
                color_continuous_scale="RdBu",
                title="Fractal Map: Win Rate vs Expected Return %"
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(yaxis_title="Avg Return (%)")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
