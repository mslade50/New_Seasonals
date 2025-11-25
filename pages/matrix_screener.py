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
SPY, QQQ, IWM, TLT, GLD, USO, UUP, HYG, XLF, XLE, XLK, XBI, SMH, ARKK, BTC-USD,
JPM, AAPL, GOOG, XOM, NVDA, TSLA, KO, UVXY, XLP, XLV, XLU, UNG, MSFT, WMT, AMD
"""

# -----------------------------------------------------------------------------
# DATA ENGINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=True)
def get_batch_data(ticker_list):
    """
    Downloads MAX history to ensure 25y window if available.
    """
    tickers = [t.strip().upper() for t in ticker_list.replace("\n", "").split(",") if t.strip()]
    tickers = list(set(tickers)) 
    
    if not tickers:
        return {}
    
    data_dict = {}
    
    progress_bar = st.progress(0)
    for i, t in enumerate(tickers):
        try:
            # Changed to "max" to cover the 25y requirement
            df = yf.download(t, period="max", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            
            # Filter to last 25 years to keep calculation relevant but deep
            start_date = pd.Timestamp.now() - pd.DateOffset(years=25)
            df = df[df.index >= start_date]
            
            if len(df) > 252: 
                data_dict[t] = df
        except:
            pass
        progress_bar.progress((i + 1) / len(tickers))
    
    progress_bar.empty()
    return data_dict

@st.cache_data(show_spinner=False)
def calculate_features(data_dict):
    """
    Calculates:
    1. 21d Trailing Rank
    2. 10d Vol-Normalized Forward Return (Sigma)
    """
    processed = {}
    
    for ticker, df in data_dict.items():
        df = df.copy()
        
        df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 1. Feature: 21d Trailing Return
        df['Ret_21d'] = df['Close'].pct_change(21)
        
        # 2. Volatility (21d Annualized) for Normalization
        # We need the daily vol to project 10 days out
        # Vol_Daily = Rolling Std Dev of Log Returns
        df['Vol_Daily'] = df['LogRet'].rolling(21).std()
        
        # 3. Target: 10d Forward Return (Raw)
        # Shift -10 to see 10 days into the future
        df['Fwd_Close'] = df['Close'].shift(-10)
        df['FwdRet_10d_Raw'] = (df['Fwd_Close'] / df['Close']) - 1.0
        
        # 4. Vol-Normalization (Z-Score)
        # Expected Move (1 Sigma) = Vol_Daily * sqrt(10)
        # Return_Sigma = Raw_Return / Expected_Move
        expected_move = df['Vol_Daily'] * np.sqrt(10)
        df['FwdRet_Sigma'] = df['FwdRet_10d_Raw'] / expected_move
        
        # 5. Rank Transformation
        df['Rank_21d'] = df['Ret_21d'].expanding(min_periods=252).rank(pct=True) * 100
        
        # Clean up
        df = df.dropna(subset=['Rank_21d', 'Vol_Daily'])
        processed[ticker] = df[['Close', 'Rank_21d', 'FwdRet_Sigma']]
        
    return processed

def run_scanner(processed_data):
    """
    Iterates through pairs.
    Logic: "Cumulative Tail Analysis"
    - If Current Rank is Extreme (e.g. 90), look at history >= 90.
    """
    results = []
    tickers = list(processed_data.keys())
    
    # 1. Get Current State
    current_states = {}
    for t in tickers:
        try:
            current_states[t] = processed_data[t].iloc[-1]['Rank_21d']
        except:
            current_states[t] = np.nan

    # 2. Pairwise Matrix Scan
    for target in tickers:
        t_df = processed_data[target]
        curr_t_rank = current_states.get(target, np.nan)
        if np.isnan(curr_t_rank): continue

        for signal in tickers:
            if target == signal: continue 
            
            s_df = processed_data[signal]
            curr_s_rank = current_states.get(signal, np.nan)
            if np.isnan(curr_s_rank): continue
            
            # --- FILTER 1: EXTREMITY CHECK ---
            # Both must be outside 20-80
            t_tail = "UPPER" if curr_t_rank > 80 else ("LOWER" if curr_t_rank < 20 else "MID")
            s_tail = "UPPER" if curr_s_rank > 80 else ("LOWER" if curr_s_rank < 20 else "MID")
            
            if t_tail == "MID" or s_tail == "MID":
                continue

            # --- ALIGNMENT ---
            aligned = pd.concat([
                t_df['FwdRet_Sigma'].rename("Outcome"),
                t_df['Rank_21d'].rename("Target_Rank"),
                s_df['Rank_21d'].rename("Signal_Rank")
            ], axis=1, join='inner')
            
            history = aligned.dropna(subset=["Outcome"])
            if history.empty: continue

            # --- FILTER 2: CUMULATIVE TAIL LOGIC ---
            # "Check all boxes that are more extreme"
            
            if t_tail == "UPPER":
                mask_t = history['Target_Rank'] >= curr_t_rank
            else:
                mask_t = history['Target_Rank'] <= curr_t_rank
                
            if s_tail == "UPPER":
                mask_s = history['Signal_Rank'] >= curr_s_rank
            else:
                mask_s = history['Signal_Rank'] <= curr_s_rank
            
            matches = history[mask_t & mask_s]
            count = len(matches)
            
            # Require minimum samples (relaxed for Super Extreme, strict otherwise)
            # Since we are using cumulative tails, samples should naturally be higher than radius
            if count < 10: continue

            # --- STATISTICS ---
            # Outcome is now in SIGMA. 
            avg_sigma = matches['Outcome'].mean()
            
            # Win Rate is strictly > 0 raw return. 
            # (Note: Sigma > 0 implies Raw Return > 0)
            win_rate = (matches['Outcome'] > 0).sum() / count * 100
            
            results.append({
                "Target": target,
                "Signal_Ticker": signal,
                "Current_Setup": f"T:{int(curr_t_rank)} | S:{int(curr_s_rank)}",
                "History_Count": count,
                "Win_Rate": win_rate,
                "Avg_Return_Sigma": avg_sigma,
                "Target_Tail": t_tail,
                "Signal_Tail": s_tail,
                # For Scatter Plot Coloring
                "Target_Rank": curr_t_rank 
            })

    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Vol-Adj Matrix Screener")
    st.title("⚡ Vol-Normalized Tail Screener")
    
    st.markdown("""
    **New Logic Applied:**
    1. **Vol-Normalization:** Returns are measured in **Standard Deviations (σ)**. Apples-to-Apples comparison (e.g. UVXY vs TLT).
    2. **Tail Logic:** We analyze all historical instances that were **more extreme** than today. (e.g., if Rank is 95, we look at 95-100).
    """)
    
    # 1. INPUTS
    with st.expander("⚙️ Screener Settings", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker_input = st.text_area("Universe", value=DEFAULT_TICKERS, height=100)
        with col2:
            st.info("Logic: Cumulative Tail Analysis")
            run_btn = st.button("Run Vol-Adj Scan", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Downloading 25y history & Normalizing Volatility..."):
            # A. Process Data
            raw_data = get_batch_data(ticker_input)
            if not raw_data:
                st.error("No valid data found.")
                return
            
            processed = calculate_features(raw_data)
            
            # B. Run Algorithm
            df_results = run_scanner(processed)
            
            if df_results.empty:
                st.warning("No dual-tail opportunities found.")
                return
            
            # C. Formatting & Display
            st.divider()
            
            # Sort by absolute Sigma strength
            bullish = df_results[df_results['Avg_Return_Sigma'] > 0].sort_values(by="Avg_Return_Sigma", ascending=False)
            bearish = df_results[df_results['Avg_Return_Sigma'] < 0].sort_values(by="Avg_Return_Sigma", ascending=True)

            # --- DISPLAY BULLISH ---
            st.subheader(f"🟢 Top Bullish Signals (High +σ)")
            st.markdown("Expectancy measured in **Standard Deviations** of a 10-day move.")
            st.dataframe(
                bullish.head(20).style.format({
                    "Win_Rate": "{:.1f}%",
                    "Avg_Return_Sigma": "+{:.2f}σ",
                    "Target_Rank": "{:.0f}"
                }).background_gradient(subset=["Win_Rate"], cmap="Greens", vmin=50, vmax=80),
                use_container_width=True,
                column_order=["Target", "Signal_Ticker", "Avg_Return_Sigma", "Win_Rate", "History_Count", "Current_Setup"]
            )

            # --- DISPLAY BEARISH ---
            st.subheader(f"🔴 Top Bearish Signals (High -σ)")
            st.dataframe(
                bearish.head(20).style.format({
                    "Win_Rate": "{:.1f}%",
                    "Avg_Return_Sigma": "{:.2f}σ",
                    "Target_Rank": "{:.0f}"
                }).background_gradient(subset=["Win_Rate"], cmap="Reds_r", vmin=20, vmax=50),
                use_container_width=True,
                column_order=["Target", "Signal_Ticker", "Avg_Return_Sigma", "Win_Rate", "History_Count", "Current_Setup"]
            )
            
            # --- SCATTER SUMMARY ---
            st.divider()
            st.subheader("Map of Vol-Adjusted Edge")
            
            fig = px.scatter(
                df_results, 
                x="Win_Rate", 
                y="Avg_Return_Sigma", 
                hover_data=["Target", "Signal_Ticker", "History_Count", "Current_Setup"],
                color="Target_Rank", 
                color_continuous_scale="RdBu",
                title="Screener Results: Win Rate vs Expected Sigma"
            )
            fig.add_vline(x=50, line_dash="dash", line_color="gray")
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(yaxis_title="Expected Move (Sigma)")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
