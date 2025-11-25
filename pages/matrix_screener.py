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
    Downloads and processes data for ALL tickers at once.
    Returns a Dictionary of DataFrames {ticker: df}.
    """
    tickers = [t.strip().upper() for t in ticker_list.replace("\n", "").split(",") if t.strip()]
    tickers = list(set(tickers)) # Remove duplicates
    
    if not tickers:
        return {}
    
    data_dict = {}
    
    progress_bar = st.progress(0)
    for i, t in enumerate(tickers):
        try:
            df = yf.download(t, period="25y", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            
            if len(df) > 252: # Ensure enough history
                data_dict[t] = df
        except:
            pass
        progress_bar.progress((i + 1) / len(tickers))
    
    progress_bar.empty()
    return data_dict

@st.cache_data(show_spinner=False)
def calculate_features(data_dict):
    """
    Calculates 21d Trailing Rank and 10d Forward Return
    """
    processed = {}
    
    for ticker, df in data_dict.items():
        df = df.copy()
        
        # 1. Feature: 21d Trailing Return
        df['Ret_21d'] = df['Close'].pct_change(21)
        
        # 2. Target: 10d Forward Return
        df['FwdRet_10d'] = (df['Close'].shift(-10) / df['Close'] - 1) * 100
        
        # 3. Rank Transformation (Expanding Window)
        df['Rank_21d'] = df['Ret_21d'].expanding(min_periods=252).rank(pct=True) * 100
        
        # Drop initial NaN
        df = df.dropna(subset=['Rank_21d'])
        processed[ticker] = df[['Close', 'Rank_21d', 'FwdRet_10d']]
        
    return processed

def run_scanner(processed_data, rank_radius=2.0):
    """
    Iterates through every pair. 
    Applies logic:
    1. FILTER: Both Target AND Signal must be in Tails (<20 or >80).
    2. FILTER: Min samples 15, unless BOTH are Super Extreme (<5 or >95).
    """
    results = []
    tickers = list(processed_data.keys())
    
    # 1. Get Current State (Today's Ranks)
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
            
            # --- LOGIC FILTER 1: THE "DOUBLE TAIL" REQUIREMENT ---
            # Both tickers must be outside the 20-80 zone.
            t_is_tail = (curr_t_rank < 25 or curr_t_rank > 75)
            s_is_tail = (curr_s_rank < 25 or curr_s_rank > 75)
            
            if not (t_is_tail and s_is_tail):
                continue

            # --- ALIGNMENT ---
            aligned = pd.concat([
                t_df['FwdRet_10d'].rename("Outcome"),
                t_df['Rank_21d'].rename("Target_Rank"),
                s_df['Rank_21d'].rename("Signal_Rank")
            ], axis=1, join='inner')
            
            history = aligned.dropna(subset=["Outcome"])
            if history.empty: continue

            # --- HISTORICAL MATCHING ---
            t_min, t_max = curr_t_rank - rank_radius, curr_t_rank + rank_radius
            s_min, s_max = curr_s_rank - rank_radius, curr_s_rank + rank_radius
            
            matches = history[
                (history['Target_Rank'] >= t_min) & (history['Target_Rank'] <= t_max) &
                (history['Signal_Rank'] >= s_min) & (history['Signal_Rank'] <= s_max)
            ]
            
            count = len(matches)
            
            # --- LOGIC FILTER 2: SAMPLE SIZE vs SUPER EXTREME ---
            # Exception applies only if BOTH are <5 or >95
            t_is_super = (curr_t_rank < 5 or curr_t_rank > 95)
            s_is_super = (curr_s_rank < 5 or curr_s_rank > 95)
            
            # If both are super extreme, allow 5 samples. Otherwise require 15.
            min_samples = 5 if (t_is_super and s_is_super) else 15
            
            if count < min_samples: continue

            # --- STATISTICS ---
            avg_return = matches['Outcome'].mean()
            win_rate = (matches['Outcome'] > 0).sum() / count * 100
            
            results.append({
                "Target": target,
                "Signal_Ticker": signal,
                "Current_Setup": f"T:{int(curr_t_rank)} | S:{int(curr_s_rank)}",
                "History_Count": count,
                "Win_Rate": win_rate,
                "Avg_Return_10d": avg_return,
                "Signal_Rank": curr_s_rank,
                "Target_Rank": curr_t_rank
            })

    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Market Matrix Screener")
    st.title("⚡ 10-Day Opportunity Screener")
    
    st.markdown("""
    **Strict Filtering Logic:**
    1. **Double Tail Filter:** Shows ONLY pairs where **BOTH** tickers are currently in the tails (<20 or >80).
    2. **Significance:** Requires 15+ historical matches, unless **BOTH** tickers are Super Extreme (<5 or >95), then allows 5+.
    """)
    
    # 1. INPUTS
    with st.expander("⚙️ Screener Settings", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker_input = st.text_area("Universe (Comma Separated)", value=DEFAULT_TICKERS, height=100)
        with col2:
            radius = st.slider("Similarity Radius", 1, 10, 2, help="Radius +/- Rank. Default is 2 (Very Strict).")
            run_btn = st.button("Run Matrix Scan", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Downloading data & crunching matrices..."):
            # A. Process Data
            raw_data = get_batch_data(ticker_input)
            if not raw_data:
                st.error("No valid data found.")
                return
            
            processed = calculate_features(raw_data)
            
            # B. Run Algorithm
            df_results = run_scanner(processed, rank_radius=radius)
            
            if df_results.empty:
                st.warning("No opportunities found. Try increasing the Radius or relaxing the strictness if markets are quiet.")
                return
            
            # C. Formatting & Display
            st.divider()
            
            # Split into Bullish / Bearish
            bullish = df_results[df_results['Avg_Return_10d'] > 0].sort_values(by="Avg_Return_10d", ascending=False)
            bearish = df_results[df_results['Avg_Return_10d'] < 0].sort_values(by="Avg_Return_10d", ascending=True)

            # --- DISPLAY BULLISH ---
            st.subheader(f"🟢 Top Bullish Setups (Next 10 Days)")
            st.dataframe(
                bullish.head(20).style.format({
                    "Win_Rate": "{:.1f}%",
                    "Avg_Return_10d": "+{:.2f}%",
                    "Signal_Rank": "{:.0f}",
                    "Target_Rank": "{:.0f}"
                }).background_gradient(subset=["Win_Rate"], cmap="Greens", vmin=50, vmax=80),
                use_container_width=True,
                column_order=["Target", "Signal_Ticker", "Avg_Return_10d", "Win_Rate", "History_Count", "Current_Setup"]
            )

            # --- DISPLAY BEARISH ---
            st.subheader(f"🔴 Top Bearish Setups (Next 10 Days)")
            st.dataframe(
                bearish.head(20).style.format({
                    "Win_Rate": "{:.1f}%",
                    "Avg_Return_10d": "{:.2f}%",
                    "Signal_Rank": "{:.0f}",
                    "Target_Rank": "{:.0f}"
                }).background_gradient(subset=["Win_Rate"], cmap="Reds_r", vmin=20, vmax=50),
                use_container_width=True,
                column_order=["Target", "Signal_Ticker", "Avg_Return_10d", "Win_Rate", "History_Count", "Current_Setup"]
            )
            
            # --- SCATTER SUMMARY ---
            st.divider()
            st.subheader("Map of Opportunities")
            
            fig = px.scatter(
                df_results, 
                x="Win_Rate", 
                y="Avg_Return_10d", 
                hover_data=["Target", "Signal_Ticker", "History_Count", "Current_Setup"],
                color="Target_Rank", # Color by how extended the target is
                color_continuous_scale="RdBu",
                title="Screener Results: Win Rate vs Expected Return (Color = Target Rank)"
            )
            fig.add_vline(x=50, line_dash="dash", line_color="gray")
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
