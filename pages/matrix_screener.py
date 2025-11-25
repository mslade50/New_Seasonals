import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DEFAULT_TICKERS = "SPY, QQQ, IWM, TLT, GLD, USO, UUP, HYG, XLF, XLE, XLK, XBI, SMH, ARKK, BTC-USD"

# -----------------------------------------------------------------------------
# DATA ENGINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=True)
def get_batch_data(ticker_list):
    """
    Downloads and processes data for ALL tickers at once.
    Returns a Dictionary of DataFrames {ticker: df}.
    """
    tickers = [t.strip().upper() for t in ticker_list.split(",") if t.strip()]
    
    if not tickers:
        return {}
    
    data_dict = {}
    
    # We download individually to ensure clean data structures for each
    # (Batch downloading with yfinance can sometimes create complex MultiIndices that are hard to align)
    progress_bar = st.progress(0)
    for i, t in enumerate(tickers):
        try:
            df = yf.download(t, period="5y", progress=False, auto_adjust=True)
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
    Calculates the specific metrics needed for the screener:
    - 21d Trailing Return (Ranked 0-100)
    - 10d Forward Return (Raw %)
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

def run_scanner(processed_data, rank_radius=15.0):
    """
    Iterates through every pair (Target vs Signal).
    Finds historical matches similar to TODAY's ranks.
    """
    results = []
    
    tickers = list(processed_data.keys())
    
    # 1. Get Current State (Today's Ranks)
    current_states = {}
    for t in tickers:
        try:
            # Get the very last row (Current)
            current_states[t] = processed_data[t].iloc[-1]['Rank_21d']
        except:
            current_states[t] = np.nan

    # 2. Pairwise Matrix Scan
    total_pairs = len(tickers) * (len(tickers) - 1)
    
    # We iterate: TARGET (The thing we want to trade) vs SIGNAL (The indicator)
    for target in tickers:
        t_df = processed_data[target]
        curr_t_rank = current_states.get(target, np.nan)
        
        if np.isnan(curr_t_rank): continue

        for signal in tickers:
            if target == signal: continue # Skip self-comparison (usually correlates 1.0)
            
            s_df = processed_data[signal]
            curr_s_rank = current_states.get(signal, np.nan)
            
            if np.isnan(curr_s_rank): continue
            
            # --- ALIGNMENT ---
            # Inner join on Index (Date)
            # We need: T_FwdRet (Outcome), T_Rank (Context), S_Rank (Signal)
            aligned = pd.concat([
                t_df['FwdRet_10d'].rename("Outcome"),
                t_df['Rank_21d'].rename("Target_Rank"),
                s_df['Rank_21d'].rename("Signal_Rank")
            ], axis=1, join='inner')
            
            # Remove the last 10 rows for statistics (as they don't have outcomes yet)
            history = aligned.dropna(subset=["Outcome"])
            
            if history.empty: continue

            # --- FILTERING ---
            # Find historical dates where ranks were similar to today (+/- radius)
            t_min, t_max = curr_t_rank - rank_radius, curr_t_rank + rank_radius
            s_min, s_max = curr_s_rank - rank_radius, curr_s_rank + rank_radius
            
            matches = history[
                (history['Target_Rank'] >= t_min) & (history['Target_Rank'] <= t_max) &
                (history['Signal_Rank'] >= s_min) & (history['Signal_Rank'] <= s_max)
            ]
            
            count = len(matches)
            if count < 10: continue # Not statistically significant
            
            # --- STATISTICS ---
            avg_return = matches['Outcome'].mean()
            win_rate = (matches['Outcome'] > 0).sum() / count * 100
            
            # Expected Value (Rough approx)
            ev = avg_return
            
            results.append({
                "Target": target,
                "Signal_Ticker": signal,
                "Current_Setup": f"T:{int(curr_t_rank)} | S:{int(curr_s_rank)}",
                "History_Count": count,
                "Win_Rate": win_rate,
                "Avg_Return_10d": avg_return,
                "Signal_Rank": curr_s_rank, # For sorting/display
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
    This tool scans **every pairwise relationship** in your list. 
    It checks: *"Given where Ticker A and Ticker B are trading **right now** (21d Ranks), 
    what usually happens to Ticker A over the next 10 days?"*
    """)
    
    # 1. INPUTS
    with st.expander("⚙️ Screener Settings", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker_input = st.text_area("Universe (Comma Separated)", value=DEFAULT_TICKERS, height=68)
        with col2:
            radius = st.slider("Similarity Radius (Rank)", 5, 25, 15, help="How strict should we be when finding historical matches? Lower = Stricter/Fewer matches.")
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
                st.warning("No significant relationships found with current settings.")
                return
            
            # C. Formatting & Display
            st.divider()
            
            # Split into Bullish / Bearish
            bullish = df_results[df_results['Avg_Return_10d'] > 0].sort_values(by="Avg_Return_10d", ascending=False)
            bearish = df_results[df_results['Avg_Return_10d'] < 0].sort_values(by="Avg_Return_10d", ascending=True)

            # --- DISPLAY BULLISH ---
            st.subheader(f"🟢 Top Bullish Setups (Next 10 Days)")
            st.markdown("look for high **Win Rates** (>65%) and high **Avg Return**.")
            
            st.dataframe(
                bullish.head(15).style.format({
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
                bearish.head(15).style.format({
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
            st.markdown("The chart below plots all detected relationships. **Top Right** = Strong Bullish Edge.")
            
            # Simple Scatter Plot
            import plotly.express as px
            fig = px.scatter(
                df_results, 
                x="Win_Rate", 
                y="Avg_Return_10d", 
                hover_data=["Target", "Signal_Ticker", "History_Count"],
                color="Avg_Return_10d",
                color_continuous_scale="RdBu",
                title="Screener Results: Win Rate vs Expected Return"
            )
            fig.add_vline(x=50, line_dash="dash", line_color="gray")
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
