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

# -----------------------------------------------------------------------------
# DATA ENGINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=True)
def get_batch_data(ticker_list):
    """
    Downloads MAX history (last 25y).
    """
    tickers = [t.strip().upper() for t in ticker_list.replace("\n", "").split(",") if t.strip()]
    tickers = list(set(tickers)) 
    
    if not tickers:
        return {}
    
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
    return data_dict

@st.cache_data(show_spinner=False)
def calculate_features(data_dict):
    """
    Calculates 21d Trailing Rank, 10d Vol-Normalized (Sigma), AND 10d Raw Returns
    """
    processed = {}
    
    for ticker, df in data_dict.items():
        df = df.copy()
        
        df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 1. Feature: 21d Trailing Return
        df['Ret_21d'] = df['Close'].pct_change(21)
        
        # 2. Volatility (21d Annualized) for Normalization
        df['Vol_Daily'] = df['LogRet'].rolling(21).std()
        
        # 3. Target: 10d Forward Return (Raw)
        df['Fwd_Close'] = df['Close'].shift(-10)
        df['FwdRet_10d_Raw'] = (df['Fwd_Close'] / df['Close']) - 1.0
        
        # 4. Vol-Normalization (Z-Score)
        expected_move = df['Vol_Daily'] * np.sqrt(10)
        df['FwdRet_Sigma'] = df['FwdRet_10d_Raw'] / expected_move
        
        # 5. Rank Transformation
        df['Rank_21d'] = df['Ret_21d'].expanding(min_periods=252).rank(pct=True) * 100
        
        df = df.dropna(subset=['Rank_21d', 'Vol_Daily'])
        
        processed[ticker] = df[['Close', 'Rank_21d', 'FwdRet_Sigma', 'FwdRet_10d_Raw']]
        
    return processed

def run_scanner(processed_data):
    """
    Iterates through pairs using Cumulative Tail Analysis.
    Detailed Hierarchical Logic applied in the loop.
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
            
            # --- HIERARCHICAL PREDICTOR LOGIC ---
            
            # 1. DEFAULT BASELINE: Everyone listens to SPY (Market) and TLT (Rates)
            allowed_signals = ["SPY", "TLT"]
            
            # 2. SECTOR SPECIFIC OVERRIDES/ADDITIONS
            
            if target == "SPY":
                # Market listens to Macro Assets
                allowed_signals = ["TLT", "USO", "UUP", "GLD"]
            
            elif target == "XOM":
                # Energy listens to Oil
                allowed_signals = ["SPY", "TLT", "USO"]
                
            elif target == "JPM":
                # Banks listen to Financials ETF
                allowed_signals = ["SPY", "TLT", "XLF"]
                
            elif target in ["NVDA", "AMD"]:
                # Semis listen to SMH
                allowed_signals = ["SPY", "TLT", "SMH"]
            
            elif target == "SLV":
                # Silver listens to Gold
                allowed_signals = ["SPY", "TLT", "GLD"]
                
            elif target == "GLD":
                # Gold listens to Silver (rare but valid correlation)
                allowed_signals = ["SPY", "TLT", "SLV"]
            
            elif target == "CEF":
                # Sprott Physical (Gold/Silver Trust) listens to both
                allowed_signals = ["SPY", "TLT", "GLD", "SLV"]
                
            elif target == "UNG":
                # Gas listens to Oil
                allowed_signals = ["SPY", "TLT", "USO"]
                
            elif target == "USO":
                # Oil listens to Gas
                allowed_signals = ["SPY", "TLT", "UNG"]
            
            # CHECK: Is this signal allowed?
            if signal not in allowed_signals:
                continue
            
            s_df = processed_data[signal]
            curr_s_rank = current_states.get(signal, np.nan)
            if np.isnan(curr_s_rank): continue
            
            # --- FILTER 1: EXTREMITY CHECK (Double Tail) ---
            # Both must be outside 25-75
            t_tail = "UPPER" if curr_t_rank > 75 else ("LOWER" if curr_t_rank < 25 else "MID")
            s_tail = "UPPER" if curr_s_rank > 75 else ("LOWER" if curr_s_rank < 25 else "MID")
            
            if t_tail == "MID" or s_tail == "MID":
                continue

            # --- ALIGNMENT ---
            aligned = pd.concat([
                t_df['FwdRet_Sigma'].rename("Outcome_Sigma"),
                t_df['FwdRet_10d_Raw'].rename("Outcome_Raw"),
                t_df['Rank_21d'].rename("Target_Rank"),
                s_df['Rank_21d'].rename("Signal_Rank")
            ], axis=1, join='inner')
            
            history = aligned.dropna(subset=["Outcome_Sigma"])
            if history.empty: continue

            # --- FILTER 2: CUMULATIVE TAIL LOGIC ---
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
            
            if count < 10: continue

            # --- STATISTICS ---
            avg_sigma = matches['Outcome_Sigma'].mean()
            avg_raw_pct = matches['Outcome_Raw'].mean() * 100 
            
            win_rate = (matches['Outcome_Raw'] > 0).sum() / count * 100
            
            results.append({
                "Target": target,
                "Signal_Ticker": signal,
                "Current_Setup": f"T:{int(curr_t_rank)} | S:{int(curr_s_rank)}",
                "History_Count": count,
                "Win_Rate": win_rate,
                "Avg_Return_Sigma": avg_sigma,
                "Avg_Return_Pct": avg_raw_pct,
                "Target_Rank": curr_t_rank 
            })

    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Hierarchical Matrix Screener")
    st.title("⚡ Vol-Normalized Tail Screener")
    
    st.markdown("""
    **Specific Relationships Configured:**
    * **Financials (JPM)** $\leftarrow$ XLF
    * **Semis (NVDA, AMD)** $\leftarrow$ SMH
    * **Metals (GLD, SLV, CEF)** $\leftarrow$ Cross-Correlated
    * **Energy (USO, UNG)** $\leftarrow$ Cross-Correlated
    * **General** $\leftarrow$ SPY, TLT
    """)
    
    # 1. INPUTS
    with st.expander("⚙️ Screener Settings", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker_input = st.text_area("Universe", value=DEFAULT_TICKERS, height=100)
        with col2:
            st.info("Filtering: Sigma > 0.25σ")
            run_btn = st.button("Run Vol-Adj Scan", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Processing Hierarchical Matrix..."):
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
            
            # --- FILTERING LOGIC ---
            SIGMA_THRESHOLD = 0.25
            
            bullish = df_results[df_results['Avg_Return_Sigma'] > SIGMA_THRESHOLD].sort_values(by="Avg_Return_Sigma", ascending=False)
            bearish = df_results[df_results['Avg_Return_Sigma'] < -SIGMA_THRESHOLD].sort_values(by="Avg_Return_Sigma", ascending=True)

            # --- DISPLAY BULLISH ---
            st.subheader(f"🟢 Top Bullish Signals (> +{SIGMA_THRESHOLD}σ)")
            if not bullish.empty:
                st.dataframe(
                    bullish.head(20).style.format({
                        "Win_Rate": "{:.1f}%",
                        "Avg_Return_Sigma": "+{:.2f}σ",
                        "Avg_Return_Pct": "+{:.2f}%",
                        "Target_Rank": "{:.0f}"
                    }).background_gradient(subset=["Avg_Return_Sigma"], cmap="Greens", vmin=0, vmax=1.5),
                    use_container_width=True,
                    column_order=["Target", "Signal_Ticker", "Avg_Return_Pct", "Avg_Return_Sigma", "Win_Rate", "History_Count", "Current_Setup"]
                )
            else:
                st.info(f"No signals found > {SIGMA_THRESHOLD}σ")

            # --- DISPLAY BEARISH ---
            st.subheader(f"🔴 Top Bearish Signals (< -{SIGMA_THRESHOLD}σ)")
            if not bearish.empty:
                st.dataframe(
                    bearish.head(20).style.format({
                        "Win_Rate": "{:.1f}%",
                        "Avg_Return_Sigma": "{:.2f}σ",
                        "Avg_Return_Pct": "{:.2f}%",
                        "Target_Rank": "{:.0f}"
                    }).background_gradient(subset=["Avg_Return_Sigma"], cmap="Reds_r", vmin=-1.5, vmax=0),
                    use_container_width=True,
                    column_order=["Target", "Signal_Ticker", "Avg_Return_Pct", "Avg_Return_Sigma", "Win_Rate", "History_Count", "Current_Setup"]
                )
            else:
                st.info(f"No signals found < -{SIGMA_THRESHOLD}σ")
            
            # --- SCATTER SUMMARY ---
            st.divider()
            st.subheader("Map of Vol-Adjusted Edge (All Signals)")
            
            fig = px.scatter(
                df_results, 
                x="Win_Rate", 
                y="Avg_Return_Pct", 
                hover_data=["Target", "Signal_Ticker", "Avg_Return_Sigma", "History_Count", "Current_Setup"],
                color="Avg_Return_Sigma", 
                color_continuous_scale="RdBu",
                title="Screener Results: Win Rate vs Expected Return %"
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig.update_layout(yaxis_title="Avg Return (%)")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
