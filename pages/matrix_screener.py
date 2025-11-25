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
            col_ret = f'Ret_{w}d'
            df[col_ret] = df['Close'].pct_change(w)
            
            col_rank = f'Rank_{w}d'
            df[col_rank] = df[col_ret].expanding(min_periods=252).rank(pct=True) * 100

        # 3. Calculate Forward Outcomes (The Payoff)
        for w in FWD_WINDOWS:
            col_fwd_raw = f'FwdRet_{w}d_Raw'
            df[col_fwd_raw] = (df['Close'].shift(-w) / df['Close']) - 1.0
            
            col_fwd_sigma = f'FwdRet_{w}d_Sigma'
            expected_move = df['Vol_Daily'] * np.sqrt(w)
            df[col_fwd_sigma] = df[col_fwd_raw] / expected_move

        cols_to_keep = ['Close', 'Vol_Daily'] + \
                       [f'Rank_{w}d' for w in TRAIL_WINDOWS] + \
                       [f'FwdRet_{w}d_Raw' for w in FWD_WINDOWS] + \
                       [f'FwdRet_{w}d_Sigma' for w in FWD_WINDOWS]
                       
        df = df.dropna(subset=['Vol_Daily']) 
        processed[ticker] = df[cols_to_keep]
        
    return processed

def run_scanner(processed_data):
    """
    Iterates through Pairs -> Target Windows -> Signal Windows -> Forward Windows.
    Calculates ALPHA (Excess Sigma).
    """
    results = []
    tickers = list(processed_data.keys())
    
    # --- PRE-CALCULATE BASELINES ---
    baselines = {} 
    for t in tickers:
        baselines[t] = {}
        for fwd_w in FWD_WINDOWS:
            col = f'FwdRet_{fwd_w}d_Sigma'
            baselines[t][fwd_w] = processed_data[t][col].mean()

    # Pre-fetch current ranks for lookups
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
            
            # Align indices once per pair to save time
            common_idx = t_df.index.intersection(s_df.index)
            if len(common_idx) < 252: continue
            
            t_subset = t_df.loc[common_idx]
            s_subset = s_df.loc[common_idx]

            # --- NESTED WINDOW LOOPS (125 Combinations) ---
            
            # 1. Target Lookback Loop
            for t_w in TRAIL_WINDOWS:
                curr_t_rank = current_ranks.get(target, {}).get(t_w, np.nan)
                if np.isnan(curr_t_rank): continue
                
                t_tail = "UPPER" if curr_t_rank > 75 else ("LOWER" if curr_t_rank < 25 else "MID")
                if t_tail == "MID": continue
                
                t_rank_col = f'Rank_{t_w}d'
                hist_t_ranks = t_subset[t_rank_col]
                
                if t_tail == "UPPER": mask_t = hist_t_ranks >= curr_t_rank
                else:                 mask_t = hist_t_ranks <= curr_t_rank

                # 2. Signal Lookback Loop (Decoupled from Target)
                for s_w in TRAIL_WINDOWS:
                    curr_s_rank = current_ranks.get(signal, {}).get(s_w, np.nan)
                    if np.isnan(curr_s_rank): continue
                    
                    s_tail = "UPPER" if curr_s_rank > 75 else ("LOWER" if curr_s_rank < 25 else "MID")
                    if s_tail == "MID": continue
                    
                    s_rank_col = f'Rank_{s_w}d'
                    hist_s_ranks = s_subset[s_rank_col]
                    
                    if s_tail == "UPPER": mask_s = hist_s_ranks >= curr_s_rank
                    else:                 mask_s = hist_s_ranks <= curr_s_rank
                    
                    # Combined Mask
                    valid_dates_mask = mask_t & mask_s
                    valid_count = valid_dates_mask.sum()
                    
                    if valid_count < 10: continue
                
                    # 3. Forward Payoff Loop
                    for fwd_w in FWD_WINDOWS:
                        outcome_sigma_col = f'FwdRet_{fwd_w}d_Sigma'
                        outcome_raw_col   = f'FwdRet_{fwd_w}d_Raw'
                        
                        outcomes_sigma = t_subset[valid_dates_mask][outcome_sigma_col].dropna()
                        outcomes_raw   = t_subset[valid_dates_mask][outcome_raw_col].dropna()
                        
                        real_count = len(outcomes_sigma)
                        if real_count < 10: continue

                        cond_avg_sigma = outcomes_sigma.mean()
                        
                        # Alpha Calculation
                        baseline_sigma = baselines.get(target, {}).get(fwd_w, 0.0)
                        alpha_sigma = cond_avg_sigma - baseline_sigma

                        avg_raw_pct = outcomes_raw.mean() * 100
                        win_rate    = (outcomes_raw > 0).sum() / real_count * 100
                        
                        results.append({
                            "Target": target,
                            "Signal": signal,
                            "T_Lookback": f"{t_w}d", # Setup Part 1
                            "S_Lookback": f"{s_w}d", # Setup Part 2
                            "Fwd_Horizon": f"{fwd_w}d", # Payoff
                            "Setup_Ranks": f"T:{int(curr_t_rank)}|S:{int(curr_s_rank)}",
                            "History": real_count,
                            "Win_Rate": win_rate,
                            "Excess_Sigma": alpha_sigma,
                            "Raw_Sigma": cond_avg_sigma,
                            "Exp_Return": avg_raw_pct,
                        })

    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Multi-Fractal Alpha")
    st.title("⚡ Multi-Timeframe Alpha Scanner")
    
    st.markdown("""
    **Scanning 125 Combinations per Pair:**
    * **Decoupled Setup:** Target Lookback ($T_{win}$) and Signal Lookback ($S_{win}$) are independent.
    * **Example:** *Is SPY (63d Trend) affected by TLT (2d Shock)?*
    * **Alpha:** Excess Sigma vs Unconditional History.
    """)
    
    # 1. INPUTS
    with st.expander("⚙️ Screener Settings", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker_input = st.text_area("Universe", value=DEFAULT_TICKERS, height=100)
        with col2:
            st.info("Filtering: |Excess| > 0.25σ") 
            run_btn = st.button("Run Alpha Scan", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Calculating 125x Combinations per pair (this may take 15s)..."):
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
            
            # Filter Threshold (Excess Return)
            ALPHA_THRESHOLD = 0.25
            
            bullish = df_results[df_results['Excess_Sigma'] > ALPHA_THRESHOLD].sort_values(by="Excess_Sigma", ascending=False)
            bearish = df_results[df_results['Excess_Sigma'] < -ALPHA_THRESHOLD].sort_values(by="Excess_Sigma", ascending=True)

            # --- DISPLAY BULLISH ---
            st.subheader(f"🟢 Top Bullish Alpha (Excess > +{ALPHA_THRESHOLD}σ)")
            if not bullish.empty:
                st.dataframe(
                    bullish.head(50).style.format({
                        "Win_Rate": "{:.1f}%",
                        "Excess_Sigma": "+{:.2f}σ",
                        "Raw_Sigma": "{:.2f}σ",
                        "Exp_Return": "+{:.2f}%",
                    }).background_gradient(subset=["Excess_Sigma"], cmap="Greens", vmin=0, vmax=1.0),
                    use_container_width=True,
                    column_order=["Target", "Signal", "T_Lookback", "S_Lookback", "Fwd_Horizon", "Excess_Sigma", "Exp_Return", "Win_Rate", "History", "Setup_Ranks"]
                )
            else:
                st.info(f"No signals found > {ALPHA_THRESHOLD}σ")

            # --- DISPLAY BEARISH ---
            st.subheader(f"🔴 Top Bearish Alpha (Excess < -{ALPHA_THRESHOLD}σ)")
            if not bearish.empty:
                st.dataframe(
                    bearish.head(50).style.format({
                        "Win_Rate": "{:.1f}%",
                        "Excess_Sigma": "{:.2f}σ",
                        "Raw_Sigma": "{:.2f}σ",
                        "Exp_Return": "{:.2f}%",
                    }).background_gradient(subset=["Excess_Sigma"], cmap="Reds_r", vmin=-1.0, vmax=0),
                    use_container_width=True,
                    column_order=["Target", "Signal", "T_Lookback", "S_Lookback", "Fwd_Horizon", "Excess_Sigma", "Exp_Return", "Win_Rate", "History", "Setup_Ranks"]
                )
            else:
                st.info(f"No signals found < -{ALPHA_THRESHOLD}σ")
            
            # --- SCATTER SUMMARY ---
            st.divider()
            st.subheader("Map of Excess Edge")
            
            fig = px.scatter(
                df_results, 
                x="Win_Rate", 
                y="Exp_Return", 
                hover_data=["Target", "Signal", "T_Lookback", "S_Lookback", "Fwd_Horizon", "Excess_Sigma", "Raw_Sigma", "History"],
                color="Excess_Sigma", 
                color_continuous_scale="RdBu",
                title="Alpha Map: Win Rate vs Total Expected Return % (Color = Excess Sigma)"
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(yaxis_title="Avg Total Return (%)")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
