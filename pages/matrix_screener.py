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

# Removed 5d and 2d as requested
TRAIL_WINDOWS = [10, 21, 63]
FWD_WINDOWS   = [5, 10, 21, 63]

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
            
            # --- HIERARCHICAL LOGIC (TLT Removed) ---
            allowed_signals = ["SPY"]
            
            if target == "SPY": allowed_signals = ["USO", "UUP", "GLD"]
            elif target == "XOM": allowed_signals = ["SPY", "USO"]
            elif target == "JPM": allowed_signals = ["SPY", "XLF"]
            elif target in ["NVDA", "AMD"]: allowed_signals = ["SPY", "SMH"]
            elif target == "SLV": allowed_signals = ["SPY", "GLD"]
            elif target == "GLD": allowed_signals = ["SPY", "SLV"]
            elif target == "CEF": allowed_signals = ["SPY", "GLD", "SLV"]
            elif target == "UNG": allowed_signals = ["SPY", "USO"]
            elif target == "USO": allowed_signals = ["SPY", "UNG"]
            
            if signal not in allowed_signals: continue
            
            s_df = processed_data[signal]
            
            common_idx = t_df.index.intersection(s_df.index)
            if len(common_idx) < 252: continue
            
            t_subset = t_df.loc[common_idx]
            s_subset = s_df.loc[common_idx]

            # --- NESTED WINDOW LOOPS ---
            for t_w in TRAIL_WINDOWS:
                curr_t_rank = current_ranks.get(target, {}).get(t_w, np.nan)
                if np.isnan(curr_t_rank): continue
                
                t_tail = "UPPER" if curr_t_rank > 75 else ("LOWER" if curr_t_rank < 25 else "MID")
                if t_tail == "MID": continue
                
                hist_t_ranks = t_subset[f'Rank_{t_w}d']
                if t_tail == "UPPER": mask_t = hist_t_ranks >= curr_t_rank
                else:                 mask_t = hist_t_ranks <= curr_t_rank

                for s_w in TRAIL_WINDOWS:
                    curr_s_rank = current_ranks.get(signal, {}).get(s_w, np.nan)
                    if np.isnan(curr_s_rank): continue
                    
                    s_tail = "UPPER" if curr_s_rank > 75 else ("LOWER" if curr_s_rank < 25 else "MID")
                    if s_tail == "MID": continue
                    
                    hist_s_ranks = s_subset[f'Rank_{s_w}d']
                    if s_tail == "UPPER": mask_s = hist_s_ranks >= curr_s_rank
                    else:                 mask_s = hist_s_ranks <= curr_s_rank
                    
                    # Combined Mask
                    valid_dates_mask = mask_t & mask_s
                    valid_count = valid_dates_mask.sum()
                    if valid_count < 10: continue
                
                    for fwd_w in FWD_WINDOWS:
                        outcomes_sigma = t_subset[valid_dates_mask][f'FwdRet_{fwd_w}d_Sigma'].dropna()
                        outcomes_raw   = t_subset[valid_dates_mask][f'FwdRet_{fwd_w}d_Raw'].dropna()
                        
                        real_count = len(outcomes_sigma)
                        if real_count < 10: continue

                        cond_avg_sigma = outcomes_sigma.mean()
                        baseline_sigma = baselines.get(target, {}).get(fwd_w, 0.0)
                        alpha_sigma = cond_avg_sigma - baseline_sigma

                        avg_raw_pct = outcomes_raw.mean() * 100
                        win_rate    = (outcomes_raw > 0).sum() / real_count * 100
                        
                        results.append({
                            "Target": target,
                            "Signal": signal,
                            "T_Lookback": f"{t_w}d", 
                            "S_Lookback": f"{s_w}d",
                            "Fwd_Horizon": f"{fwd_w}d", 
                            "Setup_Ranks": f"T:{int(curr_t_rank)}|S:{int(curr_s_rank)}",
                            "History": real_count,
                            "Win_Rate": win_rate,
                            "Excess_Sigma": alpha_sigma,
                            "Raw_Sigma": cond_avg_sigma,
                            "Exp_Return": avg_raw_pct,
                        })

    return pd.DataFrame(results)

def generate_ensemble(df_results, alpha_threshold=0.25):
    """
    Aggregates signals.
    CRITICAL CHANGE: For Bearish Scores, we ONLY count signals where Exp_Return is NEGATIVE.
    """
    if df_results.empty: return pd.DataFrame()
    
    # 1. Bullish Candidates: Positive Excess Alpha
    bull_mask = df_results['Excess_Sigma'] > alpha_threshold
    # Ideally Bullish should also have Positive Return, but Alpha usually implies it.
    
    # 2. Bearish Candidates: Negative Excess Alpha AND Negative Absolute Return
    # This prevents shorting strong uptrends just because they are "less strong".
    bear_mask = (df_results['Excess_Sigma'] < -alpha_threshold) & (df_results['Exp_Return'] < 0)
    
    # Combine
    valid_signals = df_results[bull_mask | bear_mask].copy()
    
    if valid_signals.empty: return pd.DataFrame()
    
    ensemble = valid_signals.groupby(['Target', 'Fwd_Horizon']).agg({
        'Excess_Sigma': 'sum',      
        'Win_Rate': 'mean',         
        'Exp_Return': 'mean',       
        'Signal': 'nunique',        
        'T_Lookback': 'count'       
    }).reset_index()
    
    ensemble.rename(columns={
        'Excess_Sigma': 'Conviction_Score',
        'Signal': 'Confirming_Tickers',
        'T_Lookback': 'Total_Signals'
    }, inplace=True)
    
    ensemble['Abs_Score'] = ensemble['Conviction_Score'].abs()
    ensemble = ensemble.sort_values('Abs_Score', ascending=False).drop(columns=['Abs_Score'])
    
    return ensemble

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Multi-Fractal Alpha")
    st.title("⚡ Multi-Timeframe Alpha Scanner")
    
    st.markdown("""
    **Scanning Combinations per Pair:** (Removed 2d & 5d windows)
    * **Alpha Logic:** Excess Sigma vs Unconditional History.
    * **Bearish Safety Filter:** We *only* flag a Bearish setup if the **Total Expected Return is Negative**. 
      *(We ignore 'Relative Weakness' in strong uptrends).*
    * **Note:** TLT has been removed as a signal source.
    """)
    
    with st.expander("⚙️ Screener Settings", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker_input = st.text_area("Universe", value=DEFAULT_TICKERS, height=100)
        with col2:
            st.info("Filtering: |Excess| > 0.25σ") 
            run_btn = st.button("Run Ensemble Scan", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Crunching combinations & Building Ensemble..."):
            raw_data, fetch_time = get_batch_data(ticker_input)
            if not raw_data:
                st.error("No valid data found.")
                return
            st.success(f"✅ Data Refreshed: {fetch_time}")
            
            processed = calculate_features(raw_data)
            df_results = run_scanner(processed)
            
            if df_results.empty:
                st.warning("No opportunities found matching criteria.")
                return
            
            # --- GENERATE ENSEMBLE ---
            ensemble_df = generate_ensemble(df_results, alpha_threshold=0.25)
            
            # Separate
            top_bulls = ensemble_df[ensemble_df['Conviction_Score'] > 0].head(5)
            top_bears = ensemble_df[ensemble_df['Conviction_Score'] < 0].head(5)
            
            # --- DISPLAY ENSEMBLE ---
            st.header("🏆 Top 5 High-Conviction Setups")
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("🚀 Strongest Bullish Confluence")
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
                    st.info("No strong bullish confluence found.")
                    
            with c2:
                st.subheader("🐻 Strongest Bearish Confluence")
                st.caption("Filtered: Only shows setups with Negative Expected Return")
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
                    st.info("No strong bearish confluence found (Signals may exist but didn't pass the Negative Return safety filter).")

            st.divider()
            
            # --- DISPLAY DETAILED TABLE ---
            st.subheader("🔎 Detailed Signal Logs")
            
            ALPHA_THRESHOLD = 0.25
            
            # Bullish Details: Excess > 0.25
            bullish_details = df_results[df_results['Excess_Sigma'] > ALPHA_THRESHOLD].sort_values(by="Excess_Sigma", ascending=False)
            
            # Bearish Details: Excess < -0.25 AND Exp_Return < 0 (Safety Filter)
            bearish_details = df_results[
                (df_results['Excess_Sigma'] < -ALPHA_THRESHOLD) & 
                (df_results['Exp_Return'] < 0)
            ].sort_values(by="Excess_Sigma", ascending=True)

            tab1, tab2 = st.tabs(["Individual Bull Signals", "Individual Bear Signals (Safe)"])
            
            with tab1:
                st.dataframe(
                    bullish_details.head(100).style.format({
                        "Win_Rate": "{:.1f}%", "Excess_Sigma": "+{:.2f}σ", "Exp_Return": "+{:.2f}%"
                    }).background_gradient(subset=["Excess_Sigma"], cmap="Greens", vmin=0, vmax=1.0),
                    use_container_width=True,
                    column_order=["Target", "Signal", "T_Lookback", "S_Lookback", "Fwd_Horizon", "Excess_Sigma", "Exp_Return", "Win_Rate", "History"]
                )
                
            with tab2:
                st.dataframe(
                    bearish_details.head(100).style.format({
                        "Win_Rate": "{:.1f}%", "Excess_Sigma": "{:.2f}σ", "Exp_Return": "{:.2f}%"
                    }).background_gradient(subset=["Excess_Sigma"], cmap="Reds_r", vmin=-1.0, vmax=0),
                    use_container_width=True,
                    column_order=["Target", "Signal", "T_Lookback", "S_Lookback", "Fwd_Horizon", "Excess_Sigma", "Exp_Return", "Win_Rate", "History"]
                )
            
            # --- SCATTER SUMMARY ---
            st.divider()
            
            fig = px.scatter(
                df_results, 
                x="Win_Rate", 
                y="Exp_Return", 
                hover_data=["Target", "Signal", "T_Lookback", "S_Lookback", "Fwd_Horizon", "Excess_Sigma", "Raw_Sigma", "History"],
                color="Excess_Sigma", 
                color_continuous_scale="RdBu",
                title="Alpha Map: Win Rate vs Total Expected Return %"
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(yaxis_title="Avg Total Return (%)")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
