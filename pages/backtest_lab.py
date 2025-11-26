import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import plotly.express as px
import plotly.graph_objects as go
import random
import itertools

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BACKTEST_TICKERS = ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "AMD", "MSFT", "GOOG", "AMZN", "META", "NFLX", "TLT", "USO", "GLD"]
TEST_HORIZONS = [5, 10, 21]

# -----------------------------------------------------------------------------
# DATA ENGINE (Optimized for Multi-Horizon)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=True)
def get_backtest_data(ticker_list):
    """
    Downloads data and Pre-Calculates ALL features/ranks vectorized.
    Now calculates Targets/Baselines for 5d, 10d, and 21d.
    """
    tickers = list(set([t.strip().upper() for t in ticker_list]))
    data_dict = {}
    
    start_date = "2015-01-01"
    
    progress_bar = st.progress(0)
    for i, t in enumerate(tickers):
        try:
            df = yf.download(t, start=start_date, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            
            # --- VECTORIZED FEATURE CALC ---
            df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Vol_Daily'] = df['LogRet'].rolling(21).std()
            
            # Variables to Rank
            vars_to_rank = []
            
            for w in [5, 10, 21, 63]:
                df[f'Ret_{w}d'] = df['Close'].pct_change(w)
                vars_to_rank.append(f'Ret_{w}d')
                
            df['RealVol_21d'] = df['LogRet'].rolling(21).std() * np.sqrt(252) * 100
            vars_to_rank.append('RealVol_21d')
            
            df['VolRatio_5d'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(63).mean()
            vars_to_rank.append('VolRatio_5d')

            # Calculate Ranks
            rank_cols = []
            for v in vars_to_rank:
                r_col = v + "_Rank"
                df[r_col] = df[v].expanding(min_periods=252).rank(pct=True) * 100
                rank_cols.append(r_col)
            
            # --- MULTI-HORIZON TARGETS ---
            for h in TEST_HORIZONS:
                # Raw Return (for PnL)
                df[f'Target_Return_{h}d'] = df['Close'].shift(-h) / df['Close'] - 1
                
                # Sigma Return (for Signal Baseline)
                expected_move = df['Vol_Daily'] * np.sqrt(h)
                df[f'Target_Sigma_{h}d'] = df[f'Target_Return_{h}d'] / expected_move

                # Baselines
                # Full History (Expanding)
                df[f'Baseline_Full_{h}d'] = df[f'Target_Sigma_{h}d'].expanding(min_periods=252).mean()
                # Recent Regime (Rolling 2y)
                df[f'Baseline_Recent_{h}d'] = df[f'Target_Sigma_{h}d'].rolling(500).mean()
                
                # Shift Baselines to be available at decision time
                df[f'Baseline_Full_{h}d'] = df[f'Baseline_Full_{h}d'].shift(h)
                df[f'Baseline_Recent_{h}d'] = df[f'Baseline_Recent_{h}d'].shift(h)

            df = df.dropna()
            data_dict[t] = {
                "df": df,
                "rank_cols": rank_cols
            }
            
        except Exception as e:
            pass
        progress_bar.progress((i + 1) / len(tickers))
    
    progress_bar.empty()
    return data_dict

# -----------------------------------------------------------------------------
# SIMULATION ENGINE
# -----------------------------------------------------------------------------
def run_simulation(data_dict, horizon, random_months=4):
    """
    Simulates the "Race" for a specific HORIZON.
    """
    valid_dates = list(data_dict[BACKTEST_TICKERS[0]]["df"].index)
    valid_dates = [d for d in valid_dates if d.year >= 2018 and d.year < 2025]
    
    if len(valid_dates) < 100: return pd.DataFrame()

    # Generate Random Start Dates
    start_indices = []
    attempts = 0
    while len(start_indices) < random_months and attempts < 100:
        idx = random.randint(0, len(valid_dates) - 40) # Ensure room for 21d hold
        if all(abs(idx - existing) > 60 for existing in start_indices):
            start_indices.append(idx)
        attempts += 1
    
    results = []

    for start_idx in start_indices:
        # Simulate 20 trading days (approx 1 month)
        test_indices = range(start_idx, start_idx + 20)
        
        for i in test_indices:
            date = valid_dates[i]
            daily_candidates = []
            
            for ticker, info in data_dict.items():
                df = info["df"]
                rank_cols = info["rank_cols"]
                
                if date not in df.index: continue
                row = df.loc[date]
                
                # --- HORIZON SPECIFIC DATA ---
                target_return = row[f'Target_Return_{horizon}d']
                baseline_full = row[f'Baseline_Full_{horizon}d']
                baseline_recent = row[f'Baseline_Recent_{horizon}d']
                
                # --- SIGNAL GENERATION ---
                bull_score_full = 0
                bear_score_full = 0
                bull_score_regime = 0
                bear_score_regime = 0
                
                feature_pairs = list(itertools.combinations(rank_cols, 2))
                
                for r1, r2 in feature_pairs:
                    val1, val2 = row[r1], row[r2]
                    
                    is_bull = (val1 < 25 and val2 < 25) 
                    is_bear = (val1 > 75 and val2 > 75) 
                    
                    if is_bull: 
                        bull_score_full += 1
                        # REGIME CHECK: Only buy if Recent Baseline >= Full Baseline (Momentum is stable/up)
                        if baseline_recent >= baseline_full: 
                            bull_score_regime += 1 
                            
                    if is_bear: 
                        bear_score_full += 1
                        # REGIME CHECK: Only short if Recent Baseline <= Full Baseline (Momentum is down)
                        if baseline_recent <= baseline_full:
                            bear_score_regime += 1
                
                net_full = bull_score_full - bear_score_full
                net_regime = bull_score_regime - bear_score_regime
                
                daily_candidates.append({
                    "Ticker": ticker,
                    "Net_Full": net_full,
                    "Net_Regime": net_regime,
                    "Realized_Return": target_return
                })
            
            if not daily_candidates: continue
            day_df = pd.DataFrame(daily_candidates)
            
            # STRATEGY A (NAIVE)
            day_df = day_df.sort_values("Net_Full", ascending=False)
            longs_A = day_df.head(5)
            shorts_A = day_df.tail(3)
            pnl_A = (longs_A['Realized_Return'].mean() - shorts_A['Realized_Return'].mean()) / 2
            
            # STRATEGY B (REGIME)
            day_df_reg = day_df.sort_values("Net_Regime", ascending=False)
            longs_B = day_df_reg.head(5)
            shorts_B = day_df_reg.tail(3)
            pnl_B = (longs_B['Realized_Return'].mean() - shorts_B['Realized_Return'].mean()) / 2
            
            # Clean NaNs
            pnl_A = 0.0 if np.isnan(pnl_A) else pnl_A
            pnl_B = 0.0 if np.isnan(pnl_B) else pnl_B
            
            results.append({
                "Date": date,
                "Period": f"{valid_dates[start_idx].strftime('%b %Y')}",
                "PnL_Naive": pnl_A,
                "PnL_Regime": pnl_B
            })
            
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Backtest Lab")
    st.title("🧪 The Multi-Timeframe Lab")
    
    st.markdown("""
    **Experiment:** Testing **Naive vs. Regime-Aware** models across **5d, 10d, and 21d** holding periods.
    
    * **Sampling:** Random 1-Month trading blocks from 2018-2024.
    * **Strategy:** Daily Long Top 5 / Short Top 3.
    * **Goal:** Does "Regime Filtering" improve performance on longer hold times?
    """)
    
    if st.button("Run Multi-Horizon Backtest", type="primary"):
        
        with st.spinner("Preprocessing Data (Vectorized)..."):
            data_dict = get_backtest_data(BACKTEST_TICKERS)
        
        if not data_dict:
            st.error("Data download failed.")
            return
            
        # Create Tabs for Results
        tab5, tab10, tab21 = st.tabs(["5-Day Horizon", "10-Day Horizon", "21-Day Horizon"])
        
        # Loop through Horizons
        for horizon, tab in zip(TEST_HORIZONS, [tab5, tab10, tab21]):
            with tab:
                with st.spinner(f"Simulating {horizon}d Trades..."):
                    res_df = run_simulation(data_dict, horizon=horizon, random_months=5)
                
                if res_df.empty:
                    st.warning("No trades generated.")
                    continue
                
                # Cumulative PnL
                res_df['Cum_Naive'] = res_df['PnL_Naive'].cumsum() * 100
                res_df['Cum_Regime'] = res_df['PnL_Regime'].cumsum() * 100
                
                total_naive = res_df['Cum_Naive'].iloc[-1]
                total_regime = res_df['Cum_Regime'].iloc[-1]
                delta = total_regime - total_naive
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Naive Return", f"{total_naive:.2f}%")
                c2.metric("Regime-Aware Return", f"{total_regime:.2f}%")
                c3.metric("Alpha (Improvement)", f"{delta:.2f}%", delta_color="normal")
                
                # Plot
                res_df['Trade_Count'] = range(len(res_df))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res_df['Trade_Count'], y=res_df['Cum_Naive'], name="Naive", line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=res_df['Trade_Count'], y=res_df['Cum_Regime'], name="Regime-Aware", line=dict(color='red', width=3)))
                
                # Annotate Periods
                periods = res_df['Period'].unique()
                for p in periods:
                    start_iloc = res_df[res_df['Period'] == p].index[0]
                    x_pos = res_df.loc[start_iloc, 'Trade_Count']
                    fig.add_vline(x=x_pos, line_dash="dot", line_color="gray", opacity=0.3)
                    fig.add_annotation(x=x_pos, y=0, text=p, showarrow=False, yshift=10, textangle=-90)

                fig.update_layout(
                    title=f"Cumulative PnL ({horizon}-Day Hold)",
                    xaxis_title="Trading Days",
                    yaxis_title="Return (%)",
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
