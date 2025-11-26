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
# We use a slightly tighter universe for the backtest to ensure speed
BACKTEST_TICKERS = ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "AMD", "MSFT", "GOOG", "AMZN", "META", "NFLX", "TLT", "USO", "GLD"]

# The specific horizon we are testing (Standardizing on 5d for speed)
TEST_HORIZON = 5 

# -----------------------------------------------------------------------------
# DATA ENGINE (Optimized for Backtesting)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=True)
def get_backtest_data(ticker_list):
    """
    Downloads data and Pre-Calculates ALL features/ranks vectorized.
    """
    tickers = list(set([t.strip().upper() for t in ticker_list]))
    data_dict = {}
    
    # Download 10y of data (covers the random sampling window)
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
            
            # Returns & Vol
            for w in [5, 10, 21, 63]:
                df[f'Ret_{w}d'] = df['Close'].pct_change(w)
                vars_to_rank.append(f'Ret_{w}d')
                
            df['RealVol_21d'] = df['LogRet'].rolling(21).std() * np.sqrt(252) * 100
            vars_to_rank.append('RealVol_21d')
            
            df['VolRatio_5d'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(63).mean()
            vars_to_rank.append('VolRatio_5d')

            # Calculate Ranks (Expanding)
            rank_cols = []
            for v in vars_to_rank:
                r_col = v + "_Rank"
                # Min periods 252 ensures we have 1 year of data before ranking
                df[r_col] = df[v].expanding(min_periods=252).rank(pct=True) * 100
                rank_cols.append(r_col)
            
            # Calculate Forward Returns (The Truth)
            # Raw Return (for PnL)
            df['Target_Return'] = df['Close'].shift(-TEST_HORIZON) / df['Close'] - 1
            
            # Sigma Return (for Signal)
            expected_move = df['Vol_Daily'] * np.sqrt(TEST_HORIZON)
            df['Target_Sigma'] = df['Target_Return'] / expected_move

            # --- PRE-CALCULATE BASELINES (Vectorized) ---
            # 1. Full History Baseline (Expanding Mean of Sigma)
            df['Baseline_Full'] = df['Target_Sigma'].expanding(min_periods=252).mean()
            
            # 2. Recent Regime Baseline (Rolling 2yr Mean of Sigma)
            # Using 2yr (500d) for faster regime detection in backtest
            df['Baseline_Recent'] = df['Target_Sigma'].rolling(500).mean()
            
            # Shift Baselines by +Horizon because at time T we don't know the return of T
            df['Baseline_Full'] = df['Baseline_Full'].shift(TEST_HORIZON)
            df['Baseline_Recent'] = df['Baseline_Recent'].shift(TEST_HORIZON)

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
def run_simulation(data_dict, random_months=4):
    """
    Simulates the "Race" between Naive and Regime-Aware models.
    """
    # 1. Generate Random Start Dates (from 2018 to 2024)
    # We look for valid trading days
    valid_dates = list(data_dict[BACKTEST_TICKERS[0]]["df"].index)
    
    # Filter for dates where we likely have data for everyone
    valid_dates = [d for d in valid_dates if d.year >= 2018 and d.year < 2025]
    
    if len(valid_dates) < 100: return pd.DataFrame()

    # Pick N random indices, ensuring they are spaced out (at least 60 days apart)
    start_indices = []
    attempts = 0
    while len(start_indices) < random_months and attempts < 100:
        idx = random.randint(0, len(valid_dates) - 30)
        # Check spacing
        if all(abs(idx - existing) > 60 for existing in start_indices):
            start_indices.append(idx)
        attempts += 1
    
    results = []

    # 2. The Loop
    for start_idx in start_indices:
        
        # We simulate a "Month" (20 trading days)
        test_indices = range(start_idx, start_idx + 20)
        
        for i in test_indices:
            date = valid_dates[i]
            
            daily_candidates = []
            
            # Scan every ticker for this specific day
            for ticker, info in data_dict.items():
                df = info["df"]
                rank_cols = info["rank_cols"]
                
                if date not in df.index: continue
                
                row = df.loc[date]
                
                # Get Stats
                # Since we shifted baselines, 'Baseline_Full' at date T *is* the mean of (0...T-Horizon)
                # But 'Target_Sigma' is the realized future return. We need to measure Alpha using
                # hypothetical expected returns based on historical Conditional Means.
                # NOTE: For speed in this specific script, we are approximating the "Signal Strength"
                # by simply checking if the *Current Ranks* are extreme. 
                
                # --- FAST SIGNAL GENERATION ---
                # We sum the "Extremity" of all features to generate a conviction score
                # If Rank > 80: +1 Score. If Rank < 20: -1 Score.
                
                bull_score_full = 0
                bear_score_full = 0
                
                bull_score_regime = 0
                bear_score_regime = 0
                
                # We check all pairs of features (approx 15 pairs)
                feature_pairs = list(itertools.combinations(rank_cols, 2))
                
                baseline_full = row['Baseline_Full']
                baseline_recent = row['Baseline_Recent']
                
                for r1, r2 in feature_pairs:
                    val1, val2 = row[r1], row[r2]
                    
                    # Double Tail Check
                    is_bull = (val1 < 25 and val2 < 25) # Oversold = Bullish (Mean Rev) for Returns
                    is_bear = (val1 > 75 and val2 > 75) # Overbought = Bearish
                    
                    # Logic adjustment: High Rank Returns usually mean Mean Reversion (Bearish) next?
                    # Let's assume Mean Reversion Logic: 
                    # High Rank (>75) -> Expect Down
                    # Low Rank (<25) -> Expect Up
                    
                    if is_bull: # Both < 25
                        bull_score_full += 1
                        # Regime Check: Only count if Recent Baseline is NOT significantly higher than Full 
                        # (If Recent Baseline is high, it means we are in a strong uptrend, so "Low Rank" might not be a buy)
                        if baseline_recent >= baseline_full: 
                            bull_score_regime += 1 # Trend is up, buy the dip
                            
                    if is_bear: # Both > 75
                        bear_score_full += 1
                        # Regime Check: Only short if Recent Baseline is lower (Downtrend)
                        if baseline_recent <= baseline_full:
                            bear_score_regime += 1
                
                # Net Scores
                net_full = bull_score_full - bear_score_full
                net_regime = bull_score_regime - bear_score_regime
                
                daily_candidates.append({
                    "Ticker": ticker,
                    "Net_Full": net_full,
                    "Net_Regime": net_regime,
                    "Realized_Return": row['Target_Return']
                })
            
            # --- PORTFOLIO CONSTRUCTION ---
            if not daily_candidates: continue
            day_df = pd.DataFrame(daily_candidates)
            
            # STRATEGY A (NAIVE): Top 5 Longs (High Net Full), Top 3 Shorts (Low Net Full)
            day_df = day_df.sort_values("Net_Full", ascending=False)
            longs_A = day_df.head(5)
            shorts_A = day_df.tail(3)
            
            # STRATEGY B (REGIME): Top 5 Longs (High Net Regime), Top 3 Shorts
            day_df_reg = day_df.sort_values("Net_Regime", ascending=False)
            longs_B = day_df_reg.head(5)
            shorts_B = day_df_reg.tail(3)
            
            # Calc PnL (Equal Weight)
            # Long PnL = Return. Short PnL = -Return.
            
            pnl_A = (longs_A['Realized_Return'].mean() - shorts_A['Realized_Return'].mean()) / 2
            pnl_B = (longs_B['Realized_Return'].mean() - shorts_B['Realized_Return'].mean()) / 2
            
            # Handle NaN (if no trades)
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
    st.title("🧪 The Backtest Lab")
    
    st.markdown("""
    **Hypothesis Test:** Is "Signal Stability" actually helpful?
    
    We compare two strategies over **Random 1-Month Samples** from the last 5 years:
    1.  **🔵 Naive Model:** Trades purely based on Full History (25y) signals.
    2.  **🔴 Regime-Aware Model:** Only takes trades where the Recent Baseline (2y) confirms the signal.
    
    *Strategy: Long Top 5 / Short Top 3 daily. Holding Period: 5 Days.*
    """)
    
    # Run Button
    if st.button("Run Randomized Backtest", type="primary"):
        
        with st.spinner("Preprocessing Data (This happens once)..."):
            data_dict = get_backtest_data(BACKTEST_TICKERS)
        
        if not data_dict:
            st.error("Data download failed.")
            return
            
        with st.spinner("Simulating Random Market Periods..."):
            res_df = run_simulation(data_dict, random_months=5)
        
        if res_df.empty:
            st.error("Simulation produced no trades (likely insufficient data overlap).")
            return
            
        # --- RESULTS ANALYSIS ---
        
        # Cumulative PnL
        res_df['Cum_Naive'] = res_df['PnL_Naive'].cumsum() * 100
        res_df['Cum_Regime'] = res_df['PnL_Regime'].cumsum() * 100
        
        # Final Stats
        total_naive = res_df['Cum_Naive'].iloc[-1]
        total_regime = res_df['Cum_Regime'].iloc[-1]
        
        delta = total_regime - total_naive
        
        # KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("Naive Model Return", f"{total_naive:.2f}%")
        c2.metric("Regime-Aware Return", f"{total_regime:.2f}%")
        c3.metric("Improvement (Alpha)", f"{delta:.2f}%", delta_color="normal")
        
        # Plot
        st.subheader("Equity Curve Comparison")
        
        # We want to show gaps between periods visually
        # Create a sequential index for plotting to avoid plotting the gaps as flat lines
        res_df['Trade_Count'] = range(len(res_df))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res_df['Trade_Count'], y=res_df['Cum_Naive'], name="Naive (Full History)", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=res_df['Trade_Count'], y=res_df['Cum_Regime'], name="Regime-Aware (Stable)", line=dict(color='red', width=3)))
        
        # Add visual bands for the different periods
        periods = res_df['Period'].unique()
        # Find start index of each period
        for p in periods:
            start_iloc = res_df[res_df['Period'] == p].index[0]
            # Map back to Trade_Count
            x_pos = res_df.loc[start_iloc, 'Trade_Count']
            fig.add_vline(x=x_pos, line_dash="dot", line_color="gray", opacity=0.5)
            fig.add_annotation(x=x_pos, y=0, text=p, showarrow=False, yshift=20, textangle=-90)

        fig.update_layout(
            title="Cumulative PnL (Sequential Trade Days across Random Epochs)",
            xaxis_title="Trading Days Active",
            yaxis_title="Cumulative Return (%)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Breakdown Table
        st.subheader("Performance by Sample Period")
        period_stats = res_df.groupby("Period").agg({
            "PnL_Naive": "sum",
            "PnL_Regime": "sum"
        }).reset_index()
        
        period_stats['Naive %'] = (period_stats['PnL_Naive'] * 100).map("{:.2f}%".format)
        period_stats['Regime %'] = (period_stats['PnL_Regime'] * 100).map("{:.2f}%".format)
        period_stats['Winner'] = np.where(period_stats['PnL_Regime'] > period_stats['PnL_Naive'], "Regime-Aware", "Naive")
        
        st.dataframe(
            period_stats[['Period', 'Naive %', 'Regime %', 'Winner']],
            use_container_width=True
        )

if __name__ == "__main__":
    main()
