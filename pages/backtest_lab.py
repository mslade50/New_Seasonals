import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import plotly.graph_objects as go
import itertools

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BACKTEST_TICKERS = ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "AMD", "MSFT", "GOOG", "AMZN", "META", "NFLX", "TLT", "USO", "GLD"]
TEST_HORIZONS = [5, 10, 21]

# -----------------------------------------------------------------------------
# DATA ENGINE (Fully Vectorized)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=True)
def get_backtest_data(ticker_list):
    """
    Downloads data and Pre-Calculates SCORES for the entire history.
    This avoids calculating logic inside the simulation loop.
    """
    tickers = list(set([t.strip().upper() for t in ticker_list]))
    
    # We store a "Master Panel" for each horizon to make the loop instant
    horizon_panels = {h: [] for h in TEST_HORIZONS}
    
    start_date = "2015-01-01"
    
    progress_bar = st.progress(0)
    for i, t in enumerate(tickers):
        try:
            df = yf.download(t, start=start_date, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            
            # 1. Basic Features
            df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Vol_Daily'] = df['LogRet'].rolling(21).std()
            
            # 2. Ranks (Vectorized)
            rank_cols = []
            for w in [5, 10, 21, 63]:
                col = f'Ret_{w}d'
                df[col] = df['Close'].pct_change(w)
                df[col + '_Rank'] = df[col].expanding(min_periods=252).rank(pct=True) * 100
                rank_cols.append(col + '_Rank')
                
            df['RealVol_21d'] = df['LogRet'].rolling(21).std() * np.sqrt(252) * 100
            df['RealVol_21d_Rank'] = df['RealVol_21d'].expanding(min_periods=252).rank(pct=True) * 100
            rank_cols.append('RealVol_21d_Rank')
            
            df['VolRatio_5d'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(63).mean()
            df['VolRatio_5d_Rank'] = df['VolRatio_5d'].expanding(min_periods=252).rank(pct=True) * 100
            rank_cols.append('VolRatio_5d_Rank')

            # 3. Vectorized Score Calculation
            # Instead of looping combos daily, we sum boolean masks for the whole DF
            # This is 1000x faster.
            
            # Create all pairs
            pairs = list(itertools.combinations(rank_cols, 2))
            
            # Pre-compute Bull/Bear masks for all pairs
            # Bull = Both < 25. Bear = Both > 75.
            bull_matrix = pd.DataFrame(0, index=df.index, columns=range(len(pairs)))
            bear_matrix = pd.DataFrame(0, index=df.index, columns=range(len(pairs)))
            
            for idx, (r1, r2) in enumerate(pairs):
                # Bullish Condition: Oversold Confluence
                bull_matrix[idx] = ((df[r1] < 25) & (df[r2] < 25)).astype(int)
                # Bearish Condition: Overbought Confluence
                bear_matrix[idx] = ((df[r1] > 75) & (df[r2] > 75)).astype(int)
            
            # Sum rows to get daily scores
            df['Raw_Bull_Score'] = bull_matrix.sum(axis=1)
            df['Raw_Bear_Score'] = bear_matrix.sum(axis=1)
            df['Net_Score_Full'] = df['Raw_Bull_Score'] - df['Raw_Bear_Score']

            # 4. Process Horizons
            for h in TEST_HORIZONS:
                # Calculate Forward Returns (Sigma Normalized)
                # This enables Risk-Parity PnL
                expected_vol = df['Vol_Daily'] * np.sqrt(h)
                
                # We cap expected vol at reasonable limits to avoid infinity on flat days
                expected_vol = expected_vol.replace(0, np.nan).fillna(method='ffill')
                
                fwd_ret = df['Close'].shift(-h) / df['Close'] - 1
                
                # This is the PnL of 1 unit of risk
                df[f'Sigma_Return_{h}d'] = fwd_ret / expected_vol
                
                # Baselines for Regime
                df[f'Baseline_Full_{h}d'] = df[f'Sigma_Return_{h}d'].expanding(min_periods=252).mean().shift(h)
                df[f'Baseline_Recent_{h}d'] = df[f'Sigma_Return_{h}d'].rolling(500).mean().shift(h)
                
                # Determine Regime Score (Vectorized)
                # If Recent Baseline > Full Baseline, we trust Bulls, ignore Bears (Up Regime)
                up_regime = df[f'Baseline_Recent_{h}d'] >= df[f'Baseline_Full_{h}d']
                down_regime = df[f'Baseline_Recent_{h}d'] < df[f'Baseline_Full_{h}d']
                
                # Regime Logic:
                # If Up Regime: Bull signals count. Bear signals ignored (or reduced).
                # If Down Regime: Bear signals count. Bull signals ignored.
                
                # Constructing Net_Score_Regime
                # We use numpy where for speed
                regime_bulls = np.where(up_regime, df['Raw_Bull_Score'], 0)
                regime_bears = np.where(down_regime, df['Raw_Bear_Score'], 0)
                
                df[f'Net_Score_Regime_{h}d'] = regime_bulls - regime_bears
                
                # Store lightweight version for panel
                subset = df[['Net_Score_Full', f'Net_Score_Regime_{h}d', f'Sigma_Return_{h}d']].copy()
                subset['Ticker'] = t
                horizon_panels[h].append(subset)
            
        except Exception as e:
            pass
        progress_bar.progress((i + 1) / len(tickers))
    
    # Concat all tickers into one Master Panel per horizon
    final_panels = {}
    for h in TEST_HORIZONS:
        if horizon_panels[h]:
            final_panels[h] = pd.concat(horizon_panels[h])
    
    progress_bar.empty()
    return final_panels

# -----------------------------------------------------------------------------
# SIMULATION ENGINE
# -----------------------------------------------------------------------------
def run_full_simulation(panel_df, horizon):
    """
    Simulates the entire history using the pre-computed scores.
    """
    # Pivot so we have Dates as Index and Tickers as Columns for vector sorting
    # We need separate DFs for Full Score, Regime Score, and Returns
    
    score_full = panel_df.pivot(columns='Ticker', values='Net_Score_Full')
    score_regime = panel_df.pivot(columns='Ticker', values=f'Net_Score_Regime_{horizon}d')
    returns = panel_df.pivot(columns='Ticker', values=f'Sigma_Return_{horizon}d')
    
    # Align dates
    common_index = score_full.index.intersection(returns.index)
    score_full = score_full.loc[common_index]
    score_regime = score_regime.loc[common_index]
    returns = returns.loc[common_index]
    
    # Filter for valid dates (post 2016 for ranks to stabilize)
    valid_dates = score_full.index[score_full.index.year >= 2016]
    
    # Container for daily PnL
    pnl_naive = []
    pnl_regime = []
    dates = []
    
    # Loop is still necessary for daily sorting, but data is pre-prepped
    # This loop runs 2500 times but does very light work
    
    for d in valid_dates:
        # Get row slices
        row_full = score_full.loc[d]
        row_regime = score_regime.loc[d]
        row_ret = returns.loc[d]
        
        # NAIVE STRATEGY
        # Sort by Score High to Low
        # Top 5 Longs, Bottom 3 Shorts
        sorted_full = row_full.sort_values(ascending=False)
        longs_idx = sorted_full.head(5).index
        shorts_idx = sorted_full.tail(3).index
        
        # PnL = Average Sigma of Longs - Average Sigma of Shorts
        # This effectively models a portfolio rebalanced daily where each position is 1 Unit of Risk
        day_pnl_naive = (row_ret[longs_idx].mean() - row_ret[shorts_idx].mean()) / 2
        
        # REGIME STRATEGY
        sorted_regime = row_regime.sort_values(ascending=False)
        longs_idx_reg = sorted_regime.head(5).index
        shorts_idx_reg = sorted_regime.tail(3).index
        
        day_pnl_regime = (row_ret[longs_idx_reg].mean() - row_ret[shorts_idx_reg].mean()) / 2
        
        # Handle NaNs (if data missing for that day)
        if np.isnan(day_pnl_naive): day_pnl_naive = 0
        if np.isnan(day_pnl_regime): day_pnl_regime = 0
        
        pnl_naive.append(day_pnl_naive)
        pnl_regime.append(day_pnl_regime)
        dates.append(d)
        
    results = pd.DataFrame({
        "Date": dates,
        "Daily_Sigma_Naive": pnl_naive,
        "Daily_Sigma_Regime": pnl_regime
    }).set_index("Date")
    
    return results

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Full History Backtest")
    st.title("🧪 Full History Lab (Vol-Adjusted)")
    
    st.markdown("""
    **Testing Protocol:**
    1.  **Timeline:** Full history (2016 - Present). No random sampling.
    2.  **Position Sizing:** **Risk Parity (Sigma).** We bet 1 Standard Deviation of risk per trade.
        * *Example:* If TSLA Vol is 4% and TLT Vol is 1%, we trade 4x more notional in TLT.
    3.  **Benchmark:** * 🔵 **Naive:** Trades all signals based on 25y history.
        * 🔴 **Regime-Aware:** Only trades signals confirmed by the last 2y trend.
    """)
    
    if st.button("Run Full Simulation", type="primary"):
        
        with st.spinner("Vectorizing Scores for entire history (this is fast)..."):
            master_panels = get_backtest_data(BACKTEST_TICKERS)
        
        if not master_panels:
            st.error("Data processing failed.")
            return
            
        # Create Tabs for Results
        tab5, tab10, tab21 = st.tabs(["5-Day Horizon", "10-Day Horizon", "21-Day Horizon"])
        
        for horizon, tab in zip(TEST_HORIZONS, [tab5, tab10, tab21]):
            with tab:
                if horizon not in master_panels:
                    st.warning("No data for this horizon.")
                    continue
                    
                with st.spinner(f"Simulating {horizon}d History..."):
                    res_df = run_full_simulation(master_panels[horizon], horizon)
                
                # Cumulative Sum (Sum of Sigmas Captured)
                res_df['Cum_Naive'] = res_df['Daily_Sigma_Naive'].cumsum()
                res_df['Cum_Regime'] = res_df['Daily_Sigma_Regime'].cumsum()
                
                total_naive = res_df['Cum_Naive'].iloc[-1]
                total_regime = res_df['Cum_Regime'].iloc[-1]
                
                # Display Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Naive Total Payoff", f"{total_naive:.2f} R", help="Total Risk Units captured")
                c2.metric("Regime Total Payoff", f"{total_regime:.2f} R")
                delta = total_regime - total_naive
                c3.metric("Regime Improvement", f"{delta:.2f} R", delta_color="normal")
                
                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Cum_Naive'], name="Naive Model", line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Cum_Regime'], name="Regime-Aware", line=dict(color='red', width=2)))
                
                fig.update_layout(
                    title=f"Cumulative Performance (in Risk Units)",
                    yaxis_title="Cumulative Sigma (R)",
                    xaxis_title="Year",
                    height=500,
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Monthly Stats Heatmap (Optional but nice)
                with st.expander("Monthly Breakdown"):
                    monthly = res_df.resample('M').sum()
                    monthly['Year'] = monthly.index.year
                    monthly['Month'] = monthly.index.strftime('%b')
                    
                    pivot = monthly.pivot(index='Year', columns='Month', values='Daily_Sigma_Regime')
                    st.dataframe(pivot.style.background_gradient(cmap='RdBu', vmin=-2, vmax=2).format("{:.2f}"), use_container_width=True)

if __name__ == "__main__":
    main()
