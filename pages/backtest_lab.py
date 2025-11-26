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
# DATA ENGINE (Grand Unified Vectorization)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=True)
def get_backtest_data(ticker_list):
    """
    Downloads data and Pre-Calculates SCORES for 3 Models.
    """
    tickers = list(set([t.strip().upper() for t in ticker_list]))
    
    # Store panels
    horizon_panels = {h: [] for h in TEST_HORIZONS}
    
    start_date = "2015-01-01"
    
    progress_bar = st.progress(0)
    for i, t in enumerate(tickers):
        try:
            df = yf.download(t, start=start_date, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            
            # 1. Features
            df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Vol_Daily'] = df['LogRet'].rolling(21).std()
            
            # 2. Ranks
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

            # 3. Raw Signal Matrices (1 = Trigger, 0 = No)
            pairs = list(itertools.combinations(rank_cols, 2))
            
            bull_matrix = pd.DataFrame(0, index=df.index, columns=range(len(pairs)))
            bear_matrix = pd.DataFrame(0, index=df.index, columns=range(len(pairs)))
            
            for idx, (r1, r2) in enumerate(pairs):
                bull_matrix[idx] = ((df[r1] < 25) & (df[r2] < 25)).astype(int)
                bear_matrix[idx] = ((df[r1] > 75) & (df[r2] > 75)).astype(int)
            
            # --- MODEL A: NAIVE (Raw Sums) ---
            df['Score_Naive_Bull'] = bull_matrix.sum(axis=1)
            df['Score_Naive_Bear'] = bear_matrix.sum(axis=1)
            df['Net_Score_Naive']  = df['Score_Naive_Bull'] - df['Score_Naive_Bear']
            
            # --- PREP FOR WEIGHTING ---
            bull_counts = bull_matrix.cumsum()
            bear_counts = bear_matrix.cumsum()
            bull_weights = np.tanh(bull_counts / 50.0)
            bear_weights = np.tanh(bear_counts / 50.0)

            # 4. Process Horizons (Regime Logic)
            for h in TEST_HORIZONS:
                expected_vol = df['Vol_Daily'] * np.sqrt(h)
                expected_vol = expected_vol.replace(0, np.nan).fillna(method='ffill')
                fwd_ret = df['Close'].shift(-h) / df['Close'] - 1
                
                # Risk Unit Return (Sigma)
                df[f'Sigma_Return_{h}d'] = fwd_ret / expected_vol
                
                # --- REGIME CALCULATION ---
                full_base = df[f'Sigma_Return_{h}d'].expanding(min_periods=252).mean().shift(h)
                recent_base = df[f'Sigma_Return_{h}d'].rolling(500).mean().shift(h)
                
                # REGIME FILTER MASKS (0 or 1)
                # 1 if signal is stable, 0 if decaying
                valid_bull_regime = (recent_base >= full_base).astype(int)
                valid_bear_regime = (recent_base <= full_base).astype(int)
                
                # --- MODEL B: REGIME ONLY (Equal Weight) ---
                # We mask the raw matrix with the regime filter
                # Note: Multiply series aligns on index
                
                # Bull Score = Sum of (Matrix * RegimeMask)
                # Since RegimeMask is per Ticker (not per signal), we multiply the final sum
                # If Regime is invalid, the entire Bull Score for that day becomes 0.
                
                df['Score_RegimeEqW_Bull'] = df['Score_Naive_Bull'] * valid_bull_regime
                df['Score_RegimeEqW_Bear'] = df['Score_Naive_Bear'] * valid_bear_regime
                df[f'Net_Score_RegimeEqW_{h}d'] = df['Score_RegimeEqW_Bull'] - df['Score_RegimeEqW_Bear']
                
                # --- MODEL C: OPTIMIZED (Regime + Weighted) ---
                # Apply regime filter to weights, then multiply matrix
                final_bull_weights = bull_weights.multiply(valid_bull_regime, axis=0)
                final_bear_weights = bear_weights.multiply(valid_bear_regime, axis=0)
                
                df['Score_Opt_Bull'] = (bull_matrix * final_bull_weights).sum(axis=1)
                df['Score_Opt_Bear'] = (bear_matrix * final_bear_weights).sum(axis=1)
                
                df[f'Net_Score_Optimized_{h}d'] = df['Score_Opt_Bull'] - df['Score_Opt_Bear']
                
                # Store Panel
                subset = df[[
                    'Net_Score_Naive', 
                    f'Net_Score_RegimeEqW_{h}d',
                    f'Net_Score_Optimized_{h}d', 
                    f'Sigma_Return_{h}d'
                ]].copy()
                subset['Ticker'] = t
                horizon_panels[h].append(subset)
            
        except Exception as e:
            pass
        progress_bar.progress((i + 1) / len(tickers))
    
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
    Simulates 3 Strategies.
    """
    score_naive = panel_df.pivot(columns='Ticker', values='Net_Score_Naive')
    score_reg   = panel_df.pivot(columns='Ticker', values=f'Net_Score_RegimeEqW_{horizon}d')
    score_opt   = panel_df.pivot(columns='Ticker', values=f'Net_Score_Optimized_{horizon}d')
    returns     = panel_df.pivot(columns='Ticker', values=f'Sigma_Return_{horizon}d')
    
    common_index = score_naive.index.intersection(returns.index)
    score_naive = score_naive.loc[common_index]
    score_reg   = score_reg.loc[common_index]
    score_opt   = score_opt.loc[common_index]
    returns     = returns.loc[common_index]
    
    valid_dates = score_naive.index[score_naive.index.year >= 2016]
    
    pnl_naive = []
    pnl_reg = []
    pnl_opt = []
    dates = []
    
    for d in valid_dates:
        row_naive = score_naive.loc[d]
        row_reg   = score_reg.loc[d]
        row_opt   = score_opt.loc[d]
        row_ret   = returns.loc[d]
        
        # MODEL A: NAIVE
        s_naive = row_naive.sort_values(ascending=False)
        pnl_A = (row_ret[s_naive.head(5).index].mean() - row_ret[s_naive.tail(3).index].mean()) / 2
        
        # MODEL B: REGIME (EQUAL WEIGHT)
        s_reg = row_reg.sort_values(ascending=False)
        pnl_B = (row_ret[s_reg.head(5).index].mean() - row_ret[s_reg.tail(3).index].mean()) / 2
        
        # MODEL C: OPTIMIZED (REGIME + WEIGHTED)
        s_opt = row_opt.sort_values(ascending=False)
        pnl_C = (row_ret[s_opt.head(5).index].mean() - row_ret[s_opt.tail(3).index].mean()) / 2
        
        if np.isnan(pnl_A): pnl_A = 0
        if np.isnan(pnl_B): pnl_B = 0
        if np.isnan(pnl_C): pnl_C = 0
        
        pnl_naive.append(pnl_A)
        pnl_reg.append(pnl_B)
        pnl_opt.append(pnl_C)
        dates.append(d)
        
    results = pd.DataFrame({
        "Date": dates,
        "Sigma_Naive": pnl_naive,
        "Sigma_Regime": pnl_reg,
        "Sigma_Opt": pnl_opt
    }).set_index("Date")
    
    return results

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Three-Model Backtest")
    st.title("🧪 The 3-Model Showdown")
    
    st.markdown("""
    **Isolating the Alpha:**
    
    * 🔵 **Naive:** Equal Weight. Ignores Regime.
    * 🟠 **Regime (EqW):** Equal Weight. **Regime Filter Only** (Kills decaying signals).
    * 🟣 **Optimized:** **Regime Filter + Sample Weighting** (Kills decaying + Dampens noise).
    """)
    
    if st.button("Run 3-Model Simulation", type="primary"):
        
        with st.spinner("Vectorizing 3 Models..."):
            master_panels = get_backtest_data(BACKTEST_TICKERS)
        
        if not master_panels:
            st.error("Data processing failed.")
            return
            
        tab5, tab10, tab21 = st.tabs(["5-Day Horizon", "10-Day Horizon", "21-Day Horizon"])
        
        for horizon, tab in zip(TEST_HORIZONS, [tab5, tab10, tab21]):
            with tab:
                if horizon not in master_panels:
                    st.warning("No data.")
                    continue
                    
                with st.spinner(f"Simulating {horizon}d..."):
                    res_df = run_full_simulation(master_panels[horizon], horizon)
                
                res_df['Cum_Naive'] = res_df['Sigma_Naive'].cumsum()
                res_df['Cum_Regime'] = res_df['Sigma_Regime'].cumsum()
                res_df['Cum_Opt'] = res_df['Sigma_Opt'].cumsum()
                
                # Stats
                final_naive = res_df['Cum_Naive'].iloc[-1]
                final_reg = res_df['Cum_Regime'].iloc[-1]
                final_opt = res_df['Cum_Opt'].iloc[-1]
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Naive Payoff", f"{final_naive:.2f} R")
                c2.metric("Regime (EqW) Payoff", f"{final_reg:.2f} R", delta=f"{final_reg - final_naive:.2f} vs Naive")
                c3.metric("Optimized Payoff", f"{final_opt:.2f} R", delta=f"{final_opt - final_reg:.2f} vs Regime")
                
                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Cum_Naive'], name="Naive", line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Cum_Regime'], name="Regime Only (EqW)", line=dict(color='orange', width=2, dash='dot')))
                fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Cum_Opt'], name="Optimized (Regime + Wgt)", line=dict(color='purple', width=3)))
                
                fig.update_layout(
                    title=f"Performance Attribution ({horizon}d Hold)",
                    yaxis_title="Cumulative Sigma (R)",
                    height=550,
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
