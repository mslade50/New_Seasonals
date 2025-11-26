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
    Downloads data and Pre-Calculates SCORES for Naive vs Optimized.
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
            
            # --- PREP FOR MODEL B: WEIGHTS ---
            # Sample Size Weights (Tanh)
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
                # We use the Returns Baseline to proxy "Is the edge decaying?"
                # Ideally we check Alpha per signal, but vectorized, we check Alpha per Ticker/Horizon
                # If Ticker's Recent Sigma Baseline < Full Sigma Baseline, we kill Bulls.
                
                full_base = df[f'Sigma_Return_{h}d'].expanding(min_periods=252).mean().shift(h)
                recent_base = df[f'Sigma_Return_{h}d'].rolling(500).mean().shift(h)
                
                # REGIME FILTER MASKS (0 or 1)
                # Bull Valid? Only if Recent >= Full (Momentum maintained)
                valid_bull_regime = (recent_base >= full_base).astype(int)
                # Bear Valid? Only if Recent <= Full (Weakness maintained)
                valid_bear_regime = (recent_base <= full_base).astype(int)
                
                # --- MODEL B: OPTIMIZED ---
                # Score = (Matrix * SampleWeight * RegimeFilter).sum()
                
                # We apply the Regime Filter to the *Weights* directly for speed
                # If regime is invalid, weight becomes 0.
                
                # Note: valid_bull_regime is a Series, bull_weights is a DataFrame. 
                # We multiply column-wise (axis=0) but pandas aligns on index automatically.
                
                final_bull_weights = bull_weights.multiply(valid_bull_regime, axis=0)
                final_bear_weights = bear_weights.multiply(valid_bear_regime, axis=0)
                
                df['Score_Opt_Bull'] = (bull_matrix * final_bull_weights).sum(axis=1)
                df['Score_Opt_Bear'] = (bear_matrix * final_bear_weights).sum(axis=1)
                
                df[f'Net_Score_Optimized_{h}d'] = df['Score_Opt_Bull'] - df['Score_Opt_Bear']
                
                # Store Panel
                subset = df[[
                    'Net_Score_Naive', 
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
    Simulates Naive vs Optimized Strategies.
    """
    score_naive = panel_df.pivot(columns='Ticker', values='Net_Score_Naive')
    score_opt   = panel_df.pivot(columns='Ticker', values=f'Net_Score_Optimized_{horizon}d')
    returns     = panel_df.pivot(columns='Ticker', values=f'Sigma_Return_{horizon}d')
    
    common_index = score_naive.index.intersection(returns.index)
    score_naive = score_naive.loc[common_index]
    score_opt = score_opt.loc[common_index]
    returns = returns.loc[common_index]
    
    valid_dates = score_naive.index[score_naive.index.year >= 2016]
    
    pnl_naive = []
    pnl_opt = []
    dates = []
    
    for d in valid_dates:
        row_naive = score_naive.loc[d]
        row_opt   = score_opt.loc[d]
        row_ret   = returns.loc[d]
        
        # STRATEGY A: NAIVE
        sorted_naive = row_naive.sort_values(ascending=False)
        longs_A = sorted_naive.head(5).index
        shorts_A = sorted_naive.tail(3).index
        day_pnl_A = (row_ret[longs_A].mean() - row_ret[shorts_A].mean()) / 2
        
        # STRATEGY B: OPTIMIZED (Weighted + Regime)
        sorted_opt = row_opt.sort_values(ascending=False)
        longs_B = sorted_opt.head(5).index
        shorts_B = sorted_opt.tail(3).index
        day_pnl_B = (row_ret[longs_B].mean() - row_ret[shorts_B].mean()) / 2
        
        if np.isnan(day_pnl_A): day_pnl_A = 0
        if np.isnan(day_pnl_B): day_pnl_B = 0
        
        pnl_naive.append(day_pnl_A)
        pnl_opt.append(day_pnl_B)
        dates.append(d)
        
    results = pd.DataFrame({
        "Date": dates,
        "Daily_Sigma_Naive": pnl_naive,
        "Daily_Sigma_Opt": pnl_opt
    }).set_index("Date")
    
    return results

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Grand Unified Backtest")
    st.title("🧪 The 'Grand Unified' Lab")
    
    st.markdown("""
    **The Final Boss Battle:**
    
    * 🔵 **Naive Model:** * Equal Weighting (1.0).
        * Ignores Regime Changes.
    
    * 🟣 **Optimized Model (Grand Unified):** * **Sample Weighting:** Signals weighted by `Tanh` (Low sample = Low weight).
        * **Regime Filter:** Decaying signals (Recent < Historic) are zeroed out.
        * **Alpha Logic:** Uses Vol-Adjusted Returns (Sigma).
    """)
    
    if st.button("Run Unified Simulation", type="primary"):
        
        with st.spinner("Crunching vector matrices (Weights + Regimes)..."):
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
                
                res_df['Cum_Naive'] = res_df['Daily_Sigma_Naive'].cumsum()
                res_df['Cum_Opt'] = res_df['Daily_Sigma_Opt'].cumsum()
                
                total_naive = res_df['Cum_Naive'].iloc[-1]
                total_opt = res_df['Cum_Opt'].iloc[-1]
                delta = total_opt - total_naive
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Naive Payoff", f"{total_naive:.2f} R")
                c2.metric("Optimized Payoff", f"{total_opt:.2f} R")
                c3.metric("Improvement", f"{delta:.2f} R", delta_color="normal")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Cum_Naive'], name="Naive Model", line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Cum_Opt'], name="Optimized (Weighted + Regime)", line=dict(color='purple', width=3)))
                
                fig.update_layout(
                    title=f"Grand Unified Performance ({horizon}d Hold)",
                    yaxis_title="Cumulative Sigma (R)",
                    xaxis_title="Year",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
