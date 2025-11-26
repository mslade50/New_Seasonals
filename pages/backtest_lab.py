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
# DATA ENGINE (Vectorized with Weighting)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=True)
def get_backtest_data(ticker_list):
    """
    Downloads data and Pre-Calculates SCORES (Standard vs Weighted).
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

            # 3. Vectorized Signal Generation
            pairs = list(itertools.combinations(rank_cols, 2))
            
            # Initialize Boolean Matrices (Rows=Dates, Cols=Pairs)
            # 1 = Signal Triggered, 0 = No Signal
            bull_matrix = pd.DataFrame(0, index=df.index, columns=range(len(pairs)))
            bear_matrix = pd.DataFrame(0, index=df.index, columns=range(len(pairs)))
            
            for idx, (r1, r2) in enumerate(pairs):
                # Bullish: Oversold Confluence (<25)
                bull_matrix[idx] = ((df[r1] < 25) & (df[r2] < 25)).astype(int)
                # Bearish: Overbought Confluence (>75)
                bear_matrix[idx] = ((df[r1] > 75) & (df[r2] > 75)).astype(int)
            
            # --- METHOD A: STANDARD (Equal Weight) ---
            df['Score_Standard_Bull'] = bull_matrix.sum(axis=1)
            df['Score_Standard_Bear'] = bear_matrix.sum(axis=1)
            df['Net_Score_Standard']  = df['Score_Standard_Bull'] - df['Score_Standard_Bear']
            
            # --- METHOD B: WEIGHTED (Tanh Decay) ---
            # 1. Calculate Historical Counts (Cumulative Sum)
            bull_counts = bull_matrix.cumsum()
            bear_counts = bear_matrix.cumsum()
            
            # 2. Calculate Dynamic Weights
            # Tanh(Count / 50) -> 10 samples = 0.2, 50 samples = 0.76
            bull_weights = np.tanh(bull_counts / 50.0)
            bear_weights = np.tanh(bear_counts / 50.0)
            
            # 3. Apply Weights
            df['Score_Weighted_Bull'] = (bull_matrix * bull_weights).sum(axis=1)
            df['Score_Weighted_Bear'] = (bear_matrix * bear_weights).sum(axis=1)
            df['Net_Score_Weighted']  = df['Score_Weighted_Bull'] - df['Score_Weighted_Bear']

            # 4. Process Horizons
            for h in TEST_HORIZONS:
                expected_vol = df['Vol_Daily'] * np.sqrt(h)
                expected_vol = expected_vol.replace(0, np.nan).fillna(method='ffill')
                
                fwd_ret = df['Close'].shift(-h) / df['Close'] - 1
                
                # Risk Unit Return (Sigma)
                df[f'Sigma_Return_{h}d'] = fwd_ret / expected_vol
                
                # Store Data for Master Panel
                subset = df[[
                    'Net_Score_Standard', 
                    'Net_Score_Weighted', 
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
    Simulates Standard vs Weighted Strategies.
    """
    # Pivot Data
    score_std = panel_df.pivot(columns='Ticker', values='Net_Score_Standard')
    score_wgt = panel_df.pivot(columns='Ticker', values='Net_Score_Weighted')
    returns   = panel_df.pivot(columns='Ticker', values=f'Sigma_Return_{horizon}d')
    
    # Align
    common_index = score_std.index.intersection(returns.index)
    score_std = score_std.loc[common_index]
    score_wgt = score_wgt.loc[common_index]
    returns   = returns.loc[common_index]
    
    # Filter for valid dates (post 2016 for ranks to stabilize)
    valid_dates = score_std.index[score_std.index.year >= 2016]
    
    pnl_std = []
    pnl_wgt = []
    dates = []
    
    for d in valid_dates:
        row_std = score_std.loc[d]
        row_wgt = score_wgt.loc[d]
        row_ret = returns.loc[d]
        
        # STRATEGY A: STANDARD (Equal Weight)
        sorted_std = row_std.sort_values(ascending=False)
        longs_std = sorted_std.head(5).index
        shorts_std = sorted_std.tail(3).index
        day_pnl_std = (row_ret[longs_std].mean() - row_ret[shorts_std].mean()) / 2
        
        # STRATEGY B: WEIGHTED (Sample Size Damping)
        sorted_wgt = row_wgt.sort_values(ascending=False)
        longs_wgt = sorted_wgt.head(5).index
        shorts_wgt = sorted_wgt.tail(3).index
        day_pnl_wgt = (row_ret[longs_wgt].mean() - row_ret[shorts_wgt].mean()) / 2
        
        if np.isnan(day_pnl_std): day_pnl_std = 0
        if np.isnan(day_pnl_wgt): day_pnl_wgt = 0
        
        pnl_std.append(day_pnl_std)
        pnl_wgt.append(day_pnl_wgt)
        dates.append(d)
        
    results = pd.DataFrame({
        "Date": dates,
        "Daily_Sigma_Std": pnl_std,
        "Daily_Sigma_Wgt": pnl_wgt
    }).set_index("Date")
    
    return results

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Weighting Lab")
    st.title("🧪 The Weighting Experiment")
    
    st.markdown("""
    **Hypothesis:** Does penalizing signals with low sample sizes (using `Tanh` damping) improve performance?
    
    * 🔵 **Standard Model:** All signals count as 1.0 (Equal Weight).
    * 🔴 **Weighted Model:** Signals are weighted by `Tanh(History/50)`. 
      *(10 samples = 0.2 weight, 50 samples = 0.76 weight)*.
    """)
    
    if st.button("Run Weighting Experiment", type="primary"):
        
        with st.spinner("Vectorizing Standard vs Weighted Scores..."):
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
                    
                with st.spinner(f"Simulating {horizon}d History..."):
                    res_df = run_full_simulation(master_panels[horizon], horizon)
                
                res_df['Cum_Std'] = res_df['Daily_Sigma_Std'].cumsum()
                res_df['Cum_Wgt'] = res_df['Daily_Sigma_Wgt'].cumsum()
                
                total_std = res_df['Cum_Std'].iloc[-1]
                total_wgt = res_df['Cum_Wgt'].iloc[-1]
                delta = total_wgt - total_std
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Standard Return", f"{total_std:.2f} R")
                c2.metric("Weighted Return", f"{total_wgt:.2f} R")
                c3.metric("Improvement", f"{delta:.2f} R", delta_color="normal")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Cum_Std'], name="Standard (Equal Weight)", line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Cum_Wgt'], name="Weighted (Sample Damping)", line=dict(color='red', width=2)))
                
                fig.update_layout(
                    title=f"Standard vs Weighted Performance ({horizon}d Hold)",
                    yaxis_title="Cumulative Sigma (R)",
                    xaxis_title="Year",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
