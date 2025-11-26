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
# Expanded Universe (~60 Assets) to allow for Top 20 Longs / Top 5 Shorts
BACKTEST_TICKERS = [
    # Indices/Macro
    "SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "GLD", "SLV", "USO", "UUP", "BTC-USD",
    # Sectors
    "XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLU", "XLI", "XLB", "XBI", "SMH", "KRE", "GDX",
    # Mega Caps & High Vol
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "NFLX", "AMD", "INTC",
    "JPM", "BAC", "WFC", "GS", "MS",
    "XOM", "CVX", "COP",
    "LLY", "UNH", "JNJ", "PFE",
    "HD", "MCD", "NKE", "SBUX",
    "CAT", "DE", "BA", "LMT", "RTX",
    "PLTR", "COIN", "MSTR"
]

TEST_HORIZONS = [2, 5, 10, 21]
LONG_COUNT = 20
SHORT_COUNT = 5

# -----------------------------------------------------------------------------
# DATA ENGINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=True)
def get_backtest_data(ticker_list):
    tickers = list(set([t.strip().upper() for t in ticker_list]))
    horizon_panels = {h: [] for h in TEST_HORIZONS}
    start_date = "2016-01-01" # Need data for rank calculation
    
    progress_bar = st.progress(0)
    
    # Download Batch (Much faster than loop)
    try:
        all_data = yf.download(tickers, start=start_date, progress=False, auto_adjust=True, group_by='ticker')
    except:
        st.error("Download failed.")
        return {}

    for i, t in enumerate(tickers):
        try:
            # Handle multi-level column structure from yf batch download
            if isinstance(all_data.columns, pd.MultiIndex):
                if t not in all_data.columns.levels[0]: continue
                df = all_data[t].copy()
            else:
                df = all_data.copy() # Single ticker case

            df = df.dropna()
            if df.empty: continue

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

            # 3. Signals
            pairs = list(itertools.combinations(rank_cols, 2))
            
            bull_matrix = pd.DataFrame(0, index=df.index, columns=range(len(pairs)))
            bear_matrix = pd.DataFrame(0, index=df.index, columns=range(len(pairs)))
            
            for idx, (r1, r2) in enumerate(pairs):
                bull_matrix[idx] = ((df[r1] < 25) & (df[r2] < 25)).astype(int)
                bear_matrix[idx] = ((df[r1] > 75) & (df[r2] > 75)).astype(int)
            
            # MODEL A: NAIVE
            df['Score_Naive'] = bull_matrix.sum(axis=1) - bear_matrix.sum(axis=1)
            
            # WEIGHTS
            bull_counts = bull_matrix.cumsum()
            bear_counts = bear_matrix.cumsum()
            bull_weights = np.tanh(bull_counts / 50.0)
            bear_weights = np.tanh(bear_counts / 50.0)

            # 4. Horizons
            for h in TEST_HORIZONS:
                expected_vol = df['Vol_Daily'] * np.sqrt(h)
                expected_vol = expected_vol.replace(0, np.nan).fillna(method='ffill')
                
                fwd_ret = df['Close'].shift(-h) / df['Close'] - 1
                df[f'Sigma_Return_{h}d'] = fwd_ret / expected_vol
                
                # Regime
                full_base = df[f'Sigma_Return_{h}d'].expanding(min_periods=252).mean().shift(h)
                recent_base = df[f'Sigma_Return_{h}d'].rolling(500).mean().shift(h)
                
                valid_bull = (recent_base >= full_base).astype(int)
                valid_bear = (recent_base <= full_base).astype(int)
                
                # MODEL B: REGIME (EQUAL WEIGHT)
                # We multiply the *Total* scores by the regime validity of the TICKER
                # (Simplification: If Ticker is in Downtrend regime, ignore ALL bull signals for it)
                df['Score_Regime_Bull'] = bull_matrix.sum(axis=1) * valid_bull
                df['Score_Regime_Bear'] = bear_matrix.sum(axis=1) * valid_bear
                df[f'Score_Regime_{h}d'] = df['Score_Regime_Bull'] - df['Score_Regime_Bear']
                
                # MODEL C: OPTIMIZED (REGIME + WEIGHTED)
                final_bull_w = bull_weights.multiply(valid_bull, axis=0)
                final_bear_w = bear_weights.multiply(valid_bear, axis=0)
                
                s_bull = (bull_matrix * final_bull_w).sum(axis=1)
                s_bear = (bear_matrix * final_bear_w).sum(axis=1)
                df[f'Score_Opt_{h}d'] = s_bull - s_bear
                
                subset = df[[
                    'Score_Naive', 
                    f'Score_Regime_{h}d',
                    f'Score_Opt_{h}d', 
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
    Simulates 3 Strategies with Top 20 Long / Top 5 Short.
    Separates PnL into Long and Short components.
    """
    # Pivots
    s_naive = panel_df.pivot(columns='Ticker', values='Score_Naive')
    s_reg   = panel_df.pivot(columns='Ticker', values=f'Score_Regime_{horizon}d')
    s_opt   = panel_df.pivot(columns='Ticker', values=f'Score_Opt_{horizon}d')
    returns = panel_df.pivot(columns='Ticker', values=f'Sigma_Return_{horizon}d')
    
    common = s_naive.index.intersection(returns.index)
    s_naive, s_reg, s_opt, returns = s_naive.loc[common], s_reg.loc[common], s_opt.loc[common], returns.loc[common]
    
    valid_dates = s_naive.index[s_naive.index.year >= 2018] # Start 2018 to allow regime calc
    
    results = {
        "Date": [],
        "Naive_L": [], "Naive_S": [], "Naive_Net": [],
        "Regime_L": [], "Regime_S": [], "Regime_Net": [],
        "Opt_L": [], "Opt_S": [], "Opt_Net": []
    }
    
    for d in valid_dates:
        r_ret = returns.loc[d]
        
        # Helper to calc L/S PnL
        def calc_day(scores):
            # Sort descending
            ranked = scores.sort_values(ascending=False)
            
            # Top Longs
            longs = ranked.head(LONG_COUNT).index
            # Bottom Shorts
            shorts = ranked.tail(SHORT_COUNT).index
            
            # Check for overlap (rare with 60 tickers, but safe to handle)
            shorts = [s for s in shorts if s not in longs]
            
            l_pnl = r_ret[longs].mean()
            s_pnl = -1 * r_ret[shorts].mean() # Short PnL is inverse of return
            
            if np.isnan(l_pnl): l_pnl = 0
            if np.isnan(s_pnl): s_pnl = 0
            
            return l_pnl, s_pnl, (l_pnl + s_pnl) # Net is sum of both components

        # NAIVE
        nl, ns, nn = calc_day(s_naive.loc[d])
        
        # REGIME
        rl, rs, rn = calc_day(s_reg.loc[d])
        
        # OPT
        ol, os, on = calc_day(s_opt.loc[d])
        
        results["Date"].append(d)
        results["Naive_L"].append(nl); results["Naive_S"].append(ns); results["Naive_Net"].append(nn)
        results["Regime_L"].append(rl); results["Regime_S"].append(rs); results["Regime_Net"].append(rn)
        results["Opt_L"].append(ol); results["Opt_S"].append(os); results["Opt_Net"].append(on)

    return pd.DataFrame(results).set_index("Date")

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Portfolio Breakdown")
    st.title("🧪 Long/Short Portfolio Lab")
    
    st.markdown(f"""
    **Configuration:**
    * **Universe:** ~60 Liquid Assets (Sectors, Indices, Mega-Caps)
    * **Sizing:** Risk Parity (1 $\sigma$ Risk Units)
    * **Structure:** Long Top **{LONG_COUNT}** / Short Bottom **{SHORT_COUNT}**
    """)
    
    if st.button("Run Portfolio Simulation", type="primary"):
        
        with st.spinner(f"Processing {len(BACKTEST_TICKERS)} tickers..."):
            master_panels = get_backtest_data(BACKTEST_TICKERS)
        
        if not master_panels:
            st.error("Data error.")
            return
            
        tab5, tab10, tab21 = st.tabs(["5-Day Horizon", "10-Day Horizon", "21-Day Horizon"])
        
        for horizon, tab in zip(TEST_HORIZONS, [tab5, tab10, tab21]):
            with tab:
                if horizon not in master_panels:
                    st.warning("No data.")
                    continue
                    
                with st.spinner(f"Simulating {horizon}d History..."):
                    res = run_full_simulation(master_panels[horizon], horizon)
                
                # Cumulative calcs
                for col in res.columns:
                    res[f"Cum_{col}"] = res[col].cumsum()
                
                # 1. TOTAL PERFORMANCE CHART
                st.subheader("🏆 Net Performance (Long + Short)")
                
                final_naive = res['Cum_Naive_Net'].iloc[-1]
                final_reg = res['Cum_Regime_Net'].iloc[-1]
                final_opt = res['Cum_Opt_Net'].iloc[-1]
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Naive Net", f"{final_naive:.2f} R")
                c2.metric("Regime (EqW) Net", f"{final_reg:.2f} R", delta=f"{final_reg - final_naive:.2f}")
                c3.metric("Weighted Net", f"{final_opt:.2f} R", delta=f"{final_opt - final_reg:.2f}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res.index, y=res['Cum_Naive_Net'], name="Naive", line=dict(color='blue', width=1)))
                fig.add_trace(go.Scatter(x=res.index, y=res['Cum_Regime_Net'], name="Regime (EqW)", line=dict(color='orange', width=2)))
                fig.add_trace(go.Scatter(x=res.index, y=res['Cum_Opt_Net'], name="Weighted", line=dict(color='purple', width=2)))
                fig.update_layout(height=450, title=f"Cumulative Net PnL ({horizon}d)", yaxis_title="Sigma (R)")
                st.plotly_chart(fig, use_container_width=True)
                
                # 2. DECOMPOSITION
                st.markdown("---")
                st.subheader("🔎 Performance Decomposition")
                
                col_L, col_S = st.columns(2)
                
                with col_L:
                    st.markdown("**Long Side Performance**")
                    fig_l = go.Figure()
                    fig_l.add_trace(go.Scatter(x=res.index, y=res['Cum_Naive_L'], name="Naive Long", line=dict(color='lightblue')))
                    fig_l.add_trace(go.Scatter(x=res.index, y=res['Cum_Regime_L'], name="Regime Long", line=dict(color='orange')))
                    fig_l.add_trace(go.Scatter(x=res.index, y=res['Cum_Opt_L'], name="Weighted Long", line=dict(color='violet')))
                    fig_l.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig_l, use_container_width=True)
                    
                with col_S:
                    st.markdown("**Short Side Performance**")
                    fig_s = go.Figure()
                    fig_s.add_trace(go.Scatter(x=res.index, y=res['Cum_Naive_S'], name="Naive Short", line=dict(color='lightblue')))
                    fig_s.add_trace(go.Scatter(x=res.index, y=res['Cum_Regime_S'], name="Regime Short", line=dict(color='orange')))
                    fig_s.add_trace(go.Scatter(x=res.index, y=res['Cum_Opt_S'], name="Weighted Short", line=dict(color='violet')))
                    fig_s.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig_s, use_container_width=True)

if __name__ == "__main__":
    main()
