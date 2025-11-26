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
# Reverted to original "Primary Components" list
BACKTEST_TICKERS = ["SPY", "QQQ", "IWM", "SLV", "DIA", "AAPL", "AMD", "MSFT", "SMH", "HD", "KO", "UNG", "TLT", "USO", "GLD"]

# 2d, 5d, 10d, 21d Horizons
TEST_HORIZONS = [2, 5, 10, 21]

# Original Sizing
LONG_COUNT = 5
SHORT_COUNT = 1

# -----------------------------------------------------------------------------
# DATA ENGINE (Vectorized)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=True)
def get_backtest_data(ticker_list):
    tickers = list(set([t.strip().upper() for t in ticker_list]))
    horizon_panels = {h: [] for h in TEST_HORIZONS}
    start_date = "2016-01-01"
    
    progress_bar = st.progress(0)
    
    try:
        all_data = yf.download(tickers, start=start_date, progress=False, auto_adjust=True, group_by='ticker')
    except:
        return {}

    for i, t in enumerate(tickers):
        try:
            if isinstance(all_data.columns, pd.MultiIndex):
                if t not in all_data.columns.levels[0]: continue
                df = all_data[t].copy()
            else:
                df = all_data.copy()

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
            
            # NAIVE
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
                
                # REGIME (Equal Weight)
                df['Score_Regime_Bull'] = bull_matrix.sum(axis=1) * valid_bull
                df['Score_Regime_Bear'] = bear_matrix.sum(axis=1) * valid_bear
                df[f'Score_Regime_{h}d'] = df['Score_Regime_Bull'] - df['Score_Regime_Bear']
                
                # OPTIMIZED (Regime + Weighted)
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
# SIMULATION ENGINE (With Attribution)
# -----------------------------------------------------------------------------
def run_full_simulation(panel_df, horizon):
    # Pivots
    s_naive = panel_df.pivot(columns='Ticker', values='Score_Naive')
    s_reg   = panel_df.pivot(columns='Ticker', values=f'Score_Regime_{horizon}d')
    s_opt   = panel_df.pivot(columns='Ticker', values=f'Score_Opt_{horizon}d')
    returns = panel_df.pivot(columns='Ticker', values=f'Sigma_Return_{horizon}d')
    
    common = s_naive.index.intersection(returns.index)
    s_naive, s_reg, s_opt, returns = s_naive.loc[common], s_reg.loc[common], s_opt.loc[common], returns.loc[common]
    
    valid_dates = s_naive.index[s_naive.index.year >= 2018]
    
    results = {
        "Date": [],
        "Naive_Net": [], "Regime_Net": [], "Opt_Net": []
    }
    
    # Attribution Trackers (Dictionary of Tickers)
    attr_regime = {t: 0.0 for t in s_naive.columns}
    attr_opt = {t: 0.0 for t in s_naive.columns}
    
    for d in valid_dates:
        r_ret = returns.loc[d]
        
        # Helper
        def get_pnl_and_pos(scores):
            ranked = scores.sort_values(ascending=False)
            longs = ranked.head(LONG_COUNT).index
            shorts = ranked.tail(SHORT_COUNT).index
            shorts = [s for s in shorts if s not in longs]
            
            # PnL Calculation
            l_mean = r_ret[longs].mean()
            s_mean = -1 * r_ret[shorts].mean()
            
            if np.isnan(l_mean): l_mean = 0
            if np.isnan(s_mean): s_mean = 0
            
            return l_mean + s_mean, longs, shorts

        # NAIVE
        pnl_n, _, _ = get_pnl_and_pos(s_naive.loc[d])
        
        # REGIME (Track Attribution)
        pnl_r, l_r, s_r = get_pnl_and_pos(s_reg.loc[d])
        
        # For attribution, we split the day's PnL equally among the basket
        # Longs get +Ret / Count, Shorts get -Ret / Count
        # Simplified: We just add raw return to the ticker bucket
        for t in l_r: attr_regime[t] += r_ret[t] if not np.isnan(r_ret[t]) else 0
        for t in s_r: attr_regime[t] -= r_ret[t] if not np.isnan(r_ret[t]) else 0
        
        # OPTIMIZED (Track Attribution)
        pnl_o, l_o, s_o = get_pnl_and_pos(s_opt.loc[d])
        
        for t in l_o: attr_opt[t] += r_ret[t] if not np.isnan(r_ret[t]) else 0
        for t in s_o: attr_opt[t] -= r_ret[t] if not np.isnan(r_ret[t]) else 0
        
        results["Date"].append(d)
        results["Naive_Net"].append(pnl_n)
        results["Regime_Net"].append(pnl_r)
        results["Opt_Net"].append(pnl_o)

    df_res = pd.DataFrame(results).set_index("Date")
    return df_res, attr_regime, attr_opt

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Portfolio Breakdown")
    st.title("🧪 Long/Short Portfolio Lab")
    
    st.markdown(f"""
    **Configuration:**
    * **Universe:** 15 Primary Assets ({", ".join(BACKTEST_TICKERS[:5])}...)
    * **Structure:** Long Top **{LONG_COUNT}** / Short Bottom **{SHORT_COUNT}**
    * **Sizing:** Risk Parity (1 $\sigma$ Risk Units)
    """)
    
    if st.button("Run Portfolio Simulation", type="primary"):
        
        with st.spinner(f"Processing {len(BACKTEST_TICKERS)} tickers..."):
            master_panels = get_backtest_data(BACKTEST_TICKERS)
        
        if not master_panels:
            st.error("Data error.")
            return
            
        tab2, tab5, tab10, tab21 = st.tabs(["2-Day", "5-Day", "10-Day", "21-Day"])
        
        for horizon, tab in zip(TEST_HORIZONS, [tab2, tab5, tab10, tab21]):
            with tab:
                if horizon not in master_panels:
                    st.warning("No data.")
                    continue
                    
                with st.spinner(f"Simulating {horizon}d History..."):
                    res, attr_reg, attr_opt = run_full_simulation(master_panels[horizon], horizon)
                
                for col in res.columns:
                    res[f"Cum_{col}"] = res[col].cumsum()
                
                # 1. TOTAL PERFORMANCE
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
                fig.update_layout(height=400, title=f"Cumulative Net PnL ({horizon}d)", yaxis_title="Sigma (R)")
                st.plotly_chart(fig, use_container_width=True)
                
                # 2. TICKER ATTRIBUTION (Bar Chart)
                st.markdown("---")
                st.subheader("📊 Ticker Attribution (Optimized Strategy)")
                
                # Create DF from dict
                attr_df = pd.DataFrame.from_dict(attr_opt, orient='index', columns=['Total_Sigma'])
                attr_df = attr_df.sort_values('Total_Sigma', ascending=False)
                
                # Color bars by Profit/Loss
                colors = ['green' if x > 0 else 'red' for x in attr_df['Total_Sigma']]
                
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=attr_df.index, 
                    y=attr_df['Total_Sigma'],
                    marker_color=colors
                ))
                
                fig_bar.update_layout(
                    title=f"PnL Contribution by Ticker (Weighted Strategy, {horizon}d)",
                    yaxis_title="Total Sigma (R)",
                    xaxis_tickangle=-45,
                    height=400
                )
                st.plotly_chart(fig_bar, use_container_width=True)

if __name__ == "__main__":
    main()
