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
# 15 Primary Assets
BACKTEST_TICKERS = ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "AMD", "MSFT", "GOOG", "AMZN", "META", "NFLX", "TLT", "USO", "GLD"]

TEST_HORIZONS = [2, 5, 10, 21]
LONG_COUNT = 5
SHORT_COUNT = 3

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
# SIMULATION ENGINE (Deep Attribution)
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
    
    # --- ATTRIBUTION STRUCTURE ---
    # Structure: { 'Naive': {'SPY': {'Long': 0.0, 'Short': 0.0}, ...}, 'Regime': ... }
    tickers = s_naive.columns
    attribution = {
        'Naive':  {t: {'Long': 0.0, 'Short': 0.0} for t in tickers},
        'Regime': {t: {'Long': 0.0, 'Short': 0.0} for t in tickers},
        'Opt':    {t: {'Long': 0.0, 'Short': 0.0} for t in tickers}
    }
    
    for d in valid_dates:
        r_ret = returns.loc[d]
        
        # --- CALCULATION FUNCTION ---
        def process_model(scores, model_name):
            ranked = scores.sort_values(ascending=False)
            longs = ranked.head(LONG_COUNT).index
            shorts = ranked.tail(SHORT_COUNT).index
            shorts = [s for s in shorts if s not in longs]
            
            # 1. PnL Calculation (Mean Return)
            l_mean = r_ret[longs].mean() if len(longs) > 0 else 0
            s_mean = -1 * r_ret[shorts].mean() if len(shorts) > 0 else 0
            net_pnl = l_mean + s_mean
            
            # 2. Attribution Calculation (Normalized by basket size)
            # This ensures Sum(Attribution) == PnL
            if len(longs) > 0:
                weight_l = 1.0 / len(longs)
                for t in longs:
                    if not np.isnan(r_ret[t]):
                        attribution[model_name][t]['Long'] += (r_ret[t] * weight_l)
            
            if len(shorts) > 0:
                weight_s = 1.0 / len(shorts)
                for t in shorts:
                    if not np.isnan(r_ret[t]):
                        # Short PnL is (-1 * Ret)
                        attribution[model_name][t]['Short'] += (-1 * r_ret[t] * weight_s)
            
            return net_pnl

        # RUN MODELS
        results["Naive_Net"].append(process_model(s_naive.loc[d], 'Naive'))
        results["Regime_Net"].append(process_model(s_reg.loc[d], 'Regime'))
        results["Opt_Net"].append(process_model(s_opt.loc[d], 'Opt'))
        
        results["Date"].append(d)

    df_res = pd.DataFrame(results).set_index("Date")
    return df_res, attribution

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Backtest Lab")
    st.title("🧪 Long/Short Portfolio Lab")
    
    st.markdown(f"""
    **Configuration:**
    * **Universe:** 15 Primary Assets
    * **Structure:** Long Top **{LONG_COUNT}** / Short Bottom **{SHORT_COUNT}**
    * **Sizing:** Risk Parity (1 $\sigma$ Risk Units)
    """)
    
    if st.button("Run Simulation", type="primary"):
        
        with st.spinner(f"Vectorizing Data..."):
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
                    
                with st.spinner(f"Simulating {horizon}d..."):
                    # Run Sim and Get Complex Attribution Dict
                    res, attr_dict = run_full_simulation(master_panels[horizon], horizon)
                
                for col in res.columns:
                    res[f"Cum_{col}"] = res[col].cumsum()
                
                # 1. TOTAL PERFORMANCE CHART
                st.subheader("🏆 Cumulative Net PnL")
                
                final_naive = res['Cum_Naive_Net'].iloc[-1]
                final_reg = res['Cum_Regime_Net'].iloc[-1]
                final_opt = res['Cum_Opt_Net'].iloc[-1]
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Naive", f"{final_naive:.2f} R")
                c2.metric("Regime (EqW)", f"{final_reg:.2f} R", delta=f"{final_reg - final_naive:.2f}")
                c3.metric("Weighted", f"{final_opt:.2f} R", delta=f"{final_opt - final_reg:.2f}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res.index, y=res['Cum_Naive_Net'], name="Naive", line=dict(color='blue', width=1)))
                fig.add_trace(go.Scatter(x=res.index, y=res['Cum_Regime_Net'], name="Regime (EqW)", line=dict(color='orange', width=2)))
                fig.add_trace(go.Scatter(x=res.index, y=res['Cum_Opt_Net'], name="Weighted", line=dict(color='purple', width=2)))
                fig.update_layout(height=450, title=f"Net Performance ({horizon}d)", yaxis_title="Sigma (R)")
                st.plotly_chart(fig, use_container_width=True)
                
                # 2. SEGMENTED ATTRIBUTION
                st.markdown("---")
                st.subheader("📊 Segmented Attribution (Long vs Short)")
                
                # Control to switch attribution view
                attr_model = st.selectbox(f"Select Model for {horizon}d Breakdown:", ["Naive", "Regime", "Opt"], key=f"sel_{horizon}")
                
                # Process Data for Plotting
                # We need a DF with Index=Ticker, Col=Long_PnL, Col=Short_PnL
                
                model_data = attr_dict[attr_model]
                attr_rows = []
                for t, vals in model_data.items():
                    attr_rows.append({
                        "Ticker": t,
                        "Long_PnL": vals['Long'],
                        "Short_PnL": vals['Short'],
                        "Total_PnL": vals['Long'] + vals['Short']
                    })
                
                df_attr = pd.DataFrame(attr_rows).sort_values("Total_PnL", ascending=False)
                
                # Stacked Bar Chart
                fig_bar = go.Figure()
                
                fig_bar.add_trace(go.Bar(
                    x=df_attr['Ticker'], 
                    y=df_attr['Long_PnL'], 
                    name='Long PnL',
                    marker_color='green'
                ))
                
                fig_bar.add_trace(go.Bar(
                    x=df_attr['Ticker'], 
                    y=df_attr['Short_PnL'], 
                    name='Short PnL',
                    marker_color='red'
                ))
                
                # Add Total Scatter point
                fig_bar.add_trace(go.Scatter(
                    x=df_attr['Ticker'],
                    y=df_attr['Total_PnL'],
                    mode='markers',
                    marker=dict(symbol='diamond', size=10, color='black'),
                    name='Net Total'
                ))

                fig_bar.update_layout(
                    title=f"PnL Drivers: {attr_model} Model ({horizon}d)",
                    barmode='relative', # Allows stacking positives and negatives correctly
                    yaxis_title="Total Sigma (R)",
                    height=500
                )
                st.plotly_chart(fig_bar, use_container_width=True)

if __name__ == "__main__":
    main()
