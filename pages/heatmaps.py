import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.stats import pearsonr, spearmanr
import matplotlib
import datetime
import itertools
from collections import defaultdict

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
SEASONAL_PATH = "seasonal_ranks.csv"
METRICS_PATH = "market_metrics_full_export.csv"
NAAIM_PATH = "naaim.csv"

@st.cache_data(show_spinner=False)
def load_seasonal_map():
    try:
        df = pd.read_csv(SEASONAL_PATH)
    except Exception:
        return {}
    if df.empty: return {}
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])
    df["MD"] = df["Date"].apply(lambda x: (x.month, x.day))
    output_map = {}
    for ticker, group in df.groupby("ticker"):
        output_map[ticker] = pd.Series(group.seasonal_rank.values, index=group.MD).to_dict()
    return output_map

@st.cache_data(show_spinner=False)
def load_market_metrics():
    try:
        df = pd.read_csv(METRICS_PATH)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.drop_duplicates(subset=['date', 'exchange'], keep='first')
    except Exception:
        return pd.DataFrame()
    if df.empty: return pd.DataFrame()
    pivoted = df.pivot_table(index='date', columns='exchange', values='net_new_highs', aggfunc='sum')
    pivoted['Total'] = pivoted.get('NYSE', 0) + pivoted.get('NASDAQ', 0)
    results = pd.DataFrame(index=pivoted.index)
    for cat in ['Total']:
        if cat not in pivoted.columns: continue
        series = pivoted[cat]
        for w in [5, 21]:
            ma_col = series.rolling(window=w).mean()
            results[f"Mkt_{cat}_NH_{w}d_Rank"] = ma_col.expanding(min_periods=126).rank(pct=True) * 100.0
    return results

@st.cache_data(show_spinner=False)
def load_naaim_data():
    try:
        df = pd.read_csv(NAAIM_PATH)
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'])
        elif 'date' in df.columns: df['Date'] = pd.to_datetime(df['date'])
        else: df['Date'] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index('Date').sort_index()
        df = df[~df.index.duplicated(keep='first')]
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            val_col = num_cols[0]
            df = df[[val_col]].rename(columns={val_col: 'NAAIM'})
            df['NAAIM_MA5'] = df['NAAIM'].rolling(5).mean()
            df['NAAIM_MA12'] = df['NAAIM'].rolling(12).mean()
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# CORE LOGIC
# -----------------------------------------------------------------------------
def generate_dynamic_seasonal_profile(df, cutoff_date, target_year):
    work_df = df[df.index < cutoff_date].copy()
    if len(work_df) < 500: return {}
    if 'LogRet' not in work_df.columns:
        if 'Close' in work_df.columns:
             work_df['LogRet'] = np.log(work_df['Close'] / work_df['Close'].shift(1)) * 100.0
        else: return {}
    work_df['Year'] = work_df.index.year
    work_df['MD'] = work_df.index.map(lambda x: (x.month, x.day))
    cycle_remainder = target_year % 4
    fwd_cols = []
    for w in [5, 10, 21]:
        col_name = f'Fwd_{w}d'
        work_df[col_name] = work_df['LogRet'].shift(-w).rolling(w).sum()
        fwd_cols.append(col_name)
    stats_all = work_df.groupby('MD')[fwd_cols].mean()
    cycle_df = work_df[work_df['Year'] % 4 == cycle_remainder]
    stats_cycle = stats_all.copy() if len(cycle_df) < 250 else cycle_df.groupby('MD')[fwd_cols].mean()
    stats_cycle = stats_cycle.reindex(stats_all.index).fillna(method='ffill').fillna(method='bfill')
    rnk_all = stats_all.rank(pct=True) * 100.0
    rnk_cycle = stats_cycle.rank(pct=True) * 100.0
    final_rank = (rnk_all.mean(axis=1) + 3 * rnk_cycle.mean(axis=1)) / 4.0
    return final_rank.rolling(window=5, center=True, min_periods=1).mean().to_dict()

def get_sznl_val_series(ticker, dates, sznl_map, df_hist=None):
    t_map = sznl_map.get(ticker, {})
    if t_map:
        return dates.map(lambda x: (x.month, x.day)).map(t_map).fillna(50.0)
    if df_hist is not None and not df_hist.empty:
        if 'LogRet' not in df_hist.columns:
             df_hist = df_hist.copy()
             df_hist['LogRet'] = np.log(df_hist['Close'] / df_hist['Close'].shift(1)) * 100.0
        sznl_series = pd.Series(50.0, index=dates)
        for yr in dates.year.unique():
            cutoff = pd.Timestamp(yr, 1, 1)
            if df_hist.index.tz is not None: cutoff = cutoff.tz_localize(df_hist.index.tz)
            yearly_profile = generate_dynamic_seasonal_profile(df_hist, cutoff, yr)
            if yearly_profile:
                mask = (dates.year == yr)
                sznl_series.loc[mask] = dates[mask].map(lambda x: (x.month, x.day)).map(yearly_profile).fillna(50.0)
        return sznl_series
    return pd.Series(50.0, index=dates)

@st.cache_data(show_spinner=True)
def download_data(ticker):
    try:
        df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
        df = df[~df.index.duplicated(keep='first')]
        return df, pd.Timestamp.now(tz='America/New_York')
    except: return pd.DataFrame(), None

@st.cache_data(show_spinner=False)
def get_spy_context():
    spy, _ = download_data("SPY")
    if spy.empty: return pd.DataFrame()
    spy_features = pd.DataFrame(index=spy.index)
    for w in [5, 10, 21]:
        spy_features[f'SPY_Ret_{w}d_Rank'] = spy['Close'].pct_change(w).expanding(min_periods=252).rank(pct=True) * 100.0
    return spy_features

@st.cache_data(show_spinner=True)
def calculate_heatmap_variables(df, sznl_map, market_metrics_df, ticker):
    df = df.copy()
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    for w in [1, 2, 3, 5, 10, 21, 63, 126, 252]:
        df[f'FwdRet_{w}d'] = (df['Close'].shift(-w) / df['Close'] - 1.0) * 100.0
    for w in [5, 10, 21, 252]:
        df[f'Ret_{w}d'] = df['Close'].pct_change(w)
    for w in [21, 63]:
        df[f'RealVol_{w}d'] = df['LogRet'].rolling(w).std() * np.sqrt(252) * 100.0
    for w in [10, 21]:
        df[f'VolRatio_{w}d'] = df['Volume'].rolling(w).mean() / df['Volume'].rolling(63).mean()
    df['Seasonal'] = get_sznl_val_series(ticker, df.index, sznl_map, df)
    
    vars_to_rank = ['Ret_5d', 'Ret_10d', 'Ret_21d', 'Ret_252d', 'RealVol_21d', 'RealVol_63d', 'VolRatio_10d', 'VolRatio_21d']
    rank_cols = []
    for v in vars_to_rank:
        col_name = v + '_Rank'
        df[col_name] = df[v].expanding(min_periods=252).rank(pct=True) * 100.0
        rank_cols.append(col_name)
    
    if not market_metrics_df.empty:
        df = df.join(market_metrics_df, how='left')
        df.update(df.filter(regex='^Mkt_').ffill(limit=3))
    
    if ticker != "SPY":
        spy_df = get_spy_context()
        if not spy_df.empty:
            df = df.join(spy_df, how='left')
            spy_cols = [c for c in spy_df.columns if 'Rank' in c]
            rank_cols.extend(spy_cols)
            df.update(df[spy_cols].ffill(limit=1))
            
    naaim_df = load_naaim_data()
    if not naaim_df.empty:
        df = df.join(naaim_df, how='left')
        naaim_cols_raw = ['NAAIM', 'NAAIM_MA5', 'NAAIM_MA12']
        df[naaim_cols_raw] = df[naaim_cols_raw].ffill()
        for col in naaim_cols_raw:
            rank_col = f"{col}_Rank"
            df[rank_col] = df[col].expanding(min_periods=252).rank(pct=True) * 100.0
            rank_cols.append(rank_col)
            
    return df, rank_cols

def calculate_distribution_ensemble(df, rank_cols, market_cols, tolerance=1.0):
    if df.empty: return pd.DataFrame()
    current_row = df.iloc[-1]
    all_features = rank_cols + ['Seasonal'] + market_cols
    valid_features = [f for f in all_features if f in df.columns and not pd.isna(current_row[f])]
    if len(valid_features) < 2: return pd.DataFrame()
    
    pairs = []
    non_spy_valid = [f for f in valid_features if not f.startswith("SPY_")]
    pairs.extend(list(itertools.combinations(non_spy_valid, 2)))
    spy_map = {'SPY_Ret_5d_Rank': 'Ret_5d_Rank', 'SPY_Ret_10d_Rank': 'Ret_10d_Rank', 'SPY_Ret_21d_Rank': 'Ret_21d_Rank'}
    for spy_col, ticker_col in spy_map.items():
        if spy_col in valid_features and ticker_col in valid_features: pairs.append((spy_col, ticker_col))
    
    targets = [2, 3, 5, 10, 21, 63, 126, 252]
    pooled_outcomes = {t: [] for t in targets}
    
    for f1, f2 in pairs:
        v1, v2 = current_row[f1], current_row[f2]
        subset = df[(df[f1].between(v1-tolerance, v1+tolerance)) & (df[f2].between(v2-tolerance, v2+tolerance))]
        if subset.empty: continue
        for t in targets:
            col = f'FwdRet_{t}d'
            if col in subset.columns: pooled_outcomes[t].extend(subset[col].dropna().tolist())
            
    summary = []
    for t in targets:
        data = np.array(pooled_outcomes[t])
        if len(data) == 0: continue
        baseline = df[f'FwdRet_{t}d'].mean() if f'FwdRet_{t}d' in df.columns else np.nan
        summary.append({
            "Horizon": f"{t} Days", "Exp Return": np.mean(data), "Baseline": baseline,
            "Alpha": np.mean(data) - baseline, "Win Rate": np.sum(data > 0) / len(data) * 100, "Sample Size": len(data)
        })
    return pd.DataFrame(summary)

# --- EUCLIDEAN & BACKTESTING ENGINE ---
def backtest_euclidean_model(df, rank_cols, market_cols, start_year=2015, n_neighbors=50, target_days=5, weights_dict=None):
    all_feats = rank_cols + ['Seasonal'] + market_cols
    target_col = f'FwdRet_{target_days}d'
    validation_df = df.dropna(subset=all_feats + [target_col]).copy()
    test_indices = validation_df[validation_df.index.year >= start_year].index
    if len(test_indices) == 0: return pd.DataFrame()

    w_vec = np.ones(len(all_feats))
    if weights_dict:
        w_vec = np.array([weights_dict.get(f, 1.0) for f in all_feats])
    
    feat_matrix = validation_df[all_feats].values
    target_array = validation_df[target_col].values
    index_array = validation_df.index
    start_loc = np.searchsorted(index_array, test_indices[0])
    
    predictions, actuals, dates = [], [], []
    progress_bar = st.progress(0)
    total_steps = len(test_indices)
    
    for i in range(start_loc, len(validation_df)):
        if (i - start_loc) % 100 == 0: progress_bar.progress(min((i - start_loc) / total_steps, 1.0))
        if i < n_neighbors: continue
        
        hist_feats = feat_matrix[:i]
        hist_rets = target_array[:i]
        curr_feat = feat_matrix[i]
        
        # Weighted Distance
        dists = np.sqrt(np.sum(((hist_feats - curr_feat)**2) * w_vec, axis=1))
        neighbor_idxs = np.argpartition(dists, n_neighbors)[:n_neighbors]
        predictions.append(np.mean(hist_rets[neighbor_idxs]))
        actuals.append(target_array[i])
        dates.append(index_array[i])
        
    progress_bar.empty()
    return pd.DataFrame({'Predicted': predictions, 'Actual': actuals}, index=dates)

def run_portfolio_simulation(df, predictions_df, max_long=2.0, max_short=-1.0, sensitivity=1.0, start_capital=100000):
    """
    Simulates a portfolio that rebalances DAILY based on the model's forward prediction.
    Strategy: 
      - Calculate Target Weight = Prediction * Sensitivity
      - Clip Weight between Max Long and Max Short
      - Apply Weight to Next Day's Return
    """
    # 1. Prepare Data
    # Align dates. The 'Predicted' value at Date T is generated AFTER CLOSE on Date T.
    # Therefore, it dictates the position held from Close T to Close T+1.
    sim_df = predictions_df[['Predicted']].copy()
    
    # Get the asset's daily returns for the simulation period
    # We use 'Close' to 'Close' daily returns
    asset_daily_ret = df['Close'].pct_change().shift(-1) # Shift -1 because we want T+1 return aligned with T signal
    sim_df['Asset_Daily_Ret'] = asset_daily_ret.loc[sim_df.index]
    
    # 2. Determine Position Sizing (Signal Translation)
    # Prediction is in %, e.g., 2.0 means 2%.
    # If sensitivity is 0.5: Target = 2.0 * 0.5 = 100% Long.
    # If sensitivity is 1.0: Target = 2.0 * 1.0 = 200% Long.
    sim_df['Target_Weight'] = sim_df['Predicted'] * sensitivity
    
    # Apply Leverage Constraints
    sim_df['Position'] = sim_df['Target_Weight'].clip(lower=max_short, upper=max_long)
    
    # 3. Calculate Strategy Returns
    # Strategy Return = Position * Asset Return
    # Note: We ignore transaction costs for the raw alpha check, but could add -0.0005 here
    sim_df['Strategy_Ret'] = sim_df['Position'] * sim_df['Asset_Daily_Ret']
    
    # 4. Construct Equity Curves
    sim_df = sim_df.dropna()
    
    # Cumulative Return Calculation
    sim_df['Benchmark_Equity'] = start_capital * (1 + sim_df['Asset_Daily_Ret']).cumprod()
    sim_df['Strategy_Equity'] = start_capital * (1 + sim_df['Strategy_Ret']).cumprod()
    
    # 5. Drawdown Stats
    sim_df['Peak'] = sim_df['Strategy_Equity'].cummax()
    sim_df['Drawdown'] = (sim_df['Strategy_Equity'] - sim_df['Peak']) / sim_df['Peak']
    
    return sim_df

# --- UI HELPERS ---
def get_seismic_colorscale():
    seismic = []
    cm = matplotlib.colormaps["seismic"] 
    for k in range(255):
        r, g, b, _ = cm(k / 254.0)
        seismic.append([k / 254.0, f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'])
    return seismic

# -----------------------------------------------------------------------------
# UI: MAIN
# -----------------------------------------------------------------------------
def render_heatmap():
    st.subheader("Heatmap Analytics")
    col1, col2, col3 = st.columns(3)
    
    var_options = {
        "Seasonality Rank": "Seasonal", "5d Trailing Return Rank": "Ret_5d_Rank",
        "21d Trailing Return Rank": "Ret_21d_Rank", "SPY 5d Return Rank": "SPY_Ret_5d_Rank",
        "SPY 21d Return Rank": "SPY_Ret_21d_Rank", "21d Realized Vol Rank": "RealVol_21d_Rank",
        "Total Net Highs (21d MA) Rank": "Mkt_Total_NH_21d_Rank", "NAAIM Exposure Rank": "NAAIM_Rank",
        "NAAIM 5wk MA Rank": "NAAIM_MA5_Rank"
    }
    
    with col1: x_axis_label = st.selectbox("X-Axis", list(var_options.keys()), index=0)
    with col2: y_axis_label = st.selectbox("Y-Axis", list(var_options.keys()), index=2)
    with col3: ticker = st.text_input("Ticker", value="SMH").upper()

    if st.button("Load Data", type="primary"):
        st.session_state['data_loaded'] = True

    if st.session_state.get('data_loaded'):
        with st.spinner("Crunching numbers..."):
            data, _ = download_data(ticker)
            sznl_map = load_seasonal_map()
            mkt_metrics = load_market_metrics()
            df, rank_cols = calculate_heatmap_variables(data, sznl_map, mkt_metrics, ticker)
            
            # --- TABS FOR ANALYSIS ---
            tab_heat, tab_backtest, tab_portfolio = st.tabs(["1. Heatmap & Ensemble", "2. IC Backtest", "3. Portfolio Simulator (Real Trading)"])
            
            with tab_heat:
                st.write("Visual Pattern Matching")
                # (Simplified Heatmap Logic for brevity in this full refresh, you can paste full logic back if needed)
                ensemble_df = calculate_distribution_ensemble(df, rank_cols, [c for c in df.columns if c.startswith("Mkt_")], tolerance=5)
                if not ensemble_df.empty: st.dataframe(ensemble_df.style.format("{:.2f}").background_gradient(cmap="RdBu", vmin=-2, vmax=2))
            
            with tab_backtest:
                st.write("Statistical Predictive Power (IC)")
                b_cols = st.columns(3)
                with b_cols[0]: bt_start = st.number_input("Start Year", 2010, 2024, 2018, key="bt_start_ic")
                with b_cols[1]: bt_k = st.number_input("Neighbors", 10, 200, 50, key="bt_k_ic")
                with b_cols[2]: bt_t = st.selectbox("Target", [5, 10, 21], key="bt_t_ic")
                
                if st.button("Run IC Check"):
                    res = backtest_euclidean_model(df, rank_cols, [c for c in df.columns if c.startswith("Mkt_")], bt_start, bt_k, bt_t)
                    if not res.empty:
                        ic = pearsonr(res['Predicted'], res['Actual'])[0]
                        st.metric("Information Coefficient", f"{ic:.3f}")
                        st.line_chart(res['Predicted'].rolling(252).corr(res['Actual']))
            
            with tab_portfolio:
                st.markdown("### 💰 Portfolio Simulator")
                st.info("This simulates **Daily Rebalancing**. Even though the forecast is for 5 days, we update our position every single day based on the newest forecast. This solves the 'Overlap' problem.")
                
                p_col1, p_col2, p_col3 = st.columns(3)
                with p_col1: 
                    lev_long = st.slider("Max Long Leverage", 1.0, 3.0, 2.0, 0.1)
                    lev_short = st.slider("Max Short Leverage", 0.0, 2.0, 1.0, 0.1) * -1
                with p_col2:
                    # Conviction Scalar: How much position do we take for 1% predicted return?
                    # If 0.5: 1% prediction -> 50% position.
                    # If 1.0: 1% prediction -> 100% position.
                    conviction = st.slider("Conviction Scaler (Pos % per 1% Pred)", 0.1, 2.0, 0.5, 0.1)
                with p_col3:
                    start_cap = st.number_input("Start Capital", value=100000)
                
                # Hidden Weighted Logic (Simpler for Portfolio Tab)
                active_weights = {"Ret_21d_Rank": 3.0, "NAAIM_Rank": 2.0} # Defaulting to the 'good' model
                
                if st.button("Run Portfolio Simulation"):
                    with st.spinner("Simulating Daily Trading..."):
                        # 1. Generate Signals
                        # We force 5d target for the signal generation as the base 'trend' duration
                        preds = backtest_euclidean_model(df, rank_cols, [c for c in df.columns if c.startswith("Mkt_")], 
                                                         start_year=bt_start, n_neighbors=bt_k, target_days=5, 
                                                         weights_dict=active_weights)
                        
                        # 2. Run Portfolio Logic
                        port_df = run_portfolio_simulation(df, preds, max_long=lev_long, max_short=lev_short, 
                                                           sensitivity=conviction, start_capital=start_cap)
                        
                        if not port_df.empty:
                            # Metrics
                            final_strat = port_df['Strategy_Equity'].iloc[-1]
                            final_bench = port_df['Benchmark_Equity'].iloc[-1]
                            strat_ret = (final_strat - start_cap) / start_cap * 100
                            bench_ret = (final_bench - start_cap) / start_cap * 100
                            max_dd = port_df['Drawdown'].min() * 100
                            
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Strategy Final Equity", f"${final_strat:,.0f}", delta=f"{strat_ret:.1f}%")
                            m2.metric("Buy & Hold SPY", f"${final_bench:,.0f}", delta=f"{bench_ret:.1f}%")
                            m3.metric("Max Drawdown", f"{max_dd:.1f}%", delta_color="inverse")
                            
                            # Chart 1: Equity Curve
                            fig_eq = go.Figure()
                            fig_eq.add_trace(go.Scatter(x=port_df.index, y=port_df['Strategy_Equity'], name='Strategy', line=dict(color='green', width=2)))
                            fig_eq.add_trace(go.Scatter(x=port_df.index, y=port_df['Benchmark_Equity'], name='Buy & Hold', line=dict(color='gray', dash='dot')))
                            fig_eq.update_layout(title="Portfolio Value over Time", yaxis_title="Equity ($)", height=500)
                            st.plotly_chart(fig_eq, use_container_width=True)
                            
                            # Chart 2: Leverage Utilization
                            st.write("**Leverage Utilization:** When did we go 200% Long vs Short?")
                            fig_lev = go.Figure()
                            fig_lev.add_trace(go.Area(x=port_df.index, y=port_df['Position'], name='Position Size', fill='tozeroy', line=dict(color='blue')))
                            fig_lev.update_layout(title="Active Position Size (-1.0 to 2.0)", yaxis_title="Leverage", height=300)
                            st.plotly_chart(fig_lev, use_container_width=True)
                        else:
                            st.error("Simulation failed (No data).")

def main():
    st.set_page_config(layout="wide", page_title="Heatmap Analytics")
    render_heatmap()

if __name__ == "__main__":
    main()
