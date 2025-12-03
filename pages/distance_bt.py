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

def run_portfolio_simulation(df, predictions_df, max_long=2.0, max_short=-1.0, 
                             sensitivity=1.0, start_capital=100000, 
                             slippage_bps=0.0, rebalance_weekly=False):
    """
    Simulates portfolio with friction and optional weekly rebalancing.
    """
    sim_df = predictions_df[['Predicted']].copy()
    
    if rebalance_weekly:
        is_thursday = sim_df.index.day_name() == 'Thursday'
        sim_df.loc[~is_thursday, 'Predicted'] = np.nan
        sim_df['Predicted'] = sim_df['Predicted'].ffill()
    
    asset_daily_ret = df['Close'].pct_change().shift(-1)
    sim_df['Asset_Daily_Ret'] = asset_daily_ret.loc[sim_df.index]
    
    sim_df['Target_Weight'] = sim_df['Predicted'] * sensitivity
    sim_df['Position'] = sim_df['Target_Weight'].clip(lower=max_short, upper=max_long)
    
    pos_change = sim_df['Position'].diff().abs().fillna(0)
    cost = pos_change * (slippage_bps / 10000.0)
    
    sim_df['Strategy_Ret_Gross'] = sim_df['Position'] * sim_df['Asset_Daily_Ret']
    sim_df['Strategy_Ret_Net'] = sim_df['Strategy_Ret_Gross'] - cost
    
    sim_df = sim_df.dropna()
    sim_df['Benchmark_Equity'] = start_capital * (1 + sim_df['Asset_Daily_Ret']).cumprod()
    sim_df['Strategy_Equity'] = start_capital * (1 + sim_df['Strategy_Ret_Net']).cumprod()
    
    sim_df['Peak'] = sim_df['Strategy_Equity'].cummax()
    sim_df['Drawdown'] = (sim_df['Strategy_Equity'] - sim_df['Peak']) / sim_df['Peak']
    
    return sim_df

def calc_metrics(series):
    """Calculates CAGR, Sharpe, Sortino, Vol"""
    if series.empty: return 0, 0, 0, 0
    mean_ret = series.mean()
    std_ret = series.std()
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0
    downside = series[series < 0].std()
    sortino = (mean_ret / downside * np.sqrt(252)) if downside > 0 else 0
    vol = std_ret * np.sqrt(252) * 100
    total_ret = (1 + series).prod()
    days = len(series)
    years = days / 252 
    cagr = (total_ret**(1/years) - 1) * 100 if years > 0 else 0
    return cagr, sharpe, sortino, vol

def display_net_zero_check(results_df, model_name="Model"):
    if results_df.empty: return
    st.markdown("---")
    st.subheader(f"🧹 'Net Zero' Reality Check ({model_name})")
    drift = results_df['Actual'].rolling(window=252).mean()
    excess_pred = results_df['Predicted'] - drift
    excess_actual = results_df['Actual'] - drift
    valid_check = pd.DataFrame({'Excess_Pred': excess_pred, 'Excess_Actual': excess_actual, 'Predicted': results_df['Predicted'], 'Actual': results_df['Actual']}).dropna()
    if not valid_check.empty:
        ic_pearson, _ = pearsonr(valid_check['Predicted'], valid_check['Actual'])
        alpha_ic, _ = pearsonr(valid_check['Excess_Pred'], valid_check['Excess_Actual'])
        c1, c2, c3 = st.columns(3)
        c1.metric("Original IC", f"{ic_pearson:.3f}")
        c2.metric("Alpha IC", f"{alpha_ic:.3f}", delta_color="inverse" if alpha_ic < 0.02 else "normal")
        c3.metric("Dependency on Trend", f"{((ic_pearson - alpha_ic)/ic_pearson*100):.1f}%")

# --- UI HELPERS ---
def get_seismic_colorscale():
    seismic = []
    cm = matplotlib.colormaps["seismic"] 
    for k in range(255):
        r, g, b, _ = cm(k / 254.0)
        seismic.append([k / 254.0, f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'])
    return seismic

# -----------------------------------------------------------------------------
# FEATURE INSPECTION TOOLS
# -----------------------------------------------------------------------------
def calculate_feature_correlations(df, feature_cols, target_col='FwdRet_5d'):
    target_corrs = []
    data = df.dropna(subset=feature_cols + [target_col])
    for f in feature_cols:
        corr, _ = spearmanr(data[f], data[target_col])
        target_corrs.append({'Feature': f, 'Target Correlation': corr})
    target_df = pd.DataFrame(target_corrs).sort_values(by='Target Correlation', key=abs, ascending=False)
    corr_matrix = data[feature_cols].corr(method='spearman')
    return target_df, corr_matrix

def run_permutation_importance(df, rank_cols, market_cols, start_year, n_neighbors, target_days):
    baseline_res = backtest_euclidean_model(df, rank_cols, market_cols, start_year=start_year, n_neighbors=n_neighbors, target_days=target_days)
    if baseline_res.empty: return pd.DataFrame()
    baseline_ic, _ = spearmanr(baseline_res['Predicted'], baseline_res['Actual'])
    features = rank_cols + ['Seasonal'] + market_cols
    importance_data = []
    prog = st.progress(0)
    n_feats = len(features)
    for i, feat in enumerate(features):
        prog.progress((i / n_feats))
        shuffled_df = df.copy()
        shuffled_df[feat] = np.random.permutation(shuffled_df[feat].values)
        perm_res = backtest_euclidean_model(shuffled_df, rank_cols, market_cols, start_year=start_year, n_neighbors=n_neighbors, target_days=target_days)
        if not perm_res.empty:
            perm_ic, _ = spearmanr(perm_res['Predicted'], perm_res['Actual'])
            importance_data.append({'Feature': feat, 'Shuffled IC': perm_ic, 'Importance (IC Drop)': baseline_ic - perm_ic})
    prog.empty()
    return pd.DataFrame(importance_data).sort_values('Importance (IC Drop)', ascending=False)

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
                ensemble_df = calculate_distribution_ensemble(df, rank_cols, [c for c in df.columns if c.startswith("Mkt_")], tolerance=5)
                if not ensemble_df.empty:
                    num_cols = ensemble_df.select_dtypes(include=[np.number]).columns
                    st.dataframe(ensemble_df.style.format("{:.2f}", subset=num_cols).background_gradient(cmap="RdBu", vmin=-2, vmax=2, subset=['Alpha']))
            
            with tab_backtest:
                st.write("Statistical Predictive Power (IC)")
                b_cols = st.columns(3)
                with b_cols[0]: bt_start = st.number_input("Start Year", 2010, 2024, 2018, key="bt_start_ic")
                with b_cols[1]: bt_k = st.number_input("Neighbors", 10, 200, 50, key="bt_k_ic")
                with b_cols[2]: bt_t = st.selectbox("Target", [5, 10, 21], key="bt_t_ic")
                
                # --- ADD FEATURE INSPECTION BUTTONS HERE FOR CONVENIENCE ---
                target_col_name = f'FwdRet_{bt_t}d'
                active_feats = rank_cols + ['Seasonal'] + [c for c in df.columns if c.startswith("Mkt_")]
                
                if st.checkbox("Show Feature Inspector", value=False):
                    icols = st.columns(2)
                    with icols[0]:
                        if st.button("Calc Correlations"):
                            target_corr_df, corr_matrix = calculate_feature_correlations(df, active_feats, target_col=target_col_name)
                            fig_pred = go.Figure(go.Bar(x=target_corr_df['Target Correlation'], y=target_corr_df['Feature'], orientation='h', marker=dict(color=target_corr_df['Target Correlation'], colorscale='RdBu', cmid=0)))
                            fig_pred.update_layout(title=f"Individual Feature vs {target_col_name}", height=500, yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig_pred, use_container_width=True)
                            
                            st.write("**Redundancy Matrix**")
                            fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index, colorscale='RdBu', zmin=-1, zmax=1))
                            fig_corr.update_layout(height=600)
                            st.plotly_chart(fig_corr, use_container_width=True)
                    with icols[1]:
                        if st.button("Run Permutation Importance"):
                            with st.spinner("Shuffling features..."):
                                imp_df = run_permutation_importance(df, rank_cols, [c for c in df.columns if c.startswith("Mkt_")], bt_start, bt_k, bt_t)
                                if not imp_df.empty:
                                    fig_imp = go.Figure(go.Bar(x=imp_df['Importance (IC Drop)'], y=imp_df['Feature'], orientation='h', marker=dict(color=imp_df['Importance (IC Drop)'], colorscale='Viridis')))
                                    fig_imp.update_layout(title=f"Feature Importance (Drop in IC)", height=500, yaxis={'categoryorder':'total ascending'})
                                    st.plotly_chart(fig_imp, use_container_width=True)

                if st.button("Run IC Check"):
                    res = backtest_euclidean_model(df, rank_cols, [c for c in df.columns if c.startswith("Mkt_")], bt_start, bt_k, bt_t)
                    if not res.empty:
                        ic = pearsonr(res['Predicted'], res['Actual'])[0]
                        st.metric("Information Coefficient", f"{ic:.3f}")
                        st.line_chart(res['Predicted'].rolling(252).corr(res['Actual']))
                        display_net_zero_check(res, "Standard Model")
            
            with tab_portfolio:
                st.markdown("### 💰 Realistic Portfolio Simulator")
                st.info("Runs 4 scenarios simultaneously to show the impact of Slippage and Turnover.")
                
                # --- NEW: Simulation Specific Controls ---
                sim_c1, sim_c2, sim_c3 = st.columns(3)
                with sim_c1: sim_start = st.number_input("Sim Start Year", 2010, 2024, 2018, key="sim_start")
                with sim_c2: sim_k = st.number_input("Neighbors (k)", 10, 200, 50, key="sim_k")
                with sim_c3: sim_target = st.selectbox("Target Horizon (Signal)", [5, 10, 21], index=0, key="sim_target")

                st.markdown("---")
                
                p_col1, p_col2, p_col3 = st.columns(3)
                with p_col1: 
                    lev_long = st.slider("Max Long", 1.0, 3.0, 2.0, 0.1)
                    lev_short = st.slider("Max Short", 0.0, 2.0, 1.0, 0.1) * -1
                with p_col2:
                    conviction = st.slider("Conviction (Pos % per 1% Pred)", 0.1, 2.0, 0.5, 0.1)
                    slippage = st.number_input("Slippage (bps per trade)", value=5, min_value=0)
                with p_col3:
                    start_cap = st.number_input("Start Capital", value=100000)
                
                active_weights = {"Ret_21d_Rank": 3.0, "NAAIM_Rank": 2.0} 
                
                if st.button("Run Multi-Scenario Simulation"):
                    with st.spinner(f"Simulating using {sim_target}d forecasts..."):
                        # Use local simulation parameters, NOT the IC tab parameters
                        preds = backtest_euclidean_model(df, rank_cols, [c for c in df.columns if c.startswith("Mkt_")], 
                                                         start_year=sim_start, n_neighbors=sim_k, target_days=sim_target, 
                                                         weights_dict=active_weights)
                        
                        # 1. Daily No Slip
                        daily_raw = run_portfolio_simulation(df, preds, lev_long, lev_short, conviction, start_cap, slippage_bps=0, rebalance_weekly=False)
                        # 2. Daily Slip
                        daily_slip = run_portfolio_simulation(df, preds, lev_long, lev_short, conviction, start_cap, slippage_bps=slippage, rebalance_weekly=False)
                        # 3. Weekly No Slip
                        weekly_raw = run_portfolio_simulation(df, preds, lev_long, lev_short, conviction, start_cap, slippage_bps=0, rebalance_weekly=True)
                        # 4. Weekly Slip
                        weekly_slip = run_portfolio_simulation(df, preds, lev_long, lev_short, conviction, start_cap, slippage_bps=slippage, rebalance_weekly=True)
                        
                        if not daily_raw.empty:
                            # Benchmark
                            bench_series = daily_raw['Benchmark_Equity']
                            bench_ret_series = daily_raw['Asset_Daily_Ret']
                            
                            # Calculate Benchmark Metrics
                            b_cagr, b_sharpe, b_sortino, b_vol = calc_metrics(bench_ret_series)
                            b_final = bench_series.iloc[-1]
                            b_ret = (b_final - start_cap)/start_cap * 100
                            b_peak = bench_series.cummax()
                            b_dd = ((bench_series - b_peak)/b_peak).min() * 100
                            
                            # Metrics Table
                            metrics = []
                            # Add Benchmark first
                            metrics.append({
                                "Scenario": "Buy & Hold SPY", 
                                "Final Equity": f"${b_final:,.0f}", 
                                "Total Return": f"{b_ret:.1f}%", 
                                "CAGR": f"{b_cagr:.1f}%",
                                "Max Drawdown": f"{b_dd:.1f}%",
                                "Sharpe": b_sharpe,
                                "Sortino": b_sortino,
                                "Vol": f"{b_vol:.1f}%"
                            })
                            
                            scenarios = {
                                "Daily (No Slip)": daily_raw,
                                f"Daily ({slippage}bps)": daily_slip,
                                "Weekly (No Slip)": weekly_raw,
                                f"Weekly ({slippage}bps)": weekly_slip
                            }
                            
                            for name, s_df in scenarios.items():
                                final = s_df['Strategy_Equity'].iloc[-1]
                                ret = (final - start_cap)/start_cap * 100
                                dd = s_df['Drawdown'].min() * 100
                                cagr, sharpe, sortino, vol = calc_metrics(s_df['Strategy_Ret_Net'])
                                
                                metrics.append({
                                    "Scenario": name, 
                                    "Final Equity": f"${final:,.0f}", 
                                    "Total Return": f"{ret:.1f}%", 
                                    "CAGR": f"{cagr:.1f}%",
                                    "Max Drawdown": f"{dd:.1f}%",
                                    "Sharpe": sharpe,
                                    "Sortino": sortino,
                                    "Vol": f"{vol:.1f}%"
                                })
                            
                            metrics_df = pd.DataFrame(metrics)
                            # Formatting
                            st.dataframe(
                                metrics_df.style.format({
                                    "Sharpe": "{:.2f}", 
                                    "Sortino": "{:.2f}"
                                }).background_gradient(subset=["Sharpe", "Sortino"], cmap="RdYlGn", vmin=0, vmax=2.5)
                            )
                            
                            # Comparison Chart
                            fig_comp = go.Figure()
                            fig_comp.add_trace(go.Scatter(x=daily_raw.index, y=bench_series, name="Buy & Hold", line=dict(color='grey', dash='dot')))
                            fig_comp.add_trace(go.Scatter(x=daily_raw.index, y=daily_raw['Strategy_Equity'], name="Daily (No Slip)", line=dict(color='green', width=1)))
                            fig_comp.add_trace(go.Scatter(x=daily_slip.index, y=daily_slip['Strategy_Equity'], name=f"Daily ({slippage}bps)", line=dict(color='lightgreen', width=2)))
                            fig_comp.add_trace(go.Scatter(x=weekly_raw.index, y=weekly_raw['Strategy_Equity'], name="Weekly (No Slip)", line=dict(color='blue', width=1)))
                            fig_comp.add_trace(go.Scatter(x=weekly_slip.index, y=weekly_slip['Strategy_Equity'], name=f"Weekly ({slippage}bps)", line=dict(color='cyan', width=2)))
                            
                            fig_comp.update_layout(title="Impact of Friction & Rebalancing Frequency (Log Scale)", yaxis_title="Equity ($)", height=600, yaxis_type="log")
                            st.plotly_chart(fig_comp, use_container_width=True)
                            
                        else:
                            st.error("Simulation failed.")

def main():
    st.set_page_config(layout="wide", page_title="Heatmap Analytics")
    render_heatmap()

if __name__ == "__main__":
    main()
