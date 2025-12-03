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
NAAIM_PATH = "naaim.csv"  # <--- NEW CONSTANT

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
        output_map[ticker] = pd.Series(
            group.seasonal_rank.values, index=group.MD
        ).to_dict()
    return output_map

@st.cache_data(show_spinner=False)
def load_market_metrics():
    try:
        df = pd.read_csv(METRICS_PATH)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
    except Exception:
        return pd.DataFrame()

    if df.empty: return pd.DataFrame()

    pivoted = df.pivot_table(
        index='date', 
        columns='exchange', 
        values='net_new_highs', 
        aggfunc='sum'
    )
    
    pivoted['Total'] = pivoted.get('NYSE', 0) + pivoted.get('NASDAQ', 0)
    
    windows = [5, 21] 
    categories = ['Total']
    
    results = pd.DataFrame(index=pivoted.index)
    
    for cat in categories:
        if cat not in pivoted.columns: continue
        series = pivoted[cat]
        for w in windows:
            ma_col = series.rolling(window=w).mean()
            rank_col_name = f"Mkt_{cat}_NH_{w}d_Rank"
            results[rank_col_name] = ma_col.expanding(min_periods=126).rank(pct=True) * 100.0

    return results

@st.cache_data(show_spinner=False)
def load_naaim_data():
    """Loads NAAIM CSV, calcs Weekly MAs, and preps for daily merge."""
    try:
        df = pd.read_csv(NAAIM_PATH)
        # Attempt to identify date column and value column
        # Assuming typical format, but let's be robust
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        elif 'date' in df.columns:
             df['Date'] = pd.to_datetime(df['date'])
        else:
             # Fallback: assume first column is date
             df['Date'] = pd.to_datetime(df.iloc[:, 0])

        df = df.set_index('Date').sort_index()
        
        # Identify the numeric column (usually 'Naaim Number' or similar)
        # We take the first numeric column we find
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            val_col = num_cols[0]
            df = df[[val_col]].rename(columns={val_col: 'NAAIM'})
        else:
            return pd.DataFrame()

        # Calculate MAs on the WEEKLY data (before merging to daily)
        # This ensures '5 period' means 5 weeks, not 5 days
        df['NAAIM_MA5'] = df['NAAIM'].rolling(5).mean()
        df['NAAIM_MA12'] = df['NAAIM'].rolling(12).mean()

        return df
    except Exception as e:
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# DYNAMIC SEASONALITY & DATA
# -----------------------------------------------------------------------------
def generate_dynamic_seasonal_profile(df, cutoff_date, target_year):
    work_df = df[df.index < cutoff_date].copy()
    if len(work_df) < 500: return {}

    if 'LogRet' not in work_df.columns:
        if 'Close' in work_df.columns:
             work_df['LogRet'] = np.log(work_df['Close'] / work_df['Close'].shift(1)) * 100.0
        else:
             return {}
    
    work_df['Year'] = work_df.index.year
    work_df['MD'] = work_df.index.map(lambda x: (x.month, x.day))
    
    cycle_remainder = target_year % 4
    
    windows = [5, 10, 21]
    fwd_cols = []
    for w in windows:
        col_name = f'Fwd_{w}d'
        work_df[col_name] = work_df['LogRet'].shift(-w).rolling(w).sum()
        fwd_cols.append(col_name)

    stats_all = work_df.groupby('MD')[fwd_cols].mean()
    cycle_df = work_df[work_df['Year'] % 4 == cycle_remainder]
    
    if len(cycle_df) < 250: 
        stats_cycle = stats_all.copy()
    else:
        stats_cycle = cycle_df.groupby('MD')[fwd_cols].mean()
    
    stats_cycle = stats_cycle.reindex(stats_all.index).fillna(method='ffill').fillna(method='bfill')
    
    rnk_all = stats_all.rank(pct=True) * 100.0
    rnk_cycle = stats_cycle.rank(pct=True) * 100.0
    
    avg_rank_all = rnk_all.mean(axis=1)
    avg_rank_cycle = rnk_cycle.mean(axis=1)
    
    final_rank = (avg_rank_all + 3 * avg_rank_cycle) / 4.0
    final_rank_smooth = final_rank.rolling(window=5, center=True, min_periods=1).mean()
    
    return final_rank_smooth.to_dict()

def get_sznl_val_series(ticker, dates, sznl_map, df_hist=None):
    t_map = sznl_map.get(ticker, {})
    if t_map:
        mds = dates.map(lambda x: (x.month, x.day))
        return mds.map(t_map).fillna(50.0)

    if df_hist is not None and not df_hist.empty:
        if 'LogRet' not in df_hist.columns:
             df_hist = df_hist.copy()
             df_hist['LogRet'] = np.log(df_hist['Close'] / df_hist['Close'].shift(1)) * 100.0

        sznl_series = pd.Series(50.0, index=dates)
        unique_years = dates.year.unique()
        
        for yr in unique_years:
            cutoff = pd.Timestamp(yr, 1, 1)
            if df_hist.index.tz is not None: 
                cutoff = cutoff.tz_localize(df_hist.index.tz)

            yearly_profile = generate_dynamic_seasonal_profile(df_hist, cutoff, yr)
            
            if yearly_profile:
                mask = (dates.year == yr)
                year_mds = dates[mask].map(lambda x: (x.month, x.day))
                sznl_series.loc[mask] = year_mds.map(yearly_profile).fillna(50.0)
                
        return sznl_series

    return pd.Series(50.0, index=dates)

@st.cache_data(show_spinner=True)
def download_data(ticker):
    if not ticker: return pd.DataFrame(), None
    try:
        df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        
        fetch_time = pd.Timestamp.now(tz='America/New_York')
        return df, fetch_time
    except:
        return pd.DataFrame(), None

@st.cache_data(show_spinner=False)
def get_spy_context():
    """Download and calculate SPY rank features separately."""
    try:
        spy, _ = download_data("SPY")
        if spy.empty: return pd.DataFrame()
        
        spy = spy.copy()
        
        # Calculate Ranks for 5, 10, 21 days
        spy_features = pd.DataFrame(index=spy.index)
        for w in [5, 10, 21]:
            col_ret = spy['Close'].pct_change(w)
            col_rank = f'SPY_Ret_{w}d_Rank'
            spy_features[col_rank] = col_ret.expanding(min_periods=252).rank(pct=True) * 100.0
            
        return spy_features
    except:
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# FEATURE ENGINEERING & ENSEMBLE LOGIC
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def calculate_heatmap_variables(df, sznl_map, market_metrics_df, ticker):
    df = df.copy()
    
    # 1. BASE
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))

    # 2. FORWARD TARGETS
    for w in [1, 2, 3, 5, 10, 21, 63, 126, 252]:
        df[f'FwdRet_{w}d'] = (df['Close'].shift(-w) / df['Close'] - 1.0) * 100.0

    # 3. PREDICTOR VARIABLES
    for w in [5, 10, 21, 252]:
        df[f'Ret_{w}d'] = df['Close'].pct_change(w)

    for w in [21, 63]:
        df[f'RealVol_{w}d'] = df['LogRet'].rolling(w).std() * np.sqrt(252) * 100.0
    
    for w in [10, 21]:
        df[f'VolRatio_{w}d'] = df['Volume'].rolling(w).mean() / df['Volume'].rolling(63).mean()

    # Seasonality
    df['Seasonal'] = get_sznl_val_series(ticker, df.index, sznl_map, df)

    # 4. RANK TRANSFORMATION
    vars_to_rank = [
        'Ret_5d', 'Ret_10d', 'Ret_21d', 'Ret_252d', 
        'RealVol_21d', 'RealVol_63d', 
        'VolRatio_10d', 'VolRatio_21d'
    ]
    
    rank_cols = []
    for v in vars_to_rank:
        col_name = v + '_Rank'
        df[col_name] = df[v].expanding(min_periods=252).rank(pct=True) * 100.0
        rank_cols.append(col_name)

    # 5. MERGE MARKET METRICS
    if not market_metrics_df.empty:
        df = df.join(market_metrics_df, how='left')
        df.update(df.filter(regex='^Mkt_').ffill(limit=3))
        
    # 6. MERGE SPY CONTEXT
    if ticker != "SPY":
        spy_df = get_spy_context()
        if not spy_df.empty:
            df = df.join(spy_df, how='left')
            spy_cols = [c for c in spy_df.columns if 'Rank' in c]
            rank_cols.extend(spy_cols)
            df.update(df[spy_cols].ffill(limit=1))

    # 7. MERGE NAAIM DATA
    naaim_df = load_naaim_data()
    if not naaim_df.empty:
        # We join NAAIM data. Since NAAIM is weekly, we forward fill.
        df = df.join(naaim_df, how='left')
        
        # Explicitly forward fill the NAAIM columns
        naaim_cols_raw = ['NAAIM', 'NAAIM_MA5', 'NAAIM_MA12']
        df[naaim_cols_raw] = df[naaim_cols_raw].ffill()
        
        # Calculate Percentile Ranks for the 3 metrics
        for col in naaim_cols_raw:
            rank_col = f"{col}_Rank"
            # Expanding rank gives us the percentile vs history
            df[rank_col] = df[col].expanding(min_periods=252).rank(pct=True) * 100.0
            rank_cols.append(rank_col)

    return df, rank_cols

def calculate_distribution_ensemble(df, rank_cols, market_cols, tolerance=1.0):
    if df.empty: return pd.DataFrame()
    
    current_row = df.iloc[-1]
    
    # We now include 'NAAIM' related columns in valid features check
    all_features = rank_cols + ['Seasonal'] + market_cols
    # Ensure any new rank cols (like NAAIM) are picked up
    valid_features = [f for f in all_features if f in df.columns and not pd.isna(current_row[f])]
    
    if len(valid_features) < 2: return pd.DataFrame()
    
    targets = [2, 3, 5, 10, 21, 63, 126, 252]
    pooled_outcomes = {t: [] for t in targets}
    
    pairs = []
    
    # 1. Standard Pairs (No SPY cols)
    non_spy_valid = [f for f in valid_features if not f.startswith("SPY_")]
    pairs.extend(list(itertools.combinations(non_spy_valid, 2)))
    
    # 2. Specific SPY Pairs (SPY_Xd vs Ret_Xd ONLY)
    spy_map = {
        'SPY_Ret_5d_Rank': 'Ret_5d_Rank',
        'SPY_Ret_10d_Rank': 'Ret_10d_Rank',
        'SPY_Ret_21d_Rank': 'Ret_21d_Rank'
    }
    
    for spy_col, ticker_col in spy_map.items():
        if spy_col in valid_features and ticker_col in valid_features:
            pairs.append((spy_col, ticker_col))
    
    for f1, f2 in pairs:
        v1 = current_row[f1]
        v2 = current_row[f2]
        
        mask = (
            (df[f1] >= v1 - tolerance) & (df[f1] <= v1 + tolerance) &
            (df[f2] >= v2 - tolerance) & (df[f2] <= v2 + tolerance)
        )
        
        subset = df[mask]
        
        if subset.empty: continue
        
        for t in targets:
            col = f'FwdRet_{t}d'
            if col not in subset.columns: continue
            
            vals = subset[col].dropna().tolist()
            if vals:
                pooled_outcomes[t].extend(vals)
            
    summary = []
    for t in targets:
        col = f'FwdRet_{t}d'
        if col in df.columns:
            baseline = df[col].mean()
        else:
            baseline = np.nan
            
        data = np.array(pooled_outcomes[t])
        
        if len(data) == 0:
            summary.append({
                "Horizon": f"{t} Days", 
                "Exp Return": np.nan,
                "Baseline": baseline,
                "Alpha": np.nan,
                "Implied Vol (IV)": np.nan,
                "Win Rate": np.nan, 
                "Sample Size": 0
            })
            continue
            
        grand_mean = np.mean(data)
        std_dev = np.std(data)
        iv_est = std_dev * np.sqrt(252 / t)
        pos_ratio = np.sum(data > 0) / len(data)
        alpha = grand_mean - baseline if not np.isnan(baseline) else np.nan
        
        summary.append({
            "Horizon": f"{t} Days",
            "Exp Return": grand_mean,
            "Baseline": baseline,
            "Alpha": alpha,
            "Implied Vol (IV)": iv_est,
            "Win Rate": pos_ratio * 100,
            "Sample Size": len(data)
        })
        
    return pd.DataFrame(summary)

# --- METHOD 2: EUCLIDEAN NEAREST NEIGHBORS ---
def calculate_euclidean_forecast(df, rank_cols, market_cols, n_neighbors=50):
    current_row = df.iloc[-1]
    all_feats = rank_cols + ['Seasonal'] + market_cols
    valid_feats = [f for f in all_feats if f in df.columns and not pd.isna(current_row[f])]
    
    if not valid_feats: return pd.DataFrame()
    
    target_vec = current_row[valid_feats].astype(float).values
    history = df.iloc[:-1].dropna(subset=valid_feats).copy()
    
    if history.empty: return pd.DataFrame()
    
    diff = history[valid_feats].values - target_vec
    dist_sq = np.sum(diff**2, axis=1)
    history['Euclidean_Dist'] = np.sqrt(dist_sq)
    
    n_take = min(len(history), n_neighbors)
    nearest = history.nsmallest(n_take, 'Euclidean_Dist')
    
    targets = [2, 3, 5, 10, 21, 63, 126, 252]
    summary = []
    
    for t in targets:
        col = f'FwdRet_{t}d'
        if col not in df.columns: continue
        
        baseline = df[col].mean()
        outcomes = nearest[col].dropna()
        
        if outcomes.empty:
            summary.append({
                "Horizon": f"{t} Days", "Exp Return": np.nan, 
                "Baseline": baseline, "Alpha": np.nan, 
                "Implied Vol (IV)": np.nan, "Win Rate": np.nan, 
                "Sample Size": 0
            })
            continue
            
        grand_mean = outcomes.mean()
        std_dev = outcomes.std()
        iv_est = std_dev * np.sqrt(252 / t)
        win_rate = (outcomes > 0).mean() * 100
        alpha = grand_mean - baseline
        
        summary.append({
            "Horizon": f"{t} Days",
            "Exp Return": grand_mean,
            "Baseline": baseline,
            "Alpha": alpha,
            "Implied Vol (IV)": iv_est,
            "Win Rate": win_rate,
            "Sample Size": len(outcomes)
        })
        
    return pd.DataFrame(summary)

def get_euclidean_details(df, rank_cols, market_cols, n_neighbors=50, target_days=5):
    """
    Returns the specific neighbor rows for the Euclidean calculation.
    """
    current_row = df.iloc[-1]
    all_feats = rank_cols + ['Seasonal'] + market_cols
    valid_feats = [f for f in all_feats if f in df.columns and not pd.isna(current_row[f])]
    
    if not valid_feats: return pd.DataFrame()
    
    target_vec = current_row[valid_feats].astype(float).values
    history = df.iloc[:-1].dropna(subset=valid_feats).copy()
    
    if history.empty: return pd.DataFrame()
    
    diff = history[valid_feats].values - target_vec
    dist_sq = np.sum(diff**2, axis=1)
    history['Euclidean_Dist'] = np.sqrt(dist_sq)
    
    n_take = min(len(history), n_neighbors)
    nearest = history.nsmallest(n_take, 'Euclidean_Dist').copy()
    
    # Format Table Data
    ret_col = f'FwdRet_{target_days}d'
    if ret_col in nearest.columns:
        nearest['Fwd Return'] = nearest[ret_col]
    else:
        nearest['Fwd Return'] = np.nan
        
    fwd_vol_series = df['LogRet'].rolling(target_days).std().shift(-target_days) * np.sqrt(252) * 100
    nearest['Fwd Realized Vol'] = fwd_vol_series.loc[nearest.index]
    
    output = nearest[['Euclidean_Dist', 'Close', 'Fwd Return', 'Fwd Realized Vol']].copy()
    output.columns = ['Euclidean Distance', 'Close Price', 'Fwd Return', 'Fwd Realized Vol']
    
    return output.dropna(subset=['Fwd Return']).sort_values('Euclidean Distance', ascending=True)

# --- CHARTING & TABLE HELPERS ---
def get_feature_shorthand(name):
    n = name.replace("_Rank", "")
    if n == "Seasonal": return "szn"
    if n.startswith("Ret_"): return n.replace("Ret_", "").replace("d", "") + "dr"
    if n.startswith("VolRatio_"): return n.replace("VolRatio_", "").replace("d", "") + "drv"
    if n.startswith("RealVol_"): return n.replace("RealVol_", "").replace("d", "") + "dv"
    if n.startswith("Mkt_"): return n.split("_")[-1] + "nh"
    if n.startswith("SPY_"): return n.replace("SPY_Ret_", "spy") + "dr"
    if n.startswith("NAAIM"): 
        if "MA5" in n: return "naaim5"
        if "MA12" in n: return "naaim12"
        return "naaim"
    return n[:4]

def get_detailed_match_table(df, rank_cols, market_cols, tolerance=5.0, target_days=5):
    if df.empty: return pd.DataFrame()
    current_row = df.iloc[-1]
    all_features = rank_cols + ['Seasonal'] + market_cols
    valid_features = [f for f in all_features if f in df.columns and not pd.isna(current_row[f])]
    if len(valid_features) < 2: return pd.DataFrame()
    
    pairs = []
    # Use logic from ensemble: Standard Pairs + Specific SPY Pairs
    non_spy_valid = [f for f in valid_features if not f.startswith("SPY_")]
    pairs.extend(list(itertools.combinations(non_spy_valid, 2)))
    
    spy_map = {
        'SPY_Ret_5d_Rank': 'Ret_5d_Rank',
        'SPY_Ret_10d_Rank': 'Ret_10d_Rank',
        'SPY_Ret_21d_Rank': 'Ret_21d_Rank'
    }
    for spy_col, ticker_col in spy_map.items():
        if spy_col in valid_features and ticker_col in valid_features:
            pairs.append((spy_col, ticker_col))

    date_to_pairs = defaultdict(list)
    
    for f1, f2 in pairs:
        v1 = current_row[f1]
        v2 = current_row[f2]
        mask = (
            (df[f1] >= v1 - tolerance) & (df[f1] <= v1 + tolerance) &
            (df[f2] >= v2 - tolerance) & (df[f2] <= v2 + tolerance)
        )
        subset = df[mask]
        
        if not subset.empty:
            pair_str = f"{get_feature_shorthand(f1)} & {get_feature_shorthand(f2)}"
            for date in subset.index:
                date_to_pairs[date].append(pair_str)
            
    if not date_to_pairs: return pd.DataFrame()

    dates = list(date_to_pairs.keys())
    scores = [len(date_to_pairs[d]) for d in dates]
    pair_strings = []
    for d in dates:
        unique_pairs = list(set(date_to_pairs[d]))
        display_str = ", ".join(unique_pairs[:3])
        if len(unique_pairs) > 3: display_str += f" (+{len(unique_pairs)-3} more)"
        pair_strings.append(display_str)
    
    match_df = pd.DataFrame({
        'Similarity Score': scores,
        'Trigger Pairs': pair_strings
    }, index=dates)
    match_df.index.name = 'Date'
    
    ret_col = f'FwdRet_{target_days}d'
    if ret_col in df.columns: match_df['Fwd Return'] = df.loc[match_df.index, ret_col]
    else: match_df['Fwd Return'] = np.nan

    fwd_vol_series = df['LogRet'].rolling(target_days).std().shift(-target_days) * np.sqrt(252) * 100
    match_df['Fwd Realized Vol'] = fwd_vol_series.loc[match_df.index]
    match_df['Close Price'] = df.loc[match_df.index, 'Close']
    
    return match_df.dropna(subset=['Fwd Return']).sort_values(by=['Similarity Score', 'Date'], ascending=[False, False])

# -----------------------------------------------------------------------------
# HEATMAP UTILS
# -----------------------------------------------------------------------------
def get_seismic_colorscale():
    seismic = []
    cm = matplotlib.colormaps["seismic"] 
    for k in range(255):
        r, g, b, _ = cm(k / 254.0)
        seismic.append([k / 254.0, f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'])
    return seismic

@st.cache_data
def build_bins_quantile(x, y, nx=30, ny=30):
    x = pd.Series(x, dtype=float).dropna()
    y = pd.Series(y, dtype=float).dropna()
    if len(x) < 10 or len(y) < 10:
        return np.linspace(0,100,nx+1), np.linspace(0,100,ny+1)
    x_edges = np.unique(np.quantile(x, np.linspace(0, 1, nx + 1)))
    y_edges = np.unique(np.quantile(y, np.linspace(0, 1, ny + 1)))
    if len(x_edges) < 3: x_edges = np.linspace(x.min(), x.max(), max(3, nx + 1))
    if len(y_edges) < 3: y_edges = np.linspace(y.min(), y.max(), max(3, ny + 1))
    return x_edges, y_edges

@st.cache_data
def grid_mean(df_sub, xcol, ycol, zcol, x_edges, y_edges):
    x = df_sub[xcol].to_numpy(dtype=float)
    y = df_sub[ycol].to_numpy(dtype=float)
    z = df_sub[zcol].to_numpy(dtype=float)
    m = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    x = x[m]; y = y[m]; z = z[m]

    if len(x) == 0:
        return np.array([]), np.array([]), np.full((len(y_edges)-1, len(x_edges)-1), np.nan)

    xi = np.digitize(x, x_edges) - 1
    yi = np.digitize(y, y_edges) - 1
    nx = len(x_edges) - 1; ny = len(y_edges) - 1

    sums    = np.zeros((ny, nx), float)
    counts = np.zeros((ny, nx), float)

    valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    xi = xi[valid]; yi = yi[valid]; zv = z[valid]
    
    for X, Y, Z in zip(xi, yi, zv):
        sums[Y, X]  += Z
        counts[Y, X]+= 1.0

    with np.errstate(invalid='ignore', divide='ignore'):
        mean = sums / counts
    mean[counts == 0] = np.nan

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centers, y_centers, mean

def nan_neighbor_fill(Z, k=3):
    Z = np.asarray(Z, float)
    kernel = np.ones((k, k), dtype=float)
    valid = np.isfinite(Z).astype(float)
    sum_vals    = convolve2d(np.nan_to_num(Z), kernel, mode='same', boundary='symm')
    count_vals = convolve2d(valid, kernel, mode='same', boundary='symm')
    Z_out = Z.copy()
    mask = np.isnan(Z) & (count_vals > 0)
    Z_out[mask] = sum_vals[mask] / count_vals[mask]
    return Z_out

def smooth_display(Z, sigma=1.2):
    Z = np.asarray(Z, float)
    mask = np.isfinite(Z)
    Z_fill = np.where(mask, Z, 0.0)
    w = gaussian_filter(mask.astype(float), sigma=sigma, mode="nearest")
    z = gaussian_filter(Z_fill, sigma=sigma, mode="nearest")
    out = z / np.maximum(w, 1e-9)
    return out
# -----------------------------------------------------------------------------
# BACKTESTING ENGINE
# -----------------------------------------------------------------------------
def backtest_euclidean_model(df, rank_cols, market_cols, start_year=2015, n_neighbors=50, target_days=5):
    """
    Performs a walk-forward analysis. For every day from start_year to present,
    it finds the N nearest neighbors using ONLY past data, predicts the return,
    and stores it for comparison.
    """
    # Filter for relevant columns
    all_feats = rank_cols + ['Seasonal'] + market_cols
    target_col = f'FwdRet_{target_days}d'
    
    # We need data that has the target column (FwdRet) valid
    # (Actually for the 'current' day we don't need it valid to make a prediction, 
    # but to score the backtest we need the actual result)
    validation_df = df.dropna(subset=all_feats + [target_col]).copy()
    
    # Filter to start year, but ensure we have enough history prior to it
    test_indices = validation_df[validation_df.index.year >= start_year].index
    
    if len(test_indices) == 0:
        return pd.DataFrame()

    predictions = []
    actuals = []
    dates = []
    
    # Progress Bar in Streamlit
    progress_bar = st.progress(0)
    total_steps = len(test_indices)
    
    # Convert to numpy for speed
    # We use the full validation_df for the "history" pool
    # BUT we only iterate through the test_indices for the "target"
    
    feat_matrix = validation_df[all_feats].values
    target_array = validation_df[target_col].values
    index_array = validation_df.index
    
    # Map test indices to integer locations in the numpy array
    # This is much faster than pandas indexing inside a loop
    start_loc = np.searchsorted(index_array, test_indices[0])
    
    for i in range(start_loc, len(validation_df)):
        # Update progress
        curr_step = i - start_loc
        if curr_step % 50 == 0:
            progress_bar.progress(min(curr_step / total_steps, 1.0))
            
        # 1. Define History (Strictly < Current Index i)
        # We need at least n_neighbors to make a prediction
        if i < n_neighbors: continue

        hist_feats = feat_matrix[:i]
        hist_rets = target_array[:i]
        
        # 2. Define Target (The day we are predicting FOR)
        curr_feat = feat_matrix[i]
        curr_actual = target_array[i]
        
        # 3. Euclidean Distance (Vectorized)
        # axis=1 means sum across columns (features)
        dists = np.sqrt(np.sum((hist_feats - curr_feat)**2, axis=1))
        
        # 4. Find Neighbors
        # argpartition is faster than sort for just finding top k
        if len(dists) < n_neighbors: continue
        
        # Get indices of k smallest distances
        # Note: argpartition puts the k smallest in the first k positions, but not sorted
        neighbor_idxs = np.argpartition(dists, n_neighbors)[:n_neighbors]
        
        # 5. Forecast
        pred_ret = np.mean(hist_rets[neighbor_idxs])
        
        dates.append(index_array[i])
        predictions.append(pred_ret)
        actuals.append(curr_actual)
        
    progress_bar.empty()
    
    results = pd.DataFrame({
        'Predicted': predictions,
        'Actual': actuals
    }, index=dates)
    
    return results
# -----------------------------------------------------------------------------
# UI: RENDER
# -----------------------------------------------------------------------------
def render_heatmap():
    st.subheader("Heatmap Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    var_options = {
        "Seasonality Rank": "Seasonal",
        # Price
        "5d Trailing Return Rank": "Ret_5d_Rank",
        "10d Trailing Return Rank": "Ret_10d_Rank",
        "21d Trailing Return Rank": "Ret_21d_Rank",
        "252d Trailing Return Rank": "Ret_252d_Rank",
        # SPY Context
        "SPY 5d Return Rank": "SPY_Ret_5d_Rank",
        "SPY 10d Return Rank": "SPY_Ret_10d_Rank",
        "SPY 21d Return Rank": "SPY_Ret_21d_Rank",
        # Vol
        "21d Realized Vol Rank": "RealVol_21d_Rank",
        "63d Realized Vol Rank": "RealVol_63d_Rank",
        # Volume
        "10d Rel. Volume Rank": "VolRatio_10d_Rank", 
        "21d Rel. Volume Rank": "VolRatio_21d_Rank",
        # Market Breadth
        "Total Net Highs (5d MA) Rank": "Mkt_Total_NH_5d_Rank",
        "Total Net Highs (21d MA) Rank": "Mkt_Total_NH_21d_Rank",
        # NAAIM
        "NAAIM Exposure Rank": "NAAIM_Rank",
        "NAAIM 5wk MA Rank": "NAAIM_MA5_Rank",
        "NAAIM 12wk MA Rank": "NAAIM_MA12_Rank"
    }

    target_options = {
        "5d Forward Return": "FwdRet_5d",
        "10d Forward Return": "FwdRet_10d",
        "21d Forward Return": "FwdRet_21d",
        "63d Forward Return": "FwdRet_63d",
        "126d Forward Return": "FwdRet_126d",
        "252d Forward Return": "FwdRet_252d",
    }

    with col1:
        x_axis_label = st.selectbox("X-Axis Variable", list(var_options.keys()), index=0, key="hm_x")
        y_axis_label = st.selectbox("Y-Axis Variable", list(var_options.keys()), index=2, key="hm_y")
    
    with col2:
        z_axis_label = st.selectbox("Target (Z-Axis)", list(target_options.keys()), index=2, key="hm_z")
        ticker = st.text_input("Ticker", value="SMH", key="hm_ticker").upper()
        
        # --- 3rd Dimension Filter ---
        st.markdown("---")
        st.write(" **3rd Dimension Filter**")
        filter_opts_list = ["None"] + list(var_options.keys())
        filter_label = st.selectbox("Filter by Condition:", filter_opts_list, index=0, key="hm_filter_var")
        
        filter_range = (0.0, 100.0)
        if filter_label != "None":
            filter_range = st.slider(
                f"Keep Data where {filter_label} is:", 
                0.0, 100.0, (50.0, 100.0), step=1.0, key="hm_filter_rng"
            )
    
    with col3:
        smooth_sigma = st.slider("Smoothing (Sigma)", 0.5, 3.0, 1.2, 0.1, key="hm_smooth")
        bins = st.slider("Grid Resolution (Bins)", 10, 50, 28, key="hm_bins")
        ensemble_tol = st.slider("Ensemble Similarity Tolerance (± Rank)", 1, 25, 5, 1, key="ens_tol")
        analysis_start = st.date_input("Analysis Start Date", value=datetime.date(2000, 1, 1), key="hm_start_date")
    
    st.markdown("---")

    if st.button("Generate Heatmap & Ensemble", type="primary", use_container_width=True, key="hm_gen"):
        st.session_state['hm_data'] = True 
    
    if st.session_state.get('hm_data'):
        with st.spinner(f"Processing {ticker}..."):
            
            # 1. Download & Calc
            data, fetch_time = download_data(ticker)
            if data.empty:
                st.error("No data found.")
                return

            last_dt = data.index[-1]
            time_str = fetch_time.strftime('%I:%M %p EST') if fetch_time else "N/A"
            st.info(f"Price Data Current as of: {last_dt.strftime('%Y-%m-%d')} | Last Update: {time_str}")
                
            sznl_map = load_seasonal_map()
            market_metrics_df = load_market_metrics()
            
            df, rank_cols = calculate_heatmap_variables(data, sznl_map, market_metrics_df, ticker)
            
            # 2. Time Slice
            start_ts = pd.to_datetime(analysis_start)
            if df.index.tz is not None: start_ts = start_ts.tz_localize(df.index.tz)
            
            df_filtered = df[df.index >= start_ts].copy()
            
            if df_filtered.empty:
                st.error("No data found after start date.")
                return
            
            # --- APPLY 3RD DIMENSION FILTER ---
            filter_desc_str = ""
            if filter_label != "None":
                f_col = var_options[filter_label]
                if f_col in df_filtered.columns:
                    count_before = len(df_filtered)
                    mask = (df_filtered[f_col] >= filter_range[0]) & (df_filtered[f_col] <= filter_range[1])
                    df_filtered = df_filtered[mask]
                    count_after = len(df_filtered)
                    
                    filter_desc_str = f" | Filter: {filter_label} [{filter_range[0]}-{filter_range[1]}] (n={count_after})"
                    st.success(f"Filter Active: Reduced dataset from {count_before} to {count_after} datapoints.")
                    
                    if count_after < 50:
                        st.warning("⚠️ Warning: Filter is very strict. Heatmap may look sparse or empty.")
                else:
                    st.error(f"Filter column {f_col} not found.")

            # 3. Heatmap Prep
            xcol = var_options[x_axis_label]
            ycol = var_options[y_axis_label]
            zcol = target_options[z_axis_label]
            
            clean_df = df_filtered.dropna(subset=[xcol, ycol, zcol])
            
            if not clean_df.empty:
                x_edges, y_edges = build_bins_quantile(clean_df[xcol], clean_df[ycol], nx=bins, ny=bins)
                x_centers, y_centers, Z = grid_mean(clean_df, xcol, ycol, zcol, x_edges, y_edges)
                
                Z_filled = nan_neighbor_fill(Z)
                Z_smooth = smooth_display(Z_filled, sigma=smooth_sigma)
                
                limit = np.nanmax(np.abs(Z_smooth))
                colorscale = get_seismic_colorscale()
                z_units = "%" 

                fig = go.Figure(data=go.Heatmap(
                    z=Z_smooth, x=x_centers, y=y_centers,
                    colorscale=colorscale, 
                    zmin=-limit, zmax=limit,
                    zmid=0, 
                    reversescale=True, 
                    colorbar=dict(title=f"{zcol} {z_units}"),
                    hovertemplate=f"<b>{x_axis_label}</b>: %{{x:.1f}}<br><b>{y_axis_label}</b>: %{{y:.1f}}<br><b>Value</b>: %{{z:.2f}}{z_units}<extra></extra>"
                ))
                
                title_text = f"{ticker}: {z_axis_label}{filter_desc_str}"
                
                fig.update_layout(
                    title=title_text,
                    xaxis_title=x_axis_label + " (0-100)",
                    yaxis_title=y_axis_label + " (0-100)",
                    height=600, template="plotly_white",
                    dragmode=False 
                )
                
                # Check if current day passes filter
                last_row = df.iloc[-1]
                curr_x = last_row.get(xcol, np.nan)
                curr_y = last_row.get(ycol, np.nan)
                
                passes_filter = True
                if filter_label != "None":
                    f_col = var_options[filter_label]
                    curr_f = last_row.get(f_col, np.nan)
                    if pd.isna(curr_f) or not (filter_range[0] <= curr_f <= filter_range[1]):
                        passes_filter = False

                if not np.isnan(curr_x) and not np.isnan(curr_y):
                    fig.add_vline(x=curr_x, line_width=2, line_dash="dash", line_color="black")
                    fig.add_hline(y=curr_y, line_width=2, line_dash="dash", line_color="black")
                    
                    marker_text = "Current"
                    if not passes_filter:
                        marker_text = "Current (Excluded by Filter)"
                        
                    fig.add_annotation(x=curr_x, y=curr_y, text=marker_text, showarrow=True, arrowhead=1, ax=30, ay=-30, bgcolor="white")
                
                st.plotly_chart(fig, use_container_width=True, key="heatmap_plot")
                
                # --- HISTORICAL EXPLORER (Uses the FILTERED DF) ---
                st.divider()
                st.markdown("### 🔎 Historical Data Explorer (Filtered)")
                f_col1, f_col2 = st.columns(2)
                with f_col1: x_range = st.slider(f"Inner Filter: {x_axis_label}", 0.0, 100.0, (0.0, 100.0), step=1.0)
                with f_col2: y_range = st.slider(f"Inner Filter: {y_axis_label}", 0.0, 100.0, (0.0, 100.0), step=1.0)
                
                mask_explorer = (clean_df[xcol] >= x_range[0]) & (clean_df[xcol] <= x_range[1]) & \
                                (clean_df[ycol] >= y_range[0]) & (clean_df[ycol] <= y_range[1])
                explorer_df = clean_df[mask_explorer].copy()
                
                if not explorer_df.empty:
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Matching Instances", len(explorer_df))
                    s2.metric(f"Avg Target", f"{explorer_df[zcol].mean():.2f}{z_units}")
                    win_rate = (explorer_df[zcol] > 0).sum() / len(explorer_df) * 100
                    s3.metric("Win Rate (>0)", f"{win_rate:.1f}%")
                    
                    st.write("#### Matching Dates")
                    display_cols = ['Close', xcol, ycol, zcol]
                    if filter_label != "None":
                        display_cols.append(var_options[filter_label])

                    table_df = explorer_df[display_cols].sort_index(ascending=False)
                    # FIX: Format only numeric columns AND KEEP GRADIENT
                    num_cols = table_df.select_dtypes(include=[np.number]).columns
                    st.dataframe(
                        table_df.style.format("{:.2f}", subset=num_cols).background_gradient(subset=[zcol], cmap="RdBu", vmin=-limit, vmax=limit),
                        use_container_width=True, height=400
                    )
                else:
                    st.warning("No matches found in this region.")
                    
            else:
                st.error("Insufficient data for Heatmap (Filter might be too strict).")

            st.divider()
            
            market_cols = [c for c in df.columns if c.startswith("Mkt_")]
            
            # Method 1 (Uses Filtered DF to show stats for the specific regime)
            st.subheader(f"🤖 Method 1: Pairwise Ensemble (Box Filter)")
            ensemble_df = calculate_distribution_ensemble(df_filtered, rank_cols, market_cols, tolerance=ensemble_tol)
            if not ensemble_df.empty:
                # FIX: Format only numeric columns AND KEEP GRADIENT
                num_cols = ensemble_df.select_dtypes(include=[np.number]).columns
                st.dataframe(
                    ensemble_df.style.format("{:.2f}", subset=num_cols)
                    .background_gradient(subset=['Alpha'], cmap="RdBu", vmin=-2, vmax=2),
                    use_container_width=True
                )
            else:
                st.warning("Not enough data for ensemble.")

            # Method 2 (Uses Full DF to find neighbors for TODAY)
            st.subheader(f"🧪 Method 2: Multi-Factor Similarity")
            
            k_neighbors = 50
            euclidean_df = calculate_euclidean_forecast(df, rank_cols, market_cols, n_neighbors=k_neighbors)
            if not euclidean_df.empty:
                # FIX: Format only numeric columns AND KEEP GRADIENT
                num_cols = euclidean_df.select_dtypes(include=[np.number]).columns
                st.dataframe(
                    euclidean_df.style.format("{:.2f}", subset=num_cols)
                    .background_gradient(subset=['Alpha'], cmap="RdBu", vmin=-2, vmax=2),
                    use_container_width=True
                )
                
                st.divider()
                st.subheader("🔮 Forecast Distribution Analysis (Euclidean Method)")
                dist_days = st.selectbox("Distribution Horizon (Days)", [2, 3, 5, 10, 21, 63, 126, 252], index=2)
                
                euc_data = get_euclidean_details(df, rank_cols, market_cols, n_neighbors=k_neighbors, target_days=dist_days)
                raw_returns = euc_data['Fwd Return'].tolist() if not euc_data.empty else []
                
                if raw_returns:
                    current_price = df['Close'].iloc[-1]
                    sim_prices = [current_price * (1 + (r / 100)) for r in raw_returns]
                    mean_price = np.mean(sim_prices)
                    std_price = np.std(sim_prices)
                    
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(
                        x=sim_prices, nbinsx=50, marker_color='lightgray', opacity=0.6, name='Simulated Prices'
                    ))
                    fig_dist.add_vline(x=current_price, line_width=3, line_color="black", annotation_text="Current")
                    fig_dist.add_vline(x=mean_price, line_width=3, line_dash="dot", line_color="blue", annotation_text="Mean")
                    
                    fig_dist.update_layout(
                        title=f"Projected Price Distribution ({dist_days} Days) | Based on {len(raw_returns)} Nearest Neighbors",
                        xaxis_title="Price", yaxis_title="Frequency", template="plotly_white", showlegend=False
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    st.subheader(f"📜 Historical Match Details ({dist_days} Days)")
                    tab_pairwise, tab_euc = st.tabs(["1. Pairwise Matches (Filtered Regime)", "2. Euclidean Matches (Raw Similarity)"])
                    
                    with tab_pairwise:
                        # Pairwise uses the FILTERED dataframe
                        match_df = get_detailed_match_table(df_filtered, rank_cols, market_cols, tolerance=ensemble_tol, target_days=dist_days)
                        if not match_df.empty:
                            # FIX: Explicit subset for formatting (excluding 'Trigger Pairs' string column) AND KEEP GRADIENT
                            num_cols = match_df.select_dtypes(include=[np.number]).columns
                            st.dataframe(match_df.style.format("{:.2f}", subset=num_cols).background_gradient(subset=['Fwd Return'], cmap="RdBu", vmin=-5, vmax=5), use_container_width=True)
                        else:
                            st.info("No pairwise matches found in this filtered regime.")
                            
                    with tab_euc:
                        # Euclidean uses the EUCLIDEAN data (which came from full df)
                        num_cols = euc_data.select_dtypes(include=[np.number]).columns
                        st.dataframe(euc_data.style.format("{:.2f}", subset=num_cols).background_gradient(subset=['Fwd Return'], cmap="RdBu", vmin=-5, vmax=5), use_container_width=True)
                    # ... existing code for Method 2 tables ...

            st.markdown("---")
            st.subheader("🧪 Walk-Forward Backtest (Information Coefficient)")
            st.write("This runs a strict walk-forward test. For every day in the past, it pretends it doesn't know the future, finds neighbors, and generates a prediction.")

            b_col1, b_col2, b_col3 = st.columns(3)
            with b_col1:
                bt_start_year = st.number_input("Backtest Start Year", 2010, 2024, 2018)
            with b_col2:
                bt_neighbors = st.number_input("Neighbors (k)", 10, 200, 50)
            with b_col3:
                bt_target = st.selectbox("Target Horizon", [5, 10, 21], index=0)

            if st.button("Run Backtest (Computationally Intensive)"):
                with st.spinner("Running Walk-Forward Simulation... this may take a moment..."):
                    bt_res = backtest_euclidean_model(df, rank_cols, market_cols, 
                                                      start_year=bt_start_year, 
                                                      n_neighbors=bt_neighbors, 
                                                      target_days=bt_target)
                    
                    if not bt_res.empty:
                        # 1. Overall Stats
                        ic_pearson, _ = pearsonr(bt_res['Predicted'], bt_res['Actual'])
                        ic_rank, _ = spearmanr(bt_res['Predicted'], bt_res['Actual'])
                        
                        st.success(f"Backtest Complete ({len(bt_res)} days)")
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Information Coeff (IC)", f"{ic_pearson:.3f}")
                        m2.metric("Rank IC", f"{ic_rank:.3f}")
                        
                        # Hit Rate: If we predicted +ve, was it +ve? If -ve, was it -ve?
                        same_sign = np.sign(bt_res['Predicted']) == np.sign(bt_res['Actual'])
                        hit_rate = same_sign.mean() * 100
                        m3.metric("Directional Hit Rate", f"{hit_rate:.1f}%")

                        # Win/Loss Ratio
                        # If we took the trade (Pred > 0), what was avg return?
                        long_trades = bt_res[bt_res['Predicted'] > 0]['Actual']
                        short_trades = bt_res[bt_res['Predicted'] < 0]['Actual'] # Assuming we short
                        avg_long = long_trades.mean() if len(long_trades) > 0 else 0
                        m4.metric("Avg Return when Pred > 0", f"{avg_long:.2f}%")

                        # 2. Rolling IC Chart
                        st.subheader(f"Rolling {bt_target}d Information Coefficient (252d Window)")
                        rolling_ic = bt_res['Predicted'].rolling(252).corr(bt_res['Actual'])
                        
                        fig_ic = go.Figure()
                        fig_ic.add_trace(go.Scatter(x=rolling_ic.index, y=rolling_ic, mode='lines', name='Rolling IC', line=dict(color='purple')))
                        fig_ic.add_hline(y=0, line_dash="dash", line_color="black")
                        fig_ic.add_hline(y=0.05, line_dash="dot", line_color="green", annotation_text="Good Signal (>0.05)")
                        fig_ic.update_layout(height=300, title="Signal Consistency over Time", yaxis_title="Correlation")
                        st.plotly_chart(fig_ic, use_container_width=True)

                        # 3. Scatter Plot
                        st.subheader("Prediction vs. Reality")
                        fig_scat = go.Figure(data=go.Scattergl(
                            x=bt_res['Predicted'],
                            y=bt_res['Actual'],
                            mode='markers',
                            marker=dict(color=bt_res['Actual'], colorscale='RdBu', opacity=0.5, size=5),
                            text=bt_res.index.strftime('%Y-%m-%d')
                        ))
                        fig_scat.update_layout(
                            xaxis_title=f"Predicted {bt_target}d Return",
                            yaxis_title=f"Actual {bt_target}d Return",
                            height=500,
                            title="Scatter: Do high predictions lead to high returns?"
                        )
                        st.plotly_chart(fig_scat, use_container_width=True)

                    else:
                        st.warning("Not enough data to run backtest.")

def main():
    st.set_page_config(layout="wide", page_title="Heatmap Analytics")
    render_heatmap()

if __name__ == "__main__":
    main()
