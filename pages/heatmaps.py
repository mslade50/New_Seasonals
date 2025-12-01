import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import matplotlib
import datetime
import itertools

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
SEASONAL_PATH = "seasonal_ranks.csv"
METRICS_PATH = "market_metrics_full_export.csv"

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
    
    # Only keep the metrics strictly requested
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

# -----------------------------------------------------------------------------
# FEATURE ENGINEERING & ENSEMBLE LOGIC
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def calculate_heatmap_variables(df, sznl_map, market_metrics_df, ticker):
    df = df.copy()
    
    # 1. BASE
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))

    # 2. FORWARD TARGETS
    for w in [5, 10, 21, 63, 252]:
        df[f'FwdRet_{w}d'] = (df['Close'].shift(-w) / df['Close'] - 1.0) * 100.0

    # 3. PREDICTOR VARIABLES
    
    # Momentum (Removed 63d, Kept others)
    for w in [5, 10, 21, 252]:
        df[f'Ret_{w}d'] = df['Close'].pct_change(w)

    # Realized Vol
    for w in [21, 63]:
        df[f'RealVol_{w}d'] = df['LogRet'].rolling(w).std() * np.sqrt(252) * 100.0
    
    # Relative Volume (Added 10d back)
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

    return df, rank_cols

def calculate_distribution_ensemble(df, rank_cols, market_cols):
    """
    "True Distribution" Approach:
    1. Identifies all valid feature pairs.
    2. For each pair, finds ALL historical rows where both ranks are within +/- 5 of current.
    3. Pools all these raw return outcomes into a single distribution (Bag of Returns).
    4. Calculates stats on this pooled distribution.
    5. Compares against the Baseline (Global Average) to find Alpha.
    """
    if df.empty: return pd.DataFrame()
    
    current_row = df.iloc[-1]
    
    # Combine internal ranks + seasonal + market ranks
    all_features = rank_cols + ['Seasonal'] + market_cols
    # Only use features that exist in DF and have valid current data
    valid_features = [f for f in all_features if f in df.columns and not pd.isna(current_row[f])]
    
    if len(valid_features) < 2: return pd.DataFrame()
    
    targets = [5, 10, 21, 63, 252]
    # Initialize the "Bag of Returns"
    pooled_outcomes = {t: [] for t in targets}
    
    pairs = list(itertools.combinations(valid_features, 2))
    
    # STRICT TOLERANCE
    TOLERANCE = 5.0 
    
    for f1, f2 in pairs:
        v1 = current_row[f1]
        v2 = current_row[f2]
        
        # Create mask for history: Box Filter +/- 5
        mask = (
            (df[f1] >= v1 - TOLERANCE) & (df[f1] <= v1 + TOLERANCE) &
            (df[f2] >= v2 - TOLERANCE) & (df[f2] <= v2 + TOLERANCE)
        )
        
        subset = df[mask]
        
        if subset.empty: continue
        
        for t in targets:
            col = f'FwdRet_{t}d'
            if col not in subset.columns: continue
            
            # Extract raw returns and add to the pool
            vals = subset[col].dropna().tolist()
            if vals:
                pooled_outcomes[t].extend(vals)
            
    # Aggregate Statistics
    summary = []
    for t in targets:
        col = f'FwdRet_{t}d'
        
        # Calculate Global Baseline (Full history mean for this ticker)
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
                "Win Rate": np.nan, 
                "Sample Size": 0
            })
            continue
            
        grand_mean = np.mean(data)
        std_dev = np.std(data)
        pos_ratio = np.sum(data > 0) / len(data)
        
        # Calculate Alpha
        alpha = grand_mean - baseline if not np.isnan(baseline) else np.nan
        
        summary.append({
            "Horizon": f"{t} Days",
            "Exp Return": grand_mean,
            "Baseline": baseline,
            "Alpha": alpha,
            "Std Dev": std_dev,
            "Win Rate": pos_ratio * 100,
            "Sample Size": len(data)
        })
        
    return pd.DataFrame(summary)

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
# UI: RENDER
# -----------------------------------------------------------------------------
def render_heatmap():
    st.subheader("Heatmap Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # STRIPPED PREDICTOR OPTIONS
        var_options = {
            "Seasonality Rank": "Seasonal",
            # Price
            "5d Trailing Return Rank": "Ret_5d_Rank",
            "10d Trailing Return Rank": "Ret_10d_Rank",
            "21d Trailing Return Rank": "Ret_21d_Rank",
            # "63d Trailing Return Rank": "Ret_63d_Rank",  <-- REMOVED
            "252d Trailing Return Rank": "Ret_252d_Rank",
            # Vol
            "21d Realized Vol Rank": "RealVol_21d_Rank",
            "63d Realized Vol Rank": "RealVol_63d_Rank",
            # Volume
            "10d Rel. Volume Rank": "VolRatio_10d_Rank", # <-- ADDED
            "21d Rel. Volume Rank": "VolRatio_21d_Rank",
        }
        
        var_options["Total Net Highs (5d MA) Rank"] = "Mkt_Total_NH_5d_Rank"
        var_options["Total Net Highs (21d MA) Rank"] = "Mkt_Total_NH_21d_Rank"

        x_axis_label = st.selectbox("X-Axis Variable", list(var_options.keys()), index=0, key="hm_x")
        y_axis_label = st.selectbox("Y-Axis Variable", list(var_options.keys()), index=2, key="hm_y")
    
    with col2:
        # STRIPPED TARGET OPTIONS
        target_options = {
            "5d Forward Return": "FwdRet_5d",
            "10d Forward Return": "FwdRet_10d",
            "21d Forward Return": "FwdRet_21d",
            "63d Forward Return": "FwdRet_63d",
            "252d Forward Return": "FwdRet_252d",
        }
        z_axis_label = st.selectbox("Target (Z-Axis)", list(target_options.keys()), index=2, key="hm_z")
        ticker = st.text_input("Ticker", value="SPY", key="hm_ticker").upper()
    
    with col3:
        smooth_sigma = st.slider("Smoothing (Sigma)", 0.5, 3.0, 1.2, 0.1, key="hm_smooth")
        bins = st.slider("Grid Resolution (Bins)", 10, 50, 28, key="hm_bins")
        analysis_start = st.date_input("Analysis Start Date", value=datetime.date(2000, 1, 1), key="hm_start_date")
        
    st.markdown("---")

    if st.button("Generate Heatmap & Ensemble", type="primary", use_container_width=True, key="hm_gen"):
        st.session_state['hm_data'] = True 
        
    if st.session_state.get('hm_data'):
        with st.spinner(f"Processing {ticker}..."):
            
            data, fetch_time = download_data(ticker)
            if data.empty:
                st.error("No data found.")
                return

            last_dt = data.index[-1]
            time_str = fetch_time.strftime('%I:%M %p EST') if fetch_time else "N/A"
            st.info(f"Price Data Current as of: {last_dt.strftime('%Y-%m-%d')} | Last Update: {time_str}")
                
            sznl_map = load_seasonal_map()
            market_metrics_df = load_market_metrics()
            
            # --- MAIN CALCULATION ---
            df, rank_cols = calculate_heatmap_variables(data, sznl_map, market_metrics_df, ticker)
            
            start_ts = pd.to_datetime(analysis_start)
            if df.index.tz is not None: start_ts = start_ts.tz_localize(df.index.tz)
            
            df_filtered = df[df.index >= start_ts]
            
            if df_filtered.empty:
                st.error("No data found after start date.")
                return
            
            # --- HEATMAP RENDER ---
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
                
                fig.update_layout(
                    title=f"{ticker} ({analysis_start} - Present): {z_axis_label}",
                    xaxis_title=x_axis_label + " (0-100)",
                    yaxis_title=y_axis_label + " (0-100)",
                    height=600, template="plotly_white",
                    dragmode=False 
                )
                
                last_row = df.iloc[-1]
                curr_x = last_row.get(xcol, np.nan)
                curr_y = last_row.get(ycol, np.nan)
                if not np.isnan(curr_x) and not np.isnan(curr_y):
                    fig.add_vline(x=curr_x, line_width=2, line_dash="dash", line_color="black")
                    fig.add_hline(y=curr_y, line_width=2, line_dash="dash", line_color="black")
                    fig.add_annotation(x=curr_x, y=curr_y, text="Current", showarrow=True, arrowhead=1, ax=30, ay=-30, bgcolor="white")
                
                st.plotly_chart(fig, use_container_width=True, key="heatmap_plot")
                
                # --- HISTORICAL DATA EXPLORER ---
                st.divider()
                st.markdown("### 🔎 Historical Data Explorer")
                f_col1, f_col2 = st.columns(2)
                with f_col1: x_range = st.slider(f"Filter: {x_axis_label} (Rank)", 0.0, 100.0, (0.0, 100.0), step=1.0)
                with f_col2: y_range = st.slider(f"Filter: {y_axis_label} (Rank)", 0.0, 100.0, (0.0, 100.0), step=1.0)
                
                mask = (clean_df[xcol] >= x_range[0]) & (clean_df[xcol] <= x_range[1]) & \
                       (clean_df[ycol] >= y_range[0]) & (clean_df[ycol] <= y_range[1])
                filtered_df = clean_df[mask].copy()
                
                if not filtered_df.empty:
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Matching Instances", len(filtered_df))
                    s2.metric(f"Avg Target", f"{filtered_df[zcol].mean():.2f}{z_units}")
                    
                    win_rate = (filtered_df[zcol] > 0).sum() / len(filtered_df) * 100
                    s3.metric("Win Rate (>0)", f"{win_rate:.1f}%")
                    
                    st.write("#### Matching Dates")
                    display_cols = ['Close', xcol, ycol, zcol]
                    table_df = filtered_df[display_cols].sort_index(ascending=False)
                    st.dataframe(
                        table_df.style.format({
                            "Close": "${:.2f}", xcol: "{:.1f}", ycol: "{:.1f}", zcol: "{:.2f}" + z_units
                        }).background_gradient(subset=[zcol], cmap="RdBu", vmin=-limit, vmax=limit),
                        use_container_width=True, height=400
                    )
                else:
                    st.warning("No matches found for these filters.")
                    
            else:
                st.error("Insufficient data for Heatmap.")

            # --- ENSEMBLE TABLE ---
            st.divider()
            st.subheader(f"🤖 Grand Ensemble Forecast (True Distribution Mode)")
            st.markdown(f"""
            **Methodology:**
            1. Scans ALL valid variable pairs (e.g., Seasonality vs Momentum, Volatility vs Net Highs).
            2. For every pair, finds ALL historical dates where **both** metrics were within **±5** rank points of today.
            3. Pools ALL those historical return outcomes into a single distribution to calculate expected value.
            4. **Alpha** is calculated by comparing the Ensemble Expectation vs. The Ticker's Global Average for that horizon.
            """)
            
            market_cols = [c for c in df.columns if c.startswith("Mkt_")]
            ensemble_df = calculate_distribution_ensemble(df, rank_cols, market_cols)
            
            if not ensemble_df.empty:
                st.dataframe(
                    ensemble_df.style.format({
                        "Exp Return": "{:.2f}%",
                        "Baseline": "{:.2f}%",
                        "Alpha": "{:+.2f}%",
                        "Std Dev": "{:.2f}",
                        "Win Rate": "{:.1f}%",
                        "Sample Size": "{:,.0f}"
                    }).background_gradient(subset=['Alpha'], cmap="RdBu", vmin=-2, vmax=2),
                    use_container_width=True
                )
            else:
                st.warning("Not enough historical depth (matches within ±5) to generate an ensemble forecast.")

def main():
    st.set_page_config(layout="wide", page_title="Heatmap Analytics")
    render_heatmap()

if __name__ == "__main__":
    main()
