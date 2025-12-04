import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import matplotlib
import datetime
import pytz # Added for Time Zone conversion

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
CSV_PATH = "seasonal_ranks.csv" # Ensure this matches your exported filename

# --- SIDEBAR: REFRESH BUTTON ---
# This must be placed before data loading to effectively clear the cache before use
with st.sidebar:
    st.header("Data Control")
    st.write("If data looks stale (yesterday's date), click below:")
    if st.button("Force Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()

@st.cache_data(show_spinner=False)
def load_seasonal_map():
    """
    Loads seasonal ranks with EXACT DATE matching (YYYY-MM-DD).
    """
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        return {}

    if df.empty: return {}

    # Ensure Date is parsed as datetime objects
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])
    
    # Sort to ensure chronological order (optional but good practice)
    df = df.sort_values("Date")

    output_map = {}
    for ticker, group in df.groupby("ticker"):
        # Map: Timestamp -> Rank
        output_map[ticker] = pd.Series(
            group.seasonal_rank.values, index=group.Date
        ).to_dict()
    return output_map

def get_sznl_val_series(ticker, dates, sznl_map):
    """
    Looks up seasonal rank using EXACT DATES from the index.
    """
    t_map = sznl_map.get(ticker, {})
    if not t_map:
        return pd.Series(50.0, index=dates)
    
    # 1. Normalize the input index to Midnight (remove time component)
    # 2. Remove Timezone if present (to match CSV's naive dates)
    normalized_dates = pd.to_datetime(dates).normalize()
    if normalized_dates.tz is not None:
        normalized_dates = normalized_dates.tz_localize(None)

    # Strict Lookup: If date doesn't exist in CSV, return 50 (Neutral)
    return normalized_dates.map(t_map).fillna(50.0)

@st.cache_data(show_spinner=False)
def download_data(ticker):
    if not ticker: return pd.DataFrame()
    try:
        # Added auto_adjust=True to handle splits/dividends and get cleaner Close prices
        df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        
        # Ensure index is standard
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
            
        return df
    except:
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# COLOR GENERATOR (Seismic)
# -----------------------------------------------------------------------------
def get_seismic_colorscale():
    seismic = []
    cm = matplotlib.colormaps["seismic"] 
    for k in range(255):
        r, g, b, _ = cm(k / 254.0)
        seismic.append([k / 254.0, f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'])
    return seismic

# -----------------------------------------------------------------------------
# CALCULATION ENGINES
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=True)
def calc_target_metrics(df, sznl_map, ticker):
    """
    Calculates metrics for TICKER 1 (The Target).
    Used for X-Axis (Condition) and Z-Axis (Result).
    """
    df = df.copy()
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))

    # --- Z-AXIS: FORWARD TARGETS ---
    for w in [5, 10, 21, 63]:
        df[f'FwdRet_{w}d'] = (df['Close'].shift(-w) / df['Close'] - 1.0) * 100.0

    for w in [2, 5, 10, 21]:
        curr_vol = df['LogRet'].rolling(w).std() * np.sqrt(252) * 100.0
        fwd_vol = df['LogRet'].rolling(w).std().shift(-w) * np.sqrt(252) * 100.0
        df[f'FwdVolDelta_{w}d'] = fwd_vol - curr_vol

    # --- X-AXIS: INTERNAL METRICS ---
    # Trailing Returns
    for w in [5, 10, 21, 63, 126, 252]:
        df[f'Ret_{w}d'] = df['Close'].pct_change(w)

    # Realized Volatility
    for w in [2, 5, 10, 21, 63]:
        df[f'RealVol_{w}d'] = df['LogRet'].rolling(w).std() * np.sqrt(252) * 100.0
    
    df['VolChange_2d']  = df['RealVol_2d']  - df['RealVol_63d']
    df['VolChange_5d']  = df['RealVol_5d']  - df['RealVol_63d']
    df['VolChange_10d'] = df['RealVol_10d'] - df['RealVol_63d']
    df['VolChange_21d'] = df['RealVol_21d'] - df['RealVol_63d']

    for w in [5, 10, 21]:
        df[f'VolRatio_{w}d'] = df['Volume'].rolling(w).mean() / df['Volume'].rolling(63).mean()

    # --- UPDATED SEASONAL CALL ---
    df['Seasonal'] = get_sznl_val_series(ticker, df.index, sznl_map)

    # Rank Transformations (For X Axis usage)
    vars_to_rank = [
        'Ret_5d', 'Ret_10d', 'Ret_21d', 'Ret_63d', 'Ret_126d', 'Ret_252d',
        'RealVol_21d', 'RealVol_63d', 
        'VolChange_2d', 'VolChange_5d', 'VolChange_10d', 'VolChange_21d',
        'VolRatio_5d', 'VolRatio_10d', 'VolRatio_21d'
    ]
    
    for v in vars_to_rank:
        df[v + '_Rank'] = df[v].expanding(min_periods=252).rank(pct=True) * 100.0

    return df

@st.cache_data(show_spinner=True)
def calc_signal_metrics(df):
    """
    Calculates metrics for TICKER 2 (The Signal).
    Used ONLY for Y-Axis.
    """
    df = df.copy()
    
    # Calculate Returns and Ranks for the requested windows
    windows = [2, 5, 10, 21, 63, 126, 252]
    
    for w in windows:
        # Calculate raw return
        col_name = f'Ret_{w}d'
        df[col_name] = df['Close'].pct_change(w)
        
        # Calculate Percentile Rank (0-100)
        # Prefix with T2_ so we don't overwrite Ticker 1 data later
        rank_name = f'T2_{col_name}_Rank'
        df[rank_name] = df[col_name].expanding(min_periods=252).rank(pct=True) * 100.0
        
    return df

# -----------------------------------------------------------------------------
# HEATMAP UTILS (Binning & Smoothing)
# -----------------------------------------------------------------------------
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
    sum_vals   = convolve2d(np.nan_to_num(Z), kernel, mode='same', boundary='symm')
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
# UI: CROSS ASSET HEATMAP
# -----------------------------------------------------------------------------
def render_cross_asset_heatmap():
    st.header("Cross-Asset Correlation Heatmap")
    st.markdown("""
    Analyze how the historical performance of **Ticker 2** (Y-Axis) impacts the future performance of **Ticker 1** (Z-Axis).
    """)
    
    # 1. SELECTION
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Target (Analysis)")
        ticker1 = st.text_input("Target Ticker (T1)", value="SPY", key="ca_t1").upper()
        
        t1_options = {
            "Seasonality Rank": "Seasonal",
            "5d Trailing Return Rank": "Ret_5d_Rank",
            "21d Trailing Return Rank": "Ret_21d_Rank",
            "63d Trailing Return Rank": "Ret_63d_Rank",
            "21d Realized Vol Rank": "RealVol_21d_Rank",
            "Change in Vol (21d) Rank": "VolChange_21d_Rank",
        }
        x_axis_label = st.selectbox("X-Axis (Ticker 1 State)", list(t1_options.keys()), index=0, key="ca_x")

        target_options = {
            "5d Forward Return": "FwdRet_5d",
            "10d Forward Return": "FwdRet_10d",
            "21d Forward Return": "FwdRet_21d",
            "63d Forward Return": "FwdRet_63d",
            "21d Fwd Vol Change": "FwdVolDelta_21d",
        }
        z_axis_label = st.selectbox("Z-Axis (Target Outcome)", list(target_options.keys()), index=2, key="ca_z")

    with col2:
        st.subheader("2. Signal (Influence)")
        ticker2 = st.text_input("Signal Ticker (T2)", value="TLT", key="ca_t2").upper()
        
        t2_options = {
            f"{ticker2} 2d Return Rank": "T2_Ret_2d_Rank",
            f"{ticker2} 5d Return Rank": "T2_Ret_5d_Rank",
            f"{ticker2} 10d Return Rank": "T2_Ret_10d_Rank",
            f"{ticker2} 21d Return Rank": "T2_Ret_21d_Rank",
            f"{ticker2} 63d Return Rank": "T2_Ret_63d_Rank",
            f"{ticker2} 126d Return Rank": "T2_Ret_126d_Rank",
            f"{ticker2} 252d Return Rank": "T2_Ret_252d_Rank",
        }
        y_axis_label = st.selectbox("Y-Axis (Ticker 2 State)", list(t2_options.keys()), index=3, key="ca_y")

    with col3:
        st.subheader("3. Settings")
        smooth_sigma = st.slider("Smoothing", 0.5, 3.0, 1.2, 0.1, key="ca_smooth")
        bins = st.slider("Grid Resolution", 10, 50, 28, key="ca_bins")
        analysis_start = st.date_input("Start Date", value=datetime.date(2005, 1, 1), key="ca_start")
        
    st.markdown("---")

    if st.button("Generate Cross-Asset Matrix", type="primary", use_container_width=True):
        st.session_state['ca_run'] = True 
        
    if st.session_state.get('ca_run'):
        if ticker1 == ticker2:
            st.warning("Please choose two different tickers for cross-asset analysis.")
            return

        with st.spinner(f"Aligning Data: {ticker1} vs {ticker2}..."):
            
            # --- DATA PREP ---
            df1 = download_data(ticker1)
            df2 = download_data(ticker2)
            
            if df1.empty or df2.empty:
                st.error("Could not download data.")
                return

            sznl_map = load_seasonal_map()
            df1_calc = calc_target_metrics(df1, sznl_map, ticker1)
            df2_calc = calc_signal_metrics(df2)
            
            t2_cols_to_keep = [c for c in df2_calc.columns if c.startswith("T2_")]
            common_idx = df1_calc.index.intersection(df2_calc.index)
            
            final_df = df1_calc.loc[common_idx].copy()
            df2_subset = df2_calc.loc[common_idx, t2_cols_to_keep]
            final_df = final_df.join(df2_subset)
            
            start_ts = pd.to_datetime(analysis_start)
            if final_df.index.tz is not None: start_ts = start_ts.tz_localize(final_df.index.tz)
            final_df = final_df[final_df.index >= start_ts]

            xcol = t1_options[x_axis_label]
            ycol = t2_options[y_axis_label]
            zcol = target_options[z_axis_label]
            
            # --- HEATMAP GENERATION (Strictly requires Z) ---
            heatmap_df = final_df.dropna(subset=[xcol, ycol, zcol])
            
            if heatmap_df.empty:
                st.error("Insufficient data for heatmap.")
                return

            x_edges, y_edges = build_bins_quantile(heatmap_df[xcol], heatmap_df[ycol], nx=bins, ny=bins)
            x_centers, y_centers, Z = grid_mean(heatmap_df, xcol, ycol, zcol, x_edges, y_edges)
            
            Z_filled = nan_neighbor_fill(Z)
            Z_smooth = smooth_display(Z_filled, sigma=smooth_sigma)
            
            limit = np.nanmax(np.abs(Z_smooth))
            colorscale = get_seismic_colorscale()
            z_units = "%" if "Ret" in zcol else " Vol"

            # --- TIMESTAMP LOGIC ---
            # Get the exact timestamp of the last data point
            last_ts = final_df.index[-1]
            
            # Convert to US/Eastern if aware, otherwise just format YYYY-MM-DD
            tz_eastern = pytz.timezone('US/Eastern')
            
            if last_ts.tzinfo is not None:
                # If data is timezone aware (e.g. intraday or localized daily), convert to EST
                date_str = last_ts.astimezone(tz_eastern).strftime('%Y-%m-%d %H:%M %Z')
            else:
                # If naive (common for daily data), treat as date only
                date_str = last_ts.strftime('%Y-%m-%d')

            fig = go.Figure(data=go.Heatmap(
                z=Z_smooth, x=x_centers, y=y_centers,
                colorscale=colorscale, 
                zmin=-limit, zmax=limit, zmid=0,
                reversescale=True, 
                colorbar=dict(title=f"{zcol} {z_units}"),
                hovertemplate=f"<b>{ticker1} {x_axis_label}</b>: %{{x:.1f}}<br><b>{ticker2} Return Rank</b>: %{{y:.1f}}<br><b>{ticker1} Outcome</b>: %{{z:.2f}}{z_units}<extra></extra>"
            ))
            
            fig.update_layout(
                title=f"Impact of {ticker2} on {ticker1} Forward Returns",
                xaxis_title=f"{ticker1}: {x_axis_label} (0-100)",
                yaxis_title=f"{ticker2}: {y_axis_label.split(ticker2)[-1].strip()} (0-100)",
                height=650, template="plotly_white",
                dragmode=False 
            )
            
            # Add Current Status Marker (from most recent available data row)
            last_row = final_df.iloc[-1]
            curr_x = last_row.get(xcol, np.nan)
            curr_y = last_row.get(ycol, np.nan)
            
            if not np.isnan(curr_x) and not np.isnan(curr_y):
                fig.add_vline(x=curr_x, line_width=2, line_dash="dash", line_color="black")
                fig.add_hline(y=curr_y, line_width=2, line_dash="dash", line_color="black")
                fig.add_annotation(
                    x=curr_x, y=curr_y, 
                    text=f"<b>Current ({date_str})</b><br>{ticker1}: {curr_x:.0f}<br>{ticker2}: {curr_y:.0f}", 
                    showarrow=True, arrowhead=1, ax=40, ay=-40, bgcolor="white",
                    bordercolor="black", borderwidth=1
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 8. HISTORICAL EXPLORER (UPDATED: Includes recent dates)
            st.markdown(f"### 🔎 {ticker1} vs {ticker2} Historical Explorer")
            f_col1, f_col2 = st.columns(2)
            with f_col1: x_range = st.slider(f"Filter: {ticker1} {x_axis_label}", 0.0, 100.0, (0.0, 100.0), step=1.0)
            with f_col2: y_range = st.slider(f"Filter: {ticker2} Return Rank", 0.0, 100.0, (0.0, 100.0), step=1.0)
            
            # Use final_df (includes Z=NaN)
            table_df = final_df.dropna(subset=[xcol, ycol]).copy()
            
            mask = (table_df[xcol] >= x_range[0]) & (table_df[xcol] <= x_range[1]) & \
                   (table_df[ycol] >= y_range[0]) & (table_df[ycol] <= y_range[1])
            filtered_df = table_df[mask].copy()
            
            if not filtered_df.empty:
                # Separate stats calculation (ignoring NaNs for stats)
                completed_trades = filtered_df.dropna(subset=[zcol])
                
                s1, s2, s3 = st.columns(3)
                s1.metric("Matching Instances", len(filtered_df), help="Includes pending trades")
                
                if not completed_trades.empty:
                    s2.metric(f"Avg {ticker1} Return", f"{completed_trades[zcol].mean():.2f}{z_units}")
                    win_rate = (completed_trades[zcol] > 0).sum() / len(completed_trades) * 100
                    s3.metric(f"{ticker1} Win Rate", f"{win_rate:.1f}%")
                else:
                    s2.metric("Avg Return", "N/A")
                    s3.metric("Win Rate", "N/A")
                
                st.write("#### Matching Dates (Including Pending)")
                
                # Format Data for Table
                display_cols = ['Close', xcol, ycol, zcol]
                table_display = filtered_df[display_cols].sort_index(ascending=False)
                table_display.columns = [f"{ticker1} Price", x_axis_label, f"{ticker2} Rank", "Outcome"]
                
                st.dataframe(
                    table_display.style.format({
                        f"{ticker1} Price": "${:.2f}", 
                        x_axis_label: "{:.1f}", 
                        f"{ticker2} Rank": "{:.1f}", 
                        "Outcome": "{:.2f}" + z_units
                    }).background_gradient(subset=["Outcome"], cmap="RdBu", vmin=-limit, vmax=limit),
                    use_container_width=True, height=400
                )
            else:
                st.warning("No matches found in this region.")
                
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Cross Asset Analysis")
    render_cross_asset_heatmap()
