import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import matplotlib
import datetime

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
    """
    Loads market metrics, calculates Moving Averages of Net Highs, 
    and then converts them to Percentile Ranks.
    """
    try:
        df = pd.read_csv(METRICS_PATH)
        # Handle date parsing flexibly
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
    except Exception:
        return pd.DataFrame()

    if df.empty: return pd.DataFrame()

    # 1. Pivot to get Date index and Exchange columns
    # We need to handle potential duplicates or just standard pivoting
    pivoted = df.pivot_table(
        index='date', 
        columns='exchange', 
        values='net_new_highs', 
        aggfunc='sum'
    )
    
    # 2. Create 'Total' (NYSE + NASDAQ)
    # Fill NaN with 0 for addition, though usually data aligns
    pivoted['Total'] = pivoted.get('NYSE', 0) + pivoted.get('NASDAQ', 0)
    
    # 3. Calculate MAs and Ranks
    # We want the Percentile Rank of the Trailing X Day MA of Net Highs
    windows = [5, 10, 21, 63, 252]
    categories = ['Total', 'NYSE', 'NASDAQ']
    
    results = pd.DataFrame(index=pivoted.index)
    
    for cat in categories:
        if cat not in pivoted.columns: 
            continue
            
        series = pivoted[cat]
        
        for w in windows:
            # A. Calculate Moving Average
            ma_col = series.rolling(window=w).mean()
            
            # B. Calculate Percentile Rank (0-100)
            # using expanding window so we don't look ahead
            rank_col_name = f"Mkt_{cat}_NH_{w}d_Rank"
            results[rank_col_name] = ma_col.expanding(min_periods=126).rank(pct=True) * 100.0

    return results

def get_sznl_val_series(ticker, dates, sznl_map):
    t_map = sznl_map.get(ticker, {})
    if not t_map:
        return pd.Series(50.0, index=dates)
    
    mds = dates.map(lambda x: (x.month, x.day))
    return mds.map(t_map).fillna(50.0)

@st.cache_data(show_spinner=True)
def download_data(ticker):
    if not ticker: return pd.DataFrame()
    try:
        # We need MAX history for accurate percentile ranking
        df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df
    except:
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# COLOR GENERATOR (Seismic)
# -----------------------------------------------------------------------------
def get_seismic_colorscale():
    """Generates the seismic color scale: Dark Blue (Pos) <-> White (0) <-> Dark Red (Neg)."""
    seismic = []
    # Use Matplotlib to get the exact gradient
    cm = matplotlib.colormaps["seismic"] 
    for k in range(255):
        r, g, b, _ = cm(k / 254.0)
        seismic.append([k / 254.0, f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'])
    return seismic

# -----------------------------------------------------------------------------
# CALCULATION ENGINE (Cached)
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=True)
def calculate_heatmap_variables(df, sznl_map, market_metrics_df, ticker):
    """Calculates advanced vars and ranks them 0-100 for the heatmap axes."""
    df = df.copy()
    
    # --- 1. BASE CALCULATIONS ---
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))

    # --- 2. FORWARD TARGETS (Z-AXIS) ---
    
    # A. Forward Returns
    for w in [5, 10, 21, 63]:
        df[f'FwdRet_{w}d'] = (df['Close'].shift(-w) / df['Close'] - 1.0) * 100.0

    # B. Forward Volatility DELTA (Expansion/Contraction)
    # Logic: Future Volatility - Current Volatility
    for w in [2, 5, 10, 21]:
        # Current Realized Vol
        curr_vol = df['LogRet'].rolling(w).std() * np.sqrt(252) * 100.0
        # Future Realized Vol (Shifted back)
        fwd_vol = df['LogRet'].rolling(w).std().shift(-w) * np.sqrt(252) * 100.0
        
        df[f'FwdVolDelta_{w}d'] = fwd_vol - curr_vol

    # --- 3. X/Y VARIABLES (Raw Calculation) ---
    
    # Trailing Returns
    for w in [5, 10, 21, 63, 126, 252]:
        df[f'Ret_{w}d'] = df['Close'].pct_change(w)

    # Realized Volatility (Annualized)
    for w in [2, 5, 10, 21, 63]:
        df[f'RealVol_{w}d'] = df['LogRet'].rolling(w).std() * np.sqrt(252) * 100.0
    
    # Change in Realized Vol (Short Term vs 63d Baseline)
    df['VolChange_2d']  = df['RealVol_2d']  - df['RealVol_63d']
    df['VolChange_5d']  = df['RealVol_5d']  - df['RealVol_63d']
    df['VolChange_10d'] = df['RealVol_10d'] - df['RealVol_63d']
    df['VolChange_21d'] = df['RealVol_21d'] - df['RealVol_63d']

    # Volume Ratios
    for w in [5, 10, 21]:
        df[f'VolRatio_{w}d'] = df['Volume'].rolling(w).mean() / df['Volume'].rolling(63).mean()

    # Seasonality
    df['Seasonal'] = get_sznl_val_series(ticker, df.index, sznl_map)

    # --- 4. RANK TRANSFORMATION (Percentiles 0-100) ---
    vars_to_rank = [
        'Ret_5d', 'Ret_10d', 'Ret_21d', 'Ret_63d', 'Ret_126d', 'Ret_252d',
        'RealVol_21d', 'RealVol_63d', 
        'VolChange_2d', 'VolChange_5d', 'VolChange_10d', 'VolChange_21d',
        'VolRatio_5d', 'VolRatio_10d', 'VolRatio_21d'
    ]
    
    for v in vars_to_rank:
        df[v + '_Rank'] = df[v].expanding(min_periods=252).rank(pct=True) * 100.0

    # --- 5. MERGE MARKET METRICS ---
    # Merge the pre-calculated market ranks into the main dataframe
    if not market_metrics_df.empty:
        # We use left join to keep stock data shape
        df = df.join(market_metrics_df, how='left')
        # Optional: Forward fill if market data has gaps (e.g. holidays differ slightly)
        df.update(df.filter(regex='^Mkt_').ffill(limit=3))

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
# UI: RENDER HEATMAP
# -----------------------------------------------------------------------------
def render_heatmap():
    st.subheader("Heatmap Analysis")
    
    # 1. SELECTION
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Build options dictionary
        var_options = {
            "Seasonality Rank": "Seasonal",
            # Price
            "5d Trailing Return Rank": "Ret_5d_Rank",
            "10d Trailing Return Rank": "Ret_10d_Rank",
            "21d Trailing Return Rank": "Ret_21d_Rank",
            "63d Trailing Return Rank": "Ret_63d_Rank",
            "126d Trailing Return Rank": "Ret_126d_Rank",
            "252d Trailing Return Rank": "Ret_252d_Rank",
            # Vol
            "21d Realized Vol Rank": "RealVol_21d_Rank",
            "63d Realized Vol Rank": "RealVol_63d_Rank",
            "Change in Vol (2d-63d) Rank": "VolChange_2d_Rank",
            "Change in Vol (5d-63d) Rank": "VolChange_5d_Rank",
            "Change in Vol (10d-63d) Rank": "VolChange_10d_Rank",
            "Change in Vol (21d-63d) Rank": "VolChange_21d_Rank",
            # Volume
            "5d Rel. Volume Rank": "VolRatio_5d_Rank",
            "10d Rel. Volume Rank": "VolRatio_10d_Rank",
            "21d Rel. Volume Rank": "VolRatio_21d_Rank",
        }
        
        # Add MARKET METRICS to options
        # Categories: Total, NYSE, NASDAQ
        # Windows: 5, 10, 21, 63, 252
        for cat in ['Total', 'NYSE', 'NASDAQ']:
            for w in [5, 10, 21, 63, 252]:
                label = f"{cat} Net Highs ({w}d MA) Rank"
                key = f"Mkt_{cat}_NH_{w}d_Rank"
                var_options[label] = key

        x_axis_label = st.selectbox("X-Axis Variable", list(var_options.keys()), index=0, key="hm_x")
        y_axis_label = st.selectbox("Y-Axis Variable", list(var_options.keys()), index=3, key="hm_y")
    
    with col2:
        target_options = {
            "5d Forward Return": "FwdRet_5d",
            "10d Forward Return": "FwdRet_10d",
            "21d Forward Return": "FwdRet_21d",
            "63d Forward Return": "FwdRet_63d",
            # DELTA METRICS (Centered at 0)
            "2d Fwd Vol Change (Expansion/Contraction)": "FwdVolDelta_2d",
            "5d Fwd Vol Change (Expansion/Contraction)": "FwdVolDelta_5d",
            "10d Fwd Vol Change (Expansion/Contraction)": "FwdVolDelta_10d",
            "21d Fwd Vol Change (Expansion/Contraction)": "FwdVolDelta_21d",
        }
        z_axis_label = st.selectbox("Target (Z-Axis)", list(target_options.keys()), index=2, key="hm_z")
        ticker = st.text_input("Ticker", value="SPY", key="hm_ticker").upper()
    
    with col3:
        smooth_sigma = st.slider("Smoothing (Sigma)", 0.5, 3.0, 1.2, 0.1, key="hm_smooth")
        bins = st.slider("Grid Resolution (Bins)", 10, 50, 28, key="hm_bins")
        analysis_start = st.date_input("Analysis Start Date", value=datetime.date(2000, 1, 1), key="hm_start_date")
        
    st.markdown("---")

    if st.button("Generate Heatmap", type="primary", use_container_width=True, key="hm_gen"):
        st.session_state['hm_data'] = True 
        if 'heatmap_selection' in st.session_state: del st.session_state['heatmap_selection']
        
    if st.session_state.get('hm_data'):
        with st.spinner(f"Processing {ticker}..."):
            
            data = download_data(ticker)
            if data.empty:
                st.error("No data found.")
                return
                
            sznl_map = load_seasonal_map()
            
            # LOAD MARKET METRICS
            market_metrics_df = load_market_metrics()
            
            df = calculate_heatmap_variables(data, sznl_map, market_metrics_df, ticker)
            
            start_ts = pd.to_datetime(analysis_start)
            if df.index.tz is not None: start_ts = start_ts.tz_localize(df.index.tz)
            df = df[df.index >= start_ts]
            
            if df.empty:
                st.error("No data found after start date.")
                return
            
            xcol = var_options[x_axis_label]
            ycol = var_options[y_axis_label]
            zcol = target_options[z_axis_label]
            
            # Check if columns exist (safe guard for missing market metrics)
            if xcol not in df.columns or ycol not in df.columns:
                st.error(f"Missing data for selected variables. Ensure market metrics CSV is present.")
                return

            clean_df = df.dropna(subset=[xcol, ycol, zcol])
            if clean_df.empty:
                st.error("Insufficient data.")
                return

            x_edges, y_edges = build_bins_quantile(clean_df[xcol], clean_df[ycol], nx=bins, ny=bins)
            x_centers, y_centers, Z = grid_mean(clean_df, xcol, ycol, zcol, x_edges, y_edges)
            
            Z_filled = nan_neighbor_fill(Z)
            Z_smooth = smooth_display(Z_filled, sigma=smooth_sigma)
            
            # --- SCALING ---
            # All Z targets are now Deltas/Returns, so they oscillate around 0.
            limit = np.nanmax(np.abs(Z_smooth))
            colorscale = get_seismic_colorscale()
            z_units = "%" if "Ret" in zcol else " Vol"

            fig = go.Figure(data=go.Heatmap(
                z=Z_smooth, x=x_centers, y=y_centers,
                colorscale=colorscale, 
                zmin=-limit, zmax=limit,
                zmid=0, # Force center at 0
                reversescale=True, 
                colorbar=dict(title=f"{zcol} {z_units}"),
                hovertemplate=f"<b>{x_axis_label}</b>: %{{x:.1f}}<br><b>{y_axis_label}</b>: %{{y:.1f}}<br><b>Value</b>: %{{z:.2f}}{z_units}<extra></extra>"
            ))
            
            fig.update_layout(
                title=f"{ticker} ({analysis_start} - Present): {z_axis_label}",
                xaxis_title=x_axis_label + " (0-100)",
                yaxis_title=y_axis_label + " (0-100)",
                height=650, template="plotly_white",
                dragmode=False 
            )
            
            last_row = df.iloc[-1]
            curr_x = last_row.get(xcol, np.nan)
            curr_y = last_row.get(ycol, np.nan)
            if not np.isnan(curr_x) and not np.isnan(curr_y):
                fig.add_vline(x=curr_x, line_width=2, line_dash="dash", line_color="black")
                fig.add_hline(y=curr_y, line_width=2, line_dash="dash", line_color="black")
                fig.add_annotation(x=curr_x, y=curr_y, text="Current", showarrow=True, arrowhead=1, ax=30, ay=-30, bgcolor="white")
            
            st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="heatmap_plot")
            
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
                st.warning("No matches found.")

def main():
    st.set_page_config(layout="wide", page_title="Heatmap Analytics")
    render_heatmap()

if __name__ == "__main__":
    main()
