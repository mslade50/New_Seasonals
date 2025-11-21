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
CSV_PATH = "seasonal_ranks.csv"

@st.cache_data(show_spinner=False)
def load_seasonal_map():
    try:
        df = pd.read_csv(CSV_PATH)
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
# CALCULATION ENGINE
# -----------------------------------------------------------------------------

def calculate_heatmap_variables(df, sznl_map, ticker):
    """Calculates advanced vars and ranks them 0-100 for the heatmap axes."""
    df = df.copy()
    
    # --- FORWARD RETURNS (Z-AXIS) ---
    # Logic: (Future Price - Current Price) / Current Price
    for w in [5, 10, 21, 63]:
        df[f'FwdRet_{w}d'] = (df['Close'].shift(-w) / df['Close'] - 1.0) * 100.0

    # --- X/Y VARIABLES (Raw Calculation) ---
    
    # 1. Trailing Returns
    for w in [5, 10, 21, 63, 126, 252]:
        df[f'Ret_{w}d'] = df['Close'].pct_change(w)

    # 2. Realized Volatility (Annualized)
    # Log Returns for Vol calc
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Vol Windows: 2d, 5d, 10d, 21d, 63d
    for w in [2, 5, 10, 21, 63]:
        df[f'RealVol_{w}d'] = df['LogRet'].rolling(w).std() * np.sqrt(252) * 100.0
    
    # 3. Change in Realized Vol (Short Term vs 63d Baseline)
    # Positive = Vol expanding vs trend. Negative = Vol contracting vs trend.
    df['VolChange_2d']  = df['RealVol_2d']  - df['RealVol_63d']
    df['VolChange_5d']  = df['RealVol_5d']  - df['RealVol_63d']
    df['VolChange_10d'] = df['RealVol_10d'] - df['RealVol_63d']
    df['VolChange_21d'] = df['RealVol_21d'] - df['RealVol_63d']

    # 4. Volume Ratios (Relative Volume)
    for w in [5, 10, 21]:
        # Volume relative to 63d baseline
        df[f'VolRatio_{w}d'] = df['Volume'].rolling(w).mean() / df['Volume'].rolling(63).mean()

    # 5. Seasonality
    df['Seasonal'] = get_sznl_val_series(ticker, df.index, sznl_map)

    # --- RANK TRANSFORMATION (Percentiles 0-100) ---
    # We use Expanding Rank (min 252) to normalize everything to 0-100 scale robustly.
    vars_to_rank = [
        'Ret_5d', 'Ret_10d', 'Ret_21d', 'Ret_63d', 'Ret_126d', 'Ret_252d',
        'RealVol_21d', 'RealVol_63d', 
        'VolChange_2d', 'VolChange_5d', 'VolChange_10d', 'VolChange_21d',
        'VolRatio_5d', 'VolRatio_10d', 'VolRatio_21d'
    ]
    
    for v in vars_to_rank:
        df[v + '_Rank'] = df[v].expanding(min_periods=252).rank(pct=True) * 100.0

    return df

# -----------------------------------------------------------------------------
# HEATMAP UTILS (Binning & Smoothing)
# -----------------------------------------------------------------------------

def build_bins_quantile(x, y, nx=30, ny=30):
    """Quantile-based edges to minimize empty cells."""
    x = pd.Series(x, dtype=float).dropna()
    y = pd.Series(y, dtype=float).dropna()

    if len(x) < 10 or len(y) < 10:
        return np.linspace(0,100,nx+1), np.linspace(0,100,ny+1)

    x_edges = np.unique(np.quantile(x, np.linspace(0, 1, nx + 1)))
    y_edges = np.unique(np.quantile(y, np.linspace(0, 1, ny + 1)))

    # Fallback if data is too discrete (ties in quantiles)
    if len(x_edges) < 3: x_edges = np.linspace(x.min(), x.max(), max(3, nx + 1))
    if len(y_edges) < 3: y_edges = np.linspace(y.min(), y.max(), max(3, ny + 1))
    
    return x_edges, y_edges

def grid_mean(df_sub, xcol, ycol, zcol, x_edges, y_edges):
    """Aggregates Z values into the X/Y grid."""
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

    sums   = np.zeros((ny, nx), float)
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
    """Fills small holes in the grid to prevent smoothing artifacts."""
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
    """Mask-aware Gaussian smoothing using Scipy."""
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
        # Map Display Name -> DataFrame Column
        var_options = {
            "Seasonality Rank": "Seasonal",
            "5d Trailing Return Rank": "Ret_5d_Rank",
            "10d Trailing Return Rank": "Ret_10d_Rank",
            "21d Trailing Return Rank": "Ret_21d_Rank",
            "63d Trailing Return Rank": "Ret_63d_Rank",
            "126d Trailing Return Rank": "Ret_126d_Rank",
            "252d Trailing Return Rank": "Ret_252d_Rank",
            "21d Realized Vol Rank": "RealVol_21d_Rank",
            "63d Realized Vol Rank": "RealVol_63d_Rank",
            "Change in Vol (2d-63d) Rank": "VolChange_2d_Rank",
            "Change in Vol (5d-63d) Rank": "VolChange_5d_Rank",
            "Change in Vol (10d-63d) Rank": "VolChange_10d_Rank",
            "Change in Vol (21d-63d) Rank": "VolChange_21d_Rank",
            "5d Rel. Volume Rank": "VolRatio_5d_Rank",
            "10d Rel. Volume Rank": "VolRatio_10d_Rank",
            "21d Rel. Volume Rank": "VolRatio_21d_Rank"
        }
        
        x_axis_label = st.selectbox("X-Axis Variable", list(var_options.keys()), index=0)
        y_axis_label = st.selectbox("Y-Axis Variable", list(var_options.keys()), index=3)
    
    with col2:
        target_options = {
            "5d Forward Return": "FwdRet_5d",
            "10d Forward Return": "FwdRet_10d",
            "21d Forward Return": "FwdRet_21d",
            "63d Forward Return": "FwdRet_63d",
        }
        z_axis_label = st.selectbox("Target (Z-Axis)", list(target_options.keys()), index=2)
        ticker = st.text_input("Ticker", value="SPY").upper()
    
    with col3:
        smooth_sigma = st.slider("Smoothing (Sigma)", 0.5, 3.0, 1.2, 0.1)
        bins = st.slider("Grid Resolution (Bins)", 10, 50, 28)
        
    st.markdown("---")

    # Use session state to persist chart after interaction
    if st.button("Generate Heatmap", type="primary", use_container_width=True):
        st.session_state['hm_data'] = True 
        
    if st.session_state.get('hm_data'):
        with st.spinner(f"Processing {ticker}..."):
            
            # 1. DATA FETCH
            data = download_data(ticker)
            if data.empty:
                st.error("No data found.")
                return
                
            sznl_map = load_seasonal_map()
            df = calculate_heatmap_variables(data, sznl_map, ticker)
            
            # Get column names
            xcol = var_options[x_axis_label]
            ycol = var_options[y_axis_label]
            zcol = target_options[z_axis_label]
            
            # Drop NaNs for accurate binning
            clean_df = df.dropna(subset=[xcol, ycol, zcol])
            if clean_df.empty:
                st.error("Insufficient data for these variables.")
                return

            # 2. BINNING & GRIDDING
            x_edges, y_edges = build_bins_quantile(clean_df[xcol], clean_df[ycol], nx=bins, ny=bins)
            x_centers, y_centers, Z = grid_mean(clean_df, xcol, ycol, zcol, x_edges, y_edges)
            
            # 3. FILL & SMOOTH
            Z_filled = nan_neighbor_fill(Z)
            Z_smooth = smooth_display(Z_filled, sigma=smooth_sigma)
            
            # 4. COLOR SCALING (Seismic Reversed: Red=Neg, Blue=Pos)
            limit = np.nanmax(np.abs(Z_smooth))
            colorscale = get_seismic_colorscale()
            
            # 5. PLOT CONFIG
            fig = go.Figure(data=go.Heatmap(
                z=Z_smooth,
                x=x_centers,
                y=y_centers,
                colorscale=colorscale, 
                zmin=-limit, zmax=limit,
                reversescale=True, # Matplotlib Seismic is Blue->Red. We want Red->Blue.
                colorbar=dict(title="Fwd Return %"),
                hovertemplate=
                f"<b>{x_axis_label}</b>: %{{x:.1f}}<br>" +
                f"<b>{y_axis_label}</b>: %{{y:.1f}}<br>" +
                f"<b>Fwd Return</b>: %{{z:.2f}}%<extra></extra>"
            ))
            
            fig.update_layout(
                title=f"{ticker}: {z_axis_label}",
                xaxis_title=x_axis_label + " (0-100)",
                yaxis_title=y_axis_label + " (0-100)",
                height=650, template="plotly_white",
                clickmode='event+select' # Enable Click Interaction
            )
            
            # Current Position Crosshair
            last_row = df.iloc[-1]
            curr_x = last_row.get(xcol, np.nan)
            curr_y = last_row.get(ycol, np.nan)
            
            if not np.isnan(curr_x) and not np.isnan(curr_y):
                fig.add_vline(x=curr_x, line_width=2, line_dash="dash", line_color="black")
                fig.add_hline(y=curr_y, line_width=2, line_dash="dash", line_color="black")
                fig.add_annotation(x=curr_x, y=curr_y, text="Current", 
                                   showarrow=True, arrowhead=1, ax=30, ay=-30, bgcolor="white")
            
            # 6. RENDER & INTERACTION
            # on_select='rerun' captures the click event and reloads the script to show drill-down
            event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")
            
            # ---------------------------------------------------------------------
            # DRILL DOWN LOGIC
            # ---------------------------------------------------------------------
            st.markdown("### 🖱️ Drill Down: Click Area Details")
            
            selected_points = event.selection.get("points", [])
            
            if selected_points:
                pt = selected_points[0]
                click_x = pt["x"]
                click_y = pt["y"]
                
                # Find which bin edges encompass the clicked center
                # We find the nearest center index
                x_idx = np.abs(x_centers - click_x).argmin()
                y_idx = np.abs(y_centers - click_y).argmin()
                
                # Define the range for that bin
                x_min, x_max = x_edges[x_idx], x_edges[x_idx+1]
                y_min, y_max = y_edges[y_idx], y_edges[y_idx+1]
                
                # Filter Data
                mask = (
                    (clean_df[xcol] >= x_min) & (clean_df[xcol] <= x_max) &
                    (clean_df[ycol] >= y_min) & (clean_df[ycol] <= y_max)
                )
                drill_df = clean_df[mask].copy()
                
                if not drill_df.empty:
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.info(f"**Selected Bin Range**\n\n"
                                f"X: {x_min:.1f} - {x_max:.1f}\n\n"
                                f"Y: {y_min:.1f} - {y_max:.1f}")
                        st.metric("Avg Return in Bin", f"{drill_df[zcol].mean():.2f}%")
                        st.metric("Count", len(drill_df))
                        
                    with c2:
                        st.write("##### Historical Occurrences")
                        display_cols = ['Close', xcol, ycol, zcol]
                        
                        # Display sorted by date (descending)
                        style_df = drill_df[display_cols].sort_index(ascending=False)
                        
                        st.dataframe(
                            style_df.style.format({
                                "Close": "${:.2f}",
                                xcol: "{:.1f}",
                                ycol: "{:.1f}",
                                zcol: "{:.2f}%"
                            }).background_gradient(subset=[zcol], cmap="RdBu", vmin=-5, vmax=5),
                            use_container_width=True,
                            height=300
                        )
                else:
                    st.warning("No exact data points found in this smoothed region.")
            else:
                st.caption("Click any cell in the heatmap to see the specific dates and returns.")

def main():
    st.set_page_config(layout="wide", page_title="Heatmap Analytics")
    render_heatmap()

if __name__ == "__main__":
    main()
