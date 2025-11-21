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
    """Generates the seismic color scale exactly as requested."""
    seismic = []
    cm = matplotlib.colormaps["seismic"]
    for k in range(255):
        r, g, b, _ = cm(k / 254.0)
        # Plotly expects standard CSS rgb strings
        seismic.append([k / 254.0, f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'])
    return seismic

# -----------------------------------------------------------------------------
# CALCULATION ENGINE
# -----------------------------------------------------------------------------

def calculate_heatmap_variables(df, sznl_map, ticker):
    """Calculates advanced vars and ranks them 0-100 for the heatmap axes."""
    df = df.copy()
    
    # --- FIX: FORWARD RETURNS ---
    # (Future Price / Current Price) - 1
    for w in [5, 10, 21, 63]:
        df[f'FwdRet_{w}d'] = (df['Close'].shift(-w) / df['Close'] - 1.0) * 100.0

    # --- X/Y VARIABLES (Raw Calculation) ---
    
    # 1. Trailing Returns
    for w in [5, 10, 21, 63, 126, 252]:
        df[f'Ret_{w}d'] = df['Close'].pct_change(w)

    # 2. Realized Volatility (Annualized)
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    for w in [21, 63]:
        df[f'RealVol_{w}d'] = df['LogRet'].rolling(w).std() * np.sqrt(252) * 100.0
    
    # 3. Change in Realized Vol (21d vs 63d)
    df['VolChange'] = df['RealVol_21d'] - df['RealVol_63d']

    # 4. Volume Ratios
    for w in [5, 10, 21]:
        # Volume relative to 63d baseline
        df[f'VolRatio_{w}d'] = df['Volume'].rolling(w).mean() / df['Volume'].rolling(63).mean()

    # 5. Seasonality
    df['Seasonal'] = get_sznl_val_series(ticker, df.index, sznl_map)

    # --- RANK TRANSFORMATION (Percentiles 0-100) ---
    vars_to_rank = [
        'Ret_5d', 'Ret_10d', 'Ret_21d', 'Ret_63d', 'Ret_126d', 'Ret_252d',
        'RealVol_21d', 'RealVol_63d', 'VolChange',
        'VolRatio_5d', 'VolRatio_10d', 'VolRatio_21d'
    ]
    
    for v in vars_to_rank:
        # Use Expanding Rank to mimic the user's dimension requirement
        # We convert to 0-100 scale
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

    if len(x_edges) < 3: x_edges = np.linspace(x.min(), x.max(), max(3, nx + 1))
    if len(y_edges) < 3: y_edges = np.linspace(y.min(), y.max(), max(3, ny + 1))
    
    return x_edges, y_edges

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
    """Fills small holes in the grid."""
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
    """Mask-aware Gaussian smoothing."""
    Z = np.asarray(Z, float)
    mask = np.isfinite(Z)
    Z_fill = np.where(mask, Z, 0.0)
    w = gaussian_filter(mask.astype(float), sigma=sigma, mode="nearest")
    z = gaussian_filter(Z_fill, sigma=sigma, mode="nearest")
    out = z / np.maximum(w, 1e-9)
    return out

# -----------------------------------------------------------------------------
# UI RENDER
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="Heatmap Analytics")
    st.title("Heatmap Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Variable Mapping for Display -> Column Name
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
            "Change in Vol (21d-63d) Rank": "VolChange_Rank",
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

    if st.button("Generate Heatmap", type="primary", use_container_width=True):
        st.info(f"Downloading full history for {ticker}...")
        data = download_data(ticker)
        
        if data.empty:
            st.error("No data found for ticker.")
            return
            
        sznl_map = load_seasonal_map()
        df = calculate_heatmap_variables(data, sznl_map, ticker)
        
        xcol = var_options[x_axis_label]
        ycol = var_options[y_axis_label]
        zcol = target_options[z_axis_label]
        
        # Clean data for binning
        clean_df = df.dropna(subset=[xcol, ycol, zcol])
        if clean_df.empty:
            st.error("Not enough data to calculate ranks/returns.")
            return

        # Build Edges (Quantile based)
        x_edges, y_edges = build_bins_quantile(clean_df[xcol], clean_df[ycol], nx=bins, ny=bins)
        
        # Grid Mean
        x_centers, y_centers, Z = grid_mean(clean_df, xcol, ycol, zcol, x_edges, y_edges)
        
        # Fill & Smooth
        Z_filled = nan_neighbor_fill(Z)
        Z_smooth = smooth_display(Z_filled, sigma=smooth_sigma)
        
        # Colors (Center on 0)
        # If max abs val is 5%, scale goes -5% to +5%
        limit = np.nanmax(np.abs(Z_smooth))
        
        # Generate 'Seismic' Scale
        colorscale = get_seismic_colorscale()
        
        fig = go.Figure(data=go.Heatmap(
            z=Z_smooth,
            x=x_centers,
            y=y_centers,
            colorscale=colorscale,
            zmin=-limit, zmax=limit,
            reversescale=True, # Seismic is usually Blue(Low)->Red(High). 
                               # We want Red(Low/Neg) -> Blue(High/Pos).
                               # So we reverse it.
            colorbar=dict(title="Fwd Return %")
        ))
        
        fig.update_layout(
            title=f"{ticker}: {z_axis_label}",
            xaxis_title=x_axis_label,
            yaxis_title=y_axis_label,
            height=700,
            template="plotly_white"
        )
        
        # Add crosshair for "Current Positioning"
        last_row = df.iloc[-1]
        current_x = last_row[xcol]
        current_y = last_row[ycol]
        
        fig.add_vline(x=current_x, line_width=2, line_dash="dash", line_color="black")
        fig.add_hline(y=current_y, line_width=2, line_dash="dash", line_color="black")
        
        # Add annotation for current position
        fig.add_annotation(
            x=current_x, y=current_y,
            text="Current",
            showarrow=True,
            arrowhead=1,
            ax=30, ay=-30,
            bgcolor="white", bordercolor="black"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats Table
        st.markdown("### Distribution Statistics")
        stats_df = clean_df[[xcol, ycol, zcol]].describe()
        st.dataframe(stats_df.style.format("{:.2f}"))

if __name__ == "__main__":
    main()
