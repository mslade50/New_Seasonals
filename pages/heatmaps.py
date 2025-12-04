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
NAAIM_PATH = "naaim.csv"

# -----------------------------------------------------------------------------
# DATA LOADERS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_seasonal_map():
    """
    Loads seasonal ranks with EXACT DATE matching (YYYY-MM-DD).
    """
    try:
        df = pd.read_csv(SEASONAL_PATH)
    except Exception:
        return {}
        
    if df.empty: return {}
    
    # Ensure Date is parsed as datetime objects
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])
    
    # Sort to ensure chronological order
    df = df.sort_values("Date")
    
    output_map = {}
    for ticker, group in df.groupby("ticker"):
        # Map: Timestamp -> Rank (Exact Match)
        output_map[ticker] = pd.Series(
            group.seasonal_rank.values, index=group.Date
        ).to_dict()
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
        # Robust Date Parsing
        date_col = None
        for c in df.columns:
            if 'date' in c.lower():
                date_col = c
                break
        if not date_col: date_col = df.columns[0]
        
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        df = df[~df.index.duplicated(keep='first')]
        
        # Identify Value Column
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

@st.cache_data(show_spinner=True)
def download_data(ticker):
    try:
        df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
        df = df[~df.index.duplicated(keep='first')]
        return df, pd.Timestamp.now(tz='America/New_York')
    except: return pd.DataFrame(), None

# -----------------------------------------------------------------------------
# CALCULATIONS
# -----------------------------------------------------------------------------
def get_sznl_val_series(ticker, dates, sznl_map):
    # Simplified: Only use the map, fallback to 50 if missing
    t_map = sznl_map.get(ticker, {})
    if t_map:
        return dates.map(lambda x: (x.month, x.day)).map(t_map).fillna(50.0)
    return pd.Series(50.0, index=dates)

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
    
    # 1. Price Features
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. Forward Returns (The Targets)
    # This creates NaNs at the end of the dataframe equal to 'w'
    for w in [1, 2, 3, 5, 10, 21, 63]:
        df[f'FwdRet_{w}d'] = (df['Close'].shift(-w) / df['Close'] - 1.0) * 100.0

    # 3. Trailing Features
    for w in [5, 10, 21, 252]:
        df[f'Ret_{w}d'] = df['Close'].pct_change(w)
    
    for w in [21, 63]:
        df[f'RealVol_{w}d'] = df['LogRet'].rolling(w).std() * np.sqrt(252) * 100.0
        
    for w in [10, 21]:
        df[f'VolRatio_{w}d'] = df['Volume'].rolling(w).mean() / df['Volume'].rolling(63).mean()
        
    df['Seasonal'] = get_sznl_val_series(ticker, df.index, sznl_map)

    # 4. Rank Transformations
    vars_to_rank = ['Ret_5d', 'Ret_10d', 'Ret_21d', 'Ret_252d', 'RealVol_21d', 'RealVol_63d', 'VolRatio_10d', 'VolRatio_21d']
    for v in vars_to_rank:
        df[v + '_Rank'] = df[v].expanding(min_periods=252).rank(pct=True) * 100.0

    # 5. External Data Merges
    if not market_metrics_df.empty:
        df = df.join(market_metrics_df, how='left')
        df.update(df.filter(regex='^Mkt_').ffill(limit=3))

    if ticker != "SPY":
        spy_df = get_spy_context()
        if not spy_df.empty:
            df = df.join(spy_df, how='left')
            spy_cols = [c for c in spy_df.columns if 'Rank' in c]
            df.update(df[spy_cols].ffill(limit=1))

    naaim_df = load_naaim_data()
    if not naaim_df.empty:
        df = df.join(naaim_df, how='left')
        naaim_cols = ['NAAIM', 'NAAIM_MA5', 'NAAIM_MA12']
        df[naaim_cols] = df[naaim_cols].ffill()
        for col in naaim_cols:
            df[f"{col}_Rank"] = df[col].expanding(min_periods=252).rank(pct=True) * 100.0

    return df

# -----------------------------------------------------------------------------
# HEATMAP MATH & UTILS
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
    
    # Pre-filter for valid data
    m = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    x = x[m]; y = y[m]; z = z[m]

    if len(x) == 0:
        return np.array([]), np.array([]), np.full((len(y_edges)-1, len(x_edges)-1), np.nan)

    xi = np.digitize(x, x_edges) - 1
    yi = np.digitize(y, y_edges) - 1
    nx = len(x_edges) - 1; ny = len(y_edges) - 1

    sums = np.zeros((ny, nx), float)
    counts = np.zeros((ny, nx), float)

    valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    xi = xi[valid]; yi = yi[valid]; zv = z[valid]
    
    for X, Y, Z in zip(xi, yi, zv):
        sums[Y, X] += Z
        counts[Y, X] += 1.0

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
    sum_vals = convolve2d(np.nan_to_num(Z), kernel, mode='same', boundary='symm')
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
# MAIN UI
# -----------------------------------------------------------------------------
def render_heatmap():
    st.subheader("Data Inspector & Heatmap")
    
    # 1. Inputs
    c1, c2, c3 = st.columns(3)
    with c1: ticker = st.text_input("Ticker", value="SMH").upper()
    with c2: st.write("") # Spacer
    with c3: 
        if st.button("Load Data", type="primary", use_container_width=True):
            st.session_state['data_loaded'] = True

    if st.session_state.get('data_loaded'):
        with st.spinner(f"Loading {ticker}..."):
            # Load Data
            data, fetch_time = download_data(ticker)
            if data.empty:
                st.error("No data found.")
                return
            
            sznl_map = load_seasonal_map()
            mkt_metrics = load_market_metrics()
            
            # Calculate Features
            df = calculate_heatmap_variables(data, sznl_map, mkt_metrics, ticker)
            
            st.success(f"Data Loaded: {len(df)} rows. Last Date: {df.index[-1].date()}")
            
            st.markdown("---")
            
            # 2. Variable Selection
            var_options = {
                "Seasonality Rank": "Seasonal",
                "5d Return Rank": "Ret_5d_Rank", "10d Return Rank": "Ret_10d_Rank",
                "21d Return Rank": "Ret_21d_Rank", "252d Return Rank": "Ret_252d_Rank",
                "SPY 5d Return Rank": "SPY_Ret_5d_Rank", "SPY 10d Return Rank": "SPY_Ret_10d_Rank",
                "SPY 21d Return Rank": "SPY_Ret_21d_Rank",
                "21d Realized Vol Rank": "RealVol_21d_Rank", "63d Realized Vol Rank": "RealVol_63d_Rank",
                "10d Rel. Volume Rank": "VolRatio_10d_Rank", 
                "Net Highs (5d) Rank": "Mkt_Total_NH_5d_Rank", "Net Highs (21d) Rank": "Mkt_Total_NH_21d_Rank",
                "NAAIM Rank": "NAAIM_Rank", "NAAIM 5w Rank": "NAAIM_MA5_Rank"
            }
            target_options = {
                "1d Fwd Return": "FwdRet_1d", "5d Fwd Return": "FwdRet_5d",
                "10d Fwd Return": "FwdRet_10d", "21d Fwd Return": "FwdRet_21d",
                "63d Fwd Return": "FwdRet_63d"
            }

            row1 = st.columns(3)
            with row1[0]: x_label = st.selectbox("X-Axis (Feature)", list(var_options.keys()), index=0)
            with row1[1]: y_label = st.selectbox("Y-Axis (Feature)", list(var_options.keys()), index=2)
            with row1[2]: z_label = st.selectbox("Z-Axis (Target)", list(target_options.keys()), index=1)
            
            # 3. Filtering Logic
            st.markdown("#### Filter Logic (Optional)")
            row2 = st.columns(3)
            with row2[0]:
                filter_label = st.selectbox("Filter Variable", ["None"] + list(var_options.keys()))
            
            df_filtered = df.copy()
            if filter_label != "None":
                with row2[1]:
                    f_min, f_max = st.slider("Filter Range", 0.0, 100.0, (0.0, 100.0))
                
                f_col = var_options[filter_label]
                if f_col in df.columns:
                    df_filtered = df_filtered[(df_filtered[f_col] >= f_min) & (df_filtered[f_col] <= f_max)]
                    st.caption(f"Filtered Data Points: {len(df_filtered)}")

            with row2[2]:
                smooth = st.slider("Smoothing", 0.5, 3.0, 1.2)
                bins = st.slider("Grid Bins", 10, 50, 25)

            # 4. Generate Heatmap
            xcol, ycol, zcol = var_options[x_label], var_options[y_label], target_options[z_label]
            
            # CRITICAL STEP: Remove Forward Data Leaks
            # We strictly drop rows where the Target (Z) is NaN.
            # E.g., if FwdRet_21d, the last 21 days are NaN. Drop them.
            clean_df = df_filtered.dropna(subset=[xcol, ycol, zcol])
            
            if clean_df.empty:
                st.warning("No data available for these selections.")
                return

            # Grid Calculation
            x_edges, y_edges = build_bins_quantile(clean_df[xcol], clean_df[ycol], nx=bins, ny=bins)
            x_centers, y_centers, Z = grid_mean(clean_df, xcol, ycol, zcol, x_edges, y_edges)
            
            # Display Processing
            Z_filled = nan_neighbor_fill(Z)
            Z_smooth = smooth_display(Z_filled, sigma=smooth)
            
            # Plot
            limit = np.nanmax(np.abs(Z_smooth))
            fig = go.Figure(data=go.Heatmap(
                z=Z_smooth, x=x_centers, y=y_centers,
                colorscale=get_seismic_colorscale(),
                zmin=-limit, zmax=limit, zmid=0, reversescale=True,
                colorbar=dict(title=f"{zcol} (%)"),
                hovertemplate=f"{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.1f}}<br>Return: %{{z:.2f}}%<extra></extra>"
            ))
            
            fig.update_layout(
                title=f"{ticker}: {z_label} Heatmap (Filtered)",
                xaxis_title=x_label, yaxis_title=y_label,
                height=600, template="plotly_white"
            )
            
            # Add Current Marker
            last_row = df.iloc[-1]
            curr_x = last_row.get(xcol, np.nan)
            curr_y = last_row.get(ycol, np.nan)
            
            if not np.isnan(curr_x) and not np.isnan(curr_y):
                fig.add_vline(x=curr_x, line_dash="dash", line_color="black")
                fig.add_hline(y=curr_y, line_dash="dash", line_color="black")
                fig.add_annotation(x=curr_x, y=curr_y, text="Current", bgcolor="white", showarrow=True)

            st.plotly_chart(fig, use_container_width=True)
            
            # 5. Data Explorer Table (Raw data backing the map)
            st.divider()
            st.markdown("### Raw Data (Used in Map)")
            st.caption("Rows with NaN targets (the most recent days) are strictly excluded from this view.")
            
            display_cols = ['Close', xcol, ycol, zcol]
            explorer = clean_df[display_cols].sort_index(ascending=False)
            
            st.dataframe(
                explorer.style.format("{:.2f}").background_gradient(subset=[zcol], cmap="RdBu", vmin=-limit, vmax=limit),
                use_container_width=True
            )

def main():
    st.set_page_config(layout="wide", page_title="Simple Heatmap")
    render_heatmap()

if __name__ == "__main__":
    main()
