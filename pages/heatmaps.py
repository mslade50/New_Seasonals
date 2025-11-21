import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import matplotlib
import datetime

# ... [Keep all previous Imports, Constants, Data Utils, and Calculation Engine functions exactly the same] ...
# ... [Copy: SECTOR_ETFS, INDEX_ETFS, INTERNATIONAL_ETFS, CSV_PATH] ...
# ... [Copy: load_seasonal_map, get_sznl_val_series, download_data] ...
# ... [Copy: get_seismic_colorscale, calculate_heatmap_variables] ...
# ... [Copy: build_bins_quantile, grid_mean, nan_neighbor_fill, smooth_display] ...

# -----------------------------------------------------------------------------
# UI: RENDER HEATMAP (Updated with Drill-Down)
# -----------------------------------------------------------------------------
def render_heatmap():
    st.subheader("Heatmap Analysis")
    
    # 1. SELECTION
    col1, col2, col3 = st.columns(3)
    
    with col1:
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

    if st.button("Generate Heatmap", type="primary", use_container_width=True):
        st.session_state['hm_data'] = True # Flag to keep data active
        
    # Check if we have active data to plot
    if st.session_state.get('hm_data'):
        with st.spinner(f"Processing {ticker}..."):
            # DATA FETCH (Cached)
            data = download_data(ticker)
            if data.empty:
                st.error("No data found.")
                return
                
            sznl_map = load_seasonal_map()
            df = calculate_heatmap_variables(data, sznl_map, ticker)
            
            xcol = var_options[x_axis_label]
            ycol = var_options[y_axis_label]
            zcol = target_options[z_axis_label]
            
            clean_df = df.dropna(subset=[xcol, ycol, zcol])
            if clean_df.empty:
                st.error("Insufficient data.")
                return

            # BINNING
            x_edges, y_edges = build_bins_quantile(clean_df[xcol], clean_df[ycol], nx=bins, ny=bins)
            x_centers, y_centers, Z = grid_mean(clean_df, xcol, ycol, zcol, x_edges, y_edges)
            
            # SMOOTHING
            Z_filled = nan_neighbor_fill(Z)
            Z_smooth = smooth_display(Z_filled, sigma=smooth_sigma)
            
            # PLOT
            limit = np.nanmax(np.abs(Z_smooth))
            colorscale = get_seismic_colorscale()
            
            fig = go.Figure(data=go.Heatmap(
                z=Z_smooth, x=x_centers, y=y_centers,
                colorscale=colorscale, zmin=-limit, zmax=limit,
                reversescale=True, colorbar=dict(title="Fwd Return %"),
                hovertemplate=
                f"<b>{x_axis_label}</b>: %{{x:.1f}}<br>" +
                f"<b>{y_axis_label}</b>: %{{y:.1f}}<br>" +
                f"<b>Fwd Return</b>: %{{z:.2f}}%<extra></extra>"
            ))
            
            fig.update_layout(
                title=f"{ticker}: {z_axis_label}",
                xaxis_title=x_axis_label, yaxis_title=y_axis_label,
                height=650, template="plotly_white",
                clickmode='event+select' # Enable clicking
            )
            
            # Current Position Crosshair
            last_row = df.iloc[-1]
            fig.add_vline(x=last_row[xcol], line_width=2, line_dash="dash", line_color="black")
            fig.add_hline(y=last_row[ycol], line_width=2, line_dash="dash", line_color="black")
            fig.add_annotation(x=last_row[xcol], y=last_row[ycol], text="Current", 
                               showarrow=True, arrowhead=1, ax=30, ay=-30, bgcolor="white")
            
            # RENDER WITH SELECTION EVENT
            # on_select='rerun' will allow us to capture the click in st.session_state
            event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")
            
            # ---------------------------------------------------------------------
            # DRILL DOWN LOGIC
            # ---------------------------------------------------------------------
            st.markdown("### 🖱️ Drill Down: Selected Bin Details")
            
            selected_points = event.selection.get("points", [])
            
            if selected_points:
                # 1. Get clicked coordinates (Centroids)
                pt = selected_points[0]
                click_x = pt["x"]
                click_y = pt["y"]
                
                # 2. Find which bin edges wrap this centroid
                # np.abs(centers - click) finds nearest center index
                x_idx = np.abs(x_centers - click_x).argmin()
                y_idx = np.abs(y_centers - click_y).argmin()
                
                # 3. Define Range
                x_min, x_max = x_edges[x_idx], x_edges[x_idx+1]
                y_min, y_max = y_edges[y_idx], y_edges[y_idx+1]
                
                # 4. Filter Dataframe
                mask = (
                    (clean_df[xcol] >= x_min) & (clean_df[xcol] <= x_max) &
                    (clean_df[ycol] >= y_min) & (clean_df[ycol] <= y_max)
                )
                drill_df = clean_df[mask].copy()
                
                # 5. Display
                if not drill_df.empty:
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.info(f"**Selected Range:**\n\n"
                                f"{x_axis_label}: {x_min:.1f} - {x_max:.1f}\n\n"
                                f"{y_axis_label}: {y_min:.1f} - {y_max:.1f}")
                        st.metric("Avg Return in Bin", f"{drill_df[zcol].mean():.2f}%")
                        st.metric("Occurrences", len(drill_df))
                        
                    with c2:
                        st.write("##### Historical Instances in this Bin")
                        display_cols = ['Close', xcol, ycol, zcol]
                        
                        # Format table for readability
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
                    st.warning("No exact data points found in the smoothed region (smoothing may interpolate empty spaces).")
            else:
                st.caption("Click any cell in the heatmap above to see the specific historical dates and returns associated with that area.")

# ... [Main function remains the same] ...
def main():
    st.set_page_config(layout="wide", page_title="Heatmap Analytics")
    st.title("Heatmap Analytics")
    # ...
    render_heatmap()

if __name__ == "__main__":
    main()
