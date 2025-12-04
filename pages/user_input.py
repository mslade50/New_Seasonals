import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objs as go
import streamlit as st
from datetime import date, timedelta

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def compute_atr(df, window=14):
    df = df.copy()
    df["previous_close"] = df["Close"].shift(1)
    df["TR"] = df[["High", "previous_close"]].max(axis=1) - df[["Low", "previous_close"]].min(axis=1)
    df["ATR"] = df["TR"].rolling(window=window).mean()
    df["ATR%"] = (df["ATR"] / df["Close"]) * 100
    return df

# -----------------------------------------------------------------------------
# MAIN CHART LOGIC
# -----------------------------------------------------------------------------

def calculate_path(df, cycle_label, cycle_start_mapping):
    """
    Calculates the average cumulative log return path for a given dataframe and cycle.
    """
    if df.empty:
        return pd.Series()

    if cycle_label == "All Years":
        cycle_data = df.copy()
    else:
        cycle_start = cycle_start_mapping[cycle_label]
        years_in_cycle = [cycle_start + i * 4 for i in range(30)] 
        cycle_data = df[df["year"].isin(years_in_cycle)].copy()

    # Normalize week numbers if needed
    if "week_of_month_5day" in cycle_data.columns:
        cycle_data.loc[cycle_data["week_of_month_5day"] > 4, "week_of_month_5day"] = 4

    # Group by day count and calculate mean path
    avg_path = (
        cycle_data.groupby("day_count")["log_return"]
        .mean()
        .cumsum()
        .apply(np.exp) - 1
    )
    return avg_path

def seasonals_chart(ticker, cycle_label, enable_time_travel, reference_year, show_all_years_line):
    cycle_start_mapping = {
        "Election": 1952,
        "Pre-Election": 1951,
        "Post-Election": 1953,
        "Midterm": 1950
    }

    # Data Fetching
    end_date_fetch = dt.datetime.now() + timedelta(days=5)
    spx = yf.download(ticker, period="max", end=end_date_fetch, progress=False)

    if spx.empty:
        st.error(f"No data found for {ticker}.")
        return
    
    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = spx.columns.get_level_values(0)

    # Feature Engineering
    spx["log_return"] = np.log(spx["Close"] / spx["Close"].shift(1))
    spx["year"] = spx.index.year
    spx["month"] = spx.index.month
    spx["day_count"] = spx.groupby("year").cumcount() + 1
    
    current_year = dt.date.today().year

    # -------------------------------------------------------------------------
    # 1. CURRENT MODEL CONSTRUCTION (Always Visible)
    # -------------------------------------------------------------------------
    # Use all completed years < current_year
    df_all_history = spx[spx["year"] < current_year].copy()
    
    # Path: Current Cycle Avg
    path_current_avg = calculate_path(df_all_history, cycle_label, cycle_start_mapping)
    
    # Path: Current "All Years" Avg (Optional)
    path_current_all_years = pd.Series()
    if show_all_years_line:
        path_current_all_years = calculate_path(df_all_history, "All Years", cycle_start_mapping)

    # Path: Current Realized
    df_current_year = spx[spx["year"] == current_year].copy()
    path_current_realized = pd.Series()
    if not df_current_year.empty:
        df_current_year["log_return"] = np.log(df_current_year["Close"] / df_current_year["Close"].shift(1))
        path_current_realized = df_current_year["log_return"].cumsum().apply(np.exp) - 1

    # -------------------------------------------------------------------------
    # 2. TIME TRAVEL CONSTRUCTION (Conditional)
    # -------------------------------------------------------------------------
    path_historical_avg = pd.Series()
    path_ref_realized = pd.Series()
    
    if enable_time_travel:
        # Historical Pool: Strictly < Reference Year
        df_historical_pool = spx[spx["year"] < reference_year].copy()
        
        # Path: Historical Model Avg
        path_historical_avg = calculate_path(df_historical_pool, cycle_label, cycle_start_mapping)
        
        # Path: Historical Realized
        df_ref_year = spx[spx["year"] == reference_year].copy()
        if not df_ref_year.empty:
            df_ref_year["log_return"] = np.log(df_ref_year["Close"] / df_ref_year["Close"].shift(1))
            path_ref_realized = df_ref_year["log_return"].cumsum().apply(np.exp) - 1

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    fig = go.Figure()

    # A. Current Cycle Model (Orange)
    fig.add_trace(go.Scatter(
        x=path_current_avg.index,
        y=path_current_avg.values,
        mode="lines",
        name=f"Current Model ({cycle_label})",
        line=dict(color="#FF8C00", width=3) 
    ))

    # B. Current All Years Model (Light Blue - Optional)
    if not path_current_all_years.empty:
        fig.add_trace(go.Scatter(
            x=path_current_all_years.index,
            y=path_current_all_years.values,
            mode="lines",
            name="Current Model (All Years)",
            line=dict(color="lightblue", width=1, dash='dot')
        ))

    # C. Historical Model (Gold Dashed - Time Travel)
    if enable_time_travel and not path_historical_avg.empty:
        fig.add_trace(go.Scatter(
            x=path_historical_avg.index,
            y=path_historical_avg.values,
            mode="lines",
            name=f"Model in {reference_year} (Pre-{reference_year} Data)",
            line=dict(color="#FCD12A", width=2, dash='dash')
        ))

    # D. Current Realized (Green)
    if not path_current_realized.empty:
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(path_current_realized) + 1),
            y=path_current_realized.values,
            mode="lines",
            name=f"{current_year} Realized (YTD)",
            line=dict(color="#39FF14", width=2)
        ))

    # E. Historical Realized (Cyan - Time Travel)
    if enable_time_travel and not path_ref_realized.empty:
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(path_ref_realized) + 1),
            y=path_ref_realized.values,
            mode="lines",
            name=f"{reference_year} Realized",
            line=dict(color="#00FFFF", width=2)
        ))

    # -------------------------------------------------------------------------
    # MARKERS & DATES
    # -------------------------------------------------------------------------
    # Logic: If Time Travel, context is Ref Year. If not, context is Today.
    
    today = dt.date.today()
    
    if enable_time_travel:
        # Context: Reference Year
        target_year = reference_year
        # Map today's M/D to Ref Year
        try:
            target_date_start = dt.date(target_year, today.month, today.day)
        except ValueError:
            target_date_start = dt.date(target_year, 2, 28)
        
        # For lookup, we need a dataframe representing that year
        df_context = spx[spx["year"] == target_year]
        # We plot dots on the HISTORICAL model if comparison is active
        model_to_plot_on = path_historical_avg if not path_historical_avg.empty else path_current_avg
        marker_color = "#FCD12A" # Gold
        
    else:
        # Context: Current Year (Today)
        target_year = current_year
        target_date_start = today
        # For lookup, we use current year data (to find day count)
        df_context = spx[spx["year"] == target_year]
        model_to_plot_on = path_current_avg
        marker_color = "white"

    # Find Day Count
    day_count_marker = None
    if not df_context.empty:
        # Search sorted to find closest trading day
        closest_idx = df_context.index.searchsorted(dt.datetime.combine(target_date_start, dt.time.min))
        if closest_idx >= len(df_context):
            closest_idx = len(df_context) - 1
        
        day_count_marker = df_context.iloc[closest_idx]["day_count"]

        # Plot Current Day Marker
        if day_count_marker in model_to_plot_on.index:
            y_val = model_to_plot_on.get(day_count_marker)
            
            fig.add_trace(go.Scatter(
                x=[day_count_marker],
                y=[y_val],
                mode="markers",
                name=f"Current Date ({target_date_start.strftime('%b %d')})",
                marker=dict(color=marker_color, size=8, line=dict(width=1, color="black")),
                showlegend=False
            ))

            # Projections
            future_dates = pd.bdate_range(start=target_date_start, periods=30)
            offsets = [5, 10, 21]
            
            for offset in offsets:
                target_idx = day_count_marker + offset
                if target_idx in model_to_plot_on.index:
                    proj_y = model_to_plot_on.get(target_idx)
                    
                    if offset < len(future_dates):
                        d_label = future_dates[offset].strftime("%b %d")
                    else:
                        d_label = "N/A"
                        
                    fig.add_trace(go.Scatter(
                        x=[target_idx],
                        y=[proj_y],
                        mode="markers",
                        name=f"T+{offset} ({d_label})", # Legend shows date
                        marker=dict(color=marker_color, size=6, symbol="circle"),
                    ))

    # Layout
    title_suffix = f"vs {reference_year}" if enable_time_travel else ""
    fig.update_layout(
        height=1200,
        title=f"Seasonal Analysis: {ticker} {title_suffix}",
        xaxis_title="Trading Day of Year",
        yaxis_title="Cumulative Return",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(
            bgcolor="rgba(20,20,20,0.8)", 
            font=dict(color="white"),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # DETAILED HISTORY TABLE
    # -------------------------------------------------------------------------
    st.divider()
    
    # Text logic
    if enable_time_travel:
        st.subheader(f"📜 Historical Returns (Pre-{reference_year})")
        st.caption(f"Stats exclude {reference_year} and later to prevent look-ahead bias.")
        cutoff_year = reference_year
    else:
        st.subheader(f"📜 Historical Returns (Pre-{current_year})")
        st.caption(f"Stats exclude incomplete current year ({current_year}).")
        cutoff_year = current_year

    if day_count_marker:
        spx_full = spx.copy()
        
        # Calculate Fwd Returns on full dataset
        spx_full['Fwd_5d'] = spx_full['Close'].shift(-5) / spx_full['Close'] - 1
        spx_full['Fwd_10d'] = spx_full['Close'].shift(-10) / spx_full['Close'] - 1
        spx_full['Fwd_21d'] = spx_full['Close'].shift(-21) / spx_full['Close'] - 1

        daily_snapshots = spx_full[spx_full['day_count'] == day_count_marker].copy()

        # FILTER: Strictly < cutoff_year
        display_df = daily_snapshots[daily_snapshots['year'] < cutoff_year].copy()

        if not display_df.empty:
            display_df = display_df[['year', 'Fwd_5d', 'Fwd_10d', 'Fwd_21d']]
            
            display_df['Fwd_5d'] = display_df['Fwd_5d'] * 100
            display_df['Fwd_10d'] = display_df['Fwd_10d'] * 100
            display_df['Fwd_21d'] = display_df['Fwd_21d'] * 100
            display_df = display_df.sort_values('year', ascending=False)

            # Cycle Years logic
            if cycle_label != "All Years":
                start_yr = cycle_start_mapping.get(cycle_label)
                highlight_years = [start_yr + i * 4 for i in range(30)] 
            else:
                highlight_years = []

            # --- STATS ---
            st.markdown(f"##### 🎯 Fwd Return Statistics")
            
            def calculate_stats_row(sub_df):
                if sub_df.empty:
                    return {k: np.nan for k in ["n", "5_median", "5_mean", "5_pospct", "10_median", "10_mean", "10_pospct", "21_median", "21_mean", "21_pospct"]}
                
                res = {"n": int(len(sub_df))}
                for d in [5, 10, 21]:
                    col = f"Fwd_{d}d"
                    res[f"{d}_median"] = sub_df[col].median()
                    res[f"{d}_mean"] = sub_df[col].mean()
                    res[f"{d}_pospct"] = (sub_df[col] > 0).mean() * 100
                return res

            stats_all = calculate_stats_row(display_df)
            
            if cycle_label != "All Years":
                df_cycle = display_df[display_df['year'].isin(highlight_years)]
                stats_cycle = calculate_stats_row(df_cycle)
                cycle_row_name = f"{cycle_label} Cycle"
            else:
                stats_cycle = stats_all
                cycle_row_name = f"All Years"

            stats_df = pd.DataFrame([stats_all, stats_cycle], index=[f"All History (<{cutoff_year})", cycle_row_name])
            
            ordered_cols = ["n"]
            for d in [5, 10, 21]:
                ordered_cols.extend([f"{d}_median", f"{d}_mean", f"{d}_pospct"])
            stats_df = stats_df[ordered_cols]

            # Custom Formatting
            def color_pos_pct(val):
                if pd.isna(val): return ''
                if val > 80:
                    return 'color: #90ee90; font-weight: bold;' 
                elif val < 25:
                    return 'color: #ffcccb; font-weight: bold;' 
                return ''

            st.dataframe(
                stats_df.style.format({
                    "n": "{:.0f}",
                    "5_median": "{:.2f}%", "5_mean": "{:.2f}%", "5_pospct": "{:.1f}%",
                    "10_median": "{:.2f}%", "10_mean": "{:.2f}%", "10_pospct": "{:.1f}%",
                    "21_median": "{:.2f}%", "21_mean": "{:.2f}%", "21_pospct": "{:.1f}%",
                })
                .map(color_pos_pct, subset=["5_pospct", "10_pospct", "21_pospct"]),
                use_container_width=True
            )

            # --- TABLE RENDER ---
            def highlight_year_cell(val):
                if val in highlight_years:
                    return 'background-color: #d4af37; color: black; font-weight: bold;' 
                return ''

            st.dataframe(
                display_df.style
                .format({
                    "year": "{:.0f}",
                    "Fwd_5d": "{:+.2f}%",
                    "Fwd_10d": "{:+.2f}%",
                    "Fwd_21d": "{:+.2f}%"
                })
                .background_gradient(subset=["Fwd_5d", "Fwd_10d", "Fwd_21d"], cmap="RdYlGn", vmin=-5, vmax=5)
                .map(highlight_year_cell, subset=['year']),
                use_container_width=True,
                height=400,
                hide_index=True
            )
        else:
            st.warning(f"No historical data available prior to {cutoff_year}.")

# -----------------------------------------------------------------------------
# APP ENTRY POINT
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Seasonality Analysis")
    st.title("📊 Presidential Cycle Seasonality")

    # UI Layout
    col1, col2, col3, col4 = st.columns([1, 1, 1.5, 1])
    
    with col1:
        ticker = st.text_input("Ticker", value="SPY").upper()
    
    with col2:
        cycle_label = st.selectbox(
            "Cycle Type",
            ["All Years", "Election", "Pre-Election", "Post-Election", "Midterm"],
            index=3
        )
    
    with col3:
        # Time Travel Logic
        c3_1, c3_2 = st.columns([1, 1])
        with c3_1:
            enable_time_travel = st.checkbox("Enable Time Travel", value=False)
        with c3_2:
            current_year = dt.date.today().year
            if enable_time_travel:
                reference_year = st.number_input("Compare vs Year", min_value=1950, max_value=current_year, value=current_year-1)
            else:
                reference_year = current_year # Placeholder
                st.write("") # Spacer

    with col4:
        st.write("")
        show_all_years_line = st.checkbox("Overlay 'All Years' Avg", value=False)

    if st.button("Run Analysis", type="primary", use_container_width=True):
        try:
            with st.spinner("Calculating cycle stats..."):
                seasonals_chart(ticker, cycle_label, enable_time_travel, reference_year, show_all_years_line)
        except Exception as e:
            st.error(f"Error generating chart: {e}")

if __name__ == "__main__":
    main()
