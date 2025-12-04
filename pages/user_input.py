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

def summarize_data(df, include_atr=True):
    if df.empty:
        return {
            "Avg Return (%)": np.nan, 
            "Median Daily Return (%)": np.nan, 
            "Avg ATR%": np.nan if include_atr else np.nan,
            "% Pos": np.nan
        }

    included_years = df["year"].unique()

    if "week_of_month_5day" in df.columns:
        grouped_returns = df[df["year"].isin(included_years)].groupby(["year", "month", "week_of_month_5day"])["log_return"].sum() * 100
    else:
        grouped_returns = df[df["year"].isin(included_years)].groupby(["year", "month"])["log_return"].sum() * 100

    return {
        "Avg Return (%)": grouped_returns.mean(),
        "Median Daily Return (%)": grouped_returns.median(),
        "Avg ATR%": df["ATR%"].mean() if include_atr else np.nan,
    }

# -----------------------------------------------------------------------------
# MAIN CHART LOGIC
# -----------------------------------------------------------------------------

def seasonals_chart(ticker, cycle_label, reference_year, show_all_years_line=False):
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
    spx["trading_day_of_month"] = spx.groupby([spx.index.year, spx.index.month]).cumcount() + 1
    spx["week_of_month_5day"] = (spx["trading_day_of_month"] - 1) // 5 + 1
    spx["day_count"] = spx.groupby("year").cumcount() + 1

    # -------------------------------------------------------------------------
    # TIME TRAVEL: HISTORICAL POOL
    # -------------------------------------------------------------------------
    # The pool for AVERAGES must strictly be data PRIOR to the reference year
    historical_pool = spx[spx["year"] < reference_year].copy()
    
    # We DO NOT grab ref_year_data for plotting the green line, 
    # unless it is the actual current year (optional, but user asked for "predicted path not realized")
    # So we will focus purely on the average construction.

    # Cycle Filtering on HISTORICAL POOL
    if cycle_label == "All Years":
        cycle_data = historical_pool.copy()
    else:
        cycle_start = cycle_start_mapping[cycle_label]
        years_in_cycle = [cycle_start + i * 4 for i in range(30)] 
        cycle_data = historical_pool[historical_pool["year"].isin(years_in_cycle)].copy()

    cycle_data = compute_atr(cycle_data)
    cycle_data.loc[cycle_data["week_of_month_5day"] > 4, "week_of_month_5day"] = 4

    # --- Plotting Average Path ---
    avg_path = (
        cycle_data.groupby("day_count")["log_return"]
        .mean()
        .cumsum()
        .apply(np.exp) - 1
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=avg_path.index,
        y=avg_path.values,
        mode="lines",
        name=f"Avg Path ({cycle_label})",
        line=dict(color="orange")
    ))

    # Optional Overlay: All Years (Historical only)
    if show_all_years_line:
        all_avg_path = (
            historical_pool.groupby("day_count")["log_return"]
            .mean()
            .cumsum()
            .apply(np.exp) - 1
        )
        fig.add_trace(go.Scatter(
            x=all_avg_path.index,
            y=all_avg_path.values,
            mode="lines",
            name="All Years Avg Path",
            line=dict(color="lightblue", width=1, dash='dot')
        ))

    # -------------------------------------------------------------------------
    # TIME TRAVEL: LOCATING "TODAY" IN THE REFERENCE YEAR
    # -------------------------------------------------------------------------
    today = dt.date.today()
    
    # Construct the proxy date in the reference year
    # Handle Leap Year edge case: if today is Feb 29 and ref year isn't leap, map to Feb 28
    try:
        ref_date = dt.date(reference_year, today.month, today.day)
    except ValueError:
        ref_date = dt.date(reference_year, 2, 28)

    # We need to find the 'day_count' (1-252) that corresponds to this date (or closest trading day) in the ref year
    # We can't use the historical_pool because that excludes the ref year.
    # We use the raw 'spx' data just to find the correct index integer for that year.
    ref_year_calendar = spx[spx["year"] == reference_year]
    
    current_day_count_val = None
    
    if not ref_year_calendar.empty:
        # Find the row with the closest date <= ref_date
        # Since 'day_count' is monotonic, we can just find the index
        closest_idx = ref_year_calendar.index.searchsorted(dt.datetime.combine(ref_date, dt.time.min))
        
        # Ensure we don't go out of bounds
        if closest_idx >= len(ref_year_calendar):
            closest_idx = len(ref_year_calendar) - 1
            
        current_day_count_val = ref_year_calendar.iloc[closest_idx]["day_count"]
        
        # Plot Marker on the AVERAGE Line (Where the seasonal model says we should be)
        avg_path_y_value = avg_path.get(current_day_count_val)
        
        if avg_path_y_value is not None:
            fig.add_trace(go.Scatter(
                x=[current_day_count_val],
                y=[avg_path_y_value],
                mode="markers",
                name=f"Current Date ({ref_date.strftime('%b %d')})",
                marker=dict(color="white", size=8, line=dict(width=1, color="black")),
            ))

            # ---------------------------------------------------------------------
            # PROJECTIONS: T+5, T+10, T+21
            # ---------------------------------------------------------------------
            # Generate Business Days forward from the Ref Date
            # Use pandas bdate_range to handle weekends properly
            future_dates = pd.bdate_range(start=ref_date, periods=30)
            
            projection_offsets = [5, 10, 21]
            colors = ["#FFD700", "#FFD700", "#FFD700"] # Gold/Yellow
            
            for i, offset in enumerate(projection_offsets):
                target_idx = current_day_count_val + offset
                
                # Retrieve the projected Y value from the Average Path
                if target_idx in avg_path.index:
                    proj_y = avg_path.get(target_idx)
                    
                    # Calculate the calendar date for the legend
                    if offset < len(future_dates):
                        target_date_obj = future_dates[offset]
                        date_label = target_date_obj.strftime("%b %d")
                    else:
                        date_label = "N/A"

                    # Plot just the Dot
                    fig.add_trace(go.Scatter(
                        x=[target_idx],
                        y=[proj_y],
                        mode="markers",
                        name=f"T+{offset} ({date_label})", # Date goes in Legend
                        marker=dict(color=colors[i], size=6, symbol="circle"),
                    ))

    # Chart Layout
    fig.update_layout(
        title=f"Predicted Seasonal Path: {ticker} (Using Data < {reference_year})",
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
    # DETAILED YEAR-BY-YEAR TABLE
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader(f"📜 Detailed History: Day #{current_day_count_val} to Fwd Returns")
    st.caption(f"Showing historical outcomes for Day #{current_day_count_val} prior to {reference_year}.")

    if current_day_count_val:
        # Use full dataset to calculate forward returns
        spx_full = spx.copy()
        
        spx_full['Fwd_5d'] = spx_full['Close'].shift(-5) / spx_full['Close'] - 1
        spx_full['Fwd_10d'] = spx_full['Close'].shift(-10) / spx_full['Close'] - 1
        spx_full['Fwd_21d'] = spx_full['Close'].shift(-21) / spx_full['Close'] - 1

        # Extract only the specific trading day from past years
        daily_snapshots = spx_full[spx_full['day_count'] == current_day_count_val].copy()

        # FILTER: STRICTLY BEFORE REFERENCE YEAR
        # If I am in 2025 looking at 2018, I do not want to see 2019, 2020 etc in the table.
        # I also do NOT want to see 2018 in the table (as per "include data up to end of year BEFORE").
        display_df = daily_snapshots[daily_snapshots['year'] < reference_year].copy()

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

            # --- SUMMARY STATISTICS TABLE ---
            st.markdown("##### 🎯 Fwd Return Statistics (Historical)")
            
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

            stats_df = pd.DataFrame([stats_all, stats_cycle], index=[f"All History (<{reference_year})", cycle_row_name])
            
            ordered_cols = ["n"]
            for d in [5, 10, 21]:
                ordered_cols.extend([f"{d}_median", f"{d}_mean", f"{d}_pospct"])
            stats_df = stats_df[ordered_cols]

            # Custom Formatting Function for Pos Pct
            def color_pos_pct(val):
                if pd.isna(val): return ''
                if val > 80:
                    return 'color: #90ee90; font-weight: bold;' # Light Green
                elif val < 25:
                    return 'color: #ffcccb; font-weight: bold;' # Light Red
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

            # --- MAIN DATAFRAME RENDER ---
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
            st.warning(f"No historical data available prior to {reference_year} for Day #{current_day_count_val}.")

# -----------------------------------------------------------------------------
# APP ENTRY POINT
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Seasonality Analysis")
    st.title("📊 Presidential Cycle Seasonality")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        ticker = st.text_input("Ticker", value="SPY").upper()
    with col2:
        cycle_label = st.selectbox(
            "Cycle Type",
            ["All Years", "Election", "Pre-Election", "Post-Election", "Midterm"],
            index=3
        )
    with col3:
        # Time Travel Feature
        current_year = dt.date.today().year
        reference_year = st.number_input("View Year (Time Travel)", min_value=1950, max_value=current_year + 1, value=current_year)
    with col4:
        st.write("") 
        show_all_years_line = st.checkbox("Overlay 'All Years' Avg", value=False)

    if st.button("Run Analysis", type="primary", use_container_width=True):
        try:
            with st.spinner("Fetching data and calculating cycle stats..."):
                seasonals_chart(ticker, cycle_label, reference_year, show_all_years_line)
        except Exception as e:
            st.error(f"Error generating chart: {e}")

if __name__ == "__main__":
    main()
