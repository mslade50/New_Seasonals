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

def get_current_trading_info():
    """
    Returns the current trading day of the month and week of month 
    based on SPY data for the current month.
    """
    today = dt.date.today()
    start_of_month = dt.date(today.year, today.month, 1)
    # Fetch a buffer to ensure we catch today if market is open/closed
    current_data = yf.download("SPY", start=start_of_month, end=today + timedelta(days=1), progress=False) 
    
    if isinstance(current_data.columns, pd.MultiIndex):  
        current_data.columns = current_data.columns.get_level_values(0)
        
    if not current_data.empty:
        current_data["trading_day_of_month"] = np.arange(1, len(current_data) + 1)
        current_data["week_of_month_5day"] = (current_data["trading_day_of_month"] - 1) // 5 + 1
        current_data.loc[current_data["week_of_month_5day"] > 4, "week_of_month_5day"] = 4
        
        current_trading_day_of_month = current_data["trading_day_of_month"].iloc[-1]
        current_week_of_month = current_data["week_of_month_5day"].iloc[-1]
        return current_trading_day_of_month, current_week_of_month
    else:
        return None, None

# -----------------------------------------------------------------------------
# MAIN CHART LOGIC
# -----------------------------------------------------------------------------

def seasonals_chart(ticker, cycle_label, show_all_years_line=False):
    cycle_start_mapping = {
        "Election": 1952,
        "Pre-Election": 1951,
        "Post-Election": 1953,
        "Midterm": 1950
    }

    # Data Fetching
    end_date = dt.datetime(2024, 12, 30)
    this_yr_end = dt.date.today() + timedelta(days=1)
    spx = yf.download(ticker, period="max", end=end_date, progress=False)

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
    
    # Calculate Day of Year (Trading Day Count)
    spx["day_count"] = spx.groupby("year").cumcount() + 1

    # Cycle Filtering
    if cycle_label == "All Years":
        cycle_data = spx.copy()
    else:
        cycle_start = cycle_start_mapping[cycle_label]
        years_in_cycle = [cycle_start + i * 4 for i in range(25)] # Extended range just in case
        cycle_data = spx[spx["year"].isin(years_in_cycle)].copy()

    cycle_data = compute_atr(cycle_data)
    cycle_data.loc[cycle_data["week_of_month_5day"] > 4, "week_of_month_5day"] = 4

    # Current Time Info
    now = dt.date.today()
    current_month = now.month
    next_month = current_month + 1 if current_month < 12 else 1
    current_trading_day_of_month, current_week_of_month = get_current_trading_info()

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

    # Optional Overlay: All Years
    if show_all_years_line:
        all_data = spx.copy()
        all_avg_path = (
            all_data.groupby("day_count")["log_return"]
            .mean()
            .cumsum()
            .apply(np.exp) - 1
        )
        fig.add_trace(go.Scatter(
            x=all_avg_path.index,
            y=all_avg_path.values,
            mode="lines",
            name="All Years Avg Path",
            line=dict(color="lightblue")
        ))

    # Current Year Data & Logic
    current_year_data = yf.download(ticker, start=dt.datetime(now.year, 1, 1), end=this_yr_end, progress=False)
    
    if isinstance(current_year_data.columns, pd.MultiIndex):
        current_year_data.columns = current_year_data.columns.get_level_values(0)

    # Determine Current Trading Day Count (e.g., Day 230 of the year)
    current_day_count_val = len(current_year_data) if not current_year_data.empty else None

    # Plot Current Day Marker
    avg_path_y_value = avg_path.get(current_day_count_val)
    if avg_path_y_value is not None:
        fig.add_trace(go.Scatter(
            x=[current_day_count_val],
            y=[avg_path_y_value],
            mode="markers",
            name="Current Day on Avg Path",
            marker=dict(color="white", size=7),
            showlegend=False
        ))
    
    # Plot This Year's Path
    if not current_year_data.empty:
        current_year_data["log_return"] = np.log(current_year_data["Close"] / current_year_data["Close"].shift(1))
        this_year_path = current_year_data["log_return"].cumsum().apply(np.exp) - 1
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(this_year_path) + 1),
            y=this_year_path.values,
            mode="lines",
            name=f"{now.year} (YTD)",
            line=dict(color="green", width=2)
        ))

    # --- 5D Concordance Logic (Existing) ---
    annotation_text = "Not enough data"
    if not current_year_data.empty:
        min_len = min(len(this_year_path), len(avg_path))
        if min_len >= 10:
            try:
                actual_5d = this_year_path.rolling(5).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
                avg_5d = avg_path.rolling(5).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
                
                comparison_df = pd.DataFrame({
                    "actual_5d": actual_5d.iloc[:min_len].values,
                    "avg_5d": avg_5d.iloc[:min_len].values
                }).dropna()
                
                # Filter out flat signals
                threshold = 0.001
                comparison_df = comparison_df[
                    (comparison_df["actual_5d"].abs() > threshold) &
                    (comparison_df["avg_5d"].abs() > threshold)
                ]
                
                if not comparison_df.empty:
                    concordance = (np.sign(comparison_df["actual_5d"]) == np.sign(comparison_df["avg_5d"])).mean()
                    annotation_text = f"5D Concordance: {concordance:.0%}"
            except Exception as e:
                annotation_text = "Concordance Error"

    # Chart Layout
    fig.update_layout(
        title=f"{ticker} - {cycle_label} Cycle Average",
        xaxis_title="Trading Day of Year",
        yaxis_title="Cumulative Return",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=1.0, y=1.05,
        showarrow=False,
        font=dict(color="white", size=12),
        align="right"
    )

    st.plotly_chart(fig, use_container_width=True)
    # -------------------------------------------------------------------------
    # NEW: DETAILED YEAR-BY-YEAR TABLE WITH HIGHLIGHTING
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader(f"📜 Detailed History: Day #{current_day_count_val} to Fwd Returns")
    st.caption(f"Table lists performance for every year. Rows highlighted in **Gold** indicate **{cycle_label}** years.")

   
    # -------------------------------------------------------------------------
    # NEW: DETAILED YEAR-BY-YEAR TABLE WITH HIGHLIGHTING
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader(f"📜 Detailed History: Day #{current_day_count_val} to Fwd Returns")
    st.caption(f"Table lists performance for every year. **Cycle years ({cycle_label})** are highlighted in the Year column.")

    if current_day_count_val:
        # 1. Calculate Forward Returns on the ENTIRE dataset (spx)
        spx_full = spx.copy()
        
        # Calculate returns looking forward from the current day count
        spx_full['Fwd_5d'] = spx_full['Close'].shift(-5) / spx_full['Close'] - 1
        spx_full['Fwd_10d'] = spx_full['Close'].shift(-10) / spx_full['Close'] - 1
        spx_full['Fwd_21d'] = spx_full['Close'].shift(-21) / spx_full['Close'] - 1

        # 2. Filter for the specific trading day matching TODAY
        daily_snapshots = spx_full[spx_full['day_count'] == current_day_count_val].copy()

        if not daily_snapshots.empty:
            # 3. Prepare the Dataframe for display
            display_df = daily_snapshots[['year', 'Fwd_5d', 'Fwd_10d', 'Fwd_21d']].copy()
            
            # Convert to percentages
            display_df['Fwd_5d'] = display_df['Fwd_5d'] * 100
            display_df['Fwd_10d'] = display_df['Fwd_10d'] * 100
            display_df['Fwd_21d'] = display_df['Fwd_21d'] * 100

            # Sort by Year Descending
            display_df = display_df.sort_values('year', ascending=False)

            # 4. Define Cycle Years for Highlighting
            if cycle_label != "All Years":
                start_yr = cycle_start_mapping.get(cycle_label)
                highlight_years = [start_yr + i * 4 for i in range(30)] 
            else:
                highlight_years = []

            # 5. Styling Function (Apply ONLY to 'year' column)
            def highlight_year_cell(val):
                if val in highlight_years:
                    # distinct color for the Year cell to indicate cycle match
                    return 'background-color: #d4af37; color: black; font-weight: bold;' 
                return ''

            # 6. Render Dataframe
            st.dataframe(
                display_df.style
                .format({
                    "year": "{:.0f}",
                    "Fwd_5d": "{:+.2f}%",
                    "Fwd_10d": "{:+.2f}%",
                    "Fwd_21d": "{:+.2f}%"
                })
                # A. Apply Gradient to Return Columns (Works on ALL rows)
                .background_gradient(subset=["Fwd_5d", "Fwd_10d", "Fwd_21d"], cmap="RdYlGn", vmin=-5, vmax=5)
                # B. Apply Highlight to Year Column ONLY
                .map(highlight_year_cell, subset=['year']),
                use_container_width=True,
                height=500,
                hide_index=True
            )
        else:
            st.warning(f"No historical data available for Day #{current_day_count_val}.")

    # -------------------------------------------------------------------------
    # EXISTING SUMMARY TABLES
    # -------------------------------------------------------------------------
    st.divider()
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📅 Periodic Summary")
        summary_rows = []
        timeframes = {
            "This Month": (current_month),
            "Next Month": (next_month),
            "This Week": (current_month, current_week_of_month),
            "Next Week": (next_month, 1)
        }

        for label, params in timeframes.items():
            if "Month" in label:
                month = params
                time_data = cycle_data[(cycle_data["month"] == month)]
            else:
                month, week = params
                time_data = cycle_data[
                    (cycle_data["month"] == month) & 
                    (cycle_data["week_of_month_5day"] == week)
                ]

            if not time_data.empty:
                stats = summarize_data(time_data, include_atr=False)
                sample_size = time_data["year"].nunique()
                summary_rows.append([label, stats["Avg Return (%)"], stats["Median Daily Return (%)"], sample_size])
            else:
                summary_rows.append([label, np.nan, np.nan, np.nan])

        high_level_df = pd.DataFrame(summary_rows, columns=["Timeframe", "Mean", "Median", "Count"]).set_index("Timeframe")

        st.dataframe(high_level_df.style.format({
            "Mean": "{:.1f}%", 
            "Median": "{:.1f}%", 
            "Count": "{:.0f}"
        }), use_container_width=True)

    with c2:
        st.subheader("🗓️ Monthly Breakdown")
        month_return_rows = []
        for month in [current_month, next_month]:
            time_data = cycle_data[(cycle_data["month"] == month)]
            if not time_data.empty:
                monthly_returns = time_data.groupby("year")["log_return"].sum() * 100
                for year, ret in monthly_returns.items():
                    month_return_rows.append([year, month, ret])

        if month_return_rows:
            month_returns_df = pd.DataFrame(month_return_rows, columns=["Year", "Month", "Return (%)"]).sort_values(by=["Year", "Month"])
            month_returns_df["Month"] = month_returns_df["Month"].apply(lambda x: dt.date(1900, x, 1).strftime('%B'))
            st.dataframe(
                month_returns_df.style.format({"Return (%)": "{:.1f}%"})
                .background_gradient(subset=["Return (%)"], cmap="RdYlGn", vmin=-5, vmax=5),
                use_container_width=True,
                height=300
            )
        else:
            st.write("No historical data found for selected months.")

# -----------------------------------------------------------------------------
# APP ENTRY POINT
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Seasonality Analysis")
    st.title("📊 Presidential Cycle Seasonality")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        ticker = st.text_input("Ticker", value="SPY").upper()
    with col2:
        cycle_label = st.selectbox(
            "Cycle Type",
            ["All Years", "Election", "Pre-Election", "Post-Election", "Midterm"],
            index=3
        )
    with col3:
        st.write("") # Spacer
        show_all_years_line = st.checkbox("Overlay 'All Years' Average", value=False)

    if st.button("Run Analysis", type="primary", use_container_width=True):
        try:
            with st.spinner("Fetching data and calculating cycle stats..."):
                seasonals_chart(ticker, cycle_label, show_all_years_line)
        except Exception as e:
            st.error(f"Error generating chart: {e}")

if __name__ == "__main__":
    main()
