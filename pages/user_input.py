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

def get_trading_info_from_date(target_date):
    """
    Returns the trading day of the month and week of month 
    based on SPY data for the month of the target_date.
    """
    start_of_month = dt.date(target_date.year, target_date.month, 1)
    # Fetch a buffer
    current_data = yf.download("SPY", start=start_of_month, end=target_date + timedelta(days=5), progress=False) 
    
    if isinstance(current_data.columns, pd.MultiIndex):  
        current_data.columns = current_data.columns.get_level_values(0)
        
    if not current_data.empty:
        current_data["trading_day_of_month"] = np.arange(1, len(current_data) + 1)
        current_data["week_of_month_5day"] = (current_data["trading_day_of_month"] - 1) // 5 + 1
        current_data.loc[current_data["week_of_month_5day"] > 4, "week_of_month_5day"] = 4
        
        # Find row for target date (or nearest previous if weekend/holiday provided, though we usually pass trading days)
        # For simplicity, we take the last available row if date matches or is after
        return current_data["trading_day_of_month"].iloc[-1], current_data["week_of_month_5day"].iloc[-1]
    else:
        return None, None

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
    # Allow fetching slightly past today to ensure we have current data
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
    
    # Calculate Day of Year (Trading Day Count)
    spx["day_count"] = spx.groupby("year").cumcount() + 1

    # -------------------------------------------------------------------------
    # TIME TRAVEL LOGIC
    # -------------------------------------------------------------------------
    # 1. Historical Data (For calculating Averages): Strictly < Reference Year
    historical_pool = spx[spx["year"] < reference_year].copy()
    
    # 2. "Current" Data (For plotting the green line): == Reference Year
    ref_year_data = spx[spx["year"] == reference_year].copy()

    # Cycle Filtering on HISTORICAL POOL
    if cycle_label == "All Years":
        cycle_data = historical_pool.copy()
    else:
        cycle_start = cycle_start_mapping[cycle_label]
        # We only want years that are in the cycle AND in the historical pool
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

    # Plot Reference Year Path (The "Current" Line)
    # Determine Current Trading Day Count based on the Reference Year data
    current_day_count_val = None
    last_ref_date = None

    if not ref_year_data.empty:
        ref_year_data["log_return"] = np.log(ref_year_data["Close"] / ref_year_data["Close"].shift(1))
        this_year_path = ref_year_data["log_return"].cumsum().apply(np.exp) - 1
        
        current_day_count_val = len(ref_year_data)
        last_ref_date = ref_year_data.index[-1]

        fig.add_trace(go.Scatter(
            x=np.arange(1, len(this_year_path) + 1),
            y=this_year_path.values,
            mode="lines",
            name=f"{reference_year} Path",
            line=dict(color="green", width=2)
        ))

        # Plot Current Day Marker
        avg_path_y_value = avg_path.get(current_day_count_val)
        if avg_path_y_value is not None:
            fig.add_trace(go.Scatter(
                x=[current_day_count_val],
                y=[avg_path_y_value],
                mode="markers",
                name="Current Day on Avg Path",
                marker=dict(color="white", size=8, line=dict(width=1, color="black")),
                showlegend=False
            ))

            # ---------------------------------------------------------------------
            # PROJECTIONS: T+5, T+10, T+21
            # ---------------------------------------------------------------------
            # We generate future BUSINESS days starting from the last date of ref data
            # This handles "Time Travel" correctly (showing dates relative to that year)
            future_bdays = pd.bdate_range(start=last_ref_date, periods=25) 
            # periods=25 gives us enough buffer for T+21. Index 0 is Start Date.
            
            projection_offsets = [5, 10, 21]
            
            for offset in projection_offsets:
                target_idx = current_day_count_val + offset
                
                # Check if target is within the available avg_path x-axis (approx 252 days)
                if target_idx in avg_path.index:
                    proj_y = avg_path.get(target_idx)
                    
                    # Calculate Real Date
                    # future_bdays[offset] is the T+offset date
                    proj_date_obj = future_bdays[offset]
                    proj_date_str = proj_date_obj.strftime("%b %d")

                    fig.add_trace(go.Scatter(
                        x=[target_idx],
                        y=[proj_y],
                        mode="markers+text",
                        name=f"T+{offset}",
                        marker=dict(color="yellow", size=6, symbol="circle"),
                        text=f"<b>T+{offset}</b><br>{proj_date_str}",
                        textposition="top center",
                        textfont=dict(size=10, color="yellow"),
                        showlegend=False
                    ))

    # Chart Layout
    fig.update_layout(
        title=f"{ticker} - {cycle_label} Cycle (Ref Year: {reference_year})",
        xaxis_title="Trading Day of Year",
        yaxis_title="Cumulative Return",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True)      
    
    # -------------------------------------------------------------------------
    # DETAILED YEAR-BY-YEAR TABLE
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader(f"📜 Detailed History: Day #{current_day_count_val} to Fwd Returns")
    st.caption(f"Calculated from historical data available up to {reference_year}.")

    if current_day_count_val:
        # Use FULL dataset for forward returns, but filter out 'future' years relative to reference
        # Note: We include the reference year in the table to see how it ended up, 
        # provided we aren't strict about 'blind' testing in the table view. 
        # But usually backtesting implies we only know up to ref year. 
        # Let's show all years so user can see what happened in previous cycles.
        spx_full = spx.copy()
        
        # Forward returns
        spx_full['Fwd_5d'] = spx_full['Close'].shift(-5) / spx_full['Close'] - 1
        spx_full['Fwd_10d'] = spx_full['Close'].shift(-10) / spx_full['Close'] - 1
        spx_full['Fwd_21d'] = spx_full['Close'].shift(-21) / spx_full['Close'] - 1

        # Snapshot at day count
        daily_snapshots = spx_full[spx_full['day_count'] == current_day_count_val].copy()

        if not daily_snapshots.empty:
            display_df = daily_snapshots[['year', 'Fwd_5d', 'Fwd_10d', 'Fwd_21d']].copy()
            
            # Filter: Don't show years > reference_year + 1 (Just to keep it relevant to the 'time travel')
            # Or show all? Let's show all available in the dataset but highlight the cut-off
            
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
            st.markdown("##### 🎯 Fwd Return Statistics")
            
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

            # Important: Stats should likely only include years < reference_year 
            # so the "Stats" aren't "cheating" if this is a backtest tool.
            stats_universe = display_df[display_df['year'] < reference_year]

            stats_all = calculate_stats_row(stats_universe)
            
            if cycle_label != "All Years":
                df_cycle = stats_universe[stats_universe['year'].isin(highlight_years)]
                stats_cycle = calculate_stats_row(df_cycle)
                cycle_row_name = f"{cycle_label} Cycle (Pre-{reference_year})"
            else:
                stats_cycle = stats_all
                cycle_row_name = f"All Years (Pre-{reference_year})"

            stats_df = pd.DataFrame([stats_all, stats_cycle], index=[f"All History (Pre-{reference_year})", cycle_row_name])
            
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
            st.warning(f"No historical data available for Day #{current_day_count_val}.")

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
