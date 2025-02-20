import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objs as go
import streamlit as st
from datetime import date, timedelta

def compute_atr(df, window=14):
    df = df.copy()
    df["previous_close"] = df["Close"].shift(1)
    df["TR"] = df[["High", "previous_close"]].max(axis=1) - df[["Low", "previous_close"]].min(axis=1)
    df["ATR"] = df["TR"].rolling(window=window).mean()
    df["ATR%"] = (df["ATR"] / df["Close"]) * 100
    return df

def summarize_data(df, include_atr=True):
    if df.empty:
        # Return with ATR and % Pos if include_atr is True, else still return these columns as NaN
        if include_atr:
            return {
                "Avg Return (%)": np.nan, 
                "Median Daily Return (%)": np.nan, 
                "Avg ATR%": np.nan,
                "% Pos": np.nan
            }
        else:
            return {
                "Avg Return (%)": np.nan, 
                "Median Daily Return (%)": np.nan, 
                "Avg ATR%": np.nan,  # even if include_atr=False, we return the column as NaN for consistency
                "% Pos": np.nan
            }
    daily_returns = df["log_return"] * 100  # convert to percentage
    pos_percentage = (daily_returns > 0).sum() / len(daily_returns) * 100
    if include_atr:
        return {
            "Avg Return (%)": daily_returns.mean(),
            "Median Daily Return (%)": daily_returns.median(),
            "Avg ATR%": df["ATR%"].mean(),
            "% Pos": pos_percentage
        }
    else:
        # Even if include_atr=False, let's still return ATR% as NaN so all tables have same columns
        return {
            "Avg Return (%)": daily_returns.mean(),
            "Median Daily Return (%)": daily_returns.median(),
            "Avg ATR%": np.nan,  # no ATR in original daily request, but we add it as NaN for consistency
            "% Pos": pos_percentage
        }

def get_current_trading_info():
    # Determine today's trading day_of_month and week_of_month_5day
    today = dt.date.today()
    start_of_month = dt.date(today.year, today.month, 1)
    current_data = yf.download("SPY", start=start_of_month, end=today + timedelta(days=1)) 
    if isinstance(current_data.columns, pd.MultiIndex):  
        current_data.columns = current_data.columns.get_level_values(0)
    if not current_data.empty:
        current_data["trading_day_of_month"] = np.arange(1, len(current_data) + 1)
        current_data["week_of_month_5day"] = (current_data["trading_day_of_month"] - 1) // 5 + 1
        # Merge week 5 into week 4
        current_data.loc[current_data["week_of_month_5day"] > 4, "week_of_month_5day"] = 4
        current_trading_day_of_month = current_data["trading_day_of_month"].iloc[-1]
        current_week_of_month = current_data["week_of_month_5day"].iloc[-1]
        return current_trading_day_of_month, current_week_of_month
    else:
        return None, None

def seasonals_chart(ticker, cycle_label):
    cycle_start_mapping = {
        "Election": 1952,
        "Pre-Election": 1951,
        "Post-Election": 1953,
        "Midterm": 1950
    }
    cycle_start = cycle_start_mapping[cycle_label]

    end_date = dt.datetime(2024, 12, 30)
    this_yr_end = dt.date.today() + timedelta(days=1)
    spx = yf.download(ticker, period="max", end=end_date)

    if spx.empty:
        st.error(f"No data found for {ticker}.")
        return
    
    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = spx.columns.get_level_values(0)

    spx["log_return"] = np.log(spx["Close"] / spx["Close"].shift(1))
    spx["year"] = spx.index.year
    spx["month"] = spx.index.month
    spx["trading_day_of_month"] = spx.groupby([spx.index.year, spx.index.month]).cumcount() + 1
    spx["week_of_month_5day"] = (spx["trading_day_of_month"] - 1) // 5 + 1

    # Only include data from years in the selected presidential cycle
    years_in_cycle = [cycle_start + i * 4 for i in range(19)]
    cycle_data = spx[spx["year"].isin(years_in_cycle)]
    cycle_data = compute_atr(cycle_data)

    cycle_data.loc[cycle_data["week_of_month_5day"] > 4, "week_of_month_5day"] = 4

    now = dt.date.today()
    current_month = now.month
    next_month = current_month + 1 if current_month < 12 else 1
    current_year = now.year
    current_day_of_month, current_week_of_month = get_current_trading_info()

    # --- PLOT THE CHART (Always) ---
    cycle_data["day_count"] = cycle_data.groupby("year").cumcount() + 1
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

    current_year_data = yf.download(ticker, start=dt.datetime(this_yr_end.year, 1, 1), end=this_yr_end)
    if not current_year_data.empty:
        current_year_data["log_return"] = np.log(current_year_data["Close"] / current_year_data["Close"].shift(1))
        this_year_path = current_year_data["log_return"].cumsum().apply(np.exp) - 1
        fig.add_trace(go.Scatter(
            x=avg_path.index,
            y=this_year_path.values,
            mode="lines",
            name="This Year",
            line=dict(color="green")
        ))

    fig.update_layout(
        title=f"{ticker} - {cycle_label} Cycle Average",
        xaxis_title="Trading Day",
        yaxis_title="Cumulative Return",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig)

    # --- COMPUTE & SHOW HIGH-LEVEL SUMMARY TABLE (Always) ---
    summary_rows = []
    timeframes = {
        "This Month": (current_month, current_year),
        "Next Month": (next_month, current_year),
        "This Week": (current_month, current_week_of_month),
        "Next Week": (next_month, 1)  # Assume next month starts with week 1
    }

    for label, (month, week) in timeframes.items():
        if "Month" in label:
            time_data = cycle_data[(cycle_data["month"] == month) & (cycle_data["year"].isin(years_in_cycle))]
        else:
            time_data = cycle_data[
                (cycle_data["month"] == month) &
                (cycle_data["week_of_month_5day"] == week) &
                (cycle_data["year"].isin(years_in_cycle))
            ]

        if not time_data.empty:
            stats = summarize_data(time_data, include_atr=False)
            sample_size = len(time_data)  # Count of observations
            summary_rows.append([label, stats["Avg Return (%)"], stats["Median Daily Return (%)"], stats["% Pos"], sample_size])
        else:
            summary_rows.append([label, np.nan, np.nan, np.nan, np.nan])

    high_level_df = pd.DataFrame(summary_rows, columns=["Timeframe", "Mean", "Median", "% Pos", "Sample Size"]).set_index("Timeframe")

    st.subheader("High-Level Summary")
    st.dataframe(high_level_df.style.format({
        "Mean": "{:.1f}%", 
        "Median": "{:.1f}%", 
        "% Pos": "{:.1f}%", 
        "Sample Size": "{:.0f}"  # No decimals for count
    }))

    # Print current trading day/week of the month at the end
    st.write(f"Today is the {current_day_of_month}-th trading day of this month and we are currently in week {current_week_of_month} of this month.")



st.title("Presidential Cycle Seasonality Chart")

ticker = st.text_input("Enter a stock ticker:", value="AAPL")
cycle_label = st.selectbox(
    "Select the presidential cycle type:",
    ["Election", "Pre-Election", "Post-Election", "Midterm"],
    index=2  # "Post-Election" is the 3rd item (zero-based index)
)

if st.button("Plot"):
    try:
        seasonals_chart(ticker, cycle_label)
    except Exception as e:
        st.error(f"Error generating chart: {e}")
