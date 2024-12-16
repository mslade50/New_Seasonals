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

def seasonals_chart(ticker, cycle_label, show_tables):
    cycle_start_mapping = {
        "Election": 1952,
        "Pre-Election": 1951,
        "Post-Election": 1953,
        "Midterm": 1950
    }
    cycle_start = cycle_start_mapping[cycle_label]

    end_date = dt.datetime(2023, 12, 30)
    this_yr_end = dt.date.today() + timedelta(days=1)
    spx1 = yf.Ticker(ticker)
    spx = spx1.history(period="max", end=end_date)

    if spx.empty:
        st.error(f"No data found for {ticker}.")
        return

    spx["log_return"] = np.log(spx["Close"] / spx["Close"].shift(1))
    spx["trading_day"] = spx.groupby(spx.index.year).cumcount() + 1
    spx["year"] = spx.index.year
    spx["month"] = spx.index.month
    spx["week_of_month"] = (spx.index.day - 1) // 7 + 1

    years_in_cycle = [cycle_start + i * 4 for i in range(19)]
    cycle_data = spx[spx["year"].isin(years_in_cycle)]
    cycle_data = compute_atr(cycle_data)

    avg_path = (
        cycle_data.groupby("trading_day")["log_return"]
        .mean()
        .cumsum()
        .apply(np.exp) - 1
    )

    current_year_data = yf.download(ticker, start=dt.datetime(this_yr_end.year, 1, 1), end=this_yr_end)
    if not current_year_data.empty:
        current_year_data["log_return"] = np.log(current_year_data["Close"] / current_year_data["Close"].shift(1))
        this_year_path = (
            current_year_data["log_return"]
            .cumsum()
            .apply(np.exp) - 1
        )
        current_trading_day = len(current_year_data)
    else:
        this_year_path = pd.Series(dtype=float)
        current_trading_day = None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=avg_path.index,
        y=avg_path.values,
        mode="lines",
        name=f"Avg Path ({cycle_label})",
        line=dict(color="orange")
    ))

    fig.add_trace(go.Scatter(
        x=avg_path.index,
        y=this_year_path.values,
        mode="lines",
        name="This Year",
        line=dict(color="green")
    ))

    avg_path_y_value = avg_path[current_trading_day] if current_trading_day is not None and current_trading_day in avg_path.index else None
    if avg_path_y_value is not None:
        fig.add_trace(go.Scatter(
            x=[current_trading_day],
            y=[avg_path_y_value],
            mode="markers",
            name="Current Day on Avg Path",
            marker=dict(color="white", size=7),
            showlegend=False
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

    if show_tables:
        # 1) Average and median return by month for each of the cycles, and ATR%
        monthly_data = cycle_data.copy()
        monthly_data["monthly_return"] = monthly_data.groupby(["year", "month"])["log_return"].transform("sum")

        monthly_returns_by_month = monthly_data.groupby("month")["monthly_return"].agg(["mean", "median"])
        monthly_returns_by_month.columns = ["Avg Monthly Return", "Median Monthly Return"]

        atr_by_year_month = monthly_data.groupby(["year", "month"])["ATR%"].mean().reset_index()
        atr_by_month = atr_by_year_month.groupby("month")["ATR%"].mean().to_frame("Avg ATR%")

        summary_table_1 = monthly_returns_by_month.join(atr_by_month, on="month")
        st.subheader("Table 1: Monthly Summary Stats (All Cycle Years)")
        st.dataframe(summary_table_1.style.format("{:.4f}"))

        # 2) Current month/week and next month
        now = dt.date.today()
        current_month = now.month
        current_week_of_month = (now.day - 1) // 7 + 1
        next_month = current_month + 1 if current_month < 12 else 1

        current_week_data = cycle_data[
            (cycle_data["month"] == current_month) &
            (cycle_data["week_of_month"] == current_week_of_month)
        ]
        current_week_summary = {}
        if not current_week_data.empty:
            current_week_summary["Avg Return"] = current_week_data["log_return"].sum()
            current_week_summary["Median Daily Return"] = current_week_data["log_return"].median()
            current_week_summary["Avg ATR%"] = current_week_data["ATR%"].mean()
        else:
            current_week_summary["Avg Return"] = np.nan
            current_week_summary["Median Daily Return"] = np.nan
            current_week_summary["Avg ATR%"] = np.nan

        current_week_df = pd.DataFrame([current_week_summary], index=["Current Week of Current Month"])

        next_month_data = cycle_data[cycle_data["month"] == next_month]
        next_month_summary = {}
        if not next_month_data.empty:
            next_month_summary["Avg Return"] = next_month_data["log_return"].sum()
            next_month_summary["Median Daily Return"] = next_month_data["log_return"].median()
            next_month_summary["Avg ATR%"] = next_month_data["ATR%"].mean()
        else:
            next_month_summary["Avg Return"] = np.nan
            next_month_summary["Median Daily Return"] = np.nan
            next_month_summary["Avg ATR%"] = np.nan

        next_month_df = pd.DataFrame([next_month_summary], index=["Next Month (All Weeks)"])

        st.subheader("Table 2: Current Week of Current Month & Next Month Stats")
        combined_table_2 = pd.concat([current_week_df, next_month_df])
        st.dataframe(combined_table_2.style.format("{:.4f}"))

        # 3) Current month (all weeks)
        current_month_data = cycle_data[cycle_data["month"] == current_month]
        current_month_summary = {}
        if not current_month_data.empty:
            current_month_summary["Avg Monthly Return"] = current_month_data["log_return"].sum()
            current_month_summary["Median Daily Return"] = current_month_data["log_return"].median()
            current_month_summary["Avg ATR%"] = current_month_data["ATR%"].mean()
        else:
            current_month_summary["Avg Monthly Return"] = np.nan
            current_month_summary["Median Daily Return"] = np.nan
            current_month_summary["Avg ATR%"] = np.nan

        current_month_df = pd.DataFrame([current_month_summary], index=["Current Month (All Weeks)"])
        st.subheader("Table 3: Current Month (All Weeks) Stats")
        st.dataframe(current_month_df.style.format("{:.4f}"))

st.title("Presidential Cycle Seasonality Chart")

ticker = st.text_input("Enter a stock ticker:", value="AAPL")
cycle_label = st.selectbox(
    "Select the presidential cycle type:",
    ["Election", "Pre-Election", "Post-Election", "Midterm"]
)

show_tables = st.sidebar.checkbox("Show Summary Tables")

if st.button("Plot"):
    try:
        seasonals_chart(ticker, cycle_label, show_tables)
    except Exception as e:
        st.error(f"Error generating chart: {e}")
