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

def get_current_trading_info():
    today = dt.date.today()
    start_of_month = dt.date(today.year, today.month, 1)
    
    try:
        current_data = yf.download("SPY", start=start_of_month, end=today + timedelta(days=1))
        
        if current_data.empty:
            st.error("Error: No data retrieved from Yahoo Finance.")
            return None, None

        # Ensure correct column format if MultiIndex is returned
        if isinstance(current_data.columns, pd.MultiIndex):
            current_data.columns = current_data.columns.get_level_values(0)

        # Add trading day and week of month
        current_data["trading_day_of_month"] = np.arange(1, len(current_data) + 1)
        current_data["week_of_month_5day"] = (current_data["trading_day_of_month"] - 1) // 5 + 1
        current_data.loc[current_data["week_of_month_5day"] > 4, "week_of_month_5day"] = 4

        current_trading_day_of_month = current_data["trading_day_of_month"].iloc[-1]
        current_week_of_month = current_data["week_of_month_5day"].iloc[-1]
        return current_trading_day_of_month, current_week_of_month

    except Exception as e:
        st.error(f"Error fetching SPY data: {e}")
        return None, None

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

    try:
        spx = yf.download(ticker, period="max", end=end_date)
        
        if spx.empty:
            st.error(f"No data found for {ticker}.")
            return

        # Handle MultiIndex columns
        if isinstance(spx.columns, pd.MultiIndex):
            spx.columns = spx.columns.get_level_values(0)

        # Add necessary columns
        spx["log_return"] = np.log(spx["Close"] / spx["Close"].shift(1))
        spx["year"] = spx.index.year
        spx["month"] = spx.index.month
        spx["trading_day_of_month"] = spx.groupby([spx.index.year, spx.index.month]).cumcount() + 1
        spx["week_of_month_5day"] = (spx["trading_day_of_month"] - 1) // 5 + 1

        years_in_cycle = [cycle_start + i * 4 for i in range(19)]
        cycle_data = spx[spx["year"].isin(years_in_cycle)]
        cycle_data = compute_atr(cycle_data)

        # Fix week merging
        cycle_data.loc[cycle_data["week_of_month_5day"] > 4, "week_of_month_5day"] = 4

        current_year_data = yf.download(ticker, start=dt.datetime(this_yr_end.year, 1, 1), end=this_yr_end)
        
        if current_year_data.empty:
            st.warning(f"Warning: No data available for {ticker} in {this_yr_end.year}.")
            this_year_path = pd.Series(dtype=float)
            current_trading_day = None
        else:
            current_year_data["log_return"] = np.log(current_year_data["Close"] / current_year_data["Close"].shift(1))
            this_year_path = current_year_data["log_return"].cumsum().apply(np.exp) - 1
            current_trading_day = len(current_year_data)

        # Compute cycle path
        cycle_data["day_count"] = cycle_data.groupby("year").cumcount() + 1
        avg_path = cycle_data.groupby("day_count")["log_return"].mean().cumsum().apply(np.exp) - 1

        # Plot chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=avg_path.index, y=avg_path.values, mode="lines", name=f"Avg Path ({cycle_label})", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=avg_path.index, y=this_year_path.values, mode="lines", name="This Year", line=dict(color="green")))

        if current_trading_day is not None and current_trading_day in avg_path.index:
            avg_path_y_value = avg_path[current_trading_day]
            fig.add_trace(go.Scatter(x=[current_trading_day], y=[avg_path_y_value], mode="markers", name="Current Day on Avg Path", marker=dict(color="white", size=7), showlegend=False))

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

    except Exception as e:
        st.error(f"Error generating chart: {e}")

st.title("Presidential Cycle Seasonality Chart")

ticker = st.text_input("Enter a stock ticker:", value="AAPL")
cycle_label = st.selectbox("Select the presidential cycle type:", ["Election", "Pre-Election", "Post-Election", "Midterm"])
show_tables = st.sidebar.checkbox("Show Summary Tables")

if st.button("Plot"):
    try:
        seasonals_chart(ticker, cycle_label, show_tables)
    except Exception as e:
        st.error(f"Error generating chart: {e}")
