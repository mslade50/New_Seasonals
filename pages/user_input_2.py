import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objs as go
import streamlit as st
from datetime import date
from datetime import timedelta

def seasonals_chart(ticker, cycle_label):
    """
    Plot the average historical path of the ticker for the selected cycle type 
    using trading day indexing and log returns, and compare it to the current year's path.

    :param ticker: Stock ticker symbol.
    :param cycle_label: The presidential cycle type (Election, Pre-Election, Post-Election, Midterm).
    """

    # Map cycle_label to cycle_start
    cycle_start_mapping = {
        "Election": 1952,
        "Pre-Election": 1951,
        "Post-Election": 1953,
        "Midterm": 1950
    }
    cycle_start = cycle_start_mapping[cycle_label]

    # Fetch historical data for the ticker
    end_date = dt.datetime(2023, 12, 30)
    this_yr_end = dt.date.today() + timedelta(days=1)
    spx1 = yf.Ticker(ticker)
    spx = spx1.history(period="max", end=end_date)

    # Ensure data exists
    if spx.empty:
        st.error(f"No data found for {ticker}.")
        return

    # Calculate log returns and assign trading day index
    spx["log_return"] = np.log(spx["Close"] / spx["Close"].shift(1))
    spx["trading_day"] = spx.groupby(spx.index.year).cumcount() + 1
    spx["year"] = spx.index.year

    # Define years in the selected presidential cycle
    years_in_cycle = [cycle_start + i * 4 for i in range(19)]

    # Filter data for the selected cycle years
    cycle_data = spx[spx["year"].isin(years_in_cycle)]

    # Calculate average path for the selected cycle years
    avg_path = (
        cycle_data.groupby("trading_day")["log_return"]
        .mean()
        .cumsum()
        .apply(np.exp) - 1
    )

    # Get the current year's data
    current_year_data = yf.download(ticker, start=dt.datetime(this_yr_end.year, 1, 1), end=this_yr_end)

    if not current_year_data.empty:
        current_year_data["log_return"] = np.log(current_year_data["Close"] / current_year_data["Close"].shift(1))
        this_year_path = (
            current_year_data["log_return"]
            .cumsum()
            .apply(np.exp) - 1
        )
        current_trading_day = len(current_year_data)
        current_ytd_value = this_year_path.iloc[-1]
    else:
        this_year_path = pd.Series(dtype=float)
        current_trading_day = None
        current_ytd_value = None

    # Plot the average path, this year's path, and the white dot for the current trading day
    fig = go.Figure()

    # Add average path
    fig.add_trace(go.Scatter(
        x=avg_path.index, 
        y=avg_path.values, 
        mode="lines", 
        name=f"Avg Path ({cycle_label})",
        line=dict(color="yellow")
    ))

    # Add this year's path if it exists
    if not this_year_path.empty:
        fig.add_trace(go.Scatter(
            x=this_year_path.index, 
            y=this_year_path.values, 
            mode="lines", 
            name="This Year",
            line=dict(color="green", width=2)
        ))

    # Add white dot for the current trading day if it exists
    if current_trading_day is not None and current_ytd_value is not None:
        fig.add_trace(go.Scatter(
            x=[current_trading_day], 
            y=[current_ytd_value], 
            mode="markers", 
            name="Current Day",
            marker=dict(color="white", size=10),
            showlegend=False  # Exclude from legend
        ))

    # Update layout
    fig.update_layout(
        title=f"{ticker} - {cycle_label} Cycle Average vs This Year",
        xaxis_title="Trading Day",
        yaxis_title="Cumulative Return",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
    )

    # Remove gridlines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig)




st.title("Presidential Cycle Seasonality Chart")

# User Input for Ticker
ticker = st.text_input("Enter a stock ticker:", value="AAPL")

# Dropdown for Presidential Cycle Type
cycle_label = st.selectbox(
    "Select the presidential cycle type:",
    ["Election", "Pre-Election", "Post-Election", "Midterm"]
)

# Plot Button
if st.button("Plot"):
    try:
        seasonals_chart(ticker, cycle_label)
    except Exception as e:
        st.error(f"Error generating chart: {e}")
