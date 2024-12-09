def seasonals_chart(ticker, cycle_label):
    """
    Plot the average historical path of the ticker for the selected cycle type 
    using trading day indexing and log returns, and compare it to the current year's path.

    :param ticker: Stock ticker symbol.
    :param cycle_label: The presidential cycle type (Election, Pre-Election, Post-Election, Midterm).
    """
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import datetime as dt
    import plotly.graph_objs as go
    import streamlit as st

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
    this_yr_end = dt.date.today()
    spx1 = yf.Ticker(ticker)
    spx = spx1.history(period="max", end=end_date)

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

    # Get this year's path
    this_year = spx[spx["year"] == this_yr_end.year]
    this_year_path = (
        this_year.groupby("trading_day")["log_return"]
        .sum()
        .cumsum()
        .apply(np.exp) - 1
    )

    # Plot the average path vs this year's path
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=avg_path.index, y=avg_path.values, mode="lines", name=f"Avg Path ({cycle_label})"))
    fig.add_trace(go.Scatter(x=this_year_path.index, y=this_year_path.values, mode="lines", name="This Year"))

    fig.update_layout(
        title=f"{ticker} - {cycle_label} Cycle Average vs This Year",
        xaxis_title="Trading Day",
        yaxis_title="Cumulative Return",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    )

    st.plotly_chart(fig)
