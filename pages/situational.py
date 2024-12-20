import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objs as go

# Load the dates CSV from the repo
@st.cache
def load_event_dates():
    url = "https://raw.githubusercontent.com/mslade50/New_Seasonals/main/market_dates.csv"
    return pd.read_csv(url)

# Function to calculate metrics
def calculate_metrics(data, event_dates, shift_days=0):
    # Ensure event_dates are sorted and unique
    event_dates = sorted(event_dates.dropna().unique())
    
    # Apply shifting
    event_dates = pd.to_datetime(event_dates) + pd.to_timedelta(shift_days, unit="D")

    # Filter the data for relevant event dates
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date'])
    event_data = data[data['Date'].isin(event_dates)]

    # Initialize results
    avg_return = None
    avg_daily_range = None
    avg_daily_range_14_ratio = None
    avg_5d_fwd_return = None
    individual_returns = None
    backtest_table = None

    if not event_data.empty:
        # Create a column for the previous day's close
        data['Previous Close'] = data['Close'].shift(1)

        # Filter again to align previous close
        event_data = data[data['Date'].isin(event_dates)].dropna(subset=['Previous Close'])

        # Calculate returns relative to the previous day
        event_data['1D Return'] = (event_data['Close'] - event_data['Previous Close']) / event_data['Previous Close'] * 100

        # Calculate daily range as a percentage of close
        event_data['Daily Range'] = (event_data['High'] - event_data['Low']) / event_data['Close'] * 100

        # Calculate the rolling 14-day average of the daily range
        data['14D Avg Range'] = data['Daily Range'].rolling(window=14, min_periods=1).mean().shift(1)  # Shift back 1 day

        # Merge the 14-day average into event_data
        event_data = event_data.merge(data[['Date', '14D Avg Range']], on='Date', how='left')
        event_data['Range/14D Avg'] = event_data['Daily Range'] / event_data['14D Avg Range']

        # Calculate the 5-day forward return
        event_data['5D Fwd Return'] = (
            event_data['Close'].shift(-5) - event_data['Close']
        ) / event_data['Close'] * 100

        # Compute averages
        avg_return = event_data['1D Return'].mean()
        avg_daily_range = event_data['Daily Range'].mean()
        avg_daily_range_14_ratio = event_data['Range/14D Avg'].mean()
        avg_5d_fwd_return = event_data['5D Fwd Return'].mean()

        # Get individual returns for plotting
        individual_returns = event_data[['Date', '1D Return']].dropna()

        # Backtest calculations
        backtest_table = pd.DataFrame()
        for days in [1, 3, 5, 10]:
            backtest_table[f"{days}D Return"] = (
                (event_data['Close'].shift(-days) - event_data['Close']) / event_data['Close'] * 100
            ).dropna()

    return avg_return, avg_daily_range, avg_daily_range_14_ratio, avg_5d_fwd_return, individual_returns, backtest_table

# Main Streamlit app
def main():
    st.title("Market Event Analysis Dashboard")

    # Load event dates
    dates_df = load_event_dates()

    # User input: Event type
    event_type = st.selectbox("Select Event Type", options=list(dates_df['Event'].unique()) + ["Opex", "First Day of Month", "Last Day of Month"])

    # User input: Month
    month = st.selectbox(
        "Select Month (Optional)", 
        options=["All"] + list(range(1, 13)),
        format_func=lambda x: "All" if x == "All" else pd.to_datetime(f"2022-{x}-01").strftime('%B')
    )

    # User input: Year of Presidential Cycle
    cycle_year = st.selectbox(
        "Select Year of Presidential Cycle (Optional)", 
        options=["All", "Election", "Post-Election", "Midterm", "Pre-Election"]
    )

    # User input: Stock ticker
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, SPY):")

    # User input: Shift days
    shift_days = st.number_input("Shift Event Dates by Trading Days", min_value=-10, max_value=10, value=0)

    # Trigger calculation
    if st.button("Calculate Metrics"):
        if ticker:
            # Load stock data
            data = yf.download(ticker, start="1990-01-01", progress=False)
            data.index = pd.to_datetime(data.index)

            # Filter for the selected event type
            if event_type in ["Opex", "First Day of Month", "Last Day of Month"]:
                if event_type == "Opex":
                    dates_df = dates_df[dates_df['Event'] == "Opex"]
                elif event_type == "First Day of Month":
                    dates_df = pd.DataFrame({"Date": data.index[data.index.is_month_start]})
                elif event_type == "Last Day of Month":
                    dates_df = pd.DataFrame({"Date": data.index[data.index.is_month_end]})
            else:
                dates_df = dates_df[dates_df['Event'] == event_type]

            # Apply month filter
            if month != "All":
                dates_df = dates_df[dates_df['Date'].apply(lambda x: pd.to_datetime(x).month) == month]

            # Apply year of presidential cycle filter
            if cycle_year != "All":
                dates_df = dates_df[dates_df['Cycle'] == cycle_year]

            # Extract relevant dates
            event_dates = pd.to_datetime(dates_df['Date'])

            # Calculate metrics
            avg_return, avg_daily_range, avg_daily_range_14_ratio, avg_5d_fwd_return, individual_returns, backtest_table = calculate_metrics(data, event_dates, shift_days)

            # Display results
            if avg_return is not None:
                st.write(f"### Metrics for {event_type} on {ticker}")
                st.write(f"**Average Return (%)**: {avg_return:.2f}")
                st.write(f"**Average Daily Range (%)**: {avg_daily_range:.2f}")
                st.write(f"**Average Daily Range / 14D Average Ratio**: {avg_daily_range_14_ratio:.2f}")
                st.write(f"**Average 5-Day Forward Return (%)**: {avg_5d_fwd_return:.2f}")

                # Plot bar chart of individual returns
                if not individual_returns.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=individual_returns['Date'],
                        y=individual_returns['1D Return'],
                        name="1-Day Returns",
                        marker=dict(color="blue")
                    ))
                    fig.update_layout(
                        title=f"1-Day Returns for {event_type} Events on {ticker}",
                        xaxis_title="Date",
                        yaxis_title="1-Day Return (%)",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig)

                # Display backtest table
                if backtest_table is not None:
                    st.write("### Backtest Results")
                    st.write(backtest_table.describe())

            else:
                st.error("No matching event dates found in the stock data.")
        else:
            st.error("Please enter a valid stock ticker.")

# Run the app
if __name__ == "__main__":
    main()
