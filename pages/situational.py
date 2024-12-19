import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# Load the dates CSV from the repo
@st.cache
def load_event_dates():
    url = "https://raw.githubusercontent.com/mslade50/New_Seasonals/main/market_dates.csv"
    return pd.read_csv(url)

# Function to calculate metrics
def calculate_metrics(data, event_dates):
    # Filter the data for relevant event dates
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date'])
    event_data = data[data['Date'].isin(event_dates)]

    # Initialize results
    avg_return = None
    avg_daily_range = None
    avg_daily_range_14_ratio = None
    avg_5d_fwd_return = None

    if not event_data.empty:
        # Calculate metrics
        event_data['Daily Range'] = (event_data['High'] - event_data['Low']) / event_data['Close'] * 100
        event_data['14D Avg Range'] = (
            event_data['Daily Range']
            .rolling(window=14, min_periods=1)
            .mean()
        )
        event_data['Range/14D Avg'] = event_data['Daily Range'] / event_data['14D Avg Range']
        event_data['1D Return'] = (event_data['Close'] - event_data['Close'].shift(1)) / event_data['Close'].shift(1) * 100
        event_data['5D Fwd Return'] = (
            event_data['Close'].shift(-5) - event_data['Close']
        ) / event_data['Close'] * 100

        # Compute averages
        avg_return = event_data['1D Return'].mean()
        avg_daily_range = event_data['Daily Range'].mean()
        avg_daily_range_14_ratio = event_data['Range/14D Avg'].mean()
        avg_5d_fwd_return = event_data['5D Fwd Return'].mean()

    return avg_return, avg_daily_range, avg_daily_range_14_ratio, avg_5d_fwd_return

# Main Streamlit app
def main():
    st.title("Market Event Analysis Dashboard")

    # Load event dates
    dates_df = load_event_dates()

    # User input: Event type
    event_type = st.selectbox("Select Event Type", options=dates_df['Event'].unique())

    # User input: Stock ticker
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, SPY):")

    # Trigger calculation
    if st.button("Calculate Metrics"):
        if ticker:
            # Load stock data
            data = yf.download(ticker, start="1990-01-01", progress=False)
            data.index = pd.to_datetime(data.index)

            # Filter for the selected event type
            event_dates = dates_df[dates_df['Event'] == event_type]['Date']
            event_dates = pd.to_datetime(event_dates)

            # Calculate metrics
            avg_return, avg_daily_range, avg_daily_range_14_ratio, avg_5d_fwd_return = calculate_metrics(data, event_dates)

            # Display results
            if avg_return is not None:
                st.write(f"### Metrics for {event_type} on {ticker}")
                st.write(f"**Average Return (%)**: {avg_return:.2f}")
                st.write(f"**Average Daily Range (%)**: {avg_daily_range:.2f}")
                st.write(f"**Average Daily Range / 14D Average Ratio**: {avg_daily_range_14_ratio:.2f}")
                st.write(f"**Average 5-Day Forward Return (%)**: {avg_5d_fwd_return:.2f}")
            else:
                st.error("No matching event dates found in the stock data.")
        else:
            st.error("Please enter a valid stock ticker.")

# Run the app
if __name__ == "__main__":
    main()
