import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objs as go

@st.cache_data
def load_event_dates():
    url = "https://raw.githubusercontent.com/mslade50/New_Seasonals/main/market_dates.csv"
    return pd.read_csv(url)

def calculate_metrics(event_data, shift_days=0):
    event_data = event_data.copy()
    event_data['Date'] = pd.to_datetime(event_data['Date'], errors='coerce')
    event_data = event_data.dropna(subset=['Date'])

    if event_data.empty:
        st.warning("No event data available for the selected filters.")
        return None, None, None, None, None, None

    # Add required calculations
    event_data['Previous Close'] = event_data['Close'].shift(1)
    event_data['Daily Range'] = (event_data['High'] - event_data['Low']) / event_data['Close'] * 100
    event_data['14D Avg Range'] = event_data['Daily Range'].rolling(window=14, min_periods=1).mean().shift(1)
    event_data['Range/14D Avg'] = event_data['Daily Range'] / event_data['14D Avg Range']

    # Drop rows with missing values
    event_data = event_data.dropna(subset=['Previous Close', 'Daily Range', '14D Avg Range'])

    if event_data.empty:
        st.warning("Insufficient data after applying rolling calculations.")
        return None, None, None, None, None, None

    # Calculate metrics
    event_data['1D Return'] = (event_data['Close'] - event_data['Previous Close']) / event_data['Previous Close'] * 100
    event_data['5D Fwd Return'] = (event_data['Close'].shift(-5) - event_data['Close']) / event_data['Close'] * 100

    # Compute averages
    avg_return = event_data['1D Return'].mean()
    avg_daily_range = event_data['Daily Range'].mean()
    avg_daily_range_14_ratio = event_data['Range/14D Avg'].mean()
    avg_5d_fwd_return = event_data['5D Fwd Return'].mean()

    # Prepare for plotting
    individual_returns = event_data[['Date', '1D Return']].dropna()

    # Backtest
    backtest_table = pd.DataFrame()
    for days in [1, 3, 5, 10]:
        backtest_table[f"{days}D Return"] = (
            (event_data['Close'].shift(-days) - event_data['Close']) / event_data['Close'] * 100
        ).dropna()

    return avg_return, avg_daily_range, avg_daily_range_14_ratio, avg_5d_fwd_return, individual_returns, backtest_table

def main():
    st.title("Market Event Analysis Dashboard")
    dates_df = load_event_dates()
    dates_df['Date'] = pd.to_datetime(dates_df['Date'], errors='coerce')
    dates_df = dates_df.dropna(subset=['Date'])
    dates_df['Year'] = dates_df['Date'].dt.year

    def map_presidential_cycle(year):
        if year % 4 == 0:
            return "Election"
        elif year % 4 == 1:
            return "Post-Election"
        elif year % 4 == 2:
            return "Midterm"
        elif year % 4 == 3:
            return "Pre-Election"

    dates_df['Cycle'] = dates_df['Year'].apply(map_presidential_cycle)

    # User inputs
    event_type = st.selectbox("Select Event Type", options=list(dates_df['Event'].unique()) + ["Opex", "First Day of Month", "Last Day of Month"])
    month = st.selectbox("Select Month (Optional)", options=["All"] + list(range(1, 13)), format_func=lambda x: "All" if x == "All" else pd.to_datetime(f"2022-{x}-01").strftime('%B'))
    cycle_year = st.selectbox("Select Year of Presidential Cycle (Optional)", options=["All", "Election", "Post-Election", "Midterm", "Pre-Election"])
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, SPY):")
    shift_days = st.number_input("Shift Event Dates by Trading Days", min_value=-10, max_value=10, value=0)

    if st.button("Calculate Metrics") and ticker:
        data = yf.download(ticker, start="1990-01-01", progress=False).reset_index()
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

        # Apply event filters
        if event_type == "Opex":
            dates_df = dates_df[dates_df['Event'] == "Opex"]
        elif event_type == "First Day of Month":
            dates_df = pd.DataFrame({"Date": data.loc[data['Date'].dt.is_month_start, 'Date']})
        elif event_type == "Last Day of Month":
            dates_df = pd.DataFrame({"Date": data.loc[data['Date'].dt.is_month_end, 'Date']})
        else:
            dates_df = dates_df[dates_df['Event'] == event_type]

        if month != "All":
            dates_df = dates_df[dates_df['Date'].dt.month == month]
        if cycle_year != "All":
            dates_df = dates_df[dates_df['Cycle'] == cycle_year]

        event_data = data[data['Date'].isin(dates_df['Date'])]
        avg_return, avg_daily_range, avg_daily_range_14_ratio, avg_5d_fwd_return, individual_returns, backtest_table = calculate_metrics(event_data, shift_days)

        if avg_return is not None:
            st.write(f"### Metrics for {event_type} on {ticker}")
            st.write(f"**Average Return (%)**: {avg_return:.2f}")
            st.write(f"**Average Daily Range (%)**: {avg_daily_range:.2f}")
            st.write(f"**Average Daily Range / 14D Average Ratio**: {avg_daily_range_14_ratio:.2f}")
            st.write(f"**Average 5-Day Forward Return (%)**: {avg_5d_fwd_return:.2f}")

            # Plot results
            if not individual_returns.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=individual_returns['Date'], y=individual_returns['1D Return'], name="1-Day Returns", marker=dict(color="blue")))
                fig.update_layout(title=f"1-Day Returns for {event_type} Events on {ticker}", xaxis_title="Date", yaxis_title="1-Day Return (%)", template="plotly_dark")
                st.plotly_chart(fig)

            if backtest_table is not None:
                st.write("### Backtest Results")
                st.write(backtest_table.describe())
        else:
            st.error("No matching event data found.")

if __name__ == "__main__":
    main()
