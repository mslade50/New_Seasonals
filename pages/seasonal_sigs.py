import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objs as go
import streamlit as st
from datetime import date, timedelta
import os

# --- Configuration ---
CSV_FILE_PATH = "seasonal_screener_results.csv" 
CYCLE_START_MAPPING = {
    "Election": 1952,
    "Pre-Election": 1951,
    "Post-Election": 1953,
    "Midterm": 1950
}
DEFAULT_PIVOT_PERIOD = 20

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (retained)
# -----------------------------------------------------------------------------

def get_current_trading_info():
    """
    Returns the current trading day of the month and week of month 
    based on SPY data for the current month.
    """
    today = dt.date.today()
    start_of_month = dt.date(today.year, today.month, 1)
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

def calculate_pivot_levels(df, period=DEFAULT_PIVOT_PERIOD):
    """
    Calculates Pivot Highs and Pivot Lows based on a lookback window.
    """
    df = df.copy()
    
    # Calculate Pivot High (Highest price over the lookback window)
    df['PivotHigh'] = df['High'].rolling(window=period * 2 + 1, center=True, min_periods=period).max()
    df.loc[df['High'] != df['PivotHigh'], 'PivotHigh'] = np.nan
    
    # Calculate Pivot Low (Lowest price over the lookback window)
    df['PivotLow'] = df['Low'].rolling(window=period * 2 + 1, center=True, min_periods=period).min()
    df.loc[df['Low'] != df['PivotLow'], 'PivotLow'] = np.nan

    return df


# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------

def plot_seasonal_paths(ticker, cycle_label):
    """Plots the seasonal average path (Cycle and All Years)."""
    
    st.subheader(f"📈 {ticker} Seasonal Average Path: {cycle_label}")
    
    # Data Fetching
    end_date = dt.datetime(date.today().year, 12, 30) 
    spx = yf.download(ticker, period="max", end=end_date, progress=False)
    
    if spx.empty:
        st.error(f"No data found for {ticker} for seasonality analysis.")
        return

    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = spx.columns.get_level_values(0)

    # Feature Engineering (log_return, year, month, day_count)
    spx["log_return"] = np.log(spx["Close"] / spx["Close"].shift(1))
    spx["year"] = spx.index.year
    spx["day_count"] = spx.groupby("year").cumcount() + 1 # Calculate Trading Day Count

    # --- 1. Cycle Data Average Path ---
    if cycle_label == "All Years":
        cycle_data = spx.copy()
        line_name = "All Years Avg Path"
    else:
        cycle_start = CYCLE_START_MAPPING.get(cycle_label)
        if cycle_start is None:
            st.error(f"Invalid cycle type '{cycle_label}' for seasonality analysis.")
            return

        years_in_cycle = [cycle_start + i * 4 for i in range((date.today().year - cycle_start) // 4 + 1)] 
        cycle_data = spx[spx["year"].isin(years_in_cycle)].copy()
        line_name = f"Avg Path ({cycle_label})"
        
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
        name=line_name,
        line=dict(color="orange", width=3)
    ))

    # --- 2. Overlay: All Years Average Path (Defaulted to True) ---
    if cycle_label != "All Years":
        all_avg_path = (
            spx.groupby("day_count")["log_return"]
            .mean()
            .cumsum()
            .apply(np.exp) - 1
        )
        fig.add_trace(go.Scatter(
            x=all_avg_path.index,
            y=all_avg_path.values,
            mode="lines",
            name="All Years Avg Path",
            line=dict(color="lightblue", width=1, dash='dash')
        ))

    # --- 3. Current Day Marker ---
    current_year_data = yf.download(ticker, start=dt.datetime(date.today().year, 1, 1), end=date.today() + timedelta(days=1), progress=False)
    current_day_count_val = len(current_year_data) if not current_year_data.empty else None

    avg_path_y_value = avg_path.get(current_day_count_val)
    if avg_path_y_value is not None:
        fig.add_trace(go.Scatter(
            x=[current_day_count_val],
            y=[avg_path_y_value],
            mode="markers",
            name="Current Day",
            marker=dict(color="white", size=10, line=dict(width=2, color='red')),
            showlegend=True
        ))
        
    # --- 4. Chart Layout ---
    fig.update_layout(
        xaxis_title="Trading Day of Year",
        yaxis_title="Cumulative Return (%)",
        yaxis_tickformat=".2%",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_candlestick_and_mas(ticker):
    """Plots the Candle Chart with MAs and Pivot Points."""
    st.subheader("🕯️ Price Action & Technicals")
    
    # Determine chart period: YTD or last 6 months, whichever is longer.
    today = date.today()
    ytd_start = date(today.year, 1, 1)
    six_months_ago = today - timedelta(days=6 * 30)
    chart_start_date = min(ytd_start, six_months_ago)

    # Calculate full fetch start date: Need 200 days *before* the chart starts
    ma_lookback_days = 200 * 1.5 # 200 days, plus a buffer for weekends/holidays
    full_fetch_start = chart_start_date - timedelta(days=ma_lookback_days) 

    # Fetch Data (including lookback for MAs)
    df_full = yf.download(ticker, start=full_fetch_start, end=today + timedelta(days=1), progress=False)

    if df_full.empty:
        st.error(f"No price data found for {ticker} for the current period.")
        return

    if isinstance(df_full.columns, pd.MultiIndex):
        df_full.columns = df_full.columns.get_level_values(0)
    
    # Calculate Moving Averages on the full dataset
    df_full['MA_10'] = df_full['Close'].rolling(window=10).mean()
    df_full['MA_200'] = df_full['Close'].rolling(window=200).mean()
    
    # Calculate Pivot Levels on the full dataset
    df_full = calculate_pivot_levels(df_full)
    
    # Filter the data frame back to the desired charting period
    df = df_full[df_full.index >= pd.to_datetime(chart_start_date)].copy()
    
    # Reset index for Plotly to treat x-axis as continuous trading days (removes gaps)
    df = df.reset_index()
    
    # --- Plotting ---
    fig = go.Figure()
    
    # 1. Candlestick Chart (Black/White colors)
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price',
        increasing_line_color='black',  # Green/Increasing -> Black
        decreasing_line_color='white', # Red/Decreasing -> White
        line=dict(width=1),
        opacity=0.9
    ))
    
    # 2. Moving Averages (10d Purple, 200d Red)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_10'], line=dict(color='purple', width=1.5), name='10-Day MA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_200'], line=dict(color='red', width=2), name='200-Day MA'))
    
    # 3. Pivot Levels (Plot horizontal lines for most recent pivots)
    # We use the full index (Dates) to calculate the last pivot, then map to the continuous index
    
    # Identify the last non-NaN Pivot High and Low from the *full* set
    last_ph = df_full['PivotHigh'].dropna().iloc[-1] if not df_full['PivotHigh'].dropna().empty else None
    last_pl = df_full['PivotLow'].dropna().iloc[-1] if not df_full['PivotLow'].dropna().empty else None
    
    if last_ph:
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=[last_ph] * len(df), 
            mode='lines', 
            name=f'Recent Pivot High ({last_ph:.2f})', 
            line=dict(color='orange', width=1, dash='dot')
        ))
    if last_pl:
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=[last_pl] * len(df), 
            mode='lines', 
            name=f'Recent Pivot Low ({last_pl:.2f})', 
            line=dict(color='orange', width=1, dash='dot')
        ))
        
    # Chart Layout
    fig.update_layout(
        title=f"{ticker} Price Action ({chart_start_date} to Present)",
        xaxis_title="Trading Day Index (Gaps Removed)",
        yaxis_title="Price",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        height=500,
        xaxis_rangeslider_visible=False 
    )
    
    # Map the custom index back to dates for better x-axis labels
    # We use the original Date column for tick text
    tickvals = np.linspace(df.index.min(), df.index.max(), 10, dtype=int)
    fig.update_xaxes(
        tickmode='array',
        tickvals=tickvals,
        ticktext=[df['Date'].iloc[i].strftime('%b %Y') for i in tickvals]
    )
    
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# STREAMLIT PAGE ENTRY POINT (Iterates through all tickers)
# -----------------------------------------------------------------------------

def seasonal_signals_page():
    st.set_page_config(layout="wide", page_title="Seasonal Signals - All Tickers")
    st.title("💡 Seasonal Signals")
    
    # Load Tickers and their associated Cycle Types from the CSV
    try:
        if not os.path.exists(CSV_FILE_PATH):
            st.error(f"Required file '{CSV_FILE_PATH}' not found. Please ensure it's in the correct directory.")
            return

        df_screener = pd.read_csv(CSV_FILE_PATH)
        
        # We need the Ticker and the Type (which we assume is the desired cycle)
        analysis_list = df_screener[['Ticker', 'Type']].drop_duplicates().to_dict('records')
        
        if not analysis_list:
            st.warning("Screener CSV is empty, please ensure it contains 'Ticker' and 'Type' columns.")
            return

    except Exception as e:
        st.error(f"Error loading {CSV_FILE_PATH}: {e}")
        return
    
    
    st.info(f"Displaying {len(analysis_list)} tickers found in `{CSV_FILE_PATH}`.")
    
    # --- Main Content Loop ---
    for item in analysis_list:
        ticker = item['Ticker'].upper()
        # Use the 'Type' column from the CSV as the primary cycle for the seasonal chart
        cycle_label = item['Type'] 
        
        # Check if cycle_label is valid (Election, Pre-Election, Post-Election, Midterm, or All Years)
        if cycle_label not in CYCLE_START_MAPPING and cycle_label != "All Years":
            st.warning(f"Skipping {ticker}: CSV 'Type' value '{cycle_label}' is not a recognized cycle. Falling back to 'Post-Election'.")
            cycle_label = "Post-Election" # Fallback to a common cycle

        st.markdown(f"## {ticker} Analysis (Cycle: {cycle_label})")
        
        # We will plot the Seasonal Chart with the Cycle Type specified in the CSV
        try:
            with st.container():
                plot_seasonal_paths(ticker, cycle_label)
        except Exception as e:
            st.error(f"Error generating Seasonal Chart for {ticker}: {e}")

        st.divider()
        
        # Then plot the Candlestick chart
        try:
            with st.container():
                plot_candlestick_and_mas(ticker)
        except Exception as e:
            st.error(f"Error generating Candle Chart for {ticker}: {e}")
        
        st.markdown("---") # Strong separation between tickers

if __name__ == "__main__":
    seasonal_signals_page()
