import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objs as go
import streamlit as st
from datetime import date, timedelta
import os
import ta # Technical Analysis library

# --- Configuration ---
# Assuming the CSV is in the root of the repo, or accessible relative path
CSV_FILE_PATH = "seasonal_screener_results.csv" 
CYCLE_START_MAPPING = {
    "Election": 1952,
    "Pre-Election": 1951,
    "Post-Election": 1953,
    "Midterm": 1950
}
DEFAULT_PIVOT_PERIOD = 20  # Matches your pinescript leftLenH/L and rightLenH/L

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (Copied/Adapted from your provided code)
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
    This logic replicates the spirit of your Pine Script ta.pivothigh/pivotlow 
    using a simple rolling max/min for the pivot *level* value.
    Note: A full Pine Script replication requires a more complex check 
    for local extremum points (leftLen & rightLen). For simplicity and 
    common Python libraries, we'll use a rolling max/min to establish the 
    highest/lowest points over the specified window (leftLen + rightLen).
    
    We'll treat the Pine Script logic of `leftLenH=20` and `rightLenH=20`
    as finding the high/low over a 41-bar window (20 before, 20 after, 1 current).
    However, since we can't look 'future' in the past data, we'll simplify 
    it to a rolling lookback of `period * 2 + 1` for the High/Low.
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

def plot_seasonal_paths(ticker, cycle_label, show_all_years_line=False):
    """Plots the seasonal average path (Cycle and All Years)."""
    
    st.subheader(f"📈 Seasonal Average Path: {cycle_label}")
    
    # Data Fetching
    end_date = dt.datetime(date.today().year, 12, 30) # Use a future end date for consistency with original code
    spx = yf.download(ticker, period="max", end=end_date, progress=False)
    
    if spx.empty:
        st.error(f"No data found for {ticker} for seasonality analysis.")
        return

    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = spx.columns.get_level_values(0)

    # Feature Engineering (log_return, year, month, day_count)
    spx["log_return"] = np.log(spx["Close"] / spx["Close"].shift(1))
    spx["year"] = spx.index.year
    # spx["month"] = spx.index.month
    spx["day_count"] = spx.groupby("year").cumcount() + 1 # Calculate Trading Day Count

    # --- 1. Cycle Data Average Path ---
    if cycle_label == "All Years":
        cycle_data = spx.copy()
    else:
        cycle_start = CYCLE_START_MAPPING[cycle_label]
        # Calculate years in cycle (e.g., 1952, 1956, 1960...)
        years_in_cycle = [cycle_start + i * 4 for i in range((date.today().year - cycle_start) // 4 + 1)] 
        cycle_data = spx[spx["year"].isin(years_in_cycle)].copy()
        
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
        line=dict(color="orange", width=3)
    ))

    # --- 2. Optional Overlay: All Years Average Path ---
    if show_all_years_line and cycle_label != "All Years":
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
        title=f"{ticker} Seasonal Average Path",
        xaxis_title="Trading Day of Year",
        yaxis_title="Cumulative Return (%)",
        yaxis_tickformat=".2%", # Format Y-axis as percentage
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_candlestick_and_mas(ticker):
    """Plots the Candle Chart with MAs and Pivot Points."""
    st.subheader("🕯️ Price Action & Technicals")
    
    # Determine start date: YTD or last 6 months, whichever is longer.
    today = date.today()
    ytd_start = date(today.year, 1, 1)
    six_months_ago = today - timedelta(days=6 * 30) # Approximation of 6 months
    
    start_date = min(ytd_start, six_months_ago)

    # Fetch Data
    df = yf.download(ticker, start=start_date, end=today + timedelta(days=1), progress=False)

    if df.empty:
        st.error(f"No price data found for {ticker} for the current period.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Calculate Moving Averages
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate Pivot Levels (using simplified rolling max/min for level)
    df = calculate_pivot_levels(df)
    
    # --- Plotting ---
    fig = go.Figure()
    
    # 1. Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price',
        increasing_line_color='green', decreasing_line_color='red'
    ))
    
    # 2. Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_10'], line=dict(color='yellow', width=1.5), name='10-Day MA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_200'], line=dict(color='blue', width=2), name='200-Day MA'))
    
    # 3. Pivot Levels (Plot horizontal lines for each non-NaN pivot)
    # We only plot the *current* or *most recent* pivot level as a line extending forward.
    pivot_level_plots = []
    
    # Identify the last non-NaN Pivot High and Low
    last_ph = df['PivotHigh'].dropna().iloc[-1] if not df['PivotHigh'].dropna().empty else None
    last_pl = df['PivotLow'].dropna().iloc[-1] if not df['PivotLow'].dropna().empty else None
    
    if last_ph:
        pivot_level_plots.append(go.Scatter(
            x=[df.index[0], df.index[-1]], 
            y=[last_ph, last_ph], 
            mode='lines', 
            name=f'Recent Pivot High ({last_ph:.2f})', 
            line=dict(color='orange', width=1, dash='dot')
        ))
    if last_pl:
        pivot_level_plots.append(go.Scatter(
            x=[df.index[0], df.index[-1]], 
            y=[last_pl, last_pl], 
            mode='lines', 
            name=f'Recent Pivot Low ({last_pl:.2f})', 
            line=dict(color='orange', width=1, dash='dot')
        ))

    # Add pivot lines to the figure
    for trace in pivot_level_plots:
        fig.add_trace(trace)
        
    # Chart Layout
    fig.update_layout(
        title=f"{ticker} Price Action ({start_date} to Present)",
        xaxis_title="Date",
        yaxis_title="Price",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        height=500,
        xaxis_rangeslider_visible=False # Hide the bottom range slider
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# STREAMLIT PAGE ENTRY POINT
# -----------------------------------------------------------------------------

def seasonal_signals_page():
    st.set_page_config(layout="wide", page_title="Seasonal Signals")
    st.title("💡 Seasonal Signals")
    
    # Load Tickers from the CSV
    try:
        if not os.path.exists(CSV_FILE_PATH):
            st.error(f"Required file '{CSV_FILE_PATH}' not found. Please ensure it's in the correct directory.")
            return

        df_screener = pd.read_csv(CSV_FILE_PATH)
        screener_tickers = df_screener['Ticker'].unique().tolist()
        screener_cycles = df_screener['Type'].unique().tolist()
        # Add a default if the CSV is empty
        if not screener_tickers:
            screener_tickers = ["SPY", "QQQ", "DIA"]
            screener_cycles = ["Post-Election"]
            st.info("Screener CSV is empty, using default tickers and cycles.")

    except Exception as e:
        st.error(f"Error loading {CSV_FILE_PATH}: {e}")
        screener_tickers = ["SPY", "QQQ", "DIA"]
        screener_cycles = ["Post-Election"]
    
    
    # --- Side Bar Controls ---
    st.sidebar.title("Configuration")
    
    # Ticker selection (from CSV)
    ticker = st.sidebar.selectbox("Select Ticker", screener_tickers, index=0).upper()
    
    # Cycle selection
    cycle_label = st.sidebar.selectbox(
        "Seasonal Cycle Type",
        ["All Years"] + screener_cycles,
        index=0 if "All Years" in screener_cycles else 1
    )
    
    # Overlay checkbox for Seasonal Chart
    show_all_years_line = st.sidebar.checkbox("Overlay 'All Years' Average Path", value=False)
    
    
    # --- Main Content ---
    
    st.markdown(f"**Analysis for:** **{ticker}** | **Cycle:** **{cycle_label}**")
    st.divider()

    # 1. Seasonal Chart
    plot_seasonal_paths(ticker, cycle_label, show_all_years_line)
    
    st.divider()
    
    # 2. Candlestick Chart with Technicals
    plot_candlestick_and_mas(ticker)
    

if __name__ == "__main__":
    seasonal_signals_page()
