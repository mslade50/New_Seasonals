import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objs as go
import streamlit as st
from datetime import date, timedelta
from pandas.tseries.offsets import BusinessDay
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
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def get_current_presidential_cycle():
    """
    Determines the current Presidential Cycle based on the current year.
    Returns: 'Election', 'Post-Election', 'Midterm', or 'Pre-Election'
    """
    year = date.today().year
    remainder = year % 4
    
    if remainder == 0:
        return "Election"
    elif remainder == 1:
        return "Post-Election"
    elif remainder == 2:
        return "Midterm"
    elif remainder == 3:
        return "Pre-Election"
    return "All Years"

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
            cycle_start = 1953 # Fallback if something goes wrong
            
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
    
    # Plot Main Cycle Path
    fig.add_trace(go.Scatter(
        x=avg_path.index,
        y=avg_path.values,
        mode="lines",
        name=line_name,
        line=dict(color="orange", width=3)
    ))

    # --- 2. Overlay: All Years Average Path (Always ON) ---
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

    # --- 3. Current Day + Projection Markers ---
    current_year_data = yf.download(ticker, start=dt.datetime(date.today().year, 1, 1), end=date.today() + timedelta(days=1), progress=False)
    current_day_count_val = len(current_year_data) if not current_year_data.empty else None

    if current_day_count_val:
        # T (Current)
        val_t = avg_path.get(current_day_count_val)
        
        # T + 5
        idx_t5 = current_day_count_val + 5
        val_t5 = avg_path.get(idx_t5)
        
        # T + 21
        idx_t21 = current_day_count_val + 21
        val_t21 = avg_path.get(idx_t21)

        # Plot Current Day (Red)
        if val_t is not None:
            fig.add_trace(go.Scatter(
                x=[current_day_count_val], y=[val_t],
                mode="markers", name="Current Day",
                marker=dict(color="red", size=10, line=dict(width=2, color='white'))
            ))
            
        # Plot T+5 (Green)
        if val_t5 is not None:
            fig.add_trace(go.Scatter(
                x=[idx_t5], y=[val_t5],
                mode="markers", name="T+5 (Proj)",
                marker=dict(color="#00FF00", size=8, symbol="circle")
            ))
            
        # Plot T+21 (Blue)
        if val_t21 is not None:
            fig.add_trace(go.Scatter(
                x=[idx_t21], y=[val_t21],
                mode="markers", name="T+21 (Proj)",
                marker=dict(color="#00BFFF", size=8, symbol="circle")
            ))
        
    # --- 4. Chart Layout ---
    fig.update_layout(
        xaxis_title="Trading Day of Year",
        yaxis_title="Cumulative Return (%)",
        yaxis_tickformat=".2%",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0)"
        )
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_candlestick_and_mas(ticker):
    """Plots the Candle Chart with MAs and Pivot Points."""
    st.subheader("🕯️ Price Action & Technicals")
    
    # Determine chart period
    today = date.today()
    ytd_start = date(today.year, 1, 1)
    six_months_ago = today - timedelta(days=6 * 30)
    chart_start_date = min(ytd_start, six_months_ago)

    # MA Lookback (400 calendar days to ensure we get 200 trading days)
    ma_lookback_days = 400 
    full_fetch_start = chart_start_date - timedelta(days=ma_lookback_days) 

    # Fetch Data
    df_full = yf.download(ticker, start=full_fetch_start, end=today + timedelta(days=1), progress=False)

    if df_full.empty:
        st.error(f"No price data found for {ticker} for the current period.")
        return

    if isinstance(df_full.columns, pd.MultiIndex):
        df_full.columns = df_full.columns.get_level_values(0)
    
    # Calculate Moving Averages
    df_full['MA_10'] = df_full['Close'].rolling(window=10).mean()
    df_full['MA_200'] = df_full['Close'].rolling(window=200).mean()
    
    # Calculate Pivot Levels
    df_full = calculate_pivot_levels(df_full)
    
    # Filter for display
    df = df_full[df_full.index >= pd.to_datetime(chart_start_date)].copy()
    
    if df.empty:
         st.warning(f"No data available for chart period starting {chart_start_date}")
         return

    # Reset index for gap removal
    df = df.reset_index()
    
    # --- Future Dates Calculation for Annotation ---
    # Using BusinessDay to approximate trading days (skips weekends)
    date_t5 = today + BusinessDay(5)
    date_t21 = today + BusinessDay(21)
    
    # --- Plotting ---
    fig = go.Figure()
    
    # 1. Candlestick Chart
    # UP = White fill, White line
    # DOWN = Black fill, White line (to see outline on black bg)
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price',
        increasing=dict(line=dict(color='white', width=1), fillcolor='white'), 
        decreasing=dict(line=dict(color='white', width=1), fillcolor='black'),
        opacity=1.0
    ))
    
    # 2. Moving Averages (10d Purple, 200d Red)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_10'], line=dict(color='purple', width=1.5), name='10-Day MA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_200'], line=dict(color='red', width=2), name='200-Day MA'))
    
    # 3. Pivot Levels
    last_ph = df_full['PivotHigh'].dropna().iloc[-1] if not df_full['PivotHigh'].dropna().empty else None
    last_pl = df_full['PivotLow'].dropna().iloc[-1] if not df_full['PivotLow'].dropna().empty else None
    
    if last_ph:
        fig.add_trace(go.Scatter(
            x=df.index, y=[last_ph] * len(df), mode='lines', 
            name=f'Recent Pivot High ({last_ph:.2f})', 
            line=dict(color='orange', width=1, dash='dot')
        ))
    if last_pl:
        fig.add_trace(go.Scatter(
            x=df.index, y=[last_pl] * len(df), mode='lines', 
            name=f'Recent Pivot Low ({last_pl:.2f})', 
            line=dict(color='orange', width=1, dash='dot')
        ))
        
    # Chart Layout
    fig.update_layout(
        title=f"{ticker} Price Action ({chart_start_date} to Present)",
        xaxis_title="Trading Days",
        yaxis_title="Price",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        height=500,
        xaxis_rangeslider_visible=False 
    )

    # --- Add Future Date Annotations ---
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98, # Top Left
        text=f"<b>Targets:</b><br>T+5 Date: {date_t5.strftime('%Y-%m-%d')}<br>T+21 Date: {date_t21.strftime('%Y-%m-%d')}",
        showarrow=False,
        font=dict(size=12, color="yellow"),
        align="left",
        bgcolor="rgba(50,50,50,0.8)",
        bordercolor="white",
        borderwidth=1
    )
    
    # X-Axis Ticks
    tickvals = np.linspace(df.index.min(), df.index.max(), 10, dtype=int)
    ticktext = [df['Date'].iloc[i].strftime('%b %Y') if 'Date' in df.columns else df.index[i] for i in tickvals]
    
    fig.update_xaxes(tickmode='array', tickvals=tickvals, ticktext=ticktext)
    
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# STREAMLIT PAGE ENTRY POINT
# -----------------------------------------------------------------------------

def seasonal_signals_page():
    st.set_page_config(layout="wide", page_title="Seasonal Signals - All Tickers")
    st.title("💡 Seasonal Signals")
    
    # Load Tickers
    try:
        if not os.path.exists(CSV_FILE_PATH):
            st.error(f"Required file '{CSV_FILE_PATH}' not found.")
            return

        df_screener = pd.read_csv(CSV_FILE_PATH)
        # We assume the CSV has 'Ticker' and 'Type' (Type = Bear/Bull)
        analysis_list = df_screener[['Ticker', 'Type']].drop_duplicates().to_dict('records')
        
        if not analysis_list:
            st.warning("Screener CSV is empty.")
            return

    except Exception as e:
        st.error(f"Error loading {CSV_FILE_PATH}: {e}")
        return
    
    st.info(f"Displaying {len(analysis_list)} tickers from `{CSV_FILE_PATH}`.")
    
    # --- Auto-Detect Current Cycle (Fixes the Bear/Bull Error) ---
    current_cycle = get_current_presidential_cycle()

    # --- Main Content Loop ---
    for item in analysis_list:
        ticker = item['Ticker'].upper()
        signal_type = item['Type'] # Keep this for display, but don't use for cycle logic
        
        # Header now shows the Screened Signal + The Cycle being used
        st.markdown(f"## {ticker} (Screened: {signal_type}) | Cycle: {current_cycle}")
        
        # 1. Seasonal Chart (Uses the auto-detected current_cycle)
        try:
            with st.container():
                plot_seasonal_paths(ticker, current_cycle)
        except Exception as e:
            st.error(f"Error generating Seasonal Chart for {ticker}: {e}")

        st.divider()
        
        # 2. Candle Chart
        try:
            with st.container():
                plot_candlestick_and_mas(ticker)
        except Exception as e:
            st.error(f"Error generating Candle Chart for {ticker}: {e}")
        
        st.markdown("---")

if __name__ == "__main__":
    seasonal_signals_page()
