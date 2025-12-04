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

def calculate_pivot_levels(df, period=DEFAULT_PIVOT_PERIOD):
    df = df.copy()
    df['PivotHigh'] = df['High'].rolling(window=period * 2 + 1, center=True, min_periods=period).max()
    df.loc[df['High'] != df['PivotHigh'], 'PivotHigh'] = np.nan
    df['PivotLow'] = df['Low'].rolling(window=period * 2 + 1, center=True, min_periods=period).min()
    df.loc[df['Low'] != df['PivotLow'], 'PivotLow'] = np.nan
    return df

# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------

def plot_seasonal_paths(ticker, cycle_label, stats_row=None):
    """Plots the seasonal average path (Cycle and All Years) with CSV stats header."""
    
    # --- Display CSV Stats Header ---
    if stats_row is not None:
        st.caption(f"📊 **Historical {cycle_label} Stats (from Screener)**")
        
        # safely get values, default to 0 if column missing
        def get_val(col): return stats_row.get(col, np.nan)
        
        # We try to grab Cycle stats first, fall back to All if needed or preferred
        # Assuming column names from your screenshot: Seas_Cyc_Avg_5d, Seas_Cyc_Med_5d, Seas_Cyc_Win_5d
        
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("5d Avg", f"{get_val('Seas_Cyc_Avg_5d'):.2f}%")
        c2.metric("5d Med", f"{get_val('Seas_Cyc_Med_5d'):.2f}%")
        c3.metric("5d Win%", f"{get_val('Seas_Cyc_Win_5d'):.0f}%")
        
        c4.metric("21d Avg", f"{get_val('Seas_Cyc_Avg_21d'):.2f}%")
        c5.metric("21d Med", f"{get_val('Seas_Cyc_Med_21d'):.2f}%")
        c6.metric("21d Win%", f"{get_val('Seas_Cyc_Win_21d'):.0f}%")

    st.subheader(f"📈 {ticker} Seasonal Average Path: {cycle_label}")
    
    # Data Fetching
    end_date = dt.datetime(date.today().year, 12, 30) 
    spx = yf.download(ticker, period="max", end=end_date, progress=False)
    
    if spx.empty:
        st.error(f"No data found for {ticker} for seasonality analysis.")
        return

    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = spx.columns.get_level_values(0)

    # Feature Engineering
    spx["log_return"] = np.log(spx["Close"] / spx["Close"].shift(1))
    spx["year"] = spx.index.year
    spx["day_count"] = spx.groupby("year").cumcount() + 1 

    # --- 1. Cycle Data Average Path ---
    if cycle_label == "All Years":
        cycle_data = spx.copy()
        line_name = "All Years Avg Path"
    else:
        cycle_start = CYCLE_START_MAPPING.get(cycle_label)
        if cycle_start is None: cycle_start = 1953
            
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
        x=avg_path.index, y=avg_path.values,
        mode="lines", name=line_name,
        line=dict(color="orange", width=3)
    ))

    # --- 2. Overlay: All Years Average Path ---
    if cycle_label != "All Years":
        all_avg_path = (
            spx.groupby("day_count")["log_return"]
            .mean()
            .cumsum()
            .apply(np.exp) - 1
        )
        fig.add_trace(go.Scatter(
            x=all_avg_path.index, y=all_avg_path.values,
            mode="lines", name="All Years Avg Path",
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
                mode="markers", name="T+5 (Path Proj)",
                marker=dict(color="#00FF00", size=8, symbol="circle")
            ))
        # Plot T+21 (Blue)
        if val_t21 is not None:
            fig.add_trace(go.Scatter(
                x=[idx_t21], y=[val_t21],
                mode="markers", name="T+21 (Path Proj)",
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


def plot_candlestick_and_mas(ticker, stats_row=None):
    """Plots the Candle Chart with MAs, Pivot Points, Open Days, and Price Targets."""
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
    last_close_idx = df.index[-1]
    last_close_price = df['Close'].iloc[-1]
    
    # --- Future Dates Calculation ---
    date_t5 = today + BusinessDay(5)
    date_t21 = today + BusinessDay(21)
    
    # --- Price Targets from CSV Stats ---
    # We will plot targets at T+5 and T+21 relative to the last integer index
    idx_t5 = last_close_idx + 5
    idx_t21 = last_close_idx + 21
    
    t5_avg_target = None
    t5_med_target = None
    t21_avg_target = None
    t21_med_target = None

    if stats_row is not None:
        # Helper to get float from row
        def get_f(col): return float(stats_row.get(col, 0))
        
        # Calculate Prices: Last Close * (1 + Return/100)
        t5_avg_target = last_close_price * (1 + get_f('Seas_Cyc_Avg_5d') / 100)
        t5_med_target = last_close_price * (1 + get_f('Seas_Cyc_Med_5d') / 100)
        
        t21_avg_target = last_close_price * (1 + get_f('Seas_Cyc_Avg_21d') / 100)
        t21_med_target = last_close_price * (1 + get_f('Seas_Cyc_Med_21d') / 100)

    # --- Plotting ---
    fig = go.Figure()
    
    # 1. Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price',
        increasing=dict(line=dict(color='white', width=1), fillcolor='white'), 
        decreasing=dict(line=dict(color='white', width=1), fillcolor='black'),
        opacity=1.0
    ))
    
    # 2. Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_10'], line=dict(color='purple', width=1.5), name='10-Day MA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_200'], line=dict(color='red', width=2), name='200-Day MA'))
    
    # 3. Pivot Levels
    last_ph = df_full['PivotHigh'].dropna().iloc[-1] if not df_full['PivotHigh'].dropna().empty else None
    last_pl = df_full['PivotLow'].dropna().iloc[-1] if not df_full['PivotLow'].dropna().empty else None
    
    if last_ph:
        fig.add_trace(go.Scatter(
            x=df.index, y=[last_ph] * len(df), mode='lines', 
            name=f'Pivot High ({last_ph:.2f})', line=dict(color='orange', width=1, dash='dot')
        ))
    if last_pl:
        fig.add_trace(go.Scatter(
            x=df.index, y=[last_pl] * len(df), mode='lines', 
            name=f'Pivot Low ({last_pl:.2f})', line=dict(color='orange', width=1, dash='dot')
        ))
        
    # 4. Projected Targets (Dots)
    # T+5
    if t5_avg_target:
        fig.add_trace(go.Scatter(
            x=[idx_t5], y=[t5_avg_target], mode='markers', name='T+5 Avg Tgt',
            marker=dict(color='magenta', size=8, symbol='diamond')
        ))
    if t5_med_target:
        fig.add_trace(go.Scatter(
            x=[idx_t5], y=[t5_med_target], mode='markers', name='T+5 Med Tgt',
            marker=dict(color='cyan', size=8, symbol='diamond')
        ))
        
    # T+21
    if t21_avg_target:
        fig.add_trace(go.Scatter(
            x=[idx_t21], y=[t21_avg_target], mode='markers', name='T+21 Avg Tgt',
            marker=dict(color='magenta', size=8, symbol='diamond')
        ))
    if t21_med_target:
        fig.add_trace(go.Scatter(
            x=[idx_t21], y=[t21_med_target], mode='markers', name='T+21 Med Tgt',
            marker=dict(color='cyan', size=8, symbol='diamond')
        ))

    # --- Layout & X-Axis Extension ---
    # Extend X-axis to create "Open" days
    # Range = [0, Current_Last_Index + 25]
    fig.update_layout(
        title=f"{ticker} Price Action ({chart_start_date} to Present)",
        xaxis_title="Trading Days (with Projections)",
        yaxis_title="Price",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        height=500,
        xaxis_rangeslider_visible=False,
        xaxis_range=[0, last_close_idx + 25] # Extend view
    )

    # --- Annotations & Ticks ---
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text=f"<b>Targets:</b><br>T+5 Date: {date_t5.strftime('%Y-%m-%d')}<br>T+21 Date: {date_t21.strftime('%Y-%m-%d')}",
        showarrow=False, font=dict(size=12, color="yellow"), align="left",
        bgcolor="rgba(50,50,50,0.8)", bordercolor="white", borderwidth=1
    )
    
    # Create Tick Text (Past Dates + Future Dates)
    tickvals_past = np.linspace(0, last_close_idx, 8, dtype=int)
    ticktext_past = [df['Date'].iloc[i].strftime('%b %Y') if 'Date' in df.columns else str(i) for i in tickvals_past]
    
    # Add manual ticks for the projection days
    tickvals = np.concatenate([tickvals_past, [idx_t5, idx_t21]])
    ticktext = ticktext_past + [date_t5.strftime('%m-%d'), date_t21.strftime('%m-%d')]
    
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
        # We need the whole row now, not just Ticker/Type
        if df_screener.empty:
            st.warning("Screener CSV is empty.")
            return
            
        # Get unique tickers to iterate, but keep the row data
        # We assume one row per ticker in the screener results
        unique_tickers = df_screener['Ticker'].unique()

    except Exception as e:
        st.error(f"Error loading {CSV_FILE_PATH}: {e}")
        return
    
    st.info(f"Displaying {len(unique_tickers)} tickers from `{CSV_FILE_PATH}`.")
    
    # --- Auto-Detect Current Cycle ---
    current_cycle = get_current_presidential_cycle()

    # --- Main Content Loop ---
    for ticker in unique_tickers:
        ticker = ticker.upper()
        
        # Get the row for this ticker
        row = df_screener[df_screener['Ticker'] == ticker].iloc[0]
        signal_type = row.get('Type', 'N/A')
        
        # Header
        st.markdown(f"## {ticker} (Screened: {signal_type}) | Cycle: {current_cycle}")
        
        # 1. Seasonal Chart (Pass the stats row)
        try:
            with st.container():
                plot_seasonal_paths(ticker, current_cycle, stats_row=row)
        except Exception as e:
            st.error(f"Error generating Seasonal Chart for {ticker}: {e}")

        st.divider()
        
        # 2. Candle Chart (Pass the stats row for target dots)
        try:
            with st.container():
                plot_candlestick_and_mas(ticker, stats_row=row)
        except Exception as e:
            st.error(f"Error generating Candle Chart for {ticker}: {e}")
        
        st.markdown("---")

if __name__ == "__main__":
    seasonal_signals_page()
