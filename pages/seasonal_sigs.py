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
PIVOT_LINE_LENGTH = 252 # How far forward pivot lines extend (trading days)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def get_current_presidential_cycle():
    year = date.today().year
    remainder = year % 4
    if remainder == 0: return "Election"
    elif remainder == 1: return "Post-Election"
    elif remainder == 2: return "Midterm"
    elif remainder == 3: return "Pre-Election"
    return "All Years"

def calculate_pivot_levels(df, period=DEFAULT_PIVOT_PERIOD):
    df = df.copy()
    # We use a centered rolling window. 
    # Note: 'center=True' with a window of period*2+1 means we look back 'period' and forward 'period'.
    # Because pandas handles edges by shrinking the window (if min_periods is met), 
    # we get "unconfirmed" pivots at the end of the data. 
    # We will filter these unconfirmed pivots out in the plotting function.
    
    # Pivot High
    df['PivotHigh'] = df['High'].rolling(window=period * 2 + 1, center=True, min_periods=period).max()
    df.loc[df['High'] != df['PivotHigh'], 'PivotHigh'] = np.nan
    
    # Pivot Low
    df['PivotLow'] = df['Low'].rolling(window=period * 2 + 1, center=True, min_periods=period).min()
    df.loc[df['Low'] != df['PivotLow'], 'PivotLow'] = np.nan
    
    return df

# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------

def plot_seasonal_paths(ticker, cycle_label, stats_row=None):
    # --- Stats Header ---
    if stats_row is not None:
        def get_val(col): return stats_row.get(col, np.nan)
        st.caption(f"📊 **Historical {cycle_label} Stats (from Screener)**")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("5d Avg", f"{get_val('Seas_Cyc_Avg_5d'):.2f}%")
        c2.metric("5d Med", f"{get_val('Seas_Cyc_Med_5d'):.2f}%")
        c3.metric("5d Win%", f"{get_val('Seas_Cyc_Win_5d'):.0f}%")
        c4.metric("21d Avg", f"{get_val('Seas_Cyc_Avg_21d'):.2f}%")
        c5.metric("21d Med", f"{get_val('Seas_Cyc_Med_21d'):.2f}%")
        c6.metric("21d Win%", f"{get_val('Seas_Cyc_Win_21d'):.0f}%")

    st.subheader(f"📈 {ticker} Seasonal Average Path: {cycle_label}")
    
    # Fetch Data
    # Ensure we fetch enough data to include current year
    end_date = dt.datetime.now() # <--- CHANGED: Use now() to ensure we get up to the minute/day
    spx = yf.download(ticker, period="max", end=end_date, progress=False)
    
    if spx.empty:
        st.error(f"No data found for {ticker}.")
        return

    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = spx.columns.get_level_values(0)

    # Engineering
    spx["log_return"] = np.log(spx["Close"] / spx["Close"].shift(1))
    spx["year"] = spx.index.year
    spx["day_count"] = spx.groupby("year").cumcount() + 1 

    # --- Cycle Path ---
    if cycle_label == "All Years":
        cycle_data = spx.copy()
        line_name = "All Years Avg Path"
    else:
        cycle_start = CYCLE_START_MAPPING.get(cycle_label, 1953)
        years_in_cycle = [cycle_start + i * 4 for i in range((date.today().year - cycle_start) // 4 + 1)] 
        cycle_data = spx[spx["year"].isin(years_in_cycle)].copy()
        line_name = f"Avg Path ({cycle_label})"
        
    avg_path = (cycle_data.groupby("day_count")["log_return"].mean().cumsum().apply(np.exp) - 1)

    # --- NEW: Realized YTD Path Logic --- <--- ADDED SECTION
    current_year = date.today().year
    df_current_year = spx[spx["year"] == current_year].copy()
    
    realized_path = pd.Series(dtype=float)
    if not df_current_year.empty:
        # Calculate cumulative return matching the logic of the average path
        # We set index to day_count so it aligns with the x-axis of the average path
        realized_path = df_current_year.set_index("day_count")["log_return"].cumsum().apply(np.exp) - 1

    # --- Plotting ---
    fig = go.Figure()
    
    # 1. Seasonal Average Line
    fig.add_trace(go.Scatter(x=avg_path.index, y=avg_path.values, mode="lines", name=line_name, line=dict(color="orange", width=3)))

    # 2. Realized YTD Line (Added) <--- ADDED SECTION
    if not realized_path.empty:
        fig.add_trace(go.Scatter(
            x=realized_path.index, 
            y=realized_path.values, 
            mode="lines", 
            name=f"{current_year} Realized", 
            line=dict(color="#39FF14", width=2) # Neon Green
        ))

    # --- All Years Overlay ---
    if cycle_label != "All Years":
        all_avg_path = (spx.groupby("day_count")["log_return"].mean().cumsum().apply(np.exp) - 1)
        fig.add_trace(go.Scatter(x=all_avg_path.index, y=all_avg_path.values, mode="lines", name="All Years Avg Path", line=dict(color="lightblue", width=1, dash='dash')))

    # --- Current Day Markers ---
    # We still use this to find the current day count index
    current_day_count = realized_path.index[-1] if not realized_path.empty else None

    if current_day_count:
        val_t = avg_path.get(current_day_count)
        val_t5 = avg_path.get(current_day_count + 5)
        val_t21 = avg_path.get(current_day_count + 21)

        if val_t is not None:
            fig.add_trace(go.Scatter(x=[current_day_count], y=[val_t], mode="markers", name="Curr Day (Avg)", marker=dict(color="red", size=10, line=dict(width=2, color='white'))))
        if val_t5 is not None:
            fig.add_trace(go.Scatter(x=[current_day_count + 5], y=[val_t5], mode="markers", name="T+5 (Path Proj)", marker=dict(color="#00FF00", size=8, symbol="circle")))
        if val_t21 is not None:
            fig.add_trace(go.Scatter(x=[current_day_count + 21], y=[val_t21], mode="markers", name="T+21 (Path Proj)", marker=dict(color="#00BFFF", size=8, symbol="circle")))
        
    fig.update_layout(
        xaxis_title="Trading Day of Year", yaxis_title="Cumulative Return (%)", yaxis_tickformat=".2%",
        plot_bgcolor="black", paper_bgcolor="black", font=dict(color="white"), height=500,
        showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_candlestick_and_mas(ticker, stats_row=None):
    st.subheader("🕯️ Price Action & Technicals")
    
    # 1. Timeline Definitions
    today = date.today()
    ytd_start = date(today.year, 1, 1)
    six_months_ago = today - timedelta(days=6 * 30)
    chart_visible_start = min(ytd_start, six_months_ago)

    # 2. Deep Fetch for MA and Pivots
    # We fetch 750 days (approx 2 years) to ensure 200MA is ready 
    # and to find pivots that originated a year ago but extend into view.
    fetch_start_date = chart_visible_start - timedelta(days=750) 

    df_full = yf.download(ticker, start=fetch_start_date, end=today + timedelta(days=1), progress=False)

    if df_full.empty:
        st.error(f"No price data found for {ticker}.")
        return

    if isinstance(df_full.columns, pd.MultiIndex):
        df_full.columns = df_full.columns.get_level_values(0)
    
    # 3. Calculations on FULL Dataset
    df_full['MA_10'] = df_full['Close'].rolling(window=10).mean()
    df_full['MA_200'] = df_full['Close'].rolling(window=200).mean()
    df_full = calculate_pivot_levels(df_full)
    
    # 4. Prepare Display Data (Reset Index to handle gaps)
    # Subset for the visible area
    mask_visible = df_full.index >= pd.to_datetime(chart_visible_start)
    df_display = df_full[mask_visible].copy()
    
    if df_display.empty:
         st.warning("No data in visible range.")
         return

    df_display = df_display.reset_index()
    last_close_idx = df_display.index[-1]
    last_close_price = df_display['Close'].iloc[-1]
    
    # Map for relative indexing of historical lines
    full_idx_map = {date: idx for idx, date in enumerate(df_full.index)}
    start_date_in_full_idx = full_idx_map.get(df_display['Date'].iloc[0])

    # --- Plotting ---
    fig = go.Figure()
    
    # Candles
    fig.add_trace(go.Candlestick(
        x=df_display.index,
        open=df_display['Open'], high=df_display['High'], low=df_display['Low'], close=df_display['Close'],
        name='Price',
        increasing=dict(line=dict(color='white', width=1), fillcolor='white'), 
        decreasing=dict(line=dict(color='white', width=1), fillcolor='black'),
        opacity=1.0
    ))
    
    # MAs
    fig.add_trace(go.Scatter(x=df_display.index, y=df_display['MA_10'], line=dict(color='purple', width=1.5), name='10-Day MA'))
    fig.add_trace(go.Scatter(x=df_display.index, y=df_display['MA_200'], line=dict(color='red', width=2), name='200-Day MA'))
    
    # --- PINE SCRIPT STYLE PIVOTS ---
    # FILTER: "Confirm" pivots. 
    # A pivot at index 'i' is only confirmed if we have 'period' days of data AFTER it.
    # Therefore, we discard any pivots occurring in the last 'period' days.
    
    last_valid_pivot_idx = len(df_full) - 1 - DEFAULT_PIVOT_PERIOD
    
    # Get all indices in df_full where we have a Pivot High, filtered by validity
    pivot_high_indices = np.where(~np.isnan(df_full['PivotHigh']))[0]
    pivot_high_indices = pivot_high_indices[pivot_high_indices <= last_valid_pivot_idx]
    
    pivot_low_indices = np.where(~np.isnan(df_full['PivotLow']))[0]
    pivot_low_indices = pivot_low_indices[pivot_low_indices <= last_valid_pivot_idx]

    def add_pivot_lines(indices, is_high):
        color = 'orange'
        for p_idx in indices:
            price_val = df_full['PivotHigh'].iloc[p_idx] if is_high else df_full['PivotLow'].iloc[p_idx]
            
            # Start and End relative to df_full
            start_full = p_idx
            end_full = p_idx + PIVOT_LINE_LENGTH
            
            # Convert to Display Coordinates
            x_start = start_full - start_date_in_full_idx
            x_end = end_full - start_date_in_full_idx
            
            # Only draw if the line extends into the visible chart
            if x_end > 0:
                fig.add_trace(go.Scatter(
                    x=[x_start, x_end], 
                    y=[price_val, price_val],
                    mode='lines',
                    line=dict(color=color, width=1, dash='solid'),
                    hoverinfo='skip',
                    showlegend=False
                ))

    add_pivot_lines(pivot_high_indices, True)
    add_pivot_lines(pivot_low_indices, False)

    # --- Targets / Dots ---
    idx_t5 = last_close_idx + 5
    idx_t21 = last_close_idx + 21
    
    if stats_row is not None:
        def get_f(col): return float(stats_row.get(col, 0))
        tgt_5_avg = last_close_price * (1 + get_f('Seas_Cyc_Avg_5d') / 100)
        tgt_21_avg = last_close_price * (1 + get_f('Seas_Cyc_Avg_21d') / 100)
        tgt_5_med = last_close_price * (1 + get_f('Seas_Cyc_Med_5d') / 100)
        tgt_21_med = last_close_price * (1 + get_f('Seas_Cyc_Med_21d') / 100)

        fig.add_trace(go.Scatter(x=[idx_t5], y=[tgt_5_avg], mode='markers', marker=dict(color='magenta', size=8, symbol='diamond'), name='T+5 Avg'))
        fig.add_trace(go.Scatter(x=[idx_t5], y=[tgt_5_med], mode='markers', marker=dict(color='cyan', size=8, symbol='diamond'), name='T+5 Med'))
        fig.add_trace(go.Scatter(x=[idx_t21], y=[tgt_21_avg], mode='markers', marker=dict(color='magenta', size=8, symbol='diamond'), name='T+21 Avg'))
        fig.add_trace(go.Scatter(x=[idx_t21], y=[tgt_21_med], mode='markers', marker=dict(color='cyan', size=8, symbol='diamond'), name='T+21 Med'))

    # --- Layout ---
    date_t5 = today + BusinessDay(5)
    date_t21 = today + BusinessDay(21)

    fig.update_layout(
        title=f"{ticker} Price Action ({chart_visible_start} to Present)",
        xaxis_title="Trading Days", yaxis_title="Price",
        plot_bgcolor="black", paper_bgcolor="black", font=dict(color="white"), height=500,
        xaxis_rangeslider_visible=False,
        xaxis_range=[0, last_close_idx + 25] 
    )
    
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98,
        text=f"<b>Targets:</b><br>T+5: {date_t5.strftime('%Y-%m-%d')}<br>T+21: {date_t21.strftime('%Y-%m-%d')}",
        showarrow=False, font=dict(size=12, color="yellow"), align="left",
        bgcolor="rgba(50,50,50,0.8)", bordercolor="white", borderwidth=1
    )

    tickvals_past = np.linspace(0, last_close_idx, 8, dtype=int)
    ticktext_past = [df_display['Date'].iloc[i].strftime('%b %Y') for i in tickvals_past]
    tickvals = np.concatenate([tickvals_past, [idx_t5, idx_t21]])
    ticktext = ticktext_past + [date_t5.strftime('%m-%d'), date_t21.strftime('%m-%d')]
    
    fig.update_xaxes(tickmode='array', tickvals=tickvals, ticktext=ticktext)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def seasonal_signals_page():
    st.set_page_config(layout="wide", page_title="Seasonal Signals")
    st.title("💡 Seasonal Signals")
    
    if not os.path.exists(CSV_FILE_PATH):
        st.error(f"File '{CSV_FILE_PATH}' not found.")
        return

    df_screener = pd.read_csv(CSV_FILE_PATH)
    if df_screener.empty: return
    
    unique_tickers = df_screener['Ticker'].unique()
    current_cycle = get_current_presidential_cycle()
    
    st.info(f"Loaded {len(unique_tickers)} tickers. Current Cycle: {current_cycle}")

    for ticker in unique_tickers:
        ticker = ticker.upper()
        row = df_screener[df_screener['Ticker'] == ticker].iloc[0]
        st.markdown(f"## {ticker} | Screened: {row.get('Type', 'N/A')} | Cycle: {current_cycle}")
        
        try:
            with st.container(): plot_seasonal_paths(ticker, current_cycle, stats_row=row)
        except Exception as e: st.error(str(e))

        st.divider()

        try:
            with st.container(): plot_candlestick_and_mas(ticker, stats_row=row)
        except Exception as e: st.error(str(e))
        
        st.markdown("---")

if __name__ == "__main__":
    seasonal_signals_page()
