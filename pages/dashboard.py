import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. DATA ENGINE (Optimized for Speed)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_market_data():
    """
    Fetches multi-asset data for the 4 quadrants.
    """
    # Universe definition
    macro_tickers = ['SPY', 'TLT', 'UUP', 'GLD', 'BTC-USD', '^VIX']
    sectors = ['XLK', 'XLE', 'XLF', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLB', 'IYR', 'SMH']
    actives = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'AMZN', 'MSFT', 'META', 'GOOGL', 'NFLX', 'COIN']
    
    all_tickers = list(set(macro_tickers + sectors + actives))
    
    # Download data (Last 1 year to establish smooth moving averages)
    data = yf.download(all_tickers, period="1y", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
    
    # Reformat into a clean multi-index DataFrame
    df_dict = {}
    for t in all_tickers:
        try:
            df_dict[t] = data[t]['Close']
        except KeyError:
            pass
            
    closes = pd.DataFrame(df_dict)
    
    # Get Volume for the Structure/Gamma proxy
    vol_dict = {}
    for t in ['SPY', 'QQQ', 'IWM']: # Major indices only for structure
        try:
            vol_df = yf.download(t, period="60d", interval="1d", progress=False, auto_adjust=True)
            vol_dict[t] = vol_df
        except:
            pass
            
    return closes, vol_dict

# -----------------------------------------------------------------------------
# 2. CALCULATION ENGINES
# -----------------------------------------------------------------------------
def calculate_regime(df):
    """Quadrant 1: Normalizes prices for the 'Race' chart."""
    # Lookback 20 days (approx 1 trading month)
    lookback = 20
    subset = df[['SPY', 'TLT', 'UUP', 'GLD', 'BTC-USD']].tail(lookback).copy()
    # Normalize to start at 0%
    normalized = (subset / subset.iloc[0] - 1) * 100
    return normalized

def calculate_rrg(df, sectors, benchmark='SPY'):
    """Quadrant 2: Approximates RRG (Relative Rotation Graph)."""
    # RS Ratio = (Price / Benchmark) / MovingAverage(Price / Benchmark)
    # RS Momentum = ROC(RS Ratio)
    
    rrg_data = []
    
    for s in sectors:
        if s not in df.columns: continue
        
        # Relative Strength
        rs = df[s] / df[benchmark]
        
        # JdK RS-Ratio (Simplified: 10d vs 60d trend)
        # Using a 10-day MA of RS normalized
        rs_trend = rs.rolling(window=20).mean()
        rs_ratio = 100 * ((rs / rs_trend) - 1) # Centered at 0
        
        # JdK RS-Momentum (Rate of change of the Ratio)
        rs_mom = rs_ratio.diff(5) # 5-day change in the ratio
        
        curr_ratio = rs_ratio.iloc[-1]
        curr_mom = rs_mom.iloc[-1]
        
        # Determine Quadrant
        if curr_ratio > 0 and curr_mom > 0: quad = "Leading (Green)"
        elif curr_ratio > 0 and curr_mom < 0: quad = "Weakening (Yellow)"
        elif curr_ratio < 0 and curr_mom < 0: quad = "Lagging (Red)"
        else: quad = "Improving (Blue)"
        
        rrg_data.append({
            'Ticker': s,
            'RS_Ratio': curr_ratio,
            'RS_Momentum': curr_mom,
            'Quadrant': quad
        })
        
    return pd.DataFrame(rrg_data)

def calculate_z_scanner(df, tickers):
    """Quadrant 3: Z-Score and Volatility Scanner."""
    results = []
    
    for t in tickers:
        if t not in df.columns: continue
        
        series = df[t].tail(252) # 1 year history for robust stats
        current_price = series.iloc[-1]
        
        # Z-Score (20d)
        roll_mean = series.rolling(20).mean()
        roll_std = series.rolling(20).std()
        z_score = (current_price - roll_mean.iloc[-1]) / roll_std.iloc[-1]
        
        # 5-Day Return
        ret_5d = (current_price / series.iloc[-6] - 1) * 100
        
        # Historical Volatility (Proxy for IV since IV isn't free)
        log_ret = np.log(series / series.shift(1))
        hv_20 = log_ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
        
        results.append({
            'Ticker': t,
            'Price': current_price,
            'Z_Score': z_score,
            '5d_Change': ret_5d,
            'Vol_Ann%': hv_20
        })
        
    return pd.DataFrame(results)

def calculate_structure(vol_data, ticker='SPY'):
    """Quadrant 4: Volume Nodes (Proxy for Gamma Levels)."""
    df = vol_data.get(ticker)
    if df is None: return pd.DataFrame()
    
    # Bucket prices into $2 increments for SPY
    bucket_size = 2
    df['Price_Bucket'] = (df['Close'] // bucket_size) * bucket_size
    
    # Sum volume by bucket
    profile = df.groupby('Price_Bucket')['Volume'].sum().reset_index()
    
    # Find Current Price
    curr_price = df['Close'].iloc[-1]
    
    return profile, curr_price

# -----------------------------------------------------------------------------
# 3. VISUALIZATION ENGINE
# -----------------------------------------------------------------------------
def render_macro_chart(df):
    fig = px.line(df, x=df.index, y=df.columns, 
                  title="20-Day Normalized Performance (Risk Regime)",
                  color_discrete_map={'SPY': 'green', 'TLT': 'red', 'UUP': 'blue', 'GLD': 'gold', 'BTC-USD': 'orange'})
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig

def render_rrg_chart(df):
    # Scatter plot
    fig = px.scatter(df, x="RS_Ratio", y="RS_Momentum", text="Ticker", color="Quadrant",
                     title="Relative Rotation (Sector vs SPY)",
                     color_discrete_map={
                         "Leading (Green)": "#00FF00",
                         "Weakening (Yellow)": "#FFFF00",
                         "Lagging (Red)": "#FF0000",
                         "Improving (Blue)": "#0000FF"
                     })
    
    fig.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig.add_hline(y=0, line_width=1, line_color="white")
    fig.add_vline(x=0, line_width=1, line_color="white")
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    return fig

def render_structure_chart(profile, curr_price, ticker):
    fig = px.bar(profile, y='Price_Bucket', x='Volume', orientation='h',
                 title=f"{ticker} Volume Structure (Gamma Proxy)")
    
    fig.add_hline(y=curr_price, line_dash="dash", line_color="cyan", annotation_text="Current Price")
    
    # Style to look like a profile
    fig.update_traces(marker_color='rgba(255, 255, 255, 0.3)', width=1.5)
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="Price Level")
    return fig

# -----------------------------------------------------------------------------
# 4. MAIN PAGE RENDERER
# -----------------------------------------------------------------------------
def render_swing_dashboard_page():
    st.header("⚡ Unconstrained Swing Command Center")
    st.markdown("Returns: 0-10d Horizon | No Mandate (Equities, Fx, Rates, Cmdty)")
    
    # Load Data
    with st.spinner("Scanning Global Markets..."):
        closes, vol_dict = fetch_market_data()
        
    # --- ROW 1: THE MACRO VIEW ---
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("1. The Weather (Regime)")
        macro_df = calculate_regime(closes)
        st.plotly_chart(render_macro_chart(macro_df), use_container_width=True)
        
        # VIX Overlay
        curr_vix = closes['^VIX'].iloc[-1]
        vix_ma = closes['^VIX'].rolling(20).mean().iloc[-1]
        regime = "COMPLACENT" if curr_vix < vix_ma else "FEARFUL"
        color = "green" if regime == "COMPLACENT" else "red"
        st.caption(f"VIX Regime: :{color}[{regime}] (Spot: {curr_vix:.2f} vs 20d Avg: {vix_ma:.2f})")

    with c2:
        st.subheader("2. The Rotation (Flow)")
        sectors = ['XLK', 'XLE', 'XLF', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLB', 'SMH']
        rrg_df = calculate_rrg(closes, sectors)
        st.plotly_chart(render_rrg_chart(rrg_df), use_container_width=True)

    st.divider()

    # --- ROW 2: THE EXECUTION VIEW ---
    c3, c4 = st.columns([2, 1])
    
    with c3:
        st.subheader("3. The Anomaly Scanner (Z-Scores)")
        active_tickers = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'AMZN', 'MSFT', 'META', 'SPY', 'QQQ', 'IWM', 'TLT', 'GLD']
        scanner_df = calculate_z_scanner(closes, active_tickers)
        
        # Custom Styling for the Dataframe
        def color_zscore(val):
            color = 'white'
            if val > 2.0: color = '#ff4b4b' # Red
            elif val < -2.0: color = '#4caf50' # Green
            return f'color: {color}'

        formatted_df = scanner_df.sort_values(by='Z_Score', key=abs, ascending=False).style.applymap(
            color_zscore, subset=['Z_Score']
        ).format({
            'Price': "{:.2f}",
            'Z_Score': "{:.2f}σ",
            '5d_Change': "{:+.2f}%",
            'Vol_Ann%': "{:.1f}%"
        })
        
        st.dataframe(formatted_df, use_container_width=True, height=300)
        st.info("Strategy: Fade > 2.5σ (Reversion). Join 0σ with high Vol (Trend).")

    with c4:
        st.subheader("4. The Structure (Levels)")
        struct_ticker = st.selectbox("Select Asset", ["SPY", "QQQ", "IWM"])
        profile, curr_px = calculate_structure(vol_dict, struct_ticker)
        if not profile.empty:
            st.plotly_chart(render_structure_chart(profile, curr_px, struct_ticker), use_container_width=True)
        else:
            st.warning("Volume data unavailable.")

# For testing independently
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Swing Dashboard")
    render_swing_dashboard_page()
