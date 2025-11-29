import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
MACRO_TICKERS = ['SPY', 'TLT', 'UUP', 'GLD', 'BTC-USD', '^VIX']
SECTOR_TICKERS = ['XLK', 'XLE', 'XLF', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLB', 'IYR', 'SMH']
ACTIVE_TICKERS = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'AMZN', 'MSFT', 'META', 'GOOGL', 'NFLX', 'COIN']
STRUCT_TICKERS = ['SPY', 'QQQ', 'IWM']

# -----------------------------------------------------------------------------
# 1. DATA ENGINE (ROBUST)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_market_data():
    """
    Fetches multi-asset data handles yfinance MultiIndex variations safely.
    Returns: (closes_df, volume_dict)
    """
    all_tickers = list(set(MACRO_TICKERS + SECTOR_TICKERS + ACTIVE_TICKERS))
    
    # 1. Bulk Download (Auto Adjust handles splits/divs)
    data = yf.download(all_tickers, period="1y", interval="1d", progress=False, auto_adjust=True)
    
    # 2. Extract 'Close' Column Safely
    if isinstance(data.columns, pd.MultiIndex):
        # check if 'Close' is in level 0 or level 1 (yfinance changes this often)
        if 'Close' in data.columns.get_level_values(0):
            closes = data.xs('Close', level=0, axis=1)
        elif 'Close' in data.columns.get_level_values(1):
            closes = data.xs('Close', level=1, axis=1)
        else:
            # Fallback if structure is weird
            closes = data.iloc[:, :len(all_tickers)] 
    else:
        # If single level index (rare with multiple tickers but possible)
        closes = data['Close'] if 'Close' in data.columns else data

    # Clean column names (remove any remaining tuples)
    closes.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in closes.columns]

    # 3. Fetch Volume Data for Structure (Individual fetches to ensure clean format)
    vol_dict = {}
    for t in STRUCT_TICKERS:
        try:
            vol_df = yf.download(t, period="60d", interval="1d", progress=False, auto_adjust=True)
            # Flatten if multi-index
            if isinstance(vol_df.columns, pd.MultiIndex):
                vol_df.columns = vol_df.columns.get_level_values(0)
            vol_dict[t] = vol_df
        except Exception:
            pass
            
    return closes, vol_dict

# -----------------------------------------------------------------------------
# 2. CALCULATION ENGINES
# -----------------------------------------------------------------------------
def calculate_regime(df):
    """Quadrant 1: Normalizes prices for the 'Race' chart."""
    lookback = 20
    # Only use tickers that actually exist in the dataframe to prevent KeyErrors
    valid_targets = [t for t in ['SPY', 'TLT', 'UUP', 'GLD', 'BTC-USD'] if t in df.columns]
    
    if not valid_targets:
        return pd.DataFrame()

    subset = df[valid_targets].tail(lookback).copy()
    
    # Normalize to start at 0%
    if len(subset) > 0:
        normalized = (subset / subset.iloc[0] - 1) * 100
        return normalized
    return pd.DataFrame()

def calculate_rrg(df, sectors, benchmark='SPY'):
    """Quadrant 2: Approximates RRG (Relative Rotation Graph)."""
    rrg_data = []
    
    if benchmark not in df.columns:
        return pd.DataFrame()

    for s in sectors:
        if s not in df.columns: continue
        
        # Relative Strength
        rs = df[s] / df[benchmark]
        
        # JdK RS-Ratio (Simplified)
        rs_trend = rs.rolling(window=20).mean()
        rs_ratio = 100 * ((rs / rs_trend) - 1)
        
        # JdK RS-Momentum
        rs_mom = rs_ratio.diff(5)
        
        if len(rs_ratio) > 0:
            curr_ratio = rs_ratio.iloc[-1]
            curr_mom = rs_mom.iloc[-1]
            
            # Determine Quadrant
            if curr_ratio > 0 and curr_mom > 0: quad = "Leading (Green)"
            elif curr_ratio > 0 and curr_mom < 0: quad = "Weakening (Yellow)"
            elif curr_ratio < 0 and curr_mom < 0: quad = "Lagging (Red)"
            else: quad = "Improving (Blue)"
            
            rrg_data.append({
                'Ticker': s, 'RS_Ratio': curr_ratio, 'RS_Momentum': curr_mom, 'Quadrant': quad
            })
        
    return pd.DataFrame(rrg_data)

def calculate_z_scanner(df, tickers):
    """Quadrant 3: Z-Score and Volatility Scanner."""
    results = []
    
    for t in tickers:
        if t not in df.columns: continue
        
        series = df[t].dropna()
        if len(series) < 20: continue # Skip if not enough data

        current_price = series.iloc[-1]
        
        # Z-Score (20d)
        roll_mean = series.rolling(20).mean()
        roll_std = series.rolling(20).std()
        
        if pd.isna(roll_std.iloc[-1]): continue

        z_score = (current_price - roll_mean.iloc[-1]) / roll_std.iloc[-1]
        
        # 5-Day Return
        ret_5d = 0.0
        if len(series) > 5:
            ret_5d = (current_price / series.iloc[-6] - 1) * 100
        
        # Historical Volatility
        log_ret = np.log(series / series.shift(1))
        hv_20 = log_ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
        
        results.append({
            'Ticker': t,
            # Use .item() to avoid FutureWarning when casting Series to float
            'Price': float(current_price) if isinstance(current_price, (float, int)) else current_price.item(),
            'Z_Score': float(z_score) if isinstance(z_score, (float, int)) else z_score.item(),
            '5d_Change': ret_5d,
            'Vol_Ann%': hv_20
        })
        
    return pd.DataFrame(results)

def calculate_structure(vol_data, ticker='SPY'):
    """Quadrant 4: Volume Nodes."""
    df = vol_data.get(ticker)
    if df is None or df.empty: return pd.DataFrame(), 0
    
    # Bucket prices (Adaptive bucket size based on price)
    curr_price = df['Close'].iloc[-1]
    bucket_size = max(0.5, round(curr_price * 0.005)) # ~0.5% bucket size
    
    df = df.copy()
    df['Price_Bucket'] = (df['Close'] // bucket_size) * bucket_size
    
    profile = df.groupby('Price_Bucket')['Volume'].sum().reset_index()
    return profile, curr_price

# -----------------------------------------------------------------------------
# 3. VISUALIZATION ENGINE
# -----------------------------------------------------------------------------
def render_macro_chart(df):
    if df.empty: return None
    fig = px.line(df, x=df.index, y=df.columns, 
                  title="20-Day Normalized Performance (Risk Regime)",
                  color_discrete_map={'SPY': 'green', 'TLT': 'red', 'UUP': 'blue', 'GLD': 'gold', 'BTC-USD': 'orange'})
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig

def render_rrg_chart(df):
    if df.empty: return None
    fig = px.scatter(df, x="RS_Ratio", y="RS_Momentum", text="Ticker", color="Quadrant",
                     title="Relative Rotation (Sector vs SPY)",
                     color_discrete_map={
                         "Leading (Green)": "#00FF00", "Weakening (Yellow)": "#FFFF00",
                         "Lagging (Red)": "#FF0000", "Improving (Blue)": "#0000FF"
                     })
    fig.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig.add_hline(y=0, line_width=1, line_color="white")
    fig.add_vline(x=0, line_width=1, line_color="white")
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    return fig

def render_structure_chart(profile, curr_price, ticker):
    fig = px.bar(profile, y='Price_Bucket', x='Volume', orientation='h',
                 title=f"{ticker} Volume Structure")
    fig.add_hline(y=curr_price, line_dash="dash", line_color="cyan", annotation_text="Current")
    fig.update_traces(marker_color='rgba(255, 255, 255, 0.3)', width=1.5)
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="Price Level")
    return fig

# -----------------------------------------------------------------------------
# 4. UI RENDERER
# -----------------------------------------------------------------------------
def render_swing_dashboard_page():
    st.header("⚡ Unconstrained Swing Command Center")
    st.markdown("Returns: 0-10d Horizon | No Mandate")
    
    with st.spinner("Scanning Global Markets..."):
        closes, vol_dict = fetch_market_data()
        
    if closes.empty:
        st.error("Market data could not be retrieved. Try upgrading yfinance: `pip install --upgrade yfinance`")
        return

    # --- ROW 1 ---
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("1. The Weather (Regime)")
        macro_df = calculate_regime(closes)
        if not macro_df.empty:
            st.plotly_chart(render_macro_chart(macro_df), use_container_width=True)
            # VIX Check
            if '^VIX' in closes.columns:
                curr_vix = closes['^VIX'].iloc[-1]
                vix_ma = closes['^VIX'].rolling(20).mean().iloc[-1]
                regime = "COMPLACENT" if curr_vix < vix_ma else "FEARFUL"
                color = "green" if regime == "COMPLACENT" else "red"
                st.caption(f"VIX Regime: :{color}[{regime}] (Spot: {curr_vix:.2f} vs 20d Avg: {vix_ma:.2f})")
        else:
            st.warning("Insufficient data for Regime.")

    with c2:
        st.subheader("2. The Rotation (Flow)")
        rrg_df = calculate_rrg(closes, SECTOR_TICKERS)
        if not rrg_df.empty:
            st.plotly_chart(render_rrg_chart(rrg_df), use_container_width=True)
        else:
            st.warning("Insufficient data for RRG.")

    st.divider()

    # --- ROW 2 ---
    c3, c4 = st.columns([2, 1])
    with c3:
        st.subheader("3. The Anomaly Scanner (Z-Scores)")
        # Filter actives + macro for scanner
        scanner_list = list(set(ACTIVE_TICKERS + ['SPY', 'TLT', 'GLD']))
        scanner_df = calculate_z_scanner(closes, scanner_list)
        
        if not scanner_df.empty:
            def color_zscore(val):
                color = 'white'
                if val > 2.0: color = '#ff4b4b'
                elif val < -2.0: color = '#4caf50'
                return f'color: {color}'

            formatted_df = scanner_df.sort_values(by='Z_Score', key=abs, ascending=False).style.map(
                color_zscore, subset=['Z_Score']
            ).format({
                'Price': "{:.2f}", 'Z_Score': "{:.2f}σ", '5d_Change': "{:+.2f}%", 'Vol_Ann%': "{:.1f}%"
            })
            st.dataframe(formatted_df, use_container_width=True, height=300)
            st.info("Strategy: Fade > 2.5σ (Reversion). Join 0σ with high Vol (Trend).")
        else:
            st.warning("No data for Scanner.")

    with c4:
        st.subheader("4. The Structure (Levels)")
        struct_ticker = st.selectbox("Select Asset", STRUCT_TICKERS)
        profile, curr_px = calculate_structure(vol_dict, struct_ticker)
        if not profile.empty:
            st.plotly_chart(render_structure_chart(profile, curr_px, struct_ticker), use_container_width=True)
        else:
            st.warning("Volume data unavailable.")

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Swing Dashboard")
    render_swing_dashboard_page()
