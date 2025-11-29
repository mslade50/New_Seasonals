import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
MACRO_TICKERS = ['SPY', 'TLT', 'DX-Y.NYB', 'GLD', 'BTC-USD', '^VIX', '^VIX3M', 'AUDJPY=X']
SECTOR_TICKERS = ['XLK', 'XLE', 'XLF', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLB', 'SMH']
ACTIVE_TICKERS = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'AMZN', 'MSFT', 'META', 'GOOGL', 'NFLX', 'COIN']

# -----------------------------------------------------------------------------
# 1. DATA ENGINE (Robust "Swiss Cheese" Fix)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_market_data():
    """Fetches data and fixes gaps between Crypto/Stocks/Forex."""
    all_tickers = list(set(MACRO_TICKERS + SECTOR_TICKERS + ACTIVE_TICKERS))
    
    # Download with auto_adjust=False to keep raw Close for Indices
    data = yf.download(all_tickers, period="1y", interval="1d", progress=False, auto_adjust=False)
    
    # Robust Column Extraction
    # Prioritize Adj Close, fallback to Close
    target_col = 'Adj Close' if 'Adj Close' in data.columns.get_level_values(0) else 'Close'
    
    if isinstance(data.columns, pd.MultiIndex):
        if target_col in data.columns.get_level_values(0):
            closes = data.xs(target_col, level=0, axis=1)
        elif target_col in data.columns.get_level_values(1):
            closes = data.xs(target_col, level=1, axis=1)
        else:
            # Fallback
            try: closes = data.xs('Close', level=0, axis=1)
            except: closes = data.iloc[:, :len(all_tickers)]
    else:
        closes = data[target_col] if target_col in data.columns else data

    # Clean Columns and Fill Gaps (The Fix)
    closes.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in closes.columns]
    closes = closes.ffill().bfill()
    
    return closes

# -----------------------------------------------------------------------------
# 2. CALCULATION ENGINES
# -----------------------------------------------------------------------------
def calculate_macro_regime(df):
    """Q1: Macro Weather & VIX Curve."""
    targets = ['SPY', 'TLT', 'DX-Y.NYB', 'AUDJPY=X']
    valid_targets = [t for t in targets if t in df.columns]
    
    if not valid_targets: return pd.DataFrame(), "Unknown"

    subset = df[valid_targets].tail(20).copy()
    normalized = (subset / subset.iloc[0] - 1) * 100
    
    vix_state = "Neutral"
    if '^VIX' in df.columns and '^VIX3M' in df.columns:
        spot = df['^VIX'].iloc[-1]
        term = df['^VIX3M'].iloc[-1]
        if term != 0:
            ratio = spot / term
            if ratio > 1.05: vix_state = "INVERTED (Risk Off)"
            elif ratio < 0.9: vix_state = "STEEP (Risk On)"
            else: vix_state = "FLAT (Caution)"
        
    return normalized, vix_state

def calculate_rrg(df, sectors, benchmark='SPY'):
    """Q2: Relative Rotation Graph."""
    rrg_data = []
    if benchmark not in df.columns: return pd.DataFrame()

    for s in sectors:
        if s not in df.columns: continue
        rs = df[s] / df[benchmark]
        
        rs_trend = rs.rolling(window=20).mean()
        rs_ratio = 100 * ((rs / rs_trend) - 1)
        rs_mom = rs_ratio.diff(5)
        
        if len(rs_ratio) > 0:
            curr_ratio = rs_ratio.iloc[-1]
            curr_mom = rs_mom.iloc[-1]
            if curr_ratio > 0 and curr_mom > 0: quad = "Leading"
            elif curr_ratio > 0 and curr_mom < 0: quad = "Weakening"
            elif curr_ratio < 0 and curr_mom < 0: quad = "Lagging"
            else: quad = "Improving"
            
            rrg_data.append({'Ticker': s, 'RS_Ratio': curr_ratio, 'RS_Momentum': curr_mom, 'Quadrant': quad})
            
    return pd.DataFrame(rrg_data)

def calculate_vol_scanner(df, tickers):
    """Q3: The Volatility Physics Engine."""
    results = []
    for t in tickers:
        if t not in df.columns: continue
        series = df[t].dropna()
        if len(series) < 30: continue

        curr = series.iloc[-1]
        
        # Z-Score (Potential Energy)
        mean_20 = series.rolling(20).mean().iloc[-1]
        std_20 = series.rolling(20).std().iloc[-1]
        z_score = (curr - mean_20) / std_20 if std_20 != 0 else 0
        
        # Vol Ratio (Kinetic Energy)
        log_ret = np.log(series / series.shift(1))
        vol_5d = log_ret.rolling(5).std().iloc[-1]
        vol_20d = log_ret.rolling(20).std().iloc[-1]
        vol_ratio = vol_5d / vol_20d if vol_20d != 0 else 1.0
        
        results.append({
            'Ticker': t, 
            'Price': float(curr) if isinstance(curr, (float, int)) else curr.item(),
            'Z_Score': float(z_score) if isinstance(z_score, (float, int)) else z_score.item(),
            'Vol_Ratio': vol_ratio
        })
        
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 3. VISUALIZATION ENGINE
# -----------------------------------------------------------------------------
def plot_macro(df):
    if df.empty: return None
    fig = px.line(df, x=df.index, y=df.columns, title="Global Risk Flow (20d)",
                  color_discrete_map={'SPY':'#4caf50', 'TLT':'#f44336', 'DX-Y.NYB':'#9e9e9e', 'AUDJPY=X':'#03a9f4'})
    fig.update_layout(height=280, margin=dict(l=10,r=10,t=30,b=10), hovermode="x unified")
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
    return fig

def plot_rrg(df):
    if df.empty: return None
    fig = px.scatter(df, x="RS_Ratio", y="RS_Momentum", text="Ticker", color="Quadrant",
                     title="Relative Rotation (Flow)",
                     color_discrete_map={
                         "Leading": "#4caf50", "Weakening": "#ffeb3b",
                         "Lagging": "#f44336", "Improving": "#2196f3"
                     })
    fig.update_traces(textposition='top center', marker=dict(size=14, line=dict(width=1, color='black')))
    fig.add_hline(y=0, line_color="white", opacity=0.3)
    fig.add_vline(x=0, line_color="white", opacity=0.3)
    fig.update_layout(height=280, margin=dict(l=10,r=10,t=30,b=10), showlegend=False)
    return fig

def plot_vol_landscape(df):
    """
    Visualizes the Physics: Kinetic Energy (X) vs Potential Energy (Y).
    """
    if df.empty: return None
    
    # Custom Color Scale based on Vol Ratio (Red = Explosive, Blue = Quiet)
    fig = px.scatter(df, x="Vol_Ratio", y="Z_Score", text="Ticker", 
                     title="Volatility Physics Map (Kinetic vs. Potential)",
                     color="Vol_Ratio", color_continuous_scale="RdBu_r")
    
    fig.update_traces(textposition='top center', marker=dict(size=15, line=dict(width=1, color='black')))
    
    # Add Reference Lines
    fig.add_vline(x=1.0, line_dash="dash", line_color="white", opacity=0.3, annotation_text="Normal Vol")
    fig.add_hline(y=2.0, line_dash="dot", line_color="red", opacity=0.5, annotation_text="+2σ (Stretch)")
    fig.add_hline(y=-2.0, line_dash="dot", line_color="green", opacity=0.5, annotation_text="-2σ (Stretch)")
    fig.add_hline(y=0, line_width=1, line_color="white", opacity=0.2)
    
    fig.update_layout(
        height=400, 
        xaxis_title="Vol Ratio (Kinetic Energy)", 
        yaxis_title="Z-Score (Potential Energy)",
        margin=dict(l=10,r=10,t=30,b=10)
    )
    return fig

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD
# -----------------------------------------------------------------------------
def render_dashboard():
    st.set_page_config(layout="wide", page_title="Swing Command")
    st.title("⚡ Unconstrained Swing Command Center")
    
    with st.spinner("Initializing Data Feeds..."):
        closes = fetch_market_data()
    
    if closes.empty:
        st.error("Data fetch failed. Check yfinance.")
        return

    # --- ROW 1: THE MACRO VIEW ---
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("1. Macro Regime")
        macro_df, vix_state = calculate_macro_regime(closes)
        
        # KPI Row
        k1, k2 = st.columns(2)
        k1.metric("VIX Curve", vix_state, delta_color="inverse")
        if 'DX-Y.NYB' in closes.columns:
            curr = closes['DX-Y.NYB'].iloc[-1]
            prev = closes['DX-Y.NYB'].iloc[-2]
            k2.metric("DXY Index", f"{curr:.2f}", f"{(curr/prev-1)*100:.2f}%")
            
        st.plotly_chart(plot_macro(macro_df), use_container_width=True)

    with c2:
        st.subheader("2. Rotation (RRG)")
        rrg_df = calculate_rrg(closes, SECTOR_TICKERS)
        if not rrg_df.empty:
            st.plotly_chart(plot_rrg(rrg_df), use_container_width=True)
        else:
            st.warning("Insufficient RRG Data")

    st.divider()

    # --- ROW 2: THE EXECUTION VIEW (Expanded) ---
    st.subheader("3. Volatility Physics (Execution)")
    
    c3, c4 = st.columns([2, 1])
    
    with c3:
        # The New Visual
        scanner_df = calculate_vol_scanner(closes, ACTIVE_TICKERS)
        if not scanner_df.empty:
            st.plotly_chart(plot_vol_landscape(scanner_df), use_container_width=True)
        else:
            st.warning("No Scanner Data")
            
    with c4:
        # The Raw Data
        st.caption("Active Watchlist Metrics")
        if not scanner_df.empty:
            # Sort by absolute Z-Score (Most extended first)
            display_df = scanner_df.copy()
            display_df['Abs_Z'] = display_df['Z_Score'].abs()
            display_df = display_df.sort_values('Abs_Z', ascending=False).drop(columns=['Abs_Z'])
            
            st.dataframe(
                display_df[['Ticker', 'Z_Score', 'Vol_Ratio']].style
                .background_gradient(subset=['Vol_Ratio'], cmap='RdYlGn_r', vmin=0.5, vmax=1.5)
                .format({'Z_Score': "{:.2f}σ", 'Vol_Ratio': "{:.2f}"}),
                use_container_width=True,
                height=350
            )

if __name__ == "__main__":
    render_dashboard()
