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
# 1. DATA ENGINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_market_data():
    """Fetches data and fixes gaps (Swiss Cheese problem)."""
    all_tickers = list(set(MACRO_TICKERS + SECTOR_TICKERS + ACTIVE_TICKERS))
    
    # Auto_adjust=False to handle Indices raw data better
    data = yf.download(all_tickers, period="1y", interval="1d", progress=False, auto_adjust=False)
    
    # Robust Column Extraction
    target_col = 'Adj Close' if 'Adj Close' in data.columns.get_level_values(0) else 'Close'
    
    if isinstance(data.columns, pd.MultiIndex):
        if target_col in data.columns.get_level_values(0):
            closes = data.xs(target_col, level=0, axis=1)
        elif target_col in data.columns.get_level_values(1):
            closes = data.xs(target_col, level=1, axis=1)
        else:
            try: closes = data.xs('Close', level=0, axis=1)
            except: closes = data.iloc[:, :len(all_tickers)]
    else:
        closes = data[target_col] if target_col in data.columns else data

    # Fix Columns & Gaps
    closes.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in closes.columns]
    closes = closes.ffill().bfill()
    
    return closes

# -----------------------------------------------------------------------------
# 2. CALCULATION ENGINES
# -----------------------------------------------------------------------------
def calculate_macro_regime(df):
    """Q1: Macro Weather."""
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

def calculate_rrg_smooth(df, sectors, benchmark='SPY', tail_len=15):
    """
    Q2: RRG with Smoothing.
    Applies a rolling mean to RS Ratio/Momentum to create smooth tails.
    """
    if benchmark not in df.columns: return pd.DataFrame()

    frames = []
    
    for s in sectors:
        if s not in df.columns: continue
        
        # 1. Raw Calculations
        rs = df[s] / df[benchmark]
        rs_trend = rs.rolling(window=20).mean()
        rs_ratio = 100 * ((rs / rs_trend) - 1)
        rs_mom = rs_ratio.diff(5)
        
        # 2. THE SMOOTHING FIX
        # We apply a 3-day rolling mean to the coordinates.
        # This removes the "jitter" and makes the tail curvy.
        rs_ratio = rs_ratio.rolling(3).mean()
        rs_mom = rs_mom.rolling(3).mean()
        
        # 3. Extract tail
        if len(rs_ratio) < tail_len: continue
        
        subset_ratio = rs_ratio.tail(tail_len)
        subset_mom = rs_mom.tail(tail_len)
        
        # 4. Determine Quadrant (Current Day)
        curr_r = subset_ratio.iloc[-1]
        curr_m = subset_mom.iloc[-1]
        
        if curr_r > 0 and curr_m > 0: quad = "Leading"
        elif curr_r > 0 and curr_m < 0: quad = "Weakening"
        elif curr_r < 0 and curr_m < 0: quad = "Lagging"
        else: quad = "Improving"

        tmp = pd.DataFrame({
            'Ticker': s,
            'RS_Ratio': subset_ratio,
            'RS_Momentum': subset_mom,
            'Quadrant': quad
        })
        frames.append(tmp)
        
    if not frames: return pd.DataFrame()
    return pd.concat(frames)

def calculate_vol_scanner(df, tickers):
    """Q3: Vol Physics."""
    results = []
    for t in tickers:
        if t not in df.columns: continue
        series = df[t].dropna()
        if len(series) < 30: continue

        curr = series.iloc[-1]
        mean_20 = series.rolling(20).mean().iloc[-1]
        std_20 = series.rolling(20).std().iloc[-1]
        z_score = (curr - mean_20) / std_20 if std_20 != 0 else 0
        
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
    fig.update_layout(height=350, margin=dict(l=10,r=10,t=40,b=10), hovermode="x unified")
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
    return fig

def plot_rrg_tails(df):
    """RRG with increased height and better spacing."""
    if df.empty: return None
    
    fig = go.Figure()
    
    color_map = {
        "Leading": "#4caf50", "Weakening": "#ffeb3b",
        "Lagging": "#f44336", "Improving": "#2196f3"
    }
    
    tickers = df['Ticker'].unique()
    
    for t in tickers:
        t_data = df[df['Ticker'] == t]
        current_quad = t_data['Quadrant'].iloc[-1]
        color = color_map.get(current_quad, "white")
        
        # Tail (Smoothed Line)
        fig.add_trace(go.Scatter(
            x=t_data['RS_Ratio'], 
            y=t_data['RS_Momentum'],
            mode='lines',
            line=dict(color=color, width=2),
            opacity=0.5,
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Head (Marker)
        head = t_data.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[head['RS_Ratio']], 
            y=[head['RS_Momentum']],
            mode='markers+text',
            marker=dict(color=color, size=12, line=dict(width=1, color='black')),
            text=[t],
            textposition="top center",
            name=t,
            showlegend=False
        ))

    # Crosshairs & Annotations
    fig.add_hline(y=0, line_color="white", opacity=0.2)
    fig.add_vline(x=0, line_color="white", opacity=0.2)
    
    # Place labels in corners with padding
    fig.add_annotation(x=4, y=4, text="LEADING", showarrow=False, font=dict(color="#4caf50", size=16, weight="bold"), opacity=0.4)
    fig.add_annotation(x=-4, y=4, text="IMPROVING", showarrow=False, font=dict(color="#2196f3", size=16, weight="bold"), opacity=0.4)
    fig.add_annotation(x=-4, y=-4, text="LAGGING", showarrow=False, font=dict(color="#f44336", size=16, weight="bold"), opacity=0.4)
    fig.add_annotation(x=4, y=-4, text="WEAKENING", showarrow=False, font=dict(color="#ffeb3b", size=16, weight="bold"), opacity=0.4)

    fig.update_layout(
        title="Relative Rotation (Smoothed Flow)",
        xaxis_title="RS Ratio (Trend)",
        yaxis_title="RS Momentum (Velocity)",
        height=600, # INCREASED HEIGHT FOR BREATHING ROOM
        margin=dict(l=20,r=20,t=50,b=20),
        xaxis=dict(range=[-5, 5], constrain='domain'),
        yaxis=dict(range=[-5, 5], scaleanchor="x", scaleratio=1),
        plot_bgcolor='rgba(0,0,0,0)' # Transparent background
    )
    
    return fig

def plot_vol_landscape(df):
    if df.empty: return None
    fig = px.scatter(df, x="Vol_Ratio", y="Z_Score", text="Ticker", 
                     title="Volatility Physics (Kinetic vs Potential)",
                     color="Vol_Ratio", color_continuous_scale="RdBu_r")
    fig.update_traces(textposition='top center', marker=dict(size=14, line=dict(width=1, color='black')))
    fig.add_vline(x=1.0, line_dash="dash", line_color="white", opacity=0.3)
    fig.add_hline(y=2.0, line_dash="dot", line_color="red", opacity=0.5)
    fig.add_hline(y=-2.0, line_dash="dot", line_color="green", opacity=0.5)
    fig.update_layout(height=400, margin=dict(l=10,r=10,t=40,b=10))
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
        st.error("Data fetch failed.")
        return

    # --- ROW 1 ---
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("1. Macro Regime")
        macro_df, vix_state = calculate_macro_regime(closes)
        k1, k2 = st.columns(2)
        k1.metric("VIX Curve", vix_state, delta_color="inverse")
        if 'DX-Y.NYB' in closes.columns:
            curr = closes['DX-Y.NYB'].iloc[-1]
            prev = closes['DX-Y.NYB'].iloc[-2]
            k2.metric("DXY Index", f"{curr:.2f}", f"{(curr/prev-1)*100:.2f}%")
        st.plotly_chart(plot_macro(macro_df), use_container_width=True)

    with c2:
        st.subheader("2. Sector Rotation (Flow)")
        # Call smoothed version
        rrg_df = calculate_rrg_smooth(closes, SECTOR_TICKERS, tail_len=15)
        if not rrg_df.empty:
            st.plotly_chart(plot_rrg_tails(rrg_df), use_container_width=True)
        else:
            st.warning("Insufficient RRG Data")

    st.divider()

    # --- ROW 2 ---
    st.subheader("3. Volatility Physics (Execution)")
    c3, c4 = st.columns([2, 1])
    with c3:
        scanner_df = calculate_vol_scanner(closes, ACTIVE_TICKERS)
        if not scanner_df.empty:
            st.plotly_chart(plot_vol_landscape(scanner_df), use_container_width=True)
            
    with c4:
        st.caption("Active Watchlist Metrics")
        if not scanner_df.empty:
            display_df = scanner_df.sort_values('Vol_Ratio', ascending=False)
            st.dataframe(
                display_df[['Ticker', 'Z_Score', 'Vol_Ratio']].style
                .background_gradient(subset=['Vol_Ratio'], cmap='RdYlGn_r', vmin=0.5, vmax=1.5)
                .format({'Z_Score': "{:.2f}σ", 'Vol_Ratio': "{:.2f}"}),
                use_container_width=True, height=350
            )

if __name__ == "__main__":
    render_dashboard()
