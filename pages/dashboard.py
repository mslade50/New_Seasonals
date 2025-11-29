import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Updated: Using DX-Y.NYB for Dollar Index instead of UUP
MACRO_TICKERS = ['SPY', 'TLT', 'DX-Y.NYB', 'GLD', 'BTC-USD', '^VIX', '^VIX3M', 'AUDJPY=X']
SECTOR_TICKERS = ['XLK', 'XLE', 'XLF', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLB', 'SMH']
ACTIVE_TICKERS = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'AMZN', 'MSFT', 'META', 'GOOGL', 'NFLX', 'COIN']

# -----------------------------------------------------------------------------
# 1. DATA ENGINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_market_data():
    """Fetches data handling MultiIndex robustness."""
    all_tickers = list(set(MACRO_TICKERS + SECTOR_TICKERS + ACTIVE_TICKERS))
    
    # Download
    data = yf.download(all_tickers, period="1y", interval="1d", progress=False, auto_adjust=True)
    
    # Robust Column Extraction
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            closes = data.xs('Close', level=0, axis=1)
        elif 'Close' in data.columns.get_level_values(1):
            closes = data.xs('Close', level=1, axis=1)
        else:
            closes = data.iloc[:, :len(all_tickers)]
    else:
        closes = data['Close'] if 'Close' in data.columns else data

    # Clean Columns
    closes.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in closes.columns]
    
    return closes

# -----------------------------------------------------------------------------
# 2. CALCULATION ENGINES
# -----------------------------------------------------------------------------
def calculate_macro_regime(df):
    """Q1: Macro Weather (Risk Gauge & VIX Curve)."""
    # 1. Normalized Performance (20d)
    # Using DX-Y.NYB for Dollar
    targets = ['SPY', 'TLT', 'DX-Y.NYB', 'AUDJPY=X']
    valid_targets = [t for t in targets if t in df.columns]
    
    if not valid_targets: return pd.DataFrame(), "Unknown"

    subset = df[valid_targets].tail(20).copy()
    normalized = (subset / subset.iloc[0] - 1) * 100
    
    # 2. VIX Term Structure
    vix_state = "Neutral"
    if '^VIX' in df.columns and '^VIX3M' in df.columns:
        spot = df['^VIX'].iloc[-1]
        term = df['^VIX3M'].iloc[-1]
        ratio = spot / term
        
        if ratio > 1.05: vix_state = "INVERTED (Risk Off)"
        elif ratio < 0.9: vix_state = "STEEP (Risk On)"
        else: vix_state = "FLAT (Caution)"
        
    return normalized, vix_state

def calculate_rrg(df, sectors, benchmark='SPY'):
    """Q2: Relative Rotation Graph (Trend vs Momentum)."""
    rrg_data = []
    if benchmark not in df.columns: return pd.DataFrame()

    for s in sectors:
        if s not in df.columns: continue
        
        # Relative Strength Calculation
        rs = df[s] / df[benchmark]
        
        # JdK Logic Approximation
        # RS-Ratio: Where is price relative to trend?
        rs_trend = rs.rolling(window=20).mean()
        rs_ratio = 100 * ((rs / rs_trend) - 1)
        
        # RS-Momentum: Is the ratio rising or falling?
        rs_mom = rs_ratio.diff(5)
        
        if len(rs_ratio) > 0:
            curr_ratio = rs_ratio.iloc[-1]
            curr_mom = rs_mom.iloc[-1]
            
            # Quadrant Logic
            if curr_ratio > 0 and curr_mom > 0: quad = "Leading" # Green
            elif curr_ratio > 0 and curr_mom < 0: quad = "Weakening" # Yellow
            elif curr_ratio < 0 and curr_mom < 0: quad = "Lagging" # Red
            else: quad = "Improving" # Blue
            
            rrg_data.append({
                'Ticker': s, 'RS_Ratio': curr_ratio, 'RS_Momentum': curr_mom, 'Quadrant': quad
            })
            
    return pd.DataFrame(rrg_data)

def calculate_pro_scanner(df, tickers):
    """Q3: Vol Compression & Z-Scores."""
    results = []
    for t in tickers:
        if t not in df.columns: continue
        series = df[t].dropna()
        if len(series) < 30: continue

        curr = series.iloc[-1]
        
        # Z-Score
        mean_20 = series.rolling(20).mean().iloc[-1]
        std_20 = series.rolling(20).std().iloc[-1]
        z_score = (curr - mean_20) / std_20 if std_20 != 0 else 0
        
        # Vol Ratio (Short/Long)
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

def calculate_beta_scenario(df, tickers, shock_pct):
    """Q4: Scenario Slider Logic."""
    results = []
    if 'SPY' not in df.columns: return pd.DataFrame()
    
    spy_ret = df['SPY'].pct_change()
    
    for t in tickers:
        if t not in df.columns or t == 'SPY': continue
        asset_ret = df[t].pct_change()
        
        # Beta calc
        common = pd.concat([asset_ret, spy_ret], axis=1).dropna()
        if len(common) < 20: continue
        
        cov = np.cov(common.iloc[:,0], common.iloc[:,1])
        beta = cov[0, 1] / cov[1, 1]
        
        proj = beta * shock_pct
        results.append({'Ticker': t, 'Beta': beta, 'Proj_Move%': proj})
        
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 3. VISUALIZATION
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

def plot_scenario(df, shock):
    if df.empty: return None
    df['Color'] = df['Proj_Move%'].apply(lambda x: '#f44336' if x < 0 else '#4caf50')
    fig = px.bar(df, x='Ticker', y='Proj_Move%', color='Color',
                 title=f"Beta Impact: If SPY moves {shock}%", color_discrete_map="identity")
    fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=10))
    return fig

# -----------------------------------------------------------------------------
# 4. MAIN PAGE
# -----------------------------------------------------------------------------
def render_master_dashboard():
    st.title("⚡ Unconstrained Swing Command Center")
    
    with st.spinner("Fetching Global Data..."):
        closes = fetch_market_data()
    
    if closes.empty:
        st.error("No data. Check connection or yfinance version.")
        return

    # --- ROW 1: THE VIEW FROM 30,000 FT ---
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("1. Macro Regime")
        macro_df, vix_state = calculate_macro_regime(closes)
        
        # KPIs
        m1, m2 = st.columns(2)
        m1.metric("VIX Curve", vix_state)
        if 'DX-Y.NYB' in closes.columns:
            dxy_val = closes['DX-Y.NYB'].iloc[-1]
            dxy_chg = (dxy_val / closes['DX-Y.NYB'].iloc[-2] - 1) * 100
            m2.metric("DXY Index", f"{dxy_val:.2f}", f"{dxy_chg:.2f}%")
            
        st.plotly_chart(plot_macro(macro_df), use_container_width=True)

    with c2:
        st.subheader("2. Sector Rotation (RRG)")
        rrg_df = calculate_rrg(closes, SECTOR_TICKERS)
        if not rrg_df.empty:
            st.plotly_chart(plot_rrg(rrg_df), use_container_width=True)
        else:
            st.warning("RRG Data Unavailable")

    st.divider()

    # --- ROW 2: THE OPPORTUNITY & RISK ---
    c3, c4 = st.columns([2, 1])
    
    with c3:
        st.subheader("3. Interactive Scenario")
        # Slider is the input for the chart below
        shock = st.slider("Market Shock Simulation (SPY %)", -5.0, 5.0, -2.0, 0.5)
        
        scenario_df = calculate_beta_scenario(closes, ACTIVE_TICKERS, shock)
        if not scenario_df.empty:
            st.plotly_chart(plot_scenario(scenario_df, shock), use_container_width=True)
            
    with c4:
        st.subheader("4. Vol Scanner")
        scanner_df = calculate_pro_scanner(closes, ACTIVE_TICKERS)
        if not scanner_df.empty:
            # Highlight interesting setups
            st.dataframe(
                scanner_df[['Ticker', 'Z_Score', 'Vol_Ratio']]
                .sort_values('Vol_Ratio')
                .style.background_gradient(subset=['Vol_Ratio'], cmap='RdYlGn_r', vmin=0.5, vmax=1.5)
                .format({'Z_Score': "{:.2f}σ", 'Vol_Ratio': "{:.2f}"}),
                use_container_width=True,
                height=300
            )

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Swing Master")
    render_master_dashboard()
