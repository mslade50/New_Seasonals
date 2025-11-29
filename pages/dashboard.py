import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Added ^VIX3M for term structure and AUDJPY=X for global risk
MACRO_TICKERS = ['SPY', 'TLT', 'UUP', 'GLD', 'BTC-USD', '^VIX', '^VIX3M', 'AUDJPY=X']
SECTOR_TICKERS = ['XLK', 'XLE', 'XLF', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLB', 'SMH']
ACTIVE_TICKERS = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'AMZN', 'MSFT', 'META', 'GOOGL', 'NFLX', 'COIN']
STRUCT_TICKERS = ['SPY', 'QQQ', 'IWM']

# -----------------------------------------------------------------------------
# 1. DATA ENGINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_market_data():
    all_tickers = list(set(MACRO_TICKERS + SECTOR_TICKERS + ACTIVE_TICKERS))
    
    # Download 1 Year of data
    data = yf.download(all_tickers, period="1y", interval="1d", progress=False, auto_adjust=True)
    
    # Robust Column Extraction (Handles MultiIndex variations)
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            closes = data.xs('Close', level=0, axis=1)
        elif 'Close' in data.columns.get_level_values(1):
            closes = data.xs('Close', level=1, axis=1)
        else:
            closes = data.iloc[:, :len(all_tickers)]
    else:
        closes = data['Close'] if 'Close' in data.columns else data

    closes.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in closes.columns]

    # Structure Data (Volume)
    vol_dict = {}
    for t in STRUCT_TICKERS:
        try:
            vol_df = yf.download(t, period="60d", interval="1d", progress=False, auto_adjust=True)
            if isinstance(vol_df.columns, pd.MultiIndex):
                vol_df.columns = vol_df.columns.get_level_values(0)
            vol_dict[t] = vol_df
        except:
            pass
            
    return closes, vol_dict

# -----------------------------------------------------------------------------
# 2. CALCULATION ENGINES
# -----------------------------------------------------------------------------
def calculate_macro_regime(df):
    """
    Quadrant 1: 
    1. Returns normalized Risk Proxy (AUD/JPY vs SPY vs TLT)
    2. Returns VIX Term Structure state
    """
    # 1. Normalized Lines
    targets = ['SPY', 'TLT', 'AUDJPY=X', 'GLD']
    valid_targets = [t for t in targets if t in df.columns]
    
    if not valid_targets: return pd.DataFrame(), "Unknown"

    subset = df[valid_targets].tail(20).copy()
    normalized = (subset / subset.iloc[0] - 1) * 100
    
    # 2. VIX Term Structure (Contango vs Backwardation)
    vix_state = "Neutral"
    if '^VIX' in df.columns and '^VIX3M' in df.columns:
        spot_vix = df['^VIX'].iloc[-1]
        term_vix = df['^VIX3M'].iloc[-1]
        
        # Calculate Ratio
        ratio = spot_vix / term_vix
        
        if ratio > 1.05: vix_state = "INVERTED (High Fear)"
        elif ratio < 0.9: vix_state = "STEEP (Bullish/Calm)"
        else: vix_state = "FLAT (Caution)"
        
    return normalized, vix_state

def calculate_beta_scenario(df, tickers, shock_pct):
    """
    Interactive Element: Calculates projected move based on Beta to SPY.
    """
    results = []
    if 'SPY' not in df.columns: return pd.DataFrame()
    
    spy_ret = df['SPY'].pct_change()
    
    for t in tickers:
        if t not in df.columns or t == 'SPY': continue
        
        asset_ret = df[t].pct_change()
        
        # Calculate Beta (60-day lookback)
        cov_matrix = np.cov(asset_ret.tail(60), spy_ret.tail(60))
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        
        # Projected Move
        proj_move = beta * shock_pct
        
        results.append({
            'Ticker': t,
            'Beta': beta,
            'Proj_Move%': proj_move
        })
        
    return pd.DataFrame(results)

def calculate_pro_scanner(df, tickers):
    """
    Quadrant 3: Adds 'Vol Compression' proxy for IV Rank.
    """
    results = []
    
    for t in tickers:
        if t not in df.columns: continue
        series = df[t].dropna()
        if len(series) < 30: continue

        curr = series.iloc[-1]
        
        # Z-Score
        mean_20 = series.rolling(20).mean().iloc[-1]
        std_20 = series.rolling(20).std().iloc[-1]
        z_score = (curr - mean_20) / std_20
        
        # Volatility Compression (Short Vol / Long Vol)
        # If Ratio < 0.7, Vol is compressed (Breakout imminent)
        # If Ratio > 1.5, Vol is expanded (Reversion likely)
        log_ret = np.log(series / series.shift(1))
        vol_5d = log_ret.rolling(5).std().iloc[-1]
        vol_20d = log_ret.rolling(20).std().iloc[-1]
        vol_ratio = vol_5d / vol_20d if vol_20d != 0 else 1.0
        
        results.append({
            'Ticker': t,
            'Price': curr,
            'Z_Score': z_score,
            'Vol_Ratio': vol_ratio,
            'Vol_Ann%': vol_20d * np.sqrt(252) * 100
        })
        
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 3. VISUALIZATION
# -----------------------------------------------------------------------------
def plot_macro_complex(df):
    fig = px.line(df, x=df.index, y=df.columns, title="Risk-On Gauge (20d)",
                  color_discrete_map={'SPY':'green', 'TLT':'red', 'AUDJPY=X':'cyan', 'GLD':'gold'})
    fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=10), showlegend=True)
    return fig

def plot_scenario_impact(df, shock_val):
    if df.empty: return None
    
    # Color mapping for damage
    df['Color'] = df['Proj_Move%'].apply(lambda x: 'red' if x < 0 else 'green')
    
    fig = px.bar(df, x='Ticker', y='Proj_Move%', color='Color',
                 title=f"Portfolio Impact if SPY moves {shock_val}%",
                 color_discrete_map={'red':'#ff4b4b', 'green':'#4caf50'})
    
    fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=10), showlegend=False)
    fig.add_hline(y=0, line_color="white", line_width=1)
    return fig

# -----------------------------------------------------------------------------
# 4. DASHBOARD RENDER
# -----------------------------------------------------------------------------
def render_pro_dashboard():
    st.title("⚡ Unconstrained Swing Command Center")
    
    # Fetch Data
    with st.spinner("Connecting to Data Feeds..."):
        closes, vol_dict = fetch_market_data()
        
    if closes.empty:
        st.error("Data fetch failed. Update yfinance.")
        return

    # --- ROW 1: MACRO & REGIME ---
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("1. Macro Weather Station")
        macro_df, vix_state = calculate_macro_regime(closes)
        
        # KPI Metric Row
        k1, k2, k3 = st.columns(3)
        k1.metric("Market Regime", vix_state)
        
        # Dollar Strength
        if 'UUP' in closes.columns:
            uup_chg = (closes['UUP'].iloc[-1] / closes['UUP'].iloc[-2] - 1) * 100
            k2.metric("USD Proxy (UUP)", f"{closes['UUP'].iloc[-1]:.2f}", f"{uup_chg:.2f}%")
        
        # Risk Proxy
        if 'AUDJPY=X' in closes.columns:
            aud_chg = (closes['AUDJPY=X'].iloc[-1] / closes['AUDJPY=X'].iloc[-2] - 1) * 100
            k3.metric("Risk Sentiment (AUD/JPY)", f"{closes['AUDJPY=X'].iloc[-1]:.2f}", f"{aud_chg:.2f}%")
            
        st.plotly_chart(plot_macro_complex(macro_df), use_container_width=True)

    with c2:
        st.subheader("2. Vol Compression Scanner")
        st.caption("Ratio < 0.7 = Squeeze/Breakout | > 1.3 = Overextended")
        
        scanner_df = calculate_pro_scanner(closes, ACTIVE_TICKERS)
        
        # Filter for interesting setups
        if not scanner_df.empty:
            df_display = scanner_df[['Ticker', 'Z_Score', 'Vol_Ratio']].sort_values('Vol_Ratio')
            
            st.dataframe(
                df_display.style.background_gradient(subset=['Vol_Ratio'], cmap='coolwarm', vmin=0.5, vmax=1.5)
                .format({'Z_Score': "{:.2f}", 'Vol_Ratio': "{:.2f}"}),
                use_container_width=True,
                height=300
            )

    st.divider()

    # --- ROW 2: SCENARIO ANALYSIS (The "Killer Feature") ---
    st.subheader("3. Dynamic Risk Scenario")
    
    s1, s2 = st.columns([1, 3])
    with s1:
        st.markdown("### 🎛️ Simulation")
        shock_input = st.slider("SPY Shock (%)", min_value=-5.0, max_value=5.0, value=-2.0, step=0.5)
        st.info("Calculates Beta-weighted impact on active watchlist.")
        
    with s2:
        scenario_df = calculate_beta_scenario(closes, ACTIVE_TICKERS, shock_input)
        st.plotly_chart(plot_scenario_impact(scenario_df, shock_input), use_container_width=True)

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Swing Pro")
    render_pro_dashboard()
