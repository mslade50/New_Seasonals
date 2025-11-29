import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Added ^IRX (3 Month Treasury) and ^TNX (10 Year Treasury) for Yield Curve
MACRO_TICKERS = ['SPY', 'TLT', 'DX-Y.NYB', 'GLD', 'BTC-USD', '^VIX', '^VIX3M', 'AUDJPY=X', '^IRX', '^TNX']
INTERNALS_TICKERS = ['HYG', 'IEF', 'HG=F', 'GC=F', 'XLY', 'XLP', 'RSP']
SECTOR_TICKERS = ['XLK', 'XLE', 'XLF', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLB', 'SMH']
ACTIVE_TICKERS = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'AMZN', 'MSFT', 'META', 'GOOGL', 'NFLX', 'COIN']

# -----------------------------------------------------------------------------
# 1. DATA ENGINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_market_data():
    all_tickers = list(set(MACRO_TICKERS + INTERNALS_TICKERS + SECTOR_TICKERS + ACTIVE_TICKERS + ['SPY']))
    data = yf.download(all_tickers, period="2y", interval="1d", progress=False, auto_adjust=False) # Increased to 2y for regression training
    
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

    closes.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in closes.columns]
    closes = closes.ffill().bfill()
    return closes

# -----------------------------------------------------------------------------
# 2. CALCULATION ENGINES (Standard)
# -----------------------------------------------------------------------------
def calculate_macro_regime(df):
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

def calculate_internals(df):
    internals = pd.DataFrame(index=df.index)
    if 'HYG' in df.columns and 'IEF' in df.columns: internals['Credit_Spreads'] = df['HYG'] / df['IEF']
    if 'HG=F' in df.columns and 'GC=F' in df.columns: internals['Copper_Gold'] = df['HG=F'] / df['GC=F']
    if 'XLY' in df.columns and 'XLP' in df.columns: internals['Discretionary_Staples'] = df['XLY'] / df['XLP']
    if 'RSP' in df.columns and 'SPY' in df.columns: internals['Breadth'] = df['RSP'] / df['SPY']
    
    if not internals.empty:
        subset = internals.tail(20)
        return (subset / subset.iloc[0] - 1) * 100
    return pd.DataFrame()

def calculate_rrg_smooth(df, sectors, benchmark='SPY', tail_len=15):
    if benchmark not in df.columns: return pd.DataFrame()
    frames = []
    for s in sectors:
        if s not in df.columns: continue
        rs = df[s] / df[benchmark]
        rs_trend = rs.rolling(window=20).mean()
        rs_ratio = 100 * ((rs / rs_trend) - 1)
        rs_mom = rs_ratio.diff(5)
        
        # Smooth
        rs_ratio = rs_ratio.rolling(5).mean()
        rs_mom = rs_mom.rolling(5).mean()
        
        if len(rs_ratio) < tail_len: continue
        subset_ratio = rs_ratio.tail(tail_len)
        subset_mom = rs_mom.tail(tail_len)
        
        curr_r = subset_ratio.iloc[-1]
        curr_m = subset_mom.iloc[-1]
        
        if curr_r > 0 and curr_m > 0: quad = "Leading"
        elif curr_r > 0 and curr_m < 0: quad = "Weakening"
        elif curr_r < 0 and curr_m < 0: quad = "Lagging"
        else: quad = "Improving"

        tmp = pd.DataFrame({'Ticker': s, 'RS_Ratio': subset_ratio, 'RS_Momentum': subset_mom, 'Quadrant': quad})
        frames.append(tmp)
    if not frames: return pd.DataFrame()
    return pd.concat(frames)

def calculate_vol_scanner(df, tickers):
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
        results.append({'Ticker': t, 'Price': float(curr), 'Z_Score': float(z_score), 'Vol_Ratio': vol_ratio})
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 3. NEW ENGINE: PREDICTIVE MODELS (Regression)
# -----------------------------------------------------------------------------
def run_predictive_models(df, ticker='SPY'):
    """
    1. HAR-RV Model: Predicts Future Volatility using Past Vol (Daily, Weekly, Monthly components).
    2. Macro-Return Model: Predicts Future Returns using VIX Slope & Yield Curve.
    """
    if ticker not in df.columns: return None, None
    
    # --- DATA PREP ---
    model_df = pd.DataFrame()
    model_df['Close'] = df[ticker]
    model_df['LogRet'] = np.log(df[ticker] / df[ticker].shift(1))
    
    # Realized Volatility (Rolling 5d window for "This Week's Vol")
    model_df['RV_Daily'] = model_df['LogRet'].rolling(window=1).std() * 100 # Proxy for daily
    model_df['RV_Weekly'] = model_df['LogRet'].rolling(window=5).std() * np.sqrt(5) * 100
    model_df['RV_Monthly'] = model_df['LogRet'].rolling(window=22).std() * np.sqrt(22) * 100
    
    # Features for Return Model
    if '^VIX' in df.columns and '^VIX3M' in df.columns:
        model_df['VIX_Slope'] = df['^VIX'] / df['^VIX3M']
    else:
        model_df['VIX_Slope'] = 1.0 # Neutral filler
        
    if '^TNX' in df.columns and '^IRX' in df.columns:
        # TNX is 10y yield, IRX is 13-week yield
        model_df['Yield_Curve'] = df['^TNX'] - df['^IRX']
    else:
        model_df['Yield_Curve'] = 0.0
        
    model_df = model_df.dropna()
    
    # --- MODEL 1: HAR-RV (Forecasting Volatility) ---
    # Target: Next 5 days average volatility
    # Shift target backward so today's row predicts the future
    model_df['Target_Vol'] = model_df['RV_Weekly'].shift(-5)
    
    # Features: Past Daily, Past Weekly, Past Monthly Vol
    features_vol = ['RV_Daily', 'RV_Weekly', 'RV_Monthly']
    
    train_df = model_df.dropna()
    if len(train_df) > 100:
        X_vol = train_df[features_vol]
        y_vol = train_df['Target_Vol']
        
        reg_vol = LinearRegression().fit(X_vol, y_vol)
        current_feats = model_df[features_vol].iloc[-1].values.reshape(1, -1)
        pred_vol = reg_vol.predict(current_feats)[0]
        r2_vol = reg_vol.score(X_vol, y_vol)
    else:
        pred_vol, r2_vol = 0, 0

    # --- MODEL 2: Macro Return Forecast ---
    # Target: Next 5 days Cumulative Return
    model_df['Target_Ret'] = model_df['Close'].pct_change(5).shift(-5) * 100
    
    features_ret = ['VIX_Slope', 'Yield_Curve']
    
    train_df = model_df.dropna()
    if len(train_df) > 100:
        X_ret = train_df[features_ret]
        y_ret = train_df['Target_Ret']
        
        reg_ret = LinearRegression().fit(X_ret, y_ret)
        current_feats_ret = model_df[features_ret].iloc[-1].values.reshape(1, -1)
        pred_ret = reg_ret.predict(current_feats_ret)[0]
        r2_ret = reg_ret.score(X_ret, y_ret)
    else:
        pred_ret, r2_ret = 0, 0
        
    return {
        'Vol_Forecast': pred_vol,
        'Vol_R2': r2_vol,
        'Ret_Forecast': pred_ret,
        'Ret_R2': r2_ret
    }

# -----------------------------------------------------------------------------
# 4. VISUALIZATION
# -----------------------------------------------------------------------------
def plot_macro(df):
    if df.empty: return None
    fig = px.line(df, x=df.index, y=df.columns, title="Global Risk Flow (20d)",
                  color_discrete_map={'SPY':'#4caf50', 'TLT':'#f44336', 'DX-Y.NYB':'#9e9e9e', 'AUDJPY=X':'#03a9f4'})
    fig.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10), hovermode="x unified")
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
    return fig

def plot_internals(df):
    if df.empty: return None
    fig = px.line(df, x=df.index, y=df.columns, title="Divergence Hunter (Internals)",
                  color_discrete_map={'Credit_Spreads':'#4caf50', 'Copper_Gold':'#ff9800', 
                                      'Discretionary_Staples':'#2196f3', 'Breadth':'#9e9e9e'})
    fig.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10), hovermode="x unified")
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
    return fig

def plot_rrg_tails(df):
    if df.empty: return None
    fig = go.Figure()
    color_map = {"Leading": "#4caf50", "Weakening": "#ffeb3b", "Lagging": "#f44336", "Improving": "#2196f3"}
    tickers = df['Ticker'].unique()
    for t in tickers:
        t_data = df[df['Ticker'] == t]
        current_quad = t_data['Quadrant'].iloc[-1]
        color = color_map.get(current_quad, "white")
        fig.add_trace(go.Scatter(x=t_data['RS_Ratio'], y=t_data['RS_Momentum'], mode='lines',
                                 line=dict(color=color, width=2, shape='spline', smoothing=1.3),
                                 opacity=0.6, hoverinfo='skip', showlegend=False))
        head = t_data.iloc[-1]
        fig.add_trace(go.Scatter(x=[head['RS_Ratio']], y=[head['RS_Momentum']], mode='markers+text',
                                 marker=dict(color=color, size=12, line=dict(width=1, color='black')),
                                 text=[t], textposition="top center", name=t, showlegend=False))
    fig.add_hline(y=0, line_color="white", opacity=0.2)
    fig.add_vline(x=0, line_color="white", opacity=0.2)
    fig.add_annotation(x=4, y=4, text="LEADING", showarrow=False, font=dict(color="#4caf50", size=14), opacity=0.4)
    fig.add_annotation(x=-4, y=-4, text="LAGGING", showarrow=False, font=dict(color="#f44336", size=14), opacity=0.4)
    fig.update_layout(title="Relative Rotation (Liquid Flow)", height=500, margin=dict(l=20,r=20,t=40,b=20),
                      xaxis=dict(range=[-5, 5], constrain='domain'), yaxis=dict(range=[-5, 5], scaleanchor="x", scaleratio=1),
                      plot_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_vol_landscape(df):
    if df.empty: return None
    fig = px.scatter(df, x="Vol_Ratio", y="Z_Score", text="Ticker", title="Volatility Physics Map",
                     color="Vol_Ratio", color_continuous_scale="RdBu_r")
    fig.update_traces(textposition='top center', marker=dict(size=14, line=dict(width=1, color='black')))
    fig.add_vline(x=1.0, line_dash="dash", line_color="white", opacity=0.3)
    fig.add_hline(y=2.0, line_dash="dot", line_color="red", opacity=0.5)
    fig.add_hline(y=-2.0, line_dash="dot", line_color="green", opacity=0.5)
    fig.update_layout(height=400, margin=dict(l=10,r=10,t=40,b=10))
    return fig

# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD
# -----------------------------------------------------------------------------
def render_dashboard():
    st.set_page_config(layout="wide", page_title="Swing Command")
    st.title("⚡ Unconstrained Swing Command Center")
    
    with st.spinner("Crunching Global Data & Regression Models..."):
        closes = fetch_market_data()
        pred_metrics = run_predictive_models(closes, 'SPY')
    
    if closes.empty:
        st.error("Data fetch failed.")
        return

    # --- ROW 1: MACRO & INTERNALS ---
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("1. Macro & Internals")
        tab_macro, tab_internal = st.tabs(["Global Flows", "Divergence Hunter"])
        
        with tab_macro:
            macro_df, vix_state = calculate_macro_regime(closes)
            # Regressed Metrics display
            k1, k2, k3 = st.columns(3)
            k1.metric("VIX Curve", vix_state)
            
            # Predictive Output
            if pred_metrics:
                ret_color = "normal" if pred_metrics['Ret_Forecast'] > 0 else "inverse"
                k2.metric("Pred. 5d Return", f"{pred_metrics['Ret_Forecast']:.2f}%", f"R²: {pred_metrics['Ret_R2']:.2f}")
                k3.metric("Pred. 5d Vol", f"{pred_metrics['Vol_Forecast']:.2f}%", f"HAR Model")
            
            st.plotly_chart(plot_macro(macro_df), use_container_width=True)
            
        with tab_internal:
            internals_df = calculate_internals(closes)
            if not internals_df.empty:
                st.info("Logic: If SPY is rising but these lines are falling -> DIVERGENCE (Risk Off)")
                st.plotly_chart(plot_internals(internals_df), use_container_width=True)
            else:
                st.warning("Insufficient Internal Data")

    with c2:
        st.subheader("2. Sector Rotation (Flow)")
        rrg_df = calculate_rrg_smooth(closes, SECTOR_TICKERS, tail_len=15)
        if not rrg_df.empty:
            st.plotly_chart(plot_rrg_tails(rrg_df), use_container_width=True)
        else:
            st.warning("Insufficient RRG Data")

    st.divider()

    # --- ROW 2: EXECUTION ---
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
