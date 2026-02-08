"""
regime_analysis.py — Market Regime Conditioning Analysis
=========================================================
Answers: "In what market environments does our system perform best/worst?"

Computes external market state variables (sector dispersion, correlation,
breadth, VIX term structure) and conditions the strat_backtester trade log
on each variable to find statistically meaningful performance splits.

LOCATION: pages/regime_analysis.py (Streamlit multipage app)
DEPENDENCY: Imports from strat_backtester.py and strategy_config.py

Author: McKinley (built with Claude)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import time
import sys
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# PATH SETUP - add both repo root (for strategy_config) and pages/ (for sibling imports)
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add repo root for strategy_config.py
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add pages/ directory so we can import strat_backtester as a sibling module
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from strategy_config import _STRATEGY_BOOK_RAW, ACCOUNT_VALUE
except ImportError:
    st.error("Could not find strategy_config.py in the root directory.")
    st.stop()

# Import portfolio sim functions from strat_backtester (sibling in pages/)
try:
    from strat_backtester import (
        download_historical_data,
        precompute_all_indicators,
        generate_candidates_fast,
        process_signals_fast,
        get_daily_mtm_series,
        build_price_matrix,
        load_seasonal_map
    )
except ImportError as e:
    st.error(f"Could not import from strat_backtester.py: {e}")
    st.stop()


# =============================================================================
# CONSTANTS
# =============================================================================

# Core GICS Sector SPDRs for dispersion calculation
# 11 sectors = the standard decomposition of S&P 500
SECTOR_ETFS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']

# Additional regime tickers
REGIME_TICKERS = ['SPY', '^VIX', 'VIX3M']

# All tickers needed for regime analysis (deduplicated)
ALL_REGIME_TICKERS = list(set(SECTOR_ETFS + REGIME_TICKERS))


# =============================================================================
# REGIME VARIABLE COMPUTATION
# =============================================================================

@st.cache_data(show_spinner="Downloading regime data...")
def download_regime_data(start_date: str, end_date: str) -> dict:
    """
    Download price data for all regime-relevant tickers.
    Returns dict of {ticker: DataFrame}.
    """
    regime_dict = {}
    
    # Download sector ETFs + SPY in batch
    tickers_to_download = SECTOR_ETFS + ['SPY']
    
    try:
        raw = yf.download(tickers_to_download, start=start_date, end=end_date, 
                          auto_adjust=True, threads=True, progress=False)
        
        if raw.empty:
            st.warning("Failed to download regime data.")
            return regime_dict
        
        # Handle MultiIndex columns from yfinance
        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in tickers_to_download:
                try:
                    t_df = raw.xs(ticker, level=1, axis=1).copy()
                    t_df.columns = [c.capitalize() for c in t_df.columns]
                    t_df = t_df.dropna(subset=['Close'])
                    if not t_df.empty:
                        regime_dict[ticker] = t_df
                except KeyError:
                    continue
        else:
            # Single ticker edge case (shouldn't happen with batch)
            raw.columns = [c.capitalize() for c in raw.columns]
            regime_dict[tickers_to_download[0]] = raw
    except Exception as e:
        st.warning(f"Batch download failed: {e}. Trying individual downloads...")
        for ticker in tickers_to_download:
            try:
                t_df = yf.download(ticker, start=start_date, end=end_date,
                                   auto_adjust=True, progress=False)
                if isinstance(t_df.columns, pd.MultiIndex):
                    t_df.columns = t_df.columns.get_level_values(0)
                t_df.columns = [c.capitalize() for c in t_df.columns]
                if not t_df.empty:
                    regime_dict[ticker] = t_df
            except Exception:
                continue
    
    # Download VIX separately (index tickers sometimes need individual calls)
    for vix_ticker in ['^VIX', 'VIX3M']:
        try:
            v_df = yf.download(vix_ticker, start=start_date, end=end_date,
                               auto_adjust=True, progress=False)
            if isinstance(v_df.columns, pd.MultiIndex):
                v_df.columns = v_df.columns.get_level_values(0)
            v_df.columns = [c.capitalize() for c in v_df.columns]
            if not v_df.empty:
                regime_dict[vix_ticker] = v_df
        except Exception:
            continue
    
    return regime_dict


def compute_sector_dispersion(regime_dict: dict, windows: list = [1, 5, 21]) -> pd.DataFrame:
    """
    Cross-sectional standard deviation of sector ETF returns.
    
    For each day and each lookback window:
    - Compute trailing N-day return for each sector ETF
    - Take stdev across all 11 sectors = that day's dispersion reading
    
    Returns DataFrame indexed by date with columns like 'disp_1d', 'disp_5d', 'disp_21d'
    """
    # Build a returns matrix: rows=dates, cols=sector tickers
    sector_closes = {}
    for ticker in SECTOR_ETFS:
        if ticker in regime_dict:
            sector_closes[ticker] = regime_dict[ticker]['Close']
    
    if len(sector_closes) < 8:
        st.warning(f"Only {len(sector_closes)}/11 sector ETFs available. Dispersion may be less reliable.")
    
    if len(sector_closes) < 3:
        return pd.DataFrame()
    
    close_matrix = pd.DataFrame(sector_closes)
    
    dispersion = pd.DataFrame(index=close_matrix.index)
    
    for w in windows:
        # Trailing N-day return for each sector
        returns = close_matrix.pct_change(w, fill_method=None)
        # Cross-sectional stdev on each day
        dispersion[f'disp_{w}d'] = returns.std(axis=1)
        # Also compute the percentile rank (rolling) for easier interpretation
        dispersion[f'disp_{w}d_rank'] = dispersion[f'disp_{w}d'].expanding(min_periods=63).rank(pct=True) * 100
    
    return dispersion


def compute_correlation_regime(regime_dict: dict, window: int = 21) -> pd.Series:
    """
    Rolling average pairwise correlation of sector ETF daily returns.
    
    High correlation = everything moving together (risk-on/off regime).
    Low correlation = idiosyncratic moves (stock picker's market).
    """
    sector_closes = {}
    for ticker in SECTOR_ETFS:
        if ticker in regime_dict:
            sector_closes[ticker] = regime_dict[ticker]['Close']
    
    if len(sector_closes) < 5:
        return pd.Series(dtype=float)
    
    close_matrix = pd.DataFrame(sector_closes)
    daily_rets = close_matrix.pct_change().dropna()
    
    # Rolling correlation matrix -> extract mean pairwise correlation
    avg_corr = pd.Series(dtype=float, index=daily_rets.index)
    
    for i in range(window, len(daily_rets)):
        window_rets = daily_rets.iloc[i - window:i]
        corr_matrix = window_rets.corr()
        # Extract upper triangle (exclude diagonal)
        n = len(corr_matrix)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        pairwise_corrs = corr_matrix.values[mask]
        avg_corr.iloc[i] = np.nanmean(pairwise_corrs)
    
    return avg_corr


def compute_vix_term_structure(regime_dict: dict) -> pd.Series:
    """
    VIX / VIX3M ratio. 
    > 1.0 = backwardation (fear, near-term stress)
    < 1.0 = contango (complacency, normal)
    """
    if '^VIX' not in regime_dict or 'VIX3M' not in regime_dict:
        return pd.Series(dtype=float)
    
    vix = regime_dict['^VIX']['Close']
    vix3m = regime_dict['VIX3M']['Close']
    
    # Align indices
    combined = pd.DataFrame({'VIX': vix, 'VIX3M': vix3m}).dropna()
    
    if combined.empty:
        return pd.Series(dtype=float)
    
    ratio = combined['VIX'] / combined['VIX3M']
    return ratio


def compute_breadth(regime_dict: dict, universe_dict: dict, window: int = 20) -> pd.Series:
    """
    % of tickers in the trading universe above their 20 SMA.
    Uses the actual portfolio's price data to be maximally relevant.
    """
    above_sma = {}
    
    for ticker, df in universe_dict.items():
        if df is None or df.empty:
            continue
        temp = df.copy()
        if isinstance(temp.columns, pd.MultiIndex):
            temp.columns = [c[0] if isinstance(c, tuple) else c for c in temp.columns]
        temp.columns = [c.capitalize() for c in temp.columns]
        
        if 'Close' not in temp.columns:
            continue
        
        sma = temp['Close'].rolling(window).mean()
        above_sma[ticker] = (temp['Close'] > sma).astype(float)
    
    if not above_sma:
        return pd.Series(dtype=float)
    
    above_df = pd.DataFrame(above_sma)
    # % of tickers above SMA on each day
    breadth = above_df.mean(axis=1) * 100
    
    return breadth


def compute_signal_density(sig_df: pd.DataFrame) -> pd.Series:
    """
    How many signals fired on each entry date.
    High density = broad market dislocation (correlated signals).
    Low density = idiosyncratic opportunity.
    """
    if sig_df.empty:
        return pd.Series(dtype=float)
    
    return sig_df.groupby('Date').size()


# =============================================================================
# ANALYSIS ENGINE
# =============================================================================

def bucket_and_analyze(sig_df: pd.DataFrame, regime_series: pd.Series, 
                       regime_name: str, n_buckets: int = 2,
                       use_percentile: bool = True) -> pd.DataFrame:
    """
    Merge a regime variable onto the trade log by entry date,
    then split into buckets and compare performance.
    
    Args:
        sig_df: Trade log with 'Date', 'PnL', 'Risk $', 'Strategy' columns
        regime_series: Daily series of the regime variable
        regime_name: Human-readable name for display
        n_buckets: Number of buckets (2 = median split, 3 = terciles, etc.)
        use_percentile: If True, bucket by percentile rank rather than raw value
    
    Returns:
        DataFrame with one row per bucket showing performance metrics
    """
    if sig_df.empty or regime_series.empty:
        return pd.DataFrame()
    
    df = sig_df.copy()
    
    # Map regime value to each trade's entry date
    df['regime_val'] = df['Date'].map(regime_series)
    df = df.dropna(subset=['regime_val'])
    
    if len(df) < 20:
        return pd.DataFrame()
    
    # Compute R-Multiple for each trade
    df['R_Multiple'] = np.where(df['Risk $'] > 0, df['PnL'] / df['Risk $'], 0)
    
    # Create buckets
    if n_buckets == 2:
        median_val = df['regime_val'].median()
        df['bucket'] = np.where(df['regime_val'] <= median_val, 'Low', 'High')
        bucket_order = ['Low', 'High']
    elif n_buckets == 3:
        df['bucket'] = pd.qcut(df['regime_val'], 3, labels=['Low', 'Mid', 'High'], duplicates='drop')
        bucket_order = ['Low', 'Mid', 'High']
    else:
        df['bucket'] = pd.qcut(df['regime_val'], n_buckets, duplicates='drop')
        bucket_order = sorted(df['bucket'].unique())
    
    results = []
    for bucket in bucket_order:
        b_df = df[df['bucket'] == bucket]
        if len(b_df) < 5:
            continue
        
        wins = b_df[b_df['PnL'] > 0]
        losses = b_df[b_df['PnL'] <= 0]
        
        avg_r = b_df['R_Multiple'].mean()
        win_rate = len(wins) / len(b_df)
        avg_pnl = b_df['PnL'].mean()
        total_pnl = b_df['PnL'].sum()
        
        # Profit factor
        gross_profit = wins['PnL'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['PnL'].sum()) if len(losses) > 0 else 1
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Regime value range in this bucket
        val_min = b_df['regime_val'].min()
        val_max = b_df['regime_val'].max()
        
        results.append({
            'Bucket': bucket,
            'Regime Range': f"{val_min:.3f} - {val_max:.3f}",
            'Trades': len(b_df),
            'Win Rate': win_rate,
            'Avg R': avg_r,
            'Avg PnL': avg_pnl,
            'Total PnL': total_pnl,
            'Profit Factor': pf,
        })
    
    return pd.DataFrame(results)


def analyze_by_strategy(sig_df: pd.DataFrame, regime_series: pd.Series,
                        regime_name: str) -> pd.DataFrame:
    """
    Same as bucket_and_analyze but broken out by strategy.
    Only uses 2-bucket (median) split for sample size.
    """
    if sig_df.empty or regime_series.empty:
        return pd.DataFrame()
    
    df = sig_df.copy()
    df['regime_val'] = df['Date'].map(regime_series)
    df = df.dropna(subset=['regime_val'])
    
    if len(df) < 20:
        return pd.DataFrame()
    
    df['R_Multiple'] = np.where(df['Risk $'] > 0, df['PnL'] / df['Risk $'], 0)
    
    # Global median split (same threshold for all strategies)
    median_val = df['regime_val'].median()
    df['bucket'] = np.where(df['regime_val'] <= median_val, 'Low', 'High')
    
    results = []
    for strat in df['Strategy'].unique():
        s_df = df[df['Strategy'] == strat]
        
        for bucket in ['Low', 'High']:
            b_df = s_df[s_df['bucket'] == bucket]
            if len(b_df) < 5:
                continue
            
            results.append({
                'Strategy': strat,
                'Regime': bucket,
                'Trades': len(b_df),
                'Win Rate': (b_df['PnL'] > 0).mean(),
                'Avg R': b_df['R_Multiple'].mean(),
                'Avg PnL': b_df['PnL'].mean(),
                'Total PnL': b_df['PnL'].sum(),
            })
    
    return pd.DataFrame(results)


def compute_statistical_significance(sig_df: pd.DataFrame, regime_series: pd.Series) -> dict:
    """
    Welch's t-test between High and Low regime bucket R-multiples.
    Returns t-stat, p-value, and interpretation.
    """
    df = sig_df.copy()
    df['regime_val'] = df['Date'].map(regime_series)
    df = df.dropna(subset=['regime_val'])
    
    if len(df) < 20:
        return {'t_stat': np.nan, 'p_value': np.nan, 'significant': False, 'interpretation': 'Insufficient data'}
    
    df['R_Multiple'] = np.where(df['Risk $'] > 0, df['PnL'] / df['Risk $'], 0)
    
    median_val = df['regime_val'].median()
    low = df[df['regime_val'] <= median_val]['R_Multiple']
    high = df[df['regime_val'] > median_val]['R_Multiple']
    
    if len(low) < 10 or len(high) < 10:
        return {'t_stat': np.nan, 'p_value': np.nan, 'significant': False, 'interpretation': 'Insufficient data per bucket'}
    
    t_stat, p_value = scipy_stats.ttest_ind(low, high, equal_var=False)
    
    significant = p_value < 0.05
    
    if significant:
        better = "Low dispersion" if low.mean() > high.mean() else "High dispersion"
        interpretation = f"Statistically significant (p={p_value:.4f}). {better} regime performs better."
    else:
        interpretation = f"Not statistically significant (p={p_value:.4f}). No reliable difference between regimes."
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'significant': significant,
        'interpretation': interpretation,
        'n_low': len(low),
        'n_high': len(high),
        'mean_low': low.mean(),
        'mean_high': high.mean()
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_regime_timeseries_chart(regime_series: pd.Series, daily_pnl: pd.Series,
                                   regime_name: str, starting_equity: float) -> go.Figure:
    """
    Dual-panel chart:
    - Top: Equity curve
    - Bottom: Regime variable with shading for high/low
    """
    equity = starting_equity + daily_pnl.cumsum()
    
    # Align dates
    common_dates = equity.index.intersection(regime_series.index)
    if len(common_dates) < 20:
        return None
    
    equity_aligned = equity.loc[common_dates]
    regime_aligned = regime_series.loc[common_dates]
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        subplot_titles=('Portfolio Equity', f'{regime_name}'),
        vertical_spacing=0.12,
        shared_xaxes=True
    )
    
    # Panel 1: Equity
    fig.add_trace(go.Scatter(
        x=equity_aligned.index, y=equity_aligned,
        name='Equity', line=dict(color='#00CC00', width=2)
    ), row=1, col=1)
    
    fig.add_hline(y=starting_equity, line_dash="dot", line_color="gray",
                  opacity=0.5, row=1, col=1)
    
    # Panel 2: Regime variable with median line
    median_val = regime_aligned.median()
    
    fig.add_trace(go.Scatter(
        x=regime_aligned.index, y=regime_aligned,
        name=regime_name, line=dict(color='#FF9900', width=1.5)
    ), row=2, col=1)
    
    fig.add_hline(y=median_val, line_dash="dash", line_color="white",
                  opacity=0.7, row=2, col=1,
                  annotation_text=f"Median: {median_val:.4f}")
    
    # Shade high-dispersion periods
    high_regime = regime_aligned > regime_aligned.quantile(0.75)
    
    fig.update_layout(
        height=600,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_bucket_comparison_chart(bucket_df: pd.DataFrame, regime_name: str) -> go.Figure:
    """
    Side-by-side bar chart comparing performance metrics across regime buckets.
    """
    if bucket_df.empty:
        return None
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Avg R-Multiple', 'Win Rate', 'Avg PnL ($)')
    )
    
    colors = {'Low': '#00CC00', 'Mid': '#FF9900', 'High': '#CC0000'}
    
    for _, row in bucket_df.iterrows():
        bucket = row['Bucket']
        color = colors.get(bucket, '#888888')
        
        fig.add_trace(go.Bar(
            x=[bucket], y=[row['Avg R']],
            name=bucket, marker_color=color, showlegend=False,
            text=[f"{row['Avg R']:.3f}"], textposition='outside'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=[bucket], y=[row['Win Rate'] * 100],
            name=bucket, marker_color=color, showlegend=False,
            text=[f"{row['Win Rate']:.1%}"], textposition='outside'
        ), row=1, col=2)
        
        fig.add_trace(go.Bar(
            x=[bucket], y=[row['Avg PnL']],
            name=bucket, marker_color=color, showlegend=False,
            text=[f"${row['Avg PnL']:,.0f}"], textposition='outside'
        ), row=1, col=3)
    
    fig.update_layout(
        height=350,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        title_text=f"Performance by {regime_name} Regime"
    )
    
    return fig


def create_strategy_heatmap(strat_df: pd.DataFrame, metric: str = 'Avg R') -> go.Figure:
    """
    Heatmap: rows = strategies, cols = regime buckets, values = metric.
    Quick visual scan of which strategies suffer most in which regimes.
    """
    if strat_df.empty:
        return None
    
    pivot = strat_df.pivot_table(index='Strategy', columns='Regime', values=metric, aggfunc='first')
    
    if pivot.empty:
        return None
    
    # Ensure column order
    col_order = [c for c in ['Low', 'High'] if c in pivot.columns]
    pivot = pivot[col_order]
    
    # Add a "Delta" column
    if 'Low' in pivot.columns and 'High' in pivot.columns:
        pivot['Delta (L-H)'] = pivot['Low'] - pivot['High']
        pivot = pivot.sort_values('Delta (L-H)', ascending=False)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale='RdYlGn',
        text=[[f"{v:.3f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont={"size": 12},
        zmid=0
    ))
    
    fig.update_layout(
        height=max(300, len(pivot) * 40 + 100),
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        title_text=f"{metric} by Strategy × Regime"
    )
    
    return fig


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    st.set_page_config(page_title="Regime Analysis", layout="wide")
    st.title("🌡️ Market Regime Conditioning")
    st.caption("Conditioning portfolio performance on external market state variables to identify when the system's edge strengthens or degrades.")
    
    # -------------------------------------------------------------------------
    # SIDEBAR CONFIG
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        starting_equity = st.number_input("Starting Equity ($)", 
                                           value=ACCOUNT_VALUE, step=50000)
        
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input("Start Year", 2000, 2025, 2015)
        with col2:
            end_date = st.date_input("End Date", datetime.date.today())
        
        user_start_date = datetime.date(start_year, 1, 1)
        
        st.divider()
        st.subheader("Regime Variables")
        
        run_sector_disp = st.checkbox("Sector Dispersion", value=True, 
                                       help="Cross-sectional stdev of sector ETF returns")
        run_correlation = st.checkbox("Avg Pairwise Correlation", value=True,
                                      help="Rolling 21d avg correlation across sectors")
        run_vix_term = st.checkbox("VIX Term Structure", value=True,
                                    help="VIX/VIX3M ratio (backwardation vs contango)")
        run_breadth = st.checkbox("Market Breadth", value=True,
                                   help="% of universe above 20 SMA")
        run_density = st.checkbox("Signal Density", value=True,
                                   help="# of signals per day (internal)")
        
        st.divider()
        n_buckets = st.radio("Split Method", [2, 3], index=0, 
                              format_func=lambda x: f"{x} buckets ({'Median' if x == 2 else 'Terciles'})")
        
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # -------------------------------------------------------------------------
    # MAIN EXECUTION
    # -------------------------------------------------------------------------
    if not run_button:
        st.info("👈 Configure settings and click 'Run Analysis' to begin.")
        
        with st.expander("ℹ️ What this page does"):
            st.markdown("""
            This tool answers: **"In what market environments does our system perform best/worst?"**
            
            It runs the full portfolio backtest (same as the Strategy Backtester), then overlays 
            external market state variables and splits performance into regime buckets.
            
            **Variables tested:**
            - **Sector Dispersion:** How spread out are sector returns? High = macro rotation, low = everything correlated
            - **Avg Correlation:** Rolling pairwise correlation across sectors. High = herding, low = stock-picking market
            - **VIX Term Structure:** VIX/VIX3M ratio. > 1 = near-term fear (backwardation)
            - **Market Breadth:** % of our universe above 20 SMA. Broad vs narrow participation
            - **Signal Density:** Internal metric — how many signals fired that day?
            
            **Methodology:** For each variable, trades are split at the median (or into terciles) and 
            we compare Avg R-multiple, Win Rate, and P&L. A Welch's t-test checks statistical significance.
            """)
        return
    
    # -------------------------------------------------------------------------
    # STEP 1: Run the portfolio backtest (reuse strat_backtester pipeline)
    # -------------------------------------------------------------------------
    st.header("1️⃣ Running Portfolio Backtest")
    
    import copy
    strategies = copy.deepcopy(_STRATEGY_BOOK_RAW)
    
    # Gather all unique tickers
    all_tickers = set()
    for strat in strategies:
        for t in strat['universe_tickers']:
            all_tickers.add(t.replace('.', '-'))
    
    progress = st.progress(0, text="Downloading historical data...")
    
    master_dict = download_historical_data(
        list(all_tickers), 
        start_date="2000-01-01"
    )
    
    progress.progress(25, text="Loading seasonal data...")
    
    # Load seasonal data (same function as strat_backtester uses)
    sznl_map = load_seasonal_map()
    
    # VIX series
    vix_series = None
    vix_df = master_dict.get('^VIX')
    if vix_df is not None and not vix_df.empty:
        temp = vix_df.copy()
        if isinstance(temp.columns, pd.MultiIndex):
            temp.columns = [c[0] if isinstance(c, tuple) else c for c in temp.columns]
        temp.columns = [c.capitalize() for c in temp.columns]
        vix_series = temp['Close']
    
    progress.progress(35, text="Computing indicators...")
    
    processed = precompute_all_indicators(master_dict, strategies, sznl_map, vix_series)
    
    progress.progress(50, text="Generating signal candidates...")
    
    candidates, signal_data = generate_candidates_fast(processed, strategies, sznl_map, user_start_date)
    
    if not candidates:
        st.error("No signal candidates generated. Check date range and strategy config.")
        st.stop()
    
    progress.progress(65, text="Processing trades with MTM sizing...")
    
    # process_signals_fast takes candidates, signal_data, processed_dict, strategies, starting_equity
    sig_df = process_signals_fast(candidates, signal_data, processed, strategies, starting_equity)
    
    # Filter to start date
    sig_df = sig_df[sig_df['Date'] >= pd.Timestamp(user_start_date)]
    
    if sig_df.empty:
        st.error("No trades after filtering. Check date range.")
        st.stop()
    
    # Get daily P&L for equity curve
    daily_pnl = get_daily_mtm_series(sig_df, master_dict, start_date=user_start_date)
    
    st.success(f"Backtest complete: **{len(sig_df)}** trades across **{sig_df['Strategy'].nunique()}** strategies")
    
    progress.progress(70, text="Downloading regime data...")
    
    # -------------------------------------------------------------------------
    # STEP 2: Compute regime variables
    # -------------------------------------------------------------------------
    st.header("2️⃣ Computing Regime Variables")
    
    regime_dict = download_regime_data(str(user_start_date), str(end_date))
    
    # Also make sector ETFs available from master_dict (already downloaded with full history)
    # These use auto_adjust=False like the main pipeline, but for returns-based
    # calculations (dispersion, correlation) the adjustment method doesn't matter
    for ticker in SECTOR_ETFS + ['SPY']:
        if ticker not in regime_dict and ticker in master_dict:
            temp = master_dict[ticker].copy()
            if isinstance(temp.columns, pd.MultiIndex):
                temp.columns = [c[0] if isinstance(c, tuple) else c for c in temp.columns]
            temp.columns = [c.capitalize() for c in temp.columns]
            regime_dict[ticker] = temp
    
    regime_variables = {}  # {name: pd.Series}
    
    if run_sector_disp:
        disp_df = compute_sector_dispersion(regime_dict, windows=[1, 5, 21])
        if not disp_df.empty:
            regime_variables['Sector Dispersion (5d)'] = disp_df['disp_5d']
            regime_variables['Sector Dispersion (1d)'] = disp_df['disp_1d']
            regime_variables['Sector Dispersion (21d)'] = disp_df['disp_21d']
    
    if run_correlation:
        avg_corr = compute_correlation_regime(regime_dict, window=21)
        if not avg_corr.empty:
            regime_variables['Avg Sector Correlation (21d)'] = avg_corr
    
    if run_vix_term:
        vix_ratio = compute_vix_term_structure(regime_dict)
        if not vix_ratio.empty:
            regime_variables['VIX Term Structure'] = vix_ratio
    
    if run_breadth:
        breadth = compute_breadth(regime_dict, master_dict, window=20)
        if not breadth.empty:
            regime_variables['Breadth (% > 20 SMA)'] = breadth
    
    if run_density:
        density = compute_signal_density(sig_df)
        if not density.empty:
            regime_variables['Signal Density (# same day)'] = density
    
    progress.progress(85, text="Analyzing performance by regime...")
    
    if not regime_variables:
        st.error("No regime variables could be computed. Check data availability.")
        st.stop()
    
    st.success(f"Computed **{len(regime_variables)}** regime variables")
    
    # -------------------------------------------------------------------------
    # STEP 3: Condition performance on each regime variable
    # -------------------------------------------------------------------------
    st.header("3️⃣ Performance by Market Regime")
    
    # Summary card for quick scanning
    summary_rows = []
    
    for regime_name, regime_series in regime_variables.items():
        st.subheader(f"📊 {regime_name}")
        
        # Statistical significance test
        sig_test = compute_statistical_significance(sig_df, regime_series)
        
        # Bucket analysis (portfolio level)
        bucket_df = bucket_and_analyze(sig_df, regime_series, regime_name, n_buckets=n_buckets)
        
        if bucket_df.empty:
            st.warning(f"Insufficient data to analyze {regime_name}")
            continue
        
        # Display significance
        if sig_test['significant']:
            st.success(f"✅ {sig_test['interpretation']}")
        else:
            st.info(f"⚪ {sig_test['interpretation']}")
        
        # Add to summary
        summary_rows.append({
            'Variable': regime_name,
            't-stat': sig_test.get('t_stat', np.nan),
            'p-value': sig_test.get('p_value', np.nan),
            'Significant?': '✅' if sig_test.get('significant', False) else '⚪',
            'Low Avg R': sig_test.get('mean_low', np.nan),
            'High Avg R': sig_test.get('mean_high', np.nan),
            'n (Low)': sig_test.get('n_low', 0),
            'n (High)': sig_test.get('n_high', 0),
        })
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Bucket performance table
            st.dataframe(
                bucket_df.style.format({
                    'Win Rate': '{:.1%}',
                    'Avg R': '{:.3f}',
                    'Avg PnL': '${:,.0f}',
                    'Total PnL': '${:,.0f}',
                    'Profit Factor': '{:.2f}',
                }),
                use_container_width=True
            )
        
        with col2:
            # Bar comparison chart
            bar_fig = create_bucket_comparison_chart(bucket_df, regime_name)
            if bar_fig:
                st.plotly_chart(bar_fig, use_container_width=True)
        
        # Timeseries overlay (equity vs regime)
        ts_fig = create_regime_timeseries_chart(regime_series, daily_pnl, regime_name, starting_equity)
        if ts_fig:
            with st.expander(f"📈 Equity vs {regime_name} Timeseries"):
                st.plotly_chart(ts_fig, use_container_width=True)
        
        # Per-strategy breakdown
        strat_df = analyze_by_strategy(sig_df, regime_series, regime_name)
        if not strat_df.empty:
            with st.expander(f"🔍 {regime_name} — Per-Strategy Breakdown"):
                # Heatmap
                heatmap_fig = create_strategy_heatmap(strat_df, metric='Avg R')
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                
                # Raw data table
                st.dataframe(
                    strat_df.style.format({
                        'Win Rate': '{:.1%}',
                        'Avg R': '{:.3f}',
                        'Avg PnL': '${:,.0f}',
                        'Total PnL': '${:,.0f}',
                    }),
                    use_container_width=True
                )
        
        st.divider()
    
    progress.progress(100, text="Complete!")
    
    # -------------------------------------------------------------------------
    # STEP 4: Summary Dashboard
    # -------------------------------------------------------------------------
    st.header("4️⃣ Summary: Which Regime Variables Matter?")
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        
        # Sort by absolute t-stat (strongest signal first)
        summary_df['abs_t'] = summary_df['t-stat'].abs()
        summary_df = summary_df.sort_values('abs_t', ascending=False).drop(columns=['abs_t'])
        
        st.dataframe(
            summary_df.style.format({
                't-stat': '{:.3f}',
                'p-value': '{:.4f}',
                'Low Avg R': '{:.3f}',
                'High Avg R': '{:.3f}',
            }),
            use_container_width=True
        )
        
        # Actionable recommendations
        significant_vars = [r for r in summary_rows if r.get('Significant?') == '✅']
        
        st.subheader("📋 Recommendations")
        
        if significant_vars:
            st.markdown("**Statistically significant regime variables found.** Consider these for Phase 2 (advisory dashboard) before Phase 3 (auto-sizing).")
            
            for sv in significant_vars:
                better = "Low" if sv.get('Low Avg R', 0) > sv.get('High Avg R', 0) else "High"
                worse = "High" if better == "Low" else "Low"
                spread = abs(sv.get('Low Avg R', 0) - sv.get('High Avg R', 0))
                
                st.markdown(f"""
                **{sv['Variable']}:** {better} regime outperforms by **{spread:.3f}R** per trade. 
                Consider reducing sizing when {sv['Variable']} is in the {worse} regime.
                (n={sv.get('n (Low)', 0)} low, {sv.get('n (High)', 0)} high)
                """)
        else:
            st.info("""
            No statistically significant regime variables found at p < 0.05. 
            This could mean: (a) your strategies are genuinely robust across regimes (good!), 
            (b) the sample size is too small to detect real differences, or 
            (c) you need to test different variable formulations.
            
            **Next steps:** Try extending the backtest period, or test interaction effects 
            (e.g., high dispersion + cold streak together).
            """)
        
        st.divider()
        st.caption("""
        **Methodology notes:**
        - All splits use the in-sample median (or tercile boundaries). For out-of-sample testing, 
          use a rolling median that updates daily — do not optimize the threshold.
        - Statistical test: Welch's unequal-variance t-test on R-multiples between regime buckets.
        - Sector dispersion = cross-sectional stdev of trailing N-day returns across 11 SPDR sector ETFs.
        - Signal density is an internal variable (not market state) but useful for diagnosing correlation risk.
        """)


if __name__ == "__main__":
    main()
