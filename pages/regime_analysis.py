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
                       regime_name: str, extreme_pct: int = 10) -> pd.DataFrame:
    """
    Merge a regime variable onto the trade log by entry date,
    then isolate the extreme tails and compare against the middle.
    
    Buckets:
    - Bottom N% of regime variable
    - Middle (everything else)
    - Top N% of regime variable
    
    Args:
        sig_df: Trade log with 'Date', 'PnL', 'Risk $', 'Strategy' columns
        regime_series: Daily series of the regime variable
        regime_name: Human-readable name for display
        extreme_pct: Percentile cutoff for tails (e.g., 10 = top/bottom 10%)
    
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
    
    # Compute percentile thresholds from the regime series itself (not the trade sample)
    # This avoids conditioning bias from only looking at days with trades
    lower_thresh = regime_series.quantile(extreme_pct / 100)
    upper_thresh = regime_series.quantile(1 - extreme_pct / 100)
    
    # Assign buckets
    df['bucket'] = 'Middle'
    df.loc[df['regime_val'] <= lower_thresh, 'bucket'] = f'Bottom {extreme_pct}%'
    df.loc[df['regime_val'] >= upper_thresh, 'bucket'] = f'Top {extreme_pct}%'
    
    bucket_order = [f'Bottom {extreme_pct}%', 'Middle', f'Top {extreme_pct}%']
    
    results = []
    for bucket in bucket_order:
        b_df = df[df['bucket'] == bucket]
        if len(b_df) < 3:
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
            'Regime Range': f"{val_min:.4f} – {val_max:.4f}",
            'Trades': len(b_df),
            'Win Rate': win_rate,
            'Avg R': avg_r,
            'Avg PnL': avg_pnl,
            'Total PnL': total_pnl,
            'Profit Factor': pf,
        })
    
    return pd.DataFrame(results)


def analyze_by_strategy(sig_df: pd.DataFrame, regime_series: pd.Series,
                        regime_name: str, extreme_pct: int = 10) -> pd.DataFrame:
    """
    Same as bucket_and_analyze but broken out by strategy.
    Compares Bottom N% vs Top N% for each strategy individually.
    """
    if sig_df.empty or regime_series.empty:
        return pd.DataFrame()
    
    df = sig_df.copy()
    df['regime_val'] = df['Date'].map(regime_series)
    df = df.dropna(subset=['regime_val'])
    
    if len(df) < 20:
        return pd.DataFrame()
    
    df['R_Multiple'] = np.where(df['Risk $'] > 0, df['PnL'] / df['Risk $'], 0)
    
    # Use regime series thresholds (not trade-conditioned)
    lower_thresh = regime_series.quantile(extreme_pct / 100)
    upper_thresh = regime_series.quantile(1 - extreme_pct / 100)
    
    bottom_label = f'Bottom {extreme_pct}%'
    top_label = f'Top {extreme_pct}%'
    
    df['bucket'] = 'Middle'
    df.loc[df['regime_val'] <= lower_thresh, 'bucket'] = bottom_label
    df.loc[df['regime_val'] >= upper_thresh, 'bucket'] = top_label
    
    results = []
    for strat in df['Strategy'].unique():
        s_df = df[df['Strategy'] == strat]
        
        for bucket in [bottom_label, 'Middle', top_label]:
            b_df = s_df[s_df['bucket'] == bucket]
            if len(b_df) < 3:
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


def compute_statistical_significance(sig_df: pd.DataFrame, regime_series: pd.Series,
                                      extreme_pct: int = 10) -> dict:
    """
    Welch's t-test comparing R-multiples in each extreme tail vs the middle.
    Returns results for both tails independently.
    """
    df = sig_df.copy()
    df['regime_val'] = df['Date'].map(regime_series)
    df = df.dropna(subset=['regime_val'])
    
    if len(df) < 20:
        return {'bottom': {}, 'top': {}, 'tails_combined': {}}
    
    df['R_Multiple'] = np.where(df['Risk $'] > 0, df['PnL'] / df['Risk $'], 0)
    
    lower_thresh = regime_series.quantile(extreme_pct / 100)
    upper_thresh = regime_series.quantile(1 - extreme_pct / 100)
    
    bottom = df[df['regime_val'] <= lower_thresh]['R_Multiple']
    middle = df[(df['regime_val'] > lower_thresh) & (df['regime_val'] < upper_thresh)]['R_Multiple']
    top = df[df['regime_val'] >= upper_thresh]['R_Multiple']
    
    results = {}
    
    # Bottom tail vs middle
    if len(bottom) >= 5 and len(middle) >= 10:
        t_stat, p_val = scipy_stats.ttest_ind(bottom, middle, equal_var=False)
        diff = bottom.mean() - middle.mean()
        results['bottom'] = {
            'label': f'Bottom {extreme_pct}%',
            't_stat': t_stat, 'p_value': p_val,
            'significant': p_val < 0.05,
            'tail_avg_r': bottom.mean(), 'middle_avg_r': middle.mean(),
            'diff': diff, 'n_tail': len(bottom), 'n_middle': len(middle),
            'interpretation': f"Bottom {extreme_pct}% {'outperforms' if diff > 0 else 'underperforms'} middle by {abs(diff):.3f}R (p={p_val:.4f})"
        }
    
    # Top tail vs middle
    if len(top) >= 5 and len(middle) >= 10:
        t_stat, p_val = scipy_stats.ttest_ind(top, middle, equal_var=False)
        diff = top.mean() - middle.mean()
        results['top'] = {
            'label': f'Top {extreme_pct}%',
            't_stat': t_stat, 'p_value': p_val,
            'significant': p_val < 0.05,
            'tail_avg_r': top.mean(), 'middle_avg_r': middle.mean(),
            'diff': diff, 'n_tail': len(top), 'n_middle': len(middle),
            'interpretation': f"Top {extreme_pct}% {'outperforms' if diff > 0 else 'underperforms'} middle by {abs(diff):.3f}R (p={p_val:.4f})"
        }
    
    # Combined: either tail is actionable?
    either_sig = any(r.get('significant', False) for r in results.values())
    results['any_significant'] = either_sig
    
    return results


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
    
    colors = {}
    for _, row in bucket_df.iterrows():
        b = row['Bucket']
        if 'Bottom' in b:
            colors[b] = '#0088FF'   # Blue for bottom tail
        elif 'Top' in b:
            colors[b] = '#CC0000'   # Red for top tail
        else:
            colors[b] = '#666666'   # Gray for middle
    
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
    
    # Ensure column order (tails on outside, middle in center)
    col_order = [c for c in pivot.columns if 'Bottom' in c] + \
                [c for c in pivot.columns if c == 'Middle'] + \
                [c for c in pivot.columns if 'Top' in c]
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]
    
    # Add a "Tail Spread" column (bottom tail R minus top tail R)
    bottom_cols = [c for c in pivot.columns if 'Bottom' in c]
    top_cols = [c for c in pivot.columns if 'Top' in c]
    if bottom_cols and top_cols:
        pivot['Tail Spread'] = pivot[bottom_cols[0]] - pivot[top_cols[0]]
        pivot = pivot.sort_values('Tail Spread', ascending=False)
    
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
# LAGGED REGIME ANALYSIS — "Can we see it coming?"
# =============================================================================

def lagged_bucket_and_analyze(sig_df: pd.DataFrame, regime_series: pd.Series,
                               regime_name: str, extreme_pct: int = 10,
                               lag_days: int = 1) -> pd.DataFrame:
    """
    Same as bucket_and_analyze, but uses the regime value from N days BEFORE
    the signal date. This tests whether yesterday's regime reading predicts
    today's trade quality — the only version that's actually tradeable.
    """
    if sig_df.empty or regime_series.empty:
        return pd.DataFrame()
    
    # Shift the regime series forward by lag_days (so looking up date T gives
    # you the regime value from date T - lag_days)
    lagged_series = regime_series.shift(lag_days)
    lagged_series = lagged_series.dropna()
    
    if lagged_series.empty:
        return pd.DataFrame()
    
    # Use the same bucketing logic but on the lagged series
    return bucket_and_analyze(sig_df, lagged_series, regime_name, extreme_pct=extreme_pct)


def lagged_statistical_significance(sig_df: pd.DataFrame, regime_series: pd.Series,
                                     extreme_pct: int = 10, lag_days: int = 1) -> dict:
    """Stat sig test using lagged regime values."""
    lagged_series = regime_series.shift(lag_days).dropna()
    if lagged_series.empty:
        return {'bottom': {}, 'top': {}, 'any_significant': False}
    return compute_statistical_significance(sig_df, lagged_series, extreme_pct=extreme_pct)


def simulate_regime_filter(sig_df: pd.DataFrame, regime_series: pd.Series,
                            extreme_pct: int = 10, lag_days: int = 1,
                            action: str = 'skip') -> dict:
    """
    What-if simulation: replay the trade log but skip or haircut trades
    where the PRIOR day's regime was in the extreme tail.
    
    Args:
        sig_df: Full trade log
        regime_series: Raw (unlagged) regime series
        extreme_pct: Tail cutoff
        lag_days: How many days before the signal to check
        action: 'skip' = remove trades entirely, 'haircut' = reduce PnL by 65%
    
    Returns:
        Dict with baseline vs filtered portfolio metrics
    """
    if sig_df.empty or regime_series.empty:
        return {}
    
    df = sig_df.copy()
    
    # Lag the regime series
    lagged = regime_series.shift(lag_days).dropna()
    
    # Map lagged regime value to each trade's signal date
    df['regime_prev'] = df['Date'].map(lagged)
    df = df.dropna(subset=['regime_prev'])
    
    if df.empty:
        return {}
    
    # Compute tail threshold from the full regime series (not lagged, to keep
    # the threshold consistent — lag only affects which day we look up)
    upper_thresh = regime_series.quantile(1 - extreme_pct / 100)
    
    # Flag trades where prior day was in the danger zone
    df['flagged'] = df['regime_prev'] >= upper_thresh
    n_flagged = df['flagged'].sum()
    n_total = len(df)
    
    if n_flagged == 0:
        return {'no_flags': True}
    
    # --- Baseline metrics (all trades) ---
    baseline_pnl = df['PnL'].sum()
    baseline_trades = n_total
    baseline_avg_r = (df['PnL'] / df['Risk $']).mean()
    baseline_winners = (df['PnL'] > 0).sum()
    baseline_wr = baseline_winners / n_total
    
    # --- Filtered metrics ---
    if action == 'skip':
        # Remove flagged trades entirely
        filtered = df[~df['flagged']]
        filtered_pnl = filtered['PnL'].sum()
        filtered_trades = len(filtered)
        filtered_avg_r = (filtered['PnL'] / filtered['Risk $']).mean() if len(filtered) > 0 else 0
        filtered_wr = (filtered['PnL'] > 0).mean() if len(filtered) > 0 else 0
        # PnL of the skipped trades (what we avoided)
        skipped_pnl = df[df['flagged']]['PnL'].sum()
        skipped_avg_r = (df[df['flagged']]['PnL'] / df[df['flagged']]['Risk $']).mean()
        action_label = "Skip flagged trades"
    else:
        # Haircut: reduce risk on flagged trades by 65% (keep 35%)
        filtered = df.copy()
        haircut_factor = 0.35
        filtered.loc[filtered['flagged'], 'PnL'] = filtered.loc[filtered['flagged'], 'PnL'] * haircut_factor
        filtered.loc[filtered['flagged'], 'Risk $'] = filtered.loc[filtered['flagged'], 'Risk $'] * haircut_factor
        filtered_pnl = filtered['PnL'].sum()
        filtered_trades = n_total  # same trade count
        filtered_avg_r = (filtered['PnL'] / filtered['Risk $']).mean()
        filtered_wr = (filtered['PnL'] > 0).mean()
        skipped_pnl = df[df['flagged']]['PnL'].sum() * (1 - haircut_factor)
        skipped_avg_r = (df[df['flagged']]['PnL'] / df[df['flagged']]['Risk $']).mean()
        action_label = f"Haircut flagged trades to {haircut_factor:.0%} size"
    
    return {
        'action_label': action_label,
        'n_total': n_total,
        'n_flagged': n_flagged,
        'pct_flagged': n_flagged / n_total,
        'baseline_pnl': baseline_pnl,
        'baseline_trades': baseline_trades,
        'baseline_avg_r': baseline_avg_r,
        'baseline_wr': baseline_wr,
        'filtered_pnl': filtered_pnl,
        'filtered_trades': filtered_trades,
        'filtered_avg_r': filtered_avg_r,
        'filtered_wr': filtered_wr,
        'pnl_delta': filtered_pnl - baseline_pnl,
        'pnl_delta_pct': (filtered_pnl - baseline_pnl) / abs(baseline_pnl) if baseline_pnl != 0 else 0,
        'skipped_pnl': skipped_pnl,
        'skipped_avg_r': skipped_avg_r,
        'upper_thresh': upper_thresh,
    }

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
        st.subheader("Tail Thresholds")
        extreme_pct = st.selectbox("Extreme Cutoff", [5, 10, 20], index=1,
                                    format_func=lambda x: f"Top/Bottom {x}%",
                                    help="Isolate the top and bottom N% of each regime variable")
        
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
        
        # Statistical significance test (tails vs middle)
        sig_test = compute_statistical_significance(sig_df, regime_series, extreme_pct=extreme_pct)
        
        # Bucket analysis (portfolio level)
        bucket_df = bucket_and_analyze(sig_df, regime_series, regime_name, extreme_pct=extreme_pct)
        
        if bucket_df.empty:
            st.warning(f"Insufficient data to analyze {regime_name}")
            continue
        
        # Display significance for each tail
        for tail_key in ['bottom', 'top']:
            tail_result = sig_test.get(tail_key, {})
            if not tail_result:
                continue
            label = tail_result.get('label', tail_key)
            if tail_result.get('significant', False):
                st.success(f"✅ **{label}:** {tail_result['interpretation']}")
            else:
                st.info(f"⚪ **{label}:** {tail_result['interpretation']}")
        
        # Add to summary
        for tail_key in ['bottom', 'top']:
            tail_result = sig_test.get(tail_key, {})
            if tail_result:
                summary_rows.append({
                    'Variable': regime_name,
                    'Tail': tail_result.get('label', tail_key),
                    't-stat': tail_result.get('t_stat', np.nan),
                    'p-value': tail_result.get('p_value', np.nan),
                    'Significant?': '✅' if tail_result.get('significant', False) else '⚪',
                    'Tail Avg R': tail_result.get('tail_avg_r', np.nan),
                    'Middle Avg R': tail_result.get('middle_avg_r', np.nan),
                    'Delta R': tail_result.get('diff', np.nan),
                    'n (Tail)': tail_result.get('n_tail', 0),
                    'n (Middle)': tail_result.get('n_middle', 0),
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
        strat_df = analyze_by_strategy(sig_df, regime_series, regime_name, extreme_pct=extreme_pct)
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
                'Tail Avg R': '{:.3f}',
                'Middle Avg R': '{:.3f}',
                'Delta R': '{:+.3f}',
            }),
            use_container_width=True
        )
        
        # Actionable recommendations
        significant_rows = [r for r in summary_rows if r.get('Significant?') == '✅']
        
        st.subheader("📋 Recommendations")
        
        if significant_rows:
            st.markdown(f"**Statistically significant tail effects found at the {extreme_pct}% level.** "
                       "Consider these for the advisory dashboard before auto-sizing.")
            
            for sv in significant_rows:
                delta = sv.get('Delta R', 0)
                direction = "outperforms" if delta > 0 else "underperforms"
                
                st.markdown(f"""
                **{sv['Variable']} — {sv['Tail']}:** {direction} middle by **{abs(delta):.3f}R** per trade 
                (p={sv.get('p-value', 0):.4f}, n={sv.get('n (Tail)', 0)} tail trades vs {sv.get('n (Middle)', 0)} middle)
                """)
        else:
            st.info(f"""
            No statistically significant effects found at the top/bottom {extreme_pct}% level. 
            This could mean: (a) your strategies are genuinely robust across regime extremes (good!), 
            (b) the extreme buckets have too few trades for statistical power — try widening to top/bottom 20%, or 
            (c) the tested variables don't capture the regime shifts that matter to your book.
            
            **Next steps:** Try a wider cutoff, extend the backtest period, or test interaction effects 
            (e.g., high dispersion + cold streak together).
            """)
        
        st.divider()
        st.caption("""
        **Methodology notes:**
        - All splits use in-sample percentile boundaries. For production, 
          use a rolling percentile that updates daily — do not optimize the threshold.
        - Statistical test: Welch's unequal-variance t-test on R-multiples between regime buckets.
        - Sector dispersion = cross-sectional stdev of trailing N-day returns across 11 SPDR sector ETFs.
        - Signal density is an internal variable (not market state) but useful for diagnosing correlation risk.
        """)
    
    # -------------------------------------------------------------------------
    # STEP 5: Lagged Analysis — "Can yesterday's reading predict today's trade?"
    # -------------------------------------------------------------------------
    st.header("5️⃣ Lagged Analysis: Can We See It Coming?")
    st.caption("The previous analysis uses same-day regime values. This section checks whether "
               "**yesterday's** reading predicts today's trade quality — the only version that's tradeable.")
    
    # Only run lagged analysis on variables that showed significance in same-day
    significant_vars = {}
    for regime_name, regime_series in regime_variables.items():
        # Skip signal density (it's same-day by definition, can't be lagged meaningfully)
        if 'Density' in regime_name:
            continue
        sig_test = compute_statistical_significance(sig_df, regime_series, extreme_pct=extreme_pct)
        if sig_test.get('any_significant', False):
            significant_vars[regime_name] = regime_series
    
    if not significant_vars:
        st.info("No same-day variables were significant, so lagged analysis is skipped. "
                "A variable needs to matter same-day before testing whether it's predictive with a lag.")
    else:
        st.markdown(f"Testing **{len(significant_vars)}** variables that were significant same-day: "
                   f"{', '.join(significant_vars.keys())}")
        
        lagged_summary = []
        
        for regime_name, regime_series in significant_vars.items():
            st.subheader(f"⏪ {regime_name} (Lagged 1 Day)")
            
            # Lagged stat sig
            lag_sig = lagged_statistical_significance(sig_df, regime_series, 
                                                       extreme_pct=extreme_pct, lag_days=1)
            
            # Lagged bucket analysis
            lag_buckets = lagged_bucket_and_analyze(sig_df, regime_series, regime_name,
                                                     extreme_pct=extreme_pct, lag_days=1)
            
            if lag_buckets.empty:
                st.warning(f"Insufficient data for lagged analysis of {regime_name}")
                continue
            
            # Display results side by side with same-day for comparison
            col_lag, col_sim = st.columns([1, 1])
            
            with col_lag:
                st.markdown("**Lagged Performance (prior day's reading → today's trade)**")
                
                any_lag_sig = False
                for tail_key in ['bottom', 'top']:
                    tail_result = lag_sig.get(tail_key, {})
                    if not tail_result:
                        continue
                    label = tail_result.get('label', tail_key)
                    if tail_result.get('significant', False):
                        st.success(f"✅ **{label}:** {tail_result['interpretation']}")
                        any_lag_sig = True
                    else:
                        st.info(f"⚪ **{label}:** {tail_result['interpretation']}")
                
                st.dataframe(
                    lag_buckets.style.format({
                        'Win Rate': '{:.1%}',
                        'Avg R': '{:.3f}',
                        'Avg PnL': '${:,.0f}',
                        'Total PnL': '${:,.0f}',
                        'Profit Factor': '{:.2f}',
                    }),
                    use_container_width=True
                )
                
                # Track for summary
                for tail_key in ['bottom', 'top']:
                    tr = lag_sig.get(tail_key, {})
                    if tr:
                        lagged_summary.append({
                            'Variable': regime_name,
                            'Tail': tr.get('label', ''),
                            'Same-Day Sig?': '✅',  # we only test vars that were same-day sig
                            'Lagged Sig?': '✅' if tr.get('significant', False) else '⚪',
                            'Lagged Delta R': tr.get('diff', np.nan),
                            'p-value (lagged)': tr.get('p_value', np.nan),
                            'n (Tail)': tr.get('n_tail', 0),
                        })
            
            with col_sim:
                st.markdown("**What-If Simulation: Skip vs Haircut flagged trades**")
                
                # Run both simulations (skip and haircut) using top tail only
                # (high dispersion / high correlation is the danger zone)
                sim_skip = simulate_regime_filter(sig_df, regime_series, 
                                                   extreme_pct=extreme_pct, lag_days=1,
                                                   action='skip')
                sim_haircut = simulate_regime_filter(sig_df, regime_series,
                                                      extreme_pct=extreme_pct, lag_days=1,
                                                      action='haircut')
                
                if sim_skip and not sim_skip.get('no_flags', False):
                    st.markdown(f"Trades flagged by prior-day top {extreme_pct}%: "
                               f"**{sim_skip['n_flagged']}** / {sim_skip['n_total']} "
                               f"({sim_skip['pct_flagged']:.1%})")
                    
                    st.markdown(f"Those flagged trades averaged **{sim_skip['skipped_avg_r']:.3f}R** "
                               f"and totaled **${sim_skip['skipped_pnl']:,.0f}** PnL")
                    
                    # Comparison table
                    sim_data = {
                        'Scenario': ['Baseline (all trades)', 
                                     f'Skip flagged ({sim_skip["n_flagged"]} trades)',
                                     f'Haircut to 35% size'],
                        'Trades': [sim_skip['baseline_trades'], 
                                   sim_skip['filtered_trades'],
                                   sim_haircut['filtered_trades']],
                        'Total PnL': [sim_skip['baseline_pnl'], 
                                      sim_skip['filtered_pnl'],
                                      sim_haircut['filtered_pnl']],
                        'Avg R': [sim_skip['baseline_avg_r'], 
                                  sim_skip['filtered_avg_r'],
                                  sim_haircut['filtered_avg_r']],
                        'Win Rate': [sim_skip['baseline_wr'], 
                                     sim_skip['filtered_wr'],
                                     sim_haircut['filtered_wr']],
                        'PnL Δ vs Baseline': [0, 
                                              sim_skip['pnl_delta'],
                                              sim_haircut['pnl_delta']],
                    }
                    sim_df = pd.DataFrame(sim_data)
                    
                    st.dataframe(
                        sim_df.style.format({
                            'Total PnL': '${:,.0f}',
                            'Avg R': '{:.3f}',
                            'Win Rate': '{:.1%}',
                            'PnL Δ vs Baseline': '${:+,.0f}',
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No trades were flagged by the prior-day filter.")
            
            st.divider()
        
        # Lagged summary table
        if lagged_summary:
            st.subheader("📋 Lagged Analysis Summary")
            st.caption("Does the same-day signal survive a 1-day lag? If yes → tradeable. If no → contemporaneous only (diagnostic, not actionable).")
            
            lag_summary_df = pd.DataFrame(lagged_summary)
            lag_summary_df = lag_summary_df.sort_values('p-value (lagged)')
            
            st.dataframe(
                lag_summary_df.style.format({
                    'Lagged Delta R': '{:+.3f}',
                    'p-value (lagged)': '{:.4f}',
                }),
                use_container_width=True
            )


if __name__ == "__main__":
    main()
