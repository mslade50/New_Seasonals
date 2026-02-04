"""
Equity Curve Analysis Functions
Add these to strat_backtester.py to analyze adaptive sizing opportunities
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def analyze_equity_curve_effects(daily_pnl_series, starting_equity, ma_window=20, bb_std=2.0):
    """
    Analyzes whether equity curve state predicts forward returns.
    
    Returns a dict with:
    - autocorr: Autocorrelation of daily returns (%) at various lags
    - ma_analysis: Forward returns when equity > MA vs < MA
    - bb_analysis: Forward returns by Bollinger band position
    - streak_analysis: Forward returns after winning/losing streaks
    
    Args:
        daily_pnl_series: pd.Series of daily P&L (index=date, values=$ P&L)
        starting_equity: Starting account value
        ma_window: Lookback for moving average (default 20)
        bb_std: Standard deviations for Bollinger bands (default 2.0)
    """
    if daily_pnl_series.empty or len(daily_pnl_series) < ma_window + 10:
        return None
    
    # Build equity curve (in $)
    equity = starting_equity + daily_pnl_series.cumsum()
    
    # Convert to % returns for analysis (this normalizes across time periods)
    daily_returns_pct = (daily_pnl_series / equity.shift(1)) * 100  # As percentage
    
    # Helper function for safe metric calculation
    def safe_metrics(series, mask):
        """Safely calculate metrics, handling NaN and empty series."""
        filtered = series[mask].dropna()
        count = len(filtered)
        if count == 0:
            return {'count': 0, 'avg_fwd_ret_pct': 0, 'win_rate': 0, 'total_ret_pct': 0}
        return {
            'count': int(count),
            'avg_fwd_ret_pct': float(filtered.mean()) if pd.notna(filtered.mean()) else 0,
            'win_rate': float((filtered > 0).mean()) if count > 0 else 0,
            'total_ret_pct': float(filtered.sum()) if pd.notna(filtered.sum()) else 0
        }
    
    results = {}
    
    # =========================================================================
    # 1. AUTOCORRELATION ANALYSIS
    # Question: Does yesterday's return predict today's?
    # =========================================================================
    autocorr = {}
    for lag in [1, 2, 3, 5]:
        if len(daily_returns_pct.dropna()) > lag + 10:
            ac_val = daily_returns_pct.autocorr(lag=lag)
            if pd.notna(ac_val):
                autocorr[f'lag_{lag}'] = ac_val
    results['autocorr'] = autocorr
    
    # =========================================================================
    # 2. EQUITY vs MOVING AVERAGE
    # Question: Do returns differ when equity > 20d MA vs below?
    # =========================================================================
    equity_ma = equity.rolling(ma_window).mean()
    
    above_ma = (equity > equity_ma).fillna(False).shift(1).fillna(False)
    below_ma = ~above_ma
    
    # Forward 1-day returns (next day's return %)
    fwd_1d = daily_returns_pct.shift(-1)
    
    valid_mask = equity_ma.notna().shift(1).fillna(False)
    
    ma_analysis = {
        'above_ma': safe_metrics(fwd_1d, above_ma & valid_mask),
        'below_ma': safe_metrics(fwd_1d, below_ma & valid_mask)
    }
    results['ma_analysis'] = ma_analysis
    
    # =========================================================================
    # 3. BOLLINGER BAND POSITION
    # =========================================================================
    equity_std = equity.rolling(ma_window).std()
    upper_bb = equity_ma + (bb_std * equity_std)
    lower_bb = equity_ma - (bb_std * equity_std)
    
    def get_bb_zone(row_idx):
        if pd.isna(equity_ma.iloc[row_idx]) or pd.isna(equity_std.iloc[row_idx]):
            return 'insufficient_data'
        eq = equity.iloc[row_idx]
        ma = equity_ma.iloc[row_idx]
        std = equity_std.iloc[row_idx]
        if std == 0:
            return 'middle'
        z_score = (eq - ma) / std
        if z_score > bb_std:
            return 'above_upper'
        elif z_score > 1:
            return 'upper_half'
        elif z_score > -1:
            return 'middle'
        elif z_score > -bb_std:
            return 'lower_half'
        else:
            return 'below_lower'
    
    bb_zones = pd.Series([get_bb_zone(i) for i in range(len(equity))], index=equity.index)
    bb_zones_shifted = bb_zones.shift(1)
    
    bb_analysis = {}
    for zone in ['above_upper', 'upper_half', 'middle', 'lower_half', 'below_lower']:
        mask = bb_zones_shifted == zone
        if mask.sum() > 5:
            bb_analysis[zone] = safe_metrics(fwd_1d, mask)
    results['bb_analysis'] = bb_analysis
    
    # =========================================================================
    # 4. RECENT STREAK ANALYSIS
    # =========================================================================
    is_winner = (daily_returns_pct > 0).astype(int)
    
    streak_groups = (is_winner != is_winner.shift()).cumsum()
    streak_length = is_winner.groupby(streak_groups).cumcount() + 1
    streak_length = streak_length * is_winner.replace(0, -1)
    
    streak_shifted = streak_length.shift(1)
    
    def get_streak_bucket(s):
        if pd.isna(s):
            return None
        if s >= 3:
            return 'win_3plus'
        elif s >= 1:
            return 'win_1_2'
        elif s <= -3:
            return 'loss_3plus'
        else:
            return 'loss_1_2'
    
    streak_buckets = streak_shifted.apply(get_streak_bucket)
    
    streak_analysis = {}
    for bucket in ['win_3plus', 'win_1_2', 'loss_1_2', 'loss_3plus']:
        mask = streak_buckets == bucket
        if mask.sum() > 5:
            streak_analysis[bucket] = safe_metrics(fwd_1d, mask)
    results['streak_analysis'] = streak_analysis
    
    # =========================================================================
    # 5. RECENT DRAWDOWN ANALYSIS
    # =========================================================================
    running_max = equity.expanding().max()
    drawdown_pct = (equity - running_max) / running_max
    dd_shifted = drawdown_pct.shift(1)
    
    def get_dd_bucket(dd):
        if pd.isna(dd):
            return None
        if dd >= 0:
            return 'at_high'
        elif dd > -0.02:
            return 'dd_0_2pct'
        elif dd > -0.05:
            return 'dd_2_5pct'
        else:
            return 'dd_5plus_pct'
    
    dd_buckets = dd_shifted.apply(get_dd_bucket)
    
    dd_analysis = {}
    for bucket in ['at_high', 'dd_0_2pct', 'dd_2_5pct', 'dd_5plus_pct']:
        mask = dd_buckets == bucket
        if mask.sum() > 5:
            dd_analysis[bucket] = safe_metrics(fwd_1d, mask)
    results['dd_analysis'] = dd_analysis
    
    # =========================================================================
    # 6. YESTERDAY'S RETURN MAGNITUDE
    # =========================================================================
    ret_shifted = daily_returns_pct.shift(1)
    ret_pctile = ret_shifted.rank(pct=True)
    
    def get_ret_bucket(pct):
        if pd.isna(pct):
            return None
        if pct >= 0.9:
            return 'top_10pct'
        elif pct >= 0.75:
            return 'top_25pct'
        elif pct <= 0.1:
            return 'bottom_10pct'
        elif pct <= 0.25:
            return 'bottom_25pct'
        else:
            return 'middle_50pct'
    
    ret_buckets = ret_pctile.apply(get_ret_bucket)
    
    yesterday_analysis = {}
    for bucket in ['top_10pct', 'top_25pct', 'middle_50pct', 'bottom_25pct', 'bottom_10pct']:
        mask = ret_buckets == bucket
        if mask.sum() > 5:
            yesterday_analysis[bucket] = safe_metrics(fwd_1d, mask)
    results['yesterday_analysis'] = yesterday_analysis
    
    return results

def create_equity_curve_analysis_figure(analysis_results, starting_equity):
    """
    Creates a Plotly figure visualizing the equity curve analysis.
    
    Returns a plotly figure with subplots showing:
    - Autocorrelation bars
    - MA analysis comparison
    - BB zone performance
    - Streak analysis
    """
    if analysis_results is None:
        return None
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Autocorrelation by Lag',
            'Equity vs 20d MA',
            'Bollinger Band Zone',
            'After Winning/Losing Streak',
            'Drawdown Depth',
            "Yesterday's P&L Magnitude"
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.15
    )
    
    # Color scheme
    pos_color = '#00CC00'
    neg_color = '#CC0000'
    neutral_color = '#0066CC'
    
    # =========================================================================
    # 1. Autocorrelation (Row 1, Col 1)
    # =========================================================================
    autocorr = analysis_results.get('autocorr', {})
    if autocorr:
        lags = [int(k.split('_')[1]) for k in autocorr.keys()]
        values = list(autocorr.values())
        colors = [pos_color if v > 0 else neg_color for v in values]
        
        fig.add_trace(
            go.Bar(x=[f'Lag {l}' for l in lags], y=values, marker_color=colors, 
                   text=[f'{v:.3f}' for v in values], textposition='outside'),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
    
    # =========================================================================
    # 2. MA Analysis (Row 1, Col 2)
    # =========================================================================
    ma_data = analysis_results.get('ma_analysis', {})
    if ma_data:
        categories = ['Above MA', 'Below MA']
        avg_pnls = [
            ma_data.get('above_ma', {}).get('avg_fwd_pnl', 0),
            ma_data.get('below_ma', {}).get('avg_fwd_pnl', 0)
        ]
        counts = [
            ma_data.get('above_ma', {}).get('count', 0),
            ma_data.get('below_ma', {}).get('count', 0)
        ]
        colors = [pos_color if v > 0 else neg_color for v in avg_pnls]
        
        fig.add_trace(
            go.Bar(x=categories, y=avg_pnls, marker_color=colors,
                   text=[f'${v:,.0f}<br>n={int(c)}' for v, c in zip(avg_pnls, counts)],
                   textposition='outside'),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2)
    
    # =========================================================================
    # 3. BB Zone Analysis (Row 1, Col 3)
    # =========================================================================
    bb_data = analysis_results.get('bb_analysis', {})
    if bb_data:
        zone_order = ['above_upper', 'upper_half', 'middle', 'lower_half', 'below_lower']
        zone_labels = ['Above Upper', 'Upper Half', 'Middle', 'Lower Half', 'Below Lower']
        avg_pnls = [bb_data.get(z, {}).get('avg_fwd_pnl', 0) for z in zone_order]
        counts = [bb_data.get(z, {}).get('count', 0) for z in zone_order]
        colors = [pos_color if v > 0 else neg_color for v in avg_pnls]
        
        fig.add_trace(
            go.Bar(x=zone_labels, y=avg_pnls, marker_color=colors,
                   text=[f'${v:,.0f}<br>n={int(c)}' for v, c in zip(avg_pnls, counts)],
                   textposition='outside'),
            row=1, col=3
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=3)
    
    # =========================================================================
    # 4. Streak Analysis (Row 2, Col 1)
    # =========================================================================
    streak_data = analysis_results.get('streak_analysis', {})
    if streak_data:
        streak_order = ['loss_3plus', 'loss_1_2', 'win_1_2', 'win_3plus']
        streak_labels = ['Loss 3+', 'Loss 1-2', 'Win 1-2', 'Win 3+']
        avg_pnls = [streak_data.get(s, {}).get('avg_fwd_pnl', 0) for s in streak_order]
        counts = [streak_data.get(s, {}).get('count', 0) for s in streak_order]
        colors = [pos_color if v > 0 else neg_color for v in avg_pnls]
        
        fig.add_trace(
            go.Bar(x=streak_labels, y=avg_pnls, marker_color=colors,
                   text=[f'${v:,.0f}<br>n={int(c)}' for v, c in zip(avg_pnls, counts)],
                   textposition='outside'),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    
    # =========================================================================
    # 5. Drawdown Analysis (Row 2, Col 2)
    # =========================================================================
    dd_data = analysis_results.get('dd_analysis', {})
    if dd_data:
        dd_order = ['at_high', 'dd_0_2pct', 'dd_2_5pct', 'dd_5plus_pct']
        dd_labels = ['At High', '0-2% DD', '2-5% DD', '5%+ DD']
        avg_pnls = [dd_data.get(d, {}).get('avg_fwd_pnl', 0) for d in dd_order]
        counts = [dd_data.get(d, {}).get('count', 0) for d in dd_order]
        colors = [pos_color if v > 0 else neg_color for v in avg_pnls]
        
        fig.add_trace(
            go.Bar(x=dd_labels, y=avg_pnls, marker_color=colors,
                   text=[f'${v:,.0f}<br>n={int(c)}' for v, c in zip(avg_pnls, counts)],
                   textposition='outside'),
            row=2, col=2
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=2)
    
    # =========================================================================
    # 6. Yesterday's P&L Analysis (Row 2, Col 3)
    # =========================================================================
    yest_data = analysis_results.get('yesterday_analysis', {})
    if yest_data:
        yest_order = ['bottom_10pct', 'bottom_25pct', 'middle_50pct', 'top_25pct', 'top_10pct']
        yest_labels = ['Worst 10%', 'Worst 25%', 'Middle 50%', 'Best 25%', 'Best 10%']
        avg_pnls = [yest_data.get(y, {}).get('avg_fwd_pnl', 0) for y in yest_order]
        counts = [yest_data.get(y, {}).get('count', 0) for y in yest_order]
        colors = [pos_color if v > 0 else neg_color for v in avg_pnls]
        
        fig.add_trace(
            go.Bar(x=yest_labels, y=avg_pnls, marker_color=colors,
                   text=[f'${v:,.0f}<br>n={int(c)}' for v, c in zip(avg_pnls, counts)],
                   textposition='outside'),
            row=2, col=3
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=3)
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Equity Curve Regime Analysis: Does Recent Performance Predict Tomorrow?",
        title_x=0.5,
        margin=dict(t=80, b=40)
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_yaxes(title_text="Avg Next-Day P&L ($)", row=1, col=2)
    fig.update_yaxes(title_text="Avg Next-Day P&L ($)", row=1, col=3)
    fig.update_yaxes(title_text="Avg Next-Day P&L ($)", row=2, col=1)
    fig.update_yaxes(title_text="Avg Next-Day P&L ($)", row=2, col=2)
    fig.update_yaxes(title_text="Avg Next-Day P&L ($)", row=2, col=3)
    
    return fig


def generate_sizing_recommendations(analysis_results):
    """
    Based on analysis results, generate plain-English recommendations for adaptive sizing.
    
    Returns a list of (recommendation_text, confidence) tuples.
    """
    recommendations = []
    
    if analysis_results is None:
        return [("Insufficient data for analysis", "N/A")]
    
    # 1. Check autocorrelation
    autocorr = analysis_results.get('autocorr', {})
    lag1 = autocorr.get('lag_1', 0)
    if lag1 > 0.15:
        recommendations.append((
            f"TREND-FOLLOW: Positive autocorrelation ({lag1:.2f}) suggests sizing UP after winning days.",
            "Medium" if lag1 > 0.2 else "Low"
        ))
    elif lag1 < -0.15:
        recommendations.append((
            f"MEAN-REVERT: Negative autocorrelation ({lag1:.2f}) suggests sizing DOWN after winning days.",
            "Medium" if lag1 < -0.2 else "Low"
        ))
    
    # 2. Check MA analysis
    ma_data = analysis_results.get('ma_analysis', {})
    above_pnl = ma_data.get('above_ma', {}).get('avg_fwd_pnl', 0)
    below_pnl = ma_data.get('below_ma', {}).get('avg_fwd_pnl', 0)
    above_n = ma_data.get('above_ma', {}).get('count', 0)
    below_n = ma_data.get('below_ma', {}).get('count', 0)
    
    if above_n > 30 and below_n > 30:
        diff = above_pnl - below_pnl
        if diff > 100:  # $100 difference is meaningful
            recommendations.append((
                f"EQUITY FILTER: Returns are ${diff:,.0f}/day better when equity > 20d MA. Consider reducing size when below.",
                "Medium" if diff > 200 else "Low"
            ))
        elif diff < -100:
            recommendations.append((
                f"CONTRARIAN: Returns are ${-diff:,.0f}/day better when equity < 20d MA. Consider mean-reversion sizing.",
                "Medium" if diff < -200 else "Low"
            ))
    
    # 3. Check drawdown analysis
    dd_data = analysis_results.get('dd_analysis', {})
    at_high = dd_data.get('at_high', {}).get('avg_fwd_pnl', 0)
    in_dd = dd_data.get('dd_5plus_pct', {}).get('avg_fwd_pnl', 0)
    
    if dd_data.get('at_high', {}).get('count', 0) > 20 and dd_data.get('dd_5plus_pct', {}).get('count', 0) > 20:
        if in_dd > at_high + 100:
            recommendations.append((
                f"DRAWDOWN OPPORTUNITY: Returns are better (${in_dd:,.0f} vs ${at_high:,.0f}) during 5%+ drawdowns. Don't cut size in drawdowns.",
                "Medium"
            ))
        elif at_high > in_dd + 100:
            recommendations.append((
                f"PROTECT GAINS: Returns deteriorate in drawdowns (${in_dd:,.0f} vs ${at_high:,.0f}). Consider reducing size during drawdowns.",
                "Medium"
            ))
    
    # 4. Check yesterday analysis
    yest_data = analysis_results.get('yesterday_analysis', {})
    best_10 = yest_data.get('top_10pct', {}).get('avg_fwd_pnl', 0)
    worst_10 = yest_data.get('bottom_10pct', {}).get('avg_fwd_pnl', 0)
    
    if yest_data.get('top_10pct', {}).get('count', 0) > 20 and yest_data.get('bottom_10pct', {}).get('count', 0) > 20:
        if worst_10 > best_10 + 100:
            recommendations.append((
                f"POST-BAD DAY OPPORTUNITY: After worst 10% days, avg next-day is ${worst_10:,.0f}. Consider sizing UP after bad days.",
                "Medium"
            ))
        elif best_10 > worst_10 + 100:
            recommendations.append((
                f"MOMENTUM: After best 10% days, avg next-day is ${best_10:,.0f}. Momentum persists - don't cut after wins.",
                "Medium"
            ))
    
    if not recommendations:
        recommendations.append((
            "NO CLEAR EDGE: Analysis doesn't show strong predictive relationships. Standard sizing recommended.",
            "N/A"
        ))
    
    return recommendations
