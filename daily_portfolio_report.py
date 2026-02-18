"""
daily_portfolio_report.py

Daily Portfolio Health Report - Runs at 5 PM ET

Generates:
1. Equity curve with Bollinger Bands + 20 SMA (12 months)
2. Daily P&L bar chart (green/red)
3. Open positions table with current MTM P&L
4. Automated sizing recommendations
5. Today's entered & exited positions

Author: McKinley
Last Modified: 2026-02-04
"""

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import time
import sys
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gspread

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from strategy_config import STRATEGY_BOOK, ACCOUNT_VALUE
except ImportError:
    print("❌ Could not import strategy_config.py")
    STRATEGY_BOOK = []
    ACCOUNT_VALUE = 750000

# Import backtesting functions from strat_backtester
from pages.strat_backtester import (
    download_historical_data,
    load_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
    process_signals_fast,
    get_daily_mtm_series
)

# -----------------------------------------------------------------------------
# 1. AUTHENTICATION (Same as daily_scan.py)
# -----------------------------------------------------------------------------

def get_google_client():
    """Authenticate with Google Sheets."""
    try:
        if "GCP_JSON" in os.environ:
            creds_dict = json.loads(os.environ["GCP_JSON"])
            return gspread.service_account_from_dict(creds_dict)
        elif os.path.exists("credentials.json"):
            return gspread.service_account(filename='credentials.json')
        else:
            print("❌ No Google credentials found")
            return None
    except Exception as e:
        print(f"❌ Auth Error: {e}")
        return None


# -----------------------------------------------------------------------------
# 2. PORTFOLIO SIMULATION (12-Month Lite Backtest)
# -----------------------------------------------------------------------------

def run_12month_backtest(starting_equity=None):
    """
    Run a lightweight 12-month backtest of the full strategy book.
    
    CRITICAL: Downloads data since 2000 for accurate percentile calculations,
    but only backtests the last 12 months for the report.
    
    Returns: (signals_df, equity_series, daily_pnl_series, master_dict)
    """
    if starting_equity is None:
        starting_equity = ACCOUNT_VALUE
    
    print("📊 Running 12-month portfolio backtest...")
    
    # FIXED: Download from 2000 for percentile accuracy
    data_start_date = datetime.date(2000, 1, 1)
    
    # But only backtest last 12 months
    end_date = datetime.date.today()
    backtest_start_date = end_date - datetime.timedelta(days=365)
    
    print(f"   Data range: {data_start_date} to {end_date}")
    print(f"   Backtest range: {backtest_start_date} to {end_date}")
    
    # 1. Load seasonal data
    sznl_map = load_seasonal_map()
    
    # 2. Gather all tickers from strategy book
    all_tickers = set()
    for strat in STRATEGY_BOOK:
        all_tickers.update(strat['universe_tickers'])
    
    # Add market/VIX tickers
    all_tickers.add('SPY')
    all_tickers.add('^VIX')
    
    print(f"   Downloading {len(all_tickers)} tickers from 2000...")
    master_dict = download_historical_data(list(all_tickers), start_date=data_start_date.strftime('%Y-%m-%d'))
    
    if not master_dict:
        print("❌ Failed to download data")
        return None, None, None, None
    
    # 3. Prepare market/VIX series
    spy_df = master_dict.get('SPY')
    vix_df = master_dict.get('^VIX')
    vix_series = None
    
    if vix_df is not None and not vix_df.empty:
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        vix_df.columns = [c.capitalize() for c in vix_df.columns]
        vix_series = vix_df['Close']
    
    # 4. Precompute indicators (uses all data since 2000 for percentiles)
    print("   Computing indicators (percentiles use full history)...")
    processed_dict = precompute_all_indicators(master_dict, STRATEGY_BOOK, sznl_map, vix_series)
    
    # 5. Generate candidates (only for last 12 months)
    print(f"   Finding signals since {backtest_start_date}...")
    candidates, signal_data = generate_candidates_fast(processed_dict, STRATEGY_BOOK, sznl_map, backtest_start_date)
    
    if not candidates:
        print("⚠️ No signals found in 12-month period")
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float), master_dict
    
    # 6. Process signals with MTM sizing
    print("   Processing trades...")
    sig_df = process_signals_fast(candidates, signal_data, processed_dict, STRATEGY_BOOK, starting_equity)
    
    if sig_df.empty:
        print("⚠️ No valid trades executed")
        return sig_df, pd.Series(dtype=float), pd.Series(dtype=float), master_dict
    
    # 7. Calculate equity curve (only for backtest period)
    print("   Calculating equity curve...")
    daily_pnl = get_daily_mtm_series(sig_df, master_dict, start_date=backtest_start_date)
    equity_series = starting_equity + daily_pnl.cumsum()
    
    print(f"✅ Backtest complete: {len(sig_df)} trades, {len(equity_series)} days")
    
    return sig_df, equity_series, daily_pnl, master_dict


# -----------------------------------------------------------------------------
# 3. CHART GENERATION
# -----------------------------------------------------------------------------

def create_portfolio_chart(equity_series, daily_pnl_series, starting_equity):
    """
    Creates a dual-panel chart:
    - Top: Equity curve with Bollinger Bands + 20 SMA
    - Bottom: Daily P&L bars (green/red)
    
    Returns: plotly figure object
    """
    if equity_series.empty:
        print("⚠️ Cannot create chart - empty equity series")
        return None
    
    # Calculate Bollinger Bands
    sma_20 = equity_series.rolling(20).mean()
    std_20 = equity_series.rolling(20).std()
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Portfolio Equity (12 Months)', 'Daily P&L'),
        vertical_spacing=0.1
    )
    
    # --- Panel 1: Equity Curve ---
    # Bollinger Bands (fill area)
    fig.add_trace(go.Scatter(
        x=upper_bb.index, y=upper_bb,
        name='Upper BB',
        line=dict(color='rgba(150,150,150,0.3)', width=1, dash='dash'),
        showlegend=True
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=lower_bb.index, y=lower_bb,
        name='Lower BB',
        line=dict(color='rgba(150,150,150,0.3)', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(200,200,200,0.2)',
        showlegend=True
    ), row=1, col=1)
    
    # 20 SMA
    fig.add_trace(go.Scatter(
        x=sma_20.index, y=sma_20,
        name='20-Day SMA',
        line=dict(color='#0066CC', width=2)
    ), row=1, col=1)
    
    # Equity line
    fig.add_trace(go.Scatter(
        x=equity_series.index, y=equity_series,
        name='Portfolio Equity',
        line=dict(color='#00FF00', width=2.5),
        mode='lines'
    ), row=1, col=1)
    
    # Starting equity reference line
    fig.add_hline(
        y=starting_equity,
        line_dash="dot",
        line_color="white",
        opacity=0.5,
        annotation_text=f"Start: ${starting_equity:,.0f}",
        annotation_position="right",
        row=1, col=1
    )
    
    # --- Panel 2: Daily P&L Bars ---
    colors = ['#00CC00' if p >= 0 else '#CC0000' for p in daily_pnl_series]
    fig.add_trace(go.Bar(
        x=daily_pnl_series.index,
        y=daily_pnl_series,
        marker_color=colors,
        showlegend=False,
        hovertemplate='%{x|%Y-%m-%d}<br>P&L: $%{y:,.0f}<extra></extra>'
    ), row=2, col=1)
    
    # Zero line for P&L
    fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.3, row=2, col=1)
    
    # Layout
    fig.update_layout(
        height=800,
        title_text="12-Month Portfolio Health Dashboard",
        title_font_size=20,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(color='white')
    )
    
    # Y-axis formatting
    fig.update_yaxes(title_text="Equity ($)", tickformat="$,.0f", row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Daily P&L ($)", tickformat="$,.0f", row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
    
    return fig


def save_chart_as_png(fig, filepath='/tmp/portfolio_health.png'):
    """
    Saves plotly figure as PNG file.
    Requires kaleido: pip install kaleido
    """
    try:
        fig.write_image(filepath, width=1400, height=800, scale=2)
        print(f"✅ Chart saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Failed to save chart: {e}")
        print("   Make sure 'kaleido' is installed: pip install kaleido")
        return None


# -----------------------------------------------------------------------------
# 4. OPEN POSITIONS QUERY (FIXED)
# -----------------------------------------------------------------------------

def get_open_positions_from_backtest(sig_df, master_dict):
    """
    Get open positions from backtest signals - IDENTICAL to strat_backtester.py logic.
    Includes ALL columns from the UI table.
    """
    if sig_df.empty:
        return pd.DataFrame()
    
    today = pd.Timestamp(datetime.date.today())
    
    # Filter for open positions (same as strat_backtester)
    open_mask = sig_df['Time Stop'] >= today
    open_df = sig_df[open_mask].copy()
    
    if open_df.empty:
        return pd.DataFrame()
    
    print(f"   Found {len(open_df)} open positions from backtest")
    
    # Calculate current prices and P&L (exact same logic as strat_backtester)
    current_prices, open_pnls, current_values = [], [], []
    
    for row in open_df.itertuples():
        t_clean = row.Ticker.replace('.', '-')
        t_df = master_dict.get(t_clean)
        
        if t_df is not None and not t_df.empty:
            if isinstance(t_df.columns, pd.MultiIndex):
                t_df.columns = t_df.columns.get_level_values(0)
            t_df.columns = [c.capitalize() for c in t_df.columns]
            last_close = t_df['Close'].iloc[-1]
        else:
            # Fallback: download just this ticker
            temp = yf.download(row.Ticker, period='1d', progress=False)
            if not temp.empty:
                if isinstance(temp.columns, pd.MultiIndex):
                    temp.columns = temp.columns.get_level_values(0)
                temp.columns = [c.capitalize() for c in temp.columns]
                last_close = temp['Close'].iloc[-1]
            else:
                last_close = row.Price
        
        # Calculate P&L (same as strat_backtester)
        if row.Action == 'BUY':
            pnl = (last_close - row.Price) * row.Shares
        else:
            pnl = (row.Price - last_close) * row.Shares
        
        current_prices.append(last_close)
        open_pnls.append(pnl)
        current_values.append(last_close * row.Shares)
    
    # Build result dataframe with ALL columns from strat_backtester
    result = pd.DataFrame({
        'Date': open_df['Date'].values,
        'Entry Date': open_df['Entry Date'].values,
        'Exit Date': open_df['Exit Date'].values,
        'Exit Type': open_df['Exit Type'].values,
        'Time Stop': open_df['Time Stop'].values,
        'Strategy': open_df['Strategy'].values,
        'Ticker': open_df['Ticker'].values,
        'Action': open_df['Action'].values,
        'Entry Criteria': open_df['Entry Criteria'].values,
        'Price': open_df['Price'].values,
        'Shares': open_df['Shares'].values,
        'PnL': open_df['PnL'].values,
        'ATR': open_df['ATR'].values,
        'T+1 Open': open_df['T+1 Open'].values,
        'Signal Close': open_df['Signal Close'].values,
        'Range %': open_df['Range %'].values,
        'Equity at Signal': open_df['Equity at Signal'].values,
        'Risk $': open_df['Risk $'].values,
        'Risk bps': open_df['Risk bps'].values,
        'Current Price': current_prices,
        'Open PnL': open_pnls,
        'Mkt Value': current_values
    })
    
    return result


# -----------------------------------------------------------------------------
# 4b. TODAY'S ENTERED & EXITED POSITIONS
# -----------------------------------------------------------------------------

def get_todays_activity(sig_df, master_dict):
    """
    Get positions that were ENTERED today and positions that were EXITED today.
    Returns: (entered_today_df, exited_today_df) with same columns as open positions.
    """
    if sig_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    today = pd.Timestamp(datetime.date.today())

    # --- Entered Today: Entry Date == today ---
    entered_mask = sig_df['Entry Date'].apply(
        lambda x: pd.Timestamp(x).normalize() == today
    )
    entered_df = sig_df[entered_mask].copy()

    # --- Exited Today: Exit Date == today AND position is closed (Time Stop <= today) ---
    exited_mask = (
        sig_df['Exit Date'].apply(lambda x: pd.Timestamp(x).normalize() == today)
        & sig_df['Time Stop'].apply(lambda x: pd.Timestamp(x).normalize() <= today)
    )
    exited_df = sig_df[exited_mask].copy()

    print(f"   Today's activity: {len(entered_df)} entries, {len(exited_df)} exits")

    # Build output frames with same schema as open positions
    results = []
    for label, subset in [('entered', entered_df), ('exited', exited_df)]:
        if subset.empty:
            results.append(pd.DataFrame())
            continue

        current_prices, open_pnls, current_values = [], [], []
        for row in subset.itertuples():
            t_clean = row.Ticker.replace('.', '-')
            t_df = master_dict.get(t_clean)
            if t_df is not None and not t_df.empty:
                if isinstance(t_df.columns, pd.MultiIndex):
                    t_df.columns = t_df.columns.get_level_values(0)
                t_df.columns = [c.capitalize() for c in t_df.columns]
                last_close = t_df['Close'].iloc[-1]
            else:
                temp = yf.download(row.Ticker, period='1d', progress=False)
                if not temp.empty:
                    if isinstance(temp.columns, pd.MultiIndex):
                        temp.columns = temp.columns.get_level_values(0)
                    temp.columns = [c.capitalize() for c in temp.columns]
                    last_close = temp['Close'].iloc[-1]
                else:
                    last_close = row.Price

            if row.Action == 'BUY':
                pnl = (last_close - row.Price) * row.Shares
            else:
                pnl = (row.Price - last_close) * row.Shares

            current_prices.append(last_close)
            open_pnls.append(pnl)
            current_values.append(last_close * row.Shares)

        result = pd.DataFrame({
            'Date': subset['Date'].values,
            'Entry Date': subset['Entry Date'].values,
            'Exit Date': subset['Exit Date'].values,
            'Exit Type': subset['Exit Type'].values,
            'Time Stop': subset['Time Stop'].values,
            'Strategy': subset['Strategy'].values,
            'Ticker': subset['Ticker'].values,
            'Action': subset['Action'].values,
            'Entry Criteria': subset['Entry Criteria'].values,
            'Price': subset['Price'].values,
            'Shares': subset['Shares'].values,
            'PnL': subset['PnL'].values,
            'ATR': subset['ATR'].values,
            'T+1 Open': subset['T+1 Open'].values,
            'Signal Close': subset['Signal Close'].values,
            'Range %': subset['Range %'].values,
            'Equity at Signal': subset['Equity at Signal'].values,
            'Risk $': subset['Risk $'].values,
            'Risk bps': subset['Risk bps'].values,
            'Current Price': current_prices,
            'Open PnL': open_pnls,
            'Mkt Value': current_values
        })
        results.append(result)

    return results[0], results[1]


# -----------------------------------------------------------------------------
# 4c. TRAILING STRATEGY PERFORMANCE STATS
# -----------------------------------------------------------------------------

def calculate_trailing_strategy_stats(sig_df):
    """
    Calculate strategy performance stats for trailing 3, 6, and 12 month periods.

    Returns: dict with keys '3M', '6M', '12M', each containing a DataFrame with:
        Strategy, Trades, Win Rate, Profit Factor, Total PnL, Avg PnL, SQN
    """
    if sig_df.empty:
        return {'3M': pd.DataFrame(), '6M': pd.DataFrame(), '12M': pd.DataFrame()}

    today = pd.Timestamp(datetime.date.today())

    periods = {
        '3M': today - pd.Timedelta(days=63),   # ~3 months of trading days
        '6M': today - pd.Timedelta(days=126),  # ~6 months
        '12M': today - pd.Timedelta(days=252)  # ~12 months
    }

    results = {}

    for period_name, cutoff_date in periods.items():
        # Filter signals by Entry Date within the period
        period_df = sig_df[sig_df['Entry Date'] >= cutoff_date].copy()

        if period_df.empty:
            results[period_name] = pd.DataFrame()
            continue

        stats = []

        for strat in period_df['Strategy'].unique():
            strat_df = period_df[period_df['Strategy'] == strat]

            count = len(strat_df)
            if count == 0:
                continue

            winners = strat_df[strat_df['PnL'] > 0]
            losers = strat_df[strat_df['PnL'] < 0]

            win_rate = len(winners) / count if count > 0 else 0

            gross_profit = winners['PnL'].sum() if not winners.empty else 0
            gross_loss = abs(losers['PnL'].sum()) if not losers.empty else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

            total_pnl = strat_df['PnL'].sum()
            avg_pnl = strat_df['PnL'].mean()
            std_pnl = strat_df['PnL'].std()
            sqn = (avg_pnl / std_pnl * np.sqrt(count)) if std_pnl > 0 else 0

            stats.append({
                'Strategy': strat,
                'Trades': count,
                'Win Rate': win_rate,
                'Profit Factor': profit_factor,
                'Total PnL': total_pnl,
                'Avg PnL': avg_pnl,
                'SQN': sqn
            })

        # Add portfolio total
        if stats:
            total_count = len(period_df)
            total_winners = period_df[period_df['PnL'] > 0]
            total_losers = period_df[period_df['PnL'] < 0]
            total_win_rate = len(total_winners) / total_count if total_count > 0 else 0
            total_gross_profit = total_winners['PnL'].sum() if not total_winners.empty else 0
            total_gross_loss = abs(total_losers['PnL'].sum()) if not total_losers.empty else 0
            total_pf = total_gross_profit / total_gross_loss if total_gross_loss > 0 else float('inf') if total_gross_profit > 0 else 0
            total_pnl = period_df['PnL'].sum()
            total_avg = period_df['PnL'].mean()
            total_std = period_df['PnL'].std()
            total_sqn = (total_avg / total_std * np.sqrt(total_count)) if total_std > 0 else 0

            stats.append({
                'Strategy': 'TOTAL',
                'Trades': total_count,
                'Win Rate': total_win_rate,
                'Profit Factor': total_pf,
                'Total PnL': total_pnl,
                'Avg PnL': total_avg,
                'SQN': total_sqn
            })

        results[period_name] = pd.DataFrame(stats)

    return results


# -----------------------------------------------------------------------------
# 4d. RECENT EXITS (Last 5 Trading Days)
# -----------------------------------------------------------------------------

def get_recent_exits(sig_df, master_dict, trading_days=5):
    """
    Get positions that were EXITED in the last N trading days.
    Returns DataFrame with same columns as today's activity tables.
    """
    if sig_df.empty:
        return pd.DataFrame()

    today = pd.Timestamp(datetime.date.today())

    # Get unique trading dates from SPY to determine last N trading days
    spy_df = master_dict.get('SPY')
    if spy_df is not None and not spy_df.empty:
        trading_dates = spy_df.index.normalize().unique()
        trading_dates = trading_dates[trading_dates <= today]
        trading_dates = sorted(trading_dates, reverse=True)

        if len(trading_dates) >= trading_days:
            cutoff_date = trading_dates[trading_days - 1]
        else:
            cutoff_date = trading_dates[-1] if trading_dates else today - pd.Timedelta(days=7)
    else:
        # Fallback: use calendar days
        cutoff_date = today - pd.Timedelta(days=7)

    # Filter for positions exited in the period (Exit Date >= cutoff AND Time Stop <= today)
    exited_mask = (
        sig_df['Exit Date'].apply(lambda x: pd.Timestamp(x).normalize() >= cutoff_date)
        & sig_df['Exit Date'].apply(lambda x: pd.Timestamp(x).normalize() <= today)
        & sig_df['Time Stop'].apply(lambda x: pd.Timestamp(x).normalize() <= today)
    )
    exited_df = sig_df[exited_mask].copy()

    # Exclude today's exits (those are shown separately)
    exited_df = exited_df[exited_df['Exit Date'].apply(lambda x: pd.Timestamp(x).normalize() < today)]

    if exited_df.empty:
        return pd.DataFrame()

    print(f"   Found {len(exited_df)} exits in last {trading_days} trading days")

    # Build output frame with same schema as today's activity
    current_prices, open_pnls, current_values = [], [], []
    for row in exited_df.itertuples():
        t_clean = row.Ticker.replace('.', '-')
        t_df = master_dict.get(t_clean)
        if t_df is not None and not t_df.empty:
            if isinstance(t_df.columns, pd.MultiIndex):
                t_df.columns = t_df.columns.get_level_values(0)
            t_df.columns = [c.capitalize() for c in t_df.columns]
            last_close = t_df['Close'].iloc[-1]
        else:
            temp = yf.download(row.Ticker, period='1d', progress=False)
            if not temp.empty:
                if isinstance(temp.columns, pd.MultiIndex):
                    temp.columns = temp.columns.get_level_values(0)
                temp.columns = [c.capitalize() for c in temp.columns]
                last_close = temp['Close'].iloc[-1]
            else:
                last_close = row.Price

        # For closed positions, Open PnL = Realized PnL (the position is closed)
        current_prices.append(last_close)
        open_pnls.append(row.PnL)  # Use realized PnL since position is closed
        current_values.append(last_close * row.Shares)

    result = pd.DataFrame({
        'Date': exited_df['Date'].values,
        'Entry Date': exited_df['Entry Date'].values,
        'Exit Date': exited_df['Exit Date'].values,
        'Exit Type': exited_df['Exit Type'].values,
        'Time Stop': exited_df['Time Stop'].values,
        'Strategy': exited_df['Strategy'].values,
        'Ticker': exited_df['Ticker'].values,
        'Action': exited_df['Action'].values,
        'Entry Criteria': exited_df['Entry Criteria'].values,
        'Price': exited_df['Price'].values,
        'Shares': exited_df['Shares'].values,
        'PnL': exited_df['PnL'].values,
        'ATR': exited_df['ATR'].values,
        'T+1 Open': exited_df['T+1 Open'].values,
        'Signal Close': exited_df['Signal Close'].values,
        'Range %': exited_df['Range %'].values,
        'Equity at Signal': exited_df['Equity at Signal'].values,
        'Risk $': exited_df['Risk $'].values,
        'Risk bps': exited_df['Risk bps'].values,
        'Current Price': current_prices,
        'Open PnL': open_pnls,
        'Mkt Value': current_values
    })

    # Sort by Exit Date descending (most recent first)
    result = result.sort_values('Exit Date', ascending=False)

    return result


# -----------------------------------------------------------------------------
# 5. SIZING RECOMMENDATIONS ENGINE
# -----------------------------------------------------------------------------

def generate_sizing_recommendations(equity_series, daily_pnl_series, starting_equity):
    """
    Analyzes recent portfolio performance and generates sizing recommendations.
    
    Returns: dict with {
        'summary': str,
        'recommendations': list[str],
        'metrics': dict
    }
    """
    if equity_series.empty:
        return {
            'summary': 'Insufficient data for analysis',
            'recommendations': ['⚠️ No equity data available'],
            'metrics': {}
        }
    
    # Current state
    current_equity = equity_series.iloc[-1]
    sma_20 = equity_series.rolling(20).mean().iloc[-1]
    std_20 = equity_series.rolling(20).std().iloc[-1]
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Performance metrics
    total_return_pct = ((current_equity - starting_equity) / starting_equity) * 100
    
    # Recent performance (last 30 days)
    last_30d = daily_pnl_series.tail(30)
    if len(last_30d) > 0:
        avg_daily_pnl = last_30d.mean()
        std_daily_pnl = last_30d.std()
        win_rate_30d = (last_30d > 0).sum() / len(last_30d)
        best_day = last_30d.max()
        worst_day = last_30d.min()
        
        # Calculate PnL by day of week (full 12-month period)
        pnl_by_dow = {}
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        for i, dow_name in enumerate(dow_names):
            dow_mask = daily_pnl_series.index.dayofweek == i
            dow_pnl = daily_pnl_series[dow_mask]
            if len(dow_pnl) > 0:
                pnl_by_dow[dow_name] = dow_pnl.sum()
            else:
                pnl_by_dow[dow_name] = 0
    else:
        avg_daily_pnl = 0
        std_daily_pnl = 0
        win_rate_30d = 0
        best_day = 0
        worst_day = 0
        pnl_by_dow = {dow: 0 for dow in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']}
    
    # Drawdown analysis
    running_max = equity_series.expanding().max()
    drawdown_series = (equity_series - running_max) / running_max * 100
    current_dd = drawdown_series.iloc[-1]
    max_dd = drawdown_series.min()
    
    # Recent peak analysis (60 days)
    recent_peak = equity_series.tail(60).max()
    recent_dd = ((current_equity - recent_peak) / recent_peak) * 100
    
    # Generate recommendations
    recommendations = []
    
    # 1. Bollinger Band Position
    if current_equity > upper_bb:
        recommendations.append("🟡 **CAUTION:** Equity above upper Bollinger Band - portfolio may be overextended")
        recommendations.append("   → Consider reducing position sizes by 20-30% or tightening entry criteria")
    elif current_equity < lower_bb:
        recommendations.append("🔴 **ALERT:** Equity below lower Bollinger Band - portfolio underperforming")
        recommendations.append("   → Consider reducing size by 30-50% until equity recovers to 20 SMA")
    elif current_equity > sma_20:
        recommendations.append("✅ **HEALTHY:** Equity above 20-day SMA and within normal bands")
        recommendations.append("   → Continue current sizing strategy")
    else:
        recommendations.append("🟠 **WARNING:** Equity below 20-day SMA but within bands")
        recommendations.append("   → Consider reducing size by 10-20% defensively")
    
    # 2. Drawdown-based recommendations
    if abs(current_dd) > 15:
        recommendations.append(f"🛑 **SEVERE DRAWDOWN:** Currently down {abs(current_dd):.1f}% from peak")
        recommendations.append("   → STOP TRADING or reduce size to minimum (10-20% of normal)")
    elif abs(current_dd) > 10:
        recommendations.append(f"🔴 **MAJOR DRAWDOWN:** Currently down {abs(current_dd):.1f}% from peak")
        recommendations.append("   → Cut size by 50% until equity recovers above 20 SMA")
    elif abs(recent_dd) > 5:
        recommendations.append(f"🟠 **MODERATE DRAWDOWN:** Down {abs(recent_dd):.1f}% from recent high (60d)")
        recommendations.append("   → Reduce size by 20-30% and tighten entry criteria")
    
    # 3. Win rate check
    if win_rate_30d < 0.45:
        recommendations.append(f"⚠️ **LOW WIN RATE:** Only {win_rate_30d:.1%} winners in last 30 days")
        recommendations.append("   → Review entry criteria and consider taking a break")
    elif win_rate_30d > 0.65:
        recommendations.append(f"💪 **STRONG WIN RATE:** {win_rate_30d:.1%} winners in last 30 days")
        recommendations.append("   → Performance is solid, maintain or slightly increase size")
    
    # 4. Streak analysis
    recent_streak = 0
    for pnl in reversed(list(daily_pnl_series.tail(10))):
        if (pnl > 0 and recent_streak >= 0) or (pnl < 0 and recent_streak <= 0):
            recent_streak += 1 if pnl > 0 else -1
        else:
            break
    
    if recent_streak >= 5:
        recommendations.append(f"🎯 **HOT STREAK:** {recent_streak} winning days in a row")
        recommendations.append("   → Stay disciplined, don't overtrade the hot hand")
    elif recent_streak <= -5:
        recommendations.append(f"❄️ **COLD STREAK:** {abs(recent_streak)} losing days in a row")
        recommendations.append("   → Take a break and review your process")
    
    # Overall summary
    if current_equity > starting_equity * 1.10:
        summary = "🚀 STRONG PERFORMANCE - Portfolio up significantly"
    elif current_equity > starting_equity * 1.05:
        summary = "📈 POSITIVE PERFORMANCE - Portfolio grinding higher"
    elif current_equity > starting_equity:
        summary = "➡️ FLAT TO POSITIVE - Portfolio slightly ahead"
    elif current_equity > starting_equity * 0.95:
        summary = "⚠️ SLIGHT DRAWDOWN - Portfolio slightly below starting level"
    else:
        summary = "🔴 SIGNIFICANT DRAWDOWN - Portfolio needs attention"
    
    return {
        'summary': summary,
        'recommendations': recommendations if recommendations else ['✅ No specific actions needed - continue monitoring'],
        'metrics': {
            'current_equity': current_equity,
            'total_return_pct': total_return_pct,
            'current_vs_sma': ((current_equity - sma_20) / sma_20) * 100,
            'current_dd_pct': current_dd,
            'max_dd_pct': max_dd,
            'recent_dd_pct': recent_dd,
            'win_rate_30d': win_rate_30d,
            'avg_daily_pnl': avg_daily_pnl,
            'best_day': best_day,
            'worst_day': worst_day,
            # New metrics
            'sharpe_ratio': avg_daily_pnl / std_daily_pnl if std_daily_pnl > 0 else 0,
            'std_daily_pnl': std_daily_pnl,
            'plus_1std': avg_daily_pnl + std_daily_pnl,
            'plus_2std': avg_daily_pnl + (2 * std_daily_pnl),
            'minus_1std': avg_daily_pnl - std_daily_pnl,
            'minus_2std': avg_daily_pnl - (2 * std_daily_pnl),
            # Percentage versions (as % of equity)
            'plus_1std_pct': ((avg_daily_pnl + std_daily_pnl) / starting_equity) * 100 if starting_equity > 0 else 0,
            'plus_2std_pct': ((avg_daily_pnl + (2 * std_daily_pnl)) / starting_equity) * 100 if starting_equity > 0 else 0,
            'minus_1std_pct': ((avg_daily_pnl - std_daily_pnl) / starting_equity) * 100 if starting_equity > 0 else 0,
            'minus_2std_pct': ((avg_daily_pnl - (2 * std_daily_pnl)) / starting_equity) * 100 if starting_equity > 0 else 0,
            # Day of week breakdown
            'pnl_by_dow': pnl_by_dow
        }
    }


# -----------------------------------------------------------------------------
# 6. EMAIL GENERATION (SLIMMED TABLE FOR GMAIL)
# -----------------------------------------------------------------------------

def _format_activity_table_html(df, label):
    """
    Formats an entered/exited positions dataframe into HTML for the email.
    Uses the same slimmed column set and styling as the open positions table.
    Returns HTML string (summary bar + table), or a 'no activity' placeholder.
    """
    if df.empty:
        return f"<div style='color: #aaa; padding: 20px; text-align: center;'>No positions {label.lower()} today</div>"

    # Summary stats
    total_long = df[df['Action'] == 'BUY']['Mkt Value'].sum()
    total_short = df[df['Action'] != 'BUY']['Mkt Value'].sum()
    net_exposure = total_long - total_short
    total_pnl = df['Open PnL'].sum()
    long_count = (df['Action'] == 'BUY').sum()
    short_count = len(df) - long_count

    summary_html = f"""
    <div style="background: #1a1a1a; padding: 12px; border-radius: 6px; margin: 10px 0; font-size: 13px; color: #fff;">
        <span style="color: #aaa;">Positions: </span><strong style="color: #fff;">{len(df)}</strong>
        <span style="margin-left: 15px; color: #aaa;">Long: </span><strong style="color: #00CC00;">{long_count}</strong>
        <span style="margin-left: 15px; color: #aaa;">Short: </span><strong style="color: #CC0000;">{short_count}</strong>
        <span style="margin-left: 15px; color: #aaa;">Total Long: </span><strong style="color: #fff;">${total_long:,.0f}</strong>
        <span style="margin-left: 15px; color: #aaa;">Total Short: </span><strong style="color: #fff;">${total_short:,.0f}</strong>
        <span style="margin-left: 15px; color: #aaa;">Net Exposure: </span><strong style="color: #fff;">${net_exposure:+,.0f}</strong>
        <span style="margin-left: 15px; color: #aaa;">Total P&L: </span>
        <strong style="color: {'#00CC00' if total_pnl >= 0 else '#CC0000'};">${total_pnl:,.0f}</strong>
    </div>
    """

    # Same slimmed columns as open positions table
    tbl = df[[
        'Entry Date', 'Time Stop', 'Strategy', 'Ticker', 'Price', 'Shares',
        'PnL', 'Risk $', 'Risk bps', 'Current Price', 'Mkt Value'
    ]].copy()

    tbl['Entry Date'] = tbl['Entry Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    tbl['Time Stop'] = tbl['Time Stop'].apply(lambda x: x.strftime('%Y-%m-%d'))
    tbl['Price'] = tbl['Price'].apply(lambda x: f"${x:.2f}")
    tbl['Current Price'] = tbl['Current Price'].apply(lambda x: f"${x:.2f}")
    tbl['PnL'] = tbl['PnL'].apply(
        lambda x: f'<span style="color: {"#00CC00" if x >= 0 else "#CC0000"}; font-weight: bold;">${x:,.0f}</span>'
    )
    tbl['Mkt Value'] = tbl['Mkt Value'].apply(lambda x: f"${x:,.0f}")
    tbl['Risk $'] = tbl['Risk $'].apply(lambda x: f"${x:,.0f}")
    tbl['Shares'] = tbl['Shares'].apply(lambda x: f"{x:,}")

    table_html = tbl.to_html(index=False, escape=False, classes='positions-table')
    return summary_html + table_html


def send_portfolio_email(chart_path, open_positions_df, sizing_analysis, metrics,
                         entered_today_df=None, exited_today_df=None,
                         trailing_stats=None, recent_exits_df=None):
    """
    Sends portfolio health email with chart attachment and HTML tables.
    """
    sender_email = os.environ.get("EMAIL_USER")
    sender_password = os.environ.get("EMAIL_PASS")
    receiver_email = "mckinleyslade@gmail.com"
    
    if not sender_email or not sender_password:
        print("⚠️ Email credentials not found - skipping email")
        return
    
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Build metrics section with Sharpe and std dev metrics
    metrics_html = f"""
    <div style="background: #1a1a1a; padding: 15px; border-radius: 8px; margin: 20px 0;">
        <h3 style="color: #fff; margin-top: 0;">📊 Key Metrics (12 Months)</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
            <div>
                <div style="color: #aaa; font-size: 12px;">Total Return</div>
                <div style="color: {'#00CC00' if metrics['total_return_pct'] >= 0 else '#CC0000'}; font-size: 20px; font-weight: bold;">
                    {metrics['total_return_pct']:+.1f}%
                </div>
            </div>
            <div>
                <div style="color: #aaa; font-size: 12px;">Current Equity</div>
                <div style="color: #fff; font-size: 20px; font-weight: bold;">
                    ${metrics['current_equity']:,.0f}
                </div>
            </div>
            <div>
                <div style="color: #aaa; font-size: 12px;">Sharpe Ratio</div>
                <div style="color: {'#00CC00' if metrics['sharpe_ratio'] >= 1.0 else '#FFA500' if metrics['sharpe_ratio'] >= 0.5 else '#CC0000'}; font-size: 20px; font-weight: bold;">
                    {metrics['sharpe_ratio']:.2f}
                </div>
            </div>
            <div>
                <div style="color: #aaa; font-size: 12px;">vs 20-Day SMA</div>
                <div style="color: {'#00CC00' if metrics['current_vs_sma'] >= 0 else '#CC0000'}; font-size: 20px; font-weight: bold;">
                    {metrics['current_vs_sma']:+.1f}%
                </div>
            </div>
            <div>
                <div style="color: #aaa; font-size: 12px;">Current Drawdown</div>
                <div style="color: {'#CC0000' if abs(metrics['current_dd_pct']) > 5 else '#FFA500'}; font-size: 20px; font-weight: bold;">
                    {metrics['current_dd_pct']:.1f}%
                </div>
            </div>
            <div>
                <div style="color: #aaa; font-size: 12px;">Max Drawdown (12M)</div>
                <div style="color: #CC0000; font-size: 20px; font-weight: bold;">
                    {metrics['max_dd_pct']:.1f}%
                </div>
            </div>
            <div>
                <div style="color: #aaa; font-size: 12px;">Win Rate (30D)</div>
                <div style="color: {'#00CC00' if metrics['win_rate_30d'] >= 0.55 else '#FFA500'}; font-size: 20px; font-weight: bold;">
                    {metrics['win_rate_30d']:.1%}
                </div>
            </div>
            <div>
                <div style="color: #aaa; font-size: 12px;">Avg Daily P&L</div>
                <div style="color: {'#00CC00' if metrics['avg_daily_pnl'] >= 0 else '#CC0000'}; font-size: 20px; font-weight: bold;">
                    ${metrics['avg_daily_pnl']:,.0f}
                </div>
            </div>
            <div>
                <div style="color: #aaa; font-size: 12px;">Daily Std Dev</div>
                <div style="color: #fff; font-size: 20px; font-weight: bold;">
                    ${metrics['std_daily_pnl']:,.0f}
                </div>
            </div>
        </div>
        
        <!-- Standard Deviation Metrics -->
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #333;">
            <div style="color: #aaa; font-size: 13px; margin-bottom: 8px;">Standard Deviation Thresholds (% of Equity)</div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-size: 12px;">
                <div>
                    <span style="color: #aaa;">+2σ:</span>
                    <span style="color: #00CC00; font-weight: bold;">
                        {metrics['plus_2std_pct']:+.2f}%
                    </span>
                </div>
                <div>
                    <span style="color: #aaa;">+1σ:</span>
                    <span style="color: #00CC00; font-weight: bold;">
                        {metrics['plus_1std_pct']:+.2f}%
                    </span>
                </div>
                <div>
                    <span style="color: #aaa;">-1σ:</span>
                    <span style="color: #CC0000; font-weight: bold;">
                        {metrics['minus_1std_pct']:.2f}%
                    </span>
                </div>
                <div>
                    <span style="color: #aaa;">-2σ:</span>
                    <span style="color: #CC0000; font-weight: bold;">
                        {metrics['minus_2std_pct']:.2f}%
                    </span>
                </div>
            </div>
        </div>
    </div>
    """
    
    # Build recommendations section
    recs_html = "<ul style='margin: 10px 0; padding-left: 20px;'>"
    for rec in sizing_analysis['recommendations']:
        recs_html += f"<li style='margin: 8px 0; color: #fff;'>{rec}</li>"
    recs_html += "</ul>"
    
    # Build open positions table - SLIMMED FOR GMAIL (15 columns)
    if not open_positions_df.empty:
        # Calculate summary stats
        total_long = open_positions_df[open_positions_df['Action'] == 'BUY']['Mkt Value'].sum()
        total_short = open_positions_df[open_positions_df['Action'] != 'BUY']['Mkt Value'].sum()
        net_exposure = total_long - total_short
        total_pnl = open_positions_df['Open PnL'].sum()
        long_count = (open_positions_df['Action'] == 'BUY').sum()
        short_count = len(open_positions_df) - long_count
        
        positions_summary = f"""
        <div style="background: #1a1a1a; padding: 12px; border-radius: 6px; margin: 10px 0; font-size: 13px; color: #fff;">
            <span style="color: #aaa;">Positions: </span><strong style="color: #fff;">{len(open_positions_df)}</strong>
            <span style="margin-left: 15px; color: #aaa;">Long: </span><strong style="color: #00CC00;">{long_count}</strong>
            <span style="margin-left: 15px; color: #aaa;">Short: </span><strong style="color: #CC0000;">{short_count}</strong>
            <span style="margin-left: 15px; color: #aaa;">Total Long: </span><strong style="color: #fff;">${total_long:,.0f}</strong>
            <span style="margin-left: 15px; color: #aaa;">Total Short: </span><strong style="color: #fff;">${total_short:,.0f}</strong>
            <span style="margin-left: 15px; color: #aaa;">Net Exposure: </span><strong style="color: #fff;">${net_exposure:+,.0f}</strong>
            <span style="margin-left: 15px; color: #aaa;">Total P&L: </span>
            <strong style="color: {'#00CC00' if total_pnl >= 0 else '#CC0000'};">${total_pnl:,.0f}</strong>
        </div>
        """
        
        # SLIMMED DOWN: 11 columns (removed Date, Exit Type, Entry Criteria, Equity at Signal)
        pos_table = open_positions_df[[
            'Entry Date', 'Time Stop', 'Strategy', 'Ticker', 'Price', 'Shares', 
            'PnL', 'Risk $', 'Risk bps', 'Current Price', 'Mkt Value'
        ]].copy()
        
        # Format dates
        pos_table['Entry Date'] = pos_table['Entry Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        pos_table['Time Stop'] = pos_table['Time Stop'].apply(lambda x: x.strftime('%Y-%m-%d'))
        
        # Format numbers with PnL color coding
        pos_table['Price'] = pos_table['Price'].apply(lambda x: f"${x:.2f}")
        pos_table['Current Price'] = pos_table['Current Price'].apply(lambda x: f"${x:.2f}")
        # PnL with conditional formatting (green if positive, red if negative)
        pos_table['PnL'] = pos_table['PnL'].apply(
            lambda x: f'<span style="color: {"#00CC00" if x >= 0 else "#CC0000"}; font-weight: bold;">${x:,.0f}</span>'
        )
        pos_table['Mkt Value'] = pos_table['Mkt Value'].apply(lambda x: f"${x:,.0f}")
        pos_table['Risk $'] = pos_table['Risk $'].apply(lambda x: f"${x:,.0f}")
        pos_table['Shares'] = pos_table['Shares'].apply(lambda x: f"{x:,}")
        
        positions_html = pos_table.to_html(index=False, escape=False, classes='positions-table')
    else:
        positions_summary = "<div style='color: #aaa; padding: 20px; text-align: center;'>No open positions</div>"
        positions_html = ""

    # Build today's activity section
    if entered_today_df is None:
        entered_today_df = pd.DataFrame()
    if exited_today_df is None:
        exited_today_df = pd.DataFrame()

    entered_html = _format_activity_table_html(entered_today_df, "Entered")
    exited_html = _format_activity_table_html(exited_today_df, "Exited")

    todays_activity_html = f"""
    <div class="section">
        <h2>📅 Today's Activity</h2>
        <h3 style="color: #00CC00; margin-bottom: 5px;">🟢 Entered Today</h3>
        {entered_html}
        <h3 style="color: #CC0000; margin-top: 20px; margin-bottom: 5px;">🔴 Exited Today</h3>
        {exited_html}
    </div>
    """

    # Build trailing strategy stats HTML
    trailing_stats_html = ""
    if trailing_stats:
        trailing_stats_html = """
    <div class="section">
        <h2>📈 Strategy Performance by Period</h2>
        """
        for period_name in ['3M', '6M', '12M']:
            period_df = trailing_stats.get(period_name, pd.DataFrame())
            if not period_df.empty:
                # Format the dataframe for display
                display_df = period_df.copy()
                display_df['Win Rate'] = display_df['Win Rate'].apply(lambda x: f"{x:.1%}")
                display_df['Profit Factor'] = display_df['Profit Factor'].apply(
                    lambda x: f"{x:.2f}" if x != float('inf') else "∞"
                )
                display_df['Total PnL'] = display_df['Total PnL'].apply(
                    lambda x: f'<span style="color: {"#00CC00" if x >= 0 else "#CC0000"};">${x:,.0f}</span>'
                )
                display_df['Avg PnL'] = display_df['Avg PnL'].apply(
                    lambda x: f'<span style="color: {"#00CC00" if x >= 0 else "#CC0000"};">${x:,.0f}</span>'
                )
                display_df['SQN'] = display_df['SQN'].apply(lambda x: f"{x:.2f}")

                table_html = display_df.to_html(index=False, escape=False, classes='positions-table')
                trailing_stats_html += f"""
        <h3 style="color: #4CAF50; margin-top: 15px; margin-bottom: 5px;">{period_name} Performance</h3>
        {table_html}
        """
            else:
                trailing_stats_html += f"""
        <h3 style="color: #4CAF50; margin-top: 15px; margin-bottom: 5px;">{period_name} Performance</h3>
        <div style="color: #aaa; padding: 10px;">No trades in this period</div>
        """
        trailing_stats_html += "</div>"

    # Build recent exits HTML (last 5 trading days)
    recent_exits_html = ""
    if recent_exits_df is not None and not recent_exits_df.empty:
        # Calculate summary stats
        total_pnl = recent_exits_df['PnL'].sum()
        winners = (recent_exits_df['PnL'] > 0).sum()
        losers = (recent_exits_df['PnL'] < 0).sum()
        win_rate = winners / len(recent_exits_df) if len(recent_exits_df) > 0 else 0

        summary_bar = f"""
        <div style="background: #1a1a1a; padding: 12px; border-radius: 6px; margin: 10px 0; font-size: 13px; color: #fff;">
            <span style="color: #aaa;">Exits: </span><strong style="color: #fff;">{len(recent_exits_df)}</strong>
            <span style="margin-left: 15px; color: #aaa;">Winners: </span><strong style="color: #00CC00;">{winners}</strong>
            <span style="margin-left: 15px; color: #aaa;">Losers: </span><strong style="color: #CC0000;">{losers}</strong>
            <span style="margin-left: 15px; color: #aaa;">Win Rate: </span><strong style="color: #fff;">{win_rate:.1%}</strong>
            <span style="margin-left: 15px; color: #aaa;">Total PnL: </span>
            <strong style="color: {'#00CC00' if total_pnl >= 0 else '#CC0000'};">${total_pnl:,.0f}</strong>
        </div>
        """

        # Slimmed table columns
        tbl = recent_exits_df[[
            'Entry Date', 'Exit Date', 'Strategy', 'Ticker', 'Action', 'Price', 'Shares',
            'PnL', 'Risk $'
        ]].copy()

        tbl['Entry Date'] = tbl['Entry Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        tbl['Exit Date'] = tbl['Exit Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        tbl['Price'] = tbl['Price'].apply(lambda x: f"${x:.2f}")
        tbl['PnL'] = tbl['PnL'].apply(
            lambda x: f'<span style="color: {"#00CC00" if x >= 0 else "#CC0000"}; font-weight: bold;">${x:,.0f}</span>'
        )
        tbl['Risk $'] = tbl['Risk $'].apply(lambda x: f"${x:,.0f}")
        tbl['Shares'] = tbl['Shares'].apply(lambda x: f"{x:,}")

        table_html = tbl.to_html(index=False, escape=False, classes='positions-table')

        recent_exits_html = f"""
    <div class="section">
        <h2>📋 Recent Exits (Last 5 Trading Days)</h2>
        {summary_bar}
        {table_html}
    </div>
        """
    else:
        recent_exits_html = """
    <div class="section">
        <h2>📋 Recent Exits (Last 5 Trading Days)</h2>
        <div style="color: #aaa; padding: 20px; text-align: center;">No exits in the last 5 trading days</div>
    </div>
        """

    # Assemble email
    subject = f"📊 Portfolio Health Report - {date_str}"
    
    html_content = f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #0e1117; color: #fff; }}
                .container {{ max-width: 1000px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #1a237e, #283593); padding: 25px; border-radius: 8px; text-align: center; margin-bottom: 20px; color: #ffffff; }}
                .section {{ background: #1a1a1a; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                
                /* White text in tables */
                .positions-table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin-top: 10px; 
                    font-size: 11px;
                    color: #ffffff;
                }}
                .positions-table th {{ 
                    background: #2a2a2a; 
                    padding: 8px; 
                    text-align: left; 
                    border-bottom: 2px solid #444;
                    color: #ffffff;
                    font-weight: bold;
                    font-size: 10px;
                }}
                .positions-table td {{ 
                    padding: 6px; 
                    border-bottom: 1px solid #333;
                    color: #ffffff;
                }}
                .positions-table tr:hover {{ 
                    background: #252525; 
                }}
                
                h2 {{ color: #fff; margin-top: 0; }}
                h3 {{ color: #aaa; margin-top: 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 style="margin: 0; font-size: 28px;">📊 Portfolio Health Report</h1>
                    <div style="font-size: 14px; opacity: 0.8; margin-top: 5px;">{date_str}</div>
                </div>

                <div class="section">
                    <h2>📈 12-Month Equity Curve</h2>
                    <p style="color: #aaa; font-size: 13px; margin-top: -10px;">Starting Equity: ${ACCOUNT_VALUE:,} | Data from 2000 for percentiles</p>
                    <img src="cid:equity_chart" style="max-width: 100%; border-radius: 8px;">
                </div>

                {metrics_html}

                <div class="section">
                    <h2>💼 Open Positions</h2>
                    {positions_summary}
                    {positions_html}
                </div>
                
                {todays_activity_html}

                {recent_exits_html}

                {trailing_stats_html}

                <div style="text-align: center; padding: 20px; color: #666; font-size: 12px; border-top: 1px solid #333; margin-top: 30px;">
                    Generated by daily_portfolio_report.py | {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                </div>
            </div>
        </body>
    </html>
    """
    
    # Create message
    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    
    # Attach HTML
    msg.attach(MIMEText(html_content, "html"))
    
    # Attach chart image
    if chart_path and os.path.exists(chart_path):
        with open(chart_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-ID', '<equity_chart>')
            msg.attach(img)
    
    # Send
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print(f"📧 Email sent successfully to {receiver_email}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


# -----------------------------------------------------------------------------
# 7. MAIN EXECUTION
# -----------------------------------------------------------------------------

def main():
    """
    Main execution function - runs the full portfolio health report.
    """
    print("=" * 70)
    print("📊 DAILY PORTFOLIO HEALTH REPORT")
    print(f"   Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    CURRENT_ACCOUNT_SIZE = ACCOUNT_VALUE
    
    try:
        # 1. Run 12-month backtest (uses data since 2000 for percentiles)
        signals_df, equity_series, daily_pnl, master_dict = run_12month_backtest(starting_equity=CURRENT_ACCOUNT_SIZE)
        
        if signals_df is None or equity_series is None or equity_series.empty:
            print("❌ Backtest failed - cannot generate report")
            return
        
        # 2. Generate chart
        print("\n📊 Generating charts...")
        fig = create_portfolio_chart(equity_series, daily_pnl, CURRENT_ACCOUNT_SIZE)
        
        if fig is None:
            print("❌ Failed to create chart")
            return
        
        chart_path = save_chart_as_png(fig, filepath='/tmp/portfolio_health.png')
        
        if not chart_path:
            print("❌ Failed to save chart - check kaleido installation")
            return
        
        # 3. Get open positions - FIXED: Use backtest data (same as strat_backtester)
        print("\n💼 Calculating open positions from backtest...")
        open_positions = get_open_positions_from_backtest(signals_df, master_dict)
        
        # 3b. Get today's entered & exited positions
        print("\n📅 Checking today's activity...")
        entered_today, exited_today = get_todays_activity(signals_df, master_dict)

        # 3c. Get recent exits (last 5 trading days)
        print("\n📋 Getting recent exits...")
        recent_exits = get_recent_exits(signals_df, master_dict, trading_days=5)

        # 3d. Calculate trailing strategy stats (3M, 6M, 12M)
        print("\n📈 Calculating trailing strategy performance...")
        trailing_stats = calculate_trailing_strategy_stats(signals_df)

        # 4. Generate sizing recommendations
        print("\n🎯 Analyzing performance...")
        sizing_analysis = generate_sizing_recommendations(equity_series, daily_pnl, CURRENT_ACCOUNT_SIZE)

        print("\n" + "=" * 70)
        print(f"   {sizing_analysis['summary']}")
        print("=" * 70)
        for rec in sizing_analysis['recommendations']:
            print(f"   {rec}")
        print("=" * 70)

        # 5. Send email
        print("\n📧 Sending email report...")
        send_portfolio_email(
            chart_path=chart_path,
            open_positions_df=open_positions,
            sizing_analysis=sizing_analysis,
            metrics=sizing_analysis['metrics'],
            entered_today_df=entered_today,
            exited_today_df=exited_today,
            trailing_stats=trailing_stats,
            recent_exits_df=recent_exits
        )
        
        print("\n✅ Portfolio health report completed successfully!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
