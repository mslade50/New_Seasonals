import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import time
from pandas.tseries.offsets import BusinessDay
import plotly.graph_objects as go
import sys
import os

# -----------------------------------------------------------------------------
# IMPORT STRATEGY BOOK FROM ROOT
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from strategy_config import STRATEGY_BOOK
except ImportError:
    st.error("Could not find strategy_config.py in the root directory.")
    STRATEGY_BOOK = []

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
PRIMARY_SZNL_PATH = "sznl_ranks.csv"      
BACKUP_SZNL_PATH = "seasonal_ranks.csv"   

@st.cache_resource 
def load_seasonal_map():
    def load_raw_csv(path):
        try:
            df = pd.read_csv(path)
            if 'ticker' not in df.columns or 'Date' not in df.columns:
                return pd.DataFrame()
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            df = df.dropna(subset=["Date", "ticker"])
            df["Date"] = df["Date"].dt.normalize()
            df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
            return df
        except Exception:
            return pd.DataFrame()

    df_primary = load_raw_csv(PRIMARY_SZNL_PATH)
    df_backup = load_raw_csv(BACKUP_SZNL_PATH)

    if df_primary.empty and df_backup.empty:
        return {}
    elif df_primary.empty:
        final_df = df_backup
    elif df_backup.empty:
        final_df = df_primary
    else:
        final_df = pd.concat([df_primary, df_backup], axis=0)
        final_df = final_df.drop_duplicates(subset=['ticker', 'Date'], keep='first')

    output_map = {}
    final_df = final_df.sort_values(by="Date")
    for ticker, group in final_df.groupby("ticker"):
        series = group.set_index("Date")["seasonal_rank"]
        output_map[ticker] = series
        
    return output_map

def get_sznl_val_series(ticker, dates, sznl_map):
    ticker = ticker.upper()
    t_series = sznl_map.get(ticker)
    if t_series is None and ticker == "^GSPC":
        t_series = sznl_map.get("SPY")
    if t_series is None:
        return pd.Series(50.0, index=dates)
    return dates.map(t_series).fillna(50.0)

# -----------------------------------------------------------------------------
# HELPER: BATCH DOWNLOADER
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def download_historical_data(tickers, start_date="2000-01-01"):
    if not tickers: return {}
    clean_tickers = list(set([str(t).strip().upper().replace('.', '-') for t in tickers]))
    data_dict = {}
    CHUNK_SIZE = 50 
    total = len(clean_tickers)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, total, CHUNK_SIZE):
        chunk = clean_tickers[i : i + CHUNK_SIZE]
        current_progress = min((i + CHUNK_SIZE) / total, 1.0)
        status_text.text(f"📥 Downloading batch {i+1}-{min(i+CHUNK_SIZE, total)} of {total}...")
        progress_bar.progress(current_progress)
        try:
            df = yf.download(chunk, start=start_date, group_by='ticker', auto_adjust=False, progress=False, threads=True)
            if df.empty: continue
            if len(chunk) == 1:
                ticker = chunk[0]
                if 'Close' in df.columns:
                    df.index = df.index.tz_localize(None)
                    data_dict[ticker] = df
            else:
                available_tickers = df.columns.levels[0]
                for t in available_tickers:
                    try:
                        t_df = df[t].copy()
                        if t_df.empty or 'Close' not in t_df.columns: continue
                        t_df.index = t_df.index.tz_localize(None)
                        data_dict[t] = t_df
                    except Exception:
                        continue
            time.sleep(0.25)
        except Exception:
            continue

    progress_bar.empty()
    status_text.empty()
    return data_dict

# -----------------------------------------------------------------------------
# ENGINE
# -----------------------------------------------------------------------------
def calculate_indicators(df, sznl_map, ticker, market_series=None):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean() 
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    is_open_gap = (df['Low'] > df['High'].shift(1)).astype(int)
    df['GapCount_21'] = is_open_gap.rolling(21).sum() 
    df['GapCount_10'] = is_open_gap.rolling(10).sum()
    df['GapCount_5'] = is_open_gap.rolling(5).sum() 

    denom = (df['High'] - df['Low'])
    df['RangePct'] = np.where(denom == 0, 0.5, (df['Close'] - df['Low']) / denom)

    for window in [5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window)
        df[f'rank_ret_{window}d'] = df[f'ret_{window}d'].expanding(min_periods=50).rank(pct=True) * 100.0
        
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    
    vol_ma = df['Volume'].rolling(63).mean()
    df['vol_ratio'] = df['Volume'] / vol_ma
    df['vol_ma'] = vol_ma
    cond_vol_ma = df['Volume'] > vol_ma
    cond_vol_up = df['Volume'] > df['Volume'].shift(1)
    df['Vol_Spike'] = cond_vol_ma & cond_vol_up
    
    cond_green = df['Close'] > df['Open']
    is_accumulation = (df['Vol_Spike'] & cond_green).astype(int)
    df['AccCount_21'] = is_accumulation.rolling(21).sum()
    
    cond_red = df['Close'] < df['Open']
    is_distribution = (df['Vol_Spike'] & cond_red).astype(int)
    df['DistCount_21'] = is_distribution.rolling(21).sum()
    
    vol_ma_10 = df['Volume'].rolling(10).mean()
    df['vol_ratio_10d'] = vol_ma_10 / vol_ma
    df['vol_ratio_10d_rank'] = df['vol_ratio_10d'].expanding(min_periods=50).rank(pct=True) * 100.0
    
    df['Sznl'] = get_sznl_val_series(ticker, df.index, sznl_map)
    df['Mkt_Sznl_Ref'] = get_sznl_val_series("^GSPC", df.index, sznl_map)
    
    if not df.empty:
        start_ts = df.index[0]
        df['age_years'] = (df.index - start_ts).days / 365.25
    else:
        df['age_years'] = 0.0

    if market_series is not None:
        df['Market_Above_SMA200'] = market_series.reindex(df.index, method='ffill').fillna(False)
    
    rolling_high = df['High'].shift(1).rolling(252).max()
    rolling_low = df['Low'].shift(1).rolling(252).min()
    df['is_52w_high'] = df['High'] > rolling_high
    df['is_52w_low'] = df['Low'] < rolling_low
        
    return df

def get_historical_mask(df, params, sznl_map):
    mask = pd.Series(True, index=df.index)

    if params.get('use_dow_filter', False):
        allowed = params.get('allowed_days', [])
        mask &= df.index.dayofweek.isin(allowed)

    if 'allowed_cycles' in params:
        allowed_cycles = params['allowed_cycles']
        if allowed_cycles and len(allowed_cycles) < 4:
            mask &= (df.index.year % 4).isin(allowed_cycles)

    mask &= (df['Close'] >= params.get('min_price', 0))
    mask &= (df['vol_ma'] >= params.get('min_vol', 0))
    mask &= (df['age_years'] >= params.get('min_age', 0))
    mask &= (df['age_years'] <= params.get('max_age', 100))

    if 'ATR' in df.columns:
        atr_pct = (df['ATR'] / df['Close']) * 100
        min_atr = params.get('min_atr_pct', 0.0)
        max_atr = params.get('max_atr_pct', 1000.0)
        mask &= (atr_pct >= min_atr) & (atr_pct <= max_atr)

    trend_opt = params.get('trend_filter', 'None')
    if trend_opt == "Price > 200 SMA":
        mask &= (df['Close'] > df['SMA200'])
    elif trend_opt == "Price > Rising 200 SMA":
        mask &= (df['Close'] > df['SMA200']) & (df['SMA200'] > df['SMA200'].shift(1))
    elif "Market" in trend_opt or "SPY" in trend_opt:
        if 'Market_Above_SMA200' in df.columns:
            is_above = df['Market_Above_SMA200']
            if ">" in trend_opt: mask &= is_above
            elif "<" in trend_opt: mask &= ~is_above

    if 'ma_consec_filters' in params:
        for maf in params['ma_consec_filters']:
            length = maf['length']
            col_name = f"SMA{length}"
            if col_name not in df.columns: continue
            if maf['logic'] == 'Above': cond = df['Close'] > df[col_name]
            elif maf['logic'] == 'Below': cond = df['Close'] < df[col_name]
            else: continue
            consec = maf.get('consec', 1)
            if consec > 1: cond = (cond.rolling(consec).sum() == consec)
            mask &= cond

    if params.get('use_range_filter', False):
        rn_val = df['RangePct'] * 100
        mask &= (rn_val >= params.get('range_min', 0)) & (rn_val <= params.get('range_max', 100))

    if 'perf_filters' in params:
        for pf in params['perf_filters']:
            col = f"rank_ret_{pf['window']}d"
            consec = pf.get('consecutive', 1)
            if pf['logic'] == '<': cond_f = df[col] < pf['thresh']
            else: cond_f = df[col] > pf['thresh']
            if consec > 1: cond_f = (cond_f.rolling(consec).sum() == consec)
            mask &= cond_f
        if params.get('perf_first_instance', False):
            lookback = params.get('perf_lookback', 21)
            prev_inst = mask.shift(1).rolling(lookback).sum()
            mask &= (prev_inst == 0)

    elif params.get('use_perf_rank', False):
        col = f"rank_ret_{params['perf_window']}d"
        if params['perf_logic'] == '<': raw = df[col] < params['perf_thresh']
        else: raw = df[col] > params['perf_thresh']
        consec = params.get('perf_consecutive', 1)
        if consec > 1: persist = (raw.rolling(consec).sum() == consec)
        else: persist = raw
        mask &= persist
        if params.get('perf_first_instance', False):
            lookback = params.get('perf_lookback', 21)
            prev_inst = mask.shift(1).rolling(lookback).sum()
            mask &= (prev_inst == 0)

    if params.get('use_gap_filter', False):
        lookback = params.get('gap_lookback', 21)
        col_name = f'GapCount_{lookback}' if f'GapCount_{lookback}' in df.columns else 'GapCount_21'
        g_val = df.get(col_name, 0)
        g_thresh = params.get('gap_thresh', 0)
        g_logic = params.get('gap_logic', '>')
        if g_logic == ">": mask &= (g_val > g_thresh)
        elif g_logic == "<": mask &= (g_val < g_thresh)
        elif g_logic == "=": mask &= (g_val == g_thresh)

    if params.get('use_acc_count_filter', False):
        window = params.get('acc_count_window', 21)
        col_name = f'AccCount_{window}'
        if col_name in df.columns:
            acc_thresh = params.get('acc_count_thresh', 0)
            acc_logic = params.get('acc_count_logic', '=')
            if acc_logic == '=': mask &= (df[col_name] == acc_thresh)
            elif acc_logic == '>': mask &= (df[col_name] > acc_thresh)
            elif acc_logic == '<': mask &= (df[col_name] < acc_thresh)

    if params.get('use_dist_count_filter', False):
        window = params.get('dist_count_window', 21)
        col_name = f'DistCount_{window}'
        if col_name in df.columns:
            dist_thresh = params.get('dist_count_thresh', 0)
            dist_logic = params.get('dist_count_logic', '>')
            if dist_logic == '=': mask &= (df[col_name] == dist_thresh)
            elif dist_logic == '>': mask &= (df[col_name] > dist_thresh)
            elif dist_logic == '<': mask &= (df[col_name] < dist_thresh)

    if params.get('use_dist_filter', False):
        ma_type = params.get('dist_ma_type', 'SMA 200')
        ma_col = ma_type.replace(" ", "") 
        if ma_col in df.columns:
            atr = df['ATR'].replace(0, np.nan)
            dist_units = (df['Close'] - df[ma_col]) / atr
            d_logic = params.get('dist_logic', 'Between')
            d_min = params.get('dist_min', 0)
            d_max = params.get('dist_max', 0)
            if d_logic == "Greater Than (>)": mask &= (dist_units > d_min)
            elif d_logic == "Less Than (<)": mask &= (dist_units < d_max)
            elif d_logic == "Between": mask &= (dist_units >= d_min) & (dist_units <= d_max)

    if params['use_sznl']:
        if params['sznl_logic'] == '<': raw_sznl = df['Sznl'] < params['sznl_thresh']
        else: raw_sznl = df['Sznl'] > params['sznl_thresh']
        sznl_cond = raw_sznl
        if params.get('sznl_first_instance', False):
            lookback = params.get('sznl_lookback', 21)
            prev = sznl_cond.shift(1).rolling(lookback).sum()
            sznl_cond &= (prev == 0)
        mask &= sznl_cond

    if params.get('use_market_sznl', False):
        mkt_ranks = df['Mkt_Sznl_Ref']
        if params['market_sznl_logic'] == '<': mask &= (mkt_ranks < params['market_sznl_thresh'])
        else: mask &= (mkt_ranks > params['market_sznl_thresh'])

    if params['use_52w']:
        if params['52w_type'] == 'New 52w High': cond_52 = df['is_52w_high']
        else: cond_52 = df['is_52w_low']
        if params.get('52w_first_instance', True):
            lookback = params.get('52w_lookback', 21)
            prev = cond_52.shift(1).rolling(lookback).sum()
            cond_52 &= (prev == 0)
        mask &= cond_52

    if params.get('exclude_52w_high', False):
        mask &= (~df['is_52w_high'])

    if params['use_vol']:
        mask &= (df['vol_ratio'] > params['vol_thresh'])

    if params.get('use_vol_rank'):
        val = df['vol_ratio_10d_rank']
        if params['vol_rank_logic'] == '<': mask &= (val < params['vol_rank_thresh'])
        else: mask &= (val > params['vol_rank_thresh'])

    return mask.fillna(False)

def calculate_trade_result(df, signal_date, action, shares, entry_price, hold_days):
    start_idx = df.index.searchsorted(signal_date)
    if start_idx >= len(df) - 1:
        return 0, signal_date
    window = df.iloc[start_idx+1 : start_idx+1+hold_days].copy()
    if window.empty: 
        return 0, signal_date
    exit_row = window.iloc[-1]
    exit_price = exit_row['Close']
    exit_date = exit_row.name
    if action == "BUY": pnl = (exit_price - entry_price) * shares
    elif action == "SELL SHORT": pnl = (entry_price - exit_price) * shares
    else: pnl = 0
    return pnl, exit_date

# -----------------------------------------------------------------------------
# CORE LOGIC: DAILY MARK-TO-MARKET PNL
# -----------------------------------------------------------------------------
def get_daily_mtm_series(sig_df, master_dict):
    """
    Returns a Series with index = Date and value = Total Daily PnL ($)
    for the trades in sig_df, calculated Mark-to-Market.
    """
    if sig_df.empty: return pd.Series(dtype=float)
    
    min_date = sig_df['Date'].min()
    max_date = max(sig_df['Exit Date'].max(), pd.Timestamp.today())
    all_dates = pd.date_range(start=min_date, end=max_date, freq='B') 
    
    daily_pnl = pd.Series(0.0, index=all_dates)
    
    for idx, trade in sig_df.iterrows():
        ticker = trade['Ticker'].replace('.', '-')
        action = trade['Action']
        shares = trade['Shares']
        entry_date = trade['Date']
        exit_date = trade['Exit Date']
        
        t_df = master_dict.get(ticker)
        if t_df is None or t_df.empty:
            daily_pnl[exit_date] += trade['PnL'] # Fallback
            continue
            
        trade_dates = all_dates[(all_dates > entry_date) & (all_dates <= exit_date)]
        relevant_prices = t_df['Close'].reindex(pd.concat([pd.Series([entry_date]), pd.Series(trade_dates)])).ffill()
        
        if len(relevant_prices) < 2: continue
        price_diffs = relevant_prices.diff().dropna()
        if action == "SELL SHORT": price_diffs = -price_diffs
        
        daily_dollar_pnl = price_diffs * shares
        
        # Add to master series
        for d, val in daily_dollar_pnl.items():
            if d in daily_pnl.index:
                daily_pnl[d] += val
                
    return daily_pnl

def calculate_mark_to_market_curve(sig_df, master_dict):
    """Calculates cumulative equity curve using the helper above."""
    daily_pnl = get_daily_mtm_series(sig_df, master_dict)
    if daily_pnl.empty: return pd.DataFrame(columns=['Equity'])
    return daily_pnl.cumsum().to_frame(name='Equity')

def calculate_daily_exposure(sig_df):
    if sig_df.empty: return pd.DataFrame()
    min_date = sig_df['Date'].min()
    max_date = sig_df['Exit Date'].max()
    all_dates = pd.date_range(start=min_date, end=max_date)
    exposure_df = pd.DataFrame(0.0, index=all_dates, columns=['Long Exposure ($)', 'Short Exposure ($)'])
    for idx, row in sig_df.iterrows():
        trade_dates = pd.date_range(start=row['Date'], end=row['Exit Date'])
        dollar_val = row['Price'] * row['Shares']
        if row['Action'] == 'BUY':
            exposure_df.loc[exposure_df.index.isin(trade_dates), 'Long Exposure ($)'] += dollar_val
        elif row['Action'] == 'SELL SHORT':
            exposure_df.loc[exposure_df.index.isin(trade_dates), 'Short Exposure ($)'] += dollar_val
    exposure_df['Net Exposure ($)'] = exposure_df['Long Exposure ($)'] - exposure_df['Short Exposure ($)']
    return exposure_df

def calculate_performance_stats(sig_df, master_dict):
    """
    Calculates stats. 
    SQN / Profit Factor = Trade-based (Realized).
    Sharpe Ratio = Time-based (Mark-to-Market).
    """
    stats = []
    
    def get_metrics(df, name, master_dict):
        if df.empty: return None
        
        # 1. Trade-Based Stats (Realized)
        count = len(df)
        total_pnl = df['PnL'].sum()
        gross_profit = df[df['PnL'] > 0]['PnL'].sum()
        gross_loss = abs(df[df['PnL'] < 0]['PnL'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan
        avg_pnl = df['PnL'].mean()
        std_pnl = df['PnL'].std()
        sqn = (avg_pnl / std_pnl * np.sqrt(count)) if std_pnl != 0 else 0
        
        # 2. Time-Based Stats (Mark-to-Market Sharpe)
        # Generate the daily MtM PnL series for this specific subset of trades
        daily_mtm_pnl = get_daily_mtm_series(df, master_dict)
        
        if not daily_mtm_pnl.empty:
            daily_mean = daily_mtm_pnl.mean()
            daily_std = daily_mtm_pnl.std()
            sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std != 0 else 0
        else:
            sharpe = 0

        return {
            "Strategy": name,
            "Trades": count,
            "Total PnL": total_pnl,
            "Sharpe Ratio (MtM)": sharpe,
            "Profit Factor": profit_factor,
            "SQN": sqn
        }
        
    strategies = sig_df['Strategy'].unique()
    for strat in strategies:
        strat_df = sig_df[sig_df['Strategy'] == strat]
        m = get_metrics(strat_df, strat, master_dict)
        if m: stats.append(m)
        
    total_m = get_metrics(sig_df, "TOTAL PORTFOLIO", master_dict)
    if total_m: stats.append(total_m)
    return pd.DataFrame(stats)

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="Strategy Backtest Lab")
    st.sidebar.header("⚙️ Backtest Settings")

    if st.sidebar.button("🔴 Force Clear Cache & Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        if 'backtest_data' in st.session_state:
            del st.session_state['backtest_data']
        st.rerun()
    
    current_year = datetime.date.today().year
    selected_year = st.sidebar.slider("Select Start Year", 2000, current_year, current_year - 2)
    default_date = datetime.date(selected_year, 1, 1)
    
    with st.sidebar.form("backtest_form"):
        user_start_date = st.date_input("Backtest Start Date", value=default_date, min_value=datetime.date(2000, 1, 1))
        st.caption(f"Data download buffer: 365 days prior to {user_start_date}.")
        run_btn = st.form_submit_button("⚡ Run Backtest")

    st.title("⚡ Strategy Backtest Lab")
    st.markdown(f"**Selected Start Date:** {user_start_date}")
    st.markdown("---")

    if run_btn:
        sznl_map = load_seasonal_map()
        if 'backtest_data' not in st.session_state:
            st.session_state['backtest_data'] = {}

        long_term_tickers = set()
        for strat in STRATEGY_BOOK:
            for t in strat['universe_tickers']:
                long_term_tickers.add(t)
            s = strat['settings']
            if s.get('use_market_sznl'): long_term_tickers.add(s.get('market_ticker', '^GSPC'))
            if "Market" in s.get('trend_filter', ''): long_term_tickers.add(s.get('market_ticker', 'SPY'))
            if "SPY" in s.get('trend_filter', ''): long_term_tickers.add("SPY")

        long_term_list = [t.replace('.', '-') for t in long_term_tickers]
        existing_keys = set(st.session_state['backtest_data'].keys())
        missing_long = list(set(long_term_list) - existing_keys)
        
        if missing_long:
            st.write(f"📥 Downloading **Deep History (from 2000)** for {len(missing_long)} tickers...")
            data_long = download_historical_data(missing_long, start_date="2000-01-01")
            st.session_state['backtest_data'].update(data_long)
            st.success("✅ Download Batch Complete.")

        master_dict = st.session_state['backtest_data']
        downloaded_keys = set(master_dict.keys())
        requested_set = set(long_term_list)
        failed_tickers = requested_set - downloaded_keys
        
        if failed_tickers:
            st.error(f"⚠️ {len(failed_tickers)} Tickers failed to download.")
            with st.expander("View Failed Tickers"):
                st.write(list(failed_tickers))

        all_signals = []
        progress_bar = st.progress(0)
        
        for i, strat in enumerate(STRATEGY_BOOK):
            progress_bar.progress((i + 1) / len(STRATEGY_BOOK))
            
            strat_mkt_ticker = strat['settings'].get('market_ticker', 'SPY')
            mkt_df = master_dict.get(strat_mkt_ticker)
            if mkt_df is None: mkt_df = master_dict.get('SPY')
            
            market_series = None
            if mkt_df is not None:
                temp_mkt = mkt_df.copy()
                temp_mkt['SMA200'] = temp_mkt['Close'].rolling(200).mean()
                market_series = temp_mkt['Close'] > temp_mkt['SMA200']

            strat_universe_clean = [t.replace('.', '-') for t in strat['universe_tickers']]
            missing_in_strat = [t for t in strat_universe_clean if t not in master_dict]
            if missing_in_strat: st.warning(f"Strategy **'{strat['name']}'** skipping {len(missing_in_strat)} tickers.")

            for ticker in strat['universe_tickers']:
                t_clean = ticker.replace('.', '-')
                df = master_dict.get(t_clean)
                if df is None or len(df) < 200: continue
                
                try:
                    df = calculate_indicators(df, sznl_map, t_clean, market_series)
                    mask = get_historical_mask(df, strat['settings'], sznl_map)
                    cutoff_ts = pd.Timestamp(user_start_date)
                    mask = mask[mask.index >= cutoff_ts]
                    if not mask.any(): continue

                    true_dates = mask[mask].index
                    last_exit_date = None
                    
                    for d in true_dates:
                        if last_exit_date is not None and d <= last_exit_date: continue
                        row = df.loc[d]
                        atr = row['ATR']
                        risk = strat['execution']['risk_per_trade']
                        
                        if strat['name'] == "Overbot Vol Spike":
                            vol_ratio = row.get('vol_ratio', 0)
                            if vol_ratio > 2.0: risk = 675
                            elif vol_ratio > 1.5: risk = 525
                        if strat['name'] == "Weak Close Decent Sznls":
                            sznl_val = row.get('Sznl', 0)
                            if sznl_val >= 65: risk = risk * 1.5
                            elif sznl_val >= 50: risk = risk * 1.0
                            elif sznl_val >= 33: risk = risk * 0.66
                        
                        entry_type = strat['settings'].get('entry_type', 'Signal Close')
                        entry_idx = df.index.get_loc(d)
                        if entry_idx + 1 >= len(df): continue 
                        entry_row = df.iloc[entry_idx + 1]

                        if entry_type == 'T+1 Close':
                            entry = entry_row['Close']; entry_date = entry_row.name 
                        elif entry_type == 'T+1 Open':
                            entry = entry_row['Open']; entry_date = entry_row.name 
                        elif entry_type == "Limit (Open +/- 0.5 ATR)":
                            limit_offset = 0.5 * atr
                            if strat['settings']['trade_direction'] == 'Short':
                                limit_price = entry_row['Open'] + limit_offset
                                if entry_row['High'] < limit_price: continue 
                                entry = limit_price
                            else:
                                limit_price = entry_row['Open'] - limit_offset
                                if entry_row['Low'] > limit_price: continue
                                entry = limit_price
                            entry_date = entry_row.name
                        else: 
                            entry = row['Close']; entry_date = d 
                        
                        direction = strat['settings'].get('trade_direction', 'Long')
                        if direction == 'Long':
                            stop_price = entry - (atr * strat['execution']['stop_atr'])
                            dist = entry - stop_price
                            action = "BUY"
                        else:
                            stop_price = entry + (atr * strat['execution']['stop_atr'])
                            dist = stop_price - entry
                            action = "SELL SHORT"
                        
                        shares = int(risk / dist) if dist > 0 else 0
                        pnl, exit_date = calculate_trade_result(df, entry_date, action, shares, entry, strat['execution']['hold_days'])

                        try:
                            e_idx = df.index.get_loc(entry_date)
                            ts_idx = e_idx + strat['execution']['hold_days']
                            time_stop_date = df.index[ts_idx] if ts_idx < len(df) else entry_date + BusinessDay(strat['execution']['hold_days'])
                        except: time_stop_date = pd.NaT

                        all_signals.append({
                            "Date": d.date(), "Exit Date": exit_date.date(), "Time Stop": time_stop_date, 
                            "Strategy": strat['name'], "Ticker": ticker, "Action": action,
                            "Entry Criteria": entry_type, "Price": entry, "Shares": shares,
                            "PnL": pnl, "ATR": atr, "Range %": row['RangePct'] * 100
                        })
                        last_exit_date = exit_date
                except Exception: continue
        
        progress_bar.empty()
        
        if all_signals:
            sig_df = pd.DataFrame(all_signals)
            sig_df['Date'] = pd.to_datetime(sig_df['Date'])
            sig_df['Exit Date'] = pd.to_datetime(sig_df['Exit Date'])
            sig_df['Time Stop'] = pd.to_datetime(sig_df['Time Stop'])
            sig_df = sig_df.sort_values(by="Exit Date")

            today = pd.Timestamp(datetime.date.today())
            open_mask = sig_df['Time Stop'] >= today
            open_df = sig_df[open_mask].copy()

            if not open_df.empty:
                current_prices, open_pnls, current_values = [], [], []
                for idx, row in open_df.iterrows():
                    ticker = row['Ticker']
                    t_df = master_dict.get(ticker.replace('.', '-'))
                    last_close = t_df.iloc[-1]['Close'] if t_df is not None and not t_df.empty else row['Price']
                    if row['Action'] == 'BUY':
                        pnl = (last_close - row['Price']) * row['Shares']
                        val = last_close * row['Shares']
                    else:
                        pnl = (row['Price'] - last_close) * row['Shares']
                        val = last_close * row['Shares']
                    current_prices.append(last_close); open_pnls.append(pnl); current_values.append(val)

                open_df['Current Price'] = current_prices; open_df['Open PnL'] = open_pnls; open_df['Mkt Value'] = current_values
                total_long = open_df[open_df['Action'] == 'BUY']['Mkt Value'].sum()
                total_short = open_df[open_df['Action'] == 'SELL SHORT']['Mkt Value'].sum()
                net_exposure = total_long - total_short
                total_open_pnl = open_df['Open PnL'].sum()

                st.divider()
                st.subheader("💼 Current Exposure (Active Positions)")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("# Positions", len(open_df))
                m2.metric("Total Long", f"${total_long:,.0f}")
                m3.metric("Total Short", f"${total_short:,.0f}")
                m4.metric("Net Exposure", f"${net_exposure:,.0f}")
                m5.metric("Total Open PnL", f"${total_open_pnl:,.2f}", delta_color="normal", delta=f"{total_open_pnl:,.2f}")
                st.dataframe(open_df.style.format({"Date": "{:%Y-%m-%d}", "Time Stop": "{:%Y-%m-%d}", "Price": "${:.2f}", "Current Price": "${:.2f}", "Open PnL": "${:,.2f}", "Range %": "{:.1f}%"}), use_container_width=True)
            else: st.info("No active positions (Time Stop >= Today).")

            st.divider()
            st.subheader("📊 Strategy Performance Metrics")
            # --- UPDATED FUNCTION CALL ---
            stats_df = calculate_performance_stats(sig_df, master_dict)
            st.dataframe(stats_df.style.format({"Total PnL": "${:,.2f}", "Sharpe Ratio (MtM)": "{:.2f}", "Profit Factor": "{:.2f}", "SQN": "{:.2f}"}), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Total Portfolio PnL (Mark-to-Market)")
                df_eq = calculate_mark_to_market_curve(sig_df, master_dict)
                if not df_eq.empty:
                    df_eq['SMA20'] = df_eq['Equity'].rolling(window=20).mean()
                    df_eq['StdDev'] = df_eq['Equity'].rolling(window=20).std()
                    df_eq['Upper'] = df_eq['SMA20'] + (2 * df_eq['StdDev'])
                    df_eq['Lower'] = df_eq['SMA20'] - (2 * df_eq['StdDev'])
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq['Upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                    fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq['Lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 255, 255, 0.1)', name='Bollinger Band (20, 2)', hoverinfo='skip'))
                    fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq['SMA20'], mode='lines', name='SMA 20', line=dict(color='orange', width=1, dash='dot')))
                    fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq['Equity'], mode='lines', name='Total PnL', line=dict(color='#00FF00', width=2)))
                    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified", xaxis_title="Date", yaxis_title="Cumulative PnL ($)")
                    st.plotly_chart(fig, use_container_width=True)
                else: st.write("No trades to plot.")
            with col2:
                st.subheader("📉 Cumulative Realized PnL by Strategy")
                strat_pnl = sig_df.pivot_table(index='Exit Date', columns='Strategy', values='PnL', aggfunc='sum').fillna(0)
                st.line_chart(strat_pnl.cumsum())

            st.subheader("⚖️ Portfolio Exposure Over Time")
            exposure_df = calculate_daily_exposure(sig_df)
            if not exposure_df.empty: st.line_chart(exposure_df)

            st.subheader("📜 Historical Signal Log")
            st.dataframe(sig_df.sort_values(by="Date", ascending=False).style.format({"Price": "${:.2f}", "PnL": "${:.2f}", "Date": "{:%Y-%m-%d}", "Exit Date": "{:%Y-%m-%d}", "Time Stop": "{:%Y-%m-%d}", "Range %": "{:.1f}%"}), use_container_width=True, height=400)
        else: st.warning(f"No signals found in the backtest period starting from {user_start_date}.")
    else: st.info("👈 Please select a start year/date and click 'Run Backtest' in the sidebar to begin.")

if __name__ == "__main__":
    main()
