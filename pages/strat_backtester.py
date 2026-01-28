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
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# IMPORT STRATEGY BOOK FROM ROOT
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from strategy_config import _STRATEGY_BOOK_RAW, ACCOUNT_VALUE
except ImportError:
    st.error("Could not find strategy_config.py in the root directory.")
    _STRATEGY_BOOK_RAW = []
    ACCOUNT_VALUE = 150000

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


def calculate_indicators(df, sznl_map, ticker, market_series=None, vix_series=None, gap_window=21, custom_sma_lengths=None, acc_window=None, dist_window=None):
    df = df.copy()
    df.sort_index(inplace=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    
    # Pre-extract numpy arrays for speed
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    open_prices = df['Open'].values
    volume = df['Volume'].values
    
    # Vectorized SMA calculations
    close_series = df['Close']
    for window in [10, 20, 50, 100, 200]:
        df[f'SMA{window}'] = close_series.rolling(window).mean()
    
    if custom_sma_lengths:
        for length in custom_sma_lengths:
            col_name = f"SMA{length}"
            if col_name not in df.columns:
                df[col_name] = close_series.rolling(length).mean()

    df['EMA8'] = close_series.ewm(span=8, adjust=False).mean()
    df['EMA11'] = close_series.ewm(span=11, adjust=False).mean()
    df['EMA21'] = close_series.ewm(span=21, adjust=False).mean()
    
    for window in [5, 10, 21]:
        df[f'ret_{window}d'] = close_series.pct_change(window)
        df[f'rank_ret_{window}d'] = df[f'ret_{window}d'].expanding(min_periods=252).rank(pct=True) * 100.0
    
    # ATR calculation - vectorized
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    true_range = np.maximum(np.maximum(high_low, high_close), low_close)
    df['ATR'] = pd.Series(true_range, index=df.index).rolling(14).mean()
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
    
    df['Sznl'] = get_sznl_val_series(ticker, df.index, sznl_map)
    df['Mkt_Sznl_Ref'] = get_sznl_val_series("^GSPC", df.index, sznl_map)
    
    rolling_high = df['High'].shift(1).rolling(252).max()
    rolling_low = df['Low'].shift(1).rolling(252).min()
    df['is_52w_high'] = df['High'] > rolling_high
    df['is_52w_low'] = df['Low'] < rolling_low
    
    vol_ma = df['Volume'].rolling(63).mean()
    df['vol_ma'] = vol_ma
    df['vol_ratio'] = df['Volume'] / vol_ma
    
    vol_ma_10 = df['Volume'].rolling(10).mean()
    df['vol_ratio_10d'] = vol_ma_10 / vol_ma
    df['vol_ratio_10d_rank'] = df['vol_ratio_10d'].expanding(min_periods=252).rank(pct=True) * 100.0
    
    # Vectorized accumulation/distribution day calculation
    vol_gt_prev = volume > np.roll(volume, 1)
    vol_gt_ma = df['Volume'].values > vol_ma.values
    is_green = close > open_prices
    is_red = close < open_prices

    df['is_acc_day'] = (is_green & vol_gt_prev & vol_gt_ma).astype(int)
    df['is_dist_day'] = (is_red & vol_gt_prev & vol_gt_ma).astype(int)
    
    if acc_window:
        df[f'AccCount_{acc_window}'] = df['is_acc_day'].rolling(acc_window).sum()
    if dist_window:
        df[f'DistCount_{dist_window}'] = df['is_dist_day'].rolling(dist_window).sum()

    if not df.empty:
        start_ts = df.index[0]
        df['age_years'] = (df.index - start_ts).days / 365.25
    else:
        df['age_years'] = 0.0
        
    if market_series is not None:
        df['Market_Above_SMA200'] = market_series.reindex(df.index, method='ffill').fillna(False)

    if vix_series is not None:
        df['VIX_Value'] = vix_series.reindex(df.index, method='ffill').fillna(0)
    else:
        df['VIX_Value'] = 0.0

    # Vectorized RangePct
    denom = high - low
    df['RangePct'] = np.where(denom == 0, 0.5, (close - low) / denom)

    df['DayOfWeekVal'] = df.index.dayofweek
    is_open_gap = (df['Low'] > df['High'].shift(1)).astype(int)
    df['GapCount'] = is_open_gap.rolling(gap_window).sum()

    piv_len = 20 
    roll_max = df['High'].rolling(window=piv_len*2+1, center=True).max()
    df['is_pivot_high'] = (df['High'] == roll_max)
    roll_min = df['Low'].rolling(window=piv_len*2+1, center=True).min()
    df['is_pivot_low'] = (df['Low'] == roll_min)

    df['LastPivotHigh'] = np.where(df['is_pivot_high'], df['High'], np.nan)
    df['LastPivotHigh'] = df['LastPivotHigh'].shift(piv_len).ffill()
    df['LastPivotLow'] = np.where(df['is_pivot_low'], df['Low'], np.nan)
    df['LastPivotLow'] = df['LastPivotLow'].shift(piv_len).ffill()
    
    df['PrevHigh'] = df['High'].shift(1)
    df['PrevLow'] = df['Low'].shift(1)

    return df


def get_historical_mask(df, params, sznl_map, ticker_name="UNK"):
    """Optimized mask generation - builds conditions list and combines once."""
    n = len(df)
    conditions = []
    
    # Trend filter
    trend_opt = params.get('trend_filter', 'None')
    if trend_opt == "Price > 200 SMA":
        conditions.append(df['Close'].values > df['SMA200'].values)
    elif trend_opt == "Price > Rising 200 SMA":
        sma200 = df['SMA200'].values
        conditions.append((df['Close'].values > sma200) & (sma200 > np.roll(sma200, 1)))
    elif trend_opt == "Not Below Declining 200 SMA":
        sma200 = df['SMA200'].values
        conditions.append(~((df['Close'].values < sma200) & (sma200 < np.roll(sma200, 1))))
    elif trend_opt == "Price < 200 SMA":
        conditions.append(df['Close'].values < df['SMA200'].values)
    elif trend_opt == "Price < Falling 200 SMA":
        sma200 = df['SMA200'].values
        conditions.append((df['Close'].values < sma200) & (sma200 < np.roll(sma200, 1)))
    elif "Market" in trend_opt and 'Market_Above_SMA200' in df.columns:
        if trend_opt == "Market > 200 SMA":
            conditions.append(df['Market_Above_SMA200'].values)
        elif trend_opt == "Market < 200 SMA":
            conditions.append(~df['Market_Above_SMA200'].values)

    # Price/volume/age filters
    conditions.append(df['Close'].values >= params.get('min_price', 0))
    conditions.append(df['vol_ma'].values >= params.get('min_vol', 0))
    conditions.append(df['age_years'].values >= params.get('min_age', 0))
    conditions.append(df['age_years'].values <= params.get('max_age', 100))
    
    if 'ATR_Pct' in df.columns:
        atr_pct = df['ATR_Pct'].values
        conditions.append((atr_pct >= params.get('min_atr_pct', 0)) & (atr_pct <= params.get('max_atr_pct', 1000)))
        
    if params.get('require_close_gt_open', False):
        conditions.append(df['Close'].values > df['Open'].values)

    # Breakout mode
    bk_mode = params.get('breakout_mode', 'None')
    if bk_mode == "Close > Prev Day High":
        conditions.append(df['Close'].values > df['PrevHigh'].values)
    elif bk_mode == "Close < Prev Day Low":
        conditions.append(df['Close'].values < df['PrevLow'].values)

    # Range filter
    if params.get('use_range_filter', False):
        rn_val = df['RangePct'].values * 100
        conditions.append((rn_val >= params.get('range_min', 0)) & (rn_val <= params.get('range_max', 100)))

    # Day of week filter
    if params.get('use_dow_filter', False):
        allowed = params.get('allowed_days', [])
        if allowed:
            conditions.append(np.isin(df['DayOfWeekVal'].values, allowed))

    # Presidential cycle filter
    if 'allowed_cycles' in params:
        allowed = params['allowed_cycles']
        if allowed and len(allowed) < 4:
            year_rems = df.index.year % 4
            conditions.append(np.isin(year_rems, allowed))

    # Gap filter
    if params.get('use_gap_filter', False):
        g_val = df['GapCount'].values
        g_thresh = params.get('gap_thresh', 0)
        g_logic = params.get('gap_logic', '>')
        if g_logic == ">":
            conditions.append(g_val > g_thresh)
        elif g_logic == "<":
            conditions.append(g_val < g_thresh)
        elif g_logic == "=":
            conditions.append(g_val == g_thresh)

    # Accumulation count filter
    if params.get('use_acc_count_filter', False):
        col = f"AccCount_{params['acc_count_window']}"
        if col in df.columns:
            col_vals = df[col].values
            if params['acc_count_logic'] == ">":
                conditions.append(col_vals > params['acc_count_thresh'])
            elif params['acc_count_logic'] == "<":
                conditions.append(col_vals < params['acc_count_thresh'])
            elif params['acc_count_logic'] == "=":
                conditions.append(col_vals == params['acc_count_thresh'])

    # Distribution count filter
    if params.get('use_dist_count_filter', False):
        col = f"DistCount_{params['dist_count_window']}"
        if col in df.columns:
            col_vals = df[col].values
            if params['dist_count_logic'] == ">":
                conditions.append(col_vals > params['dist_count_thresh'])
            elif params['dist_count_logic'] == "<":
                conditions.append(col_vals < params['dist_count_thresh'])
            elif params['dist_count_logic'] == "=":
                conditions.append(col_vals == params['dist_count_thresh'])

    # Performance filters (legacy list-based)
    for pf in params.get('perf_filters', []):
        col = f"rank_ret_{pf['window']}d"
        col_vals = df[col].values
        if pf['logic'] == '<':
            cond = col_vals < pf['thresh']
        else:
            cond = col_vals > pf['thresh']
        if pf['consecutive'] > 1:
            cond = pd.Series(cond).rolling(pf['consecutive']).sum().values == pf['consecutive']
        conditions.append(cond)

    # Performance rank filter
    if params.get('use_perf_rank', False):
        col = f"rank_ret_{params['perf_window']}d"
        col_vals = df[col].values
        if params['perf_logic'] == '<':
            raw = col_vals < params['perf_thresh']
        else:
            raw = col_vals > params['perf_thresh']
        
        consec = params.get('perf_consecutive', 1)
        if consec > 1:
            persist = pd.Series(raw).rolling(consec).sum().values == consec
        else:
            persist = raw
        
        if params.get('perf_first_instance', False):
            lookback = params.get('perf_lookback', 21)
            persist_series = pd.Series(persist, index=df.index)
            prev_inst = persist_series.shift(1).rolling(lookback).sum()
            persist = persist & (prev_inst.values == 0)
        
        conditions.append(persist)

    # MA consecutive filters
    for f in params.get('ma_consec_filters', []):
        col = f"SMA{f['length']}"
        if col in df.columns:
            if f['logic'] == 'Above':
                cond = df['Close'].values > df[col].values
            else:
                cond = df['Close'].values < df[col].values
            if f['consec'] > 1:
                cond = pd.Series(cond).rolling(f['consec']).sum().values == f['consec']
            conditions.append(cond)

    # Seasonal filter
    if params.get('use_sznl', False):
        sznl_vals = df['Sznl'].values
        if params['sznl_logic'] == '<':
            cond_s = sznl_vals < params['sznl_thresh']
        else:
            cond_s = sznl_vals > params['sznl_thresh']
        
        if params.get('sznl_first_instance', False):
            lookback = params.get('sznl_lookback', 21)
            cond_series = pd.Series(cond_s, index=df.index)
            prev = cond_series.shift(1).rolling(lookback).max().fillna(0)
            cond_s = cond_s & (prev.values == 0)
        conditions.append(cond_s)

    # Market seasonal filter
    if params.get('use_market_sznl', False):
        mkt_sznl = df['Mkt_Sznl_Ref'].values
        if params['market_sznl_logic'] == '<':
            conditions.append(mkt_sznl < params['market_sznl_thresh'])
        else:
            conditions.append(mkt_sznl > params['market_sznl_thresh'])

    # 52-week high/low filter
    if params.get('use_52w', False):
        if params['52w_type'] == 'New 52w High':
            cond_52 = df['is_52w_high'].values.copy()
        else:
            cond_52 = df['is_52w_low'].values.copy()
        
        if params.get('52w_lag', 0) > 0:
            cond_52 = np.roll(cond_52, params['52w_lag'])
            cond_52[:params['52w_lag']] = False

        if params.get('52w_first_instance', False):
            lookback = params.get('52w_lookback', 21)
            cond_series = pd.Series(cond_52, index=df.index)
            prev = cond_series.shift(1).rolling(lookback).max().fillna(0)
            cond_52 = cond_52 & (prev.values == 0)
        
        conditions.append(cond_52)

    # Exclude 52w high
    if params.get('exclude_52w_high', False):
        conditions.append(~df['is_52w_high'].values)

    # Volume filters
    if params.get('vol_gt_prev', False):
        conditions.append(df['Volume'].values > np.roll(df['Volume'].values, 1))
    
    if params.get('use_vol', False):
        conditions.append(df['vol_ratio'].values > params['vol_thresh'])

    if params.get('use_vol_rank', False):
        col_vals = df['vol_ratio_10d_rank'].values
        if params['vol_rank_logic'] == ">":
            conditions.append(col_vals > params['vol_rank_thresh'])
        elif params['vol_rank_logic'] == "<":
            conditions.append(col_vals < params['vol_rank_thresh'])

    # MA distance filter
    if params.get('use_ma_dist_filter', False):
        ma_col_map = {"SMA 10": "SMA10", "SMA 20": "SMA20", "SMA 50": "SMA50", "SMA 100": "SMA100", "SMA 200": "SMA200", 
                      "EMA 8": "EMA8", "EMA 11": "EMA11", "EMA 21": "EMA21"}
        ma_target = ma_col_map.get(params['dist_ma_type'])
        if ma_target and ma_target in df.columns:
            dist_val = (df['Close'].values - df[ma_target].values) / df['ATR'].values
            if params['dist_logic'] == "Greater Than (>)":
                conditions.append(dist_val > params['dist_min'])
            elif params['dist_logic'] == "Less Than (<)":
                conditions.append(dist_val < params['dist_max'])
            elif params['dist_logic'] == "Between":
                conditions.append((dist_val >= params['dist_min']) & (dist_val <= params['dist_max']))

    # VIX filter
    if params.get('use_vix_filter', False) and 'VIX_Value' in df.columns:
        vix_vals = df['VIX_Value'].values
        conditions.append((vix_vals >= params.get('vix_min', 0)) & (vix_vals <= params.get('vix_max', 100)))

    # MA touch filter
    if params.get('use_ma_touch', False):
        ma_col_map = {"SMA 10": "SMA10", "SMA 20": "SMA20", "SMA 50": "SMA50", "SMA 100": "SMA100", "SMA 200": "SMA200", 
                      "EMA 8": "EMA8", "EMA 11": "EMA11", "EMA 21": "EMA21"}
        ma_target = ma_col_map.get(params.get('ma_touch_type'))
        
        if ma_target and ma_target in df.columns:
            ma_series = df[ma_target]
            ma_vals = ma_series.values
            direction = params.get('trade_direction', 'Long')
            slope_lookback = params.get('ma_slope_days', 20)
            untested_lookback = params.get('ma_untested_days', 50)

            if direction == 'Long':
                is_slope_ok = ma_vals > np.roll(ma_vals, 1)
            else:
                is_slope_ok = ma_vals < np.roll(ma_vals, 1)
            
            if slope_lookback > 1:
                is_slope_ok = pd.Series(is_slope_ok).rolling(slope_lookback).sum().values == slope_lookback
            
            if untested_lookback > 0:
                if direction == 'Long':
                    was_untested = (df['Low'] > ma_series).shift(1).rolling(untested_lookback).min().values == 1.0
                else:
                    was_untested = (df['High'] < ma_series).shift(1).rolling(untested_lookback).min().values == 1.0
                was_untested = np.nan_to_num(was_untested, nan=False).astype(bool)
            else:
                was_untested = np.ones(n, dtype=bool)

            if direction == 'Long':
                touched_today = df['Low'].values <= ma_vals
            else:
                touched_today = df['High'].values >= ma_vals

            conditions.append(is_slope_ok & was_untested & touched_today)

    # Combine all conditions
    if conditions:
        # Handle NaN values in conditions
        combined = np.ones(n, dtype=bool)
        for cond in conditions:
            cond_arr = np.asarray(cond)
            # Replace NaN with False
            cond_arr = np.where(np.isnan(cond_arr.astype(float)), False, cond_arr)
            combined = combined & cond_arr.astype(bool)
        return pd.Series(combined, index=df.index)
    else:
        return pd.Series(True, index=df.index)


@st.cache_data(show_spinner=False)
def precompute_all_indicators(_master_dict, _strategies, _sznl_map, _vix_series):
    processed = {}
    
    spy_df = _master_dict.get('SPY')
    market_series = None
    if spy_df is not None:
        temp = spy_df.copy()
        if isinstance(temp.columns, pd.MultiIndex):
            temp.columns = [c[0] if isinstance(c, tuple) else c for c in temp.columns]
        temp.columns = [c.capitalize() for c in temp.columns]
        temp['SMA200'] = temp['Close'].rolling(200).mean()
        market_series = temp['Close'] > temp['SMA200']
    
    ticker_params = {}
    for strat in _strategies:
        settings = strat['settings']
        gap_win = settings.get('gap_lookback', 21)
        acc_win = settings.get('acc_count_window') if settings.get('use_acc_count_filter') else None
        dist_win = settings.get('dist_count_window') if settings.get('use_dist_count_filter') else None
        req_custom_mas = list(set([f['length'] for f in settings.get('ma_consec_filters', [])]))
        
        for ticker in strat['universe_tickers']:
            t_clean = ticker.replace('.', '-')
            if t_clean not in ticker_params:
                ticker_params[t_clean] = {'gap': gap_win, 'acc': acc_win, 'dist': dist_win, 'mas': set(req_custom_mas)}
            else:
                ticker_params[t_clean]['gap'] = max(ticker_params[t_clean]['gap'], gap_win)
                if acc_win: ticker_params[t_clean]['acc'] = acc_win
                if dist_win: ticker_params[t_clean]['dist'] = dist_win
                ticker_params[t_clean]['mas'].update(req_custom_mas)
    
    # Process tickers (can be parallelized if needed)
    for t_clean, params in ticker_params.items():
        df = _master_dict.get(t_clean)
        if df is None or len(df) < 200:
            continue
        try:
            processed[t_clean] = calculate_indicators(
                df, _sznl_map, t_clean, market_series, _vix_series,
                gap_window=params['gap'],
                acc_window=params['acc'],
                dist_window=params['dist'],
                custom_sma_lengths=list(params['mas'])
            )
        except Exception:
            continue
    
    return processed


def generate_candidates_fast(processed_dict, strategies, sznl_map, user_start_date):
    candidates = []
    signal_data = {}
    cutoff_ts = pd.Timestamp(user_start_date)
    
    for strat_idx, strat in enumerate(strategies):
        settings = strat['settings']
        
        for ticker in strat['universe_tickers']:
            t_clean = ticker.replace('.', '-')
            df = processed_dict.get(t_clean)
            if df is None:
                continue
            
            try:
                mask = get_historical_mask(df, settings, sznl_map, ticker)
                mask = mask[mask.index >= cutoff_ts]
                if not mask.any():
                    continue
                
                true_indices = mask[mask].index
                
                for signal_date in true_indices:
                    signal_idx = df.index.get_loc(signal_date)
                    
                    key = (t_clean, signal_idx)
                    if key not in signal_data:
                        row = df.iloc[signal_idx]
                        signal_data[key] = {
                            'atr': row['ATR'],
                            'close': row['Close'],
                            'open': row['Open'],
                            'high': row['High'],
                            'low': row['Low'],
                            'vol_ratio': row.get('vol_ratio', 0),
                            'sznl': row.get('Sznl', 50),
                            'range_pct': row['RangePct'] * 100
                        }
                    
                    candidates.append((
                        signal_date.value,
                        ticker,
                        t_clean,
                        strat_idx,
                        signal_idx
                    ))
            except Exception:
                continue
    
    return candidates, signal_data


def build_price_matrix(processed_dict, tickers):
    """Build a unified price matrix for fast lookups."""
    all_dates = set()
    for t in tickers:
        df = processed_dict.get(t)
        if df is not None:
            all_dates.update(df.index)
    
    if not all_dates:
        return pd.DataFrame()
    
    all_dates = sorted(all_dates)
    price_data = {}
    
    for t in tickers:
        df = processed_dict.get(t)
        if df is not None:
            price_data[t] = df['Close'].reindex(all_dates, method='ffill')
    
    return pd.DataFrame(price_data, index=all_dates)


def process_signals_fast(candidates, signal_data, processed_dict, strategies, starting_equity):
    """
    Process candidates chronologically with dynamic sizing based on REAL-TIME MTM equity.
    Optimized version using price matrix for faster lookups.
    """
    if not candidates:
        return pd.DataFrame()
    
    candidates.sort(key=lambda x: x[0])
    
    # Build price matrix for all relevant tickers
    all_tickers = set(c[2] for c in candidates)
    price_matrix = build_price_matrix(processed_dict, all_tickers)
    
    # Track open positions for MTM
    open_positions = []
    realized_pnl = 0.0
    position_last_exit = {}
    
    results = []
    
    for cand in candidates:
        signal_ts, ticker, t_clean, strat_idx, signal_idx = cand
        signal_date = pd.Timestamp(signal_ts)
        
        strat = strategies[strat_idx]
        settings = strat['settings']
        strat_name = strat['name']
        
        pos_key = (strat_name, ticker)
        
        last_exit_ts = position_last_exit.get(pos_key)
        if last_exit_ts is not None and signal_ts <= last_exit_ts:
            continue
        
        df = processed_dict[t_clean]
        row_data = signal_data[(t_clean, signal_idx)]
        
        atr = row_data['atr']
        if pd.isna(atr) or atr <= 0:
            continue
        
        entry_type = settings.get('entry_type', 'Signal Close')
        hold_days = strat['execution']['hold_days']
        
        if signal_idx + 1 >= len(df):
            continue
        
        entry_row = df.iloc[signal_idx + 1]
        
        valid_entry = True
        entry_price = None
        entry_date = None
        
        if entry_type == 'T+1 Close':
            entry_price = entry_row['Close']
            entry_date = entry_row.name
        elif entry_type == 'T+1 Open':
            entry_price = entry_row['Open']
            entry_date = entry_row.name
        elif entry_type == 'T+1 Close if < Signal Close':
            if entry_row['Close'] < row_data['close']:
                entry_price = entry_row['Close']
                entry_date = entry_row.name
            else:
                valid_entry = False
        elif entry_type == "Limit (Open +/- 0.5 ATR)":
            limit_offset = 0.5 * atr
            if settings['trade_direction'] == 'Short':
                limit_price = entry_row['Open'] + limit_offset
                if entry_row['High'] >= limit_price:
                    entry_price = limit_price
                else:
                    valid_entry = False
            else:
                limit_price = entry_row['Open'] - limit_offset
                if entry_row['Low'] <= limit_price:
                    entry_price = limit_price
                else:
                    valid_entry = False
            entry_date = entry_row.name
        elif "Persistent" in entry_type:
            limit_offset = 0.5 * atr
            limit_base = row_data['close']
            found_fill = False
            search_end = min(signal_idx + 1 + hold_days, len(df))
            
            for i in range(signal_idx + 1, search_end):
                check_row = df.iloc[i]
                if settings['trade_direction'] == 'Short':
                    limit_price = limit_base + limit_offset
                    if check_row['High'] >= limit_price:
                        entry_price = limit_price
                        entry_date = check_row.name
                        hold_days = max(1, strat['execution']['hold_days'] - (i - signal_idx))
                        found_fill = True
                        break
                else:
                    limit_price = limit_base - limit_offset
                    if check_row['Low'] <= limit_price:
                        entry_price = limit_price
                        entry_date = check_row.name
                        hold_days = max(1, strat['execution']['hold_days'] - (i - signal_idx))
                        found_fill = True
                        break
            valid_entry = found_fill
        else:
            entry_price = row_data['close']
            entry_date = signal_date
        
        if not valid_entry or entry_price is None or pd.isna(entry_price):
            continue
        
        # --- CALCULATE REAL-TIME MTM EQUITY (optimized) ---
        still_open = []
        for pos in open_positions:
            if signal_date >= pos['exit_date']:
                # Position has closed - realize P&L
                if pos['t_clean'] in price_matrix.columns:
                    exit_price = price_matrix.loc[pos['exit_date'], pos['t_clean']] if pos['exit_date'] in price_matrix.index else pos['entry_price']
                else:
                    pos_df = processed_dict.get(pos['t_clean'])
                    exit_price = pos_df.iloc[pos['exit_idx']]['Close'] if pos_df is not None else pos['entry_price']
                
                if pos['direction'] == 'Long':
                    realized_pnl += (exit_price - pos['entry_price']) * pos['shares']
                else:
                    realized_pnl += (pos['entry_price'] - exit_price) * pos['shares']
            else:
                still_open.append(pos)
        
        open_positions = still_open
        
        # Calculate unrealized P&L using price matrix
        unrealized_pnl = 0.0
        for pos in open_positions:
            if pos['t_clean'] in price_matrix.columns and signal_date in price_matrix.index:
                current_price = price_matrix.loc[signal_date, pos['t_clean']]
            else:
                pos_df = processed_dict.get(pos['t_clean'])
                if pos_df is None:
                    continue
                price_slice = pos_df[pos_df.index <= signal_date]
                if price_slice.empty:
                    continue
                current_price = price_slice.iloc[-1]['Close']
            
            if pd.isna(current_price):
                continue
                
            if pos['direction'] == 'Long':
                unrealized_pnl += (current_price - pos['entry_price']) * pos['shares']
            else:
                unrealized_pnl += (pos['entry_price'] - current_price) * pos['shares']
        
        # REAL-TIME MTM EQUITY
        current_equity = starting_equity + realized_pnl + unrealized_pnl
        
        risk_bps = strat['execution']['risk_bps']
        base_risk = current_equity * risk_bps / 10000
        
        if strat_name == "Overbot Vol Spike":
            vol_ratio = row_data['vol_ratio']
            if vol_ratio > 2.0:
                base_risk = current_equity * 45 / 10000
            elif vol_ratio > 1.5:
                base_risk = current_equity * 35 / 10000
        
        if strat_name == "Weak Close Decent Sznls":
            sznl_val = row_data['sznl']
            if sznl_val >= 65:
                base_risk *= 1.5
            elif sznl_val >= 33:
                base_risk *= 0.66 if sznl_val < 50 else 1.0
        
        direction = settings.get('trade_direction', 'Long')
        stop_atr = strat['execution']['stop_atr']
        
        dist = atr * stop_atr
        action = "BUY" if direction == 'Long' else "SELL SHORT"
        
        if pd.isna(dist) or dist <= 0:
            continue

        try:
            shares = int(base_risk / dist)
        except (ValueError, OverflowError):
            shares = 0
            
        if shares == 0:
            continue
        
        entry_idx = df.index.get_loc(entry_date)
        exit_idx = min(entry_idx + hold_days, len(df) - 1)
        exit_row = df.iloc[exit_idx]
        exit_price = exit_row['Close']
        exit_date = exit_row.name
        
        if action == "BUY":
            pnl = ((exit_price - entry_price) * shares).round(0)
        else:
            pnl = ((entry_price - exit_price) * shares).round(0)
        
        target_ts_idx = entry_idx + hold_days
        if target_ts_idx < len(df):
            time_stop_date = df.index[target_ts_idx]
        else:
            time_stop_date = entry_date + BusinessDay(hold_days)
        
        open_positions.append({
            'ticker': ticker,
            't_clean': t_clean,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'shares': shares,
            'direction': 'Long' if action == 'BUY' else 'Short',
            'exit_idx': exit_idx,
            'exit_date': exit_date,
            'strat_name': strat_name
        })
        
        position_last_exit[pos_key] = exit_date.value
        
        results.append({
            "Date": signal_date,
            "Entry Date": entry_date,
            "Exit Date": exit_date,
            "Time Stop": time_stop_date,
            "Strategy": strat_name,
            "Ticker": ticker,
            "Action": action,
            "Entry Criteria": entry_type,
            "Price": entry_price,
            "Shares": shares,
            "PnL": pnl,
            "ATR": atr,
            "T+1 Open": entry_row['Open'],
            "Signal Close": row_data['close'],
            "Range %": row_data['range_pct'],
            "Equity at Signal": current_equity,
            "Risk $": base_risk,
            "Risk bps": risk_bps
        })
            
    if not results:
        return pd.DataFrame()
    
    sig_df = pd.DataFrame(results)
    
    sig_df['Date'] = pd.to_datetime(sig_df['Date'])
    sig_df['Entry Date'] = pd.to_datetime(sig_df['Entry Date'])
    sig_df['Exit Date'] = pd.to_datetime(sig_df['Exit Date'])
    sig_df['Time Stop'] = pd.to_datetime(sig_df['Time Stop'])
    
    return sig_df.sort_values(by="Exit Date")


def get_daily_mtm_series(sig_df, master_dict, start_date=None):
    """Optimized daily MTM calculation using vectorized operations."""
    if sig_df.empty:
        return pd.Series(dtype=float)
    
    if start_date is not None:
        min_date = pd.Timestamp(start_date)
    else:
        min_date = sig_df['Entry Date'].min()
    
    max_date = max(sig_df['Exit Date'].max(), pd.Timestamp.today())
    all_dates = pd.date_range(start=min_date, end=max_date, freq='B')
    daily_pnl = pd.Series(0.0, index=all_dates)
    
    # Pre-process price data
    price_cache = {}
    for ticker in sig_df['Ticker'].unique():
        t_clean = ticker.replace('.', '-')
        t_df = master_dict.get(t_clean)
        if t_df is not None and not t_df.empty:
            temp_df = t_df.copy()
            if isinstance(temp_df.columns, pd.MultiIndex):
                temp_df.columns = [c[0] if isinstance(c, tuple) else c for c in temp_df.columns]
            temp_df.columns = [c.capitalize() for c in temp_df.columns]
            price_cache[ticker] = temp_df['Close'].reindex(all_dates, method='ffill')
    
    # Process trades using itertuples (faster than iterrows)
    for trade in sig_df.itertuples():
        ticker = trade.Ticker
        action = trade.Action
        shares = trade.Shares
        entry_date = trade._3  # Entry Date
        exit_date = trade._4   # Exit Date  
        entry_price = trade.Price
        
        if ticker not in price_cache:
            if exit_date in daily_pnl.index:
                daily_pnl[exit_date] += trade.PnL
            continue
        
        closes = price_cache[ticker]
        trade_dates = all_dates[(all_dates >= entry_date) & (all_dates <= exit_date)]
        
        if len(trade_dates) == 0:
            continue
        
        trade_closes = closes.loc[trade_dates]
        if trade_closes.empty:
            continue
        
        # First day P&L
        first_date = trade_dates[0]
        if first_date in trade_closes.index and not pd.isna(trade_closes[first_date]):
            if action == "BUY":
                daily_pnl[first_date] += (trade_closes[first_date] - entry_price) * shares
            else:
                daily_pnl[first_date] += (entry_price - trade_closes[first_date]) * shares
        
        # Subsequent days - daily changes
        if len(trade_dates) > 1:
            diffs = trade_closes.diff().dropna()
            if action == "SELL SHORT":
                diffs = -diffs
            for d, val in (diffs * shares).items():
                if d in daily_pnl.index and not pd.isna(val):
                    daily_pnl[d] += val
    
    return daily_pnl


def calculate_mark_to_market_curve(sig_df, master_dict, starting_equity, start_date=None):
    daily_pnl = get_daily_mtm_series(sig_df, master_dict, start_date)
    if daily_pnl.empty:
        return pd.DataFrame(columns=['Equity'])
    
    equity_curve = starting_equity + daily_pnl.cumsum()
    
    return equity_curve.to_frame(name='Equity')


def calculate_daily_exposure(sig_df, starting_equity=None):
    if sig_df.empty:
        return pd.DataFrame()
    min_date = sig_df['Date'].min()
    max_date = sig_df['Exit Date'].max()
    all_dates = pd.date_range(start=min_date, end=max_date)
    exposure_df = pd.DataFrame(0.0, index=all_dates, columns=['Long Exposure', 'Short Exposure'])
    
    for row in sig_df.itertuples():
        trade_dates = pd.date_range(start=row.Date, end=row._4)  # Exit Date
        dollar_val = row.Price * row.Shares
        col = 'Long Exposure' if row.Action == 'BUY' else 'Short Exposure'
        exposure_df.loc[exposure_df.index.isin(trade_dates), col] += dollar_val
    
    exposure_df['Net Exposure'] = exposure_df['Long Exposure'] - exposure_df['Short Exposure']
    exposure_df['Gross Exposure'] = exposure_df['Long Exposure'] + exposure_df['Short Exposure']
    
    if starting_equity is not None:
        equity_series = pd.Series(starting_equity, index=all_dates)
        
        for date in all_dates:
            closed_trades = sig_df[sig_df['Exit Date'] <= date]
            realized_pnl = closed_trades['PnL'].sum()
            equity_series[date] = starting_equity + realized_pnl
        
        exposure_df['Long Exposure %'] = (exposure_df['Long Exposure'] / equity_series) * 100
        exposure_df['Short Exposure %'] = (exposure_df['Short Exposure'] / equity_series) * 100
        exposure_df['Net Exposure %'] = (exposure_df['Net Exposure'] / equity_series) * 100
        exposure_df['Gross Exposure %'] = (exposure_df['Gross Exposure'] / equity_series) * 100
        
        return exposure_df[['Long Exposure %', 'Short Exposure %', 'Net Exposure %', 'Gross Exposure %']]
    
    return exposure_df


def calculate_annual_stats(daily_pnl_series, starting_equity):
    if daily_pnl_series.empty:
        return pd.DataFrame()
    
    equity_series = starting_equity + daily_pnl_series.cumsum()
    daily_rets = equity_series.pct_change().fillna(0)
    
    yearly_stats = []
    for year_date, rets in daily_rets.resample('YE'):
        year = year_date.year
        if rets.empty:
            continue
        
        year_mask = equity_series.index.year == year
        year_eq = equity_series[year_mask]
        if len(year_eq) < 2:
            continue
            
        year_start = year_eq.iloc[0]
        year_end = year_eq.iloc[-1]
        
        total_ret_pct = (year_end - year_start) / year_start if year_start != 0 else 0
        total_ret_dollar = year_end - year_start
        
        std_dev = rets.std() * np.sqrt(252)
        mean_ret = rets.mean() * 252
        sharpe = mean_ret / std_dev if std_dev != 0 else 0
        
        neg_rets = rets[rets < 0]
        downside_std = np.sqrt((neg_rets**2).mean()) * np.sqrt(252) if len(neg_rets) > 0 else 0
        sortino = mean_ret / downside_std if downside_std != 0 else 0
        
        running_max = year_eq.expanding().max()
        drawdown = (year_eq - running_max) / running_max
        max_dd = drawdown.min()
        
        yearly_stats.append({
            "Year": year,
            "Total Return ($)": total_ret_dollar,
            "Total Return (%)": total_ret_pct,
            "Max Drawdown": max_dd,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Std Dev": std_dev
        })
    
    return pd.DataFrame(yearly_stats)


def calculate_performance_stats(sig_df, master_dict, starting_equity, start_date=None):
    """Calculate performance stats - removed Sharpe/Sortino from per-strategy breakdown."""
    stats = []
    
    def get_metrics(df, name):
        if df.empty:
            return None
        
        count = len(df)
        total_pnl = df['PnL'].sum()
        gross_profit = df[df['PnL'] > 0]['PnL'].sum()
        gross_loss = abs(df[df['PnL'] < 0]['PnL'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan
        avg_pnl = df['PnL'].mean()
        std_pnl = df['PnL'].std()
        sqn = (avg_pnl / std_pnl * np.sqrt(count)) if std_pnl != 0 else 0
        
        return {
            "Strategy": name, "Trades": count, "Total PnL": total_pnl,
            "Profit Factor": profit_factor, "SQN": sqn
        }
    
    for strat in sig_df['Strategy'].unique():
        m = get_metrics(sig_df[sig_df['Strategy'] == strat], strat)
        if m:
            stats.append(m)
    
    total_m = get_metrics(sig_df, "TOTAL PORTFOLIO")
    if total_m:
        stats.append(total_m)
    
    return pd.DataFrame(stats)


def analyze_signal_density(sig_df, window_days=0):
    if sig_df.empty:
        return pd.DataFrame(), sig_df
    
    df = sig_df.copy()
    signal_counts = df.groupby('Date').size().to_dict()
    df['Signals Same Day'] = df['Date'].map(signal_counts)
    
    if window_days > 0:
        def count_nearby_signals(date):
            nearby = df[(df['Date'] >= date - pd.Timedelta(days=window_days)) & 
                       (df['Date'] <= date + pd.Timedelta(days=window_days))]
            return len(nearby)
        df['Signals in Window'] = df['Date'].apply(count_nearby_signals)
        density_col = 'Signals in Window'
    else:
        density_col = 'Signals Same Day'
    
    def get_density_bucket(count):
        if count == 1:
            return '1 (Isolated)'
        elif count <= 3:
            return '2-3 (Low)'
        elif count <= 6:
            return '4-6 (Moderate)'
        elif count <= 10:
            return '7-10 (High)'
        else:
            return '11+ (Clustered)'
    
    df['Density Bucket'] = df[density_col].apply(get_density_bucket)
    df['R-Multiple'] = df['PnL'] / df['Risk $']
    
    results = []
    bucket_order = ['1 (Isolated)', '2-3 (Low)', '4-6 (Moderate)', '7-10 (High)', '11+ (Clustered)']
    
    for bucket in bucket_order:
        bucket_df = df[df['Density Bucket'] == bucket]
        if bucket_df.empty:
            continue
        
        n_trades = len(bucket_df)
        total_pnl = bucket_df['PnL'].sum()
        total_risk = bucket_df['Risk $'].sum()
        
        winners = bucket_df[bucket_df['PnL'] > 0]
        losers = bucket_df[bucket_df['PnL'] <= 0]
        
        win_rate = len(winners) / n_trades if n_trades > 0 else 0
        avg_r = bucket_df['R-Multiple'].mean()
        
        avg_win_r = winners['R-Multiple'].mean() if len(winners) > 0 else 0
        avg_loss_r = losers['R-Multiple'].mean() if len(losers) > 0 else 0
        
        pnl_per_risk = total_pnl / total_risk if total_risk > 0 else 0
        
        gross_profit = winners['PnL'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['PnL'].sum()) if len(losers) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
        
        results.append({
            'Density': bucket,
            'Trades': n_trades,
            '% of Trades': n_trades / len(df),
            'Win Rate': win_rate,
            'Avg R': avg_r,
            'Avg Win R': avg_win_r,
            'Avg Loss R': avg_loss_r,
            'PnL/$ Risk': pnl_per_risk,
            'Profit Factor': profit_factor,
            'Total PnL': total_pnl,
            '% of PnL': total_pnl / df['PnL'].sum() if df['PnL'].sum() != 0 else 0
        })
    
    return pd.DataFrame(results), df


def analyze_density_by_strategy(sig_df):
    if sig_df.empty:
        return pd.DataFrame()
    
    df = sig_df.copy()
    signal_counts = df.groupby('Date').size().to_dict()
    df['Signals Same Day'] = df['Date'].map(signal_counts)
    df['R-Multiple'] = df['PnL'] / df['Risk $']
    df['Is Isolated'] = df['Signals Same Day'] <= 3
    
    results = []
    
    for strat in df['Strategy'].unique():
        strat_df = df[df['Strategy'] == strat]
        
        isolated = strat_df[strat_df['Is Isolated']]
        clustered = strat_df[~strat_df['Is Isolated']]
        
        iso_r = isolated['R-Multiple'].mean() if len(isolated) > 0 else np.nan
        clu_r = clustered['R-Multiple'].mean() if len(clustered) > 0 else np.nan
        
        iso_wr = (isolated['PnL'] > 0).mean() if len(isolated) > 0 else np.nan
        clu_wr = (clustered['PnL'] > 0).mean() if len(clustered) > 0 else np.nan
        
        if not np.isnan(iso_r) and not np.isnan(clu_r) and clu_r != 0:
            edge = (iso_r - clu_r) / abs(clu_r)
        else:
            edge = np.nan
        
        results.append({
            'Strategy': strat,
            'Isolated Trades': len(isolated),
            'Clustered Trades': len(clustered),
            'Isolated Avg R': iso_r,
            'Clustered Avg R': clu_r,
            'Isolated WR': iso_wr,
            'Clustered WR': clu_wr,
            'Isolation Edge': edge,
            'Size Up Isolated?': 'Yes ✓' if (not np.isnan(edge) and edge > 0.15) else 'No'
        })
    
    return pd.DataFrame(results).sort_values('Isolation Edge', ascending=False)


def calculate_capital_efficiency(sig_df, strategies):
    if sig_df.empty:
        return pd.DataFrame()
    
    strat_settings = {s['name']: s for s in strategies}
    results = []
    
    for strat_name in sig_df['Strategy'].unique():
        strat_df = sig_df[sig_df['Strategy'] == strat_name].copy()
        
        if strat_df.empty:
            continue
        
        n_trades = len(strat_df)
        total_pnl = strat_df['PnL'].sum()
        total_risk = strat_df['Risk $'].sum()
        
        winners = strat_df[strat_df['PnL'] > 0]
        losers = strat_df[strat_df['PnL'] <= 0]
        win_rate = len(winners) / n_trades if n_trades > 0 else 0
        
        strat_df['Hold Days'] = (strat_df['Exit Date'] - strat_df['Entry Date']).dt.days
        avg_hold_days = strat_df['Hold Days'].mean()
        
        current_bps = strat_settings.get(strat_name, {}).get('execution', {}).get('risk_bps', 0)
        
        pnl_per_risk = total_pnl / total_risk if total_risk > 0 else 0
        capital_turns_per_year = 252 / avg_hold_days if avg_hold_days > 0 else 0
        annualized_ror = pnl_per_risk * capital_turns_per_year
        
        date_range = (strat_df['Date'].max() - strat_df['Date'].min()).days
        years = date_range / 365.25 if date_range > 0 else 1
        signals_per_year = n_trades / years if years > 0 else n_trades
        
        total_portfolio_risk = sig_df['Risk $'].sum()
        risk_contribution = total_risk / total_portfolio_risk if total_portfolio_risk > 0 else 0
        
        total_portfolio_pnl = sig_df['PnL'].sum()
        pnl_contribution = total_pnl / total_portfolio_pnl if total_portfolio_pnl > 0 else 0
        
        efficiency_ratio = pnl_contribution / risk_contribution if risk_contribution > 0 else 0
        
        results.append({
            'Strategy': strat_name,
            'Trades': n_trades,
            'Signals/Yr': signals_per_year,
            'Avg Days': avg_hold_days,
            'Win Rate': win_rate,
            'Total $ Risked': total_risk,
            'Total PnL': total_pnl,
            'PnL/$ Risk': pnl_per_risk,
            'Ann. Turns': capital_turns_per_year,
            'Ann. RoR': annualized_ror,
            '% of Risk': risk_contribution,
            '% of PnL': pnl_contribution,
            'Efficiency': efficiency_ratio,
            'Current Bps': current_bps,
        })
    
    df = pd.DataFrame(results)
    
    if df.empty:
        return df
    
    total_current_bps = df['Current Bps'].sum()
    df['RoR Weight'] = df['Ann. RoR'] / df['Ann. RoR'].sum() if df['Ann. RoR'].sum() > 0 else 1 / len(df)
    df['Suggested Bps'] = (df['RoR Weight'] * total_current_bps).round(0).astype(int)
    df['Suggested Bps'] = df['Suggested Bps'].clip(lower=10)
    df['Bps Δ'] = df['Suggested Bps'] - df['Current Bps']
    
    df = df.sort_values('Ann. RoR', ascending=False)
    
    return df


# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Strategy Backtest Lab v3")
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
        starting_equity = st.number_input("Starting Equity ($)", value=150000, step=10000)
        st.caption(f"Data buffer: 365 days prior to {user_start_date}.")
        st.markdown("---")
        st.markdown("**🔄 Dynamic Position Sizing**")
        st.caption("Sizes scale with MTM equity (bps of current value including unrealized P&L).")
        run_btn = st.form_submit_button("⚡ Run Backtest")

    st.title("⚡ Strategy Backtest Lab v3")
    st.markdown(f"**Start:** {user_start_date} | **Equity:** ${starting_equity:,.0f}")
    st.info("💡 **v3:** Position sizes scale dynamically with real-time MTM equity (realized + unrealized P&L).")
    st.markdown("---")

    if run_btn:
        sznl_map = load_seasonal_map()
        if 'backtest_data' not in st.session_state:
            st.session_state['backtest_data'] = {}

        import copy
        strategies = [copy.deepcopy(s) for s in _STRATEGY_BOOK_RAW]

        long_term_tickers = set()
        for strat in strategies:
            long_term_tickers.update(strat['universe_tickers'])
            s = strat['settings']
            if s.get('use_market_sznl'):
                long_term_tickers.add(s.get('market_ticker', '^GSPC'))
            if "Market" in s.get('trend_filter', ''):
                long_term_tickers.add(s.get('market_ticker', 'SPY'))
            if s.get('use_vix_filter'):
                long_term_tickers.add('^VIX')
        long_term_tickers.add('SPY')

        long_term_list = [t.replace('.', '-') for t in long_term_tickers]
        existing = set(st.session_state['backtest_data'].keys())
        missing = list(set(long_term_list) - existing)
        
        if missing:
            st.write(f"📥 Downloading {len(missing)} tickers...")
            data = download_historical_data(missing, start_date="2000-01-01")
            st.session_state['backtest_data'].update(data)
            st.success("✅ Download complete.")

        master_dict = st.session_state['backtest_data']
        
        vix_df = master_dict.get('^VIX')
        vix_series = None
        if vix_df is not None and not vix_df.empty:
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = [c[0] if isinstance(c, tuple) else c for c in vix_df.columns]
            vix_df.columns = [c.capitalize() for c in vix_df.columns]
            vix_series = vix_df['Close']

        st.write("📊 **Phase 1:** Computing indicators...")
        t0 = time.time()
        processed_dict = precompute_all_indicators(master_dict, strategies, sznl_map, vix_series)
        st.write(f"   Processed {len(processed_dict)} tickers in {time.time()-t0:.1f}s")

        st.write("🔍 **Phase 2:** Finding signals...")
        t0 = time.time()
        candidates, signal_data = generate_candidates_fast(processed_dict, strategies, sznl_map, user_start_date)
        st.write(f"   Found {len(candidates):,} candidates in {time.time()-t0:.1f}s")

        st.write("📈 **Phase 3:** Processing with dynamic MTM-based sizing...")
        t0 = time.time()
        sig_df = process_signals_fast(candidates, signal_data, processed_dict, strategies, starting_equity)
        st.write(f"   Executed {len(sig_df):,} trades in {time.time()-t0:.1f}s")

        if not sig_df.empty:
            st.success(f"✅ Backtest complete: {len(sig_df):,} trades")
            
            today = pd.Timestamp(datetime.date.today())
            open_mask = sig_df['Time Stop'] >= today
            open_df = sig_df[open_mask].copy()

            if not open_df.empty:
                current_prices, open_pnls, current_values = [], [], []
                for row in open_df.itertuples():
                    t_df = master_dict.get(row.Ticker.replace('.', '-'))
                    if t_df is not None and not t_df.empty:
                        if isinstance(t_df.columns, pd.MultiIndex):
                            t_df.columns = [c[0] if isinstance(c, tuple) else c for c in t_df.columns]
                        t_df.columns = [c.capitalize() for c in t_df.columns]
                        last_close = t_df['Close'].iloc[-1]
                    else:
                        last_close = row.Price
                    
                    if row.Action == 'BUY':
                        pnl = (last_close - row.Price) * row.Shares
                    else:
                        pnl = (row.Price - last_close) * row.Shares
                    
                    current_prices.append(last_close)
                    open_pnls.append(pnl)
                    current_values.append(last_close * row.Shares)

                open_df['Current Price'] = current_prices
                open_df['Open PnL'] = open_pnls
                open_df['Mkt Value'] = current_values
                
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
                m5.metric("Total Open PnL", f"${total_open_pnl:,.0f}", delta_color="normal", delta=f"{total_open_pnl:,.0f}")
                display_cols = [
                    "Entry Date", "Time Stop", "Strategy", "Ticker", "Action",
                    "Price", "Current Price", "Shares", "Mkt Value", "PnL", 
                    "ATR", "Risk $", "Equity at Signal"
                ]
                # FIXED FORMATTING: PnL as integer, ATR to 2 decimals, Exit Date without time, Mkt Value as $
                st.dataframe(open_df.style.format({
                    "Entry Date": "{:%Y-%m-%d}", 
                    "Time Stop": "{:%Y-%m-%d}", 
                    "Price": "${:.2f}", 
                    "Current Price": "${:.2f}", 
                    "PnL": "${:,.0f}",
                    "ATR": "{:.2f}",
                    "Mkt Value": "${:,.0f}",
                    "Shares": "{:.0f}",
                    "Risk $": "${:,.0f}",
                    "Equity at Signal": "${:,.0f}", 
                }), use_container_width=True)
            else:
                st.divider()
                st.info("No active positions (Time Stop >= Today).")

            st.divider()
            
            st.subheader("📐 Dynamic Sizing Analysis")
            cols = st.columns(4)
            cols[0].metric("Avg Risk/Trade", f"${sig_df['Risk $'].mean():,.0f}")
            cols[1].metric("Risk Range", f"${sig_df['Risk $'].min():,.0f} - ${sig_df['Risk $'].max():,.0f}")
            final_eq = starting_equity + sig_df['PnL'].sum()
            cols[2].metric("Final Equity", f"${final_eq:,.0f}", delta=f"{(final_eq/starting_equity-1)*100:.1f}%")
            equity_growth = sig_df['Equity at Signal'].max() / sig_df['Equity at Signal'].min()
            cols[3].metric("Peak/Min Equity Ratio", f"{equity_growth:.2f}x", help="Shows how much your sizing scaled")
            
            st.subheader("📅 Annual Performance")
            port_daily_pnl = get_daily_mtm_series(sig_df, master_dict, start_date=user_start_date)
            annual_df = calculate_annual_stats(port_daily_pnl, starting_equity)
            if not annual_df.empty:
                st.dataframe(annual_df.style.format({
                    "Total Return ($)": "${:,.0f}", "Total Return (%)": "{:.1%}", "Max Drawdown": "{:.1%}",
                    "Sharpe Ratio": "{:.2f}", "Sortino Ratio": "{:.2f}", "Std Dev": "{:.1%}"
                }), use_container_width=True)

            st.subheader("📊 Strategy Metrics")
            stats_df = calculate_performance_stats(sig_df, master_dict, starting_equity, start_date=user_start_date)
            st.dataframe(stats_df.style.format({
                "Total PnL": "${:,.0f}",
                "Profit Factor": "{:.2f}", "SQN": "{:.2f}"
            }), use_container_width=True)

            st.subheader("💰 Capital Efficiency & Sizing Analysis")
            efficiency_df = calculate_capital_efficiency(sig_df, strategies)
            
            if not efficiency_df.empty:
                col1, col2, col3 = st.columns(3)
                
                most_efficient = efficiency_df.iloc[0]['Strategy']
                best_ror = efficiency_df.iloc[0]['Ann. RoR']
                col1.metric("Most Efficient Strategy", most_efficient, delta=f"{best_ror:.1%} Ann. RoR")
                
                under_allocated = efficiency_df[efficiency_df['Efficiency'] > 1.2]
                over_allocated = efficiency_df[efficiency_df['Efficiency'] < 0.8]
                col2.metric("Under-allocated", f"{len(under_allocated)} strategies")
                col3.metric("Over-allocated", f"{len(over_allocated)} strategies")
                
                display_cols = ['Strategy', 'Trades', 'Signals/Yr', 'Avg Days', 'Win Rate', 
                               'PnL/$ Risk', 'Ann. RoR', '% of Risk', '% of PnL', 'Efficiency',
                               'Current Bps', 'Suggested Bps', 'Bps Δ']
                
                st.dataframe(efficiency_df[display_cols].style.format({
                    'Signals/Yr': '{:.1f}',
                    'Avg Days': '{:.1f}',
                    'Win Rate': '{:.1%}',
                    'PnL/$ Risk': '{:.2f}',
                    'Ann. RoR': '{:.1%}',
                    '% of Risk': '{:.1%}',
                    '% of PnL': '{:.1%}',
                    'Efficiency': '{:.2f}x',
                    'Current Bps': '{:.0f}',
                    'Suggested Bps': '{:.0f}',
                    'Bps Δ': '{:+.0f}'
                }), use_container_width=True)

            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Portfolio Equity (MTM) - Log Scale")
                df_eq = calculate_mark_to_market_curve(sig_df, master_dict, starting_equity, start_date=user_start_date)
                
                if not df_eq.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_eq.index, 
                        y=df_eq['Equity'], 
                        mode='lines', 
                        name='Portfolio Equity', 
                        line=dict(color='#00FF00', width=2)
                    ))
                    
                    # Add starting equity reference line
                    fig.add_hline(y=starting_equity, line_dash="dash", line_color="gray", 
                                  annotation_text=f"Start: ${starting_equity:,.0f}")
                    
                    fig.update_layout(
                        height=400,
                        margin=dict(l=10, r=10, t=30, b=10),
                        yaxis_type="log",
                        yaxis_title="Equity ($)",
                        yaxis=dict(
                            tickformat="$,.0f",
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No trades to plot.")
            
            with col2:
                st.subheader("📉 PnL by Strategy (Interactive)")
                strat_pnl = sig_df.pivot_table(index='Exit Date', columns='Strategy', values='PnL', aggfunc='sum')
                
                if not strat_pnl.empty:
                    strat_pnl_cum = strat_pnl.fillna(0).cumsum()
                    
                    fig_strat = go.Figure()
                    
                    for column in strat_pnl_cum.columns:
                        fig_strat.add_trace(go.Scatter(
                            x=strat_pnl_cum.index,
                            y=strat_pnl_cum[column],
                            mode='lines',
                            name=str(column)
                        ))
                        
                    fig_strat.update_layout(
                        height=400, 
                        margin=dict(l=10, r=10, t=30, b=10),
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    st.plotly_chart(fig_strat, use_container_width=True)
                else:
                    st.info("No strategy data available.")

            st.subheader("⚖️ Exposure Over Time (% of Equity)")
            exposure_df = calculate_daily_exposure(sig_df, starting_equity=starting_equity)
            if not exposure_df.empty:
                # Interactive Plotly chart for exposure
                fig_exposure = go.Figure()
                
                colors = {
                    'Long Exposure %': '#00CC00',      # Green
                    'Short Exposure %': '#CC0000',     # Red
                    'Net Exposure %': '#0066CC',       # Blue
                    'Gross Exposure %': '#FF9900'      # Orange
                }
                
                for col in exposure_df.columns:
                    fig_exposure.add_trace(go.Scatter(
                        x=exposure_df.index,
                        y=exposure_df[col],
                        mode='lines',
                        name=col,
                        line=dict(color=colors.get(col, '#888888'), width=1.5),
                        hovertemplate=f'{col}: %{{y:.1f}}%<extra></extra>'
                    ))
                
                fig_exposure.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
                
                fig_exposure.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=30, b=10),
                    hovermode="x unified",
                    yaxis_title="Exposure (%)",
                    xaxis_title="Date",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_exposure, use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Gross Exposure", f"{exposure_df['Gross Exposure %'].mean():.1f}%")
                col2.metric("Max Gross Exposure", f"{exposure_df['Gross Exposure %'].max():.1f}%")
                col3.metric("Avg Net Exposure", f"{exposure_df['Net Exposure %'].mean():.1f}%")
                col4.metric("Max Net Exposure", f"{exposure_df['Net Exposure %'].max():.1f}%")

            st.subheader("📜 Trade Log")
            display_cols = ["Date", "Entry Date", "Exit Date", "Strategy", "Ticker", "Action",
                          "Entry Criteria", "Signal Close", "T+1 Open", "Price", "Shares", "PnL", 
                          "ATR", "Equity at Signal", "Risk $"]
            # FIXED FORMATTING for Trade Log as well
            st.dataframe(sig_df[display_cols].sort_values("Date", ascending=False).style.format({
                "Price": "${:.2f}", 
                "Signal Close": "${:.2f}",
                "T+1 Open": "${:.2f}",
                "PnL": "${:,.0f}", 
                "Date": "{:%Y-%m-%d}",
                "Entry Date": "{:%Y-%m-%d}", 
                "Exit Date": "{:%Y-%m-%d}",
                "ATR": "{:.2f}",
                "Equity at Signal": "${:,.0f}", 
                "Risk $": "${:,.0f}",
                "Shares": "{:.0f}"
            }), use_container_width=True, height=400)
        else:
            st.warning(f"No signals found starting from {user_start_date}.")
    else:
        st.info("👈 Configure settings and click 'Run Backtest' to begin.")


if __name__ == "__main__":
    main()
