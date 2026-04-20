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

from indicators import calculate_indicators, get_sznl_val_series

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# -----------------------------------------------------------------------------
# IMPORT STRATEGY BOOK FROM ROOT
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from strategy_config import _STRATEGY_BOOK_RAW, ACCOUNT_VALUE, DAILY_RISK_CAP_BPS
except ImportError:
    # st.error("Could not find strategy_config.py in the root directory.")
    _STRATEGY_BOOK_RAW = []
    ACCOUNT_VALUE = 150000
    DAILY_RISK_CAP_BPS = 0

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
PRIMARY_SZNL_PATH = "sznl_ranks.csv"
BACKUP_SZNL_PATH = "seasonal_ranks.csv"
ATR_SZNL_PATH = "atr_seasonal_ranks.parquet"

ATR_SZNL_WINDOWS = [5, 10, 21, 63, 126, 252]
ATR_SZNL_COLS = [f"atr_sznl_{w}d" for w in ATR_SZNL_WINDOWS]

# OLV cooldown: block re-fires within N trading days of the last primary signal.
# Mirrors live-scan gate in daily_scan.py (load_olv_cooldown). Primary MOC + its
# LOC companion both fire within the same candidate iteration; cooldown applies
# to the NEXT candidate on the same ticker.
OLV_STRATEGY_NAME = "Oversold Low Volume"
OLV_COOLDOWN_DAYS = 20


@st.cache_resource
def load_atr_seasonal_map():
    """Load ATR-normalized seasonal ranks. Returns {ticker: DataFrame with 6 rank columns}."""
    path = os.path.join(parent_dir, ATR_SZNL_PATH)
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_parquet(path)
        df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
        output = {}
        for ticker, group in df.groupby('ticker'):
            output[ticker] = group.set_index('Date')[ATR_SZNL_COLS].sort_index()
        return output
    except Exception:
        return {}


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

# -----------------------------------------------------------------------------
# HELPER: BATCH DOWNLOADER
# -----------------------------------------------------------------------------
def _parse_yf_result(df, chunk, data_dict):
    """Parse a yfinance download result into per-ticker DataFrames."""
    if df is None or df.empty:
        return
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0).unique().tolist()
        price_cols = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                      'open', 'high', 'low', 'close', 'adj close', 'volume'}
        if set(lvl0) & price_cols:
            tickers_in_data = df.columns.get_level_values(1).unique().tolist()
            for t in tickers_in_data:
                try:
                    t_df = df.xs(t, level=1, axis=1).copy()
                    t_df.columns = [str(c).capitalize() for c in t_df.columns]
                    if 'Close' not in t_df.columns or t_df['Close'].dropna().empty:
                        continue
                    data_dict[str(t).upper()] = t_df
                except Exception:
                    continue
        else:
            for t in lvl0:
                try:
                    t_df = df[t].copy()
                    if t_df.empty or 'Close' not in t_df.columns:
                        continue
                    data_dict[str(t).upper()] = t_df
                except Exception:
                    continue
    else:
        if len(chunk) == 1:
            ticker = chunk[0]
            df.columns = [str(c).capitalize() for c in df.columns]
            if 'Close' in df.columns:
                data_dict[ticker] = df


def _download_chunk(chunk, start_date, max_retries=2):
    """Download a single chunk with retries; returns a dict. Isolated so one
    failed chunk (rate-limited or a single bad ticker) can't hang the whole run."""
    out = {}
    for attempt in range(max_retries):
        try:
            df = yf.download(
                chunk, start=start_date, group_by='ticker',
                auto_adjust=False, progress=False, threads=True,
            )
            _parse_yf_result(df, chunk, out)
            if out:
                return out
        except Exception:
            pass
        if attempt < max_retries - 1:
            time.sleep(1.0)
    # Last resort: fall back to per-ticker download so one bad ticker doesn't nuke the chunk.
    for t in chunk:
        if t in out:
            continue
        try:
            df = yf.download(t, start=start_date, auto_adjust=False, progress=False, threads=False)
            _parse_yf_result(df, [t], out)
        except Exception:
            continue
    return out


def download_historical_data(tickers, start_date="2000-01-01"):
    if not tickers: return {}
    clean_tickers = list(set([str(t).strip().upper().replace('.', '-') for t in tickers]))
    data_dict = {}
    CHUNK_SIZE = 40
    MAX_WORKERS = 3  # Parallel chunks — I/O bound, so threading helps; cap to avoid rate limits.
    total = len(clean_tickers)
    chunks = [clean_tickers[i:i + CHUNK_SIZE] for i in range(0, total, CHUNK_SIZE)]
    total_batches = len(chunks)
    progress_bar = st.progress(0)
    status_text = st.empty()
    completed = 0

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_download_chunk, c, start_date): c for c in chunks}
        for fut in as_completed(futures):
            chunk_result = fut.result()
            data_dict.update(chunk_result)
            completed += 1
            status_text.text(f"📥 Downloaded batch {completed}/{total_batches} ({len(data_dict)}/{total} tickers)...")
            progress_bar.progress(min(completed / total_batches, 1.0))

    progress_bar.empty()
    status_text.empty()
    return data_dict


def update_stale_cache(cache_dir, tickers, data_dict):
    """Incrementally update cached parquets — only download from last cached date."""
    stale_tickers = {}  # {start_date_str: [tickers]}
    today_str = datetime.date.today().strftime('%Y-%m-%d')

    for t in tickers:
        if t in data_dict:
            cached_df = data_dict[t]
            if cached_df is not None and not cached_df.empty:
                last_date = cached_df.index.max()
                if pd.Timestamp(last_date).date() >= datetime.date.today() - datetime.timedelta(days=2):
                    continue  # fresh enough
                start = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                stale_tickers.setdefault(start, []).append(t)

    if not stale_tickers:
        return 0

    total_updated = 0
    for start_date, tickers_batch in stale_tickers.items():
        new_data = {}
        CHUNK_SIZE = 40
        for i in range(0, len(tickers_batch), CHUNK_SIZE):
            chunk = tickers_batch[i : i + CHUNK_SIZE]
            try:
                df = yf.download(chunk, start=start_date, group_by='ticker', auto_adjust=False, progress=False, threads=True)
                _parse_yf_result(df, chunk, new_data)
            except Exception:
                pass
            time.sleep(0.5)

        for t, new_df in new_data.items():
            if t in data_dict and not new_df.empty:
                existing = data_dict[t]
                combined = pd.concat([existing, new_df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
                data_dict[t] = combined
                try:
                    combined.to_parquet(os.path.join(cache_dir, f"{t}.parquet"))
                except Exception:
                    pass
                total_updated += 1

    return total_updated


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

    # Range in ATR filter
    if params.get('use_range_atr_filter', False):
        if 'range_in_atr' in df.columns:
            ria = df['range_in_atr'].values
            logic = params.get('range_atr_logic', 'Between')
            if logic == '>':
                conditions.append(ria > params.get('range_atr_min', 0))
            elif logic == '<':
                conditions.append(ria < params.get('range_atr_max', 99))
            elif logic == 'Between':
                conditions.append((ria >= params.get('range_atr_min', 0)) & (ria <= params.get('range_atr_max', 99)))

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
        lookback = params.get('gap_lookback', 21)
        col_name = f'GapCount_{lookback}' if f'GapCount_{lookback}' in df.columns else 'GapCount'
        g_val = df[col_name].values
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

    # ATR seasonal rank filters
    for asf in params.get('atr_sznl_filters', []):
        col = f"atr_sznl_{asf['window']}d"
        if col in df.columns:
            vals = df[col].values
            logic = asf.get('logic', '>')
            if logic == '<':
                conditions.append(vals < asf['thresh'])
            elif logic == '>':
                conditions.append(vals > asf['thresh'])
            elif logic == 'Between':
                conditions.append((vals >= asf['thresh']) & (vals <= asf.get('thresh_max', 100.0)))

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

    # Trailing 52w high filter
    if params.get('use_recent_52w', False):
        r52w_lookback = params.get('recent_52w_lookback', 21)
        recent_52w_mask = df['is_52w_high'].rolling(window=r52w_lookback, min_periods=1).max().astype(bool).values
        if params.get('recent_52w_invert', False):
            conditions.append(~recent_52w_mask)
        else:
            conditions.append(recent_52w_mask)

    # Trailing 52w low filter
    if params.get('use_recent_52w_low', False):
        r52w_low_lookback = params.get('recent_52w_low_lookback', 21)
        recent_52w_low_mask = df['is_52w_low'].rolling(window=r52w_low_lookback, min_periods=1).max().astype(bool).values
        if params.get('recent_52w_low_invert', False):
            conditions.append(~recent_52w_low_mask)
        else:
            conditions.append(recent_52w_low_mask)

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
    # Today's return in ATR terms (used by Overbot Vol Spike)
    if params.get('use_today_return', False):
        prev_close = np.roll(df['Close'].values, 1)
        daily_ret_atr = (df['Close'].values - prev_close) / df['ATR'].values
        conditions.append(
            (daily_ret_atr >= params.get('return_min', -100)) &
            (daily_ret_atr <= params.get('return_max', 100))
        )

    # ATR-normalized return filter (used by Weak close > 20d MA)
    if params.get('use_atr_ret_filter', False):
        prev_close = np.roll(df['Close'].values, 1)
        atr_ret = (df['Close'].values - prev_close) / df['ATR'].values
        conditions.append(
            (atr_ret >= params.get('atr_ret_min', -100)) &
            (atr_ret <= params.get('atr_ret_max', 100))
        )
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

    # Reference ticker filter
    if params.get('use_ref_ticker_filter', False) and params.get('ref_filters'):
        for rf in params['ref_filters']:
            col = f"Ref_rank_ret_{rf['window']}d"
            if col in df.columns:
                col_vals = df[col].values
                if rf['logic'] == '<':
                    conditions.append(col_vals < rf['thresh'])
                elif rf['logic'] == '>':
                    conditions.append(col_vals > rf['thresh'])

    # Cross-sectional rank filter
    if params.get('use_xsec_filter', False) and params.get('xsec_filters'):
        for xf in params['xsec_filters']:
            col = f"xsec_rank_ret_{xf['window']}d"
            if col in df.columns:
                col_vals = df[col].values
                if xf['logic'] == '<':
                    conditions.append(col_vals < xf['thresh'])
                elif xf['logic'] == '>':
                    conditions.append(col_vals > xf['thresh'])
                elif xf['logic'] == 'Between':
                    conditions.append((col_vals >= xf['thresh']) & (col_vals <= xf.get('thresh_max', 100.0)))

    # OR filter groups (at least one condition in each group must be true)
    for group in params.get('or_filter_groups', []):
        group_masks = []
        for cond in group:
            ctype = cond.get('type', 'perf')
            window = cond['window']
            logic = cond['logic']
            thresh = cond['thresh']
            if ctype == 'perf':
                col = f"rank_ret_{window}d"
            elif ctype == 'xsec':
                col = f"xsec_rank_ret_{window}d"
            else:
                continue
            if col not in df.columns:
                continue
            col_vals = df[col].values
            if logic == '<':
                group_masks.append(col_vals < thresh)
            elif logic == '>':
                group_masks.append(col_vals > thresh)
        if group_masks:
            # OR: any condition in the group passing is sufficient
            combined_or = np.zeros(n, dtype=bool)
            for m in group_masks:
                combined_or = combined_or | np.nan_to_num(m, nan=0).astype(bool)
            conditions.append(combined_or)

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


# Bump when indicators.py changes in a way that invalidates old cache files.
INDICATOR_CACHE_VERSION = "v1"


def _indicator_cache_path(t_clean, df, params_sig):
    """Per-ticker indicator cache file path. Key = ticker + row count +
    last date + params signature + indicator version."""
    import hashlib
    last_date = df.index[-1].strftime('%Y%m%d') if len(df) else 'empty'
    first_date = df.index[0].strftime('%Y%m%d') if len(df) else 'empty'
    key_str = f"{len(df)}|{first_date}|{last_date}|{params_sig}|{INDICATOR_CACHE_VERSION}"
    key_hash = hashlib.md5(key_str.encode()).hexdigest()[:10]
    cache_dir = os.path.join(parent_dir, "data", "bt_indicator_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{t_clean}_{key_hash}.parquet")


def precompute_all_indicators(_master_dict, _strategies, _sznl_map, _vix_series, _atr_sznl_map=None):
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
    
    # Precompute reference ticker ranks for strategies that use ref_ticker_filter
    ref_ticker_ranks_map = {}  # {ref_ticker: {window: rank_series}}
    for strat in _strategies:
        settings = strat['settings']
        if settings.get('use_ref_ticker_filter', False) and settings.get('ref_filters'):
            ref_ticker = settings.get('ref_ticker', 'IWM').replace('.', '-')
            if ref_ticker not in ref_ticker_ranks_map:
                ref_df = _master_dict.get(ref_ticker)
                if ref_df is not None and len(ref_df) >= 200:
                    try:
                        ref_calc = calculate_indicators(ref_df.copy(), _sznl_map, ref_ticker, market_series, _vix_series)
                        ref_ranks = {}
                        for rf in settings['ref_filters']:
                            col = f"rank_ret_{rf['window']}d"
                            if col in ref_calc.columns:
                                ref_ranks[rf['window']] = ref_calc[col]
                        ref_ticker_ranks_map[ref_ticker] = ref_ranks
                    except Exception:
                        pass

    # Build a merged ref_ticker_ranks dict across all strategies
    all_ref_ticker_ranks = {}
    for ref_ranks in ref_ticker_ranks_map.values():
        all_ref_ticker_ranks.update(ref_ranks)

    # Build cross-sectional rank matrices (if any strategy uses xsec filters or or_filter_groups)
    xsec_windows_needed = set()
    for strat in _strategies:
        s = strat['settings']
        if s.get('use_xsec_filter', False):
            for xf in s.get('xsec_filters', []):
                xsec_windows_needed.add(xf['window'])
        for group in s.get('or_filter_groups', []):
            for cond in group:
                if cond.get('type') == 'xsec':
                    xsec_windows_needed.add(cond['window'])

    xsec_rank_matrices = None
    if xsec_windows_needed:
        RANK_MIN_PERIODS = 252
        # Disk cache: xsec matrices are slow (O(n^2) expanding rank per ticker per window,
        # previously run serially) but depend only on the universe + latest date, so they
        # cache well. Key = (sorted universe, sorted windows, latest shared date, version).
        import hashlib as _hashlib
        _universe_sig = tuple(sorted(_master_dict.keys()))
        _windows_sig = tuple(sorted(xsec_windows_needed))
        _latest_dates = [df.index[-1] for df in _master_dict.values() if df is not None and len(df) > 0]
        _latest = max(_latest_dates).strftime('%Y%m%d') if _latest_dates else 'empty'
        _xsec_key = f"{len(_universe_sig)}|{_latest}|{_windows_sig}|{INDICATOR_CACHE_VERSION}"
        _xsec_hash = _hashlib.md5(_xsec_key.encode()).hexdigest()[:10]
        _xsec_dir = os.path.join(parent_dir, "data", "bt_xsec_cache")
        os.makedirs(_xsec_dir, exist_ok=True)
        _xsec_cache_path = os.path.join(_xsec_dir, f"xsec_{_xsec_hash}.pkl")

        xsec_rank_matrices = None
        if os.path.exists(_xsec_cache_path):
            try:
                import pickle
                with open(_xsec_cache_path, 'rb') as _f:
                    xsec_rank_matrices = pickle.load(_f)
            except Exception:
                xsec_rank_matrices = None

        if xsec_rank_matrices is None:
            _x_status = st.empty()
            _x_status.text(f"⚙️ Computing xsec rank matrices for {len(_master_dict)} tickers...")

            # Parallelize per-ticker expanding rank (the O(n^2) hot path). These are
            # numpy/pandas ops that release the GIL, so threads genuinely help.
            def _compute_xsec_for_ticker(item):
                ticker, df = item
                if df is None or 'Close' not in df.columns or len(df) < 50:
                    return ticker, None
                close = df['Close']
                out = {}
                for w in xsec_windows_needed:
                    ret = close.pct_change(w)
                    out[w] = ret.expanding(min_periods=RANK_MIN_PERIODS).rank(pct=True) * 100.0
                return ticker, out

            rank_dict = {}
            with ThreadPoolExecutor(max_workers=8) as _xpool:
                for ticker, out in _xpool.map(_compute_xsec_for_ticker, _master_dict.items()):
                    if out is None:
                        continue
                    for w, s in out.items():
                        rank_dict.setdefault(w, {})[ticker] = s

            xsec_rank_matrices = {}
            for w in xsec_windows_needed:
                if rank_dict.get(w):
                    mat = pd.DataFrame(rank_dict[w])
                    xsec_rank_matrices[w] = mat.rank(axis=1, pct=True) * 100.0

            try:
                import pickle
                with open(_xsec_cache_path, 'wb') as _f:
                    pickle.dump(xsec_rank_matrices, _f)
            except Exception:
                pass
            _x_status.empty()

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

    # Process tickers in parallel. Base indicators (per-ticker only) are disk-cached
    # in data/bt_indicator_cache so reruns skip the expensive O(n^2) expanding ranks.
    # Cross-ticker deps (xsec rank matrices, ref ticker ranks) are applied post-load.
    def _compute_one(args):
        t_clean, params = args
        df = _master_dict.get(t_clean)
        if df is None or len(df) < 200:
            return None

        params_sig = (
            params['gap'], params['acc'], params['dist'],
            tuple(sorted(params['mas'])),
        )
        cache_path = _indicator_cache_path(t_clean, df, params_sig)

        t_df = None
        if os.path.exists(cache_path):
            try:
                t_df = pd.read_parquet(cache_path)
            except Exception:
                t_df = None

        if t_df is None:
            try:
                # Compute base indicators WITHOUT xsec / ref cols so the cache is
                # reusable across different strategy-set configurations.
                t_df = calculate_indicators(
                    df, _sznl_map, t_clean, market_series, _vix_series,
                    gap_window=params['gap'],
                    acc_window=params['acc'],
                    dist_window=params['dist'],
                    custom_sma_lengths=list(params['mas']),
                    ref_ticker_ranks=None,
                    xsec_rank_matrices=None,
                )
                try:
                    t_df.to_parquet(cache_path)
                except Exception:
                    pass
            except Exception:
                return None

        return (t_clean, t_df)

    items = list(ticker_params.items())
    total = len(items)
    progress = st.progress(0.0) if total > 20 else None
    status = st.empty() if total > 20 else None
    completed = 0
    cache_hits = 0
    update_every = max(1, total // 50)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_compute_one, it): it[0] for it in items}
        for fut in as_completed(futures):
            result = fut.result()
            completed += 1
            if progress is not None and (completed % update_every == 0 or completed == total):
                progress.progress(min(completed / total, 1.0))
                status.text(f"⚙️ Indicators: {completed:,}/{total:,} tickers...")
            if result is None:
                continue
            t_clean, t_df = result

            # --- Cross-ticker layer: apply xsec ranks + ref ticker ranks on top ---
            if xsec_rank_matrices is not None:
                for window, mat in xsec_rank_matrices.items():
                    col = f'xsec_rank_ret_{window}d'
                    if t_clean in mat.columns:
                        t_df[col] = mat[t_clean].reindex(t_df.index).fillna(50.0)
                    else:
                        t_df[col] = 50.0
            if all_ref_ticker_ranks:
                for window, series in all_ref_ticker_ranks.items():
                    t_df[f'Ref_rank_ret_{window}d'] = series.reindex(
                        t_df.index, method='ffill'
                    ).fillna(50.0)

            # Merge ATR seasonal ranks onto the processed DataFrame
            if _atr_sznl_map and t_clean in _atr_sznl_map:
                atr_ranks = _atr_sznl_map[t_clean]
                for col in ATR_SZNL_COLS:
                    if col in atr_ranks.columns:
                        t_df[col] = atr_ranks[col].reindex(t_df.index, method='ffill').fillna(50.0)
                    else:
                        t_df[col] = 50.0
            else:
                for col in ATR_SZNL_COLS:
                    t_df[col] = 50.0
            processed[t_clean] = t_df

    if progress is not None:
        progress.empty()
        status.empty()

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
    
    COMPLETE FIX v4:
    - Entry: GTC limits now use FIXED limit price (based on T+1 Open)
    - Entry: Signal Close now uses correct date/price
    - EXIT: Checks for stop loss and take profit hits
    - Vol Spike: 3-branch ATH/52w overlay from 2026-02-06+ with LOC companion
    """
    if not candidates:
        return pd.DataFrame()
    
    candidates.sort(key=lambda x: x[0])

    # Build price matrix for all relevant tickers (forward-filled by ticker).
    all_tickers = set(c[2] for c in candidates)
    price_matrix = build_price_matrix(processed_dict, all_tickers)

    # Numpy-backed lookup: O(1) access to close price for (date, ticker)
    # vs ~100k pandas `.loc` calls over the course of a 25yr backtest.
    if not price_matrix.empty:
        _pm_values = price_matrix.values
        _pm_date_idx = {d: i for i, d in enumerate(price_matrix.index)}
        _pm_ticker_idx = {t: i for i, t in enumerate(price_matrix.columns)}
    else:
        _pm_values = None
        _pm_date_idx = {}
        _pm_ticker_idx = {}

    def _mtm_price(ticker, date):
        """O(1) price lookup with fallback to processed_dict slice."""
        row = _pm_date_idx.get(date)
        col = _pm_ticker_idx.get(ticker)
        if row is not None and col is not None and _pm_values is not None:
            p = _pm_values[row, col]
            if not pd.isna(p):
                return float(p)
        # Fallback: ticker missing from price_matrix or date out of range
        pos_df = processed_dict.get(ticker)
        if pos_df is None:
            return None
        sub = pos_df.index.asof(date)
        if pd.isna(sub):
            return None
        p = pos_df.at[sub, 'Close']
        return float(p) if not pd.isna(p) else None

    # Track open positions for MTM
    open_positions = []
    realized_pnl = 0.0
    position_last_exit = {}
    last_signal_ts = None
    # OLV cooldown: {t_clean: signal_idx of last primary signal}. Subsequent OLV
    # candidates within OLV_COOLDOWN_DAYS trading days of last primary are skipped.
    olv_last_signal_idx = {}

    # Progress updates — Streamlit Cloud kills silent scripts; also user feedback.
    _total_cands = len(candidates)
    _progress_bar = st.progress(0.0) if _total_cands > 500 else None
    _status = st.empty() if _total_cands > 500 else None
    _update_every = max(250, _total_cands // 200)

    results = []

    for _cand_i, cand in enumerate(candidates):
        if _progress_bar is not None and _cand_i % _update_every == 0:
            _progress_bar.progress(min(_cand_i / _total_cands, 1.0))
            _status.text(f"⚙️ Processing signal {_cand_i:,}/{_total_cands:,}…")

        signal_ts, ticker, t_clean, strat_idx, signal_idx = cand
        signal_date = pd.Timestamp(signal_ts)
        
        strat = strategies[strat_idx]
        settings = strat['settings']
        strat_name = strat['name']
        execution = strat['execution']
        
        # Only block overlapping positions when max_one_pos is True
        max_one_pos = settings.get('max_one_pos', False)
        if max_one_pos:
            pos_key = (strat_name, ticker)
            last_exit_ts = position_last_exit.get(pos_key)
            if last_exit_ts is not None and signal_ts <= last_exit_ts:
                continue

        # OLV cooldown gate — block re-fires within 20 trading days of last primary
        if strat_name == OLV_STRATEGY_NAME:
            _last_idx = olv_last_signal_idx.get(t_clean)
            if _last_idx is not None and (signal_idx - _last_idx) <= OLV_COOLDOWN_DAYS:
                continue
        
        df = processed_dict[t_clean]
        row_data = signal_data[(t_clean, signal_idx)]
        
        atr = row_data['atr']
        if pd.isna(atr) or atr <= 0:
            continue
        
        # ========== ENTRY LOGIC SECTION ==========
        entry_type = settings.get('entry_type', 'T+1 Open')
        hold_days = execution['hold_days']
        
        # Normalize entry type for matching (case-insensitive)
        entry_type_upper = entry_type.upper()
        is_signal_close = 'SIGNAL CLOSE' in entry_type_upper
        is_t1_close = entry_type == 'T+1 Close'
        is_t1_open = entry_type == 'T+1 Open'
        is_t1_conditional = 'T+1 CLOSE IF' in entry_type_upper
        is_limit_open_atr = 'LIMIT' in entry_type_upper and 'OPEN' in entry_type_upper and 'ATR' in entry_type_upper
        is_persistent = 'PERSISTENT' in entry_type_upper or 'GTC' in entry_type_upper
        is_limit_close_anchored = ('LIMIT ORDER' in entry_type_upper) and 'ATR' in entry_type_upper and not is_limit_open_atr
        
        # Skip if we need T+1 data but don't have it
        if signal_idx + 1 >= len(df) and not is_signal_close:
            continue
        
        valid_entry = True
        entry_price = None
        entry_date = None
        entry_row = df.iloc[signal_idx + 1] if signal_idx + 1 < len(df) else None
        
        # --- SIGNAL CLOSE: Enter at today's close ---
        if is_signal_close:
            entry_price = row_data['close']
            entry_date = signal_date
        
        # --- T+1 CLOSE ---
        elif is_t1_close:
            entry_price = entry_row['Close']
            entry_date = entry_row.name
        
        # --- T+1 OPEN ---
        elif is_t1_open:
            entry_price = entry_row['Open']
            entry_date = entry_row.name
        
        # --- T+1 CLOSE IF < SIGNAL CLOSE ---
        elif is_t1_conditional:
            if entry_row['Close'] < row_data['close']:
                entry_price = entry_row['Close']
                entry_date = entry_row.name
            else:
                valid_entry = False
        
        # --- LIMIT (OPEN +/- ATR) with GTC: Fixed limit based on Signal Close ---
        elif is_limit_open_atr and is_persistent:
            base_price = row_data['close']
            limit_offset = 0.5 * atr
            
            if settings['trade_direction'] == 'Long':
                limit_price = base_price - limit_offset
            else:
                limit_price = base_price + limit_offset
            
            found_fill = False
            search_end = min(signal_idx + 1 + hold_days, len(df))
            
            for i in range(signal_idx + 1, search_end):
                check_row = df.iloc[i]
                day_open, day_low, day_high = check_row['Open'], check_row['Low'], check_row['High']
                
                if settings['trade_direction'] == 'Long':
                    if day_open < limit_price:
                        entry_price = day_open
                        entry_date = check_row.name
                        hold_days = max(1, execution['hold_days'] - (i - signal_idx))
                        found_fill = True
                        break
                    elif day_low <= limit_price:
                        entry_price = limit_price
                        entry_date = check_row.name
                        hold_days = max(1, execution['hold_days'] - (i - signal_idx))
                        found_fill = True
                        break
                else:
                    if day_open > limit_price:
                        entry_price = day_open
                        entry_date = check_row.name
                        hold_days = max(1, execution['hold_days'] - (i - signal_idx))
                        found_fill = True
                        break
                    elif day_high >= limit_price:
                        entry_price = limit_price
                        entry_date = check_row.name
                        hold_days = max(1, execution['hold_days'] - (i - signal_idx))
                        found_fill = True
                        break
            
            valid_entry = found_fill
        
        # --- LIMIT (OPEN +/- ATR) Single Day: Limit anchored to Signal Close ---
        elif is_limit_open_atr:
            limit_offset = 0.5 * atr
            limit_base = row_data['close']

            if settings['trade_direction'] == 'Long':
                limit_price = limit_base - limit_offset
                if entry_row['Open'] < limit_price:
                    entry_price = entry_row['Open']
                elif entry_row['Low'] <= limit_price:
                    entry_price = limit_price
                else:
                    valid_entry = False
            else:
                limit_price = limit_base + limit_offset
                if entry_row['Open'] > limit_price:
                    entry_price = entry_row['Open']
                elif entry_row['High'] >= limit_price:
                    entry_price = limit_price
                else:
                    valid_entry = False
            entry_date = entry_row.name
        
        # --- PERSISTENT LIMIT anchored to SIGNAL CLOSE ---
        elif is_persistent or is_limit_close_anchored:
            limit_offset = 0.5 * atr
            limit_base = row_data['close']
            
            if settings['trade_direction'] == 'Long':
                limit_price = limit_base - limit_offset
            else:
                limit_price = limit_base + limit_offset
            
            found_fill = False
            search_end = min(signal_idx + 1 + hold_days, len(df))
            
            for i in range(signal_idx + 1, search_end):
                check_row = df.iloc[i]
                day_open, day_low, day_high = check_row['Open'], check_row['Low'], check_row['High']
                
                if settings['trade_direction'] == 'Long':
                    if day_open < limit_price:
                        entry_price = day_open
                        entry_date = check_row.name
                        hold_days = max(1, execution['hold_days'] - (i - signal_idx))
                        found_fill = True
                        break
                    elif day_low <= limit_price:
                        entry_price = limit_price
                        entry_date = check_row.name
                        hold_days = max(1, execution['hold_days'] - (i - signal_idx))
                        found_fill = True
                        break
                else:
                    if day_open > limit_price:
                        entry_price = day_open
                        entry_date = check_row.name
                        hold_days = max(1, execution['hold_days'] - (i - signal_idx))
                        found_fill = True
                        break
                    elif day_high >= limit_price:
                        entry_price = limit_price
                        entry_date = check_row.name
                        hold_days = max(1, execution['hold_days'] - (i - signal_idx))
                        found_fill = True
                        break
            
            valid_entry = found_fill
        
        # --- DEFAULT: T+1 Open ---
        else:
            if entry_row is not None:
                entry_price = entry_row['Open']
                entry_date = entry_row.name
            else:
                valid_entry = False
        
        if not valid_entry or entry_price is None or pd.isna(entry_price):
            continue
        
        # ========== EXIT LOGIC SECTION ==========
        direction = settings.get('trade_direction', 'Long')
        stop_atr = execution['stop_atr']
        tgt_atr = execution['tgt_atr']
        use_stop = execution.get('use_stop_loss', True)
        use_target = execution.get('use_take_profit', True)
        
        # Calculate stop and target prices
        if direction == 'Long':
            stop_price = entry_price - (atr * stop_atr)
            tgt_price = entry_price + (atr * tgt_atr)
        else:
            stop_price = entry_price + (atr * stop_atr)
            tgt_price = entry_price - (atr * tgt_atr)
        
        # Determine entry_idx
        if entry_date == signal_date:
            entry_idx = signal_idx
        else:
            entry_idx = df.index.get_loc(entry_date)
        
        # Default: time-based exit
        max_exit_idx = min(entry_idx + hold_days, len(df) - 1)
        exit_idx = max_exit_idx
        exit_price = df.iloc[exit_idx]['Close']
        exit_date = df.index[exit_idx]
        exit_type = "Time"
        
        # Check for stop/target hits day by day
        if use_stop or use_target:
            for check_idx in range(entry_idx + 1, max_exit_idx + 1):
                check_row = df.iloc[check_idx]
                day_low = check_row['Low']
                day_high = check_row['High']
                
                if direction == 'Long':
                    if use_stop and day_low <= stop_price:
                        exit_price = stop_price
                        exit_date = check_row.name
                        exit_idx = check_idx
                        exit_type = "Stop"
                        break
                    if use_target and day_high >= tgt_price:
                        exit_price = tgt_price
                        exit_date = check_row.name
                        exit_idx = check_idx
                        exit_type = "Target"
                        break
                else:
                    if use_stop and day_high >= stop_price:
                        exit_price = stop_price
                        exit_date = check_row.name
                        exit_idx = check_idx
                        exit_type = "Stop"
                        break
                    if use_target and day_low <= tgt_price:
                        exit_price = tgt_price
                        exit_date = check_row.name
                        exit_idx = check_idx
                        exit_type = "Target"
                        break
        
        # ========== CALCULATE REAL-TIME MTM EQUITY ==========
        # Only recompute when signal date changes — all same-day signals share one equity snapshot
        if signal_ts != last_signal_ts:
            still_open = []
            for pos in open_positions:
                if signal_date >= pos['exit_date']:
                    exit_price_mtm = _mtm_price(pos['t_clean'], pos['exit_date'])
                    if exit_price_mtm is None:
                        # Last-resort fallback: use processed_dict by integer idx.
                        pos_df = processed_dict.get(pos['t_clean'])
                        exit_price_mtm = pos_df.iloc[pos['exit_idx']]['Close'] if pos_df is not None else pos['entry_price']

                    if pos['direction'] == 'Long':
                        realized_pnl += (exit_price_mtm - pos['entry_price']) * pos['shares']
                    else:
                        realized_pnl += (pos['entry_price'] - exit_price_mtm) * pos['shares']
                else:
                    still_open.append(pos)

            open_positions = still_open

            # Calculate unrealized P&L
            unrealized_pnl = 0.0
            for pos in open_positions:
                current_price = _mtm_price(pos['t_clean'], signal_date)
                if current_price is None:
                    continue

                if pos['direction'] == 'Long':
                    unrealized_pnl += (current_price - pos['entry_price']) * pos['shares']
                else:
                    unrealized_pnl += (pos['entry_price'] - current_price) * pos['shares']

            current_equity = starting_equity + realized_pnl + unrealized_pnl
            last_signal_ts = signal_ts
        
        risk_bps = execution['risk_bps']
        base_risk = current_equity * risk_bps / 10000

        # ========== STRATEGY-SPECIFIC RISK ADJUSTMENTS ==========
        _vol_spike_skip_primary = False
        # Overbot Vol Spike: always emit a LOC companion alongside the primary
        # short. LOC uses the same signal criteria — only differs in entry:
        # fills at T+1 Close when T+1 Close > Signal Close.
        _vol_spike_emit_loc = (strat_name == "Overbot Vol Spike")
        # Oversold Low Volume: emit a LOC companion (long) on T+1 Close when
        # T+1 Close < Signal Close (buying the continued dip).
        _olv_emit_loc = (strat_name == OLV_STRATEGY_NAME)

        # Record this primary for OLV cooldown bookkeeping. Filters upstream
        # already vetted the signal; LOC companion fires within this iteration
        # and is NOT gated by cooldown.
        if strat_name == OLV_STRATEGY_NAME:
            olv_last_signal_idx[t_clean] = signal_idx

        if strat_name == "Weak Close Decent Sznls":
            sznl_val = row_data['sznl']
            if sznl_val >= 65:
                base_risk *= 1.5
            elif sznl_val >= 33:
                base_risk *= 0.66 if sznl_val < 50 else 1.0

        dist = atr * stop_atr
        action = "BUY" if direction == 'Long' else "SELL SHORT"

        if pd.isna(dist) or dist <= 0:
            continue

        # ========== PRIMARY TRADE ==========
        if not _vol_spike_skip_primary:
            try:
                shares = int(base_risk / dist)
            except (ValueError, OverflowError):
                shares = 0

            if shares > 0:
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
                    'ticker': ticker, 't_clean': t_clean,
                    'entry_date': entry_date, 'entry_price': entry_price,
                    'shares': shares, 'direction': 'Long' if action == 'BUY' else 'Short',
                    'exit_idx': exit_idx, 'exit_date': exit_date, 'strat_name': strat_name
                })
                if max_one_pos:
                    position_last_exit[(strat_name, ticker)] = exit_date.value

                t1_open = entry_row['Open'] if entry_row is not None else entry_price

                results.append({
                    "Date": signal_date, "Entry Date": entry_date,
                    "Exit Date": exit_date, "Exit Type": exit_type,
                    "Time Stop": time_stop_date, "Strategy": strat_name,
                    "Ticker": ticker, "Action": action,
                    "Entry Criteria": entry_type, "Price": entry_price,
                    "Exit Price": exit_price,
                    "Shares": shares, "PnL": pnl, "ATR": atr,
                    "stop_atr": stop_atr, "tgt_atr": tgt_atr,
                    "T+1 Open": t1_open, "Signal Close": row_data['close'],
                    "Range %": row_data['range_pct'],
                    "Equity at Signal": current_equity,
                    "Risk $": base_risk, "Risk bps": risk_bps
                })

        # ========== LOC COMPANION TRADE (Vol Spike) ==========
        # Same signal criteria as primary; only entry differs. Fills at T+1 Close
        # if T+1 Close > Signal Close (simple confirmation the fade didn't invalidate).
        if _vol_spike_emit_loc:
            loc_threshold = row_data['close']
            t1_idx = signal_idx + 1

            if t1_idx < len(df):
                t1_close = df.iloc[t1_idx]['Close']

                if t1_close > loc_threshold:
                    # LOC fills at T+1 close
                    loc_entry_price = t1_close
                    loc_entry_date = df.index[t1_idx]
                    loc_entry_idx = t1_idx

                    # Base risk for LOC (always standard bps, not reduced)
                    loc_risk = current_equity * risk_bps / 10000
                    loc_shares = int(loc_risk / dist) if dist > 0 else 0

                    if loc_shares > 0:
                        # Stop/target from LOC entry price
                        if direction == 'Long':
                            loc_stop = loc_entry_price - (atr * stop_atr)
                            loc_tgt = loc_entry_price + (atr * tgt_atr)
                        else:
                            loc_stop = loc_entry_price + (atr * stop_atr)
                            loc_tgt = loc_entry_price - (atr * tgt_atr)

                        # Exit logic for LOC position
                        loc_hold = execution['hold_days']
                        loc_max_exit_idx = min(loc_entry_idx + loc_hold, len(df) - 1)
                        loc_exit_idx = loc_max_exit_idx
                        loc_exit_price = df.iloc[loc_exit_idx]['Close']
                        loc_exit_date = df.index[loc_exit_idx]
                        loc_exit_type = "Time"

                        use_stop_loc = execution.get('use_stop_loss', True)
                        use_tgt_loc = execution.get('use_take_profit', True)

                        if use_stop_loc or use_tgt_loc:
                            for ci in range(loc_entry_idx + 1, loc_max_exit_idx + 1):
                                cr = df.iloc[ci]
                                if direction == 'Long':
                                    if use_stop_loc and cr['Low'] <= loc_stop:
                                        loc_exit_price, loc_exit_date, loc_exit_idx, loc_exit_type = loc_stop, cr.name, ci, "Stop"
                                        break
                                    if use_tgt_loc and cr['High'] >= loc_tgt:
                                        loc_exit_price, loc_exit_date, loc_exit_idx, loc_exit_type = loc_tgt, cr.name, ci, "Target"
                                        break
                                else:
                                    if use_stop_loc and cr['High'] >= loc_stop:
                                        loc_exit_price, loc_exit_date, loc_exit_idx, loc_exit_type = loc_stop, cr.name, ci, "Stop"
                                        break
                                    if use_tgt_loc and cr['Low'] <= loc_tgt:
                                        loc_exit_price, loc_exit_date, loc_exit_idx, loc_exit_type = loc_tgt, cr.name, ci, "Target"
                                        break

                        if action == "BUY":
                            loc_pnl = ((loc_exit_price - loc_entry_price) * loc_shares).round(0)
                        else:
                            loc_pnl = ((loc_entry_price - loc_exit_price) * loc_shares).round(0)

                        loc_ts_idx = loc_entry_idx + loc_hold
                        if loc_ts_idx < len(df):
                            loc_time_stop = df.index[loc_ts_idx]
                        else:
                            loc_time_stop = loc_entry_date + BusinessDay(loc_hold)

                        open_positions.append({
                            'ticker': ticker, 't_clean': t_clean,
                            'entry_date': loc_entry_date, 'entry_price': loc_entry_price,
                            'shares': loc_shares, 'direction': 'Long' if action == 'BUY' else 'Short',
                            'exit_idx': loc_exit_idx, 'exit_date': loc_exit_date,
                            'strat_name': strat_name + " (LOC)"
                        })
                        if max_one_pos:
                            position_last_exit[(strat_name, ticker)] = loc_exit_date.value

                        results.append({
                            "Date": signal_date, "Entry Date": loc_entry_date,
                            "Exit Date": loc_exit_date, "Exit Type": loc_exit_type,
                            "Time Stop": loc_time_stop, "Strategy": strat_name + " (LOC)",
                            "Ticker": ticker, "Action": action,
                            "Entry Criteria": "LOC (T+1 Close > Signal Close)",
                            "Price": loc_entry_price, "Exit Price": loc_exit_price,
                            "Shares": loc_shares,
                            "PnL": loc_pnl, "ATR": atr,
                            "stop_atr": stop_atr, "tgt_atr": tgt_atr,
                            "T+1 Open": df.iloc[t1_idx]['Open'],
                            "Signal Close": row_data['close'],
                            "Range %": row_data['range_pct'],
                            "Equity at Signal": current_equity,
                            "Risk $": loc_risk, "Risk bps": risk_bps
                        })

        # ========== LOC COMPANION TRADE (OLV) ==========
        # OLV is Long: companion fires when T+1 Close < Signal Close (buying
        # the continued dip). Cooldown applies to the NEXT primary, not this
        # companion (which shares the current iteration's signal).
        if _olv_emit_loc:
            loc_threshold = row_data['close']
            t1_idx = signal_idx + 1

            if t1_idx < len(df):
                t1_close = df.iloc[t1_idx]['Close']

                if t1_close < loc_threshold:
                    loc_entry_price = t1_close
                    loc_entry_date = df.index[t1_idx]
                    loc_entry_idx = t1_idx

                    loc_risk = current_equity * risk_bps / 10000
                    loc_shares = int(loc_risk / dist) if dist > 0 else 0

                    if loc_shares > 0:
                        # Long stops/targets from LOC entry price
                        loc_stop = loc_entry_price - (atr * stop_atr)
                        loc_tgt = loc_entry_price + (atr * tgt_atr)

                        loc_hold = execution['hold_days']
                        loc_max_exit_idx = min(loc_entry_idx + loc_hold, len(df) - 1)
                        loc_exit_idx = loc_max_exit_idx
                        loc_exit_price = df.iloc[loc_exit_idx]['Close']
                        loc_exit_date = df.index[loc_exit_idx]
                        loc_exit_type = "Time"

                        use_stop_loc = execution.get('use_stop_loss', True)
                        use_tgt_loc = execution.get('use_take_profit', True)

                        if use_stop_loc or use_tgt_loc:
                            for ci in range(loc_entry_idx + 1, loc_max_exit_idx + 1):
                                cr = df.iloc[ci]
                                if use_stop_loc and cr['Low'] <= loc_stop:
                                    loc_exit_price, loc_exit_date, loc_exit_idx, loc_exit_type = loc_stop, cr.name, ci, "Stop"
                                    break
                                if use_tgt_loc and cr['High'] >= loc_tgt:
                                    loc_exit_price, loc_exit_date, loc_exit_idx, loc_exit_type = loc_tgt, cr.name, ci, "Target"
                                    break

                        loc_pnl = ((loc_exit_price - loc_entry_price) * loc_shares).round(0)

                        loc_ts_idx = loc_entry_idx + loc_hold
                        if loc_ts_idx < len(df):
                            loc_time_stop = df.index[loc_ts_idx]
                        else:
                            loc_time_stop = loc_entry_date + BusinessDay(loc_hold)

                        open_positions.append({
                            'ticker': ticker, 't_clean': t_clean,
                            'entry_date': loc_entry_date, 'entry_price': loc_entry_price,
                            'shares': loc_shares, 'direction': 'Long',
                            'exit_idx': loc_exit_idx, 'exit_date': loc_exit_date,
                            'strat_name': strat_name + " (LOC)"
                        })

                        results.append({
                            "Date": signal_date, "Entry Date": loc_entry_date,
                            "Exit Date": loc_exit_date, "Exit Type": loc_exit_type,
                            "Time Stop": loc_time_stop, "Strategy": strat_name + " (LOC)",
                            "Ticker": ticker, "Action": "BUY",
                            "Entry Criteria": "LOC (T+1 Close < Signal Close)",
                            "Price": loc_entry_price, "Exit Price": loc_exit_price,
                            "Shares": loc_shares,
                            "PnL": loc_pnl, "ATR": atr,
                            "stop_atr": stop_atr, "tgt_atr": tgt_atr,
                            "T+1 Open": df.iloc[t1_idx]['Open'],
                            "Signal Close": row_data['close'],
                            "Range %": row_data['range_pct'],
                            "Equity at Signal": current_equity,
                            "Risk $": loc_risk, "Risk bps": risk_bps
                        })

    if _progress_bar is not None:
        _progress_bar.empty()
        _status.empty()

    if not results:
        return pd.DataFrame()

    sig_df = pd.DataFrame(results)

    sig_df['Date'] = pd.to_datetime(sig_df['Date'])
    sig_df['Entry Date'] = pd.to_datetime(sig_df['Entry Date'])
    sig_df['Exit Date'] = pd.to_datetime(sig_df['Exit Date'])
    sig_df['Time Stop'] = pd.to_datetime(sig_df['Time Stop'])

    # Global aggregate daily risk cap across ALL strategies.
    # Mirrors daily_scan.py / local_overflow_scan.py: for each signal date,
    # if total Risk $ from all strategies' signals exceeds DAILY_RISK_CAP_BPS
    # of equity at that date, scale every signal's Shares + PnL + Risk $ down.
    # Cap scales dynamically with "Equity at Signal" (first signal's equity
    # on that date is representative).
    if DAILY_RISK_CAP_BPS and len(sig_df) > 0:
        cap_bps = DAILY_RISK_CAP_BPS
        sig_df = sig_df.sort_values(by="Date").reset_index(drop=True)
        grouped = sig_df.groupby('Date', sort=False)
        for date, idx in grouped.groups.items():
            rows = sig_df.loc[idx]
            day_equity = float(rows['Equity at Signal'].iloc[0])
            cap_dollars = day_equity * cap_bps / 10000.0
            total_risk = float(rows['Risk $'].sum())
            if total_risk > cap_dollars > 0:
                scale = cap_dollars / total_risk
                sig_df.loc[idx, 'Shares']  = (sig_df.loc[idx, 'Shares']  * scale).round().astype(int)
                sig_df.loc[idx, 'PnL']     = (sig_df.loc[idx, 'PnL']     * scale).round()
                sig_df.loc[idx, 'Risk $']  = sig_df.loc[idx, 'Risk $']   * scale

    return sig_df.sort_values(by="Exit Date")

def get_daily_mtm_series(sig_df, master_dict, start_date=None):
    """
    Optimized daily MTM calculation using vectorized operations.
    
    FIXED: Use explicit column access instead of positional (_3, _4)
    to handle new "Exit Type" column.
    """
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
    
    # Process trades using array-based iteration (much faster than iterrows)
    _tickers = sig_df['Ticker'].values
    _actions = sig_df['Action'].values
    _shares = sig_df['Shares'].values
    _entry_dates = pd.to_datetime(sig_df['Entry Date']).values
    _exit_dates = pd.to_datetime(sig_df['Exit Date']).values
    _entry_prices = sig_df['Price'].values
    _pnl_values = sig_df['PnL'].values

    for i in range(len(sig_df)):
        ticker = _tickers[i]
        action = _actions[i]
        shares = float(_shares[i])
        entry_date = _entry_dates[i]
        exit_date = _exit_dates[i]
        entry_price = float(_entry_prices[i])

        if ticker not in price_cache:
            exit_ts = pd.Timestamp(exit_date)
            if exit_ts in daily_pnl.index:
                daily_pnl[exit_ts] += float(_pnl_values[i])
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

        # Subsequent days - vectorized daily changes
        if len(trade_dates) > 1:
            diffs = trade_closes.diff().dropna()
            if action == "SELL SHORT":
                diffs = -diffs
            pnl_contrib = (diffs * shares).dropna()
            common_idx = pnl_contrib.index.intersection(daily_pnl.index)
            if len(common_idx) > 0:
                daily_pnl[common_idx] += pnl_contrib[common_idx]

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
    n_days = len(all_dates)

    _dates = pd.to_datetime(sig_df['Date'].values)
    _exit_dates = pd.to_datetime(sig_df['Exit Date'].values)
    _prices = sig_df['Price'].values.astype(float)
    _shares_arr = sig_df['Shares'].values.astype(float)
    _actions = sig_df['Action'].values

    # Map trade dates to integer positions in all_dates with a vectorized searchsorted.
    _all_vals = all_dates.values.astype('datetime64[ns]')
    start_idx = np.searchsorted(_all_vals, _dates.values.astype('datetime64[ns]'))
    end_idx = np.searchsorted(_all_vals, _exit_dates.values.astype('datetime64[ns]'))

    dollar_vals = _prices * _shares_arr
    long_mask = _actions == 'BUY'

    # Difference-array trick: for each trade, +val at entry, -val at exit+1, then cumsum.
    # Turns O(trades × days) into O(trades + days).
    long_delta = np.zeros(n_days + 1, dtype=float)
    short_delta = np.zeros(n_days + 1, dtype=float)
    np.add.at(long_delta,  start_idx[long_mask],      dollar_vals[long_mask])
    np.add.at(long_delta,  end_idx[long_mask] + 1,   -dollar_vals[long_mask])
    np.add.at(short_delta, start_idx[~long_mask],     dollar_vals[~long_mask])
    np.add.at(short_delta, end_idx[~long_mask] + 1,  -dollar_vals[~long_mask])

    long_exp = long_delta[:n_days].cumsum()
    short_exp = short_delta[:n_days].cumsum()

    exposure_df = pd.DataFrame(
        {'Long Exposure': long_exp, 'Short Exposure': short_exp},
        index=all_dates,
    )

    exposure_df['Net Exposure'] = exposure_df['Long Exposure'] - exposure_df['Short Exposure']
    exposure_df['Gross Exposure'] = exposure_df['Long Exposure'] + exposure_df['Short Exposure']

    if starting_equity is not None:
        pnl_by_exit = sig_df.groupby('Exit Date')['PnL'].sum()
        cum_realized = pnl_by_exit.reindex(all_dates, fill_value=0).cumsum()
        equity_series = starting_equity + cum_realized
        
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

def create_portfolio_breakdown_charts(sig_df):
    """Create cycle year, monthly, and day-of-week bar charts using log % returns."""
    if sig_df.empty:
        return None, None, None

    df = sig_df.copy()
    df['Exit Date'] = pd.to_datetime(df['Exit Date'])
    df['Log Return'] = np.log1p(df['PnL'] / df['Equity at Signal']) * 100  # in log %

    def make_bar_chart(agg_df, title):
        colors = ['#00CC00' if v >= 0 else '#CC0000' for v in agg_df['LogPnL']]
        fig = go.Figure(go.Bar(
            x=agg_df['Label'], y=agg_df['LogPnL'],
            marker_color=colors,
            text=[f"{v:+.2f}%<br>{n:.0f} trades<br>{w:.0%} WR" for v, n, w in
                  zip(agg_df['LogPnL'], agg_df['Trades'], agg_df['WinRate'])],
            textposition='outside'
        ))
        fig.update_layout(
            title=title,
            yaxis_title="Cumulative Log Return (%)", height=400,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        return fig

    def agg_group(grouped):
        return grouped.agg(
            LogPnL=('Log Return', 'sum'),
            Trades=('Log Return', 'count'),
            WinRate=('PnL', lambda x: (x > 0).mean())
        )

    # --- Presidential Cycle ---
    cycle_map = {0: 'Election', 1: 'Post-Election', 2: 'Midterm', 3: 'Pre-Election'}
    df['Cycle'] = df['Exit Date'].dt.year % 4
    df['Cycle Label'] = df['Cycle'].map(cycle_map)
    cycle_agg = agg_group(df.groupby('Cycle Label'))
    cycle_agg = cycle_agg.reindex(['Post-Election', 'Midterm', 'Pre-Election', 'Election'])
    cycle_agg['Label'] = cycle_agg.index
    fig_cycle = make_bar_chart(cycle_agg, "Log Return by Presidential Cycle Year")

    # --- Monthly ---
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df['Month'] = df['Exit Date'].dt.month
    month_agg = agg_group(df.groupby('Month'))
    month_agg = month_agg.reindex(range(1, 13))
    month_agg['Label'] = month_names
    fig_month = make_bar_chart(month_agg, "Log Return by Month")

    # --- Day of Week ---
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    df['DOW'] = df['Exit Date'].dt.dayofweek
    dow_agg = agg_group(df.groupby('DOW'))
    dow_agg = dow_agg.reindex(range(5))
    dow_agg['Label'] = dow_names
    fig_dow = make_bar_chart(dow_agg, "Log Return by Day of Week (Exit Day)")

    return fig_cycle, fig_month, fig_dow


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
    selected_year = st.sidebar.slider("Select Start Year", 1997, current_year, current_year - 2)
    default_date = datetime.date(selected_year, 1, 1)
    
    with st.sidebar.form("backtest_form"):
        user_start_date = st.date_input("Backtest Start Date", value=default_date, min_value=datetime.date(1997, 1, 1))
        starting_equity = st.number_input("Starting Equity ($)", value=ACCOUNT_VALUE, step=10000)
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
        atr_sznl_map = load_atr_seasonal_map()
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

        # Step 1: Load from disk cache (always check for anything not in session_state)
        bt_cache_dir = os.path.join(parent_dir, "data", "bt_price_cache")
        os.makedirs(bt_cache_dir, exist_ok=True)
        loaded = 0
        for t in long_term_list:
            if t not in st.session_state['backtest_data']:
                pq_path = os.path.join(bt_cache_dir, f"{t}.parquet")
                if os.path.exists(pq_path):
                    try:
                        t_df = pd.read_parquet(pq_path)
                        st.session_state['backtest_data'][t] = t_df
                        loaded += 1
                    except Exception:
                        pass
        if loaded:
            st.caption(f"Loaded {loaded} tickers from disk cache")

        # Step 2: Download anything still missing (no disk cache exists)
        existing = set(st.session_state['backtest_data'].keys())
        missing = list(set(long_term_list) - existing)

        if missing:
            st.write(f"📥 Downloading {len(missing)} new tickers (have {len(existing)} cached)...")
            data = download_historical_data(missing, start_date="2000-01-01")
            st.session_state['backtest_data'].update(data)
            for t, t_df in data.items():
                try:
                    t_df.to_parquet(os.path.join(bt_cache_dir, f"{t}.parquet"))
                except Exception:
                    pass
            st.success(f"✅ Downloaded {len(data)} new tickers, saved to disk cache.")

        # Step 3: Incrementally update stale cached data (only fetches delta)
        stale_count = update_stale_cache(bt_cache_dir, long_term_list, st.session_state['backtest_data'])
        if stale_count:
            st.caption(f"🔄 Updated {stale_count} stale tickers with recent data")

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
        processed_dict = precompute_all_indicators(master_dict, strategies, sznl_map, vix_series, atr_sznl_map)
        st.write(f"   Processed {len(processed_dict)} tickers in {time.time()-t0:.1f}s")

        st.write("🔍 **Phase 2:** Finding signals...")
        t0 = time.time()
        candidates, signal_data = generate_candidates_fast(processed_dict, strategies, sznl_map, user_start_date)
        st.write(f"   Found {len(candidates):,} candidates in {time.time()-t0:.1f}s")

        st.write("📈 **Phase 3:** Processing with dynamic MTM-based sizing...")
        t0 = time.time()
        sig_df = process_signals_fast(
            candidates, signal_data, processed_dict, strategies, starting_equity
        )
        st.write(f"   Executed {len(sig_df):,} trades in {time.time()-t0:.1f}s")

        if not sig_df.empty:
            st.success(f"✅ Backtest complete: {len(sig_df):,} trades")

            # Cache sig_df for cross-page analysis (e.g., risk_dashboard.py regime analysis)
            try:
                cache_dir = os.path.join(parent_dir, "data")
                os.makedirs(cache_dir, exist_ok=True)
                sig_df_cache_path = os.path.join(cache_dir, "backtest_sig_df.parquet")
                sig_df.to_parquet(sig_df_cache_path, index=False)
                st.caption(f"📦 Cached {len(sig_df):,} trades to data/backtest_sig_df.parquet")
            except Exception as e:
                st.caption(f"⚠️ Could not cache sig_df: {e}")

            today = pd.Timestamp(datetime.date.today())
            open_mask = sig_df['Time Stop'] >= today
            open_df = sig_df[open_mask].copy()

            if not open_df.empty:
                current_prices, open_pnls, current_values = [], [], []
                for row in open_df.itertuples():
                    last_close = None
                    ticker_raw = row.Ticker
                    t_clean = ticker_raw.replace('.', '-')

                    for t_key in [t_clean, ticker_raw, t_clean.upper(), ticker_raw.upper()]:
                        t_df = master_dict.get(t_key)
                        if t_df is not None and not t_df.empty:
                            temp = t_df.copy()
                            if isinstance(temp.columns, pd.MultiIndex):
                                temp.columns = [c[0] if isinstance(c, tuple) else c for c in temp.columns]
                            temp.columns = [c.capitalize() for c in temp.columns]
                            if 'Close' in temp.columns:
                                close_clean = temp['Close'].dropna()
                                if not close_clean.empty:
                                    last_close = close_clean.iloc[-1]
                                    break

                    if last_close is None or pd.isna(last_close):
                        for t_key in [t_clean, ticker_raw]:
                            p_df = processed_dict.get(t_key)
                            if p_df is not None and not p_df.empty and 'Close' in p_df.columns:
                                close_clean = p_df['Close'].dropna()
                                if not close_clean.empty:
                                    last_close = close_clean.iloc[-1]
                                    break

                    if last_close is None or pd.isna(last_close):
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
                # Filter out low-sample strategies — they clutter the chart with noise
                MIN_SAMPLE = 200
                strat_counts = sig_df.groupby('Strategy').size()
                eligible_strats = strat_counts[strat_counts >= MIN_SAMPLE].index.tolist()
                excluded = strat_counts[strat_counts < MIN_SAMPLE]
                filtered_sig = sig_df[sig_df['Strategy'].isin(eligible_strats)]
                strat_pnl = filtered_sig.pivot_table(index='Exit Date', columns='Strategy', values='PnL', aggfunc='sum')

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
                    if len(excluded) > 0:
                        st.caption(
                            f"Hidden (n < {MIN_SAMPLE}): "
                            + ", ".join(f"{s} ({c})" for s, c in excluded.items())
                        )
                else:
                    st.info(f"No strategy has >= {MIN_SAMPLE} trades.")

            st.subheader("⚖️ Exposure Over Time (% of Equity)")
            exposure_df = calculate_daily_exposure(sig_df, starting_equity=starting_equity)
            if not exposure_df.empty:
                # Interactive Plotly chart for exposure
                fig_exposure = go.Figure()
                
                colors = {
                    'Long Exposure %': '#00CC00',       # Green
                    'Short Exposure %': '#CC0000',      # Red
                    'Net Exposure %': '#0066CC',        # Blue
                    'Gross Exposure %': '#FF9900'       # Orange
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

            st.divider()
            st.subheader("📊 Portfolio Breakdown")
            fig_cycle, fig_month, fig_dow = create_portfolio_breakdown_charts(sig_df)

            if fig_cycle:
                st.plotly_chart(fig_cycle, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    if fig_month:
                        st.plotly_chart(fig_month, use_container_width=True)
                with col2:
                    if fig_dow:
                        st.plotly_chart(fig_dow, use_container_width=True)
            # -----------------------------------------------------------------
            # DRAWDOWN DEEP DIVE
            # -----------------------------------------------------------------
            st.divider()
            st.subheader("🔍 Drawdown Deep Dive")
            st.caption("Worst rolling 3-month periods: what happened, what was the market doing, and which strategies drove the losses.")

            dd_window = st.selectbox("Rolling window", ["63 days (~3 mo)", "42 days (~2 mo)", "126 days (~6 mo)"],
                                     index=0, key='dd_window')
            dd_window_days = {'63 days (~3 mo)': 63, '42 days (~2 mo)': 42, '126 days (~6 mo)': 126}[dd_window]

            daily_pnl = get_daily_mtm_series(sig_df, master_dict, start_date=user_start_date)
            if not daily_pnl.empty:
                equity_curve = starting_equity + daily_pnl.cumsum()
                rolling_return = equity_curve - equity_curve.shift(dd_window_days)
                rolling_return = rolling_return.dropna()

                if not rolling_return.empty:
                    # Find top 5 worst periods (non-overlapping)
                    worst_periods = []
                    remaining = rolling_return.copy()
                    for _ in range(5):
                        if remaining.empty:
                            break
                        worst_end = remaining.idxmin()
                        worst_start = worst_end - pd.tseries.offsets.BusinessDay(dd_window_days)
                        worst_pnl = remaining[worst_end]
                        if pd.isna(worst_pnl):
                            break
                        worst_periods.append({
                            'start': worst_start,
                            'end': worst_end,
                            'pnl': worst_pnl,
                        })
                        # Exclude window + 63 business day buffer to force distinct drawdowns
                        cooloff_end = worst_end + pd.tseries.offsets.BusinessDay(63)
                        mask = (remaining.index >= worst_start) & (remaining.index <= cooloff_end)
                        remaining = remaining[~mask]

                    if worst_periods:
                        # Get SPY data for market context
                        spy_close = None
                        spy_raw = master_dict.get('SPY')
                        if spy_raw is not None and not spy_raw.empty:
                            tmp = spy_raw.copy()
                            if isinstance(tmp.columns, pd.MultiIndex):
                                tmp.columns = tmp.columns.get_level_values(0)
                            tmp.columns = [c.capitalize() for c in tmp.columns]
                            spy_close = tmp['Close']

                        for rank, wp in enumerate(worst_periods, 1):
                            w_start, w_end, w_pnl = wp['start'], wp['end'], wp['pnl']
                            eq_start = equity_curve.asof(w_start)
                            pct_loss = w_pnl / eq_start * 100 if eq_start > 0 else 0

                            with st.expander(
                                f"#{rank}: {w_start.strftime('%Y-%m-%d')} to {w_end.strftime('%Y-%m-%d')} — "
                                f"**${w_pnl:,.0f}** ({pct_loss:+.1f}%)",
                                expanded=(rank == 1)
                            ):
                                # Market context
                                if spy_close is not None:
                                    spy_start_val = spy_close.asof(w_start)
                                    spy_end_val = spy_close.asof(w_end)
                                    spy_ret = (spy_end_val / spy_start_val - 1) * 100 if spy_start_val > 0 else 0
                                    spy_high = spy_close.loc[w_start:w_end].max()
                                    spy_low = spy_close.loc[w_start:w_end].min()
                                    spy_dd = (spy_low / spy_high - 1) * 100
                                    mc1, mc2, mc3, mc4 = st.columns(4)
                                    mc1.metric("SPY Return", f"{spy_ret:+.1f}%")
                                    mc2.metric("SPY Peak-to-Trough", f"{spy_dd:+.1f}%")
                                    mc3.metric("Portfolio Loss", f"${w_pnl:,.0f}")
                                    mc4.metric("Portfolio Drawdown", f"{pct_loss:+.1f}%")

                                # Equity + SPY overlay chart
                                from plotly.subplots import make_subplots
                                period_eq = equity_curve.loc[w_start:w_end].dropna()
                                if not period_eq.empty:
                                    fig_dd = make_subplots(specs=[[{"secondary_y": True}]])
                                    fig_dd.add_trace(go.Scatter(
                                        x=period_eq.index, y=period_eq.values,
                                        name='Portfolio Equity', line=dict(color='#EF553B', width=2),
                                    ), secondary_y=False)
                                    if spy_close is not None:
                                        spy_period = spy_close.loc[w_start:w_end].dropna()
                                        if not spy_period.empty:
                                            fig_dd.add_trace(go.Scatter(
                                                x=spy_period.index, y=spy_period.values,
                                                name='SPY', line=dict(color='#636EFA', width=1.5, dash='dot'),
                                            ), secondary_y=True)
                                    fig_dd.update_layout(
                                        height=300, margin=dict(l=10, r=10, t=10, b=10),
                                        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
                                        hovermode='x unified',
                                    )
                                    fig_dd.update_yaxes(title_text='Equity ($)', tickformat='$,.0f', secondary_y=False)
                                    fig_dd.update_yaxes(title_text='SPY', tickformat='$,.0f', secondary_y=True)
                                    st.plotly_chart(fig_dd, use_container_width=True)

                                # Strategy attribution for this period
                                period_trades = sig_df[
                                    (sig_df['Entry Date'] <= w_end) & (sig_df['Exit Date'] >= w_start)
                                ].copy()

                                if not period_trades.empty:
                                    strat_attr = period_trades.groupby('Strategy').agg(
                                        Trades=('PnL', 'count'),
                                        Total_PnL=('PnL', 'sum'),
                                        Win_Rate=('PnL', lambda x: (x > 0).mean()),
                                        Avg_PnL=('PnL', 'mean'),
                                    ).sort_values('Total_PnL')

                                    st.markdown("**Strategy Attribution**")
                                    col_chart, col_table = st.columns([1, 1])

                                    with col_chart:
                                        fig_attr = go.Figure(go.Bar(
                                            x=strat_attr['Total_PnL'].values,
                                            y=strat_attr.index,
                                            orientation='h',
                                            marker_color=['#EF553B' if v < 0 else '#00CC96' for v in strat_attr['Total_PnL']],
                                            hovertemplate='%{y}: $%{x:,.0f}<extra></extra>',
                                        ))
                                        fig_attr.update_layout(
                                            height=max(200, len(strat_attr) * 35),
                                            margin=dict(l=10, r=10, t=10, b=10),
                                            xaxis_title='PnL ($)', xaxis_tickformat='$,.0f',
                                        )
                                        st.plotly_chart(fig_attr, use_container_width=True)

                                    with col_table:
                                        st.dataframe(strat_attr.style.format({
                                            'Total_PnL': '${:,.0f}',
                                            'Win_Rate': '{:.0%}',
                                            'Avg_PnL': '${:,.0f}',
                                        }), use_container_width=True)

                                    # Worst individual trades in this period
                                    worst_trades = period_trades.nsmallest(10, 'PnL')[
                                        ['Date', 'Strategy', 'Ticker', 'Action', 'PnL', 'Risk $']
                                    ].copy()
                                    if not worst_trades.empty:
                                        st.markdown("**Worst Trades in Period**")
                                        st.dataframe(worst_trades.style.format({
                                            'PnL': '${:,.0f}',
                                            'Risk $': '${:,.0f}',
                                            'Date': '{:%Y-%m-%d}',
                                        }), use_container_width=True, hide_index=True)
                                else:
                                    st.info("No trades overlapped with this period.")
                    else:
                        st.info("Could not identify distinct drawdown periods.")
                else:
                    st.info("Insufficient data for rolling drawdown analysis.")
            else:
                st.info("No daily PnL data available.")

            st.subheader("📜 Trade Log")
            display_cols = ["Date", "Entry Date", "Exit Date", "Exit Type", "Strategy", "Ticker", "Action",
                          "Entry Criteria", "Signal Close", "T+1 Open", "Price", "Shares", "PnL", 
                          "ATR", "Equity at Signal", "Risk $"]
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
