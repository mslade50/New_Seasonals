import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import time
from pandas.tseries.offsets import BusinessDay
import plotly.graph_objects as go
import plotly.express as px
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
    from strategy_config import _STRATEGY_BOOK_RAW, ACCOUNT_VALUE, CSV_UNIVERSE, LIQUID_PLUS_COMMODITIES
except ImportError:
    # st.error("Could not find strategy_config.py in the root directory.")
    _STRATEGY_BOOK_RAW = []
    CSV_UNIVERSE = []
    LIQUID_PLUS_COMMODITIES = []
    ACCOUNT_VALUE = 150000

# Strategies that the overflow scanner runs against the broader CSV_UNIVERSE.
# When the "Run on Overflow Universe" UI toggle is on, strat_backtester swaps
# universe_tickers to CSV_UNIVERSE for these (mirrors local_overflow_scan.py).
OVERFLOW_ELIGIBLE_STRATEGIES = {
    "Oversold Low Volume",
    "Overbot Vol Spike",
    "LT Trend ST OS",
    "St OS Sznl",
    "52wh Breakout",
}

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
PRIMARY_SZNL_PATH = "sznl_ranks.csv"
BACKUP_SZNL_PATH = "seasonal_ranks.csv"
ATR_SZNL_PATH = "atr_seasonal_ranks.parquet"

ATR_SZNL_WINDOWS = [5, 10, 21, 63, 126, 252]
ATR_SZNL_COLS = [f"atr_sznl_{w}d" for w in ATR_SZNL_WINDOWS]

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
        thresh_max = pf.get('thresh_max', 100.0)
        if pf['logic'] == '<':
            cond = col_vals < pf['thresh']
        elif pf['logic'] == 'Between':
            cond = (col_vals >= pf['thresh']) & (col_vals <= thresh_max)
        elif pf['logic'] == 'Not Between':
            # NaN comparisons return False on both branches → NaN excluded.
            cond = (col_vals < pf['thresh']) | (col_vals > thresh_max)
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

    # Risk dial filters (e.g. 63d dial 10d-avg < 50). Loads rd2_fragility.parquet
    # once per process; fails closed if missing/stale or column unavailable.
    dial_filters = params.get('dial_filters', [])
    if dial_filters:
        try:
            import os as _os
            _here = _os.path.dirname(_os.path.abspath(__file__))
            _frag_path = _os.path.join(_here, '..', 'data', 'rd2_fragility.parquet')
            frag_df = pd.read_parquet(_frag_path) if _os.path.exists(_frag_path) else None
            if frag_df is not None:
                frag_df.index = pd.to_datetime(frag_df.index).normalize()
                try: frag_df.index = frag_df.index.tz_localize(None)
                except (TypeError, AttributeError): pass
        except Exception:
            frag_df = None
        for df_filter in dial_filters:
            dial_col = df_filter.get('dial')
            if frag_df is None or dial_col not in frag_df.columns:
                # File missing or column unavailable → skip the filter for this
                # backtest (pass-through, don't penalize). daily_scan still
                # fail-closes for live safety.
                continue
            win = max(1, int(df_filter.get('window', 1)))
            dial_series = frag_df[dial_col]
            if win > 1:
                dial_series = dial_series.rolling(win, min_periods=win).mean()
            aligned = dial_series.reindex(df.index, method='ffill').values
            thresh = float(df_filter.get('thresh', 0))
            logic = df_filter.get('logic', '>')
            if   logic == '>':  cond_d = aligned > thresh
            elif logic == '<':  cond_d = aligned < thresh
            elif logic == '>=': cond_d = aligned >= thresh
            elif logic == '<=': cond_d = aligned <= thresh
            else:               cond_d = np.ones(len(df), dtype=bool)
            # NaN dial (pre-2016 or missing) → pass through, treat as if
            # the dial filter doesn't apply for those trades.
            cond_d = np.where(np.isnan(aligned), True, cond_d)
            conditions.append(cond_d)

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

    # Transition / crossover filter — fires when a series was in a "from" state
    # at any point in the past `within_days` AND is in a "to" state TODAY.
    # Enables trend-following and rotation entries like
    # "xsec 252d rank was <50 in last 5 days, is >65 today". Series options:
    #   series_type='perf'      → rank_ret_{window}d        (per-ticker perf rank)
    #   series_type='xsec'      → xsec_rank_ret_{window}d   (cross-sectional rank)
    #   series_type='atr_sznl'  → atr_sznl_{window}d        (ATR-seasonal rank)
    # Each filter: {series_type, window, from_logic, from_thresh,
    #               to_logic, to_thresh, within_days}.
    for tf in params.get('transition_filters', []):
        st_type = tf.get('series_type', 'perf')
        win = tf['window']
        if st_type == 'xsec':
            col = f"xsec_rank_ret_{win}d"
        elif st_type == 'atr_sznl':
            col = f"atr_sznl_{win}d"
        else:
            col = f"rank_ret_{win}d"
        if col not in df.columns:
            continue

        col_vals = df[col].values.astype(float)

        def _logic_mask(vals, logic, thresh):
            if logic == '<':
                return vals < thresh
            if logic == '<=':
                return vals <= thresh
            if logic == '>':
                return vals > thresh
            if logic == '>=':
                return vals >= thresh
            return np.zeros_like(vals, dtype=bool)

        to_mask = _logic_mask(col_vals, tf.get('to_logic', '>'), tf.get('to_thresh', 0.0))
        from_mask = _logic_mask(col_vals, tf.get('from_logic', '<'), tf.get('from_thresh', 0.0))

        within = max(1, int(tf.get('within_days', 5)))
        # "from state occurred in last N days before today" — rolling max over a
        # trailing window of length `within`, shifted by 1 so today is excluded.
        from_series = pd.Series(from_mask.astype(float), index=df.index)
        from_rolling = from_series.rolling(within, min_periods=1).max().shift(1).fillna(0).values.astype(bool)

        conditions.append(to_mask & from_rolling)

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
        # Transition filters may request xsec series — register their windows too.
        for tf in s.get('transition_filters', []):
            if tf.get('series_type') == 'xsec':
                xsec_windows_needed.add(tf['window'])
        # Signal-exit filters (during held position) may also reference xsec.
        for sf in s.get('signal_exit_filters', []):
            if sf.get('series_type') == 'xsec':
                xsec_windows_needed.add(sf['window'])

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
                            'range_pct': row['RangePct'] * 100,
                            'atr_sznl_5d': row.get('atr_sznl_5d', 50.0),
                            'rank_ret_126d': row.get('rank_ret_126d', 50.0),
                            'rank_ret_252d': row.get('rank_ret_252d', 50.0),
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


def process_signals_fast(candidates, signal_data, processed_dict, strategies, starting_equity, cap_bps=None, flat_sizing=False):
    """
    Process candidates chronologically with dynamic sizing based on REAL-TIME MTM equity.

    flat_sizing=True forces every trade to size off `starting_equity` (no
    compounding). Useful for evaluating per-strategy edge without recent
    equity-bloated trades dominating the time-weighted PnL.
    
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

    # Track per-strategy placed-order risk for the post-loop daily cap.
    # Mirrors order_staging.py: cap is applied to ALL orders that would be
    # staged, not just the ones whose limit happens to fill that day.
    placed_risk_by_strat_date = {}

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

        # ========== STAGING-TIME (mirrors order_staging.py) ==========
        # MTM equity, OVS gap-tier, and sizing all happen BEFORE the entry-fill
        # check so we can track "would-be placed risk" for the per-strategy
        # daily cap regardless of whether the limit actually fills. This matches
        # how prod's order_staging caps total daily risk at staging time, with
        # no knowledge of intraday fill outcomes.

        # --- 1. MTM equity (only recompute on new signal_ts) ---
        if signal_ts != last_signal_ts:
            still_open = []
            for pos in open_positions:
                if signal_date >= pos['exit_date']:
                    exit_price_mtm = _mtm_price(pos['t_clean'], pos['exit_date'])
                    if exit_price_mtm is None:
                        pos_df = processed_dict.get(pos['t_clean'])
                        exit_price_mtm = pos_df.iloc[pos['exit_idx']]['Close'] if pos_df is not None else pos['entry_price']
                    if pos['direction'] == 'Long':
                        realized_pnl += (exit_price_mtm - pos['entry_price']) * pos['shares']
                    else:
                        realized_pnl += (pos['entry_price'] - exit_price_mtm) * pos['shares']
                else:
                    still_open.append(pos)
            open_positions = still_open
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

        # --- 2. OVS gap-tier (staging decision; can drop signal entirely) ---
        # Reachable 252D values for OVS are <65 or >95 (the Not Between filter
        # excludes 65-95 and NaN at signal time). Any other case here means
        # the filter wasn't applied — fail closed by skipping.
        _ovs_size_mult = 1.0
        if strat_name == "Overbot Vol Spike" and entry_row is not None:
            _sig_close = row_data['close']
            _t1_open = float(entry_row['Open'])
            _decisive_gap = _sig_close + 0.25 * atr
            _r252 = row_data.get('rank_ret_252d', None)
            if pd.isna(_t1_open) or _t1_open <= _sig_close:
                continue  # no gap → no order placed
            elif _t1_open <= _decisive_gap:
                # Mild gap — only take if 252D < 65 (weak long-term, fade thesis strong)
                if pd.notna(_r252) and _r252 < 65:
                    _ovs_size_mult = 0.7
                else:
                    continue  # mild gap + (252D > 95 or NaN or 65-95) → skip
            else:
                _ovs_size_mult = 1.0  # decisive gap → full size

        # --- 3. Sizing (base_risk × strategy multipliers × ladder × gap_mult) ---
        risk_bps = execution['risk_bps']
        equity_for_sizing = starting_equity if flat_sizing else current_equity
        base_risk = equity_for_sizing * risk_bps / 10000
        _vol_spike_skip_primary = False
        if strat_name == "Weak Close Decent Sznls":
            sznl_val = row_data['sznl']
            if sznl_val >= 65:
                base_risk *= 1.5
            elif sznl_val >= 33:
                base_risk *= 0.66 if sznl_val < 50 else 1.0
        if strat_name == "Overbot Vol Spike":
            _atr_sznl_5d = row_data.get('atr_sznl_5d', None)
            if pd.notna(_atr_sznl_5d) and _atr_sznl_5d < 30:
                base_risk *= 1.3
        ladder_mults = execution.get('ladder_multipliers')
        if ladder_mults:
            open_count = sum(
                1 for p in open_positions
                if p['t_clean'] == t_clean and p['strat_name'] == strat_name
            )
            rung_idx = min(open_count, len(ladder_mults) - 1)
            base_risk *= ladder_mults[rung_idx]
        if _ovs_size_mult != 1.0:
            base_risk *= _ovs_size_mult

        # --- 4. Track placed risk for the per-strategy daily cap ---
        placed_risk_by_strat_date[(signal_date, strat_name)] = (
            placed_risk_by_strat_date.get((signal_date, strat_name), 0) + base_risk
        )

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
        
        # --- LIMIT (OPEN +/- ATR) Single Day: Limit anchored to T+1 Open ---
        # Mirrors live REL_OPEN order: limit price is set relative to the T+1
        # open once it's known. Short fills only if intraday price rallies to
        # open + offset; long fills only if price drops to open - offset.
        # No fallback to T+1 Open itself — if the limit isn't touched, skip.
        elif is_limit_open_atr:
            _limit_mult = 0.75 if '0.75' in entry_type else 0.5
            limit_offset = _limit_mult * atr
            t1_open = entry_row['Open']

            if settings['trade_direction'] == 'Long':
                limit_price = t1_open - limit_offset
                if entry_row['Low'] <= limit_price:
                    entry_price = limit_price
                else:
                    valid_entry = False
            else:
                limit_price = t1_open + limit_offset
                if entry_row['High'] >= limit_price:
                    entry_price = limit_price
                else:
                    valid_entry = False
            entry_date = entry_row.name
        
        # --- PERSISTENT LIMIT anchored to SIGNAL CLOSE ---
        elif is_persistent or is_limit_close_anchored:
            # Parse offset multiplier from entry_type string. Defaults to 0.5
            # for backward-compat with configs that don't specify a multiplier.
            if '0.25' in entry_type:
                _pers_mult = 0.25
            elif '1 ATR' in entry_type or '1.0 ATR' in entry_type:
                _pers_mult = 1.0
            elif '0.75' in entry_type:
                _pers_mult = 0.75
            else:
                _pers_mult = 0.5
            limit_offset = _pers_mult * atr
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

        # OVS gap-tier and sizing already handled in the staging-time block
        # above (before the entry-fill check). base_risk and _ovs_size_mult
        # are computed there so the per-strategy daily cap can be applied to
        # all placed orders, not just filled ones.

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
        
        # Signal-deactivate exit: re-evaluate filter(s) each held day; exit at
        # that day's close if any filter fires. Complements stop/target. Useful
        # for trend-following or rotation strategies that should exit when the
        # entry thesis no longer holds (e.g., "xsec 252d rank drops below 50").
        # Stop/target take priority — signal-deact only fires if neither hit.
        signal_exit_filters = settings.get('signal_exit_filters', [])
        use_signal_exit = bool(signal_exit_filters)

        def _signal_exit_fires(day_row):
            for sf in signal_exit_filters:
                st_type = sf.get('series_type', 'perf')
                win = sf['window']
                if st_type == 'xsec':
                    col = f"xsec_rank_ret_{win}d"
                elif st_type == 'atr_sznl':
                    col = f"atr_sznl_{win}d"
                else:
                    col = f"rank_ret_{win}d"
                if col not in day_row.index:
                    continue
                val = day_row[col]
                if pd.isna(val):
                    continue
                logic = sf.get('logic', '<')
                thresh = float(sf.get('thresh', 0.0))
                if logic == '<' and val < thresh: return True
                if logic == '<=' and val <= thresh: return True
                if logic == '>' and val > thresh: return True
                if logic == '>=' and val >= thresh: return True
                if logic == 'Between':
                    tmax = float(sf.get('thresh_max', 100.0))
                    if thresh <= val <= tmax: return True
            return False

        # Check for stop/target/signal-deact hits day by day
        if use_stop or use_target or use_signal_exit:
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

                if use_signal_exit and _signal_exit_fires(check_row):
                    exit_price = check_row['Close']
                    exit_date = check_row.name
                    exit_idx = check_idx
                    exit_type = "SignalDeact"
                    break
        
        # MTM equity, sizing, and OVS gap-tier mult were all computed in the
        # staging-time block above (before the entry-fill check). We arrive
        # here only if the limit actually filled, and proceed to open the
        # position using the already-computed base_risk.

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

    # Per-strategy daily risk backstop — each strategy independently capped at
    # cap_bps of MTM equity per signal date. Scaling is computed against
    # PLACED-order risk (every signal that survived gap-tier filtering),
    # not filled-trade risk. This mirrors how order_staging.py applies the
    # cap at staging time, before knowing intraday fill outcomes — orders
    # get scaled by cap/total_placed regardless of whether their limits
    # eventually fill that day.
    # cap_bps=None falls back to 250; 0 disables.
    _effective_cap = 250 if cap_bps is None else cap_bps
    if _effective_cap and len(sig_df) > 0:
        sig_df = sig_df.sort_values(by="Date").reset_index(drop=True)
        for (date, strat), grp_idx in sig_df.groupby(['Date', 'Strategy'], sort=False).groups.items():
            placed_total = placed_risk_by_strat_date.get((date, strat), 0.0)
            if placed_total <= 0:
                continue
            day_equity = (
                float(starting_equity) if flat_sizing
                else float(sig_df.loc[grp_idx, 'Equity at Signal'].iloc[0])
            )
            cap_dollars = day_equity * _effective_cap / 10000.0
            if placed_total > cap_dollars:
                scale = cap_dollars / placed_total
                sig_df.loc[grp_idx, 'Shares'] = (sig_df.loc[grp_idx, 'Shares'] * scale).round().astype(int)
                sig_df.loc[grp_idx, 'PnL']    = (sig_df.loc[grp_idx, 'PnL']    * scale).round()
                sig_df.loc[grp_idx, 'Risk $'] = sig_df.loc[grp_idx, 'Risk $']  * scale

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


def build_strategy_correlation_matrix(sig_df, master_dict, min_trades=30, mode='calendar'):
    """Daily-return correlation matrix across strategies.

    Walks sig_df grouped by Strategy, builds per-strategy daily MTM P&L via
    get_daily_mtm_series, aligns all series on a common date index, and
    returns the Pearson correlation matrix.

    mode='calendar' — correlate on all trading days (zero days included).
      Answers: "how does the portfolio equity curve co-move?" Standard for
      book-level vol / diversification analysis.
    mode='active'   — correlate only on days where at least one of the two
      strategies has a non-zero return. Filters out the trivially-correlated
      double-zero days. Answers: "when either strategy is deployed, do they
      agree?" More useful for identifying genuinely redundant signals.

    Strategies with fewer than `min_trades` trades are excluded (noise).
    Returns (corr_df, pnl_df, eligible_list). corr_df is None if <2 eligible.
    """
    if sig_df.empty or 'Strategy' not in sig_df.columns:
        return None, None, []

    # Fold 3x ETF Overbot Fade into Overbot Vol Spike for correlation purposes.
    # Same thesis (multi-horizon overbought fade, short bias) on a different
    # universe slice — treating them as distinct rows just inflates a 0.9+
    # correlation cell that tells you nothing. Pool their daily P&L under one
    # row before counting eligibility / computing corr.
    sig_df = sig_df.copy()
    sig_df['Strategy'] = sig_df['Strategy'].replace(
        {'3x ETF Overbot Fade': 'Overbot Vol Spike'}
    )

    counts = sig_df.groupby('Strategy').size()
    eligible = counts[counts >= min_trades].index.tolist()
    if len(eligible) < 2:
        return None, None, eligible

    min_date = pd.to_datetime(sig_df['Entry Date']).min()
    pnl_by_strat = {}
    for strat in eligible:
        sub = sig_df[sig_df['Strategy'] == strat]
        series = get_daily_mtm_series(sub, master_dict, start_date=min_date)
        pnl_by_strat[strat] = series

    pnl_df = pd.DataFrame(pnl_by_strat).fillna(0.0)

    if mode == 'active':
        # Day is kept if ANY strategy had a non-zero P&L that day — drops
        # double-zero rows that inflate positive correlation.
        pnl_df = pnl_df[(pnl_df != 0).any(axis=1)]
        if len(pnl_df) < 10:
            return None, pnl_df, eligible

    corr_df = pnl_df.corr()
    return corr_df, pnl_df, eligible


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

    # Full-lifespan total row — same metrics computed across the entire series.
    if yearly_stats and len(equity_series) >= 2:
        tot_start = equity_series.iloc[0]
        tot_end = equity_series.iloc[-1]
        tot_ret_pct = (tot_end - tot_start) / tot_start if tot_start != 0 else 0
        tot_ret_dollar = tot_end - tot_start
        tot_std = daily_rets.std() * np.sqrt(252)
        tot_mean = daily_rets.mean() * 252
        tot_sharpe = tot_mean / tot_std if tot_std != 0 else 0
        tot_neg = daily_rets[daily_rets < 0]
        tot_downside = np.sqrt((tot_neg**2).mean()) * np.sqrt(252) if len(tot_neg) > 0 else 0
        tot_sortino = tot_mean / tot_downside if tot_downside != 0 else 0
        tot_running_max = equity_series.expanding().max()
        tot_dd = ((equity_series - tot_running_max) / tot_running_max).min()
        yearly_stats.append({
            "Year": "Total",
            "Total Return ($)": tot_ret_dollar,
            "Total Return (%)": tot_ret_pct,
            "Max Drawdown": tot_dd,
            "Sharpe Ratio": tot_sharpe,
            "Sortino Ratio": tot_sortino,
            "Std Dev": tot_std,
        })

    return pd.DataFrame(yearly_stats)


def calculate_performance_stats(sig_df, master_dict, starting_equity, start_date=None):
    """Per-strategy metrics including Sharpe/Sortino computed only over the
    days each strategy has open positions (time-in-market basis). This
    avoids penalizing strategies that are idle most of the time and gives
    a comparable risk-adjusted return across strategies with very different
    duty cycles.
    """
    stats = []

    def _time_in_market_metrics(strat_df):
        """Sharpe + Sortino over days the strategy has any open position."""
        if strat_df.empty:
            return np.nan, np.nan
        daily_pnl = get_daily_mtm_series(strat_df, master_dict, start_date=start_date)
        if daily_pnl.empty:
            return np.nan, np.nan
        # Mark each day as "in market" if any trade in strat_df spans that day
        in_market = pd.Series(False, index=daily_pnl.index)
        entry_dates = pd.to_datetime(strat_df['Entry Date']).values
        exit_dates  = pd.to_datetime(strat_df['Exit Date']).values
        for ed, xd in zip(entry_dates, exit_dates):
            in_market.loc[(in_market.index >= ed) & (in_market.index <= xd)] = True
        pnl_im = daily_pnl[in_market]
        if len(pnl_im) < 5 or starting_equity <= 0:
            return np.nan, np.nan
        rets = pnl_im / float(starting_equity)
        mean_r = rets.mean()
        std_r = rets.std()
        sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else np.nan
        downside = rets[rets < 0]
        if len(downside) >= 2 and downside.std() > 0:
            sortino = mean_r / downside.std() * np.sqrt(252)
        else:
            sortino = np.nan
        return sharpe, sortino

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
        sharpe_im, sortino_im = _time_in_market_metrics(df)
        return {
            "Strategy": name, "Trades": count, "Total PnL": total_pnl,
            "Profit Factor": profit_factor, "SQN": sqn,
            "Sharpe (TIM)": sharpe_im, "Sortino (TIM)": sortino_im,
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
        st.markdown("---")
        use_overflow_universe = st.checkbox(
            f"🌊 Include Overflow Universe ({len(CSV_UNIVERSE):,} tickers)",
            value=False,
            help=(
                "Expand eligible strategies to CSV_UNIVERSE (sznl_ranks.csv, "
                f"~{len(CSV_UNIVERSE):,} tickers). Liquid and overflow tickers "
                "receive identical OVS sizing — overflow-specific haircuts and "
                "mild-gap multipliers were retired in prod. "
                f"Affects: {', '.join(sorted(OVERFLOW_ELIGIBLE_STRATEGIES))}. "
                "Other strategies keep their default universes."
            ),
        )
        cap_bps_input = st.number_input(
            f"Per-strategy daily risk backstop (bps, 0 = off)",
            min_value=0, max_value=1000,
            value=250, step=25,
            help=(
                "Per-signal-date, per-strategy cap on total risk, in bps of "
                "MTM equity. Each strategy independently constrained — OVS, "
                "OLV, 52wh, etc. each get their own bucket. Default 250 bps "
                "mirrors order_staging.MAX_DAILY_RISK_PCT (2.5%) but scoped "
                "per strategy rather than pooled. Set to 0 to disable."
            ),
        )
        flat_sizing_input = st.checkbox(
            "Flat sizing (no compounding)",
            value=False,
            help=(
                "Size every trade off the BACKTEST starting equity instead of "
                "live MTM equity. Eliminates the recency bias where late-period "
                "trades dominate total PnL because compounded equity made each "
                "trade much larger. Useful for evaluating per-strategy edge "
                "across the full history on equal terms."
            ),
        )
        use_master_parquet = st.checkbox(
            "📦 Use master parquet (data_provider)",
            value=True,
            help=(
                "Read OHLCV from data/master_prices.parquet — the shared "
                "source backed by scripts/build_master_prices.py + "
                "scripts/update_master_prices.py. No per-run yfinance calls. "
                "Identical data to pages/backtester.py when both use master. "
                "Uncheck to fall back to legacy yfinance + data/bt_price_cache "
                "(slower, hits yfinance every run)."
            ),
        )
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

        if use_overflow_universe and CSV_UNIVERSE:
            # Union CSV_UNIVERSE (sznl_ranks.csv) with seasonal_ranks.csv tickers
            # so the strat_backtester overflow universe matches what legacy
            # backtester runs under "All CSV + Overflow Extras". sznl_map keys
            # are already the union of both CSVs (see load_seasonal_map).
            _overflow_tickers = sorted(set(CSV_UNIVERSE) | set(sznl_map.keys()))
            _swapped = []
            for s in strategies:
                if s['name'] in OVERFLOW_ELIGIBLE_STRATEGIES:
                    s['universe_tickers'] = _overflow_tickers
                    _swapped.append(s['name'])
            if _swapped:
                st.info(
                    f"🌊 Overflow universe active — swapped {len(_swapped)} strategies to "
                    f"CSV_UNIVERSE ∪ seasonal_ranks ({len(_overflow_tickers):,} tickers): {', '.join(_swapped)}"
                )

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

        if use_master_parquet:
            import data_provider
            if not data_provider.has_master():
                st.error(
                    "Master parquet not found at data/master_prices.parquet. "
                    "Run `python scripts/build_master_prices.py` first, or "
                    "uncheck 'Use master parquet' to fall back to yfinance."
                )
                return
            t0_load = time.time()
            master_dict = data_provider.get_history(long_term_list)
            st.session_state['backtest_data'] = master_dict
            n_missing = len(set(long_term_list) - set(master_dict.keys()))
            st.caption(
                f"📦 Loaded {len(master_dict):,} tickers from master parquet "
                f"in {time.time()-t0_load:.1f}s "
                f"({n_missing} of {len(long_term_list)} requested missing — "
                "see scripts/audit_master_prices.py)"
            )
        else:
            # Legacy yfinance + per-ticker bt_price_cache flow.
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
        if cap_bps_input == 0:
            st.info("🚫 Aggregate risk backstop disabled — signals execute at raw per-strategy sizing.")
        elif cap_bps_input != 250:
            st.info(f"⚖️ Aggregate risk backstop overridden: {cap_bps_input} bps (prod default: 250 bps).")
        t0 = time.time()
        sig_df = process_signals_fast(
            candidates, signal_data, processed_dict, strategies, starting_equity,
            cap_bps=cap_bps_input,
            flat_sizing=flat_sizing_input,
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

            # Cache consolidated closes for MTM replay in fragility_sizing_lab
            try:
                closes = {}
                for ticker in sig_df['Ticker'].unique():
                    t_clean = ticker.replace('.', '-')
                    t_df = master_dict.get(t_clean)
                    if t_df is None:
                        t_df = master_dict.get(ticker)
                    if t_df is None or t_df.empty:
                        continue
                    tmp = t_df.copy()
                    if isinstance(tmp.columns, pd.MultiIndex):
                        tmp.columns = [c[0] if isinstance(c, tuple) else c for c in tmp.columns]
                    tmp.columns = [c.capitalize() for c in tmp.columns]
                    if 'Close' in tmp.columns:
                        closes[ticker] = tmp['Close']
                if closes:
                    closes_df = pd.DataFrame(closes).sort_index()
                    closes_path = os.path.join(cache_dir, "backtest_closes.parquet")
                    closes_df.to_parquet(closes_path)
                    st.caption(f"📦 Cached {closes_df.shape[1]} close series to data/backtest_closes.parquet")
                else:
                    st.caption("⚠️ No closes found in master_dict — MTM replay in Fragility Lab will fall back to stair-step.")
            except Exception as e:
                st.caption(f"⚠️ Could not cache closes: {e}")

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
            st.caption("Sharpe / Sortino are time-in-market — computed only over days each strategy has open positions, normalized by starting equity.")
            stats_df = calculate_performance_stats(sig_df, master_dict, starting_equity, start_date=user_start_date)
            st.dataframe(stats_df.style.format({
                "Total PnL": "${:,.0f}",
                "Profit Factor": "{:.2f}", "SQN": "{:.2f}",
                "Sharpe (TIM)": "{:.2f}", "Sortino (TIM)": "{:.2f}",
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

            # ========== STRATEGY CORRELATION MATRIX ==========
            st.subheader("🔗 Strategy Correlation Matrix (Daily P&L)")
            corr_c1, corr_c2 = st.columns([1, 1])
            with corr_c1:
                corr_mode = st.radio(
                    "Correlation mode", ["Calendar (all days)", "Active (non-zero days)"],
                    horizontal=True, index=0, key="corr_mode",
                    help="Calendar: correlate on all trading days including shared zero-return days (standard for portfolio vol analysis). Active: correlate only on days where at least one strategy had P&L — filters out trivially-correlated double-zero days, more useful for spotting genuinely redundant signals.",
                )
            with corr_c2:
                min_trades_corr = st.number_input(
                    "Min trades for inclusion", min_value=5, max_value=500, value=30, step=5,
                    key="corr_min_trades",
                )
            _mode_key = 'active' if 'Active' in corr_mode else 'calendar'
            corr_df, _pnl_df, _eligible = build_strategy_correlation_matrix(
                sig_df, master_dict, min_trades=int(min_trades_corr), mode=_mode_key,
            )
            if corr_df is None or corr_df.empty or len(corr_df) < 2:
                st.info(f"Need ≥2 strategies with ≥{int(min_trades_corr)} trades each for a correlation matrix.")
            else:
                # Mask the diagonal so the self-correlation 1.0 doesn't dominate
                # the red end of the scale. NaN cells render gray.
                np.fill_diagonal(corr_df.values, np.nan)
                # Heatmap — zero-centered diverging scale
                fig_corr = px.imshow(
                    corr_df.values,
                    x=corr_df.columns, y=corr_df.index,
                    text_auto='.2f', aspect='auto',
                    color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                    labels=dict(color='Correlation'),
                )
                fig_corr.update_traces(zmid=0)
                fig_corr.update_layout(
                    height=max(350, 40 * len(corr_df) + 120),
                    margin=dict(l=120, r=30, t=20, b=120),
                    xaxis=dict(tickangle=45),
                    plot_bgcolor='#888',
                )
                st.plotly_chart(fig_corr, use_container_width=True)

                # Diversification score per strategy — avg correlation to OTHER strategies.
                # Lower = more independent = better diversifier. Diagonal already NaN'd above.
                avg_corr = corr_df.mean(axis=1).sort_values()
                max_corr = corr_df.max(axis=1)
                max_corr_with = corr_df.idxmax(axis=1)
                div_df = pd.DataFrame({
                    'Strategy': avg_corr.index,
                    'Avg Corr (vs others)': avg_corr.values,
                    'Max Corr': [max_corr[s] for s in avg_corr.index],
                    'Most Correlated With': [max_corr_with[s] for s in avg_corr.index],
                }).reset_index(drop=True)

                def _color_avg(v):
                    if v < 0.2: return 'background-color: #1a4d1a; color: #9fe09f;'     # green
                    if v < 0.4: return 'background-color: #4d4d1a; color: #e0e09f;'     # yellow-green
                    if v < 0.6: return 'background-color: #663f1a; color: #e0b89f;'     # orange
                    return 'background-color: #661a1a; color: #e09f9f;'                 # red

                st.markdown("**Diversification score** — avg correlation vs other strategies. Lower = better diversifier.")
                st.dataframe(
                    div_df.style
                        .format({'Avg Corr (vs others)': '{:.2f}', 'Max Corr': '{:.2f}'})
                        .map(_color_avg, subset=['Avg Corr (vs others)']),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "Rules of thumb for book-level diversification: avg corr < 0.2 → strong diversifier, include at full size. "
                    "0.2-0.4 → useful diversifier, size at 75-100%. 0.4-0.6 → partial overlap, consider 50-75% of normal sizing. "
                    ">0.6 → largely redundant with another strategy in the book; either reduce sizing or retire one of the pair."
                )

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
            # (Portfolio Breakdown and Drawdown Deep Dive removed; kept Trade Log below.)
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
