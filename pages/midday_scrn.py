import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import gspread
from pandas.tseries.offsets import BusinessDay
import time
import pytz
import sys
import os
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

# Define a "Trading Day" offset that skips Weekends AND US Holidays
TRADING_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

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
CSV_PATH = "sznl_ranks.csv" 

@st.cache_resource 
def load_seasonal_map():
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        return {}
    if df.empty: return {}
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])
    df["Date"] = df["Date"].dt.normalize()
    output_map = {}
    for ticker, group in df.groupby("ticker"):
        series = group.set_index("Date")["seasonal_rank"].sort_index()
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

def is_after_market_close():
    tz = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(tz)
    return now.hour >= 16

# -----------------------------------------------------------------------------
# GOOGLE SHEET FUNCTIONS (Kept for EOD Batch Mode)
# -----------------------------------------------------------------------------
def save_moc_orders(signals_list, strategy_book, sheet_name='moc_orders'):
    # ... (Keep existing implementation) ...
    pass

def save_staging_orders(signals_list, strategy_book, sheet_name='Order_Staging'):
    # ... (Keep existing implementation) ...
    pass

def save_signals_to_gsheet(new_dataframe, sheet_name='Trade_Signals_Log'):
    # ... (Keep existing implementation) ...
    pass

# -----------------------------------------------------------------------------
# ENGINE
# -----------------------------------------------------------------------------

def calculate_indicators(df, sznl_map, ticker, market_series=None):
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    
    # --- MAs ---
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean() 
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    # --- Gap Count ---
    is_open_gap = (df['Low'] > df['High'].shift(1)).astype(int)
    df['GapCount_21'] = is_open_gap.rolling(21).sum() 

    # --- Candle Range Location % ---
    denom = (df['High'] - df['Low'])
    df['RangePct'] = np.where(denom == 0, 0.5, (df['Close'] - df['Low']) / denom)

    # --- Perf Ranks ---
    for window in [5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window)
        df[f'rank_ret_{window}d'] = df[f'ret_{window}d'].expanding(min_periods=50).rank(pct=True) * 100.0
        
    # --- ATR ---
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100

    # --- Volume ---
    vol_ma = df['Volume'].rolling(63).mean()
    df['vol_ratio'] = df['Volume'] / vol_ma
    df['vol_ma'] = vol_ma
    
    # Vol Rank
    vol_ma_10 = df['Volume'].rolling(10).mean()
    df['vol_ratio_10d'] = vol_ma_10 / vol_ma
    df['vol_ratio_10d_rank'] = df['vol_ratio_10d'].expanding(min_periods=50).rank(pct=True) * 100.0

    # --- Seasonality ---
    df['Sznl'] = get_sznl_val_series(ticker, df.index, sznl_map)
    df['Mkt_Sznl_Ref'] = get_sznl_val_series("^GSPC", df.index, sznl_map)
    
    # --- Age ---
    if not df.empty:
        start_ts = df.index[0]
        df['age_years'] = (df.index - start_ts).days / 365.25
    else:
        df['age_years'] = 0.0

    if market_series is not None:
        df['Market_Above_SMA200'] = market_series.reindex(df.index, method='ffill').fillna(False)
    
    # --- 52w High/Low ---
    rolling_high = df['High'].shift(1).rolling(252).max()
    rolling_low = df['Low'].shift(1).rolling(252).min()
    df['is_52w_high'] = df['High'] > rolling_high
    df['is_52w_low'] = df['Low'] < rolling_low
        
    return df
    
# -----------------------------------------------------------------------------
# REFACTORED SIGNAL AUDITOR
# -----------------------------------------------------------------------------

def check_signal(df, params, sznl_map):
    last_row = df.iloc[-1]
    
    # 0. Day of Week Filter
    if params.get('use_dow_filter', False):
        allowed = params.get('allowed_days', [])
        current_day = last_row.name.dayofweek
        if current_day not in allowed: return False

    # 0b. Cycle Year Filter (NEW)
    # 0=Election, 1=Post, 2=Mid, 3=Pre
    if 'allowed_cycles' in params:
        allowed_cycles = params['allowed_cycles']
        # Only apply if the list isn't empty and isn't all 4 cycles (which implies no filter)
        if allowed_cycles and len(allowed_cycles) < 4:
            current_year = last_row.name.year
            cycle_rem = current_year % 4
            if cycle_rem not in allowed_cycles: return False

    # 1. Liquidity Gates
    if last_row['Close'] < params.get('min_price', 0): return False
    if last_row['vol_ma'] < params.get('min_vol', 0): return False
    if last_row['age_years'] < params.get('min_age', 0): return False
    if last_row['age_years'] > params.get('max_age', 100): return False

    min_atr = params.get('min_atr_pct', 0.0)
    max_atr = params.get('max_atr_pct', 1000.0)
    
    current_atr_pct = last_row.get('ATR_Pct', 0)
    if pd.isna(current_atr_pct): return False

    if current_atr_pct < min_atr: return False
    if current_atr_pct > max_atr: return False

    # 2. Trend Filter (Global)
    trend_opt = params.get('trend_filter', 'None')
    if trend_opt == "Price > 200 SMA":
        if not (last_row['Close'] > last_row['SMA200']): return False
    elif trend_opt == "Price > Rising 200 SMA":
        prev_row = df.iloc[-2]
        if not ((last_row['Close'] > last_row['SMA200']) and (last_row['SMA200'] > prev_row['SMA200'])): return False
    elif "Market" in trend_opt or "SPY" in trend_opt:
        if 'Market_Above_SMA200' in df.columns:
            is_above = last_row['Market_Above_SMA200']
            if ">" in trend_opt and not is_above: return False
            if "<" in trend_opt and is_above: return False

    # 2b. MA Consecutive Filters
    if 'ma_consec_filters' in params:
        for maf in params['ma_consec_filters']:
            length = maf['length']
            col_name = f"SMA{length}" # e.g. SMA20
            
            # Ensure column exists (calculate_indicators handles 10, 20, 50, 200)
            if col_name not in df.columns: continue 
            
            # Logic Check
            if maf['logic'] == 'Above':
                mask = df['Close'] > df[col_name]
            elif maf['logic'] == 'Below':
                mask = df['Close'] < df[col_name]
            
            # Consecutive Check
            consec = maf.get('consec', 1)
            if consec > 1:
                mask = mask.rolling(consec).sum() == consec
            
            if not mask.iloc[-1]: return False

    # 3. Candle Range Filter
    if params.get('use_range_filter', False):
        rn_val = last_row['RangePct'] * 100
        r_min = params.get('range_min', 0)
        r_max = params.get('range_max', 100)
        if not (rn_val >= r_min and rn_val <= r_max): return False

    # 4. Perf Rank
    if 'perf_filters' in params:
        combined_cond = pd.Series(True, index=df.index)
        for pf in params['perf_filters']:
            col = f"rank_ret_{pf['window']}d"
            consec = pf.get('consecutive', 1)
            
            if pf['logic'] == '<': cond_f = df[col] < pf['thresh']
            else: cond_f = df[col] > pf['thresh']
            
            if consec > 1: cond_f = cond_f.rolling(consec).sum() == consec
            combined_cond = combined_cond & cond_f
        
        final_perf = combined_cond
        
        if params.get('perf_first_instance', False):
            lookback = params.get('perf_lookback', 21)
            prev_inst = final_perf.shift(1).rolling(lookback).sum()
            final_perf = final_perf & (prev_inst == 0)
            
        if not final_perf.iloc[-1]: return False

    elif params.get('use_perf_rank', False):
        col = f"rank_ret_{params['perf_window']}d"
        if params['perf_logic'] == '<': raw = df[col] < params['perf_thresh']
        else: raw = df[col] > params['perf_thresh']
        
        consec = params.get('perf_consecutive', 1)
        if consec > 1: persist = raw.rolling(consec).sum() == consec
        else: persist = raw
        
        final_perf = persist
        if params.get('perf_first_instance', False):
            lookback = params.get('perf_lookback', 21)
            prev_inst = final_perf.shift(1).rolling(lookback).sum()
            final_perf = final_perf & (prev_inst == 0)
            
        if not final_perf.iloc[-1]: return False

    # 5. Gap/Acc/Dist Filters
    if params.get('use_gap_filter', False):
        lookback = params.get('gap_lookback', 21)
        col_name = f'GapCount_{lookback}' if f'GapCount_{lookback}' in df.columns else 'GapCount_21'
        gap_val = last_row.get(col_name, 0)
        g_logic = params.get('gap_logic', '>')
        g_thresh = params.get('gap_thresh', 0)
        if g_logic == ">" and not (gap_val > g_thresh): return False
        if g_logic == "<" and not (gap_val < g_thresh): return False
        if g_logic == "=" and not (gap_val == g_thresh): return False

    if params.get('use_acc_count_filter', False):
        window = params.get('acc_count_window', 21)
        col_name = f'AccCount_{window}'
        if col_name in df.columns:
            acc_val = last_row[col_name]
            acc_logic = params.get('acc_count_logic', '=')
            acc_thresh = params.get('acc_count_thresh', 0)
            if acc_logic == "=" and not (acc_val == acc_thresh): return False
            if acc_logic == ">" and not (acc_val > acc_thresh): return False
            if acc_logic == "<" and not (acc_val < acc_thresh): return False

    if params.get('use_dist_count_filter', False):
        window = params.get('dist_count_window', 21)
        col_name = f'DistCount_{window}'
        if col_name in df.columns:
            dist_val = last_row[col_name]
            dist_logic = params.get('dist_count_logic', '>')
            dist_thresh = params.get('dist_count_thresh', 0)
            if dist_logic == "=" and not (dist_val == dist_thresh): return False
            if dist_logic == ">" and not (dist_val > dist_thresh): return False
            if dist_logic == "<" and not (dist_val < dist_thresh): return False

    # 6. Distance Filter
    if params.get('use_dist_filter', False):
        ma_type = params.get('dist_ma_type', 'SMA 200')
        ma_col = ma_type.replace(" ", "") 
        if ma_col in df.columns:
            ma_val = last_row[ma_col]
            atr = last_row['ATR']
            close = last_row['Close']
            if atr > 0: dist_units = (close - ma_val) / atr
            else: dist_units = 0
            d_logic = params.get('dist_logic', 'Between')
            d_min = params.get('dist_min', 0)
            d_max = params.get('dist_max', 0)
            if d_logic == "Greater Than (>)" and not (dist_units > d_min): return False
            if d_logic == "Less Than (<)" and not (dist_units < d_max): return False
            if d_logic == "Between":
                if not (dist_units >= d_min and dist_units <= d_max): return False

    # 7. Seasonality
    if params['use_sznl']:
        if params['sznl_logic'] == '<': raw_sznl = df['Sznl'] < params['sznl_thresh']
        else: raw_sznl = df['Sznl'] > params['sznl_thresh']
        
        final_sznl = raw_sznl
        if params.get('sznl_first_instance', False):
            lookback = params.get('sznl_lookback', 21)
            prev = final_sznl.shift(1).rolling(lookback).sum()
            final_sznl = final_sznl & (prev == 0)
        if not final_sznl.iloc[-1]: return False

    if params.get('use_market_sznl', False):
        mkt_ticker = params.get('market_ticker', '^GSPC')
        mkt_series_ref = sznl_map.get(mkt_ticker)
        if mkt_series_ref is None and mkt_ticker == '^GSPC':
             mkt_series_ref = sznl_map.get('SPY')
        mkt_ranks = get_sznl_val_series(mkt_ticker, df.index, sznl_map)
        
        if params['market_sznl_logic'] == '<': mkt_cond = mkt_ranks < params['market_sznl_thresh']
        else: mkt_cond = mkt_ranks > params['market_sznl_thresh']
        if not mkt_cond[-1]: return False

    # 8. 52w
    if params['use_52w']:
        if params['52w_type'] == 'New 52w High': cond_52 = df['is_52w_high']
        else: cond_52 = df['is_52w_low']
        if params.get('52w_first_instance', True):
            lookback = params.get('52w_lookback', 21)
            prev = cond_52.shift(1).rolling(lookback).sum()
            cond_52 = cond_52 & (prev == 0)
        if not cond_52.iloc[-1]: return False
        
    # 8b. Exclude 52w High
    if params.get('exclude_52w_high', False):
        if last_row['is_52w_high']: return False

    # 9. Volume (Ratio ONLY)
    if params['use_vol']:
        # Only check if magnitude > threshold (Ignore if it's > or < yesterday)
        if not (last_row['vol_ratio'] > params['vol_thresh']): return False

    if params.get('use_vol_rank'):
        val = last_row['vol_ratio_10d_rank']
        if params['vol_rank_logic'] == '<':
            if not (val < params['vol_rank_thresh']): return False
        else:
            if not (val > params['vol_rank_thresh']): return False
            
    return True

def get_signal_breakdown(df, params, sznl_map):
    """
    Categorizes failures into 'Stable' (Trend/Sznl) vs 'Volatile' (Range/5d/MA).
    """
    last_row = df.iloc[-1]
    audit = {
        'Stable_Fail': 0, 
        'Volatile_Fail': 0, 
        'Volatile_Items': [],
        'Stable_Items': []
    }
    
    # --- HELPER ---
    def log(key, value, passed, is_volatile=False):
        """
        is_volatile: If True, a failure is added to 'Volatile_Fail'.
                     If False, a failure is added to 'Stable_Fail' (Critical).
        """
        audit[key] = value
        audit[f"{key}_Pass"] = "✅" if passed else "❌"
        
        if not passed:
            if is_volatile:
                audit['Volatile_Fail'] += 1
                audit['Volatile_Items'].append(key)
            else:
                audit['Stable_Fail'] += 1
                audit['Stable_Items'].append(key)

    # -----------------------------------------------------------
    # 1. STABLE CHECKS (Must Pass)
    # -----------------------------------------------------------
    
    # Liquidity
    liq_pass = (
        last_row['Close'] >= params.get('min_price', 0) and 
        last_row['vol_ma'] >= params.get('min_vol', 0) and
        last_row['age_years'] >= params.get('min_age', 0)
    )
    log("Liquidity", f"${last_row['Close']:.2f}", liq_pass, is_volatile=False)

    # Day of Week / Cycle
    if params.get('use_dow_filter', False):
        allowed = params.get('allowed_days', [])
        day_num = last_row.name.dayofweek 
        log("DOW", last_row.name.strftime("%A"), day_num in allowed, is_volatile=False)

    if 'allowed_cycles' in params:
        allowed_cycles = params['allowed_cycles']
        if allowed_cycles and len(allowed_cycles) < 4:
            current_year = last_row.name.year
            cond = (current_year % 4) in allowed_cycles
            log("Cycle", f"{current_year}", cond, is_volatile=False)

    # Trend (Global - 200 SMA)
    trend_opt = params.get('trend_filter', 'None')
    if trend_opt != 'None':
        trend_res = True
        if trend_opt == "Price > 200 SMA":
            trend_res = last_row['Close'] > last_row['SMA200']
        elif "Market" in trend_opt or "SPY" in trend_opt:
            if 'Market_Above_SMA200' in df.columns:
                is_above = last_row['Market_Above_SMA200']
                trend_res = is_above if ">" in trend_opt else not is_above
        log("Trend", trend_opt, trend_res, is_volatile=False)

    # Perf Rank (10d and 21d are STABLE. 5d is VOLATILE)
    if 'perf_filters' in params:
        for pf in params['perf_filters']:
            window = pf['window']
            col = f"rank_ret_{window}d"
            val = last_row.get(col, 0)
            thresh = pf['thresh']
            
            if pf['logic'] == '<': cond = val < thresh
            else: cond = val > thresh
            
            # --- CUSTOM LOGIC: 5d is Volatile, others are Stable ---
            if window <= 5:
                # SPECIAL CHECK: If 5d fails, is it "close"? 
                # User wants "within 15%". I interpret this as 15 rank points.
                is_vol = True
                
                if not cond:
                    # If it failed, check distance. If distance > 15, mark as STABLE failure (too far gone).
                    dist = abs(val - thresh)
                    if dist > 15.0:
                        is_vol = False # Downgrade to Critical Failure
                
                log(f"Perf_{window}d", f"{val:.1f}", cond, is_volatile=is_vol)
            else:
                log(f"Perf_{window}d", f"{val:.1f}", cond, is_volatile=False)

    # Seasonality (Stable)
    if params['use_sznl']:
        val = last_row['Sznl']
        if params['sznl_logic'] == '<': cond = val < params['sznl_thresh']
        else: cond = val > params['sznl_thresh']
        log("Seasonality", f"{val:.1f}", cond, is_volatile=False)

    if params.get('use_market_sznl', False):
        val = last_row['Mkt_Sznl_Ref']
        if params['market_sznl_logic'] == '<': cond = val < params['market_sznl_thresh']
        else: cond = val > params['market_sznl_thresh']
        log("Mkt_Sznl", f"{val:.1f}", cond, is_volatile=False)

    # Volume (Stable)
    if params.get('use_vol', False):
        ratio = last_row['vol_ratio']
        cond = (ratio > params['vol_thresh'])
        log("Vol_Ratio", f"x{ratio:.2f}", cond, is_volatile=False)
        
    if params.get('use_vol_rank', False):
        ratio = last_row['vol_ratio_10d_rank']
        cond = (ratio < params['vol_rank_thresh']) if params['vol_rank_logic'] == '<' else (ratio > params['vol_rank_thresh'])
        log("Vol_Rank", f"{ratio:.1f}", cond, is_volatile=False)

    # -----------------------------------------------------------
    # 2. VOLATILE CHECKS (Allowed to Fail)
    # -----------------------------------------------------------

    # RANGE FILTER (Volatile)
    if params.get('use_range_filter', False):
        rng = last_row['RangePct'] * 100
        r_min = params.get('range_min', 0)
        r_max = params.get('range_max', 100)
        cond = (rng >= r_min) and (rng <= r_max)
        log("Range_Loc", f"{rng:.1f}%", cond, is_volatile=True)
        
    # MA Specific Filters (Volatile - e.g. Close > 20 SMA)
    if 'ma_consec_filters' in params:
        for maf in params['ma_consec_filters']:
            length = maf['length']
            val = last_row.get(f"SMA{length}", 0)
            logic = maf['logic']
            cond = (last_row['Close'] > val) if logic == 'Above' else (last_row['Close'] < val)
            log(f"Price_vs_SMA{length}", f"{val:.2f}", cond, is_volatile=True)

    # Final Result Text
    total_fails = audit['Stable_Fail'] + audit['Volatile_Fail']
    audit['Result'] = "✅ SIGNAL" if total_fails == 0 else ""
    
    return audit

# -----------------------------------------------------------------------------
# MAIN APP (Updated Monitor Section)
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="Pro Strategy Screener")
    st.sidebar.title("Configuration")
    mode = st.sidebar.radio("Scanner Mode", ["Intraday Monitor", "EOD Batch Runner"])
    
    sznl_map = load_seasonal_map()
    if 'master_data_dict' not in st.session_state: st.session_state['master_data_dict'] = {}

    # --- INTRADAY MONITOR ---
    if mode == "Intraday Monitor":
        st.title("👀 Intraday Signal Monitor (MOC)")
        st.markdown("""
        **Strict Logic Applied:** 1. **MUST PASS:** Seasonality, Trend (200d), 10d/21d Perf, Liquidity.
        2. **CAN FAIL (Watchlist):** Range %, Close vs SMA, 5d Perf (if within 15pts).
        """)
        
        col_tol1, col_tol2 = st.columns(2)
        # Fix: Slider max is strict, logic ensures we don't overflow
        max_volatile_fails = col_tol1.slider("Max Allowed 'Volatile' Failures:", 0, 3, 2)
        
        if st.button("🔎 Scan Potential Signals", type="primary"):
            st.cache_data.clear() 
            
            moc_strategies = [s for s in STRATEGY_BOOK if "Signal Close" in s['settings'].get('entry_type', '')]
            
            # Build Universe
            universe = set()
            for s in moc_strategies:
                universe.update(s['universe_tickers'])
                if s['settings'].get('use_market_sznl'): universe.add(s['settings'].get('market_ticker', '^GSPC'))
                if "Market" in s['settings'].get('trend_filter', ''): universe.add('SPY')
            
            universe = list(universe)
            data_dict = download_historical_data(universe, start_date="2023-01-01") 
            st.session_state['master_data_dict'].update(data_dict)
            
            results = []
            progress_bar = st.progress(0)
            
            for i, strat in enumerate(moc_strategies):
                # Market Context
                mkt_ticker = strat['settings'].get('market_ticker', 'SPY')
                mkt_df = st.session_state['master_data_dict'].get(mkt_ticker)
                if mkt_df is None: mkt_df = st.session_state['master_data_dict'].get('SPY')
                
                market_series = None
                if mkt_df is not None:
                    market_series = mkt_df['Close'] > mkt_df['Close'].rolling(200).mean()

                for ticker in strat['universe_tickers']:
                    t_clean = ticker.replace('.', '-')
                    df = st.session_state['master_data_dict'].get(t_clean)
                    if df is None or len(df) < 50: continue
                    
                    try:
                        calc_df = calculate_indicators(df, sznl_map, t_clean, market_series)
                        audit = get_signal_breakdown(calc_df, strat['settings'], sznl_map)
                        
                        # --- STRICT FILTERING LOGIC ---
                        
                        # 1. Structural/Stable Failures = IMMEDIATE REJECTION
                        if audit['Stable_Fail'] > 0:
                            continue
                            
                        # 2. Volatile Failures = CHECK TOLERANCE
                        v_fails = audit['Volatile_Fail']
                        
                        if v_fails <= max_volatile_fails:
                            
                            # Determine Status
                            if v_fails == 0:
                                status = "✅ CONFIRMED"
                                color = "green"
                                sort_key = 0
                            else:
                                missing_str = ", ".join(audit['Volatile_Items'])
                                status = f"⚠️ MISSING: {missing_str}"
                                color = "orange"
                                sort_key = 1
                                
                            results.append({
                                "Strategy": strat['name'],
                                "Ticker": ticker,
                                "Price": calc_df['Close'].iloc[-1],
                                "Range%": calc_df['RangePct'].iloc[-1] * 100,
                                "Status": status,
                                "Failures": audit['Volatile_Items'],
                                "Full_Audit": audit,
                                "_sort": sort_key
                            })
                            
                    except Exception as e:
                        continue
                
                progress_bar.progress((i + 1) / len(moc_strategies))
            progress_bar.empty()
            
            if not results:
                st.info("No signals found matching strict criteria.")
            else:
                res_df = pd.DataFrame(results)
                st.subheader(f"Found {len(res_df)} Candidates")
                
                # Sort: Confirmed (0) -> Watchlist (1), then Ticker
                res_df.sort_values(by=['_sort', 'Ticker'], ascending=[True, True], inplace=True)
                
                for _, row in res_df.iterrows():
                    is_conf = row['_sort'] == 0
                    icon = "🟢" if is_conf else "🟠"
                    
                    label = f"{icon} **{row['Ticker']}** | {row['Strategy']} | {row['Status']}"
                    
                    with st.expander(label, expanded=is_conf):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Price", f"${row['Price']:.2f}")
                        c2.metric("Range Loc", f"{row['Range%']:.1f}%")
                        c3.caption(f"Strategy: {row['Strategy']}")
                        
                        # Filter audit to only show relevant rows
                        audit_data = row['Full_Audit']
                        check_rows = []
                        # Priority sort: Show Failed items first
                        keys = sorted(audit_data.keys(), key=lambda x: 0 if x in row['Failures'] else 1)
                        
                        for k in keys:
                            if "_Pass" in k:
                                param = k.replace("_Pass", "")
                                if param in ["Stable_Fail", "Volatile_Fail", "Volatile_Items", "Stable_Items"]: continue
                                
                                val = audit_data.get(param, "-")
                                status_icon = audit_data.get(k)
                                
                                # Highlight the missing volatile items
                                if param in row['Failures']:
                                    status_icon = "⚠️ WATCH"
                                    
                                check_rows.append({"Condition": param, "Value": val, "Status": status_icon})
                                
                        st.dataframe(pd.DataFrame(check_rows), hide_index=True, use_container_width=True)

    # --- EOD BATCH RUNNER (Keep existing) ---
    elif mode == "EOD Batch Runner":
        # ... (Same as before) ...
        st.title("⚡ EOD Batch Screener")
        if st.button("Run EOD Batch", type="primary"):
            run_eod_batch_logic(sznl_map)

# ... (rest of file/helper functions remain same) ...
