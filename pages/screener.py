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

# -----------------------------------------------------------------------------
# IMPORT STRATEGY BOOK FROM ROOT
# -----------------------------------------------------------------------------
# Get the absolute path of the current file (pages/screener.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (root of the repo)
parent_dir = os.path.dirname(current_dir)
# Add parent directory to sys.path
sys.path.append(parent_dir)

# Now we can import the file from the root
try:
    from strategy_config import STRATEGY_BOOK
except ImportError:
    st.error("Could not find strategy_config.py in the root directory.")
    STRATEGY_BOOK = []
# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
CSV_PATH = "sznl_ranks.csv" # Ensure this matches your file name

@st.cache_resource 
def load_seasonal_map():
    """
    Loads the CSV and creates a dictionary of TimeSeries for each ticker.
    Structure: { 'SPY': pd.Series(index=Datetime, data=Rank) }
    """
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        return {}

    if df.empty: return {}

    # Ensure valid dates
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])
     
    # Normalize to midnight (remove time component if present)
    df["Date"] = df["Date"].dt.normalize()
     
    output_map = {}
    # Group by ticker and create a sorted Series for each
    for ticker, group in df.groupby("ticker"):
        # We set the index to Date and ensure it is sorted for 'asof' lookups
        series = group.set_index("Date")["seasonal_rank"].sort_index()
        output_map[ticker] = series

    return output_map

def get_sznl_val_series(ticker, dates, sznl_map):
    """
    Looks up the seasonal rank for the specific dates provided.
    Includes logic to fallback to SPY if ^GSPC is requested but not found.
    """
    ticker = ticker.upper()
    
    # CHANGE: Don't default to {}, default to None. 
    # This avoids the "ambiguous truth value" error when checking if it exists.
    t_series = sznl_map.get(ticker)
    
    # FALLBACK: If looking for ^GSPC but not in map, try SPY
    if t_series is None and ticker == "^GSPC":
        t_series = sznl_map.get("SPY")

    # If still None, return default 50s
    if t_series is None:
        return pd.Series(50.0, index=dates)
        
    return dates.map(t_series).fillna(50.0)
def save_staging_orders(signals_list, strategy_book, sheet_name='Order_Staging'):
    """
    Saves instructions for the Python Execution Engine.
    UPDATED: Handles "T+1 Close if < Signal Close" as a Limit Order (Close - 0.01).
    """
    if not signals_list: return

    # 1. Convert to DataFrame
    df = pd.DataFrame(signals_list)
    
    # 2. Create a lookup for Strategy Settings
    strat_map = {s['id']: s for s in strategy_book}
    
    staging_data = []
    
    for _, row in df.iterrows():
        strat = strat_map.get(row['Strategy_ID'])
        if not strat: continue
        
        settings = strat['settings']
        
        # --- A. DECODE ENTRY INSTRUCTION ---
        entry_mode = settings.get('entry_type', 'Signal Close')
        
        # Defaults
        entry_instruction = "MKT" 
        offset_atr = 0.0
        limit_price = 0.0 # NEW: Explicit Limit Price field
        tif_instruction = "DAY" # NEW: Time in Force (Day, GTC, OPG)

        # 1. ATR LIMIT ENTRY (e.g. Open - 0.5 ATR)
        if "Limit" in entry_mode and "ATR" in entry_mode:
            entry_instruction = "REL_OPEN" 
            if "0.5" in entry_mode: offset_atr = 0.5
            tif_instruction = "DAY"
            
        # 2. MARKET ON OPEN
        elif "T+1 Open" in entry_mode:
            entry_instruction = "MOO" 
            tif_instruction = "OPG"
            
        # 3. MARKET ON CLOSE (Signal Close)
        elif "Signal Close" in entry_mode:
            entry_instruction = "MOC" # Changed from MKT to MOC for clarity, or keep MKT
            tif_instruction = "DAY"

        # 4. *** NEW: OVERSOLD LOW VOLUME LOGIC ***
        elif "T+1 Close if < Signal Close" in entry_mode:
            entry_instruction = "LMT"
            # Logic: We want a Limit Order at Signal Close - 0.01
            # Note: row['Entry'] currently holds the Signal Close price from the main loop
            limit_price = row['Entry'] - 0.01
            tif_instruction = "DAY" # Good for the Day

        # --- B. PARENT ACTION ---
        ib_action = "SELL" if "SHORT" in row['Action'] else "BUY"

        staging_data.append({
            "Scan_Date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "Symbol": row['Ticker'],
            "SecType": "STK",
            "Exchange": "SMART",
            "Action": ib_action,
            "Quantity": row['Shares'],
            
            # THE INSTRUCTIONS
            "Order_Type": entry_instruction,  # MOO, REL_OPEN, MKT, LMT
            "Limit_Price": round(limit_price, 2), # NEW COLUMN: Specific Limit Price
            "Offset_ATR_Mult": offset_atr,    # e.g., 0.5
            "TIF": tif_instruction,           # NEW COLUMN: DAY or OPG
            "Frozen_ATR": round(row['ATR'], 2), 
            
            # EXIT DATA (Time Stop Only)
            "Time_Exit_Date": str(row['Time Exit']),
            "Strategy_Ref": strat['name']
        })

    df_stage = pd.DataFrame(staging_data)

    try:
        # Load Credentials
        if "gcp_service_account" in st.secrets:
            creds_dict = st.secrets["gcp_service_account"]
            gc = gspread.service_account_from_dict(creds_dict)
        else:
            gc = gspread.service_account(filename='credentials.json')

        sh = gc.open("Trade_Signals_Log")
        
        # Create/Open Tab
        try:
            worksheet = sh.worksheet(sheet_name)
        except:
            worksheet = sh.add_worksheet(title=sheet_name, rows=100, cols=20)

        # Clear and Overwrite
        worksheet.clear()
        data_to_write = [df_stage.columns.tolist()] + df_stage.astype(str).values.tolist()
        worksheet.update(values=data_to_write)
        st.toast(f"🤖 Instructions Staged! ({len(df_stage)} rows)")
        
    except Exception as e:
        st.error(f"❌ Staging Sheet Error: {e}")
        
def save_signals_to_gsheet(new_dataframe, sheet_name='Trade_Signals_Log'):
    if new_dataframe.empty: return

    # Prepare Data
    df_new = new_dataframe.copy()
    cols_to_round = ['Entry', 'Stop', 'Target', 'ATR']
    existing_cols = [c for c in cols_to_round if c in df_new.columns]
    df_new[existing_cols] = df_new[existing_cols].astype(float).round(2)
    df_new['Date'] = df_new['Date'].astype(str) 
    df_new["Scan_Timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cols = ['Scan_Timestamp'] + [c for c in df_new.columns if c != 'Scan_Timestamp']
    df_new = df_new[cols]

    try:
        if "gcp_service_account" in st.secrets:
            creds_dict = st.secrets["gcp_service_account"]
            gc = gspread.service_account_from_dict(creds_dict)
        else:
            gc = gspread.service_account(filename='credentials.json')

        sh = gc.open(sheet_name)
        worksheet = sh.sheet1 
        
        existing_data = worksheet.get_all_values()
        if existing_data:
            headers = existing_data[0]
            df_existing = pd.DataFrame(existing_data[1:], columns=headers)
        else:
            df_existing = pd.DataFrame()

        if not df_existing.empty:
            df_existing = df_existing.reindex(columns=df_new.columns)
            combined = pd.concat([df_existing, df_new])
        else:
            combined = df_new

        # Drop Duplicates (Same Ticker/Date/Strategy)
        combined = combined.drop_duplicates(subset=['Ticker', 'Date', 'Strategy_ID'], keep='last')
        
        worksheet.clear()
        data_to_write = [combined.columns.tolist()] + combined.astype(str).values.tolist()
        worksheet.update(values=data_to_write)
        st.toast(f"✅ Synced! Sheet now has {len(combined)} total rows.")
        
    except Exception as e:
        st.error(f"❌ Google Sheet Error: {e}")

def is_after_market_close():
    """Checks if the current time is 4:00 PM EST/EDT or later."""
    tz = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(tz)
    # Market closes at 16:00
    return now.hour >= 16
    
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
    df['GapCount_10'] = is_open_gap.rolling(10).sum()
    df['GapCount_5'] = is_open_gap.rolling(5).sum() 

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
    # --- Volume, Accumulation & Distribution Logic ---
    vol_ma = df['Volume'].rolling(63).mean()
    df['vol_ratio'] = df['Volume'] / vol_ma
    df['vol_ma'] = vol_ma
    
    # Base Conditions
    cond_vol_ma = df['Volume'] > vol_ma
    cond_vol_up = df['Volume'] > df['Volume'].shift(1)
    
    # Create explicit Vol Spike Column (True/False)
    df['Vol_Spike'] = cond_vol_ma & cond_vol_up
    
    # 1. Accumulation (Green + Spike)
    cond_green = df['Close'] > df['Open']
    is_accumulation = (df['Vol_Spike'] & cond_green).astype(int)
    df['AccCount_21'] = is_accumulation.rolling(21).sum()
    
    # 2. Distribution (Red + Spike)
    cond_red = df['Close'] < df['Open']
    is_distribution = (df['Vol_Spike'] & cond_red).astype(int)
    df['DistCount_21'] = is_distribution.rolling(21).sum()
    
    # --- Volume Rank ---
    vol_ma_10 = df['Volume'].rolling(10).mean()
    df['vol_ratio_10d'] = vol_ma_10 / vol_ma
    df['vol_ratio_10d_rank'] = df['vol_ratio_10d'].expanding(min_periods=50).rank(pct=True) * 100.0
    
    # --- Seasonality ---
    df['Sznl'] = get_sznl_val_series(ticker, df.index, sznl_map)
    df['Mkt_Sznl_Ref'] = get_sznl_val_series("^GSPC", df.index, sznl_map)
    
    # --- Age & Market Regime ---
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
    
def check_signal(df, params, sznl_map):
    last_row = df.iloc[-1]
    
    # 0. Day of Week Filter
    if params.get('use_dow_filter', False):
        allowed = params.get('allowed_days', [])
        current_day = last_row.name.dayofweek
        if current_day not in allowed: return False

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

    # 2b. MA Consecutive Filters (NEW - For Ugly Monday)
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
        
    # 8b. Exclude 52w High (NEW)
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
    Runs the exact check_signal logic but returns a dictionary of 
    Values and Pass/Fail statuses for inspection.
    """
    last_row = df.iloc[-1]
    audit = {}
    all_passed = True
    
    # Helper to log result
    def log(key, value, condition_met, details=""):
        nonlocal all_passed
        audit[key] = value
        audit[f"{key}_Pass"] = "✅" if condition_met else "❌"
        if not condition_met: all_passed = False

    # -----------------------------------------------------------
    # 0. DAY OF WEEK FILTER (NEWLY ADDED TO INSPECTOR)
    # -----------------------------------------------------------
    if params.get('use_dow_filter', False):
        allowed = params.get('allowed_days', [])
        # 0=Mon, 1=Tue, ... 6=Sun
        day_num = last_row.name.dayofweek 
        day_name = last_row.name.strftime("%A") 
        
        cond = day_num in allowed
        
        # Format "Monday (0) vs Allowed [0]"
        log("DOW_Check", f"{day_name} (Req: {allowed})", cond)

    # 1. Liquidity
    liq_pass = (
        last_row['Close'] >= params.get('min_price', 0) and 
        last_row['vol_ma'] >= params.get('min_vol', 0) and
        last_row['age_years'] >= params.get('min_age', 0)
    )
    log("Liquidity", f"${last_row['Close']:.2f} / {int(last_row['vol_ma'])} vol", liq_pass)

    # 2. Trend (Global)
    trend_opt = params.get('trend_filter', 'None')
    trend_res = True
    trend_val = "None"
    
    if trend_opt == "Price > 200 SMA":
        trend_res = last_row['Close'] > last_row['SMA200']
        trend_val = f"{last_row['Close']:.2f} vs {last_row['SMA200']:.2f}"
    elif "Market" in trend_opt or "SPY" in trend_opt:
        if 'Market_Above_SMA200' in df.columns:
            is_above = last_row['Market_Above_SMA200']
            if ">" in trend_opt: trend_res = is_above
            elif "<" in trend_opt: trend_res = not is_above
            trend_val = "Mkt > SMA200" if is_above else "Mkt < SMA200"
            
    log("Trend", trend_val, trend_res)

    # 2b. MA Specific Filters
    if 'ma_consec_filters' in params:
        for i, maf in enumerate(params['ma_consec_filters']):
            length = maf['length']
            col_name = f"SMA{length}"
            
            val = last_row.get(col_name, 0)
            
            # Logic Check
            if maf['logic'] == 'Above':
                mask = df['Close'] > df[col_name]
                symbol = ">"
            else:
                mask = df['Close'] < df[col_name]
                symbol = "<"
            
            # Consecutive
            consec = maf.get('consec', 1)
            consec_text = ""
            if consec > 1:
                mask = mask.rolling(consec).sum() == consec
                consec_text = f" ({consec}d)"
                
            cond = mask.iloc[-1]
            log(f"MA_Check_{i}", f"Price {symbol} SMA{length}{consec_text}", cond)

    # 3. Perf Rank
    if 'perf_filters' in params:
        for i, pf in enumerate(params['perf_filters']):
            window = pf['window']
            col = f"rank_ret_{window}d"
            val = last_row.get(col, 0)
            
            if pf['logic'] == '<': cond = val < pf['thresh']
            else: cond = val > pf['thresh']
            
            consec = pf.get('consecutive', 1)
            consec_str = ""
            if consec > 1:
                if pf['logic'] == '<': s_cond = df[col] < pf['thresh']
                else: s_cond = df[col] > pf['thresh']
                recent_sum = s_cond.rolling(consec).sum().iloc[-1]
                cond = (recent_sum == consec)
                consec_str = f" ({int(recent_sum)}/{consec} days)"

            log(f"Perf_{window}d", f"{val:.1f}{consec_str}", cond)

    # 4. Seasonality
    if params['use_sznl']:
        val = last_row['Sznl']
        if params['sznl_logic'] == '<': cond = val < params['sznl_thresh']
        else: cond = val > params['sznl_thresh']
        log("Sznl", f"{val:.1f}", cond)

    # 5. Market Seasonality
    if params.get('use_market_sznl', False):
        val = last_row['Mkt_Sznl_Ref']
        if params['market_sznl_logic'] == '<': cond = val < params['market_sznl_thresh']
        else: cond = val > params['market_sznl_thresh']
        log("Mkt_Sznl", f"{val:.1f}", cond)

    # 6. Volume
    if params.get('use_vol', False):
        ratio = last_row['vol_ratio']
        cond = (ratio > params['vol_thresh'])
        log("Vol_Spike", f"x{ratio:.2f}", cond)
        
    if params.get('use_vol_rank', False):
        ratio = last_row['vol_ratio_10d_rank']
        logic = params.get('vol_rank_logic', '<')
        thresh = params['vol_rank_thresh']
        
        # Dynamically check logic based on strategy settings
        if logic == '<':
            cond = (ratio < thresh)
        else:
            cond = (ratio > thresh)
            
        log(f"Vol_Rank ({logic} {thresh})", f"{ratio:.2f}", cond)
    # -----------------------------------------------------------
    # 7. RANGE FILTER (EXPANDED INSPECTOR)
    # -----------------------------------------------------------
    if params.get('use_range_filter', False):
        rng = last_row['RangePct'] * 100
        r_min = params.get('range_min', 0)
        r_max = params.get('range_max', 100)
        
        cond = (rng >= r_min) and (rng <= r_max)
        
        # Explicit output: e.g., "12.5% (Req: 0.0-15.0%)"
        log("Range_Loc", f"{rng:.1f}% (Req: {r_min}-{r_max}%)", cond)
    # -----------------------------------------------------------
        
    # 8. Accumulation Count
    if params.get('use_acc_count_filter', False):
        val = last_row[f"AccCount_{params['acc_count_window']}"]
        thresh = params['acc_count_thresh']
        logic = params['acc_count_logic']
        
        if logic == "=": cond = val == thresh
        elif logic == ">": cond = val > thresh
        elif logic == "<": cond = val < thresh
        log("Acc_Count", f"{int(val)}", cond)
        
    # 9. Exclude 52w High
    if params.get('exclude_52w_high', False):
        is_high = last_row['is_52w_high']
        log("Not_52w_High", f"{is_high}", not is_high)

    audit['Result'] = "✅ SIGNAL" if all_passed else ""
    return audit
# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False) # Cache results so re-runs are instant
def download_historical_data(tickers, start_date="2000-01-01"):
    """
    Downloads data in chunks of 50 to avoid rate limits and memory crashes.
    Returns a dictionary: { 'TICKER': pd.DataFrame }
    """
    if not tickers: return {}
    
    # Deduplicate and clean
    clean_tickers = list(set([str(t).strip().upper().replace('.', '-') for t in tickers]))
    
    data_dict = {}
    CHUNK_SIZE = 50 
    total = len(clean_tickers)
    
    # UI Elements for progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, total, CHUNK_SIZE):
        chunk = clean_tickers[i : i + CHUNK_SIZE]
        
        # Update UI
        current_progress = min((i + CHUNK_SIZE) / total, 1.0)
        status_text.text(f"📥 Downloading batch {i+1}-{min(i+CHUNK_SIZE, total)} of {total}...")
        progress_bar.progress(current_progress)
        
        try:
            # Download Chunk
            df = yf.download(
                chunk, 
                start=start_date, 
                group_by='ticker', 
                auto_adjust=False, # Keep original Close/Adj Close logic usually
                progress=False, 
                threads=True
            )
            
            if df.empty: continue
            
            # CASE A: Single Ticker in Chunk (Yahoo returns Flat Index)
            if len(chunk) == 1:
                ticker = chunk[0]
                # If yahoo failed, df might be empty or missing cols
                if 'Close' in df.columns:
                    # Clean Index
                    df.index = df.index.tz_localize(None)
                    data_dict[ticker] = df
            
            # CASE B: Multiple Tickers (Yahoo returns MultiIndex)
            else:
                # Iterate through the columns levels to extract valid dfs
                # Note: yf.download with group_by='ticker' makes level 0 the ticker
                available_tickers = df.columns.levels[0]
                
                for t in available_tickers:
                    try:
                        t_df = df[t].copy()
                        # specific check to ensure it's not empty
                        if t_df.empty or 'Close' not in t_df.columns: continue
                        
                        t_df.index = t_df.index.tz_localize(None)
                        data_dict[t] = t_df
                    except Exception:
                        continue
            
            # Sleep briefly to be nice to the API
            time.sleep(0.25)
            
        except Exception as e:
            st.warning(f"⚠️ Error downloading chunk starting {chunk[0]}: {e}")
            continue

    # Cleanup UI
    progress_bar.empty()
    status_text.empty()
    
    return data_dict

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Production Strategy Screener")
    st.title("⚡ Daily Strategy Screener (Batch Optimized)")
    st.markdown("---")
    
    sznl_map = load_seasonal_map()
    
    # Initialize Session State for Data if not exists
    if 'master_data_dict' not in st.session_state:
        st.session_state['master_data_dict'] = {}

    if st.button("Run All Strategies", type="primary", use_container_width=True):
        
        # 1. Gather ALL tickers first (Efficiency Step)
        all_required_tickers = set()
        for strat in STRATEGY_BOOK:
            all_required_tickers.update(strat['universe_tickers'])
            
            # Add Market/Trend Tickers
            s = strat['settings']
            if s.get('use_market_sznl'): all_required_tickers.add(s.get('market_ticker', '^GSPC'))
            if "Market" in s.get('trend_filter', ''): all_required_tickers.add(s.get('market_ticker', 'SPY'))
            if "SPY" in s.get('trend_filter', ''): all_required_tickers.add("SPY")
            
        # Clean names
        all_required_tickers = {t.replace('.', '-') for t in all_required_tickers}

        # 2. Check what is missing from session state
        existing_keys = set(st.session_state['master_data_dict'].keys())
        missing_tickers = list(all_required_tickers - existing_keys)
        
        # 3. Download Missing Data (Batch Mode)
        if missing_tickers:
            st.info(f"Need to fetch history (from 2000-01-01) for {len(missing_tickers)} new tickers.")
            new_data_dict = download_historical_data(missing_tickers, start_date="2000-01-01")
            
            # Merge into session state
            st.session_state['master_data_dict'].update(new_data_dict)
            st.success(f"✅ Data initialized. Total tickers in memory: {len(st.session_state['master_data_dict'])}")

        # 4. Run Strategies
        master_dict = st.session_state['master_data_dict']
        
        # --- NEW: Initialize a master list to collect ALL signals ---
        all_staging_signals = [] 
        # ------------------------------------------------------------

        for i, strat in enumerate(STRATEGY_BOOK):
            with st.expander(f"Strategy: {strat['name']} (Grade: {strat['stats']['grade']})", expanded=False):
                
                # Stats Header
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric("Win Rate", strat['stats']['win_rate'])
                s2.metric("Expectancy", strat['stats']['expectancy'])
                s3.metric("Profit Factor", strat['stats']['profit_factor'])
                s4.metric("Direction", strat['settings'].get('trade_direction', 'Long'))
                s5.metric("Risk Unit", f"${strat['execution']['risk_per_trade']}")
                
                # Prepare Market Series for this specific strategy
                strat_mkt_ticker = strat['settings'].get('market_ticker', 'SPY')
                market_series = None
                
                # Try to get market data from dict
                mkt_df = master_dict.get(strat_mkt_ticker)
                if mkt_df is None: mkt_df = master_dict.get('SPY') # Fallback
                
                if mkt_df is not None:
                    # Calculate SMA200 for Market
                    temp_mkt = mkt_df.copy()
                    temp_mkt['SMA200'] = temp_mkt['Close'].rolling(200).mean()
                    market_series = temp_mkt['Close'] > temp_mkt['SMA200']

                signals = []
                
                # Iterate Universe
                for ticker in strat['universe_tickers']:
                    t_clean = ticker.replace('.', '-')
                    
                    df = master_dict.get(t_clean)
                    if df is None: continue
                    if len(df) < 250: continue
                    
                    try:
                        # Copy to avoid contaminating master dict
                        calc_df = df.copy() 
                        
                        calc_df = calculate_indicators(calc_df, sznl_map, t_clean, market_series)
                        
                        if check_signal(calc_df, strat['settings'], sznl_map):
                            last_row = calc_df.iloc[-1]
                            
                            # Entry Confirmation Check
                            entry_conf_bps = strat['settings'].get('entry_conf_bps', 0)
                            entry_mode = strat['settings'].get('entry_type', 'Signal Close')
                            if entry_mode == 'Signal Close' and entry_conf_bps > 0:
                                threshold = last_row['Open'] * (1 + entry_conf_bps/10000.0)
                                if last_row['High'] < threshold: continue

                            atr = last_row['ATR']
                            # --- DYNAMIC RISK SIZING (VOLATILITY MULTIPLIER) ---
                            # Default to the strategy settings
                            risk = strat['execution']['risk_per_trade']
                            
                            # Check if this is the specific strategy and apply multipliers
                            if strat['name'] == "Overbot Vol Spike":
                                # Ensure we look at the ratio safely
                                vol_ratio = last_row.get('vol_ratio', 0)
                                
                                if vol_ratio > 2.0:
                                    risk = 675  # High conviction
                                elif vol_ratio > 1.5:
                                    risk = 525  # Medium conviction
                                # else: stays at default 400 (Low conviction > 1.25)
                            if strat['name'] == "Weak Close Decent Sznls":
                                sznl_val = last_row.get('Sznl', 0)
                                
                                if sznl_val >= 65:
                                    risk = risk * 1.5   # High conviction (e.g. 400 * 1.5 = 600)
                                elif sznl_val >= 50:
                                    risk = risk * 1.0   # Standard (e.g. 400)
                                elif sznl_val >= 33:
                                    risk = risk * 0.66
                                    
                            entry = last_row['Close']
                            direction = strat['settings'].get('trade_direction', 'Long')
                            stop_atr = strat['execution']['stop_atr']
                            tgt_atr = strat['execution']['tgt_atr']
                            
                            if direction == 'Long':
                                stop_price = entry - (atr * stop_atr)
                                tgt_price = entry + (atr * tgt_atr)
                                dist = entry - stop_price
                                action = "BUY"
                            else:
                                stop_price = entry + (atr * stop_atr)
                                tgt_price = entry - (atr * tgt_atr)
                                dist = stop_price - entry
                                action = "SELL SHORT"
                            
                            shares = int(risk / dist) if dist > 0 else 0
                            exit_date = (last_row.name + BusinessDay(strat['execution']['hold_days'])).date()
                            
                            signals.append({
                                "Strategy_ID": strat['id'],
                                "Ticker": ticker,
                                "Date": last_row.name.date(),
                                "Action": action,
                                "Shares": shares,
                                "Entry": entry,
                                "Stop": stop_price,
                                "Target": tgt_price,
                                "Time Exit": exit_date,
                                "ATR": atr
                            })
                    except Exception as e:
                        continue
                
                # --- PROCESSING SIGNALS ---
                if signals:
                    # 1. Add to the Master List (for Staging later)
                    all_staging_signals.extend(signals)
                    
                    st.success(f"✅ Found {len(signals)} Actionable Signals")
                    sig_df = pd.DataFrame(signals)
                    
                    if is_after_market_close():
                        save_signals_to_gsheet(sig_df, sheet_name='Trade_Signals_Log')
                    else:
                        st.info("🕒 Signals displayed but NOT stored (it is before 4:00 PM EST).")
                    
                    # 3. Display UI
                    st.dataframe(sig_df.style.format({"Entry": "${:.2f}", "Stop": "${:.2f}", "Target": "${:.2f}", "ATR": "{:.2f}"}), use_container_width=True)
                    
                    clip = ""
                    for s in signals:
                        clip += f"{s['Action']} {s['Shares']} {s['Ticker']} @ MKT. Stop: {s['Stop']:.2f}. Target: {s['Target']:.2f}. Exit Date: {s['Time Exit']}.\n"
                    st.text_area(f"Clipboard", clip, height=80)
                else:
                    st.caption("No signals found.")

        # --- FINAL STEP (OUTSIDE THE LOOP) ---
        # Save to Order_Staging only ONCE so we don't overwrite previous loop iterations
        if all_staging_signals:
            if is_after_market_close():
                st.divider()
                st.subheader("🚀 Staging Execution Orders")
                save_staging_orders(all_staging_signals, STRATEGY_BOOK, sheet_name='Order_Staging')
            else:
                st.warning("⚠️ Execution staging skipped: Orders only stage after 4:00 PM EST.")
        # -------------------------------------
    # -------------------------------------------------------------------------
    # DEBUG: DEEP DIVE & SCORECARD
    # -------------------------------------------------------------------------
    st.markdown("---")
    
    # 1. Allow User Input (Defaulting to LUV to preserve original behavior)
    debug_ticker = st.text_input("Enter Ticker for Deep Dive:", value="LUV").upper().strip()
    
    st.subheader(f"🔎 {debug_ticker} Deep Dive (Debug & Scorecard)")
    
    if st.button(f"Inspect {debug_ticker} Data & Signals"):
        # 2. Fetch Data
        st.write(f"Fetching {debug_ticker}, SPY, & ^GSPC data...")
        
        # Ensure we ask for the user ticker + market refs (using set to avoid duplicates)
        debug_tickers = list(set([debug_ticker, "^GSPC", "SPY"]))
        
        debug_raw = yf.download(debug_tickers, start='2000-01-01', group_by='ticker', progress=False)
        
        # 3. Extract & Calculate
        # We check if the user's ticker exists in the downloaded data
        if debug_ticker in debug_raw.columns.levels[0]:
            target_df = debug_raw[debug_ticker].copy()
            
            # Market Regime (SPY > 200 SMA)
            spy_df = debug_raw["SPY"].copy()
            spy_df['SMA200'] = spy_df['Close'].rolling(200).mean()
            market_series = spy_df['Close'] > spy_df['SMA200']
            
            # Run Indicators (Pass the dynamic ticker name to the function)
            target_df = calculate_indicators(target_df, sznl_map, debug_ticker, market_series)

            # -----------------------------------------------------------------
            # PART A: DATA INDICATORS (Modified View)
            # -----------------------------------------------------------------
            cols_to_show = [
                'Close', 'Volume', 
                'vol_ratio_10d_rank', 'Vol_Spike',         # Added Vol_Spike
                'AccCount_21', 'DistCount_21',    # Added Counts
                'Sznl', 'Mkt_Sznl_Ref', 
                'rank_ret_5d', 'rank_ret_21d',    # Kept Ranks, Removed Raw %
                'SMA200', 'Market_Above_SMA200'
            ]
            final_cols = [c for c in cols_to_show if c in target_df.columns]
            
            st.write("### 1. Underlying Data Indicators (Last 5 Days)")
            st.dataframe(
                target_df[final_cols].tail(5).style.format({
                    'Close': '{:.2f}',
                    'Volume': '{:,.0f}',
                    'vol_ratio_10d_rank': '{:.2f}',
                    'Vol_Spike': '{}',           # Displays True/False
                    'AccCount_21': '{:.0f}',     # Integer format
                    'DistCount_21': '{:.0f}',    # Integer format
                    'Sznl': '{:.2f}',
                    'Mkt_Sznl_Ref': '{:.2f}',
                    'rank_ret_5d': '{:.2f}',
                    'rank_ret_21d': '{:.2f}',
                    'SMA200': '{:.2f}'
                })
            )

            # -----------------------------------------------------------------
            # PART B: STRATEGY SCORECARD
            # -----------------------------------------------------------------
            st.write(f"### 2. Strategy Scorecard (Date: {target_df.index[-1].date()})")
            
            scorecard = []
            for strat in STRATEGY_BOOK:
                # Check signals on the dynamic dataframe
                passed = check_signal(target_df, strat['settings'], sznl_map)
                
                scorecard.append({
                    "Strategy Name": strat['name'],
                    "Direction": strat['settings'].get('trade_direction', 'Long'),
                    "Pass (1) / Fail (0)": "✅ 1 (PASS)" if passed else "0 (FAIL)"
                })
            
            st.table(pd.DataFrame(scorecard))

            # -----------------------------------------------------------------
            # PART C: INTEGRITY CHECKS
            # -----------------------------------------------------------------
            last_row = target_df.iloc[-1]
            if last_row['Mkt_Sznl_Ref'] == 50.0:
                    st.error("⚠️ Note: ^GSPC Rank is 50.0. This may indicate missing seasonal data.")
            else:
                    st.caption("✅ Seasonal Data Integrity: OK")

        else:
            st.error(f"Could not download data for {debug_ticker}. Please check the symbol.")
        # ... (After the "Deep Dive" IF block ends) ...
    
    st.markdown("---")
    st.subheader("🕵️‍♀️ Historical Strategy Inspector (Last 21 Days)")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        insp_ticker = st.text_input("Inspector Ticker:", value="AMD").upper().strip()
    with c2:
        # Create a mapping of names to Strategy Objects
        strat_map_ui = {s['name']: s for s in STRATEGY_BOOK}
        selected_strat_name = st.selectbox("Select Strategy to Audit:", list(strat_map_ui.keys()))
        
    if st.button("Run Historical Audit"):
        target_strat = strat_map_ui[selected_strat_name]
        
        # 1. Fetch Data
        if insp_ticker not in st.session_state['master_data_dict']:
             # Quick fetch if not in memory
             st.write(f"Fetching {insp_ticker}...")
             tmp_dict = download_historical_data([insp_ticker, "SPY", "^GSPC"])
             st.session_state['master_data_dict'].update(tmp_dict)

        # 2. Get DataFrames
        df_insp = st.session_state['master_data_dict'].get(insp_ticker)
        
        # Get Market Refs
        mkt_ticker = target_strat['settings'].get('market_ticker', 'SPY')
        df_mkt = st.session_state['master_data_dict'].get(mkt_ticker)
        if df_mkt is None: df_mkt = st.session_state['master_data_dict'].get('SPY')

        if df_insp is not None and len(df_insp) > 50:
            # 3. Prep Data (Calculate Indicators)
            # Market SMA Series
            mkt_series = None
            if df_mkt is not None:
                temp_mkt = df_mkt.copy()
                temp_mkt['SMA200'] = temp_mkt['Close'].rolling(200).mean()
                mkt_series = temp_mkt['Close'] > temp_mkt['SMA200']

            calc_df = df_insp.copy()
            calc_df = calculate_indicators(calc_df, sznl_map, insp_ticker, mkt_series)
            
            # 4. Loop Logic (The Audit)
            audit_rows = []
            
            # We look at the last 21 trading days
            days_to_audit = 252
            # Ensure we have enough data
            if len(calc_df) > days_to_audit:
                subset_indices = range(len(calc_df) - days_to_audit, len(calc_df))
                
                for idx in subset_indices:
                    # We create a slice up to that specific day to simulate "point in time"
                    # Note: calculate_indicators is already done, so we don't need to re-calc MAs.
                    # We just need to ensure things like 'consecutive' logic work.
                    # Since consecutive logic checks .rolling().sum(), looking at the row 'idx'
                    # in the fully calculated dataframe is technically looking-ahead IF 
                    # the indicator calculation itself used future data (which it doesn't, it uses rolling).
                    # So we can pass the truncated dataframe ending at idx.
                    
                    slice_df = calc_df.iloc[:idx+1] 
                    row_date = slice_df.index[-1].date()
                    
                    # Run the breakdown
                    breakdown = get_signal_breakdown(slice_df, target_strat['settings'], sznl_map)
                    breakdown['Date'] = row_date
                    audit_rows.append(breakdown)
            
            # 5. Display
            if audit_rows:
                audit_df = pd.DataFrame(audit_rows)
                
                # Reorder cols to put Date and Result first
                cols = ['Date', 'Result'] + [c for c in audit_df.columns if c not in ['Date', 'Result']]
                audit_df = audit_df[cols]
                audit_df = audit_df.sort_values(by='Date', ascending=False)
                st.write(f"**Audit Results for {insp_ticker} on '{selected_strat_name}'**")
                
                # Dynamic Coloring for the Result Column
                def highlight_signal(s):
                    is_signal = s == "✅ SIGNAL"
                    return ['background-color: #d4edda; color: green; font-weight: bold' if is_signal else '' for _ in s]

                st.dataframe(audit_df.style.apply(lambda x: ['background-color: #e6fffa' if x.name == 'Result' and v == "✅ SIGNAL" else '' for v in x], axis=0), use_container_width=True)
            else:
                st.warning("Not enough data to audit.")
        else:
            st.error("Ticker data not found or insufficient history.")

if __name__ == "__main__":
    main()
