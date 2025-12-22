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

# -----------------------------------------------------------------------------
# ORDER SAVING UTILS
# -----------------------------------------------------------------------------
def save_moc_orders(signals_list, strategy_book, sheet_name='moc_orders'):
    try:
        if "gcp_service_account" in st.secrets:
            creds_dict = st.secrets["gcp_service_account"]
            gc = gspread.service_account_from_dict(creds_dict)
        else:
            gc = gspread.service_account(filename='credentials.json')

        sh = gc.open("Trade_Signals_Log")
        try: worksheet = sh.worksheet(sheet_name)
        except: worksheet = sh.add_worksheet(title=sheet_name, rows=100, cols=20)
        worksheet.clear()

        moc_data = []
        if signals_list:
            strat_map = {s['id']: s for s in strategy_book}
            for row in pd.DataFrame(signals_list).to_dict('records'):
                strat = strat_map.get(row['Strategy_ID'])
                if not strat: continue
                settings = strat['settings']
                entry_mode = settings.get('entry_type', 'Signal Close')

                if "Signal Close" in entry_mode:
                    ib_action = "SELL" if "SHORT" in row['Action'] else "BUY"
                    moc_data.append({
                        "Scan_Date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "Symbol": row['Ticker'],
                        "SecType": "STK",
                        "Exchange": "SMART",
                        "Action": ib_action,
                        "Quantity": row['Shares'],
                        "Order_Type": "MOC", 
                        "Strategy_Ref": strat['name'],
                        "Exit_Date": str(row['Time Exit']) 
                    })

        if moc_data:
            df_moc = pd.DataFrame(moc_data)
            data_to_write = [df_moc.columns.tolist()] + df_moc.astype(str).values.tolist()
            worksheet.update(values=data_to_write)
            st.success(f"🚀 Staged {len(df_moc)} MOC Orders with Exit Dates!")
        else:
            headers = ["Scan_Date", "Symbol", "SecType", "Exchange", "Action", "Quantity", "Order_Type", "Strategy_Ref", "Exit_Date"]
            worksheet.update(values=[headers])
            st.toast(f"🧹 '{sheet_name}' cleared.")
        
    except Exception as e:
        st.error(f"❌ MOC Staging Error: {e}")
        
def save_staging_orders(signals_list, strategy_book, sheet_name='Order_Staging'):
    if not signals_list: return
    df = pd.DataFrame(signals_list)
    strat_map = {s['id']: s for s in strategy_book}
    staging_data = []
    
    for _, row in df.iterrows():
        strat = strat_map.get(row['Strategy_ID'])
        if not strat: continue
        settings = strat['settings']
        entry_mode = settings.get('entry_type', 'Signal Close')
        
        if "Signal Close" in entry_mode: continue
        
        entry_instruction = "MKT" 
        offset_atr = 0.0
        limit_price = 0.0 
        tif_instruction = "DAY" 

        if "Limit" in entry_mode and "ATR" in entry_mode:
            entry_instruction = "REL_OPEN" 
            if "0.5" in entry_mode: offset_atr = 0.5
            tif_instruction = "DAY"
        elif "T+1 Open" in entry_mode:
            entry_instruction = "MOO" 
            tif_instruction = "OPG"
        elif "T+1 Close if < Signal Close" in entry_mode:
            entry_instruction = "LMT"
            limit_price = row['Entry'] - 0.01
            tif_instruction = "DAY" 

        ib_action = "SELL" if "SHORT" in row['Action'] else "BUY"
        staging_data.append({
            "Scan_Date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "Symbol": row['Ticker'],
            "SecType": "STK",
            "Exchange": "SMART",
            "Action": ib_action,
            "Quantity": row['Shares'],
            "Order_Type": entry_instruction,  
            "Limit_Price": round(limit_price, 2), 
            "Offset_ATR_Mult": offset_atr,     
            "TIF": tif_instruction,            
            "Frozen_ATR": round(row['ATR'], 2), 
            "Time_Exit_Date": str(row['Time Exit']),
            "Strategy_Ref": strat['name']
        })

    if not staging_data: return

    df_stage = pd.DataFrame(staging_data)
    try:
        if "gcp_service_account" in st.secrets:
            creds_dict = st.secrets["gcp_service_account"]
            gc = gspread.service_account_from_dict(creds_dict)
        else:
            gc = gspread.service_account(filename='credentials.json')

        sh = gc.open("Trade_Signals_Log")
        try: worksheet = sh.worksheet(sheet_name)
        except: worksheet = sh.add_worksheet(title=sheet_name, rows=100, cols=20)

        worksheet.clear()
        data_to_write = [df_stage.columns.tolist()] + df_stage.astype(str).values.tolist()
        worksheet.update(values=data_to_write)
        st.toast(f"🤖 Instructions Staged! ({len(df_stage)} rows)")
    except Exception as e:
        st.error(f"❌ Staging Sheet Error: {e}")
        
def save_signals_to_gsheet(new_dataframe, sheet_name='Trade_Signals_Log'):
    if new_dataframe.empty: return
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

        combined = combined.drop_duplicates(subset=['Ticker', 'Date', 'Strategy_ID'], keep='last')
        worksheet.clear()
        data_to_write = [combined.columns.tolist()] + combined.astype(str).values.tolist()
        worksheet.update(values=data_to_write)
        st.toast(f"✅ Synced! Sheet now has {len(combined)} total rows.")
    except Exception as e:
        st.error(f"❌ Google Sheet Error: {e}")

def is_after_market_close():
    tz = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(tz)
    return now.hour >= 16
    
# -----------------------------------------------------------------------------
# ENGINE
# -----------------------------------------------------------------------------

def calculate_indicators(df, sznl_map, ticker, market_series=None, vix_series=None, gap_window=21, custom_sma_lengths=None, acc_window=None, dist_window=None):
    df = df.copy()
    df.sort_index(inplace=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    
    # --- Standard SMAs ---
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA100'] = df['Close'].rolling(100).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    # --- Dynamic SMAs ---
    if custom_sma_lengths:
        for length in custom_sma_lengths:
            col_name = f"SMA{length}"
            if col_name not in df.columns:
                df[col_name] = df['Close'].rolling(length).mean()

    # --- EMAs ---
    df['EMA8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA11'] = df['Close'].ewm(span=11, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # --- Perf Ranks ---
    for window in [5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window)
        df[f'rank_ret_{window}d'] = df[f'ret_{window}d'].expanding(min_periods=252).rank(pct=True) * 100.0
    
    # --- ATR ---
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
    
    # --- Seasonality ---
    df['Sznl'] = get_sznl_val_series(ticker, df.index, sznl_map)
    df['Mkt_Sznl_Ref'] = get_sznl_val_series("^GSPC", df.index, sznl_map)
    
    # --- 52w High/Low ---
    rolling_high = df['High'].shift(1).rolling(252).max()
    rolling_low = df['Low'].shift(1).rolling(252).min()
    df['is_52w_high'] = df['High'] > rolling_high
    df['is_52w_low'] = df['Low'] < rolling_low
    
    # --- Volume ---
    vol_ma = df['Volume'].rolling(63).mean()
    df['vol_ma'] = vol_ma
    df['vol_ratio'] = df['Volume'] / vol_ma
    
    vol_ma_10 = df['Volume'].rolling(10).mean()
    df['vol_ratio_10d'] = vol_ma_10 / vol_ma
    df['vol_ratio_10d_rank'] = df['vol_ratio_10d'].expanding(min_periods=252).rank(pct=True) * 100.0
    
    # --- Acc/Dist ---
    vol_gt_prev = df['Volume'] > df['Volume'].shift(1)
    vol_gt_ma = df['Volume'] > df['vol_ma']
    is_green = df['Close'] > df['Open']
    is_red = df['Close'] < df['Open']

    df['is_acc_day'] = (is_green & vol_gt_prev & vol_gt_ma).astype(int)
    df['is_dist_day'] = (is_red & vol_gt_prev & vol_gt_ma).astype(int)
    
    # Dynamic Window Accumulation
    if acc_window:
        df[f'AccCount_{acc_window}'] = df['is_acc_day'].rolling(acc_window).sum()
    if dist_window:
        df[f'DistCount_{dist_window}'] = df['is_dist_day'].rolling(dist_window).sum()

    # --- Age & External ---
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

    # --- Range & Gaps ---
    denom = (df['High'] - df['Low'])
    df['RangePct'] = np.where(denom == 0, 0.5, (df['Close'] - df['Low']) / denom)
    df['DayOfWeekVal'] = df.index.dayofweek
    
    is_open_gap = (df['Low'] > df['High'].shift(1)).astype(int)
    df['GapCount'] = is_open_gap.rolling(gap_window).sum()

    # --- Pivots ---
    piv_len = 20 
    roll_max = df['High'].rolling(window=piv_len*2+1, center=True).max()
    df['is_pivot_high'] = (df['High'] == roll_max)
    roll_min = df['Low'].rolling(window=piv_len*2+1, center=True).min()
    df['is_pivot_low'] = (df['Low'] == roll_min)

    df['LastPivotHigh'] = np.where(df['is_pivot_high'], df['High'], np.nan)
    df['LastPivotHigh'] = df['LastPivotHigh'].shift(piv_len).ffill()
    df['LastPivotLow'] = np.where(df['is_pivot_low'], df['Low'], np.nan)
    df['LastPivotLow'] = df['LastPivotLow'].shift(piv_len).ffill()
    
    # --- Breakout Helpers ---
    df['PrevHigh'] = df['High'].shift(1)
    df['PrevLow'] = df['Low'].shift(1)

    return df
    
def check_signal(df, params, sznl_map):
    last_row = df.iloc[-1]
    
    # 0. Day of Week Filter
    if params.get('use_dow_filter', False):
        allowed = params.get('allowed_days', [])
        current_day = last_row.name.dayofweek
        if current_day not in allowed: return False

    # 0b. Cycle Year Filter
    if 'allowed_cycles' in params:
        allowed_cycles = params['allowed_cycles']
        if allowed_cycles and len(allowed_cycles) < 4:
            current_year = last_row.name.year
            cycle_rem = current_year % 4
            if cycle_rem not in allowed_cycles: return False

    # 1. Liquidity Gates
    if last_row['Close'] < params.get('min_price', 0): return False
    if last_row['vol_ma'] < params.get('min_vol', 0): return False
    if last_row['age_years'] < params.get('min_age', 0): return False
    if last_row['age_years'] > params.get('max_age', 100): return False

    if 'ATR_Pct' in df.columns:
        current_atr_pct = last_row['ATR_Pct']
        if current_atr_pct < params.get('min_atr_pct', 0.0): return False
        if current_atr_pct > params.get('max_atr_pct', 1000.0): return False

    if params.get('require_close_gt_open', False):
        if not (last_row['Close'] > last_row['Open']): return False

    # 2. Trend Filter (Global)
    trend_opt = params.get('trend_filter', 'None')
    if trend_opt == "Price > 200 SMA":
        if not (last_row['Close'] > last_row['SMA200']): return False
    elif trend_opt == "Price > Rising 200 SMA":
        prev_row = df.iloc[-2]
        if not ((last_row['Close'] > last_row['SMA200']) and (last_row['SMA200'] > prev_row['SMA200'])): return False
    elif trend_opt == "Not Below Declining 200 SMA":
        prev_row = df.iloc[-2]
        is_below = last_row['Close'] < last_row['SMA200']
        is_falling = last_row['SMA200'] < prev_row['SMA200']
        if is_below and is_falling: return False
    elif trend_opt == "Price < 200 SMA":
        if not (last_row['Close'] < last_row['SMA200']): return False
    elif "Market" in trend_opt and 'Market_Above_SMA200' in df.columns:
        is_above = last_row['Market_Above_SMA200']
        if ">" in trend_opt and not is_above: return False
        if "<" in trend_opt and is_above: return False

    # 2b. MA Consecutive Filters
    if 'ma_consec_filters' in params:
        for maf in params['ma_consec_filters']:
            length = maf['length']
            col_name = f"SMA{length}"
            if col_name not in df.columns: continue
            
            # Create boolean series
            if maf['logic'] == 'Above': mask = df['Close'] > df[col_name]
            elif maf['logic'] == 'Below': mask = df['Close'] < df[col_name]
            
            consec = maf.get('consec', 1)
            if consec > 1:
                # Check sum of last 'consec' days
                recent = mask.rolling(consec).sum()
                if recent.iloc[-1] != consec: return False
            else:
                if not mask.iloc[-1]: return False

    # 3. Breakout Mode
    bk_mode = params.get('breakout_mode', 'None')
    if bk_mode == "Close > Prev Day High":
        if not (last_row['Close'] > last_row['PrevHigh']): return False
    elif bk_mode == "Close < Prev Day Low":
        if not (last_row['Close'] < last_row['PrevLow']): return False

    # 4. Range %
    if params.get('use_range_filter', False):
        rn_val = last_row['RangePct'] * 100
        if not (rn_val >= params.get('range_min', 0) and rn_val <= params.get('range_max', 100)): return False

    # 5. Perf Rank (Dual Support)
    # A. LIST BASED
    if 'perf_filters' in params:
        for pf in params['perf_filters']:
            col = f"rank_ret_{pf['window']}d"
            consec = pf.get('consecutive', 1)
            
            if pf['logic'] == '<': cond_series = df[col] < pf['thresh']
            else: cond_series = df[col] > pf['thresh']
            
            if consec > 1:
                recent_sum = cond_series.rolling(consec).sum().iloc[-1]
                if recent_sum != consec: return False
            else:
                if not cond_series.iloc[-1]: return False
            
            if params.get('perf_first_instance', False):
                lookback = params.get('perf_lookback', 21)
                # Ensure previous 'lookback' days were NOT triggered
                # We check the window [today-lookback-1 : today-1]
                # easier way: rolling sum of condition shifted by 1
                prev_sum = cond_series.shift(1).rolling(lookback).sum().iloc[-1]
                if prev_sum > 0: return False

    # B. SINGULAR BASED
    if params.get('use_perf_rank', False):
        col = f"rank_ret_{params['perf_window']}d"
        thresh = params['perf_thresh']
        
        if params['perf_logic'] == '<': raw_series = df[col] < thresh
        else: raw_series = df[col] > thresh
        
        consec = params.get('perf_consecutive', 1)
        if consec > 1:
            recent_sum = raw_series.rolling(consec).sum().iloc[-1]
            if recent_sum != consec: return False
        else:
            if not raw_series.iloc[-1]: return False
            
        if params.get('perf_first_instance', False):
            lookback = params.get('perf_lookback', 21)
            prev_sum = raw_series.shift(1).rolling(lookback).sum().iloc[-1]
            if prev_sum > 0: return False

    # 6. Gap/Acc/Dist Filters
    if params.get('use_gap_filter', False):
        # We use the 'GapCount' column which is pre-calculated with the correct window
        gap_val = last_row.get('GapCount', 0)
        g_logic = params.get('gap_logic', '>')
        g_thresh = params.get('gap_thresh', 0)
        if g_logic == ">" and not (gap_val > g_thresh): return False
        if g_logic == "<" and not (gap_val < g_thresh): return False
        if g_logic == "=" and not (gap_val == g_thresh): return False

    if params.get('use_acc_count_filter', False):
        col = f"AccCount_{params['acc_count_window']}"
        if col in df.columns:
            val = last_row[col]
            thresh = params['acc_count_thresh']
            logic = params['acc_count_logic']
            if logic == "=" and not (val == thresh): return False
            if logic == ">" and not (val > thresh): return False
            if logic == "<" and not (val < thresh): return False

    if params.get('use_dist_count_filter', False):
        col = f"DistCount_{params['dist_count_window']}"
        if col in df.columns:
            val = last_row[col]
            thresh = params['dist_count_thresh']
            logic = params['dist_count_logic']
            if logic == "=" and not (val == thresh): return False
            if logic == ">" and not (val > thresh): return False
            if logic == "<" and not (val < thresh): return False

    # 7. MA Distance Filter
    if params.get('use_dist_filter', False):
        ma_map = {"SMA 10": "SMA10", "SMA 20": "SMA20", "SMA 50": "SMA50", "SMA 200": "SMA200"}
        ma_col = ma_map.get(params['dist_ma_type'])
        if ma_col and ma_col in df.columns:
            dist_units = (last_row['Close'] - last_row[ma_col]) / last_row['ATR']
            d_min = params.get('dist_min', 0)
            d_max = params.get('dist_max', 0)
            d_logic = params.get('dist_logic', 'Between')
            
            if d_logic == "Greater Than (>)" and not (dist_units > d_min): return False
            if d_logic == "Less Than (<)" and not (dist_units < d_max): return False
            if d_logic == "Between" and not (dist_units >= d_min and dist_units <= d_max): return False

    # 8. MA Touch Logic
    if params.get('use_ma_touch', False):
        ma_map = {"SMA 10": "SMA10", "SMA 20": "SMA20", "SMA 50": "SMA50", "SMA 200": "SMA200"}
        ma_col = ma_map.get(params['ma_touch_type'])
        if ma_col and ma_col in df.columns:
            ma_series = df[ma_col]
            direction = params.get('trade_direction', 'Long')
            
            # A. Check Slope (Rising/Falling)
            slope_days = params.get('ma_slope_days', 20)
            if direction == 'Long': 
                slope_ok = (ma_series > ma_series.shift(1)).rolling(slope_days).sum().iloc[-1] == slope_days
            else:
                slope_ok = (ma_series < ma_series.shift(1)).rolling(slope_days).sum().iloc[-1] == slope_days
            if not slope_ok: return False
            
            # B. Check Untested
            untested_days = params.get('ma_untested_days', 50)
            if direction == 'Long':
                # Lows must have been ABOVE MA for 'untested_days' before today
                was_untested = (df['Low'] > ma_series).shift(1).rolling(untested_days).min().iloc[-1] == 1.0
            else:
                was_untested = (df['High'] < ma_series).shift(1).rolling(untested_days).min().iloc[-1] == 1.0
            if not was_untested: return False
            
            # C. Check Touch Today
            if direction == 'Long':
                if not (last_row['Low'] <= last_row[ma_col]): return False
            else:
                if not (last_row['High'] >= last_row[ma_col]): return False

    # 9. VIX Filter
    if params.get('use_vix_filter', False):
        vix_val = last_row.get('VIX_Value', 0)
        if not (vix_val >= params.get('vix_min', 0) and vix_val <= params.get('vix_max', 100)): return False

    # 10. Seasonality
    if params['use_sznl']:
        val = last_row['Sznl']
        cond_series = (df['Sznl'] < params['sznl_thresh']) if params['sznl_logic'] == '<' else (df['Sznl'] > params['sznl_thresh'])
        
        if not cond_series.iloc[-1]: return False
        
        if params.get('sznl_first_instance', False):
            lookback = params.get('sznl_lookback', 21)
            prev_sum = cond_series.shift(1).rolling(lookback).sum().iloc[-1]
            if prev_sum > 0: return False

    if params.get('use_market_sznl', False):
        val = last_row['Mkt_Sznl_Ref']
        if params['market_sznl_logic'] == '<': 
            if not (val < params['market_sznl_thresh']): return False
        else: 
            if not (val > params['market_sznl_thresh']): return False

    # 11. 52w
    if params['use_52w']:
        c_52 = df['is_52w_high'] if params['52w_type'] == 'New 52w High' else df['is_52w_low']
        
        # Apply Lag
        if params.get('52w_lag', 0) > 0:
            c_52 = c_52.shift(params['52w_lag']).fillna(False)
            
        if not c_52.iloc[-1]: return False
        
        if params.get('52w_first_instance', True):
            lookback = params.get('52w_lookback', 21)
            prev_sum = c_52.shift(1).rolling(lookback).sum().iloc[-1]
            if prev_sum > 0: return False
            
    if params.get('exclude_52w_high', False):
        if last_row['is_52w_high']: return False

    # 12. Volume
    if params['use_vol']:
        if not (last_row['vol_ratio'] > params['vol_thresh']): return False

    if params.get('use_vol_rank'):
        val = last_row['vol_ratio_10d_rank']
        if params['vol_rank_logic'] == '<':
            if not (val < params['vol_rank_thresh']): return False
        else:
            if not (val > params['vol_rank_thresh']): return False
            
    return True
    
def get_signal_breakdown(df, params, sznl_map):
    last_row = df.iloc[-1]
    audit = {}
    all_passed = True
    
    def log(key, value, condition_met):
        nonlocal all_passed
        audit[key] = value
        audit[f"{key}_Pass"] = "✅" if condition_met else "❌"
        if not condition_met: all_passed = False

    # 1. Trend
    trend_opt = params.get('trend_filter', 'None')
    trend_res = True
    if trend_opt == "Price > 200 SMA": trend_res = last_row['Close'] > last_row['SMA200']
    log("Trend", f"{trend_opt}", trend_res)

    # 2. Perf Rank
    if params.get('use_perf_rank', False):
        col = f"rank_ret_{params['perf_window']}d"
        val = last_row.get(col, 0)
        cond = val < params['perf_thresh'] if params['perf_logic'] == '<' else val > params['perf_thresh']
        log(f"Perf_{params['perf_window']}d", f"{val:.1f}", cond)

    # 3. Seasonality
    if params['use_sznl']:
        val = last_row['Sznl']
        cond = val < params['sznl_thresh'] if params['sznl_logic'] == '<' else val > params['sznl_thresh']
        log("Sznl", f"{val:.1f}", cond)

    audit['Result'] = "✅ SIGNAL" if all_passed else ""
    return audit

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
                    except: continue
            time.sleep(0.25)
        except: continue

    progress_bar.empty()
    status_text.empty()
    return data_dict

def main():
    st.set_page_config(layout="wide", page_title="Production Strategy Screener")
    st.title("⚡ Daily Strategy Screener (Batch Optimized)")
    st.markdown("---")
    
    sznl_map = load_seasonal_map()
    if 'master_data_dict' not in st.session_state:
        st.session_state['master_data_dict'] = {}

    if st.button("Run All Strategies", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        
        all_required_tickers = set()
        for strat in STRATEGY_BOOK:
            all_required_tickers.update(strat['universe_tickers'])
            s = strat['settings']
            if s.get('use_market_sznl'): all_required_tickers.add(s.get('market_ticker', '^GSPC'))
            if "Market" in s.get('trend_filter', ''): all_required_tickers.add(s.get('market_ticker', 'SPY'))
            if "SPY" in s.get('trend_filter', ''): all_required_tickers.add("SPY")
            if s.get('use_vix_filter'): all_required_tickers.add('^VIX')
            
        all_required_tickers = {t.replace('.', '-') for t in all_required_tickers}

        existing_keys = set(st.session_state['master_data_dict'].keys())
        missing_tickers = list(all_required_tickers - existing_keys)
        
        if missing_tickers:
            st.info(f"Need to fetch history for {len(missing_tickers)} tickers.")
            new_data_dict = download_historical_data(missing_tickers, start_date="2000-01-01")
            st.session_state['master_data_dict'].update(new_data_dict)
            st.success(f"✅ Data initialized. Total tickers: {len(st.session_state['master_data_dict'])}")

        master_dict = st.session_state['master_data_dict']
        vix_df = master_dict.get('^VIX')
        vix_series = vix_df['Close'] if vix_df is not None and not vix_df.empty else None
        
        all_staging_signals = [] 

        for i, strat in enumerate(STRATEGY_BOOK):
            with st.expander(f"Strategy: {strat['name']}", expanded=False):
                settings = strat['settings']
                strat_mkt_ticker = settings.get('market_ticker', 'SPY')
                mkt_df = master_dict.get(strat_mkt_ticker)
                if mkt_df is None: mkt_df = master_dict.get('SPY')
                
                market_series = None
                if mkt_df is not None:
                    temp_mkt = mkt_df.copy()
                    temp_mkt['SMA200'] = temp_mkt['Close'].rolling(200).mean()
                    market_series = temp_mkt['Close'] > temp_mkt['SMA200']

                # Extract Params
                gap_win = settings.get('gap_lookback', 21)
                acc_win = settings.get('acc_count_window') if settings.get('use_acc_count_filter') else None
                dist_win = settings.get('dist_count_window') if settings.get('use_dist_count_filter') else None
                req_custom_mas = list(set([f['length'] for f in settings.get('ma_consec_filters', [])]))
                
                if settings.get('use_ma_touch'):
                    ma_type = settings.get('ma_touch_type', '')
                    if 'SMA' in ma_type and ma_type not in ["SMA 10", "SMA 20", "SMA 50", "SMA 100", "SMA 200"]:
                        try:
                            val = int(ma_type.replace("SMA", "").strip())
                            req_custom_mas.append(val)
                        except: pass

                signals = []
                for ticker in strat['universe_tickers']:
                    t_clean = ticker.replace('.', '-')
                    df = master_dict.get(t_clean)
                    if df is None or len(df) < 250: continue
                    
                    try:
                        # PASS DYNAMIC PARAMS
                        calc_df = calculate_indicators(
                            df, sznl_map, t_clean, market_series, vix_series,
                            gap_window=gap_win,
                            acc_window=acc_win,
                            dist_window=dist_win,
                            custom_sma_lengths=req_custom_mas
                        )
                        
                        if check_signal(calc_df, settings, sznl_map):
                            last_row = calc_df.iloc[-1]
                            
                            # Entry Conf Check
                            entry_conf_bps = settings.get('entry_conf_bps', 0)
                            entry_mode = settings.get('entry_type', 'Signal Close')
                            if entry_mode == 'Signal Close' and entry_conf_bps > 0:
                                threshold = last_row['Open'] * (1 + entry_conf_bps/10000.0)
                                if last_row['High'] < threshold: continue

                            atr = last_row['ATR']
                            risk = strat['execution']['risk_per_trade']
                            
                            # Specific Overrides
                            if strat['name'] == "Overbot Vol Spike":
                                vol_ratio = last_row.get('vol_ratio', 0)
                                if vol_ratio > 2.0: risk = 675 
                                elif vol_ratio > 1.5: risk = 525 
                            if strat['name'] == "Weak Close Decent Sznls":
                                sznl_val = last_row.get('Sznl', 0)
                                if sznl_val >= 65: risk = risk * 1.5 
                                elif sznl_val >= 50: risk = risk * 1.0 
                                elif sznl_val >= 33: risk = risk * 0.66
                                    
                            entry = last_row['Close']
                            direction = settings.get('trade_direction', 'Long')
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
                            exit_date = (last_row.name + (strat['execution']['hold_days'] * TRADING_DAY)).date()
                            
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
                    except: continue
                
                if signals:
                    all_staging_signals.extend(signals)
                    st.success(f"✅ Found {len(signals)} Signals")
                    sig_df = pd.DataFrame(signals)
                    st.dataframe(sig_df.style.format({"Entry": "${:.2f}", "Stop": "${:.2f}", "Target": "${:.2f}", "ATR": "{:.2f}"}), use_container_width=True)
                else:
                    st.caption("No signals found.")

        if all_staging_signals:
            st.divider()
            st.subheader("🚀 Order Staging")
            
            save_moc_orders(all_staging_signals, STRATEGY_BOOK, sheet_name='moc_orders')
            
            if is_after_market_close():
                st.success("🌑 **Market Closed:** Auto-staging full batch.")
                save_staging_orders(all_staging_signals, STRATEGY_BOOK, sheet_name='Order_Staging')
                save_signals_to_gsheet(pd.DataFrame(all_staging_signals), sheet_name='Trade_Signals_Log')
            else:
                st.info("🕒 Market Open. Standard staging disabled until 4:00 PM EST.")

    # -------------------------------------------------------------------------
    # INSPECTOR UI
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🕵️‍♀️ Historical Strategy Inspector (Last 21 Days)")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        insp_ticker = st.text_input("Inspector Ticker:", value="AMD").upper().strip()
    with c2:
        strat_map_ui = {s['name']: s for s in STRATEGY_BOOK}
        selected_strat_name = st.selectbox("Select Strategy to Audit:", list(strat_map_ui.keys()))
        
    if st.button("Run Historical Audit"):
        target_strat = strat_map_ui[selected_strat_name]
        
        if insp_ticker not in st.session_state.get('master_data_dict', {}):
             tmp_dict = download_historical_data([insp_ticker, "SPY", "^GSPC", "^VIX"])
             st.session_state.setdefault('master_data_dict', {}).update(tmp_dict)

        df_insp = st.session_state['master_data_dict'].get(insp_ticker)
        
        settings = target_strat['settings']
        mkt_ticker = settings.get('market_ticker', 'SPY')
        df_mkt = st.session_state['master_data_dict'].get(mkt_ticker)
        if df_mkt is None: df_mkt = st.session_state['master_data_dict'].get('SPY')
        
        vix_df = st.session_state['master_data_dict'].get('^VIX')
        vix_series = vix_df['Close'] if vix_df is not None else None

        if df_insp is not None and len(df_insp) > 50:
            mkt_series = None
            if df_mkt is not None:
                temp_mkt = df_mkt.copy()
                temp_mkt['SMA200'] = temp_mkt['Close'].rolling(200).mean()
                mkt_series = temp_mkt['Close'] > temp_mkt['SMA200']

            gap_win = settings.get('gap_lookback', 21)
            acc_win = settings.get('acc_count_window') if settings.get('use_acc_count_filter') else None
            dist_win = settings.get('dist_count_window') if settings.get('use_dist_count_filter') else None
            req_custom_mas = list(set([f['length'] for f in settings.get('ma_consec_filters', [])]))
            
            if settings.get('use_ma_touch'):
                ma_type = settings.get('ma_touch_type', '')
                if 'SMA' in ma_type and ma_type not in ["SMA 10", "SMA 20", "SMA 50", "SMA 100", "SMA 200"]:
                    try: val = int(ma_type.replace("SMA", "").strip()); req_custom_mas.append(val)
                    except: pass

            calc_df = calculate_indicators(
                df_insp, sznl_map, insp_ticker, mkt_series, vix_series,
                gap_window=gap_win, acc_window=acc_win, dist_window=dist_win, custom_sma_lengths=req_custom_mas
            )
            
            audit_rows = []
            days_to_audit = 21 
            
            if len(calc_df) > days_to_audit:
                subset_indices = range(len(calc_df) - days_to_audit, len(calc_df))
                for idx in subset_indices:
                    slice_df = calc_df.iloc[:idx+1] 
                    row_date = slice_df.index[-1].date()
                    breakdown = get_signal_breakdown(slice_df, settings, sznl_map)
                    breakdown['Date'] = row_date
                    audit_rows.append(breakdown)
            
            if audit_rows:
                audit_df = pd.DataFrame(audit_rows)
                cols = ['Date', 'Result'] + [c for c in audit_df.columns if c not in ['Date', 'Result']]
                audit_df = audit_df[cols].sort_values(by='Date', ascending=False)
                st.write(f"**Audit Results for {insp_ticker}**")
                st.dataframe(audit_df.style.apply(lambda x: ['background-color: #e6fffa' if x.name == 'Result' and v == "✅ SIGNAL" else '' for v in x], axis=0), use_container_width=True)
            else:
                st.warning("Not enough data to audit.")
        else:
            st.error("Ticker data not found.")

if __name__ == "__main__":
    main()
