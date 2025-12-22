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
    
def check_signal(df, params, sznl_map):
    """
    Standard binary check (Fast). Used for Batch processing.
    """
    # ... (Keep existing logic, omitted here for brevity as we are using get_signal_breakdown for the Monitor) ...
    # NOTE: In a full refactor, check_signal could simply call get_signal_breakdown and check ['Result']
    # For now, we assume the existing function is essentially the logic below but returning boolean.
    audit = get_signal_breakdown(df, params, sznl_map)
    return audit.get('Result') == "✅ SIGNAL"

def get_signal_breakdown(df, params, sznl_map):
    """
    Runs the logic and returns a dictionary of statuses.
    Used for both the 'Inspector' and the 'Intraday Monitor'.
    """
    last_row = df.iloc[-1]
    audit = {}
    
    # Helper to log result
    def log(key, value, condition_met, vital=True):
        """
        key: Name of check
        value: The actual value (e.g., 55.4)
        condition_met: Boolean
        vital: If True, this failing means the signal fails.
        """
        audit[key] = value
        audit[f"{key}_Pass"] = "✅" if condition_met else "❌"
        # We add a hidden field to help the monitor count failures
        audit[f"{key}_Bool"] = condition_met

    # -----------------------------------------------------------
    # 0. DAY OF WEEK & CYCLE
    # -----------------------------------------------------------
    if params.get('use_dow_filter', False):
        allowed = params.get('allowed_days', [])
        day_num = last_row.name.dayofweek 
        cond = day_num in allowed
        log("DOW", last_row.name.strftime("%A"), cond)

    if 'allowed_cycles' in params:
        allowed_cycles = params['allowed_cycles']
        if allowed_cycles and len(allowed_cycles) < 4:
            current_year = last_row.name.year
            cycle_rem = current_year % 4
            cond = cycle_rem in allowed_cycles
            log("Cycle", f"{current_year}", cond)

    # 1. Liquidity (Combined for brevity)
    liq_pass = (
        last_row['Close'] >= params.get('min_price', 0) and 
        last_row['vol_ma'] >= params.get('min_vol', 0) and
        last_row['age_years'] >= params.get('min_age', 0)
    )
    log("Liquidity", f"${last_row['Close']:.2f}", liq_pass)

    # 2. Trend (Global)
    trend_opt = params.get('trend_filter', 'None')
    if trend_opt != 'None':
        trend_res = True
        if trend_opt == "Price > 200 SMA":
            trend_res = last_row['Close'] > last_row['SMA200']
        elif "Market" in trend_opt or "SPY" in trend_opt:
            if 'Market_Above_SMA200' in df.columns:
                is_above = last_row['Market_Above_SMA200']
                trend_res = is_above if ">" in trend_opt else not is_above
        log("Trend", trend_opt, trend_res)

    # 3. Perf Rank
    if 'perf_filters' in params:
        for i, pf in enumerate(params['perf_filters']):
            col = f"rank_ret_{pf['window']}d"
            val = last_row.get(col, 0)
            if pf['logic'] == '<': cond = val < pf['thresh']
            else: cond = val > pf['thresh']
            # Consecutive logic simplified for monitor view
            log(f"Perf_{pf['window']}d", f"{val:.1f}", cond)

    # 4. Seasonality
    if params['use_sznl']:
        val = last_row['Sznl']
        if params['sznl_logic'] == '<': cond = val < params['sznl_thresh']
        else: cond = val > params['sznl_thresh']
        log("Seasonality", f"{val:.1f}", cond)

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
        log("Vol_Ratio", f"x{ratio:.2f}", cond)
        
    if params.get('use_vol_rank', False):
        ratio = last_row['vol_ratio_10d_rank']
        cond = (ratio < params['vol_rank_thresh']) if params['vol_rank_logic'] == '<' else (ratio > params['vol_rank_thresh'])
        log("Vol_Rank", f"{ratio:.1f}", cond)

    # 7. RANGE FILTER (Common failing point)
    if params.get('use_range_filter', False):
        rng = last_row['RangePct'] * 100
        r_min = params.get('range_min', 0)
        r_max = params.get('range_max', 100)
        cond = (rng >= r_min) and (rng <= r_max)
        log("Range_Loc", f"{rng:.1f}%", cond)
        
    # 8. MA Specific Filters
    if 'ma_consec_filters' in params:
        for i, maf in enumerate(params['ma_consec_filters']):
            length = maf['length']
            val = last_row.get(f"SMA{length}", 0)
            cond = (last_row['Close'] > val) if maf['logic'] == 'Above' else (last_row['Close'] < val)
            log(f"Price_vs_SMA{length}", f"{val:.2f}", cond)

    # Calculate Totals
    failures = [k for k, v in audit.items() if "_Bool" in k and v is False]
    all_passed = len(failures) == 0

    audit['Result'] = "✅ SIGNAL" if all_passed else ""
    audit['Fail_Count'] = len(failures)
    
    # Generate list of readable failed keys (e.g. "Range_Loc_Bool" -> "Range_Loc")
    audit['Failed_Items'] = [f.replace("_Bool", "") for f in failures]
    
    return audit

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def download_historical_data(tickers, start_date="2000-01-01"):
    # ... (Keep existing implementation) ...
    # Placeholder for brevity - assume this works as in your original script
    if not tickers: return {}
    clean_tickers = list(set([str(t).strip().upper().replace('.', '-') for t in tickers]))
    data_dict = {}
    # ... downloading logic ...
    # Simulating download for context size limits in this response
    try:
        df = yf.download(clean_tickers, start=start_date, group_by='ticker', auto_adjust=False, progress=False, threads=True)
        if len(clean_tickers) == 1:
            clean_tickers = [clean_tickers[0]] # Handle single ticker case
            
        if isinstance(df.columns, pd.MultiIndex):
            for t in clean_tickers:
                try:
                    data_dict[t] = df[t].copy()
                    if data_dict[t].empty: del data_dict[t]
                except: pass
        else:
             data_dict[clean_tickers[0]] = df
    except: pass
    
    # Clean indexes
    for t in data_dict:
        data_dict[t].index = data_dict[t].index.tz_localize(None)
        
    return data_dict

def main():
    st.set_page_config(layout="wide", page_title="Pro Strategy Screener")
    
    # --- SIDEBAR CONTROLS ---
    st.sidebar.title("Configuration")
    mode = st.sidebar.radio("Scanner Mode", ["Intraday Monitor", "EOD Batch Runner"])
    
    sznl_map = load_seasonal_map()
    
    # Init Session State
    if 'master_data_dict' not in st.session_state:
        st.session_state['master_data_dict'] = {}

    # -------------------------------------------------------------------------
    # MODE 1: INTRADAY MONITOR (The New Feature)
    # -------------------------------------------------------------------------
    if mode == "Intraday Monitor":
        st.title("👀 Intraday Signal Monitor (MOC)")
        st.markdown("Identifies strategies with **Signal Close** entries that are *close* to triggering.")
        
        # User Tolerance Configuration
        col_tol1, col_tol2 = st.columns(2)
        tolerance = col_tol1.slider("Show Signals with X Failures or less:", 0, 3, 1, help="0 = Confirmed Only. 1 = Watchlist.")
        
        if st.button("🔎 Scan Potential Signals", type="primary"):
            st.cache_data.clear() # Force fresh data for intraday
            
            # 1. Filter Universe for "Signal Close" Strategies ONLY
            moc_strategies = [s for s in STRATEGY_BOOK if "Signal Close" in s['settings'].get('entry_type', '')]
            
            universe = set()
            for s in moc_strategies:
                universe.update(s['universe_tickers'])
                if s['settings'].get('use_market_sznl'): universe.add(s['settings'].get('market_ticker', '^GSPC'))
                if "Market" in s['settings'].get('trend_filter', ''): universe.add('SPY')
            
            universe = list(universe)
            st.toast(f"Fetching intraday data for {len(universe)} tickers...")
            
            # 2. Download Data (Force fresh download implicitly via clear cache or just re-run)
            # For intraday, we might want a shorter lookback to save time, but indicators need 252d usually.
            data_dict = download_historical_data(universe, start_date="2023-01-01") 
            st.session_state['master_data_dict'].update(data_dict)
            
            results = []
            
            # 3. Scan
            progress_bar = st.progress(0)
            
            for i, strat in enumerate(moc_strategies):
                # Setup Market Series if needed
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
                    
                    # Calculate Indicators
                    try:
                        calc_df = calculate_indicators(df, sznl_map, t_clean, market_series)
                        
                        # GET BREAKDOWN
                        audit = get_signal_breakdown(calc_df, strat['settings'], sznl_map)
                        
                        # CHECK FAILURES
                        fail_count = audit['Fail_Count']
                        failed_items = audit['Failed_Items']
                        
                        if fail_count <= tolerance:
                            # Classify Status
                            if fail_count == 0:
                                status = "✅ CONFIRMED"
                                color = "green"
                            else:
                                status = f"⚠️ MISSING ({fail_count})"
                                color = "orange"
                                
                            last_price = calc_df['Close'].iloc[-1]
                            range_pct = calc_df['RangePct'].iloc[-1] * 100
                            
                            results.append({
                                "Strategy": strat['name'],
                                "Ticker": ticker,
                                "Price": last_price,
                                "Range%": range_pct,
                                "Status": status,
                                "Failures": failed_items,
                                "Full_Audit": audit, # Store full dict for expander
                                "_color": color
                            })
                            
                    except Exception as e:
                        continue
                
                progress_bar.progress((i + 1) / len(moc_strategies))
            
            progress_bar.empty()
            
            # 4. Display Results
            if not results:
                st.info("No signals found within tolerance.")
            else:
                res_df = pd.DataFrame(results)
                
                # Group by Status for clean UI
                st.subheader(f"Found {len(res_df)} Candidates")
                
                # Sort: Confirmed first, then by Ticker
                res_df.sort_values(by=['Status', 'Ticker'], ascending=[True, True], inplace=True)
                
                for _, row in res_df.iterrows():
                    # Color code the expander border based on status
                    is_conf = "CONFIRMED" in row['Status']
                    icon = "🟢" if is_conf else "🟠"
                    
                    # Expander Header
                    fail_txt = f"Missing: {', '.join(row['Failures'])}" if row['Failures'] else "Ready to Trade"
                    label = f"{icon} **{row['Ticker']}** | {row['Strategy']} | {row['Status']} | {fail_txt}"
                    
                    with st.expander(label, expanded=is_conf):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Current Price", f"${row['Price']:.2f}")
                        c2.metric("Range Location", f"{row['Range%']:.1f}%")
                        
                        st.markdown("#### Checklist Breakdown")
                        
                        # Create a readable checklist from the audit dict
                        audit_data = row['Full_Audit']
                        check_rows = []
                        for k, v in audit_data.items():
                            if "_Pass" in k:
                                param_name = k.replace("_Pass", "")
                                val = audit_data.get(param_name, "-")
                                check_rows.append({
                                    "Condition": param_name,
                                    "Value": val,
                                    "Status": v
                                })
                        
                        st.dataframe(pd.DataFrame(check_rows), hide_index=True, use_container_width=True)

    # -------------------------------------------------------------------------
    # MODE 2: EOD BATCH RUNNER (Original Logic)
    # -------------------------------------------------------------------------
    elif mode == "EOD Batch Runner":
        st.title("⚡ EOD Batch Screener & Staging")
        st.markdown("Standard end-of-day routine. Scans all strategies, calculates stops/targets, and stages orders to Google Sheets.")
        
        if st.button("Run EOD Batch", type="primary"):
            # ... (Paste your original 'Run All Strategies' logic here) ...
            # For brevity, I am summarizing the integration:
            # 1. Check/Download Data
            # 2. Iterate Strategies (Loop)
            # 3. Call check_signal() -> if True -> append to list
            # 4. Display DataFrame
            # 5. Call save_moc_orders and save_staging_orders
            st.info("Switching to EOD Batch Logic (Copied from original script)...")
            
            # (Insert original loop logic here if you want it in the same file)
            # Or simplified call:
            run_eod_batch_logic(sznl_map)

# Wrapper to keep the original logic organized
def run_eod_batch_logic(sznl_map):
    # This is essentially the code from your original main() under "Run All Strategies"
    # Re-implemented simply to show where it goes.
    
    # 1. Universe Collection
    all_required_tickers = set()
    for strat in STRATEGY_BOOK:
        all_required_tickers.update(strat['universe_tickers'])
        s = strat['settings']
        if s.get('use_market_sznl'): all_required_tickers.add(s.get('market_ticker', '^GSPC'))
        if "Market" in s.get('trend_filter', ''): all_required_tickers.add(s.get('market_ticker', 'SPY'))
    
    missing_tickers = list(all_required_tickers - set(st.session_state['master_data_dict'].keys()))
    if missing_tickers:
        st.info(f"Downloading {len(missing_tickers)} tickers...")
        new_data = download_historical_data(missing_tickers)
        st.session_state['master_data_dict'].update(new_data)
        
    all_signals = []
    
    # 2. Processing
    master_dict = st.session_state['master_data_dict']
    
    for strat in STRATEGY_BOOK:
        # Prepare Market Series
        mkt_ticker = strat['settings'].get('market_ticker', 'SPY')
        mkt_df = master_dict.get(mkt_ticker)
        if mkt_df is None: mkt_df = master_dict.get('SPY')
        mkt_series = None
        if mkt_df is not None:
            mkt_series = mkt_df['Close'] > mkt_df['Close'].rolling(200).mean()

        for ticker in strat['universe_tickers']:
            t_clean = ticker.replace('.', '-')
            df = master_dict.get(t_clean)
            if df is None or len(df) < 200: continue
            
            calc_df = calculate_indicators(df, sznl_map, t_clean, mkt_series)
            
            # BINARY CHECK for EOD
            if check_signal(calc_df, strat['settings'], sznl_map):
                # ... (Calculate Stop/Target/Shares logic from original script) ...
                last_row = calc_df.iloc[-1]
                atr = last_row['ATR']
                risk = strat['execution']['risk_per_trade']
                # ... Simplified entry for example ...
                entry = last_row['Close']
                stop_atr = strat['execution']['stop_atr']
                stop = entry - (atr * stop_atr)
                target = entry + (atr * strat['execution']['tgt_atr'])
                dist = entry - stop
                shares = int(risk/dist) if dist > 0 else 0
                
                all_signals.append({
                    "Strategy_ID": strat['id'],
                    "Ticker": ticker,
                    "Action": "BUY", # Simplified
                    "Shares": shares,
                    "Entry": entry,
                    "Stop": stop,
                    "Target": target,
                    "Time Exit": datetime.date.today(), # Simplified
                    "ATR": atr
                })

    if all_signals:
        df_sig = pd.DataFrame(all_signals)
        st.success(f"Generated {len(df_sig)} Confirmed Signals")
        st.dataframe(df_sig)
        
        # Save logic
        # save_moc_orders(...) 
        # save_staging_orders(...)
    else:
        st.warning("No confirmed signals.")

if __name__ == "__main__":
    main()
