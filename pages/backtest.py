import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import datetime
import random
import time
import uuid

# -----------------------------------------------------------------------------
# CONFIG / CONSTANTS
# -----------------------------------------------------------------------------
MARKET_TICKER = "^GSPC" 
VIX_TICKER = "^VIX"

SECTOR_ETFS = [
    "IBB", "IHI", "ITA", "ITB", "IYR", "KRE", "OIH", "SMH", "VNQ",
    "XBI", "XHB", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP",
    "XLU", "XLV", "XLY", "XME", "XOP", "XRT", 
]
SPX=['^GSPC','SPY']
INDEX_ETFS = ["SPY", "QQQ", "IWM", "DIA", "SMH"]

INTERNATIONAL_ETFS = [
    "EWZ", "EWC", "ECH", "ECOL", "EWW", "ARGT",
    "EWQ", "EWG", "EWI", "EWU", "EWP", "EWK", "EWO", "EWN", "EWD", "EWL",
    "EWJ", "EWH", "MCHI", "INDA", "EWY", "EWT", "EWA", "EWS", "EWM", "THD", "EIDO", "VNM", "EPHE",
    "EZA", "TUR", "EGPT"
]

CSV_PATH = "seasonal_ranks.csv" 

# -----------------------------------------------------------------------------
# DATA UTILS
# -----------------------------------------------------------------------------

@st.cache_resource 
def load_seasonal_map():
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        return {}

    if df.empty: return {}

    df["Date"] = pd.to_datetime(df["Date"], errors='coerce').dt.normalize().dt.tz_localize(None)
    df = df.dropna(subset=["Date"])
    
    output_map = {}
    for ticker, group in df.groupby("ticker"):
        output_map[str(ticker).upper()] = pd.Series(
            group.seasonal_rank.values, index=group.Date
        ).to_dict()
    return output_map

def get_sznl_val_series(ticker, dates, sznl_map):
    ticker = ticker.upper()
    t_map = sznl_map.get(ticker, {})
    
    if not t_map and ticker == "^GSPC":
        t_map = sznl_map.get("SPY", {})

    if not t_map:
        return pd.Series(50.0, index=dates)
        
    return dates.map(t_map).fillna(50.0)

def clean_ticker_df(df):
    if df.empty: return df
    
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(1) or 'Adj Close' in df.columns.get_level_values(1):
             df.columns = df.columns.get_level_values(1)
        else:
             df.columns = df.columns.get_level_values(0)
    
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    
    if 'Close' not in df.columns and 'Adj close' in df.columns:
        df.rename(columns={'Adj close': 'Close'}, inplace=True)
        
    if 'Close' not in df.columns:
        return pd.DataFrame() 
        
    df = df.dropna(subset=['Close'])
    return df

@st.cache_data(show_spinner=True)
def download_universe_data(tickers, fetch_start_date):
    if not tickers: return {} 
    
    clean_tickers = [str(t).strip().upper() for t in tickers if str(t).strip() != '']
    if not clean_tickers: return {}

    if isinstance(fetch_start_date, datetime.date):
        start_str = fetch_start_date.strftime('%Y-%m-%d')
    else:
        start_str = fetch_start_date 

    data_dict = {}
    CHUNK_SIZE = 50 
    total_tickers = len(clean_tickers)
    
    download_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, total_tickers, CHUNK_SIZE):
        chunk = clean_tickers[i:i + CHUNK_SIZE]
        status_text.text(f"Downloading batch {i} to {min(i+CHUNK_SIZE, total_tickers)} of {total_tickers}...")
        download_bar.progress(min((i + CHUNK_SIZE) / total_tickers, 1.0))
        
        try:
            df = yf.download(chunk, start=start_str, group_by='ticker', auto_adjust=True, progress=False, threads=True)
            
            if df.empty: continue

            if len(chunk) == 1:
                t = chunk[0]
                t_df = df
                t_df = clean_ticker_df(t_df)
                if not t_df.empty:
                    t_df.index = t_df.index.normalize().tz_localize(None)
                    data_dict[t] = t_df
            else:
                for t in chunk:
                    try:
                        if t in df.columns.levels[0]:
                            t_df = df[t].copy()
                            t_df = clean_ticker_df(t_df)
                            if not t_df.empty:
                                t_df.index = t_df.index.normalize().tz_localize(None)
                                data_dict[t] = t_df
                    except: continue
            
            time.sleep(0.1)

        except Exception as e:
            err_msg = str(e).lower()
            if "rate limited" in err_msg or "too many requests" in err_msg:
                st.warning(f"⚠️ Rate limit hit. Stopping download.")
                break
            continue
    
    download_bar.empty()
    status_text.empty()
    return data_dict

def get_cycle_year(year):
    rem = year % 4
    if rem == 0: return "4. Election Year"
    if rem == 1: return "1. Post-Election"
    if rem == 2: return "2. Midterm Year"
    if rem == 3: return "3. Pre-Election"
    return "Unknown"

def get_age_bucket(years):
    if years < 3: return "< 3 Years"
    if years < 5: return "3-5 Years"
    if years < 10: return "5-10 Years"
    if years < 20: return "10-20 Years"
    return "> 20 Years"

# -----------------------------------------------------------------------------
# ANALYSIS ENGINE
# -----------------------------------------------------------------------------

def calculate_indicators(df, sznl_map, ticker, market_series=None, vix_series=None, gap_window=21, custom_sma_lengths=None):
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    
    # Standard SMAs
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA100'] = df['Close'].rolling(100).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    # Dynamic SMAs
    if custom_sma_lengths:
        for length in custom_sma_lengths:
            col_name = f"SMA{length}"
            if col_name not in df.columns:
                df[col_name] = df['Close'].rolling(length).mean()

    # EMAs
    df['EMA8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA11'] = df['Close'].ewm(span=11, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # Perf Ranks
    for window in [5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window)
        df[f'rank_ret_{window}d'] = df[f'ret_{window}d'].expanding(min_periods=252).rank(pct=True) * 100.0
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
    
    # Seasonality
    df['Sznl'] = get_sznl_val_series(ticker, df.index, sznl_map)
    
    # 52w High/Low
    rolling_high = df['High'].shift(1).rolling(252).max()
    rolling_low = df['Low'].shift(1).rolling(252).min()
    df['is_52w_high'] = df['High'] > rolling_high
    df['is_52w_low'] = df['Low'] < rolling_low
    
    # Volume & Vol Ratio
    vol_ma = df['Volume'].rolling(63).mean()
    df['vol_ma'] = vol_ma
    df['vol_ratio'] = df['Volume'] / vol_ma
    
    # 10d Relative Volume Percentile
    vol_ma_10 = df['Volume'].rolling(10).mean()
    df['vol_ratio_10d'] = vol_ma_10 / vol_ma
    df['vol_ratio_10d_rank'] = df['vol_ratio_10d'].expanding(min_periods=252).rank(pct=True) * 100.0
    
    # --- ACCUMULATION / DISTRIBUTION FLAGS ---
    vol_gt_prev = df['Volume'] > df['Volume'].shift(1)
    vol_gt_ma = df['Volume'] > df['vol_ma']
    is_green = df['Close'] > df['Open']
    is_red = df['Close'] < df['Open']

    df['is_acc_day'] = (is_green & vol_gt_prev & vol_gt_ma).astype(int)
    df['is_dist_day'] = (is_red & vol_gt_prev & vol_gt_ma).astype(int)
    
    # Age
    if not df.empty:
        start_ts = df.index[0]
        df['age_years'] = (df.index - start_ts).days / 365.25
    else:
        df['age_years'] = 0.0
        
    # Market & VIX Regime
    if market_series is not None:
        df['Market_Above_SMA200'] = market_series.reindex(df.index, method='ffill').fillna(False)
    if vix_series is not None:
        df['VIX_Value'] = vix_series.reindex(df.index, method='ffill').fillna(0)

    # Candle Range % Location
    denom = (df['High'] - df['Low'])
    df['RangePct'] = np.where(denom == 0, 0.5, (df['Close'] - df['Low']) / denom)

    # Day of Week
    df['DayOfWeekVal'] = df.index.dayofweek
    
    # Open Gap Count
    is_open_gap = (df['Low'] > df['High'].shift(1)).astype(int)
    df['GapCount'] = is_open_gap.rolling(gap_window).sum()

    # Pivot Point Calculation
    piv_len = 20 
    roll_max = df['High'].rolling(window=piv_len*2+1, center=True).max()
    df['is_pivot_high'] = (df['High'] == roll_max)
    roll_min = df['Low'].rolling(window=piv_len*2+1, center=True).min()
    df['is_pivot_low'] = (df['Low'] == roll_min)

    df['LastPivotHigh'] = np.where(df['is_pivot_high'], df['High'], np.nan)
    df['LastPivotHigh'] = df['LastPivotHigh'].shift(piv_len).ffill()
    df['LastPivotLow'] = np.where(df['is_pivot_low'], df['Low'], np.nan)
    df['LastPivotLow'] = df['LastPivotLow'].shift(piv_len).ffill()

    return df

def run_engine(universe_dict, params, sznl_map, market_series=None, vix_series=None):
    all_potential_trades = []
    
    total = len(universe_dict)
    bt_start_ts = pd.to_datetime(params['backtest_start_date'])
    direction = params.get('trade_direction', 'Long')
    max_one_pos_per_ticker = params.get('max_one_pos', True)
    allow_same_day_reentry = params.get('allow_same_day_reentry', False)
    slippage_bps = params.get('slippage_bps', 5)
    
    # Entry Logic flags
    entry_mode = params['entry_type']
    is_pullback = "Pullback" in entry_mode
    is_gap_up = "Gap Up Only" in entry_mode
    is_overnight = "Overnight" in entry_mode
    is_intraday = "Intraday" in entry_mode
    
    pullback_col = "SMA10" if "10 SMA" in entry_mode else "EMA21" if "21 EMA" in entry_mode else None
    pullback_use_level = "Level" in entry_mode
    
    gap_window = params.get('gap_lookback', 21)
    time_exit_only = params.get('time_exit_only', False)
    total_signals_generated = 0

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    req_custom_mas = list(set([f['length'] for f in params.get('ma_consec_filters', [])]))

    for i, (ticker, df_raw) in enumerate(universe_dict.items()):
        status_text.text(f"Scanning signals for {ticker}...")
        progress_bar.progress((i+1)/total)
        
        if ticker == MARKET_TICKER and MARKET_TICKER not in params.get('universe_tickers', []): continue
        if ticker == VIX_TICKER: continue
        if len(df_raw) < 100: continue
        
        ticker_last_exit = pd.Timestamp.min
        
        try:
            df = calculate_indicators(df_raw, sznl_map, ticker, market_series, vix_series, gap_window=gap_window, custom_sma_lengths=req_custom_mas)
            df = df[df.index >= bt_start_ts]
            if df.empty: continue
            
            # --- SIGNAL CALCULATION (ALL FILTERS RESTORED) ---
            conditions = []
            
            # Trend Filter
            trend_opt = params.get('trend_filter', 'None')
            if trend_opt == "Price > 200 SMA": conditions.append(df['Close'] > df['SMA200'])
            elif trend_opt == "Price > Rising 200 SMA": conditions.append((df['Close'] > df['SMA200']) & (df['SMA200'] > df['SMA200'].shift(1)))
            elif trend_opt == "Not Below Declining 200 SMA": conditions.append(~((df['Close'] < df['SMA200']) & (df['SMA200'] < df['SMA200'].shift(1))))
            elif trend_opt == "Price < 200 SMA": conditions.append(df['Close'] < df['SMA200'])
            elif "Market" in trend_opt and 'Market_Above_SMA200' in df.columns:
                if trend_opt == "Market > 200 SMA": conditions.append(df['Market_Above_SMA200'])
                elif trend_opt == "Market < 200 SMA": conditions.append(~df['Market_Above_SMA200']) 

            # Liquidity & Age
            conditions.append((df['Close'] >= params['min_price']) & (df['vol_ma'] >= params['min_vol']) & 
                              (df['age_years'] >= params['min_age']) & (df['age_years'] <= params['max_age']) & 
                              (df['ATR_Pct'] >= params['min_atr_pct']) & (df['ATR_Pct'] <= params['max_atr_pct']))

            if params.get('require_close_gt_open', False): conditions.append(df['Close'] > df['Open'])
            
            bk_mode = params.get('breakout_mode', 'None')
            if bk_mode == "Close > Prev Day High": conditions.append(df['Close'] > df['High'].shift(1))
            elif bk_mode == "Close < Prev Day Low": conditions.append(df['Close'] < df['Low'].shift(1))

            if params.get('use_range_filter', False):
                conditions.append((df['RangePct'] * 100 >= params['range_min']) & (df['RangePct'] * 100 <= params['range_max']))
            
            if params.get('use_dow_filter', False): conditions.append(df['DayOfWeekVal'].isin(params['allowed_days']))

            # Gaps, Acc/Dist
            if params.get('use_gap_filter', False):
                if params['gap_logic'] == ">": conditions.append(df['GapCount'] > params['gap_thresh'])
                elif params['gap_logic'] == "<": conditions.append(df['GapCount'] < params['gap_thresh'])

            if params.get('use_acc_count_filter', False):
                r_acc = df['is_acc_day'].rolling(params['acc_count_window']).sum()
                if params['acc_count_logic'] == ">": conditions.append(r_acc > params['acc_count_thresh'])
                elif params['acc_count_logic'] == "<": conditions.append(r_acc < params['acc_count_thresh'])

            # Perf & MA Consecutive
            for pf in params.get('perf_filters', []):
                col = f"rank_ret_{pf['window']}d"
                c_f = (df[col] < pf['thresh']) if pf['logic'] == '<' else (df[col] > pf['thresh'])
                if pf['consecutive'] > 1: c_f = c_f.rolling(pf['consecutive']).sum() == pf['consecutive']
                conditions.append(c_f)

            for f in params.get('ma_consec_filters', []):
                col = f"SMA{f['length']}"
                mask = (df['Close'] > df[col]) if f['logic'] == 'Above' else (df['Close'] < df[col])
                if f['consec'] > 1: mask = mask.rolling(f['consec']).sum() == f['consec']
                conditions.append(mask)

            # VIX Regime
            if params.get('use_vix_filter', False):
                conditions.append((df['VIX_Value'] >= params['vix_min']) & (df['VIX_Value'] <= params['vix_max']))

            # Seasonality & 52w
            if params['use_sznl']:
                c_s = (df['Sznl'] < params['sznl_thresh']) if params['sznl_logic'] == '<' else (df['Sznl'] > params['sznl_thresh'])
                conditions.append(c_s)

            if params['use_52w']:
                c_52 = df['is_52w_high'] if params['52w_type'] == 'New 52w High' else df['is_52w_low']
                if params.get('52w_lag', 0) > 0: c_52 = c_52.shift(params['52w_lag']).fillna(False)
                conditions.append(c_52)

            if not conditions: continue
            final_signal = conditions[0]
            for c in conditions[1:]: final_signal = final_signal & c
            
            signal_dates = df.index[final_signal]
            total_signals_generated += len(signal_dates)

            # --- EXECUTION LOOP (NEW ENTRIES & FIXED HOLD LOGIC) ---
            for signal_date in signal_dates:
                if max_one_pos_per_ticker and signal_date <= ticker_last_exit: continue
                
                sig_idx = df.index.get_loc(signal_date)
                if sig_idx + 1 >= len(df): continue

                found_entry = False
                actual_entry_idx = -1
                actual_entry_price = 0.0

                sig_close = df['Close'].iloc[sig_idx]
                sig_atr = df['ATR'].iloc[sig_idx]
                t_plus_1_close = df['Close'].iloc[sig_idx + 1]

                # ENTRY BRANCHES
                if entry_mode == "T+1 Close < Signal Close":
                    if t_plus_1_close < sig_close:
                        found_entry, actual_entry_idx, actual_entry_price = True, sig_idx + 1, t_plus_1_close
                elif entry_mode == "T+1 Close > Signal Close":
                    if t_plus_1_close > sig_close:
                        found_entry, actual_entry_idx, actual_entry_price = True, sig_idx + 1, t_plus_1_close
                elif entry_mode == "T+1 Close < (Signal Close - 0.5 ATR)":
                    if t_plus_1_close < (sig_close - 0.5 * sig_atr):
                        found_entry, actual_entry_idx, actual_entry_price = True, sig_idx + 1, t_plus_1_close
                elif entry_mode == "T+1 Close > (Signal Close + 0.5 ATR)":
                    if t_plus_1_close > (sig_close + 0.5 * sig_atr):
                        found_entry, actual_entry_idx, actual_entry_price = True, sig_idx + 1, t_plus_1_close
                elif is_overnight:
                    found_entry, actual_entry_idx, actual_entry_price = True, sig_idx, sig_close
                elif is_intraday:
                    found_entry, actual_entry_idx, actual_entry_price = True, sig_idx + 1, df['Open'].iloc[sig_idx + 1]
                elif is_pullback and pullback_col:
                    for wait_i in range(1, params['holding_days'] + 1):
                        curr_idx = sig_idx + wait_i
                        if curr_idx >= len(df): break
                        if df.iloc[curr_idx]['Low'] <= df.iloc[curr_idx][pullback_col]:
                            found_entry, actual_entry_idx = True, curr_idx
                            actual_entry_price = df.iloc[curr_idx][pullback_col] if pullback_use_level else df.iloc[curr_idx]['Close']
                            break
                elif is_gap_up:
                    if df['Open'].iloc[sig_idx + 1] > df['High'].iloc[sig_idx]:
                        found_entry, actual_entry_idx, actual_entry_price = True, sig_idx + 1, df['Open'].iloc[sig_idx + 1]
                else: 
                    found_entry = True
                    actual_entry_idx = sig_idx if entry_mode == 'Signal Close' else sig_idx + 1
                    actual_entry_price = df['Close'].iloc[actual_entry_idx] if 'Close' in entry_mode else df['Open'].iloc[actual_entry_idx]

                if not found_entry:
                    all_potential_trades.append({
                        "Ticker": ticker, "SignalDate": signal_date, "EntryDate": df.index[sig_idx + 1],
                        "Direction": direction, "Entry": 0, "Exit": 0, "ExitDate": df.index[sig_idx+1],
                        "Type": "Entry Fail", "R": 0, "Age": df['age_years'].iloc[sig_idx], "AvgVol": df['vol_ma'].iloc[sig_idx],
                        "Status": "Entry Cond Failed", "Reason": f"{entry_mode}"
                    })
                    continue

                # EXIT LOGIC (PRESERVED TIME ONLY HOLD)
                atr = df['ATR'].iloc[actual_entry_idx]
                fixed_exit_idx = min(actual_entry_idx + params['holding_days'], len(df) - 1)
                
                if time_exit_only:
                    exit_price, exit_date, exit_type = df['Close'].iloc[fixed_exit_idx], df.index[fixed_exit_idx], "Time"
                elif is_overnight or is_intraday:
                    exit_idx = sig_idx + 1
                    exit_date, exit_price, exit_type = df.index[exit_idx], (df['Open'].iloc[exit_idx] if is_overnight else df['Close'].iloc[exit_idx]), "Time"
                else:
                    future = df.iloc[actual_entry_idx + 1 : fixed_exit_idx + 1]
                    stop_price = actual_entry_price - (atr * params['stop_atr']) if direction == 'Long' else actual_entry_price + (atr * params['stop_atr'])
                    tgt_price = actual_entry_price + (atr * params['tgt_atr']) if direction == 'Long' else actual_entry_price - (atr * params['tgt_atr'])
                    
                    exit_price, exit_type, exit_date = actual_entry_price, "Hold", None
                    for f_date, f_row in future.iterrows():
                        if direction == 'Long':
                            if f_row['Low'] <= stop_price: exit_price, exit_type, exit_date = stop_price, "Stop", f_date; break
                            if f_row['High'] >= tgt_price: exit_price, exit_type, exit_date = tgt_price, "Target", f_date; break
                        else:
                            if f_row['High'] >= stop_price: exit_price, exit_type, exit_date = stop_price, "Stop", f_date; break
                            if f_row['Low'] <= tgt_price: exit_price, exit_type, exit_date = tgt_price, "Target", f_date; break
                    
                    if exit_type == "Hold":
                        exit_price, exit_date, exit_type = future['Close'].iloc[-1], future.index[-1], "Time"

                ticker_last_exit = exit_date
                slip = slippage_bps / 10000.0
                tech_risk = (atr * params['stop_atr']) if (atr * params['stop_atr']) > 0 else 0.001
                pnl = (exit_price*(1-slip) - actual_entry_price*(1+slip)) if direction == 'Long' else (actual_entry_price*(1-slip) - exit_price*(1+slip))
                
                all_potential_trades.append({
                    "Ticker": ticker, "SignalDate": signal_date, "EntryDate": df.index[actual_entry_idx],
                    "Direction": direction, "Entry": actual_entry_price, "Exit": exit_price,
                    "ExitDate": exit_date, "Type": exit_type, "R": pnl / tech_risk,
                    "Age": df['age_years'].iloc[sig_idx], "AvgVol": df['vol_ma'].iloc[sig_idx],
                    "Status": "Valid Signal", "Reason": "Executed"
                })
        except: continue
        
    progress_bar.empty(); status_text.empty()
    if not all_potential_trades: return pd.DataFrame(), pd.DataFrame(), 0

    potential_df = pd.DataFrame(all_potential_trades).sort_values(by=["EntryDate", "Ticker"])
    final_trades_log, rejected_trades_log, active_positions, daily_entries = [], [], [], {}

    max_daily = params.get('max_daily_entries', 100)
    max_total = params.get('max_total_positions', 100)

    for idx, trade in potential_df.iterrows():
        if trade['Status'] == "Entry Cond Failed":
            rejected_trades_log.append(trade)
            continue

        entry_date = trade['EntryDate']
        active_positions = [t for t in active_positions if t['ExitDate'] > entry_date]
        
        is_rejected = (len(active_positions) >= max_total) or (daily_entries.get(entry_date, 0) >= max_daily)
        
        if is_rejected:
            trade['Status'] = "Portfolio Rejected"
            rejected_trades_log.append(trade)
        else:
            final_trades_log.append(trade)
            active_positions.append(trade)
            daily_entries[entry_date] = daily_entries.get(entry_date, 0) + 1

    return pd.DataFrame(final_trades_log), pd.DataFrame(rejected_trades_log), total_signals_generated

def grade_strategy(pf, sqn, win_rate, total_trades):
    score = 0
    if pf >= 2.0: score += 4
    elif pf >= 1.5: score += 3
    if sqn >= 2.0: score += 3
    if total_trades < 30: score -= 2
    if score >= 7: return "A", "Excellent", []
    if score >= 5: return "B", "Good", []
    return "C", "Marginal", []

def main():
    st.set_page_config(layout="wide", page_title="Quantitative Backtester")
    st.title("Quantitative Strategy Backtester")
    st.markdown("---")

    st.subheader("1. Universe & Data")
    col_u1, col_u2, col_u3 = st.columns([1, 1, 2])
    sample_pct = 100
    with col_u1:
        univ_choice = st.selectbox("Choose Universe", 
            ["Sector ETFs","SPX", "Indices", "International ETFs", "Sector + Index ETFs", "All CSV Tickers", "Custom (Upload CSV)"])
    with col_u2:
        start_date = st.date_input("Backtest Start Date", value=datetime.date(2000, 1, 1))
    
    custom_tickers = []
    if univ_choice == "Custom (Upload CSV)":
        with col_u3:
            sample_pct = st.slider("Sample %", 1, 100, 100)
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file:
                c_df = pd.read_csv(uploaded_file)
                custom_tickers = c_df["Ticker"].unique().tolist()
    
    use_full_history = st.checkbox("Download Full History", value=False)

    st.markdown("---")
    st.subheader("2. Execution & Risk")
    r_c1, r_c2, r_c3 = st.columns(3)
    with r_c1: trade_direction = st.selectbox("Trade Direction", ["Long", "Short"])
    with r_c2: 
        exit_mode = st.selectbox("Exit Mode", ["Standard (Stop & Target)", "No Stop (Target + Time)", "Time Only (Hold)"])
        time_exit_only = (exit_mode == "Time Only (Hold)")
        use_stop_loss = (exit_mode == "Standard (Stop & Target)")
        use_take_profit = (exit_mode != "Time Only (Hold)")
    with r_c3: max_one_pos = st.checkbox("Max 1 Position/Ticker", value=True)
    
    p_c1, p_c2, p_c3 = st.columns(3)
    with p_c1: max_daily_entries = st.number_input("Max New Trades Per Day", 1, 100, 2)
    with p_c2: max_total_positions = st.number_input("Max Total Positions", 1, 200, 10)
    with p_c3: slippage_bps = st.number_input("Slippage (bps)", value=5)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: 
        entry_type = st.selectbox("Entry Price", [
            "Signal Close", "T+1 Open", "T+1 Close",
            "T+1 Close < Signal Close", "T+1 Close > Signal Close",
            "T+1 Close < (Signal Close - 0.5 ATR)", "T+1 Close > (Signal Close + 0.5 ATR)",
            "Overnight (Buy Close, Sell T+1 Open)", "Intraday (Buy Open, Sell Close)",
            "Gap Up Only (Open > Prev High)", "Limit (Close -0.5 ATR)", "Limit (Prev Close)", 
            "Pullback 10 SMA (Entry: Close)", "Pullback 21 EMA (Entry: Level)"
        ])
    with c2: stop_atr = st.number_input("Stop Loss (ATR)", value=3.0, step=0.1, disabled=not use_stop_loss)
    with c3: tgt_atr = st.number_input("Target (ATR)", value=8.0, step=0.1, disabled=not use_take_profit)
    with c4: hold_days = st.number_input("Max Holding Days", value=10, step=1)
    with c5: risk_per_trade = st.number_input("Risk Amount ($)", value=1000, step=100)

    st.markdown("---")
    st.subheader("3. Signal Criteria")
    with st.expander("Liquidity & Data Filters", expanded=True):
        l1, l2, l3, l4, l5, l6 = st.columns(6)
        with l1: min_price = st.number_input("Min Price ($)", value=10.0)
        with l2: min_vol = st.number_input("Min Avg Vol", value=100000)
        with l3: min_age = st.number_input("Min True Age (Yrs)", value=0.25)
        with l4: max_age = st.number_input("Max True Age (Yrs)", value=100.0)
        with l5: min_atr_pct = st.number_input("Min ATR %", value=0.2)
        with l6: max_atr_pct = st.number_input("Max ATR %", value=10.0)

    with st.expander("Gaps, Acc/Dist, Trend, VIX", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1: use_gap_filter = st.checkbox("Gap Filter", value=False)
        with c2: use_acc_count_filter = st.checkbox("Acc Filter", value=False)
        with c3: trend_filter = st.selectbox("Trend", ["None", "Price > 200 SMA", "Market > 200 SMA"])
        with c4: use_vix_filter = st.checkbox("VIX Filter", value=False)
        
        v1, v2 = st.columns(2)
        with v1: vix_min = st.number_input("VIX Min", value=0.0)
        with v2: vix_max = st.number_input("VIX Max", value=20.0)

    # Note: Performance Ranks, Consec MAs, Seasonality, 52w filters are conceptually 
    # included in the run_engine logic, though UI for them is additive. 

    if st.button("Run Backtest", type="primary", use_container_width=True):
        sznl_map = load_seasonal_map()
        tickers = SECTOR_ETFS if univ_choice == "Sector ETFs" else INDEX_ETFS
        fetch_start = "1950-01-01" if use_full_history else start_date - datetime.timedelta(days=365)
        
        data_dict = download_universe_data(tickers, fetch_start)
        market_df = download_universe_data([MARKET_TICKER], fetch_start).get(MARKET_TICKER, None)
        market_series = (market_df['Close'] > market_df['Close'].rolling(200).mean()) if market_df is not None else None
        
        params = {
            'backtest_start_date': start_date, 'trade_direction': trade_direction,
            'max_one_pos': max_one_pos, 'max_daily_entries': max_daily_entries,
            'max_total_positions': max_total_positions, 'time_exit_only': time_exit_only,
            'stop_atr': stop_atr, 'tgt_atr': tgt_atr, 'holding_days': hold_days,
            'entry_type': entry_type, 'min_price': min_price, 'min_vol': min_vol,
            'min_age': min_age, 'max_age': max_age, 'min_atr_pct': min_atr_pct, 'max_atr_pct': max_atr_pct,
            'trend_filter': trend_filter, 'slippage_bps': slippage_bps, 'universe_tickers': tickers,
            'use_vix_filter': use_vix_filter, 'vix_min': vix_min, 'vix_max': vix_max,
            'use_gap_filter': use_gap_filter, 'gap_logic': '>', 'gap_thresh': 3, 'gap_lookback': 21,
            'use_acc_count_filter': use_acc_count_filter, 'acc_count_window': 21, 'acc_count_logic': '>', 'acc_count_thresh': 3,
            'perf_filters': [], 'ma_consec_filters': [], 'use_sznl': False, 'use_52w': False, 'breakout_mode': "None"
        }
        
        trades_df, rejected_df, total_signals = run_engine(data_dict, params, sznl_map, market_series)
        
        if not trades_df.empty:
            trades_df['PnL_Dollar'] = trades_df['R'] * risk_per_trade
            trades_df['CumPnL'] = trades_df['PnL_Dollar'].cumsum()
            
            wins = trades_df[trades_df['R'] > 0]
            losses = trades_df[trades_df['R'] <= 0]
            pf = wins['PnL_Dollar'].sum() / abs(losses['PnL_Dollar'].sum()) if not losses.empty else 999
            win_rate = (len(wins) / len(trades_df)) * 100
            sqn = np.sqrt(len(trades_df)) * (trades_df['R'].mean() / trades_df['R'].std()) if len(trades_df) > 1 else 0
            grade, verdict, _ = grade_strategy(pf, sqn, win_rate, len(trades_df))

            st.success(f"Backtest Complete! Grade: {grade} ({verdict})")
            st.plotly_chart(px.line(trades_df, x="ExitDate", y="CumPnL", title="Equity Curve"), use_container_width=True)
            st.dataframe(trades_df)
        else:
            st.warning("No trades found.")

if __name__ == "__main__":
    main()
