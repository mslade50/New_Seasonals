import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import gspread
import time
import json
import os
from pandas.tseries.offsets import BusinessDay

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & STRATEGY BOOK
# -----------------------------------------------------------------------------
# (Paste your STRATEGY_BOOK list here exactly as it is in screener.py)
# For brevity in this answer, I am assuming you will copy the STRATEGY_BOOK 
# variable from your existing script and paste it right here.
STRATEGY_BOOK = [
    # ... PASTE YOUR FULL STRATEGY_BOOK LIST HERE ...
]

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS (Refactored for Automation)
# -----------------------------------------------------------------------------

def get_google_client():
    """
    Authenticates with Google Sheets using Environment Variables (GitHub) 
    or a local JSON file.
    """
    try:
        # 1. GitHub Actions (Secret named GCP_JSON)
        if "GCP_JSON" in os.environ:
            creds_dict = json.loads(os.environ["GCP_JSON"])
            return gspread.service_account_from_dict(creds_dict)
        
        # 2. Local File Fallback
        elif os.path.exists("credentials.json"):
            return gspread.service_account(filename='credentials.json')
            
        else:
            print("❌ Error: No credentials found (GCP_JSON env var or credentials.json).")
            return None
    except Exception as e:
        print(f"❌ Auth Error: {e}")
        return None

def load_seasonal_map(csv_path="sznl_ranks.csv"):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        print(f"⚠️ Warning: Could not find {csv_path}")
        return {}

    if df.empty: return {}
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce').dt.normalize()
    df = df.dropna(subset=["Date"])
    
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

def download_historical_data(tickers, start_date="2000-01-01"):
    if not tickers: return {}
    clean_tickers = list(set([str(t).strip().upper().replace('.', '-') for t in tickers]))
    
    data_dict = {}
    CHUNK_SIZE = 50 
    total = len(clean_tickers)
    
    print(f"📥 Downloading data for {total} tickers...")
    
    for i in range(0, total, CHUNK_SIZE):
        chunk = clean_tickers[i : i + CHUNK_SIZE]
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
        except Exception as e:
            print(f"⚠️ Batch Error: {e}")
            
    return data_dict

def calculate_indicators(df, sznl_map, ticker, market_series=None):
    # (Copy the EXACT calculate_indicators function from your screener.py)
    # No changes needed, just paste it here.
    # ...
    # [PASTE calculate_indicators HERE]
    return df

def check_signal(df, params, sznl_map):
    # (Copy the EXACT check_signal function from your screener.py)
    # No changes needed, just paste it here.
    # ...
    # [PASTE check_signal HERE]
    return True

# -----------------------------------------------------------------------------
# 3. SAVING FUNCTIONS (Refactored)
# -----------------------------------------------------------------------------

def save_signals_to_gsheet(new_dataframe, sheet_name='Trade_Signals_Log'):
    if new_dataframe.empty: return
    
    # Clean Data
    df_new = new_dataframe.copy()
    cols_to_round = ['Entry', 'Stop', 'Target', 'ATR']
    existing_cols = [c for c in cols_to_round if c in df_new.columns]
    df_new[existing_cols] = df_new[existing_cols].astype(float).round(2)
    df_new['Date'] = df_new['Date'].astype(str) 
    df_new["Scan_Timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cols = ['Scan_Timestamp'] + [c for c in df_new.columns if c != 'Scan_Timestamp']
    df_new = df_new[cols]

    gc = get_google_client()
    if not gc: return

    try:
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

        # Dedup
        combined = combined.drop_duplicates(subset=['Ticker', 'Date', 'Strategy_ID'], keep='last')
        
        worksheet.clear()
        data_to_write = [combined.columns.tolist()] + combined.astype(str).values.tolist()
        worksheet.update(values=data_to_write)
        print(f"✅ Signals Log Synced! ({len(combined)} rows)")
        
    except Exception as e:
        print(f"❌ Google Sheet Error: {e}")

def save_staging_orders(signals_list, strategy_book, sheet_name='Order_Staging'):
    if not signals_list: return
    
    df = pd.DataFrame(signals_list)
    strat_map = {s['id']: s for s in strategy_book}
    staging_data = []
    
    for _, row in df.iterrows():
        strat = strat_map.get(row['Strategy_ID'])
        if not strat: continue
        settings = strat['settings']
        
        # Entry Logic
        entry_mode = settings.get('entry_type', 'Signal Close')
        entry_instruction = "MKT"
        offset_atr = 0.0
        if "Limit" in entry_mode and "ATR" in entry_mode:
            entry_instruction = "REL_OPEN" 
            if "0.5" in entry_mode: offset_atr = 0.5
        elif "T+1 Open" in entry_mode:
            entry_instruction = "MOO" 
        
        ib_action = "SELL" if "SHORT" in row['Action'] else "BUY"

        staging_data.append({
            "Scan_Date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "Symbol": row['Ticker'],
            "SecType": "STK",
            "Exchange": "SMART",
            "Action": ib_action,
            "Quantity": row['Shares'],
            "Order_Type": entry_instruction, 
            "Offset_ATR_Mult": offset_atr,   
            "Frozen_ATR": round(row['ATR'], 2),
            "Time_Exit_Date": str(row['Time Exit']),
            "Strategy_Ref": strat['name']
        })

    df_stage = pd.DataFrame(staging_data)
    gc = get_google_client()
    if not gc: return

    try:
        sh = gc.open("Trade_Signals_Log") # Using same workbook? Or separate? Adjust if needed.
        try:
            worksheet = sh.worksheet(sheet_name)
        except:
            worksheet = sh.add_worksheet(title=sheet_name, rows=100, cols=20)

        worksheet.clear()
        data_to_write = [df_stage.columns.tolist()] + df_stage.astype(str).values.tolist()
        worksheet.update(values=data_to_write)
        print(f"🤖 Orders Staged! ({len(df_stage)} orders)")
        
    except Exception as e:
        print(f"❌ Staging Error: {e}")

# -----------------------------------------------------------------------------
# 4. MAIN EXECUTION
# -----------------------------------------------------------------------------
def run_daily_scan():
    print("--- Starting Daily Automated Scan ---")
    sznl_map = load_seasonal_map()
    
    # 1. Gather Tickers
    all_tickers = set()
    for strat in STRATEGY_BOOK:
        all_tickers.update(strat['universe_tickers'])
        s = strat['settings']
        if s.get('use_market_sznl'): all_tickers.add(s.get('market_ticker', '^GSPC'))
        if "Market" in s.get('trend_filter', ''): all_tickers.add(s.get('market_ticker', 'SPY'))
        if "SPY" in s.get('trend_filter', ''): all_tickers.add("SPY")
    
    # 2. Download Data
    master_dict = download_historical_data(list(all_tickers))
    
    all_signals = []

    # 3. Run Strategies
    for strat in STRATEGY_BOOK:
        print(f"Running: {strat['name']}...")
        
        # Prepare Market Series
        mkt_ticker = strat['settings'].get('market_ticker', 'SPY')
        mkt_df = master_dict.get(mkt_ticker)
        if mkt_df is None: mkt_df = master_dict.get('SPY')
        
        market_series = None
        if mkt_df is not None:
            temp_mkt = mkt_df.copy()
            temp_mkt['SMA200'] = temp_mkt['Close'].rolling(200).mean()
            market_series = temp_mkt['Close'] > temp_mkt['SMA200']

        signals = []
        for ticker in strat['universe_tickers']:
            t_clean = ticker.replace('.', '-')
            df = master_dict.get(t_clean)
            if df is None or len(df) < 250: continue
            
            try:
                calc_df = calculate_indicators(df.copy(), sznl_map, t_clean, market_series)
                if check_signal(calc_df, strat['settings'], sznl_map):
                    # ... (PASTE YOUR SIGNAL ENTRY/RISK LOGIC FROM SCREENER.PY HERE) ...
                    # This is the block that calculates risk, shares, targets, etc.
                    # Copy from "last_row = calc_df.iloc[-1]" down to "signals.append({...})"
                    pass 
            except Exception:
                continue
        
        if signals:
            all_signals.extend(signals)
            print(f"  -> Found {len(signals)} signals.")

    # 4. Save Results
    if all_signals:
        df_sig = pd.DataFrame(all_signals)
        save_signals_to_gsheet(df_sig)
        save_staging_orders(all_signals, STRATEGY_BOOK)
    else:
        print("No signals found today.")

    print("--- Scan Complete ---")

if __name__ == "__main__":
    run_daily_scan()
