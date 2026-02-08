"""
verify_fills.py — Post-Close Signal Fill Verification (Layer 1)
================================================================
Runs daily after market close via GitHub Actions (~5:00 PM ET).
Reads the Trade_Signals_Log, checks if limit prices would have been hit
using yfinance daily OHLC, and marks each signal with a Fill_Status.

Fill Statuses:
    FILLED        — Price confirmed hit during the session (or MOC/MOO assumed filled)
    EXPIRED       — Order window passed without fill
    PENDING       — GTC order still live (Time Exit in future, price not yet hit)
    INVALIDATED   — Manual override (user sets this for pulled MOC orders, etc.)
    MANUAL_REVIEW — Could not determine order type or missing data

This script NEVER overwrites INVALIDATED status (manual override is sacred).

REQUIRES:
    - Google Sheets credentials (GCP_JSON env var or local credentials.json)
    - strategy_config.py in parent directory (for historical signal lookups)
    - yfinance (pip install yfinance)

TIMING NOTE:
    Run AFTER market close. For DAY limit orders staged today (execution = tomorrow),
    this script will mark them PENDING today and verify them tomorrow.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import os
import json
import datetime
import time
import sys
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

TRADING_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# =============================================================================
# CONFIGURATION
# =============================================================================

SHEET_NAME = "Trade_Signals_Log"

# How far back to verify untagged signals on first run (calendar days)
MAX_LOOKBACK_DAYS = 45

# Rate limit pause between yfinance batches (seconds)
YF_BATCH_PAUSE = 0.5

# =============================================================================
# IMPORT STRATEGY BOOK (for historical signals missing Entry_Type_Short)
# =============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

try:
    from strategy_config import STRATEGY_BOOK
except ImportError:
    print("⚠️  strategy_config.py not found. Historical signals without Entry_Type_Short → MANUAL_REVIEW.")
    STRATEGY_BOOK = []


# =============================================================================
# AUTH — Mirrors daily_scan.py
# =============================================================================

def get_google_client():
    """Authenticate with Google Sheets. Priority: env var → local file."""
    try:
        if "GCP_JSON" in os.environ:
            creds_dict = json.loads(os.environ["GCP_JSON"])
            return gspread.service_account_from_dict(creds_dict)
        local_paths = [
            os.path.join(current_dir, 'credentials.json'),
            'credentials.json',
        ]
        for path in local_paths:
            if os.path.exists(path):
                return gspread.service_account(filename=path)
        print("❌ No credentials found (GCP_JSON env var or credentials.json).")
        return None
    except Exception as e:
        print(f"❌ Auth Error: {e}")
        return None


# =============================================================================
# STRATEGY LOOKUP — Used for signals that predate the Entry_Type_Short column
# =============================================================================

def build_strategy_map() -> dict:
    """
    Build lookup: Strategy_ID -> order classification details.
    Mirrors the staging logic in daily_scan.py save_staging_orders().
    """
    smap = {}
    for s in STRATEGY_BOOK:
        settings = s.get('settings', {})
        entry_type = settings.get('entry_type', 'Signal Close')

        order_class = 'MOC'
        offset = 0.25
        tif = 'DAY'

        if 'Limit' in entry_type and 'ATR' in entry_type:
            if 'Persistent' in entry_type:
                order_class = 'REL_CLOSE'
                tif = 'GTC'
            else:
                order_class = 'REL_OPEN'
                tif = 'DAY'
            if '0.5' in entry_type:
                offset = 0.5
        elif 'LOC' in entry_type:
            order_class = 'LOC'
        elif 'T+1 Open' in entry_type:
            order_class = 'MOO'
            tif = 'OPG'
        elif 'T+1 Close if < Signal Close' in entry_type:
            order_class = 'LMT'
        elif 'Signal Close' in entry_type:
            order_class = 'MOC'

        smap[s['id']] = {
            'order_class': order_class,
            'offset': offset,
            'tif': tif,
            'entry_type_raw': entry_type,
        }
    return smap


def classify_order(row, strategy_map: dict) -> tuple:
    """
    Determine (order_class, atr_offset, tif) for a signal row.
    Priority: 1) Entry_Type_Short column  2) Strategy config lookup  3) UNKNOWN
    
    Returns:
        (order_class, offset, tif) — e.g. ('REL_OPEN', 0.25, 'DAY')
    """
    # --- Priority 1: Entry_Type_Short column (present if daily_scan preserves it) ---
    ets = str(row.get('Entry_Type_Short', '')).strip()
    if ets and ets not in ('', 'nan', 'None'):
        ets_upper = ets.upper()
        if 'MOC' in ets_upper or 'SIGNAL CLOSE' in ets_upper:
            return 'MOC', 0.0, 'DAY'
        elif 'MOO' in ets_upper:
            return 'MOO', 0.0, 'OPG'
        elif 'PERS' in ets_upper or 'REL_CLOSE' in ets_upper or 'GTC' in ets_upper:
            # Persistent ATR Limit → anchored to signal close, GTC
            offset = 0.5 if '0.5' in ets else 0.25
            return 'REL_CLOSE', offset, 'GTC'
        elif 'REL_OPEN' in ets_upper or 'ATR LMT' in ets_upper:
            offset = 0.5 if '0.5' in ets else 0.25
            return 'REL_OPEN', offset, 'DAY'
        elif 'LOC' in ets_upper:
            return 'LOC', 0.0, 'DAY'
        elif 'LMT' in ets_upper:
            return 'LMT', 0.0, 'DAY'

    # --- Priority 2: Strategy config lookup ---
    strat_id = str(row.get('Strategy_ID', '')).strip()
    if strat_id in strategy_map:
        info = strategy_map[strat_id]
        return info['order_class'], info['offset'], info['tif']

    return 'UNKNOWN', 0.0, 'DAY'


# =============================================================================
# PRICE DATA
# =============================================================================

def fetch_price_data(tickers: list, start_date, end_date) -> dict:
    """
    Fetch daily OHLC for tickers from yfinance.
    Returns: {ticker: DataFrame[Open, High, Low, Close]}
    
    Handles the yfinance MultiIndex column trap.
    """
    if not tickers:
        return {}

    # Buffer to handle edge-of-range lookups
    start_dt = pd.to_datetime(start_date) - pd.Timedelta(days=7)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=7)

    price_data = {}
    ticker_list = sorted(set(tickers))
    batch_size = 20

    for i in range(0, len(ticker_list), batch_size):
        batch = ticker_list[i:i + batch_size]
        try:
            df = yf.download(batch, start=start_dt, end=end_dt, auto_adjust=True, progress=False)
            if df.empty:
                continue

            # ── yfinance MultiIndex trap ──
            if isinstance(df.columns, pd.MultiIndex):
                for ticker in batch:
                    try:
                        t_df = df.xs(ticker, level=1, axis=1).copy()
                        if isinstance(t_df.columns, pd.MultiIndex):
                            t_df.columns = t_df.columns.get_level_values(0)
                        t_df.columns = [str(c).capitalize() for c in t_df.columns]
                        if not t_df.dropna(how='all').empty:
                            price_data[ticker] = t_df.dropna(subset=['Close'])
                    except (KeyError, TypeError):
                        pass
            else:
                # Single ticker download
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [str(c).capitalize() for c in df.columns]
                if len(batch) == 1 and not df.dropna(how='all').empty:
                    price_data[batch[0]] = df.dropna(subset=['Close'])

        except Exception as e:
            print(f"  ⚠️  yfinance error for batch {batch[:3]}...: {e}")

        time.sleep(YF_BATCH_PAUSE)

    return price_data


# =============================================================================
# FILL CHECK ENGINE
# =============================================================================

def check_fill(order_class: str, action: str, signal_close: float, atr: float,
               offset: float, limit_price_override: float,
               ticker_prices: pd.DataFrame, signal_date, exit_date, tif: str) -> tuple:
    """
    Core fill-check logic for a single signal.
    
    Returns:
        (fill_status, fill_date, fill_price)
        fill_status: 'FILLED' | 'EXPIRED' | 'PENDING' | 'MANUAL_REVIEW'
    """
    today = datetime.date.today()
    is_buy = 'SHORT' not in str(action).upper() and str(action).upper() != 'SELL'

    if ticker_prices is None or ticker_prices.empty:
        return 'MANUAL_REVIEW', None, None

    # Normalize index for date comparison
    prices = ticker_prices.copy()
    prices.index = pd.to_datetime(prices.index)
    signal_dt = pd.to_datetime(signal_date)
    exit_dt = pd.to_datetime(exit_date) if exit_date else None

    # ── MOC: Fills at close on signal date ──
    # Assumed filled. User can manually override to INVALIDATED.
    if order_class == 'MOC':
        return 'FILLED', signal_date, signal_close

    # ── MOO: Fills at T+1 open ──
    if order_class == 'MOO':
        t1 = signal_dt + TRADING_DAY
        t1_rows = prices[prices.index.normalize() == t1.normalize()]
        if not t1_rows.empty:
            fill_px = round(float(t1_rows['Open'].iloc[0]), 2)
            return 'FILLED', t1.date(), fill_px
        elif t1.date() <= today:
            return 'EXPIRED', None, None
        else:
            return 'PENDING', None, None

    # ── Limit-based orders ──
    # Determine the window of dates to check
    t1 = signal_dt + TRADING_DAY

    if tif == 'GTC' and exit_dt:
        last_check = exit_dt
    else:
        last_check = t1  # DAY: only check T+1

    window = prices[
        (prices.index.normalize() >= t1.normalize()) &
        (prices.index.normalize() <= pd.to_datetime(last_check).normalize())
    ]

    if window.empty:
        if pd.to_datetime(last_check).date() < today:
            return 'EXPIRED', None, None
        return 'PENDING', None, None

    # ── Calculate limit price ──
    limit_price = None

    if order_class == 'REL_CLOSE':
        # Anchored to signal close
        buffer = atr * offset
        limit_price = (signal_close - buffer) if is_buy else (signal_close + buffer)

    elif order_class == 'REL_OPEN':
        # Anchored to T+1 open — calc limit from that day's open
        t1_open = float(window['Open'].iloc[0])
        buffer = atr * offset
        limit_price = (t1_open - buffer) if is_buy else (t1_open + buffer)
        # DAY order: only check this single day
        if tif != 'GTC':
            window = window.iloc[:1]

    elif order_class == 'LOC':
        # Limit on close — compare closing price, not intraday
        limit_price = limit_price_override if limit_price_override else signal_close
        for idx, day in window.iterrows():
            close_px = float(day['Close'])
            if (is_buy and close_px <= limit_price) or (not is_buy and close_px >= limit_price):
                return 'FILLED', idx.date(), round(close_px, 2)
        if pd.to_datetime(last_check).date() < today:
            return 'EXPIRED', None, None
        return 'PENDING', None, None

    elif order_class == 'LMT':
        # Pre-calculated limit (e.g., "T+1 Close if < Signal Close" → Entry - 0.01)
        limit_price = limit_price_override if limit_price_override else (signal_close - 0.01)

    if limit_price is None:
        return 'MANUAL_REVIEW', None, None

    limit_price = round(limit_price, 2)

    # ── Check each day in window for fill ──
    for idx, day in window.iterrows():
        low = float(day['Low'])
        high = float(day['High'])

        if is_buy and low <= limit_price:
            return 'FILLED', idx.date(), limit_price
        elif not is_buy and high >= limit_price:
            return 'FILLED', idx.date(), limit_price

    # No fill found
    if pd.to_datetime(last_check).date() < today:
        return 'EXPIRED', None, None
    return 'PENDING', None, None


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_fill_verification():
    """
    Full post-close verification workflow.
    1. Load signals log from Google Sheets
    2. Identify signals needing verification (no status, or PENDING)
    3. Fetch yfinance data for relevant tickers/dates
    4. Apply fill logic per order type
    5. Write Fill_Status, Fill_Date, Fill_Price back to sheet
    """
    print(f"\n{'='*60}")
    print(f"  FILL VERIFICATION — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    # ── 1. Load signals ──
    gc = get_google_client()
    if not gc:
        return

    sh = gc.open(SHEET_NAME)
    worksheet = sh.sheet1
    all_values = worksheet.get_all_values()

    if len(all_values) < 2:
        print("ℹ️  No signals in log.")
        return

    headers = all_values[0]
    df = pd.DataFrame(all_values[1:], columns=headers)
    print(f"📋 Loaded {len(df)} signals from log.")

    # ── 2. Initialize new columns if missing ──
    for col in ['Fill_Status', 'Fill_Date', 'Fill_Price']:
        if col not in df.columns:
            df[col] = ''

    # ── 3. Filter to signals needing verification ──
    # Never touch: FILLED, EXPIRED, INVALIDATED
    frozen_statuses = {'FILLED', 'EXPIRED', 'INVALIDATED'}
    needs_check_mask = ~df['Fill_Status'].astype(str).str.strip().isin(frozen_statuses)

    # Apply lookback cutoff for first-run backfill
    df['_signal_date'] = pd.to_datetime(df['Date'], errors='coerce')
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=MAX_LOOKBACK_DAYS)
    date_ok_mask = df['_signal_date'] >= cutoff

    check_mask = needs_check_mask & date_ok_mask
    indices_to_check = df.index[check_mask].tolist()

    if not indices_to_check:
        print("✅ All signals within lookback window already verified.")
        df.drop(columns=['_signal_date'], inplace=True)
        return

    print(f"🔍 {len(indices_to_check)} signals need verification.\n")

    # ── 4. Build strategy lookup for historical signals ──
    strategy_map = build_strategy_map()

    # ── 5. Determine tickers and date range, fetch prices ──
    check_df = df.loc[indices_to_check]
    tickers_needed = check_df['Ticker'].unique().tolist()

    min_signal_date = check_df['_signal_date'].min()
    exit_dates = pd.to_datetime(check_df['Time Exit'], errors='coerce')
    max_date = max(
        exit_dates.max() if not exit_dates.isna().all() else pd.Timestamp.today(),
        pd.Timestamp.today()
    )

    print(f"📡 Fetching price data for {len(tickers_needed)} tickers "
          f"({min_signal_date.date()} → {max_date.date()})...")
    price_data = fetch_price_data(tickers_needed, min_signal_date, max_date)
    print(f"   ✅ Got data for {len(price_data)} / {len(tickers_needed)} tickers.\n")

    # ── 6. Verify each signal ──
    updates = 0
    status_summary = {'FILLED': 0, 'EXPIRED': 0, 'PENDING': 0, 'MANUAL_REVIEW': 0}

    for idx in indices_to_check:
        row = df.loc[idx]

        # Skip INVALIDATED even if it somehow got here
        if str(row['Fill_Status']).strip() == 'INVALIDATED':
            continue

        ticker = str(row['Ticker']).upper().strip()
        order_class, offset, tif = classify_order(row, strategy_map)

        if order_class == 'UNKNOWN':
            df.at[idx, 'Fill_Status'] = 'MANUAL_REVIEW'
            status_summary['MANUAL_REVIEW'] += 1
            updates += 1
            print(f"  ❓ {ticker:<6} | UNKNOWN order type | Strategy: {row.get('Strategy_ID', '?')} → MANUAL_REVIEW")
            continue

        # Parse signal values
        signal_close = float(row['Entry']) if str(row.get('Entry', '')).strip() not in ('', 'nan') else 0.0
        atr = float(row['ATR']) if str(row.get('ATR', '')).strip() not in ('', 'nan') else 0.0
        action = str(row['Action'])
        signal_date = df.at[idx, '_signal_date'].date() if pd.notna(df.at[idx, '_signal_date']) else None
        exit_date_raw = pd.to_datetime(row['Time Exit'], errors='coerce')
        exit_date = exit_date_raw.date() if pd.notna(exit_date_raw) else None

        # Get explicit limit price if available (for LOC, LMT)
        limit_px = None
        lp_str = str(row.get('Limit_Price', '')).strip()
        if lp_str not in ('', 'nan', 'None', '0', '0.0'):
            try:
                limit_px = float(lp_str)
            except ValueError:
                limit_px = None

        if signal_date is None:
            df.at[idx, 'Fill_Status'] = 'MANUAL_REVIEW'
            status_summary['MANUAL_REVIEW'] += 1
            updates += 1
            continue

        ticker_prices = price_data.get(ticker)

        status, fill_date, fill_price = check_fill(
            order_class=order_class,
            action=action,
            signal_close=signal_close,
            atr=atr,
            offset=offset,
            limit_price_override=limit_px,
            ticker_prices=ticker_prices,
            signal_date=signal_date,
            exit_date=exit_date,
            tif=tif,
        )

        df.at[idx, 'Fill_Status'] = status
        df.at[idx, 'Fill_Date'] = str(fill_date) if fill_date else ''
        df.at[idx, 'Fill_Price'] = str(round(fill_price, 2)) if fill_price else ''

        status_summary[status] = status_summary.get(status, 0) + 1
        updates += 1

        icon = {'FILLED': '✅', 'EXPIRED': '⛔', 'PENDING': '⏳', 'MANUAL_REVIEW': '❓'}.get(status, '?')
        extra = f"@ ${fill_price:.2f}" if fill_price else ""
        print(f"  {icon} {ticker:<6} | {order_class:<10} | {tif:<4} | {status:<15} {extra}")

    # ── 7. Write back to sheet ──
    df.drop(columns=['_signal_date'], inplace=True)

    if updates > 0:
        print(f"\n📤 Writing {updates} updates back to Google Sheets...")
        df_clean = df.fillna('')
        data_to_write = [df_clean.columns.tolist()] + df_clean.astype(str).values.tolist()
        worksheet.clear()
        worksheet.update(values=data_to_write)
        print("✅ Signals log updated.\n")

    # ── 8. Summary ──
    print(f"{'='*40}")
    print(f"  VERIFICATION SUMMARY")
    print(f"{'='*40}")
    for s, count in sorted(status_summary.items()):
        if count > 0:
            print(f"  {s}: {count}")

    total_in_log = len(df)
    filled_total = len(df[df['Fill_Status'] == 'FILLED'])
    expired_total = len(df[df['Fill_Status'] == 'EXPIRED'])
    pending_total = len(df[df['Fill_Status'] == 'PENDING'])
    print(f"\n  Log totals: {filled_total} filled, {expired_total} expired, "
          f"{pending_total} pending, {total_in_log} total")
    print()


if __name__ == "__main__":
    run_fill_verification()
