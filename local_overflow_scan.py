"""
Local Overflow Scanner
======================
Scans the extended ticker universe (CSV_UNIVERSE minus LIQUID_PLUS_COMMODITIES)
for strategy signals. Runs locally via Task Scheduler to complement the GitHub
Actions daily scan which covers the core liquid universe.

Uses a local parquet cache for price data — full download on first run,
incremental append on subsequent runs. This keeps yfinance calls minimal.

Usage:
    python local_overflow_scan.py              # scan all overflow strategies
    python local_overflow_scan.py --rebuild    # force full cache rebuild
    python local_overflow_scan.py --dry-run    # scan but don't email/sheet
"""

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import time
import pytz
import sys
import os
import argparse
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from indicators import calculate_indicators
import copy
from strategy_config import (
    STRATEGY_BOOK, CSV_UNIVERSE, LIQUID_PLUS_COMMODITIES,
    ACCOUNT_VALUE, build_strategy_book
)

# Overflow runs these strategies against the extended universe.
# We deep-copy each and swap universe_tickers to CSV_UNIVERSE so the
# overflow filter (strat['universe_tickers'] ∩ OVERFLOW_TICKERS) actually
# yields the 870 extended names instead of an empty intersection.
OVERFLOW_STRATEGIES = []
for _s in STRATEGY_BOOK:
    if _s['name'] == "Overbot Vol Spike":
        _s_copy = copy.deepcopy(_s)
        _s_copy['universe_tickers'] = CSV_UNIVERSE
        OVERFLOW_STRATEGIES.append(_s_copy)
    elif _s['name'] == "LT Trend ST OS":
        _s_copy = copy.deepcopy(_s)
        _s_copy['universe_tickers'] = CSV_UNIVERSE
        OVERFLOW_STRATEGIES.append(_s_copy)
    elif _s['name'] == "Oversold Low Volume":
        _s_copy = copy.deepcopy(_s)
        _s_copy['universe_tickers'] = CSV_UNIVERSE
        OVERFLOW_STRATEGIES.append(_s_copy)
    elif _s['name'] == "St OS Sznl":
        _s_copy = copy.deepcopy(_s)
        _s_copy['universe_tickers'] = CSV_UNIVERSE
        OVERFLOW_STRATEGIES.append(_s_copy)
    elif _s['name'] == "52wh Breakout":
        _s_copy = copy.deepcopy(_s)
        _s_copy['universe_tickers'] = CSV_UNIVERSE
        OVERFLOW_STRATEGIES.append(_s_copy)
from daily_scan import (
    check_signal, load_seasonal_map,
    get_entry_type_short, get_sizing_variable, build_live_filters,
    generate_oversold_lv_companion,
    load_olv_cooldown, load_open_position_counts,
    OLV_STRATEGY_NAME, OLV_COOLDOWN_DAYS,
    load_atr_seasonal_map, ATR_SZNL_COLS,
    DAILY_RISK_CAP_BPS,
)
import json as _json
import gspread as _gspread


def get_google_client():
    """Local credential lookup for Task Scheduler runs.

    Checks GCP_JSON env var, project-dir credentials.json, and OneDrive
    fallback. daily_scan's version only checks cwd, which fails when this
    script runs from Task Scheduler.
    """
    try:
        if "GCP_JSON" in os.environ:
            creds_dict = _json.loads(os.environ["GCP_JSON"])
            return _gspread.service_account_from_dict(creds_dict)

        candidate_paths = [
            os.path.join(current_dir, "credentials.json"),
            "credentials.json",
            os.path.expanduser(r"~\OneDrive\credentials.json"),
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                return _gspread.service_account(filename=path)

        print(f"❌ No credentials found. Checked: {candidate_paths}")
        return None
    except Exception as e:
        print(f"❌ Auth Error: {e}")
        return None

TRADING_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# --- Cache Config ---
CACHE_DIR = os.path.join(current_dir, "data")
CACHE_FILE = os.path.join(CACHE_DIR, "overflow_price_cache.parquet")
FRAG_CACHE = os.path.join(CACHE_DIR, "rd2_fragility.parquet")
FRAG_CACHE_TS = os.path.join(CACHE_DIR, "rd2_fragility_ts.parquet")

# Overflow universe = CSV tickers not already covered by the Actions scan
OVERFLOW_TICKERS = sorted(set(CSV_UNIVERSE) - set(LIQUID_PLUS_COMMODITIES))


# ============================================================================
# PARQUET CACHE MANAGER
# ============================================================================

def load_cache():
    """Load the parquet price cache. Returns dict {ticker: DataFrame} or empty dict."""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        store = pd.read_parquet(CACHE_FILE)
        data_dict = {}
        for ticker in store['ticker'].unique():
            t_df = store[store['ticker'] == ticker].drop(columns=['ticker']).copy()
            t_df.index = pd.to_datetime(t_df['date'])
            t_df = t_df.drop(columns=['date']).sort_index()
            data_dict[ticker] = t_df
        print(f"📦 Loaded cache: {len(data_dict)} tickers, last date {store['date'].max()}")
        return data_dict
    except Exception as e:
        print(f"⚠️ Cache load failed: {e}")
        return {}


def save_cache(data_dict):
    """Save data_dict to parquet cache."""
    frames = []
    for ticker, df in data_dict.items():
        t_df = df.copy()
        t_df['ticker'] = ticker
        t_df['date'] = t_df.index
        frames.append(t_df)
    if frames:
        store = pd.concat(frames, ignore_index=True)
        store.to_parquet(CACHE_FILE, index=False)
        print(f"💾 Cache saved: {len(data_dict)} tickers, {len(store)} rows")


def download_batch(tickers, start_date="2000-01-01"):
    """Download OHLCV data for a list of tickers. Returns dict {ticker: DataFrame}."""
    clean_tickers = list(set([str(t).strip().upper().replace('.', '-') for t in tickers]))
    if not clean_tickers:
        return {}

    data_dict = {}
    CHUNK_SIZE = 20
    total = len(clean_tickers)
    print(f"📥 Downloading {total} tickers from {start_date}...")

    for i in range(0, total, CHUNK_SIZE):
        chunk = clean_tickers[i:i + CHUNK_SIZE]
        batch_num = i // CHUNK_SIZE + 1
        total_batches = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"   Batch {batch_num}/{total_batches} ({len(chunk)} tickers)...")

        for attempt in range(3):
            try:
                df = yf.download(
                    chunk, start=start_date, group_by='ticker',
                    auto_adjust=True, progress=False, threads=True
                )
                if df.empty:
                    break

                if len(chunk) == 1:
                    t = chunk[0]
                    t_df = df.dropna(subset=['Close']) if 'Close' in df.columns else pd.DataFrame()
                    if not t_df.empty:
                        t_df.index = t_df.index.normalize()
                        if t_df.index.tz is not None:
                            t_df.index = t_df.index.tz_localize(None)
                        t_df.columns = [c.capitalize() for c in t_df.columns]
                        data_dict[t] = t_df
                else:
                    for t in chunk:
                        try:
                            if isinstance(df.columns, pd.MultiIndex):
                                if t in df.columns.get_level_values(0):
                                    t_df = df[t].copy()
                                else:
                                    continue
                            else:
                                continue
                            t_df = t_df.dropna(subset=['Close']) if 'Close' in t_df.columns else pd.DataFrame()
                            if not t_df.empty:
                                t_df.index = t_df.index.normalize()
                                if t_df.index.tz is not None:
                                    t_df.index = t_df.index.tz_localize(None)
                                t_df.columns = [c.capitalize() for c in t_df.columns]
                                data_dict[t] = t_df
                        except Exception:
                            continue
                break  # success
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                else:
                    print(f"   ⚠️ Batch failed after 3 attempts: {e}")

        time.sleep(1.5)  # rate limit

    print(f"   ✓ Downloaded {len(data_dict)}/{total} tickers")
    return data_dict


def update_cache(data_dict, force_rebuild=False, expected_data_date=None):
    """Load cache, download missing/stale data, merge and save.

    A ticker is stale if its last cached bar predates `expected_data_date`
    (typically yesterday's trading close pre-market, today's close post-open).
    Falling back to a 5-day threshold when no date is provided keeps ad-hoc
    runs from re-fetching the whole universe unnecessarily.

    Returns the fully updated data_dict.
    """
    if force_rebuild:
        print("🔄 Force rebuild — downloading full history...")
        fresh = download_batch(OVERFLOW_TICKERS)
        save_cache(fresh)
        return fresh

    cached = load_cache()

    # Find tickers that need downloading
    missing = [t for t in OVERFLOW_TICKERS if t not in cached]
    stale = []

    if cached:
        if expected_data_date is not None:
            cutoff = pd.Timestamp(expected_data_date)
            for t, df in cached.items():
                if t not in OVERFLOW_TICKERS:
                    continue
                if df.empty or df.index.max() < cutoff:
                    stale.append(t)
        else:
            today = pd.Timestamp.today().normalize()
            for t, df in cached.items():
                if t not in OVERFLOW_TICKERS:
                    continue
                if df.empty or (today - df.index.max()).days > 5:
                    stale.append(t)

    to_download_full = missing
    to_update = stale

    if to_download_full:
        print(f"📥 {len(to_download_full)} new tickers to download (full history)...")
        fresh = download_batch(to_download_full)
        cached.update(fresh)

    if to_update:
        # Only fetch recent data for stale tickers
        oldest_stale = min(cached[t].index.max() for t in to_update if t in cached and not cached[t].empty)
        fetch_from = (oldest_stale - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        print(f"🔄 {len(to_update)} stale tickers — fetching from {fetch_from}...")
        updates = download_batch(to_update, start_date=fetch_from)
        for t, new_df in updates.items():
            if t in cached and not cached[t].empty:
                # Append new rows, drop overlapping dates
                combined = pd.concat([cached[t], new_df])
                combined = combined[~combined.index.duplicated(keep='last')]
                cached[t] = combined.sort_index()
            else:
                cached[t] = new_df

    # Filter to only overflow tickers
    result = {t: cached[t] for t in OVERFLOW_TICKERS if t in cached and not cached[t].empty}

    if to_download_full or to_update:
        save_cache(result)
    else:
        print("✅ Cache is current — no downloads needed")

    return result


# ============================================================================
# APPEND-ONLY GOOGLE SHEETS (safe — never clears or overwrites)
# ============================================================================

def append_signals_to_gsheet(signals_list, sheet_name='Trade_Signals_Log'):
    """Append overflow signals to the Trade_Signals_Log sheet.

    APPEND-ONLY: uses worksheet.append_rows() so it never clears or
    overwrites existing data from the Actions scan.
    """
    if not signals_list:
        return

    gc = get_google_client()
    if not gc:
        return

    df = pd.DataFrame(signals_list)

    # Drop email-only fields (mirrors daily_scan.save_signals_to_gsheet — keep
    # _is_companion / _parent_strategy so companion rows are flagged in the log).
    cols_to_drop = [
        'Setup_Type', 'Setup_Timeframe', 'Setup_Thesis', 'Setup_Filters',
        'Exit_Primary', 'Exit_Stop', 'Exit_Target', 'Exit_Notes',
        'Live_Filters', 'Entry_Type',
        'Notional', 'Days_To_Exit', 'Use_Stop', 'Use_Target', 'Sizing_Variable',
        'Scan_Date',
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Round and tag
    for c in ['Entry', 'Stop', 'Target', 'ATR']:
        if c in df.columns:
            df[c] = df[c].astype(float).round(2)
    df['Date'] = df['Date'].astype(str)
    df['Scan_Timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        sh = gc.open(sheet_name)
        worksheet = sh.sheet1

        # Align columns to the sheet's actual header row — append_rows writes
        # positionally, so mismatched key order would shift every field.
        header_row = worksheet.row_values(1)
        if header_row:
            df = df.reindex(columns=header_row)

        rows = df.astype(str).values.tolist()
        worksheet.append_rows(rows, value_input_option='RAW')
        print(f"✅ Appended {len(rows)} rows to {sheet_name}")
    except Exception as e:
        print(f"❌ Google Sheet append error: {e}")


def append_orders_to_gsheet(signals_list, workbook_name='Trade_Signals_Log', tab_name='Overflow'):
    """Write overflow orders to a dedicated tab in the Trade_Signals_Log workbook.

    Emits the same 20-column schema as daily_scan.save_staging_orders so the
    order_staging.py consumer can concat the two tabs and run the same
    enrichment path.

    Always clears the tab at start of run — even if no signals are found —
    so stale rows from a prior run never linger and get re-submitted.
    """
    gc = get_google_client()
    if not gc:
        return

    strat_map = {s['id']: s for s in STRATEGY_BOOK}
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")

    headers = [
        "Scan_Date", "Symbol", "SecType", "Exchange", "Action", "Quantity",
        "Order_Type", "Limit_Price", "Offset_ATR_Mult", "TIF",
        "Frozen_ATR", "Signal_Close", "Time_Exit_Date", "Strategy_Ref",
        "Tgt_ATR_Mult", "Stop_ATR_Mult", "Use_Target", "Use_Stop",
        "Hold_Days", "Trade_Direction",
    ]

    rows = []
    for sig in (signals_list or []):
        entry_mode = sig.get('Entry_Type', '')

        # LOC companion: clean LOC row (mirrors daily_scan companion handling)
        if sig.get('_is_companion', False) is True:
            if "LOC" not in entry_mode:
                continue
            ib_action = "SELL" if "SHORT" in sig.get('Action', '') else "BUY"
            trade_dir = "Short" if "SHORT" in sig.get('Action', '') else "Long"
            rows.append([
                today_str,
                sig.get('Ticker', ''),
                "STK", "SMART",
                ib_action,
                sig.get('Shares', 0),
                "LOC",
                round(float(sig.get('Limit_Price', sig.get('Entry', 0))), 2),
                0.0,
                "DAY",
                round(float(sig.get('ATR', 0)), 2),
                round(float(sig.get('Entry', 0)), 2),
                str(sig.get('Time Exit', '')),
                sig.get('Strategy_Name', 'Companion'),
                0.0, 0.0, False, False,
                sig.get('Days_To_Exit', 0),
                trade_dir,
            ])
            continue

        strat = strat_map.get(sig.get('Strategy_ID'))
        if not strat:
            continue

        settings = strat['settings']
        execution = strat['execution']
        entry_mode = settings.get('entry_type', 'Signal Close')

        # Overflow intentionally does not stage MOC primaries
        if entry_mode == "Signal Close":
            continue

        # Defaults
        entry_instruction = "MKT"
        offset_atr = 0.0
        limit_price = 0.0
        tif_instruction = "DAY"

        # 1. ATR Limit entry (Close-anchored GTC/Persistent vs Open-anchored DAY)
        if "Limit" in entry_mode and "ATR" in entry_mode:
            is_persistent = "Persistent" in entry_mode or "GTC" in entry_mode
            if is_persistent:
                entry_instruction = "REL_CLOSE"
                tif_instruction = "GTC"
            else:
                entry_instruction = "REL_OPEN"
                tif_instruction = "DAY"
            # Order matters: 0.75 / 0.25 before 0.5 (substring-safe); 1 ATR last.
            if "0.75" in entry_mode:
                offset_atr = 0.75
            elif "0.25" in entry_mode:
                offset_atr = 0.25
            elif "0.5" in entry_mode:
                offset_atr = 0.5
            elif "1 ATR" in entry_mode:
                offset_atr = 1.0

        elif "LOC" in entry_mode:
            entry_instruction = "LOC"
            limit_price = sig.get('Limit_Price', sig.get('Entry', 0))
            tif_instruction = "DAY"

        # 2. Market on Open
        elif "T+1 Open" in entry_mode:
            entry_instruction = "MOO"
            tif_instruction = "OPG"

        # 3. Conditional Close
        elif "T+1 Close if < Signal Close" in entry_mode:
            entry_instruction = "LMT"
            limit_price = float(sig.get('Entry', 0)) - 0.01
            tif_instruction = "DAY"

        ib_action = "SELL" if "SHORT" in sig.get('Action', '') else "BUY"

        # Bracket metadata from strategy config (multipliers, not prices)
        use_target = execution.get('use_take_profit', False)
        use_stop = execution.get('use_stop_loss', False)
        tgt_atr_mult = execution.get('tgt_atr', 0.0)
        stop_atr_mult = execution.get('stop_atr', 0.0)
        hold_days = execution.get('hold_days', 0)
        trade_direction = settings.get('trade_direction', 'Long')

        rows.append([
            today_str,
            sig.get('Ticker', ''),
            "STK", "SMART",
            ib_action,
            sig.get('Shares', 0),
            entry_instruction,
            round(float(limit_price), 2),
            offset_atr,
            tif_instruction,
            round(float(sig.get('ATR', 0)), 2),
            round(float(sig.get('Entry', 0)), 2),
            str(sig.get('Time Exit', '')),
            strat['name'],
            tgt_atr_mult,
            stop_atr_mult,
            use_target,
            use_stop,
            hold_days,
            trade_direction,
        ])

    try:
        sh = gc.open(workbook_name)
        try:
            worksheet = sh.worksheet(tab_name)
        except Exception:
            worksheet = sh.add_worksheet(title=tab_name, rows=200, cols=len(headers))

        worksheet.clear()
        data = [headers] + [[str(v) for v in r] for r in rows]
        worksheet.update(range_name='A1', values=data)
        if rows:
            print(f"✅ Wrote {len(rows)} orders to {workbook_name}!{tab_name}")
        else:
            print(f"🧹 No overflow orders — '{tab_name}' cleared (headers only)")
    except Exception as e:
        print(f"❌ Order staging write error: {e}")


# ============================================================================
# EMAIL
# ============================================================================

def send_overflow_email(signals_list, error_tickers=None, scan_time_sec=0):
    """
    Sends an HTML email summary of overflow scan signals.
    Format mirrors daily_scan.send_email_summary exactly — gradient header,
    quick summary table, per-signal detail cards, companion annotation,
    grouped error breakdown, footer.
    """
    sender_email = os.environ.get("EMAIL_USER")
    sender_password = os.environ.get("EMAIL_PASS")
    receiver_email = "mckinleyslade@gmail.com"

    if not sender_email or not sender_password:
        print("⚠️ Email credentials (EMAIL_USER/EMAIL_PASS) not found. Skipping email.")
        return

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    email_signals = list(signals_list) if signals_list else []

    # Count unique LOGICAL signals (primary + companion on same ticker = 1 signal)
    _seen_logical = set()
    for s in email_signals:
        base_strat = s.get('_parent_strategy', s.get('Strategy_Name', s['Strategy_ID']))
        _seen_logical.add((s['Ticker'], base_strat))
    signal_count = len(_seen_logical)

    _primary_signals = [s for s in email_signals if not s.get('_is_companion', False)]
    _companion_map = {s['Ticker']: s for s in email_signals if s.get('_is_companion', False)}

    # Error tickers section — grouped by reason
    error_html = ""
    if not error_tickers:
        error_html = '<div style="margin-top: 20px; font-size: 12px; color: #888;">✅ All tickers successfully parsed</div>'
    else:
        from collections import defaultdict
        by_reason = defaultdict(list)
        for tk, reason in error_tickers:
            by_reason[reason].append(tk)

        error_rows = []
        for reason, tickers in sorted(by_reason.items()):
            ticker_str = ", ".join(sorted(tickers))
            error_rows.append(
                f"<tr><td style='padding: 4px 8px; color: #888; font-size: 12px; border-bottom: 1px solid #eee;'>{reason}</td>"
                f"<td style='padding: 4px 8px; color: #999; font-size: 11px; border-bottom: 1px solid #eee;'>{ticker_str}</td></tr>"
            )

        error_html = f"""
        <div style="margin-top: 20px; padding: 15px; background: #fafafa; border: 1px solid #eee; border-radius: 6px;">
            <div style="font-size: 12px; color: #888; margin-bottom: 8px;">⚠️ <strong>{len(error_tickers)} ticker(s) skipped</strong></div>
            <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                {"".join(error_rows)}
            </table>
        </div>
        """

    if not email_signals:
        subject = f"📉 Overflow Scan: NO SIGNALS ({date_str})"
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; background-color: #f5f5f5; padding: 20px;">
                <div style="max-width: 700px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px;">
                    <h2 style="color: #333; margin-top: 0;">Overflow Strategy Scan: {date_str}</h2>
                    <p style="color: #666;">The scan completed successfully.</p>
                    <p style="font-size: 18px; color: #888;"><strong>Result:</strong> No signals found matching criteria today.</p>
                    {error_html}
                </div>
            </body>
        </html>
        """
    else:
        subject = f"🚀 {signal_count} OVERFLOW SIGNAL{'S' if signal_count > 1 else ''} ({date_str})"

        signal_cards = []
        for sig in _primary_signals:
            _companion = _companion_map.get(sig['Ticker']) if not sig.get('_is_companion', False) else None

            header_color = "#2e7d32" if sig['Action'] == "BUY" else "#c62828"
            action_emoji = "📈" if sig['Action'] == "BUY" else "📉"

            # Key filters with live values
            live_filters = sig.get('Live_Filters', [])
            if live_filters:
                filters_html_parts = []
                for filter_desc, live_val, is_binary in live_filters:
                    if is_binary:
                        filters_html_parts.append(
                            f"<li style='margin: 4px 0; color: #444;'>{filter_desc} <span style='color: #2e7d32; font-weight: bold;'>{live_val}</span></li>"
                        )
                    else:
                        filters_html_parts.append(
                            f"<li style='margin: 4px 0; color: #444;'>{filter_desc}, <span style='color: #1565c0; font-weight: bold;'>{live_val}</span></li>"
                        )
                filters_html = "".join(filters_html_parts)
            else:
                static_filters = sig.get('Setup_Filters', [])
                if static_filters:
                    filters_html = "".join([f"<li style='margin: 4px 0; color: #444;'>{f}</li>" for f in static_filters])
                else:
                    filters_html = "<li style='color: #999;'>No filter details available</li>"

            # Exit section — suppress stop/target for time-exit-only strategies
            use_stop = sig.get('Use_Stop', True)
            use_target = sig.get('Use_Target', True)
            exit_primary = sig.get('Exit_Primary', '')
            if 'time stop' in exit_primary.lower() or 'time exit' in exit_primary.lower():
                if 'stop' not in exit_primary.lower().replace('time stop', ''):
                    use_stop = False
                if 'target' not in exit_primary.lower():
                    use_target = False

            exit_parts = []
            if use_stop:
                exit_parts.append(f"Stop: ${sig['Stop']:.2f}")
            if use_target:
                exit_parts.append(f"Target: ${sig['Target']:.2f}")

            if exit_parts:
                exit_prices_str = " | ".join(exit_parts)
                exit_prices_html = f"<div style='color: #666; font-size: 12px; margin-top: 5px;'>{exit_prices_str}</div>"
            else:
                exit_prices_html = ""

            exit_notes = sig.get('Exit_Notes', '')
            sizing_var = sig.get('Sizing_Variable', '')

            notes_parts = []
            if sizing_var:
                notes_parts.append(f"📊 {sizing_var}")
            if exit_notes:
                notes_parts.append(f"⚡ {exit_notes}")

            if _companion:
                comp_price = _companion.get('Limit_Price', 0)
                comp_shares = _companion.get('Shares', 0)
                notes_parts.append(f"📋 Also staged: LOC {comp_shares:,} shares @ >${comp_price:.2f} (Close + 0.5 ATR)")

            if notes_parts:
                notes_html = "<div style='font-size: 12px; color: #ff9800; margin-top: 8px;'>" + "<br>".join(notes_parts) + "</div>"
            else:
                notes_html = ""

            thesis = sig.get('Setup_Thesis', '')
            thesis_html = f"<div style='font-style: italic; color: #555; margin: 10px 0; padding: 10px; background: #f9f9f9; border-left: 3px solid #2196f3;'>{thesis}</div>" if thesis else ""

            entry_type = sig.get('Entry_Type', 'Signal Close')
            limit_price = sig.get('Limit_Price')
            is_open_based = "Open" in entry_type and "Limit" in entry_type
            if is_open_based:
                entry_display = entry_type
            elif "Signal Close" in entry_type or "T+1 Close" in entry_type:
                entry_display = f"{entry_type} @ ${sig['Entry']:.2f}"
            elif limit_price and "Close" in entry_type:
                entry_display = f"{entry_type} @ ${limit_price:.2f}"
            else:
                entry_display = f"{entry_type} @ ${sig['Entry']:.2f}"

            notional = sig.get('Notional', 0)
            days_to_exit = sig.get('Days_To_Exit', 0)

            card_html = f"""
            <div style="border: 1px solid #ddd; border-radius: 8px; margin-bottom: 20px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="background: {header_color}; color: white; padding: 15px;">
                    <div style="font-size: 18px; font-weight: bold;">
                        {action_emoji} {sig.get('Strategy_Name', sig['Strategy_ID'])}
                    </div>
                    <div style="font-size: 13px; opacity: 0.9; margin-top: 3px;">
                        {sig.get('Setup_Type', 'Custom')} | {sig.get('Setup_Timeframe', 'Swing')}
                    </div>
                </div>

                <div style="padding: 15px; background: #fafafa; border-bottom: 1px solid #eee;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 24px; font-weight: bold; color: #333;">{sig['Ticker']}</span>
                            <span style="color: #666; margin-left: 10px; font-size: 14px;">
                                {sig['Action']} {sig['Shares']:,} shares
                            </span>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 14px; color: #333;"><strong>${sig['Risk_Amt']:,.0f}</strong> risk</div>
                            <div style="font-size: 12px; color: #888;">${notional:,.0f} notional</div>
                        </div>
                    </div>
                    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px dashed #ddd; font-size: 13px; color: #555;">
                        <strong>Entry:</strong> {entry_display}
                        <span style="margin-left: 20px;"><strong>Exit:</strong> {sig['Time Exit']} ({days_to_exit}d)</span>
                    </div>
                </div>

                {thesis_html}

                <div style="padding: 15px;">
                    <div style="font-weight: bold; color: #333; margin-bottom: 8px; font-size: 14px;">
                        🎯 WHY IT FLAGGED:
                    </div>
                    <ul style="margin: 0; padding-left: 20px; font-size: 13px;">
                        {filters_html}
                    </ul>
                </div>

                <div style="padding: 15px; background: #f5f5f5; border-top: 1px solid #eee;">
                    <div style="font-weight: bold; color: #333; font-size: 13px;">
                        🚪 EXIT: {sig.get('Exit_Primary', f'{days_to_exit}-day time stop')}
                    </div>
                    {exit_prices_html}
                    {notes_html}
                </div>

                <div style="padding: 10px 15px; background: #333; color: #aaa; font-size: 11px;">
                    📊 {sig['Stats']}
                </div>
            </div>
            """
            signal_cards.append(card_html)

        all_cards_html = "".join(signal_cards)

        # Quick summary table
        df = pd.DataFrame(email_signals)
        summary_rows = []
        for _, row in df.iterrows():
            color = "#2e7d32" if row['Action'] == "BUY" else "#c62828"
            entry_short = row.get('Entry_Type_Short', 'MOC')
            risk_amt = row.get('Risk_Amt', 0)
            summary_rows.append(f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>{row['Ticker']}</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee; color: {color};">{row['Action']}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">{row['Shares']:,}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee; font-family: monospace;">{entry_short}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>${risk_amt:,.0f}</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee; color: #666; font-size: 12px;">{row.get('Strategy_Name', row['Strategy_ID'][:25])}</td>
                </tr>
            """)
        summary_table = f"""
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 25px; font-size: 13px;">
            <tr style="background: #f0f0f0;">
                <th style="padding: 10px; text-align: left;">Ticker</th>
                <th style="padding: 10px; text-align: left;">Action</th>
                <th style="padding: 10px; text-align: left;">Shares</th>
                <th style="padding: 10px; text-align: left;">Entry</th>
                <th style="padding: 10px; text-align: left;">$ Risk</th>
                <th style="padding: 10px; text-align: left;">Strategy</th>
            </tr>
            {"".join(summary_rows)}
        </table>
        """

        total_risk = sum(s.get('Risk_Amt', 0) for s in email_signals)
        long_notional = sum(s.get('Notional', 0) for s in email_signals if s['Action'] == 'BUY')
        short_notional = sum(s.get('Notional', 0) for s in email_signals if s['Action'] != 'BUY')
        net_notional = long_notional - short_notional
        long_count = len({(s['Ticker'], s.get('_parent_strategy', s.get('Strategy_Name'))) for s in email_signals if s['Action'] == 'BUY'})
        short_count = signal_count - long_count

        if net_notional >= 0:
            net_notional_str = f"+${net_notional:,.0f}"
        else:
            net_notional_str = f"-${abs(net_notional):,.0f}"

        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; background-color: #f5f5f5; padding: 20px;">
                <div style="max-width: 700px; margin: 0 auto;">
                    <div style="background: linear-gradient(135deg, #1a237e, #283593); color: white; padding: 25px; border-radius: 8px 8px 0 0; text-align: center;">
                        <h1 style="margin: 0; font-size: 24px;">Overflow Strategy Scan</h1>
                        <div style="font-size: 14px; opacity: 0.8; margin-top: 5px;">{date_str}</div>
                        <div style="font-size: 28px; margin-top: 10px;">🎯 {signal_count} Signal{'s' if signal_count > 1 else ''}</div>
                        <div style="font-size: 14px; margin-top: 8px; opacity: 0.9;">
                            {long_count} Long | {short_count} Short | ${total_risk:,.0f} Risk | {net_notional_str} Net Exposure
                        </div>
                    </div>

                    <div style="background: white; padding: 20px; border-bottom: 1px solid #ddd;">
                        <h3 style="margin-top: 0; color: #333;">⚡ Quick Summary</h3>
                        {summary_table}
                    </div>

                    <div style="background: white; padding: 20px; border-radius: 0 0 8px 8px;">
                        <h3 style="color: #333;">📋 Signal Details</h3>
                        {all_cards_html}
                    </div>

                    {error_html}

                    <div style="text-align: center; padding: 15px; color: #888; font-size: 12px;">
                        Check Google Sheet for staging details
                    </div>
                </div>
            </body>
        </html>
        """

    body = html_content
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print(f"📧 Email sent: {subject}")
    except Exception as e:
        print(f"⚠️ Email failed: {e}")


# ============================================================================
# MAIN SCAN
# ============================================================================

def run_overflow_scan(dry_run=False, force_rebuild=False):
    """Run the overflow scan against the extended universe."""
    start_time = time.time()
    print("=" * 60)
    print(f"OVERFLOW SCAN — {len(OVERFLOW_TICKERS)} tickers")
    print(f"Strategies: {[s['name'] for s in OVERFLOW_STRATEGIES]}")
    print("=" * 60)

    # 1. Load seasonal map
    sznl_map = load_seasonal_map()

    # 2. Compute expected data date first so cache-staleness knows the target
    eastern = pytz.timezone('America/New_York')
    now_eastern = datetime.datetime.now(eastern)
    current_date = now_eastern.date()
    market_open_time = now_eastern.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_time = now_eastern.replace(hour=16, minute=0, second=0, microsecond=0)
    # Intraday partial = market open AND today's session not yet closed.
    # Used to relax the LT Trend ST OS volume threshold from 1.25× to 1.0×.
    is_intraday_partial = (market_open_time <= now_eastern < market_close_time)

    if now_eastern < market_open_time:
        expected_data_date = (pd.Timestamp(current_date) - TRADING_DAY).date()
        print(f"🌅 Morning Run: data cutoff at {expected_data_date}")
    else:
        expected_data_date = current_date
        print(f"☀️ Day Run: allowing data through {expected_data_date}")
    if is_intraday_partial:
        print(f"🕐 Intraday partial-bar window — LT Trend ST OS will use vol_thresh 1.0× (else 1.25×)")

    # 3. Update cache — any ticker whose last bar is before expected_data_date is stale
    master_dict = update_cache({}, force_rebuild=force_rebuild, expected_data_date=expected_data_date)
    if not master_dict:
        print("❌ No data available — aborting")
        return

    validated_dict = {}
    for ticker, df in master_dict.items():
        if df is None or df.empty:
            continue
        last_row_date = df.index[-1].date()
        if now_eastern < market_open_time and last_row_date > expected_data_date:
            df = df[df.index.date <= expected_data_date]
        if not df.empty:
            validated_dict[ticker] = df
    master_dict = validated_dict
    print(f"✅ {len(master_dict)} tickers validated")

    # 4. Fragility sizing
    frag_mult = 1.0
    FRAG_THRESHOLD = 25
    FRAG_MAX_MULT = 1.25
    FRAG_MIN_MULT = 0.10
    frag_path = FRAG_CACHE if os.path.exists(FRAG_CACHE) else (FRAG_CACHE_TS if os.path.exists(FRAG_CACHE_TS) else None)
    if frag_path:
        try:
            frag_df = pd.read_parquet(frag_path)
            if '63d' in frag_df.columns:
                frag_series = frag_df['63d'].dropna().rolling(10, min_periods=1).mean()
                if not frag_series.empty:
                    frag_score = float(frag_series.iloc[-1])
                    if frag_score <= FRAG_THRESHOLD:
                        frag_mult = FRAG_MAX_MULT - (frag_score / FRAG_THRESHOLD) * (FRAG_MAX_MULT - 1.0) if FRAG_THRESHOLD > 0 else FRAG_MAX_MULT
                    else:
                        frag_mult = max(FRAG_MIN_MULT, 1.0 - ((frag_score - FRAG_THRESHOLD) / (100 - FRAG_THRESHOLD)) * (1 - FRAG_MIN_MULT))
                    print(f"🛡️ Fragility: {frag_score:.1f} → {frag_mult:.2f}x")
        except Exception as e:
            print(f"⚠️ Fragility load failed: {e}")

    # 5. VIX series
    vix_series = None
    # Try to get VIX from main Actions data or download fresh
    try:
        vix_raw = yf.download("^VIX", start="2020-01-01", progress=False)
        if not vix_raw.empty:
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_raw.columns = vix_raw.columns.get_level_values(0)
            vix_raw.columns = [c.capitalize() for c in vix_raw.columns]
            vix_raw.index = vix_raw.index.normalize()
            if vix_raw.index.tz is not None:
                vix_raw.index = vix_raw.index.tz_localize(None)
            vix_series = vix_raw['Close']
    except Exception:
        pass

    # 6. Build cross-sectional rank matrices
    xsec_rank_matrices = None
    xsec_windows_needed = set()
    for strat in OVERFLOW_STRATEGIES:
        s = strat['settings']
        if s.get('use_xsec_filter', False):
            for xf in s.get('xsec_filters', []):
                xsec_windows_needed.add(xf['window'])
        for group in s.get('or_filter_groups', []):
            for cond in group:
                if cond.get('type') == 'xsec':
                    xsec_windows_needed.add(cond['window'])
    if xsec_windows_needed:
        print(f"📊 Computing cross-sectional ranks (windows: {sorted(xsec_windows_needed)})...")
        RANK_MIN_PERIODS = 252
        rank_dict = {}
        for ticker, df in master_dict.items():
            if df is None or 'Close' not in df.columns or len(df) < 50:
                continue
            for w in xsec_windows_needed:
                ret = df['Close'].pct_change(w)
                temporal_pctile = ret.expanding(min_periods=RANK_MIN_PERIODS).rank(pct=True) * 100.0
                rank_dict.setdefault(w, {})[ticker] = temporal_pctile
        xsec_rank_matrices = {}
        for w in xsec_windows_needed:
            if rank_dict.get(w):
                mat = pd.DataFrame(rank_dict[w])
                xsec_rank_matrices[w] = mat.rank(axis=1, pct=True) * 100.0
        print(f"   ✓ {len(next(iter(xsec_rank_matrices.values())).columns)} tickers ranked")

    # 7. Run strategies
    all_signals = []
    error_tickers = []  # (ticker, reason) tuples

    # OLV cooldown — shared across main + overflow via Trade_Signals_Log
    olv_cooldown = load_olv_cooldown(lookback_trading_days=OLV_COOLDOWN_DAYS + 5)
    olv_cutoff = (pd.Timestamp(datetime.date.today()) - TRADING_DAY * OLV_COOLDOWN_DAYS).date()
    if olv_cooldown:
        print(f"🛑 OLV cooldown: {len(olv_cooldown)} tickers with signals since {olv_cutoff}")

    # Ladder sizing — count currently-held filled positions per (ticker, strategy).
    ladder_strats = {s['name'] for s in OVERFLOW_STRATEGIES if s['execution'].get('ladder_multipliers')}
    ladder_counts = load_open_position_counts(ladder_strats)
    if ladder_counts:
        print(f"📈 Ladder: {len(ladder_counts)} open positions tracked")

    # ATR seasonal ranks — load once if any overflow strategy uses them
    _uses_atr_sznl = (
        any(s['settings'].get('atr_sznl_filters') for s in OVERFLOW_STRATEGIES)
        or any(s['name'] == "Overbot Vol Spike" for s in OVERFLOW_STRATEGIES)  # uses atr_sznl_5d for the 1.5x sizer
    )
    atr_sznl_map = load_atr_seasonal_map() if _uses_atr_sznl else {}
    if _uses_atr_sznl:
        if atr_sznl_map:
            print(f"📊 Loaded ATR seasonal ranks: {len(atr_sznl_map)} tickers")
        else:
            print(f"⚠️ atr_seasonal_ranks.parquet not found — atr_sznl_filters will match nothing")

    for strat in OVERFLOW_STRATEGIES:
        strat_name = strat['name']
        # Only scan overflow tickers
        overflow_in_strat = [t for t in strat['universe_tickers'] if t in master_dict and t in OVERFLOW_TICKERS]
        if not overflow_in_strat:
            continue

        print(f"\n▶ {strat_name} ({len(overflow_in_strat)} overflow tickers)...")

        # Market series
        mkt_ticker = strat['settings'].get('market_ticker', 'SPY')
        mkt_df = master_dict.get(mkt_ticker)
        if mkt_df is None:
            # Download SPY/market ticker on the fly
            try:
                mkt_raw = yf.download(mkt_ticker.replace('.', '-'), start="2020-01-01", progress=False)
                if not mkt_raw.empty:
                    if isinstance(mkt_raw.columns, pd.MultiIndex):
                        mkt_raw.columns = mkt_raw.columns.get_level_values(0)
                    mkt_raw.columns = [c.capitalize() for c in mkt_raw.columns]
                    mkt_raw.index = mkt_raw.index.normalize()
                    if mkt_raw.index.tz is not None:
                        mkt_raw.index = mkt_raw.index.tz_localize(None)
                    mkt_df = mkt_raw
            except Exception:
                pass

        market_series = None
        if mkt_df is not None:
            temp_mkt = mkt_df.copy()
            temp_mkt['SMA200'] = temp_mkt['Close'].rolling(200).mean()
            market_series = temp_mkt['Close'] > temp_mkt['SMA200']

        # Ref ticker ranks
        ref_ticker_ranks = None
        ref_settings = strat['settings']
        if ref_settings.get('use_ref_ticker_filter', False) and ref_settings.get('ref_filters'):
            ref_ticker_key = ref_settings.get('ref_ticker', 'IWM').replace('.', '-')
            ref_df = master_dict.get(ref_ticker_key)
            if ref_df is not None and len(ref_df) > 250:
                ref_calc = calculate_indicators(ref_df.copy(), sznl_map, ref_ticker_key, market_series, vix_series)
                ref_ticker_ranks = {}
                for rf in ref_settings['ref_filters']:
                    col = f'rank_ret_{rf["window"]}d'
                    if col in ref_calc.columns:
                        ref_ticker_ranks[rf['window']] = ref_calc[col]

        signals = []
        for ticker in overflow_in_strat:
            t_clean = ticker.replace('.', '-')
            df = master_dict.get(t_clean)
            if df is None:
                error_tickers.append((t_clean, "No data returned"))
                continue
            if len(df) < 250:
                error_tickers.append((t_clean, f"Insufficient history ({len(df)} bars)"))
                continue

            try:
                calc_df = calculate_indicators(
                    df.copy(), sznl_map, t_clean, market_series, vix_series,
                    ref_ticker_ranks=ref_ticker_ranks,
                    xsec_rank_matrices=xsec_rank_matrices
                )

                # Merge ATR seasonal ranks (if strategy needs them)
                if atr_sznl_map and t_clean in atr_sznl_map:
                    _atr_ranks = atr_sznl_map[t_clean]
                    _dates = calc_df.index.normalize()
                    for _col in ATR_SZNL_COLS:
                        if _col in _atr_ranks.columns:
                            calc_df[_col] = _atr_ranks[_col].reindex(_dates).values

                # LT Trend ST OS intraday volume relaxation — see daily_scan.py
                # for the rationale. Drops vol_thresh from config 1.25× to 1.0×
                # while market is open through 4 PM ET.
                _eff_settings = strat['settings']
                if (is_intraday_partial
                        and strat['name'] == "LT Trend ST OS"
                        and _eff_settings.get('use_vol')):
                    _eff_settings = dict(_eff_settings)
                    _eff_settings['vol_thresh'] = 1.0

                if check_signal(calc_df, _eff_settings, sznl_map, ticker=t_clean):
                    last_row = calc_df.iloc[-1]

                    # Cooldown gate: OLV suppresses re-fires within 20 trading days
                    if strat_name == OLV_STRATEGY_NAME:
                        last_sig = olv_cooldown.get(t_clean)
                        if last_sig is not None and last_sig >= olv_cutoff:
                            print(f"   ⏭ OLV cooldown: {t_clean} (last signal {last_sig})")
                            continue

                    # Entry confirmation
                    entry_conf_bps = strat['settings'].get('entry_conf_bps', 0)
                    entry_mode = strat['settings'].get('entry_type', 'Signal Close')

                    if entry_mode == 'Signal Close' and entry_conf_bps > 0:
                        threshold = last_row['Open'] * (1 + entry_conf_bps / 10000.0)
                        if last_row['High'] < threshold:
                            continue

                    atr = last_row['ATR']
                    base_risk = strat['execution']['risk_per_trade']
                    risk = base_risk
                    sizing_note = "Standard (1.0x)"

                    # Overbot Vol Spike: 1.5x when 5d ATR seasonal rank is in the
                    # bottom quartile — weak short-horizon seasonal reinforces the
                    # fade thesis. Mirrors the rule in daily_scan.py.
                    if strat['name'] == "Overbot Vol Spike":
                        _atr_sznl_5d = last_row.get('atr_sznl_5d', None)
                        if _atr_sznl_5d is not None and pd.notna(_atr_sznl_5d) and _atr_sznl_5d < 25:
                            risk = risk * 1.5
                            sizing_note = f"ATR Sznl 5d {_atr_sznl_5d:.0f} < 25 → 1.5x"

                    # Overbot Vol Spike: 0.5x when 126D or 252D rank > 65 (leader
                    # penalty — fade thesis weaker against established strength).
                    if strat['name'] == "Overbot Vol Spike":
                        _r126 = last_row.get('rank_ret_126d', None)
                        _r252 = last_row.get('rank_ret_252d', None)
                        _is_leader_126 = _r126 is not None and pd.notna(_r126) and _r126 > 65
                        _is_leader_252 = _r252 is not None and pd.notna(_r252) and _r252 > 65
                        if _is_leader_126 or _is_leader_252:
                            risk = risk * 0.5
                            _which = []
                            if _is_leader_126: _which.append(f"126D={_r126:.0f}")
                            if _is_leader_252: _which.append(f"252D={_r252:.0f}")
                            sizing_note = f"{sizing_note} | OVS leader ({', '.join(_which)}>65) → 0.5x"

                    # OVS: terminal flat 5 bps when BOTH leader (126D/252D > 65)
                    # AND strong 5D ATR seasonal (> 65). Weakest fade setup — tiny
                    # lottery-ticket sizing. Overrides prior OVS multipliers.
                    if strat['name'] == "Overbot Vol Spike":
                        _atr_5d_t = last_row.get('atr_sznl_5d', None)
                        _r126_t = last_row.get('rank_ret_126d', None)
                        _r252_t = last_row.get('rank_ret_252d', None)
                        _is_leader_t = ((pd.notna(_r126_t) and _r126_t > 65)
                                        or (pd.notna(_r252_t) and _r252_t > 65))
                        _is_high_sznl_t = pd.notna(_atr_5d_t) and _atr_5d_t > 65
                        if _is_leader_t and _is_high_sznl_t:
                            risk = ACCOUNT_VALUE * 5 / 10000.0
                            sizing_note = f"OVS leader+high-sznl (5D ATR={_atr_5d_t:.0f}>65) → flat 5 bps (${risk:.0f})"

                    # Fragility adjustment
                    if frag_mult != 1.0:
                        risk = risk * frag_mult
                        sizing_note += f" | Frag {frag_mult:.2f}x"

                    # Ladder sizing — scale up on repeat signals when prior
                    # positions are still open. Companion LOC stays flat.
                    ladder_mults = strat['execution'].get('ladder_multipliers')
                    companion_risk_override = None
                    if ladder_mults:
                        open_count = ladder_counts.get((t_clean, strat['name']), 0)
                        rung_idx = min(open_count, len(ladder_mults) - 1)
                        ladder_mult = ladder_mults[rung_idx]
                        loc_mult = strat['execution'].get('loc_companion_multiplier')
                        if loc_mult is not None:
                            companion_risk_override = risk * loc_mult
                        risk = risk * ladder_mult
                        sizing_note += f" | Ladder rung {rung_idx + 1} ({ladder_mult:.2f}x, {open_count} open)"

                    # Calculate prices & shares
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
                    hold_days = strat['execution']['hold_days']

                    # Entry date & exit date
                    if entry_mode == "Signal Close":
                        effective_entry_date = last_row.name
                    else:
                        effective_entry_date = last_row.name + TRADING_DAY
                    exit_date = (effective_entry_date + (TRADING_DAY * hold_days)).date()

                    # Limit price
                    limit_price = 0.0
                    if "0.75 ATR" in entry_mode:
                        limit_price = (entry + 0.75 * atr) if direction == 'Short' else (entry - 0.75 * atr)
                    elif "0.25 ATR" in entry_mode:
                        limit_price = (entry + 0.25 * atr) if direction == 'Short' else (entry - 0.25 * atr)
                    elif "0.5 ATR" in entry_mode:
                        limit_price = (entry + 0.5 * atr) if direction == 'Short' else (entry - 0.5 * atr)
                    elif "1 ATR" in entry_mode:
                        limit_price = (entry + 1.0 * atr) if direction == 'Short' else (entry - 1.0 * atr)

                    risk_bps = strat['execution'].get('risk_bps', 0)
                    sizing_with_risk = f"{sizing_note} | Risk: {risk_bps}bps (${risk:.0f})"

                    setup_block = strat.get('setup', {})
                    exit_block = strat.get('exit_summary', {})

                    signal_dict = {
                        "Date": last_row.name.strftime('%Y-%m-%d'),
                        "Ticker": t_clean,
                        "Strategy_ID": strat['id'],
                        "Strategy_Name": strat_name,
                        "Action": action,
                        "Scan_Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Shares": shares,
                        "Risk_Amt": risk,
                        "Sizing_Notes": sizing_with_risk,
                        "Stats": f"WR: {strat['stats']['win_rate']} | PF: {strat['stats']['profit_factor']}",
                        "Entry": round(entry, 2),
                        "Stop": round(stop_price, 2),
                        "Target": round(tgt_price, 2),
                        "Time Exit": str(exit_date),
                        "ATR": round(atr, 2),
                        "Entry_Type": entry_mode,
                        "Entry_Type_Short": get_entry_type_short(entry_mode, limit_price),
                        "Limit_Price": round(limit_price, 2),
                        "Notional": shares * entry,
                        "Days_To_Exit": hold_days,
                        "Use_Stop": strat['execution'].get('use_stop_loss', False),
                        "Use_Target": strat['execution'].get('use_take_profit', False),
                        "Setup_Type": setup_block.get('type', 'Custom'),
                        "Setup_Timeframe": setup_block.get('timeframe', 'Swing'),
                        "Setup_Thesis": setup_block.get('thesis', ''),
                        "Setup_Filters": setup_block.get('key_filters', []),
                        "Live_Filters": build_live_filters(strat, last_row, calc_df),
                        "Exit_Primary": exit_block.get('primary_exit', ''),
                        "Exit_Stop": exit_block.get('stop_logic', ''),
                        "Exit_Target": exit_block.get('target_logic', ''),
                        "Exit_Notes": exit_block.get('notes', ''),
                        "Sizing_Variable": get_sizing_variable(strat_name, last_row)
                    }

                    signals.append(signal_dict)

                    if strat_name == OLV_STRATEGY_NAME:
                        companion = generate_oversold_lv_companion(signal_dict, strat, last_row, override_risk=companion_risk_override)
                        if companion:
                            signals.append(companion)

            except Exception as e:
                error_tickers.append((t_clean, str(e)[:80]))
                continue

        if signals:
            all_signals.extend(signals)
            print(f"   → {len(signals)} signals")

    # Global aggregate daily risk cap across ALL strategies.
    # If total Risk_Amt across today's overflow signals exceeds DAILY_RISK_CAP_BPS,
    # scale every signal's Shares / Risk_Amt / Notional down proportionally.
    if all_signals and ACCOUNT_VALUE > 0:
        cap_dollars = ACCOUNT_VALUE * DAILY_RISK_CAP_BPS / 10000.0
        total_risk = sum(float(s.get('Risk_Amt', 0) or 0) for s in all_signals)
        if total_risk > cap_dollars > 0:
            scale = cap_dollars / total_risk
            for s in all_signals:
                s['Shares'] = int(s.get('Shares', 0) * scale)
                s['Risk_Amt'] = float(s.get('Risk_Amt', 0) or 0) * scale
                entry_px = s.get('Entry') or 0
                s['Notional'] = s['Shares'] * float(entry_px)
                s['Sizing_Notes'] = f"{s.get('Sizing_Notes', '')} | Daily cap {DAILY_RISK_CAP_BPS}bps: {scale:.2f}x"
            print(f"\n>>> Global risk cap hit: {len(all_signals)} signals scaled by {scale:.2f}x "
                  f"(${total_risk:,.0f} -> ${cap_dollars:,.0f})\n")

    # Dedup error tickers across strategies
    seen_errors = set()
    unique_errors = []
    for tk, reason in error_tickers:
        if (tk, reason) not in seen_errors:
            seen_errors.add((tk, reason))
            unique_errors.append((tk, reason))

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"SCAN COMPLETE: {len(all_signals)} signals, {len(unique_errors)} errors, {elapsed:.0f}s")
    print("=" * 60)

    if dry_run:
        print("🏜️ DRY RUN — skipping email and Google Sheets")
        for sig in all_signals:
            print(f"   {sig['Ticker']:6s} | {sig['Action']:10s} | {sig['Strategy_Name']}")
        return

    # 8. Save & notify
    #    - Trade_Signals_Log is append-only (no clear)
    #    - Overflow staging tab is always cleared+rewritten, even on zero-signal
    #      days, so stale rows can't linger and get re-submitted by order_staging.
    if all_signals:
        try:
            append_signals_to_gsheet(all_signals)
        except Exception as e:
            print(f"⚠️ Google Sheets failed: {e}")

    try:
        append_orders_to_gsheet(all_signals)
    except Exception as e:
        print(f"⚠️ Order staging failed: {e}")

    send_overflow_email(all_signals, error_tickers=unique_errors)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local overflow scanner")
    parser.add_argument("--rebuild", action="store_true", help="Force full cache rebuild")
    parser.add_argument("--dry-run", action="store_true", help="Scan but don't email/sheet")
    args = parser.parse_args()

    run_overflow_scan(dry_run=args.dry_run, force_rebuild=args.rebuild)
