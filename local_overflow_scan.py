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
from strategy_config import (
    STRATEGY_BOOK, CSV_UNIVERSE, LIQUID_PLUS_COMMODITIES,
    ACCOUNT_VALUE, build_strategy_book
)
from daily_scan import (
    check_signal, load_seasonal_map, generate_vol_spike_companion,
    get_entry_type_short, get_sizing_variable, build_live_filters,
    save_signals_to_gsheet, save_staging_orders, save_moc_orders
)

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


def update_cache(data_dict, force_rebuild=False):
    """Load cache, download missing/stale data, merge and save.

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
    today = pd.Timestamp.today().normalize()

    if cached:
        # Check staleness: if last cached date is >3 trading days old, update
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
# EMAIL
# ============================================================================

def send_overflow_email(signals_list, error_count=0, scan_time_sec=0):
    """Send a summary email for overflow scan results."""
    sender_email = os.environ.get("EMAIL_USER")
    sender_password = os.environ.get("EMAIL_PASS")
    receiver_email = "mckinleyslade@gmail.com"

    if not sender_email or not sender_password:
        print("⚠️ EMAIL_USER / EMAIL_PASS not set — skipping email")
        return

    eastern = pytz.timezone('America/New_York')
    now = datetime.datetime.now(eastern)
    date_str = now.strftime("%Y-%m-%d %H:%M ET")

    if signals_list:
        subject = f"🔍 {len(signals_list)} OVERFLOW SIGNAL(S) ({date_str})"
        rows = []
        for sig in signals_list:
            rows.append(f"""
            <tr>
                <td style="padding:6px;border:1px solid #444;">{sig['Ticker']}</td>
                <td style="padding:6px;border:1px solid #444;">{sig['Action']}</td>
                <td style="padding:6px;border:1px solid #444;">{sig.get('Shares', 0):,}</td>
                <td style="padding:6px;border:1px solid #444;">{sig.get('Entry_Type_Short', sig.get('Entry_Type', ''))}</td>
                <td style="padding:6px;border:1px solid #444;">${sig.get('Risk_Amt', 0):,.0f}</td>
                <td style="padding:6px;border:1px solid #444;">{sig['Strategy_Name']}</td>
            </tr>""")

        body = f"""
        <div style="font-family:monospace;background:#0e1117;color:#e0e0e0;padding:20px;">
            <h2>Overflow Scan: {len(signals_list)} Signal(s)</h2>
            <p style="color:#888;">Extended universe ({len(OVERFLOW_TICKERS)} tickers) · Scan time: {scan_time_sec:.0f}s</p>
            <table style="border-collapse:collapse;width:100%;margin-top:10px;">
                <tr style="background:#1a1a2e;">
                    <th style="padding:6px;border:1px solid #444;">Ticker</th>
                    <th style="padding:6px;border:1px solid #444;">Action</th>
                    <th style="padding:6px;border:1px solid #444;">Shares</th>
                    <th style="padding:6px;border:1px solid #444;">Entry</th>
                    <th style="padding:6px;border:1px solid #444;">Risk</th>
                    <th style="padding:6px;border:1px solid #444;">Strategy</th>
                </tr>
                {"".join(rows)}
            </table>
            {f'<p style="color:#ff9800;margin-top:10px;">⚠️ {error_count} tickers had errors</p>' if error_count else ''}
        </div>
        """
    else:
        subject = f"📉 Overflow Scan: NO SIGNALS ({date_str})"
        body = f"""
        <div style="font-family:monospace;background:#0e1117;color:#e0e0e0;padding:20px;">
            <h2>Overflow Scan: No Signals</h2>
            <p style="color:#888;">Scanned {len(OVERFLOW_TICKERS)} tickers · {scan_time_sec:.0f}s</p>
        </div>
        """

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
    print(f"Strategies: {[s['name'] for s in STRATEGY_BOOK]}")
    print("=" * 60)

    # 1. Load seasonal map
    sznl_map = load_seasonal_map()

    # 2. Update cache and get data
    master_dict = update_cache({}, force_rebuild=force_rebuild)
    if not master_dict:
        print("❌ No data available — aborting")
        return

    # 3. Date validation (same as daily_scan)
    eastern = pytz.timezone('America/New_York')
    now_eastern = datetime.datetime.now(eastern)
    current_date = now_eastern.date()
    market_open_time = now_eastern.replace(hour=9, minute=30, second=0, microsecond=0)

    if now_eastern < market_open_time:
        expected_data_date = (pd.Timestamp(current_date) - TRADING_DAY).date()
        print(f"🌅 Morning Run: data cutoff at {expected_data_date}")
    else:
        expected_data_date = current_date
        print(f"☀️ Day Run: allowing data through {expected_data_date}")

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
    for strat in STRATEGY_BOOK:
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
    error_count = 0

    for strat in STRATEGY_BOOK:
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
            if df is None or len(df) < 250:
                error_count += 1
                continue

            try:
                calc_df = calculate_indicators(
                    df.copy(), sznl_map, t_clean, market_series, vix_series,
                    ref_ticker_ranks=ref_ticker_ranks,
                    xsec_rank_matrices=xsec_rank_matrices
                )

                if check_signal(calc_df, strat['settings'], sznl_map, ticker=t_clean):
                    last_row = calc_df.iloc[-1]

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

                    # Fragility adjustment
                    if frag_mult != 1.0:
                        risk = risk * frag_mult
                        sizing_note += f" | Frag {frag_mult:.2f}x"

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
                    if "Signal Close" in entry_mode:
                        effective_entry_date = last_row.name
                    else:
                        effective_entry_date = last_row.name + TRADING_DAY
                    exit_date = (effective_entry_date + (TRADING_DAY * hold_days)).date()

                    # Limit price
                    limit_price = 0.0
                    if "0.5 ATR" in entry_mode:
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

                    # Vol Spike companion
                    if strat_name == "Overbot Vol Spike":
                        signals.append(signal_dict)
                        companion = generate_vol_spike_companion(signal_dict, strat, last_row)
                        if companion:
                            signals.append(companion)
                    else:
                        signals.append(signal_dict)

            except Exception as e:
                error_count += 1
                continue

        if signals:
            all_signals.extend(signals)
            print(f"   → {len(signals)} signals")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"SCAN COMPLETE: {len(all_signals)} signals, {error_count} errors, {elapsed:.0f}s")
    print("=" * 60)

    if dry_run:
        print("🏜️ DRY RUN — skipping email and Google Sheets")
        for sig in all_signals:
            print(f"   {sig['Ticker']:6s} | {sig['Action']:10s} | {sig['Strategy_Name']}")
        return

    # 8. Save & notify
    if all_signals:
        df_sig = pd.DataFrame(all_signals)
        try:
            save_signals_to_gsheet(df_sig)
            print("✅ Signals logged to Google Sheets")
        except Exception as e:
            print(f"⚠️ Google Sheets failed: {e}")

        try:
            save_moc_orders(all_signals, STRATEGY_BOOK, sheet_name='moc_orders')
            save_staging_orders(all_signals, STRATEGY_BOOK, sheet_name='Order_Staging')
            print("✅ Orders staged")
        except Exception as e:
            print(f"⚠️ Order staging failed: {e}")

    send_overflow_email(all_signals, error_count, elapsed)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local overflow scanner")
    parser.add_argument("--rebuild", action="store_true", help="Force full cache rebuild")
    parser.add_argument("--dry-run", action="store_true", help="Scan but don't email/sheet")
    args = parser.parse_args()

    run_overflow_scan(dry_run=args.dry_run, force_rebuild=args.rebuild)
