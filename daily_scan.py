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
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from indicators import calculate_indicators, get_sznl_val_series
from filters import (
    ETF_ATR_EXEMPT,
    FRAG_STALE_TD,
    check_signal_live,
    get_fragility_df_cached as _get_fragility_df_cached,
    live_signal_mask,
    recency_prior_from_mask,
)
from earnings_filter import load_earnings_dates_map, in_blackout, signed_offset
from exposure_leg import compute_exposure_targets, save_state
import pc_fear

# NYSE trading-day offset (2026-07-16: was USFederalHolidayCalendar, which
# marks Columbus/Veterans Day while NYSE trades — the morning after each,
# Friday's already-traded signals were re-staged — and misses Good Friday).
# order_staging.py back-computes entry expiry with an IDENTICAL calendar;
# see trading_calendar.py + tests/test_trading_calendar.py.
from trading_calendar import TRADING_DAY

# -----------------------------------------------------------------------------
# IMPORT STRATEGY BOOK
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from strategy_config import (
        STRATEGY_BOOK, ACCOUNT_VALUE, SPOT_TO_TRADEABLE,
        CSV_UNIVERSE, LIQUID_PLUS_COMMODITIES,
        CROSS_STRATEGY_OVERLAP_OVERRIDES,
        GLOBAL_RISK_MULTIPLIER,
        same_day_derate_mult,
    )
except ImportError:
    print("[ERROR] Could not find strategy_config.py in the root directory.")
    STRATEGY_BOOK = []
    ACCOUNT_VALUE = 0
    SPOT_TO_TRADEABLE = {}
    CSV_UNIVERSE = []
    LIQUID_PLUS_COMMODITIES = []
    GLOBAL_RISK_MULTIPLIER = 1.0
    CROSS_STRATEGY_OVERLAP_OVERRIDES = []

    def same_day_derate_mult(execution, n_signals):
        return 1.0

# Dynamic overflow universe (Layer C). When data/overflow_universe.parquet is
# absent, the loaders return the caller-supplied fallback / {} so the scan
# behaves exactly as before the universe was bootstrapped.
try:
    from overflow_universe import (
        load_overflow_universe, load_overflow_universe_full, load_overflow_meta,
        filter_by_addv, adv_share_cap, ADV_PARTICIPATION_CAP,
    )
except ImportError:
    print("[WARN] overflow_universe.py not importable — using static overflow tier.")

    def load_overflow_universe(fallback=None, **_kw):
        return list(fallback) if fallback is not None else []

    def load_overflow_universe_full(static_fallback=None, **_kw):
        return sorted(set(static_fallback or []))

    def load_overflow_meta(**_kw):
        return {}

    def filter_by_addv(tickers, strategy_name, meta):
        return list(tickers)

    def adv_share_cap(addv_63d, entry_price, participation=0.02):
        return None

    ADV_PARTICIPATION_CAP = 0.02

# Master prices parquet — full CSV_UNIVERSE history, built/maintained by
# scripts/update_master_prices.py. Used for the overflow scope to avoid
# 870+ ticker yfinance pulls.
MASTER_PRICES_PATH = os.path.join(current_dir, "data", "master_prices.parquet")

# Strategies the overflow scope expands to CSV_UNIVERSE − LIQUID_PLUS_COMMODITIES.
# Mirrors local_overflow_scan.OVERFLOW_STRATEGIES + daily_portfolio_report.OVERFLOW_ELIGIBLE.
OVERFLOW_ELIGIBLE_STRATEGIES = {
    "Overbot Vol Spike",
    "LT Trend ST OS",
    "Oversold Low Volume",
    "St OS Sznl",
    "52wh Breakout",
    "ATR Extended Gap Up",  # added 2026-06-09; native 60 bps on overflow (no override)
}

# Per-strategy bps overrides for the overflow tier. OVS uses path-1 nominal
# (40 bps) for both universes — see strategy_config.py + order_staging.py.
OVERFLOW_RISK_OVERRIDES = {
    "Oversold Low Volume": 25,  # vs liquid 35 (signal-recency ladder applies on both tiers)
}

# ATR-normalized seasonal ranks (built by build_atr_seasonal_ranks.py)
ATR_SZNL_PATH = os.path.join(current_dir, "atr_seasonal_ranks.parquet")
ATR_SZNL_WINDOWS = [5, 10, 21, 63, 126, 252]
ATR_SZNL_COLS = [f"atr_sznl_{w}d" for w in ATR_SZNL_WINDOWS]


def build_effective_strategy_book(scope='liquid', moc_only=False):
    """Build the strategy list daily_scan iterates against, given a scope.

    scope='liquid' (default, matches today's GHA behavior): every entry in
        STRATEGY_BOOK as-is, scanning each strategy's native universe_tickers
        (typically LIQUID_PLUS_COMMODITIES).

    scope='overflow': only the 6 overflow-eligible strategies, with their
        universe swapped to CSV_UNIVERSE − LIQUID_PLUS_COMMODITIES and
        per-strategy risk_bps overrides applied (e.g. OLV 35 → 25).

    scope='all': liquid pass + overflow pass concatenated. The same strategy
        name may appear twice (once per tier), each scanning its own ticker
        set. Signals are tagged with `_scan_source` so downstream code can
        stamp Scan_Source on the staged row.

    moc_only=True restricts to strategies with entry_type='Signal Close'.
        Overflow tier is auto-excluded (it doesn't stage MOC by convention,
        see save_moc_orders). Used by intraday GHA runs to skip the bulk
        of the work — limit-entry strategies don't change between morning
        and post-close so re-running them intraday is wasted compute.

    Each entry gets `_scan_source` set to 'Liquid' or 'Overflow'.
    """
    import copy as _copy
    liquid_set = set(LIQUID_PLUS_COMMODITIES)
    # Comprehensive overflow universe = dynamic screen ∪ legacy static tier
    # (CSV_UNIVERSE − LIQUID_PLUS_COMMODITIES). respect_active=True keeps live on the
    # static tier alone until OVERFLOW_UNIVERSE_ACTIVE is set; once active, the full
    # union is scanned. Falls back to the static tier when the parquet is absent.
    _static_overflow = sorted(set(CSV_UNIVERSE) - liquid_set)
    overflow_tickers = load_overflow_universe_full(static_fallback=_static_overflow, respect_active=True)
    overflow_meta = load_overflow_meta()  # {} when gate OFF / no parquet → ADDV gate is a no-op

    def _is_moc(s):
        return str(s['settings'].get('entry_type', '')).strip() == 'Signal Close'

    book = []
    if scope in ('liquid', 'all'):
        for s in STRATEGY_BOOK:
            if moc_only and not _is_moc(s):
                continue
            ws = _copy.deepcopy(s)
            ws['_scan_source'] = 'Liquid'
            book.append(ws)

    if scope in ('overflow', 'all') and not moc_only:
        for s in STRATEGY_BOOK:
            if s['name'] not in OVERFLOW_ELIGIBLE_STRATEGIES:
                continue
            ws = _copy.deepcopy(s)
            # Per-strategy ADDV floor (R-T3): OVS shorts need deeper liquidity
            # than patient GTC-limit strategies. No-op when overflow_meta is {}.
            ws['universe_tickers'] = filter_by_addv(overflow_tickers, s['name'], overflow_meta)
            if s['name'] in OVERFLOW_RISK_OVERRIDES:
                new_bps = OVERFLOW_RISK_OVERRIDES[s['name']] * GLOBAL_RISK_MULTIPLIER
                ws['execution']['risk_bps'] = new_bps
                ws['execution']['risk_per_trade'] = ACCOUNT_VALUE * new_bps / 10000
            ws['_scan_source'] = 'Overflow'
            book.append(ws)

    return book


def load_master_prices_dict(tickers):
    """Load price history for `tickers` from master_prices.parquet → {ticker: DataFrame}.

    Used by the overflow scope to avoid yfinance bulk-pulling ~870 tickers
    on every run. Returns empty dict if the parquet is missing — caller
    should fall back to download_historical_data() in that case.
    """
    if not os.path.exists(MASTER_PRICES_PATH):
        return {}
    wanted = set(t.strip().upper().replace('.', '-') for t in tickers)
    # Predicate + column pushdown so we never materialize the full parquet in
    # memory (critical once master_prices grows to thousands of tickers × 20y —
    # a naive full read risks OOM on a 7 GB GHA runner). master_prices stores
    # tickers uppercased with '.'→'-' already, so the filter matches directly.
    _cols = ['ticker', 'date', 'Open', 'High', 'Low', 'Close', 'Volume']
    try:
        df = pd.read_parquet(
            MASTER_PRICES_PATH,
            columns=_cols,
            filters=[('ticker', 'in', list(wanted))],
        )
    except Exception as e:
        # Older pyarrow / engines may not support the filters kwarg — fall back
        # to a full read + in-memory filter (correct, just heavier).
        print(f"⚠️ pushdown read failed ({e}); falling back to full read")
        try:
            df = pd.read_parquet(MASTER_PRICES_PATH)
        except Exception as e2:
            print(f"⚠️ Failed to load {MASTER_PRICES_PATH}: {e2}")
            return {}
    if df.empty:
        return {}
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    df = df[df['ticker'].isin(wanted)]
    if df.empty:
        return {}
    df['date'] = pd.to_datetime(df['date'])
    out = {}
    for tkr, grp in df.groupby('ticker'):
        sub = grp.set_index('date').sort_index()
        sub = sub[[c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in sub.columns]]
        if sub.index.tz is not None:
            sub.index = sub.index.tz_localize(None)
        sub.index = sub.index.normalize()
        out[tkr] = sub
    return out


def load_atr_seasonal_map():
    """Load ATR-normalized seasonal ranks. Returns {ticker: DataFrame} or {} on failure."""
    if not os.path.exists(ATR_SZNL_PATH):
        return {}
    try:
        df = pd.read_parquet(ATR_SZNL_PATH)
    except Exception as e:
        print(f"⚠️ Failed to load {ATR_SZNL_PATH}: {e}")
        return {}
    if df.empty:
        return {}
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    output = {}
    for ticker, group in df.groupby('ticker'):
        output[str(ticker).upper()] = group.set_index('Date')[ATR_SZNL_COLS].sort_index()
    return output

# -----------------------------------------------------------------------------
# 1. AUTHENTICATION & HELPERS
# -----------------------------------------------------------------------------

def get_google_client():
    """
    Authenticates with Google Sheets using Environment Variables (GitHub Actions) 
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


def send_email_summary(signals_list, error_tickers=None, scope_label=None,
                       pc_state=None):
    """
    Sends an HTML email summary of the signals using Gmail SMTP.
    Card-based layout showing full signal criteria with LIVE values.

    The Risk Dial header and the Exposure Leg block were removed 2026-07-16
    (per McKinley): the dial state lives on the site risk tab's Sizing State
    hero and the daily risk email; the exposure leg still computes and writes
    exposure_state.json on the AM run (the site reads it), it just no longer
    renders here. Per-signal band tilts still appear in each signal's
    Sizing notes.
    """
    sender_email = os.environ.get("EMAIL_USER")
    sender_password = os.environ.get("EMAIL_PASS")
    receiver_email = "mckinleyslade@gmail.com"

    if not sender_email or not sender_password:
        print("⚠️ Email credentials (EMAIL_USER/EMAIL_PASS) not found. Skipping email.")
        return

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Filter out companion signals from email (they go to staging only)
    # All staged orders go to email (companions included in summary table)
    email_signals = list(signals_list) if signals_list else []
    
    # Count unique LOGICAL signals (primary + companion on same ticker = 1 signal)
    _seen_logical = set()
    for s in email_signals:
        base_strat = s.get('_parent_strategy', s.get('Strategy_Name', s['Strategy_ID']))
        _seen_logical.add((s['Ticker'], base_strat))
    signal_count = len(_seen_logical)
    
    # Separate for card generation: one card per logical signal
    _primary_signals = [s for s in email_signals if not s.get('_is_companion', False)]
    _companion_map = {s['Ticker']: s for s in email_signals if s.get('_is_companion', False)}
    
    # Build error tickers section (shared across both branches)
    error_html = ""
    if not error_tickers:
        error_html = '<div style="margin-top: 20px; font-size: 12px; color: #888;">✅ All tickers successfully parsed</div>'
    elif error_tickers:
        # Group by reason for compact display
        from collections import defaultdict
        by_reason = defaultdict(list)
        for ticker, reason in error_tickers:
            by_reason[reason].append(ticker)

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

    # P/C fear-state liveness footnote (2026-08-05): always rendered so a
    # stale feed is visible in every scan email, not just when a family
    # signal happens to fire.
    pc_html = ""
    if pc_state:
        if pc_state.get('state') == 'stale':
            _pc_txt = (f"🚨 P/C data STALE (last {pc_state.get('data_date')}, "
                       f"{pc_state.get('age_bd')} bd old) — family fragility "
                       f"bands failing closed to incumbent 0.25x tables")
            _pc_color = "#c62828"
        else:
            _pc_txt = (f"🧭 P/C fear state: {pc_state['pct']:.0f}%ile 10d-MA "
                       f"equity put/call (data through {pc_state['data_date']}, "
                       f"{pc_state['age_bd']} bd old) — fear "
                       f"{pc_state['state'].upper()}; family bands "
                       f"{'1.25x <50 / 1.0x >=50' if pc_state['state'] == 'on' else '1.0x <50 / ZERO >=50'}")
            _pc_color = "#666"
        pc_html = (f'<div style="margin-top: 12px; font-size: 12px; '
                   f'color: {_pc_color};">{_pc_txt}</div>')

    # Event-sleeve mini cards (2026-08-06): one card per calendar trade
    # (T1-T4) showing today's staged/skipped action, open positions with
    # their scheduled exits, or the next armed window. Best effort — the
    # email must never fail on sleeve status.
    event_html = ""
    try:
        from event_sleeve import sleeve_status_cards
        _kind_color = {"staged": "#1565c0", "open": "#2e7d32",
                       "skipped": "#8d6e63", "armed": "#888",
                       "error": "#c62828"}
        _cards = []
        for _c in sleeve_status_cards():
            _col = _kind_color.get(_c["kind"], "#888")
            _name = _c["trade"].replace("_", " ").title().replace("Fomc", "FOMC")
            _cards.append(
                f'<div style="border: 1px solid #eee; border-left: 3px solid '
                f'{_col}; border-radius: 4px; padding: 8px 10px; margin-top: 6px;">'
                f'<div style="font-size: 12px; color: #333;"><strong>{_name}'
                f'</strong> <span style="color: {_col};">{_c["status"]}</span></div>'
                f'<div style="font-size: 11px; color: #888; margin-top: 2px;">'
                f'{_c["rule"]} {_c["evidence"]}</div></div>')
        if _cards:
            event_html = (
                '<div style="margin-top: 16px;">'
                '<div style="font-size: 12px; color: #666;">📅 '
                '<strong>Event sleeve</strong> (calendar trades — '
                'event_moo.py places at 9:05 AM)</div>'
                + "".join(_cards) + "</div>")
    except Exception as _e:
        event_html = (f'<div style="margin-top: 12px; font-size: 11px; '
                      f'color: #c62828;">Event sleeve status unavailable: '
                      f'{_e}</div>')

    _scope_suffix = f" — {scope_label}" if scope_label else ""
    if not email_signals:
        subject = f"📉 Scan Result: NO SIGNALS ({date_str}){_scope_suffix}"
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; background-color: #f5f5f5; padding: 20px;">
                <div style="max-width: 700px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px;">
                    <h2 style="color: #333; margin-top: 0;">Daily Strategy Scan: {date_str}</h2>
                    <p style="color: #666;">The scan completed successfully.</p>
                    <p style="font-size: 18px; color: #888;"><strong>Result:</strong> No signals found matching criteria today.</p>
                    {pc_html}
                    {event_html}
                    {error_html}
                </div>
            </body>
        </html>
        """
    else:
        subject = f"🚀 {signal_count} SIGNAL{'S' if signal_count > 1 else ''} ({date_str}){_scope_suffix}"
        
        # Build card-based HTML for each signal
        signal_cards = []
        
        for sig in _primary_signals:
            # Check if this signal has a companion order
            _companion = _companion_map.get(sig['Ticker']) if not sig.get('_is_companion', False) else None
            
            # Header color based on action
            header_color = "#2e7d32" if sig['Action'] == "BUY" else "#c62828"
            action_emoji = "📈" if sig['Action'] == "BUY" else "📉"
            
            # Build the key filters bullet list WITH LIVE VALUES
            live_filters = sig.get('Live_Filters', [])
            if live_filters:
                filters_html_parts = []
                for filter_desc, live_val, is_binary in live_filters:
                    if is_binary:
                        # Binary filter - just show checkmark
                        filters_html_parts.append(
                            f"<li style='margin: 4px 0; color: #444;'>{filter_desc} <span style='color: #2e7d32; font-weight: bold;'>{live_val}</span></li>"
                        )
                    else:
                        # Numeric filter - show value after comma
                        filters_html_parts.append(
                            f"<li style='margin: 4px 0; color: #444;'>{filter_desc}, <span style='color: #1565c0; font-weight: bold;'>{live_val}</span></li>"
                        )
                filters_html = "".join(filters_html_parts)
            else:
                # Fallback to static filters if live not available
                static_filters = sig.get('Setup_Filters', [])
                if static_filters:
                    filters_html = "".join([f"<li style='margin: 4px 0; color: #444;'>{f}</li>" for f in static_filters])
                else:
                    filters_html = "<li style='color: #999;'>No filter details available</li>"
            
            # Build exit section - only show stop/target if actually used
            # Smart detection: check explicit flags OR infer from exit_primary text
            use_stop = sig.get('Use_Stop', True)
            use_target = sig.get('Use_Target', True)
            
            # Also check if exit_primary suggests time-only exit
            exit_primary = sig.get('Exit_Primary', '')
            if 'time stop' in exit_primary.lower() or 'time exit' in exit_primary.lower():
                # If it says "X-day time stop" without mentioning stop/target, suppress them
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
            
            # Exit notes (dynamic sizing info)
            exit_notes = sig.get('Exit_Notes', '')
            sizing_var = sig.get('Sizing_Variable', '')
            sizing_notes = sig.get('Sizing_Notes', '')

            # Combine exit notes with sizing variable if present
            notes_parts = []
            if sizing_var:
                notes_parts.append(f"📊 {sizing_var}")
            # Surface the multiplier chain (ATR sznl 1.5x, Frag, Ladder rung, etc.)
            # so we can see at-a-glance why the staged risk isn't just base 1.0x.
            if sizing_notes and sizing_notes != "Standard (1.0x)" and "Standard (1.0x) |" not in sizing_notes:
                notes_parts.append(f"⚖️ Sizing: {sizing_notes}")
            if exit_notes:
                notes_parts.append(f"⚡ {exit_notes}")
            
            # Companion order info
            if _companion:
                comp_price = _companion.get('Limit_Price', 0)
                comp_shares = _companion.get('Shares', 0)
                notes_parts.append(f"📋 Also staged: LOC {comp_shares:,} shares @ >${comp_price:.2f} (Close + 0.5 ATR)")
            
            if notes_parts:
                notes_html = "<div style='font-size: 12px; color: #ff9800; margin-top: 8px;'>" + "<br>".join(notes_parts) + "</div>"
            else:
                notes_html = ""
            
            # Thesis
            thesis = sig.get('Setup_Thesis', '')
            thesis_html = f"<div style='font-style: italic; color: #555; margin: 10px 0; padding: 10px; background: #f9f9f9; border-left: 3px solid #2196f3;'>{thesis}</div>" if thesis else ""
            
            # Entry type display - don't show price for Open-based limits
            entry_type = sig.get('Entry_Type', 'Signal Close')
            limit_price = sig.get('Limit_Price')
            
            # Determine if we know the entry price
            is_open_based = "Open" in entry_type and "Limit" in entry_type
            
            if is_open_based:
                # Open-based limit - we don't know T+1 Open yet
                entry_display = entry_type  # Just show the entry type, no price
            elif "Signal Close" in entry_type or "T+1 Close" in entry_type:
                # We know the price
                entry_display = f"{entry_type} @ ${sig['Entry']:.2f}"
            elif limit_price and "Close" in entry_type:
                # Close-based limit with known price
                entry_display = f"{entry_type} @ ${limit_price:.2f}"
            else:
                # Default: show entry price if known
                entry_display = f"{entry_type} @ ${sig['Entry']:.2f}"
            
            # Notional and days
            notional = sig.get('Notional', 0)
            days_to_exit = sig.get('Days_To_Exit', 0)
            
            card_html = f"""
            <div style="border: 1px solid #ddd; border-radius: 8px; margin-bottom: 20px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <!-- Header -->
                <div style="background: {header_color}; color: white; padding: 15px;">
                    <div style="font-size: 18px; font-weight: bold;">
                        {action_emoji} {sig.get('Strategy_Name', sig['Strategy_ID'])}
                    </div>
                    <div style="font-size: 13px; opacity: 0.9; margin-top: 3px;">
                        {sig.get('Setup_Type', 'Custom')} | {sig.get('Setup_Timeframe', 'Swing')}
                    </div>
                </div>
                
                <!-- Trade Details -->
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
                
                <!-- Thesis -->
                {thesis_html}
                
                <!-- Why It Flagged -->
                <div style="padding: 15px;">
                    <div style="font-weight: bold; color: #333; margin-bottom: 8px; font-size: 14px;">
                        🎯 WHY IT FLAGGED:
                    </div>
                    <ul style="margin: 0; padding-left: 20px; font-size: 13px;">
                        {filters_html}
                    </ul>
                </div>
                
                <!-- Exit Plan -->
                <div style="padding: 15px; background: #f5f5f5; border-top: 1px solid #eee;">
                    <div style="font-weight: bold; color: #333; font-size: 13px;">
                        🚪 EXIT: {sig.get('Exit_Primary', f'{days_to_exit}-day time stop')}
                    </div>
                    {exit_prices_html}
                    {notes_html}
                </div>
                
                <!-- Footer Stats -->
                <div style="padding: 10px 15px; background: #333; color: #aaa; font-size: 11px;">
                    📊 {sig['Stats']}
                </div>
            </div>
            """
            signal_cards.append(card_html)
        
        # Combine all cards
        all_cards_html = "".join(signal_cards)
        
        # Quick summary table - Entry Type and $ Risk instead of price
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
        
        # Total risk summary - NET notional (long - short)
        total_risk = sum(s.get('Risk_Amt', 0) for s in email_signals)
        long_notional = sum(s.get('Notional', 0) for s in email_signals if s['Action'] == 'BUY')
        short_notional = sum(s.get('Notional', 0) for s in email_signals if s['Action'] != 'BUY')
        net_notional = long_notional - short_notional
        long_count = len({(s['Ticker'], s.get('_parent_strategy', s.get('Strategy_Name'))) for s in email_signals if s['Action'] == 'BUY'})
        short_count = signal_count - long_count
        
        # Format net notional with +/- sign
        if net_notional >= 0:
            net_notional_str = f"+${net_notional:,.0f}"
        else:
            net_notional_str = f"-${abs(net_notional):,.0f}"
        
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; background-color: #f5f5f5; padding: 20px;">
                <div style="max-width: 700px; margin: 0 auto;">
                    <!-- Header -->
                    <div style="background: linear-gradient(135deg, #1a237e, #283593); color: white; padding: 25px; border-radius: 8px 8px 0 0; text-align: center;">
                        <h1 style="margin: 0; font-size: 24px;">Daily Strategy Scan</h1>
                        <div style="font-size: 14px; opacity: 0.8; margin-top: 5px;">{date_str}</div>
                        <div style="font-size: 28px; margin-top: 10px;">🎯 {signal_count} Signal{'s' if signal_count > 1 else ''}</div>
                        <div style="font-size: 14px; margin-top: 8px; opacity: 0.9;">
                            {long_count} Long | {short_count} Short | ${total_risk:,.0f} Risk | {net_notional_str} Net Exposure
                        </div>
                    </div>

                    <!-- Quick Summary -->
                    <div style="background: white; padding: 20px; border-bottom: 1px solid #ddd;">
                        <h3 style="margin-top: 0; color: #333;">⚡ Quick Summary</h3>
                        {summary_table}
                    </div>
                    
                    <!-- Detailed Cards -->
                    <div style="background: white; padding: 20px; border-radius: 0 0 8px 8px;">
                        <h3 style="color: #333;">📋 Signal Details</h3>
                        {all_cards_html}
                    </div>
                    
                    <!-- P/C fear-state liveness -->
                    {pc_html}
                    {event_html}

                    <!-- Error Tickers -->
                    {error_html}

                    <!-- Footer -->
                    <div style="text-align: center; padding: 15px; color: #888; font-size: 12px;">
                        Check Google Sheet for staging details
                    </div>
                </div>
            </body>
        </html>
        """

    # Setup and send message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg.attach(MIMEText(html_content, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print(f"📧 Email sent successfully to {receiver_email}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


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


# -----------------------------------------------------------------------------
# 2. CALCULATION ENGINE
# -----------------------------------------------------------------------------

# ETF_ATR_EXEMPT moved to filters.py (2026-07-16)

# Trading-day staleness bound for the fragility cache (data/rd2_fragility.parquet).
# The producer (risk_report.yml → daily_risk_report.py) writes it every weekday
# post-close; a value older than this many trading days means the producer is
# broken / missed runs, so we must NOT trade through it. Sizing falls back to
# 1.0x and the dial-filter gate fails closed. Tolerates a long weekend + one
# missed run without over-triggering.
# FRAG_STALE_TD moved to filters.py (2026-07-16)

# Fragility risk bands (2026-07-02) — replaced the retired book-wide ramp
# (1.25x boost -> 0.10x floor; the boost had no edge case and the rest of the
# book shows no high-frag degradation). Per-strategy now: strategies opt in via
# execution['frag_risk_bands'] = [[lo, hi, mult], ...] on the 10d-MA 63d score
# (FAMILY4 dip-buyers + 3x Bear Fade 0.25x at >=50; the OVS [21,44) 0.75x tilt
# was removed 2026-07-03 after failing the PIT gate). Aligned with
# strat_backtester sizing 3b3, which replays the same bands point-in-time —
# unlike the old ramp, the ledger and live now agree.


def frag_band_mult(execution, frag_score, pc_state=None):
    """Sizing multiplier from the strategy's fragility band table for a
    10d-MA 63d fragility score. Bands are [lo, hi, mult], first match wins,
    mult applies when lo <= score < hi. 1.0 when the strategy has no bands,
    the DIAL score is None/stale, or no band matches (the stale-dial fail-open
    convention is unchanged book-wide).

    P/C fear-conditioned table selection (2026-08-05): strategies carrying
    execution['pc_fear_bands'] select their table by pc_state['state']
    ('on'/'off' from pc_fear.fear_state_asof); stale/absent P/C state fails
    CLOSED to the plain frag_risk_bands (the incumbent 0.25x book). Mirrored
    in strat_backtester 3b3 (frag_band_mult_at) — change together."""
    if not execution:
        return 1.0
    bands = pc_fear.select_bands(
        execution, pc_state['state'] if pc_state else 'stale')
    return pc_fear.band_mult(bands, frag_score)


def _print_ledger_provenance(path, n_rows):
    """Surface the ledger vintage the gate is about to act on. The ledger is a
    full backtest rebuild whose recent trades can flicker between vintages, and
    the R2 key can in principle be written from anywhere — so every scan logs
    who built the copy it's using and warns on stale/non-GHA vintages
    (2026-07-06: an unattributed weekend vintage false-blocked TS/USO)."""
    try:
        import pyarrow.parquet as _pq
        meta = _pq.read_schema(path).metadata or {}
        built = (meta.get(b'ledger_build_utc') or b'').decode()
        source = (meta.get(b'ledger_source') or b'').decode()
        if not built:
            print("⚠️ sector_loss_gate: ledger has no provenance metadata "
                  "(pre-2026-07-06 vintage or non-standard writer)")
            return
        print(f"   sector_loss_gate ledger: {n_rows} trades, built {built} by {source or 'unknown'}")
        age = pd.Timestamp.now(tz='UTC') - pd.Timestamp(built)
        if age > pd.Timedelta(days=4):
            print(f"⚠️ sector_loss_gate: ledger vintage is {age.days} days old — "
                  "gate may act on outdated exits (fail-open by design, not disabling)")
        if not source.startswith('gha:'):
            print(f"⚠️ sector_loss_gate: ledger was built OUTSIDE the deploy pipeline "
                  f"({source or 'unknown'}) — verify this vintage is intentional")
    except Exception:
        pass


_SLG_CACHE = {}  # key='state' → (ledger_df or None, sector_map dict)
def _sector_gate_state():
    """Closed-trade ledger + sector map for execution['sector_loss_gate'].

    The ledger (data/backtest_trades_full.parquet) is the modeled book rebuilt
    nightly by deploy_site and mirrored to R2 (pulled by daily_screener.yml),
    so at scan time it holds exits through yesterday — the same "exits strictly
    before the signal date" window the backtest gate uses. Missing ledger or
    sector map degrades to gate-off with a printed notice (fail-open: the gate
    is an overlay, not a dependency)."""
    if 'state' in _SLG_CACHE:
        return _SLG_CACHE['state']
    ledger, smap = None, {}
    try:
        _lp = os.path.join(current_dir, 'data', 'backtest_trades_full.parquet')
        if os.path.exists(_lp):
            ledger = pd.read_parquet(
                _lp, columns=['Strategy', 'Ticker', 'Exit Date', 'R_Multiple'])
            ledger['Exit Date'] = pd.to_datetime(ledger['Exit Date'])
            _print_ledger_provenance(_lp, len(ledger))
        else:
            print("⚠️ sector_loss_gate: data/backtest_trades_full.parquet missing — gate disabled")
    except Exception as e:
        print(f"⚠️ sector_loss_gate: ledger unreadable ({e}) — gate disabled")
        ledger = None
    try:
        _sp = os.path.join(current_dir, 'data', 'sector_map.parquet')
        if os.path.exists(_sp):
            _sd = pd.read_parquet(_sp)
            smap = dict(zip(_sd['ticker'].astype(str).str.upper(), _sd['sector']))
        else:
            print("⚠️ sector_loss_gate: data/sector_map.parquet missing — gate disabled")
    except Exception:
        smap = {}
    _SLG_CACHE['state'] = (ledger, smap)
    return _SLG_CACHE['state']


def sector_gate_blocked(strat_name, execution, t_clean, asof):
    """(blocked, note) for execution['sector_loss_gate'] at a signal date.
    Blocks when the strategy's realized R in the same sector over the trailing
    window_td business days is max_realized_r or worse. Mirrors the candidate
    gate in pages/strat_backtester.py — change together."""
    cfg = execution.get('sector_loss_gate') if execution else None
    if not cfg:
        return False, ''
    ledger, smap = _sector_gate_state()
    if ledger is None or not smap:
        return False, ''
    sec = smap.get(t_clean.upper())
    if not sec or sec == 'UNKNOWN':
        # no real sector info -> pass through. Never pool UNKNOWN tickers into
        # one pseudo-sector (2026-07-03 fix: USO was gated off unrelated
        # no-sector names' losses before the ETF sector table existed).
        return False, ''
    asof = pd.Timestamp(asof).normalize()
    lo = asof - pd.tseries.offsets.BDay(int(cfg['window_td']))
    sub = ledger[(ledger['Strategy'] == strat_name)
                 & (ledger['Exit Date'] >= lo) & (ledger['Exit Date'] < asof)]
    if sub.empty:
        return False, ''
    same = sub[sub['Ticker'].astype(str).str.upper().map(smap).eq(sec)]
    rsum = float(same['R_Multiple'].sum()) if len(same) else 0.0
    if rsum < float(cfg['max_realized_r']):
        # Name every contributing exit: the threshold is a knife edge and the
        # ledger rebuilds nightly, so a block must be auditable after the
        # vintage that produced it is gone (2026-07-06 TS/USO forensics).
        detail = ', '.join(
            f"{r['Ticker']} {r['Exit Date'].strftime('%m-%d')} {r['R_Multiple']:+.2f}R"
            for _, r in same.sort_values('Exit Date').iterrows())
        return True, (f"{sec}: {rsum:+.1f}R realized over last {cfg['window_td']}td "
                      f"({len(same)} exits: {detail}) < {cfg['max_realized_r']}R")
    return False, ''


# fragility cache moved to filters.py (2026-07-16)

def memoized_indicators(memo, key, src_df, sznl_map, t_key, market_series,
                        vix_series, ref_ranks, xsec_rank_matrices, atr_sznl_map):
    """Per-run indicator memo (2026-07-16). calculate_indicators used to be
    recomputed for every (strategy, ticker) pair — 5.9x redundant, ~14 min of
    the pre-market critical path. Its output depends only on the ticker frame
    plus the market-series source and ref-rank config (both folded into
    ``key`` by the caller); sznl_map / vix / xsec / atr_sznl are global per
    run. Sharing the returned frame across strategies is safe because
    check_signal and the signal-build path are strictly READ-ONLY on it —
    verified by inspection and guarded by tests/test_indicator_memo_parity.py.
    The atr_sznl merge is folded in here so the cached frame is complete."""
    got = memo.get(key)
    if got is not None:
        return got
    got = calculate_indicators(
        src_df.copy(), sznl_map, t_key, market_series, vix_series,
        ref_ticker_ranks=ref_ranks, xsec_rank_matrices=xsec_rank_matrices,
    )
    if atr_sznl_map and t_key in atr_sznl_map:
        _atr_ranks = atr_sznl_map[t_key]
        _dates = got.index.normalize()
        for _col in ATR_SZNL_COLS:
            if _col in _atr_ranks.columns:
                got[_col] = _atr_ranks[_col].reindex(_dates).values
    memo[key] = got
    return got


def check_signal(df, params, sznl_map, ticker=None):
    """Delegates to filters.check_signal_live — the single filter
    implementation shared with the engine (consolidated 2026-07-16; the
    ~420-line body that lived here is now filters.py). Live mode:
    dial_filters FAIL CLOSED on missing/stale fragility data, and the T+1
    gates are stripped (the scan stamps their specs; order_staging enforces
    them at the real T+1 open). Guards:
    tests/test_filters_consolidation.py; ship-time proof
    scratch/verify_filters_consolidation.py."""
    return check_signal_live(df, params, sznl_map=sznl_map, ticker=ticker)


# -----------------------------------------------------------------------------
# 3. SAVING FUNCTIONS
# -----------------------------------------------------------------------------

def save_moc_orders(signals_list, strategy_book, sheet_name='moc_orders'):
    """
    Saves 'Signal Close' orders to the 'moc_orders' tab with Exit_Date.
    """
    gc = get_google_client()
    if not gc: return

    try:
        sh = gc.open("Trade_Signals_Log")
        try:
            worksheet = sh.worksheet(sheet_name)
        except:
            worksheet = sh.add_worksheet(title=sheet_name, rows=100, cols=20)
        
        worksheet.clear()

        # Filter and Build Data
        moc_data = []
        if signals_list:
            strat_map = {s['id']: s for s in strategy_book}
            
            for row in pd.DataFrame(signals_list).to_dict('records'):
                strat = strat_map.get(row['Strategy_ID'])
                if not strat: continue

                # Skip overflow-tier rows: thin-liquidity tickers shouldn't be
                # MOC'd. Mirrors the original local_overflow_scan convention.
                if str(row.get('Scan_Source', 'Liquid')).strip() == 'Overflow':
                    continue

                settings = strat['settings']
                entry_mode = settings.get('entry_type', 'Signal Close')

                if entry_mode == "Signal Close":
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
            print(f"🚀 Staged {len(df_moc)} MOC Orders with Exit Dates!")
        else:
            headers = ["Scan_Date", "Symbol", "SecType", "Exchange", "Action", "Quantity", "Order_Type", "Strategy_Ref", "Exit_Date"]
            worksheet.update(values=[headers])
            print(f"🧹 '{sheet_name}' cleared.")
            
    except Exception as e:
        print(f"❌ MOC Staging Error: {e}")


def _sheets_write_with_retry(what, fn, attempts=3, backoffs=(5, 20)):
    """Run a Sheets mutation with retries; RAISE on final failure.

    Staging tabs gate live orders — a swallowed clear/update failure either
    leaves stale rows for order_staging to resubmit or a half-cleared tab,
    while the workflow stays green and the email claims signals staged.
    Raising turns the GHA run red, which is the alert channel.
    (2026-07-16 fail-loud batch.)
    """
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise RuntimeError(
                    f"Sheets write failed after {attempts} attempts — {what}: {e}"
                ) from e
            wait = backoffs[min(i, len(backoffs) - 1)]
            print(f"[WARN] {what}: attempt {i + 1}/{attempts} failed ({e}); retrying in {wait}s")
            time.sleep(wait)


def _staging_no_client(sheet_name):
    """No Sheets client: fatal in GHA (a green run with unwritten staging
    tabs leaves stale rows live), warn-and-continue locally."""
    msg = f"No Google Sheets client — cannot write staging tab '{sheet_name}'"
    if os.environ.get('GITHUB_ACTIONS'):
        raise RuntimeError(msg + " (failing loud in GHA)")
    print(f"[WARN] {msg} (local run — continuing)")


def save_staging_orders(signals_list, strategy_book, sheet_name='Order_Staging', tier_filter=None):
    """Save non-MOC orders to a Google Sheets tab.

    Excludes 'Signal Close' (those go to moc_orders).

    tier_filter: if 'Liquid' or 'Overflow', only stage rows whose Scan_Source
        matches. Used by the merged scan to write Liquid → Order_Staging and
        Overflow → Overflow without touching the other tab.

    CHANGES from previous version:
    - FIX: GTC entry types now correctly detected (was only checking "Persistent")
    - NEW: Bracket exit metadata columns (Tgt_ATR_Mult, Stop_ATR_Mult,
           Use_Target, Use_Stop, Hold_Days, Trade_Direction) so order_staging.py
           can compute exit prices anchored to the resolved entry limit price.
    - NEW: tier_filter for tier-aware tab writes (Liquid → Order_Staging,
           Overflow → Overflow). If unset, behaves as before (writes everything).
    """
    if tier_filter is not None:
        signals_list = [
            s for s in (signals_list or [])
            if str(s.get('Scan_Source', 'Liquid')).strip() == tier_filter
        ]
    if not signals_list:
        # Even on zero-signal days we clear the tab so stale rows from a
        # prior run never linger and get re-staged by order_staging.
        # A failed clear leaves yesterday's rows live — fail loud, never
        # swallow (2026-07-16).
        gc = get_google_client()
        if gc is None:
            _staging_no_client(sheet_name)
            return

        def _clear_tab():
            sh = gc.open("Trade_Signals_Log")
            try:
                ws = sh.worksheet(sheet_name)
            except gspread.WorksheetNotFound:
                return  # tab doesn't exist — nothing stale to clear
            ws.clear()

        _sheets_write_with_retry(f"clear '{sheet_name}' (zero-signal day)", _clear_tab)
        print(f"🧹 '{sheet_name}' cleared — no rows for tier_filter={tier_filter}")
        return

    df = pd.DataFrame(signals_list)
    strat_map = {s['id']: s for s in strategy_book}
    
    staging_data = []
    
    for _, row in df.iterrows():
        # Handle companion signals specially (they have their own entry type)
        if row.get('_is_companion', False) is True:
            entry_mode = row.get('Entry_Type', '')
            
            # LOC companion orders
            if "LOC" in entry_mode:
                ib_action = "SELL" if "SHORT" in row['Action'] else "BUY"
                trade_dir = "Short" if "SHORT" in row['Action'] else "Long"
                staging_data.append({
                    "Scan_Date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "Signal_Date": str(row['Date']),
                    "Symbol": row['Ticker'],
                    "SecType": "STK",
                    "Exchange": "SMART",
                    "Action": ib_action,
                    "Quantity": row['Shares'],
                    "Order_Type": "LOC",
                    "Limit_Price": round(row.get('Limit_Price', row['Entry']), 2),
                    # Manual entry-price override (always emitted empty). A user
                    # types a price here in the sheet to pin a specific limit for
                    # one signal; order_staging.py uses it verbatim as a LMT and
                    # anchors the bracket to it. Emitted empty so the column
                    # survives daily_scan's clear+rewrite.
                    "Manual_Limit": "",
                    "Offset_ATR_Mult": 0.0,
                    "TIF": "DAY",
                    "Frozen_ATR": round(row['ATR'], 2),
                    "Signal_Close": round(row['Entry'], 2),
                    "Time_Exit_Date": str(row['Time Exit']),
                    "Strategy_Ref": row.get('Strategy_Name', 'Companion'),
                    # Bracket metadata for companion orders
                    "Tgt_ATR_Mult": 0.0,
                    "Stop_ATR_Mult": 0.0,
                    "Use_Target": False,
                    "Use_Stop": False,
                    "Hold_Days": row.get('Days_To_Exit', 0),
                    "Trade_Direction": trade_dir,
                    "Rank_252D": row.get('Rank_252D', ''),
                    "Risk_Amt": float(row.get('Risk_Amt', 0) or 0),
                    "Risk_Bps": 0,
                    "Scan_Source": str(row.get('Scan_Source', 'Liquid')),
                })
            continue
        
        strat = strat_map.get(row['Strategy_ID'])
        if not strat: continue
        
        settings = strat['settings']
        execution = strat['execution']
        entry_mode = settings.get('entry_type', 'Signal Close')
        
        # *** SKIP MOC ORDERS (They go to the other sheet) ***
        if entry_mode == "Signal Close":
            continue
        
        # Defaults
        entry_instruction = "MKT" 
        offset_atr = 0.0
        limit_price = 0.0
        tif_instruction = "DAY" 

        # =====================================================
        # 1. ATR LIMIT ENTRY
        # FIX: Now checks for BOTH "Persistent" and "GTC" to
        # correctly route GTC limit orders as REL_CLOSE.
        # Previously only checked "Persistent", causing
        # "Limit (Open +/- 0.5 ATR) GTC" to fall through
        # to REL_OPEN / DAY — a silent backtest divergence.
        # =====================================================
        if "Limit" in entry_mode and "ATR" in entry_mode:
            is_persistent = "Persistent" in entry_mode or "GTC" in entry_mode
            
            if is_persistent:
                entry_instruction = "REL_CLOSE"  # Anchored to Signal Close
                tif_instruction = "GTC"           # Good til canceled (or hold_days)
            else:
                entry_instruction = "REL_OPEN"   # Anchored to T+1 Open
                tif_instruction = "DAY"
            
            # Order matters: check 0.75/0.25 before 0.5 (substring-safe). "1 ATR" last
            # so "0.5" / "0.75" / "0.25" don't accidentally match it.
            if "0.75" in entry_mode: offset_atr = 0.75
            elif "0.25" in entry_mode: offset_atr = 0.25
            elif "0.5" in entry_mode: offset_atr = 0.5
            elif "1 ATR" in entry_mode: offset_atr = 1.0

        elif "LOC" in entry_mode:
            entry_instruction = "LOC"
            limit_price = row.get('Limit_Price', row['Entry'])
            tif_instruction = "DAY" 
            
        # 2. MARKET ON OPEN
        elif "T+1 Open" in entry_mode:
            entry_instruction = "MOO" 
            tif_instruction = "OPG"
            
        # 3. CONDITIONAL CLOSE (Oversold Low Vol Logic)
        elif "T+1 Close if < Signal Close" in entry_mode:
            entry_instruction = "LMT"
            limit_price = row['Entry'] - 0.01
            tif_instruction = "DAY" 
        
        ib_action = "SELL" if "SHORT" in row['Action'] else "BUY"

        # =====================================================
        # NEW: Pull bracket exit metadata from strategy config.
        # These are ATR multipliers and flags — NOT prices.
        # order_staging.py will compute actual prices once it
        # resolves the entry limit price.
        # =====================================================
        use_target = execution.get('use_take_profit', False)
        use_stop = execution.get('use_stop_loss', False)
        # Vol-confirmed stop mode (OLV, 2026-07-20): NO resting STP leg live.
        # The stop decision is made post-close on settled bars and staged as
        # a next-open MOO exit (stage_olv_vol_confirm_exits below), so the
        # entry bracket must carry target + time legs only. use_stop_loss
        # stays True in config because stop_atr still defines the risk unit.
        if execution.get('stop_mode') == 'vol_confirm_close':
            use_stop = False
        tgt_atr_mult = execution.get('tgt_atr', 0.0)
        stop_atr_mult = execution.get('stop_atr', 0.0)
        hold_days = execution.get('hold_days', 0)
        # Entry-order live window: order_staging cancels a persistent (GTC) limit
        # if unfilled after this many trading days. Defaults to hold_days so
        # non-persistent / unspecified strategies are unaffected. OLV=3 (2026-06-24).
        fill_window_days = execution.get('fill_window_days', hold_days)
        trade_direction = settings.get('trade_direction', 'Long')

        staging_data.append({
            "Scan_Date": datetime.datetime.now().strftime("%Y-%m-%d"),
            # Signal bar's date (NOT the run date). Scan_Date restamps to the
            # run day on every rescan — the Monday AM pass restamps a Friday
            # signal to Monday — so any weekday-gated rule in order_staging
            # (MonGapKill_Weekdays) must read THIS column, never Scan_Date.
            "Signal_Date": str(row['Date']),
            "Symbol": row['Ticker'],
            "SecType": "STK",
            "Exchange": "SMART",
            "Action": ib_action,
            "Quantity": row['Shares'],
            "Order_Type": entry_instruction,
            "Limit_Price": round(limit_price, 2),
            # Manual entry-price override (always emitted empty). A user types a
            # price here in the sheet to pin a specific limit for one signal;
            # order_staging.py uses it verbatim as a LMT and anchors the bracket
            # to it. Emitted empty so the column survives the clear+rewrite.
            "Manual_Limit": "",
            "Offset_ATR_Mult": offset_atr,
            "TIF": tif_instruction,
            "Frozen_ATR": round(row['ATR'], 2),
            "Signal_Close": round(row['Entry'], 2),
            "Signal_High": round(float(row.get('Signal_High', 0) or 0), 2),
            "Time_Exit_Date": str(row['Time Exit']),
            "Strategy_Ref": strat['name'],
            # Bracket exit metadata (NEW)
            "Tgt_ATR_Mult": tgt_atr_mult,
            "Stop_ATR_Mult": stop_atr_mult,
            "Use_Target": use_target,
            "Use_Stop": use_stop,
            "Hold_Days": hold_days,
            "Fill_Window_Days": fill_window_days,
            "Trade_Direction": trade_direction,
            # 252D rank stamped for OVS gap-tier sizing in order_staging.py
            "Rank_252D": row.get('Rank_252D', ''),
            # Per-trade risk $ (post all scanner multipliers). order_staging.py
            # uses this as the canonical base for the daily cap — staging
            # recomputes size from this rather than back-deriving from Quantity.
            "Risk_Amt": float(row.get('Risk_Amt', 0) or 0),
            "Risk_Bps": int(execution.get('risk_bps', 0)),
            # OVS 2-path sizing fields (empty for non-OVS rows). order_staging
            # reads these to apply the path-1/path-2/skip decision and the
            # path-2 daily aggregate cap.
            "Path1_Bps": execution.get('path1_bps', '') if strat['name'] == "Overbot Vol Spike" else '',
            "Path2_Bps": execution.get('path2_bps', '') if strat['name'] == "Overbot Vol Spike" else '',
            "Path2_Daily_Cap_Pct": execution.get('path2_daily_cap_pct', '') if strat['name'] == "Overbot Vol Spike" else '',
            # T+1 Open filter spec (JSON list of {logic, reference, atr_offset}).
            # order_staging reads this and drops the trade if any condition
            # fails when T+1 open prints. Empty for strategies that don't use it.
            "T1_Open_Filters": (
                json.dumps(strat['settings'].get('t1_open_filters', []))
                if strat['settings'].get('use_t1_open_filter')
                else ''
            ),
            # Monday-gap kill spec. The Friday post-close scan can't see Monday's
            # open, so the scanner only STAMPS the rule; order_staging.py enforces
            # it at the IBKR T+1 session open. For a row whose Signal_Date.weekday()
            # is in MonGapKill_Weekdays (e.g. [4]=Friday), order_staging drops the
            # trade if the T+1 open gaps > MonGapKill_ATR * Frozen_ATR above
            # Signal_Close (MonGapKill_Dir='up'). Empty for strats that don't set it.
            # Gate reads Signal_Date, NOT Scan_Date — Scan_Date restamps to the
            # run day on the Monday AM rescan, which left this gate dead 2026-06-09
            # to 2026-07-16.
            "MonGapKill_ATR": (
                strat['settings'].get('t1_gap_kill_atr', '')
                if strat['settings'].get('use_t1_gap_kill') else ''
            ),
            "MonGapKill_Weekdays": (
                json.dumps(strat['settings'].get('t1_gap_kill_signal_weekdays', []))
                if strat['settings'].get('use_t1_gap_kill') else ''
            ),
            "MonGapKill_Dir": (
                strat['settings'].get('t1_gap_kill_dir', 'up')
                if strat['settings'].get('use_t1_gap_kill') else ''
            ),
            # Large-gap-up size DERATE spec (2026-07-21). Like MonGapKill this is a
            # T+1-open rule the pre-market scan can't evaluate, so it only STAMPS
            # the params; order_staging halves the qty at the IBKR open when the
            # open gaps > GapDerate_ATR * Frozen_ATR past Signal_Close in
            # GapDerate_Dir. Unlike MonGapKill (a full drop) this scales size by
            # GapDerate_Mult and is NOT weekday-gated. Empty for strats without the
            # execution['gap_size_derate'] field. Mirror: strat_backtester 3b5.
            "GapDerate_ATR": (
                execution['gap_size_derate'].get('threshold_atr', '')
                if execution.get('gap_size_derate') else ''
            ),
            "GapDerate_Mult": (
                execution['gap_size_derate'].get('mult', '')
                if execution.get('gap_size_derate') else ''
            ),
            "GapDerate_Dir": (
                execution['gap_size_derate'].get('dir', 'up')
                if execution.get('gap_size_derate') else ''
            ),
            # Tier this signal's universe sits in. save_staging_orders routes by
            # tier — Liquid rows → Order_Staging, Overflow rows → the Overflow tab
            # — and this field travels with the row so order_staging can tell the
            # tiers apart after it concatenates both tabs.
            "Scan_Source": str(row.get('Scan_Source', 'Liquid')),
            # NAV the scanner sized against. order_staging asserts this equals
            # its own hardcoded ACCOUNT_VALUE at load and ABORTS on mismatch —
            # the two constants live in different files and have drifted before
            # (the 200-vs-250 cap class). Change them together (2026-07-16).
            "Account_Value": float(ACCOUNT_VALUE),
            # Live filter readings at signal time (desc, value) — same data the
            # scan email shows. Display-only: the private site's signal cards
            # render "why it fired"; order_staging ignores the column.
            "Live_Filters": json.dumps([
                (d, v) for d, v, *_ in (row.get('Live_Filters') or [])
            ]),
        })

    # If all orders were "Signal Close", this list is empty now
    if not staging_data:
        gc = get_google_client()
        if gc is None:
            _staging_no_client(sheet_name)
            return

        def _clear_tab_moc():
            sh = gc.open("Trade_Signals_Log")
            try:
                ws = sh.worksheet(sheet_name)
            except gspread.WorksheetNotFound:
                return
            ws.clear()

        _sheets_write_with_retry(f"clear '{sheet_name}' (only MOC orders)", _clear_tab_moc)
        print(f"🧹 '{sheet_name}' cleared (only MOC orders found).")
        return

    df_stage = pd.DataFrame(staging_data)

    gc = get_google_client()
    if gc is None:
        _staging_no_client(sheet_name)
        return

    # clear+update as one retried unit: a failure between the two calls
    # leaves the tab empty (or half-written) — the retry re-runs both, and
    # a final failure RAISES so the run goes red instead of the email
    # claiming these rows were staged (2026-07-16).
    def _write_tab():
        sh = gc.open("Trade_Signals_Log")
        try:
            worksheet = sh.worksheet(sheet_name)
        except gspread.WorksheetNotFound:
            worksheet = sh.add_worksheet(title=sheet_name, rows=100, cols=20)
        # Preserve Manual_Limit pins across the clear+rewrite (2026-07-16).
        # A price typed into the sheet used to be destroyed by ANY re-scan
        # (including the 10:30 UTC fallback) with no trace. Pins are re-applied
        # by (Symbol, Strategy_Ref) to signals still present this scan; a pin
        # whose signal vanished dies with it, which is correct.
        try:
            _existing = worksheet.get_all_records()
        except Exception as _pe:
            _existing = []
            print(f"[WARN] could not read existing tab for Manual_Limit pins: {_pe}")
        _pins = {}
        for _r in _existing:
            _ml = str(_r.get('Manual_Limit', '') or '').strip()
            if _ml and _ml.lower() not in ('nan', 'none'):
                _pins[(str(_r.get('Symbol', '')).strip().upper(),
                       str(_r.get('Strategy_Ref', '')).strip())] = _ml
        if _pins and 'Manual_Limit' in df_stage.columns:
            _applied = 0
            for _i in df_stage.index:
                _k = (str(df_stage.at[_i, 'Symbol']).strip().upper(),
                      str(df_stage.at[_i, 'Strategy_Ref']).strip())
                if _k in _pins:
                    df_stage.at[_i, 'Manual_Limit'] = _pins[_k]
                    _applied += 1
            _lost = len(_pins) - _applied
            print(f"📌 Manual_Limit pins preserved: {_applied}"
                  + (f" ({_lost} pin(s) dropped — signal no longer present)" if _lost else ""))
        worksheet.clear()
        data_to_write = [df_stage.columns.tolist()] + df_stage.astype(str).values.tolist()
        worksheet.update(values=data_to_write)
        # readback: the tab was just cleared, so anything other than exactly
        # header + N rows means a truncated/failed write — raise into the retry
        got = len(worksheet.get_all_values())
        if got != len(data_to_write):
            raise RuntimeError(f"readback mismatch: {got} rows in tab, wrote {len(data_to_write)}")

    _sheets_write_with_retry(f"stage {len(df_stage)} rows to '{sheet_name}'", _write_tab)
    print(f"🤖 Instructions Staged! ({len(df_stage)} rows)")


def save_signals_to_gsheet(new_dataframe, sheet_name='Trade_Signals_Log'):
    if new_dataframe.empty: return
    
    # Clean Data - exclude the detailed setup/exit/execution fields from the main log
    df_new = new_dataframe.copy()
    
    # Drop the detailed fields (they're for email only)
    cols_to_drop = [
        'Setup_Type', 'Setup_Timeframe', 'Setup_Thesis', 'Setup_Filters',
        'Exit_Primary', 'Exit_Stop', 'Exit_Target', 'Exit_Notes',
        'Live_Filters', 'Entry_Type',
        'Notional', 'Days_To_Exit', 'Use_Stop', 'Use_Target', 'Sizing_Variable'
    ]
    df_new = df_new.drop(columns=[c for c in cols_to_drop if c in df_new.columns], errors='ignore')
    
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


# -----------------------------------------------------------------------------
# 4. MAIN EXECUTION
# -----------------------------------------------------------------------------

def get_entry_type_short(entry_mode, limit_price=None):
    """
    Returns a concise entry type label for the summary table.
    For Open-based limits, we can't show a price since T+1 Open is unknown.
    """
    if "T+1 Close if <" in entry_mode:
        return "Cond Close"
    elif entry_mode == "Signal Close":
        return "MOC"
    elif "T+1 Open" in entry_mode and "Limit" not in entry_mode:
        return "MOO"
    elif "Limit" in entry_mode:
        # Check if it's Open-based (unknown price) or Close-based (known price)
        if "Open" in entry_mode:
            # Open-based: can't show price, it depends on T+1 Open
            if "0.75" in entry_mode:
                return "Open ±0.75 ATR"
            elif "0.25" in entry_mode:
                return "Open ±0.25 ATR"
            elif "0.5" in entry_mode:
                return "Open ±0.5 ATR"
            elif "1 ATR" in entry_mode:
                return "Open ±1 ATR"
            else:
                return "Open LMT"
        elif "Persistent" in entry_mode:
            # Close-based persistent limit - can show price
            if limit_price:
                return f"LMT ${limit_price:.2f} GTC"
            return "LMT GTC"
        else:
            # Close-based day limit - can show price
            if limit_price:
                return f"LMT ${limit_price:.2f}"
            return "LMT"
    else:
        return entry_mode[:15]


def get_sizing_variable(strat_name, last_row):
    """
    Returns the key variable that drives sizing for dynamic-sized strategies.
    """
    if strat_name == "Overbot Vol Spike":
        is_ath_l10 = bool(last_row.get('is_ath', False))  # simplified; full check in main loop
        is_52w = bool(last_row.get('is_52w_high', False))
        return f"ATH Today: {'Y' if last_row.get('is_ath', False) else 'N'} | 52w High: {'Y' if is_52w else 'N'}"
    elif strat_name == "Weak Close Decent Sznls":
        sznl = last_row.get('Sznl', 0)
        return f"Seasonal Rank: {sznl:.0f}"
    else:
        return None

def build_live_filters(strat, last_row, df):
    """
    Builds a list of filter descriptions with their LIVE values from the scan.
    Returns list of tuples: (filter_description, live_value, is_binary)
    """
    live_filters = []
    settings = strat['settings']
    
    # --- Performance Rank Filters ---
    for pf in settings.get('perf_filters', []):
        window = pf['window']
        col = f"rank_ret_{window}d"
        val = last_row.get(col, 0)
        logic = pf['logic']
        thresh = pf['thresh']
        consec = pf.get('consecutive', 1)

        if logic == 'Between':
            thresh_max = pf.get('thresh_max', 100.0)
            desc = f"{window}D rank between {thresh:.0f}-{thresh_max:.0f}th %ile"
        elif logic == 'Not Between':
            thresh_max = pf.get('thresh_max', 100.0)
            desc = f"{window}D rank NOT between {thresh:.0f}-{thresh_max:.0f}th %ile"
        else:
            desc = f"{window}D rank {logic} {thresh:.0f}th %ile"
        if consec > 1:
            desc += f" ({consec}d consecutive)"
        live_filters.append((desc, f"{val:.1f}", False))
    
    # --- Single Perf Rank (legacy format) ---
    if settings.get('use_perf_rank', False):
        window = settings['perf_window']
        col = f"rank_ret_{window}d"
        val = last_row.get(col, 0)
        logic = settings['perf_logic']
        thresh = settings['perf_thresh']
        consec = settings.get('perf_consecutive', 1)
        
        desc = f"{window}D rank {logic} {thresh:.0f}th %ile"
        if consec > 1:
            desc += f" ({consec}d consecutive)"
        live_filters.append((desc, f"{val:.1f}", False))
    
    # --- Seasonality ---
    if settings.get('use_sznl', False):
        val = last_row.get('Sznl', 50)
        logic = settings['sznl_logic']
        thresh = settings['sznl_thresh']
        live_filters.append((f"Ticker seasonal {logic} {thresh:.0f}", f"{val:.0f}", False))
    
    if settings.get('use_market_sznl', False):
        val = last_row.get('Mkt_Sznl_Ref', 50)
        logic = settings['market_sznl_logic']
        thresh = settings['market_sznl_thresh']
        live_filters.append((f"Market seasonal {logic} {thresh:.0f}", f"{val:.0f}", False))

    # --- ATR Seasonal Rank Filters (ATR-normalized forward-return rank per DOY) ---
    for asf in settings.get('atr_sznl_filters', []):
        window = asf['window']
        col = f"atr_sznl_{window}d"
        val = last_row.get(col)
        logic = asf.get('logic', '>')
        thresh = asf.get('thresh', 50.0)
        thresh_max = asf.get('thresh_max', 100.0)
        consec = asf.get('consecutive', 1)

        if logic == 'Between':
            desc = f"{window}D ATR sznl rank between {thresh:.0f}-{thresh_max:.0f}"
        else:
            desc = f"{window}D ATR sznl rank {logic} {thresh:.0f}th %ile"
        if consec > 1:
            desc += f" ({consec}d consecutive)"

        val_str = f"{val:.1f}" if val is not None and pd.notna(val) else "N/A"
        live_filters.append((desc, val_str, False))

    # --- Range Filter ---
    if settings.get('use_range_filter', False):
        val = last_row.get('RangePct', 0.5) * 100
        r_min = settings.get('range_min', 0)
        r_max = settings.get('range_max', 100)
        live_filters.append((f"Close in {r_min}-{r_max}% of range", f"{val:.0f}%", False))
    
    # --- MA Consecutive Filters ---
    for maf in settings.get('ma_consec_filters', []):
        length = maf['length']
        logic = maf['logic']
        consec = maf.get('consec', 1)
        col = f"SMA{length}"
        ma_val = last_row.get(col, 0)
        close_val = last_row['Close']
        
        desc = f"Close {logic.lower()} {length} SMA"
        if consec > 1:
            desc += f" ({consec}d)"
        # Show as pass/fail since it's essentially binary
        live_filters.append((desc, "✓", True))
    
    # --- Trend Filter ---
    trend = settings.get('trend_filter', 'None')
    if trend != 'None':
        # NOTE: check "Market"/"SPY" BEFORE the generic "200 SMA" branch.
        # "Market > 200 SMA" contains the substring "200 SMA", so a 200-SMA-first
        # ordering swallows it and prints the TICKER's close vs the TICKER's own
        # 200 SMA — misleading, since this filter actually evaluates SPY's regime
        # (Market_Above_SMA200) in check_signal.
        if "Market" in trend or "SPY" in trend:
            mkt_above = last_row.get('Market_Above_SMA200', False)
            live_filters.append((trend, "✓" if mkt_above else "✗", True))
        elif "200 SMA" in trend:
            sma200 = last_row.get('SMA200', 0)
            close = last_row['Close']
            if "Price >" in trend:
                live_filters.append((f"Price > 200 SMA", f"${close:.2f} vs ${sma200:.2f}", False))
            elif "Price <" in trend:
                live_filters.append((f"Price < 200 SMA", f"${close:.2f} vs ${sma200:.2f}", False))
            else:
                live_filters.append((trend, f"${close:.2f} vs ${sma200:.2f}", False))
    
    # --- Volume Filters ---
    if settings.get('use_vol', False):
        val = last_row.get('vol_ratio', 0)
        thresh = settings['vol_thresh']
        live_filters.append((f"Volume > {thresh:.1f}x avg", f"{val:.2f}x", False))
    
    if settings.get('use_vol_rank', False):
        val = last_row.get('vol_ratio_10d_rank', 50)
        logic = settings['vol_rank_logic']
        thresh = settings['vol_rank_thresh']
        live_filters.append((f"10D vol rank {logic} {thresh:.0f}th %ile", f"{val:.0f}", False))
    
    # --- Acc/Dist Counts ---
    if settings.get('use_acc_count_filter', False):
        window = settings.get('acc_count_window', 21)
        col = f'AccCount_{window}'
        val = last_row.get(col, 0) if col in df.columns else last_row.get('AccCount_21', 0)
        logic = settings['acc_count_logic']
        thresh = settings['acc_count_thresh']
        live_filters.append((f"Acc days {logic} {thresh} in {window}d", f"{val:.0f}", False))
    
    if settings.get('use_dist_count_filter', False):
        window = settings.get('dist_count_window', 21)
        col = f'DistCount_{window}'
        val = last_row.get(col, 0) if col in df.columns else last_row.get('DistCount_21', 0)
        logic = settings['dist_count_logic']
        thresh = settings['dist_count_thresh']
        live_filters.append((f"Dist days {logic} {thresh} in {window}d", f"{val:.0f}", False))
    
    # --- 52 Week High/Low ---
    if settings.get('use_52w', False):
        type_52w = settings['52w_type']
        first_inst = settings.get('52w_first_instance', False)
        desc = type_52w
        if first_inst:
            lookback = settings.get('52w_lookback', 21)
            desc += f" (first in {lookback}d)"
        live_filters.append((desc, "✓", True))
    
    if settings.get('exclude_52w_high', False):
        live_filters.append(("NOT at 52-week high", "✓", True))

    if settings.get('use_recent_52w', False):
        prefix = "Has NOT made" if settings.get('recent_52w_invert') else "Made"
        lb = settings.get('recent_52w_lookback', 21)
        live_filters.append((f"{prefix} 52w high in last {lb}d", "✓", True))

    if settings.get('use_recent_52w_low', False):
        prefix = "Has NOT made" if settings.get('recent_52w_low_invert') else "Made"
        lb = settings.get('recent_52w_low_lookback', 21)
        live_filters.append((f"{prefix} 52w low in last {lb}d", "✓", True))

    # --- VIX Filter ---
    if settings.get('use_vix_filter', False):
        val = last_row.get('VIX_Value', 0)
        vix_min = settings.get('vix_min', 0)
        vix_max = settings.get('vix_max', 100)
        live_filters.append((f"VIX between {vix_min:.0f}-{vix_max:.0f}", f"{val:.1f}", False))
    
    # --- Today's Return Filter ---
    if settings.get('use_today_return', False):
        val = last_row.get('today_return_atr', 0)
        ret_min = settings.get('return_min', -100)
        ret_max = settings.get('return_max', 100)
        live_filters.append((f"Today's move {ret_min:.1f} to {ret_max:.1f} ATR", f"{val:.2f} ATR", False))
    # --- ATR Return Filter (new config key) ---
    if settings.get('use_atr_ret_filter', False):
        val = last_row.get('today_return_atr', 0)
        live_filters.append((f"Net change {settings.get('atr_ret_min', -100):.1f} to {settings.get('atr_ret_max', 100):.1f} ATR", f"{val:.2f} ATR", False))

    # --- Range in ATR Filter ---
    if settings.get('use_range_atr_filter', False):
        atr = last_row.get('ATR', 1)
        range_val = (last_row['High'] - last_row['Low']) / atr if atr > 0 else 0
        logic = settings.get('range_atr_logic', 'Between')
        if logic == '>':
            live_filters.append((f"Range > {settings['range_atr_min']:.1f} ATR", f"{range_val:.2f} ATR", False))
        elif logic == '<':
            live_filters.append((f"Range < {settings['range_atr_max']:.1f} ATR", f"{range_val:.2f} ATR", False))
        else:
            live_filters.append((f"Range {settings['range_atr_min']:.1f}-{settings['range_atr_max']:.1f} ATR", f"{range_val:.2f} ATR", False))

    # --- Green Candle ---
    if settings.get('require_close_gt_open', False):
        is_green = last_row['Close'] > last_row['Open']
        live_filters.append(("Close > Open", "✓" if is_green else "✗", True))

    # --- Breakout Mode ---
    bk = settings.get('breakout_mode', 'None')
    if bk != 'None':
        live_filters.append((bk, "✓", True))

    # --- Vol > Prev ---
    if settings.get('vol_gt_prev', False):
        live_filters.append(("Volume > prev day", "✓", True))

    # --- ATH Filters ---
    if settings.get('use_ath', False):
        live_filters.append((settings.get('ath_type', 'Today is ATH'), "✓" if last_row.get('is_ath', False) else "✗", True))

    if settings.get('use_recent_ath', False):
        lookback = settings.get('ath_lookback_days', 21)
        recent = bool(df['is_ath'].rolling(window=lookback, min_periods=1).max().iloc[-1])
        inverted = settings.get('recent_ath_invert', False)
        prefix = "No ATH" if inverted else "Made ATH"
        live_filters.append((f"{prefix} in last {lookback}d", "✓" if (recent != inverted) else "✗", True))

    # --- Reference Ticker ---
    if settings.get('use_ref_ticker_filter', False) and settings.get('ref_filters'):
        ref_ticker = settings.get('ref_ticker', 'IWM')
        for rf in settings['ref_filters']:
            col = f"Ref_rank_ret_{rf['window']}d"
            val = last_row.get(col, 50)
            live_filters.append((f"{ref_ticker} {rf['window']}D rank {rf['logic']} {rf['thresh']:.0f}", f"{val:.0f}", False))

    # --- Cross-Sectional Rank (e.g. XSec 252D > 50 for LT Trend ST OS) ---
    if settings.get('use_xsec_filter', False) and settings.get('xsec_filters'):
        for xf in settings['xsec_filters']:
            window = xf['window']
            logic = xf['logic']
            thresh = xf['thresh']
            col = f"xsec_rank_ret_{window}d"
            val = last_row.get(col, 50.0)
            live_filters.append((f"XSec {window}D rank {logic} {thresh:.0f}", f"{val:.0f}", False))

    # --- Risk Dial Filters (fragility score gate) ---
    for df_filter in settings.get('dial_filters', []):
        dial_col = df_filter.get('dial')
        win = int(df_filter.get('window', 1))
        logic = df_filter.get('logic', '>')
        thresh = float(df_filter.get('thresh', 0))
        frag_df = _get_fragility_df_cached()
        if frag_df is None or dial_col not in frag_df.columns:
            live_filters.append((f"{dial_col} dial ({win}d avg) {logic} {thresh:.0f}", "n/a", False))
            continue
        dial_series = frag_df[dial_col]
        if win > 1:
            dial_series = dial_series.rolling(win, min_periods=win).mean()
        try:
            signal_date = pd.Timestamp(last_row.name).normalize()
            try:
                signal_date = signal_date.tz_localize(None)
            except (TypeError, AttributeError):
                pass
            val = float(dial_series.reindex([signal_date], method='ffill').iloc[0])
        except Exception:
            val = float('nan')
        val_str = f"{val:.1f}" if not pd.isna(val) else "n/a"
        live_filters.append((f"{dial_col} dial ({win}d avg) {logic} {thresh:.0f}", val_str, False))

    # --- Day of Week ---
    if settings.get('use_dow_filter', False):
        day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
        current = day_names.get(last_row.name.dayofweek, '?')
        live_filters.append(("Day of week filter", f"{current}", False))
    # --- ATR% Filter ---
    min_atr = settings.get('min_atr_pct', 0)
    max_atr = settings.get('max_atr_pct', 100)
    if min_atr > 0 or max_atr < 10:
        val = last_row.get('ATR_Pct', 0)
        live_filters.append((f"ATR% between {min_atr:.1f}-{max_atr:.1f}%", f"{val:.2f}%", False))
    
    return live_filters

def download_historical_data(tickers, start_date="2000-01-01"):
    if not tickers: return {}
    clean_tickers = list(set([str(t).strip().upper().replace('.', '-') for t in tickers]))

    data_dict = {}
    CHUNK_SIZE = 20
    MAX_RETRIES = 3
    total = len(clean_tickers)

    print(f"📥 Downloading data for {total} tickers...")

    for i in range(0, total, CHUNK_SIZE):
        chunk = clean_tickers[i : i + CHUNK_SIZE]
        batch_num = i // CHUNK_SIZE + 1
        total_batches = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"   Batch {batch_num}/{total_batches}...")

        for attempt in range(MAX_RETRIES):
            try:
                df = yf.download(chunk, start=start_date, group_by='ticker', auto_adjust=True, progress=False, threads=True)
                if df.empty: break

                if len(chunk) == 1:
                    ticker = chunk[0]
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
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
                break  # success — exit retry loop
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"   ⏳ Rate limited — retrying batch {batch_num} in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"   ⚠️ Batch {batch_num} failed after {MAX_RETRIES} attempts: {e}")

        time.sleep(1.5)

    return data_dict


def load_open_position_counts(ladder_strategy_names):
    """Build {(ticker, strategy_name): count} of currently-held positions
    for ladder sizing.

    Reads the dedicated 'Portfolio' tab of Trade_Signals_Log, which is
    refreshed nightly by daily_portfolio_report.write_portfolio_to_sheet().
    That snapshot already reflects only truly-open positions (from the
    portfolio simulation), so we don't need to filter by fill status or
    check exit dates here — every row is an open line item.

    The Portfolio tab uses the backtest schema with a 'Strategy' column
    (not 'Strategy_Name'), and already excludes LOC companion rows.

    Returns {} on any failure so the scan proceeds without the ladder overlay.
    """
    if not ladder_strategy_names:
        return {}

    gc = get_google_client()
    if not gc:
        return {}

    try:
        sh = gc.open("Trade_Signals_Log")
        ws = sh.worksheet("Portfolio")
        rows = ws.get_all_values()
    except Exception as e:
        print(f"⚠️ Ladder position lookup failed (reading Portfolio tab): {e}")
        return {}

    if not rows or len(rows) < 2:
        return {}

    headers = rows[0]
    try:
        name_idx = headers.index("Strategy")
        ticker_idx = headers.index("Ticker")
    except ValueError:
        print(f"⚠️ Portfolio tab missing Strategy/Ticker columns (got {headers[:5]}...)")
        return {}

    target_names = set(ladder_strategy_names)
    counts = {}

    for r in rows[1:]:
        if len(r) <= max(name_idx, ticker_idx):
            continue
        strat_name = r[name_idx].strip()
        if strat_name not in target_names:
            continue
        key = (r[ticker_idx].strip(), strat_name)
        counts[key] = counts.get(key, 0) + 1

    return counts


def load_open_position_notionals(cap_strategy_names):
    """Build {(ticker, strategy_name): open entry notional $} from the
    Portfolio tab, for the per-ticker concurrent notional cap (OLV,
    2026-07-20). Notional = Shares x Price (the engine's resolved entry),
    summed across stacked legs. Same conventions as
    load_open_position_counts: nightly snapshot of truly-open positions,
    fail-OPEN — any failure returns {} and the cap simply doesn't bind
    this run (a new leg then sizes uncapped; the engine still models the
    cap point-in-time, so drift is one leg, not systemic). KNOWN BOUND
    (review 2026-07-20): unfilled working limits are NOT positions and
    don't count, and OLV's T+3 fill window means up to THREE consecutive
    days' full-size limits can be simultaneously invisible to the cap —
    worst-case concurrent notional ~3x a single leg (~84% NAV on a very
    low-ATR name) if all fill late in their windows. The engine shares
    the blindness, so ledger and live agree; a working-order-aware cap
    (eq_order_entry-side) is the eventual fix.
    """
    if not cap_strategy_names:
        return {}

    gc = get_google_client()
    if not gc:
        return {}

    try:
        sh = gc.open("Trade_Signals_Log")
        ws = sh.worksheet("Portfolio")
        rows = ws.get_all_values()
    except Exception as e:
        print(f"⚠️ Notional-cap position lookup failed (reading Portfolio tab): {e}")
        return {}

    if not rows or len(rows) < 2:
        return {}

    headers = rows[0]
    try:
        name_idx = headers.index("Strategy")
        ticker_idx = headers.index("Ticker")
        shares_idx = headers.index("Shares")
        price_idx = headers.index("Price")
    except ValueError:
        print(f"⚠️ Portfolio tab missing Strategy/Ticker/Shares/Price columns (got {headers[:6]}...)")
        return {}

    target_names = set(cap_strategy_names)
    notionals = {}
    for r in rows[1:]:
        if len(r) <= max(name_idx, ticker_idx, shares_idx, price_idx):
            continue
        strat_name = r[name_idx].strip()
        if strat_name not in target_names:
            continue
        try:
            _n = abs(float(r[shares_idx])) * float(r[price_idx])
        except (TypeError, ValueError):
            continue
        key = (r[ticker_idx].strip(), strat_name)
        notionals[key] = notionals.get(key, 0.0) + _n

    return notionals


def stage_olv_vol_confirm_exits(master_dict=None):
    """Evaluate open OLV positions against the vol-confirmed stop and stage
    next-open MOO exit rows to the 'OLV_Exits' Sheets tab (2026-07-20).

    The rule (strategy_config OLV execution, stop_mode='vol_confirm_close'):
    exit at the NEXT open iff the last settled session CLOSED at/below
    entry - stop_atr*ATR AND its volume was >= stop_vol_mult x the trailing
    20d median (ex-that-day). Quiet closes through the level are held — the
    T+10 time-exit leg still bounds every position.

    Timing: the PM bookend scan (~22:00 UTC) evaluates today's just-settled
    close and stages Execute_On = next trading day; the AM scan (~4:47 ET,
    cache has --exclude-today) re-evaluates the SAME session with corrected
    data and restages Execute_On = today. The local pre-market runner
    olv_exit_moo.py (OneDrive trading_ibkr, Task Scheduler weekdays 9:10 AM
    ET) reads the tab and places true TIF=OPG MOO sells on BOTH accounts for
    rows with Execute_On == today, clamped to the actual held position
    (belt and suspenders). Before 2026-07-30 these rows rode the 9:31
    order_staging chain, which runs AFTER the open — never a real MOO.

    Per-leg contract: the Portfolio tab carries ONE ROW PER OPEN LEG
    (engine trades), so stacked positions are evaluated independently —
    each leg against its own entry, ATR and stop level. Every leg prints an
    explicit verdict line (CONFIRMED / no breach / quiet breach / entry-day
    / stale / unusable) so a silently-skipped leg is impossible to miss in
    the scan log, and staged exits + carry-forwards are keyed per
    (ticker, Time_Exit_Date) — never collapsed per symbol.

    Basis note: entry Price and ATR come from the Portfolio tab, which the
    nightly report RE-DERIVES from the current adjusted cache each evening —
    both sides of the comparison are the same cache lineage one vintage
    apart, so the dividend-adjustment relative-level rule holds (this is
    NOT a frozen dollar level vs re-pulled history).

    The tab is ALWAYS cleared+rewritten (even to empty) so stale exit rows
    can never resubmit. Fail-open: any error leaves positions to their
    time exits and never crashes the scan.
    """
    warnings = []

    def _warn(msg):
        warnings.append(msg)
        print(f"[OLV-EXIT] WARNING: {msg}")

    olv = next((s for s in STRATEGY_BOOK if s['name'] == 'Oversold Low Volume'), None)
    if olv is None or olv['execution'].get('stop_mode') != 'vol_confirm_close':
        return warnings

    ex = olv['execution']
    stop_atr = float(ex.get('stop_atr', 1.25))
    vol_mult = float(ex.get('stop_vol_mult', 1.5))

    gc = get_google_client()
    if not gc:
        _warn("no Google client — vol-confirm exit staging SKIPPED; open OLV "
              "positions have only their T+10 time exits")
        return warnings

    try:
        sh = gc.open("Trade_Signals_Log")
        rows = sh.worksheet("Portfolio").get_all_values()
    except Exception as e:
        _warn(f"could not read Portfolio tab ({e}) — exit staging SKIPPED")
        return warnings

    positions = []
    if rows and len(rows) >= 2:
        headers = rows[0]
        try:
            idx = {c: headers.index(c) for c in ("Strategy", "Ticker", "Shares",
                                                 "Price", "ATR", "Time Stop",
                                                 "Entry Date")}
            for r in rows[1:]:
                if len(r) <= max(idx.values()):
                    continue
                if r[idx["Strategy"]].strip() != "Oversold Low Volume":
                    continue
                try:
                    positions.append({
                        "ticker": r[idx["Ticker"]].strip().upper(),
                        "shares": int(abs(float(r[idx["Shares"]]))),
                        "entry": float(r[idx["Price"]]),
                        "atr": float(r[idx["ATR"]]),
                        "entry_date": pd.to_datetime(r[idx["Entry Date"]]).normalize(),
                        # Per-leg bracket key: stacked legs are separate OCA
                        # brackets live, each with its own stop level and a
                        # time-MKT leg at this date. eq_order_entry uses it
                        # to cancel exactly the confirmed leg's bracket.
                        "time_exit": str(pd.to_datetime(r[idx["Time Stop"]]).date()),
                    })
                except (TypeError, ValueError):
                    continue
        except ValueError as e:
            _warn(f"Portfolio tab schema unexpected ({e}) — exit staging SKIPPED")
            return warnings

    # Stacked-leg sanity: legs are identified downstream by
    # (Symbol, Time_Exit_Date) — the bracket key olv_exit_moo.py matches on.
    # Two open legs sharing both cannot be told apart; surface it loudly.
    _leg_key_counts = {}
    for _p in positions:
        _k = (_p["ticker"], _p["time_exit"])
        _leg_key_counts[_k] = _leg_key_counts.get(_k, 0) + 1
    for (_sym, _tx), _n in _leg_key_counts.items():
        if _n > 1:
            _warn(f"{_sym}: {_n} open legs share Time_Exit_Date {_tx} — "
                  f"downstream bracket matching cannot distinguish them; "
                  f"verify exits manually if either confirms")

    # Previously-staged rows: carried forward PER LEG for tickers we cannot
    # re-evaluate this run (stale bar), so an AM run with one lagging feed
    # can't wipe a valid PM-staged exit that is due today. Keyed by
    # (Symbol, Time_Exit_Date) — a symbol-level key would collapse stacked
    # legs and silently drop all but one of their staged exits. Only rows
    # still in the future (Execute_On >= today) are eligible to carry.
    today_norm = pd.Timestamp.now().normalize()
    prior_rows = {}
    try:
        _prev = sh.worksheet("OLV_Exits").get_all_records()
        for _r in _prev:
            _eo = pd.to_datetime(_r.get("Execute_On"), errors="coerce")
            if pd.notna(_eo) and _eo.normalize() >= today_norm:
                _pk = (str(_r.get("Symbol", "")).strip().upper(),
                       str(_r.get("Time_Exit_Date", "")).strip())
                prior_rows[_pk] = _r
    except Exception:
        prior_rows = {}

    def _carry_forward(pos, leg_label):
        """Re-stage the leg's own previously staged exit row, if any."""
        _pk = (pos["ticker"], str(pos["time_exit"]).strip())
        row = prior_rows.pop(_pk, None)
        if row is not None:
            exit_rows.append(row)
            print(f"[OLV-EXIT] {leg_label}: carried forward previously staged exit")

    def _frame_for(tkr):
        df = (master_dict or {}).get(tkr)
        if df is None:
            try:
                _raw = pd.read_parquet("data/master_prices.parquet",
                                       filters=[("ticker", "==", tkr)])
                _raw["date"] = pd.to_datetime(_raw["date"])
                df = _raw.sort_values("date").set_index("date")
            except Exception:
                df = None
        return df

    frames = {p["ticker"]: _frame_for(p["ticker"]) for p in positions}
    # Expected settled session = freshest bar across the evaluated frames
    # (AM cache: yesterday; PM cache: today). A ticker whose last bar lags
    # this is STALE and must not be re-evaluated off old data.
    _dates = [f.index[-1] for f in frames.values() if f is not None and len(f)]
    expected_session = max(_dates) if _dates else None

    exit_rows = []
    for pos in positions:
        tkr = pos["ticker"]
        _ed = pos["entry_date"].date() if pos["entry_date"] is not None else "?"
        # Per-leg label: every verdict line names the exact leg (entry date +
        # time-exit bracket key), so stacked positions are auditable leg by
        # leg in the scan log — a silent skip is impossible.
        leg = f"{tkr} leg[entry {_ed}, texit {pos['time_exit']}]"
        df = frames.get(tkr)
        if df is None or len(df) < 25 or "Volume" not in df.columns:
            _warn(f"{leg}: no usable price/volume history — cannot evaluate "
                  f"vol-confirm stop (held; T+10 time exit bounds)")
            _carry_forward(pos, leg)
            continue
        if expected_session is not None and df.index[-1] != expected_session:
            _warn(f"{leg}: last bar {df.index[-1].date()} lags expected session "
                  f"{expected_session.date()} — stale feed, NOT re-evaluated")
            _carry_forward(pos, leg)
            continue
        # Day-2 arming convention (book-wide, 2026-06-09): the engine's
        # vol-confirm loop starts at entry_idx+1, so an entry-day close is
        # NEVER a confirm. Skip legs whose entry is the evaluation session.
        if pos["entry_date"] is not None and pos["entry_date"] >= df.index[-1].normalize():
            print(f"[OLV-EXIT] {leg}: stop arms next session, not evaluated "
                  f"on the entry-day close")
            continue
        if pos["atr"] <= 0 or pos["entry"] <= 0:
            _warn(f"{leg}: degenerate entry/ATR ({pos['entry']}/{pos['atr']}) — "
                  f"cannot compute stop level (held; T+10 time exit bounds)")
            continue
        last = df.iloc[-1]
        stop_level = pos["entry"] - stop_atr * pos["atr"]
        if last["Close"] > stop_level:
            print(f"[OLV-EXIT] {leg}: close {last['Close']:.2f} > stop "
                  f"{stop_level:.2f} — no breach, held")
            continue
        med20 = df["Volume"].iloc[-21:-1].median()
        volx = (last["Volume"] / med20) if med20 and med20 > 0 else float("nan")
        if pd.isna(volx):
            # Breach with UNVERIFIABLE volume (NaN bar / degenerate median,
            # the SOXS-feed-bug class): held by rule, but this must be loud —
            # a corrupted feed can suppress every genuine capitulation exit.
            _warn(f"{leg}: closed {last['Close']:.2f} <= stop {stop_level:.2f} "
                  f"but volume is UNVERIFIABLE (NaN/degenerate med20) — held; "
                  f"REVIEW THE FEED")
            continue
        if volx < vol_mult:
            print(f"[OLV-EXIT] {leg}: closed {last['Close']:.2f} <= stop {stop_level:.2f} "
                  f"but volume {volx:.2f}x med20 < {vol_mult}x — HELD (quiet breach)")
            continue
        confirm_date = df.index[-1]
        execute_on = (confirm_date + TRADING_DAY).date()
        exit_rows.append({
            "Symbol": tkr,
            "Action": "SELL",
            "Quantity": pos["shares"],
            "Strategy_Ref": "Oversold Low Volume",
            "Confirm_Date": str(confirm_date.date()),
            "Execute_On": str(execute_on),
            "Time_Exit_Date": pos["time_exit"],
            "Entry_Date": str(_ed),
            "Stop_Level": round(stop_level, 2),
            "Confirm_Close": round(float(last["Close"]), 2),
            "Vol_X_Med20": round(float(volx), 2),
            "Staged_At": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        print(f"[OLV-EXIT] {leg}: CONFIRMED — closed {last['Close']:.2f} <= stop "
              f"{stop_level:.2f} on {volx:.1f}x volume -> MOO exit {execute_on}")

    cols = ["Symbol", "Action", "Quantity", "Strategy_Ref", "Confirm_Date",
            "Execute_On", "Time_Exit_Date", "Entry_Date", "Stop_Level",
            "Confirm_Close", "Vol_X_Med20", "Staged_At"]

    def _write_exits():
        try:
            ws = sh.worksheet("OLV_Exits")
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title="OLV_Exits", rows=50, cols=len(cols))
        ws.clear()
        data = [cols] + [[str(r.get(c, "")) for c in cols] for r in exit_rows]
        ws.update(values=data)

    try:
        _sheets_write_with_retry(
            f"stage {len(exit_rows)} OLV vol-confirm exit(s)", _write_exits)
        _per = {}
        for _p in positions:
            _per[_p["ticker"]] = _per.get(_p["ticker"], 0) + 1
        _breakdown = ", ".join(f"{t}x{n}" if n > 1 else t
                               for t, n in sorted(_per.items())) or "none"
        print(f"[OLV-EXIT] OLV_Exits tab written: {len(exit_rows)} exit(s), "
              f"{len(positions)} open OLV leg(s) evaluated ({_breakdown})")
    except Exception as e:
        _warn(f"failed to write OLV_Exits tab ({e}) — confirmed exits NOT "
              f"staged; positions fall back to their time exits")
    return warnings


def run_daily_scan(scope='liquid', moc_only=False, dry_run=False):
    """Run the daily scan against `scope` (liquid|overflow|all).

    moc_only=True restricts to MOC strategies (entry_type='Signal Close')
    and skips the overflow tier entirely. Used by intraday GHA runs.

    See build_effective_strategy_book() for full scope semantics.
    """
    if scope not in ('liquid', 'overflow', 'all'):
        raise ValueError(f"Invalid scope: {scope!r} (expected liquid|overflow|all)")

    print(f"--- Starting Daily Automated Scan (scope={scope}, moc_only={moc_only}) ---")
    sznl_map = load_seasonal_map()

    # Build the strategy list this run iterates over. For scope=liquid (the
    # default GHA path) this is every entry in STRATEGY_BOOK. For scope=overflow
    # it's the 6 overflow-eligible strategies with universes swapped to
    # CSV_UNIVERSE − LIQUID_PLUS_COMMODITIES and per-strategy bps overrides
    # applied. scope=all is liquid + overflow concatenated.
    effective_book = build_effective_strategy_book(scope, moc_only=moc_only)
    if not effective_book:
        print(f"⚠️ scope={scope} (moc_only={moc_only}) produced an empty strategy book — nothing to scan.")
        return

    # 1. Gather Tickers
    all_tickers = set()
    for strat in effective_book:
        all_tickers.update(strat['universe_tickers'])
        # Spot-index alias: ensure tradeable ETF (SPY/QQQ) is downloaded so its
        # ATR/close/indicators are available at substitution time.
        for _spot, _tradeable in SPOT_TO_TRADEABLE.items():
            if _spot in strat['universe_tickers']:
                all_tickers.add(_tradeable)
        s = strat['settings']
        if s.get('use_market_sznl'): all_tickers.add(s.get('market_ticker', '^GSPC'))
        if "Market" in s.get('trend_filter', ''): all_tickers.add(s.get('market_ticker', 'SPY'))
        if "SPY" in s.get('trend_filter', ''): all_tickers.add("SPY")
        if s.get('use_vix_filter', False): all_tickers.add("^VIX")  # VIX for strategies that need it
        if s.get('use_ref_ticker_filter', False) and s.get('ref_ticker'):
            all_tickers.add(s['ref_ticker'].replace('.', '-'))
    
    # 2. Download Data
    # Cache-first for ALL tickers (liquid + overflow + strategy universes):
    # read OHLCV from master_prices.parquet and hit yfinance ONLY for names the
    # cache genuinely lacks. The cache is the deterministic, resilient source —
    # it retains the prior session's bars even when the morning
    # update_master_prices job can't fetch a day, so the scan never silently
    # slips to a stale evaluation date.
    #
    # History: the liquid tier (incl. the 3x-ETF universe) used to pull live
    # from yfinance every run. On 2026-06-11 the pre-market run's live pull came
    # back with a stale last bar (June 9), so EVERY liquid strategy found zero
    # signals and the morning scan then wiped the staged liquid orders (DUST/JDST
    # etc.) that the post-close scan had correctly staged. Reading from the cache
    # removes that single point of failure; yfinance is now only a fallback for
    # carets / freshly-added names not yet backfilled, so nothing disappears.
    master_dict = load_master_prices_dict(list(all_tickers))
    if master_dict:
        print(f"📦 Loaded master_prices.parquet: {len(master_dict)} tickers from cache")
    else:
        print("⚠️ master_prices.parquet unavailable — falling back to yfinance for all tickers")
    _have = {k.replace('.', '-').upper() for k in master_dict}
    _yf_tickers = [t for t in all_tickers if t.replace('.', '-').upper() not in _have]
    if _yf_tickers:
        print(f"🌐 {len(_yf_tickers)} ticker(s) not in cache — fetching live from yfinance: "
              f"{sorted(_yf_tickers)[:12]}{' ...' if len(_yf_tickers) > 12 else ''}")
        yf_dict = download_historical_data(_yf_tickers)
        master_dict.update(yf_dict)
    
    # -------------------------------------------------------------------------
    # 3. DATE VALIDATION & ENFORCEMENT (Morning vs. Day Logic)
    # -------------------------------------------------------------------------
    # This logic ensures that if you run at 5:30 AM, the script STRICTLY uses 
    # yesterday's closing data, deleting any "ghost" bars from today.
    
    eastern = pytz.timezone('America/New_York')
    now_eastern = datetime.datetime.now(eastern)
    current_date = now_eastern.date()
    
    # Define Market Open (9:30 AM EST)
    market_open_time = now_eastern.replace(hour=9, minute=30, second=0, microsecond=0)
    
    market_close_time = now_eastern.replace(hour=16, minute=0, second=0, microsecond=0)
    # Intraday partial = market is open AND today's session hasn't closed yet.
    # During this window, today's bar volume is incomplete, so strategies that
    # filter on volume can have false negatives. LT Trend ST OS specifically
    # relaxes its 1.25× volume requirement to 1.0× during this window.
    is_intraday_partial = (market_open_time <= now_eastern < market_close_time)
    is_morning_run = now_eastern < market_open_time

    if is_morning_run:
        # Morning Run (e.g. 5:30 AM): Strict cutoff at YESTERDAY'S close.
        # We must remove any partial data stamped with today's date.
        expected_data_date = (pd.Timestamp(current_date) - TRADING_DAY).date()
        print(f"🌅 Morning Run (Pre-Market): Enforcing data cutoff at {expected_data_date}")
    else:
        # Day Run (e.g. 10:00 AM): Allow today's partial bar.
        expected_data_date = current_date
        print(f"☀️ Day Run (Post-Open): Allowing data through {expected_data_date}")
    if is_intraday_partial:
        print(f"🕐 Intraday partial-bar window — LT Trend ST OS will use vol_thresh 1.0× (else 1.25×)")

    # 3c. Morning-run-only: read pre-existing overflow OVS quantities from the
    # 'Overflow' tab. When the liquid scan fires Overbot Vol Spike signals in the
    # 5 AM run, any ticker that already has an Overflow OVS row from yesterday's
    # scope=overflow run gets its daily_scan shares CAPPED to that quantity —
    # avoids doubling exposure when the same ticker fires in both tiers.
    # User-acknowledged: this may cross the aggregate bps risk threshold.
    # Must read 'Overflow' (not 'Order_Staging'): save_staging_orders routes
    # Scan_Source='Overflow' rows to the Overflow tab (tier_filter), so
    # Order_Staging only ever holds Liquid rows — the old read there was inert.
    # No-op today because the tiers are disjoint (static overflow = CSV − liquid),
    # but the control now fires correctly if a name ever lands in both.
    overflow_ovs_quantities = {}
    if is_morning_run and scope in ('liquid', 'all'):
        try:
            _gc = get_google_client()
            if _gc:
                _sh = _gc.open("Trade_Signals_Log")
                try:
                    _ws = _sh.worksheet("Overflow")
                    _rows = _ws.get_all_records()
                    for _r in _rows:
                        if str(_r.get('Strategy_Ref', '')).strip() != "Overbot Vol Spike":
                            continue
                        _tkr = str(_r.get('Symbol', '')).strip().upper()
                        try:
                            _qty = int(float(_r.get('Quantity', 0)))
                        except (ValueError, TypeError):
                            _qty = 0
                        if _tkr and _qty > 0:
                            # Multiple rows for same ticker → keep max (conservative cap)
                            overflow_ovs_quantities[_tkr] = max(overflow_ovs_quantities.get(_tkr, 0), _qty)
                    if overflow_ovs_quantities:
                        print(f"📋 Morning run: {len(overflow_ovs_quantities)} Overflow OVS tickers in Overflow tab — liquid OVS shares will cap to match")
                except Exception as _e:
                    print(f"ℹ️ No Overflow rows to read for OVS size match (or empty): {_e}")
        except Exception as e:
            print(f"⚠️ Could not read Overflow tab for OVS size match: {e}")

    validated_dict = {}
    for ticker, df in master_dict.items():
        if df is None or df.empty:
            continue
            
        # Check the date of the last row
        last_row_date = df.index[-1].date()
        
        # If the last row is newer than allowed (e.g. today's date during a morning run), trim it
        if last_row_date > expected_data_date:
            df = df.iloc[:-1]
            
        # If dataframe is empty after trimming, skip it
        if df.empty:
            continue
            
        validated_dict[ticker] = df

    # Replace the master dictionary with the strictly validated version
    master_dict = validated_dict

    # FAIL-LOUD FRESHNESS GATE (2026-07-16). The trim above only removes bars
    # NEWER than expected — it never asserts the cache actually HAS the
    # expected bar. Two consecutive updater failures leave the whole cache one
    # session stale, and the scan would silently re-detect the prior day's
    # already-traded signals with fresh Scan_Dates (order_staging would then
    # resubmit them at stale limit/ATR levels). Abort BEFORE any staging write
    # so the GHA run goes red instead. Keyed on the freshest bar across the
    # whole cache: if no ticker has the expected bar, the updater didn't run.
    # Intraday manual runs are allowed one session of slack (today's bar
    # legitimately doesn't exist until the PM updater lands it).
    if not master_dict:
        raise RuntimeError(
            "Freshness gate: zero tickers survived date validation — price "
            "cache is empty or entirely stale. Aborting before staging."
        )
    _freshest = max(df.index[-1].date() for df in master_dict.values())
    _allowed = {expected_data_date}
    if is_intraday_partial:
        _allowed.add((pd.Timestamp(expected_data_date) - TRADING_DAY).date())
    if _freshest not in _allowed:
        raise RuntimeError(
            f"Freshness gate: freshest bar in the price cache is {_freshest}, "
            f"expected {sorted(_allowed)} — updater likely failed; aborting "
            f"before any staging write so already-traded signals are not "
            f"re-staged at stale levels."
        )
    print(f"✅ Data dates validated. (Processing {len(master_dict)} tickers, freshest bar {_freshest})\n")
    # -------------------------------------------------------------------------

    # 3b. Load fragility score for sizing adjustment
    # Uses 10d MA of 63d fragility, lagged 1 day (today's score → tomorrow's trades)
    # PIT-or-nothing: rd2_fragility.parquet is the ONLY sizing source. The old
    # fallback to rd2_fragility_ts.parquet served a non-PIT recompute vintage
    # (drifts up to ~7 pts, wrong smoothing basis) into live sizing whenever the
    # PIT file was missing — removed 2026-07-16. Missing/stale PIT = 1.0x.
    FRAG_CACHE = os.path.join(current_dir, "data", "rd2_fragility.parquet")
    frag_score = None  # populated below if fragility cache loads AND is fresh

    frag_path = FRAG_CACHE if os.path.exists(FRAG_CACHE) else None
    if frag_path:
        try:
            frag_df = pd.read_parquet(frag_path)
            if '63d' in frag_df.columns:
                frag_series = frag_df['63d'].dropna().rolling(10, min_periods=1).mean()
                if not frag_series.empty:
                    # Staleness guard: a fragility reading older than FRAG_STALE_TD
                    # trading days means the producer (risk_report.yml) is broken or
                    # missed runs. Fall back to 1.0x rather than boost/throttle on a
                    # regime that may no longer hold. Leaves frag_score=None so the
                    # summary email omits the (misleading) stale reading.
                    _last_frag_dt = pd.Timestamp(frag_series.index[-1]).normalize()
                    try:
                        _last_frag_dt = _last_frag_dt.tz_localize(None)
                    except (TypeError, AttributeError):
                        pass
                    _today = pd.Timestamp.today().normalize()
                    try:
                        _age_td = int(np.busday_count(_last_frag_dt.date(), _today.date()))
                    except (ValueError, AttributeError):
                        _age_td = FRAG_STALE_TD + 1
                    if _age_td > FRAG_STALE_TD:
                        print(f"🚨 STALE FRAGILITY: last score {_last_frag_dt.date()} is "
                              f"{_age_td} trading days old (> {FRAG_STALE_TD}) — falling back "
                              f"to 1.0x sizing (producer may be broken)")
                    else:
                        frag_score = float(frag_series.iloc[-1])
                        print(f"🛡️ Fragility score: 63d 10d-MA = {frag_score:.1f} "
                              f"(per-strategy frag_risk_bands apply at sizing)")
                else:
                    print("⚠️ Fragility series empty after processing — using 1.0x")
            else:
                print("⚠️ 63d column not found in fragility cache — using 1.0x")
        except Exception as e:
            print(f"⚠️ Could not load fragility data: {e} — using 1.0x")
    else:
        print("⚠️ No fragility cache found — using 1.0x sizing")

    # 3b2. P/C FEAR STATE (2026-08-05) — selects the family band TABLE in
    # sizing 2b (pc_fear_bands carriers). Lag-1 by construction: the state
    # for the scanned bar date uses the newest cboe_putcall.parquet row dated
    # <= bar - 1 bday (the nightly scrape only ever has D-1 — measured
    # 2026-08-05). Stale (> 3 bd) fails CLOSED to the incumbent
    # frag_risk_bands table. Mirrored point-in-time in strat_backtester 3b3.
    pc_state = pc_fear.fear_state_asof(expected_data_date)
    if pc_state['state'] == 'stale':
        print(f"🚨 P/C FEAR STATE STALE (data {pc_state['data_date']}, "
              f"age {pc_state['age_bd']} bd) — family bands fail closed to "
              f"incumbent frag_risk_bands")
    else:
        print(f"🧭 P/C fear state: {pc_state['pct']:.0f}%ile "
              f"(10d-MA equity P/C, data through {pc_state['data_date']}, "
              f"{pc_state['age_bd']} bd old) — fear "
              f"{pc_state['state'].upper()}")

    # 4. Prepare VIX Series (for strategies with VIX filter)
    vix_df = master_dict.get('^VIX')
    vix_series = None
    if vix_df is not None and not vix_df.empty:
        temp_vix = vix_df.copy()
        temp_vix.columns = [c.capitalize() for c in temp_vix.columns]
        if temp_vix.index.tz is not None:
            temp_vix.index = temp_vix.index.tz_localize(None)
        vix_series = temp_vix['Close']
    
    # 4b. Build Cross-Sectional Rank Matrices (if any strategy uses xsec filters or or_filter_groups)
    xsec_rank_matrices = None
    xsec_windows_needed = set()
    for strat in effective_book:
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
        print(f"   Done — {len(next(iter(xsec_rank_matrices.values())).columns)} tickers ranked")

    all_signals = []
    error_tickers = []  # (ticker, reason) tuples for email reporting

    # Overflow universe metadata (addv_63d etc.) for the ADV participation cap.
    # {} when the parquet is absent → the cap is a no-op (current behavior).
    _overflow_meta = load_overflow_meta()

    # 4a. Load ATR seasonal ranks once if any strategy uses them
    _uses_atr_sznl = (
        any(s['settings'].get('atr_sznl_filters') for s in effective_book)
        or any(s['name'] == "Overbot Vol Spike" for s in effective_book)  # uses atr_sznl_5d for the 1.5x sizer
    )
    atr_sznl_map = load_atr_seasonal_map() if _uses_atr_sznl else {}
    if _uses_atr_sznl:
        if atr_sznl_map:
            print(f"📊 Loaded ATR seasonal ranks: {len(atr_sznl_map)} tickers")
        else:
            print(f"⚠️ atr_seasonal_ranks.parquet not found — atr_sznl_filters will match nothing")

    # 4b. Earnings calendar — used by OVS earnings blackout (±10 trading days).
    # Tickers with no earnings data (commodity ETFs, indices, futures, FX) pass
    # through automatically (NaN-as-True). Empty dict ⇒ filter is a silent no-op.
    _uses_earnings_blackout = any(
        s['execution'].get('earnings_blackout_td')
        or s['execution'].get('earnings_size_override')
        for s in effective_book
    )
    earnings_map = load_earnings_dates_map() if _uses_earnings_blackout else {}
    if _uses_earnings_blackout:
        if earnings_map:
            print(f"📊 Loaded earnings calendar: {len(earnings_map)} tickers")
        else:
            print(f"⚠️ data/earnings_calendar.parquet not found — earnings blackout disabled")

    # 4c2. Per-ticker notional-cap open state (OLV, 2026-07-20): open entry
    # notional per (ticker, strategy) for strategies carrying
    # execution['ticker_notional_cap']. Fail-open — {} means the cap simply
    # doesn't bind this run.
    _cap_strats = {s['name'] for s in effective_book
                   if s['execution'].get('ticker_notional_cap')}
    open_notionals = load_open_position_notionals(_cap_strats)

    # 4c. Ladder position counts — counts currently-held filled primary signals
    # per (ticker, strategy) so repeat signals size up on each successive day.
    ladder_strats = {s['name'] for s in effective_book if s['execution'].get('ladder_multipliers')}
    ladder_counts = load_open_position_counts(ladder_strats)
    if ladder_counts:
        print(f"📈 Ladder: {len(ladder_counts)} open ({', '.join(f'{t}/{s[:12]}={c}' for (t, s), c in list(ladder_counts.items())[:5])}{'...' if len(ladder_counts) > 5 else ''})")

    # Per-ticker indicator memo shared across the strategy loop — see
    # memoized_indicators. Keys: (ticker, market-series source, ref-config).
    _ind_memo = {}

    # 5. Run Strategies
    for strat in effective_book:
        _scan_source = strat.get('_scan_source', 'Liquid')
        print(f"Running: {strat['name']} [{_scan_source}]...")
        
        # Prepare Market Series
        mkt_ticker = strat['settings'].get('market_ticker', 'SPY')
        mkt_df = master_dict.get(mkt_ticker)
        if mkt_df is None: mkt_df = master_dict.get('SPY')
        
        market_series = None
        if mkt_df is not None:
            temp_mkt = mkt_df.copy()
            temp_mkt['SMA200'] = temp_mkt['Close'].rolling(200).mean()
            market_series = temp_mkt['Close'] > temp_mkt['SMA200']
        # Prepare Reference Ticker Ranks (if needed)
        ref_ticker_ranks = None
        _ref_memo_key = None
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
                if ref_ticker_ranks:
                    _ref_memo_key = (ref_ticker_key, tuple(sorted(ref_ticker_ranks)))
        # Indicator-memo key parts: the market series' actual source ticker
        # (mkt_ticker when present in the cache, else the SPY fallback, else
        # None) and the ref-rank config. Two strategies with identical parts
        # produce byte-identical indicator frames, so they share one.
        if master_dict.get(mkt_ticker) is not None:
            _mkt_memo_key = mkt_ticker
        elif master_dict.get('SPY') is not None:
            _mkt_memo_key = 'SPY'
        else:
            _mkt_memo_key = None
        signals = []
        for ticker in strat['universe_tickers']:
            t_clean = ticker.replace('.', '-')
            df = master_dict.get(t_clean)
            if df is None:
                error_tickers.append((t_clean, "No data returned"))
                continue
            if len(df) < 250:
                error_tickers.append((t_clean, f"Insufficient history ({len(df)} bars)"))
                continue
            
            try:
                calc_df = memoized_indicators(
                    _ind_memo, (t_clean, _mkt_memo_key, _ref_memo_key),
                    df, sznl_map, t_clean, market_series, vix_series,
                    ref_ticker_ranks, xsec_rank_matrices, atr_sznl_map)

                # LT Trend ST OS intraday volume relaxation: during partial-bar
                # window (market open through 4 PM ET), today's bar volume is
                # incomplete. Drop the 1.25× threshold to 1.0× so we don't get
                # false negatives — the assumption is the rest of the session
                # will bring volume up to the strict threshold by close.
                _eff_settings = strat['settings']
                if (is_intraday_partial
                        and strat['name'] == "LT Trend ST OS"
                        and _eff_settings.get('use_vol')):
                    _eff_settings = dict(_eff_settings)
                    _eff_settings['vol_thresh'] = 1.0

                if check_signal(calc_df, _eff_settings, sznl_map, ticker=t_clean):
                    # Earnings blackout (OVS-only currently). Reject signals
                    # within ±N trading days of earnings. NaN passes through —
                    # commodity ETFs / futures / indices have no earnings data
                    # and shouldn't be silently killed by a stock-only filter.
                    _no_earn_cov = False
                    _eb_window = strat['execution'].get('earnings_blackout_td')
                    if _eb_window and earnings_map:
                        _e_arr = earnings_map.get(t_clean.upper())
                        # Overflow tier (R-T5): an OVS short on a thin small-cap
                        # with NO earnings coverage is real gap risk. SOFT drop —
                        # still stage it, but flag the gap (Sizing_Notes +
                        # Earnings_Cov column) so it can be eyeballed before fill.
                        # Liquid tier keeps silent passthrough (commodity ETFs /
                        # indices legitimately have no earnings rows).
                        if strat.get('_scan_source') == 'Overflow' and (_e_arr is None or len(_e_arr) == 0):
                            _no_earn_cov = True
                        elif in_blackout(calc_df.index[-1], _e_arr, window=_eb_window):
                            continue

                    # Sector loss gate (OLV, 2026-07-02): skip a signal into a
                    # sector where this strategy just realized heavy losses —
                    # the dip is trending, not bouncing (June 2026 oil cluster).
                    _slg_block, _slg_note = sector_gate_blocked(
                        strat['name'], strat['execution'], t_clean, calc_df.index[-1])
                    if _slg_block:
                        print(f"   ⛔ {t_clean}: sector loss gate — {_slg_note}")
                        continue

                    # Spot-index alias: detection happens on ^GSPC/^NDX (purer price),
                    # but staging happens on SPY/QQQ. Recompute calc_df against the
                    # tradeable so all downstream values (entry, ATR, stop/target,
                    # share count, notional) reflect the ETF as a 1:1 replacement.
                    if t_clean in SPOT_TO_TRADEABLE:
                        tradeable = SPOT_TO_TRADEABLE[t_clean]
                        tradeable_clean = tradeable.replace('.', '-')
                        sub_df = master_dict.get(tradeable_clean)
                        if sub_df is None or len(sub_df) < 250:
                            error_tickers.append((t_clean, f"Signal fired but tradeable {tradeable} unavailable"))
                            continue
                        calc_df = memoized_indicators(
                            _ind_memo, (tradeable_clean, _mkt_memo_key, _ref_memo_key),
                            sub_df, sznl_map, tradeable_clean, market_series,
                            vix_series, ref_ticker_ranks, xsec_rank_matrices,
                            atr_sznl_map)
                        print(f"   🔁 {t_clean} signal → staging as {tradeable}")
                        ticker = tradeable
                        t_clean = tradeable_clean

                    last_row = calc_df.iloc[-1]

                    # 1. Entry Confirmation Check
                    entry_conf_bps = strat['settings'].get('entry_conf_bps', 0)
                    entry_mode = strat['settings'].get('entry_type', 'Signal Close')

                    if entry_mode == 'Signal Close' and entry_conf_bps > 0:
                        threshold = last_row['Open'] * (1 + entry_conf_bps/10000.0)
                        if last_row['High'] < threshold: continue

                    atr = last_row['ATR']
                    
                    # ---------------------------------------------------------
                    # 2. DYNAMIC RISK SIZING LOGIC (Synced)
                    # ---------------------------------------------------------
                    base_risk = strat['execution']['risk_per_trade']
                    risk = base_risk 

                    sizing_note = "Standard (1.0x)"

                    if strat['name'] == "Weak Close Decent Sznls":
                        sznl_val = last_row.get('Sznl', 0)
                        if sznl_val >= 65:
                            risk = risk * 1.5
                            sizing_note = f"High Sznl ({sznl_val:.0f}) = 1.5x"
                        elif sznl_val >= 50:
                            risk = risk * 1.0
                            sizing_note = f"Med Sznl ({sznl_val:.0f}) = 1.0x"
                        elif sznl_val >= 33:
                            risk = risk * 0.66
                            sizing_note = f"Low Sznl ({sznl_val:.0f}) = 0.66x"

                    # OVS sizing is governed by the 2-path scheme in
                    # order_staging.py: path-1 (decisive 0.25 ATR gap) → 40 bps,
                    # path-2 (mild gap, open ≤ close + 0.25 ATR) → 8 bps with a
                    # 1% aggregate path-2 cap, no gap → skip. Scanner stages at
                    # path-1 nominal; order_staging downsizes path-2 rows.
                    # ---------------------------------------------------------

                    # 2b. FRAGILITY RISK BANDS (per-strategy, 2026-07-02;
                    # P/C-fear table selection 2026-08-05). Strategies with
                    # execution['pc_fear_bands'] pick their band table by the
                    # fear state (3b2 above): ON -> 1.25x below dial 50 / 1.0x
                    # above; OFF -> 1.0x below / 0.0x above (signal stages at
                    # 0 shares — visible in email + tabs, never ordered);
                    # stale P/C -> incumbent frag_risk_bands. Strategies
                    # without the field keep plain frag_risk_bands; missing/
                    # stale DIAL scores still run 1.0x (fail-open, unchanged).
                    # Mirrored in strat_backtester 3b3 (point-in-time replay).
                    _fbm = frag_band_mult(strat['execution'], frag_score,
                                          pc_state=pc_state)
                    if strat['execution'].get('pc_fear_bands'):
                        risk = risk * _fbm
                        sizing_note += " | " + pc_fear.sizing_note(
                            pc_state, frag_score, _fbm)
                        if _fbm == 0.0:
                            sizing_note += " — ZEROED (hi-frag, no P/C washout)"
                            print(f"   🚫 {t_clean} {strat['name']}: zeroed "
                                  f"(dial {frag_score:.0f} >= 50, P/C fear OFF)")
                    elif _fbm != 1.0:
                        risk = risk * _fbm
                        sizing_note += f" | Frag band ({frag_score:.0f}): {_fbm:.2f}x"

                    # 2c. SIGNAL-RECENCY LADDER (OLV, 2026-07-30) — rung =
                    # count of this ticker's SIGNAL days (shared mask,
                    # fill-independent) in the trailing window_td sessions
                    # before today. Replaced the open-position-count ladder
                    # for OLV: no reset on chain exit inside the window, no
                    # blindness to still-unfilled working limits. The mult is
                    # carried into 2d — the earnings override composes with
                    # it instead of clobbering it. Mirrored in
                    # strat_backtester's candidate-recency pre-pass.
                    _recency_mult = 1.0
                    _srl = strat['execution'].get('signal_recency_ladder')
                    if _srl:
                        _srl_mask = live_signal_mask(calc_df, _eff_settings,
                                                     sznl_map, ticker=t_clean)
                        _prior = recency_prior_from_mask(
                            _srl_mask, _srl.get('window_td', 21))
                        _srl_mults = _srl['mults']
                        _rung = min(_prior, len(_srl_mults) - 1)
                        _recency_mult = float(_srl_mults[_rung])
                        risk = risk * _recency_mult
                        sizing_note += (f" | Recency rung {_rung + 1} "
                                        f"({_recency_mult:.2f}x, {_prior} prior "
                                        f"signal(s)/{_srl.get('window_td', 21)}td)")

                    # 2c-old. LADDER SIZING (open-position count) — dormant
                    # machinery, no carriers since the OLV swap above.
                    ladder_mults = strat['execution'].get('ladder_multipliers')
                    if ladder_mults:
                        open_count = ladder_counts.get((t_clean, strat['name']), 0)
                        rung_idx = min(open_count, len(ladder_mults) - 1)
                        ladder_mult = ladder_mults[rung_idx]
                        risk = risk * ladder_mult
                        sizing_note += f" | Ladder rung {rung_idx + 1} ({ladder_mult:.2f}x, {open_count} open)"

                    # 2c2. CYCLE-YEAR RISK MULTIPLIER (e.g. OVS midterm 0.75x).
                    # execution['cycle_risk_mults'] = {year%4: mult}. Applies to
                    # OVS too (unlike the fragility multiplier) — but note the
                    # OVS P1 fixed-dollar resize in order_staging.py carries its
                    # own OVS_CYCLE_MULTS mirror, since it clobbers Risk_Amt.
                    _cyc = strat['execution'].get('cycle_risk_mults')
                    if _cyc:
                        _cm = float(_cyc.get(last_row.name.year % 4, 1.0))
                        if _cm != 1.0:
                            risk = risk * _cm
                            sizing_note += f" | Cycle yr%4={last_row.name.year % 4}: {_cm:.2f}x"

                    # 2d. EARNINGS SIZE OVERRIDE — replace the BASE risk with
                    # the configured bps when signal_date sits in the offset
                    # range, then re-apply the signal-recency mult (2026-07-30:
                    # composes with 2c instead of clobbering it — a
                    # first-iteration pre-earnings OLV signal is 10 x 0.5 bps).
                    # Every OTHER multiplier (frag, tier) is still clobbered.
                    # NaN offsets (commodity ETFs / indices / futures with no
                    # earnings data) bypass the override.
                    _eo = strat['execution'].get('earnings_size_override')
                    if _eo and earnings_map:
                        _e_arr = earnings_map.get(t_clean.upper())
                        _off = signed_offset(last_row.name, _e_arr)
                        if pd.notna(_off) and _eo['min_td'] <= _off <= _eo['max_td']:
                            _ovr_bps = _eo['risk_bps']
                            _prior_note = sizing_note
                            risk = ACCOUNT_VALUE * _ovr_bps / 10000 * _recency_mult
                            sizing_note = (f"Pre-earnings override: {_ovr_bps} bps"
                                           + (f" x {_recency_mult:.2f} recency" if _recency_mult != 1.0 else "")
                                           + f" (offset {int(_off):+d} TD; default was {_prior_note})")

                    # 3. Calculate Prices & Shares
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

                    # Morning-run OVS size match: if this ticker already has an OVS
                    # order in the Overflow tab, cap the daily_scan shares to that
                    # quantity (never increase, only reduce). Keeps us from staging
                    # a full-size main order on top of an existing overflow order.
                    if (is_morning_run
                            and strat['name'] == "Overbot Vol Spike"
                            and t_clean in overflow_ovs_quantities):
                        _of_qty = overflow_ovs_quantities[t_clean]
                        if shares > _of_qty:
                            _orig_shares = shares
                            shares = _of_qty
                            # Recompute risk $ to reflect capped shares
                            risk = shares * dist
                            sizing_note = f"{sizing_note} | OVS morning-match overflow: {_orig_shares} → {shares} shares"
                            print(f"   🔄 {t_clean}: OVS shares {_orig_shares} → {shares} (capped to Overflow)")

                    # ADV participation cap (R-T3): never let a single overflow
                    # position exceed ADV_PARTICIPATION_CAP × 63d ADDV in notional.
                    # No-op when meta is absent or the name has no ADDV. Stamped
                    # onto the row so order_staging can re-enforce if it wishes.
                    _addv_63d = None
                    if strat.get('_scan_source') == 'Overflow' and _overflow_meta:
                        _info = _overflow_meta.get(t_clean.upper())
                        if _info is not None:
                            _addv_63d = _info.get('addv_63d')
                            _cap = adv_share_cap(_addv_63d, entry, ADV_PARTICIPATION_CAP)
                            if _cap is not None and 0 < _cap < shares:
                                _orig = shares
                                shares = _cap
                                risk = shares * dist
                                sizing_note = (f"{sizing_note} | ADV cap "
                                               f"({ADV_PARTICIPATION_CAP:.0%} ADDV): {_orig} → {shares} sh")

                    # Per-ticker concurrent notional cap (OLV, 2026-07-20):
                    # stacked legs in ONE single-stock ticker may not exceed
                    # pct_nav x NAV in entry notional; the new leg is scaled
                    # down (or zeroed) to fit. ETFs pass through via the
                    # exempt list. Open state = filled positions in the
                    # nightly Portfolio snapshot (load_open_position_notionals,
                    # fail-open). Mirrored point-in-time by the engine cap in
                    # strat_backtester — change together. Guard:
                    # tests/test_olv_stop_and_cap.py.
                    _tnc = strat['execution'].get('ticker_notional_cap')
                    if _tnc and shares > 0:
                        _tnc_exempt = set(_tnc.get('exempt') or ())
                        if t_clean.upper() not in _tnc_exempt:
                            _tnc_cap = float(_tnc['pct_nav']) * ACCOUNT_VALUE
                            _tnc_open = open_notionals.get((t_clean, strat['name']), 0.0)
                            _tnc_new = shares * entry
                            if _tnc_open + _tnc_new > _tnc_cap:
                                _tnc_room = max(0.0, _tnc_cap - _tnc_open)
                                _orig_sh = shares
                                shares = int(_tnc_room / entry) if entry > 0 else 0
                                risk = shares * dist
                                sizing_note = (
                                    f"{sizing_note} | Notional cap "
                                    f"{_tnc['pct_nav']:.0%} NAV "
                                    f"(${_tnc_open:,.0f} open): {_orig_sh} → {shares} sh")
                                print(f"   🧢 {t_clean}: notional cap "
                                      f"{_tnc['pct_nav']:.0%} NAV — {_orig_sh} → {shares} shares "
                                      f"(${_tnc_open:,.0f} already open)")
                                if shares <= 0:
                                    print(f"   ⛔ {t_clean}: notional cap full — signal skipped")
                                    continue

                    entry_mode = strat['settings'].get('entry_type', 'Signal Close')
                    hold_days = strat['execution']['hold_days']

                    # Determine the effective Entry Date
                    if entry_mode == "Signal Close":
                        effective_entry_date = last_row.name
                    else:
                        effective_entry_date = last_row.name + TRADING_DAY

                    # Calculate Exit Date
                    exit_date = (effective_entry_date + (TRADING_DAY * hold_days)).date()
                    
                    # Build enhanced sizing note with risk info
                    risk_bps = strat['execution'].get('risk_bps', 0)
                    sizing_with_risk = f"{sizing_note} | Risk: {risk_bps}bps (${risk:.0f})"
                    if _no_earn_cov:
                        sizing_with_risk += " | ⚠️ No earnings data — verify before fill"
                    
                    # Pull stats from strategy config
                    stats_dict = strat.get('stats', {})
                    stats_str = f"WR: {stats_dict.get('win_rate', 'N/A')} | PF: {stats_dict.get('profit_factor', 'N/A')} | Exp: {stats_dict.get('expectancy', 'N/A')}"
                    
                    # Pull setup and exit_summary for email clarity
                    setup_block = strat.get('setup', {})
                    exit_block = strat.get('exit_summary', {})
                    
                    # Check if stop/target are actually used
                    use_stop = strat['execution'].get('use_stop_loss', True)
                    use_target = strat['execution'].get('use_take_profit', True)
                    
                    # Build LIVE filter values with actual indicator readings
                    live_filters = build_live_filters(strat, last_row, calc_df)
                    
                    # Calculate limit price for limit orders
                    limit_price = None
                    if "Limit" in entry_mode and "ATR" in entry_mode:
                        if "0.75" in entry_mode:
                            limit_price = entry - (0.75 * atr) if direction == 'Long' else entry + (0.75 * atr)
                        elif "0.25" in entry_mode:
                            limit_price = entry - (0.25 * atr) if direction == 'Long' else entry + (0.25 * atr)
                        elif "0.5" in entry_mode:
                            limit_price = entry - (0.5 * atr) if direction == 'Long' else entry + (0.5 * atr)
                        elif "1 ATR" in entry_mode:
                            limit_price = entry - atr if direction == 'Long' else entry + atr
                    
                    # Calculate notional exposure
                    notional = shares * entry
                    
                    # Days until exit
                    days_to_exit = hold_days
                    
                    # Build short entry type label for summary
                    entry_type_short = get_entry_type_short(entry_mode, limit_price)
                    
                    _r252_stamp = last_row.get('rank_ret_252d', None)
                    _r252_val = float(_r252_stamp) if _r252_stamp is not None and pd.notna(_r252_stamp) else None

                    signal_dict = {
                        "Strategy_ID": strat['id'],
                        "Strategy_Name": strat['name'],
                        "Ticker": ticker,
                        "Date": last_row.name.date(),
                        "Action": action,
                        "Shares": shares,
                        "Risk_Amt": risk,
                        "Sizing_Notes": sizing_with_risk,
                        "Rank_252D": _r252_val if _r252_val is not None else '',
                        "Stats": stats_str,
                        "Entry": entry,
                        "Stop": stop_price,
                        "Target": tgt_price,
                        "Time Exit": exit_date,
                        "ATR": atr,
                        # Signal-day High (post spot-alias substitution if applicable).
                        # Stamped so order_staging can evaluate use_t1_open_filter
                        # gates that reference today's High (e.g. SPX OB Fade's
                        # "T+1 Open > High + 0.05 ATR" condition). Only High is
                        # stamped today — Signal_Close + Frozen_ATR cover the
                        # other current filter; add Open/Low here only when a
                        # future strategy references them.
                        "Signal_High": float(last_row['High']),
                        # Execution context
                        "Entry_Type": entry_mode,
                        "Entry_Type_Short": entry_type_short,
                        "Limit_Price": limit_price,
                        "Notional": notional,
                        "Addv_63d": float(_addv_63d) if _addv_63d is not None and pd.notna(_addv_63d) else '',
                        "Earnings_Cov": 'MISSING' if _no_earn_cov else '',
                        "Days_To_Exit": days_to_exit,
                        # Entry-order live window (OLV T+3). Defaults to hold_days
                        # so verify_fills bounds the GTC fill search to the same
                        # window order_staging cancels the live limit on.
                        "Fill_Window_Days": strat['execution'].get('fill_window_days', hold_days),
                        "Use_Stop": use_stop,
                        "Use_Target": use_target,
                        # Setup context
                        "Setup_Type": setup_block.get('type', 'Custom'),
                        "Setup_Timeframe": setup_block.get('timeframe', 'Swing'),
                        "Setup_Thesis": setup_block.get('thesis', ''),
                        "Setup_Filters": setup_block.get('key_filters', []),
                        "Live_Filters": live_filters,
                        "Exit_Primary": exit_block.get('primary_exit', ''),
                        "Exit_Stop": exit_block.get('stop_logic', ''),
                        "Exit_Target": exit_block.get('target_logic', ''),
                        "Exit_Notes": exit_block.get('notes', ''),
                        # Sizing context variable (for strategies with dynamic sizing)
                        "Sizing_Variable": get_sizing_variable(strat['name'], last_row),
                        # Tier this signal belongs to ('Liquid' or 'Overflow').
                        # Stamped onto the staging row so order_staging knows
                        # which universe sized it. The 6 overflow-eligible
                        # strategies appear twice in scope=all (once per tier).
                        "Scan_Source": _scan_source,
                    }

                    signals.append(signal_dict)
            except Exception as e:
                error_tickers.append((t_clean, str(e)[:80]))
                print(f"Error processing {ticker}: {e}")
                continue
        
        if signals:
            all_signals.extend(signals)
            print(f"  -> Found {len(signals)} signals.")

    # No scanner-side risk cap — order_staging.py applies the 2.5% backstop
    # post-open across all signals (incl. OVS), which is the single source of
    # truth for aggregate risk control.

    # 5b. Cross-Strategy Overlap Clamp — apply CROSS_STRATEGY_OVERLAP_OVERRIDES.
    # When a defined pair of strategies both fire on the same date and same
    # tradeable ticker (after SPOT_TO_TRADEABLE substitution), each side's
    # Shares / Risk_Amt / Notional is scaled down so the per-trade risk lands
    # at the configured clamp bps. This runs BEFORE staging so the Sheets row
    # already reflects the reduced size; order_staging.py honors what's there.
    if CROSS_STRATEGY_OVERLAP_OVERRIDES and all_signals:
        # Build (Date, TradedAs) -> {strategy names} index
        from collections import defaultdict as _dd
        _date_traded_to_strats = _dd(set)
        for _s in all_signals:
            _tkr = str(_s.get('Ticker', ''))
            _td = SPOT_TO_TRADEABLE.get(_tkr, _tkr)
            _date_traded_to_strats[(_s.get('Date'), _td)].add(_s.get('Strategy_Name'))

        # Map strategy name -> original risk_bps so we can compute the scale.
        _strat_bps = {s['name']: int(s['execution'].get('risk_bps', 0)) for s in effective_book}

        _clamp_count = 0
        for _ovr in CROSS_STRATEGY_OVERLAP_OVERRIDES:
            _pair = set(_ovr['strategies'])
            _clamp_bps = float(_ovr['risk_bps_when_overlapping'])
            # Find collision keys: (date, traded) where >= 2 strategies in _pair fired.
            _collisions = {
                _key for _key, _strats in _date_traded_to_strats.items()
                if len(_strats & _pair) >= 2
            }
            if not _collisions:
                continue
            for _s in all_signals:
                if _s.get('Strategy_Name') not in _pair:
                    continue
                _tkr = str(_s.get('Ticker', ''))
                _td = SPOT_TO_TRADEABLE.get(_tkr, _tkr)
                if (_s.get('Date'), _td) not in _collisions:
                    continue
                _orig_bps = _strat_bps.get(_s.get('Strategy_Name'), 0)
                if _orig_bps <= 0 or _clamp_bps >= _orig_bps:
                    continue
                _scale = _clamp_bps / _orig_bps
                _orig_shares = _s.get('Shares', 0)
                _s['Shares'] = int(round(_orig_shares * _scale))
                _s['Risk_Amt'] = float(_s.get('Risk_Amt', 0.0)) * _scale
                _s['Notional'] = float(_s.get('Notional', 0.0)) * _scale
                _s['Sizing_Notes'] = (
                    f"{_s.get('Sizing_Notes', '')} | "
                    f"Cross-strategy overlap clamp -> {int(_clamp_bps)} bps "
                    f"({_orig_bps}->{int(_clamp_bps)}, scale {_scale:.2f})"
                )
                _clamp_count += 1
        if _clamp_count:
            print(f"[OVERLAP CLAMP] Reduced risk on {_clamp_count} signal(s) due to cross-strategy date+tradeable collisions.")

    # 5c. Same-day signal de-rate (3x Bear ETF Overbot Fade, 2026-07-07).
    # Any strategy with execution['same_day_signal_derate'] has every one of
    # today's signals scaled by max(floor, 1 - d*(n-1)), n = that strategy's
    # signal count in this scan (per tier). Count is ex-ante (staged signals,
    # not fills): several inverse-3x names overbought at once marks a violent
    # selloff where per-trade edge degrades. Runs as a post-pass because n is
    # only known after the strategy's ticker loop. Mirrored in
    # strat_backtester sizing 3b4 — change together.
    _derate_execs = {
        s['name']: s['execution'] for s in effective_book
        if s.get('execution', {}).get('same_day_signal_derate')
    }
    if _derate_execs and all_signals:
        from collections import Counter as _Counter
        _sig_counts = _Counter(
            (_s.get('Strategy_Name'), _s.get('Scan_Source'))
            for _s in all_signals
            if _s.get('Strategy_Name') in _derate_execs
        )
        _derated = 0
        for _s in all_signals:
            _name = _s.get('Strategy_Name')
            _exe = _derate_execs.get(_name)
            if _exe is None:
                continue
            _n = _sig_counts.get((_name, _s.get('Scan_Source')), 1)
            _mult = same_day_derate_mult(_exe, _n)
            if _mult == 1.0:
                continue
            _s['Shares'] = int(round(_s.get('Shares', 0) * _mult))
            _s['Risk_Amt'] = float(_s.get('Risk_Amt', 0.0)) * _mult
            _s['Notional'] = float(_s.get('Notional', 0.0)) * _mult
            _s['Sizing_Notes'] = (
                f"{_s.get('Sizing_Notes', '')} | Same-day derate: "
                f"{_n} signals -> {_mult:.2f}x"
            )
            _derated += 1
        if _derated:
            print(f"[SAME-DAY DERATE] Scaled {_derated} signal(s) — multiple same-strategy signals today.")

    # 6. Save Results
    # Dry-run: print a summary and skip ALL side effects (no Google Sheets
    # writes, no R2, no email). Used to validate a new universe / config safely.
    if dry_run:
        print("\n=== DRY RUN (no Sheets / R2 / email writes) ===")
        if all_signals:
            _df = pd.DataFrame(all_signals)
            _by = _df.groupby(['Scan_Source', 'Strategy_Name']).size() if 'Scan_Source' in _df.columns else _df.groupby('Strategy_Name').size()
            print(f"Total signals: {len(_df)}")
            print(_by.to_string())
            if 'Earnings_Cov' in _df.columns:
                _miss = int((_df['Earnings_Cov'] == 'MISSING').sum())
                print(f"Overflow signals flagged 'no earnings coverage': {_miss}")
            print("\nSample:")
            _cols = [c for c in ['Scan_Source', 'Strategy_Name', 'Ticker', 'Action', 'Shares', 'Notional', 'Addv_63d', 'Earnings_Cov', 'Sizing_Notes'] if c in _df.columns]
            print(_df[_cols].head(25).to_string(index=False))
        else:
            print("No signals found today.")
        print(f"Errors/skips: {len(error_tickers)}")
        print("--- Dry Run Complete ---")
        return all_signals

    # IMPORTANT: --moc-only runs only scan MOC strategies (entry_type='Signal
    # Close'). They never produce limit/persistent signals, so they MUST NOT
    # touch the Order_Staging or Overflow tabs — those tabs hold persistent
    # GTC limit signals from the bookend scope=all runs that should survive
    # across intraday MOC-only runs. Touching them with empty data would
    # clear and wipe the legitimate signals already there.
    if all_signals:
        df_sig = pd.DataFrame(all_signals)
        # 1. Log to Master Sheet (APPEND MODE)
        save_signals_to_gsheet(df_sig)

        # 2. Stage MOC Orders (Signal Close) — Liquid only by convention
        # (overflow tier is too thin for safe MOC participation; save_moc_orders
        # skips rows with Scan_Source='Overflow').
        if scope in ('liquid', 'all'):
            save_moc_orders(all_signals, effective_book, sheet_name='moc_orders')

        # 3. Stage non-MOC orders to per-tier tabs (skip when moc_only —
        # those tabs are owned by the bookend scope=all runs).
        if scope in ('liquid', 'all') and not moc_only:
            save_staging_orders(
                all_signals, effective_book,
                sheet_name='Order_Staging', tier_filter='Liquid',
            )
        if scope in ('overflow', 'all') and not moc_only:
            save_staging_orders(
                all_signals, effective_book,
                sheet_name='Overflow', tier_filter='Overflow',
            )
    else:
        print("No signals found today.")
        # Clear whichever tabs THIS scope owns so stale rows don't linger.
        # moc_only runs intentionally leave Order_Staging / Overflow alone.
        if scope in ('liquid', 'all') and not moc_only:
            save_staging_orders([], effective_book, sheet_name='Order_Staging', tier_filter='Liquid')
        if scope in ('overflow', 'all') and not moc_only:
            save_staging_orders([], effective_book, sheet_name='Overflow', tier_filter='Overflow')

    # 6b. OLV vol-confirmed exit staging (2026-07-20). Runs on both bookend
    # scans regardless of signal count: the PM run evaluates today's settled
    # close, the AM run re-evaluates the same session with corrected data
    # (same convention as the risk-report AM correction). Always rewrites
    # the OLV_Exits tab so stale exits can't resubmit; fully fail-open.
    if not moc_only:
        try:
            _olv_exit_warnings = stage_olv_vol_confirm_exits(
                master_dict if 'master_dict' in dir() else None) or []
        except Exception as e:
            _olv_exit_warnings = [f"staging crashed ({e}) — positions fall back to time exits"]
            print(f"[OLV-EXIT] {_olv_exit_warnings[0]}")
        # Surface exit-pipeline problems in the daily email: the resting STP
        # this pipeline replaced failed loudly at the broker; its replacement
        # must never fail silently in a log nobody reads (review finding,
        # 2026-07-20).
        for _w in _olv_exit_warnings:
            error_tickers.append(("OLV-EXIT", _w))

    # 7. Send Email Summary
    # Deduplicate error tickers (same ticker may appear across multiple strategies)
    seen_errors = set()
    unique_errors = []
    for ticker, reason in error_tickers:
        key = (ticker, reason)
        if key not in seen_errors:
            seen_errors.add(key)
            unique_errors.append((ticker, reason))

    # Pass scope label to email subject so the run mode is visible in the
    # inbox — quick way to spot a misclassified scan at a glance.
    if moc_only:
        _scope_label = "intraday MOC-only"
    elif scope == 'all':
        _scope_label = "bookend full (scope=all)"
    elif scope == 'overflow':
        _scope_label = "scope=overflow"
    else:
        _scope_label = "scope=liquid"

    # Exposure leg — only on the AM bookend run (scope=all + UTC hour < 12).
    # The PM bookend at ~20:13 UTC and intraday MOC runs skip it.
    # Email rendering removed 2026-07-16 (per McKinley): the state still
    # computes and persists here — the site's Sizing State hero and the
    # committed exposure_state.json snapshot depend on it.
    is_am_run = (scope == 'all' and not moc_only and datetime.datetime.utcnow().hour < 12)
    if is_am_run:
        try:
            today_snap = compute_exposure_targets(
                account_value=ACCOUNT_VALUE,
                master_dict=master_dict if 'master_dict' in dir() else None,
            )
            if today_snap is not None:
                save_state(today_snap)
                print(f"[exposure] Mult={today_snap['mult']:.2f}x rule={today_snap['active_rule']} reason={today_snap['reason']}")
            else:
                print("[exposure] Fragility cache missing — exposure leg skipped.")
        except Exception as e:
            print(f"[exposure] Failed to compute exposure leg: {e}")

    send_email_summary(all_signals, error_tickers=unique_errors,
                       scope_label=_scope_label, pc_state=pc_state)

    print("--- Scan Complete ---")


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="Daily scan — liquid + overflow universes")
    _ap.add_argument(
        "--scope",
        choices=("liquid", "overflow", "all"),
        default="liquid",
        help="liquid (default, GHA path): scan LIQUID_PLUS_COMMODITIES per strategy. "
             "overflow: scan CSV_UNIVERSE − LIQUID_PLUS_COMMODITIES for the 5 "
             "overflow-eligible strategies with bps overrides. "
             "all: liquid + overflow (signals tagged with Scan_Source).",
    )
    _ap.add_argument(
        "--moc-only", action="store_true",
        help="Restrict to MOC strategies (entry_type='Signal Close'). Skips the "
             "overflow tier entirely (overflow doesn't MOC). Use for intraday "
             "GHA runs — limit-entry strategies don't change with intraday data.",
    )
    _ap.add_argument(
        "--dry-run", action="store_true",
        help="Run the full scan but skip ALL side effects (no Google Sheets, no "
             "R2, no email) — print a signal summary only. Safe for validating a "
             "new universe or config. Note: pair with OVERFLOW_UNIVERSE_ACTIVE=1 "
             "to preview the dynamic overflow universe before activating it live.",
    )
    _args = _ap.parse_args()
    run_daily_scan(scope=_args.scope, moc_only=_args.moc_only, dry_run=_args.dry_run)
