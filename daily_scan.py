import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import gspread
from pandas.tseries.offsets import BusinessDay, CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import time
import pytz
import sys
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Define a "Trading Day" offset that skips Weekends AND US Holidays
TRADING_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# -----------------------------------------------------------------------------
# IMPORT STRATEGY BOOK
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from strategy_config import STRATEGY_BOOK
except ImportError:
    print("❌ Could not find strategy_config.py in the root directory.")
    STRATEGY_BOOK = []

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


def send_email_summary(signals_list):
    """
    Sends an HTML email summary of the signals using Gmail SMTP.
    Card-based layout showing full signal criteria with LIVE values.
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
    
    # Orphaned companions (ATH routing: LOC staged but primary skipped)
    _primary_vol_spike_tickers = {s['Ticker'] for s in _primary_signals if s.get('Strategy_Name') == 'Overbot Vol Spike'}
    for ticker, comp in _companion_map.items():
        if comp.get('_parent_strategy') == 'Overbot Vol Spike' and ticker not in _primary_vol_spike_tickers:
            _primary_signals.append(comp)
    
    if not email_signals:
        subject = f"📉 Scan Result: NO SIGNALS ({date_str})"
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; background-color: #f5f5f5; padding: 20px;">
                <div style="max-width: 700px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px;">
                    <h2 style="color: #333; margin-top: 0;">Daily Strategy Scan: {date_str}</h2>
                    <p style="color: #666;">The scan completed successfully.</p>
                    <p style="font-size: 18px; color: #888;"><strong>Result:</strong> No signals found matching criteria today.</p>
                </div>
            </body>
        </html>
        """
    else:
        subject = f"🚀 {signal_count} SIGNAL{'S' if signal_count > 1 else ''} ({date_str})"
        
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
            
            # Combine exit notes with sizing variable if present
            notes_parts = []
            if sizing_var:
                notes_parts.append(f"📊 {sizing_var}")
            if exit_notes:
                notes_parts.append(f"⚡ {exit_notes}")
            
            # Companion order info
            if _companion:
                comp_price = _companion.get('Limit_Price', 0)
                comp_shares = _companion.get('Shares', 0)
                notes_parts.append(f"📋 Also staged: LOC {comp_shares:,} shares @ >${comp_price:.2f} (Close + 0.5 ATR)")
            
            # ATH routing explanation (orphaned companion shown as card)
            if sig.get('_is_companion', False) and sig.get('_parent_strategy') == 'Overbot Vol Spike':
                notes_parts.append(f"⚠️ ATH in last 10d — primary short suppressed, LOC only")
            
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


def get_sznl_val_series(ticker, dates, sznl_map):
    ticker = ticker.upper()
    t_series = sznl_map.get(ticker)
    
    if t_series is None and ticker == "^GSPC":
        t_series = sznl_map.get("SPY")

    if t_series is None:
        return pd.Series(50.0, index=dates)
        
    return dates.map(t_series).fillna(50.0)


# -----------------------------------------------------------------------------
# 2. CALCULATION ENGINE
# -----------------------------------------------------------------------------

def calculate_indicators(df, sznl_map, ticker, market_series=None, vix_series=None, ref_ticker_ranks=None):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    
    # --- MAs ---
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA100'] = df['Close'].rolling(100).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA11'] = df['Close'].ewm(span=11, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # --- Gap Count ---
    is_open_gap = (df['Low'] > df['High'].shift(1)).astype(int)
    df['GapCount_21'] = is_open_gap.rolling(21).sum() 
    df['GapCount_10'] = is_open_gap.rolling(10).sum()
    df['GapCount_5'] = is_open_gap.rolling(5).sum() 

    # --- Candle Range Location % ---
    denom = (df['High'] - df['Low'])
    df['RangePct'] = np.where(denom == 0, 0.5, (df['Close'] - df['Low']) / denom)

    # --- Perf Ranks ---
    for window in [2, 5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window, fill_method=None)
        df[f'rank_ret_{window}d'] = df[f'ret_{window}d'].expanding(min_periods=50).rank(pct=True) * 100.0
        
    # --- ATR ---
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
    
    # --- Today's Return in ATR units (vs yesterday's close) ---
    df['today_return_atr'] = (df['Close'] - df['Close'].shift(1)) / df['ATR']

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
    df['AccCount_5'] = is_accumulation.rolling(5).sum()
    df['AccCount_10'] = is_accumulation.rolling(10).sum()
    df['AccCount_42'] = is_accumulation.rolling(42).sum()
    
    # 2. Distribution (Red + Spike)
    cond_red = df['Close'] < df['Open']
    is_distribution = (df['Vol_Spike'] & cond_red).astype(int)
    df['DistCount_21'] = is_distribution.rolling(21).sum()
    df['DistCount_5'] = is_distribution.rolling(5).sum()
    df['DistCount_10'] = is_distribution.rolling(10).sum()
    df['DistCount_42'] = is_distribution.rolling(42).sum()
    
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

    # --- VIX (passed in as parameter) ---
    if vix_series is not None:
        df['VIX_Value'] = vix_series.reindex(df.index, method='ffill').fillna(0)
    else:
        df['VIX_Value'] = 0.0

    # --- 52w High/Low ---
    rolling_high = df['High'].shift(1).rolling(252).max()
    rolling_low = df['Low'].shift(1).rolling(252).min()
    df['is_52w_high'] = df['High'] > rolling_high
    df['is_52w_low'] = df['Low'] < rolling_low

    # --- All-Time High ---
    df['prior_ath'] = df['High'].shift(1).expanding().max()
    df['is_ath'] = df['High'] >= df['prior_ath']
    df['High_52w'] = df['High'].rolling(252).max()
    df['ATH_Level'] = df['High'].expanding().max()

    # Reference Ticker Ranks
    if ref_ticker_ranks is not None:
        for window, series in ref_ticker_ranks.items():
            df[f'Ref_rank_ret_{window}d'] = series.reindex(df.index, method='ffill').fillna(50.0)
            
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

    if params.get('use_today_return', False):
        today_ret = last_row.get('today_return_atr', 0)
        if pd.isna(today_ret): return False
        ret_min = params.get('return_min', -100)
        ret_max = params.get('return_max', 100)
        if not (today_ret >= ret_min and today_ret <= ret_max): return False

    if params.get('use_atr_ret_filter', False):
        today_ret = last_row.get('today_return_atr', 0)
        if pd.isna(today_ret): return False
        if not (today_ret >= params.get('atr_ret_min', -100) and today_ret <= params.get('atr_ret_max', 100)): return False

    # 2. Trend Filter (Global) - ALL OPTIONS
    trend_opt = params.get('trend_filter', 'None')
    if trend_opt == "Price > 200 SMA":
        if not (last_row['Close'] > last_row['SMA200']): return False
    elif trend_opt == "Price > Rising 200 SMA":
        prev_row = df.iloc[-2]
        if not ((last_row['Close'] > last_row['SMA200']) and (last_row['SMA200'] > prev_row['SMA200'])): return False
    elif trend_opt == "Not Below Declining 200 SMA":
        prev_row = df.iloc[-2]
        # REJECT if price is below AND the 200 SMA is falling
        is_below_declining = (last_row['Close'] < last_row['SMA200']) and (last_row['SMA200'] < prev_row['SMA200'])
        if is_below_declining: return False
    elif trend_opt == "Price < 200 SMA":
        if not (last_row['Close'] < last_row['SMA200']): return False
    elif trend_opt == "Price < Falling 200 SMA":
        prev_row = df.iloc[-2]
        if not ((last_row['Close'] < last_row['SMA200']) and (last_row['SMA200'] < prev_row['SMA200'])): return False
    elif "Market" in trend_opt or "SPY" in trend_opt:
        if 'Market_Above_SMA200' in df.columns:
            is_above = last_row['Market_Above_SMA200']
            if ">" in trend_opt and not is_above: return False
            if "<" in trend_opt and is_above: return False

    # 2b. MA Consecutive Filters
    if 'ma_consec_filters' in params:
        for maf in params['ma_consec_filters']:
            length = maf['length']
            col_name = f"SMA{length}"
            if col_name not in df.columns: continue 
            
            if maf['logic'] == 'Above':
                mask = df['Close'] > df[col_name]
            elif maf['logic'] == 'Below':
                mask = df['Close'] < df[col_name]
            
            consec = maf.get('consec', 1)
            if consec > 1:
                mask = mask.rolling(consec).sum() == consec
            
            if not mask.iloc[-1]: return False
    # 2c. Range in ATR Filter
    if params.get('use_range_atr_filter', False):
        atr = last_row.get('ATR', 0)
        if atr > 0:
            range_in_atr = (last_row['High'] - last_row['Low']) / atr
            logic = params.get('range_atr_logic', 'Between')
            if logic == '>' and not (range_in_atr > params.get('range_atr_min', 0)): return False
            if logic == '<' and not (range_in_atr < params.get('range_atr_max', 99)): return False
            if logic == 'Between' and not (range_in_atr >= params.get('range_atr_min', 0) and range_in_atr <= params.get('range_atr_max', 99)): return False

    # 2d. Require Green Candle
    if params.get('require_close_gt_open', False):
        if not (last_row['Close'] > last_row['Open']): return False

    # 2e. Breakout Mode
    bk_mode = params.get('breakout_mode', 'None')
    if bk_mode != 'None':
        prev_row = df.iloc[-2]
        if bk_mode == "Close > Prev Day High":
            if not (last_row['Close'] > prev_row['High']): return False
        elif bk_mode == "Close < Prev Day Low":
            if not (last_row['Close'] < prev_row['Low']): return False

    # 2f. Volume > Previous Day
    if params.get('vol_gt_prev', False):
        prev_row = df.iloc[-2]
        if not (last_row['Volume'] > prev_row['Volume']): return False
            
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
    if params.get('use_ma_dist_filter', False) or params.get('use_dist_filter', False):
        ma_type = params.get('dist_ma_type', 'SMA 200')
        ma_col_map = {"52-Week High": "High_52w", "All-Time High": "ATH_Level"}
        ma_col = ma_col_map.get(ma_type, ma_type.replace(" ", ""))
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
        val = last_row['Mkt_Sznl_Ref']
        if params['market_sznl_logic'] == '<': 
            if not (val < params['market_sznl_thresh']): return False
        else: 
            if not (val > params['market_sznl_thresh']): return False

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
    # 8c. ATH Filter
    if params.get('use_ath', False):
        if params.get('ath_type') == 'Today is ATH':
            if not last_row['is_ath']: return False
        else:
            if last_row['is_ath']: return False

    # 8d. Recent ATH Filter
    if params.get('use_recent_ath', False):
        ath_lookback = params.get('ath_lookback_days', 21)
        recent_ath = df['is_ath'].rolling(window=ath_lookback, min_periods=1).max().iloc[-1]
        if params.get('recent_ath_invert', False):
            if bool(recent_ath): return False  # Reject if ATH was made recently
        else:
            if not bool(recent_ath): return False  # Reject if NO ATH recently
    # 9. VIX Filter
    if params.get('use_vix_filter', False):
        vix_min = params.get('vix_min', 0)
        vix_max = params.get('vix_max', 100)
        vix_val = last_row.get('VIX_Value', 0)
        if not (vix_val >= vix_min and vix_val <= vix_max): 
            return False
    # 9b. Reference Ticker Filter
    if params.get('use_ref_ticker_filter', False) and params.get('ref_filters'):
        for rf in params['ref_filters']:
            col = f"Ref_rank_ret_{rf['window']}d"
            val = last_row.get(col, 50.0)
            if rf['logic'] == '<' and not (val < rf['thresh']): return False
            if rf['logic'] == '>' and not (val > rf['thresh']): return False
    # 10. Volume (Ratio ONLY)
    if params['use_vol']:
        if not (last_row['vol_ratio'] > params['vol_thresh']): return False

    if params.get('use_vol_rank'):
        val = last_row['vol_ratio_10d_rank']
        if params['vol_rank_logic'] == '<':
            if not (val < params['vol_rank_thresh']): return False
        else:
            if not (val > params['vol_rank_thresh']): return False
            
    return True

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
            print(f"🚀 Staged {len(df_moc)} MOC Orders with Exit Dates!")
        else:
            headers = ["Scan_Date", "Symbol", "SecType", "Exchange", "Action", "Quantity", "Order_Type", "Strategy_Ref", "Exit_Date"]
            worksheet.update(values=[headers])
            print(f"🧹 '{sheet_name}' cleared.")
            
    except Exception as e:
        print(f"❌ MOC Staging Error: {e}")


def save_staging_orders(signals_list, strategy_book, sheet_name='Order_Staging'):
    """
    Saves non-MOC orders (Limits, T+1, etc) to 'Order_Staging'.
    Excludes 'Signal Close' orders.
    """
    if not signals_list: return

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
                staging_data.append({
                    "Scan_Date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "Symbol": row['Ticker'],
                    "SecType": "STK",
                    "Exchange": "SMART",
                    "Action": ib_action,
                    "Quantity": row['Shares'],
                    "Order_Type": "LOC",
                    "Limit_Price": round(row.get('Limit_Price', row['Entry']), 2),
                    "Offset_ATR_Mult": 0.0,
                    "TIF": "DAY",
                    "Frozen_ATR": round(row['ATR'], 2),
                    "Signal_Close": round(row['Entry'], 2),
                    "Time_Exit_Date": str(row['Time Exit']),
                    "Strategy_Ref": row.get('Strategy_Name', 'Companion')
                })
            continue
        
        strat = strat_map.get(row['Strategy_ID'])
        if not strat: continue
        
        settings = strat['settings']
        entry_mode = settings.get('entry_type', 'Signal Close')
        
        # *** SKIP MOC ORDERS (They go to the other sheet) ***
        if "Signal Close" in entry_mode:
            continue
        
        # Defaults
        entry_instruction = "MKT" 
        offset_atr = 0.0
        limit_price = 0.0
        tif_instruction = "DAY" 

        # 1. ATR LIMIT ENTRY
        if "Limit" in entry_mode and "ATR" in entry_mode:
            if "Persistent" in entry_mode:
                entry_instruction = "REL_CLOSE"  # Anchored to Signal Close
                tif_instruction = "GTC"  # Good til canceled (or hold_days)
            else:
                entry_instruction = "REL_OPEN"   # Anchored to T+1 Open
                tif_instruction = "DAY"
            
            if "0.5" in entry_mode: offset_atr = 0.5

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
            "Signal_Close": round(row['Entry'], 2),
            "Time_Exit_Date": str(row['Time Exit']),
            "Strategy_Ref": strat['name']
        })

    # If all orders were "Signal Close", this list is empty now
    if not staging_data:
        gc = get_google_client()
        if gc:
            try:
                sh = gc.open("Trade_Signals_Log")
                worksheet = sh.worksheet(sheet_name)
                worksheet.clear()
                print(f"🧹 '{sheet_name}' cleared (only MOC orders found).")
            except: pass
        return

    df_stage = pd.DataFrame(staging_data)

    gc = get_google_client()
    if not gc: return

    try:
        sh = gc.open("Trade_Signals_Log")
        try:
            worksheet = sh.worksheet(sheet_name)
        except:
            worksheet = sh.add_worksheet(title=sheet_name, rows=100, cols=20)

        worksheet.clear()
        data_to_write = [df_stage.columns.tolist()] + df_stage.astype(str).values.tolist()
        worksheet.update(values=data_to_write)
        print(f"🤖 Instructions Staged! ({len(df_stage)} rows)")
        
    except Exception as e:
        print(f"❌ Staging Sheet Error: {e}")


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
    if "Signal Close" in entry_mode:
        return "MOC"
    elif "T+1 Open" in entry_mode and "Limit" not in entry_mode:
        return "MOO"
    elif "T+1 Close if <" in entry_mode:
        return "Cond Close"
    elif "Limit" in entry_mode:
        # Check if it's Open-based (unknown price) or Close-based (known price)
        if "Open" in entry_mode:
            # Open-based: can't show price, it depends on T+1 Open
            if "0.5" in entry_mode:
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

def generate_vol_spike_companion(primary_signal, strat, last_row, override_risk=None):
    """
    Generates a companion LOC (Limit-on-Close) order for Vol Spike signals.
    
    Logic: If Vol Spike fires, we also want to catch the scenario where price
    keeps running higher the next day. Stage a LOC sell order at 
    (Signal_Close + 0.5 ATR) — only fills if T+1 Close exceeds that threshold.
    
    Args:
        override_risk: If provided, use this risk instead of the primary signal's risk.
                       Used for Tuesday+ATH where primary is killed but LOC stays normal.
    
    Returns a signal dict for the companion order, or None if not applicable.
    """
    if strat['name'] != "Overbot Vol Spike":
        return None
    
    signal_close = primary_signal['Entry']
    atr = primary_signal['ATR']
    
    # LOC threshold: only fill if close > signal_close + 0.5 ATR
    loc_threshold = signal_close + (0.5 * atr)
    
    # Use override risk if provided (e.g. Tuesday+ATH), otherwise match primary
    risk = primary_signal['Risk_Amt']
    
    # For LOC, estimate entry at threshold for sizing
    direction = "Short"
    stop_atr = strat['execution']['stop_atr']
    dist = atr * stop_atr
    shares = int(risk / dist) if dist > 0 else 0
    
    if shares == 0:
        return None
    
    # Calculate exit date (same hold period as primary)
    hold_days = strat['execution']['hold_days']
    effective_entry_date = last_row.name + TRADING_DAY  # T+1 close entry
    exit_date = (effective_entry_date + (TRADING_DAY * hold_days)).date()
    
    # Stop/target from threshold price
    stop_price = loc_threshold + (atr * stop_atr)
    tgt_atr = strat['execution']['tgt_atr']
    tgt_price = loc_threshold - (atr * tgt_atr)
    
    # Build sizing notes — if override, note it's using pre-overlay risk
    sizing_notes = primary_signal['Sizing_Notes'] + " | LOC Companion"
    
    return {
        "Strategy_ID": strat['id'] + " (LOC Add)",
        "Strategy_Name": "Vol Spike LOC Add",
        "Ticker": primary_signal['Ticker'],
        "Date": primary_signal['Date'],
        "Action": "SELL SHORT",
        "Shares": shares,
        "Risk_Amt": risk,
        "Sizing_Notes": sizing_notes,
        "Stats": primary_signal['Stats'],
        "Entry": loc_threshold,  # Threshold price (actual fill at close)
        "Stop": stop_price,
        "Target": tgt_price,
        "Time Exit": exit_date,
        "ATR": atr,
        # Execution context - LOC specific
        "Entry_Type": "LOC (Limit-on-Close)",
        "Entry_Type_Short": f"LOC >${loc_threshold:.2f}",
        "Limit_Price": loc_threshold,
        "Notional": shares * loc_threshold,
        "Days_To_Exit": hold_days,
        "Use_Stop": strat['execution'].get('use_stop_loss', False),
        "Use_Target": strat['execution'].get('use_take_profit', False),
        # Setup context
        "Setup_Type": "MeanReversion",
        "Setup_Timeframe": "Overnight",
        "Setup_Thesis": "Vol Spike ran higher — catching extended move at close",
        "Setup_Filters": ["Same conditions as Vol Spike", 
                          f"T+1 Close > ${loc_threshold:.2f} (Signal Close + 0.5 ATR)"],
        "Live_Filters": [("T+1 Close must exceed", f"${loc_threshold:.2f}", False)],
        "Exit_Primary": f"{hold_days}-day time stop",
        "Exit_Stop": strat.get('exit_summary', {}).get('stop_logic', ''),
        "Exit_Target": strat.get('exit_summary', {}).get('target_logic', ''),
        "Exit_Notes": "Companion to Vol Spike — only fills if price kept running",
        "Sizing_Variable": primary_signal.get('Sizing_Variable'),
        # Internal flags
        "_is_companion": True,
        "_parent_strategy": "Overbot Vol Spike"
    }
    
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
        if "200 SMA" in trend:
            sma200 = last_row.get('SMA200', 0)
            close = last_row['Close']
            if "Price >" in trend:
                live_filters.append((f"Price > 200 SMA", f"${close:.2f} vs ${sma200:.2f}", False))
            elif "Price <" in trend:
                live_filters.append((f"Price < 200 SMA", f"${close:.2f} vs ${sma200:.2f}", False))
            else:
                live_filters.append((trend, f"${close:.2f} vs ${sma200:.2f}", False))
        elif "Market" in trend:
            mkt_above = last_row.get('Market_Above_SMA200', False)
            live_filters.append((trend, "✓" if mkt_above else "✗", True))
    
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
    CHUNK_SIZE = 50 
    total = len(clean_tickers)
    
    print(f"📥 Downloading data for {total} tickers...")
    
    for i in range(0, total, CHUNK_SIZE):
        chunk = clean_tickers[i : i + CHUNK_SIZE]
        try:
            df = yf.download(chunk, start=start_date, group_by='ticker', auto_adjust=True, progress=False, threads=True)
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


def run_daily_scan():
    print("--- Starting Daily Automated Scan (Synced) ---")
    sznl_map = load_seasonal_map()
    
    # 1. Gather Tickers
    all_tickers = set()
    for strat in STRATEGY_BOOK:
        all_tickers.update(strat['universe_tickers'])
        s = strat['settings']
        if s.get('use_market_sznl'): all_tickers.add(s.get('market_ticker', '^GSPC'))
        if "Market" in s.get('trend_filter', ''): all_tickers.add(s.get('market_ticker', 'SPY'))
        if "SPY" in s.get('trend_filter', ''): all_tickers.add("SPY")
        if s.get('use_vix_filter', False): all_tickers.add("^VIX")  # VIX for strategies that need it
        if s.get('use_ref_ticker_filter', False) and s.get('ref_ticker'):
            all_tickers.add(s['ref_ticker'].replace('.', '-'))
    
    # 2. Download Data
    master_dict = download_historical_data(list(all_tickers))
    
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
    
    if now_eastern < market_open_time:
        # Morning Run (e.g. 5:30 AM): Strict cutoff at YESTERDAY'S close.
        # We must remove any partial data stamped with today's date.
        expected_data_date = (pd.Timestamp(current_date) - TRADING_DAY).date()
        print(f"🌅 Morning Run (Pre-Market): Enforcing data cutoff at {expected_data_date}")
    else:
        # Day Run (e.g. 10:00 AM): Allow today's partial bar.
        expected_data_date = current_date
        print(f"☀️ Day Run (Post-Open): Allowing data through {expected_data_date}")

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
    print(f"✅ Data dates validated. (Processing {len(master_dict)} tickers)\n")
    # -------------------------------------------------------------------------

    # 4. Prepare VIX Series (for strategies with VIX filter)
    vix_df = master_dict.get('^VIX')
    vix_series = None
    if vix_df is not None and not vix_df.empty:
        temp_vix = vix_df.copy()
        temp_vix.columns = [c.capitalize() for c in temp_vix.columns]
        if temp_vix.index.tz is not None:
            temp_vix.index = temp_vix.index.tz_localize(None)
        vix_series = temp_vix['Close']
    
    all_signals = []

    # 5. Run Strategies
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
        # Prepare Reference Ticker Ranks (if needed)
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
        for ticker in strat['universe_tickers']:
            t_clean = ticker.replace('.', '-')
            df = master_dict.get(t_clean)
            if df is None or len(df) < 250: continue
            
            try:
                calc_df = calculate_indicators(df.copy(), sznl_map, t_clean, market_series, vix_series, ref_ticker_ranks)
                
                if check_signal(calc_df, strat['settings'], sznl_map):
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
                    
                    # Initialize overlay flags
                    # Initialize overlay flags
                    _skip_primary = False
                    _skip_loc = False

                    if strat['name'] == "Overbot Vol Spike":
                        is_ath_l10 = bool(calc_df['is_ath'].rolling(window=10, min_periods=1).max().iloc[-1])
                        is_52w_high = bool(last_row.get('is_52w_high', False))

                        if is_ath_l10:
                            # Case 1: Made ATH in last 10 days → LOC only, normal risk
                            _skip_primary = True
                            risk = base_risk
                            sizing_note = f"ATH in L10 → LOC only (1.0x)"
                        elif is_52w_high:
                            # Case 2: No ATH in L10 but 52w high today → primary only, 0.66x
                            _skip_loc = True
                            risk = base_risk * 0.66
                            sizing_note = f"52w High, no ATH L10 → Primary only (0.66x)"
                        else:
                            # Case 3: No ATH in L10, no 52w high → both primary + LOC, normal risk
                            risk = base_risk
                            sizing_note = f"No ATH L10, no 52w High → Primary + LOC (1.0x)"
                    
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
                    # ---------------------------------------------------------

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
                    entry_mode = strat['settings'].get('entry_type', 'Signal Close')
                    hold_days = strat['execution']['hold_days']

                    # Determine the effective Entry Date
                    if "Signal Close" in entry_mode:
                        effective_entry_date = last_row.name 
                    else:
                        effective_entry_date = last_row.name + TRADING_DAY

                    # Calculate Exit Date
                    exit_date = (effective_entry_date + (TRADING_DAY * hold_days)).date()
                    
                    # Build enhanced sizing note with risk info
                    risk_bps = strat['execution'].get('risk_bps', 0)
                    sizing_with_risk = f"{sizing_note} | Risk: {risk_bps}bps (${risk:.0f})"
                    
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
                        if "0.5" in entry_mode:
                            limit_price = entry - (0.5 * atr) if direction == 'Long' else entry + (0.5 * atr)
                        elif "1 ATR" in entry_mode:
                            limit_price = entry - atr if direction == 'Long' else entry + atr
                    
                    # Calculate notional exposure
                    notional = shares * entry
                    
                    # Days until exit
                    days_to_exit = hold_days
                    
                    # Build short entry type label for summary
                    entry_type_short = get_entry_type_short(entry_mode, limit_price)
                    
                    signal_dict = {
                        "Strategy_ID": strat['id'],
                        "Strategy_Name": strat['name'],
                        "Ticker": ticker,
                        "Date": last_row.name.date(),
                        "Action": action,
                        "Shares": shares,
                        "Risk_Amt": risk, 
                        "Sizing_Notes": sizing_with_risk,
                        "Stats": stats_str,
                        "Entry": entry,
                        "Stop": stop_price,
                        "Target": tgt_price,
                        "Time Exit": exit_date,
                        "ATR": atr,
                        # Execution context
                        "Entry_Type": entry_mode,
                        "Entry_Type_Short": entry_type_short,
                        "Limit_Price": limit_price,
                        "Notional": notional,
                        "Days_To_Exit": days_to_exit,
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
                        "Sizing_Variable": get_sizing_variable(strat['name'], last_row)
                    }
                    
                    if strat['name'] == "Overbot Vol Spike":
                        companion = None
                        if not _skip_loc:
                            companion = generate_vol_spike_companion(
                                signal_dict, strat, last_row
                            )

                        if not _skip_primary:
                            signals.append(signal_dict)

                        if companion:
                            signals.append(companion)
                    else:
                        signals.append(signal_dict)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
        if signals:
            all_signals.extend(signals)
            print(f"  -> Found {len(signals)} signals.")

    # 6. Save Results
    if all_signals:
        df_sig = pd.DataFrame(all_signals)
        # 1. Log to Master Sheet (APPEND MODE)
        save_signals_to_gsheet(df_sig)
        
        # 2. Stage MOC Orders (Signal Close)
        save_moc_orders(all_signals, STRATEGY_BOOK, sheet_name='moc_orders')
        
        # 3. Stage Rest of Orders (Limits, MOO)
        save_staging_orders(all_signals, STRATEGY_BOOK, sheet_name='Order_Staging')
    else:
        print("No signals found today.")

    # 7. Send Email Summary
    send_email_summary(all_signals)

    print("--- Scan Complete ---")


if __name__ == "__main__":
    run_daily_scan()
