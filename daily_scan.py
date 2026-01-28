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
    NEW FORMAT: Card-based layout showing full signal criteria for each trade.
    """
    sender_email = os.environ.get("EMAIL_USER")
    sender_password = os.environ.get("EMAIL_PASS")
    receiver_email = "mckinleyslade@gmail.com"

    if not sender_email or not sender_password:
        print("⚠️ Email credentials (EMAIL_USER/EMAIL_PASS) not found. Skipping email.")
        return

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    if not signals_list:
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
        subject = f"🚀 {len(signals_list)} SIGNALS ({date_str})"
        
        # Build card-based HTML for each signal
        signal_cards = []
        
        for sig in signals_list:
            # Header color based on action
            header_color = "#2e7d32" if sig['Action'] == "BUY" else "#c62828"
            action_emoji = "📈" if sig['Action'] == "BUY" else "📉"
            
            # Build the key filters bullet list
            filters_list = sig.get('Setup_Filters', [])
            if filters_list:
                filters_html = "".join([f"<li style='margin: 4px 0; color: #444;'>{f}</li>" for f in filters_list])
            else:
                filters_html = "<li style='color: #999;'>No filter details available</li>"
            
            # Build exit plan string
            exit_parts = []
            if sig.get('Exit_Stop'):
                exit_parts.append(f"Stop: {sig['Exit_Stop']}")
            if sig.get('Exit_Target'):
                exit_parts.append(f"Target: {sig['Exit_Target']}")
            exit_plan_str = " | ".join(exit_parts) if exit_parts else "Time exit only"
            
            # Exit notes (dynamic sizing info, etc)
            exit_notes = sig.get('Exit_Notes', '')
            exit_notes_html = f"<div style='font-size: 12px; color: #ff9800; margin-top: 5px;'>⚡ {exit_notes}</div>" if exit_notes else ""
            
            # Thesis
            thesis = sig.get('Setup_Thesis', '')
            thesis_html = f"<div style='font-style: italic; color: #555; margin: 10px 0; padding: 10px; background: #f9f9f9; border-left: 3px solid #2196f3;'>{thesis}</div>" if thesis else ""
            
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
                    <span style="font-size: 22px; font-weight: bold; color: #333;">{sig['Ticker']}</span>
                    <span style="color: #666; margin-left: 10px;">
                        {sig['Action']} {sig['Shares']} @ ${sig['Entry']:.2f}
                    </span>
                    <span style="color: #888; margin-left: 15px;">
                        Exit: {sig['Time Exit']}
                    </span>
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
                        🚪 EXIT: {sig.get('Exit_Primary', 'Time stop')}
                    </div>
                    <div style="color: #666; font-size: 12px; margin-top: 5px;">
                        {exit_plan_str}
                    </div>
                    {exit_notes_html}
                </div>
                
                <!-- Footer Stats -->
                <div style="padding: 10px 15px; background: #333; color: #aaa; font-size: 11px;">
                    <span style="margin-right: 15px;">📊 {sig['Stats']}</span>
                    <span style="color: #888;">ATR: ${sig['ATR']:.2f} | Stop: ${sig['Stop']:.2f} | Target: ${sig['Target']:.2f}</span>
                </div>
            </div>
            """
            signal_cards.append(card_html)
        
        # Combine all cards
        all_cards_html = "".join(signal_cards)
        
        # Quick summary table at top for scanning
        df = pd.DataFrame(signals_list)
        summary_rows = []
        for _, row in df.iterrows():
            color = "#2e7d32" if row['Action'] == "BUY" else "#c62828"
            summary_rows.append(f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>{row['Ticker']}</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee; color: {color};">{row['Action']}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">{row['Shares']}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee;">${row['Entry']:.2f}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">{row.get('Strategy_Name', row['Strategy_ID'][:30])}</td>
                </tr>
            """)
        summary_table = f"""
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 25px; font-size: 13px;">
            <tr style="background: #f0f0f0;">
                <th style="padding: 10px; text-align: left;">Ticker</th>
                <th style="padding: 10px; text-align: left;">Action</th>
                <th style="padding: 10px; text-align: left;">Shares</th>
                <th style="padding: 10px; text-align: left;">Entry</th>
                <th style="padding: 10px; text-align: left;">Strategy</th>
            </tr>
            {"".join(summary_rows)}
        </table>
        """
        
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; background-color: #f5f5f5; padding: 20px;">
                <div style="max-width: 700px; margin: 0 auto;">
                    <!-- Header -->
                    <div style="background: linear-gradient(135deg, #1a237e, #283593); color: white; padding: 25px; border-radius: 8px 8px 0 0; text-align: center;">
                        <h1 style="margin: 0; font-size: 24px;">Daily Strategy Scan</h1>
                        <div style="font-size: 14px; opacity: 0.8; margin-top: 5px;">{date_str}</div>
                        <div style="font-size: 28px; margin-top: 10px;">🎯 {len(signals_list)} Signal{'s' if len(signals_list) > 1 else ''} Found</div>
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

    min_atr = params.get('min_atr_pct', 0.0)
    max_atr = params.get('max_atr_pct', 1000.0)
    
    current_atr_pct = last_row.get('ATR_Pct', 0)
    if pd.isna(current_atr_pct): return False

    if current_atr_pct < min_atr: return False
    if current_atr_pct > max_atr: return False

    # 1b. Today's Return Filter (in ATR units)
    if params.get('use_today_return', False):
        today_ret = last_row.get('today_return_atr', 0)
        if pd.isna(today_ret): return False
        ret_min = params.get('return_min', -100)
        ret_max = params.get('return_max', 100)
        if not (today_ret >= ret_min and today_ret <= ret_max): return False

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
        
    # 8b. Exclude 52w High
    if params.get('exclude_52w_high', False):
        if last_row['is_52w_high']: return False

    # 9. Volume (Ratio ONLY)
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
    
    # Clean Data - exclude the new setup/exit fields from the main log
    df_new = new_dataframe.copy()
    
    # Drop the detailed setup fields (they're for email only)
    cols_to_drop = ['Setup_Type', 'Setup_Timeframe', 'Setup_Thesis', 'Setup_Filters',
                    'Exit_Primary', 'Exit_Stop', 'Exit_Target', 'Exit_Notes']
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

                    if strat['name'] == "Overbot Vol Spike":
                        total_mult = 1.0
                        reasons = []

                        # --- A. Momentum Multiplier ---
                        r5, r10 = last_row.get('rank_ret_5d', 0), last_row.get('rank_ret_10d', 0)
                        if r5 > 95 and r10 > 95:
                            total_mult *= 1.5; reasons.append("Mom>95 (1.5x)")
                        elif r5 > 90 and r10 > 90:
                            total_mult *= 1.15; reasons.append("Mom>90 (1.15x)")
                        elif r5 < 85 and r10 < 85:
                            total_mult *= 0.75; reasons.append("Weak Mom (0.75x)")

                        # --- B. Day of Week Multiplier ---
                        if last_row.name.dayofweek == 4:
                            total_mult *= 1.25
                            reasons.append("Friday (1.25x)")

                        # --- C. Volume Multiplier ---
                        vol_ratio = last_row.get('vol_ratio', 0)
                        if vol_ratio > 2.0:
                            total_mult *= 1.15
                            reasons.append(f"Vol {vol_ratio:.1f}x (1.15x)")

                        risk = base_risk * total_mult
                        sizing_note = " + ".join(reasons) + f" = {total_mult:.2f}x" if reasons else "Standard (1.0x)"
                    
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
                    
                    signals.append({
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
                        # New fields for email clarity
                        "Setup_Type": setup_block.get('type', 'Custom'),
                        "Setup_Timeframe": setup_block.get('timeframe', 'Swing'),
                        "Setup_Thesis": setup_block.get('thesis', ''),
                        "Setup_Filters": setup_block.get('key_filters', []),
                        "Exit_Primary": exit_block.get('primary_exit', ''),
                        "Exit_Stop": exit_block.get('stop_logic', ''),
                        "Exit_Target": exit_block.get('target_logic', ''),
                        "Exit_Notes": exit_block.get('notes', '')
                    })

            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
        if signals:
            all_signals.extend(signals)
            print(f"  -> Found {len(signals)} signals.")

    # 4. Save Results
    if all_signals:
        df_sig = pd.DataFrame(all_signals)
        # 1. Log to Master Sheet
        save_signals_to_gsheet(df_sig)
        
        # 2. Stage MOC Orders (Signal Close)
        save_moc_orders(all_signals, STRATEGY_BOOK, sheet_name='moc_orders')
        
        # 3. Stage Rest of Orders (Limits, MOO)
        save_staging_orders(all_signals, STRATEGY_BOOK, sheet_name='Order_Staging')
    else:
        print("No signals found today.")

    # 5. Send Email Summary
    send_email_summary(all_signals)

    print("--- Scan Complete ---")


if __name__ == "__main__":
    run_daily_scan()
