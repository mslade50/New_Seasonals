"""
daily_pm_briefing.py

Consolidated Morning PM Briefing Email — Runs AFTER the three existing reports
in GitHub Actions. Reads cached outputs rather than recomputing.

Sections:
1. Environment — from risk dashboard signal computation
2. Portfolio — from cached backtest_sig_df.parquet
3. Today's Signals — from cached signal state
4. Action Items — decision queue logic

Uses existing Gmail SMTP pattern from daily_risk_report.py.

Author: McKinley
"""

import pandas as pd
import numpy as np
import datetime
import os
import sys
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from strategy_config import _STRATEGY_BOOK_RAW, ACCOUNT_VALUE
except ImportError:
    _STRATEGY_BOOK_RAW = []
    ACCOUNT_VALUE = 750000

DATA_DIR = os.path.join(current_dir, "data")
ENV_CACHE = os.path.join(DATA_DIR, "rd2_environment.json")

# Ticker → sector (same as pm_dashboard.py)
TICKER_SECTOR = {
    'AAPL': 'Tech', 'ADBE': 'Tech', 'ADI': 'Tech', 'AMD': 'Tech', 'AVGO': 'Tech',
    'CRM': 'Tech', 'CSCO': 'Tech', 'GOOG': 'Tech', 'IBM': 'Tech', 'INTC': 'Tech',
    'META': 'Tech', 'MSFT': 'Tech', 'MU': 'Tech', 'NVDA': 'Tech', 'ORCL': 'Tech',
    'QCOM': 'Tech', 'TXN': 'Tech', 'QQQ': 'Tech', 'SMH': 'Tech', 'XLK': 'Tech',
    'ABT': 'HC', 'AMGN': 'HC', 'BMY': 'HC', 'CVS': 'HC', 'GILD': 'HC',
    'JNJ': 'HC', 'LLY': 'HC', 'MRK': 'HC', 'PFE': 'HC', 'UNH': 'HC',
    'XLV': 'HC', 'IBB': 'HC',
    'BAC': 'Fin', 'C': 'Fin', 'GS': 'Fin', 'JPM': 'Fin', 'MS': 'Fin',
    'SCHW': 'Fin', 'WFC': 'Fin', 'XLF': 'Fin', 'KRE': 'Fin',
    'AMZN': 'ConDisc', 'DIS': 'ConDisc', 'HD': 'ConDisc', 'LOW': 'ConDisc',
    'NKE': 'ConDisc', 'SBUX': 'ConDisc', 'TGT': 'ConDisc', 'XLY': 'ConDisc',
    'KO': 'Staples', 'PEP': 'Staples', 'PG': 'Staples', 'COST': 'Staples',
    'WMT': 'Staples', 'XLP': 'Staples',
    'BA': 'Indust', 'CAT': 'Indust', 'DE': 'Indust', 'HON': 'Indust',
    'LMT': 'Indust', 'UNP': 'Indust', 'XLI': 'Indust',
    'CVX': 'Energy', 'COP': 'Energy', 'XOM': 'Energy', 'XLE': 'Energy',
    'NEE': 'Util', 'DUK': 'Util', 'SO': 'Util', 'XLU': 'Util',
    'SPY': 'Index', 'DIA': 'Index', 'IWM': 'Index',
}


# ---------------------------------------------------------------------------
# 1. COMPUTE ENVIRONMENT
# ---------------------------------------------------------------------------

def compute_environment():
    """Load environment from cached JSON (written by daily_risk_report.py)."""
    print("  Loading cached environment snapshot...")
    if not os.path.exists(ENV_CACHE):
        raise RuntimeError(f"Environment cache not found at {ENV_CACHE}. Run daily_risk_report.py first.")

    with open(ENV_CACHE, 'r') as f:
        snapshot = json.load(f)

    signals_ordered = {}
    for name, sig in snapshot.get('signals', {}).items():
        signals_ordered[name] = sig

    return {
        'price_ctx': snapshot.get('price_ctx', {}),
        'signals_ordered': signals_ordered,
        'h_scores': snapshot.get('h_scores'),
        'active_count': sum(1 for s in signals_ordered.values() if s.get('on')),
    }


# ---------------------------------------------------------------------------
# 2. LOAD PORTFOLIO
# ---------------------------------------------------------------------------

def load_portfolio():
    """Load cached backtest sig_df."""
    cache_path = os.path.join(DATA_DIR, "backtest_sig_df.parquet")
    if not os.path.exists(cache_path):
        print(f"  No backtest cache at {cache_path}")
        return None, None

    sig_df = pd.read_parquet(cache_path)
    for col in ['Date', 'Entry Date', 'Exit Date', 'Time Stop']:
        if col in sig_df.columns:
            sig_df[col] = pd.to_datetime(sig_df[col])

    today = pd.Timestamp(datetime.date.today())
    open_mask = sig_df['Exit Date'] >= today
    open_df = sig_df[open_mask].copy()
    return sig_df, open_df


# ---------------------------------------------------------------------------
# 3. DECISION QUEUE
# ---------------------------------------------------------------------------

def compute_decision_queue(env, open_df):
    """Generate list of (severity, message) alerts."""
    alerts = []

    if env and env.get('h_scores'):
        frag_21d = env['h_scores'].get('21d', 0)
        if frag_21d >= 90:
            alerts.append(('CRITICAL', f"21d Fragility at {frag_21d:.0f} — Consider hedge overlay"))
        elif frag_21d >= 70:
            min_mult = max(0.5, 1.0 - (frag_21d / 100) * 0.5)
            alerts.append(('WARNING', f"21d Fragility at {frag_21d:.0f} — Consider reducing sizes to {min_mult:.0%}"))

    if env:
        active_count = env.get('active_count', 0)
        if active_count >= 3:
            active_names = [name for name, sig in env['signals_ordered'].items() if sig.get('on')]
            alerts.append(('WARNING', f"{active_count} risk signals active: {', '.join(active_names)}"))

    if open_df is not None and not open_df.empty:
        open_sect = open_df.copy()
        open_sect['Sector'] = open_sect['Ticker'].map(TICKER_SECTOR).fillna('Other')
        open_sect['Dollar Exposure'] = open_sect['Price'] * open_sect['Shares']
        total_exp = open_sect['Dollar Exposure'].sum()
        if total_exp > 0:
            sector_pcts = open_sect.groupby('Sector')['Dollar Exposure'].sum() / total_exp
            max_sector = sector_pcts.idxmax()
            max_pct = sector_pcts.max()
            if max_pct > 0.40:
                alerts.append(('WARNING', f"Portfolio is {max_pct:.0%} concentrated in {max_sector}"))

    return alerts


# ---------------------------------------------------------------------------
# 4. HTML EMAIL BUILDER
# ---------------------------------------------------------------------------

def build_briefing_email(env, sig_df, open_df, alerts):
    """Build the consolidated PM briefing HTML email."""
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    # --- Section 1: Environment ---
    env_html = ""
    if env:
        p = env['price_ctx']
        spy_price = f"${p['price']:.2f}"
        ret_12m = f"{p.get('ret_12m', 0):+.1%}" if p.get('ret_12m') is not None else "N/A"
        ext_200d = f"{p.get('extension_200d', 0):+.1%}" if p.get('extension_200d') is not None else "N/A"
        dd = f"{p.get('drawdown', 0):+.1%}" if p.get('drawdown') is not None else "N/A"
        regime = p.get('regime_label', 'Unknown')

        h = env.get('h_scores', {})
        frag_str = f"{h.get('5d', 0):.0f} / {h.get('21d', 0):.0f} / {h.get('63d', 0):.0f}"

        # Signal board
        signal_rows = ""
        for name, sig in env['signals_ordered'].items():
            is_on = sig.get('on', False)
            badge = f'<span style="background:{("#CC0000" if is_on else "#00CC00")};color:#fff;padding:2px 8px;border-radius:10px;font-size:11px;">{"ON" if is_on else "OFF"}</span>'
            detail = sig.get('detail', '') or sig.get('summary', '')
            signal_rows += f"<tr><td style='padding:4px 8px;color:#fff;border-bottom:1px solid #333;'>{name}</td><td style='padding:4px 8px;text-align:center;border-bottom:1px solid #333;'>{badge}</td><td style='padding:4px 8px;color:#aaa;font-size:12px;border-bottom:1px solid #333;'>{detail}</td></tr>"

        env_html = f"""
        <h2 style="color:#fff;margin-bottom:10px;">1. Environment</h2>
        <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:15px;margin-bottom:15px;">
            <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;">
                <div style="text-align:center;"><div style="color:#888;font-size:10px;">SPY</div><div style="color:#fff;font-size:18px;font-weight:bold;">{spy_price}</div></div>
                <div style="text-align:center;"><div style="color:#888;font-size:10px;">12M</div><div style="color:#fff;font-size:18px;">{ret_12m}</div></div>
                <div style="text-align:center;"><div style="color:#888;font-size:10px;">vs 200d</div><div style="color:#fff;font-size:18px;">{ext_200d}</div></div>
                <div style="text-align:center;"><div style="color:#888;font-size:10px;">DD</div><div style="color:#CC0000;font-size:18px;">{dd}</div></div>
                <div style="text-align:center;"><div style="color:#888;font-size:10px;">Regime</div><div style="color:#FFD700;font-size:14px;">{regime}</div></div>
                <div style="text-align:center;"><div style="color:#888;font-size:10px;">Fragility (5/21/63)</div><div style="color:#FF8C00;font-size:18px;font-weight:bold;">{frag_str}</div></div>
            </div>
        </div>
        <table style="width:100%;border-collapse:collapse;background:rgba(255,255,255,0.03);border-radius:8px;">
            <thead><tr style="border-bottom:2px solid #444;">
                <th style="padding:4px 8px;text-align:left;color:#888;font-size:10px;">Signal</th>
                <th style="padding:4px 8px;text-align:center;color:#888;font-size:10px;">Status</th>
                <th style="padding:4px 8px;text-align:left;color:#888;font-size:10px;">Detail</th>
            </tr></thead>
            <tbody>{signal_rows}</tbody>
        </table>
        """
    else:
        env_html = "<h2 style='color:#fff;'>1. Environment</h2><p style='color:#888;'>Risk data not available.</p>"

    # --- Section 2: Portfolio ---
    port_html = "<h2 style='color:#fff;margin-top:20px;'>2. Portfolio</h2>"
    if open_df is not None and not open_df.empty:
        long_count = len(open_df[open_df['Action'] == 'BUY'])
        short_count = len(open_df[open_df['Action'] == 'SELL SHORT'])
        total_risk = open_df['Risk $'].sum() if 'Risk $' in open_df.columns else 0
        total_exposure = (open_df['Price'] * open_df['Shares']).sum()

        pos_rows = ""
        for _, row in open_df.iterrows():
            sector = TICKER_SECTOR.get(row['Ticker'], '?')
            pos_rows += f"<tr><td style='padding:3px 6px;color:#fff;border-bottom:1px solid #333;font-size:12px;'>{row['Strategy']}</td><td style='padding:3px 6px;color:#fff;border-bottom:1px solid #333;font-size:12px;'>{row['Ticker']}</td><td style='padding:3px 6px;color:#888;border-bottom:1px solid #333;font-size:12px;'>{sector}</td><td style='padding:3px 6px;color:#fff;border-bottom:1px solid #333;font-size:12px;'>{row['Action']}</td><td style='padding:3px 6px;color:#fff;border-bottom:1px solid #333;font-size:12px;'>{row.get('Shares', 0):.0f}</td><td style='padding:3px 6px;color:#fff;border-bottom:1px solid #333;font-size:12px;'>{row.get('Exit Date', pd.NaT):%Y-%m-%d}</td></tr>"

        port_html += f"""
        <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:10px;margin-bottom:10px;">
            <span style="color:#fff;">Open: <b>{len(open_df)}</b> ({long_count}L / {short_count}S)</span> |
            <span style="color:#fff;">Exposure: <b>${total_exposure:,.0f}</b> ({total_exposure/ACCOUNT_VALUE:.1%})</span> |
            <span style="color:#fff;">Risk: <b>${total_risk:,.0f}</b></span>
        </div>
        <table style="width:100%;border-collapse:collapse;background:rgba(255,255,255,0.03);font-size:12px;">
            <thead><tr style="border-bottom:2px solid #444;">
                <th style="padding:3px 6px;text-align:left;color:#888;font-size:10px;">Strategy</th>
                <th style="padding:3px 6px;text-align:left;color:#888;font-size:10px;">Ticker</th>
                <th style="padding:3px 6px;text-align:left;color:#888;font-size:10px;">Sector</th>
                <th style="padding:3px 6px;text-align:left;color:#888;font-size:10px;">Side</th>
                <th style="padding:3px 6px;text-align:left;color:#888;font-size:10px;">Shares</th>
                <th style="padding:3px 6px;text-align:left;color:#888;font-size:10px;">Exit</th>
            </tr></thead>
            <tbody>{pos_rows}</tbody>
        </table>
        """
    else:
        port_html += "<p style='color:#888;'>No open positions.</p>"

    # --- Section 3: Recent Activity ---
    activity_html = "<h2 style='color:#fff;margin-top:20px;'>3. Recent Activity</h2>"
    if sig_df is not None and not sig_df.empty:
        today = pd.Timestamp(datetime.date.today())
        recent = sig_df[sig_df['Entry Date'] >= today - pd.tseries.offsets.BDay(3)].sort_values('Entry Date', ascending=False)
        if not recent.empty:
            act_rows = ""
            for _, row in recent.head(15).iterrows():
                act_rows += f"<tr><td style='padding:3px 6px;color:#fff;border-bottom:1px solid #333;font-size:12px;'>{row.get('Entry Date', pd.NaT):%Y-%m-%d}</td><td style='padding:3px 6px;color:#fff;border-bottom:1px solid #333;font-size:12px;'>{row['Strategy']}</td><td style='padding:3px 6px;color:#fff;border-bottom:1px solid #333;font-size:12px;'>{row['Ticker']}</td><td style='padding:3px 6px;color:#fff;border-bottom:1px solid #333;font-size:12px;'>{row['Action']}</td><td style='padding:3px 6px;color:#fff;border-bottom:1px solid #333;font-size:12px;'>${row.get('Price', 0):.2f}</td></tr>"
            activity_html += f"""
            <table style="width:100%;border-collapse:collapse;background:rgba(255,255,255,0.03);font-size:12px;">
                <thead><tr style="border-bottom:2px solid #444;">
                    <th style="padding:3px 6px;text-align:left;color:#888;font-size:10px;">Date</th>
                    <th style="padding:3px 6px;text-align:left;color:#888;font-size:10px;">Strategy</th>
                    <th style="padding:3px 6px;text-align:left;color:#888;font-size:10px;">Ticker</th>
                    <th style="padding:3px 6px;text-align:left;color:#888;font-size:10px;">Side</th>
                    <th style="padding:3px 6px;text-align:left;color:#888;font-size:10px;">Price</th>
                </tr></thead>
                <tbody>{act_rows}</tbody>
            </table>
            """
        else:
            activity_html += "<p style='color:#888;'>No recent entries (last 3 days).</p>"
    else:
        activity_html += "<p style='color:#888;'>No backtest data available.</p>"

    # --- Section 4: Action Items ---
    action_html = "<h2 style='color:#fff;margin-top:20px;'>4. Action Items</h2>"
    if alerts:
        for severity, msg in alerts:
            if severity == 'CRITICAL':
                color = '#CC0000'
                icon = '🔴'
            else:
                color = '#FF8C00'
                icon = '🟡'
            action_html += f"<div style='background:rgba(255,255,255,0.05);border-left:3px solid {color};padding:8px 12px;margin-bottom:8px;border-radius:4px;'><span style='color:{color};font-weight:bold;'>{icon} {severity}:</span> <span style='color:#fff;'>{msg}</span></div>"
    else:
        action_html += "<div style='background:rgba(0,204,0,0.1);border-left:3px solid #00CC00;padding:8px 12px;border-radius:4px;'><span style='color:#00CC00;font-weight:bold;'>✅ ALL CLEAR:</span> <span style='color:#fff;'>Environment is benign and portfolio is balanced.</span></div>"

    # Fragility scores for subject line
    h = env.get('h_scores', {}) if env else {}
    s5 = h.get('5d', 0)
    s21 = h.get('21d', 0)
    s63 = h.get('63d', 0)

    n_open = len(open_df) if open_df is not None else 0
    n_alerts = len(alerts)

    subject = f"PM Briefing — {date_str} | Frag: {s5:.0f}/{s21:.0f}/{s63:.0f} | {n_open} open | {n_alerts} alerts"

    html = f"""
    <html>
        <head>
            <style>
                body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #ffffff; }}
                .container {{ max-width: 900px; margin: 0 auto; background: #1a1a2e; border-radius: 12px; overflow: hidden; }}
                .header {{ background: linear-gradient(135deg, #16213e, #1a1a2e); padding: 20px; text-align: center; border-bottom: 2px solid #333; }}
                .header h1 {{ color: #fff; margin: 0; font-size: 22px; }}
                .header p {{ color: #888; margin: 5px 0 0; font-size: 12px; }}
                .body {{ padding: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 PM Briefing</h1>
                    <p>{date_str} | Generated {datetime.datetime.now().strftime('%H:%M ET')}</p>
                </div>
                <div class="body">
                    {action_html}
                    {env_html}
                    {port_html}
                    {activity_html}
                </div>
            </div>
        </body>
    </html>
    """

    return subject, html


# ---------------------------------------------------------------------------
# 5. SEND EMAIL
# ---------------------------------------------------------------------------

def send_email(subject, html_content):
    """Send the PM briefing email."""
    sender_email = os.environ.get("EMAIL_USER")
    sender_password = os.environ.get("EMAIL_PASS")
    receiver_email = "mckinleyslade@gmail.com"

    if not sender_email or not sender_password:
        print(f"  EMAIL_USER / EMAIL_PASS not set — skipping send")
        print(f"  Subject: {subject}")
        return False

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg.attach(MIMEText(html_content, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print(f"  Email sent to {receiver_email}")
        return True
    except Exception as e:
        print(f"  Failed to send email: {e}")
        return False


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DAILY PM BRIEFING")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Environment
    print("\n[1/4] Computing environment...")
    try:
        env = compute_environment()
        active = env.get('active_count', 0)
        h = env.get('h_scores', {})
        print(f"  Signals active: {active}/6 | Fragility: {h.get('5d',0):.0f}/{h.get('21d',0):.0f}/{h.get('63d',0):.0f}")
    except Exception as e:
        print(f"  Environment failed: {e}")
        env = None

    # 2. Portfolio
    print("[2/4] Loading portfolio...")
    sig_df, open_df = load_portfolio()
    n_open = len(open_df) if open_df is not None else 0
    print(f"  Open positions: {n_open}")

    # 3. Decision queue
    print("[3/4] Computing decision queue...")
    alerts = compute_decision_queue(env, open_df)
    print(f"  Alerts: {len(alerts)}")
    for severity, msg in alerts:
        print(f"    [{severity}] {msg}")

    # 4. Build & send email
    print("[4/4] Building & sending email...")
    subject, html = build_briefing_email(env, sig_df, open_df, alerts)
    send_email(subject, html)

    print("\nDone.")


if __name__ == "__main__":
    main()
