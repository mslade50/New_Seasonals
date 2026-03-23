"""
daily_risk_report.py

Daily Risk Report Email — Runs at 5:15 PM ET via GitHub Actions

Computes risk dashboard signals and fragility scores, then sends
an HTML email with inline gauge images and signal overlay chart.

Author: McKinley
"""

import pandas as pd
import numpy as np
import datetime
import os
import sys
import json
import smtplib
import tempfile
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# Path setup — ensure project root is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Must set a minimal env var so streamlit import doesn't break in headless mode
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pages.risk_dashboard_v2 import (
    refresh_all_data,
    load_cached_data,
    compute_da_signal,
    compute_vix_range_compression,
    compute_defensive_leadership,
    compute_fomc_signal,
    compute_low_ar_signal,
    compute_seasonal_divergence_signal,
    compute_price_context,
    compute_regime_multiplier,
    load_horizon_stats,
    compute_horizon_fragility,
    compute_fragility_timeseries,
    compute_similar_reading_returns,
    build_risk_dial,
    chart_signal_overlay,
    SECTOR_ETFS,
    CACHE_SPY_OHLC,
    CACHE_CLOSES,
    CACHE_SP500,
)

# Also import the decay metadata helper for DECAYING badge
from pages.risk_dashboard_v2 import _compute_decay_metadata


# ---------------------------------------------------------------------------
# 1. DATA DOWNLOAD
# ---------------------------------------------------------------------------

def download_data():
    """Download all risk data (10-year lookback) and return cached DataFrames."""
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365 * 10)).strftime("%Y-%m-%d")
    print("  Downloading risk data...")
    refresh_all_data(start_date, progress_callback=None)

    spy_df, closes, sp500_closes = load_cached_data()
    if spy_df is None or closes is None:
        raise RuntimeError("Failed to download/load risk data")
    return spy_df, closes, sp500_closes


# ---------------------------------------------------------------------------
# 2. COMPUTE SIGNALS (replicates _cached_compute_signals body)
# ---------------------------------------------------------------------------

def compute_all_signals(spy_df, closes, sp500_closes):
    """Compute all 6 signals and derived metrics."""
    spy_close = spy_df["Close"]

    sector_cols = [c for c in SECTOR_ETFS if c in closes.columns]
    sector_closes = closes[sector_cols].dropna(axis=1, how="all")
    sector_returns = sector_closes.pct_change().dropna(how="all")

    da = compute_da_signal(spy_df)
    vix_close = closes["^VIX"].dropna() if "^VIX" in closes.columns else pd.Series(dtype=float)
    vrc = compute_vix_range_compression(vix_close)
    dl = compute_defensive_leadership(sp500_closes, spy_close)
    fomc = compute_fomc_signal(spy_close)
    ar = compute_low_ar_signal(sector_returns, spy_close)
    srd = compute_seasonal_divergence_signal(spy_close)

    signals_ordered = {
        'Distribution Dominance': da,
        'VIX Range Compression': vrc,
        'Defensive Leadership': dl,
        'Pre-FOMC Rally': fomc,
        'Low Absorption Ratio': ar,
        'Seasonal Rank Divergence': srd,
    }

    price_ctx = compute_price_context(spy_close)
    regime_mult = compute_regime_multiplier(price_ctx)

    horizon_stats = load_horizon_stats()
    h_scores = None
    frag_df = None
    if horizon_stats is not None:
        h_scores = compute_horizon_fragility(
            signals_ordered, regime_mult, horizon_stats, price_ctx, spy_close
        )
        frag_df = compute_fragility_timeseries(signals_ordered, spy_close, horizon_stats)
        # 5d moving average for dial display (matches dashboard logic, line 2694)
        if frag_df is not None and len(frag_df) >= 1:
            h_scores = frag_df.rolling(5, min_periods=1).mean().iloc[-1].to_dict()

    return {
        'signals_ordered': signals_ordered,
        'price_ctx': price_ctx,
        'regime_mult': regime_mult,
        'horizon_stats': horizon_stats,
        'h_scores': h_scores,
        'frag_df': frag_df,
        'spy_close': spy_close,
    }


# ---------------------------------------------------------------------------
# 3. FORWARD RETURNS TABLE
# ---------------------------------------------------------------------------

def build_forward_returns_data(frag_df, spy_close, h_scores):
    """Compute forward returns for each horizon at current fragility reading."""
    if frag_df is None or h_scores is None:
        return {}

    results = {}
    for horizon in ['5d', '21d', '63d']:
        score = h_scores.get(horizon, 0)
        if score == 0:
            continue
        frag_series = frag_df[horizon].dropna()
        if frag_series.empty:
            continue
        ret = compute_similar_reading_returns(frag_series, spy_close, score)
        if ret is not None:
            results[horizon] = ret
    return results


# ---------------------------------------------------------------------------
# 4. IMAGE GENERATION
# ---------------------------------------------------------------------------

def generate_dial_image(h_scores, tmp_dir):
    """Generate combined 3-dial gauge image."""
    if h_scores is None:
        return None

    horizons = [('5d', '5-Day'), ('21d', '21-Day'), ('63d', '63-Day')]
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}] * 3],
        horizontal_spacing=0.05,
    )

    for i, (key, label) in enumerate(horizons):
        score = h_scores.get(key, 0)
        dial_fig = build_risk_dial(score, title=label)
        # Extract the indicator trace and add to subplot
        indicator = dial_fig.data[0]
        fig.add_trace(indicator, row=1, col=i + 1)

    fig.update_layout(
        height=200,
        width=750,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#ffffff"),
    )

    path = os.path.join(tmp_dir, "risk_dials.png")
    fig.write_image(path, scale=2)
    print(f"  Dial image saved: {path}")
    return path


def generate_overlay_image(spy_close, signals_ordered, tmp_dir):
    """Generate signal overlay chart (last 2 years)."""
    fig = chart_signal_overlay(spy_close, signals_ordered, year_filter=None)

    fig.update_layout(
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#ffffff"),
        height=500,
        width=900,
    )
    # Axis styling for dark background
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")

    path = os.path.join(tmp_dir, "signal_overlay.png")
    fig.write_image(path, scale=2)
    print(f"  Overlay image saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 5. HTML EMAIL BUILDER
# ---------------------------------------------------------------------------

def _status_badge(sig, price_ctx):
    """Return (badge_text, badge_color) for a signal."""
    if sig.get('on'):
        if sig.get('elevated'):
            return 'ELEVATED', '#FF4500'
        return 'ON', '#CC0000'

    # Check for decay
    dd = price_ctx.get('drawdown')
    spy_pct_from_high = abs(dd) if dd is not None and dd < 0 else 0.0
    decay = _compute_decay_metadata(sig, spy_pct_from_high)
    if decay is not None:
        return f"DECAYING ({decay['days_since']}d ago)", '#FFD700'

    return 'OFF', '#00CC00'


def build_html_email(computed, fwd_returns_data):
    """Build the full HTML email body."""
    price_ctx = computed['price_ctx']
    signals_ordered = computed['signals_ordered']
    h_scores = computed['h_scores']
    frag_df = computed['frag_df']

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    # --- Price Context Banner ---
    p = price_ctx
    spy_price = f"${p['price']:.2f}"
    ret_12m = f"{p['ret_12m']:+.1%}" if p['ret_12m'] is not None else "N/A"
    ext_200d = f"{p['extension_200d']:+.1%}" if p['extension_200d'] is not None else "N/A"
    dd_str = f"{p['drawdown']:+.1%}" if p['drawdown'] is not None else "N/A"
    regime = p['regime_label']

    # Regime color
    if 'drawdown' in regime.lower() or 'downtrend' in regime.lower() or 'correction' in regime.lower():
        regime_color = '#CC0000'
    elif 'extended' in regime.lower() or 'stretched' in regime.lower():
        regime_color = '#FF8C00'
    elif 'healthy' in regime.lower() or 'strong' in regime.lower():
        regime_color = '#00CC00'
    else:
        regime_color = '#FFD700'

    price_banner_html = f"""
    <div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 15px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 15px;">
            <div style="text-align: center;">
                <div style="color: #888; font-size: 11px; text-transform: uppercase;">SPY Price</div>
                <div style="color: #fff; font-size: 22px; font-weight: bold;">{spy_price}</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #888; font-size: 11px; text-transform: uppercase;">12M Return</div>
                <div style="color: {'#00CC00' if p.get('ret_12m', 0) and p['ret_12m'] >= 0 else '#CC0000'}; font-size: 22px; font-weight: bold;">{ret_12m}</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #888; font-size: 11px; text-transform: uppercase;">vs 200d SMA</div>
                <div style="color: {'#00CC00' if p.get('extension_200d', 0) and p['extension_200d'] >= 0 else '#CC0000'}; font-size: 22px; font-weight: bold;">{ext_200d}</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #888; font-size: 11px; text-transform: uppercase;">Drawdown</div>
                <div style="color: #CC0000; font-size: 22px; font-weight: bold;">{dd_str}</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #888; font-size: 11px; text-transform: uppercase;">Regime</div>
                <div style="color: {regime_color}; font-size: 16px; font-weight: bold;">{regime}</div>
            </div>
        </div>
    </div>
    """

    # --- Signal Board ---
    signal_rows = ""
    active_count = 0
    for name, sig in signals_ordered.items():
        badge_text, badge_color = _status_badge(sig, price_ctx)
        if sig.get('on'):
            active_count += 1

        detail = sig.get('detail', '') or sig.get('summary', '')
        # For OFF signals, show the summary metric instead
        if not sig.get('on') and badge_text == 'OFF':
            detail = sig.get('summary', '')

        signal_rows += f"""
        <tr>
            <td style="padding: 8px 12px; border-bottom: 1px solid #333; color: #fff; font-weight: bold; white-space: nowrap;">{name}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #333; text-align: center;">
                <span style="background: {badge_color}; color: #fff; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: bold;">{badge_text}</span>
            </td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #333; color: #aaa; font-size: 12px;">{detail}</td>
        </tr>
        """

    signal_board_html = f"""
    <div style="margin-bottom: 20px;">
        <h2 style="color: #fff; margin-bottom: 10px;">Signal Board ({active_count}/6 active)</h2>
        <table style="width: 100%; border-collapse: collapse; background: rgba(255,255,255,0.03); border-radius: 8px;">
            <thead>
                <tr style="border-bottom: 2px solid #444;">
                    <th style="padding: 8px 12px; text-align: left; color: #888; font-size: 11px; text-transform: uppercase;">Signal</th>
                    <th style="padding: 8px 12px; text-align: center; color: #888; font-size: 11px; text-transform: uppercase;">Status</th>
                    <th style="padding: 8px 12px; text-align: left; color: #888; font-size: 11px; text-transform: uppercase;">Detail</th>
                </tr>
            </thead>
            <tbody>
                {signal_rows}
            </tbody>
        </table>
    </div>
    """

    # --- Forward Returns Table ---
    fwd_returns_html = ""
    if fwd_returns_data:
        fwd_rows = ""
        for horizon in ['5d', '21d', '63d']:
            ret_data = fwd_returns_data.get(horizon)
            if ret_data is None:
                continue
            score = ret_data['current_score']
            n_episodes = ret_data['n_episodes']

            fwd_rows += f"""
            <tr style="border-bottom: 1px solid #444;">
                <td colspan="7" style="padding: 8px 12px; color: #FFD700; font-weight: bold; font-size: 13px;">
                    {horizon.upper()} Fragility = {score:.0f} | {n_episodes} historical episodes (band: {ret_data['band_low']:.0f}-{ret_data['band_high']:.0f})
                </td>
            </tr>
            <tr style="border-bottom: 1px solid #333;">
                <td style="padding: 4px 12px; color: #888; font-size: 11px; font-weight: bold;">Window</td>
                <td style="padding: 4px 12px; color: #888; font-size: 11px; font-weight: bold; text-align: right;">Mean</td>
                <td style="padding: 4px 12px; color: #888; font-size: 11px; font-weight: bold; text-align: right;">Median</td>
                <td style="padding: 4px 12px; color: #888; font-size: 11px; font-weight: bold; text-align: right;">% Negative</td>
                <td style="padding: 4px 12px; color: #888; font-size: 11px; font-weight: bold; text-align: right;">Mean Z</td>
                <td style="padding: 4px 12px; color: #888; font-size: 11px; font-weight: bold; text-align: right;">Median Z</td>
                <td style="padding: 4px 12px; color: #888; font-size: 11px; font-weight: bold; text-align: right;">Baseline</td>
            </tr>
            """

            for window, stats in ret_data['returns'].items():
                if stats is None:
                    continue
                mean_color = '#00CC00' if stats['mean'] >= 0 else '#CC0000'
                baseline_color = '#00CC00' if stats['uncond_mean'] >= 0 else '#CC0000'
                fwd_rows += f"""
                <tr>
                    <td style="padding: 4px 12px; color: #fff; border-bottom: 1px solid #2a2a2a;">{window}d</td>
                    <td style="padding: 4px 12px; color: {mean_color}; text-align: right; font-weight: bold; font-family: monospace; border-bottom: 1px solid #2a2a2a;">{stats['mean']:+.2%}</td>
                    <td style="padding: 4px 12px; color: #aaa; text-align: right; font-family: monospace; border-bottom: 1px solid #2a2a2a;">{stats['median']:+.2%}</td>
                    <td style="padding: 4px 12px; color: {'#CC0000' if stats['pct_neg'] > 0.5 else '#aaa'}; text-align: right; font-family: monospace; border-bottom: 1px solid #2a2a2a;">{stats['pct_neg']:.0%}</td>
                    <td style="padding: 4px 12px; color: {'#CC0000' if stats.get('mean_z', 0) < -1 else '#FFD700' if stats.get('mean_z', 0) < 0 else '#00CC00'}; text-align: right; font-family: monospace; border-bottom: 1px solid #2a2a2a;">{stats.get('mean_z', 0):+.2f}</td>
                    <td style="padding: 4px 12px; color: {'#CC0000' if stats.get('median_z', 0) < -1 else '#FFD700' if stats.get('median_z', 0) < 0 else '#00CC00'}; text-align: right; font-family: monospace; border-bottom: 1px solid #2a2a2a;">{stats.get('median_z', 0):+.2f}</td>
                    <td style="padding: 4px 12px; color: {baseline_color}; text-align: right; font-family: monospace; border-bottom: 1px solid #2a2a2a;">{stats['uncond_mean']:+.2%}</td>
                </tr>
                """

        if fwd_rows:
            fwd_returns_html = f"""
            <div style="margin-bottom: 20px;">
                <h2 style="color: #fff; margin-bottom: 10px;">Forward Returns at Similar Fragility</h2>
                <table style="width: 100%; border-collapse: collapse; background: rgba(255,255,255,0.03); border-radius: 8px;">
                    {fwd_rows}
                </table>
            </div>
            """

    # --- Dial scores for subject line ---
    s5 = h_scores.get('5d', 0) if h_scores else 0
    s21 = h_scores.get('21d', 0) if h_scores else 0
    s63 = h_scores.get('63d', 0) if h_scores else 0

    subject = f"Risk Report \u2014 {date_str} | Fragility: {s5:.0f}/{s21:.0f}/{s63:.0f}"

    # --- Full HTML ---
    html = f"""
    <html>
        <head>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #0d1117;
                    color: #ffffff;
                }}
                .container {{
                    max-width: 900px;
                    margin: 0 auto;
                    background: #1a1a2e;
                    border-radius: 12px;
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #16213e, #1a1a2e);
                    padding: 25px;
                    text-align: center;
                    border-bottom: 2px solid #333;
                }}
                .section {{
                    padding: 20px 25px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 style="margin: 0; font-size: 24px; color: #fff;">Risk Report</h1>
                    <div style="font-size: 14px; opacity: 0.7; margin-top: 5px;">{date_str} | {regime}</div>
                </div>

                <div class="section">
                    {price_banner_html}
                    {signal_board_html}

                    <h2 style="color: #fff; margin-bottom: 10px;">Fragility Dials (5d / 21d / 63d)</h2>
                    <div style="text-align: center;">
                        <img src="cid:risk_dials" style="max-width: 100%; border-radius: 8px;">
                    </div>

                    {fwd_returns_html}

                    <h2 style="color: #fff; margin-bottom: 10px; margin-top: 25px;">Signal Overlay (2yr)</h2>
                    <div style="text-align: center;">
                        <img src="cid:signal_overlay" style="max-width: 100%; border-radius: 8px;">
                    </div>
                </div>

                <div style="text-align: center; padding: 15px; color: #555; font-size: 11px; border-top: 1px solid #333;">
                    Generated by daily_risk_report.py | {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                </div>
            </div>
        </body>
    </html>
    """

    return subject, html


# ---------------------------------------------------------------------------
# 6. SEND EMAIL
# ---------------------------------------------------------------------------

def send_email(subject, html_content, dial_path, overlay_path):
    """Send the risk report email with inline images."""
    sender_email = os.environ.get("EMAIL_USER")
    sender_password = os.environ.get("EMAIL_PASS")
    receiver_email = "mckinleyslade@gmail.com"

    if not sender_email or not sender_password:
        print("  EMAIL_USER / EMAIL_PASS not set — skipping send")
        print(f"  Subject: {subject}")
        return False

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    msg.attach(MIMEText(html_content, "html"))

    # Attach inline images
    for img_path, cid in [(dial_path, "risk_dials"), (overlay_path, "signal_overlay")]:
        if img_path and os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-ID', f'<{cid}>')
                img.add_header('Content-Disposition', 'inline', filename=f'{cid}.png')
                msg.attach(img)

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
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DAILY RISK REPORT")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Download data
    print("\n[1/6] Downloading data...")
    spy_df, closes, sp500_closes = download_data()

    # 2. Compute signals
    print("[2/6] Computing signals...")
    computed = compute_all_signals(spy_df, closes, sp500_closes)

    # 2b. Cache fragility timeseries + environment snapshot for PM dashboard/briefing
    data_dir = os.path.join(current_dir, "data")
    frag_df = computed.get('frag_df')
    if frag_df is not None and not frag_df.empty:
        frag_cache_path = os.path.join(data_dir, "rd2_fragility.parquet")
        frag_smoothed = frag_df.rolling(5, min_periods=1).mean()
        frag_smoothed.to_parquet(frag_cache_path)
        print(f"  Cached fragility timeseries ({len(frag_smoothed)} rows)")

    # Save environment snapshot (price context + h_scores + signal summaries)
    env_snapshot = {
        'date': datetime.datetime.now().strftime('%Y-%m-%d'),
        'price_ctx': computed['price_ctx'],
        'h_scores': computed.get('h_scores'),
        'signals': {},
    }
    for name, sig in computed['signals_ordered'].items():
        env_snapshot['signals'][name] = {
            'on': bool(sig.get('on', False)),
            'detail': sig.get('detail', ''),
            'summary': sig.get('summary', ''),
        }
    env_path = os.path.join(data_dir, "rd2_environment.json")
    with open(env_path, 'w') as f:
        json.dump(env_snapshot, f, indent=2, default=str)
    print(f"  Cached environment snapshot to {env_path}")

    # 3. Forward returns
    print("[3/6] Computing forward returns...")
    fwd_returns_data = build_forward_returns_data(
        computed['frag_df'], computed['spy_close'], computed['h_scores']
    )

    # 4. Generate images
    print("[4/6] Generating images...")
    tmp_dir = tempfile.mkdtemp()
    dial_path = generate_dial_image(computed['h_scores'], tmp_dir)
    overlay_path = generate_overlay_image(
        computed['spy_close'], computed['signals_ordered'], tmp_dir
    )

    # 5. Build email
    print("[5/6] Building email...")
    subject, html = build_html_email(computed, fwd_returns_data)

    # 6. Send
    print("[6/6] Sending email...")
    send_email(subject, html, dial_path, overlay_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
