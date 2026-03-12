"""
weekly_market_rundown.py

Weekly Market Rundown PDF — Runs Sunday 11 PM UTC (6 PM ET) via GitHub Actions

Computes all risk dashboard signals, exports each chart as high-res PNG,
builds a multi-page landscape PDF, and emails it as an attachment.

Author: McKinley
"""

import pandas as pd
import numpy as np
import datetime
import os
import sys
import tempfile
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fpdf import FPDF

from pages.risk_dashboard_v2 import (
    refresh_all_data,
    load_cached_data,
    compute_da_signal,
    compute_vix_range_compression,
    compute_defensive_leadership,
    compute_fomc_signal,
    compute_low_ar_signal,
    compute_seasonal_divergence_signal,
    compute_dispersion_signal,
    compute_price_context,
    compute_regime_multiplier,
    load_horizon_stats,
    compute_horizon_fragility,
    compute_fragility_timeseries,
    compute_regime_deep_dive,
    build_risk_dial,
    chart_signal_overlay,
    chart_fragility_timeseries,
    chart_da_ratio,
    chart_vix_compression,
    chart_leadership,
    chart_fomc_signals,
    chart_ar_signal,
    chart_seasonal_divergence,
    chart_dispersion_signal,
    chart_absorption_ratio,
    _compute_decay_metadata,
    _add_signal_vlines,
    _assign_regime_bucket,
    SECTOR_ETFS,
    FOMC_DATES,
    REGIME_ORDER,
    REGIME_COLORS,
)

from daily_scan import load_seasonal_map
from indicators import get_sznl_val_series


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def _latin1_safe(text):
    """Replace non-Latin-1 characters so fpdf2 doesn't choke."""
    if not isinstance(text, str):
        return str(text)
    return text.encode("latin-1", errors="replace").decode("latin-1")


CHART_WIDTH = 1800
CHART_HEIGHT = 900
CHART_SCALE = 2
DARK_BG = {"paper_bgcolor": "#1a1a2e", "plot_bgcolor": "#16213e"}
AXIS_GRID = "rgba(128,128,128,0.2)"

# PDF dimensions (landscape letter)
PDF_W = 279.4  # mm
PDF_H = 215.9  # mm


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
# 2. COMPUTE SIGNALS
# ---------------------------------------------------------------------------

def compute_all_signals(spy_df, closes, sp500_closes):
    """Compute all 7 signals (6 base + dispersion) and derived metrics."""
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
    disp = compute_dispersion_signal(sp500_closes, spy_df, spy_close)

    signals_ordered = {
        'Distribution Dominance': da,
        'VIX Range Compression': vrc,
        'Defensive Leadership': dl,
        'Pre-FOMC Rally': fomc,
        'Low Absorption Ratio': ar,
        'Seasonal Rank Divergence': srd,
        'Dispersion Signal': disp,
    }

    price_ctx = compute_price_context(spy_close)
    regime_mult = compute_regime_multiplier(price_ctx)

    horizon_stats = load_horizon_stats()
    h_scores = None
    h_scores_10d = None
    frag_df = None
    if horizon_stats is not None:
        h_scores = compute_horizon_fragility(
            signals_ordered, regime_mult, horizon_stats, price_ctx, spy_close
        )
        frag_df = compute_fragility_timeseries(signals_ordered, spy_close, horizon_stats)
        if frag_df is not None and len(frag_df) >= 1:
            h_scores = frag_df.rolling(5, min_periods=1).mean().iloc[-1].to_dict()
            h_scores_10d = frag_df.rolling(10, min_periods=1).mean().iloc[-1].to_dict()

    # Extension vs 200d SMA — percentile rank over full history
    ext_200d_pctile = None
    if len(spy_close) >= 200:
        sma_200 = spy_close.rolling(200).mean()
        ext_series = (spy_close / sma_200 - 1).dropna()
        if len(ext_series) > 0:
            current_ext = ext_series.iloc[-1]
            ext_200d_pctile = (ext_series < current_ext).mean() * 100

    # YTD return
    ytd_return = None
    today = spy_close.index[-1]
    year_start_candidates = spy_close.loc[spy_close.index.year == today.year]
    if len(year_start_candidates) > 0:
        first_close = float(year_start_candidates.iloc[0])
        ytd_return = float(spy_close.iloc[-1]) / first_close - 1

    # Seasonal rank for SPY
    seasonal_rank = None
    try:
        sznl_map = load_seasonal_map("sznl_ranks.csv")
        spy_sznl = get_sznl_val_series("SPY", spy_close.index, sznl_map)
        seasonal_rank = float(spy_sznl.iloc[-1])
    except Exception:
        pass

    return {
        'signals_ordered': signals_ordered,
        'price_ctx': price_ctx,
        'regime_mult': regime_mult,
        'horizon_stats': horizon_stats,
        'h_scores': h_scores,
        'h_scores_10d': h_scores_10d,
        'frag_df': frag_df,
        'spy_close': spy_close,
        'spy_df': spy_df,
        'closes': closes,
        'sp500_closes': sp500_closes,
        'vix_close': vix_close,
        'ext_200d_pctile': ext_200d_pctile,
        'ytd_return': ytd_return,
        'seasonal_rank': seasonal_rank,
        'da': da,
        'vrc': vrc,
        'dl': dl,
        'fomc': fomc,
        'ar': ar,
        'srd': srd,
        'disp': disp,
    }


# ---------------------------------------------------------------------------
# 3. CHART EXPORT HELPERS
# ---------------------------------------------------------------------------

def _style_fig(fig, width=CHART_WIDTH, height=CHART_HEIGHT):
    """Apply dark theme and sizing to a chart figure."""
    fig.update_layout(
        **DARK_BG,
        font=dict(color="#ffffff", size=14),
        height=height,
        width=width,
        margin=dict(l=60, r=30, t=60, b=40),
    )
    fig.update_xaxes(gridcolor=AXIS_GRID)
    fig.update_yaxes(gridcolor=AXIS_GRID)
    return fig


def _save_fig(fig, path):
    """Write figure to PNG."""
    fig.write_image(path, scale=CHART_SCALE)
    print(f"  Saved: {os.path.basename(path)}")


def _fomc_upcoming():
    """Return True if an FOMC meeting is within 0-7 calendar days from now."""
    today = pd.Timestamp.now().normalize()
    for d in FOMC_DATES:
        delta = (d - today).days
        if 0 <= delta <= 7:
            return True
    return False


# ---------------------------------------------------------------------------
# 4. GENERATE ALL CHART IMAGES
# ---------------------------------------------------------------------------

def generate_charts(computed, tmp_dir):
    """Generate all chart PNGs and return ordered list of (path, title) tuples."""
    sigs = computed['signals_ordered']
    spy_close = computed['spy_close']
    da = computed['da']
    vrc = computed['vrc']
    dl = computed['dl']
    fomc = computed['fomc']
    ar = computed['ar']
    srd = computed['srd']
    disp = computed['disp']
    vix_close = computed['vix_close']
    frag_df = computed['frag_df']
    charts = []

    # --- Signal Overlay ---
    fig = chart_signal_overlay(spy_close, sigs, year_filter=None)
    # Show legend (signal colors), hide y-axis tick labels on bottom panel,
    # remove the chart title (page title handles it)
    fig.update_layout(
        showlegend=True,
        title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, font=dict(size=13)),
    )
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    _style_fig(fig)
    p = os.path.join(tmp_dir, "02_signal_overlay.png")
    _save_fig(fig, p)
    charts.append((p, "Signal Overlay - Gantt Timeline + SPY"))

    # --- Regime Deep Dive Table (early, right after overlay) ---
    regime_path = _generate_regime_table(computed, tmp_dir)
    if regime_path:
        charts.append((regime_path, "Regime Deep Dive - 63d Horizon, 10d Avg"))

    # --- Fragility Timeseries (63d) ---
    if frag_df is not None:
        fig = chart_fragility_timeseries(frag_df, spy_close, "63d", year_filter=None)
        _style_fig(fig)
        p = os.path.join(tmp_dir, "03_fragility_ts.png")
        _save_fig(fig, p)
        charts.append((p, "63-Day Fragility Timeseries"))

    # --- D/A Ratio ---
    if len(da.get('da_ratio', pd.Series(dtype=float)).dropna()) > 0:
        fig = chart_da_ratio(da['da_ratio'], spy_close, year_filter=None)
        _add_signal_vlines(fig, da.get('signal_history'))
        _style_fig(fig)
        p = os.path.join(tmp_dir, "04_da_ratio.png")
        _save_fig(fig, p)
        charts.append((p, "Distribution / Absorption Ratio"))

    # --- VIX Compression ---
    if len(vrc.get('compression_pctile', pd.Series(dtype=float)).dropna()) > 0:
        fig = chart_vix_compression(vix_close, vrc['compression_pctile'], year_filter=None)
        _add_signal_vlines(fig, vrc.get('signal_history'))
        _style_fig(fig)
        p = os.path.join(tmp_dir, "05_vix_compression.png")
        _save_fig(fig, p)
        charts.append((p, "VIX Level + Range Compression"))

    # --- Leadership ---
    if len(dl.get('spread', pd.Series(dtype=float)).dropna()) > 0:
        fig = chart_leadership(dl['on_breadth'], dl['off_breadth'], spy_close, year_filter=None)
        _add_signal_vlines(fig, dl.get('signal_history'))
        _style_fig(fig)
        p = os.path.join(tmp_dir, "06_leadership.png")
        _save_fig(fig, p)
        charts.append((p, "Risk-On vs Risk-Off Breadth"))

    # --- FOMC (conditional) ---
    if _fomc_upcoming() and len(fomc.get('signal_dates', [])) > 0:
        fig = chart_fomc_signals(spy_close, fomc['signal_dates'], year_filter=None)
        _style_fig(fig)
        p = os.path.join(tmp_dir, "07_fomc.png")
        _save_fig(fig, p)
        charts.append((p, "Pre-FOMC Rally Signal"))

    # --- AR Signal ---
    if len(ar.get('ar_pctile', pd.Series(dtype=float)).dropna()) > 0:
        fig = chart_ar_signal(ar['ar_pctile'], spy_close, year_filter=None)
        _add_signal_vlines(fig, ar.get('signal_history'))
        _style_fig(fig)
        p = os.path.join(tmp_dir, "08_ar_signal.png")
        _save_fig(fig, p)
        charts.append((p, "Low Absorption Ratio Signal"))

    # --- Seasonal Divergence ---
    if len(srd.get('spread', pd.Series(dtype=float)).dropna()) > 0:
        fig = chart_seasonal_divergence(srd['spread'], spy_close, year_filter=None)
        _add_signal_vlines(fig, srd.get('signal_history'))
        _style_fig(fig)
        p = os.path.join(tmp_dir, "09_seasonal_divergence.png")
        _save_fig(fig, p)
        charts.append((p, "Seasonal Rank Spread"))

    # --- Dispersion Signal ---
    if len(disp.get('composite_pctile', pd.Series(dtype=float)).dropna()) > 0:
        fig = chart_dispersion_signal(disp['composite_pctile'], spy_close, year_filter=None)
        _add_signal_vlines(fig, disp.get('signal_history'))
        _style_fig(fig)
        p = os.path.join(tmp_dir, "10_dispersion.png")
        _save_fig(fig, p)
        charts.append((p, "Cross-Sectional Dispersion"))

    # --- Absorption Ratio (raw) ---
    if len(ar.get('ar_series', pd.Series(dtype=float)).dropna()) > 0:
        fig = chart_absorption_ratio(ar['ar_series'])
        _style_fig(fig)
        p = os.path.join(tmp_dir, "11_absorption_ratio.png")
        _save_fig(fig, p)
        charts.append((p, "Absorption Ratio - Raw Time Series"))

    return charts


# ---------------------------------------------------------------------------
# 5. REGIME DEEP DIVE TABLE AS PLOTLY go.Table
# ---------------------------------------------------------------------------

def _generate_regime_table(computed, tmp_dir):
    """Build regime deep dive summary as a Plotly table figure and export PNG."""
    spy_df = computed['spy_df']
    closes = computed['closes']
    frag_df = computed['frag_df']
    h_scores = computed['h_scores']

    if frag_df is None or h_scores is None:
        return None

    cache_key = "weekly"
    try:
        deep_dive = compute_regime_deep_dive(spy_df, closes, frag_df, cache_key, sma_filter=False)
    except Exception as e:
        print(f"  Regime deep dive failed: {e}")
        return None

    if not deep_dive or '63d' not in deep_dive:
        return None

    bucket_data = deep_dive['63d'].get('10d_avg')
    if bucket_data is None:
        return None

    summary = bucket_data.get('summary')
    if summary is None or summary.empty:
        return None

    # Determine current regime from 63d h_score
    current_score = h_scores.get('63d', 50)
    current_regime = None
    for regime in REGIME_ORDER:
        lo, hi = {'Robust': (0, 20), 'Calm': (20, 40), 'Neutral': (40, 60),
                  'Elevated': (60, 80), 'Fragile': (80, 100.01)}[regime]
        if lo <= current_score < hi:
            current_regime = regime
            break

    # Build table data — metrics as rows, regimes as columns
    metrics = list(summary.columns)
    header_vals = ["Metric"] + list(REGIME_ORDER)

    # Header colors: regime-colored text
    header_font_colors = ["#ffffff"]
    header_fill_colors = ["#1a1a2e"]
    for r in REGIME_ORDER:
        header_font_colors.append(REGIME_COLORS.get(r, "#ffffff"))
        if r == current_regime:
            header_fill_colors.append("rgba(255,215,0,0.25)")
        else:
            header_fill_colors.append("#1a1a2e")

    # Build cell values and colors
    metric_col = metrics
    cell_vals = [metric_col]
    cell_colors = [["#1a1a2e"] * len(metrics)]
    cell_font_colors = [["#cccccc"] * len(metrics)]

    for regime in REGIME_ORDER:
        col_vals = []
        col_bg = []
        col_fc = []
        for m in metrics:
            val = summary.loc[regime, m] if regime in summary.index else np.nan
            if pd.isna(val):
                col_vals.append("-")
                col_fc.append("#666666")
            else:
                # Format: return-type metrics get +/- and color
                is_return = any(kw in m for kw in ['Ret', 'Fwd', 'Sharpe'])
                if is_return:
                    col_vals.append(f"{val:+.2f}" if abs(val) < 100 else f"{val:+.0f}")
                    col_fc.append("#00CC00" if val >= 0 else "#CC0000")
                else:
                    col_vals.append(f"{val:.2f}" if abs(val) < 1000 else f"{val:.0f}")
                    col_fc.append("#cccccc")
            # Gold highlight for current regime column
            if regime == current_regime:
                col_bg.append("rgba(255,215,0,0.10)")
            else:
                col_bg.append("#16213e")
        cell_vals.append(col_vals)
        cell_colors.append(col_bg)
        cell_font_colors.append(col_fc)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=header_vals,
            fill_color=header_fill_colors,
            font=dict(color=header_font_colors, size=14, family="monospace"),
            align=["left"] + ["center"] * 5,
            height=35,
            line=dict(color="#333333", width=1),
        ),
        cells=dict(
            values=cell_vals,
            fill_color=cell_colors,
            font=dict(color=cell_font_colors, size=12, family="monospace"),
            align=["left"] + ["center"] * 5,
            height=28,
            line=dict(color="#333333", width=1),
        ),
    )])

    title_text = "Regime Deep Dive - 63d Horizon, 10d Avg"
    if current_regime:
        title_text += f"  |  Current: {current_regime} ({current_score:.0f})"

    fig.update_layout(
        title=dict(text=title_text, font=dict(color="#FFD700", size=18)),
        paper_bgcolor="#1a1a2e",
        font=dict(color="#ffffff"),
        width=CHART_WIDTH,
        height=max(700, len(metrics) * 30 + 100),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    p = os.path.join(tmp_dir, "12_regime_table.png")
    _save_fig(fig, p)
    return p


# ---------------------------------------------------------------------------
# 6. COVER PAGE DIALS
# ---------------------------------------------------------------------------

def generate_dial_image(h_scores, tmp_dir):
    """Generate combined 3-dial gauge image for the cover page."""
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
        indicator = dial_fig.data[0]
        fig.add_trace(indicator, row=1, col=i + 1)

    fig.update_layout(
        height=350,
        width=1200,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#ffffff", size=14),
    )

    path = os.path.join(tmp_dir, "cover_dials.png")
    _save_fig(fig, path)
    return path


# ---------------------------------------------------------------------------
# 7. PDF BUILDER
# ---------------------------------------------------------------------------

class RundownPDF(FPDF):
    """Landscape letter PDF with dark background."""

    def __init__(self):
        super().__init__(orientation='L', unit='mm', format='letter')
        self.set_auto_page_break(auto=False)

    def _dark_page(self):
        """Fill current page with dark background."""
        self.set_fill_color(26, 26, 46)  # #1a1a2e
        self.rect(0, 0, PDF_W, PDF_H, 'F')

    def cover_page(self, computed, dial_path):
        """Build the cover page with title, market context, signal summary, and dials."""
        self.add_page()
        self._dark_page()

        price_ctx = computed['price_ctx']
        h_scores = computed['h_scores']
        h_scores_10d = computed.get('h_scores_10d')
        signals_ordered = computed['signals_ordered']
        date_str = datetime.datetime.now().strftime("%A, %B %d, %Y")

        # Title
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(255, 255, 255)
        self.set_xy(20, 15)
        self.cell(0, 12, "Weekly Market Rundown", ln=True)

        self.set_font("Helvetica", "", 14)
        self.set_text_color(170, 170, 170)
        self.set_xy(20, 30)
        self.cell(0, 8, f"Week of {date_str}", ln=True)

        # Divider
        self.set_draw_color(80, 80, 80)
        self.line(20, 42, PDF_W - 20, 42)

        # Market Context
        y = 48
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(136, 136, 136)
        self.set_xy(20, y)
        self.cell(0, 6, "MARKET CONTEXT", ln=True)
        y += 9

        p = price_ctx
        ret_12m = f"{p['ret_12m']:+.1%}" if p['ret_12m'] is not None else "N/A"
        ytd_ret = f"{computed['ytd_return']:+.1%}" if computed.get('ytd_return') is not None else "N/A"
        ext_pctile = f"{computed['ext_200d_pctile']:.0f}th" if computed.get('ext_200d_pctile') is not None else "N/A"
        d5 = p.get('days_since_5pct')
        d10 = p.get('days_since_10pct')
        days_5_str = f"{d5}d" if d5 is not None else "N/A"
        days_10_str = f"{d10}d" if d10 is not None else "N/A"
        sznl = computed.get('seasonal_rank')
        sznl_str = f"{sznl:.0f}" if sznl is not None else "N/A"

        # Row 1
        row1 = [
            ("12M Ret", ret_12m, 30),
            ("YTD", ytd_ret, 24),
            ("vs 200d %ile", ext_pctile, 32),
            ("Seasonal Rank", sznl_str, 34),
        ]
        # Row 2
        row2 = [
            ("Since 5% Corr", days_5_str, 34),
            ("Since 10% Corr", days_10_str, 34),
        ]

        self.set_font("Helvetica", "", 13)
        x = 20
        for label, val, label_w in row1:
            self.set_text_color(136, 136, 136)
            self.set_xy(x, y)
            self.cell(0, 6, f"{label}:")
            self.set_text_color(255, 255, 255)
            self.set_xy(x + label_w, y)
            self.cell(0, 6, _latin1_safe(val))
            x += label_w + 28

        y += 8
        x = 20
        for label, val, label_w in row2:
            self.set_text_color(136, 136, 136)
            self.set_xy(x, y)
            self.cell(0, 6, f"{label}:")
            self.set_text_color(255, 255, 255)
            self.set_xy(x + label_w, y)
            self.cell(0, 6, _latin1_safe(val))
            x += label_w + 28

        # Signal Summary
        y += 14
        active = [name for name, sig in signals_ordered.items() if sig.get('on')]
        active_count = len(active)
        total = len(signals_ordered)

        self.set_font("Helvetica", "B", 11)
        self.set_text_color(136, 136, 136)
        self.set_xy(20, y)
        self.cell(0, 6, f"SIGNAL BOARD ({active_count}/{total} ACTIVE)", ln=True)
        y += 9

        self.set_font("Helvetica", "", 11)
        for name, sig in signals_ordered.items():
            is_on = sig.get('on', False)
            badge = "ON" if is_on else "OFF"
            detail = sig.get('detail', '') or sig.get('summary', '')
            if not is_on:
                detail = sig.get('summary', '')

            self.set_xy(22, y)
            if is_on:
                self.set_text_color(204, 0, 0)
            else:
                self.set_text_color(0, 204, 0)
            self.cell(12, 5, badge)

            self.set_text_color(255, 255, 255)
            self.set_xy(36, y)
            self.cell(50, 5, name)

            self.set_text_color(170, 170, 170)
            self.set_xy(90, y)
            max_detail = 100
            if len(detail) > max_detail:
                detail = detail[:max_detail] + "..."
            self.cell(0, 5, _latin1_safe(detail))
            y += 6

        # Fragility Dials
        if dial_path and os.path.exists(dial_path):
            y += 5
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(136, 136, 136)
            self.set_xy(20, y)
            self.cell(0, 6, "FRAGILITY DIALS (5d / 21d / 63d)", ln=True)
            y += 8
            dial_w = 160
            dial_x = (PDF_W - dial_w) / 2
            self.image(dial_path, x=dial_x, y=y, w=dial_w)
            y += dial_w * 0.29 + 2  # approx image height ratio

        # 10d trailing average below dials
        if h_scores_10d:
            self.set_font("Helvetica", "", 12)
            parts = []
            for key, label in [('5d', '5d'), ('21d', '21d'), ('63d', '63d')]:
                score = h_scores_10d.get(key, 0)
                regime = _assign_regime_bucket(score)
                parts.append(f"{label}: {score:.0f} ({regime})")
            avg_line = "10d Trailing Avg:   " + "   |   ".join(parts)
            self.set_text_color(170, 170, 170)
            self.set_xy(20, y)
            self.cell(0, 6, avg_line, align='C')

        # Footer
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.set_xy(20, PDF_H - 12)
        self.cell(0, 5, f"Generated by weekly_market_rundown.py | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def chart_page(self, img_path, title):
        """Add a full-page chart with title."""
        self.add_page()
        self._dark_page()

        # Title bar
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(255, 215, 0)
        self.set_xy(15, 8)
        self.cell(0, 10, title)

        # Chart image — fill most of the page
        img_y = 22
        img_h = PDF_H - img_y - 10
        img_w = PDF_W - 20
        self.image(img_path, x=10, y=img_y, w=img_w, h=img_h)


def build_pdf(computed, charts, dial_path, tmp_dir):
    """Assemble the multi-page landscape PDF."""
    pdf = RundownPDF()

    # Page 1: Cover
    pdf.cover_page(computed, dial_path)

    # Pages 2+: Charts
    for img_path, title in charts:
        pdf.chart_page(img_path, title)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    pdf_path = os.path.join(tmp_dir, f"Weekly_Market_Rundown_{date_str}.pdf")
    pdf.output(pdf_path)
    print(f"  PDF saved: {pdf_path}")
    return pdf_path


# ---------------------------------------------------------------------------
# 8. EMAIL
# ---------------------------------------------------------------------------

def send_email(pdf_path, computed):
    """Send the PDF as an email attachment."""
    sender_email = os.environ.get("EMAIL_USER")
    sender_password = os.environ.get("EMAIL_PASS")
    receiver_email = "mckinleyslade@gmail.com"

    if not sender_email or not sender_password:
        print("  EMAIL_USER / EMAIL_PASS not set — skipping send")
        return False

    h_scores = computed.get('h_scores', {}) or {}
    s5 = h_scores.get('5d', 0)
    s21 = h_scores.get('21d', 0)
    s63 = h_scores.get('63d', 0)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    subject = f"Weekly Rundown - {date_str} | Fragility: {s5:.0f}/{s21:.0f}/{s63:.0f}"

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    # Attach PDF
    with open(pdf_path, "rb") as f:
        part = MIMEBase("application", "pdf")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            "attachment",
            filename=os.path.basename(pdf_path),
        )
        msg.attach(part)

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
# 9. MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("WEEKLY MARKET RUNDOWN")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Download data
    print("\n[1/5] Downloading data...")
    spy_df, closes, sp500_closes = download_data()

    # 2. Compute signals
    print("[2/5] Computing signals...")
    computed = compute_all_signals(spy_df, closes, sp500_closes)

    # 3. Generate chart images
    print("[3/5] Generating charts...")
    tmp_dir = tempfile.mkdtemp()
    dial_path = generate_dial_image(computed['h_scores'], tmp_dir)
    charts = generate_charts(computed, tmp_dir)
    print(f"  {len(charts)} chart pages generated")

    # 4. Build PDF
    print("[4/5] Building PDF...")
    pdf_path = build_pdf(computed, charts, dial_path, tmp_dir)

    # 5. Send email
    print("[5/5] Sending email...")
    send_email(pdf_path, computed)

    print("\nDone.")


if __name__ == "__main__":
    main()
