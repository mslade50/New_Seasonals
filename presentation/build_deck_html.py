"""Render a self-contained HTML slide deck from deck_spec.json.

Single source of truth: presentation/deck_spec.json. This builder turns it into
presentation/deck.html — one clean, dependency-free file that opens in any browser,
presents full-screen (arrows to navigate, N for notes, F for fullscreen), and prints
to PDF (one 1280x720 slide per page).

Figures are rendered from real numbers (the 12-strategy stat table) as inline SVG/CSS,
keyed off each slide's "figure" id. No external assets, no network.

Usage:
    python presentation/build_deck_html.py [spec.json] [out.html]
    # no args -> deck_spec.json -> deck.html (the full deck)
"""
from __future__ import annotations

import html
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "deck_spec.json"
OUT_PATH = HERE / "deck.html"

# --- The book, with real backtest stats (from strategy_config.py STRATEGY_BOOK) ---
FAM_COLOR = {
    "Dip-buy": "#2dd4bf",
    "Vol-fade": "#f59e0b",
    "Breakout": "#60a5fa",
    "Seasonal": "#a78bfa",
}
STRATS = [
    {"name": "3x ETF Overbot Fade", "fam": "Vol-fade", "dir": "Short", "hold": "2d", "wr": 79.6, "exp": 0.87, "pf": 7.58},
    {"name": "Oversold Low Volume", "fam": "Dip-buy", "dir": "Long", "hold": "10d", "wr": 69.0, "exp": 0.48, "pf": 2.82},
    {"name": "LT Trend ST OS", "fam": "Dip-buy", "dir": "Long", "hold": "1d", "wr": 68.4, "exp": 0.40, "pf": 2.91},
    {"name": "ATR Extended Gap Up", "fam": "Vol-fade", "dir": "Short", "hold": "2d", "wr": 65.2, "exp": 0.80, "pf": 3.25},
    {"name": "Indices Oversold Bounce", "fam": "Dip-buy", "dir": "Long", "hold": "2d", "wr": 64.6, "exp": 0.34, "pf": 1.90},
    {"name": "St OS Sznl", "fam": "Dip-buy", "dir": "Long", "hold": "5d", "wr": 64.5, "exp": 0.45, "pf": 2.17},
    {"name": "Monday Dip", "fam": "Dip-buy", "dir": "Long", "hold": "2d", "wr": 64.1, "exp": 0.40, "pf": 2.17},
    {"name": "SPY QQQ MonFri Reversion", "fam": "Seasonal", "dir": "Long", "hold": "2d", "wr": 62.9, "exp": 0.40, "pf": 2.21},
    {"name": "Weak Close Decent Sznls", "fam": "Seasonal", "dir": "Long", "hold": "2d", "wr": 61.3, "exp": 0.28, "pf": 1.78},
    {"name": "Overbot Vol Spike", "fam": "Vol-fade", "dir": "Short", "hold": "2d", "wr": 58.0, "exp": 0.28, "pf": 1.96},
    {"name": "52wh Breakout", "fam": "Breakout", "dir": "Long", "hold": "63d", "wr": 47.3, "exp": 0.77, "pf": 2.43},
    {"name": "Sector BO", "fam": "Breakout", "dir": "Long", "hold": "63d", "wr": 28.9, "exp": 1.17, "pf": 2.53},
]
FAMILIES = [
    ("Dip-buy reversion", "Buy oversold pullbacks in uptrenders",
     ["Oversold Low Volume", "LT Trend ST OS", "St OS Sznl", "Indices Oversold Bounce", "Monday Dip"]),
    ("Vol-spike fades", "Short overbought / blow-off exhaustion",
     ["Overbot Vol Spike", "3x ETF Overbot Fade", "ATR Extended Gap Up"]),
    ("Breakout continuation", "Ride fresh highs on conviction volume",
     ["52wh Breakout", "Sector BO"]),
    ("Seasonal & calendar tilts", "Concentrate reversion into paying windows",
     ["Weak Close Decent Sznls", "SPY QQQ MonFri Reversion"]),
]
FAM_OF = {"Dip-buy reversion": "Dip-buy", "Vol-spike fades": "Vol-fade",
          "Breakout continuation": "Breakout", "Seasonal & calendar tilts": "Seasonal"}


def esc(s: str) -> str:
    return html.escape(str(s or ""))


# ----------------------------------------------------------------------------
# Figures — each returns an HTML/SVG string placed in the slide's figure column.
# ----------------------------------------------------------------------------

def _equity(n: int, vol: float, drift: float, seed: int) -> list[float]:
    random.seed(seed)
    v, out = 100.0, []
    for _ in range(n):
        v *= 1 + random.gauss(drift, vol) / 100
        out.append(v)
    return out


def _poly(series: list[float], lo: float, hi: float, w: int, h: int, pad: int) -> str:
    n = len(series)
    xs = lambda i: pad + i * (w - 2 * pad) / (n - 1)
    ys = lambda v: h - pad - (v - lo) / (hi - lo) * (h - 2 * pad)
    return " ".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(series))


def fig_equity_curves() -> str:
    w, h, pad = 660, 360, 24
    smooth = _equity(150, 0.55, 0.20, 7)
    jagged = _equity(150, 2.7, 0.09, 11)
    # Share a terminal value: same outcome, wilder path (honest portfolio-math framing).
    jagged = [v * smooth[-1] / jagged[-1] for v in jagged]
    allv = smooth + jagged
    lo, hi = min(allv) * 0.98, max(allv) * 1.02
    p_s = _poly(smooth, lo, hi, w, h, pad)
    p_j = _poly(jagged, lo, hi, w, h, pad)
    return f"""<svg viewBox="0 0 {w} {h}" class="svgfig" role="img" aria-label="equity curves">
  <polyline points="{p_j}" fill="none" stroke="#5b6b82" stroke-width="1.6" opacity="0.85"/>
  <polyline points="{p_s}" fill="none" stroke="#2dd4bf" stroke-width="3.2"/>
  <g font-size="13" fill="#93a4ba">
    <rect x="{w-214}" y="20" width="13" height="13" fill="#2dd4bf"/>
    <text x="{w-196}" y="31">Aggregate book (smooth)</text>
    <rect x="{w-214}" y="42" width="13" height="13" fill="#5b6b82"/>
    <text x="{w-196}" y="53">One strategy (jagged)</text>
  </g>
</svg>
<div class="figcap">Edges that don't move together add up to a steadier curve, so the capital base rarely takes a deep hit.</div>"""


def fig_expectancy_strip() -> str:
    rows = sorted(STRATS, key=lambda s: s["exp"], reverse=True)
    maxe = max(s["exp"] for s in rows)
    bars = []
    for s in rows:
        wpct = s["exp"] / maxe * 100
        c = FAM_COLOR[s["fam"]]
        bars.append(
            f'<div class="ebar-row"><span class="ebar-name">{esc(s["name"])}</span>'
            f'<span class="ebar-track"><span class="ebar-fill" style="width:{wpct:.0f}%;background:{c}"></span></span>'
            f'<span class="ebar-val">+{s["exp"]:.2f}R</span></div>'
        )
    return f'<div class="ebars"><div class="ebar-head">Per-trade expectancy (R), and every strategy is positive</div>{"".join(bars)}</div>'


def fig_strategy_table() -> str:
    head = (
        "<tr><th>Strategy</th><th>Family</th><th>Dir</th><th>Hold</th>"
        "<th>Win&nbsp;%</th><th>Exp&nbsp;(R)</th><th>PF</th></tr>"
    )
    rows = []
    for s in STRATS:
        c = FAM_COLOR[s["fam"]]
        dcls = "dir-short" if s["dir"] == "Short" else "dir-long"
        rows.append(
            f'<tr><td class="t-name"><span class="dot" style="background:{c}"></span>{esc(s["name"])}</td>'
            f'<td style="color:{c}">{esc(s["fam"])}</td>'
            f'<td><span class="{dcls}">{s["dir"]}</span></td>'
            f'<td class="t-num">{esc(s["hold"])}</td>'
            f'<td class="t-num">{s["wr"]:.1f}</td>'
            f'<td class="t-num">+{s["exp"]:.2f}</td>'
            f'<td class="t-num">{s["pf"]:.2f}</td></tr>'
        )
    return f'<table class="stbl">{head}{"".join(rows)}</table>'


def fig_scatter() -> str:
    w, h = 680, 430
    x0, x1, y0, y1 = 64, 656, 372, 36
    wr_lo, wr_hi, e_lo, e_hi = 25.0, 82.0, 0.20, 1.25
    sx = lambda wr: x0 + (wr - wr_lo) / (wr_hi - wr_lo) * (x1 - x0)
    sy = lambda e: y0 - (e - e_lo) / (e_hi - e_lo) * (y0 - y1)
    grid = []
    for wr in (30, 40, 50, 60, 70, 80):
        grid.append(f'<line x1="{sx(wr):.0f}" y1="{y1}" x2="{sx(wr):.0f}" y2="{y0}" stroke="#1c2940"/>'
                    f'<text x="{sx(wr):.0f}" y="{y0+20}" font-size="12" fill="#7e90a8" text-anchor="middle">{wr}%</text>')
    for e in (0.25, 0.50, 0.75, 1.00, 1.25):
        grid.append(f'<line x1="{x0}" y1="{sy(e):.0f}" x2="{x1}" y2="{sy(e):.0f}" stroke="#1c2940"/>'
                    f'<text x="{x0-10}" y="{sy(e)+4:.0f}" font-size="12" fill="#7e90a8" text-anchor="end">{e:.2f}</text>')
    label_these = {"3x ETF Overbot Fade": (-26, -10, "end"), "Sector BO": (8, 18, "middle"),
                   "ATR Extended Gap Up": (0, -16, "middle"), "LT Trend ST OS": (-10, -14, "end"),
                   "Overbot Vol Spike": (8, 16, "start")}
    dots = []
    for s in STRATS:
        r = 7 + s["pf"] * 1.9
        c = FAM_COLOR[s["fam"]]
        cx, cy = sx(s["wr"]), sy(s["exp"])
        dots.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r:.0f}" fill="{c}" fill-opacity="0.30" stroke="{c}" stroke-width="1.6"/>')
        if s["name"] in label_these:
            dx, dy, anc = label_these[s["name"]]
            dots.append(f'<text x="{cx+dx:.0f}" y="{cy+dy:.0f}" font-size="12.5" fill="#cdd9e8" text-anchor="{anc}">{esc(s["name"])}</text>')
    return f"""<svg viewBox="0 0 {w} {h}" class="svgfig" role="img" aria-label="win rate vs expectancy">
  {''.join(grid)}
  <line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#3a4a63"/>
  <line x1="{x0}" y1="{y1}" x2="{x0}" y2="{y0}" stroke="#3a4a63"/>
  {''.join(dots)}
  <text x="{(x0+x1)/2:.0f}" y="{h-6}" font-size="13" fill="#93a4ba" text-anchor="middle">Win rate &#8594;</text>
  <text x="16" y="{(y0+y1)/2:.0f}" font-size="13" fill="#93a4ba" text-anchor="middle" transform="rotate(-90 16 {(y0+y1)/2:.0f})">Expectancy (R) &#8593;</text>
</svg>
<div class="figcap">Bubble size is profit factor. Sector BO wins just 28.9% of the time but earns the most per trade.</div>"""


def fig_families() -> str:
    cards = []
    for name, desc, members in FAMILIES:
        fam = FAM_OF[name]
        c = FAM_COLOR[fam]
        avg = sum(s["exp"] for s in STRATS if s["fam"] == fam) / max(1, sum(1 for s in STRATS if s["fam"] == fam))
        chips = "".join(f'<span class="chip" style="border-color:{c}55">{esc(m)}</span>' for m in members)
        cards.append(
            f'<div class="famcard" style="border-top:3px solid {c}">'
            f'<div class="fam-h" style="color:{c}">{esc(name)}</div>'
            f'<div class="fam-d">{esc(desc)}</div>'
            f'<div class="fam-chips">{chips}</div>'
            f'<div class="fam-stat">avg expectancy <b>+{avg:.2f}R</b> · {len(members)} strategies</div></div>'
        )
    return f'<div class="famgrid">{"".join(cards)}</div>'


def fig_setup_cause() -> str:
    rows = [
        ("Oversold dip on low volume", "Forced, thin selling with no news, so it snaps back"),
        ("Volatility spike and parabolic gap", "One-sided panic flow runs out, the overshoot reverts"),
        ("Overbought 3x leveraged ETF", "Daily rebalancing bleeds it lower on its own"),
        ("52-week-high breakout in a calm market", "Real institutional buying keeps going"),
    ]
    out = []
    for setup, cause in rows:
        out.append(
            f'<div class="sc-row"><div class="sc-setup">{esc(setup)}</div>'
            f'<div class="sc-arrow">&#8594;</div>'
            f'<div class="sc-cause">{esc(cause)}</div></div>'
        )
    return f'<div class="sc-wrap"><div class="sc-head"><span>The setup</span><span>The reason it pays</span></div>{"".join(out)}</div>'


def fig_risk_funnel() -> str:
    bands = [
        ("100%", "Per trade", "8 to 40 bps of account, $600 to $3,000 risk, ATR stop fixed at entry"),
        ("70%", "Per sleeve", "OVS mild-gap path capped at 1% of account, pro-rata"),
        ("42%", "Whole book", "Hard 2.5% daily risk cap, the worst-case day"),
    ]
    out = []
    for wpct, label, desc in bands:
        out.append(
            f'<div class="funnel-band" style="width:{wpct}">'
            f'<div class="fb-label">{esc(label)}</div><div class="fb-desc">{esc(desc)}</div></div>'
        )
    return f'<div class="funnel">{"".join(out)}<div class="funnel-floor">2.5% a day. One bad name or one bad day can\'t end the compounding.</div></div>'


def fig_evidence_cards() -> str:
    cards = [
        ("Earnings blackout", "±10 td", "Drop signals near earnings, where a surprise breaks the reversion"),
        ("Cycle-year tilt", "0.75×", "OVS in midterm years: avgR +0.19 vs +0.49 otherwise"),
        ("Stop-arming", "+33R", "Arm stops the next session instead of at the fill, over 24 years"),
    ]
    out = []
    for title, metric, desc in cards:
        out.append(
            f'<div class="ev-card"><div class="ev-metric">{esc(metric)}</div>'
            f'<div class="ev-title">{esc(title)}</div><div class="ev-desc">{esc(desc)}</div></div>'
        )
    return (f'<div class="ev-wrap">{"".join(out)}</div>'
            f'<div class="ev-foot">Each one checked by holding out years and testing on them, not by re-running the same backtest.</div>')


def fig_timeline() -> str:
    steps = [
        ("4:17 AM", "Caches refresh", "Prices + earnings pulled from R2, updated, written back"),
        ("4:47 AM", "Book scans", "~1,000 names, all 12 strategies, --scope=all"),
        ("~5 AM", "Sheets staged", "Sized candidates land in Order_Staging / Overflow"),
        ("9:28 AM", "Router submits", "Local order_staging.py → IBKR TWS"),
        ("9:30 AM", "Market open", "Stops arm next session"),
    ]
    nodes = []
    for t, label, desc in steps:
        nodes.append(
            f'<div class="tl-step"><div class="tl-dot"></div>'
            f'<div class="tl-time">{esc(t)}</div><div class="tl-label">{esc(label)}</div>'
            f'<div class="tl-desc">{esc(desc)}</div></div>'
        )
    return (f'<div class="timeline"><div class="tl-line"></div>{"".join(nodes)}</div>'
            f'<div class="tl-foot">A second run after the close re-scans the whole book at 6:00 PM ET.</div>')


def _node(label: str, sub: str = "", cls: str = "") -> str:
    s = f'<div class="nd-sub">{esc(sub)}</div>' if sub else ""
    return f'<div class="node {cls}"><div class="nd-l">{esc(label)}</div>{s}</div>'


def _arrow(label: str = "", dashed: bool = False) -> str:
    cls = "arr dashed" if dashed else "arr"
    lab = f'<span class="arr-lab">{esc(label)}</span>' if label else ""
    return f'<div class="{cls}">{lab}<span class="arr-line">&#8594;</span></div>'


def fig_data_layer() -> str:
    r2 = ('<div class="node r2"><div class="nd-l">Cloudflare R2</div>'
          '<div class="parq">master_prices.parquet</div><div class="parq">earnings_calendar.parquet</div></div>')
    return (f'<div class="dgm row">{r2}{_arrow("cache read")}{_node("daily_scan.py", "reads cache first", "accent")}'
            f'{_arrow("")}{_node("Google Sheets", "staged candidates")}</div>'
            f'<div class="dgm row sub-row">{_node("yfinance", "live", "ghost")}{_arrow("fallback only", dashed=True)}'
            f'<div class="incident">On 2026-06-11 a stale live pull zeroed the liquid tier and wiped the staged orders. Reading from the cache fixed it.</div></div>')


def fig_two_tier() -> str:
    proc = ('<div class="node accent proc"><div class="nd-l">Strategy loop</div>'
            '<div class="nd-sub">earnings blackout · ATR stop/target · risk_bps sizing</div></div>')
    tabs = ('<div class="tier-col">'
            f'{_node("Order_Staging", "Liquid · ~190 names", "tab-liquid")}'
            f'{_node("Overflow", "~870 small/mid · 6 strategies", "tab-overflow")}</div>')
    return (f'<div class="dgm row">{_node("~1,000 names", "liquid + overflow")}{_arrow()}{proc}{_arrow()}{tabs}</div>'
            f'<div class="figcap">Each run clears and rewrites only its own tab, so stale rows can\'t be re-staged, even on a day with no signals.</div>')


def fig_morning_flow() -> str:
    flow = (f'<div class="dgm row wrap">{_node("Overnight signal")}{_arrow()}'
            f'{_node("Gap check", "open vs close + 0.25 ATR", "gate")}{_arrow()}'
            f'{_node("Sized order")}{_arrow()}{_node("IBKR TWS")}{_arrow()}{_node("verify_fills.py", "post-close audit")}</div>')
    bar = ('<div class="pathbar">'
           '<div class="pb-seg pb-green">Path 1 · 40 bps<span>clear gap up</span></div>'
           '<div class="pb-seg pb-amber">Path 2 · 8 bps<span>mild gap</span></div>'
           '<div class="pb-seg pb-red">Skip<span>open at or below close</span></div></div>')
    return flow + bar


def fig_risk_split() -> str:
    left = ('<div class="node split-l"><div class="nd-l">Strategy system</div>'
            '<div class="nd-sub">WHAT &amp; WHEN to trade</div></div>')
    right = ('<div class="node split-r"><div class="nd-l">Risk overlay</div>'
             '<div class="nd-sub">is it safe to size up?</div></div>')
    return (f'<div class="dgm row split-top">{left}<div class="wall">no shared<br>imports</div>{right}</div>'
            f'<div class="dgm row split-down"><span class="downarr">&#8595;</span><span class="downarr">&#8595;</span></div>'
            f'<div class="dgm row">{_node("Gross-exposure / hedge decision", "book level, not per-trade", "accent wide-node")}</div>')


def fig_gauge() -> str:
    val = 42
    cx, cy, r = 200, 200, 150
    start, end = math.pi, 0.0
    def arc(a0, a1, color, wgt=26):
        x0, y0 = cx + r * math.cos(a0), cy - r * math.sin(a0)
        x1, y1 = cx + r * math.cos(a1), cy - r * math.sin(a1)
        large = 1 if (a0 - a1) > math.pi else 0
        return f'<path d="M {x0:.1f} {y0:.1f} A {r} {r} 0 {large} 1 {x1:.1f} {y1:.1f}" fill="none" stroke="{color}" stroke-width="{wgt}" stroke-linecap="butt"/>'
    segs = (arc(start, math.pi * 0.66, "#34d399") + arc(math.pi * 0.66, math.pi * 0.33, "#f59e0b")
            + arc(math.pi * 0.33, end, "#f87171"))
    ang = start - (val / 100) * math.pi
    nx, ny = cx + (r - 30) * math.cos(ang), cy - (r - 30) * math.sin(ang)
    needle = f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#e7eef7" stroke-width="4"/><circle cx="{cx}" cy="{cy}" r="8" fill="#e7eef7"/>'
    chips = ('<div class="q-chips">'
             '<span class="qchip clear">Liquidity real? · CLEAR</span>'
             '<span class="qchip watch">Same side? · WATCH</span>'
             '<span class="qchip clear">Correlations? · CLEAR</span></div>')
    return (f'{chips}<svg viewBox="0 0 400 250" class="svgfig gauge" role="img" aria-label="fragility gauge">{segs}{needle}'
            f'<text x="{cx}" y="{cy-34}" font-size="46" fill="#e7eef7" text-anchor="middle" font-weight="700">{val}</text>'
            f'<text x="50" y="238" font-size="14" fill="#34d399">Robust</text>'
            f'<text x="350" y="238" font-size="14" fill="#f87171" text-anchor="end">Fragile</text></svg>')


def fig_hub_spoke() -> str:
    spokes_top = ["Price updater", "Earnings calendar", "Daily scanner"]
    spokes_bot = ["Portfolio report", "Risk report", "Site build"]
    top = "".join(_node(s) for s in spokes_top)
    bot = "".join(_node(s) for s in spokes_bot)
    hub = '<div class="node r2 hub"><div class="nd-l">Cloudflare R2</div><div class="nd-sub">shared parquet caches</div></div>'
    return (f'<div class="hub-grid"><div class="hub-row">{top}</div>'
            f'<div class="hub-row hub-mid">{hub}</div><div class="hub-row">{bot}</div></div>'
            f'<div class="hub-foot">~14 scheduled jobs · 5 outputs · morning triggers beat the cron queue</div>')


def fig_pillars() -> str:
    pillars = ["Edge", "Breadth", "Bounded risk", "OOS discipline", "Automation"]
    bars = "".join(f'<div class="pillar"><div class="pl-bar"></div><div class="pl-name">{esc(p)}</div></div>' for p in pillars)
    return (f'<div class="pillars">{bars}</div>'
            f'<div class="risk-note">Honest caveat: every number here comes from a backtest, and live results will be lower '
            f'(slippage, crowding, changing regimes). The whole framework is the defense against that.</div>')


# Rolling 36-month correlation of book monthly returns (flat $750k, PnL booked
# at exit) vs SPY monthly returns, 2003-02..2026-06. Recompute: scratch corr
# script vs data/backtest_trades_full.parquet + master_prices.parquet.
CORR_STATS = {"full": 0.19, "beta": 0.12, "down_n": 91, "down_avg": 1.12, "down_pos": 73}
ROLL36_START_YEAR = 2006  # first window ends 2006-01; one value per month
ROLL36 = [
    0.14, 0.17, 0.16, 0.22, 0.28, 0.29, 0.29, 0.29, 0.31, 0.30, 0.29, 0.33, 0.32, 0.32, 0.33, 0.36,
    0.36, 0.30, 0.28, 0.29, 0.29, 0.26, 0.26, 0.48, 0.45, 0.45, 0.45, 0.43, 0.50, 0.49, 0.52, 0.34,
    0.14, 0.13, 0.17, 0.21, 0.21, 0.19, 0.21, 0.19, 0.17, 0.15, 0.19, 0.18, 0.18, 0.17, 0.18, 0.19,
    0.20, 0.16, 0.14, 0.12, 0.12, 0.10, 0.10, 0.07, 0.08, 0.08, 0.09, -0.01, -0.02, -0.03, -0.03,
    -0.04, -0.11, -0.11, -0.10, -0.00, 0.12, 0.11, 0.11, 0.02, -0.05, -0.00, -0.03, 0.13, 0.16,
    0.18, 0.16, 0.17, 0.13, 0.13, 0.13, 0.12, 0.11, 0.16, 0.15, 0.17, 0.16, 0.18, 0.15, 0.19, 0.16,
    0.20, 0.18, 0.11, 0.13, 0.13, 0.13, 0.13, 0.11, 0.12, 0.12, 0.08, 0.08, 0.10, 0.11, 0.08, 0.18,
    0.20, 0.23, 0.04, 0.02, 0.01, 0.26, 0.28, 0.28, 0.28, 0.28, 0.30, 0.31, 0.35, 0.36, 0.36, 0.37,
    0.36, 0.40, 0.41, 0.45, 0.46, 0.47, 0.57, 0.55, 0.56, 0.57, 0.58, 0.57, 0.56, 0.55, 0.55, 0.59,
    0.59, 0.59, 0.66, 0.46, 0.39, 0.39, 0.37, 0.37, 0.35, 0.15, 0.12, 0.17, 0.16, 0.12, 0.01, -0.00,
    -0.15, -0.16, -0.15, -0.05, -0.03, -0.03, -0.03, -0.09, -0.12, -0.11, -0.10, -0.10, -0.09,
    -0.12, -0.04, -0.02, 0.02, -0.04, -0.08, -0.10, -0.05, -0.03, -0.04, 0.02, 0.04, 0.06, 0.07,
    0.06, 0.08, 0.08, 0.09, 0.07, 0.08, 0.05, 0.12, 0.14, 0.15, 0.19, 0.17, 0.22, 0.20, 0.22, 0.27,
    0.25, 0.27, 0.27, 0.28, 0.26, 0.23, 0.31, 0.29, 0.30, 0.28, 0.37, 0.44, 0.45, 0.44, 0.43, 0.46,
    0.51, 0.47, 0.48, 0.45, 0.43, 0.43, 0.42, 0.42, 0.39, 0.40, 0.39, 0.34, 0.33, 0.32, 0.32, 0.31,
    0.25, 0.27, 0.24, 0.17, 0.24, 0.23, 0.22, 0.18, 0.22, 0.21, 0.22, 0.18, 0.20,
]


# Monthly cumulative book profit in $M on the flat $750k risk base (PnL booked
# at exit), 2003-01..2026-07, from data/backtest_trades_full.parquet.
BOOK_EQ_START_YEAR = 2003
BOOK_EQ = [
    -0.009, 0.007, 0.016, 0.016, 0.029, 0.047, 0.078, 0.082, 0.085, 0.098, 0.105, 0.106, 0.121,
    0.131, 0.102, 0.104, 0.112, 0.126, 0.125, 0.141, 0.152, 0.166, 0.168, 0.162, 0.146, 0.154,
    0.169, 0.156, 0.170, 0.165, 0.166, 0.185, 0.206, 0.213, 0.208, 0.220, 0.239, 0.231, 0.217,
    0.233, 0.212, 0.207, 0.207, 0.213, 0.218, 0.236, 0.260, 0.262, 0.279, 0.281, 0.299, 0.322,
    0.350, 0.354, 0.370, 0.384, 0.395, 0.413, 0.424, 0.442, 0.394, 0.399, 0.407, 0.415, 0.433,
    0.414, 0.431, 0.452, 0.482, 0.508, 0.517, 0.529, 0.519, 0.519, 0.518, 0.530, 0.530, 0.556,
    0.557, 0.599, 0.602, 0.601, 0.613, 0.633, 0.631, 0.652, 0.649, 0.658, 0.669, 0.674, 0.674,
    0.677, 0.675, 0.692, 0.742, 0.760, 0.763, 0.760, 0.773, 0.779, 0.805, 0.799, 0.804, 0.812,
    0.808, 0.818, 0.842, 0.853, 0.850, 0.841, 0.875, 0.906, 0.865, 0.901, 0.883, 0.889, 0.909,
    0.936, 0.932, 0.932, 0.946, 0.981, 1.027, 1.014, 1.070, 1.084, 1.093, 1.115, 1.110, 1.109,
    1.123, 1.124, 1.166, 1.181, 1.186, 1.220, 1.212, 1.224, 1.224, 1.243, 1.244, 1.229, 1.241,
    1.234, 1.267, 1.313, 1.294, 1.300, 1.337, 1.342, 1.358, 1.298, 1.292, 1.304, 1.299, 1.312,
    1.305, 1.303, 1.358, 1.373, 1.373, 1.381, 1.382, 1.387, 1.409, 1.394, 1.424, 1.448, 1.455,
    1.455, 1.459, 1.460, 1.485, 1.514, 1.508, 1.509, 1.536, 1.574, 1.600, 1.600, 1.620, 1.676,
    1.711, 1.722, 1.712, 1.709, 1.707, 1.722, 1.727, 1.729, 1.744, 1.761, 1.762, 1.773, 1.764,
    1.767, 1.778, 1.818, 1.851, 1.862, 1.870, 1.844, 1.854, 1.882, 1.893, 1.909, 1.923, 1.927,
    1.999, 2.066, 2.106, 2.090, 2.143, 2.167, 2.203, 2.263, 2.314, 2.390, 2.422, 2.475, 2.511,
    2.517, 2.557, 2.584, 2.601, 2.648, 2.667, 2.675, 2.675, 2.694, 2.739, 2.754, 2.750, 2.736,
    2.753, 2.754, 2.745, 2.741, 2.774, 2.792, 2.837, 2.882, 2.891, 2.911, 2.924, 2.954, 2.968,
    2.980, 2.984, 2.999, 3.030, 3.051, 3.065, 3.125, 3.207, 3.184, 3.185, 3.176, 3.199, 3.214,
    3.212, 3.210, 3.236, 3.275, 3.379, 3.414, 3.429, 3.458, 3.474, 3.497, 3.519, 3.515, 3.551,
    3.613, 3.638, 3.682, 3.756, 3.769, 3.803, 3.838, 3.835, 3.827, 3.825,
]


# Weekly cumulative book profit over the trailing 12 months, $k on the same
# flat $750k base, 2025-07-11..2026-07-06 (same recompute source as BOOK_EQ).
BOOK_EQ_12M = [
    9.9, 18.5, 17.0, 17.0, 11.8, 11.5, 21.5, 13.4, 15.2, 18.8, 18.4, 47.6, 48.2, 81.4, 99.4, 98.4,
    111.7, 105.1, 137.9, 135.1, 136.9, 154.2, 146.6, 148.4, 161.3, 180.5, 222.4, 224.3, 228.8,
    254.7, 242.1, 247.0, 266.2, 267.8, 266.1, 275.7, 277.3, 294.5, 317.8, 320.5, 325.0, 321.9,
    341.0, 328.0, 334.8, 332.4, 333.8, 341.4, 332.5, 321.3, 329.5, 324.1, 323.6,
]


def _equity_panel(series: list[float], x0: int, x1: int, y0: int, y1: int,
                  v_lo: float, v_hi: float, yticks: list, ylab, xticks: list,
                  title: str, end_label: str) -> str:
    n = len(series)
    sx = lambda i: x0 + i * (x1 - x0) / (n - 1)
    sy = lambda v: y0 - (v - v_lo) / (v_hi - v_lo) * (y0 - y1)
    out = []
    for v in yticks:
        col = "#3a4a63" if v == 0 else "#1c2940"
        out.append(f'<line x1="{x0}" y1="{sy(v):.0f}" x2="{x1}" y2="{sy(v):.0f}" stroke="{col}"/>'
                   f'<text x="{x0-7}" y="{sy(v)+4:.0f}" font-size="11.5" fill="#7e90a8" text-anchor="end">{ylab(v)}</text>')
    for i, lab in xticks:
        anc = "end" if sx(i) > x1 - 24 else ("start" if sx(i) < x0 + 24 else "middle")
        out.append(f'<line x1="{sx(i):.0f}" y1="{y1}" x2="{sx(i):.0f}" y2="{y0}" stroke="#1c2940"/>'
                   f'<text x="{sx(i):.0f}" y="{y0+18}" font-size="11.5" fill="#7e90a8" text-anchor="{anc}">{lab}</text>')
    line = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(series))
    out.append(f'<polygon points="{x0},{sy(0):.1f} {line} {x1},{sy(0):.1f}" fill="#2dd4bf" fill-opacity="0.08"/>')
    out.append(f'<polyline points="{line}" fill="none" stroke="#2dd4bf" stroke-width="2.4"/>')
    out.append(f'<circle cx="{x1}" cy="{sy(series[-1]):.0f}" r="4" fill="#2dd4bf"/>')
    out.append(f'<text x="{x1-6}" y="{sy(series[-1])-11:.0f}" font-size="12.5" fill="#d6fff8" text-anchor="end" font-weight="700">{end_label}</text>')
    out.append(f'<text x="{(x0+x1)/2:.0f}" y="{y1-10}" font-size="12.5" fill="#93a4ba" text-anchor="middle">{title}</text>')
    return "".join(out)


def fig_book_equity() -> str:
    w, h = 700, 380
    left = _equity_panel(
        BOOK_EQ, 48, 400, 330, 42, -0.1, 4.1,
        [0, 1, 2, 3, 4], lambda v: f"${v}M",
        [((yr - BOOK_EQ_START_YEAR) * 12, str(yr)) for yr in (2006, 2012, 2018, 2024)],
        "Full backtest, 2003 to 2026", f"+${BOOK_EQ[-1]:.1f}M")
    right = _equity_panel(
        BOOK_EQ_12M, 490, 688, 330, 42, -10, 375,
        [0, 100, 200, 300], lambda v: f"${v}k",
        [(0, "Jul '25"), (26, "Jan '26"), (52, "Jul '26")],
        "Last 12 months", f"+${BOOK_EQ_12M[-1]:.0f}k")
    return f"""<svg viewBox="0 0 {w} {h}" class="svgfig" role="img" aria-label="book equity curve, full period and last 12 months">
  {left}
  {right}
  <text x="{w/2:.0f}" y="{h-4}" font-size="13" fill="#93a4ba" text-anchor="middle">Cumulative profit on the fixed $750k risk base, profit booked as trades close</text>
</svg>
<div class="figcap">$3.8M across 23 years with no losing year, and +$324k over the last 12 months. The dip at the end is June 2026, the roughest stretch of the year, shown as it happened.</div>"""


def fig_rolling_corr() -> str:
    w, h = 700, 400
    x0, x1, y0, y1 = 56, 676, 344, 30
    c_lo, c_hi = -0.4, 1.05
    n = len(ROLL36)
    sx = lambda i: x0 + i * (x1 - x0) / (n - 1)
    sy = lambda c: y0 - (c - c_lo) / (c_hi - c_lo) * (y0 - y1)
    grid = []
    for c in (-0.25, 0.0, 0.25, 0.50, 0.75, 1.0):
        col = "#3a4a63" if c == 0 else "#1c2940"
        grid.append(f'<line x1="{x0}" y1="{sy(c):.0f}" x2="{x1}" y2="{sy(c):.0f}" stroke="{col}"/>'
                    f'<text x="{x0-8}" y="{sy(c)+4:.0f}" font-size="12" fill="#7e90a8" text-anchor="end">{c:+.2f}'.replace("+0.00", "0.00") + "</text>")
    for yr in range(2008, 2027, 4):
        i = (yr - ROLL36_START_YEAR) * 12
        if 0 <= i < n:
            grid.append(f'<line x1="{sx(i):.0f}" y1="{y1}" x2="{sx(i):.0f}" y2="{y0}" stroke="#1c2940"/>'
                        f'<text x="{sx(i):.0f}" y="{y0+20}" font-size="12" fill="#7e90a8" text-anchor="middle">{yr}</text>')
    line = " ".join(f"{sx(i):.1f},{sy(c):.1f}" for i, c in enumerate(ROLL36))
    avg = CORR_STATS["full"]
    return f"""<svg viewBox="0 0 {w} {h}" class="svgfig" role="img" aria-label="rolling correlation to SPY">
  {''.join(grid)}
  <line x1="{x0}" y1="{sy(1.0):.0f}" x2="{x1}" y2="{sy(1.0):.0f}" stroke="#f87171" stroke-width="1.4" stroke-dasharray="6 5" opacity="0.8"/>
  <text x="{x1}" y="{sy(1.0)-8:.0f}" font-size="12.5" fill="#f87171" text-anchor="end">holding the market = 1.00</text>
  <line x1="{x0}" y1="{sy(avg):.0f}" x2="{x1}" y2="{sy(avg):.0f}" stroke="#f59e0b" stroke-width="1.4" stroke-dasharray="6 5" opacity="0.9"/>
  <text x="{sx((2020 - ROLL36_START_YEAR) * 12):.0f}" y="{sy(avg)+22:.0f}" font-size="12.5" fill="#f59e0b" text-anchor="middle">whole-period average {avg:.2f}</text>
  <polyline points="{line}" fill="none" stroke="#2dd4bf" stroke-width="2.6"/>
  <text x="{(x0+x1)/2:.0f}" y="{h-4}" font-size="13" fill="#93a4ba" text-anchor="middle">Rolling 3-year correlation of monthly returns, book vs SPY</text>
</svg>
<div class="figcap">It wanders between roughly -0.15 and 0.65 and keeps coming back down. There is no regime where the book quietly turns into a market-tracking fund.</div>"""


# --- Strategy cards (the per-strategy deck) -----------------------------------
# One card per STRATEGY_BOOK entry, mirroring the site's signal cards: entry
# mechanics, bracket levels, risk, special handling, stats. Risk bps are the
# STAGED (GRM-scaled) values, same basis the signal cards show.
CARDS = [
    {"slug": "olv", "name": "Oversold Low Volume", "fam": "Dip-buy", "dir": "Long", "hold": "10 td",
     "entry": "GTC limit at signal close - 0.25 ATR, live T+1 to T+3 only",
     "stop": "1.25 ATR (arms day 2)", "tgt": "2.5 ATR", "texit": "10 trading days",
     "risk": "52.5 bps liquid / 37.5 overflow",
     "spec": ["Sector loss gate: no new entries in a sector where the strategy realized -2R or worse inside 10 days",
              "Ladder on repeat signals while positions are open: 0.85x / 1.00x / 1.15x",
              "Signals within 10 days before earnings trade at 15 bps instead"],
     "wr": "69.0%", "exp": "+0.48R", "pf": "2.82", "grade": ""},
    {"slug": "ltos", "name": "LT Trend ST OS", "fam": "Dip-buy", "dir": "Long", "hold": "1 td",
     "entry": "GTC limit at signal close - 0.25 ATR",
     "stop": "none (1-day clock bounds risk)", "tgt": "2.0 ATR", "texit": "1 trading day",
     "risk": "45 bps",
     "spec": ["Earnings blackout: no entries within 10 trading days of an announcement, either side"],
     "wr": "68.4%", "exp": "+0.40R", "pf": "2.91", "grade": ""},
    {"slug": "stos", "name": "St OS Sznl", "fam": "Dip-buy", "dir": "Long", "hold": "5 td",
     "entry": "GTC limit at signal close - 0.25 ATR",
     "stop": "none (5-day clock bounds risk)", "tgt": "1.5 ATR", "texit": "5 trading days",
     "risk": "60 bps", "spec": [],
     "wr": "64.5%", "exp": "+0.45R", "pf": "2.17", "grade": ""},
    {"slug": "iob", "name": "Indices Oversold Bounce", "fam": "Dip-buy", "dir": "Long", "hold": "2 td",
     "entry": "GTC limit at signal close - 0.25 ATR",
     "stop": "none (2-day clock bounds risk)", "tgt": "2.0 ATR", "texit": "2 trading days",
     "risk": "52.5 bps",
     "spec": ["Signals read off the spot indices (^GSPC / ^NDX); orders staged 1:1 in SPY / QQQ",
              "Fragility throttle: quarter size when the dial reads 50+",
              "Overlap clamp: 20 bps nominal when MonFri Reversion fires the same day on the same ticker"],
     "wr": "64.6%", "exp": "+0.34R", "pf": "1.90", "grade": ""},
    {"slug": "mdip", "name": "Monday Dip", "fam": "Dip-buy", "dir": "Long", "hold": "2 td",
     "entry": "Day limit at T+1 open - 0.25 ATR",
     "stop": "1.0 ATR (arms day 2)", "tgt": "2.0 ATR", "texit": "2 trading days",
     "risk": "45 bps",
     "spec": ["SPY / QQQ carved out to MonFri Reversion so the two never overlap",
              "Fragility throttle: quarter size when the dial reads 50+"],
     "wr": "64.1%", "exp": "+0.40R", "pf": "2.17", "grade": ""},
    {"slug": "ovs", "name": "Overbot Vol Spike", "fam": "Vol-fade", "dir": "Short", "hold": "2 td",
     "entry": "Day limit at T+1 open + 0.75 ATR",
     "stop": "none (2-day clock bounds risk)", "tgt": "2.0 ATR", "texit": "2 trading days",
     "risk": "60 bps decisive gap / 12 bps mild gap",
     "spec": ["Two-path open gate: decisive gap up = full size, mild gap = small (pooled daily cap), no gap = skip",
              "Earnings blackout: 10 trading days either side",
              "Friday entries: exit at the close if 0.25 ATR offside by 15:58",
              "Midterm election years run 0.75x"],
     "wr": "58.0%", "exp": "+0.28R", "pf": "1.96", "grade": ""},
    {"slug": "x3", "name": "3x ETF Overbot Fade", "fam": "Vol-fade", "dir": "Short", "hold": "2 td",
     "entry": "Day limit at T+1 open + 0.5 ATR",
     "stop": "none", "tgt": "none - pure 2-day time exit", "texit": "2 trading days",
     "risk": "60 bps",
     "spec": ["Collects the snap-back plus the 3x daily-reset decay; no volume or range requirement"],
     "wr": "79.6%", "exp": "+0.87R", "pf": "7.58", "grade": ""},
    {"slug": "x3b", "name": "3x Bear ETF Overbot Fade", "fam": "Vol-fade", "dir": "Short", "hold": "2 td",
     "entry": "Day limit at T+1 open + 0.5 ATR",
     "stop": "none", "tgt": "none - pure 2-day time exit", "texit": "2 trading days",
     "risk": "37.5 bps",
     "spec": ["Pilot, split out of the 3x fade 2026-07-07 with looser thresholds",
              "Shorting an overbought bear ETF is a market dip-buy, so it runs the dip-buy fragility throttle (0.25x at 50+)",
              "The leader exclusion is load-bearing: it keeps this from shorting a sustained bear market"],
     "wr": "66.7%", "exp": "+0.66R", "pf": "2.80", "grade": "Pilot"},
    {"slug": "gap", "name": "ATR Extended Gap Up", "fam": "Vol-fade", "dir": "Short", "hold": "2 td",
     "entry": "Day limit at T+1 open + 0.75 ATR",
     "stop": "none (2-day clock bounds risk)", "tgt": "4.0 ATR", "texit": "2 trading days",
     "risk": "60 bps",
     "spec": ["Long-tail risk is real (an extension can keep extending); the 2-day clock is the valve"],
     "wr": "65.2%", "exp": "+0.80R", "pf": "3.25", "grade": ""},
    {"slug": "wch", "name": "52wh Breakout", "fam": "Breakout", "dir": "Long", "hold": "63 td",
     "entry": "GTC limit at signal close - 0.25 ATR, live 10 days",
     "stop": "2.0 ATR (arms day 2)", "tgt": "8.0 ATR", "texit": "63 trading days",
     "risk": "52.5 bps",
     "spec": ["Calm-regime gate: no entries when the fragility dial (10d avg) reads 30+"],
     "wr": "47.3%", "exp": "+0.77R", "pf": "2.43", "grade": ""},
    {"slug": "sbo", "name": "Sector BO", "fam": "Breakout", "dir": "Long", "hold": "63 td",
     "entry": "GTC limit at signal close - 0.5 ATR, live 10 days",
     "stop": "1.0 ATR (arms day 2)", "tgt": "8.0 ATR", "texit": "63 trading days",
     "risk": "37.5 bps", "spec": [],
     "wr": "28.9%", "exp": "+1.17R", "pf": "2.53", "grade": ""},
    {"slug": "wcd", "name": "Weak Close Decent Sznls", "fam": "Seasonal", "dir": "Long", "hold": "2 td",
     "entry": "Day limit at T+1 open - 0.25 ATR",
     "stop": "1.0 ATR (arms day 2)", "tgt": "2.0 ATR", "texit": "2 trading days",
     "risk": "52.5 bps",
     "spec": ["Fragility throttle: quarter size when the dial reads 50+"],
     "wr": "61.3%", "exp": "+0.28R", "pf": "1.78", "grade": ""},
    {"slug": "mf", "name": "SPY QQQ MonFri Reversion", "fam": "Seasonal", "dir": "Long", "hold": "2 td",
     "entry": "Day limit at T+1 open - 0.25 ATR",
     "stop": "1.0 ATR (arms day 2)", "tgt": "2.0 ATR", "texit": "2 trading days",
     "risk": "52.5 bps",
     "spec": ["Mondays and Fridays only",
              "The time exit carries the edge; stop and target roughly cancel",
              "Fragility throttle: quarter size when the dial reads 50+",
              "Overlap clamp: 20 bps nominal when Indices Oversold Bounce fires the same day on the same ticker"],
     "wr": "62.9%", "exp": "+0.40R", "pf": "2.21", "grade": ""},
]


def make_card(c: dict):
    def fig() -> str:
        fam_c = FAM_COLOR[c["fam"]]
        dir_cls = "short" if c["dir"] == "Short" else "long"
        badges = (f'<span class="sbadge {dir_cls}">{c["dir"].upper()}</span>'
                  f'<span class="sbadge fam" style="color:{fam_c};border-color:{fam_c}66">{esc(c["fam"])}</span>'
                  f'<span class="sbadge fam">HOLD {esc(c["hold"])}</span>')
        if c["grade"]:
            badges += f'<span class="sbadge pilot">{esc(c["grade"].upper())}</span>'
        rows = "".join(
            f'<div class="srow"><span>{lab}</span><b>{esc(val)}</b></div>'
            for lab, val in (("Entry", c["entry"]), ("Stop", c["stop"]), ("Target", c["tgt"]),
                             ("Time exit", c["texit"]), ("Risk", c["risk"])))
        spec = ""
        if c["spec"]:
            spec = ('<div class="scard-spec">'
                    + "".join(f'<div class="sspec">{esc(s)}</div>' for s in c["spec"]) + "</div>")
        stats = (f'<div class="scard-stats">'
                 f'<div class="sstat"><b>{esc(c["wr"])}</b><span>win rate</span></div>'
                 f'<div class="sstat"><b>{esc(c["exp"])}</b><span>per trade</span></div>'
                 f'<div class="sstat"><b>{esc(c["pf"])}</b><span>profit factor</span></div></div>')
        return (f'<div class="scard" style="border-top:3px solid {fam_c}">'
                f'<div class="scard-head"><span class="scard-name">{esc(c["name"])}</span>{badges}</div>'
                f'<div class="scard-rows">{rows}</div>{spec}{stats}</div>')
    return fig


FIGURES = {
    "equity_curves": fig_equity_curves,
    "book_equity": fig_book_equity,
    "rolling_corr": fig_rolling_corr,
    "expectancy_strip": fig_expectancy_strip,
    "strategy_table": fig_strategy_table,
    "scatter": fig_scatter,
    "families": fig_families,
    "setup_cause": fig_setup_cause,
    "risk_funnel": fig_risk_funnel,
    "evidence_cards": fig_evidence_cards,
    "timeline": fig_timeline,
    "data_layer": fig_data_layer,
    "two_tier": fig_two_tier,
    "morning_flow": fig_morning_flow,
    "risk_split": fig_risk_split,
    "gauge": fig_gauge,
    "hub_spoke": fig_hub_spoke,
    "pillars": fig_pillars,
}
FIGURES.update({f"card_{c['slug']}": make_card(c) for c in CARDS})


# ----------------------------------------------------------------------------
# Slide assembly
# ----------------------------------------------------------------------------

def render_bullets(bullets: list[str]) -> str:
    if not bullets:
        return ""
    items = "\n".join(f"        <li>{esc(b)}</li>" for b in bullets)
    return f'      <ul class="bullets">\n{items}\n      </ul>'


def render_callout(callout: str) -> str:
    if not callout:
        return ""
    return f'      <div class="callout">{esc(callout)}</div>'


def render_figure(fig_id: str) -> str:
    if not fig_id or fig_id not in FIGURES:
        return ""
    return f'    <div class="figure">{FIGURES[fig_id]()}</div>'


def cover_slide(deck: dict) -> str:
    tag = deck.get("tagline", "")
    tag_div = f'      <div class="cover-tag">{esc(tag)}</div>\n' if tag else ""
    return (
        '<section class="slide cover">\n'
        '    <div class="cover-inner">\n'
        '      <div class="cover-eyebrow">Quantitative Equity Trading System</div>\n'
        f'      <h1>{esc(deck["deck_title"].split(":")[0])}</h1>\n'
        f'      <p class="cover-sub">{esc(deck["subtitle"])}</p>\n'
        f'{tag_div}'
        f'      <p class="cover-arc">{esc(deck.get("narrative_arc", ""))}</p>\n'
        '    </div>\n'
        '  </section>'
    )


def content_slide(s: dict, idx: int, total: int) -> str:
    fig = render_figure(s.get("figure", ""))
    left_bits = [render_bullets(s.get("bullets", [])), render_callout(s.get("callout", ""))]
    left = "\n".join(b for b in left_bits if b)
    grid_cls = "content-grid" if fig else "content-grid no-fig"
    body = [
        f'<section class="slide content" data-section="{esc(s["section"])}">',
        f'    <div class="kicker">{esc(s["section"])}</div>',
        f'    <h2>{esc(s["title"])}</h2>',
    ]
    if s.get("subtitle"):
        body.append(f'    <p class="slide-subtitle">{esc(s["subtitle"])}</p>')
    body.append(f'    <div class="{grid_cls}">')
    body.append(f'      <div class="left">\n{left}\n      </div>')
    if fig:
        body.append(fig)
    body.append("    </div>")
    if s.get("speaker_notes"):
        body.append(f'    <div class="notes"><strong>Notes.</strong> {esc(s["speaker_notes"])}</div>')
    body.append("  </section>")
    return "\n".join(body)


def build(deck: dict) -> str:
    slides = [cover_slide(deck)]
    content = [s for s in deck["slides"] if s.get("section") != "Open"]
    for i, s in enumerate(content):
        slides.append(content_slide(s, i, len(content)))
    return (TEMPLATE
            .replace("__TITLE__", esc(deck["deck_title"]))
            .replace("__SLIDES__", "\n\n  ".join(slides)))


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{
    --bg:#060b14; --slide:#0c1626; --ink:#e7eef7; --soft:#c2d0e2; --muted:#8aa0b8;
    --line:#22324a; --teal:#2dd4bf; --amber:#f59e0b; --blue:#60a5fa; --violet:#a78bfa;
    --red:#f87171; --green:#34d399; --accent:var(--teal); --maxw:1240px;
    --fz:clamp(15px,1.55vw,20px);
  }
  *{box-sizing:border-box;}
  html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
    font-size:var(--fz);line-height:1.5;-webkit-font-smoothing:antialiased;}
  .viewport{min-height:100vh;display:flex;align-items:center;justify-content:center;padding:3vh 0;}
  .slide{position:relative;display:none;flex-direction:column;justify-content:flex-start;
    width:min(95vw,var(--maxw));min-height:84vh;margin:0 auto;background:var(--slide);
    border:1px solid var(--line);border-radius:16px;padding:4.6vh 4vw;
    box-shadow:0 30px 80px rgba(0,0,0,.45);}
  .slide.active{display:flex;}

  h1{font-size:clamp(40px,6vw,76px);line-height:1.02;margin:0 0 .22em;letter-spacing:-.022em;font-weight:800;}
  h2{font-size:clamp(26px,3.4vw,44px);line-height:1.06;margin:.05em 0 .14em;letter-spacing:-.018em;font-weight:760;}
  .kicker{font-size:.66em;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin-bottom:.5em;}
  .slide-subtitle{font-size:1em;color:var(--muted);margin:.1em 0 1.1em;max-width:60ch;}

  /* Cover */
  .cover{justify-content:center;background:
    radial-gradient(1100px 520px at 78% -8%, rgba(45,212,191,.16), transparent 60%),
    linear-gradient(165deg,#0a1422 0%,#0b1828 55%,#0d2230 100%);}
  .cover-inner{max-width:50ch;}
  .cover-eyebrow{font-size:.7em;font-weight:700;letter-spacing:.22em;text-transform:uppercase;color:var(--teal);margin-bottom:1.1em;}
  .cover h1{color:#f4f9ff;}
  .cover-sub{font-size:clamp(18px,2.1vw,27px);color:var(--soft);font-weight:500;margin:.1em 0 1.3em;max-width:40ch;}
  .cover-tag{display:inline-block;font-size:1.02em;font-weight:700;color:#04221f;background:var(--teal);
    padding:.5em 1em;border-radius:999px;margin-bottom:1.5em;}
  .cover-arc{font-size:.92em;color:var(--muted);max-width:64ch;line-height:1.6;}

  /* Content grid */
  .content-grid{display:grid;grid-template-columns:0.84fr 1.16fr;gap:2.6em;align-items:start;flex:1;margin-top:.3em;}
  .content-grid.no-fig{grid-template-columns:1fr;}
  @media (max-width:860px){.content-grid{grid-template-columns:1fr;gap:1.4em;}}
  .left{display:flex;flex-direction:column;gap:1.1em;}

  ul.bullets{list-style:none;margin:.1em 0 0;padding:0;}
  ul.bullets li{position:relative;padding-left:1.35em;margin:.6em 0;color:var(--soft);font-size:.99em;max-width:48ch;}
  ul.bullets li::before{content:"";position:absolute;left:0;top:.55em;width:.58em;height:.58em;border-radius:2px;background:var(--accent);}

  .callout{background:linear-gradient(135deg,rgba(45,212,191,.16),rgba(45,212,191,.05));
    border:1px solid rgba(45,212,191,.4);color:#d6fff8;border-radius:12px;padding:.85em 1em;
    font-weight:700;font-size:1.08em;line-height:1.3;}

  .figure{display:flex;flex-direction:column;justify-content:center;gap:.7em;min-width:0;}
  .svgfig{width:100%;height:auto;display:block;}
  .figcap{font-size:.78em;color:var(--muted);line-height:1.45;max-width:54ch;}

  /* Expectancy bars */
  .ebars{display:flex;flex-direction:column;gap:.34em;}
  .ebar-head{font-size:.8em;color:var(--muted);margin-bottom:.3em;}
  .ebar-row{display:grid;grid-template-columns:13.5em 1fr 3.4em;align-items:center;gap:.6em;font-size:.82em;}
  .ebar-name{color:var(--soft);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .ebar-track{height:.72em;background:#13203200;border-radius:5px;}
  .ebar-fill{display:block;height:100%;border-radius:5px;}
  .ebar-val{color:var(--ink);font-variant-numeric:tabular-nums;text-align:right;}

  /* Strategy table */
  .stbl{width:100%;border-collapse:collapse;font-size:.82em;}
  .stbl th{text-align:right;color:var(--muted);font-weight:600;padding:.3em .5em;border-bottom:1px solid var(--line);font-size:.92em;}
  .stbl th:first-child,.stbl th:nth-child(2){text-align:left;}
  .stbl td{padding:.32em .5em;border-bottom:1px solid #16243800;text-align:right;color:var(--soft);}
  .stbl tr:nth-child(even) td{background:rgba(255,255,255,.018);}
  .t-name{text-align:left!important;color:var(--ink)!important;white-space:nowrap;}
  .t-num{font-variant-numeric:tabular-nums;}
  .dot{display:inline-block;width:.62em;height:.62em;border-radius:50%;margin-right:.5em;vertical-align:baseline;}
  .dir-long{color:var(--teal);} .dir-short{color:var(--amber);}

  /* Families */
  .famgrid{display:grid;grid-template-columns:1fr 1fr;gap:.9em;}
  .famcard{background:#0e1a2b;border:1px solid var(--line);border-radius:12px;padding:.8em .9em;}
  .fam-h{font-weight:700;font-size:.96em;}
  .fam-d{font-size:.78em;color:var(--muted);margin:.2em 0 .5em;}
  .fam-chips{display:flex;flex-wrap:wrap;gap:.3em;}
  .chip{font-size:.68em;color:var(--soft);border:1px solid var(--line);border-radius:999px;padding:.16em .6em;white-space:nowrap;}
  .fam-stat{font-size:.74em;color:var(--muted);margin-top:.55em;}
  .fam-stat b{color:var(--ink);}

  /* Setup -> cause */
  .sc-wrap{display:flex;flex-direction:column;gap:.5em;}
  .sc-head{display:flex;justify-content:space-between;font-size:.72em;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);}
  .sc-row{display:grid;grid-template-columns:1fr auto 1.05fr;align-items:center;gap:.7em;
    background:#0e1a2b;border:1px solid var(--line);border-radius:10px;padding:.6em .8em;}
  .sc-setup{font-weight:650;font-size:.86em;color:var(--ink);}
  .sc-arrow{color:var(--accent);font-size:1.2em;}
  .sc-cause{font-size:.82em;color:var(--soft);}

  /* Risk funnel */
  .funnel{display:flex;flex-direction:column;align-items:center;gap:.5em;}
  .funnel-band{background:linear-gradient(135deg,rgba(96,165,250,.18),rgba(45,212,191,.10));
    border:1px solid rgba(96,165,250,.35);border-radius:10px;padding:.6em .9em;text-align:center;min-width:46%;}
  .fb-label{font-weight:750;font-size:.92em;}
  .fb-desc{font-size:.76em;color:var(--muted);}
  .funnel-floor{margin-top:.4em;font-size:.82em;font-weight:700;color:#04221f;background:var(--teal);border-radius:8px;padding:.45em .8em;text-align:center;}

  /* Strategy cards (per-strategy deck) */
  .scard{background:#0e1a2b;border:1px solid var(--line);border-radius:14px;
    padding:1em 1.15em;display:flex;flex-direction:column;gap:.7em;}
  .scard-head{display:flex;align-items:center;gap:.5em;flex-wrap:wrap;}
  .scard-name{font-weight:800;font-size:1.02em;color:var(--ink);}
  .sbadge{font-size:.62em;font-weight:700;border-radius:999px;padding:.2em .65em;letter-spacing:.04em;
    background:rgba(96,165,250,.10);color:#a9c8f5;border:1px solid rgba(96,165,250,.3);}
  .sbadge.long{background:rgba(45,212,191,.15);color:#7ef0e2;border-color:rgba(45,212,191,.4);}
  .sbadge.short{background:rgba(245,158,11,.15);color:#fcd34d;border-color:rgba(245,158,11,.4);}
  .sbadge.pilot{background:rgba(248,113,113,.14);color:#fca5a5;border-color:rgba(248,113,113,.4);}
  .scard-rows{display:flex;flex-direction:column;gap:.32em;}
  .srow{display:grid;grid-template-columns:6.4em 1fr;gap:.7em;font-size:.8em;align-items:baseline;}
  .srow span{color:var(--muted);}
  .srow b{color:var(--soft);font-weight:600;}
  .scard-spec{border-top:1px solid var(--line);padding-top:.55em;display:flex;flex-direction:column;gap:.32em;}
  .sspec{font-size:.74em;color:var(--muted);padding-left:1em;position:relative;line-height:1.4;}
  .sspec::before{content:"";position:absolute;left:0;top:.48em;width:.45em;height:.45em;border-radius:2px;background:var(--accent);opacity:.7;}
  .scard-stats{display:flex;gap:.6em;border-top:1px solid var(--line);padding-top:.6em;}
  .sstat{flex:1;text-align:center;}
  .sstat b{display:block;font-size:1.5em;font-weight:800;color:var(--teal);line-height:1.05;}
  .sstat span{font-size:.66em;color:var(--muted);}

  /* Evidence cards */
  .ev-wrap{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.7em;}
  .ev-card{background:#0e1a2b;border:1px solid var(--line);border-radius:12px;padding:.8em;text-align:center;}
  .ev-metric{font-size:1.9em;font-weight:800;color:var(--teal);line-height:1;}
  .ev-title{font-weight:700;font-size:.86em;margin:.35em 0 .25em;}
  .ev-desc{font-size:.74em;color:var(--muted);line-height:1.35;}
  .ev-foot{margin-top:.8em;font-size:.8em;color:var(--soft);font-style:italic;text-align:center;}

  /* Timeline */
  .timeline{position:relative;display:flex;justify-content:space-between;gap:.4em;padding-top:1.2em;}
  .tl-line{position:absolute;top:1.55em;left:6%;right:6%;height:2px;background:var(--line);}
  .tl-step{position:relative;flex:1;text-align:center;}
  .tl-dot{width:14px;height:14px;border-radius:50%;background:var(--accent);margin:0 auto .6em;box-shadow:0 0 0 4px rgba(45,212,191,.15);}
  .tl-time{font-weight:750;font-size:.84em;color:var(--ink);}
  .tl-label{font-size:.8em;color:var(--soft);margin-top:.1em;}
  .tl-desc{font-size:.68em;color:var(--muted);margin-top:.25em;line-height:1.3;}
  .tl-foot{margin-top:1em;font-size:.8em;color:var(--muted);text-align:center;}

  /* Box diagrams */
  .dgm{display:flex;align-items:center;gap:.6em;margin:.3em 0;}
  .dgm.row.wrap{flex-wrap:wrap;}
  .node{background:#0e1a2b;border:1px solid var(--line);border-radius:10px;padding:.55em .8em;text-align:center;min-width:0;}
  .nd-l{font-weight:700;font-size:.86em;color:var(--ink);}
  .nd-sub{font-size:.72em;color:var(--muted);margin-top:.15em;}
  .node.accent{border-color:rgba(45,212,191,.5);background:rgba(45,212,191,.08);}
  .node.ghost{opacity:.6;border-style:dashed;}
  .node.r2{border-color:rgba(96,165,250,.5);background:rgba(96,165,250,.08);}
  .parq{font-size:.7em;color:var(--blue);margin-top:.2em;}
  .arr{display:flex;align-items:center;flex-direction:column;color:var(--muted);font-size:1.2em;}
  .arr.dashed .arr-line{opacity:.5;}
  .arr-lab{font-size:.55em;color:var(--muted);margin-bottom:-.2em;}
  .sub-row{margin-top:.6em;}
  .incident{font-size:.74em;color:#fca5a5;background:rgba(248,113,113,.1);border:1px solid rgba(248,113,113,.3);border-radius:8px;padding:.45em .7em;}
  .formula{margin-top:.7em;font-size:.82em;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#bfe9e3;
    background:#0a1422;border:1px solid var(--line);border-radius:8px;padding:.5em .7em;text-align:center;}
  .tier-col{display:flex;flex-direction:column;gap:.4em;}
  .tab-liquid{border-color:rgba(45,212,191,.45);} .tab-overflow{border-color:rgba(245,158,11,.45);}
  .proc{max-width:13em;}

  /* Path bar */
  .pathbar{display:flex;gap:.4em;margin-top:1em;}
  .pb-seg{flex:1;border-radius:8px;padding:.6em;text-align:center;font-weight:700;font-size:.82em;color:#08130f;}
  .pb-seg span{display:block;font-weight:500;font-size:.78em;opacity:.85;margin-top:.15em;}
  .pb-green{background:var(--green);} .pb-amber{background:var(--amber);} .pb-red{background:var(--red);color:#2a0a0a;}

  /* Risk split */
  .split-top{justify-content:center;gap:1em;}
  .split-l{border-color:rgba(96,165,250,.5);flex:1;} .split-r{border-color:rgba(45,212,191,.5);flex:1;}
  .wall{font-size:.66em;color:var(--muted);text-align:center;border-left:2px dashed var(--line);border-right:2px dashed var(--line);padding:0 .7em;line-height:1.2;}
  .split-down{justify-content:space-around;color:var(--muted);font-size:1.3em;}
  .wide-node{flex:1;} .accent.wide-node{width:100%;}

  /* Gauge */
  .gauge{max-width:380px;margin:0 auto;}
  .q-chips{display:flex;flex-wrap:wrap;gap:.4em;justify-content:center;margin-bottom:.3em;}
  .qchip{font-size:.74em;font-weight:650;border-radius:999px;padding:.22em .7em;}
  .qchip.clear{background:rgba(52,211,153,.14);color:#86efac;border:1px solid rgba(52,211,153,.35);}
  .qchip.watch{background:rgba(245,158,11,.14);color:#fcd34d;border:1px solid rgba(245,158,11,.35);}

  /* Hub & spoke */
  .hub-grid{display:flex;flex-direction:column;gap:.7em;}
  .hub-row{display:flex;justify-content:center;gap:.7em;}
  .hub-mid{margin:.1em 0;}
  .hub{min-width:14em;}
  .hub-foot{margin-top:.9em;font-size:.8em;color:var(--muted);text-align:center;}

  /* Pillars */
  .pillars{display:flex;align-items:flex-end;justify-content:space-between;gap:.7em;height:150px;
    border-bottom:2px solid var(--line);padding:0 .5em;}
  .pillar{flex:1;text-align:center;display:flex;flex-direction:column;justify-content:flex-end;height:100%;}
  .pl-bar{background:linear-gradient(180deg,var(--teal),rgba(45,212,191,.25));border-radius:6px 6px 0 0;height:100%;}
  .pillar:nth-child(2) .pl-bar{height:78%;} .pillar:nth-child(3) .pl-bar{height:88%;}
  .pillar:nth-child(4) .pl-bar{height:70%;} .pillar:nth-child(5) .pl-bar{height:94%;}
  .pl-name{font-size:.78em;color:var(--soft);margin-top:.5em;font-weight:600;}
  .risk-note{margin-top:1em;font-size:.8em;color:var(--muted);line-height:1.5;border-left:3px solid var(--amber);padding-left:.8em;}

  /* Notes */
  .notes{display:none;margin-top:1.4em;padding-top:.9em;border-top:1px solid var(--line);
    font-size:.8em;color:var(--muted);line-height:1.5;}
  body.show-notes .slide.active .notes{display:block;}

  /* Chrome */
  .footer{position:fixed;left:0;right:0;bottom:0;height:30px;display:flex;align-items:center;
    justify-content:space-between;padding:0 16px;font-size:11px;color:#5b6b82;pointer-events:none;z-index:5;}
  .progress{position:fixed;top:0;left:0;height:3px;background:var(--accent);z-index:6;transition:width .25s ease;}
  .hint{position:fixed;bottom:36px;right:14px;font-size:10.5px;color:#5b6b82;z-index:5;
    background:rgba(255,255,255,.04);padding:3px 8px;border-radius:6px;}

  @media print{
    @page{size:1280px 720px;margin:0;}
    :root{--fz:16.5px;}
    body{background:#060b14;}
    .viewport{display:block;padding:0;}
    .footer,.progress,.hint{display:none;}
    .slide{display:flex!important;width:1280px;min-height:720px;height:720px;
      overflow:hidden;page-break-after:always;border-radius:0;margin:0;border:none;box-shadow:none;padding:44px 64px;}
  }
</style>
</head>
<body>
<div class="progress" id="progress"></div>
<div class="viewport"><div class="deck" id="deck">
  __SLIDES__
</div></div>
<div class="footer"><span>__TITLE__</span><span id="counter"></span></div>
<div class="hint">&larr; &rarr; navigate &middot; <b>N</b> notes &middot; <b>F</b> fullscreen</div>
<script>
  const slides = Array.from(document.querySelectorAll('.slide'));
  let i = 0;
  function show(n){
    i = Math.max(0, Math.min(slides.length-1, n));
    slides.forEach((s,k)=>s.classList.toggle('active', k===i));
    document.getElementById('counter').textContent = (i+1)+' / '+slides.length;
    document.getElementById('progress').style.width = ((i+1)/slides.length*100)+'%';
    if (location.hash !== '#'+i) history.replaceState(null,'','#'+i);
  }
  function next(){ show(i+1); } function prev(){ show(i-1); }
  document.addEventListener('keydown', e=>{
    if(e.key==='ArrowRight'||e.key==='PageDown'||e.key===' '){ next(); e.preventDefault(); }
    else if(e.key==='ArrowLeft'||e.key==='PageUp'){ prev(); e.preventDefault(); }
    else if(e.key==='Home'){ show(0); } else if(e.key==='End'){ show(slides.length-1); }
    else if(e.key.toLowerCase()==='n'){ document.body.classList.toggle('show-notes'); }
    else if(e.key.toLowerCase()==='f'){ if(!document.fullscreenElement) document.documentElement.requestFullscreen(); else document.exitFullscreen(); }
  });
  document.addEventListener('click', e=>{ if(e.target.closest('a'))return; next(); });
  const fromHash = parseInt((location.hash||'').replace('#',''),10);
  show(Number.isFinite(fromHash)?fromHash:0);
</script>
</body>
</html>
"""


def main() -> None:
    spec_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SPEC_PATH
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_PATH
    if not spec_path.exists():
        raise SystemExit(f"spec not found at {spec_path}")
    deck = json.loads(spec_path.read_text(encoding="utf-8"))
    out_path.write_text(build(deck), encoding="utf-8")
    n = len([s for s in deck["slides"] if s.get("section") != "Open"])
    print(f"Wrote {out_path}  (cover + {n} content slides)")


if __name__ == "__main__":
    main()
