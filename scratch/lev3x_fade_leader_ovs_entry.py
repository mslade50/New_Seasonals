"""3x leader-expansion fade — chosen config (252d>95 + gap-up) at the OVS
0.75 ATR entry limit vs 0.5 / 1.0 (2026-07-10).

User direction: proceed with loose(80/c1) + 252d rank > 95 + decisive gap-up
(NextOpen > Close + 0.25 ATR), entry limit matched to OVS's 0.75 ATR for
book consistency if it holds up vs the neighbors. Non-bull-equity universe
is the target implementation; bull-eq shown only as the control.
"""
import copy
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data_provider
from strategy_config import STRATEGY_BOOK, ACCOUNT_VALUE, LEV3X_ALL
from pages.strat_backtester import (
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
    process_signals_fast,
)

START = pd.Timestamp("2003-01-01")

BULL_EQ = ['SPXL', 'TQQQ', 'UDOW', 'TNA', 'MIDU',
           'SOXL', 'FAS', 'TECL', 'LABU', 'CURE', 'ERX', 'DPST',
           'DRN', 'NAIL', 'RETL', 'WEBL', 'DFEN', 'YINN', 'BRZU', 'EDC', 'MEXX']
BEAR_EQ = ['SPXS', 'SQQQ', 'SDOW', 'TZA',
           'SOXS', 'FAZ', 'TECS', 'LABD', 'ERY', 'DRV', 'WEBS', 'YANG', 'EDZ']
BONDS = ['TMF', 'TMV']
CMDTY_BULL = ['NUGT', 'JNUG', 'GUSH']
CMDTY_BEAR = ['DUST', 'JDST', 'DRIP']

def klass(tk):
    if tk in BULL_EQ: return 'bull-eq'
    if tk in BEAR_EQ: return 'bear-eq'
    if tk in BONDS: return 'bond'
    if tk in CMDTY_BULL: return 'cmdty-bull'
    if tk in CMDTY_BEAR: return 'cmdty-bear'
    return '?'

def stats_line(r, label):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        print(f"{label:<52} N=   0")
        return
    pf = r[r > 0].sum() / max(1e-9, -(r[r < 0].sum()))
    print(f"{label:<52} N={len(r):>4}  win={(r > 0).mean():6.1%}  "
          f"avgR={r.mean():+.3f}  medR={r.median():+.3f}  "
          f"totR={r.sum():+8.1f}  PF={pf:5.2f}  minR={r.min():+.2f}")

base = copy.deepcopy({s["name"]: s for s in STRATEGY_BOOK}["3x ETF Overbot Fade"])

def variant(name, entry_type):
    v = copy.deepcopy(base)
    v["name"] = name
    v["universe_tickers"] = list(LEV3X_ALL)
    pf = v["settings"]["perf_filters"]
    for i in range(4):
        pf[i]["thresh"] = 80.0
    pf[3]["consecutive"] = 1
    flip = copy.deepcopy(pf[5])
    flip["logic"] = ">"
    flip["thresh"] = 95.0
    v["settings"]["perf_filters"] = pf[:4] + [flip]
    v["settings"]["use_t1_open_filter"] = True
    v["settings"]["t1_open_filters"] = [
        {"reference": "Close", "atr_offset": 0.25, "logic": ">"}]
    v["settings"]["entry_type"] = entry_type
    return v

variants = [
    variant("lim0.5", "Limit (Open +/- 0.5 ATR)"),
    variant("lim0.75 (OVS)", "Limit (Open +/- 0.75 ATR)"),
    variant("lim1.0", "Limit (Open +/- 1 ATR)"),
]

md = data_provider.get_history(list(LEV3X_ALL) + ["SPY", "^VIX"], start="2000-01-01")
vix_df = md.get("^VIX")
vix_series = None
if vix_df is not None and not vix_df.empty:
    vd = vix_df.copy()
    if isinstance(vd.columns, pd.MultiIndex):
        vd.columns = vd.columns.get_level_values(0)
    vd.columns = [c.capitalize() for c in vd.columns]
    vix_series = vd["Close"]

sznl_map = load_seasonal_map()
atr_sznl_map = load_atr_seasonal_map()
processed = precompute_all_indicators(md, variants, sznl_map, vix_series, atr_sznl_map)

ERAS = [("2010-2014", "2010-01-01", "2014-12-31"),
        ("2015-2019", "2015-01-01", "2019-12-31"),
        ("2020-2022", "2020-01-01", "2022-12-31"),
        ("2023-now ", "2023-01-01", "2099-01-01")]

results = {}
for v in variants:
    cands, sd = generate_candidates_fast(processed, [v], sznl_map, START)
    tr = process_signals_fast(cands, sd, processed, [v], ACCOUNT_VALUE, flat_sizing=True)
    if tr.empty:
        continue
    tr["Date"] = pd.to_datetime(tr["Date"])
    tr["R"] = tr["PnL"] / tr["Risk $"].replace(0, np.nan)
    tr["class"] = tr["Ticker"].map(klass)
    results[v["name"]] = tr

for name in ["lim0.5", "lim0.75 (OVS)", "lim1.0"]:
    tr = results.get(name)
    if tr is None:
        continue
    nb = tr[tr["class"] != "bull-eq"]
    print(f"\n{'=' * 84}\nB>95 + gap0.25 — {name}\n{'=' * 84}")
    stats_line(nb["R"], "  non-bull-eq (target universe)")
    stats_line(tr[tr["class"] == "bear-eq"]["R"], "    bear-eq")
    stats_line(tr[tr["class"] == "bond"]["R"], "    bond")
    stats_line(tr[tr["class"].isin(["cmdty-bull", "cmdty-bear"])]["R"], "    cmdty")
    stats_line(tr[tr["class"] == "bull-eq"]["R"], "  bull-eq (control, not traded)")
    for elbl, a, b in ERAS:
        g = nb[(nb["Date"] >= a) & (nb["Date"] <= b)]
        stats_line(g["R"], f"      {elbl} non-bull")
    # per-year for robustness eyeball
    print("  per-year (non-bull): ", end="")
    for y, g in nb.groupby(nb["Date"].dt.year):
        print(f"{y}:{g['R'].sum():+.1f}({len(g)})", end="  ")
    print()

# episode clustering: same-week trades share one episode
print(f"\n{'=' * 84}\nEPISODE VIEW — non-bull trades grouped by ISO week\n{'=' * 84}")
for name in ["lim0.5", "lim0.75 (OVS)", "lim1.0"]:
    tr = results.get(name)
    if tr is None:
        continue
    nb = tr[tr["class"] != "bull-eq"].copy()
    nb["wk"] = nb["Date"].dt.strftime("%G-W%V")
    ep = nb.groupby("wk")["R"].agg(["sum", "count", "mean"])
    t = ep["mean"].mean() / (ep["mean"].std() / np.sqrt(len(ep))) if len(ep) > 2 else np.nan
    print(f"\n  {name}: {len(nb)} trades in {len(ep)} week-episodes; "
          f"episode avgR(mean-of-means)={ep['mean'].mean():+.3f}, "
          f"episode t={t:.2f}, worst week {ep['sum'].min():+.1f}R, "
          f"best week {ep['sum'].max():+.1f}R")

# full trade list for the OVS-entry cell
tr = results.get("lim0.75 (OVS)")
if tr is not None:
    nb = tr[tr["class"] != "bull-eq"]
    cols = ["Date", "Entry Date", "Exit Date", "Ticker", "class", "Price",
            "Exit Price", "R"]
    cols = [c for c in cols if c in nb.columns]
    d = nb.sort_values("Date")[cols].copy()
    for c in ("Date", "Entry Date", "Exit Date"):
        d[c] = pd.to_datetime(d[c]).dt.strftime("%Y-%m-%d")
    print(f"\nlim0.75 (OVS) — non-bull trade list ({len(d)}):")
    with pd.option_context("display.width", 160, "display.max_rows", 400):
        print(d.to_string(index=False))
