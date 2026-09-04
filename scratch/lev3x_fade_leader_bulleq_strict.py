"""Leader-expansion fade, bull-eq rescue attempt: does the PARENT's stricter
filter stack (2/5/10d > 85, 21d > 85 x3 consecutive) fix the bull-eq class?
(2026-07-10)

Prior finding: bull-eq loses pervasively at the loose 80/consec1 thresholds
under every entry. Test the strict stack at rank>65 / >95, base entry and
gap0.25+lim0.75 entry. Bull-eq universe only.
"""
import copy
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data_provider
from strategy_config import STRATEGY_BOOK, ACCOUNT_VALUE
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

def stats_line(r, label):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        print(f"{label:<44} N=   0")
        return
    pf = r[r > 0].sum() / max(1e-9, -(r[r < 0].sum()))
    print(f"{label:<44} N={r.size:>4}  win={(r > 0).mean():6.1%}  "
          f"avgR={r.mean():+.3f}  totR={r.sum():+8.1f}  PF={pf:5.2f}")

base = copy.deepcopy({s["name"]: s for s in STRATEGY_BOOK}["3x ETF Overbot Fade"])

def variant(name, rank252_thresh, strict, gap=False):
    v = copy.deepcopy(base)
    v["name"] = name
    v["universe_tickers"] = list(BULL_EQ)
    pf = v["settings"]["perf_filters"]
    # parent defaults are already 85s + consec3 on the 21d; loosen if not strict
    if not strict:
        for i in range(4):
            pf[i]["thresh"] = 80.0
        pf[3]["consecutive"] = 1
    flip = copy.deepcopy(pf[5])
    flip["logic"] = ">"
    flip["thresh"] = rank252_thresh
    v["settings"]["perf_filters"] = pf[:4] + [flip]
    if gap:
        v["settings"]["use_t1_open_filter"] = True
        v["settings"]["t1_open_filters"] = [
            {"reference": "Close", "atr_offset": 0.25, "logic": ">"}]
        v["settings"]["entry_type"] = "Limit (Open +/- 0.75 ATR)"
    return v

variants = [
    variant("strict85/c3 base>65", 65.0, True),
    variant("strict85/c3 base>95", 95.0, True),
    variant("strict85/c3 gap075>65", 65.0, True, gap=True),
    variant("strict85/c3 gap075>95", 95.0, True, gap=True),
    variant("loose80/c1 gap075>95 (ref)", 95.0, False, gap=True),
]

md = data_provider.get_history(list(BULL_EQ) + ["SPY", "^VIX"], start="2000-01-01")
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

print("BULL-EQ universe only — strict parent filters vs loose, leader-required")
for v in variants:
    cands, sd = generate_candidates_fast(processed, [v], sznl_map, START)
    tr = process_signals_fast(cands, sd, processed, [v], ACCOUNT_VALUE, flat_sizing=True)
    if tr.empty:
        print(f"\n{v['name']}: NO TRADES")
        continue
    tr["Date"] = pd.to_datetime(tr["Date"])
    tr["R"] = tr["PnL"] / tr["Risk $"].replace(0, np.nan)
    tr = tr.sort_values("Date")
    print()
    stats_line(tr["R"], v["name"])
    tr["wk"] = tr["Date"].dt.strftime("%G-W%V")
    ep = tr.groupby("wk")["R"].sum()
    print(f"    episodes: {len(ep)} ({(ep > 0).sum()} pos / {(ep <= 0).sum()} neg), "
          f"median {ep.median():+.2f}R")
    print("    per-year: ", end="")
    for y, g in tr.groupby(tr["Date"].dt.year):
        print(f"{y}:{g['R'].sum():+.1f}({len(g)})", end="  ")
    print()
