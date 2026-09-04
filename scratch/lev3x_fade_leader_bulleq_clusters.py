"""Leader-expansion fade — are the bull-eq losses episode-clustered or
pervasive? (2026-07-10)

Dump every bull-eq trade across four signal versions (rank>65 / >95, base
0.5 entry no gap vs gap0.25+0.75 entry) with dates, grouped into ISO-week
episodes and by year/era, to see whether the class fails everywhere or the
totR is a few bad clusters.
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

def stats_line(r, label):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        print(f"{label:<44} N=   0")
        return
    pf = r[r > 0].sum() / max(1e-9, -(r[r < 0].sum()))
    print(f"{label:<44} N={len(r):>4}  win={(r > 0).mean():6.1%}  "
          f"avgR={r.mean():+.3f}  totR={r.sum():+8.1f}  PF={pf:5.2f}")

base = copy.deepcopy({s["name"]: s for s in STRATEGY_BOOK}["3x ETF Overbot Fade"])

def variant(name, rank252_thresh, gap=False):
    v = copy.deepcopy(base)
    v["name"] = name
    v["universe_tickers"] = list(BULL_EQ)
    pf = v["settings"]["perf_filters"]
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
    variant("base>65", 65.0),
    variant("base>95", 95.0),
    variant("gap075>65", 65.0, gap=True),
    variant("gap075>95", 95.0, gap=True),
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

for v in variants:
    cands, sd = generate_candidates_fast(processed, [v], sznl_map, START)
    tr = process_signals_fast(cands, sd, processed, [v], ACCOUNT_VALUE, flat_sizing=True)
    if tr.empty:
        print(f"\n{v['name']}: NO TRADES")
        continue
    tr["Date"] = pd.to_datetime(tr["Date"])
    tr["R"] = tr["PnL"] / tr["Risk $"].replace(0, np.nan)
    tr = tr.sort_values("Date")
    tr["wk"] = tr["Date"].dt.strftime("%G-W%V")

    print(f"\n{'=' * 80}\nBULL-EQ ONLY — {v['name']}\n{'=' * 80}")
    stats_line(tr["R"], "  total")
    # per-year
    print("  per-year: ", end="")
    for y, g in tr.groupby(tr["Date"].dt.year):
        print(f"{y}:{g['R'].sum():+.1f}({len(g)})", end="  ")
    print()
    # episode view
    ep = tr.groupby("wk").agg(R=("R", "sum"), n=("R", "count"),
                              tickers=("Ticker", lambda x: ",".join(x)))
    pos = (ep["R"] > 0).sum()
    print(f"  week-episodes: {len(ep)}  positive: {pos}  negative: {len(ep) - pos}")
    print(f"  episode R: mean {ep['R'].mean():+.2f}  median {ep['R'].median():+.2f}  "
          f"worst {ep['R'].min():+.1f}  best {ep['R'].max():+.1f}")
    print("  all episodes:")
    for wk, row in ep.iterrows():
        print(f"    {wk}  {row['R']:+6.2f}R  n={row['n']}  {row['tickers']}")
