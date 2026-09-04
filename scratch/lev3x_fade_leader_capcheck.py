"""Leader gap fade — when would a 2.5% daily aggregate risk cap bind? (2026-07-10)

Staging happens pre-market, so the cap operates on that day's STAGED rows
(all signals), while the gap gate + limit resolve at/after the open. Count
per-day signals two ways:
  - staged: candidates WITHOUT the gap filter (what order_staging sees)
  - gap-passed: candidates WITH the gap filter (rows still live at the open)
Then compute worst-day aggregate risk at candidate sizings vs the cap.
"""
import copy
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data_provider
from strategy_config import STRATEGY_BOOK, LEV3X_ALL
from pages.strat_backtester import (
    load_seasonal_map,
    load_atr_seasonal_map,
    precompute_all_indicators,
    generate_candidates_fast,
)

START = pd.Timestamp("2003-01-01")

BULL_EQ = ['SPXL', 'TQQQ', 'UDOW', 'TNA', 'MIDU',
           'SOXL', 'FAS', 'TECL', 'LABU', 'CURE', 'ERX', 'DPST',
           'DRN', 'NAIL', 'RETL', 'WEBL', 'DFEN', 'YINN', 'BRZU', 'EDC', 'MEXX']
UNIVERSE = [t for t in LEV3X_ALL if t not in BULL_EQ]

base = copy.deepcopy({s["name"]: s for s in STRATEGY_BOOK}["3x ETF Overbot Fade"])

def make(name, gap):
    v = copy.deepcopy(base)
    v["name"] = name
    v["universe_tickers"] = UNIVERSE
    pf = v["settings"]["perf_filters"]
    for i in range(4):
        pf[i]["thresh"] = 80.0
    pf[3]["consecutive"] = 1
    flip = copy.deepcopy(pf[5])
    flip["logic"] = ">"
    flip["thresh"] = 95.0
    v["settings"]["perf_filters"] = pf[:4] + [flip]
    if gap:
        v["settings"]["use_t1_open_filter"] = True
        v["settings"]["t1_open_filters"] = [
            {"reference": "Close", "atr_offset": 0.25, "logic": ">"}]
    v["settings"]["entry_type"] = "Limit (Open +/- 0.75 ATR)"
    return v

variants = [make("staged (no gap gate)", False), make("gap-passed", True)]

md = data_provider.get_history(UNIVERSE + ["SPY", "^VIX"], start="2000-01-01")
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
    if not cands:
        print(f"{v['name']}: no candidates")
        continue
    cd = pd.DataFrame(cands)
    date_col = "date" if "date" in cd.columns else cd.columns[0]
    tick_col = "ticker" if "ticker" in cd.columns else cd.columns[1]
    per_day = cd.groupby(date_col)[tick_col].count().sort_values(ascending=False)
    print(f"\n{v['name']}: {len(cd)} signal-days total, {per_day.index.nunique()} distinct dates")
    print(f"  days with >=2 signals: {(per_day >= 2).sum()}, >=5: {(per_day >= 5).sum()}, >=8: {(per_day >= 8).sum()}")
    print("  busiest days:")
    for d, n in per_day.head(12).items():
        tks = ",".join(sorted(cd.loc[cd[date_col] == d, tick_col]))
        print(f"    {pd.Timestamp(d):%Y-%m-%d}  n={n:>2}  {tks}")
    # cap arithmetic at candidate sizings (nominal x GRM 1.5 = effective)
    for nom in (15, 20, 25):
        eff = nom * 1.5
        worst = per_day.max() * eff
        bind_days = (per_day * eff > 250).sum()
        print(f"  @ {nom} bps nominal ({eff:.1f} eff): worst day {worst:.0f} bps eff; "
              f"days over a 250 bps cap: {bind_days}")
