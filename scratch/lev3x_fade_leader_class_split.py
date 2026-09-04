"""Leader-expansion fade (gap0.25 + lim0.75) — per-class / per-ticker
effectiveness (2026-07-10).

Chosen cell (252d>95) has tiny class Ns, so also run the 252d>65 version of
the SAME entry mechanics for a larger-sample read on where the edge lives.
Classes split by direction within asset too (TMF vs TMV, cmdty bull vs bear).
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

def klass(tk):
    if tk in BULL_EQ: return 'bull-eq'
    if tk in BEAR_EQ: return 'bear-eq'
    if tk == 'TMF': return 'bond-bull'
    if tk == 'TMV': return 'bond-bear'
    if tk in ('NUGT', 'JNUG', 'GUSH'): return 'cmdty-bull'
    if tk in ('DUST', 'JDST', 'DRIP'): return 'cmdty-bear'
    return '?'

def stats_line(r, label):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        print(f"{label:<40} N=   0")
        return
    pf = r[r > 0].sum() / max(1e-9, -(r[r < 0].sum()))
    print(f"{label:<40} N={len(r):>4}  win={(r > 0).mean():6.1%}  "
          f"avgR={r.mean():+.3f}  medR={r.median():+.3f}  "
          f"totR={r.sum():+8.1f}  PF={pf:5.2f}  minR={r.min():+.2f}")

base = copy.deepcopy({s["name"]: s for s in STRATEGY_BOOK}["3x ETF Overbot Fade"])

def variant(name, rank252_thresh):
    v = copy.deepcopy(base)
    v["name"] = name
    v["universe_tickers"] = list(LEV3X_ALL)
    pf = v["settings"]["perf_filters"]
    for i in range(4):
        pf[i]["thresh"] = 80.0
    pf[3]["consecutive"] = 1
    flip = copy.deepcopy(pf[5])
    flip["logic"] = ">"
    flip["thresh"] = rank252_thresh
    v["settings"]["perf_filters"] = pf[:4] + [flip]
    v["settings"]["use_t1_open_filter"] = True
    v["settings"]["t1_open_filters"] = [
        {"reference": "Close", "atr_offset": 0.25, "logic": ">"}]
    v["settings"]["entry_type"] = "Limit (Open +/- 0.75 ATR)"
    return v

variants = [variant("rank>95 (chosen)", 95.0), variant("rank>65 (wide)", 65.0)]

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

CLASSES = ['bull-eq', 'bear-eq', 'bond-bull', 'bond-bear', 'cmdty-bull', 'cmdty-bear']

for v in variants:
    cands, sd = generate_candidates_fast(processed, [v], sznl_map, START)
    tr = process_signals_fast(cands, sd, processed, [v], ACCOUNT_VALUE, flat_sizing=True)
    if tr.empty:
        continue
    tr["Date"] = pd.to_datetime(tr["Date"])
    tr["R"] = tr["PnL"] / tr["Risk $"].replace(0, np.nan)
    tr["class"] = tr["Ticker"].map(klass)

    print(f"\n{'=' * 84}\ngap0.25 + lim0.75 — {v['name']}\n{'=' * 84}")
    for k in CLASSES:
        stats_line(tr.loc[tr["class"] == k, "R"], f"  {k}")
    print("\n  per-ticker (all, sorted by totR):")
    for tk, g in sorted(tr.groupby("Ticker"), key=lambda x: -x[1]["R"].sum()):
        stats_line(g["R"], f"    {tk} [{klass(tk)}]")
    # dates for the small classes so the episodes are inspectable
    for k in ['bond-bull', 'bond-bear', 'cmdty-bull', 'cmdty-bear']:
        g = tr[tr["class"] == k].sort_values("Date")
        if len(g):
            rows = ", ".join(f"{d:%Y-%m-%d} {t} {r:+.2f}R"
                             for d, t, r in zip(g["Date"], g["Ticker"], g["R"]))
            print(f"\n  {k} trades: {rows}")
