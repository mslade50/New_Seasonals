"""3x leader-expansion fade — does a stop loss help? (2026-07-10)

Follow-up to scratch/lev3x_fade_leader_expansion.py. The expanded signal
(252d rank > 65 / > 95 leader REQUIREMENT) showed a tail-carried profile:
negative median at >95, PnL from crash-reversal days, worst trades -4 to
-7R. Question: does use_stop_loss=True fix the left tail or just get run
over at max panic before the reversal?

Grid: two base configs x {no stop, 1.0 ATR day2-armed, 1.0 ATR day1-armed,
1.5 ATR day2, 2.0 ATR day2}. Stop fills use the prod gap-through + slippage
model. NOTE: stop_atr also sets the sizing distance, so R stays
risk-normalized within each variant (same-move dollar PnL differs).
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

def variant(name, rank252_thresh, thresh=None, consec21=None,
            stop_atr=None, day1_stop=False):
    v = copy.deepcopy(base)
    v["name"] = name
    v["universe_tickers"] = list(LEV3X_ALL)
    pf = v["settings"]["perf_filters"]
    if thresh is not None:
        for i in range(4):
            pf[i]["thresh"] = thresh
    if consec21 is not None:
        pf[3]["consecutive"] = consec21
    flip = copy.deepcopy(pf[5])
    flip["logic"] = ">"
    flip["thresh"] = rank252_thresh
    v["settings"]["perf_filters"] = pf[:4] + [flip]
    if stop_atr is not None:
        v["execution"]["use_stop_loss"] = True
        v["execution"]["stop_atr"] = stop_atr
        if day1_stop:
            v["execution"]["stop_active_entry_day"] = True
    return v

GRID = [("nostop", dict()),
        ("stop1.0/d2", dict(stop_atr=1.0)),
        ("stop1.0/d1", dict(stop_atr=1.0, day1_stop=True)),
        ("stop1.5/d2", dict(stop_atr=1.5)),
        ("stop2.0/d2", dict(stop_atr=2.0))]

variants = []
for label, cfg in GRID:
    variants.append(variant(f"A>65 {label}", 65.0, **cfg))
for label, cfg in GRID:
    variants.append(variant(f"B>95L {label}", 95.0, thresh=80.0, consec21=1, **cfg))

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
        print(f"{v['name']}: NO TRADES")
        continue
    tr["Date"] = pd.to_datetime(tr["Date"])
    tr["R"] = tr["PnL"] / tr["Risk $"].replace(0, np.nan)
    tr["class"] = tr["Ticker"].map(klass)
    results[v["name"]] = tr

for fam, fam_label in [("A>65", "A: parent(85/c3) 252d>65"),
                       ("B>95L", "B: loose(80/c1) 252d>95")]:
    print(f"\n{'=' * 84}\n{fam_label}\n{'=' * 84}")
    for label, _ in GRID:
        tr = results.get(f"{fam} {label}")
        if tr is None:
            continue
        nb = tr[tr["class"] != "bull-eq"]
        stops = (tr["Exit Type"] == "Stop").sum() if "Exit Type" in tr else 0
        print(f"\n  -- {label}  (stop exits: {stops}/{len(tr)})")
        stats_line(tr["R"], f"    ALL 42")
        stats_line(nb["R"], f"    non-bull-eq")
        stats_line(tr[tr["class"] == "bear-eq"]["R"], f"      bear-eq")
        stats_line(tr[tr["class"] == "bond"]["R"], f"      bond")
        stats_line(tr[tr["class"] == "bull-eq"]["R"], f"      bull-eq")
        for elbl, a, b in ERAS:
            g = nb[(nb["Date"] >= a) & (nb["Date"] <= b)]
            stats_line(g["R"], f"        {elbl} non-bull")

# matched-trade view: what the 1.0 ATR day-2 stop does to the baseline tails
print(f"\n{'=' * 84}\nMATCHED TRADES — baseline tails under the 1.0 ATR day-2 stop\n{'=' * 84}")
for fam in ["A>65", "B>95L"]:
    b0, b1 = results.get(f"{fam} nostop"), results.get(f"{fam} stop1.0/d2")
    if b0 is None or b1 is None:
        continue
    m = b0.merge(b1[["Ticker", "Date", "R", "Exit Type"]], on=["Ticker", "Date"],
                 suffixes=("_ns", "_st"), how="inner")
    m["dR"] = m["R_st"] - m["R_ns"]
    m = m[m["class"] != "bull-eq"]
    print(f"\n{fam} (non-bull only, matched N={len(m)}):")
    print(f"  stop saves (dR) on baseline losers <= -1R: "
          f"{m.loc[m['R_ns'] <= -1, 'dR'].sum():+.1f}R over {int((m['R_ns'] <= -1).sum())} trades")
    print(f"  stop costs (dR) on baseline winners >= +1R: "
          f"{m.loc[m['R_ns'] >= +1, 'dR'].sum():+.1f}R over {int((m['R_ns'] >= +1).sum())} trades")
    big = m.reindex(m["R_ns"].abs().sort_values(ascending=False).index).head(15)
    cols = ["Date", "Ticker", "class", "R_ns", "R_st", "Exit Type_st"]
    d = big[cols].copy()
    d["Date"] = d["Date"].dt.strftime("%Y-%m-%d")
    print(d.to_string(index=False))
