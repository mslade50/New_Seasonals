"""3x leader-expansion fade — more selective ENTRIES instead of stops (2026-07-10)

Follow-up to lev3x_fade_leader_expansion.py / lev3x_fade_leader_stops.py.
Stops destroyed the edge (adverse excursion > 1 ATR is the normal path).
This tests demanding a BETTER PRICE / BETTER TAPE instead:
  1. wider entry limit: short at T+1 open + 0.75 / 1.0 ATR (vs prod 0.5)
  2. OVS-style decisive gap-up requirement: NextOpen > Close + 0.25 ATR
     (via use_t1_open_filter), with 0.5 and 1.0 ATR limits
No stops anywhere (time exit only, per prior finding).
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
            entry_type=None, gap_up_atr=None):
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
    if entry_type is not None:
        v["settings"]["entry_type"] = entry_type
    if gap_up_atr is not None:
        v["settings"]["use_t1_open_filter"] = True
        v["settings"]["t1_open_filters"] = [
            {"reference": "Close", "atr_offset": gap_up_atr, "logic": ">"}]
    return v

GRID = [
    ("lim0.5 (base)", dict()),
    ("lim0.75", dict(entry_type="Limit (Open +/- 0.75 ATR)")),
    ("lim1.0", dict(entry_type="Limit (Open +/- 1 ATR)")),
    ("gap0.25 + lim0.5", dict(gap_up_atr=0.25)),
    ("gap0.25 + lim1.0", dict(gap_up_atr=0.25,
                              entry_type="Limit (Open +/- 1 ATR)")),
]

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
        results[v["name"]] = pd.DataFrame()
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
        if tr is None or tr.empty:
            print(f"\n  -- {label}: NO TRADES")
            continue
        nb = tr[tr["class"] != "bull-eq"]
        print(f"\n  -- {label}")
        stats_line(tr["R"], f"    ALL 42")
        stats_line(nb["R"], f"    non-bull-eq")
        stats_line(tr[tr["class"] == "bear-eq"]["R"], f"      bear-eq")
        stats_line(tr[tr["class"] == "bond"]["R"], f"      bond")
        stats_line(tr[tr["class"] == "bull-eq"]["R"], f"      bull-eq")
        for elbl, a, b in ERAS:
            g = nb[(nb["Date"] >= a) & (nb["Date"] <= b)]
            stats_line(g["R"], f"        {elbl} non-bull")

# what does each selectivity layer keep/drop vs base? (matched on Ticker+Date)
print(f"\n{'=' * 84}\nVS BASE — retained vs dropped base trades (non-bull only)\n{'=' * 84}")
for fam in ["A>65", "B>95L"]:
    b0 = results.get(f"{fam} lim0.5 (base)")
    if b0 is None or b0.empty:
        continue
    b0nb = b0[b0["class"] != "bull-eq"]
    keys0 = set(zip(b0nb["Ticker"], b0nb["Date"]))
    print(f"\n{fam}: base non-bull N={len(b0nb)}, totR={b0nb['R'].sum():+.1f}")
    for label, _ in GRID[1:]:
        tr = results.get(f"{fam} {label}")
        if tr is None or tr.empty:
            print(f"  {label:<20} no trades")
            continue
        nb = tr[tr["class"] != "bull-eq"]
        keys1 = set(zip(nb["Ticker"], nb["Date"]))
        dropped = b0nb[~b0nb.apply(lambda x: (x["Ticker"], x["Date"]) in keys1, axis=1)]
        stats_line(dropped["R"], f"  {label:<20} DROPPED base trades (base R)")

# trade list for the most selective cells
def dump_trades(tr, label):
    cols = ["Date", "Entry Date", "Exit Date", "Ticker", "class", "Price",
            "Exit Price", "R"]
    cols = [c for c in cols if c in tr.columns]
    d = tr.sort_values("Date")[cols].copy()
    for c in ("Date", "Entry Date", "Exit Date"):
        if c in d.columns:
            d[c] = pd.to_datetime(d[c]).dt.strftime("%Y-%m-%d")
    print(f"\n{label} ({len(d)} trades)")
    with pd.option_context("display.width", 160, "display.max_rows", 400):
        print(d.to_string(index=False))

for name in ["A>65 gap0.25 + lim1.0", "B>95L gap0.25 + lim1.0"]:
    tr = results.get(name)
    if tr is not None and not tr.empty:
        dump_trades(tr[tr["class"] != "bull-eq"], f"{name} — non-bull trades")
