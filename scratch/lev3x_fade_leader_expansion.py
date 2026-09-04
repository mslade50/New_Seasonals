"""3x ETF Overbot Fade — EXPANDED signal on 252d LEADERS (2026-07-10).

Question: what do the trades the 126/252d < 65 leader exclusion currently
DROPS look like?  Flip the exclusion into a requirement:
  - keep the short-horizon overbought stack (2/5/10/21d)
  - drop the 126d filter, set 252d rank > 65 (variant A) / > 95 (variant B)

Universe slices requested: ALL 42 lev-3x ETFs, and the non-bull-equity
subset. Perf-rank filters are per-ticker time series and sizing here is
flat, so one engine run on LEV3X_ALL slices exactly by class.

Thresholds: run both the parent config (85s, 21d consec3) and the looser
bear-fade config (80s, consec1) so the bear-eq slice is comparable to the
live 3x Bear ETF Overbot Fade.
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
        print(f"{label:<46} N=   0")
        return
    pf = r[r > 0].sum() / max(1e-9, -(r[r < 0].sum()))
    print(f"{label:<46} N={len(r):>4}  win={(r > 0).mean():6.1%}  "
          f"avgR={r.mean():+.3f}  medR={r.median():+.3f}  "
          f"totR={r.sum():+8.1f}  PF={pf:5.2f}")

base = copy.deepcopy({s["name"]: s for s in STRATEGY_BOOK}["3x ETF Overbot Fade"])

def variant(name, rank252_thresh, thresh=None, consec21=None):
    v = copy.deepcopy(base)
    v["name"] = name
    v["universe_tickers"] = list(LEV3X_ALL)
    pf = v["settings"]["perf_filters"]
    # pf: [0]=2d>85 [1]=5d>85 [2]=10d>85 [3]=21d>85 consec3 [4]=126d<65 [5]=252d<65
    if thresh is not None:
        for i in range(4):
            pf[i]["thresh"] = thresh
    if consec21 is not None:
        pf[3]["consecutive"] = consec21
    # drop the 126d exclusion, flip 252d into a leader REQUIREMENT
    flip = copy.deepcopy(pf[5])
    flip["logic"] = ">"
    flip["thresh"] = rank252_thresh
    v["settings"]["perf_filters"] = pf[:4] + [flip]
    return v

variants = [
    variant("parent(85/c3) 252d>65", 65.0),
    variant("parent(85/c3) 252d>95", 95.0),
    variant("loose(80/c1) 252d>65", 65.0, thresh=80.0, consec21=1),
    variant("loose(80/c1) 252d>95", 95.0, thresh=80.0, consec21=1),
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

ERAS = [("2003-2009", "2003-01-01", "2009-12-31"),
        ("2010-2014", "2010-01-01", "2014-12-31"),
        ("2015-2019", "2015-01-01", "2019-12-31"),
        ("2020-2022", "2020-01-01", "2022-12-31"),
        ("2023-now ", "2023-01-01", "2099-01-01")]

SLICES = [
    ("ALL 42", lambda tr: tr),
    ("non-bull-eq (ex bull-eq, incl cmdty-bull)",
     lambda tr: tr[~tr["class"].isin(["bull-eq"])]),
    ("non-bull strict (bear-eq + bond + cmdty-bear)",
     lambda tr: tr[tr["class"].isin(["bear-eq", "bond", "cmdty-bear"])]),
    ("  bull-eq", lambda tr: tr[tr["class"] == "bull-eq"]),
    ("  bear-eq", lambda tr: tr[tr["class"] == "bear-eq"]),
    ("  bond", lambda tr: tr[tr["class"] == "bond"]),
    ("  cmdty-bull", lambda tr: tr[tr["class"] == "cmdty-bull"]),
    ("  cmdty-bear", lambda tr: tr[tr["class"] == "cmdty-bear"]),
]

results = {}
for v in variants:
    cands, sd = generate_candidates_fast(processed, [v], sznl_map, START)
    tr = process_signals_fast(cands, sd, processed, [v], ACCOUNT_VALUE, flat_sizing=True)
    if tr.empty:
        print(f"\n{'=' * 78}\n{v['name']}: NO TRADES\n{'=' * 78}")
        continue
    tr["Date"] = pd.to_datetime(tr["Date"])
    tr["R"] = tr["PnL"] / tr["Risk $"].replace(0, np.nan)
    tr["class"] = tr["Ticker"].map(klass)
    results[v["name"]] = tr

    print(f"\n{'=' * 78}\n{v['name']}\n{'=' * 78}")
    for lbl, fn in SLICES:
        stats_line(fn(tr)["R"], lbl)
    print("\n  eras (ALL / non-bull-eq):")
    for lbl, a, b in ERAS:
        g = tr[(tr["Date"] >= a) & (tr["Date"] <= b)]
        stats_line(g["R"], f"    {lbl} ALL")
        stats_line(g[g["class"] != "bull-eq"]["R"], f"    {lbl} non-bull")
    print("\n  per-ticker (N>=3):")
    for tk, g in sorted(tr.groupby("Ticker"), key=lambda x: -x[1]["R"].sum()):
        if len(g) >= 3:
            stats_line(g["R"], f"    {tk} [{klass(tk)}]")

def dump_trades(tr, label, max_rows=None):
    cols = ["Date", "Entry Date", "Exit Date", "Exit Type", "Ticker", "class",
            "Price", "Exit Price", "R"]
    cols = [c for c in cols if c in tr.columns]
    d = tr.sort_values("Date")[cols].copy()
    for c in ("Date", "Entry Date", "Exit Date"):
        if c in d.columns:
            d[c] = pd.to_datetime(d[c]).dt.strftime("%Y-%m-%d")
    print(f"\n{label} ({len(d)} trades)")
    with pd.option_context("display.width", 160, "display.max_rows", 500):
        if max_rows and len(d) > max_rows:
            print(d.tail(max_rows).to_string(index=False))
            print(f"  ... showing last {max_rows} of {len(d)}")
        else:
            print(d.to_string(index=False))

print(f"\n{'=' * 78}\nTRADE LISTS\n{'=' * 78}")
for name in ["parent(85/c3) 252d>95", "loose(80/c1) 252d>95"]:
    if name in results:
        dump_trades(results[name], name)
for name in ["parent(85/c3) 252d>65"]:
    if name in results:
        dump_trades(results[name], name, max_rows=40)

# per-year totR table
show = [n for n in results]
years = sorted({d.year for n in show for d in results[n]["Date"]})
print("\nper-year totR (N):")
print("year  " + "".join(f"{n:>26}" for n in show))
for y in years:
    row = f"{y}  "
    for n in show:
        g = results[n][results[n]["Date"].dt.year == y]
        row += f"{g['R'].sum():>+20.1f} ({len(g):>3})"
    print(row)
