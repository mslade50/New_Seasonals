"""3x ETF Overbot Fade — bull vs bear ETF split + looser-filter sensitivity.

Questions (2026-07-07):
  1. How does performance split between fading overbought BULL (long-equity)
     3x ETFs vs BEAR (inverse) 3x ETFs?  (Fading a bear ETF = long the market;
     fading a bull ETF = short the market. Both harvest 3x vol drag.)
  2. Should the entry filters be LOOSER on the bear/inverse names, i.e. trade
     them more liberally?

Part A: production ledger split by ETF class (sanity anchor).
Part B: engine runs — baseline split by class, then filter-relaxation variants
        on the bear-equity subset (and bull subset for symmetric comparison),
        with per-era breakdown to keep the loosening honest.
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

# Classes per the strategy_config comment blocks
BULL_EQ = ['SPXL', 'TQQQ', 'UDOW', 'TNA', 'MIDU',
           'SOXL', 'FAS', 'TECL', 'LABU', 'CURE', 'ERX', 'DPST',
           'DRN', 'NAIL', 'RETL', 'WEBL', 'DFEN', 'YINN', 'BRZU', 'EDC', 'MEXX']
BEAR_EQ = ['SPXS', 'SQQQ', 'SDOW', 'TZA',
           'SOXS', 'FAZ', 'TECS', 'LABD', 'ERY', 'DRV', 'WEBS', 'YANG', 'EDZ']
BONDS = ['TMF', 'TMV']
CMDTY_BULL = ['NUGT', 'JNUG', 'GUSH']   # commodity-sector equity bull
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

# ---------------- Part A: production ledger ----------------
print("=" * 78)
print("PART A — production ledger, '3x ETF Overbot Fade' by ETF class")
print("=" * 78)
led = pd.read_parquet(os.path.join(ROOT, "data", "backtest_trades_full.parquet"))
lf = led[led["Strategy"] == "3x ETF Overbot Fade"].copy()
lf["Signal Date"] = pd.to_datetime(lf["Signal Date"])
lf["class"] = lf["Ticker"].map(klass)
print(f"trades: {len(lf)}  ({lf['Signal Date'].min().date()} -> "
      f"{lf['Signal Date'].max().date()})")
stats_line(lf["R_Multiple"], "ALL")
for k in ['bull-eq', 'bear-eq', 'bond', 'cmdty-bull', 'cmdty-bear']:
    stats_line(lf.loc[lf["class"] == k, "R_Multiple"], f"  {k}")
print("\nper-ticker (N>=5):")
for tk, g in sorted(lf.groupby("Ticker"), key=lambda x: -x[1]["R_Multiple"].sum()):
    if len(g) >= 5:
        stats_line(g["R_Multiple"], f"  {tk} [{klass(tk)}]")

# ---------------- Part B: engine variants ----------------
base = copy.deepcopy({s["name"]: s for s in STRATEGY_BOOK}["3x ETF Overbot Fade"])

def variant(name, universe, thresh=None, consec21=None, drop_lt=False):
    v = copy.deepcopy(base)
    v["name"] = name
    v["universe_tickers"] = universe
    pf = v["settings"]["perf_filters"]
    # pf: [0]=2d>85 [1]=5d>85 [2]=10d>85 [3]=21d>85 consec3 [4]=126d<65 [5]=252d<65
    if thresh is not None:
        for i in range(4):
            pf[i]["thresh"] = thresh
    if consec21 is not None:
        pf[3]["consecutive"] = consec21
    if drop_lt:
        v["settings"]["perf_filters"] = pf[:4]
    return v

variants = [
    variant("base/all", list(LEV3X_ALL)),
    variant("base/bear-eq", BEAR_EQ),
    variant("base/bull-eq", BULL_EQ),
    # loosening ladder on bear-eq
    variant("bear-eq consec3->1", BEAR_EQ, consec21=1),
    variant("bear-eq 85->80", BEAR_EQ, thresh=80.0),
    variant("bear-eq 85->75", BEAR_EQ, thresh=75.0),
    variant("bear-eq drop 126/252", BEAR_EQ, drop_lt=True),
    variant("bear-eq 80 + consec1", BEAR_EQ, thresh=80.0, consec21=1),
    variant("bear-eq 75 + consec1 + noLT", BEAR_EQ, thresh=75.0, consec21=1, drop_lt=True),
    # same ladder on bull-eq for contrast
    variant("bull-eq consec3->1", BULL_EQ, consec21=1),
    variant("bull-eq 85->80", BULL_EQ, thresh=80.0),
    variant("bull-eq 80 + consec1", BULL_EQ, thresh=80.0, consec21=1),
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

print("\n" + "=" * 78)
print("PART B — engine variants (flat sizing, R per trade; time-exit-only strat)")
print("=" * 78)
results = {}
for v in variants:
    cands, sd = generate_candidates_fast(processed, [v], sznl_map, START)
    tr = process_signals_fast(cands, sd, processed, [v], ACCOUNT_VALUE, flat_sizing=True)
    if tr.empty:
        print(f"\n{v['name']}: no trades")
        continue
    tr["Date"] = pd.to_datetime(tr["Date"])
    tr["R"] = tr["PnL"] / tr["Risk $"].replace(0, np.nan)
    results[v["name"]] = tr
    yrs = (tr["Date"].max() - tr["Date"].min()).days / 365.25
    print()
    stats_line(tr["R"], f"{v['name']}  (~{len(tr)/max(yrs,1e-9):.0f} tr/yr)")
    for lbl, a, b in ERAS:
        g = tr[(tr["Date"] >= a) & (tr["Date"] <= b)]
        stats_line(g["R"], f"    {lbl}")

# incremental view: what do the ADDED trades (vs base/bear-eq) look like?
print("\n" + "=" * 78)
print("MARGINAL trades added by each bear-eq loosening (vs base/bear-eq)")
print("=" * 78)
if "base/bear-eq" in results:
    base_keys = set(zip(results["base/bear-eq"]["Ticker"],
                        results["base/bear-eq"]["Date"]))
    for name in ["bear-eq consec3->1", "bear-eq 85->80", "bear-eq 85->75",
                 "bear-eq drop 126/252", "bear-eq 80 + consec1",
                 "bear-eq 75 + consec1 + noLT"]:
        tr = results.get(name)
        if tr is None:
            continue
        marg = tr[~tr.apply(lambda x: (x["Ticker"], x["Date"]) in base_keys, axis=1)]
        stats_line(marg["R"], f"  {name} — added trades")

# per-year totR for the headline buckets (robustness eyeball)
print("\nper-year totR:")
show = ["base/bull-eq", "base/bear-eq", "bear-eq 80 + consec1",
        "bear-eq 75 + consec1 + noLT"]
years = sorted({d.year for n in show if n in results
                for d in results[n]["Date"]})
hdr = "year  " + "".join(f"{n:>28}" for n in show)
print(hdr)
for y in years:
    row = f"{y}  "
    for n in show:
        tr = results.get(n)
        if tr is None:
            row += f"{'':>28}"
            continue
        g = tr[tr["Date"].dt.year == y]
        row += f"{g['R'].sum():>+22.1f} ({len(g):>3})"
    print(row)
