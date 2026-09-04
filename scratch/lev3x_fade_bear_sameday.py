"""Same-day clustering profile for the proposed 3x Bear ETF Overbot Fade
(bear-eq universe, 80 threshold, consec-1). How many of the 51 trades share
a signal date, and what does a same-day cluster look like as a single bet?
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
    load_seasonal_map, load_atr_seasonal_map, precompute_all_indicators,
    generate_candidates_fast, process_signals_fast,
)

START = pd.Timestamp("2003-01-01")
BEAR_EQ = ['SPXS', 'SQQQ', 'SDOW', 'TZA',
           'SOXS', 'FAZ', 'TECS', 'LABD', 'ERY', 'DRV', 'WEBS', 'YANG', 'EDZ']

v = copy.deepcopy({s["name"]: s for s in STRATEGY_BOOK}["3x ETF Overbot Fade"])
v["name"] = "bear 80+consec1"
v["universe_tickers"] = BEAR_EQ
for i in range(4):
    v["settings"]["perf_filters"][i]["thresh"] = 80.0
v["settings"]["perf_filters"][3]["consecutive"] = 1

md = data_provider.get_history(list(LEV3X_ALL) + ["SPY", "^VIX"], start="2000-01-01")
vd = md["^VIX"].copy()
if isinstance(vd.columns, pd.MultiIndex):
    vd.columns = vd.columns.get_level_values(0)
vd.columns = [c.capitalize() for c in vd.columns]

sznl_map = load_seasonal_map()
atr_sznl_map = load_atr_seasonal_map()
processed = precompute_all_indicators(md, [v], sznl_map, vd["Close"], atr_sznl_map)

cands, sd = generate_candidates_fast(processed, [v], sznl_map, START)
tr = process_signals_fast(cands, sd, processed, [v], ACCOUNT_VALUE, flat_sizing=True)
tr["Date"] = pd.to_datetime(tr["Date"])
tr["R"] = tr["PnL"] / tr["Risk $"].replace(0, np.nan)
tr = tr.sort_values("Date")

print(f"filled trades: {len(tr)} on {tr['Date'].nunique()} distinct signal dates")

# signals per day
per_day = tr.groupby("Date").agg(n=("R", "size"), sumR=("R", "sum"),
                                 avgR=("R", "mean"),
                                 tks=("Ticker", lambda x: ",".join(x)))
dist = per_day["n"].value_counts().sort_index()
print("\nsignals-per-day distribution:")
for k, c in dist.items():
    print(f"  {k} signal(s): {c} days  ({c*k} trades)")

multi = per_day[per_day["n"] >= 2]
solo = per_day[per_day["n"] == 1]
print(f"\ntrades on multi-signal days: {int(multi['n'].sum())} of {len(tr)} "
      f"({multi['n'].sum()/len(tr):.0%})")

def day_stats(g, label):
    if len(g) == 0:
        print(f"{label:<34} N=0")
        return
    r = g["avgR"]
    print(f"{label:<34} days={len(g):>3}  day-avgR mean={r.mean():+.3f}  "
          f"med={r.median():+.3f}  win={(r > 0).mean():5.1%}  "
          f"worst daySumR={g['sumR'].min():+.2f}")

print("\nDAY-LEVEL (each day = one observation, avgR across its signals):")
day_stats(solo, "  solo-signal days")
day_stats(multi, "  multi-signal days (2+)")

t = per_day["avgR"].mean() / (per_day["avgR"].std(ddof=1) / np.sqrt(len(per_day)))
print(f"\nday-level t-stat (all {len(per_day)} days): {t:+.2f}")

print("\nall multi-signal days:")
for d, row in multi.sort_values("n", ascending=False).iterrows():
    print(f"  {d.date()}  n={int(row['n'])}  sumR={row['sumR']:+.2f}  "
          f"avgR={row['avgR']:+.2f}  [{row['tks']}]")

# aggregate same-day risk at proposed sizing (25 bps nominal, GRM 1.5)
eff_bps = 25 * 1.5
print(f"\nat 25 bps nominal ({eff_bps:.1f} effective) per trade, same-day "
      f"aggregate risk:")
for k in sorted(multi["n"].unique()):
    print(f"  {int(k)}-signal day: {k*eff_bps:.0f} bps effective "
          f"({k*eff_bps/100:.2f}% of NAV at risk on one market direction)")
