"""C3 round 1: what a maximal one-day SECTOR-ROTATION print says about SPY.

Two trigger definitions, because the surface map offers two and they are not
the same tape: (i) max-minus-min one-day sector spread across the nine
original SPDRs, (ii) the XLV-minus-XLK gap.  Both are ranked in a trailing
252-session window (lookahead-free) and also as absolute thresholds.

Both directions of SPY are measured from one cell.  The falsification target
is the pair of mechanisms in the brief: rotation as healthy broadening
(bullish) vs rotation as the last stage of a topping index (bearish).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 240)

SECT = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
px = close_panel(SECT + ["SPY", "QQQ"])
r1 = px.pct_change()

spr = (r1[SECT].max(axis=1) - r1[SECT].min(axis=1)).dropna()
vk = (r1["XLV"] - r1["XLK"]).dropna()
today = px.index[-1]

print(f"live max-min sector spread {today.date()} = {100*spr.loc[today]:.3f}pp")
print(f"  full-sample pctile {100*(spr < spr.loc[today]).mean():.2f}")
tr_spr = spr.rolling(252).rank(pct=True) * 100.0
print(f"  trailing-252 rank {tr_spr.loc[today]:.2f}")
tr_vk = vk.rolling(252).rank(pct=True) * 100.0
print(f"live XLV-XLK = {100*vk.loc[today]:.3f}pp, trailing-252 rank {tr_vk.loc[today]:.2f}")

masks = {
    "maxmin trail rank>=99": (tr_spr >= 99.0),
    "maxmin trail rank>=95": (tr_spr >= 95.0),
    "maxmin trail rank>=90": (tr_spr >= 90.0),
    "maxmin abs >= 4.23pp (today)": (spr >= spr.loc[today]),
    "XLV-XLK trail rank>=95": (tr_vk >= 95.0),
    "XLV-XLK abs >= 4.07pp": (vk >= vk.loc[today]),
}
print()
for k, m in masks.items():
    mm = m.reindex(px.index, fill_value=False)
    print(f"  {k:30s} n_days={int(mm.sum()):5d}   FIRES TODAY: {bool(mm.loc[today])}")

BASE = masks["maxmin trail rank>=95"].reindex(px.index, fill_value=False)

for h in (1, 3, 5, 10):
    battery(px, BASE, [("SPY", 1.0)], h,
            "C3 CELL: long SPY after a trailing-95th-pctile max-min sector spread",
            cost_bps=2.0,
            variants={k: v.reindex(px.index, fill_value=False)
                      for k, v in masks.items()},
            event_kinds=("cpi", "fomc"))

print("\n\n########## HORIZON SCAN, BOTH DEFINITIONS ##########")
for lbl in ("maxmin trail rank>=95", "maxmin trail rank>=99",
            "maxmin abs >= 4.23pp (today)", "XLV-XLK abs >= 4.07pp"):
    m = masks[lbl].reindex(px.index, fill_value=False)
    d = px.index[m.values]
    print(f"\n-- {lbl}  (n_days={len(d)}) --")
    show(horizon_scan(px, d, [("SPY", 1.0)], hs=(1, 2, 3, 5, 7, 10)), "SPY long")

print("\n\n########## WHAT KIND OF DAY IS IT?  (reference class) ##########")
sig = px.index[BASE.values]
sma200 = px["SPY"].rolling(200).mean()
above = px["SPY"] > sma200
hi52 = px["SPY"].rolling(252).max()
dist = px["SPY"] / hi52 - 1.0
print(f"SPY same-day ret on trigger days: mean {100*r1['SPY'].loc[sig].mean():+.3f}% "
      f"median {100*r1['SPY'].loc[sig].median():+.3f}%  frac down "
      f"{100*(r1['SPY'].loc[sig] < 0).mean():.1f}%   (all days mean "
      f"{100*r1['SPY'].mean():+.3f}%, frac down {100*(r1['SPY'] < 0).mean():.1f}%)")
print(f"live SPY 1d {100*r1['SPY'].loc[today]:+.3f}%")
print(f"trigger days above SPY 200d {100*above.loc[sig].mean():.1f}% vs base "
      f"{100*above.dropna().mean():.1f}%   (live above200d={bool(above.loc[today])})")
print(f"trigger-day distance from 52w high: mean {100*dist.loc[sig].mean():+.2f}% "
      f"median {100*dist.loc[sig].median():+.2f}%   (all days median "
      f"{100*dist.dropna().median():+.2f}%)   live {100*dist.loc[today]:+.2f}%")

print("\n\n########## GATE ATTRIBUTION: near-a-high SUBSET ##########")
print("today's version is a rotation day with SPY 1.34% off its 52w high and")
print("VIX 15.8.  the pooled trigger set is mostly panic tape, so split it.")
near = (dist > -0.03)
for h in (3, 5, 10):
    ret = fwd_lag(px["SPY"], h, 1)
    valid = ret.dropna().index
    s_all = px.index[BASE.values].intersection(valid)
    s_near = px.index[(BASE & near).fillna(False).values].intersection(valid)
    s_far = px.index[(BASE & ~near).fillna(False).values].intersection(valid)
    rows = [summarize(ret.loc[declusters(s_all, h, valid)].values, "all triggers"),
            summarize(ret.loc[declusters(s_near, h, valid)].values,
                      "SPY within 3% of 52w high  <-- today's class"),
            summarize(ret.loc[declusters(s_far, h, valid)].values, "SPY >3% off high"),
            summarize(ret.loc[valid].values, "CTRL all days")]
    show(rows, f"h={h} near-high gate attribution")

print("\n\n########## PLACEBO: does the ROTATION add anything over the DAY? ##########")
print("match on SPY's own one-day move instead of on the sector spread.")
for h in (3, 5, 10):
    ret = fwd_lag(px["SPY"], h, 1)
    valid = ret.dropna().index
    sig = px.index[BASE.values].intersection(valid)
    epi = declusters(sig, h, valid)
    lo, hi = r1["SPY"].loc[sig].quantile([0.25, 0.75])
    # ignorant rule: same distribution of SPY 1d moves, no rotation condition
    ign = r1["SPY"].between(lo, hi) & ~BASE
    ign_d = px.index[ign.fillna(False).values].intersection(valid)
    show([summarize(ret.loc[epi].values, f"rotation trigger (N={len(epi)})"),
          summarize(ret.loc[ign_d].values,
                    f"IGNORANT: SPY 1d in [{100*lo:.2f},{100*hi:.2f}]% only"),
          summarize(ret.loc[valid].values, "all days")],
         f"h={h} placebo vs an ignorant SPY-move rule")
