"""C4 round 2b -- the tight rung's two remaining kill routes.

a6 left the tight rung (TLT<=0.5, IEF<=1.0, LQD<=1.0 off 52w low) alive:
17 episodes / 5 years, +0.354pp excess, 82.4% hit, sign p 0.0101, ex-2022
+0.339% at 90%, local-control welch t +2.15, and TLT<=0.5 ALONE is -0.120% at
50% so the joint condition is load-bearing rather than decoration.

Two things can still kill it:
 (a) HORIZON FRAGILITY. The loose rung was positive only at h=1. If the tight
     rung is the same shape, the 'edge' is one bar wide and the rung was chosen
     off a ladder after the fact.
 (b) FRESHNESS. The registry's declustering rule cuts both ways: if today is
     inside an episode that already started, today is not a fresh trigger and
     the 17-episode statistic does not describe it.
Plus the overlap question: C4-tight and C1 are the SAME trade today (long TLT,
MOC tonight, h=1). That is size, not diversification.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["TLT", "IEF", "LQD"]).dropna()
idx = px.index
N = len(idx)
off = {t: ((px[t] / px[t].rolling(252).min()) - 1.0) * 100.0
       for t in ["TLT", "IEF", "LQD"]}
tight = np.ones(N, bool)
for t, k in [("TLT", 0.5), ("IEF", 1.0), ("LQD", 1.0)]:
    tight &= (off[t] <= k).values
tight &= ~np.isnan(off["TLT"].values)
trig = idx[tight]

print("=" * 100)
print("(a) HORIZON SCAN on the TIGHT rung")
print("=" * 100)
show(horizon_scan(px, trig, [("TLT", 1.0)], hs=(1, 2, 3, 5, 10, 21),
                  lag=1, min_gap=10), "tight rung, long TLT")

print("\n" + "=" * 100)
print("(b) FRESHNESS: is TODAY a fresh trigger or mid-episode?")
print("=" * 100)
print(f"  most recent trigger days: "
      f"{[str(d.date()) for d in trig[-12:]]}")
epi = declusters(trig, 10, idx)
print(f"  declustered episodes (min gap 10td): {len(epi)}")
print(f"  most recent episode start: {epi[-1].date()}")
last = trig[-1]
pos = pd.Series(range(N), index=idx)
gap = pos[last] - pos[epi[-1]]
print(f"  today's bar {idx[-1].date()} is {gap} td into the episode that")
print(f"  began {epi[-1].date()} -> a FRESH-trigger rule would NOT fire today.")
run = 1
for i in range(len(trig) - 1, 0, -1):
    if pos[trig[i]] - pos[trig[i - 1]] == 1:
        run += 1
    else:
        break
print(f"  consecutive trigger sessions ending today: {run}")

print("\n  Does day-in-episode matter? (episode-first vs later days)")
r1 = vehicle_ret(px, [("TLT", 1.0)], 1, 1)
base = r1.dropna()
first = set(epi)
fv = r1.loc[[d for d in trig if d in first]].dropna().values
lv = r1.loc[[d for d in trig if d not in first]].dropna().values
for lbl, s in [("episode-FIRST days", fv), ("LATER days in episode", lv)]:
    w = int((s > 0).sum())
    print(f"    {lbl:24s} N={len(s):3d} {100*s.mean():+.4f}% "
          f"excess {100*(s.mean()-base.mean()):+.4f}pp hit {100*w/len(s):.1f}% "
          f"signp {sign_test(w, len(s), float((base>0).mean())):.4f}")
print("  If the later days are flat, today (mid-episode) inherits the flat")
print("  number, not the +0.354pp headline.")

print("\n" + "=" * 100)
print("(c) OVERLAP: C4-tight and C1 are the same order today")
print("=" * 100)
ev = load_events()
sp = lambda k: sorted({int(idx.searchsorted(x, "left"))
                       for x in ev[ev.event == k]["date"]
                       if 0 <= int(idx.searchsorted(x, "left")) < N})
ppi_l = [p for p in sp("ppi") if 3 <= p < N and not np.isnan(r1.values[p - 2])]
cpi_all = set(sp("cpi"))
live = [p for p in ppi_l if (p - 1) in cpi_all]
both = [p for p in live if tight[p - 2]]
print(f"  C1 live-cell observations: {len(live)}")
print(f"  of those, the IG complex was at the TIGHT floor at the anchor: "
      f"{len(both)}")
if both:
    print(f"    dates: {', '.join(str(idx[p].date()) for p in both)}")
    vv = np.array([px['TLT'].values[p] / px['TLT'].values[p - 1] - 1.0
                   for p in both])
    print(f"    those observations: {100*vv.mean():+.4f}% "
          f"hit {100*(vv>0).mean():.0f}%")
print("\n  Both candidates resolve to ONE order tonight: long TLT, MOC, exit")
print("  MOC on the print. Shipping them as two ideas is doubling one bet.")
