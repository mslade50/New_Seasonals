"""Gold's mid-August seasonal against gold's stretched entry.

Engine: `E:seasonal_doy|GC=F` h5 all years 20-5 up, +0.71%, sign p 0.002 - the cleanest seasonal
number on the board. Also `P4:z10_extreme|GC=F|stretched up` fired tonight (z10 2.18, 21d rank 92)
and that cell is flat at h1 and -0.20% at h5.

So the two cells that both describe Monday's gold disagree. Which one wins when they coincide?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, fwd_ret, summarize, sign_test, era_split, cluster_note, zscore, declusters  # noqa

px = close_panel(["GC=F", "SI=F", "DX-Y.NYB"])
idx = px.index
g = px["GC=F"].dropna()

# the engine's z10 definition, matching build_pitch_state._metrics_for
r1 = g.pct_change()
z10 = (g / g.shift(10) - 1) / (r1.rolling(21).std() * np.sqrt(10))
print(f"gold close {g.iloc[-1]:.1f}  z10 {z10.iloc[-1]:+.2f}  "
      f"21d rank {100*(g.pct_change(21).iloc[-252:] < g.pct_change(21).iloc[-1]).mean():.0f}")

# seasonal anchors: session before the Aug-17 analogue, one per year
anchors = []
for y in range(2000, 2026):
    yr = g.index[g.index.year == y]
    if len(yr) < 20:
        continue
    pick = min(yr, key=lambda d: abs((d - pd.Timestamp(f"{y}-08-17")).days))
    loc = g.index.get_loc(pick)
    if loc > 0:
        anchors.append(g.index[loc - 1])
anchors = pd.DatetimeIndex(anchors)

print("\ngold, one read per year at the mid-August anchor")
for h in (1, 5, 10, 21):
    v = fwd_ret(g, h).reindex(anchors).dropna()
    up = int((v > 0).sum())
    print(f"  h{h:<3d} n {len(v):2d}  {up}-{len(v)-up}  mean {100*v.mean():+.2f}%  med {100*v.median():+.2f}%"
          f"  signp {sign_test(up, len(v)):.4f}")
v5 = fwd_ret(g, 5).reindex(anchors).dropna()
print("  h5 era:", [f"{e['label']} n {e['n']} mean {e['mean_pct']:+.2f}% up {e['hit']:.1f}%"
                    for e in era_split(v5.index, v5.values)])
print("  h5 cluster:", cluster_note(v5.index, v5.values, k=2))
print("  h5 by year:", ", ".join(f"{d.year}:{100*x:+.1f}" for d, x in zip(v5.index, v5.values)))

# unconditional control for the same instrument over the same span
base5 = fwd_ret(g, 5).dropna()
print(f"  control, every session: n {len(base5)} mean {100*base5.mean():+.3f}% "
      f"up {100*(base5>0).mean():.1f}%")

# the stretched cell, and the intersection
print("\ngold entering stretched (z10 >= 2), declustered 5 sessions")
trig = z10.index[(z10 >= 2).fillna(False)]
trig = declusters(trig, 5, g.index)
print(f"  {len(trig)} declustered episodes, first {trig[0].date()} last {trig[-1].date()}")
for h in (1, 5, 10, 21):
    v = fwd_ret(g, h).reindex(trig).dropna()
    s = summarize(v.values, f"h{h}")
    up = int((v > 0).sum())
    print(f"  h{h:<3d} n {s['n']:3d}  {up}-{s['n']-up}  mean {s['mean_pct']:+.2f}%  med {s['median_pct']:+.2f}%"
          f"  t {s['t']:+.2f}  signp {sign_test(up, s['n']):.4f}")

print("\nboth at once: stretched entry inside the seasonal month (Aug 1 to Sep 5)")
both = trig[(trig.month == 8) | ((trig.month == 9) & (trig.day <= 5))]
for h in (1, 5, 10, 21):
    v = fwd_ret(g, h).reindex(both).dropna()
    up = int((v > 0).sum())
    print(f"  h{h:<3d} n {len(v):2d}  {up}-{len(v)-up}  mean {100*v.mean():+.2f}%  med {100*v.median():+.2f}%"
          f"  signp {sign_test(up, len(v)):.4f}")
print("  dates:", ", ".join(str(d.date()) for d in both))

print("\nsame month, stretched vs not, h5")
aug = g.index[(g.index.month == 8) & (g.index.day >= 1)]
st = aug.intersection(z10.index[(z10 >= 1.5).fillna(False)])
ns = aug.difference(st)
for name, dts in [("August, z10 >= 1.5", st), ("August, everything else", ns)]:
    v = fwd_ret(g, 5).reindex(dts).dropna()
    up = int((v > 0).sum())
    print(f"  {name:26s} n {len(v):4d}  {up}-{len(v)-up}  mean {100*v.mean():+.2f}%  "
          f"med {100*v.median():+.2f}%")
