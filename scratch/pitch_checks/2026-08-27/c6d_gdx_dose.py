"""C6 round 2c -- the DOSE RESPONSE on the down-day leg.

The -2% threshold is the whole gate. If the effect is a mechanism it should
grade; if the adjacent bucket (fell 1-2%) is sign-flipped, -2% is a fitted
edge and the cell is a threshold artifact. The registry's GDX teardown already
found the dose response running backwards on the ENTRY's own axis once.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

px = close_panel(["GDX"])
g = px["GDX"]
r21 = g / g.shift(21) - 1.0
rk = rolling_on_valid(r21, lambda x: x.rolling(252).rank(pct=True) * 100.0)
r1 = g.pct_change(fill_method=None)

BUCKETS = [(-99, -4), (-4, -3), (-3, -2), (-2, -1.5), (-1.5, -1),
           (-1, 0), (0, 1), (1, 99)]

for thr in (99.0, 97.0, 95.0):
    print(f"\n\n########## dose response inside rank>={thr:g} thrusts ##########")
    base_m = (rk >= thr).fillna(False)
    print("rank days:", int(base_m.sum()))
    for h in (5, 10):
        ret = fwd_lag(g, h, 1)
        ok = ret.notna()
        rows = []
        for lo, hi in BUCKETS:
            m = base_m & (100 * r1 > lo) & (100 * r1 <= hi) & ok
            d = px.index[m.values]
            if len(d) == 0:
                rows.append({"label": f"1d ({lo},{hi}]", "n": 0}); continue
            e = declusters(d, 10, px.index)
            r = summarize(ret.loc[e].values, f"1d ({lo},{hi}]")
            r["n_days"] = len(d)
            rows.append(r)
        show(rows, f"h={h}, episodes")

# the exact arithmetic behind the <=-1 vs <=-2 discrepancy
print("\n\n########## the (-2,-1] slice, named ##########")
m99 = (rk >= 99).fillna(False)
for h in (5, 10):
    ret = fwd_lag(g, h, 1)
    a = px.index[(m99 & (r1 <= -0.02) & ret.notna()).values]
    b = px.index[(m99 & (r1 <= -0.01) & (r1 > -0.02) & ret.notna()).values]
    ea, eb = declusters(a, 10, px.index), declusters(b, 10, px.index)
    print(f"\nh={h}: 1d<=-2%  N={len(ea)} mean {100*ret.loc[ea].mean():+.3f}%")
    print(f"h={h}: 1d in (-2,-1]  N={len(eb)} mean {100*ret.loc[eb].mean():+.3f}%")
    for d in eb:
        print(f"    {d.date()}  1d {100*r1.loc[d]:+.2f}%  fwd{h} {100*ret.loc[d]:+.2f}%")
