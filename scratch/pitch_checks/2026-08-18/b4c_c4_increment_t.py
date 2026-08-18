"""C4 close-out: the surviving 3-condition cell's t is against ZERO, not
against its own control. Compute the Welch t of the INCREMENT (D vs A, D vs
the local control) -- that is the number that decides whether the divergence
adds anything to "VIX has been quiet".
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["SPY", "^VIX"]).dropna(subset=["SPY", "^VIX"])
spy, vix = px["SPY"], px["^VIX"]
r_spy, r_vix = spy.pct_change(), vix.pct_change()
rk = pct_rank(vix, 21)

A = rk <= 25
B = A & (r_vix >= 0.05)
C = A & (r_spy > -0.0075)
D = A & (r_vix >= 0.05) & (r_spy > -0.0075)
JOINT = (r_vix >= 0.05) & (r_spy > -0.0075)


def ep(m, gap):
    return declusters(px.index[m.reindex(px.index, fill_value=False).values], gap, px.index)


def welch(x, y):
    x, y = np.asarray(x), np.asarray(y)
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    return (x.mean() - y.mean()) / se


print("Welch t of the INCREMENT (episode level, long SPY, lag=1)")
for h in (3, 5, 10):
    ret = vehicle_ret(px, [("SPY", 1.0)], h, lag=1)
    v = ret.notna()
    gap = max(h, 5)
    d = ret.loc[ep(D & v, gap)].values
    a = ret.loc[ep(A & v, gap)].values
    b = ret.loc[ep(B & v, gap)].values
    c = ret.loc[ep(C & v, gap)].values
    loc = ret.loc[local_control(px.index[v.values], px.index[(D & v).values])].values
    j = ret.loc[ep(JOINT & v, gap)].values
    allv = ret[v].values
    print(f"\nh={h}  D N={len(d)} mean {100*d.mean():+.3f}%  (t vs 0 = "
          f"{d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.2f})")
    print(f"   D vs A  (rank<=25 alone, N={len(a)}): diff {100*(d.mean()-a.mean()):+.3f}pp  "
          f"welch t {welch(d, a):+.2f}")
    print(f"   D vs B  (A+pop, N={len(b)}):           diff {100*(d.mean()-b.mean()):+.3f}pp  "
          f"welch t {welch(d, b):+.2f}")
    print(f"   D vs C  (A+no-damage, N={len(c)}):     diff {100*(d.mean()-c.mean()):+.3f}pp  "
          f"welch t {welch(d, c):+.2f}")
    print(f"   D vs local +/-126td (N={len(loc)}):    diff {100*(d.mean()-loc.mean()):+.3f}pp  "
          f"welch t {welch(d, loc):+.2f}")
    print(f"   PITCHED 2-leg JOINT cell (N={len(j)}): {100*j.mean():+.3f}%  vs all days "
          f"{100*allv.mean():+.3f}%  -> edge {100*(j.mean()-allv.mean()):+.3f}pp  "
          f"welch t {welch(j, allv):+.2f}")
