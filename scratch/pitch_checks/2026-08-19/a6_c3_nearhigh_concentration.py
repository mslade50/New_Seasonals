"""C3 round 2: the near-a-high subset that a4 Part B left standing.

a2 killed the max-min-sector-spread version of C3 on reference class (in the
SPY-within-3%-of-a-52w-high subset the excess over the all-days control is
+0.016pp at h=5 and -0.002pp at h=10, i.e. exactly nothing).  a4 Part B then
showed the XLV-XLK version of the same gate reading +0.35 / +0.59 / +0.54 at
h=3/5/10 on N=22-27.  Two definitions of one trigger disagreeing is already
the finding, but the surviving one gets the concentration test before it is
called dead, plus the event window that matters for a trade put on tomorrow.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 220)

TK = ["XLV", "XLK", "SPY"]
px = close_panel(TK)
r1 = px.pct_change()
vk = (r1["XLV"] - r1["XLK"]).reindex(px.index)
dist = px["SPY"] / px["SPY"].rolling(252).max() - 1.0
NEAR = ((vk >= 0.030) & (dist > -0.03)).fillna(False)
print(f"C3 near-high cell n_days={int(NEAR.sum())}  fires today "
      f"{bool(NEAR.loc[px.index[-1]])}")

for h in (3, 5, 10):
    ret = fwd_lag(px["SPY"], h, 1)
    valid = ret.dropna().index
    e = declusters(px.index[NEAR.values].intersection(valid), h, valid)
    v = ret.loc[e].values
    base = ret.loc[valid].mean()
    o = np.argsort(-v)
    d1 = np.delete(v, o[0])
    d2 = np.delete(v, o[:2])
    m26 = np.array([d.year >= 2026 for d in e])
    print(f"\nh={h:2d}  N={len(v)}  mean {100*v.mean():+.3f}%  base "
          f"{100*base:+.3f}%  excess {100*(v.mean()-base):+.3f}pp  "
          f"rec {int((v>0).sum())}-{int((v<=0).sum())}  sign p="
          f"{sign_test(int((v>0).sum()), len(v)):.4f}")
    print(f"   {cluster_note(e, v)}")
    print(f"   drop-best excess {100*(d1.mean()-base):+.3f}pp   "
          f"drop-2 excess {100*(d2.mean()-base):+.3f}pp")
    if (~m26).sum():
        vn = v[~m26]
        print(f"   ex-2026 mean {100*vn.mean():+.3f}%  excess "
              f"{100*(vn.mean()-base):+.3f}pp  N={len(vn)}  rec "
              f"{int((vn>0).sum())}-{int((vn<=0).sum())}")
    print(f"   by year: " + ", ".join(
        f"{y}:n={len(g)} {g.mean():+.2f}"
        for y, g in pd.Series(100 * v, index=e).groupby(e.year)))
    # sign test against SPY's OWN base rate, not against a coin
    p_base = float((ret.loc[valid] > 0).mean())
    print(f"   NB base hit rate for SPY at h={h} is {100*p_base:.1f}%, so the "
          f"honest null is p={p_base:.3f}: sign p="
          f"{sign_test(int((v>0).sum()), len(v), p_base):.4f}")

print("\n\n########## EVENT WINDOW FOR A TRADE PUT ON TOMORROW ##########")
ev = load_events()
ev["date"] = pd.to_datetime(ev["date"])
win = ev[(ev["date"] >= "2026-08-19") & (ev["date"] <= "2026-09-05")]
print(win.to_string(index=False))
idx = px.index
print("\nsessions from an entry at the 2026-08-20 close:")
print("  opex 2026-08-21 = +1 session;  NVDA print 2026-08-26 = +4 sessions;")
print("  so ANY h>=4 hold from tomorrow carries the NVDA print uncompensated,")
print("  and every long-tech candidate here (C2 at h>=5, C5 at any horizon)")
print("  is short that gamma by construction.")
