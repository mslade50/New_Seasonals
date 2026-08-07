"""E6: the one cell in the batch with a pulse -- E1's trigger taken LONG at h=1, not short at h=5.

E1 h=1: 38 episodes, +0.155%, t=1.89, hit 76%, vs a +0.042% control. This script asks
whether that survives era splits, drop-best, the placebo weekday check, and the fact that
it is a direction/horizon flip discovered AFTER looking at the h=5 short result
(garden-of-forking-paths: the honest N of tried cells here is 2 signs x 4 horizons = 8).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa: F401,F403

px = load_prices(["SPY", "^VIX"])
spy = px["SPY"]["Close"]
idx = spy.index
vix = px["^VIX"]["Close"].reindex(idx).ffill()
rk5 = pct_rank(spy, 5)
rk5v = pct_rank(vix, 5)
dist = (spy / spy.rolling(252).max() - 1) * 100
full = (rk5 >= 95) & (dist >= -0.5) & (rk5v <= 25)


def fwd_entry_next(s, h):
    return s.shift(-(h + 1)) / s.shift(-1) - 1.0


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))


H = 1
f = fwd_entry_next(spy, H)
valid = rk5.notna() & rk5v.notna() & dist.notna() & f.notna()
ep = declusters(idx[full & valid], H + 1, idx)
v = f[ep].dropna().values
ctrl = f[valid].dropna().values

print("########## E1-LONG h=1 (enter MOC D+1, exit MOC D+2) ##########")
show([summarize(v, "trigger episodes"), summarize(ctrl, "ctrl same window"),
      summarize(f[f.notna()].values, "ctrl all-days")], "headline")
print(f"excess over control: {100*v.mean()-100*ctrl.mean():+.4f}%   Welch t={welch(v, ctrl):+.2f}")
print(f"bootstrap P(mean<=0): {bootstrap_p_le0(v):.4f}")
print(f"after 1bp round-trip: {100*v.mean()-0.01:+.4f}% ({(100*v.mean()-0.01)/0.01:.1f}x cost)")
j, k = int(np.argmax(v)), int(np.argmin(v))
print(f"best {ep[j].date()} {100*v[j]:+.3f}%   worst {ep[k].date()} {100*v[k]:+.3f}%")
show([summarize(np.delete(v, j), "drop-BEST"), summarize(np.delete(v, k), "drop-WORST"),
      summarize(np.sort(v)[:-3], "drop top-3")], "drop-one / drop-top-3")

show(era_split(ep, v, "2013-01-01"), "era 2013")
show(era_split(ep, v, "2018-01-01"), "era 2018")
for lo, hi in [(2000, 2008), (2008, 2013), (2013, 2018), (2018, 2022), (2022, 2027)]:
    m = (ep >= pd.Timestamp(f"{lo}-01-01")) & (ep < pd.Timestamp(f"{hi}-01-01"))
    if m.sum():
        s = summarize(v[m], f"{lo}-{hi}")
        print(f"  {s['label']:>10s} n={s['n']:3d} mean={s['mean_pct']:+.4f}% "
              f"hit={s['hit']:.0f}% worst={s['worst_pct']:+.3f}%")

# forking-paths honesty: the full 2 signs x 4 horizons grid, excess over control
print("\n########## the full sign x horizon grid actually searched ##########")
rows = []
for h in (1, 2, 3, 5, 10):
    fh = fwd_entry_next(spy, h)
    vl = rk5.notna() & rk5v.notna() & dist.notna() & fh.notna()
    e = declusters(idx[full & vl], h + 1, idx)
    vv = fh[e].dropna().values
    cc = fh[vl].dropna().values
    rows.append(dict(h=h, n=len(vv), mean=round(100*vv.mean(), 4),
                     ctrl=round(100*cc.mean(), 4),
                     excess=round(100*(vv.mean()-cc.mean()), 4),
                     t_raw=round(summarize(vv)["t"], 2), t_vs_ctrl=round(welch(vv, cc), 2),
                     hit=round(summarize(vv)["hit"], 0),
                     bootP_long=round(bootstrap_p_le0(vv), 3)))
print(pd.DataFrame(rows).to_string(index=False))
print("Bonferroni over 10 searched cells (5 horizons x 2 signs): a nominal p=0.05 needs "
      "p<0.005, i.e. |t|>~2.9.")

# placebo: does the VIX leg matter at h=1, and is the cell just 'SPY momentum'?
print("\n########## h=1 marginal contribution + weekday placebo ##########")
cA, cB, cC = rk5 >= 95, dist >= -0.5, rk5v <= 25
rows = []
for lab, m in [("rk5>=95 only", cA), ("near-high only", cB), ("VIXrk<=25 only", cC),
               ("rk5 + near-high", cA & cB), ("FULL TRIPLE", full)]:
    e = declusters(idx[m & valid], H + 1, idx)
    rows.append(summarize(f[e].dropna().values, lab))
rows.append(summarize(ctrl, "-- control --"))
show(rows, "h=1 marginal")

nxt_dow = pd.Series(idx, index=idx).shift(-1).dt.dayofweek
names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
rows = []
for dw, nm in names.items():
    e = declusters(idx[full & valid & (nxt_dow == dw)], H + 1, idx)
    vv = f[e].dropna().values
    if len(vv) < 2:
        rows.append(dict(entry_dow=nm, n=len(vv), mean=np.nan, hit=np.nan)); continue
    s = summarize(vv)
    rows.append(dict(entry_dow=nm, n=s["n"], mean=round(s["mean_pct"], 4),
                     hit=round(s["hit"], 0)))
print(pd.DataFrame(rows).to_string(index=False))
print("(today's entry is a FRIDAY -- check the Friday sub-cell specifically)")

# episode list
print("\nepisode list (h=1):")
for d, r in zip(ep, v):
    e_ = idx[idx.get_loc(d) + 1]
    print(f"  trig {d.date()} -> entry {e_.date()} ({e_.day_name()[:3]}) ret={100*r:+.3f}%")
