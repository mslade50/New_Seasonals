"""D3 round 2b -- two last numbers before the verdict is written.

  1. IS TODAY A FRESH TRIGGER? Round 1 reported cluster depth 2. The registry
     is explicit: "mid-cluster entry is not a fresh trigger -- compute cluster
     depth before quoting an episode statistic as today's expectation."
     Every episode statistic in d3/d3b is measured on FIRST days of clusters.
  2. HOW MUCH OF THE SHORT'S EDGE IS THE CURRENT YEAR? 2026 contributes
     +9.20pp of the +31.03pp total on 2 episodes. Drop it and see.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["EWZ", "SPY"]).dropna()
px = px.loc[px.index >= "2003-05-01"]
idx = px.index
r5 = {t: px[t].pct_change(5) for t in ["EWZ", "SPY"]}
d52s = px["SPY"] / px["SPY"].rolling(252).max() - 1.0
gate = ((d52s >= -0.010) & (r5["SPY"] >= 0.020)).fillna(False)
m = (gate & (r5["EWZ"] < 0)).fillna(False)

H = 5
ret = vehicle_ret(px, [("EWZ", -1.0), ("SPY", 1.0)], H, 1)
val = ret.notna()
sig = idx[m.values]
epi = declusters(idx[m.values & val.values], H, idx)

print("1. FRESHNESS")
print(f"   trigger fires today: {bool(m.iloc[-1])}")
run = 0
for v in m.values[::-1]:
    if v:
        run += 1
    else:
        break
print(f"   consecutive trigger sessions ending today: {run}")
print(f"   first day of this cluster: {idx[-run].date()} "
      f"(today is day {run} of it)")
epi_set = set(pd.DatetimeIndex(epi))
print(f"   would today be counted as an EPISODE by the same declustering "
      f"rule used for every stat above? "
      f"{'YES' if idx[-1] in epi_set else 'NO -- it is a follow-on day'}")
# what do follow-on days do, historically?
first_days = set(declusters(sig, H, idx))
follow = [d for d in sig if d not in first_days]
follow = pd.DatetimeIndex([d for d in follow if val.get(d, False)])
fd = pd.DatetimeIndex([d for d in declusters(sig, H, idx) if val.get(d, False)])
print(f"\n   FIRST days of clusters (N={len(fd)}): short-spread "
      f"{100*ret.loc[fd].mean():+.3f}%")
print(f"   FOLLOW-ON days      (N={len(follow)}): short-spread "
      f"{100*ret.loc[follow].mean():+.3f}%  <- the population today belongs to")
w = int((ret.loc[follow] > 0).sum())
print(f"   follow-on record {w}-{len(follow)-w}, sign p "
      f"{sign_test(w, len(follow)):.3f}")

print("\n2. CURRENT-YEAR DEPENDENCE (short cell, h=5 episodes)")
v = ret.loc[epi].values
yrs = pd.DatetimeIndex(epi).year
print(f"   all              {100*v.mean():+.3f}%  N={len(v)}")
for drop in (2026, 2015):
    k = yrs != drop
    vv = v[k]
    w = int((vv > 0).sum())
    print(f"   ex-{drop}          {100*vv.mean():+.3f}%  N={len(vv)}  "
          f"{w}-{len(vv)-w}  sign p {sign_test(w, len(vv)):.3f}  "
          f"boot P<=0 {bootstrap_p_le0(vv):.3f}")
k = ~np.isin(yrs, [2026, 2015])
vv = v[k]
w = int((vv > 0).sum())
print(f"   ex-2026 AND 2015 {100*vv.mean():+.3f}%  N={len(vv)}  "
      f"{w}-{len(vv)-w}  sign p {sign_test(w, len(vv)):.3f}  "
      f"boot P<=0 {bootstrap_p_le0(vv):.3f}")
print(f"   -> vs SPY-gate-only control +0.408%: the ex-2026/2015 cell "
      f"{'BEATS' if vv.mean() > 0.00408 else 'LOSES TO'} the control "
      f"by {100*vv.mean()-0.408:+.3f}pp")

print("\n3. LOSER PATHS on the short cell (what an invalidation looks like)")
pp = episode_paths(px, epi, [("EWZ", -1.0), ("SPY", 1.0)], H, 1)
fin = pp[H]
los = pp[fin < 0]
print(f"   {len(los)}/{len(pp)} episodes finish negative; mean finish "
      f"{100*los[H].mean():.2f}%, worst {100*los[H].min():.2f}% "
      f"({los[H].idxmin().date()})")
print(f"   of the losers, {int((los[1] < 0).sum())}/{len(los)} were already "
      f"red at day 1; mean day-1 mark of the losers {100*los[1].mean():.2f}%")
print(f"   of the WINNERS, {int((pp[fin >= 0][1] < 0).sum())}/"
      f"{int((fin >= 0).sum())} were red at day 1 -- so a day-1 stop would "
      f"cut winners too")
