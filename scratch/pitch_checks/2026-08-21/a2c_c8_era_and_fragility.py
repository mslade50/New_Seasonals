"""C8 round 2c: era composition and the definition-fragility number.

a2b showed the ONLY positive rung of the drawdown-gap grid is <=-25pp
(N=20, beta-neutral +1.006%, 75% hit, sign p 0.021). Two things decide it:
  1. what era those 20 episodes come from, and what is left out-of-sample
  2. the fragility step -- <=-24pp reads +1.051% and <=-23pp reads +0.237%,
     so two adjacent observations flip the cell. Name their return.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import warnings
warnings.filterwarnings("ignore")

ASOF = pd.Timestamp("2026-08-20")
H = 5
px = close_panel(["SLV", "GLD"]).loc[:ASOF].dropna()
idx = px.index
slv5, gld5 = pct_rank(px["SLV"], 5), pct_rank(px["GLD"], 5)
slv_dd = px["SLV"] / rolling_on_valid(px["SLV"], lambda x: x.rolling(252).max()) - 1.0
gld_dd = px["GLD"] / rolling_on_valid(px["GLD"], lambda x: x.rolling(252).max()) - 1.0
gap = slv_dd - gld_dd
B = float(np.polyfit(px["GLD"].pct_change().dropna(), px["SLV"].pct_change().dropna(), 1)[0])
r_bn = vehicle_ret(px, [("SLV", 1.0), ("GLD", -B)], H, 1)
r_ed = vehicle_ret(px, [("SLV", 1.0), ("GLD", -1.0)], H, 1)
thrust = ((slv5 >= 70) & (gld5 >= 70)).fillna(False)

cell = (thrust & (gap <= -0.25)).fillna(False)
epi = declusters(idx[(cell & r_bn.notna()).values], 5, idx)
v = r_bn.loc[epi].values
yrs = pd.DatetimeIndex(epi).year

print("=" * 100)
print("C8c-1  ERA COMPOSITION of the gap<=-25pp cell (the only positive rung)")
print("=" * 100)
print(f"  N={len(epi)}  beta-neutral mean {100*v.mean():+.3f}%")
for lo, hi, lbl in ((1900, 2013, "2008-2012 (GFC / silver bubble unwind)"),
                    (2013, 2021, "2013-2020"),
                    (2021, 2100, "2021+")):
    m = (yrs >= lo) & (yrs < hi)
    if m.sum() == 0:
        print(f"  {lbl:<42} N=0")
        continue
    print(f"  {lbl:<42} N={int(m.sum()):<4} mean {100*v[m].mean():+7.3f}%  "
          f"hit {100*(v[m]>0).mean():5.1f}%")
live = pd.DatetimeIndex([d for d in epi if d >= pd.Timestamp("2026-08-01")])
print(f"  episodes inside the LIVE cluster (2026-08+): {len(live)} -> "
      f"{', '.join(str(d.date()) for d in live)}")
oos = [d for d in epi if d.year >= 2013 and d not in live]
print(f"  independent evidence since 2013 excluding the live cluster: {len(oos)} episodes "
      f"({', '.join(str(d.date()) for d in oos)})")
if oos:
    ov = r_bn.loc[pd.DatetimeIndex(oos)].values
    print(f"     -> mean {100*ov.mean():+.3f}%  hit {100*(ov>0).mean():.1f}%")

print("\n" + "=" * 100)
print("C8c-2  DEFINITION FRAGILITY: name the two observations that flip the cell")
print("=" * 100)
e24 = declusters(idx[((thrust & (gap <= -0.24)).fillna(False) & r_ed.notna()).values], 5, idx)
e23 = declusters(idx[((thrust & (gap <= -0.23)).fillna(False) & r_ed.notna()).values], 5, idx)
add = e23.difference(e24)
print(f"  gap<=-24pp: N={len(e24)}  eq-dollar {100*r_ed.loc[e24].mean():+.3f}%  "
      f"bn {100*r_bn.loc[e24].mean():+.3f}%")
print(f"  gap<=-23pp: N={len(e23)}  eq-dollar {100*r_ed.loc[e23].mean():+.3f}%  "
      f"bn {100*r_bn.loc[e23].mean():+.3f}%")
print(f"  the {len(add)} episode(s) that enter between -24 and -23:")
for d in add:
    print(f"     {d.date()}  gap {100*gap.loc[d]:+.2f}pp  eq-dollar h=5 {100*r_ed.loc[d]:+.2f}%  "
          f"bn {100*r_bn.loc[d]:+.2f}%")
print(f"  today's gap is {100*gap.loc[ASOF]:.2f}pp, i.e. {100*(gap.loc[ASOF]+0.25):.2f}pp inside "
      f"the only positive rung.")
