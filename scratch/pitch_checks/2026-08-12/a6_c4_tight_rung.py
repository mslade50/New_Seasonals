"""C4 round 2 -- the one loose end.

a5's threshold ladder is monotone the RIGHT way (TLT<=0.5 +0.371% at 82.4%,
<=1.0 +0.062%, <=1.5 -0.081%, <=2.0 -0.240%) and today's tape (TLT 0.33%, IEF
0.77%, LQD 0.62% off their 52w lows) sits in the TIGHTEST rung. Picking the best
rung off a sensitivity ladder is exactly the move the registry forbids, so this
script tries to kill the tight rung on its own terms: year histogram, regime
attribution, local control, and whether the tightness is doing work or just
selecting 2022.
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
r1 = vehicle_ret(px, [("TLT", 1.0)], 1, 1)
base = r1.dropna()
bh = float((base > 0).mean())

tight = np.ones(N, bool)
for t, k in [("TLT", 0.5), ("IEF", 1.0), ("LQD", 1.0)]:
    tight &= (off[t] <= k).values
tight &= ~np.isnan(off["TLT"].values)
trig = idx[tight]
epi = declusters(trig, 10, idx)
v = r1.loc[epi].dropna()

print("=" * 100)
print("1. THE TIGHT RUNG: TLT<=0.5, IEF<=1.0, LQD<=1.0 off 52w low")
print("=" * 100)
print(f"  today: TLT {off['TLT'].iloc[-1]:.2f}  IEF {off['IEF'].iloc[-1]:.2f}  "
      f"LQD {off['LQD'].iloc[-1]:.2f}  -> IN the rung")
print(f"  days={int(tight.sum())}  episodes={len(v)}")
hist = pd.Series(1, index=trig).groupby(trig.year).sum()
print(f"  DAY histogram by year: {hist.to_dict()}")
eh = pd.Series(1, index=epi).groupby(epi.year).sum()
print(f"  EPISODE histogram by year: {eh.to_dict()}")
print(f"  distinct years: {len(eh)}   largest year share of episodes: "
      f"{100*eh.max()/eh.sum():.0f}%")

print("\n" + "=" * 100)
print("2. IS IT 2022? drop the biggest year and recompute")
print("=" * 100)
yy = pd.DatetimeIndex(v.index).year
for lbl, m in [("all episodes", np.ones(len(v), bool)),
               (f"ex-{eh.idxmax()}", yy != eh.idxmax()),
               ("ex-2022", yy != 2022)]:
    s = v.values[m]
    if len(s) < 2:
        continue
    w = int((s > 0).sum())
    print(f"  {lbl:16s} N={len(s):3d} {100*s.mean():+.4f}% "
          f"excess {100*(s.mean()-base.mean()):+.4f}pp hit {100*w/len(s):5.1f}% "
          f"signp {sign_test(w, len(s), bh):.4f} "
          f"boot P(<=0) {bootstrap_p_le0(s):.3f}")
print(f"\n  {cluster_note(pd.DatetimeIndex(v.index), v.values, k=3)}")
print(f"  episode dates: {', '.join(str(d.date()) for d in v.index)}")

print("\n" + "=" * 100)
print("3. LOCAL CONTROL -- is the rung better than the days AROUND it?")
print("=" * 100)
loc = local_control(base.index, trig, 126)
print(f"  rung episodes  {100*v.mean():+.4f}%  (N={len(v)})")
print(f"  local +/-126td {100*r1.loc[loc].mean():+.4f}%  (N={len(loc)})")
print(f"  all days       {100*base.mean():+.4f}%")
lc = r1.loc[loc].dropna().values
se = np.sqrt(v.var(ddof=1) / len(v) + lc.var(ddof=1) / len(lc))
print(f"  rung vs local welch t = {(v.mean()-lc.mean())/se:+.2f}")

print("\n" + "=" * 100)
print("4. IS THE TIGHTNESS DOING WORK, or is it TLT<=0.5 alone?")
print("=" * 100)
for lbl, mm in [("TLT<=0.5 & IEF<=1.0 & LQD<=1.0", tight),
                ("TLT<=0.5 alone", (off["TLT"] <= 0.5).values),
                ("IEF<=1.0 alone", (off["IEF"] <= 1.0).values),
                ("LQD<=1.0 alone", (off["LQD"] <= 1.0).values)]:
    mm = mm & ~np.isnan(off["TLT"].values)
    e = declusters(idx[mm], 10, idx)
    s = r1.loc[e].dropna().values
    if len(s) < 2:
        continue
    w = int((s > 0).sum())
    ey = pd.DatetimeIndex(e).year
    print(f"  {lbl:34s} epi={len(s):3d} yrs={len(set(ey)):2d} "
          f"{100*s.mean():+.4f}% hit {100*w/len(s):5.1f}% "
          f"signp {sign_test(w, len(s), bh):.4f}")

print("\n" + "=" * 100)
print("5. COST at the tight rung")
print("=" * 100)
bps = 100 * 100 * (v.mean() - base.mean())
print(f"  excess {bps:+.2f} bps vs 2.5 bps round trip = {bps/2.5:.2f}x "
      f"(bar is 5x)")
print(f"  worst episode {100*v.min():+.2f}% on "
      f"{v.index[int(np.argmin(v.values))].date()}")
