"""C1 round 2b -- the August problem.

a2 found the live cell (PPI print, CPI on the eve) is 0-for-4 in AUGUST at
-0.847% mean, and the PARENT's August is +0.020% at a 50.0% hit over N=24, the
joint-weakest month with December. Today is an August PPI print.

Before treating that as a kill, test whether MONTH is a real conditioner on this
cell at all or 12 buckets of noise. The honest test: does the month ranking
built on the FIRST half of the sample predict the SECOND half? If it does not,
0-for-4 is four coin flips and the kill is illegal. If it does, August is a
definitional-fragility kill.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["TLT"])
tl = px["TLT"].dropna()
idx = tl.index
c = tl.values
N = len(c)
d1 = np.full(N, np.nan)
d1[1:] = c[1:] / c[:-1] - 1.0
base_hit = float((d1[~np.isnan(d1)] > 0).mean())

ev = load_events()
sp = lambda k: sorted({int(idx.searchsorted(x, "left"))
                       for x in ev[ev.event == k]["date"]
                       if 0 <= int(idx.searchsorted(x, "left")) < N})
ppi_l = [p for p in sp("ppi") if 1 <= p < N and not np.isnan(d1[p])]
cpi_all = set(sp("cpi"))

v = np.array([d1[p] for p in ppi_l])
dt = pd.DatetimeIndex([idx[p] for p in ppi_l])
mo, yr = dt.month.values, dt.year.values
ceve = np.array([(p - 1) in cpi_all for p in ppi_l])

print("=" * 100)
print("1. SPLIT-HALF STABILITY OF THE MONTH PROFILE (parent cell)")
print("=" * 100)
cut = 2014
h1 = yr < cut
rows = []
for m in range(1, 13):
    a = v[h1 & (mo == m)]
    b = v[(~h1) & (mo == m)]
    rows.append({"month": m, "N_pre": len(a), "pre_pct": round(100 * a.mean(), 3),
                 "N_post": len(b), "post_pct": round(100 * b.mean(), 3)})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
r = np.corrcoef(df["pre_pct"], df["post_pct"])[0, 1]
rk = pd.Series(df["pre_pct"]).rank().corr(pd.Series(df["post_pct"]).rank())
print(f"\n  corr(pre-{cut} monthly mean, {cut}+ monthly mean) = {r:+.3f}   "
      f"rank corr = {rk:+.3f}")
print("  -> near zero means the month profile does not persist and August's")
print("     4 losses are noise, not a conditioner.")

print("\n" + "=" * 100)
print("2. IS AUGUST SPECIAL, OR IS EVERY MONTH THIS DISPERSED? (permutation)")
print("=" * 100)
means = np.array([v[mo == m].mean() for m in range(1, 13)])
obs_sd = means.std(ddof=1)
rng = np.random.default_rng(42)
sims = []
for _ in range(5000):
    sh = rng.permutation(v)
    sims.append(np.array([sh[mo == m].mean() for m in range(1, 13)]).std(ddof=1))
sims = np.array(sims)
print(f"  observed sd of the 12 monthly means = {100*obs_sd:.4f}pp")
print(f"  permutation null mean sd = {100*sims.mean():.4f}pp, "
      f"P(sim >= obs) = {(sims >= obs_sd).mean():.3f}")
aug = v[mo == 8]
w = int((aug > 0).sum())
print(f"  August parent: N={len(aug)} mean {100*aug.mean():+.4f}% hit "
      f"{100*w/len(aug):.1f}%  P(<= {w} wins | base {base_hit:.3f}) = "
      f"{sign_test(len(aug)-w, len(aug), 1-base_hit):.4f}")
rk_aug = int((means <= means[7]).sum())
print(f"  August's rank among the 12 monthly means: {rk_aug}/12")

print("\n" + "=" * 100)
print("3. THE LIVE CELL'S 4 AUGUST OBSERVATIONS, ONE BY ONE")
print("=" * 100)
liv = ceve & (mo == 8)
for i in np.where(liv)[0]:
    p = ppi_l[i]
    print(f"  {dt[i].date()}  print-day {100*v[i]:+.3f}%   "
          f"CPI-day (eve) {100*d1[p-1]:+.3f}%   yr {yr[i]}")
print(f"\n  combined mean {100*v[liv].mean():+.4f}%  "
      f"P(0 wins in 4 | base {base_hit:.3f}) = {(1-base_hit)**4:.4f}")
print("  Under the cell's OWN 63.6% hit rate, P(0 of 4) = "
      f"{(1-0.636)**4:.4f}")

print("\n" + "=" * 100)
print("4. AUGUST ACROSS THE OTHER RATES INSTRUMENTS on the parent cell")
print("   (a real August effect should show up in IEF and in yields too)")
print("=" * 100)
px2 = close_panel(["TLT", "IEF", "LQD", "^TNX"])
for t in ["TLT", "IEF", "LQD", "^TNX"]:
    s = px2[t].dropna()
    ii, cc = s.index, s.values
    vv, mm = [], []
    for p in ppi_l:
        d = idx[p]
        if d in ii:
            a = ii.get_loc(d)
            if a >= 1:
                vv.append(cc[a] / cc[a - 1] - 1.0)
                mm.append(d.month)
    vv, mm = np.array(vv), np.array(mm)
    ag, ot = vv[mm == 8], vv[mm != 8]
    print(f"  {t:5s} August N={len(ag):3d} {100*ag.mean():+.4f}% hit "
          f"{100*(ag>0).mean():5.1f}%  |  other months "
          f"{100*ot.mean():+.4f}% hit {100*(ot>0).mean():5.1f}%")

print("\n" + "=" * 100)
print("5. SEQUENTIAL / RECENCY: last 24 parent prints and last 12 live-cell obs")
print("=" * 100)
for lbl, sel in [("parent last 24", np.arange(len(v))[-24:]),
                 ("live cell last 12", np.where(ceve)[0][-12:])]:
    s = v[sel]
    w = int((s > 0).sum())
    print(f"  {lbl:20s} N={len(s)} {100*s.mean():+.4f}% hit {100*w/len(s):.1f}% "
          f"signp {sign_test(w, len(s), base_hit):.4f}  "
          f"dates {dt[sel][0].date()}..{dt[sel][-1].date()}")
