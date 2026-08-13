"""r2 - RED TEAM attack 2: price the search charge, then stress the one
statistic the case rests on (the excess-over-drift bootstrap, 0.0022).

1. sign test both ways (coin, and IHI's OWN up-base-rate) + 2-look Sidak /
   Bonferroni, then the honest wider charge (horizons actually scanned).
2. is the bootstrap inflated by overlap? -> episode gap census, min_gap
   ladder (5/10/21/63/252), block bootstrap over episodes, and a CLUSTER
   bootstrap that resamples YEARS (the real unit of dependence here).
3. bootstrap on the EXCESS series rather than the raw, since the excess is
   what the claim is.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

H = 5
c = load_prices(["IHI"])["IHI"]["Close"].dropna()
r21 = pct_rank(c, 21)
dd = c / c.rolling(252).max() - 1.0
m = ((r21 >= 99) & (dd <= -0.10)).fillna(False)
ret = fwd_lag(c, H)
valid = ret.notna()
trig = c.index[m.values & valid.values]
epi = declusters(trig, 5, c.index)
epi = epi[ret.reindex(epi).notna().values]
v = ret.loc[epi].values
span = (c.index >= trig[0]) & (c.index <= trig[-1]) & valid.values
ctrl = ret[span].values
base_hit = float((ctrl > 0).mean())
wins = int((v > 0).sum())
n = len(v)
xs = v - ctrl.mean()

print("=== 1. THE RECORD, and what the looks cost it ===")
p_coin = sign_test(wins, n)
p_base = sign_test(wins, n, base_hit)
print(f"  episodes N={n} record {wins}-{n-wins} ({100*wins/n:.1f}% hit)")
print(f"  IHI own up-base-rate over the same span = {100*base_hit:.2f}% "
      f"(N={len(ctrl)} days)")
print(f"  sign p vs a COIN          = {p_coin:.4f}")
print(f"  sign p vs IHI's OWN base  = {p_base:.4f}   <-- the honest baseline")
for L, lbl in [(2, "2 looks (fade pre-specified + long instructed)"),
               (6, "6 looks (2 dir x 3 horizon families 3/5/10)"),
               (20, "20 looks (2 dir x 10 horizons scanned in c6_dev)")]:
    print(f"  {lbl}:")
    print(f"    Sidak on coin  p = {1-(1-p_coin)**L:.4f}   "
          f"Bonferroni = {min(1, p_coin*L):.4f}")
    print(f"    Sidak on BASE  p = {1-(1-p_base)**L:.4f}   "
          f"Bonferroni = {min(1, p_base*L):.4f}")

print("\n  the horizon scan h=5 was picked from (episode mean, min_gap 5):")
hs = []
for h in range(1, 11):
    r = fwd_lag(c, h)
    tt = c.index[m.values & r.notna().values]
    e = declusters(tt, 5, c.index)
    e = e[r.reindex(e).notna().values]
    vv = r.loc[e].values
    sp = (c.index >= tt[0]) & (c.index <= tt[-1]) & r.notna().values
    hs.append({"h": h, "n": len(vv), "mean_pct": round(100*vv.mean(), 3),
               "hit": round(100*(vv > 0).mean(), 1),
               "excess_pp": round(100*(vv.mean()-r[sp].mean()), 3),
               "sign_p_base": round(sign_test(int((vv > 0).sum()), len(vv),
                                              float((r[sp] > 0).mean())), 4)})
hd = pd.DataFrame(hs)
print(hd.to_string(index=False))
print(f"  h=5 is the ARGMAX of excess_pp over 10 horizons: "
      f"{hd.loc[hd.excess_pp.idxmax(),'h'] == 5}. "
      f"best-of-10 min sign_p_base = {hd.sign_p_base.min():.4f} at h="
      f"{int(hd.loc[hd.sign_p_base.idxmin(),'h'])}")

print("\n=== 2. IS THE BOOTSTRAP INFLATED BY OVERLAP? ===")
pos = pd.Series(range(len(c.index)), index=c.index)
gaps = np.diff([pos[d] for d in epi])
print(f"  episode dates: {', '.join(str(d.date()) for d in epi)}")
print(f"  gaps between consecutive episodes (td): {list(gaps)}")
print(f"  gaps < h+1={H+1} td (holding windows would touch/overlap): "
      f"{int((gaps <= H).sum())} of {len(gaps)}")
print(f"  gaps < 252 td (same-year dependence): {int((gaps < 252).sum())}")
print("\n  min_gap ladder (raw mean / excess / bootstrap):")
rows = []
for g in (5, 10, 21, 63, 126, 252):
    e = declusters(trig, g, c.index)
    e = e[ret.reindex(e).notna().values]
    vv = ret.loc[e].values
    rows.append({"min_gap_td": g, "n": len(vv),
                 "mean_pct": round(100*vv.mean(), 3),
                 "hit": round(100*(vv > 0).mean(), 1),
                 "excess_pp": round(100*(vv.mean()-ctrl.mean()), 3),
                 "boot_P_raw_le0": round(bootstrap_p_le0(vv), 4),
                 "boot_P_excess_le0": round(bootstrap_p_le0(vv-ctrl.mean()), 4),
                 "sign_p_base": round(sign_test(int((vv > 0).sum()), len(vv),
                                                base_hit), 4)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== 3. CLUSTER BOOTSTRAP BY CALENDAR YEAR (the real unit) ===")
yrs = pd.Series(v, index=epi.year)
by = yrs.groupby(level=0).agg(["mean", "count", "sum"])
print("  episodes by year:")
print(by.assign(mean=lambda d: (100*d["mean"]).round(3),
                sum=lambda d: (100*d["sum"]).round(2)).to_string())
groups = [g.values for _, g in yrs.groupby(level=0)]
xgroups = [g.values - ctrl.mean() for _, g in yrs.groupby(level=0)]
rng = np.random.default_rng(42)


def cluster_boot(gs, nb=20000):
    K = len(gs)
    out = np.empty(nb)
    for b in range(nb):
        pick = rng.integers(0, K, K)
        out[b] = np.concatenate([gs[i] for i in pick]).mean()
    return out


for lbl, gs in [("RAW", groups), ("EXCESS", xgroups)]:
    bb = cluster_boot(gs)
    print(f"  {lbl}: year-cluster bootstrap P(mean<=0) = {(bb <= 0).mean():.4f}   "
          f"(iid episode bootstrap = "
          f"{bootstrap_p_le0(v if lbl=='RAW' else xs):.4f})   "
          f"mean {100*bb.mean():+.3f}%  2.5/97.5 "
          f"[{100*np.percentile(bb,2.5):+.3f}, {100*np.percentile(bb,97.5):+.3f}]")

print("\n=== 4. BLOCK BOOTSTRAP over the episode sequence (block=3) ===")
for lbl, arr in [("RAW", v), ("EXCESS", xs)]:
    nb, bl = 20000, 3
    nblk = int(np.ceil(n / bl))
    out = np.empty(nb)
    for b in range(nb):
        starts = rng.integers(0, n, nblk)
        samp = np.concatenate([np.take(arr, np.arange(s, s+bl), mode="wrap")
                               for s in starts])[:n]
        out[b] = samp.mean()
    print(f"  {lbl}: block bootstrap P(mean<=0) = {(out <= 0).mean():.4f}  "
          f"mean {100*out.mean():+.3f}%")

print("\n=== 5. and the cross-sectional charge from r1b, restated ===")
print("  single-ticker permutation p (IHI alone)   = 0.0440")
print("  family-wise max-stat p over 27 sector ETFs = 0.9330")
print("  Cochran Q p = 0.544, I^2 = 0.0%, common-mean excess = -0.035pp")
print("  -> whatever the within-IHI bootstrap says, the cross-section prices")
print("     this cell at ZERO and IHI's max is SUB-MEDIAN under the null max.")
