"""C3/C4/C5 round 2b — teardown of the only cells that survived b1b's tdom and
ladder controls: SLV (all-months and August), USO, FXI, GLD.

Attacks, in order of how likely each is to kill:
  1. era split — the best August years are 2009/2012/2010, the metals bull.
  2. is it SILVER or is it METALS? GLD's August ladder also ranks 1 of 17, so
     the honest question is whether the SLV cell is anything but a 2.5x-beta
     expression of an August gold seasonal. Beta-hedged residual answers it.
  3. is it the OPEX GATE or the WINDOW? the August ladder is a PLATEAU from
     off=-1 to off=+4 (2.12 / 3.66 / 3.39 / 3.26 / 2.56 / 1.88); measure the
     gate's marginal contribution rather than the level.
  4. multiplicity priced by RELOCATED-ANCHOR permutation over the grid that
     was actually walked (vehicle x horizon x {pooled, August}).
  5. tail and live-state honesty.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

VEH = ["GLD", "SLV", "TLT", "IEF", "HYG", "LQD", "USO", "XLE", "UUP", "FXI"]
px = close_panel(VEH)
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
opex = pd.DatetimeIndex([d for d in load_events(["opex"])["date"] if d in pos.index])
tdom = pd.Series(pd.Series(idx, index=idx).groupby([idx.year, idx.month])
                 .cumcount().values + 1, index=idx)
ATD = [9, 10, 11, 12, 13, 14]
aA = pd.DatetimeIndex([idx[pos[d] - 1] for d in opex if pos[d] >= 1])
augA = pd.DatetimeIndex([d for d in aA if idx[pos[d] + 1].month == 8])


def R(v, h, lag=1):
    return fwd_lag(px[v].dropna(), h, lag)


print("===== 1. era split, SLV =====")
for lbl, anc in [("all-months", aA), ("August", augA)]:
    for h in (5, 10):
        r = R("SLV", h)
        a = pd.DatetimeIndex(anc).intersection(r.dropna().index)
        v = r.loc[a]
        for cut in (2013, 2018):
            pre, post = v[v.index.year < cut], v[v.index.year >= cut]
            print(f"SLV {lbl} h={h}: pre-{cut} {100*pre.mean():+.3f}% "
                  f"(N={len(pre)}, {int((pre>0).sum())}-{int((pre<=0).sum())})  "
                  f"| {cut}+ {100*post.mean():+.3f}% (N={len(post)}, "
                  f"{int((post>0).sum())}-{int((post<=0).sum())}, "
                  f"sign p={sign_test(int((post>0).sum()), len(post)):.4f})")
    print()

print("===== 1b. era split, the other ladder survivors (h=10) =====")
for v_ in ("USO", "FXI", "GLD", "XLE"):
    for lbl, anc in [("all", aA), ("Aug", augA)]:
        r = R(v_, 10)
        a = pd.DatetimeIndex(anc).intersection(r.dropna().index)
        s = r.loc[a]
        pre, post = s[s.index.year < 2018], s[s.index.year >= 2018]
        print(f"{v_:4s} {lbl:4s} h=10: full {100*s.mean():+.3f}% (N={len(s)}) "
              f"| pre-2018 {100*pre.mean():+.3f}% (N={len(pre)}) "
              f"| 2018+ {100*post.mean():+.3f}% (N={len(post)}, "
              f"{int((post>0).sum())}-{int((post<=0).sum())})")

print("\n===== 2. SILVER or METALS? beta-hedged SLV-vs-GLD residual =====")
d = pd.DataFrame({"SLV": px["SLV"], "GLD": px["GLD"]}).dropna().pct_change().dropna()
beta = np.polyfit(d["GLD"], d["SLV"], 1)[0]
print(f"beta(SLV daily on GLD daily) = {beta:.3f}  (N={len(d)})")
for lbl, anc in [("all-months", aA), ("August", augA)]:
    for h in (5, 10):
        rs, rg = R("SLV", h), R("GLD", h)
        a = pd.DatetimeIndex(anc).intersection(rs.dropna().index)\
                                 .intersection(rg.dropna().index)
        res = rs.loc[a] - beta * rg.loc[a]
        base_i = rs.dropna().index.intersection(rg.dropna().index)
        base_i = base_i[(base_i >= a[0]) & (base_i <= a[-1])]
        if lbl == "August":
            base_i = base_i[base_i.month == 8]
        base = rs.loc[base_i] - beta * rg.loc[base_i]
        print(f"SLV-{beta:.2f}*GLD {lbl} h={h}: cell {100*res.mean():+.3f}% "
              f"(N={len(res)}, {int((res>0).sum())}-{int((res<=0).sum())}, "
              f"sign p={sign_test(int((res>0).sum()), len(res)):.4f}) "
              f"| own base {100*base.mean():+.3f}% (N={len(base)}) "
              f"| excess {100*(res.mean()-base.mean()):+.3f}pp")

print("\n===== 3. what does the OPEX GATE add over the WINDOW? =====")
print("cell = anchor(opex-1). window = every August session at anchor tdom band.")
for v_ in ("SLV", "GLD", "XLE", "FXI", "USO", "HYG"):
    for h in (5, 10):
        r = R(v_, h).dropna()
        a = pd.DatetimeIndex(augA).intersection(r.index)
        near = pd.DatetimeIndex(
            [idx[pos[d] + o] for d in a for o in (-2, -1, 1, 2, 3)
             if 0 <= pos[d] + o < len(idx)]).intersection(r.index).difference(a)
        wide = r.index[(r.index.month == 8) & (tdom.reindex(r.index).isin(ATD))
                       & (r.index >= a[0]) & (r.index <= a[-1])].difference(a)
        print(f"{v_:4s} Aug h={h:2d}: cell {100*r.loc[a].mean():+.3f}% (N={len(a)}) "
              f"| adjacent +-2/3 sessions {100*r.loc[near].mean():+.3f}% "
              f"(N={len(near)}) -> GATE ADDS {100*(r.loc[a].mean()-r.loc[near].mean()):+.3f}pp "
              f"| wide Aug tdom window {100*r.loc[wide].mean():+.3f}%")

print("\n===== 4. multiplicity: RELOCATED-ANCHOR permutation over the walked grid =====")
print("null: slide the whole monthly anchor set by a random offset in +-(9..40)")
print("sessions, recompute the SAME grid (10 vehicles x h in {3,5,10} x")
print("{pooled, August}) = 60 cells, take the max excess-vs-own-drift.")
rng = np.random.default_rng(42)
rets = {(v_, h): R(v_, h) for v_ in VEH for h in (3, 5, 10)}


def grid_max(anchor_set):
    best, arg = -9e9, None
    for v_ in VEH:
        for h in (3, 5, 10):
            r = rets[(v_, h)].dropna()
            for lbl in ("pooled", "Aug"):
                a = pd.DatetimeIndex(anchor_set).intersection(r.index)
                if lbl == "Aug":
                    a = pd.DatetimeIndex([x for x in a if x.month == 8])
                if len(a) < 15:
                    continue
                base = r[(r.index >= a[0]) & (r.index <= a[-1])]
                if lbl == "Aug":
                    base = base[base.index.month == 8]
                e = 100 * (r.loc[a].mean() - base.mean())
                if e > best:
                    best, arg = e, (v_, h, lbl)
    return best, arg


obs, oarg = grid_max(aA)
print(f"OBSERVED max cell: {oarg} at {obs:+.3f}pp")
nulls = []
for _ in range(400):
    o = int(rng.choice(list(range(-40, -8)) + list(range(9, 41))))
    anc = pd.DatetimeIndex([idx[pos[d] - 1 + o] for d in opex
                            if 0 <= pos[d] - 1 + o < len(idx)])
    b, _a = grid_max(anc)
    nulls.append(b)
nulls = np.array(nulls)
print(f"null max-of-grid: mean {nulls.mean():+.3f}pp, median "
      f"{np.median(nulls):+.3f}pp, p90 {np.percentile(nulls,90):+.3f}pp, "
      f"max {nulls.max():+.3f}pp")
print(f"P(null max >= observed {obs:.3f}pp) = {(nulls >= obs).mean():.3f}")

print("\n===== 5. tail + live-state honesty (SLV) =====")
r10 = R("SLV", 10)
a = pd.DatetimeIndex(augA).intersection(r10.dropna().index)
vals = r10.loc[a]
print("August SLV h=10 episode-by-episode:")
print("  " + "  ".join(f"{d.year}:{100*x:+.1f}" for d, x in vals.items()))
print(f"  worst {100*vals.min():+.2f}% ({vals.idxmin().year}), "
      f"median {100*vals.median():+.2f}%, mean {100*vals.mean():+.2f}%, "
      f"sd {100*vals.std(ddof=1):.2f}%")
print(f"  bootstrap P(mean<=0) = {bootstrap_p_le0(vals.values):.3f}")
print(f"  {cluster_note(vals.index, vals.values, k=3)}")
s = px["SLV"].dropna()
print(f"\nlive state 2026-08-20: SLV {s.iloc[-1]:.2f}, "
      f"21d {100*(s.iloc[-1]/s.iloc[-22]-1):+.2f}%, "
      f"63d {100*(s.iloc[-1]/s.iloc[-64]-1):+.2f}%, "
      f"252d {100*(s.iloc[-1]/s.iloc[-253]-1):+.2f}%, "
      f"pct of 252d high {100*s.iloc[-1]/s.iloc[-252:].max():.1f}%")
tr21 = s / s.shift(21) - 1.0
print("trigger-day trailing 21d on the 20 August anchors: median "
      f"{100*tr21.loc[a].median():+.2f}%, today {100*tr21.iloc[-1]:+.2f}%, "
      f"pctile of today within the trigger set "
      f"{100*(tr21.loc[a] < tr21.iloc[-1]).mean():.0f}")
