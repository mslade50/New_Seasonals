"""The three-way survived tdom (live entry tdom 5 is INSIDE the observed
support 2-5; tdom-matched excess +1.686pp of an all-days +1.802pp). So it dies
here or it lives.

Two tests, both about whether the TLT@low leg carries information or is the
max of the search I ran to find it:

  1. SUBSET PERMUTATION. Inside the parent (cmdty-high + print, 54 episodes at
     h=8), how often does a RANDOM 5-episode subset beat the three-way's
     +1.670% mean, and how often is a random 5-subset perfect? That prices the
     leg against its own parent with no distributional assumption.
  2. SEARCH COST. I searched 2 vehicles x 4 horizons x 2 forms = 16 cells and
     am quoting the best. Family-wise probability that SOME cell looks this
     good.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
px = close_panel(["DBC", "TLT", "IEF"])
raw = load_prices(["DBC", "TLT"])
IDX = px.index
state = (1.0 - raw["DBC"]["Close"] /
         rolling_on_valid(raw["DBC"]["Close"], lambda x: x.rolling(252).max())).reindex(IDX) <= 0.0025
mom = (raw["TLT"]["Close"] /
       rolling_on_valid(raw["TLT"]["Close"], lambda x: x.rolling(252).min()) - 1.0).reindex(IDX) <= 0.02


def flag(h, lag=1, k=0):
    ev = load_events(["ppi", "cpi"])["date"]
    pos, _ = anchor_positions(IDX, ev, offset=k)
    e = np.asarray(pd.DatetimeIndex([IDX[p] for p in pos]).values, dtype="datetime64[ns]")
    out = np.zeros(len(IDX), dtype=bool)
    for i in range(len(IDX)):
        if i + lag + h >= len(IDX):
            continue
        out[i] = bool(((e > np.datetime64(IDX[i + lag])) &
                       (e <= np.datetime64(IDX[i + lag + h]))).any())
    return pd.Series(out, index=IDX)


print("=" * 78)
print("THE FULL SEARCH GRID I RAN  (2 vehicles x 4 horizons x 2 forms)")
print("=" * 78)
grid = []
for tkr in ("TLT", "IEF"):
    for H in (3, 5, 8, 10):
        pr = flag(H)
        ret = vehicle_ret(px, [(tkr, -1.0)], H, 1)
        valid = ret.notna()
        for form, m in (("2-way (cmdty+print)", state & pr),
                        ("3-way (+TLT@low)", state & mom & pr)):
            dd = IDX[m.values & valid.values]
            e = declusters(dd, H, IDX)
            v = ret.loc[e].values
            if len(v) == 0:
                continue
            w = int((v > 0).sum())
            grid.append({"cell": f"SHORT {tkr} h={H} {form}", "n": len(v),
                         "mean_pct": round(100 * v.mean(), 3),
                         "hit": round(100 * (v > 0).mean(), 1),
                         "sign_p": round(sign_test(w, len(v)), 4)})
show(grid, "every cell the search touched")
best = min(grid, key=lambda r: r["sign_p"])
print(f"  BEST BY SIGN p: {best['cell']}  n={best['n']} mean {best['mean_pct']:+.3f}% "
      f"sign p {best['sign_p']}")

print("\n" + "=" * 78)
print("1. SUBSET PERMUTATION inside the parent (SHORT TLT h=8, 54 episodes)")
print("=" * 78)
H = 8
pr = flag(H)
ret = vehicle_ret(px, [("TLT", -1.0)], H, 1)
valid = ret.notna()
par = declusters(IDX[(state & pr).values & valid.values], H, IDX)
vpar = ret.loc[par].values
three = declusters(IDX[(state & mom & pr).values & valid.values], H, IDX)
v3 = ret.loc[three].values
obs, k = v3.mean(), len(v3)
print(f"parent n={len(vpar)} mean {100*vpar.mean():+.3f}% sd {100*vpar.std(ddof=1):.3f}%")
print(f"three-way n={k} mean {100*obs:+.3f}%  hit 100%")
B = 200000
draws = np.array([rng.choice(vpar, size=k, replace=False) for _ in range(B)])
p_mean = float((draws.mean(axis=1) >= obs).mean())
p_perf = float((draws > 0).all(axis=1).mean())
p_both = float(((draws.mean(axis=1) >= obs) & (draws > 0).all(axis=1)).mean())
print(f"\n  P(random {k}-subset of the parent has mean >= {100*obs:.3f}%) = {p_mean:.4f}")
print(f"  P(random {k}-subset is PERFECT {k}-for-{k})                  = {p_perf:.4f}")
print(f"  P(both)                                                = {p_both:.4f}")
print(f"  -> the TLT@low leg is worth p={p_mean:.3f} against its OWN parent, "
      f"before any search charge")

print("\n" + "=" * 78)
print("2. SEARCH COST")
print("=" * 78)
n_cells = len(grid)
fw_mean = 1 - (1 - p_mean) ** n_cells
fw_perf = 1 - (1 - p_perf) ** n_cells
print(f"  cells searched = {n_cells}")
print(f"  family-wise P(SOME cell has a subset mean this extreme) = {fw_mean:.3f}")
print(f"  family-wise P(SOME cell is perfect at this n)           = {fw_perf:.3f}")
print(f"  Bonferroni alpha for {n_cells} cells at 0.05 = {0.05/n_cells:.4f}; "
      f"the cell's own sign p is {sign_test(k, k):.4f}")
print(f"  VERDICT: sign p {sign_test(k,k):.4f} "
      f"{'CLEARS' if sign_test(k,k) < 0.05/n_cells else 'DOES NOT CLEAR'} "
      f"the Bonferroni bar for the search that found it.")

print("\n" + "=" * 78)
print("3. WHAT THE COMPLEMENT SAYS (the filter-that-does-not-filter test)")
print("=" * 78)
comp = declusters(IDX[(state & pr & ~mom).values & valid.values], H, IDX)
vc = ret.loc[comp].values
print(f"  parent               n={len(vpar):>3} mean {100*vpar.mean():+.3f}%")
print(f"  three-way (kept)     n={len(v3):>3} mean {100*obs:+.3f}%")
print(f"  complement (dropped) n={len(vc):>3} mean {100*vc.mean():+.3f}%")
print(f"  the leg drops {len(vc)} of {len(vpar)} episodes ({100*len(vc)/len(vpar):.0f}%) "
      f"and moves the complement only {100*(vc.mean()-vpar.mean()):+.3f}pp")
print(f"  i.e. the ENTIRE parent edge survives without the leg "
      f"({100*vc.mean():+.3f}% vs {100*vpar.mean():+.3f}%), which is the "
      f"signature of a lucky subset rather than a filter")
