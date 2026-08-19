"""C7c: the exact LIVE cell (dollar rank 14.3 AND gold already at rank 77),
plus the search charge on the grid the checker actually walked.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

px = close_panel(["DX-Y.NYB", "^TNX", "GLD"]).dropna(subset=["DX-Y.NYB", "^TNX", "GLD"])
dx, tnx, gld = px["DX-Y.NYB"], px["^TNX"], px["GLD"]
rk_tnx, rk_dx, rk_gld = pct_rank(tnx, 21), pct_rank(dx, 21), pct_rank(gld, 21)
r21_tnx = tnx.pct_change(21)
lvl = tnx - tnx.shift(21)
base = ((r21_tnx > 0) & (rk_tnx >= 65) & (rk_dx <= 20)).fillna(False)

def stat(mask, h, gap=21, label=""):
    ret = vehicle_ret(px, [("GLD", 1.0)], h)
    valid = px.index[ret.notna().values]
    idx = px.index[mask.reindex(px.index, fill_value=False).values & ret.notna().values]
    epi = declusters(pd.DatetimeIndex(idx), gap, valid)
    v = ret.loc[epi].values
    s = summarize(v, label)
    if s["n"]:
        s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        s["edge_pp"] = round(s["mean_pct"] - 100*ret.loc[valid].mean(), 3)
    return s, epi, v

print("=== (a) THE LIVE CELL ===")
print("live: DX rank 14.3 (<=15), TNX 21d level +0.108pt, GLD rank21 77.0, "
      "GLD 21d +8.42%")
for h in (1, 3, 5, 10):
    m = ((r21_tnx > 0) & (rk_tnx >= 65) & (rk_dx <= 20)
         & (rk_gld >= 75)).fillna(False)
    s, epi, v = stat(m, h, 21, f"h={h} joint + GLD rank>=75 (TODAY)")
    show([s])
    if h == 3:
        print("   episodes:", ", ".join(str(d.date()) for d in epi))
        print("   values:", [round(100*x, 2) for x in v])
# tighter live cell: gold strong AND dollar rank<=15
print("\n  tighter, matching today on both dials (DX rank<=15 AND GLD rank>=75):")
for h in (1, 3, 5):
    m = ((r21_tnx > 0) & (rk_tnx >= 65) & (rk_dx <= 15)
         & (rk_gld >= 75)).fillna(False)
    s, epi, v = stat(m, h, 21, f"h={h} DX<=15 & GLD>=75")
    show([s])

print("\n=== (b) does the cell just select gold bull markets? ===")
# GLD above its own 200d on trigger days vs base rate
sma200 = gld.rolling(200).mean()
above = (gld > sma200)
_, epi, _ = stat(base, 3, 21)
print("  GLD above its 200d on %.1f%% of trigger episodes vs %.1f%% base rate"
      % (100*above.reindex(epi).mean(), 100*above.dropna().mean()))
print("  TODAY: GLD %s its 200d (%.2f vs %.2f)"
      % ("above" if above.iloc[-1] else "below", gld.iloc[-1], sma200.iloc[-1]))
# residual: subtract the 200d-uptrend cell's own drift
ret3 = vehicle_ret(px, [("GLD", 1.0)], 3)
up = px.index[above.fillna(False).values & ret3.notna().values]
print("  control: GLD's h=3 drift on ALL days above its 200d = %+0.3f%% "
      "(N=%d) against the cell's %+0.3f%%"
      % (100*ret3.loc[up].mean(), len(up), 100*stat(base, 3, 21)[2].mean()))
upe = declusters(pd.DatetimeIndex(up), 21, px.index[ret3.notna().values])
print("  same control at EPISODE spacing (21td): %+0.3f%% (N=%d)"
      % (100*ret3.loc[upe].mean(), len(upe)))

print("\n=== (c) SEARCH CHARGE, rotation permutation ===")
grid = []
for ln in (10, 21, 42, 63):
    rt, rdk, rr = pct_rank(tnx, ln), pct_rank(dx, ln), tnx.pct_change(ln)
    for a_ in (55, 60, 65, 70, 75, 85):
        for b_ in (5, 10, 15, 20, 25, 30, 40):
            grid.append(((rr > 0) & (rt >= a_) & (rdk <= b_)).fillna(False).values)
grid = np.array(grid)
print("  masks walked: %d (x 4 horizons x 6 decluster gaps in round 2)" % len(grid))

def grid_ts(retv, validv, masks, gap=21):
    out = []
    idxall = px.index[validv]
    for m in masks:
        idx = px.index[m & validv]
        if len(idx) < 15:
            out.append(np.nan); continue
        e = declusters(pd.DatetimeIndex(idx), gap, idxall)
        vv = retv[px.index.get_indexer(e)]
        vv = vv[~np.isnan(vv)]
        if len(vv) < 15:
            out.append(np.nan); continue
        out.append(vv.mean()/(vv.std(ddof=1)/np.sqrt(len(vv))))
    return np.array(out)

r3 = vehicle_ret(px, [("GLD", 1.0)], 3)
obs = grid_ts(r3.values, r3.notna().values, grid)
pitched = 2.065
print("  pitched h=3 cell t=%.2f ranks %d of %d live cells; grid max %.2f"
      % (pitched, int((np.nan_to_num(obs, nan=-9) >= pitched).sum()),
         int(np.isfinite(obs).sum()), np.nanmax(obs)))
rng = np.random.default_rng(42)
n = len(px.index)
maxes = []
for _ in range(300):
    k = rng.integers(252, n - 252)
    rot = np.roll(r3.values, k)
    maxes.append(np.nanmax(grid_ts(rot, ~np.isnan(rot), grid)))
maxes = np.array(maxes)
print("  P(grid max t >= %.2f) under rotation = %.3f ; median grid max t = %.2f"
      % (pitched, (maxes >= pitched).mean(), np.median(maxes)))
