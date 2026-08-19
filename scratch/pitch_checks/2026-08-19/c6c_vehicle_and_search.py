"""C6c: (a) matched-episode vehicle test, spot DXY vs the carry-bearing ETF;
(b) era x rate-regime intersection; (c) the LIVE cell; (d) search charge.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

px = close_panel(["DX-Y.NYB", "UUP", "^TNX"]).dropna(subset=["DX-Y.NYB", "^TNX"])
dx, tnx, uup = px["DX-Y.NYB"], px["^TNX"], px["UUP"]
rk_tnx, rk_dx = pct_rank(tnx, 21), pct_rank(dx, 21)
r21_tnx, r21_dx = tnx.pct_change(21), dx.pct_change(21)
base = ((r21_tnx > 0) & (rk_tnx >= 65) & (rk_dx <= 20)).fillna(False)
H = 5

# ---------- (a) matched vehicle test ----------
print("=== (a) MATCHED-EPISODE VEHICLE TEST (short dollar, h=5) ===")
rd = vehicle_ret(px, [("DX-Y.NYB", -1.0)], H)
ru = vehicle_ret(px, [("UUP", -1.0)], H)
both = rd.notna() & ru.notna()
idx = px.index[base.values & both.values]
epi = declusters(idx, 21, px.index[both.values])
a, b = rd.loc[epi].values, ru.loc[epi].values
print("  matched episodes N=%d (UUP starts %s)" % (len(epi), uup.dropna().index[0].date()))
show([summarize(a, "short DXY SPOT index"), summarize(b, "short UUP (carry-bearing)")])
d = a - b
print("  spot MINUS carry-bearing vehicle = %+0.3fpp (%.1f bps), t=%.2f, "
      "vehicle agrees on sign in %.1f%% of episodes"
      % (100*d.mean(), 10000*d.mean(), d.mean()/(d.std(ddof=1)/np.sqrt(len(d))),
         100*((a > 0) == (b > 0)).mean()))
print("  -> the spot index carries NO interest differential; a short DX futures "
      "position pays it. Quoting the spot number overstates the tradeable edge "
      "by the gap above.")
# same on all days, to size the structural carry drag
allb = px.index[both.values]
print("  all-days control: short DXY spot %+0.4f%% vs short UUP %+0.4f%% "
      "(structural gap %.1f bps / 5td)"
      % (100*rd.loc[allb].mean(), 100*ru.loc[allb].mean(),
         10000*(rd.loc[allb].mean()-ru.loc[allb].mean())))

# ---------- (b) era x rate-regime intersection ----------
print("\n=== (b) era x rate-regime intersection (short DXY spot, h=5) ===")
v = rd.loc[epi] if False else None
idx_all = px.index[base.values & rd.notna().values]
epi_all = declusters(idx_all, 21, px.index[rd.notna().values])
va = rd.loc[epi_all].values
sec = (tnx - tnx.shift(252)).reindex(epi_all).values
yr = pd.DatetimeIndex(epi_all).year.values
rows = []
for lbl, m in [("rising yields & pre-2013", (sec >= 0) & (yr < 2013)),
               ("rising yields & 2013+", (sec >= 0) & (yr >= 2013)),
               ("rising yields & 2018+", (sec >= 0) & (yr >= 2018)),
               ("falling yields & 2013+", (sec < 0) & (yr >= 2013))]:
    s = summarize(va[m], lbl)
    if s["n"]:
        s["bps"] = round(100*s["mean_pct"], 1)
        s["sign_p"] = round(sign_test(int((va[m] > 0).sum()), int(m.sum())), 4)
    rows.append(s)
show(rows)
print("  episode dates in 'rising yields & 2013+':",
      ", ".join(str(d.date()) for d in pd.DatetimeIndex(epi_all)[(sec >= 0) & (yr >= 2013)]))

# ---------- (c) the LIVE cell ----------
print("\n=== (c) the LIVE cell: DX rank<=15 (today 14.3) AND TNX 21d level "
      "change <=+0.15pt (today +0.108) ===")
lvl = tnx - tnx.shift(21)
live = ((r21_tnx > 0) & (rk_tnx >= 65) & (rk_dx <= 15) & (lvl <= 0.15)).fillna(False)
li = px.index[live.values & rd.notna().values]
le = declusters(li, 21, px.index[rd.notna().values])
lv = rd.loc[le].values
s = summarize(lv, "LIVE cell, short DXY spot h=5")
s["bps"] = round(100*s["mean_pct"], 1)
s["sign_p"] = round(sign_test(int((lv > 0).sum()), len(lv)), 4)
show([s])
print("  episodes:", ", ".join(str(d.date()) for d in le))

# ---------- (d) search charge ----------
print("\n=== (d) SEARCH CHARGE, rotation permutation (preserves calendar + "
      "autocorrelation) ===")
grid = []
for ln in (10, 21, 42, 63):
    rt, rdk, rr = pct_rank(tnx, ln), pct_rank(dx, ln), tnx.pct_change(ln)
    for a_ in (55, 60, 65, 70, 75, 85):
        for b_ in (5, 10, 15, 20, 25, 30, 40):
            grid.append(((rr > 0) & (rt >= a_) & (rdk <= b_)).fillna(False).values)
grid = np.array(grid)
print("  grid cells walked: %d masks x 3 horizons = %d" % (len(grid), 3*len(grid)))

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

rng = np.random.default_rng(42)
obs_t = []
for h in (3, 5, 10):
    r = vehicle_ret(px, [("DX-Y.NYB", -1.0)], h)
    obs_t.append(grid_ts(r.values, r.notna().values, grid))
obs_t = np.concatenate(obs_t)
pitched = 1.734
print("  pitched cell t=%.2f ranks %d of %d live cells (max in grid %.2f)"
      % (pitched, int((np.nan_to_num(obs_t, nan=-9) >= pitched).sum()),
         int(np.isfinite(obs_t).sum()), np.nanmax(obs_t)))
r5 = vehicle_ret(px, [("DX-Y.NYB", -1.0)], 5)
n = len(px.index)
maxes = []
for _ in range(300):
    k = rng.integers(252, n - 252)
    rot = np.roll(r5.values, k)
    valid = ~np.isnan(rot)
    ts = grid_ts(rot, valid, grid)
    maxes.append(np.nanmax(ts))
maxes = np.array(maxes)
print("  rotation permutation (300 draws, h=5 grid of %d): "
      "P(grid max t >= %.2f) = %.3f ; median grid max t = %.2f"
      % (len(grid), pitched, (maxes >= pitched).mean(), np.median(maxes)))
