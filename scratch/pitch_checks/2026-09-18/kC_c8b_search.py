"""c8 round 2: the search charge, priced on THIS cell's statistic.

The [1.0,1.5) band came out of 2026-08-20's walked 125-cell grid (5 thrust
rungs x 5 distance-to-low rungs x 5 horizons, |t| so both signs are charged).
Here: rotate the RETURN series (masks fixed), take the grid max |t| each draw,
and compare the null distribution with the OBSERVED episode |t| of the selected
cell (long TLT, [1.0,1.5) within 4% of the low, h=2, gap 10). Also an extended
grid that adds the adjacent BANDS (the object that was actually selected is a
band, not a >= rung), and the lower-bound neighbour row.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kC_common import *  # noqa

px = panel(["TLT"], "TLT")
tlt = px["TLT"]
d = px.index
d1 = dret(tlt)
dist = tlt / tlt.rolling(252).min() - 1.0
L = [("TLT", 1.0)]
HS = [1, 2, 3, 5, 10]
LOW = [0.02, 0.03, 0.04, 0.06, 0.08]
THR = [0.010, 0.0125, 0.015, 0.0175, 0.020]
BANDS = [(0.010, 0.0125), (0.0125, 0.015), (0.015, 0.0175), (0.0175, 0.020), (0.010, 0.015), (0.015, 0.020)]


def m_of(lo, hi, low):
    return ((d1 >= lo) & (d1 < hi) & (dist <= low)).fillna(False).values


grid125 = {(t, 9.0, lw): m_of(t, 9.0, lw) for t in THR for lw in LOW}
gridB = dict(grid125)
gridB.update({(a, b, lw): m_of(a, b, lw) for a, b in BANDS for lw in LOW})
RET = {h: vehicle_ret(px, L, h).values for h in HS}


def tstat(arr, mask, gap=10):
    rr = pd.Series(arr, index=d)
    valid = rr.notna().values
    e = declusters(d[mask & valid], gap, d)
    if len(e) < 8:
        return np.nan
    v = rr.loc[e].values
    sd = v.std(ddof=1)
    return abs(v.mean() / (sd / np.sqrt(len(v)))) if sd > 0 else np.nan


def grid_max(shift, grid):
    best = 0.0
    for h in HS:
        arr = np.roll(RET[h], shift)
        for m in grid.values():
            t = tstat(arr, m)
            if not np.isnan(t):
                best = max(best, t)
    return best


cell_m = m_of(0.010, 0.015, 0.04)
obs_cell = {h: tstat(RET[h], cell_m) for h in HS}
print("observed |t| of THE SELECTED CELL by horizon:", {h: round(v, 3) for h, v in obs_cell.items()})
obs_g125 = grid_max(0, grid125)
obs_gB = grid_max(0, gridB)
print(f"observed grid max |t|: 125-cell {obs_g125:.2f}   extended ({len(gridB)*len(HS)}-cell) {obs_gB:.2f}")

rng = np.random.default_rng(42)
shifts = rng.integers(60, len(d) - 60, size=150)
n125 = np.array([grid_max(int(s), grid125) for s in shifts])
nB = np.array([grid_max(int(s), gridB) for s in shifts])
sel = obs_cell[2]
for lbl, nul in [("125-cell grid", n125), (f"{len(gridB)*len(HS)}-cell grid (+bands)", nB)]:
    print(f"\n{lbl}: null grid max |t| mean {nul.mean():.2f} p50 {np.median(nul):.2f} p90 {np.percentile(nul, 90):.2f} max {nul.max():.2f}")
    print(f"  P(null grid max |t| >= SELECTED CELL h=2 |t| {sel:.3f}) = {(nul >= sel).mean():.3f}   (150 rotations)")
    print(f"  P(null grid max |t| >= best-h selected-cell |t| {max(obs_cell.values()):.3f}) = {(nul >= max(obs_cell.values())).mean():.3f}")
