"""K3 c8: the only positive LIVE neighbour in k3_c8_neigh.py is the 10-day
lookback (IEF r10 <= 10 & TLT r10 >= 20, +11.8 bp h=5 on 28). Is it a real
object or a grid draw? Era, tdom control, cost multiple, overlap with the
^TNX-252-max parent (watchlist 19), and a charge for the 36-rung x 3-horizon
grid it was read from (block-shifted null on the curve return)."""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import rolling_on_valid

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
BAR = pd.Timestamp("2026-09-16")
px = close_panel(["IEF", "TLT", "^TNX"])
px = px[px.index <= BAR].dropna(subset=["IEF", "TLT"])
D = px.index
LEGS = [("IEF", 1.0), ("TLT", -0.523)]
COST = 4.423
tdom = pd.Series(D, index=D).groupby([D.year, D.month]).rank().astype(int)
tdom = pd.Series(tdom.values, index=D)
tnx = px["^TNX"]
hi = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
PB = ((tnx / hi - 1.0) >= -0.0025).reindex(D, fill_value=False)

ri = pct_rank(px["IEF"], 10)
rt = pct_rank(px["TLT"], 10)
CELL = (ri <= 10) & (rt >= 20)
print(f"live {bool(CELL.iloc[-1])}  IEF r10 {ri.iloc[-1]:.1f} TLT r10 {rt.iloc[-1]:.1f}")
for h in (3, 5, 8, 10):
    ret = vehicle_ret(px, LEGS, h, 1)
    valid = ret.notna()
    sig = D[(CELL & valid).values]
    e = declusters(sig, max(h, 5), D)
    v = ret.loc[e].values
    base = ret[valid & ~CELL]
    tm = base.groupby(tdom.reindex(base.index)).mean()
    tc = np.nanmean(tm.reindex(tdom.loc[e]).values)
    w = int((v > 0).sum())
    print(f"\nh={h}: n={len(v)} {1e4*v.mean():+.1f} bp  rec {w}-{len(v)-w} "
          f"sign p {sign_test(w, len(v)):.3f}  all-days {1e4*base.mean():+.1f}  "
          f"tdom ctl {1e4*tc:+.1f}  edge {1e4*(v.mean()-tc):+.1f}  "
          f"cost {1e4*v.mean()/COST:.2f}x  boot P<=0 {bootstrap_p_le0(v):.3f}")
    y = e.year
    for lab, m in [("pre-2018", y < 2018), ("2018+", y >= 2018),
                   ("2022-23", (y >= 2022) & (y <= 2023)),
                   ("2018+ ex22-23", (y >= 2018) & ~((y >= 2022) & (y <= 2023))),
                   ("midterm", y % 4 == 2), ("& PB (TNX max)", PB.loc[e].values),
                   ("& ~PB", ~PB.loc[e].values)]:
        vv = v[m]
        if len(vv):
            ww = int((vv > 0).sum())
            print(f"   {lab:15s} n={len(vv):2d} {1e4*vv.mean():+7.1f} bp  {ww}-{len(vv)-ww}")
    print("  ", cluster_note(e, v))
    vs = np.sort(v)[::-1]
    print(f"   drop-best {1e4*vs[1:].mean():+.1f} bp  drop-best-2 {1e4*vs[2:].mean():+.1f} bp")

# grid charge: 36 rungs x h in (3,5,10); null = circular shift of the curve
# return series against the fixed masks, max mean across rungs/horizons
rng = np.random.default_rng(7)
masks = []
for n in (3, 5, 10):
    a_ = pct_rank(px["IEF"], n)
    b_ = pct_rank(px["TLT"], n)
    for a in (2, 5, 10):
        for b in (10, 20, 30, 50):
            masks.append(((a_ <= a) & (b_ >= b)).values)
R = {h: vehicle_ret(px, LEGS, h, 1).values for h in (3, 5, 10)}
pos = np.arange(len(D))


def epi_pos(m, h):
    kp, last = [], -10 ** 9
    for p in pos[m & ~np.isnan(R[h])]:
        if p - last >= max(h, 5):
            kp.append(p)
            last = p
    return np.array(kp, dtype=int)


EP = {(i, h): epi_pos(m, h) for i, m in enumerate(masks) for h in (3, 5, 10)}
obs = 1e4 * np.nanmean(R[5][EP[(35 - 4 + 1, 5)]]) if False else None
# locate the live 10d rung index: n=10 block starts at 24; a=10 -> +8; b=20 -> +1
idx = 24 + 8 + 1
obs = 1e4 * np.nanmean(R[5][EP[(idx, 5)]])
print(f"\nobserved rung idx {idx} h=5 {obs:+.1f} bp (n={len(EP[(idx, 5)])})")
NB = 1000
mx = np.empty(NB)
L = len(D)
for k in range(NB):
    s = rng.integers(252, L - 252)
    best = -1e9
    for (i, h), ep in EP.items():
        if len(ep) < 5:
            continue
        rr = np.roll(R[h], s)[ep]
        mval = 1e4 * np.nanmean(rr)
        best = max(best, mval)
    mx[k] = best
print(f"grid charge (max over rungs with n>=5 x h 3/5/10): P(null max >= obs) "
      f"{(mx >= obs).mean():.3f}; null max median {np.median(mx):+.1f} bp")
