"""C8 round 2: the gap<=-25pp cell is the only positive rung of the grid and
today's -25.35pp sits just inside it. Probe it before crediting anything.

Round 1: the pitched cell (gap<=-20pp) is -0.014% eq-dollar / -0.019%
beta-neutral over 45 episodes, welch -0.19, 24-21. The grid rung at -25pp
reads +1.168% (N=20, t 1.99). That rung is the MAXIMUM of a 5-point grid I
built this morning, and its two neighbours are -0.014% (-20) and +0.539%
(-30), so it is non-monotone. Price it properly: beta-neutral, concentration,
era, gate attribution, and the multiplicity charge for the grid.
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
slv5 = pct_rank(px["SLV"], 5)
gld5 = pct_rank(px["GLD"], 5)
slv_dd = px["SLV"] / rolling_on_valid(px["SLV"], lambda x: x.rolling(252).max()) - 1.0
gld_dd = px["GLD"] / rolling_on_valid(px["GLD"], lambda x: x.rolling(252).max()) - 1.0
gap = slv_dd - gld_dd
d = px.pct_change().dropna()
B = float(np.polyfit(d["GLD"], d["SLV"], 1)[0])

thrust = ((slv5 >= 70) & (gld5 >= 70)).fillna(False)
cell = (thrust & (gap <= -0.25)).fillna(False)
r_ed = vehicle_ret(px, [("SLV", 1.0), ("GLD", -1.0)], H, 1)
r_bn = vehicle_ret(px, [("SLV", 1.0), ("GLD", -B)], H, 1)
r_out = fwd_lag(px["SLV"], H, 1)
epi = declusters(idx[(cell & r_ed.notna()).values], 5, idx)

print(f"today gap = {100*gap.loc[ASOF]:.2f}pp   beta = {B:.3f}")
print(f"gap<=-25pp cell: {len(epi)} episodes")
print("dates:", ", ".join(str(x.date()) for x in epi))
print()
for lbl, r in (("equal-dollar SLV-GLD", r_ed), (f"beta-neutral SLV-{B:.2f}GLD", r_bn),
               ("SLV outright", r_out)):
    v = r.loc[epi].values
    b = r.dropna()
    print(f"  {lbl:<26} mean {100*v.mean():+7.3f}%  excess {100*v.mean()-100*b.mean():+7.3f}%  "
          f"hit {100*(v>0).mean():5.1f}%  sign p {sign_test(int((v>0).sum()), len(v)):.4f}  "
          f"boot P(<=0) {bootstrap_p_le0(v):.3f}  worst {100*v.min():+.2f}%")

v = r_bn.loc[epi].values
print(f"\n  concentration (beta-neutral): {cluster_note(epi, v)}")
print(f"  concentration (eq-dollar):    {cluster_note(epi, r_ed.loc[epi].values)}")
yrs = pd.DatetimeIndex(epi).year
print("  episodes by year:", dict(pd.Series(1, index=yrs).groupby(level=0).sum()))
for y in sorted(set(yrs)):
    sub = v[yrs == y]
    print(f"     drop {y}: remaining N={len(v)-len(sub)}  mean "
          f"{100*np.delete(v, np.where(yrs == y)).mean():+7.3f}%")

print("\n  gate attribution at gap<=-25pp:")
for lbl, m in (("gap<=-25 ONLY (no thrust gate)", (gap <= -0.25).fillna(False)),
               ("gap<=-25 AND joint thrust", cell)):
    e = declusters(idx[(m & r_bn.notna()).values], 5, idx)
    x = r_bn.loc[e].values
    y = r_ed.loc[e].values
    print(f"    {lbl:<32} N={len(e):<4} beta-neutral {100*x.mean():+7.3f}% hit {100*(x>0).mean():5.1f}% "
          f"| eq-dollar {100*y.mean():+7.3f}% signp {sign_test(int((y>0).sum()), len(y)):.4f}")

print("\n  fine grid around -25pp (eq-dollar / beta-neutral, episodes):")
for g in (-0.21, -0.22, -0.23, -0.24, -0.25, -0.26, -0.27, -0.28):
    m = (thrust & (gap <= g)).fillna(False)
    e = declusters(idx[(m & r_ed.notna()).values], 5, idx)
    if len(e) < 3:
        print(f"    gap<={100*g:.0f}pp  N={len(e)}")
        continue
    print(f"    gap<={100*g:.0f}pp  N={len(e):<4} eq {100*r_ed.loc[e].mean():+7.3f}%  "
          f"bn {100*r_bn.loc[e].mean():+7.3f}%  hit {100*(r_ed.loc[e]>0).mean():5.1f}%")

print("\n  horizon scan on the -25pp cell:")
show(horizon_scan(px, epi, [("SLV", 1.0), ("GLD", -B)], hs=(1, 2, 3, 5, 10), min_gap=5),
     f"beta-neutral SLV - {B:.2f}*GLD, gap<=-25pp")
