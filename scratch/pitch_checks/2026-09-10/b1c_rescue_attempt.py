"""B1 round 2b -- an honest attempt to RESCUE the dollar cell.

The kill rests on the 2006 cluster (7 episodes, -0.769%, record 2-5) that UUP's
2007 inception hides. Before accepting it, ask three fair questions:
  1. Is the 2006 state real, or a DX-Y.NYB data artefact?
  2. Is there an EX-ANTE feature that separates 2006 from 2013/2018/2022?
     Any such feature is a rescue grid I WALK, so it is charged.
  3. What is today's cell most like -- 2006 or 2013/2018/2022?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change

pd.set_option("display.width", 210)

TICKS = ["SPY", "UUP", "DX-Y.NYB", "^TNX", "TLT", "IEF"]
raw = load_prices(TICKS)
cal = raw["SPY"].index
px = pd.DataFrame({t: raw[t]["Close"].reindex(cal) for t in TICKS})
dx, tnx = px["DX-Y.NYB"], px["^TNX"]


def at_high(s, n):
    hi = rolling_on_valid(s, lambda x: x.rolling(n).max())
    return (s >= hi - 1e-9) & s.notna() & hi.notna()


dx_r63 = pct_rank(dx, 63)
TRIG = ((dx_r63 <= 20) & at_high(tnx, 252)).fillna(False)
H = 5
r_dx = vehicle_ret(px, [("DX-Y.NYB", 1.0)], H, 1)
alld = px.index[TRIG.values & r_dx.notna().values]
epi = declusters(alld, H, px.index)

print("=" * 78)
print("1. IS THE 2006 STATE REAL? state descriptors on every episode")
print("=" * 78)
tnx_lvl = tnx
dx_lvl = dx
dx_63r = _valid_pct_change(dx, 63)
tnx_63chg = tnx - tnx.shift(63)
print(f"{'date':12s} {'DXlvl':>7s} {'DXr63':>6s} {'DX63d%':>7s} {'TNX':>6s} "
      f"{'TNX63dchg':>10s} {'fwd5d':>8s}  era")
for d in epi:
    print(f"{str(d.date()):12s} {dx_lvl.get(d, np.nan):7.2f} {dx_r63.get(d, np.nan):6.1f} "
          f"{100*dx_63r.get(d, np.nan):7.2f} {tnx_lvl.get(d, np.nan):6.3f} "
          f"{tnx_63chg.get(d, np.nan):+10.3f} {100*r_dx.get(d, np.nan):+8.3f}%  "
          f"{'2006' if d.year == 2006 else 'later'}")
print(f"\nTODAY  {dx_lvl.iloc[-1]:7.2f} {dx_r63.iloc[-1]:6.1f} "
      f"{100*dx_63r.iloc[-1]:7.2f} {tnx_lvl.iloc[-1]:6.3f} {tnx_63chg.iloc[-1]:+10.3f}")

print("\n" + "=" * 78)
print("2. RESCUE GRID (charged): can any EX-ANTE feature exclude 2006?")
print("=" * 78)
v_all = r_dx.loc[epi].values
feats = {
    "TNX level >= 3.0": (tnx >= 3.0),
    "TNX level <  3.0": (tnx < 3.0),
    "TNX 63d chg >= +0.40": (tnx_63chg >= 0.40),
    "TNX 63d chg <  +0.40": (tnx_63chg < 0.40),
    "DX 63d ret <= -2%": (dx_63r <= -0.02),
    "DX 63d ret >  -2%": (dx_63r > -0.02),
    "DX level >= 95": (dx >= 95),
    "DX level <  95": (dx < 95),
    "non-midterm year": pd.Series([d.year % 4 != 2 for d in px.index], index=px.index),
    "MIDTERM year (today)": pd.Series([d.year % 4 == 2 for d in px.index], index=px.index),
}
rows = []
for lbl, f in feats.items():
    m = (TRIG & f.reindex(px.index).fillna(False)).astype(bool)
    d_ = px.index[m.values & r_dx.notna().values]
    if len(d_) == 0:
        rows.append({"label": lbl, "n": 0})
        continue
    e = declusters(d_, H, px.index)
    v = r_dx.loc[e].values
    s = summarize(v, lbl)
    w = int((v > 0).sum())
    s["record"] = f"{w}-{len(v)-w}"
    s["n2006"] = int(sum(1 for x in e if x.year == 2006))
    s["live_today"] = bool(f.reindex(px.index).fillna(False).iloc[-1])
    rows.append(s)
show(rows, "rescue features x the trigger (DX vehicle, all history)")
print("\n  NOTE: 10 features walked here = a rescue grid. Any survivor must be")
print("  charged max-of-10, and none of them was pre-specified this morning.")

print("\n" + "=" * 78)
print("3. WHAT IS TODAY MOST LIKE? nearest-neighbour on the state descriptors")
print("=" * 78)
cols = {"dx_r63": dx_r63, "tnx": tnx, "tnx63chg": tnx_63chg, "dx63r": 100 * dx_63r}
X = pd.DataFrame({k: v for k, v in cols.items()})
today = X.iloc[-1]
sub = X.loc[epi]
sd = X.dropna().std()
dist = ((sub - today) / sd).pow(2).sum(axis=1).pow(0.5)
near = dist.sort_values().head(6)
print("  six nearest historical episodes (standardised distance):")
for d, x in near.items():
    print(f"    {d.date()}  dist={x:.2f}  fwd5d={100*r_dx.get(d, np.nan):+7.3f}%"
          f"  {'2006 CLUSTER' if d.year == 2006 else ''}")
print(f"\n  mean fwd 5d over those 6 nearest = "
      f"{100*r_dx.loc[near.index].mean():+.3f}%  "
      f"record {(r_dx.loc[near.index]>0).sum()}-{(r_dx.loc[near.index]<=0).sum()}")

print("\n" + "=" * 78)
print("4. WHAT WOULD TURN IT ON (the number to park)")
print("=" * 78)
print("  The UUP-era record is 6-0 (+0.870%, sign p 0.0156). The full-underlying")
print("  record is 8-5 (-0.007%, sign p 0.2905). For the FULL-history cell to")
print("  reach sign p <= 0.05 it needs, from 13 episodes:")
for extra in range(1, 12):
    w, n = 8 + extra, 13 + extra
    p = sign_test(w, n)
    if p <= 0.05:
        print(f"    +{extra} consecutive new WINS -> {w}-{n-w}, sign p {p:.4f}  <-- first pass")
        break
    print(f"    +{extra} consecutive new wins -> {w}-{n-w}, sign p {p:.4f}")
print("\n  The trigger has fired in 5 distinct calendar clusters in 26 years, so")
print("  that is roughly a decade of waiting. Parked, not pitched.")
