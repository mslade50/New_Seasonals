"""A5 ROUND 2 -- the label is fake, and the only live-relevant sub-cell.

Round 1 (k1_vix_thrust_contango.py), pre-specified rung rank>=90 (the live
reading's own PIT percentile is 90.1) with VIX/VIX3M <= 0.95:
  h=1  episodes n=90 +0.124%, edge over same-span drift +0.075pp, welch t +0.76
  h=3  edge -0.017pp
  h=5  edge +0.179pp
  h=10 edge +0.382pp
and the registry baseline re-derived here: plain SPY 5d <= -1% pays edge
+0.059 / +0.179 / +0.261 / +0.189pp on n=1639 day-level at h=1/3/5/10, i.e. it
BEATS the labelled cell at h=1, 3 and 5 on eighteen times the sample. Inside
the dip, the thrust contributes +0.014pp (h=1), -0.107pp (h=5), +0.002pp (h=10).

Round 2:
  1. definition fragility, already visible: the rank ladder is non-monotone and
     rank>=98 is NEGATIVE. Quantify the dose response properly.
  2. the LIVE configuration is thrust WITHOUT a dip (SPY 5d is -0.96%, a hair
     above the -1% line). That 2x2 cell was the only attractive box. It is a
     third gate found after the 2x2, so it is charged, declustered and
     era-split before anything is claimed for it.
  3. dip-threshold sensitivity: -0.96% sits between two neighbouring rungs.
  4. cost at the horizon where the contango gate actually earns its place.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = pd.Timestamp("2026-09-10")
raw = load_prices(["SPY", "^VIX", "^VIX3M"])
SP = raw["SPY"].index
PX = pd.DataFrame({t: raw[t]["Close"].reindex(SP).ffill() for t in raw})
PX = PX.rename(columns={"^VIX": "VIX", "^VIX3M": "VIX3M"})
vix, v3m = PX["VIX"], PX["VIX3M"]
ratio = vix / v3m
v5 = vix / vix.shift(5) - 1.0
spy5 = PX["SPY"] / PX["SPY"].shift(5) - 1.0
rank5 = rolling_on_valid(v5, lambda x: x.rolling(252).rank(pct=True) * 100)
CELL = ((rank5 >= 90) & (ratio <= 0.95)).fillna(False)

print("=" * 78)
print("1. DOSE RESPONSE -- a thrust claim says bigger thrust, bigger effect")
print("=" * 78)
for h in (1, 5, 10):
    r = fwd_lag(PX["SPY"], h, 1)
    valid = r.dropna().index
    base = r.loc[valid].mean()
    print(f"  h={h} (all-days {100*base:+.3f}%) -- contango held at <=0.95, "
          f"^VIX 5d rank BANDS:")
    for lo, hi in ((80, 85), (85, 90), (90, 95), (95, 98), (98, 101)):
        m = ((rank5 >= lo) & (rank5 < hi) & (ratio <= 0.95)).fillna(False)
        d = pd.DatetimeIndex(SP[m.values]).intersection(valid)
        e = declusters(d, 10, valid)
        if len(e) == 0:
            continue
        v = r.loc[e].values
        print(f"    rank [{lo},{hi}) : epi n={len(e):3d} {100*np.mean(v):+.3f}% "
              f"edge {100*(np.mean(v)-base):+.3f}pp hit {100*(v>0).mean():.0f}%")
    print(f"    RAW 5d magnitude bands (contango <=0.95):")
    for lo, hi in ((0.00, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 10.0)):
        m = ((v5 >= lo) & (v5 < hi) & (ratio <= 0.95)).fillna(False)
        d = pd.DatetimeIndex(SP[m.values]).intersection(valid)
        e = declusters(d, 10, valid)
        if len(e) == 0:
            continue
        v = r.loc[e].values
        print(f"    5d in [{100*lo:4.0f}%,{100*hi:4.0f}%): epi n={len(e):3d} "
              f"{100*np.mean(v):+.3f}% edge {100*(np.mean(v)-base):+.3f}pp "
              f"hit {100*(v>0).mean():.0f}%")

print("\n" + "=" * 78)
print("2. THE LIVE SUB-CELL: thrust + contango + NO dip (SPY 5d > -1%)")
print("=" * 78)
NODIP = (spy5 > -0.01).fillna(False)
LIVE = (CELL & NODIP)
print(f"  fires today? {bool(LIVE.loc[ASOF])}  (SPY 5d = {100*spy5.loc[ASOF]:+.2f}%)")
for h in (1, 3, 5, 10):
    r = fwd_lag(PX["SPY"], h, 1)
    valid = r.dropna().index
    base = r.loc[valid].mean()
    d = pd.DatetimeIndex(SP[LIVE.values]).intersection(valid)
    e = declusters(d, 10, valid)
    v = r.loc[e]
    w = int((v > 0).sum())
    # the correct control for a NO-DIP cell is NO-DIP days, not all days
    nd = pd.DatetimeIndex(SP[(NODIP & ~CELL).values]).intersection(valid)
    nde = declusters(nd, 10, valid)
    print(f"  h={h:2d}: epi n={len(v):3d} {100*v.mean():+.3f}% hit "
          f"{100*(v>0).mean():.1f}% sign p {sign_test(w, len(v)):.4f} | "
          f"vs NO-DIP no-thrust n={len(nde)} {100*r.loc[nde].mean():+.3f}% "
          f"-> thrust adds {100*(v.mean()-r.loc[nde].mean()):+.3f}pp "
          f"| all-days {100*base:+.3f}%")
    print(f"       {cluster_note(v.index, v.values)}")
    show(era_split(pd.DatetimeIndex(v.index), v.values), f"    h={h} era")

print("\n" + "=" * 78)
print("3. DIP-THRESHOLD SENSITIVITY -- SPY 5d is -0.96%, between two rungs")
print("=" * 78)
for h in (1, 5, 10):
    r = fwd_lag(PX["SPY"], h, 1)
    valid = r.dropna().index
    base = r.loc[valid].mean()
    parts = []
    for thr in (-0.02, -0.015, -0.01, -0.005, 0.0, 0.005):
        m = (CELL & (spy5 > thr)).fillna(False)
        d = pd.DatetimeIndex(SP[m.values]).intersection(valid)
        e = declusters(d, 10, valid)
        if len(e) < 3:
            parts.append(f"5d>{100*thr:+.1f}%: n={len(e)} -")
            continue
        v = r.loc[e]
        parts.append(f"5d>{100*thr:+.1f}%: n={len(e):3d} "
                     f"{100*v.mean():+.3f}% ({100*(v.mean()-base):+.3f})")
    print(f"  h={h:2d}  " + " | ".join(parts))

print("\n" + "=" * 78)
print("4. CHARGE THE SUB-CELL -- it was found in a 2x2 I ran, so it is a search")
print("=" * 78)
h = 10
r = fwd_lag(PX["SPY"], h, 1)
valid = r.dropna().index
d = pd.DatetimeIndex(SP[LIVE.values]).intersection(valid)
e = declusters(d, 10, valid)
obs = float(r.loc[e].mean())
base = float(r.loc[valid].mean())
print(f"  defended: h={h} thrust x contango x no-dip, n={len(e)}, "
      f"{100*obs:+.3f}%, edge {100*(obs-base):+.3f}pp")
rng = np.random.default_rng(3)
# grid actually walked to get here: 5 rank thresholds x 4 curve rungs x
# 4 horizons x 2 dip sides = 160 cells
K = 160
null_max = []
pool = r.loc[valid].values
for _ in range(3000):
    best = max(rng.choice(pool, size=len(e), replace=False).mean() - base
               for _ in range(20))
    null_max.append(best)
null_max = np.array(null_max)
# scale a 20-draw max up to K draws via the order statistic relation
print(f"  null-max over 20 draws: median {100*np.median(null_max):+.3f}pp, "
      f"95th {100*np.quantile(null_max, 0.95):+.3f}pp")
print(f"  P(20-cell no-effect grid beats the defended edge) = "
      f"{(null_max >= obs-base).mean():.4f}   "
      f"(the walk was ~{K} cells, so this is a LOWER bound on the charge)")

print("\n" + "=" * 78)
print("5. COST AT EACH HORIZON, edge basis, SPY round trip ~2 bps")
print("=" * 78)
for h in (1, 3, 5, 10):
    r = fwd_lag(PX["SPY"], h, 1)
    valid = r.dropna().index
    base = r.loc[valid].mean()
    e = declusters(pd.DatetimeIndex(SP[CELL.values]).intersection(valid), 10, valid)
    edge_bps = 100 * 100 * (r.loc[e].mean() - base)
    dipd = pd.DatetimeIndex(SP[(spy5 <= -0.01).fillna(False).values]).intersection(valid)
    dip_bps = 100 * 100 * (r.loc[dipd].mean() - base)
    print(f"  h={h:2d}: cell edge {edge_bps:+.1f} bps = {abs(edge_bps)/2:.1f}x "
          f"a 2 bps round trip  |  plain SPY 5d<=-1% edge {dip_bps:+.1f} bps "
          f"= {abs(dip_bps)/2:.1f}x, n={len(dipd)}")
