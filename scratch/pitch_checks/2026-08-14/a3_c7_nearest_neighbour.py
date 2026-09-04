"""C7 round 1 — nearest-neighbour tapes to 2026-08-13 and what the next
week did.

Feature vector (all computable at the close, no lookahead):
  f1 ^VIX LEVEL percentile (252d)          today   5.6
  f2 ^SKEW LEVEL percentile (252d)         today   2.0
  f3 SPY distance to its 52w high (%)      today  +0.00
  f4 SPY extension above its 200d SMA (%)  today +10.66
  f5 breadth: fraction of the 9 original sector SPDRs above their own 200d

Each feature is z-scored over full history, distance is Euclidean in z-space,
the K closest days are declustered to episodes and their forward distribution
is compared to the unconditional one at h=1..10 (entry lag=1).

Stated kill criteria (from the assignment): indistinguishable from
unconditional, or the neighbours collapse to one episode.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import (close_panel, declusters, show, sign_test,  # noqa: E402
                       summarize, vehicle_ret, horizon_scan, bootstrap_p_le0)

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
px = close_panel(["SPY", "^VIX", "^SKEW"] + SECTORS).dropna()
print(f"panel {px.index[0].date()} .. {px.index[-1].date()}  n={len(px)}")


def lvl_pctile(s, lb=252):
    return s.rolling(lb).rank(pct=True) * 100.0


f1 = lvl_pctile(px["^VIX"])
f2 = lvl_pctile(px["^SKEW"])
f3 = 100 * (px["SPY"] / px["SPY"].rolling(252).max() - 1.0)
f4 = 100 * (px["SPY"] / px["SPY"].rolling(200).mean() - 1.0)
above = pd.DataFrame({s: (px[s] > px[s].rolling(200).mean()).astype(float)
                      for s in SECTORS})
f5 = 100 * above.mean(axis=1)

F = pd.DataFrame({"vix_lvl_pct": f1, "skew_lvl_pct": f2, "spy_d52wh": f3,
                  "spy_ext200": f4, "breadth_pct": f5}).dropna()
print(f"feature frame {F.index[0].date()} .. {F.index[-1].date()}  n={len(F)}")

# --- UNITS ASSERTION -------------------------------------------------------
_r = px["SPY"].pct_change().dropna()
assert abs(_r).max() < 0.5, "returns must be FRACTIONS before summarize()"

TODAY = F.iloc[-1]
print("\ntoday's feature vector (2026-08-13):")
print(TODAY.round(2).to_string())

Z = (F - F.mean()) / F.std()
zt = Z.iloc[-1]
dist = np.sqrt(((Z - zt) ** 2).sum(axis=1))
dist = dist.iloc[:-1]                       # exclude today itself
dist = dist[dist.index < pd.Timestamp("2026-01-01")]   # exclude the live year

print("\nz-distance summary: min %.3f  median %.3f  max %.3f"
      % (dist.min(), dist.median(), dist.max()))

for K in (20, 40, 80, 150):
    nb = dist.nsmallest(K).index.sort_values()
    epi = declusters(pd.DatetimeIndex(nb), 21, px.index)
    yrs = pd.Series(1, index=nb).groupby(nb.year).sum()
    print(f"\n{'='*78}\nK={K} nearest days -> {len(epi)} episodes (min_gap 21td)"
          f"\n  years: {dict(yrs)}"
          f"\n  max single-year share: {100*yrs.max()/K:.0f}%"
          f"\n  worst distance kept: {dist.nsmallest(K).max():.3f}"
          f"\n{'='*78}")
    print("  episodes:", ", ".join(str(d.date()) for d in epi))

    rows = horizon_scan(px, epi, [("SPY", 1.0)], hs=(1, 2, 3, 5, 7, 10),
                        lag=1, min_gap=21)
    show(rows, f"K={K} episode-level horizon scan vs unconditional")

    # significance vs unconditional at each horizon
    print("  welch t vs all-days + sign test:")
    for h in (1, 2, 3, 5, 7, 10):
        ret = vehicle_ret(px, [("SPY", 1.0)], h, 1)
        e = pd.DatetimeIndex(epi).intersection(ret.dropna().index)
        v = ret.loc[e].values
        base = ret.dropna().values
        if len(v) < 2:
            continue
        se = np.sqrt(v.var(ddof=1) / len(v) + base.var(ddof=1) / len(base))
        w = int((v > 0).sum())
        print(f"    h={h:2d}: N={len(v):2d}  mean {100*v.mean():+.3f}%  "
              f"base {100*base.mean():+.3f}%  diff {100*(v.mean()-base.mean()):+.3f}%"
              f"  welch t {(v.mean()-base.mean())/se:+.2f}  record {w}-{len(v)-w}"
              f"  sign p {sign_test(w, len(v)):.4f}  "
              f"boot P(<=0) {bootstrap_p_le0(v):.3f}")

print("\n" + "=" * 78)
print("LEAVE-ONE-FEATURE-OUT: is the neighbour set one feature in disguise?")
print("=" * 78)
K = 40
base_nb = set(dist.nsmallest(K).index)
for drop in F.columns:
    cols = [c for c in F.columns if c != drop]
    d2 = np.sqrt(((Z[cols] - zt[cols]) ** 2).sum(axis=1))
    d2 = d2[d2.index < pd.Timestamp("2026-01-01")]
    nb2 = set(d2.nsmallest(K).index)
    ov = len(base_nb & nb2) / K
    ret = vehicle_ret(px, [("SPY", 1.0)], 5, 1)
    e2 = declusters(pd.DatetimeIndex(sorted(nb2)), 21, px.index)
    e2 = pd.DatetimeIndex(e2).intersection(ret.dropna().index)
    print(f"  drop {drop:14s} overlap with full set {100*ov:5.1f}%   "
          f"h=5 episodes N={len(e2):2d} mean {100*ret.loc[e2].mean():+.3f}%")

print("\n" + "=" * 78)
print("MIDTERM cross on the K=40 neighbour episodes")
print("=" * 78)
nb = dist.nsmallest(40).index.sort_values()
epi = declusters(pd.DatetimeIndex(nb), 21, px.index)
for h in (5, 10):
    ret = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    e = pd.DatetimeIndex(epi).intersection(ret.dropna().index)
    mid = np.array([x.year % 4 == 2 for x in e])
    show([summarize(ret.loc[e[mid]].values, f"h={h} midterm"),
          summarize(ret.loc[e[~mid]].values, f"h={h} non-midterm")],
         f"K=40 by cycle year, h={h}")

print("\n" + "=" * 78)
print("HOW CLOSE ARE THE NEIGHBOURS, REALLY? (feature values of the top 10)")
print("=" * 78)
top = dist.nsmallest(10).index
print(F.loc[top].round(2).to_string())
print("\ntoday:")
print(F.iloc[[-1]].round(2).to_string())
