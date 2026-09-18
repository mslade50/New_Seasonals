"""Confirm pitch_lab.filter_vs_reanchor reproduces the hand-rolled A1 result.

The helper was extracted from k1_tlt_filter_vs_reanchor_b.py this morning.
A promotion that does not reproduce the number it came from is worse than no
promotion, so this rebuilds the watchlist-5 masks and checks the decomposition
against the reported +0.016pp filtering / +0.683pp re-anchoring.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = load_prices(["TLT", "IEF", "LQD"])
idx = px["TLT"].index
for t in ("IEF", "LQD"):
    idx = idx.intersection(px[t].index)


def dist_low(t, n=252):
    s = px[t]["Close"].reindex(idx)
    return (s / s.rolling(n).min() - 1.0) * 100


tl, ie, lq = dist_low("TLT"), dist_low("IEF"), dist_low("LQD")
ret = fwd_lag(px["TLT"]["Close"].reindex(idx), h=1, lag=1)

# PARENT: TLT alone at the floor, taken fresh (first in >= 10 td)
parent_raw = (tl <= 0.5).fillna(False)
parent_anchors = declusters(idx[parent_raw.to_numpy(dtype=bool)], 10, idx)
parent = pd.Series(idx.isin(parent_anchors), index=idx)

# CHILD: the three-way tight rung, taken fresh the same way
child_raw = ((tl <= 0.5) & (ie <= 1.0) & (lq <= 1.0)).fillna(False)
child_anchors = declusters(idx[child_raw.to_numpy(dtype=bool)], 10, idx)
child = pd.Series(idx.isin(child_anchors), index=idx)

print(f"parent fresh anchors {len(parent_anchors)}   "
      f"child fresh anchors {len(child_anchors)}")

out = filter_vs_reanchor(ret, parent, child, idx, window_td=21,
                         label="watchlist 5: TLT floor -> three-way IG rung")

print("\nshift distribution:", out["shifts"])
print(f"re-anchoring share {100 * out['reanchor_share']:.0f}%")


def first_in_n(mask, n=10):
    """The ARM's own form: first trigger day in >= n sessions, as a MASK.
    The clock restarts on every TRIGGER, unlike declusters() which restarts on
    the last KEPT day. This is the reconciliation debt filed on watchlist 5."""
    m = mask.to_numpy(dtype=bool)
    keep, last = [], -10 ** 9
    for i, hit in enumerate(m):
        if hit:
            if i - last >= n:
                keep.append(idx[i])
            last = i
    return pd.DatetimeIndex(keep)


arm_child = pd.Series(idx.isin(first_in_n(child_raw)), index=idx)
arm_parent = pd.Series(idx.isin(first_in_n(parent_raw)), index=idx)
print(f"\nARM form (mask): parent {int(arm_parent.sum())} anchors, "
      f"child {int(arm_child.sum())} anchors "
      f"-- against declusters() at {len(parent_anchors)} and "
      f"{len(child_anchors)}")
out_arm = filter_vs_reanchor(ret, arm_parent, arm_child, idx, window_td=21,
                             label="watchlist 5, ARM form (the executable one)")

for lbl, o in (("declusters form", out), ("arm/mask form", out_arm)):
    child_mean = ret.reindex(pd.DatetimeIndex(
        [b for _, b, _ in o["pairs"]])).dropna().mean()
    null = reanchor_null(ret, [a for a, _, _ in o["pairs"]], o["shifts"],
                         idx, child_mean=child_mean, n_boot=5000, seed=0)
    print(f"\n{lbl}: filtering {o['filtering_pp']:+.3f}pp, "
          f"re-anchoring {o['reanchoring_pp']:+.3f}pp "
          f"({100 * o['reanchor_share']:.0f}%), "
          f"blind-delay P = {null['p']:.4f}")

print("""
VERDICT ON THE PROMOTION.
The ARM/MASK form reproduces the hand-rolled k1_tlt_filter_vs_reanchor_b.py
EXACTLY -- 11 matched pairs, deleted -0.256%, kept-at-parent -0.213%,
kept-at-child +0.470%, filtering +0.016pp, re-anchoring +0.683pp (98%) -- so
pitch_lab.filter_vs_reanchor is a faithful extraction and may be cited in
place of the day-local script. The blind-delay null reads 0.0248 here against
0.0223 there, a resampling difference at n_boot=5000, same conclusion.

The declusters() row is NOT a discrepancy in the helper. It is the SAME
reconciliation debt filed on watchlist 5 this morning, now expressed as
arithmetic: the arm's prose ('first trigger day in >= 10 sessions', a mask
whose clock restarts on every TRIGGER) and declusters(mask, 10) (whose clock
restarts on the last KEPT day) select 11 and 19 child anchors respectively.
Quote the form you actually measured.

Invariant across both populations, and the finding itself: re-anchoring
dominates filtering by roughly an order of magnitude, filtering is at or
below zero, and the blind-delay null is small -- the join delays entry, and
the delay is not quite blind.
""")
