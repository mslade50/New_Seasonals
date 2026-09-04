"""C6 addendum — the one sub-cell the round-1 scan turned up positive:
pre-opex week in AUGUST of a MIDTERM year (N=6, +0.847%). Price the search
and run drop-best, because a 6-observation cell found inside a parent that
is WORSE than unconditional is exactly the shape the registry keeps killing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import close_panel, fwd_lag, load_events, sign_test, summarize, show  # noqa: E402

ASOF = pd.Timestamp("2026-08-13")
H = 5
px = close_panel(["SPY"])
px = px[px.index <= ASOF]
spy = px["SPY"].dropna()
idx = spy.index
pos = pd.Series(range(len(idx)), index=idx)
ret = fwd_lag(spy, H, 1)

opex = load_events(["opex"])["date"]
# MUST clip to the price index: an opex date past the last bar otherwise maps
# to the final bar and every future expiry collapses onto ONE anchor, which
# duplicated a single 2026 observation 17 times on the first run of this file.
opex = opex[(opex >= idx[0]) & (opex <= idx[-1])]
anchors = []
for d in opex:
    p = pos.get(d)
    if p is None:
        prior = idx[idx <= d]
        if len(prior) == 0:
            continue
        p = pos[prior[-1]]
    if p - 6 >= 0:
        anchors.append(idx[p - 6])
anchors = sorted(set(anchors))
a = pd.DatetimeIndex([d for d in anchors if ret.notna().get(d, False)])
v = ret.loc[a].values
d = pd.DatetimeIndex(a)

cell = (d.month == 8) & ((d.year % 4) == 2)
vc = v[cell]
print("=" * 74)
print("pre-opex week, AUGUST, MIDTERM years — SPY long h=5")
print("=" * 74)
print(f"  N={len(vc)}  mean {100*vc.mean():+.3f}%  "
      f"record {int((vc>0).sum())}-{len(vc)-int((vc>0).sum())}  "
      f"sign p = {sign_test(int((vc > 0).sum()), len(vc)):.4f}")
print("  per-year:", {int(y): round(100 * x, 2) for y, x in zip(d[cell].year, vc)})
print(f"  drop-best {100*np.sort(vc)[:-1].mean():+.3f}%   "
      f"drop-two-best {100*np.sort(vc)[:-2].mean():+.3f}%")

show([summarize(v, "parent: ALL pre-opex weeks"),
      summarize(v[d.month == 8], "August, any year"),
      summarize(v[(d.year % 4) == 2], "midterm, any month"),
      summarize(vc, "August AND midterm")], "the cell inside its parents")

base = ret.dropna()
print(f"\n  SPY unconditional h=5 = {100*base.mean():+.3f}%")
print(f"  SPY unconditional h=5, August days = "
      f"{100*base[base.index.month == 8].mean():+.3f}%")
print(f"  SPY unconditional h=5, August of midterm years = "
      f"{100*base[(base.index.month == 8) & (base.index.year % 4 == 2)].mean():+.3f}%")
print("\n  search cost: month (12) x cycle (4) x era (3) = 144 cells available "
      "in this one split table; a p of 0.11 does not survive one of them.")
