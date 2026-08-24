"""b4 — tomorrow-specific tail risk and the exact near-miss numbers.

Today's entry is MOC 2026-08-24. A hold of h>=4 contains Jackson Hole
(2026-08-28); h>=5 contains the August month-end close (2026-08-31). Neither
anchor has ever been inside the hold of these cells, which is a statement
about how unmeasured the live hold is.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, declusters, event_in_window, load_prices, load_events,
    rolling_on_valid, show, summarize, vehicle_ret,
)
from pitch_lab import _valid_pct_change as vpc  # noqa: E402

pd.set_option("display.width", 220)


def tape_z10(close, n=10):
    return close.pct_change(n) / (close.pct_change().rolling(21).std() * np.sqrt(n))


# --- C3
px = close_panel(["FCX", "XME", "SPY"])
r5 = vpc(px["FCX"], 5)
hi = rolling_on_valid(px["FCX"], lambda x: x.rolling(252).max())
for th, lbl in ((0.15, "r5>=15%"), (0.10, "r5>=10%")):
    m = ((r5 >= th) & (px["FCX"] >= hi * (1 - 1e-9))).fillna(False)
    ret = vehicle_ret(px, [("FCX", 1.0)], 5)
    epi = declusters(px.index[m.values & ret.notna().values], 5, px.index)
    for kind in ("jackson_hole",):
        fl = event_in_window(epi, px.index, 5, 1, (kind,))
        print(f"C3 {lbl}: {int(fl.sum())} of {len(epi)} episodes ever carried "
              f"{kind} inside a 5td hold")
    mm = pd.DatetimeIndex(epi).month
    print(f"   episodes by month: "
          f"{pd.Series(mm).value_counts().sort_index().to_dict()}  "
          f"(today is August)")
    aug = mm == 8
    if aug.sum():
        show([summarize(ret.loc[epi].values[aug], f"August episodes (N={int(aug.sum())})"),
              summarize(ret.loc[epi].values[~aug], "other months")], f"C3 {lbl} month split")
    else:
        print("   NO August episode has ever occurred in this cell.")

# --- C7
COMPLEX = ["XLE", "XOP", "USO", "COP", "CVX", "VLO", "OXY", "SLB", "EOG",
           "HAL", "WMB"]
raw = load_prices(sorted(set(COMPLEX + ["SPY"])))
pan = close_panel(sorted(set(COMPLEX + ["SPY"]))).dropna(subset=["XLE", "SPY"])
IDX = pan.index
z = pd.DataFrame({t: tape_z10(raw[t]["Close"]) for t in COMPLEX}).reindex(IDX)
allv = z.notna().all(axis=1)
cnt = (z >= 2.0).sum(axis=1).where(allv)
ret = vehicle_ret(pan, [("XLE", 1.0)], 5)
for lo, hi2, lbl in ((5, 11, "count>=5 (today)"), (2, 3, "count in [2,3]")):
    m = ((cnt >= lo) & (cnt <= hi2)).fillna(False)
    epi = declusters(IDX[m.values & ret.notna().values], 10, IDX)
    fl = event_in_window(epi, IDX, 5, 1, ("jackson_hole",))
    mm = pd.DatetimeIndex(epi).month
    print(f"\nC7 {lbl}: {len(epi)} episodes, {int(fl.sum())} ever carried "
          f"jackson_hole in a 5td hold")
    print(f"   by month: {pd.Series(mm).value_counts().sort_index().to_dict()}")
    aug = mm == 8
    if aug.sum():
        show([summarize(ret.loc[epi].values[aug], f"August (N={int(aug.sum())})"),
              summarize(ret.loc[epi].values[~aug], "other months")], f"C7 {lbl} month split")
    else:
        print("   NO August episode has ever occurred in this cell.")

print("\n=== exact near-miss numbers ===")
for k in range(1, 9):
    m = (cnt == k).fillna(False)
    d = IDX[m.values & ret.notna().values]
    e = declusters(d, 10, IDX)
    if len(e) >= 5:
        v = ret.loc[e].values
        print(f"  C7 count=={k}: N_epi={len(e):3d} mean {100*v.mean():+.3f}% "
              f"hit {100*(v > 0).mean():.1f}%")
print(f"  today's count = {int(cnt.iloc[-1])}; the positive band is 2-3 "
      f"(+0.699% h=5, t=2.40, 99 episodes, +0.459pp over all days)")
