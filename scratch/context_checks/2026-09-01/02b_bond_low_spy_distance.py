"""Follow-on to 02. The joint cell is n=2 episodes, so the publishable content
is the RARITY and the company the state usually keeps, not a forward return.

Questions:
  1. On the 30 sessions IEF and LQD were both at a 252d low, where was SPY?
  2. What does the only prior analogue (April-May 2006, also a midterm year)
     look like out to 63 sessions?
  3. Widen the SPY tolerance: at what distance-from-high does the cell stop
     being unique?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TKRS = ["IEF", "LQD", "TLT", "SPY"]
px = close_panel(TKRS).dropna(subset=["IEF", "LQD", "SPY"])
W = 252
low = {t: rolling_on_valid(px[t], lambda x: x.rolling(W, min_periods=200).min())
       for t in TKRS}
spy_hi = rolling_on_valid(px["SPY"], lambda x: x.rolling(W, min_periods=200).max())

both = ((px["IEF"] <= low["IEF"] * (1 + 1e-9)) &
        (px["LQD"] <= low["LQD"] * (1 + 1e-9))).fillna(False)
dates = px.index[both.values]
dist = 100 * (px["SPY"] / spy_hi - 1.0)

print("=== 1. SPY distance from its own 252d high on the 30 both-at-low sessions ===")
d = dist.loc[dates].dropna()
print(f"n={len(d)}  median {d.median():+.2f}%  mean {d.mean():+.2f}%  "
      f"best {d.max():+.2f}%  worst {d.min():+.2f}%")
for yr, g in d.groupby(d.index.year):
    print(f"  {yr}: {len(g):>2} sessions, SPY {g.min():+.2f}% to {g.max():+.2f}% from its high")

print("\n=== 3. how the count falls as the SPY-near-high tolerance tightens ===")
for tol in (0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.025):
    m = both & (dist >= -100 * tol)
    dd = px.index[m.fillna(False).values]
    epi = declusters(dd, 21, px.index)
    print(f"  SPY within {100*tol:>4.1f}% of its high: {len(dd):>2} sessions, "
          f"{len(epi)} episodes -> {', '.join(str(x.date()) for x in epi)}")

print("\n=== 2. the 2006 analogue, day by day from 2006-05-01 ===")
anchor = pd.Timestamp("2006-05-01")
pos = px.index.get_loc(anchor)
for h in (1, 5, 10, 21, 42, 63, 126):
    if pos + h >= len(px):
        break
    line = [f"h={h:>3}"]
    for t in ["SPY", "TLT", "IEF", "LQD"]:
        r = 100 * (px[t].iloc[pos + h] / px[t].iloc[pos] - 1.0)
        line.append(f"{t} {r:+6.2f}%")
    print("  " + "  ".join(line))

print("\n  drawdown from the 2006-05-01 close, SPY:")
fwd = px["SPY"].iloc[pos:pos + 127]
trough = fwd.min()
print(f"    trough {trough:.2f} on {fwd.idxmin().date()}, "
      f"{100*(trough/px['SPY'].iloc[pos]-1):+.2f}% from the anchor close; "
      f"back above the anchor close on "
      f"{fwd[fwd > px['SPY'].iloc[pos]].index[fwd[fwd > px['SPY'].iloc[pos]].index > fwd.idxmin()][0].date()}")

print("\n=== the 2026 state, for the record ===")
last = px.index[-1]
for t in ["IEF", "LQD", "TLT"]:
    print(f"  {t}: close {px[t][last]:.2f}, 252d low {low[t][last]:.2f}, "
          f"{100*(px[t][last]/low[t][last]-1):+.2f}% above it")
print(f"  SPY: close {px['SPY'][last]:.2f}, {dist[last]:+.2f}% from its 252d high")
