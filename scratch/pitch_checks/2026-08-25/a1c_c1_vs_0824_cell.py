"""C1 collision check vs the 2026-08-24 SPY/QQQ kill's by-product cell.

That kill recorded: "on days tech's 63-day rank is bottom-quintile while the
index's is not, QQQ LONG pays +0.508% at h=5".  That is evidence FOR C1's
DIRECTION, and the brief requires checking whether C1 is simply that already
measured cell re-cut.  Two things settle it: today's state against the cell's
gates, and the day-level mask overlap in both directions.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 240)

px = close_panel(["XLK", "XLV", "QQQ", "SPY"])
qr63 = pct_rank(px["QQQ"], 63)
sr63 = pct_rank(px["SPY"], 63)
kr63 = pct_rank(px["XLK"], 63)
r5 = px.pct_change(5)
SPREAD = (r5["XLV"] - r5["XLK"]) * 100.0

print(f"today: QQQ r63={qr63.iloc[-1]:.1f}  SPY r63={sr63.iloc[-1]:.1f}  "
      f"XLK r63={kr63.iloc[-1]:.1f}  |  5d XLV-XLK {SPREAD.iloc[-1]:+.2f}pp")

CELL = ((qr63 <= 20) & (sr63 > 20)).fillna(False)   # the 08-24 by-product cell
print(f"  08-24 cell live today? {bool(CELL.iloc[-1])}")

for h in (3, 5, 10):
    for lbl, legs in (("QQQ", [("QQQ", 1.0)]), ("XLK", [("XLK", 1.0)])):
        ret = vehicle_ret(px, legs, h)
        v = ret.dropna().index
        e = declusters(px.index[CELL.values].intersection(v), h, v)
        s = summarize(ret.loc[e].values, f"h={h} long {lbl} | QQQ r63<=20 & SPY r63>20")
        s["drift_pct"] = round(100 * ret.loc[v].mean(), 3)
        s["edge_pct"] = round(s["mean_pct"] - 100 * ret.loc[v].mean(), 3)
        show([s], "")

print("\n########## MASK OVERLAP: C1 vs the 08-24 cell ##########")
for rung in (6, 8, 10):
    A = set(px.index[(SPREAD >= rung).fillna(False).values])
    B = set(px.index[CELL.values])
    i = A & B
    print(f"  C1 5d>={rung}pp n={len(A):>4d} | 08-24 cell n={len(B):>4d} | shared {len(i):>3d}"
          f"  -> {100*len(i)/len(A):5.1f}% of C1, {100*len(i)/len(B):5.1f}% of the cell")

print("\n########## AND: the C1 trigger conditioned on the 08-24 cell being ON ##########")
for h in (3, 5):
    ret = vehicle_ret(px, [("XLK", 1.0)], h)
    v = ret.dropna().index
    for lbl, m in (("C1>=8pp & cell ON", (SPREAD >= 8).fillna(False) & CELL),
                   ("C1>=8pp & cell OFF", (SPREAD >= 8).fillna(False) & ~CELL),
                   ("cell ON, no C1", CELL & ~(SPREAD >= 8).fillna(False))):
        e = declusters(px.index[m.values].intersection(v), h, v)
        s = summarize(ret.loc[e].values, f"h={h} {lbl}")
        show([s], "")
