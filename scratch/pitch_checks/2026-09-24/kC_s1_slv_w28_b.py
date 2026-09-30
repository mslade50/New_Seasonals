"""S1 follow-up for the W28 watchlist verdict (S1 is killed in round 1).

At h=1 the day-level cell has NO overlap (consecutive triggers hold disjoint
sessions), so the gap-5 episode number is a first-break-in-5-sessions claim.
Split the faithful cell into FIRST breaks (no faithful trigger in the prior 5
sessions) and FOLLOW-ON breaks, lag 0/1/2, and era.  Also the GLD-leg margin:
how often does GLD miss by <= 0.25pp (today -1.80%) and what do those pay.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
BAR = pd.Timestamp("2026-09-23")
px = close_panel(["GLD", "SLV", "GDX"]).dropna().loc[:BAR]
r1 = {t: px[t] / px[t].shift(1) - 1.0 for t in ["GLD", "SLV", "GDX"]}
LEGS = [("SLV", -1.0)]
faith = (r1["GLD"] <= -0.02) & (r1["SLV"] <= -0.02) & (r1["GDX"] <= -0.02)
prior = faith.shift(1).rolling(5).max().fillna(0).astype(bool)
first = faith & ~prior
follow = faith & prior


def rec(mask, lag, label, h=1):
    r = vehicle_ret(px, LEGS, h, lag)
    base = r.dropna()
    d = px.index[mask.values & r.notna().values]
    v = r.loc[d].values
    w = int((v > 0).sum())
    o = summarize(v, label)
    o["rec"] = f"{w}-{len(v)-w}"
    o["p_vs_downrate"] = round(sign_test(w, len(v), float((base > 0).mean())), 4)
    return o, d, v


for lag in (0, 1, 2):
    rows = [rec(faith, lag, "ALL faithful days (h=1 no overlap)")[0],
            rec(first, lag, "FIRST break (none in prior 5)")[0],
            rec(follow, lag, "FOLLOW-ON break (one in prior 5)")[0]]
    show(rows, f"h=1 lag={lag}, day level")
for lab, m in (("ALL", faith), ("FIRST", first), ("FOLLOW", follow)):
    _, d, v = rec(m, 1, lab)
    show(era_split(d, v), f"{lab} era split, h=1 lag=1")

miss = (r1["GLD"] > -0.02) & (r1["GLD"] <= -0.0175) & (r1["SLV"] <= -0.02) & (r1["GDX"] <= -0.02)
for lag in (0, 1, 2):
    show([rec(miss, lag, "GLD missed by <= 0.25pp, SLV & GDX <= -2%")[0],
          rec(miss & (r1["SLV"] <= -0.04), lag, "  ... and SLV <= -4%")[0]],
         f"near-miss GLD leg, h=1 lag={lag}")
_, d, v = rec(miss, 1, "")
print("  near-miss dates (lag1 short h=1):",
      ", ".join(f"{x.date()}({100*y:+.2f})" for x, y in zip(d, v)))
