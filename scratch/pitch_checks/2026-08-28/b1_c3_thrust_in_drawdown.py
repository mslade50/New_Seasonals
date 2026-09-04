"""C3 round 1 -- "the thrust from inside a drawdown".

State: 5-day return rank >= 90 (252d lookback) while the close sits >= 10%
below its own trailing-252d high. Live on EWZ 2026-08-27 (r5 91.3, -13.49%).

Round 1 = battery on EWZ + the ONE probe that decides it: does the drawdown
conditioner do any work over the bare 5-day thrust? The registry's silver
entry (2026-08-10) says distance-from-high is a U-shaped noise carve, so the
prior is that it does nothing. Run the gate attribution first-class.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (battery, close_panel, declusters, load_prices, pct_rank,
                       rolling_on_valid, show, summarize, vehicle_ret)

H = 10
COST = 9.0  # EWZ country ETF round trip

px = close_panel(["EWZ"])
s = load_prices(["EWZ"])["EWZ"]["Close"].dropna()
r5 = pct_rank(s, 5)
dd = s / s.rolling(252).max() - 1.0

joint = (r5 >= 90) & (dd <= -0.10)
bare = (r5 >= 90)

variants = {
    "r5>=85 & dd<=-10%": (pct_rank(s, 5) >= 85) & (dd <= -0.10),
    "r5>=90 & dd<=-10%": joint,
    "r5>=95 & dd<=-10%": (r5 >= 95) & (dd <= -0.10),
    "r5>=90 & dd<=-5%":  (r5 >= 90) & (dd <= -0.05),
    "r5>=90 & dd<=-15%": (r5 >= 90) & (dd <= -0.15),
    "r5>=90 & dd<=-20%": (r5 >= 90) & (dd <= -0.20),
    "r5>=90 BARE (no dd gate)": bare,
}

battery(px, joint, [("EWZ", 1.0)], H, "C3 EWZ thrust inside a >=10% drawdown",
        COST, variants=variants, min_gap=10)

# ---------------------------------------------------------------------------
print("\n" + "=" * 92)
print("GATE ATTRIBUTION -- does the drawdown clause do any work over the bare thrust?")
print("=" * 92)
ret = vehicle_ret(px, [("EWZ", 1.0)], H, 1)
valid = ret.dropna().index


def cell(mask, label):
    t = px.index[mask.reindex(px.index, fill_value=False).values].intersection(valid)
    epi = declusters(t, 10, valid)
    r = summarize(ret.loc[epi].values, label)
    r["n_days"] = len(t)
    return r


rows = [cell(bare, "A. bare r5>=90 (the thrust)"),
        cell(joint, "B. thrust AND dd<=-10% (C3)"),
        cell(bare & (dd > -0.10), "C. thrust AND dd>-10% (the complement)"),
        cell(bare & (dd > -0.03), "D. thrust AND within 3% of the high")]
show(rows, "gate attribution, episode level, h=10")

# U-shape check: bucket the thrust population by drawdown depth
print("\nthrust population bucketed by drawdown depth (episodes, h=10):")
b_t = px.index[bare.reindex(px.index, fill_value=False).values].intersection(valid)
b_epi = declusters(b_t, 10, valid)
dep = dd.reindex(b_epi)
buckets = [(-0.02, 0.0), (-0.05, -0.02), (-0.10, -0.05), (-0.15, -0.10),
           (-0.25, -0.15), (-1.0, -0.25)]
rows = []
for lo, hi in buckets:
    m = (dep > lo) & (dep <= hi)
    rows.append(summarize(ret.loc[b_epi[m.values]].values,
                          f"dd in ({lo:+.0%},{hi:+.0%}]"))
show(rows, "U-shape probe")

# where does today sit?
print(f"\n  today's EWZ dd = {100*dd.iloc[-1]:+.2f}%  r5 = {r5.iloc[-1]:.1f}")
