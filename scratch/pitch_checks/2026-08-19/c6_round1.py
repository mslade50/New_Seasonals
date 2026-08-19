"""C6 round 1: yields up / dollar at the bottom of its trailing year.

Trigger (all PIT, trailing-252d ranks):
  ^TNX 21d return > 0  AND  rank21(^TNX) >= 65
  AND rank21(DX-Y.NYB) <= 20

Legs measured BOTH signs on DX, plus UUP (dead vehicle, cost check only),
TLT and SPY.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

T = ["DX-Y.NYB", "UUP", "^TNX", "TLT", "SPY", "GLD"]
px = close_panel(T).dropna(subset=["DX-Y.NYB", "^TNX", "SPY"])
dx, tnx = px["DX-Y.NYB"], px["^TNX"]

r21_tnx = tnx.pct_change(21)
rk_tnx = pct_rank(tnx, 21)
rk_dx = pct_rank(dx, 21)
r21_dx = dx.pct_change(21)

base = (r21_tnx > 0) & (rk_tnx >= 65) & (rk_dx <= 20)
base = base.fillna(False)
print("trigger days:", int(base.sum()), " span",
      px.index[base][0].date(), "..", px.index[base][-1].date())
print("today's state: TNX r21 %+0.2f%% rank %.1f | DX rank %.1f r21 %+0.2f%%" %
      (100*r21_tnx.iloc[-1], rk_tnx.iloc[-1], rk_dx.iloc[-1], 100*r21_dx.iloc[-1]))
print("today's TNX 21d LEVEL change: %+0.3f pts (%.3f -> %.3f)" %
      (tnx.iloc[-1]-tnx.iloc[-22], tnx.iloc[-22], tnx.iloc[-1]))

# magnitude context of the rank gate (registry: a rank gate is not a magnitude gate)
lvl_chg = tnx - tnx.shift(21)
trig_lvl = lvl_chg[base].dropna()
print("\nTNX 21d LEVEL change on trigger days: median %+0.3f pts, "
      "today %+0.3f = pctile %.1f of the trigger distribution" %
      (trig_lvl.median(), lvl_chg.iloc[-1],
       100*(trig_lvl < lvl_chg.iloc[-1]).mean()))
print("DX 21d return on trigger days: median %+0.2f%%, today %+0.2f%% = pctile %.1f"
      % (100*r21_dx[base].median(), 100*r21_dx.iloc[-1],
         100*(r21_dx[base] < r21_dx.iloc[-1]).mean()))

variants = {
    "TNX>=55 / DX<=20": ((r21_tnx > 0) & (rk_tnx >= 55) & (rk_dx <= 20)).fillna(False),
    "TNX>=75 / DX<=20": ((r21_tnx > 0) & (rk_tnx >= 75) & (rk_dx <= 20)).fillna(False),
    "TNX>=65 / DX<=10": ((r21_tnx > 0) & (rk_tnx >= 65) & (rk_dx <= 10)).fillna(False),
    "TNX>=65 / DX<=30": ((r21_tnx > 0) & (rk_tnx >= 65) & (rk_dx <= 30)).fillna(False),
    "TNX gate ALONE":   ((r21_tnx > 0) & (rk_tnx >= 65)).fillna(False),
    "DX gate ALONE":    (rk_dx <= 20).fillna(False),
}

H = 5
for label, legs, cost in [
    ("LONG DX (dollar bounces)",  [("DX-Y.NYB", 1.0)], 1.5),
    ("SHORT DX (dollar keeps falling)", [("DX-Y.NYB", -1.0)], 1.5),
    ("LONG UUP (dead-vehicle cost check)", [("UUP", 1.0)], 6.0),
    ("LONG TLT", [("TLT", 1.0)], 2.0),
    ("LONG SPY", [("SPY", 1.0)], 2.0),
]:
    battery(px, base, legs, H, f"C6 {label}", cost, variants=variants,
            min_gap=21, event_kinds=("fomc_decision", "cpi", "nfp"))

# horizon view for both dollar signs, episode level
print("\n=== C6 horizon scan, episodes (min_gap=21) ===")
trig = px.index[base.values]
for nm, legs in [("DX long", [("DX-Y.NYB", 1.0)]),
                 ("DX short", [("DX-Y.NYB", -1.0)]),
                 ("TLT long", [("TLT", 1.0)]),
                 ("SPY long", [("SPY", 1.0)]),
                 ("GLD long", [("GLD", 1.0)])]:
    show(horizon_scan(px, trig, legs, hs=(1, 2, 3, 5, 10), min_gap=21), nm)
