"""A2 round 2 -- the gate attribution that decides whether A2 is an idea or a
re-skin of A1.

A2's only monotone dose response is on the FLATTENER (a2_fomc_yield_extreme.py:
-0.00% +0.769 / -0.25% +0.767 / -1% +0.642 / -2% +0.415 / -5% +0.226 / -10%
+0.055 / all +0.063).  But the flattener is EXACTLY A1's vehicle and the
yield-distance gate is EXACTLY A1's trigger.  So the registry's gate rule bites
on the OTHER leg: run it WITHOUT the FOMC anchor.  If the calendar leg does not
move the result, nothing may be attributed to the FOMC.

Three tests:
  1. A1's plain trigger split by FOMC-in-the-hold vs not, at h=10.
  2. All-anchor pre-FOMC drift on the flattener vs all-days: is there any FOMC
     effect at all before the yield gate is applied?
  3. The overlap: how many of A1's 50 trigger episodes already sit inside a
     pre-FOMC window?  (If the two populations are largely the same days the
     "new anchor" is a relabelling.)
Plus the family multiplicity charge on the 4 vehicles x 8 gate rungs actually
walked in a2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
ASOF = pd.Timestamp("2026-08-31")

px = close_panel(["^TNX", "TLT", "IEF", "SPY"]).dropna(how="any")
px = px[px.index <= ASOF]
idx = px.index
tnx = px["^TNX"]
hi252 = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
off_hi = tnx / hi252 - 1.0
LEVEL = off_hi >= -0.0025

d = px[["TLT", "IEF"]].pct_change().dropna()
BETA = float(np.polyfit(d["IEF"].values, d["TLT"].values, 1)[0])
FLAT = [("IEF", 1.0), ("TLT", -1.0 / BETA)]
VEH = {"TLT": [("TLT", 1.0)], "IEF": [("IEF", 1.0)], "FLAT": FLAT, "SPY": [("SPY", 1.0)]}

ev = load_events(["fomc_decision"])
fom = pd.DatetimeIndex(sorted(ev["date"].unique()))
pos, kept = anchor_positions(idx, fom, offset=-10)
anchor_set = set(idx[p] for p in pos if 0 <= p < len(idx))

POS = {dd: i for i, dd in enumerate(idx)}


def fast_decluster(sig, gap):
    keep, last = [], -10 ** 9
    for dd in sig:
        p = POS.get(dd)
        if p is None:
            continue
        if p - last >= gap:
            keep.append(dd)
            last = p
    return pd.DatetimeIndex(keep)


print("=" * 110)
print("1. A1's plain trigger, split by whether an FOMC DECISION lands in the hold")
print("=" * 110)
for h in (8, 10):
    for vk, legs in VEH.items():
        ret = vehicle_ret(px, legs, h, 1)
        valid = ret.notna()
        sig = idx[LEVEL.reindex(idx, fill_value=False).values & valid.values]
        ep = fast_decluster(sig, max(h, 10))
        fl = event_in_window(ep, idx, h, 1, ("fomc_decision",))
        a, b = ret.loc[ep].values[fl], ret.loc[ep].values[~fl]
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if len(a) > 1 and len(b) > 1 else np.nan
        print("  h=%2d %-5s | FOMC IN hold N=%2d %+.4f%% hit %.0f%% | FOMC OUT N=%2d %+.4f%% hit %.0f%%"
              " | diff %+.4fpp welch t %+.2f"
              % (h, vk, len(a), 100 * a.mean(), 100 * (a > 0).mean(),
                 len(b), 100 * b.mean(), 100 * (b > 0).mean(),
                 100 * (a.mean() - b.mean()), (a.mean() - b.mean()) / se))

print("\n" + "=" * 110)
print("2. Is there ANY pre-FOMC effect on the flattener before the yield gate?")
print("=" * 110)
for vk, legs in VEH.items():
    ret = vehicle_ret(px, legs, 10, 0)
    anc = np.array([float(ret.iloc[p]) for p in pos if 0 <= p < len(idx)])
    anc = anc[~np.isnan(anc)]
    base = ret.dropna()
    se = np.sqrt(anc.var(ddof=1) / len(anc) + base.var(ddof=1) / len(base))
    print("  %-5s all %d pre-FOMC anchors %+.4f%%  vs all-days %+.4f%%  ->  excess %+.4fpp  t %+.2f"
          % (vk, len(anc), 100 * anc.mean(), 100 * base.mean(),
             100 * (anc.mean() - base.mean()), (anc.mean() - base.mean()) / se))

print("\n" + "=" * 110)
print("3. OVERLAP -- how much of A1's population is already a pre-FOMC window?")
print("=" * 110)
ret10 = vehicle_ret(px, FLAT, 10, 1)
sig = idx[LEVEL.reindex(idx, fill_value=False).values & ret10.notna().values]
ep = fast_decluster(sig, 10)
in_anchor = np.array([dd in anchor_set for dd in ep])
fl = event_in_window(ep, idx, 10, 1, ("fomc_decision",))
print("  A1 episodes N=%d | that ARE the decision-10td session exactly: %d"
      " | that have a decision inside the 10td hold: %d (%.0f%%)"
      % (len(ep), int(in_anchor.sum()), int(fl.sum()), 100 * fl.mean()))
print("  A2's tight cell (6 anchors) that are ALSO A1 trigger days: %d"
      % sum(1 for p in pos if 0 <= p < len(idx) and bool(LEVEL.iloc[p])))
print("  -> A2's 6 observations are a SUBSET of A1's 183 trigger days by construction.")

print("\n" + "=" * 110)
print("4. FAMILY MULTIPLICITY on the a2 walk (4 vehicles x 8 gate rungs = 32 cells)")
print("   rotation permutation of the LEVEL mask over the anchor set")
print("=" * 110)
RUNGS = (0.0000, 0.0025, 0.0100, 0.0200, 0.0500, 0.1000, 0.2000, 9.99)
rets10 = {vk: vehicle_ret(px, lg, 10, 0) for vk, lg in VEH.items()}
pos_ok = [p for p in pos if 0 <= p < len(idx)]
dist_base = np.array([float(off_hi.iloc[p]) for p in pos_ok])
vals = {vk: np.array([float(rets10[vk].iloc[p]) for p in pos_ok]) for vk in VEH}
good = {vk: ~np.isnan(vals[vk]) for vk in VEH}


def grid_max_t(dvec):
    best, where = 0.0, None
    for rung in RUNGS:
        m0 = dvec >= -rung
        for vk in VEH:
            m = m0 & good[vk]
            if m.sum() < 4:
                continue
            v = vals[vk][m]
            if v.std(ddof=1) == 0:
                continue
            t = abs(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
            if t > best:
                best, where = t, (vk, rung, 100 * v.mean(), len(v))
    return best, where


obs, where = grid_max_t(dist_base)
print("  OBSERVED grid max |t| = %.2f at %s rung -%.2f%% mean %+.3f%% N=%d"
      % (obs, where[0], 100 * where[1], where[2], where[3]))
rng = np.random.default_rng(7)
NP = 2000
maxes = np.array([grid_max_t(rng.permutation(dist_base))[0] for _ in range(NP)])
print("  permutation null (%d shuffles of the yield-distance vector across anchors):"
      " median %.2f p95 %.2f" % (NP, np.median(maxes), np.percentile(maxes, 95)))
print("  P(shuffled 32-cell grid max |t| >= %.2f) = %.3f" % (obs, float((maxes >= obs).mean())))
flat_t = None
m = (dist_base >= -0.0025) & good["FLAT"]
v = vals["FLAT"][m]
flat_t = abs(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
print("  the FLAT -0.25%% cell itself: |t| = %.2f -> P(null max >= that) = %.3f"
      % (flat_t, float((maxes >= flat_t).mean())))
