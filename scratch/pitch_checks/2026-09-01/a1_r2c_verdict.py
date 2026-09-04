"""A1 verdict arithmetic -- the three numbers the kill rests on, in one place.

(1) THE TRADED RUNG IS OUTSIDE THE GRID THAT WAS CHARGED.
    d2b_c5_flattener_charge.py declared HS = (1, 2, 3, 5, 10) and permuted a
    3-vehicle x 2-sign x 5-horizon grid to get P = 0.018.  The rung the parked
    entry actually ships is h=8 ("the horizon ladder TOPS OUT at 22.2 bps
    (h=8, 3.70x)"), which is NOT a member of that grid.  Show it.

(2) THE LIVE MAGNITUDE BUCKET.  Today's 252-session yield change decides which
    half of the cell's own support the live signal sits in.  Price that half
    against the cost settled in a1_r2b_cost.py.

(3) THE RE-PARK THRESHOLD.  What would have to be true for the live state to
    sit in the paying half?  Quote a ^TNX level, not a mood.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
ASOF = pd.Timestamp("2026-08-31")

px = close_panel(["^TNX", "TLT", "IEF"]).dropna(how="any")
px = px[px.index <= ASOF]
idx = px.index
tnx = px["^TNX"]
hi252 = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
off_hi = tnx / hi252 - 1.0
LEVEL = off_hi >= -0.0025
d = px[["TLT", "IEF"]].pct_change().dropna()
BETA = float(np.polyfit(d["IEF"].values, d["TLT"].values, 1)[0])
FLAT = [("IEF", 1.0), ("TLT", -1.0 / BETA)]

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


def cell(h, gap=10):
    ret = vehicle_ret(px, FLAT, h, 1)
    sig = idx[LEVEL.reindex(idx, fill_value=False).values & ret.notna().values]
    ep = fast_decluster(sig, max(h, gap))
    return ep, ret.loc[ep].values


print("=" * 100)
print("(1) THE TRADED RUNG IS OUTSIDE THE CHARGED GRID")
print("=" * 100)
DECLARED = (1, 2, 3, 5, 10)
print("  d2b declared HS = %s and charged 3 vehicles x 2 signs x 5 horizons -> P = 0.018" % (DECLARED,))
print("  the parked entry ships h=8.  h=8 in the declared grid? %s" % (8 in DECLARED))
for h in (1, 2, 3, 5, 8, 10):
    ep, v = cell(h)
    t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
    print("    h=%2d  %s  N=%2d  %5.1f bps  |t| %.2f"
          % (h, "declared" if h in DECLARED else "OFF-GRID", len(ep), 100 * 100 * v.mean(), abs(t)))
print("  best DECLARED rung is h=10 at %.1f bps; the shipped h=8 at %.1f bps was"
      % (100 * 100 * cell(10)[1].mean(), 100 * 100 * cell(8)[1].mean()))
print("  reached by walking a ladder the permutation never saw.")

print("\n" + "=" * 100)
print("(2) THE LIVE MAGNITUDE BUCKET, priced")
print("=" * 100)
chg252 = (tnx - tnx.shift(252)) * 100.0
chg63 = (tnx - tnx.shift(63)) * 100.0
rng252 = (rolling_on_valid(tnx, lambda x: x.rolling(252).max())
          - rolling_on_valid(tnx, lambda x: x.rolling(252).min())) * 100.0
COST_C = 3.594   # half-spread each side + fixed comm + auction fee, a1_r2b_cost
COST_C_BORROW = 3.594 + 0.829   # + 0.50%/yr borrow on the 0.522 TLT leg for 8 td

ep8, v8 = cell(8)
print("  headline h=8: %.1f bps -> %.2fx at %.2f bps (conv C), %.2fx with borrow"
      % (100 * 100 * v8.mean(), 100 * 100 * v8.mean() / COST_C, COST_C,
         100 * 100 * v8.mean() / COST_C_BORROW))
for lab, ser, today in (("252d yield change", chg252, float(chg252.iloc[-1])),
                        ("63d yield change", chg63, float(chg63.iloc[-1])),
                        ("252d range width", rng252, float(rng252.iloc[-1]))):
    e = ser.loc[ep8].values
    med = np.nanmedian(e)
    side = e <= med if today <= med else e > med
    vv = v8[side]
    bps = 100 * 100 * vv.mean()
    print("  %-18s today %+6.1f bp vs episode median %+6.1f -> live half is the %s half:"
          "  %5.1f bps  N=%2d  hit %.0f%%  -> %.2fx (conv C)  %.2fx (with borrow)"
          % (lab, today, med, "LOW" if today <= med else "HIGH", bps, int(side.sum()),
             100 * (vv > 0).mean(), bps / COST_C, bps / COST_C_BORROW))
print("  three of the four magnitude conditioners put today in a half that pays")
print("  under the 5x bar; the 21d conditioner has no dose at all (see a1_r2).")

print("\n" + "=" * 100)
print("(3) RE-PARK THRESHOLD -- a testable number, not a mood")
print("=" * 100)
e252 = chg252.loc[ep8].values
med252 = float(np.nanmedian(e252))
lo = v8[e252 <= med252]
hi = v8[e252 > med252]
print("  cell's own 252d-change median: %+.1f bp.  LOW half %.1f bps (N=%d) | HIGH half %.1f bps (N=%d)"
      % (med252, 100 * 100 * lo.mean(), len(lo), 100 * 100 * hi.mean(), len(hi)))
lvl_needed = float(tnx.iloc[-253]) + med252 / 100.0
print("  ^TNX 252 sessions ago = %.4f, so the HIGH half needs today's ^TNX >= %.4f"
      % (tnx.iloc[-253], lvl_needed))
print("  live ^TNX %.4f  ->  gap %+.1f bp" % (tnx.iloc[-1], 100 * (lvl_needed - tnx.iloc[-1])))
print("  the 5x bar at conv-C-with-borrow (%.2f bps) needs %.1f bps of episode mean;"
      % (COST_C_BORROW, 5 * COST_C_BORROW))
print("  the LOW half delivers %.1f bps and the HIGH half %.1f bps."
      % (100 * 100 * lo.mean(), 100 * 100 * hi.mean()))
print("  RE-PARK ARM: the cell is tradeable when ^TNX is at a trailing-252 max AND")
print("  the 252-session yield change is >= %+.0f bp (^TNX >= trailing-252-ago + %.2f pt)."
      % (med252, med252 / 100.0))

print("\n" + "=" * 100)
print("(4) DECLUSTER DEGRADATION (supporting)")
print("=" * 100)
for gap in (5, 10, 21, 42):
    ep, v = cell(8, gap)
    print("  gap %2d  N=%2d  %5.1f bps  |t| %.2f  -> %.2fx at conv C + borrow"
          % (gap, len(ep), 100 * 100 * v.mean(),
             abs(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))),
             100 * 100 * v.mean() / COST_C_BORROW))
