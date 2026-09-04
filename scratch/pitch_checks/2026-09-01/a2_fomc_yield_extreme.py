"""A2 -- duration into the September FOMC with the ten-year at a one-year high.

00c already did count-first: 6 of 204 measurable anchors.  The brief's rule is
then explicit: either find a LOOSENED magnitude form of the conditioner with a
real count AND a real dose response, or kill on the count.

So this script does exactly two things:
  1. Print the distance-from-extreme distribution across the WHOLE anchor
     history first (registry 2026-08-31), with both percentile conventions.
  2. Ladder the gate from -0.25% out to no gate at all, on TLT / IEF / the
     flattener / SPY, and ask whether tightening the gate BUYS anything
     monotonically.  A non-monotone or flat ladder means the conditioner is not
     a conditioner and the tight cell is a 6-observation coincidence.
Entry MOC at decision-10td, exit MOC on the decision close (h=10, lag=0 from
the anchor position -- the anchor IS the entry session, so the forward return
is measured from that close).
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
lvl_incl = rolling_on_valid(tnx, lambda x: x.rolling(252).apply(
    lambda w: float((w <= w[-1]).mean()), raw=True))

d = px[["TLT", "IEF"]].pct_change().dropna()
BETA = float(np.polyfit(d["IEF"].values, d["TLT"].values, 1)[0])
FLAT = [("IEF", 1.0), ("TLT", -1.0 / BETA)]
VEH = {"TLT": [("TLT", 1.0)], "IEF": [("IEF", 1.0)], "FLAT": FLAT, "SPY": [("SPY", 1.0)]}

ev = load_events(["fomc_decision"])
fom = pd.DatetimeIndex(sorted(ev["date"].unique()))
pos, kept = anchor_positions(idx, fom, offset=-10)

print("=" * 110)
print("1. DISTANCE FROM EXTREME across the whole anchor history (print BEFORE stats)")
print("=" * 110)
dist = np.array([float(off_hi.iloc[p]) for p in pos])
ok = ~np.isnan(dist)
dist, kept2, pos2 = dist[ok], kept[ok], list(np.array(pos)[ok])
print("  measurable anchors: %d  span %s .. %s" % (len(dist), kept2[0].date(), kept2[-1].date()))
print("  off-high at the entry session (%%): min %.2f p10 %.2f p25 %.2f med %.2f p75 %.2f max %.2f"
      % tuple(100 * np.percentile(dist, [0, 10, 25, 50, 75, 100])))
print("  TODAY (2026-08-31, the live entry session) off-high = %+.5f%%" % (100 * off_hi.iloc[-1]))
for rung in (0.0000, 0.0025, 0.0100, 0.0200, 0.0500, 0.1000, 0.2000):
    m = dist >= -rung
    print("    off-high >= -%.2f%%  ->  %3d anchors (%4.1f%% of history) | midterm %d"
          % (100 * rung, int(m.sum()), 100 * m.mean(),
             int(((pd.DatetimeIndex(kept2).year % 4 == 2) & m).sum())))
print("  the 6 anchors of the tight cell:",
      [str(x.date()) for x in pd.DatetimeIndex(kept2)[dist >= -0.0025]])

print("\n" + "=" * 110)
print("2. GATE LADDER -- does tightening the conditioner BUY anything?")
print("   entry MOC at decision-10td close, exit MOC on the decision close (h=10)")
print("=" * 110)
for vk, legs in VEH.items():
    ret = vehicle_ret(px, legs, 10, 0)   # anchor session IS the entry close
    v_all = np.array([float(ret.iloc[p]) if p < len(idx) else np.nan for p in pos2])
    good = ~np.isnan(v_all)
    print("\n  --- %s ---   unconditional anchor mean %+.3f%% (N=%d) | all-days h=10 %+.3f%%"
          % (vk, 100 * np.nanmean(v_all), int(good.sum()),
             100 * float(ret.dropna().mean())))
    rows = []
    for rung in (0.0000, 0.0025, 0.0100, 0.0200, 0.0500, 0.1000, 0.2000, 9.99):
        m = good & (dist >= -rung)
        if m.sum() == 0:
            rows.append({"gate": "-%.2f%%" % (100 * rung), "n": 0})
            continue
        vv = v_all[m]
        r = summarize(vv, "off-hi>=-%.2f%%" % (100 * rung))
        w = int((vv > 0).sum())
        r["signp"] = round(sign_test(w, len(vv)), 4)
        mid = (pd.DatetimeIndex(kept2)[m].year % 4 == 2)
        r["mid_n"] = int(mid.sum())
        r["mid_mean"] = round(100 * vv[mid].mean(), 3) if mid.sum() else None
        rows.append(r)
    show(rows, "  gate ladder, %s" % vk)

print("\n" + "=" * 110)
print("3. DOSE RESPONSE the other way -- regress the anchor return on the DISTANCE")
print("   (if the conditioner is real the slope is signed and monotone)")
print("=" * 110)
for vk, legs in VEH.items():
    ret = vehicle_ret(px, legs, 10, 0)
    v_all = np.array([float(ret.iloc[p]) if p < len(idx) else np.nan for p in pos2])
    good = ~np.isnan(v_all)
    x, y = dist[good], v_all[good]
    qs = np.percentile(x, [20, 40, 60, 80])
    buckets = np.digitize(x, qs)
    line = "  %-5s quintiles of off-high (deepest -> at the high): " % vk
    line += "  ".join("%+.3f%%(N=%d)" % (100 * y[buckets == b].mean(), int((buckets == b).sum()))
                      for b in range(5))
    print(line)
    sl = np.polyfit(x, y, 1)[0]
    print("        OLS slope of return on off-high: %+.4f (return pp per 1.0 of off-high frac)"
          % (100 * sl))

print("\n" + "=" * 110)
print("4. TODAY'S JOINT STATE -- is 2026-09-16 even a normal FOMC anchor?")
print("=" * 110)
print("  live entry session 2026-08-31 is decision-10td for 2026-09-16:",
      "yes" if len(idx) - 1 == np.searchsorted(idx, pd.Timestamp("2026-08-31")) else "check")
print("  midterm anchors in the tight cell: 2022-11-02, 2006-05-10 (2 of 6)")
for vk, legs in VEH.items():
    ret = vehicle_ret(px, legs, 10, 0)
    v_all = np.array([float(ret.iloc[p]) if p < len(idx) else np.nan for p in pos2])
    good = ~np.isnan(v_all)
    tight = good & (dist >= -0.0025)
    vv = v_all[tight]
    print("  %-5s tight cell N=%d  mean %+.3f%%  record %d-%d  dates %s"
          % (vk, len(vv), 100 * vv.mean(), int((vv > 0).sum()), int((vv <= 0).sum()),
             [str(x.date()) for x in pd.DatetimeIndex(kept2)[tight]]))
    print("        per-episode: %s" % [round(100 * z, 2) for z in vv])
