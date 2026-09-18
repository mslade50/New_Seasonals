"""Round-2 decisive probe.

The h=3 forms of C7/C8/C9 are dead, but the horizon scan put C7's DBC/USO at
h=10 (+0.78/+1.40pp edge) and C8's short-duration at 10-of-10 positive
horizons. Before I sign a kill I have to know whether ANY of that belongs to
the print gate, or whether it is the commodity-momentum state with a calendar
label stapled on. Three questions:

  A. Is the print gate vacuous at long h? (a 10-td window catches a print
     almost always, so "print in window" stops being a condition)
  B. Gate attribution at the horizon where each cell peaks: state+print vs
     state-no-print, day AND episode level.
  C. Regime concentration: the inflation era. A 2021-22 (or 2007-08)
     concentration is fatal per the brief.
  D. Placebo ladder AT THE PEAK HORIZON, not just at h=3.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["DBC", "USO", "TLT", "IEF", "SPY"])
raw = load_prices(["DBC", "TLT"])
IDX = px.index


def dist_to_high(t, n=252):
    s = raw[t]["Close"]
    hi = rolling_on_valid(s, lambda x: x.rolling(n).max())
    return (1.0 - s / hi).reindex(IDX)


def dist_above_low(t, n=252):
    s = raw[t]["Close"]
    lo = rolling_on_valid(s, lambda x: x.rolling(n).min())
    return (s / lo - 1.0).reindex(IDX)


def flag_in_window(kinds, h, lag=1, k=0):
    ev = load_events(list(kinds))["date"]
    pos, _ = anchor_positions(IDX, ev, offset=k)
    e = np.asarray(pd.DatetimeIndex([IDX[p] for p in pos]).values, dtype="datetime64[ns]")
    out = np.zeros(len(IDX), dtype=bool)
    for i in range(len(IDX)):
        if i + lag + h >= len(IDX):
            continue
        out[i] = bool(((e > np.datetime64(IDX[i + lag])) &
                       (e <= np.datetime64(IDX[i + lag + h]))).any())
    return pd.Series(out, index=IDX)


LAG = 1
d = dist_to_high("DBC")
state = d <= 0.0025

print("=" * 78)
print("A. IS THE PRINT GATE VACUOUS AT LONG HORIZONS?")
print("=" * 78)
print(f"{'h':>3} {'days w/ print in window':>24} {'% of all days':>14} "
      f"{'state days w/ print':>20} {'% of state days':>16}")
for h in (1, 2, 3, 5, 8, 10):
    pr = flag_in_window(("ppi", "cpi"), h, LAG)
    tot = len(IDX)
    print(f"{h:>3} {int(pr.sum()):>24} {100*pr.mean():>13.1f}% "
          f"{int((state & pr).sum()):>20} "
          f"{100*(state & pr).sum()/max(1,int(state.sum())):>15.1f}%")

print("\n" + "=" * 78)
print("B. GATE ATTRIBUTION AT EACH CELL'S PEAK HORIZON  (day AND episode)")
print("=" * 78)
for lbl, legs, h in (("C7 DBC", [("DBC", 1.0)], 10),
                     ("C7 USO", [("USO", 1.0)], 10),
                     ("C8 SHORT TLT", [("TLT", -1.0)], 8),
                     ("C8 SHORT IEF", [("IEF", -1.0)], 5)):
    pr = flag_in_window(("ppi", "cpi"), h, LAG)
    ret = vehicle_ret(px, legs, h, LAG)
    valid = ret.notna()
    rows = []
    for m, nm in ((state & pr, "state AND print (the cell)"),
                  (state & ~pr, "state, NO print"),
                  (state, "state, ANY calendar"),
                  (pr & ~state, "print, NO state"),
                  (pd.Series(True, index=IDX), "all days")):
        dd = IDX[m.values & valid.values]
        if len(dd) == 0:
            rows.append({"label": nm + " [day]", "n": 0}); continue
        e = declusters(dd, h, IDX)
        rows.append(summarize(ret.loc[dd].values, nm + " [day]"))
        rows.append(summarize(ret.loc[e].values, nm + " [epi]"))
    show(rows, f"{lbl}  h={h}")
    dd = IDX[(state & pr).values & valid.values]
    e0 = declusters(dd, h, IDX)
    d1 = IDX[(state & ~pr).values & valid.values]
    e1 = declusters(d1, h, IDX)
    print(f"  PRINT-GATE CONTRIBUTION  day {100*(ret.loc[dd].mean()-ret.loc[d1].mean()):+.3f}pp | "
          f"epi {100*(ret.loc[e0].mean()-ret.loc[e1].mean()):+.3f}pp   "
          f"(gate keeps {len(dd)} of {len(dd)+len(d1)} state days)")
    print(f"  concentration: {cluster_note(e0, ret.loc[e0].values)}")

print("\n" + "=" * 78)
print("C. REGIME CONCENTRATION -- per-year episode sums on the live cells")
print("=" * 78)
for lbl, legs, h in (("C7 DBC h=10", [("DBC", 1.0)], 10),
                     ("C7 USO h=10", [("USO", 1.0)], 10),
                     ("C8 SHORT TLT h=8", [("TLT", -1.0)], 8),
                     ("C8 SHORT IEF h=5", [("IEF", -1.0)], 5),
                     ("C8 SHORT TLT h=3", [("TLT", -1.0)], 3)):
    pr = flag_in_window(("ppi", "cpi"), h, LAG)
    ret = vehicle_ret(px, legs, h, LAG)
    valid = ret.notna()
    dd = IDX[(state & pr).values & valid.values]
    e = declusters(dd, h, IDX)
    v = ret.loc[e].values
    yr = pd.Series(v, index=pd.DatetimeIndex(e)).groupby(pd.DatetimeIndex(e).year)
    tot = v.sum()
    print(f"\n{lbl}: total {100*tot:+.2f}pp over {len(v)} episodes, "
          f"mean {100*v.mean():+.3f}%")
    print("   by year (pp): " + ", ".join(
        f"{y}:{100*s:+.1f}" for y, s in yr.sum().items()))
    infl = pd.DatetimeIndex(e).year.isin([2007, 2008, 2021, 2022])
    print(f"   the four inflation-shock years (2007,2008,2021,2022) = "
          f"{int(infl.sum())} of {len(v)} episodes and "
          f"{100*v[infl].sum():+.2f}pp of {100*tot:+.2f}pp "
          f"({100*v[infl].sum()/tot*100 if tot else float('nan'):.0f}%)")
    rest = v[~infl]
    if len(rest):
        print(f"   EX those four years: n={len(rest)}, mean {100*rest.mean():+.3f}%, "
              f"hit {100*(rest>0).mean():.1f}%, "
              f"sign p {sign_test(int((rest>0).sum()), len(rest)):.4f}")

print("\n" + "=" * 78)
print("D. PLACEBO ANCHOR LADDER AT THE PEAK HORIZON")
print("=" * 78)
for lbl, legs, h in (("C7 DBC h=10", [("DBC", 1.0)], 10),
                     ("C7 USO h=10", [("USO", 1.0)], 10),
                     ("C8 SHORT TLT h=8", [("TLT", -1.0)], 8),
                     ("C8 SHORT IEF h=5", [("IEF", -1.0)], 5)):
    ret = vehicle_ret(px, legs, h, LAG)
    valid = ret.notna()
    rows, means = [], []
    for k in range(-5, 6):
        m = state & flag_in_window(("ppi", "cpi"), h, LAG, k)
        dd = IDX[m.values & valid.values]
        e = declusters(dd, h, IDX)
        r = summarize(ret.loc[e].values, f"k={k:+d}" + ("  <-- TRUE" if k == 0 else ""))
        rows.append(r); means.append(r.get("mean_pct", np.nan))
    show(rows, lbl)
    print(f"  true anchor ranks {int(np.sum(np.asarray(means) >= means[5]))} of 11")

print("\n" + "=" * 78)
print("E. THE LIVE JOINT STATE for C8: TLT within 2% of its OWN 252d low")
print("=" * 78)
mom = dist_above_low("TLT") <= 0.02
print(f"live TLT distance above 252d low = {100*dist_above_low('TLT').iloc[-1]:.2f}%")
for legs, nm in (([("TLT", -1.0)], "SHORT TLT"), ([("IEF", -1.0)], "SHORT IEF")):
    rows = []
    for h in (3, 5, 8, 10):
        pr = flag_in_window(("ppi", "cpi"), h, LAG)
        ret = vehicle_ret(px, legs, h, LAG)
        valid = ret.notna()
        for m, tag in ((state & pr, "state+print"),
                       (state & pr & mom, "state+print+TLT@low (FULLY LIVE)")):
            dd = IDX[m.values & valid.values]
            if len(dd) == 0:
                rows.append({"label": f"h={h} {tag}", "n": 0}); continue
            e = declusters(dd, h, IDX)
            rows.append(summarize(ret.loc[e].values, f"h={h} {tag}"))
    show(rows, nm)
