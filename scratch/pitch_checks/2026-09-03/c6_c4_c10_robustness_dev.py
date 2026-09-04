"""Round 2 for the two kills that carry the most weight.

C4: the load-bearing claim is that the COUNT carries ZERO independent
information -- COUNT_HI never fires without XLI already at its own 5-day rank
floor. That must hold at floors and counts I did not pick, or the kill is an
artefact of my thresholds.

C10: the "filter subtracts" verdict must survive the definition neighbour --
the ABS-range percentile form instead of the REL-range form -- or it is a
threshold artefact rather than a filter that does not filter.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import (close_panel, declusters, load_prices, pct_rank, show,
                       summarize, vehicle_ret)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

# ===================== C4 ==================================================
print("=" * 78)
print("C4 ROUND 2 -- does the COUNT ever carry information XLI's own rank lacks?")
print("=" * 78)
COMPLEX = ["NSC", "UNP", "CSX", "DOV", "ITW", "PH", "MMM", "HON", "SNA", "IP",
           "GE", "CAT", "EMR"]
px_ind = load_prices(COMPLEX + ["XLI", "SPY"])
xli = px_ind["XLI"]["Close"].dropna()
cal = xli.index
R = pd.DataFrame({t: pct_rank(px_ind[t]["Close"].dropna(), 5, 252).reindex(cal)
                  for t in COMPLEX}, index=cal)
avail = R.notna().sum(axis=1)
usable = avail >= 10
xr = pct_rank(xli, 5, 252)

print("\nfloor  minCount |  count-trigger days | of which XLI is NOT at that floor")
for floor in (3.0, 5.0, 8.0, 10.0, 15.0):
    for k in (6, 8, 10, 12):
        cnt = (R <= floor).sum(axis=1)
        m = (cnt >= k) & usable
        xl = xr <= floor
        n, orphan = int(m.sum()), int((m & ~xl).sum())
        print(f"{floor:5.1f}  {k:8d} | {n:19d} | {orphan:d}"
              + ("   <-- count adds days" if orphan else "   (strict subset)"))

print("\nlive-state count at each floor (2026-09-02):")
for floor in (3.0, 5.0, 6.0, 8.0, 10.0):
    print(f"  floor {floor:4.1f}: {int((R.iloc[-1] <= floor).sum())} of "
          f"{int(avail.iloc[-1])} names, XLI r5 {xr.iloc[-1]:.1f}")

px = close_panel(["XLI", "SPY"])
px = px.loc[px.index.isin(cal)]
print("\nthe consequence: parent (XLI at its own floor) vs the count subset")
for floor, k in ((5.0, 10), (8.0, 10), (10.0, 12)):
    cnt = (R <= floor).sum(axis=1)
    m = (cnt >= k) & usable
    xl = (xr <= floor)
    for h in (5, 10):
        ret = vehicle_ret(px, [("XLI", 1.0)], h, 1)
        valid = ret.notna()
        rows = []
        for lbl, mm in (("parent XLI floor alone", xl),
                        ("count-ON subset", xl & m),
                        ("count-OFF complement", xl & ~m)):
            d = px.index[mm.reindex(px.index, fill_value=False).values & valid.values]
            e = declusters(d, h, px.index)
            r = summarize(ret.loc[e].values, f"floor{floor} k{k} h{h} {lbl}")
            r["n_days"] = len(d)
            rows.append(r)
        show(rows)

# ===================== C10 =================================================
print("\n" + "=" * 78)
print("C10 ROUND 2 -- 'the filter subtracts' under the ABS-range definition")
print("=" * 78)
raw = load_prices(["^SKEW", "^VIX", "SPY"])
skew = raw["^SKEW"]["Close"].dropna()
vix = raw["^VIX"]["Close"].dropna()
spy = raw["SPY"]["Close"].dropna()
cal2 = spy.index
rng21 = vix.rolling(21).max() - vix.rolling(21).min()
REL = (rng21 / vix.rolling(21).mean()).rolling(252).rank(pct=True).mul(100).reindex(cal2)
ABS = rng21.rolling(252).rank(pct=True).mul(100).reindex(cal2)
SK = pct_rank(skew, 21, 252).reindex(cal2)
px2 = pd.DataFrame({"SPY": spy})

for form, RG, rung in (("REL-range", REL, 15.0), ("ABS-range", ABS, 15.0),
                       ("ABS-range", ABS, 5.0)):
    for h in (5, 10):
        ret = vehicle_ret(px2, [("SPY", 1.0)], h, 1)
        valid = ret.notna()
        rows = []
        cells = {"skew r21>=95 ALONE": SK >= 95,
                 "JOINT (filter ON)": (SK >= 95) & (RG <= rung),
                 "filter OFF complement": (SK >= 95) & ~(RG <= rung)}
        for lbl, m in cells.items():
            d = cal2[m.fillna(False).values & valid.values]
            e = declusters(d, h, cal2)
            r = summarize(ret.loc[e].values, f"{form}<={rung:g} h{h} {lbl}")
            r["n_days"] = len(d)
            r["edge_pct"] = round(r["mean_pct"] - 100 * ret[valid].mean(), 3) if r["n"] else np.nan
            rows.append(r)
        show(rows)

print("\nlive readings: REL-range pctile %.2f   ABS-range pctile %.2f   "
      "skew r21 %.1f" % (REL.iloc[-1], ABS.iloc[-1], SK.iloc[-1]))

# does the SPY-off-its-high leg of watchlist 6 change anything?
print("\n" + "-" * 78)
print("watchlist 6's regime trigger: SPY >1% below its 52w high (live -1.64%, "
      "clears) AND non-midterm (live FAILS)")
hi = spy.rolling(252).max()
off = (spy / hi - 1.0) < -0.01
mid = pd.Series(cal2.year % 4 == 2, index=cal2)
for h in (5, 10):
    ret = vehicle_ret(px2, [("SPY", 1.0)], h, 1)
    valid = ret.notna()
    base = (SK >= 95) & (REL <= 15)
    rows = []
    for lbl, m in (("JOINT & SPY off-high & NON-midterm", base & off & ~mid),
                   ("JOINT & SPY off-high & MIDTERM (LIVE)", base & off & mid),
                   ("JOINT & SPY off-high (both)", base & off)):
        d = cal2[m.fillna(False).values & valid.values]
        e = declusters(d, h, cal2)
        r = summarize(ret.loc[e].values, f"h{h} {lbl}")
        r["n_days"] = len(d)
        rows.append(r)
    show(rows)

print("\nDONE round 2")
