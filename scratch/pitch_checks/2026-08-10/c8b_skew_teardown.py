"""C8 round 2 -- teardown of the one cell with a round-1 pulse.

Round 1 gave h=5 episodes +0.410% (N=35) vs own drift +0.240%, bootstrap
P(mean<=0)=0.019, sign p=0.0205. Four things to break:

  1. SIGN vs MECHANISM. The pitched mechanism is "no put wall -> fragile
     tape", which predicts a NEGATIVE forward return. h=5 is POSITIVE. If the
     long is taken instead, what story is left, and is it just "SPY at a high"?
  2. CO-LINEARITY. Is crushed SKEW at a high simply LOW VIX at a high?
  3. ERA FENCE. 2018+ is 6 episodes = Jan 2018 x3 + Apr/May 2026 x3. Two
     macro episodes wearing an era label.
  4. TODAY'S CELL. Today P/C is at the 50-69th pctile, not low. The SKEW-low
     / P/C-not-low half is the half that describes today.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

import pc_fear  # noqa: E402

px = close_panel(["SPY", "^SKEW", "^VIX"]).dropna()
idx = px.index
spy, skew, vix = px["SPY"], px["^SKEW"], px["^VIX"]

skew_lvl = skew.rolling(252).rank(pct=True) * 100
vix_lvl = vix.rolling(252).rank(pct=True) * 100
spy_dist = spy / spy.rolling(252).max() - 1.0
m_hi = (spy_dist >= -0.005).fillna(False)
m_sk = (skew_lvl <= 10).fillna(False)
m_vx = (vix_lvl <= 10).fillna(False)
m = (m_hi & m_sk).fillna(False)

pcs = pc_fear.pct_series()
pc = pcs.reindex(idx, method="ffill").shift(1)


def epi_stats(mask, h, label, span_lo=None):
    r = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    val = r.notna()
    d = idx[mask.reindex(idx, fill_value=False).values & val.values]
    if span_lo is not None:
        d = d[d >= span_lo]
    if len(d) == 0:
        return {"label": label, "n": 0}, pd.DatetimeIndex([]), np.array([])
    e = declusters(d, h, idx)
    v = r.loc[e].values
    s = summarize(v, label)
    s["n_days"] = len(d)
    return s, e, v


print("=" * 78)
print("1. SIGN vs MECHANISM. 'No put wall' predicts DOWN. What does the tape do?")
print("=" * 78)
rows = []
for h in (1, 2, 3, 5, 7, 10, 15, 21):
    s, e, v = epi_stats(m, h, f"h={h}")
    r = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    s["ctrl_pct"] = round(100 * r.dropna().mean(), 3)
    s["edge_pp"] = round(s["mean_pct"] - s["ctrl_pct"], 3)
    s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(s)
show(rows, "SKEW-crush x SPY-high, LONG SPY, episode level, by horizon")
print("  The pitched mechanism is bearish. Every positive cell above is the")
print("  mechanism FAILING, and the sign of the edge over control flips at")
print("  h=7 / h=10 / h=21. An effect with a dealer-positioning mechanism")
print("  should decay monotonically, not oscillate.")

print("\n" + "=" * 78)
print("2. CO-LINEARITY: is crushed SKEW just LOW VIX at a high?")
print("=" * 78)
print(f"corr(SKEW lvl pctile, VIX lvl pctile) = "
      f"{skew_lvl.corr(vix_lvl):.3f}")
print(f"today: SKEW lvlpct {skew_lvl.iloc[-1]:.1f}, VIX lvlpct "
      f"{vix_lvl.iloc[-1]:.1f}  -> BOTH bottom-decile")
h = 5
rows = []
for lbl, mm in [
    ("SKEW-low & VIX-low & high", (m & m_vx).fillna(False)),
    ("SKEW-low, VIX NOT low, high", (m & ~m_vx).fillna(False)),
    ("VIX-low, SKEW NOT low, high", (m_vx & m_hi & ~m_sk).fillna(False)),
    ("SPY-high, neither low", (m_hi & ~m_sk & ~m_vx).fillna(False)),
    ("SPY-high alone (parent)", m_hi),
]:
    s, e, v = epi_stats(mm, h, lbl)
    rows.append(s)
r5 = vehicle_ret(px, [("SPY", 1.0)], 5, 1).dropna()
rows.append(summarize(r5.values, "CTRL all days"))
show(rows, "h=5 episodes -- which low-vol gate is doing the work?")
print("  TODAY sits in the SKEW-low & VIX-low cell.")

print("\n" + "=" * 78)
print("3. ERA FENCE: 2018+ is +1.389% on 6 episodes. Which episodes?")
print("=" * 78)
s, e, v = epi_stats(m, 5, "h=5")
post = pd.DatetimeIndex(e)[pd.DatetimeIndex(e) >= "2018-01-01"]
print("2018+ episodes:", ", ".join(f"{d.date()} {100*x:+.2f}%"
                                   for d, x in zip(post, v[np.array(e >= pd.Timestamp('2018-01-01'))])))
print("  -> two clusters, Jan 2018 and Apr/May 2026. Not an era, two episodes.")
mask_j18 = (pd.DatetimeIndex(e) >= "2018-01-01") & (pd.DatetimeIndex(e) < "2018-02-01")
mask_26 = pd.DatetimeIndex(e).year == 2026
show([summarize(v, "all episodes"),
      summarize(v[~mask_j18], "drop Jan-2018 cluster"),
      summarize(v[~mask_26], "drop 2026 cluster"),
      summarize(v[~(mask_j18 | mask_26)], "drop BOTH clusters")],
     "drop-cluster sensitivity, h=5 episodes")
yrs = pd.DatetimeIndex(e).year
top3 = pd.Series(v).groupby(yrs.values).size().sort_values(ascending=False).head(3)
print(f"  3 biggest years by episode count: {dict(top3)} = "
      f"{int(top3.sum())} of {len(v)} episodes")

print("\n" + "=" * 78)
print("4. TODAY'S CELL: P/C is NOT low today, so which half is today in?")
print("=" * 78)
pc_lo = (pc <= 10).fillna(False)
pc_lo25 = (pc <= 25).fillna(False)
ok = pc.notna()
lo_span = idx[ok][0]
for h in (5, 10):
    rows = []
    for lbl, mm in [
        ("A: SKEW-low & P/C<=10 (confirmed)", (m & pc_lo).fillna(False)),
        ("B: SKEW-low & P/C>10  <-- TODAY", (m & ~pc_lo & ok).fillna(False)),
        ("B': SKEW-low & P/C>25 (stricter)", (m & ~pc_lo25 & ok).fillna(False)),
        ("SPY-high, P/C era, no SKEW gate", (m_hi & ok).fillna(False)),
    ]:
        s, e2, v2 = epi_stats(mm, h, lbl, span_lo=lo_span)
        if s["n"]:
            s["sign_p"] = round(sign_test(int((v2 > 0).sum()), len(v2)), 4)
        rows.append(s)
    rr = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    rows.append(summarize(rr[rr.notna() & ok].values, "CTRL all days, P/C era"))
    show(rows, f"P/C halves, h={h}, episodes (P/C era 2007-11+)")
print(f"  today: SKEW lvlpct {skew_lvl.iloc[-1]:.1f}, P/C pctile "
      f"{pc.iloc[-1]:.1f} -> cell B.")

print("\n" + "=" * 78)
print("5. FRIDAY-ANCHOR PLACEBO. Today's signal date is Fri 2026-08-07,")
print("   entry MOC Mon 2026-08-10 -- one session INTO the cluster.")
print("=" * 78)
r = vehicle_ret(px, [("SPY", 1.0)], 5, 1)
d = idx[m.values & r.notna().values]
print(f"cell day-level all weekdays: {100*r.loc[d].mean():+.3f}% (N={len(d)})")
for wd, nm in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
    sel = d[d.weekday == wd]
    if len(sel):
        print(f"  anchor {nm}: {100*r.loc[sel].mean():+.3f}%  (N={len(sel)}, "
              f"hit {100*(r.loc[sel] > 0).mean():.0f}%)")

print("\n" + "=" * 78)
print("6. CLUSTER DEPTH: what does the Nth consecutive trigger session pay?")
print("   (registry: mid-cluster entry is not a fresh trigger; today is #4)")
print("=" * 78)
pos = pd.Series(range(len(idx)), index=idx)
depth = np.zeros(len(idx), dtype=int)
run = 0
for i, v_ in enumerate(m.values):
    run = run + 1 if v_ else 0
    depth[i] = run
dep = pd.Series(depth, index=idx)
rows = []
for lo, hi_ in [(1, 1), (2, 3), (4, 99)]:
    sel = idx[(dep >= lo).values & (dep <= hi_).values & m.values
              & r.notna().values]
    if len(sel):
        rows.append(summarize(r.loc[sel].values,
                              f"cluster depth {lo}-{hi_} (day-level N={len(sel)})"))
show(rows, "h=5 by cluster depth")
print(f"  TODAY'S DEPTH = {int(dep.iloc[-1])}")
