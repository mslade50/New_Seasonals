"""C3 round 1 -- long duration into CPI while bond vol sits in its bottom decile.

Two things get tested before any edge is measured:
 0. DOES THE CONDITIONER EVEN FIRE TODAY under the natural definition?
    The map calls ^MOVE "at a floor" from a 5d RETURN rank of 7.5 and
    "-37.4% below its 52w high". A floor is a LEVEL statement. ^MOVE's level
    sits at the 45.6th percentile of its own trailing year.
 1. Leg separation: MOVE-low alone, TLT-at-floor alone, both, and both with
    NO CPI anchor -- so it is visible which leg does the work.

Plus the count-first credit question (HYG at 52w high AND LQD near its 52w
low around CPI): COUNT the joint state before measuring anything.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["TLT", "IEF", "^MOVE", "HYG", "LQD", "SPY", "^TNX"])
pxm = px.dropna(subset=["TLT", "^MOVE"])
idx = pxm.index
ev = load_events(["cpi", "ppi"])


def anchor_mask(kind: str, offset: int = -2, on=None) -> pd.Series:
    on = idx if on is None else on
    m = pd.Series(False, index=on)
    for x in ev[ev.event == kind]["date"]:
        p = int(on.searchsorted(x, side="left"))
        if 0 <= p + offset < len(on):
            m.iloc[p + offset] = True
    return m


mv = pxm["^MOVE"]
tlt = pxm["TLT"]

# ---------------------------------------------------------------- 0. today
lvl_pct = mv.rolling(252).apply(lambda w: (w <= w[-1]).mean(), raw=True) * 100
r5rank = pct_rank(mv, 5)
dist_hi = mv / mv.rolling(252).max() - 1.0
dist_lo_tlt = tlt / tlt.rolling(252).min() - 1.0

print("=" * 88)
print("0. DOES THE CONDITIONER FIRE TODAY?  (a rank is not a level)")
print("=" * 88)
print(f"  ^MOVE level                 {mv.iloc[-1]:.2f}")
print(f"  ^MOVE LEVEL pctile 252d     {lvl_pct.iloc[-1]:.1f}  "
      f"-> bottom-decile floor? {'YES' if lvl_pct.iloc[-1] < 10 else 'NO'}")
print(f"  ^MOVE 5d-return rank        {r5rank.iloc[-1]:.1f}  "
      f"-> bottom decile? {'YES' if r5rank.iloc[-1] < 10 else 'NO'}")
print(f"  ^MOVE dist from 52w high    {100*dist_hi.iloc[-1]:.1f}%")
print(f"  TLT  dist from 52w low      {100*dist_lo_tlt.iloc[-1]:.2f}%")
print("\n  Of the 5849 sessions with a 252d history, how often does a "
      "5d-rank<10 coincide with a LEVEL pctile<10?")
a = r5rank < 10
b = lvl_pct < 10
both = a & b
print(f"    5d-rank<10: {int(a.sum())}   level<10: {int(b.sum())}   "
      f"BOTH: {int(both.sum())}  -> P(level floor | 5d crush) = "
      f"{100*float(both.sum()/max(a.sum(),1)):.1f}%")
print("  So 'crushed this week' and 'at a floor' are different states, and "
      "today is the FIRST, not the second.")

# ---------------------------------------------------------------- 1. legs
gates = {
    "A1 MOVE level pctile<10 (a real floor)": b,
    "A2 MOVE 5d-rank<10 (TODAY'S state)": a,
    "A3 MOVE <=-30% from 52w high": dist_hi <= -0.30,
    "B  TLT within 3% of 52w low": dist_lo_tlt <= 0.03,
}
for k, g in gates.items():
    print(f"  {k}: {int(g.sum())} days ({100*g.mean():.1f}%)  fires today: "
          f"{bool(g.iloc[-1])}")

m_cpi = anchor_mask("cpi", -2)
H = 3

print("\n" + "=" * 88)
print(f"1. LEG SEPARATION, long TLT, h={H}, entry lag=1 (MOC before the print)")
print("=" * 88)
cases = {
    "CPI anchor alone (no state gate)": m_cpi,
    "A2 MOVE-crush alone (no event)": gates["A2 MOVE 5d-rank<10 (TODAY'S state)"],
    "B TLT-floor alone (no event)": gates["B  TLT within 3% of 52w low"],
    "A2 + B (no event)": gates["A2 MOVE 5d-rank<10 (TODAY'S state)"] & gates["B  TLT within 3% of 52w low"],
    "CPI + A2 (today's stated cell, MOVE crush)": m_cpi & gates["A2 MOVE 5d-rank<10 (TODAY'S state)"],
    "CPI + A1 (the LEVEL floor version)": m_cpi & gates["A1 MOVE level pctile<10 (a real floor)"],
    "CPI + B": m_cpi & gates["B  TLT within 3% of 52w low"],
    "CPI + A2 + B (the full stated interaction)": m_cpi & gates["A2 MOVE 5d-rank<10 (TODAY'S state)"] & gates["B  TLT within 3% of 52w low"],
    "CPI + A1 + B": m_cpi & gates["A1 MOVE level pctile<10 (a real floor)"] & gates["B  TLT within 3% of 52w low"],
}
r = vehicle_ret(pxm, [("TLT", 1.0)], H, 1)
allr = r.dropna()
rows = []
for lbl, m in cases.items():
    d = idx[m.reindex(idx, fill_value=False).values & r.notna().values]
    d = declusters(d, 10, allr.index)
    if len(d) == 0:
        rows.append({"label": lbl, "n": 0})
        continue
    v = r.loc[d].values
    s = summarize(v, lbl)
    s["edge_pp"] = round(s["mean_pct"] - 100 * allr.mean(), 3)
    w = int((v > 0).sum())
    s["sign_p"] = round(sign_test(w, len(v)), 4)
    s["boot"] = round(bootstrap_p_le0(v), 3) if len(v) >= 3 else np.nan
    mid = d.year % 4 == 2
    s["mid_mean"] = round(100 * v[mid].mean(), 3) if mid.sum() else np.nan
    s["mid_n"] = int(mid.sum())
    rows.append(s)
show(rows, f"TLT long, h={H}: which leg does the work? "
           f"(all-days ctrl {100*allr.mean():+.3f}%)")

print("\nSame table for h=1 and h=5:")
for HH in (1, 5):
    r2 = vehicle_ret(pxm, [("TLT", 1.0)], HH, 1)
    a2 = r2.dropna()
    rr = []
    for lbl, m in cases.items():
        d = idx[m.reindex(idx, fill_value=False).values & r2.notna().values]
        d = declusters(d, 10, a2.index)
        if len(d) == 0:
            rr.append({"label": lbl, "n": 0})
            continue
        v = r2.loc[d].values
        s = summarize(v, lbl)
        s["edge_pp"] = round(s["mean_pct"] - 100 * a2.mean(), 3)
        rr.append(s)
    show(rr, f"h={HH} (ctrl {100*a2.mean():+.3f}%)")

print("\n" + "=" * 88)
print("2. DETAIL on the full stated interaction (CPI + MOVE crush + TLT floor)")
print("=" * 88)
full = m_cpi & gates["A2 MOVE 5d-rank<10 (TODAY'S state)"] & gates["B  TLT within 3% of 52w low"]
battery(pxm, full, [("TLT", 1.0)], h=3,
        title="TLT long into CPI, MOVE crushed, TLT at 52w floor",
        cost_bps=2.0, lag=1, min_gap=10, event_kinds=("ppi",))

print("\n" + "=" * 88)
print("3. COUNT-FIRST: HYG at its 52w high AND LQD near its 52w low, at CPI")
print("=" * 88)
pc = px.dropna(subset=["HYG", "LQD"])
ic = pc.index
hyg_hi = pc["HYG"] / pc["HYG"].rolling(252).max() - 1.0
lqd_lo = pc["LQD"] / pc["LQD"].rolling(252).min() - 1.0
print(f"TODAY: HYG {100*hyg_hi.iloc[-1]:.2f}% from 52w high, "
      f"LQD {100*lqd_lo.iloc[-1]:.2f}% above 52w low")
for hi_tol, lo_tol in [(0.000, 0.02), (0.005, 0.03), (0.01, 0.05), (0.02, 0.10)]:
    st = (hyg_hi >= -hi_tol) & (lqd_lo <= lo_tol)
    mc = anchor_mask("cpi", -2, on=ic)
    joint = st & mc
    epi = declusters(ic[joint.values], 10, ic)
    print(f"  HYG within {100*hi_tol:.1f}% of high & LQD within "
          f"{100*lo_tol:.0f}% of low: state days {int(st.sum())}, "
          f"AT a CPI anchor {int(joint.sum())}, episodes {len(epi)}"
          + (f"  dates {[str(d.date()) for d in epi]}" if len(epi) <= 12 else ""))
print("\n  VERDICT RULE (given up front): under ~8 episodes this is "
      "unmeasurable, not merely small.")
