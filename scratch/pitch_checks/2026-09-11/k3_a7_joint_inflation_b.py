"""A7 round 2 - how many INDEPENDENT episodes does the joint state actually
have, and does the one cell with a real sign test (IWM h=1, sign p 0.0068 on
'N=33') survive declustering? At h=1 the battery's min_gap defaults to h, i.e.
gap=1, which is NO declustering at all.
"""
import sys
from pathlib import Path

ROOTP = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOTP))
import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
from pitch_lab import *  # noqa: E402,F403

TK = ["SPY", "IWM", "XLE", "GLD", "TLT", "IEF", "LQD", "DBC"]
px = close_panel(TK)
cal = px["SPY"].dropna().index
px = px.reindex(cal)
SHOCK = {2007, 2008, 2021, 2022}


def near_high(t, pct):
    s = px[t]
    mx = rolling_on_valid(s, lambda x: x.rolling(252).max())
    return ((mx - s) / mx * 100 <= pct) & mx.notna()


def near_low(t, pct):
    s = px[t]
    mn = rolling_on_valid(s, lambda x: x.rolling(252).min())
    return ((s - mn) / mn * 100 <= pct) & mn.notna()


dbc_hi = near_high("DBC", 0.5)
ig_lo = near_low("IEF", 1.0) & near_low("LQD", 1.0)
joint = dbc_hi & ig_lo
sig = cal[joint.reindex(cal, fill_value=False).values]

print("=" * 78)
print("A. HOW MANY INDEPENDENT EPISODES DOES THE JOINT STATE HAVE?")
print(f"  trigger DAYS by year: "
      f"{dict(pd.Series(sig.year).value_counts().sort_index())}")
for gap in (1, 5, 10, 21, 63):
    e = declusters(sig, gap, cal)
    print(f"  gap={gap:>3} td -> {len(e):>3} episodes   "
          f"{[str(d.date()) for d in e]}")
print("\n  The state's ENTIRE pre-2026 history is two clusters: spring 2018")
print("  and Jan-Jun 2022. 2022 is one of the four inflation-shock years the")
print("  registry already named as holding >100% of the commodity-high parent.")

print("\n" + "=" * 78)
print("B. THE ONE CELL WITH A REAL SIGN TEST: LONG IWM h=1")
r = vehicle_ret(px, [("IWM", 1.0)], 1, 1)
b = r.dropna()
for gap in (1, 5, 10, 21, 63):
    e = declusters(sig.intersection(b.index), gap, cal)
    v = r.loc[e].values
    w = int((v > 0).sum())
    print(f"  gap={gap:>3}: N={len(v):>3}  mean {100*v.mean():+.3f}%  "
          f"edge {100*(v.mean()-b.mean()):+.3f}pp  hit "
          f"{100*(v>0).mean():5.1f}%  record {w}-{len(v)-w}  "
          f"sign p {sign_test(w, len(v)):.4f}")
e21 = declusters(sig.intersection(b.index), 21, cal)
print(f"  at gap=21 the 'N=33' collapses to {len(e21)} independent "
      f"observations, which is what 3 calendar clusters can support.")
print("\n  per-cluster mean (gap=21 grouping):")
for yr in sorted(set(sig.year)):
    d = sig[sig.year == yr].intersection(b.index)
    if len(d) == 0:
        continue
    print(f"    {yr}: N_days={len(d):>3}  mean {100*r.loc[d].mean():+.3f}%  "
          f"hit {100*(r.loc[d]>0).mean():.1f}%")

print("\n" + "=" * 78)
print("C. GATE ATTRIBUTION FOR IWM h=1 (does the JOINT beat both halves?)")
rows = []
for lbl, m in [("DBC hi ALONE", dbc_hi), ("IG lo ALONE", ig_lo),
               ("JOINT", joint), ("DBChi & NOT IGlo", dbc_hi & ~ig_lo),
               ("IGlo & NOT DBChi", ig_lo & ~dbc_hi),
               ("neither", ~dbc_hi & ~ig_lo)]:
    d = cal[m.reindex(cal, fill_value=False).values].intersection(b.index)
    e = declusters(d, 21, cal)
    if len(e) == 0:
        continue
    v = r.loc[e].values
    rows.append({"cell": lbl, "N_days": len(d), "N_epi21": len(e),
                 "mean_pct": round(100 * v.mean(), 3),
                 "edge_pp": round(100 * (v.mean() - b.mean()), 3),
                 "hit": round(100 * (v > 0).mean(), 1)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 78)
print("D. EX-SHOCK AND EX-2026 (what is actually out of sample?)")
d = sig.intersection(b.index)
v = r.loc[d].values
yrs = d.year
for lbl, m in [("all trigger days", np.ones(len(d), bool)),
               ("ex 2022 (shock yr)", ~np.isin(yrs, [2022])),
               ("ex 2026 (the live state)", ~np.isin(yrs, [2026])),
               ("ex shock AND ex 2026",
                ~np.isin(yrs, list(SHOCK)) & ~np.isin(yrs, [2026]))]:
    if m.sum() == 0:
        print(f"  {lbl:<26} N=0  -- EMPTY")
        continue
    w = int((v[m] > 0).sum())
    print(f"  {lbl:<26} N_days={int(m.sum()):>3}  mean "
          f"{100*v[m].mean():+.3f}%  hit {100*(v[m]>0).mean():5.1f}%  "
          f"record {w}-{int(m.sum())-w}  sign p "
          f"{sign_test(w, int(m.sum())):.4f}  years {sorted(set(yrs[m]))}")

print("\n" + "=" * 78)
print("E. COST on the one surviving-looking cell (IWM h=1 long)")
print("  IWM MOC round trip ~2 bps. Gap=21 episode mean above is the number;")
print("  a 1-session hold has to clear 10 bps to be 5x cost.")
