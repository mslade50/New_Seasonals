"""C9 round 1 -- synchronized 52w high across SPY, EFA and HYG.

THE OBVIOUS KILL: this is "SPY at a 52w high" with decoration. So the control
here is NOT the unconditional drift alone -- it is SPY-at-a-52w-high, run on
the SAME span (HYG starts 2007-04). If the EFA+HYG confirmation legs add under
~20 bps over that control, the filter does not filter and it dies.

Pre-specified: all three within 0.5% of their own 252d high on the same
session; vehicle long SPY; entry lag=1 MOC-tomorrow; h=10 and h=5 both shown
(no horizon prior stated in the map, so the h-scan is labelled a scan).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["SPY", "EFA", "HYG", "IWM", "EWJ"]
px = close_panel(TK).dropna()
idx = px.index
print(f"panel span (HYG-limited): {idx[0].date()} .. {idx[-1].date()}  n={len(idx)}")

TOL = -0.005


def near_high(s: pd.Series, tol: float = TOL) -> pd.Series:
    return (s / s.rolling(252).max() - 1.0) >= tol


nh = {t: near_high(px[t]) for t in TK}
for t in TK:
    d = 100 * (px[t].iloc[-1] / px[t].rolling(252).max().iloc[-1] - 1)
    print(f"  {t:5s} dist52wh {d:+.3f}%  near-high days {int(nh[t].sum())}")

m_spy = nh["SPY"].fillna(False)                       # CONTROL cell
m_tri = (nh["SPY"] & nh["EFA"] & nh["HYG"]).fillna(False)   # TREATMENT
m_sh = (nh["SPY"] & nh["HYG"]).fillna(False)
m_se = (nh["SPY"] & nh["EFA"]).fillna(False)
m_all5 = (nh["SPY"] & nh["EFA"] & nh["HYG"] & nh["IWM"] & nh["EWJ"]).fillna(False)

print(f"\nSPY alone {int(m_spy.sum())} | +HYG {int(m_sh.sum())} | "
      f"+EFA {int(m_se.sum())} | TRIPLE {int(m_tri.sum())} | all five "
      f"{int(m_all5.sum())}")
print(f"  conditional prob(triple | SPY at high) = "
      f"{m_tri.sum()/max(m_spy.sum(),1):.3f}  <- a filter that keeps this "
      f"share of its parent is barely a filter")

# cluster depth today
for lbl, mm in [("TRIPLE", m_tri), ("SPY alone", m_spy)]:
    run = 0
    for v in mm.values[::-1]:
        if v:
            run += 1
        else:
            break
    print(f"CLUSTER DEPTH TODAY, {lbl}: {run} consecutive sessions")

variants = {
    "tol 0.1%": (near_high(px["SPY"], -0.001) & near_high(px["EFA"], -0.001)
                 & near_high(px["HYG"], -0.001)).fillna(False),
    "tol 1.0%": (near_high(px["SPY"], -0.01) & near_high(px["EFA"], -0.01)
                 & near_high(px["HYG"], -0.01)).fillna(False),
    "tol 2.0%": (near_high(px["SPY"], -0.02) & near_high(px["EFA"], -0.02)
                 & near_high(px["HYG"], -0.02)).fillna(False),
    "GATE ATTR: SPY+HYG only": m_sh,
    "GATE ATTR: SPY+EFA only": m_se,
    "GATE ATTR: SPY alone (CONTROL)": m_spy,
    "all five at highs": m_all5,
}

for h in (5, 10):
    battery(px, m_tri, [("SPY", 1.0)], h=h,
            title=f"LONG SPY | SPY+EFA+HYG all within 0.5% of 52w high, h={h}",
            cost_bps=2.0, lag=1, min_gap=h, event_kinds=("cpi",),
            variants=variants if h == 10 else None)

print("\n" + "=" * 78)
print("HEAD TO HEAD: treatment minus SPY-alone control, SAME SPAN, episodes")
print("=" * 78)
for h in (1, 2, 3, 5, 10, 21):
    r = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    val = r.notna()
    dt_ = idx[m_tri.values & val.values]
    dc = idx[m_spy.values & val.values]
    if len(dt_) == 0:
        continue
    lo = dt_[0]
    et = declusters(dt_, h, idx)
    ec = declusters(dc[dc >= lo], h, idx)
    vt, vc = r.loc[et].values, r.loc[ec].values
    base = r[val & (idx >= lo)]
    se = np.sqrt(vt.var(ddof=1)/len(vt) + vc.var(ddof=1)/len(vc))
    print(f"h={h:2d}  TRIPLE {100*vt.mean():+.3f}% (N={len(vt)})   "
          f"SPY-ALONE {100*vc.mean():+.3f}% (N={len(vc)})   "
          f"ADD {100*(vt.mean()-vc.mean()):+.3f}pp  welch t {(vt.mean()-vc.mean())/se:+.2f}   "
          f"| uncond same span {100*base.mean():+.3f}%  "
          f"triple-over-uncond {100*(vt.mean()-base.mean()):+.3f}pp")

print("\n" + "=" * 78)
print("SCAN (multiplicity applies): horizon 1..21 on the triple cell")
print("=" * 78)
show(horizon_scan(px, idx[m_tri.values], [("SPY", 1.0)],
                  hs=(1, 2, 3, 5, 7, 10, 15, 21)), "triple-high, long SPY")

print("\n" + "=" * 78)
print("MIDTERM SPLIT + ERA + YEAR HISTOGRAM (h=10 episodes)")
print("=" * 78)
h = 10
r = vehicle_ret(px, [("SPY", 1.0)], h, 1)
val = r.notna()
e = declusters(idx[m_tri.values & val.values], h, idx)
v = r.loc[e].values
yr = pd.DatetimeIndex(e).year
mid = yr % 4 == 2
base = r[val]
bmid = base.index.year % 4 == 2
show([summarize(v[mid], f"midterm (N={int(mid.sum())})"),
      summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})"),
      summarize(base[bmid].values, "CTRL all days midterm"),
      summarize(base[~bmid].values, "CTRL all days non-midterm")], "midterm")
hist = pd.Series(100 * v, index=pd.DatetimeIndex(e)).groupby(yr).agg(
    ["count", "sum", "mean"])
print("\nyear histogram (episodes, h=10, pp):")
print(hist.round(2).to_string())
print("\n" + cluster_note(pd.DatetimeIndex(e), v, k=3))

print("\n" + "=" * 78)
print("SHORT SIDE HONESTY: is the contrarian late-cycle read there instead?")
print("=" * 78)
for h in (10, 21, 42, 63):
    rr = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    vv = rr.notna()
    ee = declusters(idx[m_tri.values & vv.values], h, idx)
    ec = declusters(idx[m_spy.values & vv.values & (idx >= idx[m_tri.values][0])], h, idx)
    a, b = rr.loc[ee].values, rr.loc[ec].values
    bb = rr[vv & (idx >= idx[m_tri.values][0])]
    print(f"h={h:2d}  triple {100*a.mean():+.3f}% (N={len(a)})  "
          f"SPY-alone {100*b.mean():+.3f}% (N={len(b)})  "
          f"uncond {100*bb.mean():+.3f}%  add {100*(a.mean()-b.mean()):+.3f}pp")
