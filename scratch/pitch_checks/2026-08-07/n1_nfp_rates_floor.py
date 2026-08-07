"""N1: NFP reaction in RATES, conditioned on the long end sitting at its floor.

The cell the 5:10 AM run never opened. Every calendar-anchored check that
morning ran on SPY; every cross-asset check ran on a chart level. This is the
intersection: a jobs print landing while TLT sits within ~2% of its 52w low.

Trade shape under test: MOC on the NFP close (today) -> MOC h td later.
Both directions reported, because the sign is the whole question.

Adversarial brief. Controls are TLT's own drift over the same span (it has a
brutal secular downtrend, so "bonds fell" must not be mistaken for an edge)
and the all-NFP baseline (so "NFP does something" must not be mistaken for
"NFP does something HERE").
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import numpy as np
import pandas as pd

PX = close_panel(["TLT", "IEF", "SPY", "XLU", "HYG", "LQD", "IWM", "UUP"]).dropna(
    subset=["TLT", "SPY"])
CAL = PX.index
POS = pd.Series(range(len(CAL)), index=CAL)
EV = load_events()
NFP = [d for d in EV.loc[EV.event == "nfp", "date"] if d in POS.index]

# distance to the trailing 252d low, in percent (0 = AT the low)
def floor_dist(s):
    return 100.0 * (s / s.rolling(252).min() - 1.0)


TLT_FLOOR = floor_dist(PX["TLT"])
print(f"TLT distance to 52w low on {CAL[-1].date()}: "
      f"{TLT_FLOOR.iloc[-1]:.2f}%   (today's gate value)")
print(f"NFP dates available: {len(NFP)} "
      f"({NFP[0].date()} .. {NFP[-1].date()})")


# FRACTIONS, not percent: summarize() scales by 100 itself. Passing percent in
# double-scales and every mean/worst column comes out 100x.
def fwd(sym, d, h):
    p = POS[d]
    if p + h >= len(CAL):
        return np.nan
    return PX[sym].iloc[p + h] / PX[sym].iloc[p] - 1.0


def drift(sym, h, span_lo=None, span_hi=None):
    s = PX[sym]
    r = s.shift(-h) / s - 1.0
    if span_lo is not None:
        r = r[(r.index >= span_lo) & (r.index <= span_hi)]
    return r.dropna()


# ---------------------------------------------------------------------------
print("\n" + "=" * 96)
print("1) THE CELL: NFP with TLT near its 52w floor, TLT forward returns")
print("=" * 96)

for thr in (1.0, 2.0, 3.0, 5.0):
    sub = [d for d in NFP if TLT_FLOOR.get(d, np.nan) <= thr]
    rows = []
    for h in (1, 2, 3, 5, 10):
        v = np.array([fwd("TLT", d, h) for d in sub])
        v = v[~np.isnan(v)]
        if len(v) < 3:
            continue
        s = summarize(v, f"TLT +{h}td")
        base = drift("TLT", h, min(sub), max(sub))
        s["ctl_own_drift"] = round(100 * base.mean(), 3)
        s["edge_vs_ctl"] = round(100 * (v.mean() - base.mean()), 3)
        allnfp = np.array([fwd("TLT", d, h) for d in NFP])
        s["ctl_all_nfp"] = round(100 * np.nanmean(allnfp), 3)
        rows.append(s)
    show(rows, f"gate: TLT within {thr:.0f}% of 52w low   (N NFPs = {len(sub)})")

# ---------------------------------------------------------------------------
print("\n" + "=" * 96)
print("2) CROSS-ASSET at the SAME trigger (h=5) - where does the reaction live?")
print("=" * 96)

GATE = 2.0
sub = [d for d in NFP if TLT_FLOOR.get(d, np.nan) <= GATE]
rows = []
for sym in ("TLT", "IEF", "LQD", "HYG", "XLU", "SPY", "IWM", "UUP"):
    if sym not in PX.columns:
        continue
    v = np.array([fwd(sym, d, 5) for d in sub])
    v = v[~np.isnan(v)]
    if len(v) < 3:
        continue
    s = summarize(v, sym)
    base = drift(sym, 5, min(sub), max(sub))
    s["ctl_own_drift"] = round(100 * base.mean(), 3)
    s["edge_vs_ctl"] = round(100 * (v.mean() - base.mean()), 3)
    s["p_le0_boot"] = bootstrap_p_le0(v)
    rows.append(s)
show(rows, f"h=5 MOC->MOC, gate TLT within {GATE:.0f}% of 52w low, N={len(sub)}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 96)
print("3) DECLUSTER + ERA + EPISODE YEARS (the two traps that killed the AM run)")
print("=" * 96)

for sym in ("TLT", "XLU", "LQD"):
    v = np.array([fwd(sym, d, 5) for d in sub])
    dts = pd.DatetimeIndex([d for d, x in zip(sub, v) if not np.isnan(x)])
    v = v[~np.isnan(v)]
    dc = declusters(dts, 21, CAL)
    vdc = np.array([fwd(sym, d, 5) for d in dc])
    vdc = vdc[~np.isnan(vdc)]
    print(f"\n--- {sym} ---")
    show([summarize(v, "day-level"), summarize(vdc, "declustered 21td")], "")
    for s in era_split(dts, v):
        print(f"    era {s['label']:<10} n={s['n']:<4} mean={s['mean_pct']:+.3f} "
              f"t={s['t']:+.2f}")
    yrs = pd.Series([d.year for d in dts]).value_counts().sort_index()
    print(f"    episode years: {dict(yrs)}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 96)
print("4) COST + TODAY'S POSITION IN THE DISTRIBUTION")
print("=" * 96)
print("  TLT round trip ~2 bps; XLU ~2 bps. Edge must clear ~10 bps (5x).")
gate_vals = np.array([TLT_FLOOR.get(d, np.nan) for d in NFP])
gate_vals = gate_vals[~np.isnan(gate_vals)]
today = TLT_FLOOR.iloc[-1]
print(f"  today's TLT floor distance {today:.2f}% is the "
      f"{100.0 * (gate_vals <= today).mean():.1f}th percentile of all NFP days")
print(f"  midterm NFPs inside the gate: "
      f"{sum(1 for d in sub if d.year % 4 == 2)} of {len(sub)}")
