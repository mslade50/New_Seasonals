"""C4 round 1 -- the whole IG duration complex pinned at 52w lows, long TLT.

Trigger (live today): TLT within 1.0% of its 52w low AND IEF within 1.5% AND
LQD within 1.5%, simultaneously.

Two questions:
 (a) does the state stand alone at any horizon, and
 (b) the one that matters -- does it ADD anything to C1 as a conditioner? The
     2026-08-10 work found a 52w-floor gate on the PPI cell did nothing
     (+0.115% -> +0.117% at a tenth the sample). RE-DERIVE, do not trust it.

Registry: h must come from the horizon scan, not be assumed; episodes not days;
year histogram; local control; and the joint-state occurrence COUNT first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["TLT", "IEF", "LQD"]).dropna()
idx = px.index
N = len(idx)


def off_low(s, win=252):
    lo = s.rolling(win).min()
    return (s / lo - 1.0) * 100.0


offs = {t: off_low(px[t]) for t in ["TLT", "IEF", "LQD"]}
print("=" * 104)
print("0. TODAY'S STATE (tape through 2026-08-11) and the trigger definition")
print("=" * 104)
for t in ["TLT", "IEF", "LQD"]:
    print(f"  {t}: {offs[t].iloc[-1]:.2f}% off its 52w low   "
          f"(last bar {idx[-1].date()})")
TR = {"TLT": 1.0, "IEF": 1.5, "LQD": 1.5}
mask = np.ones(N, bool)
for t, k in TR.items():
    mask &= (offs[t] <= k).values
mask &= ~np.isnan(offs["TLT"].values)
trig = idx[mask]
print(f"\n  joint trigger: TLT<=1.0%, IEF<=1.5%, LQD<=1.5% off 52w low")
print(f"  occurrences: {int(mask.sum())} days over {idx[0].date()}..{idx[-1].date()}")
print(f"  year histogram: "
      f"{pd.Series(1, index=trig).groupby(trig.year).sum().to_dict()}")

print("\n" + "=" * 104)
print("1. HORIZON SCAN (episodes, min gap 10td) -- where, if anywhere, is it?")
print("=" * 104)
rows = horizon_scan(px, trig, [("TLT", 1.0)], hs=(1, 2, 3, 5, 10, 21),
                    lag=1, min_gap=10)
show(rows, "outright long TLT from the joint floor state")

print("\n" + "=" * 104)
print("2. FULL BATTERY at h=1 (the recon's only positive horizon)")
print("=" * 104)
m = pd.Series(mask, index=idx)
loosen = {}
for k in [(1.5, 2.0, 2.0), (0.5, 1.0, 1.0), (2.0, 3.0, 3.0)]:
    mm = np.ones(N, bool)
    for t, kk in zip(["TLT", "IEF", "LQD"], k):
        mm &= (offs[t] <= kk).values
    mm &= ~np.isnan(offs["TLT"].values)
    loosen[f"TLT<={k[0]} IEF<={k[1]} LQD<={k[2]}"] = pd.Series(mm, index=idx)
loosen["TLT alone <=1.0%"] = pd.Series(
    (offs["TLT"] <= 1.0).values & ~np.isnan(offs["TLT"].values), index=idx)
battery(px, m, [("TLT", 1.0)], 1, "C4 IG complex at 52w floor -> long TLT",
        cost_bps=2.5, variants=loosen, min_gap=10, event_kinds=("cpi", "ppi"))

print("\n" + "=" * 104)
print("3. *** THE DECISIVE ONE: does the floor state ADD to C1? ***")
print("   re-derived, not borrowed from 2026-08-10")
print("=" * 104)
tl = px["TLT"]
d1 = tl.pct_change().values
base_hit = float((d1[1:] > 0).mean())
ev = load_events()
sp = lambda k: sorted({int(idx.searchsorted(x, "left"))
                       for x in ev[ev.event == k]["date"]
                       if 0 <= int(idx.searchsorted(x, "left")) < N})
ppi_l = [p for p in sp("ppi") if 1 <= p < N and not np.isnan(d1[p])]
cpi_all = set(sp("cpi"))
v = np.array([d1[p] for p in ppi_l])
dt = pd.DatetimeIndex([idx[p] for p in ppi_l])
ceve = np.array([(p - 1) in cpi_all for p in ppi_l])
# floor state as of the ANCHOR (2 sessions before the print) -- knowable at
# the time the trade is decided
flr = np.array([bool(mask[p - 2]) if p >= 2 else False for p in ppi_l])
flr_t = np.array([bool((offs["TLT"].values[p - 2] <= 1.0)) if p >= 2 else False
                  for p in ppi_l])


def rep(x, lbl):
    if len(x) == 0:
        return {"cell": lbl, "N": 0}
    w = int((x > 0).sum())
    return {"cell": lbl, "N": len(x), "mean_pct": round(100 * x.mean(), 4),
            "hit": round(100 * w / len(x), 1),
            "signp": round(sign_test(w, len(x), base_hit), 4),
            "worst": round(100 * x.min(), 2)}


print(pd.DataFrame([
    rep(v, "PARENT PPI print"),
    rep(v[flr], "PARENT + IG complex at floor"),
    rep(v[~flr], "PARENT + not at floor"),
    rep(v[flr_t], "PARENT + TLT alone at floor"),
    rep(v[ceve], "LIVE CELL (CPI on eve)"),
    rep(v[ceve & flr], "*** LIVE CELL + floor = TODAY ***"),
    rep(v[ceve & ~flr], "LIVE CELL, not at floor"),
]).to_string(index=False))
if (ceve & flr).sum():
    print(f"\n  today-exact dates: "
          f"{', '.join(str(d.date()) for d in dt[ceve & flr])}")
a, b = v[flr], v[~flr]
if len(a) > 1:
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    print(f"  floor-gate lift on the parent = {100*(a.mean()-b.mean()):+.4f}pp "
          f"welch t {(a.mean()-b.mean())/se:+.2f}")
print("  -> if the gate moves nothing, C4 is not a conditioner and not an idea;")
print("     say what the trade actually keys on.")

print("\n" + "=" * 104)
print("4. GATE ATTRIBUTION FOR C4 ITSELF: strip the IEF/LQD legs")
print("=" * 104)
for lbl, mm in [("TLT<=1.0 AND IEF<=1.5 AND LQD<=1.5", mask),
                ("TLT<=1.0 only", (offs["TLT"] <= 1.0).values),
                ("IEF<=1.5 only", (offs["IEF"] <= 1.5).values),
                ("LQD<=1.5 only", (offs["LQD"] <= 1.5).values)]:
    mm = mm & ~np.isnan(offs["TLT"].values)
    t_ = idx[mm]
    e = declusters(t_, 10, idx)
    r1 = vehicle_ret(px, [("TLT", 1.0)], 1, 1)
    s = r1.loc[e].dropna().values
    if len(s) < 2:
        continue
    allv = r1.dropna()
    w = int((s > 0).sum())
    print(f"  {lbl:38s} days={int(mm.sum()):4d} epi={len(s):3d} "
          f"{100*s.mean():+.4f}% excess "
          f"{100*(s.mean()-allv.mean()):+.4f}pp hit {100*w/len(s):5.1f}% "
          f"signp {sign_test(w, len(s), base_hit):.4f}")
print("  If the multi-leg join does no better than TLT alone, the extra legs")
print("  are decoration on a single condition.")
