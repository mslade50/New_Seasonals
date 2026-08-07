"""N2: red-team the NFP x rates h=3 cell, and clear the other open B1 cells.

N1 put the signal at h=3, not h=5: TLT +0.633% (gate 2%) / +0.543% (gate 3%)
with a 76% hit rate and t ~1.7-1.9, stable across gate widths. That is the
only thing worth attacking. Everything here is designed to KILL it.

Also clears the remaining CHECK cells from the surface map so they are not
left as unexamined claims: credit duration divergence (HYG vs LQD), IWM, and
the dollar leg.
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
CPI = sorted(EV.loc[EV.event == "cpi", "date"])
FLOOR = 100.0 * (PX["TLT"] / PX["TLT"].rolling(252).min() - 1.0)


def fwd(sym, d, h):
    p = POS[d]
    return np.nan if p + h >= len(CAL) else PX[sym].iloc[p + h] / PX[sym].iloc[p] - 1.0


def drift(sym, h, lo, hi):
    r = (PX[sym].shift(-h) / PX[sym] - 1.0)
    return r[(r.index >= lo) & (r.index <= hi)].dropna()


H = 3
GATE = 3.0
sub = [d for d in NFP if FLOOR.get(d, np.nan) <= GATE]
vals = np.array([fwd("TLT", d, H) for d in sub])
dts = pd.DatetimeIndex([d for d, x in zip(sub, vals) if not np.isnan(x)])
vals = vals[~np.isnan(vals)]

print("=" * 96)
print(f"RED TEAM: TLT long, NFP close -> +{H}td MOC, gate TLT within {GATE:.0f}% of 52w low")
print(f"N = {len(vals)}   span {dts.min().date()} .. {dts.max().date()}")
print("=" * 96)

print("\n--- A. is it just the h=3 pick? horizon stability ---")
rows = []
for h in (1, 2, 3, 4, 5, 6, 8, 10):
    v = np.array([fwd("TLT", d, h) for d in sub])
    v = v[~np.isnan(v)]
    s = summarize(v, f"+{h}td")
    b = drift("TLT", h, min(sub), max(sub))
    s["edge"] = round(100 * (v.mean() - b.mean()), 3)
    rows.append(s)
show(rows, "horizon sweep (a one-horizon spike is a fitted horizon)")

print("\n--- B. declustering (overlapping NFPs inflate day-level t) ---")
dc = declusters(dts, 21, CAL)
vdc = np.array([fwd("TLT", d, H) for d in dc])
vdc = vdc[~np.isnan(vdc)]
show([summarize(vals, "day-level"), summarize(vdc, "declustered 21td")], "")

print("\n--- C. era + episode concentration (the trap that killed TLT-short) ---")
for s in era_split(dts, vals):
    print(f"    {s['label']:<10} n={s['n']:<4} mean={s['mean_pct']:+.3f}  t={s['t']:+.2f}")
yrs = pd.Series([d.year for d in dts]).value_counts().sort_index()
print(f"    episode years: {dict(yrs)}")
top2 = yrs.nlargest(2)
keep = [i for i, d in enumerate(dts) if d.year not in top2.index]
print(f"    dropping the two biggest years {list(top2.index)} "
      f"({top2.sum()} of {len(dts)} obs):")
show([summarize(vals, "all"), summarize(vals[keep], "ex-top-2-years")], "")

print("\n--- D. leave-one-year-out floor ---")
loyo = []
for y in sorted(set(d.year for d in dts)):
    k = [i for i, d in enumerate(dts) if d.year != y]
    s = summarize(vals[k], f"ex-{y}")
    loyo.append(s)
worst = min(loyo, key=lambda r: r["t"])
show(loyo, "LOYO")
print(f"    LOYO floor: {worst['label']} t={worst['t']:+.2f} "
      f"mean={worst['mean_pct']:+.3f}")

print("\n--- E. midterm split (today is midterm) ---")
mid = [i for i, d in enumerate(dts) if d.year % 4 == 2]
non = [i for i, d in enumerate(dts) if d.year % 4 != 2]
show([summarize(vals[mid], f"midterm (N={len(mid)})"),
      summarize(vals[non], f"non-midterm (N={len(non)})")], "")

print("\n--- F. CPI inside the hold? (today: CPI is 3 td out, so YES) ---")
cpiset = set(CPI)
inside = np.array([any(x in cpiset for x in CAL[POS[d] + 1: POS[d] + H + 1])
                   for d in dts])
show([summarize(vals[inside], f"CPI inside (N={int(inside.sum())})"),
      summarize(vals[~inside], f"CPI outside (N={int((~inside).sum())})")], "")

print("\n--- G. bootstrap + cost ---")
print(f"    bootstrap P(mean <= 0) = {bootstrap_p_le0(vals):.3f}")
print(f"    mean {100*vals.mean():.3f}% vs ~2 bps round trip = "
      f"{100*vals.mean()/0.02:.1f}x cost")

# ---------------------------------------------------------------------------
print("\n\n" + "=" * 96)
print("OTHER OPEN CELLS FROM THE SURFACE MAP")
print("=" * 96)

print("\n--- H. credit duration divergence: HYG near 52w high, LQD near 52w low ---")
hyg_hi = 100.0 * (PX["HYG"] / PX["HYG"].rolling(252).max() - 1.0)
lqd_lo = 100.0 * (PX["LQD"] / PX["LQD"].rolling(252).min() - 1.0)
print(f"    today HYG {hyg_hi.iloc[-1]:+.2f}% off 52wh, "
      f"LQD {lqd_lo.iloc[-1]:+.2f}% off 52wl")
div = [d for d in NFP if hyg_hi.get(d, -99) >= -1.0 and lqd_lo.get(d, 99) <= 2.0]
print(f"    NFP days in that joint state: N = {len(div)}")
if len(div) >= 5:
    rows = []
    for sym in ("HYG", "LQD", "TLT"):
        v = np.array([fwd(sym, d, 5) for d in div])
        v = v[~np.isnan(v)]
        rows.append(summarize(v, sym))
    show(rows, "h=5 forward, joint credit-divergence state")
else:
    print("    -> too few occurrences to measure. Cell recorded as UNMEASURABLE,")
    print("       which is a kill, not a pass (count occurrences before edge).")

print("\n--- I. IWM at the same rates trigger ---")
v = np.array([fwd("IWM", d, 5) for d in sub])
v = v[~np.isnan(v)]
b = drift("IWM", 5, min(sub), max(sub))
s = summarize(v, "IWM +5td")
s["edge"] = round(100 * (v.mean() - b.mean()), 3)
s["p_le0_boot"] = bootstrap_p_le0(v)
show([s], "")

print("\n--- J. dollar leg (UUP) at the same trigger, both directions ---")
v = np.array([fwd("UUP", d, 5) for d in sub])
v = v[~np.isnan(v)]
b = drift("UUP", 5, min(sub), max(sub))
show([summarize(v, "UUP long"), summarize(-v, "UUP short")], "")
print(f"    edge vs own drift (long side): {100*(v.mean()-b.mean()):+.3f}pp")
print(f"    UUP round trip ~4 bps; short-side mean {100*(-v).mean():.3f}% = "
      f"{100*(-v).mean()/0.04:.1f}x cost")
