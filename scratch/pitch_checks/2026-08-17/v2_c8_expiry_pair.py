"""C8 round 1 -- the vol complex into the VIX-expiry / opex pair.

The brief's first demand: the two 307-event grids (vix_expiry -2 td and
opex -4 td) overlap heavily and CANNOT both be independent evidence.
Section 1 measures the overlap exactly and locates any genuine disagreement.

Section 2 re-measures the only TRADEABLE leg (SVXY; ^VIX is an index) with
the 2018-02-28 leverage break respected -- the surface-map grid pooled
across it, which is a standing invalidation.

Section 3 prices the book overlap with the LIVE event sleeve V4_POSTOPEX_VOL
(long SVXY 10% NAV, opex MOC -> +3 sessions MOC, every month but September),
which fires on 2026-08-21.

Section 4 charges the multiplicity of the grid this candidate came out of.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

SVXY_BREAK = pd.Timestamp("2018-03-01")
px = close_panel(["SVXY", "SPY", "QQQ", "IWM", "^VIX", "^VIX3M"])
ALL = px.index
ev = load_events()


def anchors(kind, off):
    dates = pd.to_datetime(ev.loc[ev["event"] == kind, "date"].unique())
    out = []
    for d in dates:
        pos = ALL.searchsorted(d)
        if pos >= len(ALL):
            continue
        i = pos - off
        if i < 260 or i >= len(ALL) - 12:
            continue
        out.append(ALL[i])
    return pd.DatetimeIndex(sorted(set(out)))


A_vx = anchors("vix_expiry", 2)
A_op = anchors("opex", 4)

print("=" * 78)
print("1. ARE THE TWO ANCHOR GRIDS THE SAME DAYS?")
print("=" * 78)
inter = A_vx.intersection(A_op)
print(f"  vix_expiry-2 N={len(A_vx)}   opex-4 N={len(A_op)}   "
      f"INTERSECTION N={len(inter)}  "
      f"({100*len(inter)/len(A_vx):.1f}% of the vix grid, "
      f"{100*len(inter)/len(A_op):.1f}% of the opex grid)")
only_vx = A_vx.difference(A_op)
only_op = A_op.difference(A_vx)
print(f"  vix-only N={len(only_vx)}   opex-only N={len(only_op)}")
# what the offset between the two events actually is, month by month
vxd = pd.to_datetime(ev.loc[ev.event == "vix_expiry", "date"])
opd = pd.to_datetime(ev.loc[ev.event == "opex", "date"])
pos = pd.Series(range(len(ALL)), index=ALL)
gaps = []
for d in vxd:
    p = ALL.searchsorted(d)
    if p >= len(ALL):
        continue
    # nearest opex in the same calendar month
    same = opd[(opd.dt.year == d.year) & (opd.dt.month == d.month)]
    if len(same) == 0:
        continue
    q = ALL.searchsorted(same.iloc[0])
    if q >= len(ALL):
        continue
    gaps.append(q - p)
print(f"  opex minus vix_expiry, in trading days, distribution: "
      f"{dict(pd.Series(gaps).value_counts().sort_index())}")
print("  (+2 = VIX expiry Wed of opex week -> the two anchors are the SAME "
      "Monday; anything else and they are different days)")

print("\n1b. THE SAME horizon at both anchors, on the SHARED days vs the "
      "exclusive days -- where does the 'disagreement' live?")
for tkr, lbl in (("^VIX", "^VIX (index, NOT tradeable)"), ("SPY", "SPY")):
    s = px[tkr].dropna()
    for h in (3, 5):
        r = fwd_lag(s, h, 1)
        base = 100 * r.mean()
        rows = []
        for nm, idx in (("vix-2 ALL", A_vx), ("opex-4 ALL", A_op),
                        ("SHARED days", inter), ("vix-2 ONLY", only_vx),
                        ("opex-4 ONLY", only_op)):
            d = pd.DatetimeIndex(idx).intersection(r.dropna().index)
            x = summarize(r.loc[d].values, nm)
            if x["n"]:
                x["excess_pp"] = round(x["mean_pct"] - base, 3)
            rows.append(x)
        show(rows, f"1b. {lbl} h={h}  (uncond drift {base:+.3f}%)")

print("\n" + "=" * 78)
print("2. THE ONLY TRADEABLE VOL LEG: SVXY, POST-BREAK ONLY (2018-03-01+)")
print("=" * 78)
sv = px["SVXY"].dropna()
n_pre = int((pd.DatetimeIndex(A_op).isin(sv.index) &
             (pd.DatetimeIndex(A_op) < SVXY_BREAK)).sum())
n_post = int((pd.DatetimeIndex(A_op).isin(sv.index) &
              (pd.DatetimeIndex(A_op) >= SVXY_BREAK)).sum())
print(f"  opex-4 anchors with SVXY data: {n_pre} PRE-break / {n_post} post-break "
      f"-> the surface-map grid pooled {100*n_pre/(n_pre+n_post):.0f}% of its "
      f"SVXY sample on the -1x instrument that no longer exists")

post = px.loc[px.index >= SVXY_BREAK]
for nm, idx in (("opex-4", A_op), ("vix_expiry-2", A_vx), ("SHARED", inter)):
    m = pd.Series(False, index=px.index)
    m.loc[pd.DatetimeIndex(idx)] = True
    for h in (3, 5, 8):
        battery(post, m, [("SVXY", 1.0)], h,
                f"2. long SVXY at {nm} anchor h={h} (post-break)",
                cost_bps=10.0, min_gap=max(h, 5))

print("\n" + "=" * 78)
print("3. BOOK OVERLAP -- V4_POSTOPEX_VOL is the SAME TRADE, one week later")
print("=" * 78)
print("  V4 spec (event_sleeve.EVENT_SLEEVE): long SVXY, nav_frac 0.10,")
print("  entry MOC on opex, exit MOC opex+3 td, every month except September.")
print("  C8's opex-4 long-SVXY cell enters MOC on the Tuesday of opex week and")
print("  at h>=4 is still holding on opex Friday, i.e. it ENTERS V4's position")
print("  early and doubles it.")
posn = pd.Series(range(len(ALL)), index=ALL)
v4 = set()
for d in opd:
    p = posn.get(pd.Timestamp(d))
    if p is None or d.month == 9:
        continue
    for q in range(p, min(p + 4, len(ALL))):
        v4.add(ALL[q])
for h in (3, 5, 8):
    ov = []
    for d in A_op:
        p = posn.get(d)
        if p is None or p + 1 + h >= len(ALL):
            continue
        ov.append(len(set(ALL[p + 1:p + 2 + h]) & v4))
    ov = np.array(ov)
    print(f"  h={h}: {100*(ov>0).mean():.0f}% of opex-4 anchors hold into a V4 "
          f"window, mean {ov.mean():.2f} of {h} sessions shared")

print("\n" + "=" * 78)
print("4. MULTIPLICITY CHARGE (the brief: C8 came out of a grid, so charge it)")
print("=" * 78)
n_cells = 13 * 3 * 6   # 13 tickers x 3 anchors x 6 horizons in 02_event_surface
print(f"  02_event_surface.py scanned ~{n_cells} (ticker x anchor x horizon) cells.")
print(f"  At alpha=0.05 that expects ~{0.05*n_cells:.0f} nominally significant")
print("  cells under a pure null. The SVXY opex-4 h=5 cell was READ OUT of that")
print("  grid, so it needs to clear a family-wise bar, not a nominal one.")
