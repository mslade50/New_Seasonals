"""C12 round 1: the run INTO NFP entered 7 sessions early.

Live instance: signal close 2026-08-25, entry MOC 2026-08-26, NFP 2026-09-04
which is exactly 7 sessions after the entry close.

Formulation: entry position = pos(NFP) + k - 7, exit = entry + h.  At k=0 and
h=7 the exit IS the NFP close.  The placebo ladder sweeps k in -10..+5.

Vehicle grid PRE-DECLARED: SPY, TLT, UUP, GLD.  Horizons 5..10.
Per the registry the ladder runs FIRST; a plateau kills on the spot.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

VEH = ["SPY", "TLT", "UUP", "GLD"]
H_GRID = (5, 6, 7, 8, 9, 10)
LEAD = 7  # entry sits LEAD sessions before the print

px = load_prices(VEH)
ser = {t: px[t]["Close"].dropna() for t in VEH}
nfp = load_events(["nfp"])["date"]
nfp = nfp[(nfp >= "2000-01-01") & (nfp <= "2026-08-25")].reset_index(drop=True)
print(f"NFP prints available: {len(nfp)}  {nfp.iloc[0].date()} .. {nfp.iloc[-1].date()}")


def cell(t: str, k: int, h: int):
    """Returns (dates_of_entry, fwd returns) for offset k, horizon h."""
    s = ser[t]
    idx = s.index
    v = s.values
    ent, ret = [], []
    for d in nfp:
        p = int(idx.searchsorted(d))
        if p >= len(idx):
            continue
        e = p + k - LEAD
        x = e + h
        if e < 0 or x >= len(idx):
            continue
        ent.append(idx[e])
        ret.append(v[x] / v[e] - 1.0)
    return pd.DatetimeIndex(ent), np.asarray(ret, float)


def drift_ctrl(t: str, h: int, dates: pd.DatetimeIndex | None = None):
    """CTRL-a: the instrument's own unconditional h-session drift.  When
    `dates` is passed the control is restricted to those calendar months."""
    s = ser[t]
    r = (s.shift(-h) / s - 1.0).dropna()
    if dates is not None and len(dates):
        months = set(pd.DatetimeIndex(dates).month)
        r = r[r.index.month.isin(months)]
    return r.values


print("\n" + "=" * 78)
print("1. PLACEBO OFFSET LADDER at h=7 (exit lands on the print close at k=0)")
print("=" * 78)
lad = {}
for t in VEH:
    row = {}
    for k in range(-10, 6):
        d, r = cell(t, k, 7)
        row[k] = (100 * r.mean(), len(r), 100 * (r > 0).mean())
    lad[t] = row
    best = max(row, key=lambda kk: row[kk][0])
    rank = 1 + sum(1 for kk in row if row[kk][0] > row[0][0])
    print(f"\n{t}:  true k=0 -> {row[0][0]:+.3f}%  (N={row[0][1]}, hit {row[0][2]:.1f}%)"
          f"   rank {rank} of {len(row)}   best k={best} at {row[best][0]:+.3f}%")
    print("   " + "  ".join(f"{k:+d}:{row[k][0]:+.2f}" for k in range(-10, 6)))

print("\n" + "=" * 78)
print("2. PRE-DECLARED 4-vehicle x 6-horizon GRID at the true anchor (k=0)")
print("=" * 78)
rows = []
grid = {}
for t in VEH:
    for h in H_GRID:
        d, r = cell(t, 0, h)
        c = drift_ctrl(t, h)
        cm = drift_ctrl(t, h, d)
        sm = summarize(r, f"{t} h={h}")
        sm["ctrl_all_pct"] = round(100 * c.mean(), 3)
        sm["ctrl_monthmatch_pct"] = round(100 * cm.mean(), 3)
        sm["excess_mm_pct"] = round(sm["mean_pct"] - 100 * cm.mean(), 3)
        rows.append(sm)
        grid[(t, h)] = (d, r, cm.mean())
show(rows, "grid, month-matched control")

best = max(grid, key=lambda key: 100 * grid[key][1].mean() - 100 * grid[key][2])
bd, br, bc = grid[best]
print(f"\nbest cell of {len(grid)}: {best}  excess {100*(br.mean()-bc):+.3f}pp "
      f"(raw {100*br.mean():+.3f}%, N={len(br)}, hit {100*(br>0).mean():.1f}%)")

print("\n" + "=" * 78)
print("3. ROTATION NULL for the max of the pre-declared grid")
print("=" * 78)
rng = np.random.default_rng(42)
obs = 100 * (br.mean() - bc)
null = []
for it in range(400):
    shift = int(rng.integers(-40, 41))
    if abs(shift) < 8:
        shift = 8 * (1 if shift >= 0 else -1)
    m = -1e9
    for t in VEH:
        for h in H_GRID:
            d, r = cell(t, shift, h)
            if len(r) < 10:
                continue
            cm = drift_ctrl(t, h, d)
            m = max(m, 100 * (r.mean() - cm.mean()))
    null.append(m)
null = np.asarray(null)
print(f"observed max-of-grid excess = {obs:+.3f}pp")
print(f"rotation null (400 relocations, |shift|>=8 td): "
      f"P(max >= observed) = {(null >= obs).mean():.3f}   "
      f"null mean {null.mean():+.3f}  p90 {np.percentile(null, 90):+.3f}")

print("\n" + "=" * 78)
print("4. MIDTERM split and SEPTEMBER-NFP split, h=7, all four vehicles")
print("=" * 78)
for t in VEH:
    d, r = cell(t, 0, 7)
    mid = np.array([x.year % 4 == 2 for x in d])
    sep = np.array([x.month in (8, 9) for x in d])  # entry Aug/Sep = Sep print
    sept_print = np.array([(x + pd.Timedelta(days=14)).month == 9 for x in d])
    show([summarize(r[mid], f"{t} midterm (N={int(mid.sum())})"),
          summarize(r[~mid], f"{t} non-midterm"),
          summarize(r[sept_print], f"{t} SEPT print (N={int(sept_print.sum())})"),
          summarize(r[~sept_print], f"{t} other months")], f"{t} splits h=7")

print("\n" + "=" * 78)
print("5. COST: 1 leg ~6 bps round trip, need >=5x = +0.30%")
print("=" * 78)
for t in VEH:
    d, r = cell(t, 0, 7)
    cm = drift_ctrl(t, 7, d)
    ex = 100 * (r.mean() - cm.mean())
    print(f"{t}: raw {100*r.mean():+.3f}% = {100*r.mean()*100:.1f} bps -> "
          f"{100*r.mean()*100/6:.1f}x ; month-matched excess {ex:+.3f}pp = "
          f"{ex*100:.1f} bps -> {ex*100/6:.1f}x")
