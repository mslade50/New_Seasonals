"""C9 round 2: gate attribution, max-of-12 null, era + midterm, session decomp.

Fixes a definition bug in c9_month_of_year.py: `ltd_positions` treats the
INCOMPLETE final month (2026-08, whose last available bar is 2026-08-25) as a
month-end, which minted a spurious 2026 observation in the h=3 cells.  Complete
months only here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

VEH = ["IWM", "SPY"]
px = load_prices(VEH)
ser = {t: px[t]["Close"].dropna() for t in VEH}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
          "Aug", "Sep", "Oct", "Nov", "Dec"]


def ltd_positions(idx):
    per = pd.Series(idx.to_period("M"), index=range(len(idx)))
    out = [int(g.index.max()) for _, g in per.groupby(per.values)]
    return out[:-1]           # drop the INCOMPLETE final month


def cell(t, k=0, h=3, me=3):
    s = ser[t]; idx = s.index; v = s.values
    e_, r_, m_ = [], [], []
    for p in ltd_positions(idx):
        e, x = p - me + k, p - me + k + h
        if e < 1 or x >= len(idx):
            continue
        e_.append(idx[e]); r_.append(v[x] / v[e] - 1.0); m_.append(idx[p].month)
    return pd.DatetimeIndex(e_), np.asarray(r_, float), np.asarray(m_)


print("=" * 78)
print("1. GATE ATTRIBUTION: what does the SEPTEMBER label add to a bare ME-3?")
print("=" * 78)
for t in VEH:
    for h in (3, 5):
        d, r, m = cell(t, 0, h)
        s = ser[t]
        base = (s.shift(-h) / s - 1.0).dropna()
        aug = r[m == 8]
        print(f"\n{t} h={h}:")
        print(f"  bare ME-3, ALL months pooled : {100*r.mean():+.3f}%  N={len(r)}  "
              f"hit {100*(r>0).mean():.1f}%")
        print(f"  all-days control             : {100*base.mean():+.3f}%  "
              f"N={len(base)}")
        print(f"  pooled ME-3 gate value       : {100*(r.mean()-base.mean()):+.3f}pp"
              f"  ({100*(r.mean()-base.mean())*100/6:.1f}x a 6 bp round trip)")
        print(f"  Aug-end only (INTO September): {100*aug.mean():+.3f}%  "
              f"N={len(aug)}  hit {100*(aug>0).mean():.1f}%")
        print(f"  September label discards {len(r)-len(aug)} of {len(r)} obs to "
              f"add {100*(aug.mean()-r.mean()):+.3f}pp over the pooled cell")
        comp = r[m != 8]
        print(f"  complement (11 other months) : {100*comp.mean():+.3f}%  "
              f"N={len(comp)}")

print("\n" + "=" * 78)
print("2. MAX-OF-12 NULL: is 'August-end' distinguishable inside its own family?")
print("=" * 78)
rng = np.random.default_rng(7)
for t in VEH:
    for h in (3, 5, 7, 10):
        d, r, m = cell(t, 0, h)
        ts, mus = {}, {}
        for mm in range(1, 13):
            v = r[m == mm]
            ts[mm] = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
            mus[mm] = v.mean()
        obs_t = ts[8]
        rank = 1 + sum(1 for mm in ts if ts[mm] > obs_t)
        # permutation: shuffle month labels across the same anchor returns
        mx = []
        for _ in range(3000):
            perm = rng.permutation(m)
            best = max(
                (r[perm == mm].mean()
                 / (r[perm == mm].std(ddof=1) / np.sqrt((perm == mm).sum())))
                for mm in range(1, 13))
            mx.append(best)
        mx = np.asarray(mx)
        print(f"  {t} h={h:2d}: Aug-end t={obs_t:+.2f} ranks {rank} of 12 by t; "
              f"P(max-of-12 t >= observed) = {(mx >= obs_t).mean():.3f}; "
              f"best month = {MONTHS[max(ts, key=ts.get)-1]} at t={max(ts.values()):+.2f}")

print("\n" + "=" * 78)
print("3. MIDTERM (today's state) and ERA splits, Aug-end")
print("=" * 78)
for t in VEH:
    for h in (3, 5, 7, 10):
        d, r, m = cell(t, 0, h)
        sel = m == 8
        a, dd = r[sel], d[sel]
        mid = np.array([x.year % 4 == 2 for x in dd])
        w = int((a[mid] > 0).sum())
        rows = [summarize(a, f"{t} Aug-end h={h}"),
                summarize(a[mid], f"MIDTERM (today) N={int(mid.sum())}"),
                summarize(a[~mid], "non-midterm")] + era_split(dd, a)
        show(rows)
        print(f"  midterm years: "
              f"{[(x.year, round(100*y, 2)) for x, y in zip(dd[mid], a[mid])]}")
        print(f"  midterm record {w}-{int(mid.sum())-w}\n")

print("=" * 78)
print("4. SESSION DECOMPOSITION: which session inside the ME-3 -> ME+2 window pays?")
print("=" * 78)
for t in VEH:
    s = ser[t]; idx = s.index; v = s.values
    print(f"\n{t} (Aug-end anchors only):")
    for off in range(-4, 4):
        vals = []
        for p in ltd_positions(idx):
            if idx[p].month != 8:
                continue
            a, b = p + off, p + off + 1
            if a < 1 or b >= len(idx):
                continue
            vals.append(v[b] / v[a] - 1.0)
        vals = np.asarray(vals)
        base = (s.shift(-1) / s - 1.0).dropna()
        print(f"  ME{off:+d} -> ME{off+1:+d}: {100*100*vals.mean():+7.2f} bp  "
              f"hit {100*(vals>0).mean():5.1f}%  N={len(vals)}   "
              f"(all-days {100*100*base.mean():+.2f} bp, hit "
              f"{100*(base>0).mean():.1f}%)")
