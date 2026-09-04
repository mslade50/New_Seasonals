"""C9 round 1: the turn INTO September, entered at ME-3 on IWM.

Live instance: signal close 2026-08-25 (= LTD-4 for August), entry MOC
2026-08-26 (= LTD-3, three sessions before the 08-31 month-end close).

The registry closed the month-POSITION anchor on equities (2026-08-24), on FX
(08-25) and suspended it on rates (watchlist #12).  The claim here is that the
crossing INTO September is a MONTH-OF-YEAR object.  The honest test is
therefore the 12-month table under an identical ME-3 entry, and the placebo
offset ladder around the month-end anchor.
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


def ltd_positions(idx: pd.DatetimeIndex) -> list[int]:
    """Positions of each month's LAST trading day."""
    per = pd.Series(idx.to_period("M"), index=range(len(idx)))
    return [int(g.index.max()) for _, g in per.groupby(per.values)]


def cell(t: str, k: int = 0, h: int = 3, me: int = 3):
    """Entry at LTD - me + k, exit h sessions later.  k=0, me=3, h=3 exits
    exactly on the month-end close."""
    s = ser[t]; idx = s.index; v = s.values
    ents, rets, mons = [], [], []
    for p in ltd_positions(idx):
        e = p - me + k
        x = e + h
        if e < 1 or x >= len(idx):
            continue
        ents.append(idx[e]); rets.append(v[x] / v[e] - 1.0)
        mons.append(idx[p].month)          # month whose end is being crossed
    return pd.DatetimeIndex(ents), np.asarray(rets, float), np.asarray(mons)


def drift(t: str, h: int, months=None):
    s = ser[t]
    r = (s.shift(-h) / s - 1.0).dropna()
    if months is not None:
        r = r[r.index.month.isin(months)]
    return r.values


print("=" * 78)
print("1. TWELVE-MONTH TABLE, identical ME-3 entry.  Is September distinguishable?")
print("=" * 78)
for t in VEH:
    for h in (3, 5, 7, 10):
        d, r, m = cell(t, 0, h)
        rows = []
        for mm in range(1, 13):
            sel = m == mm
            sm = summarize(r[sel], f"{MONTHS[mm-1]}-end")
            base = drift(t, h, [mm, (mm % 12) + 1])
            sm["excess_pct"] = round(sm["mean_pct"] - 100 * base.mean(), 3)
            rows.append(sm)
        df = pd.DataFrame(rows).sort_values("mean_pct", ascending=False)
        aug = [i for i, x in enumerate(df["label"]) if x == "Aug-end"][0]
        print(f"\n--- {t}, h={h} (entry ME-3) --- Aug-end (the turn INTO September) "
              f"ranks {aug+1} of 12 by mean")
        for c in df.columns:
            if df[c].dtype.kind == "f":
                df[c] = df[c].round(3)
        print(df.to_string(index=False))

print("\n" + "=" * 78)
print("2. PLACEBO OFFSET LADDER around the month-end anchor (Aug-end only, h=5)")
print("=" * 78)
for t in VEH:
    row = {}
    for k in range(-10, 6):
        d, r, m = cell(t, k, 5)
        sel = m == 8
        row[k] = (100 * r[sel].mean(), int(sel.sum()))
    rank = 1 + sum(1 for kk in row if row[kk][0] > row[0][0])
    print(f"\n{t} Aug-end: true k=0 -> {row[0][0]:+.3f}% (N={row[0][1]})  "
          f"rank {rank} of 16")
    print("   " + "  ".join(f"{k:+d}:{row[k][0]:+.2f}" for k in range(-10, 6)))

print("\n" + "=" * 78)
print("3. THE Aug-end CELL AGAINST ITS CONTROLS + midterm split")
print("=" * 78)
for t in VEH:
    for h in (3, 5, 7, 10):
        d, r, m = cell(t, 0, h)
        sel = m == 8
        a, dd = r[sel], d[sel]
        c_aug = drift(t, h, [8, 9])
        c_all = drift(t, h)
        mid = np.array([x.year % 4 == 2 for x in dd])
        rows = [summarize(a, f"{t} Aug-end ME-3 h={h} (N={len(a)})"),
                summarize(c_aug, "CTRL Aug+Sep all days"),
                summarize(c_all, "CTRL all days"),
                summarize(a[mid], f"midterm (N={int(mid.sum())})"),
                summarize(a[~mid], "non-midterm")]
        show(rows)
        w = int((a > 0).sum())
        up = float((r > 0).mean())
        print(f"  record {w}-{len(a)-w}; sign p vs the vehicle's OWN up-rate "
              f"{100*up:.1f}% = {sign_test(w, len(a), up):.4f}; "
              f"excess vs Aug+Sep control {100*(a.mean()-c_aug.mean()):+.3f}pp "
              f"= {100*(a.mean()-c_aug.mean())*100/6:.1f}x cost")
        print(f"  {cluster_note(dd, a)}")
        print(f"  years: {[(str(x.year), round(100*y,2)) for x, y in zip(dd, a)]}")
        print()

print("=" * 78)
print("4. T3 OVERLAP: does an ME-3 hold reach September opex?")
print("=" * 78)
ev = load_events(["opex"])
idx = ser["IWM"].index
for p in ltd_positions(idx):
    if idx[p].month == 8 and idx[p].year >= 2015:
        e = p - 3
        for h in (5, 10):
            x = min(e + h, len(idx) - 1)
            sep_opex = ev[(ev["date"].dt.year == idx[p].year)
                          & (ev["date"].dt.month == 9)]["date"]
            ov = bool(len(sep_opex) and (sep_opex.iloc[0] <= idx[x]))
            if h == 10:
                print(f"  {idx[p].year}: entry {idx[e].date()} h=10 exit "
                      f"{idx[x].date()} vs Sep opex "
                      f"{sep_opex.iloc[0].date() if len(sep_opex) else '?'} "
                      f"-> overlap {ov}")
