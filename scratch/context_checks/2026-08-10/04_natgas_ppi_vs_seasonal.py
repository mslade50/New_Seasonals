"""Natural gas: the sweep's only `solid` event cell against the Aug-11 seasonal.

The sweep says NG=F rises the session after the 3-td-before-PPI anchor:
n=308, +0.452%, 57.5% hit, t=2.50, 177-130, sign p 0.0051, era stable.
The seasonal_doy cell says NG=F FALLS around Aug 11: 19 of 25 down, -1.008%,
sign p 0.0073.

Both describe roughly the same calendar position, so at least one is an
artifact. PPI lands mid-month, which makes the PPI anchor close to a fixed
day-of-month, so the obvious control is day-of-month itself.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, fwd_ret, summarize, era_split, sign_test,
    cluster_note,
)

px = close_panel(["NG=F", "CL=F"])
px = px[px.index >= "1999-01-01"]
dates = px.index
ng = px["NG=F"].dropna()
print(f"NG=F {len(ng)} bars {ng.index.min().date()} -> {ng.index.max().date()}")

f1 = fwd_ret(ng, 1)


def show(idx, label):
    v = f1.reindex(pd.DatetimeIndex(sorted(set(idx)))).dropna()
    if len(v) < 3:
        print(f"  {label:52s} n={len(v)} too few")
        return None
    r = summarize(v.values, "")
    up = int((v.values > 0).sum())
    print(f"  {label:52s} n={r['n']:4d} mean {r['mean_pct']:+.3f}% "
          f"med {r['median_pct']:+.3f}% hit {r['hit']:.1f}% t={r['t']:+.2f} | "
          f"{up}-{len(v)-up} up-p {sign_test(up, len(v)):.4f}")
    return v


print("\n--- baseline ---")
allv = show(dates, "ALL sessions")

# --- the PPI cell as the engine built it -------------------------------------
ev = load_events(["ppi"])
ev = ev[(ev["date"] >= dates[0]) & (ev["date"] <= dates[-1])]
anch = []
for d in ev["date"]:
    pos = dates.searchsorted(pd.Timestamp(d))
    if pos >= len(dates) or dates[pos] != pd.Timestamp(d) or pos - 3 < 0:
        continue
    anch.append(dates[pos - 3])
anch = pd.DatetimeIndex(anch)
print(f"\n--- the PPI k3 cell (n_anchors={len(anch)}) ---")
v_ppi = show(anch, "NG=F, 3 td before a PPI")
print(f"    day-of-month of those anchors: median {int(np.median(anch.day))}, "
      f"range {anch.day.min()}-{anch.day.max()}, "
      f"IQR {int(np.percentile(anch.day, 25))}-{int(np.percentile(anch.day, 75))}")
for e in era_split(v_ppi.index, v_ppi.values):
    print(f"    era n={e['n']:4d} mean {e['mean_pct']:+.3f}% hit {e['hit']:.1f}% t={e['t']:+.2f}")
print(f"    {cluster_note(v_ppi.index, v_ppi.values)}")

# --- the day-of-month control ------------------------------------------------
print("\n--- day-of-month control: does the whole month behave this way? ---")
lo, hi = int(np.percentile(anch.day, 25)), int(np.percentile(anch.day, 75))
same_dom = dates[(dates.day >= lo) & (dates.day <= hi)]
show(same_dom, f"ALL sessions with day-of-month {lo}-{hi}")
show(same_dom.difference(anch), f"day-of-month {lo}-{hi} but NOT a PPI k3 anchor")
for d0 in range(1, 29, 3):
    sel = dates[(dates.day >= d0) & (dates.day < d0 + 3)]
    show(sel, f"   day-of-month {d0}-{d0+2}")

# --- trading-day-of-month control, the cleaner version -----------------------
print("\n--- trading-day-of-month control ---")
tdom = pd.Series(1, index=dates).groupby([dates.year, dates.month]).cumsum()
anch_tdom = tdom.reindex(anch).dropna()
print(f"    PPI k3 anchors sit at trading-day-of-month median "
      f"{int(anch_tdom.median())}, IQR {int(anch_tdom.quantile(.25))}-"
      f"{int(anch_tdom.quantile(.75))}")
for t0 in range(1, 22, 2):
    sel = dates[(tdom.values >= t0) & (tdom.values < t0 + 2)]
    show(sel, f"   trading-day-of-month {t0}-{t0+1}")

lo2, hi2 = int(anch_tdom.quantile(.25)), int(anch_tdom.quantile(.75))
band = dates[(tdom.values >= lo2) & (tdom.values <= hi2)]
show(band, f"ALL sessions at trading-day-of-month {lo2}-{hi2}")
show(band.difference(anch), f"tdom {lo2}-{hi2} but NOT a PPI k3 anchor")

# --- the August seasonal cell -------------------------------------------------
print("\n--- the Aug-11 seasonal cell ---")
aug = dates[(dates.month == 8) & (dates.day >= 9) & (dates.day <= 13)]
show(aug, "NG=F, Aug 9-13 sessions (all)")
one_per_yr = pd.DatetimeIndex(
    [g.index[len(g) // 2] for _, g in pd.Series(0, index=aug).groupby(aug.year)])
v_sz = show(one_per_yr, "NG=F, one Aug-11 anchor per year")
if v_sz is not None:
    print(f"    {cluster_note(v_sz.index, v_sz.values)}")
    for e in era_split(v_sz.index, v_sz.values):
        print(f"    era n={e['n']:3d} mean {e['mean_pct']:+.3f}% hit {e['hit']:.1f}% t={e['t']:+.2f}")

# --- whole-month August, the honest version of 'August is bad for gas' -------
print("\n--- August as a whole ---")
for m in range(1, 13):
    show(dates[dates.month == m], f"   month {m:02d}")
