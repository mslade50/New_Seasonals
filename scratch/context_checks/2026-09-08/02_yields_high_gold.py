"""The surprise from 01: GOLD rises after ^TNX prints a 252d max.

h21 26-11 up, mean +1.96%, edge +0.95pp over all days. Counterintuitive enough that
it needs an era split, a concentration check, and a control that is not just "gold
went up over the sample". Also checks the IEF h5 belly cell from the same script.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (close_panel, rolling_on_valid, fwd_ret, declusters,
                       local_control, summarize, era_split, sign_test,
                       cluster_note, show)

px = close_panel(["^TNX", "GC=F", "IEF", "TLT", "SPY"])
tnx = px["^TNX"]
mx = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
at_high = ((tnx >= mx * 0.9999) & tnx.notna() & mx.notna()).fillna(False)
dates = px.index[at_high]

def cell(sub, h, gap, label):
    f = fwd_ret(px[sub], h)
    valid = f.dropna().index
    d = pd.DatetimeIndex(dates).intersection(valid)
    e = declusters(d, gap, valid)
    v = f.loc[e].values
    up = int((v > 0).sum())
    r = summarize(v, label)
    base = f.loc[valid]
    print(f"\n### {label}")
    print(f"  n={r['n']} mean={r['mean_pct']:+.3f}% median={r['median_pct']:+.3f}% "
          f"hit={r['hit']:.1f}% t={r['t']:.2f} record {up}-{r['n']-up} "
          f"sign_p={sign_test(up, r['n']):.4f}")
    print(f"  all-days control {100*base.mean():+.3f}%  edge {r['mean_pct']-100*base.mean():+.3f}pp")
    ctl = local_control(valid, d, 126)
    cv = f.loc[ctl]
    print(f"  local +/-126td control n={len(cv)} mean={100*cv.mean():+.3f}% "
          f"hit={100*(cv>0).mean():.1f}%  edge vs local {r['mean_pct']-100*cv.mean():+.3f}pp")
    show(era_split(e, v), f"{label} era split")
    print("  " + cluster_note(e, v, k=2))
    yr = pd.Series(v, index=e).groupby(e.year).agg(['mean', 'count'])
    print("  by year (mean frac, n):")
    print("   ", {int(y): (round(100*m, 2), int(c)) for y, (m, c) in yr.iterrows()})
    return e, v

print("=" * 70)
print("GOLD after ^TNX at a 252-day high")
print("=" * 70)
for h in (5, 10, 21, 42):
    cell("GC=F", h, max(h, 5), f"GC=F h={h}")

print("\n" + "=" * 70)
print("IEF (belly) h5 after ^TNX at a 252-day high")
print("=" * 70)
cell("IEF", 5, 5, "IEF h=5")
cell("TLT", 5, 5, "TLT h=5")

# Is the gold cell just "yields up and gold up together in the 2000s"?
print("\n" + "=" * 70)
print("CONTROL: gold's h21 in the 2018+ era only, at-high vs everything else")
print("=" * 70)
f = fwd_ret(px["GC=F"], 21)
valid = f.dropna().index
d = pd.DatetimeIndex(dates).intersection(valid)
e = declusters(d, 21, valid)
for lo, hi, lab in [("1999-01-01", "2018-01-01", "pre-2018"),
                    ("2018-01-01", "2027-01-01", "2018+")]:
    m = (e >= lo) & (e < hi)
    sel = e[m]
    v = f.loc[sel].values
    if len(v) == 0:
        continue
    up = int((v > 0).sum())
    bm = (valid >= lo) & (valid < hi)
    base = f.loc[valid[bm]]
    print(f"{lab}: n={len(v)} mean={100*v.mean():+.3f}% hit={100*(v>0).mean():.1f}% "
          f"record {up}-{len(v)-up} sign_p={sign_test(up, len(v)):.4f} | "
          f"same-era all-days {100*base.mean():+.3f}% edge {100*(v.mean()-base.mean()):+.3f}pp")
