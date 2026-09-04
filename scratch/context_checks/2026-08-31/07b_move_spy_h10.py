"""The one arm of drill 07 that stood out, and the deflator that goes with it.

07 found SPY 45-13 up over the next ten sessions after a MOVE jump off a low
base (n=58, sign p 0.0000, mean +0.737%, t 1.86, edge over the local +/-126td
control +0.277pp). A 77.6% hit rate on 58 observations wants a closer look, and
so does the fact that the t-stat is only 1.86, which means tails.

It also found the deflator: a 6%+ MOVE session is 7.95% of all sessions and
2026 has produced six of them since June. "Bond vol jumped" is not news on its
own. The conditioning is what makes the cell.

Three checks: overlap (h=10 windows on 58 anchors overlap heavily, so the
episode-declustered version is the honest one), eras, and concentration.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, cluster_note, declusters, era_split,  # noqa: E402
                       fwd_ret, local_control, show, sign_test, summarize)

px = close_panel(["^MOVE", "SPY", "TLT"])
px = px[px.index >= "1999-01-01"]
mv = px["^MOVE"].dropna()
r1 = mv.pct_change()
r21 = mv.pct_change(21)
rank21 = r21.rolling(252, min_periods=252).apply(
    lambda w: 100.0 * (w[-1] > w[:-1]).mean(), raw=True)
state = (r1 >= 0.05) & (rank21 <= 33.3)


def first_in_calendar_days(mask, days=30):
    out = pd.Series(False, index=mask.index)
    last = None
    for d in mask.index[mask.fillna(False).values]:
        if last is None or (d - last).days > days:
            out.loc[d] = True
        last = d
    return out


ed = first_in_calendar_days(state, 30)
hist = ed.index[ed.values]
hist = hist[hist < pd.Timestamp("2026-08-31")]

spy = px["SPY"].dropna()
f10 = fwd_ret(spy, 10).dropna()
d = pd.DatetimeIndex(hist).intersection(f10.index)
v = f10.loc[d].values
print(f"SPY h=10, 30-day-novelty episodes: n={len(v)} "
      f"{int((v>0).sum())}-{int((v<0).sum())} up, mean {100*v.mean():+.3f}%, "
      f"median {100*np.median(v):+.3f}%, sign_p {sign_test(int((v>0).sum()), len(v)):.5f}")

print("\n=== overlap check: decluster to a 10-session minimum gap ===")
dd = declusters(d, 10, f10.index)
vd = f10.loc[dd].values
print(f"  n={len(vd)} {int((vd>0).sum())}-{int((vd<0).sum())} up, "
      f"mean {100*vd.mean():+.3f}%, sign_p {sign_test(int((vd>0).sum()), len(vd)):.5f}")
dd21 = declusters(d, 21, f10.index)
v21 = f10.loc[dd21].values
print(f"  min-gap 21: n={len(v21)} {int((v21>0).sum())}-{int((v21<0).sum())} up, "
      f"mean {100*v21.mean():+.3f}%, sign_p {sign_test(int((v21>0).sum()), len(v21)):.5f}")

print("\n=== control ===")
ctl = local_control(f10.index, d, 126)
print(f"  all days      mean {100*f10.mean():+.3f}%  hit {100*(f10>0).mean():.1f}%  n={len(f10)}")
print(f"  local +/-126  mean {100*f10.loc[ctl].mean():+.3f}%  "
      f"hit {100*(f10.loc[ctl]>0).mean():.1f}%  n={len(ctl)}")

print("\n=== eras and concentration ===")
show(era_split(d, v), "SPY h=10")
for label, m in (("pre-2018", d < pd.Timestamp("2018-01-01")),
                 ("2018+", d >= pd.Timestamp("2018-01-01"))):
    s = f10.loc[d[m]].values
    up = int((s > 0).sum())
    print(f"  {label}: {up}-{len(s)-up}, sign_p {sign_test(up, len(s)):.4f}")
print(" ", cluster_note(d, v, k=2))

print("\n=== the 13 losers, so the tail is named ===")
lo = pd.Series(v, index=d).sort_values().head(6)
for dt, x in lo.items():
    print(f"  {dt.date()} {100*x:+.2f}%")

print("\n=== MOVE itself, next session, down-side record ===")
f1 = fwd_ret(mv, 1).dropna()
d1 = pd.DatetimeIndex(hist).intersection(f1.index)
v1 = f1.loc[d1].values
dn = int((v1 < 0).sum())
print(f"  n={len(v1)}  {dn} of {len(v1)} LOWER, median {100*np.median(v1):+.2f}%, "
      f"sign_p {sign_test(dn, len(v1)):.4f}")
print(f"  base rate: MOVE lower on {100*(f1<0).mean():.1f}% of all sessions "
      f"(n={len(f1)})")
