"""The best untouched event cell in tonight's sweep: the dollar on Sep 1.

E:seasonal_doy for DX-Y.NYB reports 19-7 up all years at sign p 0.0145 on the
matching trading day of year, and 6 of 6 up in midterm years at +0.533%,
sign p 0.0156. Tomorrow is the first session of September in a midterm year.

Three things to establish before any of that publishes:
  1. Is the effect the CALENDAR SLOT (trading day of year, plus or minus 2) or
     is it just "the first session of a month", which is a different and much
     larger cell?
  2. Does it survive as a September-specific statement, i.e. is September's
     first session different from the other eleven?
  3. Era stability, and how much of the mean sits in one or two years.

Convention: lag=0 close-to-close from the anchor close (the final session of
August), so h=1 is 2026-09-01 itself.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, era_split, fwd_ret, show, sign_test, summarize  # noqa: E402

px = close_panel(["DX-Y.NYB", "EURUSD=X", "JPY=X", "SPY", "GC=F", "EEM"])
dxy = px["DX-Y.NYB"].dropna()
dxy = dxy[dxy.index >= "1999-01-01"]
idx = dxy.index

# anchor = the LAST session of a month; h=1 is the first session of the next
month = pd.Series(idx.month, index=idx)
is_month_end = month != month.shift(-1)
anchors = idx[is_month_end.values]
anchors = anchors[:-1] if anchors[-1] == idx[-1] else anchors

f1 = fwd_ret(dxy, 1)
f5 = fwd_ret(dxy, 5)

rows = []
for name, sel in [
    ("ALL month-ends -> first session", anchors),
    ("-> first session of SEPTEMBER", anchors[anchors.month == 8]),
    ("-> first session of the other 11", anchors[anchors.month != 8]),
]:
    d = pd.DatetimeIndex(sel).intersection(f1.dropna().index)
    v = f1.loc[d].values
    r = summarize(v, name)
    r["record"] = f"{int((v>0).sum())}-{int((v<0).sum())}"
    r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(r)
show(rows, "DX-Y.NYB h=1, the first session of a month")

sep = anchors[anchors.month == 8]
mid = sep[sep.year % 4 == 2]
print("\n=== September first-session detail, by year ===")
for d in sep:
    if d in f1.index and np.isfinite(f1.loc[d]):
        tag = " MIDTERM" if d.year % 4 == 2 else ""
        print(f"  {d.date()} -> {100*f1.loc[d]:+6.2f}%   5d {100*f5.loc[d]:+6.2f}%{tag}")

for label, sel in [("September, midterm years", mid),
                   ("September, non-midterm", sep[sep.year % 4 != 2])]:
    d = pd.DatetimeIndex(sel).intersection(f1.dropna().index)
    v = f1.loc[d].values
    up = int((v > 0).sum())
    print(f"\n{label}: n={len(v)} {up}-{len(v)-up} up  mean {100*v.mean():+.3f}%  "
          f"med {100*np.median(v):+.3f}%  sign_p {sign_test(up, len(v)):.4f}")

print("\n=== era split, September first session, all years ===")
d = pd.DatetimeIndex(sep).intersection(f1.dropna().index)
show(era_split(d, f1.loc[d].values), "DX-Y.NYB h=1")

print("\n=== control: is 'first session of any month' already positive? ===")
d = pd.DatetimeIndex(anchors).intersection(f1.dropna().index)
allday = f1.dropna()
print(f"  first-session cell n={len(d)} mean {100*f1.loc[d].mean():+.4f}%")
print(f"  all days           n={len(allday)} mean {100*allday.mean():+.4f}%")
print(f"  edge {100*(f1.loc[d].mean() - allday.mean()):+.4f}pp")

print("\n=== the mirror: EURUSD on the same slot (a dollar move should show up here) ===")
eur = px["EURUSD=X"].dropna()
fe = fwd_ret(eur, 1)
for label, sel in [("Sep 1 all years", sep), ("Sep 1 midterm", mid)]:
    d = pd.DatetimeIndex(sel).intersection(fe.dropna().index)
    v = fe.loc[d].values
    dn = int((v < 0).sum())
    print(f"  {label}: n={len(v)} {dn} of {len(v)} lower (dollar up), "
          f"mean {100*v.mean():+.3f}%, sign_p {sign_test(dn, len(v)):.4f}")

print("\n=== h=5, does it hold for a week? ===")
for label, sel in [("Sep 1 all years", sep), ("Sep 1 midterm", mid)]:
    d = pd.DatetimeIndex(sel).intersection(f5.dropna().index)
    v = f5.loc[d].values
    up = int((v > 0).sum())
    print(f"  DXY {label}: n={len(v)} {up}-{len(v)-up} up mean {100*v.mean():+.3f}% "
          f"sign_p {sign_test(up, len(v)):.4f}")
