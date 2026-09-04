"""The Aug-13 seasonal cells are all N=5 or 6 and the h1 and h5 halves point opposite
ways, which is what noise looks like. Widen to a mid-August window in midterm years and
see whether any shape survives at usable N. If it does not, the cells are dead and the
map says so.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, fwd_ret, sign_test, summarize  # noqa: E402

px = close_panel(["SPY", "^GSPC", "QQQ", "IWM", "TLT", "IEF"])

print("=== mid-August windows, midterm years vs everything else ===")
for tkr in ("^GSPC", "SPY", "TLT", "IEF"):
    s = px[tkr].dropna()
    for lo, hi in ((10, 18), (8, 20)):
        m = (s.index.month == 8) & (s.index.day >= lo) & (s.index.day <= hi)
        mid = m & (s.index.year % 4 == 2)
        oth = m & (s.index.year % 4 != 2)
        for h in (1, 5):
            f = fwd_ret(s, h)
            for lbl, mask in (("midterm", mid), ("other", oth)):
                a = s.index[mask].intersection(f.dropna().index)
                v = f.loc[a].values
                if len(v) == 0:
                    continue
                d = summarize(v)
                up = int((v > 0).sum())
                yrs = len(set(a.year))
                print(f"  {tkr:<6} Aug {lo}-{hi} {lbl:<8} h{h}  n={len(v):<4} "
                      f"({yrs} yrs)  mean={d['mean_pct']:+.3f}%  med={d['median_pct']:+.3f}%  "
                      f"hit={d['hit']:.1f}%  t={d['t']:+.2f}  {up}-{len(v) - up} up  "
                      f"sign p={sign_test(up, len(v)):.4f}")
        print()

print("=== year by year: the Aug 10-18 window, midterm years ===")
for tkr in ("^GSPC", "TLT"):
    s = px[tkr].dropna()
    print(f"  {tkr}")
    for y in sorted(set(s.index.year)):
        if y % 4 != 2:
            continue
        m = (s.index.year == y) & (s.index.month == 8) & \
            (s.index.day >= 10) & (s.index.day <= 18)
        idx = s.index[m]
        if len(idx) < 2:
            continue
        r = (s.loc[idx[-1]] / s.loc[idx[0]] - 1) * 100
        print(f"    {y}  {idx[0].date()} -> {idx[-1].date()}  {r:+.2f}%")

print("\n=== the engine's exact Aug-13 anchor, all years, for the record ===")
for tkr in ("SPY", "^GSPC", "TLT"):
    s = px[tkr].dropna()
    anchors = []
    for y in sorted(set(s.index.year)):
        cand = s.index[(s.index.year == y) & (s.index.month == 8) &
                       (s.index.day >= 11) & (s.index.day <= 15)]
        if len(cand):
            anchors.append(cand[0])
    a = pd.DatetimeIndex(anchors)
    for h in (1, 5):
        f = fwd_ret(s, h)
        aa = a.intersection(f.dropna().index)
        v = f.loc[aa].values
        d = summarize(v)
        up = int((v > 0).sum())
        print(f"  {tkr:<6} Aug 11-15 anchor h{h}  n={len(v):<4} mean={d['mean_pct']:+.3f}%  "
              f"hit={d['hit']:.1f}%  {up}-{len(v) - up} up  sign p={sign_test(up, len(v)):.4f}")
        mm = aa[aa.year % 4 == 2]
        v2 = f.loc[mm].values
        d2 = summarize(v2)
        up2 = int((v2 > 0).sum())
        print(f"  {tkr:<6}    midterm only  h{h}  n={len(v2):<4} mean={d2['mean_pct']:+.3f}%  "
              f"{up2}-{len(v2) - up2} up  sign p={sign_test(up2, len(v2)):.4f}")
