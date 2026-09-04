"""07 showed the Aug-13 "midterm" bond cell is a slice of a window effect that is just
as strong in non-midterm years (TLT h5 71.1% midterm vs 69.6% other). But those anchor
counts are overlapping daily observations inside the same handful of Augusts, so N=153
is not 153 independent facts. Redo it at the YEAR level, which is the honest unit, and
check whether the window is distinguishable from August as a whole.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, sign_test, summarize  # noqa: E402

px = close_panel(["TLT", "IEF", "^GSPC", "^TNX"])


def window_move(s: pd.Series, year: int, d0: int, d1: int):
    idx = s.index[(s.index.year == year) & (s.index.month == 8) &
                  (s.index.day >= d0) & (s.index.day <= d1)]
    if len(idx) < 2:
        return None
    return s.loc[idx[-1]] / s.loc[idx[0]] - 1.0


for tkr in ("TLT", "IEF", "^GSPC"):
    s = px[tkr].dropna()
    print(f"\n=== {tkr}: Aug 10 -> Aug 18, one observation per year ===")
    rows = []
    for y in sorted(set(s.index.year)):
        r = window_move(s, y, 10, 18)
        if r is None or y == 2026:
            continue
        rows.append((y, r))
    v = np.array([r for _, r in rows])
    up = int((v > 0).sum())
    d = summarize(v)
    print(f"  n={len(v)} years  mean={d['mean_pct']:+.3f}%  med={d['median_pct']:+.3f}%  "
          f"{up}-{len(v) - up} up  sign p={sign_test(up, len(v)):.4f}  t={d['t']:+.2f}")
    mid = np.array([r for y, r in rows if y % 4 == 2])
    oth = np.array([r for y, r in rows if y % 4 != 2])
    for lbl, a in (("midterm", mid), ("other", oth)):
        if len(a) == 0:
            continue
        dd = summarize(a)
        u = int((a > 0).sum())
        print(f"    {lbl:<8} n={len(a):<3} mean={dd['mean_pct']:+.3f}%  "
              f"{u}-{len(a) - u} up  sign p={sign_test(u, len(a)):.4f}")
    print("    year by year: " + "  ".join(f"{y}:{100 * r:+.1f}" for y, r in rows))

    # the rest of August, as the control the window has to beat
    ctrl = []
    for y in sorted(set(s.index.year)):
        if y == 2026:
            continue
        full = window_move(s, y, 1, 31)
        w = window_move(s, y, 10, 18)
        if full is None or w is None:
            continue
        ctrl.append(full)
    c = np.array(ctrl)
    dd = summarize(c)
    u = int((c > 0).sum())
    print(f"  control, the WHOLE of August: n={len(c)} mean={dd['mean_pct']:+.3f}%  "
          f"{u}-{len(c) - u} up  sign p={sign_test(u, len(c)):.4f}")

    # every other 7-calendar-day window in August, as a placebo
    print("  placebo windows in August (mean %, up-down):")
    for d0 in (1, 5, 10, 15, 20, 24):
        vv = [window_move(s, y, d0, d0 + 8) for y in sorted(set(s.index.year))
              if y != 2026]
        vv = np.array([x for x in vv if x is not None])
        if len(vv) < 10:
            continue
        u = int((vv > 0).sum())
        print(f"    Aug {d0:>2}-{d0 + 8:<2}  n={len(vv):<3} "
              f"mean={100 * vv.mean():+.3f}%  {u}-{len(vv) - u} up  "
              f"sign p={sign_test(u, len(vv)):.4f}")

print("\n=== and the same window measured on the 10-year YIELD ===")
s = px["^TNX"].dropna()
rows = [(y, window_move(s, y, 10, 18)) for y in sorted(set(s.index.year)) if y != 2026]
rows = [(y, r) for y, r in rows if r is not None]
v = np.array([r for _, r in rows])
dn = int((v < 0).sum())
print(f"  n={len(v)} years  mean={100 * v.mean():+.3f}%  "
      f"{dn}-{len(v) - dn} DOWN  sign p={sign_test(dn, len(v)):.4f}")
