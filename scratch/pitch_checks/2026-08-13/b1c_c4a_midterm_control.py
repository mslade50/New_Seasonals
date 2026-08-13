"""b1c - C4a: the MIDTERM claim against the same calendar control.

b1b showed the JH anchor is a 24-day subsample of the Aug 6-16 window and that
the window dies after 2018. This asks whether the midterm restriction (5-1,
+1.495%) is anything more than "the four bond-bull midterms".
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

px = close_panel(["TLT", "GLD", "DX-Y.NYB"])
r = {t: fwd_lag(px[t], 10, 1) for t in px.columns}

for t in ("TLT", "GLD", "DX-Y.NYB"):
    rr = r[t].dropna()
    aug = rr[(rr.index.month == 8) & (rr.index.day >= 6) & (rr.index.day <= 16)]
    mid = aug[aug.index.year % 4 == 2]
    non = aug[aug.index.year % 4 != 2]
    show([summarize(aug.values, "Aug6-16 ALL days"),
          summarize(mid.values, f"Aug6-16 MIDTERM yrs {sorted(set(mid.index.year))}"),
          summarize(non.values, "Aug6-16 NON-midterm"),
          summarize(mid[mid.index.year >= 2018].values, "Aug6-16 MIDTERM 2018+"),
          summarize(non[non.index.year >= 2018].values, "Aug6-16 NON-mid 2018+")],
         f"{t}: calendar-window control for the midterm claim (h=10, lag=1)")
    # per-midterm-year mean of the window
    pm = mid.groupby(mid.index.year).mean() * 100
    print("  per-midterm-year window mean %:", {int(k): round(v, 2) for k, v in pm.items()})
    print()

# the JH-anchored midterm record split by era
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
jh = load_events(["jackson_hole"])["date"]
anch = []
for d in jh:
    p = pos.get(d)
    if p is None:
        later = idx[idx >= d]
        if len(later) == 0:
            continue
        p = pos[later[0]]
    if p - 11 >= 0:
        anch.append((d.year, idx[p - 11]))
for t in ("TLT", "GLD", "DX-Y.NYB"):
    sel = [(y, d) for y, d in anch if y % 4 == 2 and not np.isnan(r[t].get(d, np.nan))]
    v = np.array([r[t][d] for _, d in sel])
    yrs = [y for y, _ in sel]
    print(f"{t} JH-anchored MIDTERM per year: "
          f"{dict(zip(yrs, np.round(100*v, 2)))}")
    pre = v[np.array(yrs) < 2018]
    post = v[np.array(yrs) >= 2018]
    print(f"   pre-2018 midterms N={len(pre)} mean {100*pre.mean():+.3f}% | "
          f"2018+ midterms N={len(post)} mean {100*post.mean():+.3f}%")
