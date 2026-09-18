"""B1 enumeration: the live SEASONAL and CYCLE cells for the week of
2026-09-08, so the surface map can give each one a verdict instead of
leaving the seasonal lane silently absent.

The state file's seasonal board is stale (asof 2026-08-05) and carries no
live tickets, so the cells are recomputed here from raw bars.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

TK = ["SPY", "IWM", "TLT", "GLD", "USO", "UUP", "EFA", "HYG", "GDX", "SLV", "XLE"]
px = close_panel(TK)
print(f"panel {px.index[0].date()} .. {px.index[-1].date()}")

# --- Cell 1: the post-Labor-Day stretch. Labor Day is the first Monday of
# September; the anchor is the first session after it. Measure h=1..10 from
# an MOC entry on that session (lag=1 -> the order rests the session after,
# which is what a pitch published this morning actually gets).
idx = px.index
is_sep = idx.month == 9
labor_anchor = []
for yr in sorted(set(idx[is_sep].year)):
    sep = idx[(idx.year == yr) & (idx.month == 9)]
    if len(sep) == 0:
        continue
    # first Monday of September = Labor Day (closed). The anchor is the first
    # session on or after that Monday, i.e. the first September session whose
    # weekday is Tuesday and which follows a >=3 calendar-day gap.
    first = sep[0]
    # walk to the session immediately after the Labor Day closure
    prev = idx[idx < sep[0]]
    cand = [d for i, d in enumerate(sep)
            if (d - (idx[idx.get_loc(d) - 1])).days >= 3 and d.day <= 8]
    if cand:
        labor_anchor.append(cand[0])
labor_anchor = pd.DatetimeIndex(sorted(set(labor_anchor)))
print(f"\nLabor-Day anchors found: {len(labor_anchor)}  "
      f"{[str(d.date()) for d in labor_anchor]}")

print("\n=== Cell 1: post-Labor-Day session, h=1..10, lag=1 MOC entry ===")
for t in TK:
    rows = []
    for h in (1, 2, 3, 5, 10):
        f = fwd_lag(px[t], h, lag=1)
        v = f.reindex(labor_anchor).dropna().to_numpy()
        allv = f.dropna().to_numpy()
        if len(v) == 0:
            continue
        s = summarize(v, f"h={h}")
        w = int((v > 0).sum())
        rows.append(f"h={h:<2} n={len(v):<3} mean {s['mean_pct']:+.3f}% "
                    f"med {s['median_pct']:+.3f}% hit {100*w/len(v):.1f}% "
                    f"sign p {sign_test(w, len(v)):.4f} | all-days "
                    f"{summarize(allv)['mean_pct']:+.3f}%")
    print(f"-- {t}")
    for r in rows:
        print("   " + r)

# --- Cell 2: September month-position. Today is trading-day-of-month 6 of
# September (2026-09-08 is the 6th September session). What does the rest of
# the month pay from that position, and does the midterm year differ?
print("\n=== Cell 2: September trading-day-of-month position ===")
sep_pos = {}
for yr in sorted(set(idx[idx.month == 9].year)):
    sep = idx[(idx.year == yr) & (idx.month == 9)]
    for i, d in enumerate(sep):
        sep_pos[d] = i + 1
tdom = pd.Series(sep_pos)
target = tdom[tdom == 6].index  # 2026-09-08 is Sep session #6 (1,2,3,4,8 = 5)
print(f"September tdom==6 anchors: {len(target)}")
for t in ("SPY", "IWM"):
    for h in (5, 10):
        f = fwd_lag(px[t], h, lag=1)
        v = f.reindex(target).dropna()
        mid = v[[d.year % 4 == 2 for d in v.index]]
        non = v[[d.year % 4 != 2 for d in v.index]]
        for lbl, arr in (("all", v), ("midterm", mid), ("non-mid", non)):
            a = arr.to_numpy()
            if len(a) == 0:
                continue
            w = int((a > 0).sum())
            print(f"{t} h={h:<3} {lbl:<8} n={len(a):<3} "
                  f"mean {summarize(a)['mean_pct']:+.3f}% hit {100*w/len(a):.1f}% "
                  f"sign p {sign_test(w, len(a)):.4f}")

# --- Cell 3: the cycle-year conditioner on the whole September month.
print("\n=== Cell 3: full-September return by cycle year (SPY, IWM) ===")
for t in ("SPY", "IWM"):
    recs = []
    for yr in sorted(set(idx.year)):
        sep = idx[(idx.year == yr) & (idx.month == 9)]
        aug = idx[(idx.year == yr) & (idx.month == 8)]
        if len(sep) < 5 or len(aug) == 0:
            continue
        r = px[t].loc[sep[-1]] / px[t].loc[aug[-1]] - 1.0
        if np.isfinite(r):
            recs.append((yr, r))
    mid = [r for y, r in recs if y % 4 == 2]
    non = [r for y, r in recs if y % 4 != 2]
    for lbl, a in (("all", [r for _, r in recs]), ("midterm", mid),
                   ("non-mid", non)):
        a = np.asarray(a)
        if len(a) == 0:
            continue
        w = int((a > 0).sum())
        print(f"{t} {lbl:<8} n={len(a):<3} mean {summarize(a)['mean_pct']:+.3f}% "
              f"hit {100*w/len(a):.1f}% sign p {sign_test(w, len(a)):.4f}")
