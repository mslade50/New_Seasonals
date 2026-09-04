"""Correct anchor: TODAY is the 4th-to-last session of August, so h1 is
tomorrow, the 3rd-to-last.

Drill 04 anchored on the 3rd-to-last session, which makes h1 the SECOND-to-last
session, not tomorrow. The context convention anchors on the session that
printed, so the cell describing tomorrow anchors on today. This redoes it at
the right offset, and the answer flips: the bond leg was never about tomorrow,
the equity leg is.

Then the obvious confound, because Jackson Hole also sits in this week every
year: is "3rd-to-last session of August" just the symposium session wearing a
calendar label? Yesterday's brief published an IWM Jackson Hole cell at 21-5,
and this cell arrives at 21-5 too, so they have to be separated before either
can be told again.

Convention: lag=0 close-to-close from the anchor close, h=1 is the next
session.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from pitch_lab import close_panel, load_events, summarize, sign_test, cluster_note

SUBJECTS = ["SPY", "^GSPC", "IWM", "QQQ", "TLT", "IEF", "HYG"]
px = close_panel(SUBJECTS).sort_index()
all_dates = px.index

pos_of = pd.Series(np.arange(len(all_dates)), index=all_dates)
ym = all_dates.to_period("M")
last_pos = pd.Series(pos_of.values, index=all_dates).groupby(ym).transform("max")
dist_me = pd.Series(last_pos.values - pos_of.values, index=all_dates)
CUR = all_dates.max().to_period("M")


def slot(month, k):
    sel = all_dates[((dist_me == k).values) & (all_dates.month == month)]
    return sel[sel.to_period("M") != CUR]


def fwd(sym, dates, h=1):
    s = px[sym].dropna()
    p = s.index.searchsorted(dates)
    ok = (p + h < len(s)) & (p < len(s))
    p = p[ok]
    used = s.index[p]
    keep = used.isin(dates)
    p, used = p[keep], used[keep]
    return used, (s.values[p + h] / s.values[p]) - 1.0


def line(lbl, dd, vv, extra=""):
    if len(vv) < 3:
        print(f"  {lbl:32s} n={len(vv):3d}  (too few)")
        return
    s = summarize(vv, "")
    up = int((vv > 0).sum())
    print(f"  {lbl:32s} n={len(vv):3d}  mean={s['mean_pct']:+6.3f}%  "
          f"{up}-{len(vv)-up} up  t={s['t']:+5.2f}  "
          f"signp={sign_test(up, len(vv)):.4f}{extra}")


A = slot(8, 3)   # today's slot: 4th-to-last August session
print(f"anchor = 4th-to-last session of August (today's slot), n={len(A)} years")
print(f"  {A.min().date()} .. {A.max().date()}")
print(f"  h1 from this anchor = the 3rd-to-last session = TOMORROW 2026-08-27")

print("\n" + "=" * 78)
print("A. h1 by subject, anchored on today's slot")
print("=" * 78)
for sym in SUBJECTS:
    d, v = fwd(sym, A, 1)
    line(sym, d, v)

print("\n" + "=" * 78)
print("B. SPY: kill attempts")
print("=" * 78)
d, v = fwd("SPY", A, 1)
s = summarize(v, "")
print(f"  median {s['median_pct']:+6.3f}%   worst {s['worst_pct']:+6.2f}%  "
      f"best {s['best_pct']:+6.2f}%  sd {s['sd_pct']:.2f}%")
print(f"  concentration: {cluster_note(d, v, k=2)}")
order = np.argsort(v)[::-1]
line("drop best 2", d[np.isin(np.arange(len(v)), order[2:])],
     np.delete(v, order[:2]))
pre = d < pd.Timestamp("2018-01-01")
line("pre-2018", d[pre], v[pre])
line("2018+", d[~pre], v[~pre])
mid = (d.year % 4 == 2)
line("midterm years", d[mid], v[mid])
line("non-midterm", d[~mid], v[~mid])

print("\n  year by year:")
for dt, val in zip(d, v):
    print(f"    {dt.date()} -> {val*100:+6.2f}%")

print("\n" + "=" * 78)
print("C. Control: same slot in the other eleven months, and all days")
print("=" * 78)
for sym in ["SPY", "IWM", "^GSPC"]:
    others = all_dates[((dist_me == 3).values) & (all_dates.month != 8)]
    do, vo = fwd(sym, others, 1)
    line(f"{sym}: 4th-to-last ex-August", do, vo)
    sa = px[sym].dropna().pct_change().shift(-1).dropna()
    print(f"  {sym+': all-days control':34s} n={len(sa):4d}  "
          f"mean={sa.mean()*100:+6.3f}%  hit={(sa > 0).mean():.1%}")

print("\n" + "=" * 78)
print("D. THE CONFOUND: is this the Jackson Hole session?")
print("=" * 78)
ev = load_events(["jackson_hole"])
jh = pd.to_datetime(sorted(ev["date"].unique()))
jh_set = set(jh)

# For each anchor year, what is the h1 session, and is it a JH day?
sd = px["SPY"].dropna()
rows = []
for dt in d:
    p = sd.index.searchsorted(dt)
    nxt = sd.index[p + 1]
    # distance in calendar days from the h1 session to that year's JH date
    same_year_jh = [j for j in jh if j.year == nxt.year]
    gap = min((abs((nxt - j).days) for j in same_year_jh), default=None)
    rows.append((dt, nxt, gap))
print(f"  {'anchor':12s} {'h1 session':12s} {'days from JH':>13s}")
for a, n, g in rows:
    print(f"  {str(a.date()):12s} {str(n.date()):12s} {str(g):>13s}")

gaps = np.array([g for _, _, g in rows if g is not None])
print(f"\n  h1 session is a Jackson Hole day itself: "
      f"{sum(1 for _, n, _ in rows if n in jh_set)} of {len(rows)}")
print(f"  h1 within 1 calendar day of JH: {(gaps <= 1).sum()} of {len(gaps)}")
print(f"  h1 within 3 calendar days of JH: {(gaps <= 3).sum()} of {len(gaps)}")

far = np.array([g is not None and g > 3 for _, _, g in rows])
if far.sum() >= 4:
    line("SPY, h1 MORE than 3d from JH", d[far], v[far])
near = ~far
if near.sum() >= 4:
    line("SPY, h1 within 3d of JH", d[near], v[near])
