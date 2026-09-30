"""Engine cell: VIX on Tuesdays in September, +1.89%, t 2.49, N 113, era stable.
Sunday's 2x2 showed the Monday cell is the weekday, not the month. Is the Tuesday
cell the month, the weekday, or a handful of crisis Tuesdays?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, sign_test, cluster_note  # noqa

px = close_panel(["^VIX", "SPY"]).dropna(subset=["^VIX"])
px = px[px.index >= "1999-01-01"]
vr = px["^VIX"].pct_change().dropna()
vr = vr[vr.index < px.index[-1] + pd.Timedelta(days=1)]
dow, mon = vr.index.dayofweek, vr.index.month


def rep(v, label):
    v = np.asarray(v, float)
    up = int((v > 0).sum())
    t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
    print(f"  {label:32} n {len(v):5}  mean {100*v.mean():+6.2f}%  median {100*np.median(v):+6.2f}%  "
          f"rec {up}-{len(v)-up}  signp_up {sign_test(up, len(v)):.4f}  t {t:+.2f}")
    return v


a = rep(vr[(dow == 1) & (mon == 9)], "Sep Tuesdays")
b = rep(vr[(dow == 1) & (mon != 9)], "Tuesdays other months")
rep(vr[(dow != 1) & (mon == 9)], "Sep non-Tuesdays")
rep(vr[(dow == 1)], "all Tuesdays")
se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
print(f"  Welch t Sep Tue vs other Tue: {(a.mean()-b.mean())/se:+.2f}")
s = vr[(dow == 1) & (mon == 9)]
print("  concentration:", cluster_note(s.index, s.values, k=3))
print("  top 6 Sep Tuesdays:", [(str(d.date()), round(100 * x, 1)) for d, x in s.sort_values(ascending=False).head(6).items()])
trim = s.sort_values().iloc[:-3]
rep(trim, "Sep Tuesdays, drop top 3")
print("  by 5y block:")
for lo in range(1999, 2027, 5):
    ss = s[(s.index.year >= lo) & (s.index.year < lo + 5)]
    if len(ss):
        print(f"    {lo}-{lo+4}: n {len(ss):3} mean {100*ss.mean():+6.2f}% up {int((ss>0).sum())}")
print("  per month, Tuesdays:")
for m in range(1, 13):
    ss = vr[(dow == 1) & (mon == m)]
    print(f"    {m:2}: n {len(ss):3} mean {100*ss.mean():+6.2f}%  median {100*ss.median():+6.2f}%")
# the Tuesday AFTER the post-Sept-expiry Monday, and Tuesdays after an up Monday for SPY
sr = px["SPY"].pct_change()
prev_up = sr.shift(1).reindex(vr.index) >= 0.01
rep(vr[(dow == 1) & prev_up.values], "Tuesdays after SPY Monday +1%+")
rep(vr[(dow == 1) & (mon == 9) & prev_up.values], "Sep Tuesdays after SPY Mon +1%+")

# The Tuesday after Labor Day is a post-long-weekend session, i.e. a Monday in VIX terms.
prev_sess = pd.Series(px.index, index=px.index).shift(1).reindex(vr.index)
post_hol = (prev_sess.dt.dayofweek == 4).values & (dow == 1)
print("\n=== excluding Tuesdays that follow a Monday holiday ===")
a2 = rep(vr[(dow == 1) & (mon == 9) & ~post_hol], "Sep Tuesdays, ex post-holiday")
rep(vr[(dow == 1) & (mon == 9) & post_hol], "Sep Tuesdays after Labor Day")
b2 = rep(vr[(dow == 1) & (mon != 9) & ~post_hol], "other Tuesdays, ex post-holiday")
rep(vr[(dow == 1) & post_hol], "all post-Monday-holiday Tuesdays")
se = np.sqrt(a2.var(ddof=1) / len(a2) + b2.var(ddof=1) / len(b2))
print(f"  Welch t: {(a2.mean()-b2.mean())/se:+.2f}")
s2 = vr[(dow == 1) & (mon == 9) & ~post_hol]
print("  concentration:", cluster_note(s2.index, s2.values, k=3))
m = s2.index < "2018-01-01"
rep(s2[m], "  pre-2018")
rep(s2[~m], "  2018+")
# late-September Tuesdays only (day >= 15), today's slot
late = s2[s2.index.day >= 15]
rep(late, "Sep Tuesdays day>=15, ex holiday")
# SPY on those same Tuesdays
spy_r = sr.reindex(s2.index)
rep(spy_r.dropna(), "SPY on Sep Tuesdays ex holiday")
