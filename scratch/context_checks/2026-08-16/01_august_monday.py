"""August Mondays: QQQ vs SPY, against all Mondays and against August's other sessions.

The engine's cell is bare: QQQ 76-40 up on 116 August Mondays, sign p 0.0005, mean only
+0.14%. A hit rate that high with a mean that small means the down days are bigger, so the
magnitudes have to be shown. And the cell has to be separated from the plain Monday effect
and from August itself.

Anchor convention: the FRIDAY close, so h1 is the Monday session itself.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, fwd_ret, summarize, sign_test, era_split, cluster_note  # noqa


TICKERS = ["QQQ", "SPY", "^GSPC", "IWM", "^VIX"]
px = close_panel(TICKERS)
idx = px.index

# the anchor is the session BEFORE a Monday, so h1 lands on the Monday
nxt_is_mon = pd.Series(idx, index=idx).shift(-1).dt.weekday == 0
nxt_is_aug = pd.Series(idx, index=idx).shift(-1).dt.month == 8
nxt_month = pd.Series(idx, index=idx).shift(-1).dt.month
nxt_wd = pd.Series(idx, index=idx).shift(-1).dt.weekday

aug_mon = (nxt_is_mon & nxt_is_aug).fillna(False)
all_mon = nxt_is_mon.fillna(False)
non_aug_mon = (nxt_is_mon & ~nxt_is_aug).fillna(False)
aug_non_mon = (~nxt_is_mon & nxt_is_aug).fillna(False)

print(f"August-Monday anchors: {int(aug_mon.sum())}   all-Monday anchors: {int(all_mon.sum())}")
print(f"first {idx[0].date()}  last {idx[-1].date()}\n")


def block(tkr: str) -> None:
    r1 = fwd_ret(px[tkr], 1)
    print("=" * 78)
    print(tkr)
    for name, m in [("August Mondays", aug_mon),
                    ("Mondays outside August", non_aug_mon),
                    ("all Mondays", all_mon),
                    ("August, other weekdays", aug_non_mon)]:
        v = r1[m.values].dropna()
        s = summarize(v.values, name)
        up = int((v > 0).sum())
        print(f"  {name:26s} n {s['n']:4d}  mean {s['mean_pct']:+.3f}%  med {s['median_pct']:+.3f}%"
              f"  hit {s['hit']:.1f}%  t {s['t']:+.2f}  {up}-{s['n']-up}  signp {sign_test(up, s['n']):.4f}")

    v = r1[aug_mon.values].dropna()
    pos, neg = v[v > 0], v[v < 0]
    print(f"  magnitudes: up days avg {100*pos.mean():+.3f}% (n {len(pos)}), "
          f"down days avg {100*neg.mean():+.3f}% (n {len(neg)})")
    va = r1[all_mon.values & ~aug_mon.values].dropna()
    pa, na = va[va > 0], va[va < 0]
    print(f"  other Mondays: up {100*pa.mean():+.3f}% (n {len(pa)}), down {100*na.mean():+.3f}% (n {len(na)})")
    print("  era:", [f"{e['label']} n {e['n']} mean {e['mean_pct']:+.3f}% hit {e['hit']:.1f}%"
                     for e in era_split(v.index, v.values)])
    print("  cluster:", cluster_note(v.index, v.values, k=2))
    mid = v[v.index.year % 4 == 2]
    mu = int((mid > 0).sum())
    print(f"  midterm years: n {len(mid)} {mu}-{len(mid)-mu} mean {100*mid.mean():+.3f}% "
          f"signp {sign_test(mu, len(mid)):.4f}")


for t in TICKERS:
    block(t)

# QQQ minus SPY on the same anchors, the piece that makes it a QQQ cell and not a Monday cell
print("\n" + "=" * 78)
print("QQQ minus SPY, same anchors")
sp = fwd_ret(px["QQQ"], 1) - fwd_ret(px["SPY"], 1)
for name, m in [("August Mondays", aug_mon), ("Mondays outside August", non_aug_mon),
                ("August, other weekdays", aug_non_mon), ("every session", pd.Series(True, index=idx))]:
    v = sp[m.values].dropna()
    if not len(v):
        continue
    s = summarize(v.values, name)
    up = int((v > 0).sum())
    print(f"  {name:26s} n {s['n']:4d}  mean {s['mean_pct']:+.3f}%  hit {s['hit']:.1f}%"
          f"  t {s['t']:+.2f}  {up}-{s['n']-up}  signp {sign_test(up, s['n']):.4f}")

# is the QQQ hit rate a Monday-of-month-position artefact
print("\nAugust Mondays by position in the month, QQQ")
v = fwd_ret(px["QQQ"], 1)[aug_mon.values].dropna()
mon_dates = pd.DatetimeIndex([d for d in idx[aug_mon.values]])
wk = pd.Series([(idx[idx.get_loc(d) + 1].day - 1) // 7 + 1 for d in v.index], index=v.index)
for w in sorted(wk.unique()):
    vv = v[wk == w]
    up = int((vv > 0).sum())
    print(f"  week {w} of August: n {len(vv):3d}  {up}-{len(vv)-up}  mean {100*vv.mean():+.3f}%")
