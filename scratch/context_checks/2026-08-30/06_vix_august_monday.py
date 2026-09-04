"""Two calendar cells on the same session, crossed with the vol state.

The engine fired, for Mondays in August (n 118):
  ^VIX  h1 +2.409%, t 2.37, era stable
  QQQ   h1 +0.130%, hit 64.4%, record 76-42, sign p 0.0011, BH pass, t 1.05

QQQ's shape is odd and worth resolving: a 64% hit rate with no mean says the
up days are small and the down days are large. And both cells are confounded
with month end, because the last Monday of August is sometimes the last
session of the month, which is exactly what Monday is.

The VIX cell also has a state to cross it with. VIX closed 53.5% below its
52-week high and VIX3M printed a 52-week LOW on Friday. A calendar cell that
says vol rises means something different from a floor than from the middle.

Convention: anchor is the session before the Monday, so h=1 is the Monday.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, summarize, show, sign_test, era_split, cluster_note,
)

px = close_panel(["^VIX", "^VIX3M", "QQQ", "SPY", "^GSPC"])
cal = px["^GSPC"].dropna().index
ym = pd.Series(cal.year * 100 + cal.month, index=cal)
COMPLETE = sorted(set(ym.values))[:-1]
final_pos = set()
for key, grp in ym.groupby(ym.values):
    if key in COMPLETE:
        final_pos.add(list(cal).index(grp.index[-1]))

# Mondays in August, anchored on the prior session
mondays = [i for i in range(1, len(cal))
           if cal[i].month == 8 and cal[i].weekday() == 0]
is_final = np.array([i in final_pos for i in mondays])
anchor = pd.DatetimeIndex([cal[i - 1] for i in mondays])
target = pd.DatetimeIndex([cal[i] for i in mondays])


def leg(sub):
    s = px[sub].dropna()
    v, keep = [], []
    for a, b in zip(anchor, target):
        if a in s.index and b in s.index:
            v.append(s.loc[b] / s.loc[a] - 1.0)
            keep.append(True)
        else:
            v.append(np.nan)
            keep.append(False)
    return np.asarray(v, float), np.asarray(keep)


print("=" * 78)
print("Q1. the two cells as the engine has them, and the month-end split")
print("=" * 78)
for sub in ["^VIX", "QQQ", "SPY"]:
    v, keep = leg(sub)
    ok = np.isfinite(v)
    rows = []
    for label, m in [("all August Mondays", ok),
                     ("...that ARE the month's last session", ok & is_final),
                     ("...that are NOT", ok & ~is_final)]:
        if m.sum() < 3:
            continue
        r = summarize(v[m], label)
        u = int((v[m] > 0).sum())
        r["record"] = f"{u}-{int(m.sum()) - u}"
        r["sign_p"] = round(sign_test(u, int(m.sum())), 4)
        rows.append(r)
    s = px[sub].dropna()
    b = summarize((s / s.shift(1) - 1.0).dropna().values, "all sessions")
    b["record"] = ""
    b["sign_p"] = np.nan
    rows.append(b)
    show(rows, f"{sub}: August Mondays")
    print("  era:", [(x["label"], x["n"], round(x["mean_pct"], 3), round(x["hit"], 1))
                     for x in era_split(anchor[ok], v[ok])])
    print("  conc:", cluster_note(anchor[ok], v[ok]))

print()
print("=" * 78)
print("Q2. QQQ's shape: a 64% hit with no mean. Size the two sides.")
print("=" * 78)
v, _ = leg("QQQ")
ok = np.isfinite(v)
x = v[ok]
print(f"  n={len(x)}  up {int((x>0).sum())} mean {100*x[x>0].mean():+.3f}%  "
      f"down {int((x<0).sum())} mean {100*x[x<0].mean():+.3f}%  "
      f"median {100*np.median(x):+.3f}%")
s = px["QQQ"].dropna()
allr = (s / s.shift(1) - 1.0).dropna().values
print(f"  baseline: up {int((allr>0).sum())} mean {100*allr[allr>0].mean():+.3f}%  "
      f"down {int((allr<0).sum())} mean {100*allr[allr<0].mean():+.3f}%  "
      f"hit {100*(allr>0).mean():.1f}%")
print("  Mondays in every OTHER month, for the control:")
mon_other = [i for i in range(1, len(cal))
             if cal[i].month != 8 and cal[i].weekday() == 0]
vo = []
sq = px["QQQ"].dropna()
for i in mon_other:
    a, b = cal[i - 1], cal[i]
    if a in sq.index and b in sq.index:
        vo.append(sq.loc[b] / sq.loc[a] - 1.0)
vo = np.asarray(vo, float)
u = int((vo > 0).sum())
print(f"    n={len(vo)} {u}-{len(vo)-u} hit {100*(vo>0).mean():.1f}% "
      f"mean {100*vo.mean():+.3f}%")

print()
print("=" * 78)
print("Q3. the VIX cell crossed with the vol floor. Split August Mondays on")
print("    where VIX sat in its own trailing-252 range at the anchor.")
print("    Friday: VIX 53.5% below its 52w high, VIX3M at a 52w low.")
print("=" * 78)
vix = px["^VIX"].dropna()
lo252 = vix.rolling(252, min_periods=200).min()
hi252 = vix.rolling(252, min_periods=200).max()
pos = ((vix - lo252) / (hi252 - lo252) * 100)
v, _ = leg("^VIX")
ok = np.isfinite(v)
pv = np.array([pos.loc[a] if a in pos.index else np.nan for a in anchor])
rows = []
for label, m in [("VIX in bottom third of its 52w range", ok & (pv <= 33)),
                 ("middle third", ok & (pv > 33) & (pv <= 66)),
                 ("top third", ok & (pv > 66))]:
    if m.sum() < 3:
        continue
    r = summarize(v[m], label)
    u = int((v[m] > 0).sum())
    r["record"] = f"{u}-{int(m.sum()) - u}"
    r["sign_p"] = round(sign_test(u, int(m.sum())), 4)
    rows.append(r)
show(rows, "^VIX on August Mondays by entry position in its 52w range")

m = ok & (pv <= 33)
if m.sum() >= 5:
    print("  era:", [(x["label"], x["n"], round(x["mean_pct"], 3), round(x["hit"], 1))
                     for x in era_split(anchor[m], v[m])])
    print("  conc:", cluster_note(anchor[m], v[m]))
    print("  years:", sorted({x.year for x in anchor[m]}))

print()
print("=" * 78)
print("today's readings")
print("=" * 78)
print(f"  ^VIX   {vix.iloc[-1]:.2f}  position in 52w range {pos.iloc[-1]:.0f}%  "
      f"(low {lo252.iloc[-1]:.2f} high {hi252.iloc[-1]:.2f})")
v3 = px["^VIX3M"].dropna()
l3 = v3.rolling(252, min_periods=200).min()
print(f"  ^VIX3M {v3.iloc[-1]:.2f}  52w low {l3.iloc[-1]:.2f}  "
      f"at the low: {bool(abs(v3.iloc[-1] - l3.iloc[-1]) < 1e-9)}")
