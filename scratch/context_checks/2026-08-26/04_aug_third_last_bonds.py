"""The cell tomorrow actually occupies: third-to-last session of August.

Drill 03 found the August month-end duration bid is not spread across the
final three sessions, it sits almost entirely on the FIRST of them, which is
tomorrow. TLT 19-6 at +0.448%, IEF 18-7 at +0.216%, equities flat.

This drill tries to kill it: year by year, era split on the slot itself
(03 only split the pooled August window), concentration, the follow-on path,
and whether the anchor is robust to shifting one session either way.

Convention: lag=0 close-to-close from the anchor close, h=1 is the next
session.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from pitch_lab import close_panel, summarize, sign_test, cluster_note

SUBJECTS = ["TLT", "IEF", "^TNX", "SPY", "HYG"]
px = close_panel(SUBJECTS).sort_index()
all_dates = px.index

pos_of = pd.Series(np.arange(len(all_dates)), index=all_dates)
ym = all_dates.to_period("M")
last_pos = pd.Series(pos_of.values, index=all_dates).groupby(ym).transform("max")
dist_me = pd.Series(last_pos.values - pos_of.values, index=all_dates)


CUR_PERIOD = all_dates.max().to_period("M")


def slot_dates(month, k):
    """Sessions k from month end. The CURRENT month is excluded: its last
    session has not happened yet, so dist_me measures distance to the last
    bar in the cache, not to the real month end. Leaving it in put
    2026-08-24 in the cell as if it were August's third-to-last session."""
    sel = all_dates[((dist_me == k).values) & (all_dates.month == month)]
    return sel[sel.to_period("M") != CUR_PERIOD]


def fwd(sym, dates, h=1):
    s = px[sym].dropna()
    p = s.index.searchsorted(dates)
    ok = (p + h < len(s)) & (p < len(s))
    p = p[ok]
    used = s.index[p]
    keep = used.isin(dates)
    p, used = p[keep], used[keep]
    return used, (s.values[p + h] / s.values[p]) - 1.0


AUG2 = slot_dates(8, 2)
print(f"anchor cell: third-to-last session of August, {len(AUG2)} years "
      f"({AUG2.min().date()} .. {AUG2.max().date()})")
print(f"tomorrow 2026-08-27 is this slot: "
      f"{pd.Timestamp('2026-08-27') in set(all_dates)} (in cache), "
      f"dist_to_month_end would be 2")

print("\n" + "=" * 76)
print("A. Year by year, h1")
print("=" * 76)
d, v = fwd("TLT", AUG2, 1)
d_ief, v_ief = fwd("IEF", AUG2, 1)
d_spy, v_spy = fwd("SPY", AUG2, 1)
ief_map = dict(zip(d_ief, v_ief))
spy_map = dict(zip(d_spy, v_spy))
print(f"{'anchor':12s} {'TLT h1':>9s} {'IEF h1':>9s} {'SPY h1':>9s}")
for dt, val in zip(d, v):
    i = ief_map.get(dt)
    s = spy_map.get(dt)
    print(f"{str(dt.date()):12s} {val*100:+8.2f}% "
          f"{(i*100 if i is not None else float('nan')):+8.2f}% "
          f"{(s*100 if s is not None else float('nan')):+8.2f}%")

print("\n" + "=" * 76)
print("B. Kill attempts on TLT / IEF")
print("=" * 76)
for sym, dd, vv in [("TLT", d, v), ("IEF", d_ief, v_ief)]:
    print(f"\n--- {sym} ---")
    s = summarize(vv, "")
    up = int((vv > 0).sum())
    print(f"  full cell        n={len(vv):3d}  mean={s['mean_pct']:+6.3f}%  "
          f"{up}-{len(vv)-up} up  t={s['t']:+5.2f}  signp={sign_test(up, len(vv)):.4f}")
    print(f"  median           {s['median_pct']:+6.3f}%   worst {s['worst_pct']:+6.2f}%  "
          f"best {s['best_pct']:+6.2f}%")
    print(f"  concentration    {cluster_note(dd, vv, k=2)}")
    # drop best 2
    order = np.argsort(vv)[::-1]
    trimmed = np.delete(vv, order[:2])
    st = summarize(trimmed, "")
    upt = int((trimmed > 0).sum())
    print(f"  drop best 2      n={len(trimmed):3d}  mean={st['mean_pct']:+6.3f}%  "
          f"{upt}-{len(trimmed)-upt} up  t={st['t']:+5.2f}")
    # era
    pre = dd < pd.Timestamp("2018-01-01")
    for lbl, mask in [("pre-2018", pre), ("2018+", ~pre)]:
        w = vv[mask]
        if len(w) < 3:
            continue
        sw = summarize(w, "")
        upw = int((w > 0).sum())
        print(f"  {lbl:9s}        n={len(w):3d}  mean={sw['mean_pct']:+6.3f}%  "
              f"{upw}-{len(w)-upw} up  t={sw['t']:+5.2f}  "
              f"signp={sign_test(upw, len(w)):.4f}")
    # cycle
    mid = (dd.year % 4 == 2)
    w = vv[mid]
    if len(w) >= 3:
        sw = summarize(w, "")
        upw = int((w > 0).sum())
        print(f"  midterm years    n={len(w):3d}  mean={sw['mean_pct']:+6.3f}%  "
              f"{upw}-{len(w)-upw} up")

print("\n" + "=" * 76)
print("C. Anchor robustness: shift the slot one session either way")
print("=" * 76)
for sym in ["TLT", "IEF", "SPY"]:
    print(f"\n--- {sym} ---")
    for k, lbl in [(3, "4th-to-last (one earlier)"),
                   (2, "3rd-to-last  <-- tomorrow"),
                   (1, "2nd-to-last (one later)")]:
        dk, vk = fwd(sym, slot_dates(8, k), 1)
        if len(vk) < 3:
            continue
        sk = summarize(vk, "")
        upk = int((vk > 0).sum())
        print(f"  {lbl:28s} n={len(vk):3d}  mean={sk['mean_pct']:+6.3f}%  "
              f"{upk}-{len(vk)-upk} up  t={sk['t']:+5.2f}  "
              f"signp={sign_test(upk, len(vk)):.4f}")

print("\n" + "=" * 76)
print("D. Follow-on: does it hold or give back? (TLT / IEF / SPY)")
print("=" * 76)
for sym in ["TLT", "IEF", "SPY"]:
    print(f"\n--- {sym} ---")
    for h in (1, 2, 3, 5, 10):
        dh, vh = fwd(sym, AUG2, h)
        sh = summarize(vh, "")
        uph = int((vh > 0).sum())
        print(f"  h={h:<3d} n={len(vh):3d}  mean={sh['mean_pct']:+6.3f}%  "
              f"{uph}-{len(vh)-uph} up  t={sh['t']:+5.2f}")

print("\n" + "=" * 76)
print("E. Control: the same slot in the other eleven months (TLT / IEF)")
print("=" * 76)
for sym in ["TLT", "IEF"]:
    others = all_dates[((dist_me == 2).values) & (all_dates.month != 8)]
    do, vo = fwd(sym, others, 1)
    so = summarize(vo, "")
    upo = int((vo > 0).sum())
    print(f"  {sym}: 3rd-to-last, ex-August  n={len(vo):3d}  "
          f"mean={so['mean_pct']:+6.3f}%  {upo}-{len(vo)-upo} up  t={so['t']:+5.2f}")
    s_all = px[sym].dropna().pct_change().shift(-1).dropna()
    print(f"       all-days control          n={len(s_all):4d}  "
          f"mean={s_all.mean()*100:+6.3f}%  hit={(s_all > 0).mean():.1%}")
