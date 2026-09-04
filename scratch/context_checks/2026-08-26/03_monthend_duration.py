"""The month-end duration bid, and whether it shows up in August.

Drill 02 established the effect is not Jackson Hole. This one asks the
question that actually matters for tomorrow: the final-3-session window is
one of the strongest calendar cells in the whole sweep, but tomorrow is an
AUGUST final-3 session, and August looked flat-to-negative there.

Two things done properly here that 02 did not:
  - declustering. 960 "sessions" are only ~320 month-end episodes of 3
    overlapping days. Episode-level counting is the honest N.
  - no Jackson Hole exclusion. 02 needed it to clean the comparison; here it
    would gut the August cell, which is the cell under test.

Convention: lag=0 close-to-close from the anchor close, h=1 is the next
session.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from pitch_lab import close_panel, summarize, sign_test

SUBJECTS = ["TLT", "IEF", "^TNX", "HYG", "SPY", "^GSPC"]
px = close_panel(SUBJECTS).sort_index()
all_dates = px.index

pos_of = pd.Series(np.arange(len(all_dates)), index=all_dates)
ym = all_dates.to_period("M")
last_pos = pd.Series(pos_of.values, index=all_dates).groupby(ym).transform("max")
dist_me = pd.Series(last_pos.values - pos_of.values, index=all_dates)

final3 = all_dates[(dist_me <= 2).values]
print(f"final-3 sessions: {len(final3)}   month-end episodes: "
      f"{len(pd.Series(final3).dt.to_period('M').unique())}")


def fwd(sym, dates, h=1):
    s = px[sym].dropna()
    p = s.index.searchsorted(dates)
    ok = (p >= 0) & (p + h < len(s)) & (p < len(s))
    p = p[ok]
    used = s.index[p]
    # only keep dates that actually exist in this series
    keep = used.isin(dates)
    p, used = p[keep], used[keep]
    return used, (s.values[p + h] / s.values[p]) - 1.0


def block(sym, dates, label, h=1):
    d, v = fwd(sym, dates, h)
    if len(v) < 4:
        print(f"  {label:34s} n={len(v):4d}  (too few)")
        return None
    s = summarize(v, label)
    up = int((v > 0).sum())
    print(f"  {label:34s} n={len(v):4d}  mean={s['mean_pct']:+7.3f}%  "
          f"{up}-{len(v)-up} up  hit={s['hit']:5.1f}%  t={s['t']:+6.2f}  "
          f"signp={sign_test(up, len(v)):.4f}")
    return {"d": d, "v": v, "up": up, "n": len(v), "mean": s["mean_pct"], "t": s["t"]}


def episode_level(sym, dates, h=1):
    """One observation per month-end episode: the mean of its 3 sessions."""
    d, v = fwd(sym, dates, h)
    if len(v) == 0:
        return None
    g = pd.Series(v, index=d).groupby(pd.Series(d).dt.to_period("M").values).mean()
    up = int((g > 0).sum())
    return {"n": len(g), "mean": g.mean() * 100, "up": up,
            "signp": sign_test(up, len(g)), "series": g}


print("\n" + "=" * 84)
print("A. The full month-end cell, then split by month")
print("=" * 84)
aug3 = final3[final3.month == 8]
non_aug3 = final3[final3.month != 8]
print(f"August final-3 sessions: {len(aug3)}   other months: {len(non_aug3)}")

for sym in SUBJECTS:
    print(f"\n--- {sym} ---")
    block(sym, final3, "final-3, all months")
    block(sym, non_aug3, "final-3, EXCLUDING August")
    block(sym, aug3, "final-3, AUGUST only")
    ep_all = episode_level(sym, final3)
    ep_aug = episode_level(sym, aug3)
    ep_non = episode_level(sym, non_aug3)
    if ep_all:
        print(f"  [episode-level] all months  n={ep_all['n']:3d} episodes  "
              f"mean={ep_all['mean']:+6.3f}%  {ep_all['up']}-{ep_all['n']-ep_all['up']} "
              f"signp={ep_all['signp']:.4f}")
    if ep_non:
        print(f"  [episode-level] ex-August   n={ep_non['n']:3d} episodes  "
              f"mean={ep_non['mean']:+6.3f}%  {ep_non['up']}-{ep_non['n']-ep_non['up']} "
              f"signp={ep_non['signp']:.4f}")
    if ep_aug:
        print(f"  [episode-level] AUGUST      n={ep_aug['n']:3d} episodes  "
              f"mean={ep_aug['mean']:+6.3f}%  {ep_aug['up']}-{ep_aug['n']-ep_aug['up']} "
              f"signp={ep_aug['signp']:.4f}")


print("\n" + "=" * 84)
print("B. TLT and IEF: every month, episode level, to see if August is special")
print("=" * 84)
for sym in ["TLT", "IEF"]:
    print(f"\n--- {sym} (episode-level h1 mean %, one obs per month-end) ---")
    rows = []
    for m in range(1, 13):
        sel = final3[final3.month == m]
        ep = episode_level(sym, sel)
        if ep:
            rows.append((m, ep["n"], ep["mean"], ep["up"], ep["signp"]))
    for m, n, mean, up, sp in rows:
        star = "   <<<" if m == 8 else ""
        print(f"  month {m:2d}  n={n:3d}  mean={mean:+6.3f}%  {up}-{n-up} up  "
              f"signp={sp:.4f}{star}")
    means = [r[2] for r in rows]
    aug_rank = sorted(means, reverse=True).index(dict((r[0], r[2]) for r in rows)[8]) + 1
    print(f"  August ranks {aug_rank} of 12 months by mean")


print("\n" + "=" * 84)
print("C. Tomorrow's exact slot: the FIRST of the final three")
print("=" * 84)
slot = {k: all_dates[(dist_me == k).values] for k in (0, 1, 2)}
for sym in ["TLT", "IEF", "SPY"]:
    print(f"\n--- {sym} ---")
    for k in (2, 1, 0):
        block(sym, slot[k], f"dist_to_month_end = {k}")
    print("  August only:")
    for k in (2, 1, 0):
        block(sym, slot[k][slot[k].month == 8], f"  August, dist = {k}")


print("\n" + "=" * 84)
print("D. Era and cycle splits on the headline cell")
print("=" * 84)
for sym in ["TLT", "IEF"]:
    print(f"\n--- {sym} ---")
    d, v = fwd(sym, final3, 1)
    pre = d < pd.Timestamp("2018-01-01")
    for lbl, mask in [("pre-2018", pre), ("2018+", ~pre)]:
        vv = v[mask]
        up = int((vv > 0).sum())
        s = summarize(vv, "")
        print(f"  all months {lbl:9s} n={len(vv):4d}  mean={s['mean_pct']:+7.3f}%  "
              f"{up}-{len(vv)-up}  t={s['t']:+6.2f}")
    da, va = fwd(sym, aug3, 1)
    prea = da < pd.Timestamp("2018-01-01")
    for lbl, mask in [("pre-2018", prea), ("2018+", ~prea)]:
        vv = va[mask]
        if len(vv) < 4:
            continue
        up = int((vv > 0).sum())
        s = summarize(vv, "")
        print(f"  AUGUST     {lbl:9s} n={len(vv):4d}  mean={s['mean_pct']:+7.3f}%  "
              f"{up}-{len(vv)-up}  t={s['t']:+6.2f}")
    mid = da.year % 4 == 2
    vv = va[mid]
    if len(vv) >= 4:
        up = int((vv > 0).sum())
        s = summarize(vv, "")
        print(f"  AUGUST     midterm   n={len(vv):4d}  mean={s['mean_pct']:+7.3f}%  "
              f"{up}-{len(vv)-up}  t={s['t']:+6.2f}")
