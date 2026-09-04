"""Precious-metals magnitude, two different states.

Gold: 21-day return +15.79%, the 98.4th percentile of its year. How rare in level
terms, and what follows.
Silver: 21-day return +17.58% while still 40% BELOW its 52-week high. A thrust
inside a drawdown is a different animal from a thrust at a high; split on it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (load_prices, summarize, sign_test, fwd_ret, declusters,
                       local_control, era_split, cluster_note)  # noqa

ASOF = pd.Timestamp("2026-08-24")
px = load_prices(["GC=F", "SI=F", "SPY"])


def block(ser, mask, label, horizons=(1, 5, 10, 21), gap=21):
    idx = ser.index
    dts = idx[mask.reindex(idx).fillna(False)]
    dts = pd.DatetimeIndex([d for d in dts if d <= ASOF])
    dc = declusters(dts, gap, idx)
    print(f"-- {label}: raw n={len(dts)}, declustered@{gap}td n={len(dc)}")
    print("   dates:", [str(x.date()) for x in dc])
    ctrl = local_control(idx, dc, 126)
    for h in horizons:
        f = fwd_ret(ser, h)
        v = f.reindex(dc).dropna()
        if len(v) < 3:
            print(f"   h{h}: n={len(v)} too few")
            continue
        st = summarize(v.values, "")
        up = int((v.values > 0).sum())
        cs = summarize(f.reindex(ctrl).dropna().values, "")
        a = summarize(f.dropna().values, "")
        print(f"   h{h:<3} n={st['n']:<3} mean {st['mean_pct']:>7.2f}%  med {st['median_pct']:>7.2f}%  "
              f"{up}-{len(v)-up} up  sign p {sign_test(up, len(v)):.4f}  t {st['t']:>5.2f} | "
              f"local ctrl {cs['mean_pct']:>6.2f}% | all {a['mean_pct']:>6.2f}%  "
              f"| worst {st['worst_pct']:.1f}% best {st['best_pct']:.1f}%")
        if h == 21:
            print("      era:", [(e['label'], e['n'], round(e.get('mean_pct', float('nan')), 2)) for e in era_split(v.index, v.values)])
            print("      ", cluster_note(v.index, v.values))


g = px["GC=F"]["Close"]
r21g = g.pct_change(21)
print(f"gold 21d today: {100*r21g.iloc[-1]:.2f}%   "
      f"pctile of full history: {100*(r21g.dropna() < r21g.iloc[-1]).mean():.2f}")
print(f"sessions with 21d >= 15%: {int((r21g >= 0.15).sum())} of {r21g.notna().sum()}")
block(g, r21g >= 0.15, "gold 21d return of 15% or more")

print()
s = px["SI=F"]["Close"]
r21s = s.pct_change(21)
dd = s / s.rolling(252).max() - 1
print(f"silver 21d today: {100*r21s.iloc[-1]:.2f}%, drawdown from 252d high {100*dd.iloc[-1]:.2f}%")
m_all = r21s >= 0.15
print(f"silver sessions with 21d >= 15%: {int(m_all.sum())}")
block(s, m_all & (dd <= -0.30), "silver 21d >= 15% while 30%+ below its 252d high")
print()
block(s, m_all & (dd > -0.30), "silver 21d >= 15% while less than 30% below its 252d high (control)")
