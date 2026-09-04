"""Nasdaq weakness underneath a broad market at its highs.

Live state 2026-08-24: QQQ 5d -3.23% (rank 7.1 of its own year), SPY 5d -1.19%,
^NYA 5d +0.04% and 0.38% from a 52-week high. Cell: QQQ's 5-day return in the
bottom decile of its trailing year on a session ^NYA closes within 0.5% of a
252-day high. What follows for QQQ, for SPY, and for the QQQ-minus-NYA spread.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (load_prices, summarize, sign_test, fwd_ret, declusters,
                       local_control, era_split, cluster_note, pct_rank)  # noqa

ASOF = pd.Timestamp("2026-08-24")
px = load_prices(["QQQ", "SPY", "^NYA", "^GSPC"])
pan = pd.concat({k: px[k]["Close"] for k in px}, axis=1).dropna()
pan = pan.loc[:ASOF]
print("panel", pan.index[0].date(), "->", pan.index[-1].date(), len(pan))

q, s, n = pan["QQQ"], pan["SPY"], pan["^NYA"]
q5 = q.pct_change(5)
rank_q5 = pct_rank(q5, 1, 252) if False else q5.rolling(252).apply(
    lambda w: (w[:-1] < w[-1]).mean() * 100, raw=True)
nya_hi = n.rolling(252).max()
near_hi = n >= nya_hi * 0.995

mask = (rank_q5 <= 10) & near_hi
dts = pan.index[mask.fillna(False)]
dc = declusters(dts, 10, pan.index)
print(f"raw n={len(dts)}  declustered@10td n={len(dc)}")
print("dates:", [str(x.date()) for x in dc])
print("live state today:", f"QQQ 5d rank {rank_q5.iloc[-1]:.1f}",
      f"NYA dist from 252d high {100*(n.iloc[-1]/nya_hi.iloc[-1]-1):.2f}%")

ctrl = local_control(pan.index, dc, 126)
for name, ser in (("QQQ", q), ("SPY", s), ("^NYA", n)):
    print(f"\n-- forward {name}")
    for h in (1, 5, 10, 21):
        f = fwd_ret(ser, h)
        v = f.reindex(dc).dropna()
        if len(v) < 3:
            continue
        st = summarize(v.values, "")
        up = int((v.values > 0).sum())
        cs = summarize(f.reindex(ctrl).dropna().values, "")
        a = summarize(f.dropna().values, "")
        print(f"   h{h:<3} n={st['n']:<3} mean {st['mean_pct']:>7.2f}%  med {st['median_pct']:>7.2f}%  "
              f"{up}-{len(v)-up} up  sign p {sign_test(up, len(v)):.4f}  t {st['t']:>5.2f} | "
              f"local ctrl {cs['mean_pct']:>6.2f}% | all days {a['mean_pct']:>6.2f}%")
        if h in (5, 21):
            print("      era:", [(e['label'], e['n'], round(e['mean_pct'], 2)) for e in era_split(v.index, v.values)])
            print("      ", cluster_note(v.index, v.values))

print("\n-- forward QQQ minus ^NYA spread (does the gap close?)")
for h in (1, 5, 10, 21):
    sp = fwd_ret(q, h) - fwd_ret(n, h)
    v = sp.reindex(dc).dropna()
    st = summarize(v.values, "")
    up = int((v.values > 0).sum())
    cs = summarize(sp.reindex(ctrl).dropna().values, "")
    a = summarize(sp.dropna().values, "")
    print(f"   h{h:<3} n={st['n']:<3} mean {st['mean_pct']:>7.2f}pp  {up}-{len(v)-up} up  "
          f"sign p {sign_test(up, len(v)):.4f}  t {st['t']:>5.2f} | local ctrl {cs['mean_pct']:>6.2f}pp "
          f"| all days {a['mean_pct']:>6.2f}pp")
