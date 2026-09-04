"""Bitcoin closed down five sessions running while the US index sits at a 52-week high.

Engine base cell: `P7b:down_streak|BTC-USD` n=66, h1 +0.31%, 41-25 up, sign p 0.032, and the
tag hint is suggestive on a 66-observation cell that is mostly overlapping days inside the same
streaks. Two things the base cell does not do: decluster the streaks, and condition on the
equity tape, which is the part that makes tonight unusual.

Note the calendar mismatch: BTC trades every day, SPY does not. Everything below is computed on
BTC's own sessions with SPY reindexed forward-filled, and the streak is counted on BTC closes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices, fwd_ret, summarize, sign_test, era_split, cluster_note, declusters  # noqa

px = load_prices(["BTC-USD", "SPY"])
b = px["BTC-USD"]["Close"].dropna()
s = px["SPY"]["Close"].dropna()
print(f"BTC {b.index[0].date()} .. {b.index[-1].date()}  n {len(b)}   last {b.iloc[-1]:,.0f}")

r = b.pct_change()
down = (r < 0).astype(int)
streak = down * 0
run = 0
vals = []
for x in down.values:
    run = run + 1 if x else 0
    vals.append(run)
streak = pd.Series(vals, index=b.index)
print(f"tonight's streak: {int(streak.iloc[-1])} down closes; "
      f"{100*(b.iloc[-1]/b.iloc[-6]-1):+.2f}% over the five")

trig_all = streak.index[streak >= 5]
trig = declusters(trig_all, 5, b.index)
print(f"streak >= 5: {len(trig_all)} raw days, {len(trig)} declustered episodes")

# where SPY sits, forward filled onto BTC's calendar
spy_on_b = s.reindex(b.index).ffill()
hi252 = spy_on_b.rolling(252).max()
near_high = (spy_on_b / hi252 - 1) >= -0.005
dist = 100 * (spy_on_b.iloc[-1] / hi252.iloc[-1] - 1)
print(f"SPY sits {dist:+.2f}% from its 252d high on the same calendar\n")

def block(name, dates):
    dates = pd.DatetimeIndex([d for d in dates if d in b.index])
    print(f"{name}   n {len(dates)}")
    for h in (1, 5, 10, 21):
        v = fwd_ret(b, h).reindex(dates).dropna()
        if len(v) < 3:
            continue
        st = summarize(v.values, f"h{h}")
        up = int((v > 0).sum())
        print(f"  h{h:<3d} n {st['n']:3d}  {up}-{st['n']-up}  mean {st['mean_pct']:+.2f}%  "
              f"med {st['median_pct']:+.2f}%  t {st['t']:+.2f}  signp {sign_test(up, st['n']):.4f}")
    v = fwd_ret(b, 5).reindex(dates).dropna()
    if len(v) >= 5:
        print("   h5 era:", [f"{e['label']} n {e['n']} mean {e['mean_pct']:+.2f}% up {e['hit']:.1f}%"
                             for e in era_split(v.index, v.values)])
        print("   h5 cluster:", cluster_note(v.index, v.values, k=2))

block("all declustered 5+ down streaks", trig)
block("... with SPY within 0.5% of a 52-week high", [d for d in trig if bool(near_high.get(d, False))])
block("... with SPY not near its high", [d for d in trig if not bool(near_high.get(d, False))])

base = fwd_ret(b, 5).dropna()
print(f"\ncontrol, every BTC session: n {len(base)} mean {100*base.mean():+.2f}% "
      f"med {100*base.median():+.2f}% up {100*(base>0).mean():.1f}%")
base1 = fwd_ret(b, 1).dropna()
print(f"control h1: n {len(base1)} mean {100*base1.mean():+.2f}% up {100*(base1>0).mean():.1f}%")

# how common is a 5-day streak at all, and how deep is this drawdown
dd = 100 * (b.iloc[-1] / b.rolling(365).max().iloc[-1] - 1)
print(f"\nBTC is {dd:.1f}% below its 365d high")
prior = [d for d in trig if 100 * (b[d] / b.rolling(365).max()[d] - 1) <= -30]
print(f"episodes that were also 30%+ below the high: {len(prior)}")
if prior:
    for h in (1, 5, 21):
        v = fwd_ret(b, h).reindex(pd.DatetimeIndex(prior)).dropna()
        up = int((v > 0).sum())
        print(f"  h{h:<3d} n {len(v):2d}  {up}-{len(v)-up}  mean {100*v.mean():+.2f}%  "
              f"med {100*v.median():+.2f}%  signp {sign_test(up, len(v)):.4f}")
    print("  dates:", ", ".join(str(d.date()) for d in prior))
