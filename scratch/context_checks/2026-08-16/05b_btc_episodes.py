"""The 11 episodes behind 05's h21 result, one line each, plus the drawdown definition check.

Two things to settle: whether the 1-10 record is cycle-top clustering with a handful of
independent regimes behind it, and which drawdown number is honest. The engine's tape block
computes a 252-ROW high, which on a 7-day-a-week series is eight months, not 52 weeks.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices, fwd_ret, sign_test, cluster_note, declusters  # noqa

px = load_prices(["BTC-USD", "SPY"])
b = px["BTC-USD"]["Close"].dropna()
s = px["SPY"]["Close"].dropna()

r = b.pct_change()
run, vals = 0, []
for x in (r < 0).astype(int).values:
    run = run + 1 if x else 0
    vals.append(run)
streak = pd.Series(vals, index=b.index)

trig = declusters(streak.index[streak >= 5], 5, b.index)
spy_on_b = s.reindex(b.index).ffill()
near = (spy_on_b / spy_on_b.rolling(252).max() - 1) >= -0.005
dates = pd.DatetimeIndex([d for d in trig if bool(near.get(d, False))])

h21 = fwd_ret(b, 21)
h5 = fwd_ret(b, 5)
dd252 = 100 * (b / b.rolling(252).max() - 1)
dd365 = 100 * (b / b.rolling(365).max() - 1)

print(f"{'date':12s} {'BTC':>10s} {'dd252':>7s} {'dd365':>7s} {'h5':>8s} {'h21':>8s}")
for d in dates:
    print(f"{str(d.date()):12s} {b[d]:10,.0f} {dd252[d]:7.1f} {dd365[d]:7.1f} "
          f"{100*h5.get(d, np.nan):8.2f} {100*h21.get(d, np.nan):8.2f}")

v = h21.reindex(dates).dropna()
up = int((v > 0).sum())
print(f"\nh21: n {len(v)}  {up}-{len(v)-up}  mean {100*v.mean():+.2f}%  med {100*v.median():+.2f}%"
      f"  down-side sign p {sign_test(len(v)-up, len(v)):.4f}")
print("cluster:", cluster_note(v.index, v.values, k=2))
print("distinct years:", sorted(set(v.index.year)))
print(f"drop the worst episode: n {len(v)-1} mean {100*v.drop(v.idxmin()).mean():+.2f}%  "
      f"{up}-{len(v)-1-up}")

base = h21.dropna()
print(f"\ncontrol, every BTC session h21: n {len(base)} mean {100*base.mean():+.2f}% "
      f"med {100*base.median():+.2f}% up {100*(base>0).mean():.1f}%")

# the same conditioning without the streak: SPY at a high, BTC not in a streak
allnear = b.index[near.fillna(False).values]
nostreak = allnear.difference(pd.DatetimeIndex(streak.index[streak >= 5]))
v2 = h21.reindex(nostreak).dropna()
print(f"SPY at a high, BTC not in a 5-day streak: n {len(v2)} mean {100*v2.mean():+.2f}% "
      f"up {100*(v2>0).mean():.1f}%")

print(f"\ntonight: BTC {b.iloc[-1]:,.0f}, dd252 {dd252.iloc[-1]:.1f}%, dd365 {dd365.iloc[-1]:.1f}%, "
      f"SPY {100*(spy_on_b.iloc[-1]/spy_on_b.rolling(252).max().iloc[-1]-1):+.2f}% from its high")
