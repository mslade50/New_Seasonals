"""Stat verify: is QQQ actually on a 5+ down-close streak tonight, and what
do 5+ streaks do next? Clean own-math version of the brief's depth-split cell
so the post's numbers are unambiguous (the brief stored one cell of a split).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    declusters, era_split, fwd_lag, fwd_ret, load_prices, sign_test, summarize,
)

qqq = load_prices(["QQQ"])["QQQ"]["Close"]
r = qqq.pct_change()
down = (r < 0).astype(int)
streak = down * (down.groupby((down != down.shift()).cumsum()).cumcount() + 1)
print("QQQ", qqq.index[-1].date(), "close", round(qqq.iloc[-1], 2),
      "| current down streak:", int(streak.iloc[-1]))
print("last 6 closes:", [round(x, 2) for x in qqq.iloc[-6:]])

# day a streak REACHES 5 (fresh trigger), declustered 5 td
hit5 = streak == 5
idx = qqq.index[hit5.fillna(False)]
idx = idx[idx < qqq.index[-1]]
trig = declusters(idx, 5, qqq.index)
print(f"\nstreak reaches 5: raw {len(idx)}, declustered {len(trig)}")

for h in (1, 3, 5, 10):
    f = fwd_lag(qqq, h).reindex(trig).dropna()   # buy next close
    f0 = fwd_ret(qqq, h).reindex(trig).dropna()  # lag-0 contrast
    s = summarize(f.values)
    nup = int((f > 0).sum())
    allc = summarize(fwd_lag(qqq, h).dropna().values)
    print(f"  lag1 h{h:<2} n={s['n']:<3} mean={s['mean_pct']:+.3f}%  {nup}-{len(f)-nup} up  "
          f"t={s['t']:+.2f}  sign_p={sign_test(nup, len(f)):.4f}  "
          f"| all {allc['mean_pct']:+.3f}%  | lag0 {summarize(f0.values)['mean_pct']:+.3f}%")

f5 = fwd_lag(qqq, 5).reindex(trig).dropna()
print("era h5 lag1:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3),
                        round(e.get("hit", np.nan), 1)) for e in era_split(f5.index, f5.values)])

# depth on the trigger day (cumulative loss over the 5 closes)
depth = (qqq / qqq.shift(5) - 1).reindex(trig)
shallow = f5.reindex(depth[depth > -0.04].index).dropna()
deep = f5.reindex(depth[depth <= -0.04].index).dropna()
for lab, fs in (("shallow >-4%", shallow), ("deep <=-4%", deep)):
    if len(fs):
        s = summarize(fs.values)
        nup = int((fs > 0).sum())
        print(f"  {lab}: n={s['n']:<3} mean={s['mean_pct']:+.3f}%  {nup}-{len(fs)-nup} up")
print("\ntonight's 5d cumulative:", round((qqq.iloc[-1] / qqq.iloc[-6] - 1) * 100, 2), "%")
