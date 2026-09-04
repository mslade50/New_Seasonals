"""GDX flush re-fires: the arm tonight actually is.

01_iwm_monthend_gdx_flush.py found the first-flush cell (21d >= +25% and a
-3% day, declustered at 10 sessions) 8-1 over five sessions. But Friday
2026-08-28 is NOT a first flush: 08-18 fired the same cell 8 sessions earlier
(and 08-26 printed -2.9%). The decluster rule keeps the first event of each
cluster, so the 8-1 record never traded a day like Friday.

This script splits every raw match into first-of-cluster vs re-fire (a match
within 10 sessions of a prior match) and grades the re-fires on their own,
lag-1, h = 1/3/5/10. Also the run-level view: what did the FIRST flush's
five-session path look like in the episodes where a second flush followed?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import fwd_lag, load_prices, sign_test, summarize  # noqa: E402

px = load_prices(["GDX", "GLD"])
g = px["GDX"]["Close"].dropna()
ret1 = g.pct_change()
ret21 = g.pct_change(21)
pos = pd.Series(np.arange(len(g)), index=g.index)

for thr in (0.20, 0.25):
    m = (ret21 >= thr) & (ret1 <= -0.03)
    raw = g.index[m.fillna(False).values]
    first, refire = [], []
    last_p = -10**9
    for d in raw:
        p = pos[d]
        if p - last_p >= 10:
            first.append(d)
        else:
            refire.append(d)
        last_p = p
    print(f"\n=== 21d >= {thr:.0%} & day <= -3%: raw {len(raw)}, first {len(first)}, refire {len(refire)} ===")
    for lab, idx in (("first", first), ("refire", refire), ("all raw", list(raw))):
        for h in (1, 3, 5, 10):
            r = fwd_lag(g, h, 1).reindex(pd.DatetimeIndex(idx)).dropna()
            if len(r) == 0:
                continue
            st = summarize(r.values)
            nup = int((r > 0).sum())
            print(f"  {lab:<8} h{h:<3} n={st['n']:<3} mean={st['mean_pct']:+.2f}% med={st['median_pct']:+.2f}% "
                  f"{nup}-{len(r)-nup} sp={sign_test(nup, len(r)):.3f} t={st['t']:+.2f} "
                  f"worst {st['worst_pct']:+.1f}% best {st['best_pct']:+.1f}%")
    r5 = fwd_lag(g, 5, 1).reindex(pd.DatetimeIndex(refire)).dropna()
    print("  refire dates h5:", [(d.date().isoformat(), round(100 * x, 1)) for d, x in r5.items()])
    r5f = fwd_lag(g, 5, 1).reindex(pd.DatetimeIndex(first)).dropna()
    print("  first dates h5: ", [(d.date().isoformat(), round(100 * x, 1)) for d, x in r5f.items()])

# how many sessions from the first flush to the re-fire, and where did the
# run end up 21 sessions after the re-fire (is a second flush the top?)
print("\n=== re-fire -> 21 sessions later, 21d >= 20% version ===")
m = (ret21 >= 0.20) & (ret1 <= -0.03)
raw = g.index[m.fillna(False).values]
last_p = -10**9
for d in raw:
    p = pos[d]
    if p - last_p < 10:
        r21 = fwd_lag(g, 21, 1).get(d, np.nan)
        r5 = fwd_lag(g, 5, 1).get(d, np.nan)
        print(f"  {d.date()}  gap {p-last_p:>2} sess  day {100*ret1[d]:+.1f}%  21d {100*ret21[d]:+.1f}%  "
              f"h5 {100*r5:+.1f}%  h21 {100*r21:+.1f}%")
    last_p = p
print(f"  tonight: 08-18 -> 08-28 gap {pos[g.index[-1]] - pos[pd.Timestamp('2026-08-18')]} sess, "
      f"day {100*ret1.iloc[-1]:+.1f}%, 21d {100*ret21.iloc[-1]:+.1f}%")
