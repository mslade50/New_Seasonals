"""Drill 03b — firm up the two ^SKEW numbers the brief will actually quote.

03 showed the S&P does slightly BETTER after a 98th-percentile 21d run-up in
^SKEW, and that the forward 21-session tail is milder, not fatter. Both are
counterintuitive enough that they need their era split and a proper baseline
before publication. 03's tail baseline used a 1-in-7 sample; redo on all days.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, fwd_ret, pct_rank, summarize, era_split, cluster_note,
    sign_test, declusters, show,
)

px = close_panel(["^SKEW", "^GSPC"])
nyse = px["^GSPC"].dropna().index
px = px.reindex(nyse)
skew, sp = px["^SKEW"].dropna(), px["^GSPC"].dropna()
rk = pct_rank(skew, 21, 252)
trig = declusters(rk.index[rk >= 95.0], 10, skew.index)
print(f"episodes: {len(trig)}   latest 21d rank {rk.iloc[-1]:.1f}")

print("\n=== h5 era split (the follow-on number) ===")
v5 = fwd_ret(sp, 5).reindex(trig).dropna()
print(f"  h5 n={len(v5)} mean {100 * v5.mean():+.3f}% "
      f"hit {100 * (v5.values > 0).mean():.1f} median {100 * v5.median():+.3f}%")
w = int((v5.values > 0).sum())
print(f"  record {w}-{len(v5) - w} up, sign p(up) {sign_test(w, len(v5)):.4f}")
for e in era_split(v5.index, v5.values):
    print(f"    era {e['label']}: n={e['n']} mean {e['mean_pct']:+.3f}% hit {e['hit']:.1f}")
print(f"    conc: {cluster_note(v5.index, v5.values)}")

print("\n=== forward 21td tail, baseline on ALL days ===")
ret = sp.pct_change()
posmap = pd.Series(range(len(sp)), index=sp.index)


def tail_stats(dates, label):
    worst, big = [], []
    for d in dates:
        p = posmap.get(d)
        if p is None or p + 22 > len(sp):
            continue
        win = ret.iloc[p + 1:p + 22].values
        if len(win) == 21 and not np.isnan(win).any():
            worst.append(win.min())
            big.append(float(win.min() < -0.02))
    return {"label": label, "n": len(worst),
            "mean_worst_day_pct": 100 * float(np.mean(worst)),
            "median_worst_day_pct": 100 * float(np.median(worst)),
            "P_any_lt_-2pct": 100 * float(np.mean(big))}


rows = [tail_stats(trig, "SKEW 21d rank>=95 (episodes)"),
        tail_stats(sp.index, "all days (full overlap)")]
show(rows, "^GSPC forward 21 sessions")

# era-split the tail claim too
for cut_lo, cut_hi, lab in ((None, "2018-01-01", "pre-2018"),
                            ("2018-01-01", None, "2018+")):
    d = trig
    if cut_lo:
        d = d[d >= cut_lo]
    if cut_hi:
        d = d[d < cut_hi]
    print(f"  {lab}: {tail_stats(d, lab)}")
