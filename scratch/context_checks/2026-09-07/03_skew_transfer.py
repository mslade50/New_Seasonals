"""Drill 03 — ^SKEW at the 98th percentile. Does the crash-hedge bid say anything
about SPY, or only about ^SKEW?

`P5b:rank21_extreme` gave ^SKEW's own forward return: n=361, -1.25% h1, 32.7%
hit, t=-7.48, era-stable, BH-pass. That is ^SKEW mean-reverting, which is
mechanical and useless to Scott. The question worth answering is the transfer:
when the tail-hedge bid is this rich, what does the index do.

^SKEW measures the price of OTM puts relative to ATM. A 98th-percentile 21d
run-up means someone paid up for crash protection over the last month. Two
readings, and they are opposite: informed hedging ahead of trouble, or hedged
positioning that removes the forced-seller channel. This drill settles it on
the tape rather than on the story.

Anchor: the session the state printed. h1 lag=0 close-to-close.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, fwd_ret, pct_rank, summarize, era_split, cluster_note,
    sign_test, declusters, local_control, show,
)

px = close_panel(["^SKEW", "^GSPC", "SPY", "^VIX"])
nyse = px["^GSPC"].dropna().index
px = px.reindex(nyse)
skew = px["^SKEW"].dropna()
print(f"^SKEW {skew.index[0].date()} .. {skew.index[-1].date()}  n={len(skew)}")
print(f"latest ^SKEW close {skew.iloc[-1]:.2f}")

# pct_rank differences internally: pass the LEVEL, not the return.
rk = pct_rank(skew, 21, 252)
print(f"latest 21d rank {rk.iloc[-1]:.1f}")

trig_all = rk.index[rk >= 95.0]
trig = declusters(trig_all, 10, skew.index)
print(f"\n21d rank >= 95: {len(trig_all)} raw sessions -> {len(trig)} episodes "
      f"(10td min gap)")
print(f"  most recent: {[str(d.date()) for d in trig[-6:]]}")

sp = px["^GSPC"].dropna()
for h in (1, 5, 10, 21):
    f = fwd_ret(sp, h)
    base = f.dropna()
    ctrl = local_control(base.index, trig, 126)
    rows = [
        summarize(f.reindex(trig).dropna().values, f"SKEW 21d rank>=95 (episodes)"),
        summarize(f.reindex(trig_all).dropna().values, "all raw days in state"),
        summarize(f.reindex(ctrl).dropna().values, "local +/-126td control"),
        summarize(base.values, "all days"),
    ]
    show(rows, f"=== ^GSPC h{h} ===")

f1 = fwd_ret(sp, 1)
v = f1.reindex(trig).dropna()
w, n = int((v.values > 0).sum()), len(v)
print(f"\nh1 record {w}-{n - w} up, sign p(up) {sign_test(w, n):.4f}")
for e in era_split(v.index, v.values):
    print(f"  era {e['label']}: n={e['n']} mean {e['mean_pct']:+.3f}% hit {e['hit']:.1f}")
print(f"  conc: {cluster_note(v.index, v.values)}")

# the level matters as much as the change: rank>=95 at a HIGH absolute SKEW
print("\n=== conditioning on the absolute level too ===")
lvl_rank = (skew.rolling(252).rank(pct=True) * 100.0)   # percentile of the LEVEL
print(f"latest absolute ^SKEW percentile (252d): {lvl_rank.iloc[-1]:.1f}")
both = rk.index[(rk >= 95.0) & (lvl_rank >= 80.0)]
both_ep = declusters(both, 10, skew.index)
f5 = fwd_ret(sp, 5)
show([summarize(f1.reindex(both_ep).dropna().values, "rank>=95 AND level>=80th, h1"),
      summarize(f5.reindex(both_ep).dropna().values, "same, h5")],
     f"^GSPC (n_episodes={len(both_ep)})")

# and the drawdown question Scott actually cares about: does a rich tail bid
# precede an unusual number of big down days?
print("\n=== next 21 sessions: worst single day, and P(any day < -2%) ===")
ret = sp.pct_change()
rows = []
for label, dates in (("SKEW rank>=95", trig), ("all days", sp.index[::7])):
    worst, anybig = [], []
    posmap = pd.Series(range(len(sp)), index=sp.index)
    for d in dates:
        p = posmap.get(d)
        if p is None or p + 22 > len(sp):
            continue
        win = ret.iloc[p + 1:p + 22].values
        if len(win) == 21:
            worst.append(win.min())
            anybig.append(float(win.min() < -0.02))
    if worst:
        rows.append({"label": label, "n": len(worst),
                     "mean_worst_day_pct": 100 * float(np.mean(worst)),
                     "P_any_day_lt_-2pct": 100 * float(np.mean(anybig))})
show(rows, "^GSPC forward 21td tail")
