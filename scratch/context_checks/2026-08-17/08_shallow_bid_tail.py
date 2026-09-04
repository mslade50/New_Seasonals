"""Pin the tail claim from scripts 05 and 06.

The shallow cell's h5 MEAN (+1.06%) is 59% carried by two anchors, 2018-01-29 and
2015-08-18, which are the run-ups to Volmageddon and the August 2015 crash. Medians
say both cells revert. So the honest claim is about the tail, not the centre, and a
tail claim has to be counted rather than asserted.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, sign_test  # noqa: E402

px = close_panel(["SPY", "^GSPC", "^VIX"]).dropna(how="all")
spx_r = px["^GSPC"] / px["^GSPC"].shift(1) - 1.0
vix = px["^VIX"].dropna()
vix_r = vix / vix.shift(1) - 1.0
common = spx_r.dropna().index.intersection(vix_r.dropna().index)

shallow = common[((vix_r.reindex(common) >= 0.05) & (spx_r.reindex(common) > -0.01)
                  & (spx_r.reindex(common) < 0)).values]
deep = common[((vix_r.reindex(common) >= 0.05) & (spx_r.reindex(common) <= -0.01)).values]
alld = vix.index

# max VIX level over the next k sessions, relative to the anchor close
def max_run_up(idx, k):
    out = []
    pos = vix.index.get_indexer(pd.DatetimeIndex(idx))
    for p in pos:
        if p < 0 or p + k >= len(vix):
            continue
        out.append(vix.iloc[p + 1:p + 1 + k].max() / vix.iloc[p] - 1.0)
    return np.array(out)


def spy_drawdown(idx, k):
    s = px["SPY"].dropna()
    out = []
    pos = s.index.get_indexer(pd.DatetimeIndex(idx))
    for p in pos:
        if p < 0 or p + k >= len(s):
            continue
        out.append(s.iloc[p + 1:p + 1 + k].min() / s.iloc[p] - 1.0)
    return np.array(out)


print("=== P(^VIX trades at least X% above the anchor close within 10 sessions) ===")
print(f"{'cell':28s} {'n':>5s} {'+25%':>8s} {'+50%':>8s} {'+75%':>8s} {'+100%':>8s} {'median max':>11s}")
for name, idx in [("shallow (SPX 0 to -1%)", shallow), ("deep (SPX <= -1%)", deep),
                  ("all days", alld)]:
    m = max_run_up(idx, 10)
    row = " ".join(f"{100*(m >= th).mean():7.1f}%" for th in (0.25, 0.50, 0.75, 1.00))
    print(f"{name:28s} {len(m):5d} {row} {100*np.median(m):10.1f}%")

print("\n=== same, 21 sessions ===")
print(f"{'cell':28s} {'n':>5s} {'+25%':>8s} {'+50%':>8s} {'+75%':>8s} {'+100%':>8s} {'median max':>11s}")
for name, idx in [("shallow (SPX 0 to -1%)", shallow), ("deep (SPX <= -1%)", deep),
                  ("all days", alld)]:
    m = max_run_up(idx, 21)
    row = " ".join(f"{100*(m >= th).mean():7.1f}%" for th in (0.25, 0.50, 0.75, 1.00))
    print(f"{name:28s} {len(m):5d} {row} {100*np.median(m):10.1f}%")

print("\n=== worst SPY drawdown within 21 sessions ===")
print(f"{'cell':28s} {'n':>5s} {'<= -3%':>8s} {'<= -5%':>8s} {'<= -10%':>8s} {'median':>9s}")
for name, idx in [("shallow (SPX 0 to -1%)", shallow), ("deep (SPX <= -1%)", deep),
                  ("all days", alld)]:
    d = spy_drawdown(idx, 21)
    row = " ".join(f"{100*(d <= -th).mean():7.1f}%" for th in (0.03, 0.05, 0.10))
    print(f"{name:28s} {len(d):5d} {row} {100*np.median(d):8.1f}%")

print("\n=== the two episodes that carry the shallow mean ===")
for d in [pd.Timestamp("2018-01-29"), pd.Timestamp("2015-08-18")]:
    p = vix.index.get_loc(d)
    nxt = vix.iloc[p + 1:p + 11]
    print(f"  {d.date()}  SPX {100*spx_r[d]:+.2f}%  VIX {100*vix_r[d]:+.1f}% "
          f"to {vix.iloc[p]:.2f}  ->  max {nxt.max():.2f} "
          f"({100*(nxt.max()/vix.iloc[p]-1):+.0f}%) within 10 sessions")

print("\n=== shallow cell, VIX level at the anchor: does a low VIX matter? ===")
rank = vix.rolling(252).apply(lambda w: 100.0 * (w <= w[-1]).mean(), raw=True)
print(f"  today's VIX 252d percentile = {rank.iloc[-1]:.0f}, level {vix.iloc[-1]:.2f}")
for lo, hi, lbl in [(0, 25, "anchor VIX pctile <= 25"), (25, 60, "25-60"), (60, 101, "> 60")]:
    sub = shallow[((rank.reindex(shallow) > lo) & (rank.reindex(shallow) <= hi)).values]
    m = max_run_up(sub, 10)
    if len(m) == 0:
        continue
    print(f"  {lbl:24s} n={len(m):4d}  P(+50% in 10d) = {100*(m >= 0.5).mean():.1f}%  "
          f"median max {100*np.median(m):+.1f}%")
