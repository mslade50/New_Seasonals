"""The 3-month bill yield closed at a 52-week high the day after an FOMC cut
decision, while the 10-year FELL 1.18%.

^IRX: 3.965, dist_52w_high -0.13%, 21d rank 99.2, z10 2.54, 21d change +7.0%.
^TNX: 4.947, -1.18% today, 21d rank 88.9, 63d rank 95.2.

The engine's P4 cell on ^IRX is unusable (its all-days control is enormous
because a yield level drifts), so compute the thing itself: how often does the
front end print a 52-week high while the long end is falling, and what has the
curve done next.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, summarize, show, sign_test, declusters, rolling_on_valid,
)

px = close_panel(["^IRX", "^FVX", "^TNX", "SPY", "IWM", "DX-Y.NYB"])
idx = px.index
irx, tnx = px["^IRX"].dropna(), px["^TNX"].dropna()
common = irx.index.intersection(tnx.index)
irx, tnx = irx.reindex(common), tnx.reindex(common)

hi252 = irx.rolling(252).max()
at_high = irx >= hi252 - 1e-9
# first such print in 30+ sessions, the engine's own novelty convention
first = at_high & (~at_high.rolling(30).max().shift(1).fillna(0).astype(bool))
tnx_dn = (tnx / tnx.shift(1) - 1.0) <= -0.01

print(f"^IRX sessions: {len(common)}  {common[0].date()} .. {common[-1].date()}")
print(f"  52w-high prints: {int(at_high.sum())}, first-in-30td: {int(first.sum())}")

spread = (tnx - irx)              # 10y minus 3m, in percentage points
print(f"  today 10y-3m spread: {spread.iloc[-1]:.3f} pp "
      f"(21td ago {spread.iloc[-22]:.3f}, 63td ago {spread.iloc[-64]:.3f})")

conds = [
    ("^IRX at a 52w high", at_high),
    ("first 52w high in 30+ td", first),
    ("first 52w high in 30+ td AND 10y down 1%+", first & tnx_dn),
]
for lab, m in conds:
    d = common[m.fillna(False).values]
    print(f"\n--- {lab}: n={len(d)} ---")
    if len(d) == 0:
        continue
    epi = declusters(d, 21, common)
    print(f"    declustered (21td): {len(epi)}; by year "
          f"{dict(pd.Series(1, index=epi).groupby(epi.year).sum())}")
    for h in (5, 21):
        fs = spread.shift(-h) - spread
        r = summarize((fs.loc[epi] / 100).values, f"10y-3m change, h={h} (pp)")
        k = int((fs.loc[epi] > 0).sum())
        n = int(fs.loc[epi].notna().sum())
        print(f"    curve h={h}: mean {fs.loc[epi].mean():+.3f} pp, "
              f"median {fs.loc[epi].median():+.3f} pp, steepened {k} of {n}, "
              f"sign p(flatten) = {sign_test(n - k, n):.4f}")
    for tkr in ["SPY", "IWM"]:
        s = px[tkr].dropna()
        for h in (5, 21):
            f = s.shift(-h) / s - 1.0
            dd = epi.intersection(f.dropna().index)
            if len(dd) < 3:
                continue
            k = int((f.loc[dd] > 0).sum())
            base = f.dropna()
            print(f"    {tkr} h={h}: n={len(dd)} mean {100 * f.loc[dd].mean():+.2f}% "
                  f"(ctl {100 * base.mean():+.2f}%) {k}-{len(dd) - k} up "
                  f"sign p(down) = {sign_test(len(dd) - k, len(dd)):.4f}")

print("\n=== how rare is a 52w-high front end with the long end 21d-rank < 95? ===")
print("    (i.e. the whole curve is not making highs together)")
rk = rolling_on_valid(tnx, lambda x: x.rolling(252).apply(
    lambda w: 100.0 * (w.iloc[-1] > w[:-1]).mean(), raw=False))
print(f"    ^TNX trailing-252 percentile today: {rk.iloc[-1]:.1f}")

print("\n=== the plain fact for the brief: 21-session change in each tenor ===")
for t in ["^IRX", "^FVX", "^TNX"]:
    s = px[t].dropna()
    print(f"  {t}: {s.iloc[-1]:.3f}  1d {100 * (s.iloc[-1] / s.iloc[-2] - 1):+.2f}%  "
          f"21d {100 * (s.iloc[-1] / s.iloc[-22] - 1):+.2f}%  "
          f"(bp: 1d {100 * (s.iloc[-1] - s.iloc[-2]):+.1f}, "
          f"21d {100 * (s.iloc[-1] - s.iloc[-22]):+.1f})")
