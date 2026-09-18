"""A2 - the exact number that would turn the rescued rung on.

Two numbers for the watchlist:
  (1) the 95th percentile of the 72-cell null-max distribution, i.e. the
      date-averaged h=5 excess the cell must EXCEED to clear its own
      multiplicity charge;
  (2) how many new fresh-trigger dates at what mean would take the
      leave-KRE-out cell from +0.436% to that bar.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

FAMILY = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLE", "XLU", "XLB",
          "SMH", "IBB", "XBI", "IHI", "KRE", "ITA", "XME", "XRT", "XHB",
          "IYR", "OIH"]
px = load_prices(FAMILY)
panel = pd.DataFrame({t: px[t]["Close"] for t in FAMILY})
R5 = pd.DataFrame({t: pct_rank(panel[t], 5) for t in FAMILY})
R63 = pd.DataFrame({t: pct_rank(panel[t], 63) for t in FAMILY})
GRID = [(a, b) for a in (2, 5, 10, 15) for b in (75, 85, 90)]
HS = (1, 2, 3, 5, 7, 10)
FW = {h: {t: fwd_lag(panel[t].dropna(), h, 1) for t in FAMILY} for h in HS}
POOL = {h: {t: FW[h][t].dropna().values for t in FAMILY} for h in HS}
DRIFT = {h: np.concatenate([POOL[h][t] for t in FAMILY]).mean() for h in HS}

cells = []
for a, b in GRID:
    for h in HS:
        cn = {}
        for t in FAMILY:
            m = ((R5[t] <= a) & (R63[t] >= b)).fillna(False)
            d = R5.index[m.values]
            f = FW[h][t]
            cn[t] = int(sum(1 for x in d
                            if x in f.index and not np.isnan(f.loc[x])))
        if sum(cn.values()) >= 5:
            cells.append((h, cn))

rng = np.random.default_rng(7)
nulls = []
for _ in range(4000):
    best = -np.inf
    for h, cn in cells:
        tot, num = 0.0, 0
        for t in FAMILY:
            n = cn[t]
            if n:
                arr = POOL[h][t]
                tot += arr[rng.integers(0, len(arr), size=n)].sum(); num += n
        if num:
            best = max(best, tot / num - DRIFT[h])
    nulls.append(best)
nulls = np.asarray(nulls)
q95 = float(np.quantile(nulls, 0.95))
print(f"K={len(cells)} cells")
print(f"null-max median  = {100*np.median(nulls):+.3f}%")
print(f"null-max 95th pct = {100*q95:+.3f}%   <= the BAR the excess must clear")
print(f"defended (all members) excess = +1.577%  -> P = "
      f"{float((nulls >= 0.01577).mean()):.4f}")
print(f"ex-KRE excess = +0.436%  -> P = {float((nulls >= 0.00436).mean()):.4f}")

# how many new dates at what mean take ex-KRE from +0.436% to the bar?
cur_n, cur_exc = 25, 0.00436
print("\nex-KRE cell has 25 dates at excess +0.436%. New dates needed:")
for k in (5, 10, 15, 20):
    need = (q95 * (cur_n + k) - cur_exc * cur_n) / k
    print(f"  {k:>2} new fresh-trigger dates must average an excess of "
          f"{100*need:+.3f}% (mean ~{100*(need + DRIFT[5]):+.2f}% at h=5)")
