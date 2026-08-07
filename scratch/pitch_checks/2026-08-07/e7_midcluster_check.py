"""E7: today is a MID-CLUSTER fire for every candidate, not a fresh trigger.

E1 has fired 08-04/05/06, E2 08-03..06, E3 08-03/05/06. The episode-level stats measure
FIRST fires. This script measures the cell today actually belongs to: the 3rd+ consecutive
fire. If the mid-cluster cell is where the (short) edge lives, E1 could be resurrected;
if it is just a smaller, noisier slice, everything stays dead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa: F401,F403

px = load_prices(["SPY", "^VIX", "AAPL"])
spy = px["SPY"]["Close"]
idx = spy.index
vix = px["^VIX"]["Close"].reindex(idx).ffill()
aapl = px["AAPL"]["Close"].reindex(idx).ffill()
rk5 = pct_rank(spy, 5)
rk5v = pct_rank(vix, 5)
dist = (spy / spy.rolling(252).max() - 1) * 100
pos = pd.Series(range(len(idx)), index=idx)


def fwd_entry_next(s, h):
    return s.shift(-(h + 1)) / s.shift(-1) - 1.0


def run_depth(name, cond, s, hs):
    """Depth = how many consecutive prior sessions the condition was also true."""
    c = cond.reindex(idx).fillna(False).astype(bool)
    depth = c.groupby((~c).cumsum()).cumcount()  # 0 on the first fire of a run
    for H in hs:
        f = fwd_entry_next(s, H)
        valid = cond.notna() & f.notna() & rk5.notna() & dist.notna()
        rows = []
        for lab, m in [("depth 0 (fresh)", depth == 0), ("depth 1", depth == 1),
                       ("depth 2+ (= TODAY)", depth >= 2),
                       ("depth 3+", depth >= 3), ("all fires", c)]:
            sel = idx[c & m & valid]
            rows.append(summarize(f[sel].values, f"{lab}"))
        rows.append(summarize(f[valid].values, "-- control --"))
        show(rows, f"{name} h={H} by cluster depth (day-level, no declustering)")
        sel = idx[c & (depth >= 2) & valid]
        v = f[sel].dropna().values
        if len(v) >= 3:
            print(f"  depth2+ bootstrap P(mean<=0) LONG : {bootstrap_p_le0(v):.4f}   "
                  f"SHORT: {bootstrap_p_le0(-v):.4f}")
            sub = pd.DatetimeIndex(sel)
            for cut in ("2013-01-01", "2018-01-01"):
                a = era_split(sub, v, cut)
                print(f"  depth2+ era {cut[:4]}: "
                      f"{a[0]['label']} n={a[0].get('n',0)} mean={a[0].get('mean_pct',float('nan')):+.3f}% | "
                      f"{a[1]['label']} n={a[1].get('n',0)} mean={a[1].get('mean_pct',float('nan')):+.3f}%")


print("########## E1 ##########")
run_depth("E1", (rk5 >= 95) & (dist >= -0.5) & (rk5v <= 25), spy, (1, 5))

print("\n########## E3 ##########")
run_depth("E3", (vix < 16) & (rk5v <= 25) & (dist >= -1.0), spy, (5, 10))

print("\n########## E2 (AAPL) ##########")
run_depth("E2", (pct_rank(aapl, 5) <= 5) & (dist >= -1.0), aapl, (5, 10))

print("\n########## today's depth for each candidate ##########")
for nm, cond in [("E1", (rk5 >= 95) & (dist >= -0.5) & (rk5v <= 25)),
                 ("E2", (pct_rank(aapl, 5) <= 5) & (dist >= -1.0)),
                 ("E3", (vix < 16) & (rk5v <= 25) & (dist >= -1.0)),
                 ("E4", (rk5 >= 90) & (dist >= -0.5))]:
    c = cond.reindex(idx).fillna(False).astype(bool)
    d = c.groupby((~c).cumsum()).cumcount()
    print(f"  {nm}: fires today={bool(c.iloc[-1])}  cluster depth today={int(d.iloc[-1])} "
          f"(0 = fresh trigger)")
