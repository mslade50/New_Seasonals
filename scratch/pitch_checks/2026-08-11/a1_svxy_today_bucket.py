"""C1 last look: the one sub-bucket that could argue for a rescue.

H1 found that in the -0.5x era the TODAY-LIKE bucket (calm VIX level + SVXY
within 2% of its 252d high) is the best post-break cell: N=19, +1.205%, hit
68.4%, sign p 0.084.  Before that counts as anything it needs the same test
that killed the parent - the SPY-beta residual - plus a count of how many
DISTINCT episodes and years those 19 events are.

Run: python scratch/pitch_checks/2026-08-11/a1_svxy_today_bucket.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_lab import SVXY_LEV_BREAK, anchor_dates, tdom_of  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, declusters, fwd_lag, load_events, sign_test, summarize,
)

warnings.filterwarnings("ignore")
px = close_panel(["SVXY", "SPY", "^VIX"])
all_dates = px.index
ev = load_events(["cpi"])
svxy, spy, vix = px["SVXY"].dropna(), px["SPY"].dropna(), px["^VIX"].dropna()
anch = declusters(anchor_dates(ev, "cpi", 2, all_dates), 5, all_dates)
anch = anch[anch.isin(svxy.index)]

vix_rank63 = vix.rolling(63).rank(pct=True) * 100
dist_high = svxy / svxy.rolling(252).max() - 1.0
today_like = (vix_rank63 <= 50) & (dist_high >= -0.02)

for h in (1, 3, 5):
    fs, fp = fwd_lag(svxy, h, lag=1), fwd_lag(spy, h, lag=1)
    j = pd.concat([fs.rename("y"), fp.rename("x")], axis=1).dropna()
    j = j[j.index >= SVXY_LEV_BREAK]
    X = np.column_stack([np.ones(len(j)), j["x"].values])
    b, *_ = np.linalg.lstsq(X, j["y"].values, rcond=None)
    r = pd.Series(j["y"].values - X @ b, index=j.index)
    sel = anch[anch >= SVXY_LEV_BREAK]
    sel = sel[sel.isin(today_like[today_like].index)]
    sel = sel[sel.isin(r.index)]
    raw, res = j["y"].reindex(sel).dropna(), r.reindex(sel).dropna()
    st = summarize(raw.values)
    print(f"h={h:<2} N={len(res):<3} years={sorted(set(sel.year))}")
    print(f"     RAW {st['mean_pct']:+.3f}% hit {st['hit']:.1f}% "
          f"signp {sign_test(int((raw.values>0).sum()), len(raw)):.4f}  ->  "
          f"beta={b[1]:+.2f} RESIDUAL {100*res.mean():+.3f}% "
          f"t={res.mean()/(res.std(ddof=1)/np.sqrt(len(res))):+.2f} "
          f"hit {100*(res>0).mean():.1f}% signp "
          f"{sign_test(int((res>0).sum()), len(res)):.4f}")
    print(f"     SPY on those days {100*j['x'].reindex(sel).dropna().mean():+.3f}% "
          f"vs all days {100*j['x'].mean():+.3f}%")
