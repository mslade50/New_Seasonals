"""C5 round 2d -- the last thing that can be wrong with the curve leg.

Round 2b/2c priced the duration-neutral position with a FULL-SAMPLE OLS beta
(TLT ~ IEF = 1.914). That is lookahead: the hedge ratio was fitted on the same
25 years the cell is measured over, and a hedge ratio is a free parameter like
any threshold. If the edge only exists at the fitted ratio it is a fit.

Tested here: an EXPANDING point-in-time beta (min 252 sessions, lag-1), a
252-session ROLLING beta, the raw daily-sd ratio (2.101), and two round-number
ratios a human would actually type (2.0 and 1.5). Whole variants only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)

px = close_panel(["^TNX", "TLT", "IEF"]).dropna(how="any")
idx = px.index
tnx = px["^TNX"]
LEVEL = (tnx / rolling_on_valid(tnx, lambda x: x.rolling(252).max()) - 1.0) >= -0.0025
rt = px["TLT"].pct_change()
ri = px["IEF"].pct_change()

# expanding + rolling PIT betas, shifted 1 so the ratio is known before entry
cov_e = ri.expanding(252).cov(rt)
var_e = ri.expanding(252).var()
beta_exp = (cov_e / var_e).shift(1)
cov_r = ri.rolling(252).cov(rt)
var_r = ri.rolling(252).var()
beta_roll = (cov_r / var_r).shift(1)
print("beta full-sample %.3f | PIT expanding today %.3f | rolling-252 today %.3f"
      % (float(np.polyfit(ri.dropna().values, rt.dropna().align(ri.dropna())[0].values, 1)[0])
         if False else 1.914, beta_exp.iloc[-1], beta_roll.iloc[-1]))
print("PIT expanding beta range over trigger days: %.2f .. %.2f"
      % (beta_exp[LEVEL].min(), beta_exp[LEVEL].max()))


def pos_ret(h, w_tlt):
    """long IEF 1.0, short w_tlt * TLT, where w_tlt may be a SERIES (PIT)."""
    a = fwd_lag(px["IEF"], h, 1)
    b = fwd_lag(px["TLT"], h, 1)
    return a - (w_tlt * b if isinstance(w_tlt, pd.Series) else w_tlt * b)


rows = []
for lab, w in [("full-sample beta 1.914 -> 0.523", 1.0 / 1.914),
               ("PIT EXPANDING beta (lag-1)", 1.0 / beta_exp),
               ("ROLLING-252 beta (lag-1)", 1.0 / beta_roll),
               ("daily-sd ratio 2.101 -> 0.476", 1.0 / 2.101),
               ("round 2.0 -> 0.500", 0.5),
               ("round 1.5 -> 0.667", 1.0 / 1.5)]:
    for h in (2, 8, 10):
        r = pos_ret(h, w)
        sig = idx[LEVEL.values & r.notna().values]
        e = declusters(sig, max(h, 10), idx)
        v = r.loc[e].values
        v = v[~np.isnan(v)]
        bps = 100 * 100 * v.mean()
        rows.append({"hedge": lab, "h": h, "n": len(v),
                     "mean_pct": round(100 * v.mean(), 3), "bps": round(bps, 1),
                     "x_cost": round(bps / 6.0, 2),
                     "hit": round(100 * (v > 0).mean(), 1),
                     "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2)})
print("\n=== hedge-ratio sensitivity (cost bar = 5x = 30 bps) ===")
print(pd.DataFrame(rows).to_string(index=False))
print("\n  If the PIT rows sit near the full-sample rows the ratio is not the fit;")
print("  the binding constraint stays COST. Bar 30 bps.")
