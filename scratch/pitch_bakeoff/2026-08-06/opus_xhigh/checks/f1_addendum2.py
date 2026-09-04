"""Multiplicity null for the N=12 SPY sub-cell: how often does the SAME search
(6 SPY cells x today's-state slice) produce |t| >= 4.54 under a null calendar?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from f1_pre_cpi import load_events, build_assets, dedupe_monthly, cell

RNG = np.random.default_rng(4242); NSIM = 4000
ev = load_events(); cpi, _ = dedupe_monthly(ev["cpi"])
A = build_assets(C.load(["SPY"])); a = A["SPY"]
close = pd.Series(a.cl, index=a.idx); n = len(a.cl)
r5 = C.pct_rank(close.pct_change(5), 252).to_numpy()
hi52 = close.rolling(252).max().to_numpy()
state = (a.cl >= hi52 * 0.999) & (r5 >= 80)
real = a.pos_of(cpi)

def search_max_t(pp):
    best = 0.0
    for k in (3, 4, 5):
        for xk in ("eve", "print"):
            e, x, r = cell(a, pp, k, xk)
            m = (e >= 1) & state[e - 1]
            if m.sum() >= 8:
                tv = abs(C.tstat(r[m]))
                if np.isfinite(tv) and tv > best: best = tv
    return best

obs = search_max_t(real)
print(f"observed max |t| over the 6-cell x today's-state search: {obs:.2f}")
nl = np.empty(NSIM)
sh = RNG.integers(6, 250, NSIM) * RNG.choice([-1, 1], NSIM)
for s in range(NSIM):
    nl[s] = search_max_t(np.sort((real + sh[s]) % n))
print(f"circular-shift null: median {np.median(nl):.2f}  p90 {np.percentile(nl,90):.2f}  "
      f"p95 {np.percentile(nl,95):.2f}  p99 {np.percentile(nl,99):.2f}")
print(f"P(null >= {obs:.2f}) = {(nl >= obs).mean():.4f}")
nl2 = np.empty(NSIM)
for s in range(NSIM):
    nl2[s] = search_max_t(np.sort(RNG.choice(np.arange(6, n - 2), len(real), replace=False)))
print(f"random-position null: median {np.median(nl2):.2f}  p95 {np.percentile(nl2,95):.2f}  "
      f"p99 {np.percentile(nl2,99):.2f}   P(null >= {obs:.2f}) = {(nl2 >= obs).mean():.4f}")
