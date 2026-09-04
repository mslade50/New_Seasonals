"""Honest FULL-search null: 24 grid cells x each asset's today's-state slice."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from f1_pre_cpi import load_events, build_assets, dedupe_monthly, cell, ASSETS

RNG = np.random.default_rng(777); NSIM = 2500
ev = load_events(); cpi, _ = dedupe_monthly(ev["cpi"])
A = build_assets(C.load(ASSETS))
ST = {}
for t, a in A.items():
    cl = pd.Series(a.cl, index=a.idx)
    r5 = C.pct_rank(cl.pct_change(5), 252).to_numpy()
    r21 = C.pct_rank(cl.pct_change(21), 252).to_numpy()
    r63 = C.pct_rank(cl.pct_change(63), 252).to_numpy()
    sma = cl.rolling(200).mean().to_numpy(); hi = cl.rolling(252).max().to_numpy()
    if t == "SPY":        s = (a.cl >= hi * 0.999) & (r5 >= 80)
    elif t == "DX-Y.NYB": s = (r5 <= 25) & (r63 >= 70)
    elif t == "TLT":      s = (r21 <= 33) & (a.cl < sma)
    else:                 s = (a.cl < sma) & (a.cl <= hi * 0.85)
    ST[t] = np.nan_to_num(s, nan=False).astype(bool)

def search(pdates):
    best, where = 0.0, None
    for t in ASSETS:
        a = A[t]; pp = a.pos_of(pdates)
        for k in (3, 4, 5):
            for xk in ("eve", "print"):
                e, x, r = cell(a, pp, k, xk)
                m = (e >= 1) & ST[t][e - 1]
                if m.sum() >= 8:
                    tv = abs(C.tstat(r[m]))
                    if np.isfinite(tv) and tv > best: best, where = tv, (t, k, xk, int(m.sum()))
    return best, where

obs, w = search(cpi)
print(f"observed max |t| over 24 cells x today's-state = {obs:.2f} at {w}")
cal = A["SPY"].idx; real = A["SPY"].pos_of(cpi); n = len(cal)
nlB = np.empty(NSIM); sh = RNG.integers(6, 250, NSIM) * RNG.choice([-1, 1], NSIM)
for s in range(NSIM):
    nlB[s] = search(pd.DatetimeIndex(cal[np.sort((real + sh[s]) % n)]))[0]
nlA = np.empty(NSIM)
for s in range(NSIM):
    nlA[s] = search(pd.DatetimeIndex(cal[np.sort(RNG.choice(np.arange(6, n-2), len(real), False))]))[0]
for nm, nl in (("circular-shift", nlB), ("random-position", nlA)):
    print(f"{nm:16s} null: median {np.median(nl):.2f}  p90 {np.percentile(nl,90):.2f}  "
          f"p95 {np.percentile(nl,95):.2f}  p99 {np.percentile(nl,99):.2f}   "
          f"P(null >= {obs:.2f}) = {(nl >= obs).mean():.4f}")
