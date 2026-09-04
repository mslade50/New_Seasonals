"""Decisive control for section 5: is SPY's 52w-high pre-CPI sub-cell about CPI
at all, or just a momentum state that pays over ANY 3-4 session MOO->MOC hold?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from f1_pre_cpi import load_events, build_assets, dedupe_monthly

ev = load_events(); cpi, _ = dedupe_monthly(ev["cpi"])
A = build_assets(C.load(["SPY"])); a = A["SPY"]
close = pd.Series(a.cl, index=a.idx); n = len(a.cl)
r5 = C.pct_rank(close.pct_change(5), 252).to_numpy()
hi52 = close.rolling(252).max().to_numpy()
state = (a.cl >= hi52 * 0.999) & (r5 >= 80)          # state at a CLOSE
cpos = set(a.pos_of(cpi).tolist())
cps = np.zeros(n, bool); cps[list(cpos)] = True
cum = np.concatenate([[0], np.cumsum(cps)])
epos = np.array(sorted(p - 4 for p in cpos if p - 4 >= 0))

print("SPY 52w-high + 5d rank>=80 state at the PRE-ENTRY close, MOO next bar -> MOC h later")
print("treatment = that state 4 sessions before a CPI print (today's geometry)")
print("control   = SAME state, any other session, window contains NO CPI print\n")
for h, lab in ((3, "eve (k=4)"), (4, "print (k=4)")):
    i = np.arange(1, n - h - 1)
    ok = state[i - 1] & np.isfinite(a.op[i]) & np.isfinite(a.cl[i + h])
    r = (a.cl[i + h] / a.op[i] - 1.0) * 100.0
    is_cpi_entry = np.isin(i, epos)
    has_cpi = (cum[i + h + 1] - cum[i]) > 0
    rows = [C.describe(f"[{lab}] state 4td pre-CPI (treat)", r[ok & is_cpi_entry]),
            C.describe(f"[{lab}] state, NO CPI in window (control)", r[ok & ~has_cpi]),
            C.describe(f"[{lab}] state, ANY session", r[ok]),
            C.describe(f"[{lab}] all bars uncond", r)]
    C.show(rows)
    t_, c_ = r[ok & is_cpi_entry], r[ok & ~has_cpi]
    vx, vy = t_.var(ddof=1)/len(t_), c_.var(ddof=1)/len(c_)
    print(f"  treat - control = {t_.mean()-c_.mean():+.4f}%   Welch t = "
          f"{(t_.mean()-c_.mean())/np.sqrt(vx+vy):+.2f}   (N {len(t_)} vs {len(c_)})\n")
