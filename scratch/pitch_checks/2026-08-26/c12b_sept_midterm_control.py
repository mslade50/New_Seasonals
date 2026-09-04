"""C12 confirmation: the LIVE cell is a SEPTEMBER print in a MIDTERM year.

The round-1 grid's month-matched control was vacuous because NFP is monthly,
so entries span all 12 months and "month-matched" == "all days".  The
2026-08-21 self-correction says a month-matched control must match ONE month.
Here the September-print subset is scored against a control restricted to the
same calendar months the entry window actually occupies.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

VEH = ["SPY", "TLT", "UUP", "GLD"]
LEAD, H = 7, 7
px = load_prices(VEH)
ser = {t: px[t]["Close"].dropna() for t in VEH}
nfp = load_events(["nfp"])["date"]
nfp = nfp[(nfp >= "2000-01-01") & (nfp <= "2026-08-25")].reset_index(drop=True)


def cell(t, k=0, h=H):
    s = ser[t]; idx = s.index; v = s.values
    ent, ret, pr = [], [], []
    for d in nfp:
        p = int(idx.searchsorted(d))
        if p >= len(idx):
            continue
        e, x = p + k - LEAD, p + k - LEAD + h
        if e < 0 or x >= len(idx):
            continue
        ent.append(idx[e]); ret.append(v[x] / v[e] - 1.0); pr.append(d)
    return pd.DatetimeIndex(ent), np.asarray(ret, float), pd.DatetimeIndex(pr)


for t in VEH:
    d, r, pr = cell(t)
    s = ser[t]
    fwd = (s.shift(-H) / s - 1.0).dropna()
    sep = np.array([x.month == 9 for x in pr])
    mid = np.array([x.year % 4 == 2 for x in pr])
    # entry window months actually occupied by the September-print subset
    emon = sorted(set(pd.DatetimeIndex(d[sep]).month))
    ctrl_sep = fwd[fwd.index.month.isin(emon)].values
    print(f"\n### {t}  (September-print entries land in months {emon})")
    rows = [summarize(r[sep], f"SEPT print (N={int(sep.sum())})"),
            summarize(ctrl_sep, f"CTRL same-month all days (N={len(ctrl_sep)})"),
            summarize(r[sep & mid], f"SEPT x MIDTERM (N={int((sep & mid).sum())})"),
            summarize(r[sep & ~mid], "SEPT x non-midterm"),
            summarize(r[~sep], "all other prints")]
    show(rows)
    a, b = r[sep], ctrl_sep
    if len(a) > 1:
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        print(f"  SEPT excess vs same-month control = {100*(a.mean()-b.mean()):+.3f}pp "
              f"welch t {(a.mean()-b.mean())/se:+.2f}   "
              f"= {100*(a.mean()-b.mean())*100/6:.1f}x a 6 bp round trip")
    sm = r[sep & mid]
    if len(sm):
        w = int((sm > 0).sum())
        print(f"  SEPT x MIDTERM record {w}-{len(sm)-w}, sign p vs the vehicle's own "
              f"up-rate {100*(r>0).mean():.1f}%: "
              f"{sign_test(w, len(sm), float((r>0).mean())):.4f}")
        print(f"  SEPT x MIDTERM years: "
              f"{[(str(x.date()), round(100*y,2)) for x, y in zip(pr[sep & mid], sm)]}")

# does the anticipation window even sit before the print, or is it drift?
print("\n### decomposition: the LAST session (the print itself) vs the run-up")
for t in VEH:
    s = ser[t]; idx = s.index; v = s.values
    pre, prn = [], []
    for d in nfp:
        p = int(idx.searchsorted(d))
        e = p - LEAD
        if e < 0 or p >= len(idx):
            continue
        pre.append(v[p - 1] / v[e] - 1.0)   # entry close -> day before print
        prn.append(v[p] / v[p - 1] - 1.0)   # the print session itself
    show([summarize(np.asarray(pre), f"{t} run-up (entry -> print eve, 6 td)"),
          summarize(np.asarray(prn), f"{t} the print session alone")])
