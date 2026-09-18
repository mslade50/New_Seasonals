"""kR round 3 on c10: is the SEP x dollar-bid-into-eve slice a threshold
artifact? Threshold sweep and continuous regression within SEP meetings,
horizon path h1..h10 for the live slice, and the announcement-session return
by SEP / r5 (mechanism a inside the slice). DX-Y.NYB, tdom-matched excess.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pxd = load_prices(["DX-Y.NYB"])
FOMC = load_events(["fomc_decision"])["date"]
FOMC = pd.DatetimeIndex(FOMC[FOMC <= pd.Timestamp("2026-09-15")])
SEP_EARLY = pd.DatetimeIndex(["2011-04-27", "2011-06-22", "2011-11-02", "2012-01-25",
                              "2012-04-25", "2012-06-20", "2012-09-13", "2012-12-12"])


def is_sep(d):
    return (d in SEP_EARLY) or (d.year >= 2013 and d.month in (3, 6, 9, 12))


s = pxd["DX-Y.NYB"]["Close"].dropna()
idx = s.index
ym = pd.Series(idx.year * 100 + idx.month, index=idx)
TD = ym.groupby(ym.values).cumcount().values + 1
POS = pd.Series(np.arange(len(s)), index=idx)
dd = pd.DatetimeIndex([d for d in FOMC if d in POS.index])
dp = np.array([POS[d] for d in dd])
EX = np.zeros(len(s), bool)
for p in dp:
    EX[max(0, p - 5):p + 6] = True
r5 = pct_rank(s, 5)
eve_r5 = np.array([r5.iloc[p - 1] for p in dp])
sep = np.array([is_sep(d) for d in dd])


def xcell(h, lag_start=0):
    r = (s.shift(-h) / s - 1.0).values
    ok = ~np.isnan(r)
    b = {j: np.nanmean(r[(TD == j) & ~EX & ok]) for j in np.unique(TD)}
    pp = dp + lag_start
    return np.array([r[p] - b[TD[p]] if ok[p] else np.nan for p in pp])


def rec(x):
    x = x[~np.isnan(x)]
    w = int((x > 0).sum())
    return f"{100 * x.mean():+.3f}% {w}-{len(x) - w} (n {len(x)})"


print(f"today eve r5 {r5.iloc[-1]:.1f}")
g = ~np.isnan(eve_r5)
print("\n1. threshold sweep within SEP meetings (DX tdomX)")
for h in (3, 5):
    x = xcell(h)
    line = [f"h={h}"]
    for thr in (50, 60, 70, 75, 80, 85, 90):
        m = sep & g & (eve_r5 >= thr)
        line.append(f">={thr}: {rec(x[m])}")
    print("  " + " | ".join(line))
    line = [f"h={h} nonSEP"]
    for thr in (50, 60, 70, 75, 80, 85, 90):
        m = ~sep & g & (eve_r5 >= thr)
        line.append(f">={thr}: {rec(x[m])}")
    print("  " + " | ".join(line))
    for lbl, m in (("SEP", sep & g), ("nonSEP", ~sep & g), ("ALL", g)):
        xx, rr = x[m], eve_r5[m]
        k = ~np.isnan(xx)
        X = np.column_stack([np.ones(k.sum()), rr[k]])
        beta, res, *_ = np.linalg.lstsq(X, xx[k], rcond=None)
        resid = xx[k] - X @ beta
        s2 = resid @ resid / (k.sum() - 2)
        cov = s2 * np.linalg.inv(X.T @ X)
        x0 = np.array([1.0, 84.9])
        pred, se = x0 @ beta, np.sqrt(x0 @ cov @ x0)
        print(f"    {lbl} h={h}: slope {100 * beta[1] * 10:+.3f}pp per 10 rank pts (t {beta[1] / np.sqrt(cov[1, 1]):+.2f}); "
              f"predicted at r5=84.9 {100 * pred:+.3f}% se {100 * se:.3f}")

print("\n2. horizon path, live slice SEP & r5>=75 vs SEP & r5<75 (DX tdomX)")
for h in range(1, 11):
    x = xcell(h)
    print(f"  h={h:2d}  SEP&hi {rec(x[sep & g & (eve_r5 >= 75)])}   SEP&lo {rec(x[sep & g & (eve_r5 < 75)])}   "
          f"nonSEP&hi {rec(x[~sep & g & (eve_r5 >= 75)])}")

print("\n3. announcement session (eve close -> D close), DX raw and tdomX")
x1 = xcell(1, lag_start=-1)
for lbl, m in (("SEP", sep), ("nonSEP", ~sep), ("SEP & r5>=75", sep & g & (eve_r5 >= 75)),
               ("SEP & r5<75", sep & g & (eve_r5 < 75)), ("nonSEP & r5>=75", ~sep & g & (eve_r5 >= 75))):
    print(f"  {lbl:18s} {rec(x1[m])}")
