"""Follow-on to drill 02: the k3->k0 window is stronger than the single day.

TLT +0.262% t 2.92 over the three sessions into opex, against +0.141% t 2.45
for the first of them. That window is exactly tomorrow's open through Friday's
close from tonight's anchor, so it is the number the brief should carry. This
script gives it the controls the single-day cell already passed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, summarize, show, sign_test, era_split, cluster_note,
)

TKRS = ["TLT", "IEF", "LQD", "^TNX", "SPY"]
px = close_panel(TKRS)
idx = px.index
ev = load_events(["opex"])
P = np.array([idx.searchsorted(d) for d in pd.DatetimeIndex(ev["date"])
              if idx.searchsorted(d) < len(idx)
              and idx[idx.searchsorted(d)] == d])

H = 3  # k3 -> k0


def window(t):
    v = px[t].values
    vals, dts = [], []
    for p in P:
        if p - H < 0 or p >= len(v):
            continue
        x = v[p] / v[p - H] - 1.0
        if not np.isnan(x):
            vals.append(x)
            dts.append(idx[p - H])
    return np.array(vals), pd.DatetimeIndex(dts)


print("=" * 74)
print("A. the 3-session run into opex vs a matched-length control")
print("=" * 74)
out = []
for t in TKRS:
    v, d = window(t)
    r = summarize(v, f"{t} k3->k0 (n={len(v)})")
    r["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
    r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    # every non-overlapping 3-session window in the same history
    s = px[t].dropna()
    all3 = (s.shift(-H) / s - 1.0).dropna()
    r["ctl_all_3d_pct"] = round(100 * all3.mean(), 3)
    r["edge_pp"] = round(r["mean_pct"] - 100 * all3.mean(), 3)
    out.append(r)
show(out, "the run into opex, all tickers")

print("\n" + "=" * 74)
print("B. TLT / IEF / LQD: era, concentration, per-year record")
print("=" * 74)
for t in ["TLT", "IEF", "LQD"]:
    v, d = window(t)
    show(era_split(d, v), f"{t}: era split at 2018")
    print(" ", cluster_note(d, v, k=2))
    by_yr = pd.Series(v).groupby(d.year.values).mean()
    print(f"  positive in {int((by_yr > 0).sum())} of {len(by_yr)} calendar years")
    # drop the two largest absolute episodes and re-test
    order = np.argsort(-np.abs(v))[:2]
    keep = np.ones(len(v), bool)
    keep[order] = False
    r = summarize(v[keep], f"{t} ex the 2 largest episodes")
    show([r], f"{t}: drop-the-biggest robustness")

print("\n" + "=" * 74)
print("C. August, and the August-midterm subset")
print("=" * 74)
for t in ["TLT", "IEF", "LQD"]:
    v, d = window(t)
    out = []
    for name, m in [
        ("all opex", np.ones(len(v), bool)),
        ("AUGUST opex", np.array(d.month == 8)),
        ("August + midterm", np.array((d.month == 8) & (d.year % 4 == 2))),
    ]:
        if m.sum() == 0:
            continue
        r = summarize(v[m], name)
        r["rec"] = f"{int((v[m] > 0).sum())}-{int((v[m] <= 0).sum())}"
        r["sign_p"] = round(sign_test(int((v[m] > 0).sum()), int(m.sum())), 4)
        out.append(r)
    show(out, f"{t}: by month bucket")

print("\n" + "=" * 74)
print("D. does the bond bid need a risk-off tape? split by SPY over the window")
print("=" * 74)
vt, dt = window("TLT")
vs, ds = window("SPY")
common = dt.intersection(ds)
mt = pd.Series(vt, index=dt).loc[common].values
ms = pd.Series(vs, index=ds).loc[common].values
for name, m in [("SPY up over the window", ms > 0), ("SPY down over the window", ms <= 0)]:
    r = summarize(mt[m], f"TLT | {name}")
    r["rec"] = f"{int((mt[m] > 0).sum())}-{int((mt[m] <= 0).sum())}"
    show([r], name)
print(f"  corr(SPY window, TLT window) = {np.corrcoef(ms, mt)[0,1]:+.3f}")
print(f"  SPY over the same window: {100*ms.mean():+.3f}%, "
      f"{int((ms>0).sum())}-{int((ms<=0).sum())}")
