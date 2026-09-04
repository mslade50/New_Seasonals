"""C7 round 2: the exact number that would turn the September dollar cell on.

Round 1 killed it on the pre-declared max-of-12 charge (DX h=3, the best of
the eight cells scanned, at P = 0.404). This script states the threshold a
future morning can re-run mechanically: the 95th percentile of the
max-of-12 permutation t distribution, and the September mean it implies.

Also: does the September DX cell survive when the SPY September window is
regressed out? Round 1 measured corr = -0.44 at h=3, so the dollar cell may
be nothing but the (already-closed, and stronger) September equity cell.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
          "Aug", "Sep", "Oct", "Nov", "Dec"]
px = load_prices(["DX-Y.NYB", "SPY"])
ser = {t: px[t]["Close"].dropna() for t in px}
rng = np.random.default_rng(7)


def ltd_positions(idx):
    per = pd.Series(idx.to_period("M"), index=range(len(idx)))
    return [int(g.index.max()) for _, g in per.groupby(per.values)][:-1]


def cell(t, h):
    s = ser[t]
    idx, v = s.index, s.values
    e_, r_, lab_ = [], [], []
    for p in ltd_positions(idx):
        if p + h >= len(idx) or p + 1 >= len(idx):
            continue
        e_.append(idx[p])
        r_.append(v[p + h] / v[p] - 1.0)
        lab_.append(idx[p + 1].month)
    return pd.DatetimeIndex(e_), np.asarray(r_, float), np.asarray(lab_)


print("1. THE TURN-ON THRESHOLD (max-of-12 permutation, 20000 draws)")
for h in (1, 3, 5):
    d, r, m = cell("DX-Y.NYB", h)
    sep = r[m == 9]
    obs_t = sep.mean() / (sep.std(ddof=1) / np.sqrt(len(sep)))
    mx = []
    for _ in range(20000):
        perm = rng.permutation(m)
        best = -1e9
        for mm in range(1, 13):
            v = r[perm == mm]
            sd = v.std(ddof=1)
            if len(v) >= 3 and sd > 0:
                best = max(best, v.mean() / (sd / np.sqrt(len(v))))
        mx.append(best)
    mx = np.asarray(mx)
    t95 = float(np.quantile(mx, 0.95))
    # September mean that t95 implies at the observed sd and N
    need_mu = t95 * sep.std(ddof=1) / np.sqrt(len(sep))
    print("  DX h=%2d: observed Sep t = %+.2f (mean %+.3f%%, N=%d, sd %.3f%%);  "
          "P(max-of-12 >= obs) = %.3f" % (h, obs_t, 100 * sep.mean(), len(sep),
                                          100 * sep.std(ddof=1),
                                          float((mx >= obs_t).mean())))
    print("       TURN-ON: needs Sep t >= %.2f, i.e. a September mean >= %+.3f%% "
          "at today's dispersion -- %.1fx the observed %+.3f%%"
          % (t95, 100 * need_mu, need_mu / sep.mean(), 100 * sep.mean()))

print()
print("2. IS THE DOLLAR CELL JUST THE (CLOSED, STRONGER) SEPTEMBER EQUITY CELL?")
for h in (1, 3, 5, 10):
    dd, rd, md = cell("DX-Y.NYB", h)
    ds, rs, ms = cell("SPY", h)
    a = pd.Series(rd[md == 9], index=dd[md == 9])
    b = pd.Series(rs[ms == 9], index=ds[ms == 9])
    j = pd.concat([a.rename("dx"), b.rename("spy")], axis=1).dropna()
    beta = np.polyfit(j["spy"], j["dx"], 1)
    resid = j["dx"] - (beta[0] * j["spy"] + beta[1])
    alpha = beta[1]
    se_a = resid.std(ddof=2) / np.sqrt(len(j))
    print("  h=%2d: DX Sep %+.3f%%  beta-to-SPY %+.2f  |  SPY-orthogonal alpha "
          "%+.3f%% (t %+.2f, N=%d)"
          % (h, 100 * j["dx"].mean(), beta[0], 100 * alpha, alpha / se_a, len(j)))
