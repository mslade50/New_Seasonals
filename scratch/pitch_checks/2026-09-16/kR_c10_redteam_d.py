"""kR round 4: family charge as minP instead of max|t|.

max|t| over rungs with n = 5..204 is dominated by the fat tails of the n=5-16
cells, so B3 in kR_c10_redteam_b.py is not a like-for-like statistic. Here each
cell's tdom-matched t is converted to a two-sided Student-t p with its own df,
and the family statistic is min p. Same seed, same null draws as round 2.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd
from scipy import stats

pxd = load_prices(["DX-Y.NYB", "TLT", "GLD", "^TNX"])
FOMC = load_events(["fomc_decision"])["date"]
FOMC = pd.DatetimeIndex(FOMC[FOMC <= pd.Timestamp("2026-09-15")])
dx = pxd["DX-Y.NYB"]["Close"].dropna()
idx = dx.index
ym = pd.Series(idx.year * 100 + idx.month, index=idx)
TD = ym.groupby(ym.values).cumcount().values + 1
POS = pd.Series(np.arange(len(idx)), index=idx)
tnx = pxd["^TNX"]["Close"].dropna()
mx = tnx.rolling(252).max()
chg21 = tnx - tnx.shift(21)
fomcA = [d for d in FOMC if len(tnx.index[tnx.index < d]) and
         not np.isnan(mx.get(tnx.index[tnx.index < d][-1], np.nan)) and d in POS.index]
dpA = np.array([POS[d] for d in fomcA])
ev = [tnx.index[tnx.index < d][-1] for d in fomcA]
tv = np.array([tnx[e] for e in ev])
tm = np.array([mx[e] for e in ev])
c21 = np.array([chg21[e] for e in ev])
dist = tv / tm - 1.0
GATES = {"ALL": np.ones(len(dpA), bool), "at_max": tv >= tm - 1e-9, "w0.5": dist >= -0.005,
         "w1.0": dist >= -0.01, "w2.0": dist >= -0.02, "thr20": c21 >= 0.20, "NOT_w2.0": dist < -0.02}
GATES["at_max&thr20"] = GATES["at_max"] & GATES["thr20"]
EX = np.zeros(len(idx), bool)
for d in FOMC:
    if d in POS.index:
        p = POS[d]
        EX[max(0, p - 5):p + 6] = True
RET, BUCK = {}, {}
for vk, tk in (("TLT", "TLT"), ("DX", "DX-Y.NYB"), ("GLD", "GLD")):
    s2 = pxd[tk]["Close"].dropna().reindex(idx)
    for h in range(1, 6):
        r = (s2.shift(-h) / s2 - 1.0).values
        ok = ~np.isnan(r)
        RET[(vk, h)] = r
        BUCK[(vk, h)] = np.array([np.nanmean(r[(TD == j) & ~EX & ok]) if ((TD == j) & ~EX & ok).any()
                                  else np.nan for j in range(1, 24)])


def pmat(X):
    n = (~np.isnan(X)).sum(axis=-1)
    m = np.nanmean(X, axis=-1)
    sd = np.nanstd(X, axis=-1, ddof=1)
    t = m / (sd / np.sqrt(n))
    return np.nan_to_num(2 * stats.t.sf(np.abs(t), np.maximum(n - 1, 1)), nan=1.0)


obs = {}
for key, r in RET.items():
    x = r[dpA] - BUCK[key][TD[dpA] - 1]
    for gk, gm in GATES.items():
        obs[(key[0], key[1], gk)] = pmat(x[gm][None, :])[0]
print("observed p: DX h3 ALL %.5f  DX h5 ALL %.5f" % (obs[("DX", 3, "ALL")], obs[("DX", 5, "ALL")]))
print("observed smallest p cells:", [(k, round(float(v), 5)) for k, v in sorted(obs.items(), key=lambda kv: kv[1])[:6]])

NPERM = 2000
prng = np.random.default_rng(20260916)
cand = np.where(~EX & (np.arange(len(idx)) < len(idx) - 6))[0]
PP = np.empty((NPERM, len(dpA)), int)
for j, p in enumerate(dpA):
    c = cand[(TD[cand] == TD[p]) & (np.abs(cand - p) <= 252)]
    PP[:, j] = c[prng.integers(0, len(c), size=NPERM)]
FAM = {"ALL only": ["ALL"], "rungs n>=30": ["ALL", "thr20", "NOT_w2.0"], "all 8 rungs": list(GATES)}
MINP = {k: np.ones(NPERM) for k in FAM}
for key, r in RET.items():
    X = r[PP] - BUCK[key][TD[PP] - 1]
    for gk, gm in GATES.items():
        pv = pmat(X[:, gm])
        for fk, rungs in FAM.items():
            if gk in rungs:
                MINP[fk] = np.minimum(MINP[fk], pv)
for fk in FAM:
    mp = MINP[fk]
    print(f"  minP {fk}: null minP 5/10/50 pct {np.percentile(mp, [5, 10, 50]).round(4)}  "
          f"P(<= DX h3) {(1 + (mp <= obs[('DX', 3, 'ALL')]).sum()) / (NPERM + 1):.4f}  "
          f"P(<= DX h5) {(1 + (mp <= obs[('DX', 5, 'ALL')]).sum()) / (NPERM + 1):.4f}")
