"""C2 BOUNDARY PROBE — the h=21 cell is the only one that got near a family-wise
threshold (permutation p = 0.0500, XLE rank 1 of 9). Decide KILL vs NEAR-MISS
honestly instead of taking the boundary at face value.

Four questions, all of which a real h=21 effect must answer:
  1. Is the h=21 result monotone in the gate (the dose response that failed at
     h=5 and h=10)?
  2. Does it survive dropping its heaviest years?
  3. Is it stable across eras and declustering gaps?
  4. Is the permutation p stable to the seed, or is 0.0500 a coin landing on
     the line?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd
import numpy as np

ASOF = pd.Timestamp("2026-08-31")
SEC9 = ["XLE", "XLK", "XLF", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
PX = load_prices(SEC9 + ["SPY"])
PX = {t: d[d.index <= ASOF] for t, d in PX.items()}
SPY_1D = PX["SPY"]["Close"].dropna().pct_change()
TOL = 0.0005
H = 21


def at_high(s, tol=TOL):
    return s >= s.rolling(252, min_periods=252).max() * (1 - tol)


xle = PX["XLE"]["Close"].dropna()
ah = at_high(xle).fillna(False)
idx_all = xle.index[ah.reindex(xle.index, fill_value=False).values]
r = fwd_lag(xle, H, 1)
dn = (ah & (SPY_1D.reindex(xle.index) < 0)).fillna(False)
t_dn = xle.index[dn.reindex(xle.index, fill_value=False).values]
e = declusters(t_dn.intersection(r.dropna().index), 10, xle.index)

print("===== 1. h=21 DOSE RESPONSE (must be monotone decreasing in SPY's move) =====")
rows = []
for lo, hi in [(-9, -1.0), (-1.0, -0.5), (-0.5, -0.25), (-0.25, 0.0),
               (0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 9)]:
    sd = 100 * SPY_1D.reindex(idx_all)
    sel = idx_all[((sd >= lo) & (sd < hi)).values].intersection(r.dropna().index)
    ep = declusters(sel, 10, xle.index)
    s = summarize(r.reindex(ep).values, f"SPY 1d [{lo},{hi})%")
    s["n_days"] = len(sel)
    rows.append(s)
show(rows, "h=21 by SPY same-day move")
j = pd.DataFrame({"spy": SPY_1D.reindex(idx_all), "f": r.reindex(idx_all)}).dropna()
print("  spearman(SPY move, XLE h=21 fwd) over at-high days = %+.4f  n=%d"
      % (j["spy"].corr(j["f"], method="spearman"), len(j)))

print("\n===== 2. YEAR-DROP ROBUSTNESS (h=21) =====")
v = r.reindex(e).values
by = pd.DataFrame({"y": e.year, "v": v}).groupby("y")["v"].agg(["count", "mean", "sum"])
by[["mean", "sum"]] = (100 * by[["mean", "sum"]]).round(2)
print(by.to_string())
drift = 100 * r.dropna().mean()
print("  XLE all-days h=21 drift = %+.3f%%" % drift)
for drop in ([], [2022], [2026], [2022, 2026], [2005], [2022, 2026, 2005]):
    keep = ~np.isin(e.year, drop) if drop else np.ones(len(e), bool)
    s = summarize(v[keep], str(drop))
    print("  drop %-20s N=%2d mean %+.3f%% EXCESS %+.3fpp hit %.1f%% t %s"
          % (str(drop), s["n"], s["mean_pct"], s["mean_pct"] - drift, s["hit"],
             f"{s['t']:+.2f}" if s["n"] > 1 else "na"))
print("  concentration:", cluster_note(e, v, k=3))
sv = np.sort(v)
print("  drop-best-2 %+.3f%%  drop-best-3 %+.3f%%  (vs full %+.3f%%)"
      % (100 * sv[:-2].mean(), 100 * sv[:-3].mean(), 100 * v.mean()))

print("\n===== 3. ERA + DECLUSTERING (h=21) =====")
show(era_split(e, v), "era split")
for gap in (5, 10, 21, 42, 63):
    ep = declusters(t_dn.intersection(r.dropna().index), gap, xle.index)
    s = summarize(r.reindex(ep).values, f"gap={gap}")
    print("  gap=%2d N=%2d mean %+.3f%% EXCESS %+.3fpp hit %.1f%% t %s"
          % (gap, s["n"], s["mean_pct"], s["mean_pct"] - drift, s["hit"],
             f"{s['t']:+.2f}" if s["n"] > 1 else "na"))

print("\n===== 4. IS FAMILY p=0.0500 STABLE ACROSS SEEDS? =====")
pools, obs = {}, {}
for t in SEC9:
    s = PX[t]["Close"].dropna()
    a = at_high(s).fillna(False)
    rr = fwd_lag(s, H, 1)
    ix = s.index[a.reindex(s.index, fill_value=False).values].intersection(rr.dropna().index)
    lab = (SPY_1D.reindex(ix) < 0).values
    val = rr.reindex(ix).values
    ok = ~np.isnan(val)
    lab, val = lab[ok], val[ok]
    if lab.sum() < 5 or (~lab).sum() < 5:
        continue
    obs[t] = val[lab].mean() - val[~lab].mean()
    pools[t] = (val, lab)
ps = []
for seed in (1, 7, 42, 99, 2026):
    rng = np.random.default_rng(seed)
    mx = np.empty(5000)
    for i in range(5000):
        mx[i] = max((val[(p := rng.permutation(lab))].mean() - val[~p].mean())
                    for val, lab in pools.values())
    ps.append(float((mx >= obs["XLE"]).mean()))
    print("  seed %5d -> family-wise p = %.4f" % (seed, ps[-1]))
print("  seed range %.4f .. %.4f  (a p that straddles 0.05 on the seed is not "
      "a 5%% result)" % (min(ps), max(ps)))

print("\n===== 5. WHAT WOULD TURN IT ON (the near-miss number, if any) =====")
print("  Full-cell h=21 excess: %+.3fpp on N=%d episodes." % (100 * v.mean() - drift, len(v)))
print("  Ex-2022/2026 excess:   %+.3fpp on N=%d."
      % (summarize(v[~np.isin(e.year, [2022, 2026])])["mean_pct"] - drift,
         int((~np.isin(e.year, [2022, 2026])).sum())))
