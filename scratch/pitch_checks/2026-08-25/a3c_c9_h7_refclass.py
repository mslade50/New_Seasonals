"""C9 round 2b - the h=7 cell against its reference class, ex-2024, and cost.

h=7 is a SHELF (h=5..h=9 all positive, +0.68 / +0.85 / +1.61 / +1.78 / +1.80)
rather than a one-horizon spike, the gate ADDS +0.976pp over the bare washout
there, and drop-best-3 is still +0.906%.  That is enough to stop the round-1
kill from being automatic, so this script runs the two tests that decide it:

  1. the mandated single-sector reference class AT h=7 (the 2026-08-19 rule:
     any single-sector claim owes a Cochran-Q-style heterogeneity read and a
     permutation max-of-k), and
  2. what is left after 2024, which is 5 of the 13 episodes and 61% of the
     total return.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 240)

SECTORS = ["XLK", "XLV", "XLP", "XLU", "XLI", "XLF", "XLY", "XLE", "XLB"]
px = close_panel(SECTORS + ["SPY"])
H = 7

print("########## 1. REFERENCE CLASS AT h=7 - identical gate on all 9 SPDRs ##########")
rows, stats = [], []
for s_ in SECTORS:
    a5 = pct_rank(px[s_], 5)
    a63 = pct_rank(px[s_], 63)
    hi = rolling_on_valid(px[s_], lambda x: x.rolling(252).max())
    dd = px[s_] / hi - 1.0
    m = ((a5 <= 5) & (a63 >= 30) & (a63 <= 60) & (dd >= -0.05)).fillna(False)
    rr = vehicle_ret(px, [(s_, 1.0)], H)
    v = rr.dropna().index
    e = declusters(px.index[m.values].intersection(v), H, v)
    r = summarize(rr.loc[e].values, f"LONG {s_}")
    if r["n"] > 1:
        r["drift_pct"] = round(100 * rr.loc[v].mean(), 3)
        r["edge_pct"] = round(r["mean_pct"] - 100 * rr.loc[v].mean(), 3)
        stats.append((s_, r["n"], r["mean_pct"] / 100, r["sd_pct"] / 100, r["t"]))
    rows.append(r)
show(rows, f"long each sector on its own washout-in-intact-trend, h={H}")

means = np.array([m for _, _, m, _, _ in stats])
ns = np.array([n for _, n, _, _, _ in stats])
sds = np.array([s for _, _, _, s, _ in stats])
ses = sds / np.sqrt(ns)
w = 1.0 / ses**2
pooled = (w * means).sum() / w.sum()
Q = (w * (means - pooled) ** 2).sum()
from scipy import stats as sps  # noqa
print(f"\n  pooled effect {100*pooled:+.3f}%   Cochran Q = {Q:.2f} on {len(stats)-1} df, "
      f"p = {1 - sps.chi2.cdf(Q, len(stats)-1):.3f}")
print(f"  -> {'heterogeneous: XLI may be special' if 1 - sps.chi2.cdf(Q, len(stats)-1) < 0.10 else 'HOMOGENEOUS: no evidence XLI differs from its peers; the honest estimate is the POOLED one'}")
ts = np.array([t for _, _, _, _, t in stats])
order = np.argsort(-np.abs(ts))
print(f"  |t| ranking: {[(stats[i][0], round(ts[i],2)) for i in order]}")
xli_t = [t for s_, _, _, _, t in stats if s_ == "XLI"][0]
print(f"  XLI ranks {[i+1 for i,j in enumerate(order) if stats[j][0]=='XLI'][0]} of {len(stats)}")

# permutation max-of-9 null with the observed per-sector episode counts
rng = np.random.default_rng(11)
maxt = []
fw = {s_: vehicle_ret(px, [(s_, 1.0)], H).dropna().values for s_ in SECTORS}
nmap = {s_: n for s_, n, _, _, _ in stats}
for _ in range(4000):
    best = 0.0
    for s_ in SECTORS:
        arr = fw[s_]
        k = nmap.get(s_, 10)
        idx = rng.integers(0, len(arr), size=k)
        v = arr[idx]
        sd = v.std(ddof=1)
        t = v.mean() / (sd / np.sqrt(k)) if sd > 0 else 0.0
        best = max(best, abs(t))
    maxt.append(best)
maxt = np.array(maxt)
print(f"  permutation max-of-9 null: P(max|t| >= {abs(xli_t):.2f}) = "
      f"{(maxt >= abs(xli_t)).mean():.3f}  (4000 draws, observed per-sector N)")

print("\n########## 2. WHAT IS LEFT WITHOUT 2024 ##########")
r5 = pct_rank(px["XLI"], 5)
r63 = pct_rank(px["XLI"], 63)
hi = rolling_on_valid(px["XLI"], lambda x: x.rolling(252).max())
d52 = px["XLI"] / hi - 1.0
MAIN = ((r5 <= 5) & (r63 >= 30) & (r63 <= 60) & (d52 >= -0.05)).fillna(False)
ret = vehicle_ret(px, [("XLI", 1.0)], H)
v = ret.dropna().index
e = declusters(px.index[MAIN.values].intersection(v), H, v)
vals = ret.loc[e].values
m24 = np.array([d.year == 2024 for d in e])
print(f"  full      N={len(vals):2d} mean {100*vals.mean():+.3f}%  rec "
      f"{int((vals>0).sum())}-{int((vals<=0).sum())}")
print(f"  2024 only N={int(m24.sum()):2d} mean {100*vals[m24].mean():+.3f}%")
print(f"  ex-2024   N={int((~m24).sum()):2d} mean {100*vals[~m24].mean():+.3f}%  rec "
      f"{int((vals[~m24]>0).sum())}-{int((vals[~m24]<=0).sum())}  sign p="
      f"{sign_test(int((vals[~m24]>0).sum()), int((~m24).sum())):.4f}")
print(f"  ex-2024 vs XLI drift {100*ret.loc[v].mean():+.3f}%  -> edge "
      f"{100*(vals[~m24].mean()-ret.loc[v].mean()):+.3f}pp")
print(f"  bootstrap P(mean<=0) full {bootstrap_p_le0(vals):.3f}   "
      f"ex-2024 {bootstrap_p_le0(vals[~m24]):.3f}")

print("\n########## 3. COST AND EVENT RISK IN THE h=7 WINDOW ##########")
print(f"  1 leg x ~5 bps round trip; episode mean {100*vals.mean():.3f}% = "
      f"{100*100*vals.mean():.1f} bps -> {100*100*vals.mean()/5:.1f}x cost")
print(f"  ex-2024: {100*100*vals[~m24].mean()/5:.1f}x cost")
fl_nfp = event_in_window(e, px.index, H, 1, ("nfp",))
fl_jh = event_in_window(e, px.index, H, 1, ("jackson_hole",))
print(f"  episodes with an NFP inside the hold: {int(fl_nfp.sum())} of {len(e)}  "
      f"mean {100*vals[fl_nfp].mean() if fl_nfp.sum() else float('nan'):+.3f}% "
      f"vs {100*vals[~fl_nfp].mean():+.3f}% without")
print(f"  episodes with Jackson Hole inside the hold: {int(fl_jh.sum())} of {len(e)}")
print("  A h=7 MOC entry today (2026-08-25) exits ~2026-09-04: it carries the NVDA")
print("  print (+1 td), the Jackson Hole speech (+3 td) AND the Sept NFP (+8 td).")

print("\n########## 4. TAPE OVER-SELECTION AT h=7 ##########")
sma200 = rolling_on_valid(px["SPY"], lambda x: x.rolling(200).mean())
above = px["SPY"] > sma200
dd = px.index[MAIN.values].intersection(above.dropna().index)
print(f"  trigger days above SPY 200d: {100*above.loc[dd].mean():.1f}% "
      f"(base {100*above.dropna().mean():.1f}%), N below = "
      f"{int((~above.loc[dd]).sum())}")
