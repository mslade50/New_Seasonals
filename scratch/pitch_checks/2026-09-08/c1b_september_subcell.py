"""C1 round 2 -- the one positive split: SEPTEMBER pair anchors on SPY.

C1's h=3 cell is dead overall (+0.001% vs +0.112% drift, month-x-tdom
excess -0.125pp, gate inverted, placebo rank 8 of 11). The single positive
sub-cell is SEPTEMBER: N=14, +0.844%, 85.7% hit, sign p 0.006.

Today is a September pair anchor, so that cell has to be priced properly
rather than waved away, and it has to be crossed with the two conditioners
the registry says own September: the pre-FOMC run-in and the 12-month
search that found the month.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd
pd.set_option("display.width", 240)
rng = np.random.default_rng(42)

px = close_panel(["SPY"])
idx = px["SPY"].dropna().index
pos = pd.Series(range(len(idx)), index=idx)
ppi = load_events(["ppi"])["date"]
cpi = set(load_events(["cpi"])["date"])
fomc = load_events(["fomc_decision"])["date"]
p_all, _ = anchor_positions(idx, ppi, 0)
pair_pos = [p for p in p_all if p + 1 < len(idx) and idx[p + 1] in cpi]
A = pd.DatetimeIndex(idx[[p - 3 for p in pair_pos if p - 3 >= 0]])

H = 3
r = fwd_lag(px["SPY"], H, lag=1)
d = A.intersection(r.dropna().index)
v = r.reindex(d).values
base = r.dropna()

print("1. the 12-month search priced")
mo = d.month
obs = {m: v[mo == m].mean() for m in range(1, 13) if (mo == m).sum() >= 3}
print("   per-month means (%):", {k: round(100 * x, 3) for k, x in obs.items()})
best = max(obs.values()); cnt = 0; NP = 5000
for _ in range(NP):
    perm = rng.permutation(v)
    cnt += (max(perm[mo == m].mean() for m in obs) >= best)
print(f"   September mean {100*v[mo==9].mean():+.3f}% n={(mo==9).sum()}")
print(f"   permutation P(SOME month with n>=3 looks this good) = {cnt/NP:.4f}")

print("\n2. is September's cell the pre-FOMC run-in? (days from entry to FOMC)")
fp, _ = anchor_positions(idx, fomc, 0)
fset = np.array(sorted(fp))
rows = []
for dt in d[mo == 9]:
    p = pos[dt] + 1                       # entry position
    nxt = fset[fset >= p]
    gap = int(nxt[0] - p) if len(nxt) else 99
    rows.append({"anchor": str(dt.date()), "entry_to_FOMC_td": gap,
                 "ret_pct": round(100 * r.loc[dt], 3)})
t = pd.DataFrame(rows)
print(t.to_string(index=False))
print(f"   FOMC inside the h=3 hold (gap<=3): {int((t.entry_to_FOMC_td<=3).sum())} of {len(t)}"
      f"  mean {t.loc[t.entry_to_FOMC_td<=3,'ret_pct'].mean():+.3f}%")
print(f"   FOMC outside (gap>3, = TODAY at 6): "
      f"{int((t.entry_to_FOMC_td>3).sum())}  mean "
      f"{t.loc[t.entry_to_FOMC_td>3,'ret_pct'].mean():+.3f}%")
sub = t.loc[t.entry_to_FOMC_td > 3, 'ret_pct'].values / 100
print(f"   -> TODAY's bucket (FOMC 6 td out, outside the hold): n={len(sub)} "
      f"mean {100*sub.mean():+.3f}% hit {100*(sub>0).mean():.1f}% "
      f"sign p {sign_test(int((sub>0).sum()), len(sub)):.4f}")

print("\n3. month-x-tdom control on the September cell alone")
tdom = pd.Series(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).cumcount().values + 1,
                 index=idx)
cells = pd.DataFrame({"m": idx.month, "d": tdom.values, "r": r.values},
                     index=idx).dropna()
cm = cells.groupby(["m", "d"])["r"].mean()
sig = cells.reindex(d[mo == 9])
exc = sig["r"].values - np.asarray(sig.set_index(["m", "d"]).index.map(cm), float)
exc = exc[~np.isnan(exc)]
print(f"   September excess vs month-x-tdom: {100*exc.mean():+.3f}pp  n={len(exc)}"
      f"  hit {100*(exc>0).mean():.1f}%")
