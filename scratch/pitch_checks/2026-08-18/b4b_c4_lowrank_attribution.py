"""C4 follow-up. b4 killed the plain joint cell (worse than all three gates at
every h>=2). ONE live-like subset survived that pass: joint & VIX 21d rank<=25,
h=10 +0.874% t=2.67 N=59. That is a THIRD condition added after seeing the
result, i.e. a search, and it needs its own gate attribution:

  A  VIX 21d rank<=25 ALONE            (the calm-regime marker)
  B  A & VIX 1d >= +5%                 (adds the pop, no spot condition)
  C  A & SPY 1d > -0.75%               (adds the no-damage leg, no pop)
  D  A & both                          (the surviving cell)
  E  local +/-126td control of D, and all days

If D does not beat A, the "divergence" is a calm-tape proxy and the whole
candidate collapses to "buy SPY when VIX has been quiet", which is drift.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["SPY", "^VIX"]).dropna(subset=["SPY", "^VIX"])
spy, vix = px["SPY"], px["^VIX"]
r_spy, r_vix = spy.pct_change(), vix.pct_change()
rk = pct_rank(vix, 21)
print(f"live: VIX 21d rank {rk.iloc[-1]:.1f}, VIX 1d {100*r_vix.iloc[-1]:+.2f}%, "
      f"SPY 1d {100*r_spy.iloc[-1]:+.2f}%")

A = rk <= 25
B = A & (r_vix >= 0.05)
C = A & (r_spy > -0.0075)
D = A & (r_vix >= 0.05) & (r_spy > -0.0075)


def ep(m, gap):
    return declusters(px.index[m.reindex(px.index, fill_value=False).values], gap, px.index)


for h in (3, 5, 10):
    ret = vehicle_ret(px, [("SPY", 1.0)], h, lag=1)
    v = ret.notna()
    gap = max(h, 5)
    loc = local_control(px.index[v.values], px.index[(D & v).values])
    rows = [
        summarize(ret.loc[ep(D & v, gap)].values, f"D joint & rank<=25 (N_days={int((D&v).sum())})"),
        summarize(ret.loc[ep(A & v, gap)].values, f"A rank<=25 ALONE (N_days={int((A&v).sum())})"),
        summarize(ret.loc[ep(B & v, gap)].values, f"B A & VIX pop (N_days={int((B&v).sum())})"),
        summarize(ret.loc[ep(C & v, gap)].values, f"C A & no-damage (N_days={int((C&v).sum())})"),
        summarize(ret.loc[loc].values, "E local +/-126td ex-D"),
        summarize(ret[v].values, "all days"),
    ]
    show(rows, f"gate attribution inside the calm-VIX regime, long SPY, h={h}")
    d, a = rows[0]["mean_pct"], rows[1]["mean_pct"]
    print(f"  D - A = {d - a:+.3f}pp   D - local = {d - rows[4]['mean_pct']:+.3f}pp")

# rank-threshold neighbours: does the 25 cut matter, or is it the grid speaking?
for h in (10,):
    ret = vehicle_ret(px, [("SPY", 1.0)], h, lag=1)
    v = ret.notna()
    rows = []
    for cut in (10, 15, 20, 25, 30, 40, 50):
        m = (rk <= cut) & (r_vix >= 0.05) & (r_spy > -0.0075) & v
        ma = (rk <= cut) & v
        rD = summarize(ret.loc[ep(m, 10)].values, f"D rank<={cut}")
        rA = summarize(ret.loc[ep(ma, 10)].values, f"A rank<={cut}")
        rD["A_mean_pct"] = rA.get("mean_pct", np.nan)
        rD["D_minus_A"] = rD.get("mean_pct", np.nan) - rA.get("mean_pct", np.nan)
        rows.append(rD)
    show(rows, f"rank-cut neighbours, h={h}  (D vs its own A control at each cut)")

# concentration + era for the surviving cell
for h in (10,):
    ret = vehicle_ret(px, [("SPY", 1.0)], h, lag=1)
    d = ep(D & ret.notna(), 10)
    vals = ret.loc[d].values
    print(f"\nD episodes h={h}: N={len(d)}  {cluster_note(d, vals)}")
    show(era_split(d, vals), "D era split 2018")
    show(era_split(d, vals, "2021-01-01"), "D era split 2021")
    yrs = pd.Series(vals, index=d).groupby(d.year).mean() * 100
    print("  per-year mean %:", {int(k): round(x, 2) for k, x in yrs.items()})
    print(f"  drop best year: ", end="")
    best = yrs.idxmax()
    keep = d.year != best
    print(summarize(vals[keep], f"ex-{best}"))
