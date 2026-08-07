"""INV2 final probe: the 2018+ cell's excess is real WITHIN 2021-2022. So the
question is whether 2021-2022 is distinguishable ex ante from 2006/2013/2016,
where the identical trigger LOST. If it isn't, the era split is a coin flip
that was called after the toss.

Also: is today part of a live cluster (2026-07-22 fired 11 td ago)?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import numpy as np
import pandas as pd

P = close_panel(["TLT", "IEF", "^TNX", "SPY"]).dropna()
idx = P.index
off_lo = P["TLT"] / P["TLT"].rolling(252).min() - 1.0
tnx63 = pct_rank(P["^TNX"], 63)
M = ((off_lo <= 0.015) & (tnx63 >= 85)).fillna(False)


def short_tlt(h):
    return -(P["TLT"].shift(-(1 + h)) / P["TLT"].shift(-1) - 1.0)


S = idx[M.values & short_tlt(10).notna().values]
E = declusters(S, 10, idx)
V = short_tlt(10).reindex(E).to_numpy()

# ex-ante context variables, all knowable on the signal close
tnx = P["^TNX"]
tlt = P["TLT"]
ctx = pd.DataFrame({
    "tnx_lvl": tnx,
    "tnx_chg_63d_bps": (tnx - tnx.shift(63)) * 100,
    "tnx_lvl_pctile_5y": tnx.rolling(1260).rank(pct=True) * 100,
    "tlt_252d_ret_pct": 100 * (tlt / tlt.shift(252) - 1),
    "tlt_dd_from_252d_hi_pct": 100 * (tlt / tlt.rolling(252).max() - 1),
    "spy_252d_ret_pct": 100 * (P["SPY"] / P["SPY"].shift(252) - 1),
})

print("=" * 100)
print("FULL-SAMPLE EPISODE TABLE, short TLT h=10, with ex-ante context")
print("=" * 100)
rows = []
for d, v in zip(E, V):
    c = ctx.loc[d]
    rows.append({"date": str(d.date()), "yr": d.year, "shortTLT_10td_pct": 100 * v,
                 "tnx_lvl": c.tnx_lvl, "tnx_63d_chg_bps": c.tnx_chg_63d_bps,
                 "tnx_5y_pctile": c.tnx_lvl_pctile_5y,
                 "tlt_1y_ret": c.tlt_252d_ret_pct, "tlt_dd_1y_hi": c.tlt_dd_from_252d_hi_pct,
                 "spy_1y_ret": c.spy_252d_ret_pct})
show(rows)

print("\nWIN/LOSS by era:")
pre = E < pd.Timestamp("2018-01-01")
print(f"  pre-2018 (2006/2013/2016): N={pre.sum()}  mean={100*V[pre].mean():+.3f}%  "
      f"hit={100*(V[pre]>0).mean():.0f}%")
print(f"  2021-2022               : N={((E.year>=2021)&(E.year<=2022)).sum()}  "
      f"mean={100*V[(E.year>=2021)&(E.year<=2022)].mean():+.3f}%  "
      f"hit={100*(V[(E.year>=2021)&(E.year<=2022)]>0).mean():.0f}%")
print(f"  2023+                   : N={(E.year>=2023).sum()}  "
      f"mean={100*V[E.year>=2023].mean():+.3f}%  hit={100*(V[E.year>=2023]>0).mean():.0f}%")

print("\nCan any ex-ante variable separate the winning era from the losing one?")
for col in ("tnx_lvl", "tnx_chg_63d_bps", "tnx_lvl_pctile_5y", "tlt_252d_ret_pct",
            "tlt_dd_from_252d_hi_pct", "spy_252d_ret_pct"):
    a = ctx.loc[E[pre], col].to_numpy()
    b = ctx.loc[E[(E.year >= 2021) & (E.year <= 2022)], col].to_numpy()
    tod = ctx[col].iloc[-1]
    ov = "OVERLAP" if (np.nanmin(b) <= np.nanmax(a) and np.nanmin(a) <= np.nanmax(b)) else "separates"
    print(f"  {col:<26s} pre2018 [{np.nanmin(a):+8.2f},{np.nanmax(a):+8.2f}]  "
          f"2021-22 [{np.nanmin(b):+8.2f},{np.nanmax(b):+8.2f}]  {ov:>9s}   TODAY={tod:+8.2f}")

print("\nTODAY's placement inside the full-sample episode distribution:")
for col in ("tnx_lvl", "tnx_chg_63d_bps", "tnx_lvl_pctile_5y", "tlt_252d_ret_pct", "spy_252d_ret_pct"):
    e = ctx.loc[E, col].to_numpy()
    tod = ctx[col].iloc[-1]
    print(f"  {col:<26s} TODAY={tod:+8.2f}  episode range [{np.nanmin(e):+8.2f},"
          f"{np.nanmax(e):+8.2f}]  pctile={100*np.nanmean(e < tod):5.1f}%")

print("\nCLUSTER CHECK: is today a fresh episode or a re-fire?")
pos = pd.Series(range(len(idx)), index=idx)
last_trig = S[-1]
print(f"  most recent trigger DAY in sample: {last_trig.date()}")
recent = S[S >= pd.Timestamp("2026-01-01")]
print(f"  2026 trigger days: {[str(d.date()) for d in recent]}")
print(f"  today (2026-08-06) index pos {len(idx)-1}; last episode start "
      f"{E[-1].date()} at pos {pos[E[-1]]} -> gap = {len(idx)-1-pos[E[-1]]} td "
      f"({'NEW episode by gap=10' if len(idx)-1-pos[E[-1]] >= 10 else 'SAME cluster'})")
print(f"  that live cluster's realized h=10: {100*V[-1]:+.2f}%")

# how often does the trigger persist? (a 'floor' can keep making new lows)
print("\nTrigger persistence: consecutive-day run lengths")
runs, cur, prev = [], 0, None
for d in S:
    p = pos[d]
    if prev is not None and p == prev + 1:
        cur += 1
    else:
        if cur:
            runs.append(cur)
        cur = 1
    prev = p
runs.append(cur)
print(f"  run lengths: {sorted(runs, reverse=True)}  median={int(np.median(runs))} "
      f"max={max(runs)}  -> the 'floor' state persists, so a short here can be early")
