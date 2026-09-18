"""Semis +2.61% (SMH) on a session the S&P fell 0.39%. Cell: SMH up 2% or
more on a day SPY closed lower. Forward SMH and SPY (lag1), declustered 5,
controls: SMH up 2%+ on any day; SPY-down days alone. Sub-cell: SMH 63d
return rank in the bottom decile at the anchor (tonight 5.2).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    cluster_note, declusters, era_split, fwd_lag, load_prices, pct_rank,
    sign_test, summarize,
)

warnings.filterwarnings("ignore")
ASOF = pd.Timestamp("2026-09-04")
raw = load_prices(["SMH", "SPY", "QQQ"])
smh, spy, qqq = raw["SMH"]["Close"].dropna(), raw["SPY"]["Close"].dropna(), raw["QQQ"]["Close"].dropna()
idx = smh.index.intersection(spy.index)
smh, spy = smh.reindex(idx), spy.reindex(idx)
r_smh, r_spy = smh.pct_change(), spy.pct_change()
rank63 = pct_rank(smh.pct_change(63), 63) if False else None
r63 = smh.pct_change(63)
rank63 = r63.rolling(252).apply(lambda w: 100 * (w[:-1] < w[-1]).mean(), raw=True)
print(f"tonight SMH {100*r_smh.iloc[-1]:+.2f}%  SPY {100*r_spy.iloc[-1]:+.2f}%  SMH 63d rank {rank63.iloc[-1]:.1f}  "
      f"spread {100*(r_smh.iloc[-1]-r_spy.iloc[-1]):+.2f}pp")
spread = r_smh - r_spy
print(f"SMH-SPY 1d spread pctile since 2000: {100*(spread.dropna() < spread.iloc[-1]).mean():.1f}")

mask = (r_smh >= 0.02) & (r_spy < 0)
days = idx[mask.values]
days = days[days <= ASOF]
dc = declusters(days, 5, idx)
print(f"SMH >= +2% on SPY-down day: {len(days)} days, {len(dc)} declustered; by year "
      f"{pd.Series(1, index=days).groupby(days.year).sum().to_dict()}")
mask_any = r_smh >= 0.02
any_dc = declusters(idx[mask_any.values & (idx <= ASOF)], 5, idx)
weak = days[rank63.reindex(days).fillna(50).values < 10]
weak_dc = declusters(weak, 5, idx)
print(f"sub-cell SMH 63d rank < 10: {len(weak)} days / {len(weak_dc)} dc: {[d.date().isoformat() for d in weak_dc]}")


def block(name, s, dates, h, lag=1, notes=False):
    f = fwd_lag(s, h, lag)
    v = f.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v) == 0:
        print(f"  {name:<48} n=0")
        return v
    st = summarize(v.values)
    nup = int((v > 0).sum())
    base = f.dropna()
    print(f"  {name:<48} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
          f"{nup}-{len(v)-nup} ({st['hit']:.1f}%)  t={st['t']:+.2f}  sp={sign_test(nup, len(v)):.4f}  "
          f"| base {100*base.mean():+.3f}% hit {100*(base>0).mean():.1f}%  worst {st['worst_pct']:+.2f}%")
    if notes:
        print("    era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3), round(e.get("hit", np.nan), 1))
                           for e in era_split(v.index, v.values)])
        print("    concentration:", cluster_note(v.index, v.values))
    return v


for h in (1, 3, 5, 10, 21):
    print(f"\n--- h={h} ---")
    block(f"SMH after cell (dc5)", smh, dc, h, notes=(h in (5, 21)))
    block(f"SMH after any +2% day (dc5)", smh, any_dc, h)
    block(f"SMH after cell, 63d rank<10 (dc5)", smh, weak_dc, h)
    block(f"SPY after cell (dc5)", spy, dc, h, notes=(h == 5))
    block(f"SMH-SPY spread after cell (dc5)", (smh / spy), dc, h)

# --- vehicle comparison for the idea: QQQ and IWM after the same cell ---
print("\n=== other vehicles after the cell (dc5) ===")
for tkr in ("QQQ", "IWM"):
    s = load_prices([tkr])[tkr]["Close"].dropna()
    for h in (5, 10):
        block(f"{tkr} after cell (dc5) h={h}", s, dc, h, notes=(h == 5))
