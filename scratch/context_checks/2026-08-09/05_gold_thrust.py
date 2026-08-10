"""Gold's week: +3.76% Friday, +8.7% over 5 sessions, 98.8th percentile, and
still 17% below its own 52w high.

The engine dropped GC=F from the fired P5 list on the per-trigger cap, so
this cell is recovered by hand (the cap logging is what made it findable).
Two questions: what follows a 5d thrust of this size, and does the answer
change when the thrust happens well BELOW the 52w high, i.e. inside a
drawdown rather than at a breakout.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import cluster_note, fwd_ret, load_prices, sign_test, show, summarize  # noqa: E402

ASOF = pd.Timestamp("2026-08-07")
px = load_prices(["GC=F", "SI=F", "DX-Y.NYB"])

close = px["GC=F"]["Close"].astype(float)
close = close[close.index <= ASOF]
r5 = close.pct_change(5, fill_method=None)
rank = r5.rolling(252).rank(pct=True) * 100
hi52 = close.rolling(252, min_periods=252).max()
dist_high = close / hi52 - 1.0

print(f"GC=F  last {close.iloc[-1]:.2f}  5d {100 * r5.iloc[-1]:+.2f}%  "
      f"rank {rank.iloc[-1]:.1f}  dist to 52w high {100 * dist_high.iloc[-1]:+.1f}%")

thrust = (rank >= 95).fillna(False)
in_drawdown = (dist_high <= -0.10).fillna(False)

for h in (1, 5):
    f = fwd_ret(close, h)
    valid = f.dropna().index

    def cell(mask: pd.Series, label: str) -> dict:
        idx = pd.DatetimeIndex(mask.index[mask.values]).intersection(valid)
        vals = f.loc[idx].values
        row = summarize(vals, label)
        if row["n"]:
            up = int((vals > 0).sum())
            row["record"] = f"{up}-{row['n'] - up}"
            row["sign_p"] = round(sign_test(max(up, row["n"] - up), row["n"]), 4)
        return row

    rows = [cell(thrust, "5d thrust, top 5%"),
            cell(thrust & in_drawdown, "thrust while >10% below the 52w high"),
            cell(thrust & ~in_drawdown, "thrust near the highs"),
            cell(pd.Series(True, index=close.index), "CTRL all days")]
    show(rows, f"GC=F forward {h}d after a 5d thrust")

idx = pd.DatetimeIndex((thrust & in_drawdown).index[(thrust & in_drawdown).values])
f1 = fwd_ret(close, 1)
sel = idx.intersection(f1.dropna().index)
print(f"\n  drawdown-thrust episodes: {len(sel)}")
print(f"  {cluster_note(sel, f1.loc[sel].values)}")
print("  most recent:", ", ".join(str(d.date()) for d in sel[-8:]))

# Does the dollar explain it? Friday's DXY z10 was -1.92.
dxy = px["DX-Y.NYB"]["Close"].astype(float)
dxy = dxy[dxy.index <= ASOF]
both = pd.Series(False, index=close.index)
dxy_down = (dxy.pct_change(5, fill_method=None) <= -0.01).reindex(close.index).fillna(False)
both.loc[:] = thrust.values & dxy_down.values
sel2 = pd.DatetimeIndex(both.index[both.values]).intersection(f1.dropna().index)
show([summarize(f1.loc[sel2].values, "thrust WITH a 5d dollar decline >1%"),
      summarize(f1.loc[sel.difference(sel2)].values, "thrust without it")],
     "gold thrust, split on the dollar")
