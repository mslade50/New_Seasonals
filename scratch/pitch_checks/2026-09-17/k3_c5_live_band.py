"""K3 c5 addendum: what does the LIVE band pay (spread rank in [10,25), the
2026-09-16 state at 17.1), and the percentile ladder for the beta-neutral
long-GDX pair, so the not-live verdict carries the number that would arm it
and whether arming would even help (dose response)."""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
BAR = pd.Timestamp("2026-09-16")
px = close_panel(["GDX", "GLD"]).dropna()
px = px[px.index <= BAR]
D = px.index
r1 = px.pct_change()
s5 = px["GDX"].pct_change(5) - px["GLD"].pct_change(5)
rank = s5.rolling(252).rank(pct=True) * 100
beta = r1["GDX"].rolling(252).cov(r1["GLD"]) / r1["GLD"].rolling(252).var()

rows = []
for lo, hi in ((0, 0.41), (0.41, 2), (2, 5), (5, 10), (10, 25), (25, 50),
               (50, 101)):
    m = (rank >= lo) & (rank < hi)
    rec = {"band": f"[{lo},{hi})", "days": int(m.sum())}
    for h in (1, 3, 5, 10):
        ret = fwd_lag(px["GDX"], h) - beta * fwd_lag(px["GLD"], h)
        sig = D[(m & ret.notna()).values]
        e = declusters(sig, max(h, 5), D)
        v = ret.loc[e].values
        rec[f"n{h}"] = len(v)
        rec[f"pct{h}"] = round(100 * v.mean(), 3)
        rec[f"hit{h}"] = round(100 * (v > 0).mean(), 1)
    rows.append(rec)
print(pd.DataFrame(rows).to_string(index=False))
print(f"live rank {rank.iloc[-1]:.2f}; live spread {100*s5.iloc[-1]:+.2f}pp; "
      f"arm (<= trailing-252 min) {100*s5.iloc[-252:-1].min():+.2f}pp; "
      f"rank<=2 needs <= {100*s5.iloc[-252:].quantile(0.02):+.2f}pp")
