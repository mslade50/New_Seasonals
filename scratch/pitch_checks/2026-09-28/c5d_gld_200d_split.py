"""C5B: the GLD>200d half that carries the pooled cell -- its dates, and how far
today's GLD sits from its 200d (the near-miss number)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

gp = close_panel(["GLD"]).dropna()
idx = gp.index
g = gp["GLD"]
dxr = pct_rank(close_panel(["DX-Y.NYB"])["DX-Y.NYB"].dropna(), 21).reindex(idx, method="ffill", limit=2)
gdd = 100 * (1 - g / g.rolling(252).max())
sma = g.rolling(200).mean()
gate = (gdd >= 15) & (dxr >= 85)
print(f"LIVE GLD {g.iloc[-1]:.2f}  200d {sma.iloc[-1]:.2f}  gap {100*(g.iloc[-1]/sma.iloc[-1]-1):+.2f}%")
for h in (5, 10):
    ret = fwd_lag(g, h)
    s = idx[(gate & (g > sma)).values & ret.notna().values]
    e = declusters(s, h, idx)
    print(f"h={h} GLD>200d episodes:", ", ".join(f"{d.date()} {100*ret[d]:+.2f}" for d in e))
    s2 = idx[(gate & (g <= sma)).values & ret.notna().values]
    e2 = declusters(s2, h, idx)
    print(f"h={h} GLD<200d years:", pd.Series(ret.loc[e2].values, index=e2.year).groupby(level=0).agg(["count", "sum"]).round(3).to_dict("index"))
