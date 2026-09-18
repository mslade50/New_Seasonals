"""Era split and concentration on the post-decision 1% pop cell from 13."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, load_events, sign_test, cluster_note  # noqa: E402

px = close_panel(["^GSPC"])
idx = px.index
g = px["^GSPC"].dropna()
r1 = g / g.shift(1) - 1.0
f5 = g.shift(-5) / g - 1.0
fomc = pd.DatetimeIndex(load_events(["fomc_decision"])["date"])
pos = pd.Series(range(len(idx)), index=idx)
after = pd.DatetimeIndex(
    [idx[pos[d] + 1] for d in fomc if d in pos.index and pos[d] + 1 < len(idx)])
after = after.intersection(r1.dropna().index)
d = after[(r1.reindex(after) >= 0.01).values].intersection(f5.dropna().index)
v = f5.loc[d]
k = int((v > 0).sum())
print(f"post-decision session up 1%+, h5: n={len(v)} mean {100*v.mean():+.3f}% "
      f"median {100*v.median():+.3f}% {k}-{len(v)-k} up "
      f"t {v.mean()/(v.std(ddof=1)/np.sqrt(len(v))):.2f} "
      f"sign p(down) = {sign_test(len(v)-k, len(v)):.4f}")
for lab, m in [("pre-2018", d < pd.Timestamp("2018-01-01")),
               ("2018+", d >= pd.Timestamp("2018-01-01"))]:
    vv = v[m]
    kk = int((vv > 0).sum())
    print(f"  {lab:<10} n={len(vv):<3} mean {100*vv.mean():+.3f}% median "
          f"{100*vv.median():+.3f}%  {kk}-{len(vv)-kk} up")
print(f"  concentration: {cluster_note(d, v.values, k=2)}")
order = np.argsort(-np.abs(v.values))[:2]
keep = np.ones(len(v), dtype=bool); keep[order] = False
w = v[keep]; kw = int((w > 0).sum())
print(f"  drop-two: n={len(w)} mean {100*w.mean():+.3f}% {kw}-{len(w)-kw} up "
      f"(dropped {[str(v.index[o].date()) for o in order]})")
