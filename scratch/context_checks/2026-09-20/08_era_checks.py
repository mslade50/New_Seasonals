"""Era stability for the two cells that could carry a [solid] tag:
the generic Monday VIX seam, and the MOVE mean-reversion leg.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, fwd_ret, declusters, summarize, sign_test  # noqa

px = close_panel(["^VIX", "^MOVE", "SPY", "^GSPC"]).dropna(subset=["^VIX"])
px = px[px.index >= "1999-01-01"]
vix = px["^VIX"]
vret = vix / vix.shift(1) - 1.0
mon = px.index.dayofweek == 0
sep = px.index.month == 9


def line(v, lbl):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    up = int((v > 0).sum())
    t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
    print(f"  {lbl:28} n {len(v):5}  mean {100*v.mean():+6.2f}%  med {100*np.median(v):+6.2f}%  "
          f"rec {up}-{len(v)-up}  signp {sign_test(max(up, len(v)-up), len(v)):.4f}  t {t:+.2f}")


print("=== the generic Monday VIX seam, by era (Mondays outside September) ===")
m = mon & ~sep
d = vret[m].dropna()
for cut in ("2008-01-01", "2013-01-01", "2018-01-01", "2022-01-01"):
    k = d.index < pd.Timestamp(cut)
    line(d.values[k], f"pre-{cut[:4]}")
    line(d.values[~k], f"{cut[:4]}+")
    print()
print("  every 5-year block:")
for lo in range(1999, 2026, 5):
    k = (d.index.year >= lo) & (d.index.year < lo + 5)
    if k.sum():
        line(d.values[k], f"{lo}-{min(lo+4, 2026)}")

print("\n=== all Mondays (incl. September) vs all non-Mondays, by era ===")
for cut in ("2013-01-01", "2018-01-01"):
    for lbl, mm in (("Mondays", mon), ("non-Mondays", ~mon)):
        dd = vret[mm].dropna()
        k = dd.index < pd.Timestamp(cut)
        line(dd.values[k], f"{lbl} pre-{cut[:4]}")
        line(dd.values[~k], f"{lbl} {cut[:4]}+")
    print()

print("=== MOVE mean reversion, by era ===")
p2 = px.dropna(subset=["^MOVE", "^VIX", "SPY"])
move = p2["^MOVE"]
move21 = move / move.shift(21) - 1.0
vv = p2["^VIX"] / p2["^VIX"].rolling(200).mean() - 1.0
mask = (move21 >= 0.10) & (vv <= -0.10)
trig = p2.index[mask.reindex(p2.index).fillna(False)]
dec = declusters(trig, 21, p2.index)
for h in (5, 21):
    f = fwd_ret(p2["^MOVE"], h)
    v = f.reindex(dec).dropna()
    print(f"  h{h}:")
    for cut in ("2013-01-01", "2018-01-01"):
        k = v.index < pd.Timestamp(cut)
        line(v.values[k], f"    pre-{cut[:4]}")
        line(v.values[~k], f"    {cut[:4]}+")
    print("   every 5-year block:")
    for lo in range(2003, 2027, 5):
        k = (v.index.year >= lo) & (v.index.year < lo + 5)
        if k.sum() >= 3:
            line(v.values[k], f"    {lo}-{min(lo+4, 2026)}")
    print()
