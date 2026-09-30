"""kC C3 round 2c: what the trade keys on once the volume gate is removed.
HYG 5d washout (r5 <= 5) signalled k sessions before the month-end, long HYG
lag=1: offset ladder k=1..8, quarter vs other months, and the duration split
(IEF r5 <= 20 = duration-driven flush, today's form, IEF r5 0.4).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
px = close_panel(["HYG", "IEF"])
px = px[px["HYG"].notna()].copy()
idx = px.index
r5 = pct_rank(px["HYG"], 5)
ief5 = pct_rank(px["IEF"], 5)
ym = idx.year * 100 + idx.month
g = pd.Series(1, index=idx).groupby(ym)
to_me = (g.transform("size") - g.cumcount() - 1).astype(int)
to_me[ym == 202609] += 2
isq = pd.Series(np.isin(idx.month, [3, 6, 9, 12]), index=idx)
wash = r5 <= 5


def rec(v):
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "n/a"
    w = int((v > 0).sum())
    return f"{100*v.mean():+.3f} {w}-{len(v)-w} p{sign_test(w, len(v)):.3f}"


rows = []
for h in (3, 5):
    ret = fwd_lag(px["HYG"], h)
    for k in range(1, 9):
        at = to_me == k
        r = {"h": h, "k(ME-k)": k,
             "wash all months": rec(ret[at & wash].values),
             "wash QE": rec(ret[at & wash & isq].values),
             "wash dur(IEF<=20)": rec(ret[at & wash & (ief5 <= 20)].values),
             "wash spread(IEF>20)": rec(ret[at & wash & (ief5 > 20)].values),
             "no-wash all": rec(ret[at & ~wash].values)}
        rows.append(r)
print("=== offset ladder: HYG r5<=5 on the ME-k close, long HYG lag=1 (one obs per month) ===")
print(pd.DataFrame(rows).to_string(index=False))

h = 5
ret = fwd_lag(px["HYG"], h)
m = to_me.isin([1, 2, 3]) & wash
dts = declusters(idx[(m & ret.notna()).values], h, idx)
print("\nME-3..-1 washout episodes (h=5), with IEF r5 and quarter flag:")
print(pd.DataFrame({"to_me": to_me[dts], "QE": isq[dts], "HYG r5": r5[dts].round(1),
                    "IEF r5": ief5[dts].round(1), "h5%": (100 * ret[dts]).round(2)}).to_string())
v = ret.loc[dts].values
dur = (ief5[dts] <= 20).values
print(f"duration-driven: {rec(v[dur])}   spread-driven: {rec(v[~dur])}")
print(f"cluster: {cluster_note(dts, v)}")
