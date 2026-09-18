"""^TNX closing AT its trailing-252 max: what the next sessions did.

Live state 2026-09-08: ^TNX 4.806 == 252d max, ^FVX also at its max, ^IRX 5d rank 92.5.
No P1 trigger fired because P1 wants the FIRST new high in 30+ days and yields have
been grinding up; the LEVEL fact is still the most relevant thing about tomorrow.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (close_panel, rolling_on_valid, fwd_ret, declusters,
                       local_control, summarize, era_split, sign_test,
                       cluster_note, show)

TK = ["^TNX", "SPY", "^GSPC", "TLT", "IEF", "^VIX", "DX-Y.NYB", "GC=F"]
px = close_panel(TK)
print("panel", px.index.min().date(), "->", px.index.max().date(), len(px), "rows")

tnx = px["^TNX"]
mx = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
at_high = (tnx >= mx * 0.9999) & tnx.notna() & mx.notna()
dates = px.index[at_high.fillna(False)]
print(f"\n^TNX at a 252d max: {len(dates)} sessions, "
      f"{dates.min().date()} -> {dates.max().date()}")
print("by year:", dict(pd.Series(1, index=dates).groupby(dates.year).sum()))

# today must be in the set or the mask is wrong
print("2026-09-08 in set:", pd.Timestamp("2026-09-08") in dates)

valid = tnx.dropna().index
epi = declusters(dates, 10, valid)
print(f"declustered at 10td: {len(epi)} episodes")

print("\n--- forward returns from an at-the-high close (lag=0, close-to-close) ---")
for sub in ["^GSPC", "SPY", "TLT", "IEF", "^TNX", "^VIX", "GC=F", "DX-Y.NYB"]:
    rows = []
    s = px[sub]
    for h in (1, 5, 10, 21):
        f = fwd_ret(s, h)
        d = pd.DatetimeIndex(dates).intersection(f.dropna().index)
        e = declusters(d, max(h, 5), f.dropna().index)
        r = summarize(f.loc[e].values, f"h={h}")
        if r["n"]:
            base = f.dropna()
            r["ctl_all_pct"] = round(100 * base.mean(), 3)
            r["edge_pct"] = round(r["mean_pct"] - 100 * base.mean(), 3)
            up = int((f.loc[e].values > 0).sum())
            r["record"] = f"{up}-{r['n']-up}"
            r["sign_p"] = round(sign_test(up, r["n"]), 4)
        rows.append(r)
    show(rows, f"{sub} after ^TNX at a 252d max")

print("\n--- the equity cell in detail: ^GSPC h=1 ---")
f1 = fwd_ret(px["^GSPC"], 1)
d1 = pd.DatetimeIndex(dates).intersection(f1.dropna().index)
v1 = f1.loc[d1].values
r = summarize(v1, "all at-high days h1")
up = int((v1 > 0).sum())
print(r)
print(f"record {up}-{len(v1)-up}  sign_p={sign_test(up, len(v1)):.4f}")
show(era_split(d1, v1), "^GSPC h1 era split")
print("cluster:", cluster_note(d1, v1, k=2))
ctl = local_control(f1.dropna().index, d1, 126)
print("local +/-126td control:", summarize(f1.loc[ctl].values, "local"))

print("\n--- and TLT h=5, the duration question ---")
f5 = fwd_ret(px["TLT"], 5)
d5 = pd.DatetimeIndex(dates).intersection(f5.dropna().index)
e5 = declusters(d5, 5, f5.dropna().index)
v5 = f5.loc[e5].values
r = summarize(v5, "TLT h5")
up = int((v5 > 0).sum())
print(r, f"record {up}-{len(v5)-up} sign_p={sign_test(up, len(v5)):.4f}")
show(era_split(e5, v5), "TLT h5 era split")
print("cluster:", cluster_note(e5, v5, k=2))
ctlt = local_control(f5.dropna().index, d5, 126)
print("local control:", summarize(f5.loc[ctlt].values, "local"))
