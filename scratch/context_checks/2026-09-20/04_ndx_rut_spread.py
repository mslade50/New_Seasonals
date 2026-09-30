"""Nasdaq flat, Russell and Dow in the 21-day bottom decile.

Friday: ^NDX 21d +0.74% (rank 50), ^RUT 21d -5.69% (rank 6), ^DJI 21d -3.33% (rank 8).
The 21d NDX-RUT spread is +6.4pp. Cell: spread in the top 5% of its trailing year
with NDX itself NOT in a drawdown state, i.e. large growth holding while small breaks.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, fwd_ret, declusters, local_control,  # noqa
                       summarize, show, sign_test, cluster_note, era_split)

T = ["^NDX", "^RUT", "^GSPC", "^DJI", "QQQ", "IWM", "SPY", "^VIX"]
px = close_panel(T).dropna(subset=["^NDX", "^RUT", "^GSPC"])
print("coverage:", px.index.min().date(), "->", px.index.max().date(), "n", len(px))

r21 = lambda s: s / s.shift(21) - 1.0
ndx21, rut21 = r21(px["^NDX"]), r21(px["^RUT"])
spread = ndx21 - rut21
srank = spread.rolling(252).rank(pct=True) * 100

print(f"LIVE: NDX 21d {100*float(ndx21.iloc[-1]):+.2f}%, RUT 21d {100*float(rut21.iloc[-1]):+.2f}%, "
      f"spread {100*float(spread.iloc[-1]):+.2f}pp, rank {float(srank.iloc[-1]):.1f}")

# the live shape: spread extreme AND the Nasdaq leg is the one holding up (ndx21 >= 0)
mask = (srank >= 95) & (ndx21 >= 0)
trig = px.index[mask.reindex(px.index).fillna(False)]
print(f"\nspread rank >= 95 AND NDX 21d >= 0: {len(trig)} raw days")
dec = declusters(trig, 21, px.index)
print(f"declustered: {len(dec)} episodes")
print("  by year:", dict(pd.Series(1, index=dec).groupby(dec.year).sum()))

ctrl = local_control(px.index, trig, 126)
for h in (1, 5, 10, 21):
    rows = []
    for name in ("^GSPC", "^NDX", "^RUT", "^DJI", "^VIX"):
        f = fwd_ret(px[name], h)
        v = f.reindex(dec).dropna().values
        r = summarize(v, f"{name} h{h}")
        up = int((v > 0).sum())
        r["rec"] = f"{up}-{len(v)-up}"
        r["sign_p"] = round(sign_test(max(up, len(v) - up), len(v)), 4)
        cv = f.reindex(ctrl).dropna().values
        r["ctrl_local"] = round(100 * cv.mean(), 3) if len(cv) else np.nan
        rows.append(r)
    # the spread itself forward: does the gap keep widening or close?
    sp = fwd_ret(px["^NDX"], h) - fwd_ret(px["^RUT"], h)
    v = sp.reindex(dec).dropna().values
    r = summarize(v, f"NDX-RUT h{h}")
    up = int((v > 0).sum())
    r["rec"] = f"{up}-{len(v)-up}"
    r["sign_p"] = round(sign_test(max(up, len(v) - up), len(v)), 4)
    cvv = sp.reindex(ctrl).dropna().values
    r["ctrl_local"] = round(100 * cvv.mean(), 3) if len(cvv) else np.nan
    rows.append(r)
    show(rows, f"h={h}")

for h in (5, 21):
    sp = (fwd_ret(px["^NDX"], h) - fwd_ret(px["^RUT"], h)).reindex(dec).dropna()
    print(f"\nNDX-RUT h{h}: {cluster_note(sp.index, sp.values)}")
    for e in era_split(sp.index, sp.values):
        print("   ", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in e.items()
                      if k in ("label", "n", "mean_pct", "hit", "t")})
    v = fwd_ret(px["^GSPC"], h).reindex(dec).dropna()
    print(f"^GSPC h{h} era:")
    for e in era_split(v.index, v.values):
        print("   ", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in e.items()
                      if k in ("label", "n", "mean_pct", "hit", "t")})
