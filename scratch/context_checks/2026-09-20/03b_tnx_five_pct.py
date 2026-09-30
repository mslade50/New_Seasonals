"""The 10-year at 4.998. How rare is a 5% handle, and what has the curve cell done
at a looser threshold where n is interpretable?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, fwd_ret, declusters, local_control,  # noqa
                       summarize, show, sign_test, cluster_note, era_split)

T = ["^TNX", "^FVX", "^IRX", "SPY", "TLT", "IWM", "QQQ", "^VIX"]
px = close_panel(T).dropna(subset=["^TNX", "SPY"])
tnx = px["^TNX"]
print("^TNX coverage:", tnx.index.min().date(), "->", tnx.index.max().date())
print(f"live close {float(tnx.iloc[-1]):.3f}")

ge5 = tnx[tnx >= 5.0]
print(f"\ncloses at or above 5.00: {len(ge5)} of {len(tnx)} sessions")
if len(ge5):
    print("  most recent:", str(ge5.index[-1].date()), round(float(ge5.iloc[-1]), 3))
    print("  by year:", dict(pd.Series(1, index=ge5.index).groupby(ge5.index.year).sum()))
last_ge5 = ge5.index[-1] if len(ge5) else None
if last_ge5 is not None:
    since = int((tnx.index > last_ge5).sum())
    print(f"  sessions since the last 5%+ close: {since}")

# how close has it come since then
post = tnx[tnx.index > last_ge5] if last_ge5 is not None else tnx
print(f"  max close since: {float(post.max()):.3f} on {str(post.idxmax().date())}")
print(f"  is Friday the highest close since? {post.idxmax() == tnx.index[-1]}")

# 52w high on the 10y yield
hi252 = tnx.rolling(252).max()
at_hi = tnx >= hi252 * 0.999
print(f"\n^TNX within 0.1% of its 252d max: live={bool(at_hi.iloc[-1])}, "
      f"dist to 252d max {100*(float(tnx.iloc[-1])/float(hi252.iloc[-1])-1):+.2f}%")


def rank21(s):
    r = s / s.shift(21) - 1.0
    return r.rolling(252).rank(pct=True) * 100


rk = {t: rank21(px[t]) for t in ("^TNX", "^FVX", "^IRX")}
for thr in (90, 85):
    m = (rk["^TNX"] >= thr) & (rk["^FVX"] >= thr) & (rk["^IRX"] >= thr)
    trig = px.index[m.reindex(px.index).fillna(False)]
    dec = declusters(trig, 21, px.index)
    print(f"\n### all three tenors 21d rank >= {thr}: {len(trig)} days, {len(dec)} episodes")
    print("   by year:", dict(pd.Series(1, index=dec).groupby(dec.year).sum()))
    ctrl = local_control(px.index, trig, 126)
    for h in (1, 5, 21):
        rows = []
        for name in ("SPY", "TLT", "^TNX", "IWM", "QQQ"):
            f = fwd_ret(px[name], h)
            v = f.reindex(dec).dropna().values
            r = summarize(v, f"{name} h{h}")
            up = int((v > 0).sum())
            r["rec"] = f"{up}-{len(v)-up}"
            r["sign_p"] = round(sign_test(max(up, len(v) - up), len(v)), 4)
            cv = f.reindex(ctrl).dropna().values
            r["ctrl_local"] = round(100 * cv.mean(), 3) if len(cv) else np.nan
            rows.append(r)
        show(rows, f"thr {thr} h={h}")
    v = fwd_ret(px["SPY"], 5).reindex(dec).dropna()
    if len(v):
        print("   SPY h5", cluster_note(v.index, v.values))
        for e in era_split(v.index, v.values):
            print("     ", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in e.items()
                            if k in ("label", "n", "mean_pct", "hit", "t")})
