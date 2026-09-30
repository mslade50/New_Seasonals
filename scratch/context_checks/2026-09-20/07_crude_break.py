"""Crude's -6.32% session, which the sweep missed on a threshold.

01_roll_seams cleared it: gap -0.83%, intraday -5.53%, range 3.15%, so the move is real.
Context: CL=F closed 95.47, still +17.7% over its 200d SMA and +8.7% over 21 days, and
it also fell 3.21% on Wednesday. Cell: a 5%+ down session while above the 200d SMA.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, load_prices, fwd_ret, declusters,  # noqa
                       local_control, summarize, show, sign_test, cluster_note, era_split)

T = ["CL=F", "SPY", "^GSPC", "^VIX", "XLE" ]
px = close_panel(T).dropna(subset=["CL=F", "SPY"])
print("coverage:", px.index.min().date(), "->", px.index.max().date(), "n", len(px))
cl = px["CL=F"]
ret1 = cl / cl.shift(1) - 1.0
sma200 = cl.rolling(200).mean()
above = cl / sma200 - 1.0
print(f"LIVE: close {float(cl.iloc[-1]):.2f}, session {100*float(ret1.iloc[-1]):+.2f}%, "
      f"vs 200d {100*float(above.iloc[-1]):+.1f}%, 21d {100*(float(cl.iloc[-1]/cl.iloc[-22])-1):+.1f}%")

# bare cell: a 5%+ down day in crude
for thr, need_above in ((-0.05, False), (-0.05, True), (-0.06, True)):
    m = (ret1 <= thr)
    if need_above:
        m = m & (above > 0.05)
    trig = px.index[m.reindex(px.index).fillna(False)]
    dec = declusters(trig, 10, px.index)
    lbl = f"{100*thr:.0f}% day" + (" while >5% over its 200d" if need_above else "")
    print(f"\n### crude {lbl}: {len(trig)} days, {len(dec)} episodes")
    if len(dec) < 5:
        print("  too thin")
        continue
    print("  by year:", dict(pd.Series(1, index=dec).groupby(dec.year).sum()))
    ctrl = local_control(px.index, trig, 126)
    for h in (1, 5, 21):
        rows = []
        for name in ("CL=F", "SPY", "^VIX"):
            f = fwd_ret(px[name], h)
            v = f.reindex(dec).dropna().values
            if not len(v):
                continue
            r = summarize(v, f"{name} h{h}")
            up = int((v > 0).sum())
            r["rec"] = f"{up}-{len(v)-up}"
            r["sign_p"] = round(sign_test(max(up, len(v) - up), len(v)), 4)
            cv = f.reindex(ctrl).dropna().values
            r["ctrl_local"] = round(100 * cv.mean(), 3) if len(cv) else np.nan
            r["ctrl_all"] = round(100 * f.dropna().values.mean(), 3)
            rows.append(r)
        show(rows, f"{lbl} h={h}")
    v = fwd_ret(px["CL=F"], 5).reindex(dec).dropna()
    print("  CL=F h5", cluster_note(v.index, v.values))
    for e in era_split(v.index, v.values):
        print("     ", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in e.items()
                        if k in ("label", "n", "mean_pct", "hit", "t")})

print("\n### how unusual is a 6%+ down day for crude at all? ###")
big = ret1[ret1 <= -0.06].dropna()
print(f"  {len(big)} sessions of {len(ret1.dropna())} ({100*len(big)/len(ret1.dropna()):.2f}%)")
print("  by year:", dict(pd.Series(1, index=big.index).groupby(big.index.year).sum()))
print("  most recent five:", [(str(d.date()), round(100 * float(v), 1)) for d, v in big.iloc[-5:].items()])

print("\n### the two-day shape: crude -3% Wed and -6% Fri, i.e. -4.6% over 5 sessions ###")
r5 = cl / cl.shift(5) - 1.0
m = (r5 <= -0.045) & (above > 0.10)
trig = px.index[m.reindex(px.index).fillna(False)]
dec = declusters(trig, 10, px.index)
print(f"  5d <= -4.5% while >10% over the 200d: {len(trig)} days, {len(dec)} episodes")
if len(dec) >= 5:
    print("  by year:", dict(pd.Series(1, index=dec).groupby(dec.year).sum()))
    ctrl = local_control(px.index, trig, 126)
    for h in (5, 21):
        rows = []
        for name in ("CL=F", "SPY"):
            f = fwd_ret(px[name], h)
            v = f.reindex(dec).dropna().values
            r = summarize(v, f"{name} h{h}")
            up = int((v > 0).sum())
            r["rec"] = f"{up}-{len(v)-up}"
            r["sign_p"] = round(sign_test(max(up, len(v) - up), len(v)), 4)
            cv = f.reindex(ctrl).dropna().values
            r["ctrl_local"] = round(100 * cv.mean(), 3) if len(cv) else np.nan
            rows.append(r)
        show(rows, f"5d break h={h}")
