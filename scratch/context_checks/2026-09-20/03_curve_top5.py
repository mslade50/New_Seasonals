"""The whole Treasury curve at a 21-day top-5% rise at once.

Friday: ^IRX 21d rank 100 and AT a 52-week high (z10 3.2), ^FVX rank 98,
^TNX rank 97 and 0.16% off its own 52w high. The engine scored ^TNX alone
(n 398, h1 -0.14%, sign p 0.049). Condition on all three tenors together.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, fwd_ret, declusters, local_control,  # noqa
                       summarize, show, sign_test, cluster_note, era_split)

T = ["^TNX", "^FVX", "^IRX", "SPY", "TLT", "IWM", "QQQ", "^VIX"]
px = close_panel(T).dropna(subset=["^TNX", "^FVX", "^IRX", "SPY"])
print("coverage:", px.index.min().date(), "->", px.index.max().date(), "n", len(px))


def rank21(s):
    r = s / s.shift(21) - 1.0
    return r.rolling(252).rank(pct=True) * 100


rk = {t: rank21(px[t]) for t in ("^TNX", "^FVX", "^IRX")}
for t, r in rk.items():
    print(f"LIVE {t} 21d rank {float(r.iloc[-1]):.1f}  level {float(px[t].iloc[-1]):.3f}")

allthree = (rk["^TNX"] >= 95) & (rk["^FVX"] >= 95) & (rk["^IRX"] >= 95)
trig = px.index[allthree.reindex(px.index).fillna(False)]
print(f"\nall three tenors at 21d rank >= 95: {len(trig)} raw days")
dec = declusters(trig, 21, px.index)
print(f"declustered (21td): {len(dec)} episodes")
print("  by year:", dict(pd.Series(1, index=dec).groupby(dec.year).sum()))

ctrl = local_control(px.index, trig, 126)
for h in (1, 5, 10, 21):
    rows = []
    for name in ("SPY", "TLT", "^TNX", "IWM", "QQQ", "^VIX"):
        f = fwd_ret(px[name], h)
        v = f.reindex(dec).dropna().values
        r = summarize(v, f"{name} h{h}")
        up = int((v > 0).sum())
        r["rec"] = f"{up}-{len(v)-up}"
        r["sign_p"] = round(sign_test(max(up, len(v) - up), len(v)), 4)
        cv = f.reindex(ctrl).dropna().values
        r["ctrl_local"] = round(100 * cv.mean(), 3) if len(cv) else np.nan
        r["ctrl_all"] = round(100 * f.dropna().values.mean(), 3)
        rows.append(r)
    show(rows, f"h={h}")

print("\n--- does the curve rise keep going? ^TNX forward by episode, h5 ---")
t5 = fwd_ret(px["^TNX"], 5).reindex(dec).dropna()
s5 = fwd_ret(px["SPY"], 5).reindex(dec).dropna()
print(pd.DataFrame({"date": [str(d.date()) for d in t5.index],
                    "tnx_h5_pct": (100 * t5.values).round(1),
                    "spy_h5_pct": (100 * s5.reindex(t5.index).values).round(2)}).to_string(index=False))

for h in (5, 21):
    v = fwd_ret(px["^TNX"], h).reindex(dec).dropna()
    print(f"\n^TNX h{h}: {cluster_note(v.index, v.values)}")
    for e in era_split(v.index, v.values):
        print("   ", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in e.items()
                      if k in ("label", "n", "mean_pct", "hit", "t")})

print("\n--- the seasonal control the engine flagged: ^TNX h5 from the Sep-21 analogue ---")
print("engine: n 26, mean -1.16%, 8 up 18 down, sign p 0.0378 (all years)")
