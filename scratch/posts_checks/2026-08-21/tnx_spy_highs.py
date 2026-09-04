"""Tonight the 10y yield closed 0.15% off its 52-week HIGH while SPY sits
1.6% off its own 52-week high. How often do both happen at once, and what
did SPY do next? Declustered, lag-1."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
import pitch_lab as pl

px = pl.load_prices(["SPY", "^TNX"])
spy = px["SPY"]["Close"]
tnx = px["^TNX"]["Close"].reindex(spy.index).ffill()

spy_high = spy.rolling(252).max()
tnx_high = tnx.rolling(252).max()
mask = (tnx / tnx_high - 1 > -0.005) & (spy / spy_high - 1 > -0.02)
trig_all = spy.index[mask.fillna(False)]
trig = pl.declusters(trig_all, 21, spy.index)
print(f"joint state days={len(trig_all)}, declustered episodes={len(trig)}")
print("episodes:", [str(d.date()) for d in trig])
for h in (5, 10, 21):
    f = pl.fwd_lag(spy, h)
    vals = f.reindex(trig).dropna().values
    if not len(vals):
        continue
    wins = int((vals > 0).sum())
    su = pl.summarize(vals)
    print(f"h={h}: N={su['n']} mean={su['mean_pct']:+.2f}% med={su['median_pct']:+.2f}% "
          f"hit={su['hit']:.0f}% sign_p={pl.sign_test(wins, len(vals)):.4f} "
          f"worst={vals.min()*100:+.1f}%")
f5 = pl.fwd_lag(spy, 5)
ctrl = pl.local_control(spy.index, trig)
cv = f5.reindex(ctrl).dropna().values
print(f"local control h=5: mean={cv.mean()*100:+.2f}% hit={(cv>0).mean()*100:.0f}% n={len(cv)}")
