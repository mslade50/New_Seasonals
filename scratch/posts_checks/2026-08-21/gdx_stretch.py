"""GDX closed tonight +37% over 21 sessions (its own 21d rank = 100th pctile
of the trailing year). What happened after prior +30%/21d months? Declustered
episodes, lag-1 forward returns. Gold's +14%/21d gets the same look."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
import pitch_lab as pl

px = pl.load_prices(["GDX", "GC=F"])

for tkr, thresh in [("GDX", 0.30), ("GC=F", 0.12)]:
    close = px[tkr]["Close"]
    r21 = close.pct_change(21)
    trig_all = r21.index[r21 >= thresh]
    trig = pl.declusters(trig_all, 21, close.index)
    print(f"\n===== {tkr}: 21d return >= {thresh:.0%} "
          f"(raw days={len(trig_all)}, declustered episodes={len(trig)})")
    print("  episodes:", [str(d.date()) for d in trig])
    for h in (5, 10, 21):
        f = pl.fwd_lag(close, h)
        vals = f.reindex(trig).dropna().values
        if not len(vals):
            continue
        wins = int((vals > 0).sum())
        su = pl.summarize(vals)
        print(f"  h={h}: N={su['n']} mean={su['mean_pct']:+.2f}% "
              f"med={su['median_pct']:+.2f}% hit={su['hit']:.0f}% "
              f"sign_p={pl.sign_test(wins, len(vals)):.4f} "
              f"worst={vals.min()*100:+.1f}% best={vals.max()*100:+.1f}%")
    # control: all other days
    f5 = pl.fwd_lag(close, 5)
    ctrl = pl.local_control(close.index, trig)
    cv = f5.reindex(ctrl).dropna().values
    print(f"  local control h=5: mean={cv.mean()*100:+.2f}% hit={(cv>0).mean()*100:.0f}% n={len(cv)}")

# context: where was the equity tape (SPY vs its 252d high) and the trigger
# ticker itself vs its own 252d high, at each GDX episode?
spy = pl.load_prices(["SPY"])["SPY"]["Close"]
spy_dd = spy / spy.rolling(252).max() - 1.0
gdx = px["GDX"]["Close"]
gdx_dd = gdx / gdx.rolling(252).max() - 1.0
r21g = gdx.pct_change(21)
print("\nGDX episode context (SPY off 52w high / GDX off its own 52w high):")
for d in pl.declusters(r21g.index[r21g >= 0.30], 21, gdx.index):
    sd = spy_dd.loc[:d].iloc[-1]
    gd = gdx_dd.loc[:d].iloc[-1]
    print(f"  {d.date()}: SPY {sd*100:+.1f}% off high, GDX {gd*100:+.1f}% off its high")
