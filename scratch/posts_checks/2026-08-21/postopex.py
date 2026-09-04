"""Today was August monthly opex. The classic cells for Monday's session and
the week after: SPY post-opex Monday, post-opex week, the August subset, and
the interaction with how the entry week went (yesterday's idea bought the
weak-week subset)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
import pitch_lab as pl

ev = pl.load_events(["opex"])
opex = pd.DatetimeIndex(sorted(ev["date"].unique()))
px = pl.load_prices(["SPY", "^VIX"])
spy = px["SPY"]["Close"]
dates = spy.index
r5_in = spy.pct_change(5)


def cell(anchors, h, label, series=None):
    s = spy if series is None else series
    d2 = s.index
    rows = []
    for d in anchors:
        pos = d2.searchsorted(d)
        if pos >= len(d2):
            continue
        if d2[pos] != d:
            pos -= 1
        if pos + h >= len(d2) or pos < 0:
            continue
        rows.append((d2[pos], s.iloc[pos + h] / s.iloc[pos] - 1.0))
    vals = np.array([r[1] for r in rows])
    if not len(vals):
        print(f"  {label}: empty")
        return None
    wins = int((vals > 0).sum())
    su = pl.summarize(vals, label)
    print(f"  {label}: N={su['n']} mean={su['mean_pct']:+.2f}% "
          f"med={su['median_pct']:+.2f}% hit={su['hit']:.0f}% "
          f"sign_p={pl.sign_test(wins, len(vals)):.4f} "
          f"worst={vals.min()*100:+.1f}%")
    return pd.Series(vals, index=pd.DatetimeIndex([r[0] for r in rows]))


hist = opex[(opex >= dates[0]) & (opex <= dates[-1])]
aug = pd.DatetimeIndex([d for d in hist if d.month == 8])

print("=== SPY from the opex close (lag-0 from opex close = enter opex MOC)")
cell(hist, 1, "next session (opex Monday), all months")
cell(hist, 5, "opex close + 5 td, all months")
cell(aug, 1, "next session, AUGUST opex only")
cell(aug, 5, "opex close + 5 td, AUGUST only")

print("\n=== conditioned on the walk-in week (5d return into opex)")
strong = pd.DatetimeIndex([d for d in hist if d in r5_in.index and r5_in.loc[:d].iloc[-1] > 0])
weak = pd.DatetimeIndex([d for d in hist if d in r5_in.index and r5_in.loc[:d].iloc[-1] <= 0])
cell(strong, 5, f"walk-in week UP (n_anchor={len(strong)})")
cell(weak, 5, f"walk-in week DOWN (n_anchor={len(weak)})")

print("\n=== ^VIX from the opex close")
vix = px["^VIX"]["Close"]
cell(hist, 3, "VIX opex close + 3 td, all", series=vix)
cell(aug, 3, "VIX opex close + 3 td, AUGUST", series=vix)

print("\n=== baseline: every 5td SPY return (overlapping)")
base = spy.pct_change(5).shift(-5).dropna()
print(f"  mean={base.mean()*100:+.2f}% med={base.median()*100:+.2f}% "
      f"hit={(base>0).mean()*100:.0f}%")
