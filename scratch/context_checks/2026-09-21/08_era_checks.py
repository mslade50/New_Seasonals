"""Base-rate sign tests and era splits for the numbers the brief will quote."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices, close_panel, fwd_ret, sign_test  # noqa

print("sign tests against measured base hit rates:")
print(f"  IXIC long-drought h21 15/19 vs 0.621: p {sign_test(15, 19, 0.621):.4f}")
print(f"  IXIC short-drought h1 21/27 vs 0.544: p {sign_test(21, 27, 0.544):.4f}")
print(f"  IXIC all P1 h1 33/47 vs 0.544: p {sign_test(33, 47, 0.544):.4f}")
print(f"  NDX +2% near high h5 22/30 vs 0.571: p {sign_test(22, 30, 0.571):.4f}")
print(f"  GSPC after 8pp spread h21 15/20 vs 0.62: p {sign_test(15, 20, 0.62):.4f}")
print(f"  BTC h5 52/77 vs 0.543: p {sign_test(52, 77, 0.543):.4f}")
print(f"  Labor Day Tuesday VIX 22/26 vs 0.5: p {sign_test(22, 26):.4f}")

# ^IXIC long-drought (>=90d) first highs: era split of h21 and h1
ix = load_prices(["^IXIC"])["^IXIC"]["Close"].astype(float)
hi = ix >= ix.rolling(252, min_periods=252).max()
hd = ix.index[hi.fillna(False).values]
ev, last = [], None
for d in hd:
    if last is not None and (d - last).days >= 90 and d >= pd.Timestamp("1999-01-01"):
        ev.append(d)
    last = d
ev = [d for d in ev if d < ix.index[-1]]
f1, f21 = fwd_ret(ix, 1).reindex(ev), fwd_ret(ix, 21).reindex(ev)
e = pd.DataFrame({"h1": 100 * f1, "h21": 100 * f21})
for lbl, m in (("pre-2018", e.index < "2018-01-01"), ("2018+", e.index >= "2018-01-01")):
    s = e[m]
    print(f"  IXIC drought>=90 {lbl}: n {len(s)} h1 {s.h1.mean():+.2f}% up {int((s.h1>0).sum())}  "
          f"h21 {s.h21.mean():+.2f}% up {int((s.h21>0).sum())}")

# VIX: Labor Day Tuesday era split
px = close_panel(["^VIX"]).dropna()
px = px[px.index >= "1999-01-01"]
vr = px["^VIX"].pct_change()
prev = pd.Series(px.index, index=px.index).shift(1)
ld = vr[(px.index.dayofweek == 1) & (px.index.month == 9) & (prev.dt.dayofweek == 4).values].dropna()
for lbl, m in (("pre-2018", ld.index < "2018-01-01"), ("2018+", ld.index >= "2018-01-01")):
    s = ld[m]
    print(f"  Labor Day Tue VIX {lbl}: n {len(s)} mean {100*s.mean():+.2f}% up {int((s>0).sum())}")
print("  2026 Labor Day Tuesday:", {str(d.date()): round(100 * x, 2) for d, x in ld[ld.index.year == 2026].items()})
