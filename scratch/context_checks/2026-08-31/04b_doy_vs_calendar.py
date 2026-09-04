"""Why the engine's dollar cell and the calendar cell disagree.

Drill 04 restated E:seasonal_doy|DX-Y.NYB ("6 of 6 up in midterm years,
+0.533%, sign p 0.0156") as the calendar fact tomorrow actually is, the first
session of September, and got 3-3 at -0.128%.

Two candidate explanations: (a) I built the wrong cell, or (b) the engine's
trading-day-of-year match lands on dates that are NOT the month boundary, so
the two cells are different populations that merely sound alike.

This prints the engine's own picked anchors so the disagreement is settled by
inspection rather than by argument.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from pitch_lab import load_prices  # noqa: E402
from seasonal_edge import _trading_doy, _window_pick_positions  # noqa: E402

ASOF = pd.Timestamp("2026-08-31")
px = load_prices(["DX-Y.NYB"])["DX-Y.NYB"]
close = px["Close"].dropna().sort_index()

doy = _trading_doy(close.index).values
years = close.index.year.values.astype(np.int64)
target = int(doy[close.index.values <= np.datetime64(ASOF)][-1])
print(f"asof {ASOF.date()} trading doy = {target}")

cv = close.values.astype(np.float64)
fwd = np.full(cv.shape, np.nan)
fwd[:-1] = cv[1:] / cv[:-1] - 1.0

for label, phase in (("all years", None), ("midterm", 2)):
    picks = _window_pick_positions(doy, years, target, ASOF.year, phase, 2, True)
    print(f"\n--- {label}: {len(picks)} picks ---")
    for p in picks:
        d = close.index[p]
        first_of_sep = (d.month == 8 and d == close.index[close.index.month == 8][-1]) \
            if False else None
        nxt = close.index[p + 1] if p + 1 < len(close.index) else None
        boundary = "MONTH BOUNDARY" if (nxt is not None and nxt.month != d.month) else ""
        print(f"  anchor {d.date()} (doy {doy[p]})  h1 {100*fwd[p]:+6.2f}%  "
              f"next bar {nxt.date() if nxt is not None else 'n/a'}  {boundary}")
    r = fwd[picks]
    r = r[~np.isnan(r)]
    print(f"  n={len(r)}  {int((r>0).sum())}-{int((r<0).sum())} up  mean {100*r.mean():+.3f}%")
