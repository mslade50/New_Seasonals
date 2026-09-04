"""Is the dollar's doy-169 cell a real early-September window or a lottery ticket?

04b settled that the engine's cell is not the month boundary in disguise: the
trading-day-of-year match lands on an actual month-end only 5 times in 26, and
once in 6 midterm years. So E:seasonal_doy|DX-Y.NYB (19-7 up all years,
sign p 0.0145; 6 of 6 in midterms at +0.533%) is a distinct effect.

Distinct is not the same as real. Two tests that a lottery ticket fails:

  1. NEIGHBOURHOOD. If doy 169 works and 165 through 173 do not, the cell is
     one column of a scan and nothing more. A genuine early-September dollar
     window shows up in the neighbours too.
  2. WINDOW. Pool the whole late-Aug-to-mid-Sep band against the rest of the
     year, one observation per year per doy, and see whether the dollar is
     simply firmer in that stretch.

Plus the era split, because 04 found the calendar version was carried entirely
by 2018+.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from pitch_lab import load_prices, sign_test  # noqa: E402
from seasonal_edge import _trading_doy  # noqa: E402

px = load_prices(["DX-Y.NYB"])["DX-Y.NYB"]
close = px["Close"].dropna().sort_index()
close = close[close.index >= "1999-01-01"]
doy = pd.Series(_trading_doy(close.index).values, index=close.index)
yr = pd.Series(close.index.year, index=close.index)
r1 = close.pct_change().shift(-1)  # h=1 from the anchor, lag 0

print("=== 1. neighbourhood: doy 160..180, h=1, all years and midterm ===")
print(" doy   all: n  rec      mean%   sign_p |  midterm: n  rec    mean%")
for d in range(160, 181):
    m = doy == d
    v = r1[m].dropna()
    vy = yr[v.index]
    up, n = int((v > 0).sum()), len(v)
    vm = v[(vy % 4 == 2).values]
    upm, nm = int((vm > 0).sum()), len(vm)
    star = "  <-- engine cell" if d == 169 else ""
    print(f" {d:4d}    {n:3d}  {up:2d}-{n-up:<2d}  {100*v.mean():+7.3f}  {sign_test(up,n):6.4f} | "
          f"    {nm:2d}  {upm}-{nm-upm}  {100*vm.mean():+7.3f}{star}")

print("\n=== 2. window: doy 160-180 pooled vs the rest of the year ===")
win = (doy >= 160) & (doy <= 180)
for label, m in (("doy 160-180", win), ("rest of year", ~win)):
    v = r1[m].dropna()
    up, n = int((v > 0).sum()), len(v)
    print(f"  {label:14s} n={n:4d}  {up}-{n-up} up  mean {100*v.mean():+.4f}%  "
          f"med {100*np.median(v):+.4f}%  sign_p {sign_test(up,n):.4f}")

print("\n=== 3. era split of the engine's own doy-169 cell ===")
v = r1[doy == 169].dropna()
vy = yr[v.index]
for label, m in (("1999-2017", (vy < 2018).values), ("2018+", (vy >= 2018).values)):
    s = v[m]
    up, n = int((s > 0).sum()), len(s)
    print(f"  {label:10s} n={n:2d}  {up}-{n-up} up  mean {100*s.mean():+.3f}%  "
          f"sign_p {sign_test(up,n):.4f}")

print("\n=== 4. how big is a typical dollar day, for scale ===")
allr = close.pct_change().dropna()
print(f"  DXY daily sd {100*allr.std():.3f}%, mean {100*allr.mean():+.4f}%, "
      f"n={len(allr)}")
print(f"  a +0.533% mean is {0.533/(100*allr.std()):.2f} daily sd")
