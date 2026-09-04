"""How the Jackson Hole cells published this week resolved on the 08-28 bar.

The brief (08-25 / 08-27) and the queue (08-26 / 08-27) carried:
  IWM symposium session 21-26 up, +0.66%
  HG=F symposium session 19-25 up, +0.82%, midterm 6-6
  ^VIX symposium session 21-26 DOWN, -2.61%
  ^VIX3M 52w low on the eve x SPY 21d rank top decile: VIX higher next day 5-5
  SPY three sessions from the 08-25 close into the symposium close: 17-26, +0.58%
Plus the declined 08-27 pitch: GDX long MOC Thu 08-27, 6-0 cell, +5.75% h=5.

This script just prints the realized moves so the resolution post carries
exact numbers off the cache, not from memory.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_prices  # noqa: E402

px = load_prices(["IWM", "SPY", "QQQ", "HG=F", "^VIX", "^VIX3M", "GDX", "GC=F", "IEF", "TLT", "DX-Y.NYB"])
for t, f in px.items():
    c = f["Close"].dropna()
    last = c.index[-1]
    r1 = 100 * (c.iloc[-1] / c.iloc[-2] - 1)
    print(f"{t:<9} bar {last.date()}  close {c.iloc[-1]:.3f}  1d {r1:+.2f}%  "
          f"O/H/L {f['Open'].iloc[-1]:.3f}/{f['High'].iloc[-1]:.3f}/{f['Low'].iloc[-1]:.3f}")

spy = px["SPY"]["Close"].dropna()
c0825 = spy.loc["2026-08-25"]
print(f"\nSPY 08-25 close {c0825:.2f} -> 08-28 close {spy.iloc[-1]:.2f}: {100*(spy.iloc[-1]/c0825-1):+.2f}% (the 3-session cell, 17-26 +0.58%)")
iwm = px["IWM"]["Close"].dropna()
print(f"IWM 08-25 -> 08-28: {100*(iwm.iloc[-1]/iwm.loc['2026-08-25']-1):+.2f}%")

gdx = px["GDX"]["Close"].dropna()
print(f"\nGDX Thu 08-27 close {gdx.loc['2026-08-27']:.2f} -> Fri {gdx.iloc[-1]:.2f}: "
      f"{100*(gdx.iloc[-1]/gdx.loc['2026-08-27']-1):+.2f}%  (pitch entry MOC Thu, exit MOC Thu 09-03)")
print(f"GDX 21d return into Thu 08-27: {100*(gdx.loc['2026-08-27']/gdx.iloc[-23]-1):+.1f}%")

ief = px["IEF"]["Close"].dropna()
print(f"\nIEF (open queue idea x20260826-1): entry Thu 08-27 close {ief.loc['2026-08-27']:.3f}, "
      f"Fri {ief.iloc[-1]:.3f} = {100*(ief.iloc[-1]/ief.loc['2026-08-27']-1):+.2f}% so far, exit Mon MOC")

vix = px["^VIX"]["Close"].dropna()
print(f"\nVIX 08-27 {vix.loc['2026-08-27']:.2f} -> 08-28 {vix.iloc[-1]:.2f}  {100*(vix.iloc[-1]/vix.loc['2026-08-27']-1):+.2f}%")
v3 = px["^VIX3M"]["Close"].dropna()
print(f"VIX3M 08-27 {v3.loc['2026-08-27']:.2f} -> {v3.iloc[-1]:.2f}  {100*(v3.iloc[-1]/v3.loc['2026-08-27']-1):+.2f}%")
