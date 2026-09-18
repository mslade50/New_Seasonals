"""c0 — cache probe: which vehicles exist, spans, and today's readings.

Answers, before any cell is built:
  - is NG=F in the cache (needed to price UNG's roll drag)?
  - what natgas-adjacent vehicles exist (BOIL, KOLD, UNL, FCG, UGA, USO, CL=F)?
  - the exact ^VIX 21d relative-range trailing-252 percentile as of 2026-09-09
  - ^VIX calendar vs SPY calendar mismatch (repo rule 8)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

CAND = ["SPY", "^VIX", "^VIX3M", "SVXY", "UVXY", "VXX", "^SKEW",
        "UNG", "UNL", "BOIL", "KOLD", "FCG", "XOP", "XLE",
        "NG=F", "CL=F", "USO", "UGA", "DBC", "QQQ", "IWM"]

mp = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "master_prices.parquet")
have = set(mp["ticker"].unique())
print("== cache membership ==")
for t in CAND:
    if t in have:
        g = mp[mp["ticker"] == t]
        d = pd.to_datetime(g["date"])
        print(f"  {t:8s} PRESENT  {d.min().date()} .. {d.max().date()}  n={len(g)}")
    else:
        print(f"  {t:8s} ABSENT")

print("\n== any ticker containing NG / gas ==")
print(sorted([t for t in have if "NG" in t.upper() or "GAS" in t.upper()])[:60])
print("\n== any '=F' futures tickers in cache ==")
print(sorted([t for t in have if "=F" in t])[:80])

# ---------------------------------------------------------------- VIX compression
px = close_panel(["SPY", "^VIX"])
vix_raw = px["^VIX"].dropna()
spy = px["SPY"].dropna()
print(f"\n== calendar check ==")
print(f"  SPY sessions {len(spy)}  ^VIX sessions {len(vix_raw)}")
extra = vix_raw.index.difference(spy.index)
print(f"  ^VIX bars SPY lacks: {len(extra)}  most recent: "
      f"{[str(d.date()) for d in extra[-8:]]}")
vix = vix_raw.reindex(spy.index)  # rule 8: reindex to SPY's calendar
print(f"  after reindex, ^VIX NaNs on SPY calendar: {int(vix.isna().sum())}")

# relative range on SPY calendar
rr = rolling_on_valid(vix, lambda x: (x.rolling(21).max() - x.rolling(21).min())
                      / x.rolling(21).mean())
rr_pct = rolling_on_valid(rr, lambda x: x.rolling(252).rank(pct=True) * 100.0)
print("\n== ^VIX 21d RELATIVE range percentile (trailing 252), SPY calendar ==")
print(rr.dropna().tail(8).round(4).to_string())
print(rr_pct.dropna().tail(8).round(3).to_string())
last = rr_pct.dropna().index[-1]
print(f"  ** as of {last.date()}: rel-range = {rr.loc[last]:.4f}, "
      f"trailing-252 pctile = {rr_pct.loc[last]:.3f} **")
print(f"  ^VIX close {vix.loc[last]:.2f}   SPY close {spy.loc[last]:.2f}")

# 5d ^VIX change on SPY's calendar (task asked to re-verify)
v5 = vix.dropna()
print(f"  ^VIX 1d {100*(v5.iloc[-1]/v5.iloc[-2]-1):+.2f}%  "
      f"5d {100*(v5.iloc[-1]/v5.iloc[-6]-1):+.2f}% (SPY calendar)")
vr = vix_raw
print(f"  ^VIX 5d on its OWN calendar {100*(vr.iloc[-1]/vr.iloc[-6]-1):+.2f}%")

# UNG readings
if "UNG" in have:
    ung = close_panel(["UNG"])["UNG"].dropna()
    lo252 = rolling_on_valid(ung, lambda x: x.rolling(252).min())
    sma200 = rolling_on_valid(ung, lambda x: x.rolling(200).mean())
    l = ung.index[-1]
    print(f"\n== UNG {l.date()} close {ung.iloc[-1]:.2f} "
          f"above252low {100*(ung.iloc[-1]/lo252.iloc[-1]-1):+.2f}%  "
          f"vs200sma {100*(ung.iloc[-1]/sma200.iloc[-1]-1):+.2f}%")
