"""E0: recompute today's (2026-08-06 close) state from the real cache.

A trigger that does not fire on the freshest bar kills that candidate for today.
Checks every condition used by E1-E4 independently.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa: F401,F403

TICKS = ["SPY", "QQQ", "^VIX", "AAPL", "MSFT", "XLU", "XLV"]
px = load_prices(TICKS)

print("=== cache coverage ===")
for t in TICKS:
    s = px[t]["Close"]
    print(f"{t:6s} first={s.index[0].date()} last={s.index[-1].date()} n={len(s)} last_close={s.iloc[-1]:.2f}")

spy = px["SPY"]["Close"]
vix = px["^VIX"]["Close"]
aapl = px["AAPL"]["Close"]
msft = px["MSFT"]["Close"]

D = spy.index[-1]
print(f"\nfreshest SPY bar: {D.date()}  weekday={D.day_name()}  (entry day = next session)")

# next session on a US federal business calendar
try:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay
    BD = CustomBusinessDay(calendar=USFederalHolidayCalendar())
except Exception:
    BD = pd.tseries.offsets.BDay()
nxt = D + BD
print(f"entry session (D+1 bday) = {nxt.date()} {nxt.day_name()}")


def state(name, s, vs_ref=None):
    r5 = s.pct_change(5)
    r21 = s.pct_change(21)
    rk5 = r5.rolling(252).rank(pct=True) * 100
    rk21 = r21.rolling(252).rank(pct=True) * 100
    rk63 = s.pct_change(63).rolling(252).rank(pct=True) * 100
    hi52 = s.rolling(252).max()
    dist = (s / hi52 - 1) * 100
    z = zscore(s, 10)
    print(f"\n--- {name} ---")
    print(f"  close        {s.iloc[-1]:.2f}")
    print(f"  ret_1d       {100*s.pct_change(1).iloc[-1]:+.2f}%")
    print(f"  ret_5d       {100*r5.iloc[-1]:+.2f}%")
    print(f"  ret_21d      {100*r21.iloc[-1]:+.2f}%")
    print(f"  rank_5d      {rk5.iloc[-1]:.1f}")
    print(f"  rank_21d     {rk21.iloc[-1]:.1f}")
    print(f"  rank_63d     {rk63.iloc[-1]:.1f}")
    print(f"  z10          {z.iloc[-1]:+.2f}")
    print(f"  vs 52w high  {dist.iloc[-1]:+.2f}%")
    return dict(rk5=rk5.iloc[-1], rk21=rk21.iloc[-1], dist=dist.iloc[-1], z=z.iloc[-1],
                close=s.iloc[-1])


sS = state("SPY", spy)
sV = state("^VIX", vix)
sA = state("AAPL", aapl)
sM = state("MSFT", msft)

print("\n=== TRIGGER FIRE CHECK on 2026-08-06 close ===")

# E1: SPY rk5 >= 95 AND within 0.5% of 52w high AND VIX rk5 <= 25
c1a = sS["rk5"] >= 95
c1b = sS["dist"] >= -0.5
c1c = sV["rk5"] <= 25
print(f"E1  SPY rk5>=95 : {c1a}  ({sS['rk5']:.1f})")
print(f"E1  SPY <=0.5% of 52wh : {c1b}  ({sS['dist']:+.2f}%)")
print(f"E1  VIX rk5<=25 : {c1c}  ({sV['rk5']:.1f})")
print(f"E1  FIRES TODAY: {bool(c1a and c1b and c1c)}")

# E2: AAPL rank5 <= 5 (cross-sectional in the brief, but we test time-series rank too)
print(f"\nE2  AAPL time-series rk5 = {sA['rk5']:.1f} (ret5 {100*aapl.pct_change(5).iloc[-1]:+.2f}%)")
print(f"E2  SPY within 1% of 52wh : {sS['dist'] >= -1.0} ({sS['dist']:+.2f}%)")
print(f"E2  MSFT time-series rk5 = {sM['rk5']:.1f}")

# cross-sectional rank of AAPL 5d return inside the cached universe (brief says 3.17 of 217)
mp = pd.read_parquet(PRICES)
mp["date"] = pd.to_datetime(mp["date"])
last = mp[mp["date"] == D]
piv = mp.pivot_table(index="date", columns="ticker", values="Close")
piv = piv.loc[:D]
liq = piv.columns[piv.loc[D].notna() & piv.shift(5).loc[D].notna()]
r5x = (piv.loc[D, liq] / piv.shift(5).loc[D, liq] - 1)
xs = r5x.rank(pct=True) * 100
print(f"E2  AAPL cross-sectional 5d-return pctile in cache universe "
      f"({len(liq)} names): {xs.get('AAPL', np.nan):.2f}")
print(f"E2  MSFT cross-sectional: {xs.get('MSFT', np.nan):.2f}")

# E3: VIX < 16 AND VIX rk5 <= 25 AND SPY within 1% of 52wh
c3a = sV["close"] < 16
print(f"\nE3  VIX < 16 : {c3a} ({sV['close']:.2f})")
print(f"E3  VIX rk5 <= 25 : {c1c} ({sV['rk5']:.1f})")
print(f"E3  SPY <=1% of 52wh : {sS['dist'] >= -1.0}")
print(f"E3  FIRES TODAY: {bool(c3a and c1c and sS['dist'] >= -1.0)}")

# E4: entry MOC on a FRIDAY, trigger measured on the entry day itself
print(f"\nE4  entry day {nxt.date()} is {nxt.day_name()} -> Friday? {nxt.dayofweek == 4}")
print(f"E4  NOTE: trigger is evaluated on the ENTRY day's close (Friday 08-07),")
print(f"    which does not exist yet. Using 08-06 as proxy: rk5>=90 {sS['rk5']>=90}, "
      f"dist<=0.5% {sS['dist']>=-0.5}")
