"""D0 -- probe. Verify the two arms independently before spending a check on them.

(1) D1's VIX 21-day RELATIVE-range trailing-252 percentile on the 2026-09-09 bar,
    computed THREE ways: (a) the parked script's own convention (rolling on ^VIX's
    OWN valid bars), (b) ^VIX reindexed to SPY's calendar first (method rule 5),
    (c) ^VIX reindexed AND the trailing-252 rank taken on SPY-calendar days only.
    If these disagree materially the gate is definition-fragile before we start.
(2) The runway from CPI 2026-09-11 to the next scheduled print under the NARROW
    event set {nfp, cpi, ppi, fomc_decision} and under a WIDER set that also
    counts vix_expiry / opex / quad_witching.
(3) D2's ^TNX 252-day maximum touch and 252-session change.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import close_panel, load_events, rolling_on_valid

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

px = close_panel(["SVXY", "^VIX", "^VIX3M", "SPY", "^TNX", "TLT", "IEF"])
print("panel last date:", px.index[-1].date(), " rows:", len(px))
for t in px.columns:
    s = px[t].dropna()
    print(f"  {t:8s} first {s.index[0].date()}  last {s.index[-1].date()}  n={len(s)}")

cal = px["SPY"].dropna().index
vix_raw = px["^VIX"].dropna()
print("\n^VIX bars NOT on SPY's calendar (last 12):",
      [str(d.date()) for d in vix_raw.index.difference(cal)][-12:])
print("count of such bars overall:", len(vix_raw.index.difference(cal)))


def relrange(v):
    rng = (rolling_on_valid(v, lambda x: x.rolling(21).max())
           - rolling_on_valid(v, lambda x: x.rolling(21).min()))
    rel = rolling_on_valid(rng / rolling_on_valid(v, lambda x: x.rolling(21).mean()),
                           lambda x: x.rolling(252).rank(pct=True) * 100)
    return rel


REL_own = relrange(px["^VIX"])                       # parked convention
vix_cal = px["^VIX"].reindex(cal).ffill()            # SPY calendar, ffill holes
REL_cal = relrange(vix_cal)

print("\n(1) rel-range percentile, last 8 SPY sessions")
tail = pd.DataFrame({"vix": px["^VIX"].reindex(cal), "REL_own_cal": REL_own.reindex(cal),
                     "REL_spy_cal": REL_cal.reindex(cal)}).dropna().tail(8)
print(tail.round(3).to_string())

v21 = vix_raw.iloc[-21:]
print(f"\n   raw 21-bar window on ^VIX's own calendar: max {v21.max():.2f} "
      f"min {v21.min():.2f} mean {v21.mean():.4f} rel {(v21.max()-v21.min())/v21.mean():.4f}")
v21c = vix_cal.dropna().iloc[-21:]
print(f"   raw 21-bar window on SPY's calendar      : max {v21c.max():.2f} "
      f"min {v21c.min():.2f} mean {v21c.mean():.4f} rel {(v21c.max()-v21c.min())/v21c.mean():.4f}")

print("\n   ^VIX 5-day change on SPY's calendar: "
      f"{100*(vix_cal.dropna().iloc[-1]/vix_cal.dropna().iloc[-6]-1):+.2f}%   "
      "on ^VIX's own calendar: "
      f"{100*(vix_raw.iloc[-1]/vix_raw.iloc[-6]-1):+.2f}%")

# ---------------------------------------------------------------------------
print("\n(2) RUNWAY under two event-set definitions")
NARROW = ("nfp", "cpi", "ppi", "fomc_decision")
WIDE = NARROW + ("vix_expiry", "opex", "quad_witching")
ev = load_events(None)
print("   event kinds available:", sorted(ev["event"].unique()))
fut = ev[(ev["date"] >= pd.Timestamp("2026-09-08")) & (ev["date"] <= pd.Timestamp("2026-10-10"))]
print(fut.to_string(index=False))

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
fwd = pd.DatetimeIndex(pd.date_range(pd.Timestamp("2026-09-01"), periods=120, freq=bd))
for name, kinds in (("NARROW", NARROW), ("WIDE", WIDE)):
    dates = pd.DatetimeIndex(sorted(ev[ev["event"].isin(kinds)]["date"].unique()))
    d0 = pd.Timestamp("2026-09-11")
    nxt = dates[dates > d0]
    rw = int(fwd.searchsorted(nxt[0]) - fwd.searchsorted(d0))
    print(f"   {name}: CPI 2026-09-11 -> next print {nxt[0].date()} "
          f"({ev[ev['date']==nxt[0]]['event'].tolist()})  runway {rw} td  "
          f"{'QUALIFIES (>=3)' if rw >= 3 else 'DISQUALIFIED'}")

# ---------------------------------------------------------------------------
print("\n(3) D2 arms")
tnx = px["^TNX"].dropna()
hi252 = rolling_on_valid(px["^TNX"], lambda x: x.rolling(252).max())
print(f"   ^TNX last {tnx.iloc[-1]:.4f} on {tnx.index[-1].date()}  "
      f"trailing-252 max {float(hi252.dropna().iloc[-1]):.4f}  "
      f"off-high {100*(tnx.iloc[-1]/float(hi252.dropna().iloc[-1])-1):+.4f}%")
print(f"   ^TNX 252 valid sessions ago = {tnx.iloc[-253]:.4f} on {tnx.index[-253].date()}"
      f"   -> 252-session change {100*(tnx.iloc[-1]-tnx.iloc[-253]):+.1f} bp")
print("   last 8 sessions of (tnx, tnx_252_ago, change_bp, required_close_for_78bp):")
for i in range(-8, 0):
    print(f"     {tnx.index[i].date()}  {tnx.iloc[i]:.4f}  ref {tnx.iloc[i-252]:.4f}  "
          f"chg {100*(tnx.iloc[i]-tnx.iloc[i-252]):+6.1f} bp  need {tnx.iloc[i-252]+0.78:.4f}")
