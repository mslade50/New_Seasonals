"""D0 - verify every trigger/calendar claim in the D1-D4 candidate specs against real data.

Recomputes: dial ma10, 52w distance, opex/vix-expiry rows, trading-day counts on a
US-federal-holiday business calendar (NOT naive weekdays), August trading-session index.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa: F401,F403

BD = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def td_between(a, b) -> int:
    """Trading sessions from a to b exclusive-of-a, holiday aware."""
    rng = pd.date_range(pd.Timestamp(a) + BD, pd.Timestamp(b), freq=BD)
    return len(rng)


px = load_prices(["SPY", "IWM"])
spy, iwm = px["SPY"], px["IWM"]
spy = spy[spy.index <= "2026-08-06"]
iwm = iwm[iwm.index <= "2026-08-06"]

frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
frag.index = pd.to_datetime(frag.index)
ma10 = frag["63d"].rolling(10).mean()

print("=== dial ===")
print(f"last dial date        {frag.index[-1].date()}")
print(f"raw 21d               {frag['21d'].iloc[-1]:.1f}   (brief says 69.7)")
print(f"raw 63d               {frag['63d'].iloc[-1]:.1f}   (brief says 76.0)")
print(f"ma10(63d) today       {ma10.iloc[-1]:.2f}  (brief says 57.2)")
print(f"ma10(63d) 21 sess ago {ma10.iloc[-22]:.2f}  (brief says 20.5)")
print(f"min ma10 trailing 21  {ma10.iloc[-21:].min():.2f}")
print(f"D1 trigger (ma10>=50 and min21<30): {bool(ma10.iloc[-1] >= 50 and ma10.iloc[-21:].min() < 30)}")

print("\n=== SPY / IWM state ===")
hi52 = spy["Close"].rolling(252).max()
d52 = spy["Close"].iloc[-1] / hi52.iloc[-1] - 1
print(f"SPY close {spy['Close'].iloc[-1]:.2f}  52w-high dist {100*d52:+.2f}%  (brief -0.36%)")
print(f"D1 proximity gate (<=1.5% below high): {bool(d52 >= -0.015)}")
hi52i = iwm["Close"].rolling(252).max()
print(f"IWM close {iwm['Close'].iloc[-1]:.2f}  52w-high dist "
      f"{100*(iwm['Close'].iloc[-1]/hi52i.iloc[-1]-1):+.2f}%  (brief -1.15%)")
print(f"SPY ret_5d {100*(spy['Close'].iloc[-1]/spy['Close'].iloc[-6]-1):+.2f}%  (brief +3.62%)")

print("\n=== calendar rows in macro_events ===")
ev = load_events()
aug = ev[(ev["date"] >= "2026-08-01") & (ev["date"] <= "2026-08-31")]
print(aug[["date", "event"]].to_string(index=False))
opex26 = set(ev[(ev.event == "opex") & (ev.date.dt.year == 2026)].date)
vix26 = set(ev[(ev.event == "vix_expiry") & (ev.date.dt.year == 2026)].date)
print(f"2026-08-21 is an opex row      : {pd.Timestamp('2026-08-21') in opex26}")
print(f"2026-08-19 is a vix_expiry row : {pd.Timestamp('2026-08-19') in vix26}")
print(f"2026-08-07 is an nfp row       : "
      f"{pd.Timestamp('2026-08-07') in set(ev[ev.event=='nfp'].date)}")

print("\n=== trading-day arithmetic (US federal holiday calendar) ===")
print(f"td 2026-08-07 close -> 2026-08-21 close : {td_between('2026-08-07','2026-08-21')} "
      f"(D2 claims 10)")
print(f"td 2026-08-06 close -> 2026-08-21 close : {td_between('2026-08-06','2026-08-21')}")
print(f"td 2026-08-14 close -> 2026-08-19 close : {td_between('2026-08-14','2026-08-19')} "
      f"(D4 claims 3)")
naive = np.busday_count(np.datetime64("2026-08-07") + 1, np.datetime64("2026-08-22"))
print(f"naive weekday count 08-07 -> 08-21      : {naive} (no holidays in window, so equal)")

# August trading-session index of 2026-08-07
aug_sessions = pd.date_range("2026-08-01", "2026-08-31", freq=BD)
pos = list(aug_sessions).index(pd.Timestamp("2026-08-07")) + 1
print(f"\n2026-08-07 is August trading session #{pos}  <-- D3 spec says '7th session'")
print(f"  August 2026 sessions 1-8: {[d.strftime('%m-%d') for d in aug_sessions[:8]]}")
print(f"  the 7th session of Aug 2026 is {aug_sessions[6].date()} (a {aug_sessions[6].strftime('%A')})")

print("\n=== CPI/PPI inside the candidate holds ===")
for name, a, b in [("D2 08-07->08-21", "2026-08-07", "2026-08-21"),
                   ("D3 h10 from 08-07", "2026-08-07", pd.Timestamp("2026-08-07") + 10 * BD),
                   ("D3 h21 from 08-07", "2026-08-07", pd.Timestamp("2026-08-07") + 21 * BD),
                   ("D4 08-14->08-19", "2026-08-14", "2026-08-19"),
                   ("D1 h10 from 08-07", "2026-08-07", pd.Timestamp("2026-08-07") + 10 * BD)]:
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    inside = ev[(ev.date > a) & (ev.date <= b) & (ev.event.isin(["cpi", "ppi", "fomc_decision", "jackson_hole"]))]
    tags = ", ".join(f"{r.event}@{r.date.date()}" for r in inside.itertuples()) or "none"
    print(f"  {name:22s} {a.date()} -> {b.date()}: {tags}")
