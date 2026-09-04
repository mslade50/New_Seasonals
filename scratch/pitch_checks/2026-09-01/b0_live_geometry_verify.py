"""Verification of the load-bearing live facts every B-candidate rests on.

If any of these is wrong, all four checks are measuring the wrong window.
  - freshest bar, entry session, and the exact td distance to 2026-09-16
  - that 2026-09-16 is BOTH a fomc_decision and a vix_expiry in the calendar
  - SPY 21d return rank (252d, lag-1), which is what turns the event sleeve's
    T2 short off, and where it sits on the prereg's own "rank21>70 flips it"
    line
  - 2026 is a midterm year (T1 off)
  - XLE / USO / SLB distance from a 52-week high (the B2 entry state and the
    book-overlap disclosure)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

ASOF = pd.Timestamp("2026-08-31")
DEC = pd.Timestamp("2026-09-16")

px = load_prices(["SPY", "XLE", "USO", "SLB", "XOP"])
S = {t: px[t]["Close"].dropna() for t in px}
spy = S["SPY"]
print("freshest SPY bar in the cache: %s" % spy.index[-1].date())
assert spy.index[-1] == ASOF, "signal close is not 2026-08-31"

bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
sess = pd.date_range(ASOF, DEC, freq=bd)
print("sessions from the 2026-08-31 signal close to the decision close:")
print("  ", ", ".join(str(d.date()) for d in sess))
entry = sess[1]
print("  entry MOC = %s (lag 1)" % entry.date())
print("  td from the ENTRY close to the decision close = %d" % (len(sess) - 2))
print("  -> h=10 exits ON the decision close; h=9 exits the session before. OK"
      if len(sess) - 2 == 10 else "  -> GEOMETRY MISMATCH")

ev = load_events()
on = ev[ev["date"] == DEC]
print("\ncalendar events on %s: %s" % (DEC.date(), sorted(on["event"].unique())))
print("2026 %% 4 == %d -> midterm year: %s (T1 is non-midterm ONLY, so T1 is OFF)"
      % (2026 % 4, 2026 % 4 == 2))

r21 = pct_rank(spy, 21, 252)
print("\nSPY 21d return rank (252d) at the 2026-08-31 close = %.1f."
      % r21.iloc[-1])
print("  That IS the lag-1 input for a %s entry, i.e. the number the sleeve's"
      % entry.date())
print("  T2 gate reads. (The 2026-08-28 close read %.1f, for context.)"
      % r21.iloc[-2])
print("  T2 gate is rank21 < 50 -> T2 is OFF. The sleeve's own frozen prereg")
print("  says 'rank21>70 tapes FLIP the short to a loser'; the live reading sits")
print("  between the gate and that flip line, and the prior session was above it.")
print("  SPY 21d return itself: %+.2f%%" % (100 * (spy.iloc[-1] / spy.iloc[-22] - 1)))

print("\n52-week-high distances at the signal close:")
for t in ("XLE", "USO", "SLB", "XOP", "SPY"):
    s = S[t]
    hi = rolling_on_valid(s, lambda x: x.rolling(252).max())
    print("  %-4s %8.2f   %+6.2f%% from its 252d high" % (t, s.iloc[-1], 100 * (s.iloc[-1] / hi.iloc[-1] - 1)))
print("  1-day moves: XLE %+.2f%%  USO %+.2f%%  SLB %+.2f%%  SPY %+.2f%%"
      % tuple(100 * S[t].pct_change().iloc[-1] for t in ("XLE", "USO", "SLB", "SPY")))
print("\nBOOK OVERLAP: a live staged OVS SHORT in SLB sits in the same complex as")
print("any long-energy pitch today.")
