"""Shared NYSE trading-day arithmetic (introduced 2026-07-16).

Every scheduling surface — scan cutoff dates, exit-date math, fill windows,
entry-expiry back-computation, T+1 fill verification, earnings-blackout
offsets — must count the SAME trading days. USFederalHolidayCalendar was
wrong in both directions for NYSE:

- It marks Columbus Day and Veterans Day as holidays while NYSE TRADES both.
  The morning after each, expected_data_date resolved to the prior Friday,
  the real Monday bar was trimmed as "too new", and Friday's already-traded
  signals were re-detected and re-staged at stale limit/ATR levels.
- It misses Good Friday, when NYSE is CLOSED — so hold windows, fill windows
  and blackout offsets around it counted one session too many.

order_staging.py and div_adjust.py (OneDrive\\trading_ibkr — outside this
repo, deliberately import-decoupled) carry an inline copy of this exact
construction and must stay in lockstep. Guard: tests/test_trading_calendar.py
asserts the repo and OneDrive calendars generate identical holiday sets and
pins the Columbus/Veterans/Good-Friday behaviors.
"""
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """Regular NYSE full-closure holidays. Half-days are trading days here
    (early closes still print a bar, which is what date arithmetic needs)."""
    rules = [
        # Jan 1 on a Saturday is NOT observed the prior Friday (NYSE Rule
        # 7.2), so sunday_to_monday — not nearest_workday.
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,   # NYSE observance since 1998; pre-1998 unused here
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


# Ad-hoc full closures in the data era. Future one-offs (mourning days,
# disasters) are announced ad hoc — append here AND in the OneDrive copy.
SPECIAL_CLOSURES = [
    "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14",  # September 11
    "2004-06-11",                # Reagan day of mourning
    "2007-01-02",                # Ford day of mourning
    "2012-10-29", "2012-10-30",  # Hurricane Sandy
    "2018-12-05",                # G.H.W. Bush day of mourning
    "2025-01-09",                # Carter day of mourning
]

NYSE_HOLIDAYS = (
    NYSEHolidayCalendar()
    .holidays(start="1990-01-01", end="2040-12-31")
    .append(pd.DatetimeIndex(SPECIAL_CLOSURES))
    .sort_values()
    .unique()
)

# Precomputed-holiday CustomBusinessDay: identical arithmetic to the
# calendar= form, materially faster on repeated offset math.
TRADING_DAY = CustomBusinessDay(holidays=list(NYSE_HOLIDAYS))
