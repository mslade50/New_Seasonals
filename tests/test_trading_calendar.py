"""NYSE trading-calendar guards (2026-07-16 switch from USFederalHolidayCalendar).

Three things must hold:
1. The calendar has NYSE semantics — Good Friday closed, Columbus Day and
   Veterans Day open. The federal calendar was wrong on all three: the
   morning after each fake holiday, expected_data_date resolved to Friday,
   the real Monday bar was trimmed as "too new", and Friday's already-traded
   signals were re-staged.
2. The OneDrive live-side copy (trading_calendar_live.py, used by
   order_staging's entry-expiry back-computation and div_adjust) generates
   an IDENTICAL holiday set — daily_scan computes exit dates with the repo
   copy and order_staging back-computes entry expiry with the live copy, so
   any divergence silently shifts live order expiries. (Skipped where the
   OneDrive dir isn't present, e.g. CI.)
3. Known-date pins so a pandas holiday-rule regression can't slip through.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_calendar import NYSE_HOLIDAYS, TRADING_DAY

IBKR_DIR = os.path.join(os.path.expanduser('~'), 'OneDrive', 'trading_ibkr')

HOLIDAY_SET = set(pd.DatetimeIndex(NYSE_HOLIDAYS))


def _is_holiday(datestr):
    return pd.Timestamp(datestr) in HOLIDAY_SET


def test_nyse_semantics_vs_federal():
    # Columbus Day + Veterans Day: NYSE TRADES (the federal calendar's error)
    assert not _is_holiday('2025-10-13')   # Columbus Day 2025
    assert not _is_holiday('2026-10-12')   # Columbus Day 2026
    assert not _is_holiday('2025-11-11')   # Veterans Day 2025
    assert not _is_holiday('2026-11-11')   # Veterans Day 2026
    # Good Friday: NYSE CLOSED (missing from the federal calendar)
    assert _is_holiday('2025-04-18')
    assert _is_holiday('2026-04-03')


def test_known_holidays_pinned():
    assert _is_holiday('2026-01-01')       # New Year's
    assert _is_holiday('2026-01-19')       # MLK
    assert _is_holiday('2026-02-16')       # Washington's Birthday
    assert _is_holiday('2026-05-25')       # Memorial Day
    assert _is_holiday('2026-06-19')       # Juneteenth
    assert _is_holiday('2026-07-03')       # July 4 observed (Sat -> Fri)
    assert _is_holiday('2026-09-07')       # Labor Day
    assert _is_holiday('2026-11-26')       # Thanksgiving
    assert _is_holiday('2026-12-25')       # Christmas
    # Juneteenth starts 2022 — NYSE traded 2021-06-18/19 era
    assert not _is_holiday('2021-06-18')
    # Jan 1 2022 fell on Saturday: NYSE did NOT observe Friday 2021-12-31
    assert not _is_holiday('2021-12-31')
    # Special closures
    assert _is_holiday('2025-01-09')       # Carter mourning
    assert _is_holiday('2018-12-05')       # Bush mourning
    assert _is_holiday('2012-10-29')       # Sandy


def test_trading_day_arithmetic():
    # Tuesday after Columbus Day 2026 minus one trading day = Monday 10/12
    assert (pd.Timestamp('2026-10-13') - TRADING_DAY) == pd.Timestamp('2026-10-12')
    # Thursday before Good Friday 2026 plus one trading day = Monday 4/6
    assert (pd.Timestamp('2026-04-02') + TRADING_DAY) == pd.Timestamp('2026-04-06')


def test_live_copy_identical():
    """order_staging back-computes entry expiry with the OneDrive copy —
    the two calendars must generate identical holiday sets."""
    if not os.path.isdir(IBKR_DIR):
        pytest.skip(f"live execution dir not present: {IBKR_DIR}")
    sys.path.insert(0, IBKR_DIR)
    try:
        import trading_calendar_live as live
    except ImportError as e:
        pytest.skip(f"trading_calendar_live not importable ({e})")
    repo_set = set(pd.DatetimeIndex(NYSE_HOLIDAYS))
    live_set = set(pd.DatetimeIndex(live.NYSE_HOLIDAYS))
    only_repo = sorted(str(d.date()) for d in repo_set - live_set)
    only_live = sorted(str(d.date()) for d in live_set - repo_set)
    assert repo_set == live_set, (
        f"repo vs live calendar drift — only in repo: {only_repo[:5]}, "
        f"only in live: {only_live[:5]} (sync both copies!)"
    )
