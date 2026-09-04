"""Standalone implementation helpers for the TLT month-end strategy.

The functions deliberately require an authoritative list of exchange sessions
for the target month. They do not approximate NYSE holidays.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Iterable, Literal

import pandas as pd


ENTRY_OFFSET = -6  # Python index: sixth-last session is T-5.
PILOT_NOTIONAL_PCT = 0.10
MAX_AGGREGATE_TLT_PCT = 0.20

Instruction = Literal["ENTER_LONG_MOC", "HOLD", "EXIT_LONG_MOC", "FLAT"]


@dataclass(frozen=True)
class MonthSchedule:
    month: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    sessions_held: int = 5


@dataclass(frozen=True)
class PilotOrder:
    shares: int
    incremental_notional: float
    incremental_nav_pct: float
    aggregate_tlt_nav_pct_after: float


def schedule_from_sessions(sessions: Iterable[object]) -> MonthSchedule:
    """Resolve T-5 entry and T exit from authoritative exchange sessions."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(sessions))).sort_values().unique()
    if len(idx) < 6:
        raise ValueError("A month needs at least six exchange sessions")
    periods = idx.to_period("M").unique()
    if len(periods) != 1:
        raise ValueError("All supplied sessions must belong to one calendar month")
    return MonthSchedule(
        month=str(periods[0]),
        entry_date=pd.Timestamp(idx[ENTRY_OFFSET]),
        exit_date=pd.Timestamp(idx[-1]),
    )


def instruction_for_session(
    session: object, schedule: MonthSchedule, *, position_open: bool
) -> Instruction:
    """Return the required end-of-day action for one exchange session."""
    session = pd.Timestamp(session).normalize()
    if session == schedule.entry_date.normalize() and not position_open:
        return "ENTER_LONG_MOC"
    if session == schedule.exit_date.normalize() and position_open:
        return "EXIT_LONG_MOC"
    if position_open:
        return "HOLD"
    return "FLAT"


def pilot_order(
    *,
    account_value: float,
    tlt_price: float,
    existing_tlt_notional: float = 0.0,
    pilot_notional_pct: float = PILOT_NOTIONAL_PCT,
    max_aggregate_tlt_pct: float = MAX_AGGREGATE_TLT_PCT,
) -> PilotOrder:
    """Size the incremental pilot while respecting aggregate TLT exposure.

    Ten percent NAV is the default because the worst historical five-session
    event in the research sample was about -4.13%, making that event roughly a
    -41 bp NAV loss before any interaction with an existing TLT allocation.
    """
    if account_value <= 0 or tlt_price <= 0:
        raise ValueError("account_value and tlt_price must be positive")
    if existing_tlt_notional < 0:
        raise ValueError("existing_tlt_notional cannot be negative")
    if not 0 <= pilot_notional_pct <= max_aggregate_tlt_pct <= 1:
        raise ValueError("Require 0 <= pilot <= max aggregate <= 1")

    desired = account_value * pilot_notional_pct
    remaining_capacity = max(
        account_value * max_aggregate_tlt_pct - existing_tlt_notional, 0.0
    )
    budget = min(desired, remaining_capacity)
    shares = floor(budget / tlt_price)
    notional = shares * tlt_price
    aggregate_after = existing_tlt_notional + notional
    return PilotOrder(
        shares=shares,
        incremental_notional=notional,
        incremental_nav_pct=notional / account_value,
        aggregate_tlt_nav_pct_after=aggregate_after / account_value,
    )

