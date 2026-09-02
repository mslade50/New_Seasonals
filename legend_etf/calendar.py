"""XNYS session helpers used by both preparation and execution.

Full-session enforcement is intentional.  The historical study only admitted
entry dates with a complete 09:30-16:00 ETF session, so known early closes are
blocked before entry rather than discovered with hindsight.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=1)
def _calendar():
    try:
        import exchange_calendars as xcals
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise RuntimeError(
            "exchange-calendars is required for exact XNYS session handling"
        ) from exc
    return xcals.get_calendar("XNYS")


def _utc_day(value: date | str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tz is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    # exchange_calendars 4.x session labels are timezone-naive UTC dates.
    return stamp.normalize()


def is_session(value: date | str | pd.Timestamp) -> bool:
    return bool(_calendar().is_session(_utc_day(value)))


def previous_session(value: date | str | pd.Timestamp) -> pd.Timestamp:
    """Return the immediately preceding XNYS session as a naive date stamp."""

    day = _utc_day(value)
    if _calendar().is_session(day):
        result = _calendar().previous_session(day)
    else:
        result = _calendar().date_to_session(day, direction="previous")
    result = pd.Timestamp(result)
    if result.tz is not None:
        result = result.tz_convert("UTC").tz_localize(None)
    return result.normalize()


def next_session(value: date | str | pd.Timestamp) -> pd.Timestamp:
    day = _utc_day(value)
    if _calendar().is_session(day):
        result = _calendar().next_session(day)
    else:
        result = _calendar().date_to_session(day, direction="next")
    result = pd.Timestamp(result)
    if result.tz is not None:
        result = result.tz_convert("UTC").tz_localize(None)
    return result.normalize()


def is_full_session(value: date | str | pd.Timestamp) -> bool:
    """True only for a regular XNYS session closing at 16:00 New York time."""

    day = _utc_day(value)
    if not _calendar().is_session(day):
        return False
    close = pd.Timestamp(_calendar().session_close(day)).tz_convert(
        "America/New_York"
    )
    return close.hour == 16 and close.minute == 0


def rth_bar_starts(
    start_session: date | str | pd.Timestamp,
    end_session: date | str | pd.Timestamp,
    *,
    frequency: str = "15min",
) -> pd.DatetimeIndex:
    """Return the exact calendar-aware XNYS RTH bar-start grid."""

    calendar = _calendar()
    sessions = calendar.sessions_in_range(
        _utc_day(start_session), _utc_day(end_session)
    )
    pieces: list[pd.DatetimeIndex] = []
    for session in sessions:
        opened = pd.Timestamp(calendar.session_open(session)).tz_convert(
            "America/New_York"
        )
        closed = pd.Timestamp(calendar.session_close(session)).tz_convert(
            "America/New_York"
        )
        pieces.append(
            pd.date_range(opened, closed, freq=frequency, inclusive="left")
        )
    if not pieces:
        return pd.DatetimeIndex([], tz="America/New_York")
    result = pieces[0]
    for piece in pieces[1:]:
        result = result.append(piece)
    return result


def session_labels(
    start_session: date | str | pd.Timestamp,
    end_session: date | str | pd.Timestamp,
) -> pd.DatetimeIndex:
    """Return exact timezone-naive XNYS session labels for a date range."""

    labels = _calendar().sessions_in_range(
        _utc_day(start_session), _utc_day(end_session)
    )
    result = pd.DatetimeIndex(labels)
    if result.tz is not None:
        result = result.tz_convert("UTC").tz_localize(None)
    return result.normalize()


def require_full_entry_session(value: date | str | pd.Timestamp) -> None:
    if not is_session(value):
        raise ValueError(f"{pd.Timestamp(value).date()} is not an XNYS session")
    if not is_full_session(value):
        raise ValueError(
            f"{pd.Timestamp(value).date()} is a scheduled XNYS early close; "
            "the Legend ETF sleeve does not trade it"
        )
