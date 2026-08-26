"""Gate scheduled Discretionary Focus runs to actual NYSE sessions.

GitHub's weekday cron also fires on exchange holidays.  Without this gate, a
holiday run could label the next session's shortlist early and consume that
session's exact-once email receipt.  Manual dispatches are intentionally
handled by the workflow expression and may still run for diagnostics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading_calendar import NYSE_HOLIDAYS  # noqa: E402


ET = ZoneInfo("America/New_York")
TARGET_LOCAL_TIME = dt.time(8, 35)
EARLIEST_LOCAL_START = dt.time(8, 25)
LATEST_LOCAL_START = dt.time(9, 10)
LATEST_LOCAL_DELIVERY = dt.time(9, 20)
_HOLIDAY_DATES = frozenset(
    stamp.date() for stamp in pd.DatetimeIndex(NYSE_HOLIDAYS)
)


def is_nyse_session(day: dt.date) -> bool:
    """Return whether *day* is a regular NYSE trading session."""
    return day.weekday() < 5 and day not in _HOLIDAY_DATES


def session_gate(now: dt.datetime | None = None) -> tuple[bool, dt.date]:
    """Resolve the ET calendar date and whether a scheduled run may proceed."""
    current = now or dt.datetime.now(dt.timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=dt.timezone.utc)
    market_date = current.astimezone(ET).date()
    return is_nyse_session(market_date), market_date


def scheduled_session_gate(
    now: dt.datetime | None,
    scheduled_cron: str,
) -> tuple[bool, dt.date]:
    """Accept only the UTC cron slot that maps to 08:35 New York.

    GitHub may start a scheduled job several minutes late, so this compares
    the triggering cron expression—not the runner's wall-clock minute.
    """
    is_session, market_date = session_gate(now)
    parts = scheduled_cron.strip().split()
    if len(parts) != 5:
        return False, market_date
    try:
        cron_minute = int(parts[0])
        cron_hour = int(parts[1])
    except ValueError:
        return False, market_date
    local_target = dt.datetime.combine(market_date, TARGET_LOCAL_TIME, tzinfo=ET)
    utc_target = local_target.astimezone(dt.timezone.utc)
    current = now or dt.datetime.now(dt.timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=dt.timezone.utc)
    local_now = current.astimezone(ET)
    inside_delivery_window = (
        EARLIEST_LOCAL_START <= local_now.time().replace(tzinfo=None) <= LATEST_LOCAL_START
    )
    matches_slot = (
        cron_minute == utc_target.minute and cron_hour == utc_target.hour
    )
    return is_session and matches_slot and inside_delivery_window, market_date


def delivery_window_gate(
    now: dt.datetime | None = None,
) -> tuple[bool, dt.date]:
    """Require the actual publish/send clock to remain pre-market."""
    current = now or dt.datetime.now(dt.timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=dt.timezone.utc)
    local = current.astimezone(ET)
    allowed = (
        is_nyse_session(local.date())
        and EARLIEST_LOCAL_START
        <= local.time().replace(tzinfo=None)
        <= LATEST_LOCAL_DELIVERY
    )
    return allowed, local.date()


def _parse_now(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def _append_github_output(path: Path, *, should_run: bool, market_date: dt.date) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"should_run={'true' if should_run else 'false'}\n")
        handle.write(f"market_date={market_date.isoformat()}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--now",
        help="ISO timestamp override for deterministic diagnostics (default: current UTC)",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=None,
        help="output file override (default: the GITHUB_OUTPUT environment path)",
    )
    parser.add_argument(
        "--scheduled-cron",
        default="",
        help="GitHub event.schedule value; gates the DST-specific UTC slot",
    )
    parser.add_argument(
        "--delivery-window",
        action="store_true",
        help="gate the actual publish/send time to 08:25-09:20 New York",
    )
    parser.add_argument(
        "--require-allowed",
        action="store_true",
        help="exit nonzero when the selected gate is closed",
    )
    args = parser.parse_args()

    parsed_now = _parse_now(args.now)
    if args.delivery_window:
        should_run, market_date = delivery_window_gate(parsed_now)
    elif args.scheduled_cron.strip():
        should_run, market_date = scheduled_session_gate(
            parsed_now, args.scheduled_cron
        )
    else:
        should_run, market_date = session_gate(parsed_now)
    output = args.github_output
    if output is None and os.environ.get("GITHUB_OUTPUT"):
        output = Path(os.environ["GITHUB_OUTPUT"])
    if output is not None:
        _append_github_output(
            output, should_run=should_run, market_date=market_date
        )

    disposition = "allowed" if should_run else "blocked"
    print(f"{market_date.isoformat()}: {disposition}")
    return 1 if args.require_allowed and not should_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
