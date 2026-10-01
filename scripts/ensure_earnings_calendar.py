"""Repair a stale canonical earnings calendar before the morning scan.

The 2026-09-30 postclose earnings refresh failed, scan_pm was skipped, and the
next scan_am died inside daily_scan.py on ``validate_freshness`` after its
side-effect boundary, so the receipt went indeterminate and no recovery ran.
This step runs first in scan_am, before that boundary:

* fresh canonical R2 calendar (the scanner's own ``validate_freshness``): exit 0;
* stale: run the normal producer once (the same command earnings_and_grades
  runs), then re-check the canonical object; exit 0 when it is fresh now;
* still stale, producer failed, or the calendar cannot be read: exit 1, a
  local failure before the boundary.

What that buys: a stale calendar is repaired whenever the producer can run; a
local-only failure can still be rescued by the immediate GitHub fallback; and
the operator is emailed. It does NOT guarantee the 05:45 retry: with fallback
allowed (production), a failed GitHub run of non-rerun-safe scan_am still ends
indeterminate/manual_review, which the retry skips.

It never relaxes the scanner's gate. The producer publishes with an
ETag-conditional write and coordinates one Alpha request per New York date
through R2: a repair here spends that date's request, so the same evening's
earnings_and_grades reuses this morning's snapshot when it succeeded, and
cannot fetch again that date when it failed.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from earnings_calendar_provider import CalendarError, validate_freshness

KEY = "earnings_calendar.parquet"
PRODUCER = ("scripts/refresh_earnings_calendar.py",)
# Inside the supervisor's 1200 s step timeout so this script, not a kill,
# reports a hung producer.
PRODUCER_TIMEOUT_SECONDS = 1080


class CalendarUnavailable(RuntimeError):
    pass


def fetch_canonical(dest: Path) -> pd.DataFrame:
    from cache_io import download_to_local, last_download_error

    if not download_to_local(KEY, str(dest)):
        raise CalendarUnavailable(f"canonical {KEY} could not be downloaded: {last_download_error()}")
    try:
        return pd.read_parquet(dest)
    except Exception as exc:  # noqa: BLE001 - any unreadable object is unavailable
        raise CalendarUnavailable(f"canonical {KEY} is unreadable: {type(exc).__name__}") from None


def staleness(frame: pd.DataFrame, now: pd.Timestamp | None = None) -> str | None:
    try:
        validate_freshness(frame, now)
    except CalendarError as exc:
        return str(exc)
    return None


def run_producer(timeout_seconds: int = PRODUCER_TIMEOUT_SECONDS) -> int:
    try:
        return subprocess.run([sys.executable, *PRODUCER], cwd=str(ROOT), timeout=timeout_seconds).returncode
    except subprocess.TimeoutExpired:
        print(f"ERROR: earnings producer exceeded {timeout_seconds}s")
        return 124


def check(workdir: Path, label: str) -> str | None:
    frame = fetch_canonical(workdir / f"{label}.parquet")
    return staleness(frame)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ensure_earnings_", ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp)
        try:
            problem = check(workdir, "before")
        except CalendarUnavailable as exc:
            print(f"ERROR: earnings calendar check failed: {exc}")
            return 1
        if problem is None:
            print("Earnings calendar fresh; no refresh needed.")
            return 0
        print(f"Earnings calendar stale ({problem}); running the earnings producer.")
        rc = run_producer()
        try:
            after = check(workdir, "after")
        except CalendarUnavailable as exc:
            print(f"ERROR: earnings calendar re-check failed after producer exit {rc}: {exc}")
            return 1
        if after is None:
            print(f"Earnings calendar repaired (producer exit {rc}); canonical calendar is fresh.")
            return 0
        print(
            f"ERROR: earnings calendar still stale after producer exit {rc}: {after}. "
            "The scan is stopped before its side-effect boundary; nothing was staged."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
