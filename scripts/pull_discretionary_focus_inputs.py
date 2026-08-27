"""Materialize the fail-closed inputs for the Discretionary Focus run.

The daily focus job is intentionally independent from the strategy scanner and
the fundamental sleeve's manual workflow.  It reads market/research inputs and
may publish a research-only attention list; it never touches Sheets, a broker,
or an order-staging payload.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cache_io import download_to_local, last_download_error  # noqa: E402


RECEIPT_KEY = "discretionary_focus/email_receipt.json"

REQUIRED = {
    "master_prices.parquet": ROOT / "data" / "master_prices.parquet",
    "overflow_prices.parquet": ROOT / "data" / "overflow_prices.parquet",
    "earnings_calendar.parquet": ROOT / "data" / "earnings_calendar.parquet",
    "symbol_master.parquet": ROOT / "data" / "symbol_master.parquet",
}

OPTIONAL = {
    # Extends the five-session earnings gate to dynamic overflow names that
    # are not yet present in the production universe's primary calendar.
    "earnings_calendar_overflow.parquet": (
        ROOT / "data" / "earnings_calendar_overflow.parquet"
    ),
    # Required by the builder whenever the technical stage produces a candidate;
    # irrelevant when the technical funnel conclusively returns zero.
    "fundamental/current/daily_report_latest.json": (
        ROOT / "data" / "fundamental" / "current" / "daily_report_latest.json"
    ),
    RECEIPT_KEY: (
        ROOT / "data" / "discretionary_focus" / "email_receipt.json"
    ),
}


def _confirmed_not_found(error: str | None) -> bool:
    text = str(error or "")
    return any(token in text for token in ("404", "NoSuchKey", "Not Found"))


def pull_inputs(*, include_optional: bool = True) -> list[str]:
    failures: list[str] = []
    for key, path in REQUIRED.items():
        if download_to_local(key, str(path)):
            print(f"ready: {key} -> {path}")
        else:
            failures.append(f"{key}: {last_download_error() or 'download failed'}")

    if failures:
        return failures

    if include_optional:
        for key, path in OPTIONAL.items():
            if download_to_local(key, str(path)):
                print(f"ready: {key} -> {path}")
            else:
                error = last_download_error() or "download failed"
                if key == RECEIPT_KEY:
                    # A missing receipt is safe only when R2 positively says
                    # the object does not exist. Auth, transport, and timeout
                    # failures must never initialize a blank exact-once state.
                    if _confirmed_not_found(error) and not path.exists():
                        print("ready: no prior Discretionary Focus email receipt")
                    else:
                        failures.append(f"{key}: {error}")
                else:
                    print(f"optional input unavailable: {key}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--required-only",
        action="store_true",
        help="skip optional research, receipt, and TradingView inputs",
    )
    args = parser.parse_args()
    failures = pull_inputs(include_optional=not args.required_only)
    if failures:
        print("Discretionary Focus inputs unavailable:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
