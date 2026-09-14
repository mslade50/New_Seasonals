"""Broker book timestamps use agent seconds or Cloudflare fallback milliseconds."""
from datetime import datetime, timezone

import pytest

from daily_execution_report import validate_book


NOW = datetime(2026, 9, 14, 20, 30, tzinfo=timezone.utc)


def book_at(stamp):
    return {"at": stamp, "accounts": [
        {"key": "primary", "positions": [], "orders": []},
    ]}


@pytest.mark.parametrize("scale", [1, 1000], ids=["agent-seconds", "cloud-milliseconds"])
@pytest.mark.parametrize("age", [0, 6.25, 300, -30])
def test_current_book_accepts_both_wire_formats(scale, age):
    book = book_at((NOW.timestamp() - age) * scale)
    original_stamp = book["at"]
    assert validate_book(book, NOW) is book["accounts"][0]
    assert book["at"] == original_stamp


@pytest.mark.parametrize("scale", [1, 1000])
@pytest.mark.parametrize("age", [300.01, 600, -30.01, -600])
def test_stale_and_future_books_still_fail(scale, age):
    with pytest.raises(RuntimeError, match="stale or has an invalid timestamp"):
        validate_book(book_at((NOW.timestamp() - age) * scale), NOW)


@pytest.mark.parametrize("stamp", [None, "invalid", {}, float("nan"), float("inf"),
                                  float("-inf"), 0, -1, True, False])
def test_missing_or_invalid_timestamp_fails(stamp):
    with pytest.raises(RuntimeError, match="timestamp"):
        validate_book(book_at(stamp), NOW)


@pytest.mark.parametrize("scale", [1, 1000])
def test_fresh_but_incomplete_primary_still_fails(scale):
    book = book_at(NOW.timestamp() * scale)
    book["accounts"][0]["error"] = "not connected"
    with pytest.raises(RuntimeError, match="snapshot is incomplete"):
        validate_book(book, NOW)
