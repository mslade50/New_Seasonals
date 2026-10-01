import pandas as pd
import hashlib
import pytest
from scripts.prepare_earnings_issuer_review import review_queue, snapshot_counts
from scripts.compare_earnings_shadow import parse_alpha_csv
from scripts.prepare_earnings_issuer_review import verified_raw


def test_legacy_windows_newlines_require_exact_original_digest():
    raw = b"symbol,date\r\nA,2026-10-13\r\n"
    digest = hashlib.sha256(raw).hexdigest()
    assert verified_raw(raw.replace(b"\r\n", b"\r\r\n"), digest) == raw.decode()
    with pytest.raises(ValueError, match="digest"):
        verified_raw(raw.replace(b"A,", b"B,"), digest)


def test_calendar_day_boundaries_union_and_share_class_alias():
    alpha = parse_alpha_csv("symbol,name,reportDate,fiscalDateEnding,estimate,currency\n"
        "A,A,2026-10-06,2026-09-30,1,USD\n"  # seven: exclude
        "B,B,2026-10-07,2026-09-30,1,USD\n"  # eight: include
        "C,C,2026-10-14,2026-09-30,1,USD\n"  # fifteen: include
        "D,D,2026-10-15,2026-09-30,1,USD\n"  # sixteen: exclude
        "BRK.B,Berkshire,2026-10-08,2026-09-30,1,USD\n"
        "OUT,Outside,2026-10-08,2026-09-30,1,USD\n")
    fmp = pd.DataFrame([dict(ticker="D", date="2026-10-13"),dict(ticker="B", date="2026-10-07")])
    queue = review_queue(alpha, {"A", "B", "C", "D", "BRK-B"}, "2026-09-29", fmp)
    assert [q["ticker"] for q in queue] == ["B", "BRK-B", "C", "D"]
    assert len(queue[0]["provider_events"]) == 2
    assert all(q["status"] == "needs_issuer_review" for q in queue)


def test_counts_are_unique_companies_inclusive_of_today_and_day14():
    alpha = parse_alpha_csv("symbol,name,reportDate,fiscalDateEnding,estimate,currency\n"
        "A,A,2026-09-29,2026-06-30,1,USD\n"
        "A,A,2026-10-13,2026-09-30,1,USD\n"
        "B,B,2026-10-14,2026-09-30,1,USD\n"
        "C,C,2026-10-13,2026-09-30,1,USD\n")
    count = snapshot_counts(alpha, {"A", "B"}, "2026-09-29")
    assert count["all_alpha"] == 2 and count["tracked"] == 1


class _ReadOnlyStore:
    def __init__(self, value):
        self.value, self.keys = value, []

    def read(self, key):
        self.keys.append(key)
        return self.value, "etag"

    def write(self, *args, **kwargs):
        raise AssertionError("issuer review must never claim or write the shared snapshot")


def test_r2_alpha_snapshot_is_read_only_and_date_checked():
    from alpha_calendar_snapshot import SnapshotError
    from scripts.prepare_earnings_issuer_review import r2_alpha_snapshot
    raw = "symbol,name,reportDate,fiscalDateEnding,estimate,currency\nA,A,2026-10-13,2026-09-30,1,USD\n"
    ready = dict(state="ready", mode="authenticated", captured_at_utc="2026-10-01T10:00:00+00:00",
                 raw=raw, sha256=hashlib.sha256(raw.encode()).hexdigest())
    store = _ReadOnlyStore(ready)
    assert r2_alpha_snapshot("2026-10-01", store) == (raw, "provider_snapshots/alpha_earnings/2026-10-01.json")
    with pytest.raises(SnapshotError, match="wrong capture date"):
        r2_alpha_snapshot("2026-10-02", _ReadOnlyStore(ready))
    with pytest.raises(SnapshotError, match="already attempted"):
        r2_alpha_snapshot("2026-10-01", _ReadOnlyStore(dict(ready, state="attempted")))
    with pytest.raises(SnapshotError, match="No shared Alpha snapshot"):
        r2_alpha_snapshot("2026-10-01", _ReadOnlyStore(None))
