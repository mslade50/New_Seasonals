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
