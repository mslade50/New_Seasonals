from datetime import datetime, timezone

import pytest

from episodic_pivot import listed_universe as lu

TARGET = "2026-09-22"
NOW = datetime(2026, 9, 22, 12, 30, tzinfo=timezone.utc)


@pytest.fixture
def evidence(monkeypatch):
    monkeypatch.setattr(lu, "MIN_DIRECTORY_ROWS", 1)
    bodies = {
        "nasdaqlisted.txt": "Symbol|Security Name|Test Issue|ETF\nNEWX|New Company Common Stock|N|N\nADRZ|Foreign Company American Depositary Shares|N|N\nTEST|Test Company|Y|N\nETFZ|Equity ETF|N|Y\nWRTZ|Issuer Warrants|N|N\nUNTZ|Issuer Units|N|N\nPREF|Issuer Preferred Shares|N|N\nDEBT|Issuer Senior Notes|N|N\nFile Creation Time: 0922202608:00|||",
        "otherlisted.txt": "ACT Symbol|Security Name|Exchange|NASDAQ Symbol|Test Issue|ETF\nBRK.B|Berkshire Common Class B|N|BRK.B|N|N\nFile Creation Time: 0922202608:00|||||",
    }
    return {"schema": lu.SCHEMA, "session_date": TARGET, "sources": {
        name: {"url": url, "content": bodies[name], "captured_at": "2026-09-22T12:15:00Z"}
        for name, url in lu.URLS.items()}}


def test_broad_listing_membership_adr_class_mapping_and_exclusions(evidence):
    rows, coverage = lu.validate_universe(evidence, TARGET, now=NOW)
    assert set(rows) == {"NEWX", "ADRZ", "BRK.B"}
    assert rows["BRK.B"]["yahoo_symbol"] == "BRK-B"
    assert rows["ADRZ"]["exchange"] == "NASDAQ"
    assert coverage["listed_equities"] == 3
    assert coverage["directories"]["nasdaqlisted.txt"]["excluded"] == {
        "test_issue": 1, "exchange_traded_fund": 1, "warrant_right_unit_preferred_or_debt": 4}


@pytest.mark.parametrize("mutation", ["missing_directory", "no_footer", "stale", "future",
                                     "future_capture", "wrong_url", "wrong_session", "missing_column",
                                     "duplicate", "bad_flag", "truncated_row"])
def test_untrusted_or_incomplete_listing_input_rejected(evidence, mutation):
    record = evidence["sources"]["nasdaqlisted.txt"]
    if mutation == "missing_directory": evidence["sources"].pop("otherlisted.txt")
    elif mutation == "no_footer": record["content"] = record["content"].split("File Creation")[0]
    elif mutation == "stale": record["content"] = record["content"].replace("09222026", "09182026")
    elif mutation == "future": record["content"] = record["content"].replace("09222026", "09232026")
    elif mutation == "future_capture": record["captured_at"] = "2026-09-22T13:00:00Z"
    elif mutation == "wrong_url": record["url"] = "https://example.com/list.txt"
    elif mutation == "wrong_session": evidence["session_date"] = "2026-09-21"
    elif mutation == "missing_column": record["content"] = record["content"].replace("Test Issue", "Unknown")
    elif mutation == "duplicate": record["content"] = record["content"].replace("ADRZ|", "NEWX|")
    elif mutation == "bad_flag": record["content"] = record["content"].replace("Stock|N|N", "Stock|?|N")
    else: record["content"] = record["content"].replace("Stock|N|N", "Stock|N")
    with pytest.raises(ValueError):
        lu.validate_universe(evidence, TARGET, now=NOW)


def test_unexpectedly_small_directory_is_not_a_valid_full_universe(evidence, monkeypatch):
    monkeypatch.setattr(lu, "MIN_DIRECTORY_ROWS", 1000)
    with pytest.raises(ValueError, match="unexpectedly small"):
        lu.validate_universe(evidence, TARGET, now=NOW)


def test_latest_prior_session_directory_is_acceptable(evidence):
    evidence["sources"]["nasdaqlisted.txt"]["content"] = evidence["sources"]["nasdaqlisted.txt"]["content"].replace("09222026", "09212026")
    assert lu.validate_universe(evidence, TARGET, now=NOW)[1]["listed_equities"] == 3


@pytest.mark.parametrize("security", ["Right", "Rights", "Unit", "Units", "Warrant", "Note", "Bond"])
def test_singular_and_plural_non_equity_descriptions_are_excluded(evidence, security):
    record = evidence["sources"]["nasdaqlisted.txt"]
    record["content"] = record["content"].replace("New Company Common Stock", "New Company - " + security)
    assert "NEWX" not in lu.validate_universe(evidence, TARGET, now=NOW)[0]


def test_same_issuer_duplicate_deduplicates_but_identity_conflict_fails(evidence):
    record = evidence["sources"]["otherlisted.txt"]
    record["content"] = record["content"].replace("BRK.B|Berkshire Common Class B|N|BRK.B", "NEWX|New Company Common Stock|N|NEWX")
    rows, coverage = lu.validate_universe(evidence, TARGET, now=NOW)
    assert len(rows) == 2 and coverage["duplicates"] == 1
    record["content"] = record["content"].replace("New Company Common Stock", "Different Company Stock")
    with pytest.raises(ValueError, match="Conflicting"):
        lu.validate_universe(evidence, TARGET, now=NOW)


def test_capture_fetches_both_official_sources_without_static_fallback(evidence, monkeypatch):
    calls = []

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    class Response:
        def __init__(self, text): self.text = text
        def raise_for_status(self): pass

    def get(url, timeout):
        calls.append((url, timeout))
        name = next(name for name, value in lu.URLS.items() if value == url)
        return Response(evidence["sources"][name]["content"])

    monkeypatch.setattr(lu, "datetime", Clock)
    monkeypatch.setattr(lu.requests, "get", get)
    captured = lu.capture_universe(TARGET)
    assert {url for url, _ in calls} == set(lu.URLS.values())
    assert len(lu.validate_universe(captured, TARGET, now=NOW)[0]) == 3
