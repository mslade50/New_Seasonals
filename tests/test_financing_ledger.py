import pytest

from fundamental.financing_ledger import (knowledge, validate_records, reconcile,
    current_receipts, financing_facts, facts_available_at)
from scripts.build_financing_ledger import imported_announcements


def record(**kwargs):
    base = dict(record_id="one", funding_id="deal", cik=1, ticker="AAA", kind="primary_equity",
        stage="receipt", status="verified", available_at="2025-05-05T12:00:00Z",
        amount_usd=100., amount_basis="net", amount_scope="incremental", receipt_id="base",
        cash_start="2025-04-15", cash_end="2025-04-15", sources=["https://www.sec.gov/test"])
    return dict(base, **kwargs)


def observation(**kwargs):
    return dict(dict(cik=1, ticker="AAA", session="2025-05-30", as_of="2025-05-30T20:30:00Z",
        balance_date="2025-03-31", group="strong_short", cash_only=False, runway_operating=12.), **kwargs)


def test_expected_close_does_not_become_cash_and_later_confirmation_does_not_leak():
    priced = record(record_id="price", stage="pricing", event_date="2025-03-28", expected_close="2025-04-15",
                    available_at="2025-03-28T20:00:00Z", amount_basis="gross")
    receipt = record()
    early = observation(session="2025-04-30", as_of="2025-04-30T20:30:00Z")
    got = reconcile(early, {"reported_liquidity": 50}, [priced, receipt])
    assert got["status"] == "funding_announced_receipt_unconfirmed"
    assert got["net_receipts_usd"] == 0
    later = reconcile(observation(), {"reported_liquidity": 50}, [priced, receipt])
    assert later["net_receipts_usd"] == 100
    assert later["reported_plus_known_net_before_burn"] == 150
    assert later["current_cash_estimate"] is None


def test_date_only_never_assumed_known_at_open_or_close():
    r = record(available_at=None, available_date="2025-05-05")
    assert knowledge(r, "2025-05-05T02:00:00Z") == "future"  # Still May4 ET.
    assert knowledge(r, "2025-05-05T20:30:00Z") == "same_day_time_unknown"
    assert knowledge(r, "2025-05-06T04:00:00Z") == "known"
    assert knowledge(record(), "2025-05-05T12:00:00Z") == "known"


def test_exact_same_day_launch_compared_to_signal():
    r = record(stage="announcement", event_date="2025-05-30", available_at="2025-05-30T20:02:00Z")
    assert reconcile(observation(), {}, [r])["pending_funding_ids"] == ["deal"]
    assert not reconcile(observation(as_of="2025-05-30T20:00:00Z"), {}, [r])["pending_funding_ids"]


def test_revisions_deduplicate_receipt_but_separate_option_tranches_add_once():
    first = record(amount_basis="gross", amount_usd=115)
    revised = record(record_id="net", available_at="2025-05-07T12:00:00Z", amount_usd=107.5)
    final = record(record_id="final", available_at="2025-08-07T12:00:00Z", amount_usd=107.2)
    option = record(record_id="option", receipt_id="option", amount_usd=20)
    got = reconcile(observation(), {"reported_liquidity": 50}, [first, revised, final, option])
    assert got["net_receipts_usd"] == 127.5
    assert got["gross_receipts_usd"] == 0
    assert set(got["added_receipt_ids"]) == {"net", "option"}


def test_cash_inside_balance_or_spanning_it_is_not_added():
    embedded = record(cash_start="2025-03-31", cash_end="2025-03-31")
    overlap = record(record_id="overlap", receipt_id="overlap", cash_start="2025-03-01", cash_end="2025-04-15")
    got = reconcile(observation(), {}, [embedded, overlap])
    assert got["embedded_receipt_ids"] == ["one"]
    assert got["overlapping_receipt_ids"] == ["overlap"]
    assert got["net_receipts_usd"] == 0
    assert got["status"] == "receipt_allocation_needs_review"


def test_gross_and_facility_capacity_never_inflate_net_bridge():
    gross = record(amount_basis="gross")
    cap = record(record_id="facility", funding_id="facility", stage="capacity", amount_basis="capacity", amount_usd=1e9)
    got = reconcile(observation(), {"reported_liquidity": 50}, [gross, cap])
    assert got["reported_plus_known_net_before_burn"] == 50
    assert got["gross_receipts_usd"] == 100
    assert got["capacity_funding_ids"] == ["facility"]


def test_other_issuers_provisional_and_unknown_receipts_are_not_net_cash():
    records = [record(cik=2), record(record_id="provisional", status="provisional"),
               record(record_id="unknown", receipt_id="unknown", amount_usd=None)]
    got = reconcile(observation(), {}, records)
    assert got["net_receipts_usd"] == 0
    assert got["unknown_amount_receipt_ids"] == ["unknown"]
    assert got["fully_reconciled"] is False
    assert got["research_eligible"] is False


def test_conflicting_receipt_assertions_fail_closed():
    rows = [record(), record(record_id="conflict", amount_usd=150)]
    chosen, conflicts = current_receipts(rows, observation()["as_of"])
    assert chosen == [] and set(conflicts) == {"one", "conflict"}
    assert reconcile(observation(), {}, rows)["status"] == "receipt_allocation_needs_review"


@pytest.mark.parametrize("change", [dict(amount_usd=-1), dict(amount_usd=float("nan")),
    dict(cash_end="2025-03-01"), dict(cash_end="2025-06-01"), dict(amount_scope="cumulative"),
    dict(sources=[]), dict(available_at="2025-05-05T12:00:00")])
def test_invalid_receipts_rejected(change):
    with pytest.raises((ValueError, KeyError)):
        validate_records([record(**change)])


def test_duplicate_identity_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_records([record(), record()])
    with pytest.raises(ValueError, match="crosses issuers"):
        validate_records([record(), record(record_id="two", cik=2)])


def test_unknowns_never_clear_a_signal():
    got = reconcile(observation(), {}, [])
    assert got["status"] == "no_documented_intervening_funding_coverage_incomplete"
    assert got["funding_coverage"] == "incomplete"
    assert not got["fully_reconciled"]


def test_import_never_attaches_later_priced_amount_to_launch():
    old = dict(event_id="id", cik=1, ticker_at_event="AAA", announcement_date="2025-05-05",
               status="verified", sources=["https://www.sec.gov/test"], gross_usd=100e6)
    got = imported_announcements({"events": [old]})[0]
    assert got["amount_usd"] is None


def test_xbrl_uses_filing_acceptance_retains_vintages_and_never_sums_tags():
    subs = {"filings": {"recent": {"accessionNumber": ["first", "later"], "form": ["10-Q", "10-Q/A"],
        "acceptanceDateTime": ["2025-05-05T12:00:00Z", "2025-06-05T12:00:00Z"]}}}
    original = dict(start="2025-01-01", end="2025-03-31", val=100, accn="first", form="10-Q")
    revised = dict(original, val=120, accn="later", form="10-Q/A")
    payload = {"cik": 1, "facts": {"us-gaap": {
        "ProceedsFromIssuanceOfCommonStock": {"units": {"USD": [original, original, revised]}},
        "ProceedsFromIssuanceOrSaleOfEquity": {"units": {"USD": [original]}}}}}
    rows = financing_facts(payload, subs)
    assert len(rows) == 3  # Exact duplicate removed, alternate concept retained.
    before = facts_available_at(rows, 1, "2025-05-30T20:30:00Z")
    assert len(before) == 2 and all(r["value"] == 100 for r in before)
    assert all(not r["summable"] for r in rows)
    assert facts_available_at(rows, 1, "2025-05-01T20:30:00Z") == []
    assert facts_available_at(rows, 2, "2025-07-01T20:30:00Z") == []


def test_date_only_source_cannot_confirm_following_day_receipt():
    with pytest.raises(ValueError, match="predates"):
        validate_records([record(available_at=None, available_date="2025-05-05",
            cash_start="2025-05-06", cash_end="2025-05-06")])


def test_later_or_contemporaneous_gross_does_not_erase_known_net():
    net = record()
    for at in ["2025-05-05T12:00:00Z", "2025-05-06T12:00:00Z"]:
        gross = record(record_id="gross", amount_basis="gross", amount_usd=110, available_at=at)
        got = reconcile(observation(), {"reported_liquidity": 50}, [net, gross])
        assert got["reported_plus_known_net_before_burn"] == 150
        assert not got["conflict_record_ids"]
