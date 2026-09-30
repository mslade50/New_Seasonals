from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from fundamental.financing_history import (date_from_release, event_triage, historical_metrics,
    label_window, rate_summary, match_controls, group_name, intervening_announcement)
from scripts.build_financing_history import (merge_submissions, baseline_securities, stratum,
                                             event_search_query, verify_frozen_inputs, save)


def test_submissions_merge_retains_old_acceptance_and_deduplicates():
    main = {"filings": {"recent": {"accessionNumber": ["new"], "filingDate": ["2025-01-01"],
        "acceptanceDateTime": ["2025-01-01T17:00:00Z"]}, "files": []}}
    archive = {"accessionNumber": ["old", "new"], "filingDate": ["2022-01-01", "2025-01-01"],
               "acceptanceDateTime": ["2022-01-01T17:00:00Z", "wrong"]}
    merged = merge_submissions(main, [archive])["filings"]["recent"]
    assert merged["accessionNumber"] == ["old", "new"]
    assert merged["acceptanceDateTime"] == ["2022-01-01T17:00:00Z", "2025-01-01T17:00:00Z"]


def test_baseline_security_identity_uses_matching_context():
    xml = b'''<xbrl xmlns:dei="http://test"><dei:TradingSymbol contextRef="a">AAA</dei:TradingSymbol>
    <dei:SecurityExchangeName contextRef="a">NASDAQ</dei:SecurityExchangeName><dei:Security12bTitle contextRef="a">Common stock</dei:Security12bTitle>
    <dei:TradingSymbol contextRef="b">AAAW</dei:TradingSymbol><dei:Security12bTitle contextRef="b">Warrants</dei:Security12bTitle></xbrl>'''
    shares, warrant = baseline_securities(xml)
    assert shares["ticker"] == "AAA" and shares["exchange"] == "NASDAQ"
    assert warrant["title"] == "Warrants" and warrant["exchange"] == ""


def test_historical_reverse_split_does_not_invent_price_eligibility():
    dates = pd.bdate_range("2023-01-01", periods=160)
    # After a future 1-for-10 reverse split Yahoo shows the old $2 as $20.
    bars = pd.DataFrame({"Open": 20., "Close": 20., "Adj Close": 20., "Volume": 1e6,
                         "Stock Splits": 0.}, index=dates)
    bars.loc[dates[-1], "Stock Splits"] = .1
    got = historical_metrics(bars, bars, dates[140])
    assert got["close"] == pytest.approx(2)
    assert got["dollar_volume_20"] == pytest.approx(20e6)
    assert not got["tradable_filter"]
    # With a future 10-for-1 forward split the historical $20 stays eligible.
    bars[["Open", "Close", "Adj Close"]] = 2.
    bars["Volume"] = 10e6
    bars.loc[dates[-1], "Stock Splits"] = 10
    assert historical_metrics(bars, bars, dates[140])["tradable_filter"]


def test_unknown_negative_never_becomes_zero():
    row = {"cik": 1, "session": "2023-01-31", "as_of": "2023-01-31T21:30:00Z"}
    covered = {"cik": 1, "start": "2022-10-01", "status": "reviewed", "through": "2026-03-31"}
    assert label_window(row, [], dict(covered, status="searched"))[0] is None
    assert label_window(row, [], dict(covered, through="2023-02-01"))[0] is None
    assert label_window(row, [], dict(covered, start="2023-02-01"))[0] is None
    assert label_window(row, [], dict(covered, cik=2))[0] is None
    assert label_window(row, [], covered)[0] == 0
    cov = dict(covered, gaps=[{"start": "2023-03-01", "end": "2023-03-10"}])
    assert label_window(row, [], cov)[0] is None


def test_date_only_same_day_and_window_boundaries():
    row = {"cik": 1, "session": "2023-01-31", "as_of": "2023-01-31T21:30:00Z"}
    event = {"cik": 1, "announcement_date": "2023-01-31", "status": "verified", "event_id": "one"}
    assert label_window(row, [event], {})[1] == "same_day_date_only"
    event["announcement_at"] = "2023-01-31T22:00:00Z"
    assert label_window(row, [event], {})[0] == 1
    event.pop("announcement_at")
    event["announcement_date"] = "2023-04-01"  # 60 calendar days inclusive.
    assert label_window(row, [event], {}, 60)[0] == 1
    assert label_window(row, [event], {}, 30)[0] is None
    event["status"] = "candidate"
    assert label_window(row, [event], {}, 60)[0] is None


def test_release_date_is_not_8k_filing_date():
    text = "Company Announces Pricing of Public Offering of Common Stock. NEW YORK, July 28, 2025 — announced today."
    assert date_from_release(text) == "2025-07-28"
    triage = event_triage(text, "EX-99.1")
    assert triage["triage"] == "primary_equity_candidate"
    assert triage["stage"] == "pricing"
    assert "status" not in triage  # Classification is not manual confirmation.
    assert date_from_release("As filed on February 31, 2025") is None


def test_match_does_not_consult_outcome_and_enforces_calipers():
    common = dict(session="2024-01-31", stratum="health", assets=100e6, dollar_volume_20=10e6, return_60=.30)
    rows = [dict(common, cik=1, group="strong_short", outcome_60=1),
            dict(common, cik=2, group="strong_longer", outcome_60=None),
            dict(common, cik=3, group="strong_longer", stratum="other", outcome_60=0),
            dict(common, cik=4, group="strong_longer", assets=1e9, outcome_60=1)]
    assert match_controls(rows)[0]["control_cik"] == 2
    rows[1]["outcome_60"] = 1
    assert match_controls(rows)[0]["control_cik"] == 2


def test_rates_count_unique_events_and_keep_unknowns_out_of_denominator():
    rows = [{"cik": 1, "outcome_60": 1, "event_60": "same"},
            {"cik": 1, "outcome_60": 1, "event_60": "same"},
            {"cik": 2, "outcome_60": None}, {"cik": 3, "outcome_60": 0}]
    got = rate_summary(rows)
    assert got["labeled_subset_rate"] == pytest.approx(2/3)
    assert got["rate"] is None  # Never present positive-biased partial labels as incidence.
    assert got["incidence_bounds"] == [.5, .75]
    assert got["issuers"] == 2 and got["unique_events"] == 1 and got["unknown"] == 1
    assert got["interval"] is None  # Too few independent issuers.


def test_strata_and_group_reject_unknown_funding():
    assert stratum(2834) == "health"
    assert stratum(7372) == "technology"
    assert stratum(5000) == "other"
    assert group_name({"funding_group": "unknown", "tradable_filter": True, "setups": ["Fresh rally"]}) == "ineligible_or_unknown"
    assert historical_metrics(pd.DataFrame(), pd.DataFrame(), "2023-01-31")["price_status"] == "unavailable"


def test_same_day_announcement_before_signal_is_not_predicted():
    row = dict(cik=1, session="2025-06-30", as_of="2025-06-30T20:30:00Z", balance_date="2025-03-31")
    ev = dict(cik=1, event_id="one", status="verified", announcement_date="2025-06-30", announcement_at="2025-06-30T20:02:00Z")
    assert label_window(row, [ev], {})[0] is None
    assert intervening_announcement(ev, row)
    assert not intervening_announcement(dict(ev, announcement_at="2025-06-30T21:00:00Z"), row)
    assert not intervening_announcement(ev, dict(row, balance_date=None))
    assert intervening_announcement(ev, dict(row, session="2025-08-29", as_of="2025-08-29T20:30:00Z", balance_date="2025-06-30"))


def test_efts_contract_rejects_literal_amendments_but_covers_all_prospectuses():
    # Live probe showed literal 8-K/A zeroed an otherwise nonempty query.
    query = event_search_query(1285550, "2023-01-01", "2025-12-31")
    assert query["ciks"] == "0001285550"
    assert "8-K/A" not in query["forms"]
    assert set(query["forms"].split(",")) >= {"8-K", "424B1", "424B2", "424B3", "424B4", "424B5", "FWP"}


def test_frozen_sample_cannot_silently_change_after_outcomes(tmp_path):
    import hashlib
    (tmp_path / "cohort.csv").write_text("cik\n1\n")
    (tmp_path / "protocol.json").write_text('{"seed":"fixed"}')
    save(tmp_path / "cohort_manifest.json", {name + "_sha256": hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
         for name, filename in [("cohort", "cohort.csv"), ("protocol", "protocol.json")]})
    verify_frozen_inputs(tmp_path)
    (tmp_path / "cohort.csv").write_text("cik\n2\n")
    with pytest.raises(ValueError, match="Frozen cohort changed"):
        verify_frozen_inputs(tmp_path)
