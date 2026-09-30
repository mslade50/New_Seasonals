import pandas as pd
import pytest

from fundamental.sec_earnings import parse_earnings_8k
from earnings_calendar_provider import build_candidate, CalendarError

URL = "https://www.sec.gov/Archives/edgar/data/1/123/report.htm"
HTML = """<main>Item 2.02. Results of Operations and Financial Condition
On September 9, 2026, Example Corporation issued a press release announcing financial results for the fiscal quarter ended July 31, 2026.
Item 9.01 Financial Statements and Exhibits.</main>"""


def proof(raw=HTML):
    return parse_earnings_8k(raw, ticker="TEST", source_url=URL,
        accepted_at="2026-09-10T20:00:00Z", captured_at="2026-09-22T15:00:00Z")


def test_announcement_is_not_filing_date():
    row = proof()
    assert row["date"] == pd.Timestamp("2026-09-09")
    assert row["fiscalDateEnding"] == "2026-07-31"
    assert row["eps_actual"] is None


@pytest.mark.parametrize("raw", ["Date of earliest event reported: September 9, 2026",
    HTML.replace("issued", "will issue"), HTML.replace("financial results", "a conference call"),
    HTML.replace("September 9, 2026", "September 12, 2026"),
    HTML + HTML.replace("September 9, 2026", "September 8, 2026")])
def test_no_guessed_confirmation(raw):
    with pytest.raises(ValueError): proof(raw)


def inputs():
    prior = pd.DataFrame([
        dict(ticker="TEST", date="2026-09-08", eps_actual=None, fiscalDateEnding="2026-07-31", event_status="expected", event_source="alpha_vantage"),
        dict(ticker="MSFT", date="2026-07-29", eps_actual=4.27, event_status="confirmed", event_source="fmp_legacy"),
    ])
    alpha = pd.DataFrame([dict(ticker="TEST", date="2026-09-30", fiscalDateEnding="2026-07-31"),
                          dict(ticker="MSFT", date="2026-10-28", fiscalDateEnding="2026-09-30")])
    return prior, alpha


def test_primary_period_corrects_elapsed_expectation_without_faking_eps():
    prior, alpha = inputs()
    candidate, _ = build_candidate(prior, alpha, pd.DataFrame([proof()]), "2026-09-22", confirmation_provider="sec")
    test = candidate[candidate.ticker.eq("TEST")]
    assert len(test) == 1 and test.iloc[0].date == pd.Timestamp("2026-09-09")
    assert test.iloc[0].event_status == "confirmed" and pd.isna(test.iloc[0].eps_actual)
    assert candidate.loc[candidate.ticker.eq("MSFT") & candidate.date.eq("2026-07-29"), "eps_actual"].iloc[0] == 4.27


def test_date_only_proof_preserves_existing_financials():
    prior, alpha = inputs()
    prior.loc[0, ["date", "eps_actual", "event_status"]] = ["2026-09-09", 0, "confirmed"]
    candidate, _ = build_candidate(prior, alpha, pd.DataFrame([proof()]), "2026-09-22", confirmation_provider="sec")
    assert candidate.loc[candidate.ticker.eq("TEST"), "eps_actual"].iloc[0] == 0


def test_missing_or_duplicate_period_proof_blocks():
    prior, alpha = inputs()
    with pytest.raises(CalendarError): build_candidate(prior, alpha, pd.DataFrame(), "2026-09-22", confirmation_provider="sec")
    for change in [{"payload_digest": "bad"}, {"source_url": "https://example.com/fake"},
                   {"accepted_at": "2026-09-24T00:00:00Z"}, {"accepted_at": "2026-09-10"},
                   {"announcement_confirmed": False}]:
        with pytest.raises(CalendarError):
            build_candidate(prior, alpha, pd.DataFrame([proof() | change]), "2026-09-22", confirmation_provider="sec")
    with pytest.raises(CalendarError):
        build_candidate(prior, alpha, pd.DataFrame([proof(), proof()]), "2026-09-22", confirmation_provider="sec")
