import json
import pandas as pd
import pytest

from fundamental.sec_statements import annual_statement_bundle, filing_news

NOW = "2026-09-22T15:00:00Z"


def fixture():
    def fact(value, **kw):
        return dict(val=value, start="2025-01-01", end="2025-12-31", form="10-K", accn="a", fy=2026, **kw)
    facts = {"cik": 1, "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [fact(100)]}},
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [fact(20)]}},
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [fact(5)]}},
        "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": [fact(10)]}},
    }}}
    submissions = {"cik": "1", "filings": {"recent": {"accessionNumber": ["a"],
        "acceptanceDateTime": ["2026-02-01T22:00:00Z"], "form": ["10-K"], "primaryDocument": ["annual.htm"], "items": [""]}}}
    return facts, submissions


def build(facts, submissions, **kwargs):
    return annual_statement_bundle(facts, submissions, ticker="TEST", as_of=kwargs.get("as_of", NOW), digest="abc")


def test_fiscal_dates_capex_sign_missing_debt_and_units():
    facts, submissions = fixture()
    bundle, report = build(facts, submissions)
    assert bundle.date.dt.year.unique().tolist() == [2025]  # fy is a filing year, not the observation year
    cash = bundle[bundle.endpoint.eq("cash-flow-statement")].iloc[0]
    assert cash.capitalExpenditure == -5 and cash.freeCashFlow == 15
    assert bundle.totalDebt.isna().all() and bundle.ebitda.isna().all()
    assert not report["minimum_history_ok"]
    assert "accession" in json.loads(cash.fact_provenance)["capitalExpenditure"]


def test_unknown_or_future_acceptance_never_becomes_historical_fact():
    facts, submissions = fixture()
    with pytest.raises(ValueError): build(facts, submissions, as_of="2026-01-31T23:59:59Z")
    submissions["filings"]["recent"]["acceptanceDateTime"] = [""]
    with pytest.raises(ValueError): build(facts, submissions)


def test_quarters_and_foreign_currency_do_not_become_annual_usd():
    facts, submissions = fixture()
    facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"][0]["start"] = "2025-10-01"
    with pytest.raises(ValueError): build(facts, submissions)
    facts, submissions = fixture()
    facts["facts"]["us-gaap"]["Revenues"]["units"]["EUR"] = facts["facts"]["us-gaap"]["Revenues"]["units"].pop("USD")
    with pytest.raises(ValueError): build(facts, submissions)


def test_conflicting_same_filing_fact_fails():
    facts, submissions = fixture()
    records = facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    records.append(records[0] | {"val": 200})
    with pytest.raises(ValueError, match="ambiguous"): build(facts, submissions)


def test_latest_accepted_restatement_wins_before_cutoff():
    facts, submissions = fixture()
    records = facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    records.append(records[0] | {"val": 90, "accn": "b"})
    recent = submissions["filings"]["recent"]
    recent["accessionNumber"].append("b"); recent["acceptanceDateTime"].append("2026-08-01T20:00:00Z")
    b, _ = build(facts, submissions)
    assert b.revenue.dropna().iloc[0] == 90
    b, _ = build(facts, submissions, as_of="2026-07-01T00:00:00Z")
    assert b.revenue.dropna().iloc[0] == 100


def test_filing_news_is_not_earnings_confirmation():
    facts, submissions = fixture()
    recent = submissions["filings"]["recent"]
    recent["acceptanceDateTime"] = ["2026-09-01T20:00:00Z"]
    recent["form"] = ["8-K"]; recent["items"] = ["2.02"]
    rows = filing_news(submissions, ticker="TEST", as_of=NOW)
    assert rows[0]["source_scope"] == "filings_only"
    assert "eps_actual" not in rows[0] and "announcement_date" not in rows[0]


def test_mismatched_annual_start_is_not_combined_with_revenue():
    facts, submissions = fixture()
    facts["facts"]["us-gaap"]["NetCashProvidedByUsedInOperatingActivities"]["units"]["USD"][0]["start"] = "2025-01-15"
    bundle, report = build(facts, submissions)
    assert bundle.operatingCashFlow.isna().all()
