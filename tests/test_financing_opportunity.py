"""Integrity tests for the research screen, not assertions of predictive edge."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fundamental.financing_opportunity import (
    ScreenPolicy, choose_financial_queue, join_funding, price_metrics, ticker_bars,
)
from scripts.build_financing_watchlist import last_completed_session


def history(n=260, growth=.008):
    dates = pd.bdate_range("2025-01-02", periods=n)
    close = 10 * (1 + growth) ** np.arange(n)
    return pd.DataFrame({"Open":close, "Close":close, "Adj Close":close,
                         "Volume":1_000_000., "Stock Splits":0.}, index=dates)


def metric(d, benchmark=None):
    return price_metrics(d, benchmark if benchmark is not None else history(len(d), 0), session=d.index[-1])


def test_returns_and_relative_returns_use_matching_sessions():
    d, b = history(), history(growth=.001)
    r = metric(d, b)
    assert r["return_60"] == pytest.approx(1.008 ** 60 - 1)
    assert r["relative_60"] == pytest.approx(1.008 ** 60 - 1.001 ** 60)
    assert r["setups"] == ["Sustained strength"]


def test_current_day_not_in_relative_volume_denominator():
    d = history(growth=0)
    d.loc[d.index[-1], ["Open", "Close", "Adj Close"]] = 12
    d.loc[d.index[-1], "Volume"] = 2_000_000
    r = metric(d)
    assert r["volume_ratio"] == 2
    assert r["max_gap_5"] == pytest.approx(.2)
    assert r["setups"] == ["Fresh rally"]


def test_adjusted_returns_and_unadjusted_dollar_turnover_are_separate():
    d = history(growth=0)
    d["Adj Close"] = np.linspace(9, 10, len(d))
    r = metric(d)
    assert r["dollar_volume_20"] == 10_000_000
    assert r["return_20"] > 0
    assert r["max_gap_5"] == 0


def test_future_bar_does_not_enter_signal():
    d = history()
    session = d.index[-2]
    r = price_metrics(d, history(growth=0), session=session)
    d.loc[d.index[-1], ["Open", "Close", "Adj Close"]] = 100_000
    assert price_metrics(d, history(growth=0), session=session) == r


@pytest.mark.parametrize("damage", ["stale", "interior", "duplicate", "short", "zero"])
def test_bad_coverage_fails_closed(damage):
    d, b = history(), history(growth=0)
    if damage == "stale": d = d.iloc[:-1]
    if damage == "interior": d = d.drop(d.index[-30])
    if damage == "duplicate": d = pd.concat([d, d.tail(1)])
    if damage == "short": d = d.tail(119)
    if damage == "zero": d.loc[d.index[-10], "Close"] = 0
    r = price_metrics(d, b, session=b.index[-1])
    assert r["price_status"] == "unavailable"
    assert not r["setups"]


def test_liquidity_and_price_floor_even_when_rallying():
    d = history()
    d["Volume"] = 100
    assert not metric(d)["setups"]
    d["Volume"] = 100_000_000
    d[["Open", "Close", "Adj Close"]] /= 100
    assert not metric(d)["setups"]


def test_multiindex_both_orders():
    d = history()
    outer = pd.concat({"ABC":d, "XYZ":d * 2}, axis=1)
    pd.testing.assert_frame_equal(ticker_bars(outer, "ABC"), d)
    pd.testing.assert_frame_equal(ticker_bars(outer.swaplevel(axis=1), "ABC"), d)
    assert ticker_bars(outer, "MISSING").empty


def test_no_burn_missing_and_stale_are_not_short_runway():
    row = dict(setups=["Fresh rally"])
    base = dict(status="calculated", balance_age_days=80, monthly_burn_6m=0,
                runway_6m=None, runway_with_capex_6m=None)
    assert join_funding(row, None)["funding_status"] == "Not checked"
    assert join_funding(row, base)["funding_status"] == "No operating burn observed"
    assert not join_funding(row, dict(base, runway_6m=2, balance_age_days=151))["candidate"]
    assert not join_funding(row, dict(base, runway_6m=2, status="partial"))["candidate"]
    assert join_funding(row, dict(base, runway_6m=24))["candidate"]
    assert not join_funding(row, dict(base, runway_6m=24.01))["candidate"]
    assert join_funding(row, dict(base, runway_with_capex_6m=12))["funding_status"].startswith("Capex")


def test_cohort_queue_deduplicates_and_balances():
    rows = [dict(ticker="A",setups=["Sustained strength", "Fresh rally"],return_20=1),
            dict(ticker="B",setups=["Sustained strength"],return_20=.9),
            dict(ticker="C",setups=["Fresh rally"],return_20=.8),
            dict(ticker="D",setups=[],return_20=2)]
    assert choose_financial_queue(rows, 3) == ["A", "B", "C"]
    assert len(choose_financial_queue(rows, 2)) == 2


def test_session_cutoff_handles_holidays_early_close_and_buffer():
    assert last_completed_session("2026-09-22T19:00:00Z") == "2026-09-21"
    assert last_completed_session("2026-09-22T20:29:59Z") == "2026-09-21"
    assert last_completed_session("2026-09-22T20:30:00Z") == "2026-09-22"
    assert last_completed_session("2026-07-03T22:00:00Z") == "2026-07-02"
    assert last_completed_session("2026-11-27T18:30:00Z") == "2026-11-27"


def review_fixture():
    rows = [dict(ticker="ABC", cik=123, financial=dict(balance_date="2026-06-30", cash=100))]
    manifest = dict(as_of="2026-09-22T20:30:00+00:00")
    review = dict(ticker="ABC", cik=123, as_of=manifest["as_of"], balance_date="2026-06-30",
                  expected_financials=dict(cash=100),
                  sources=[dict(url="https://www.sec.gov/example", published_date="2026-08-01")])
    return rows, manifest, review


@pytest.mark.parametrize("mutation", ["issuer", "period", "value", "future", "duplicate"])
def test_manual_reviews_reject_inconsistent_evidence(mutation):
    from fundamental.financing_opportunity_report import apply_reviews
    rows, manifest, review = review_fixture()
    reviews = [review]
    if mutation == "issuer": review["cik"] = 456
    if mutation == "period": review["balance_date"] = "2026-03-31"
    if mutation == "value": review["expected_financials"]["cash"] = 999
    if mutation == "future": review["sources"][0]["published_date"] = "2026-09-23"
    if mutation == "duplicate": reviews.append(dict(review))
    with pytest.raises(ValueError):
        apply_reviews(rows, reviews, manifest)


def test_same_day_source_needs_accepted_timestamp():
    from fundamental.financing_opportunity_report import apply_reviews
    rows, manifest, review = review_fixture()
    source = review["sources"][0]
    source["published_date"] = "2026-09-22"
    with pytest.raises(ValueError):
        apply_reviews(rows, [review], manifest)
    source["published_at"] = "2026-09-22T20:31:00+00:00"
    with pytest.raises(ValueError):
        apply_reviews(rows, [review], manifest)
    source["published_at"] = "2026-09-22T20:29:00+00:00"
    assert apply_reviews(rows, [review], manifest)[0]["review"] == review


def test_unknown_observations_remain_unknown_and_frozen(tmp_path):
    import csv
    from fundamental.financing_opportunity_report import freeze_observations
    path = tmp_path / "observations.csv"
    row = dict(ticker="ABC", cik=123, setups=["Fresh rally"], candidate=False,
               financial=dict(status="unavailable"), funding_status="Financial coverage gap")
    manifest = dict(as_of="2026-09-22T20:30:00+00:00", price_session="2026-09-22")
    freeze_observations(path, [row], manifest)
    with path.open() as handle:
        record = next(csv.DictReader(handle))
    assert record["funding_flag"] == ""
    assert record["funding_status"] == "Financial coverage gap"
    # User-written future outcome survives an otherwise identical report refresh.
    path.write_text(path.read_text().replace("Pending; no follow-up performed", "Follow-up completed"))
    freeze_observations(path, [row], manifest)
    assert "Follow-up completed" in path.read_text()
    row["funding_status"] = "Longer reported runway"
    with pytest.raises(ValueError, match="Frozen observations"):
        freeze_observations(path, [row], manifest)
