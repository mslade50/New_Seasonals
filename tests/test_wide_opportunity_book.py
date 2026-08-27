from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from research.opportunity_book import (
    OpportunityConfig,
    build_opportunity_book,
    normalize_prices,
    write_opportunity_book,
)


def synthetic_prices(
    tickers: list[str], periods: int = 70, identical: bool = False
) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=periods)
    n_tickers, n_dates = len(tickers), len(dates)
    ticker_idx = np.repeat(np.arange(n_tickers), n_dates)
    day_idx = np.tile(np.arange(n_dates), n_tickers)
    phase = np.zeros_like(ticker_idx, dtype=float) if identical else ticker_idx * 0.031
    drift = 0.0012 * day_idx
    wave = 0.018 * np.sin(day_idx / 8.0 + phase)
    cross_section = 0.0 if identical else ticker_idx * 0.0007
    close = 40.0 * (1.0 + drift + wave + cross_section)
    # Build an open series without an expensive per-ticker groupby.
    open_ = close * (1.0 + 0.0015 * np.sin(day_idx / 3.0 + phase))
    volume = 800_000.0 + 2_000.0 * day_idx + (0.0 if identical else 700.0 * ticker_idx)
    frame = pd.DataFrame(
        {
            "date": np.tile(dates.to_numpy(), n_tickers),
            "ticker": np.repeat(np.asarray(tickers), n_dates),
            "Open": open_,
            "High": np.maximum(open_, close) * 1.008,
            "Low": np.minimum(open_, close) * 0.992,
            "Close": close,
            "Volume": volume,
        }
    )
    # Deterministic last-day participation shocks create non-degenerate ranks.
    last = frame.groupby("ticker", sort=False).tail(1).index
    if not identical:
        frame.loc[last, "Volume"] *= 1.0 + (np.arange(n_tickers) % 9) / 4.0
    return frame


def _config(asof, **overrides) -> OpportunityConfig:
    values = {
        "asof": asof,
        "review_limit": 75,
        "deep_test_limit": 10,
        "audit_limit": 10,
        "audit_seed": 91,
        "min_history": 63,
        "max_stale_sessions": 0,
        "market_ticker": "SPY",
    }
    values.update(overrides)
    return OpportunityConfig(**values)


def test_processes_every_name_in_a_1025_ticker_universe():
    tickers = ["SPY"] + [f"T{i:04d}" for i in range(1024)]
    prices = synthetic_prices(tickers)
    asof = prices["date"].max()

    result = build_opportunity_book(prices, tickers, _config(asof))

    assert result.manifest["coverage"]["requested_count"] == 1025
    assert result.manifest["coverage"]["eligible_count"] == 1025
    assert len(result.coverage) == 1025
    assert set(result.coverage["Ticker"]) == set(tickers)
    assert result.coverage["First_Rejection"].isna().all()
    assert len(result.features) == 1025


def test_every_requested_ticker_gets_a_first_coverage_verdict():
    prices = synthetic_prices(["SPY", "FULL"], periods=70)
    short = synthetic_prices(["SHORT"], periods=15)
    prices = pd.concat([prices, short], ignore_index=True)
    tickers = ["SPY", "FULL", "SHORT", "MISSING"]

    result = build_opportunity_book(
        prices,
        tickers,
        _config(prices["date"].max(), review_limit=4, deep_test_limit=2, audit_limit=1),
    )
    coverage = result.coverage.set_index("Ticker")

    assert coverage.loc["FULL", "Research_Status"] == "ELIGIBLE"
    assert coverage.loc["SHORT", "First_Rejection"] == "INSUFFICIENT_HISTORY"
    assert coverage.loc["MISSING", "First_Rejection"] == "NO_PRICE_ROWS"
    assert result.manifest["coverage"]["first_rejection_counts"] == {
        "INSUFFICIENT_HISTORY": 1,
        "NO_PRICE_ROWS": 1,
    }


def test_ties_and_seeded_audit_sample_are_deterministic():
    tickers = ["SPY"] + [f"EQ{i:02d}" for i in range(30)]
    prices = synthetic_prices(tickers, identical=True)
    config = _config(
        prices["date"].max(),
        review_limit=12,
        deep_test_limit=5,
        audit_limit=7,
        audit_seed=2026,
    )

    first = build_opportunity_book(prices, tickers, config)
    second = build_opportunity_book(
        prices.sample(frac=1.0, random_state=8), list(reversed(tickers)), config
    )

    assert (
        first.review_queue["Ticker"].tolist() == second.review_queue["Ticker"].tolist()
    )
    assert (
        first.audit_sample["Ticker"].tolist() == second.audit_sample["Ticker"].tolist()
    )
    ranks = first.features.set_index("Ticker")
    # Equal-score ties resolve alphabetically, including SPY in its sorted slot.
    top = ranks["archetype_participation_shock_rank"].sort_values().index[0]
    assert top == min(tickers)


def test_future_rows_never_change_asof_features_or_queues():
    tickers = ["SPY"] + [f"X{i:02d}" for i in range(18)]
    full = synthetic_prices(tickers, periods=90)
    asof = sorted(full["date"].unique())[72]
    clipped = full[full["date"] <= asof].copy()
    config = _config(
        asof, review_limit=10, deep_test_limit=5, audit_limit=4, min_history=50
    )

    with_future = build_opportunity_book(full, tickers, config)
    without_future = build_opportunity_book(clipped, tickers, config)

    comparable = [c for c in with_future.features.columns if c != "sector_label"]
    assert_frame_equal(
        with_future.features[comparable],
        without_future.features[comparable],
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert with_future.review_queue.to_dict(
        "records"
    ) == without_future.review_queue.to_dict("records")
    assert with_future.audit_sample.to_dict(
        "records"
    ) == without_future.audit_sample.to_dict("records")
    assert with_future.manifest["source"]["future_rows_discarded"] > 0
    assert without_future.manifest["source"]["future_rows_discarded"] == 0
    assert with_future.manifest["source"]["price_reference_bar"] == str(
        pd.Timestamp(asof).date()
    )


def test_queue_caps_and_complete_local_artifact_bundle(tmp_path):
    tickers = ["SPY"] + [f"Z{i:02d}" for i in range(35)]
    prices = synthetic_prices(tickers)
    result = build_opportunity_book(
        prices,
        tickers,
        _config(prices["date"].max(), review_limit=9, deep_test_limit=4, audit_limit=3),
    )
    paths = write_opportunity_book(result, tmp_path / "wide")

    assert len(result.review_queue) == 9
    assert len(result.deep_test_queue) == 4
    assert len(result.audit_sample) == 3
    assert set(paths) == {
        "manifest",
        "coverage",
        "features",
        "review",
        "deep_test",
        "audit",
        "html",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert payload["research_only"] is True
    html = paths["html"].read_text(encoding="utf-8")
    assert "NOT AN INVESTMENT RECOMMENDATION" in html
    assert "there is no universal score" in payload["selection"]["method"]


def test_outputs_have_no_execution_contract_fields_or_actions():
    tickers = ["SPY"] + [f"SAFE{i}" for i in range(12)]
    prices = synthetic_prices(tickers)
    result = build_opportunity_book(
        prices,
        tickers,
        _config(prices["date"].max(), review_limit=7, deep_test_limit=3, audit_limit=2),
    )

    keys: set[str] = set()

    def walk(value):
        if isinstance(value, dict):
            keys.update(str(k).casefold() for k in value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(result.manifest)
    forbidden = {
        "action",
        "quantity",
        "order_type",
        "entry_type",
        "limit_price",
        "stop_price",
        "target_price",
        "risk_amt",
        "notional",
        "scan_source",
        "approve",
        "place_pass",
        "execute_on",
    }
    assert not keys & forbidden
    assert set(result.review_queue["Research_Priority"]) == {"REVIEW"}
    assert set(result.deep_test_queue["Research_Priority"]) == {"DEEP_TEST"}


def test_missing_ohlcv_fields_degrade_features_without_dropping_close_history():
    tickers = ["SPY", "CLOSEONLY", "COMPLETE"]
    prices = synthetic_prices(tickers).drop(columns=["Open", "High", "Low", "Volume"])

    result = build_opportunity_book(
        prices,
        tickers,
        _config(prices["date"].max(), review_limit=3, deep_test_limit=2, audit_limit=1),
    )
    coverage = result.coverage.set_index("Ticker")
    features = result.features.set_index("Ticker")

    assert coverage.loc["CLOSEONLY", "Research_Status"] == "ELIGIBLE"
    assert "gap_1d" in coverage.loc["CLOSEONLY", "Missing_Features"]
    assert pd.isna(features.loc["CLOSEONLY", "atr_14"])
    assert pd.notna(features.loc["CLOSEONLY", "ret_21d"])
    assert len(result.review_queue) == 3


def test_yfinance_price_ticker_multiindex_is_normalized():
    dates = pd.bdate_range("2026-01-02", periods=3)
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["SPY", "AAPL"]],
        names=["Price", "Ticker"],
    )
    values = (
        np.arange(len(dates) * len(columns), dtype=float).reshape(len(dates), -1) + 20
    )
    raw = pd.DataFrame(values, index=dates, columns=columns)

    normalized = normalize_prices(raw)

    assert set(normalized["Ticker"]) == {"SPY", "AAPL"}
    assert list(normalized.columns) == [
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
    assert len(normalized) == 6


def test_yfinance_ticker_price_orientation_and_case_are_normalized():
    dates = pd.bdate_range("2026-01-02", periods=3)
    columns = pd.MultiIndex.from_product(
        [["spy", "aapl"], ["Open", "High", "Low", "Close", "Volume"]],
        names=["Ticker", "Price"],
    )
    values = (
        np.arange(len(dates) * len(columns), dtype=float).reshape(len(dates), -1) + 10
    )

    normalized = normalize_prices(pd.DataFrame(values, index=dates, columns=columns))

    assert set(normalized["Ticker"]) == {"SPY", "AAPL"}
    assert len(normalized) == 6
