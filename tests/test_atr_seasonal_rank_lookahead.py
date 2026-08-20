"""Point-in-time guards for ATR-normalized seasonal ranks.

The target year's prices are out of sample. Changing any of them must leave
the complete rank surface for that target year unchanged at every horizon.
"""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import build_atr_seasonal_ranks as rank_builder
from atr_seasonal_contract import RANK_METHOD_COLUMN, RANK_METHOD_VERSION
from build_atr_seasonal_ranks import (
    FWD_WINDOWS,
    compute_ranks_for_year,
    filter_retired_tickers,
    generate_trading_dates,
    normalize_ticker,
    prepare_ticker_data,
)


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2014-01-02", "2021-12-31")
    x = np.arange(len(dates), dtype=float)
    close = 80.0 + 0.025 * x + 2.5 * np.sin(x / 17.0) + 0.7 * np.cos(x / 41.0)
    return pd.DataFrame(
        {
            "Open": close * (1.0 + 0.001 * np.sin(x / 7.0)),
            "High": close + 1.1 + 0.1 * np.cos(x / 9.0),
            "Low": close - 1.0 - 0.1 * np.sin(x / 11.0),
            "Close": close,
            "Volume": 1_000_000.0 + x,
        },
        index=dates,
    )


def test_target_year_price_mutation_cannot_change_any_rank_horizon():
    original = _prices()
    mutated = original.copy()
    in_target = mutated.index.year == 2021

    # Before the fix, late-2020 origins referenced these perturbed prices and
    # moved every target-year rank family.
    mutated.loc[in_target, ["Open", "High", "Low", "Close"]] *= 7.0

    ranks_original = compute_ranks_for_year(prepare_ticker_data(original), 2021)
    ranks_mutated = compute_ranks_for_year(prepare_ticker_data(mutated), 2021)

    assert ranks_original is not None
    assert list(ranks_original.columns) == [f"atr_sznl_{w}d" for w in FWD_WINDOWS]
    pdt.assert_frame_equal(ranks_original, ranks_mutated, check_exact=True)


def test_merge_path_rejects_a_legacy_artifact_with_the_contract_error(monkeypatch, tmp_path):
    output = tmp_path / "atr_seasonal_ranks.parquet"
    pd.DataFrame({"Date": [pd.Timestamp("2020-01-02")], "ticker": ["OLD"]}).to_parquet(
        output, index=False
    )
    monkeypatch.setattr(
        rank_builder,
        "load_master_prices_cache",
        lambda _tickers: {"TEST": _prices()},
    )
    monkeypatch.setattr(rank_builder, "load_overflow_cache", lambda: {})

    with pytest.raises(RuntimeError, match="refusing to merge corrected rows"):
        rank_builder.build_atr_ranks(
            ["TEST"],
            [2021],
            output_path=str(output),
            merge=True,
            allow_download=False,
        )


def test_no_download_fails_when_any_requested_price_source_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(rank_builder, "load_master_prices_cache", lambda _tickers: {})
    monkeypatch.setattr(rank_builder, "load_overflow_cache", lambda: {})

    with pytest.raises(RuntimeError, match="complete source coverage"):
        rank_builder.build_atr_ranks(
            ["MISSING"],
            [2021],
            output_path=str(tmp_path / "unused.parquet"),
            allow_download=False,
        )


def test_rank_calendar_uses_versioned_nyse_special_closures():
    dates = generate_trading_dates(2025)["Date"]

    assert len(dates) == 250
    assert pd.Timestamp("2025-01-09") not in set(dates)


def test_reviewed_legacy_retirements_are_filtered_from_repair_universe():
    kept, retired = filter_retired_tickers(["AAPL", "THS", "^SOX"])

    assert kept == ["AAPL"]
    assert retired == ["THS", "^SOX"]


def test_ticker_normalization_preserves_yahoo_suffix_symbols():
    assert normalize_ticker("BRK.B") == "BRK-B"
    assert normalize_ticker("DX-Y.NYB") == "DX-Y.NYB"


def test_master_cache_loader_can_reach_yahoo_suffix_symbol(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        [{
            "ticker": "DX-Y.NYB",
            "date": pd.Timestamp("2026-08-19"),
            "Open": 98.0,
            "High": 99.0,
            "Low": 97.0,
            "Close": 98.5,
            "Volume": 1_000.0,
        }]
    ).to_parquet(data_dir / "master_prices.parquet", index=False)
    monkeypatch.setattr(rank_builder, "current_dir", str(tmp_path))

    loaded = rank_builder.load_master_prices_cache(["DX-Y.NYB"])

    assert set(loaded) == {"DX-Y.NYB"}


def test_versioned_merge_preserves_existing_ticker_and_adds_requested_ticker(
    monkeypatch, tmp_path
):
    output = tmp_path / "atr_seasonal_ranks.parquet"
    old = {
        "Date": pd.Timestamp("2020-01-02"),
        "ticker": "OLD",
        RANK_METHOD_COLUMN: RANK_METHOD_VERSION,
    }
    old.update({f"atr_sznl_{window}d": 50.0 for window in FWD_WINDOWS})
    pd.DataFrame([old]).to_parquet(output, index=False)
    monkeypatch.setattr(
        rank_builder,
        "load_master_prices_cache",
        lambda _tickers: {"TEST": _prices()},
    )
    monkeypatch.setattr(rank_builder, "load_overflow_cache", lambda: {})

    rank_builder.build_atr_ranks(
        ["TEST"],
        [2021],
        output_path=str(output),
        merge=True,
        allow_download=False,
    )

    merged = pd.read_parquet(output)
    assert set(merged["ticker"]) == {"OLD", "TEST"}
    assert set(merged[RANK_METHOD_COLUMN]) == {RANK_METHOD_VERSION}
