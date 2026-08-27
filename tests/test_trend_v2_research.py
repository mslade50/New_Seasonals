from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from research.trend_v2.config import (
    FROZEN_BENCHMARK,
    FROZEN_BENCHMARK_UNIVERSE,
    PREREGISTERED_CROSS_SECTIONAL_SPEC,
    PREREGISTERED_MULTISPEED_SPECS,
)
from research.trend_v2.engine import (
    PriceData,
    apply_turnover_controls,
    backtest_next_period,
    cross_sectional_targets,
    frozen_benchmark_targets,
    multispeed_targets,
    normalize_sector_history,
    sector_neutral_percentile_ranks,
    votes_to_hysteresis,
)
from research.trend_v2.runner import write_research_artifacts


def _etf_prices(periods: int = 900) -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-02", periods=periods)
    wave = np.sin(np.arange(periods) / 17.0) * 0.001
    data = {}
    for number, ticker in enumerate(FROZEN_BENCHMARK_UNIVERSE):
        daily_return = 0.00015 + number * 0.00001 + wave * (1.0 + number / 30.0)
        data[ticker] = 50.0 * np.cumprod(1.0 + daily_return)
    return pd.DataFrame(data, index=dates)


def test_frozen_benchmark_invariance_matches_locked_rules():
    assert FROZEN_BENCHMARK_UNIVERSE == (
        "SPY", "QQQ", "IWM", "EFA", "EEM", "FXI", "VNQ",
        "GLD", "SLV", "DBC", "TLT", "LQD",
    )
    assert FROZEN_BENCHMARK.momentum_lookback_months == 12
    assert FROZEN_BENCHMARK.momentum_skip_months == 1
    assert FROZEN_BENCHMARK.moving_average_months == 10
    assert FROZEN_BENCHMARK.asset_weight_cap == 0.20

    close = _etf_prices()
    actual = frozen_benchmark_targets(close).targets

    monthly = close.resample("ME").last()
    momentum = monthly.shift(1) / monthly.shift(12) - 1.0
    above_ma = monthly > monthly.rolling(10).mean()
    eligible = monthly.notna().rolling(13).count() >= 13
    signal = momentum.gt(0.0) & above_ma & eligible
    volatility = (
        close.pct_change(fill_method=None)
        .rolling(63)
        .std()
        .mul(np.sqrt(252.0))
        .resample("ME")
        .last()
        .clip(lower=0.04)
    )
    inverse = (1.0 / volatility).where(eligible, 0.0)
    slots = inverse.div(inverse.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    expected = slots.clip(upper=0.20).mul(signal.astype(float)).fillna(0.0)
    pdt.assert_frame_equal(actual, expected)


def test_multispeed_signals_do_not_change_when_future_prices_change():
    close = _etf_prices()
    spec = PREREGISTERED_MULTISPEED_SPECS[0]
    original = multispeed_targets(close, spec).targets
    mutation_start = close.index[-100]
    mutated = close.copy()
    mutated.loc[mutation_start:, ["SPY", "QQQ"]] *= np.linspace(1.0, 4.0, 100)[:, None]
    changed = multispeed_targets(mutated, spec).targets
    last_safe_month = mutation_start.to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(1)
    pdt.assert_frame_equal(original.loc[:last_safe_month], changed.loc[:last_safe_month])


def test_hysteresis_and_turnover_controls_are_path_dependent_and_bounded():
    index = pd.date_range("2024-01-31", periods=6, freq="ME")
    votes = pd.DataFrame({"A": [0, 2, 1, 1, 0, 3]}, index=index)
    eligible = pd.DataFrame(True, index=index, columns=["A"])
    state = votes_to_hysteresis(votes, eligible, enter_votes=2, exit_votes=0)
    assert state["A"].tolist() == [False, True, True, True, False, True]

    desired = pd.DataFrame({"A": [0.0, 0.30, 0.305, 0.90, 0.90, 0.0]}, index=index)
    controlled = apply_turnover_controls(
        desired, rebalance_band=0.01, max_turnover=0.30
    )
    assert controlled.iloc[1, 0] == pytest.approx(0.30)
    assert controlled.iloc[2, 0] == pytest.approx(0.30)  # 50 bp delta is inside band
    assert controlled.iloc[3, 0] == pytest.approx(0.60)  # discretionary move capped
    assert controlled.iloc[-1, 0] == 0.0  # risk exit is never throttled


def test_sector_neutral_ranks_and_equal_sector_budgets():
    one_date = pd.DatetimeIndex(["2024-01-31"])
    columns = ["A1", "A2", "A3", "B1", "B2", "B3"]
    scores = pd.DataFrame([[1, 2, 3, 10, 20, 30]], index=one_date, columns=columns)
    sectors = pd.DataFrame(
        [["A", "A", "A", "B", "B", "B"]], index=one_date, columns=columns
    )
    ranks = sector_neutral_percentile_ranks(scores, sectors)
    assert ranks.loc[one_date[0], "A3"] == 1.0
    assert ranks.loc[one_date[0], "B3"] == 1.0
    assert ranks.loc[one_date[0], "A1"] == pytest.approx(1.0 / 3.0)
    assert ranks.loc[one_date[0], "B1"] == pytest.approx(1.0 / 3.0)

    dates = pd.bdate_range("2019-01-02", periods=420)
    rng = np.random.default_rng(7)
    market_returns = 0.0001 + 0.0015 * np.sin(np.arange(len(dates)) / 11.0)
    market = pd.Series(100.0 * np.cumprod(1.0 + market_returns), index=dates)
    stock_data = {}
    alphas = {"A1": 0.00005, "A2": 0.00010, "A3": 0.00030,
              "B1": 0.00004, "B2": 0.00009, "B3": 0.00028}
    for ticker, alpha in alphas.items():
        noise = rng.normal(0.0, 0.00005, len(dates))
        stock_data[ticker] = 40.0 * np.cumprod(1.0 + market_returns + alpha + noise)
    stock_close = pd.DataFrame(stock_data, index=dates)
    sector_history = pd.DataFrame(
        {
            "date": [dates[0]] * 6,
            "ticker": columns,
            "sector": ["A", "A", "A", "B", "B", "B"],
        }
    )
    spec = replace(
        PREREGISTERED_CROSS_SECTIONAL_SPEC,
        beta_lookback_days=20,
        residual_momentum_days=20,
        volatility_days=20,
        min_history_days=60,
        asset_weight_cap=0.60,
        max_monthly_turnover=2.0,
    )
    result = cross_sectional_targets(stock_close, market, sector_history, spec)
    last = result.targets.loc[result.targets.abs().sum(axis=1).gt(0.0)].iloc[-1]
    assert last[["A1", "A2", "A3"]].sum() == pytest.approx(0.5, abs=1e-8)
    assert last[["B1", "B2", "B3"]].sum() == pytest.approx(0.5, abs=1e-8)


def test_sector_history_does_not_backfill_a_future_classification():
    signal_dates = pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-31"])
    history = pd.DataFrame(
        {
            "date": ["2024-02-15", "2024-03-15"],
            "ticker": ["A", "A"],
            "sector": ["OLD", "NEW"],
        }
    )
    panel = normalize_sector_history(history, signal_dates, ["A"])
    assert pd.isna(panel.loc["2024-01-31", "A"])
    assert panel.loc["2024-02-29", "A"] == "OLD"
    assert panel.loc["2024-03-31", "A"] == "NEW"


def test_cost_subtraction_is_exact_and_does_not_change_gross_returns():
    index = pd.date_range("2020-01-31", periods=12, freq="ME")
    close = pd.DataFrame({"A": 100.0 * 1.01 ** np.arange(12)}, index=index)
    targets = pd.DataFrame({"A": [0.0, 0.5, 0.5, 0.0] * 3}, index=index)
    zero = backtest_next_period(targets, close, cost_bps_per_side=0.0)
    ten = backtest_next_period(targets, close, cost_bps_per_side=10.0)
    common = zero.monthly.index.intersection(ten.monthly.index)
    pdt.assert_series_equal(
        zero.monthly.loc[common, "gross_return"],
        ten.monthly.loc[common, "gross_return"],
    )
    expected_drag = ten.monthly.loc[common, "turnover"] * 0.001
    actual_drag = (
        zero.monthly.loc[common, "net_return"]
        - ten.monthly.loc[common, "net_return"]
    )
    pdt.assert_series_equal(actual_drag, expected_drag, check_names=False)


def test_artifact_writer_refuses_source_paths_and_reports_trial_count(tmp_path):
    close = _etf_prices()
    price_data = PriceData(close=close, open=None, source=None)
    benchmark_target = frozen_benchmark_targets(close)
    benchmark_backtest = backtest_next_period(
        benchmark_target.targets, close, cost_bps_per_side=5.0
    )
    from research.trend_v2.runner import TrialRun
    from research.trend_v2.engine import performance_summary

    run = TrialRun(
        name=FROZEN_BENCHMARK.name,
        family="frozen_benchmark_not_a_candidate_trial",
        specification={},
        targets=benchmark_target,
        backtest=benchmark_backtest,
        summary=performance_summary(benchmark_backtest),
    )
    with pytest.raises(ValueError):
        write_research_artifacts(
            output_dir="research/trend_v2/forbidden_output",
            prices=price_data,
            runs=[run],
            stock_family_requested=False,
        )

    output = tmp_path / "artifacts" / "trend_test"
    written = write_research_artifacts(
        output_dir=output,
        prices=price_data,
        runs=[run],
        stock_family_requested=False,
    )
    manifest = pd.read_json(written / "manifest.json", typ="series")
    assert manifest["research_only"] is True
    assert manifest["production_writes"] is False
    assert manifest["trial_accounting"]["candidate_trials_executed"] == 0
