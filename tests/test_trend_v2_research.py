from __future__ import annotations

import json
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
    BacktestResult,
    PriceData,
    TargetResult,
    _compound_cagr,
    backtest_next_period,
    cross_sectional_targets,
    frozen_benchmark_targets,
    multispeed_targets,
    normalize_membership_history,
    normalize_sector_history,
    performance_summary,
    sector_neutral_percentile_ranks,
    transition_from_drifted_weights,
    votes_to_hysteresis,
)
from research.trend_v2.runner import (
    TrialRun,
    _safe_child,
    write_research_artifacts,
)
from scripts.run_trend_v2_research import ROOT, _resolve_project_artifact_output
from trading_calendar import TRADING_DAY


def _etf_prices(periods: int = 900) -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-02", periods=periods)
    wave = np.sin(np.arange(periods) / 17.0) * 0.001
    data = {}
    for number, ticker in enumerate(FROZEN_BENCHMARK_UNIVERSE):
        daily_return = 0.00015 + number * 0.00001 + wave * (1.0 + number / 30.0)
        data[ticker] = 50.0 * np.cumprod(1.0 + daily_return)
    return pd.DataFrame(data, index=dates)


def _first_session(month: str) -> pd.Timestamp:
    period = pd.Period(month, freq="M")
    return pd.date_range(period.start_time, period.end_time, freq=TRADING_DAY)[0]


def _monthly_open_panel(months: list[str], values: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(values, index=[_first_session(month) for month in months])


def test_frozen_benchmark_invariance_matches_locked_rules():
    assert FROZEN_BENCHMARK_UNIVERSE == (
        "SPY", "QQQ", "IWM", "EFA", "EEM", "FXI", "VNQ",
        "GLD", "SLV", "DBC", "TLT", "LQD",
    )
    assert FROZEN_BENCHMARK.momentum_lookback_months == 12
    assert FROZEN_BENCHMARK.momentum_skip_months == 1
    assert FROZEN_BENCHMARK.moving_average_months == 10
    assert FROZEN_BENCHMARK.asset_weight_cap == 0.20
    assert FROZEN_BENCHMARK.rebalance_band == 0.01

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


def test_frozen_benchmark_matches_current_production_target_function():
    from trend_sleeve import compute_targets

    close = _etf_prices()
    research = frozen_benchmark_targets(close).targets.iloc[-1]
    production = compute_targets(close, use_fragility_gate=False).set_index("Ticker")
    production = production["Weight"].reindex(research.index)
    pdt.assert_series_equal(
        research.round(4),
        production,
        check_names=False,
        check_dtype=False,
    )


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


def test_hysteresis_and_execution_controls_are_path_dependent_and_bounded():
    index = pd.date_range("2024-01-31", periods=6, freq="ME")
    votes = pd.DataFrame({"A": [0, 2, 1, 1, 0, 3]}, index=index)
    eligible = pd.DataFrame(True, index=index, columns=["A"])
    state = votes_to_hysteresis(votes, eligible, enter_votes=2, exit_votes=0)
    assert state["A"].tolist() == [False, True, True, True, False, True]

    pretrade = pd.Series({"A": 0.30})
    inside_band, turnover, _ = transition_from_drifted_weights(
        pretrade,
        pd.Series({"A": 0.305}),
        rebalance_band=0.01,
        max_turnover=0.30,
        asset_weight_cap=1.0,
        gross_weight_cap=1.0,
    )
    assert inside_band["A"] == pytest.approx(0.30)
    assert turnover == 0.0

    exited, turnover, _ = transition_from_drifted_weights(
        pd.Series({"A": 0.005}),
        pd.Series({"A": 0.0}),
        rebalance_band=0.01,
        max_turnover=None,
        asset_weight_cap=1.0,
        gross_weight_cap=1.0,
    )
    assert exited["A"] == 0.0  # production band never suppresses an ON/OFF flip
    assert turnover == pytest.approx(0.005)

    capped_move, turnover, _ = transition_from_drifted_weights(
        pretrade,
        pd.Series({"A": 0.90}),
        rebalance_band=0.01,
        max_turnover=0.30,
        asset_weight_cap=1.0,
        gross_weight_cap=1.0,
    )
    assert capped_move["A"] == pytest.approx(0.60)
    assert turnover == pytest.approx(0.30)


def test_hard_name_and_gross_caps_override_soft_turnover():
    posttrade, turnover, mandatory = transition_from_drifted_weights(
        pretrade=pd.Series({"A": 0.80, "B": 0.40}),
        desired=pd.Series({"A": 0.80, "B": 0.40}),
        rebalance_band=0.0,
        max_turnover=0.01,
        asset_weight_cap=0.20,
        gross_weight_cap=0.30,
    )
    assert posttrade.abs().max() <= 0.20 + 1e-12
    assert posttrade.abs().sum() <= 0.30 + 1e-12
    assert mandatory > 0.01
    assert turnover == pytest.approx(mandatory)


def test_drift_aware_turnover_after_one_of_two_assets_doubles_is_one_third():
    target_index = pd.date_range("2023-12-31", periods=4, freq="ME")
    desired = pd.DataFrame(0.5, index=target_index, columns=["A", "B"])
    opens = _monthly_open_panel(
        ["2023-12", "2024-01", "2024-02", "2024-03", "2024-04"],
        {
            "A": [100.0, 100.0, 200.0, 200.0, 200.0],
            "B": [100.0, 100.0, 100.0, 100.0, 100.0],
        },
    )
    result = backtest_next_period(
        desired,
        close=opens,
        open_prices=opens,
        cost_bps_per_side=0.0,
        asset_weight_cap=1.0,
        gross_weight_cap=1.0,
    )
    assert result.pretrade_weights.loc["2024-02-29", "A"] == pytest.approx(2.0 / 3.0)
    assert result.pretrade_weights.loc["2024-02-29", "B"] == pytest.approx(1.0 / 3.0)
    assert result.monthly.loc["2024-02-29", "turnover"] == pytest.approx(1.0 / 3.0)


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


def test_stock_allocator_uses_actual_five_percent_cap_and_equal_feasible_sectors():
    dates = pd.bdate_range("2019-01-02", periods=420)
    market_returns = 0.0001 + 0.001 * np.sin(np.arange(len(dates)) / 13.0)
    market = pd.Series(100.0 * np.cumprod(1.0 + market_returns), index=dates)
    names_a = [f"A{i}" for i in range(5)]
    names_b = [f"B{i}" for i in range(15)]
    names = names_a + names_b
    stock_close = pd.DataFrame(
        {
            ticker: 50.0
            * np.cumprod(1.0 + market_returns + 0.00001 * (rank + 1))
            for rank, ticker in enumerate(names)
        },
        index=dates,
    )
    sectors = pd.DataFrame(
        {
            "date": [dates[0]] * len(names),
            "ticker": names,
            "sector": ["A"] * len(names_a) + ["B"] * len(names_b),
        }
    )
    spec = replace(
        PREREGISTERED_CROSS_SECTIONAL_SPEC,
        beta_lookback_days=20,
        residual_momentum_days=20,
        volatility_days=20,
        min_history_days=60,
    )
    result = cross_sectional_targets(stock_close, market, sectors, spec)
    last = result.desired_targets.loc[
        result.desired_targets.abs().sum(axis=1).gt(0.0)
    ].iloc[-1]
    assert last.abs().max() <= 0.05 + 1e-12
    assert last[names_a].sum() == pytest.approx(last[names_b].sum(), abs=1e-10)
    assert last[names_a].sum() == pytest.approx(0.10, abs=1e-10)


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

    membership = pd.DataFrame(
        {
            "date": ["2024-02-15", "2024-03-15"],
            "ticker": ["A", "A"],
            "in_universe": [True, False],
        }
    )
    member_panel = normalize_membership_history(membership, signal_dates, ["A"])
    assert member_panel.loc["2024-01-31", "A"] == np.False_
    assert member_panel.loc["2024-02-29", "A"] == np.True_
    assert member_panel.loc["2024-03-31", "A"] == np.False_


def test_cost_subtraction_is_exact_and_does_not_change_gross_returns():
    index = pd.date_range("2023-12-31", periods=5, freq="ME")
    targets = pd.DataFrame({"A": [0.5, 0.5, 0.0, 0.5, 0.5]}, index=index)
    opens = _monthly_open_panel(
        ["2023-12", "2024-01", "2024-02", "2024-03", "2024-04", "2024-05"],
        {"A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]},
    )
    result = backtest_next_period(
        targets,
        close=opens,
        open_prices=opens,
        cost_bps_per_side=10.0,
    )
    actual_drag = result.monthly["gross_return"] - result.monthly["net_return"]
    expected_drag = result.monthly["turnover"] * 0.001
    pdt.assert_series_equal(actual_drag, expected_drag, check_names=False)


def test_next_open_return_and_trade_cost_use_the_prior_month_end_target():
    dates = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-31",
            "2024-02-01",
            "2024-02-29",
            "2024-03-01",
            "2024-03-28",
        ]
    )
    close = pd.DataFrame({"A": [100, 102, 110, 115, 121, 125]}, index=dates)
    opens = pd.DataFrame({"A": [100, 101, 110, 114, 121, 124]}, index=dates)
    months = close.resample("ME").last().index
    targets = pd.DataFrame({"A": [1.0, 0.0, 0.0]}, index=months)

    result = backtest_next_period(
        targets,
        close,
        open_prices=opens,
        cost_bps_per_side=10.0,
    ).monthly

    february = result.loc[pd.Timestamp("2024-02-29")]
    assert february["gross_return"] == pytest.approx(121.0 / 110.0 - 1.0)
    assert february["turnover"] == pytest.approx(1.0)
    assert february["net_return"] == pytest.approx(121.0 / 110.0 - 1.0 - 0.001)
def test_missing_exact_first_session_open_is_not_replaced_by_delayed_open():
    index = pd.date_range("2023-12-31", periods=3, freq="ME")
    targets = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=index)
    jan_first = _first_session("2024-01")
    opens = _monthly_open_panel(
        ["2023-12", "2024-02", "2024-03"],
        {"A": [100.0, 102.0, 103.0]},
    )
    # A later January bar exists, but the exact first-session open does not.
    opens.loc[jan_first + TRADING_DAY, "A"] = 101.0
    opens = opens.sort_index()
    with pytest.raises(ValueError, match="missing exact first-session open"):
        backtest_next_period(targets, close=opens, open_prices=opens)


def test_missing_held_return_fails_loudly_instead_of_dropping_month():
    index = pd.date_range("2023-12-31", periods=4, freq="ME")
    targets = pd.DataFrame(0.5, index=index, columns=["A", "B"])
    opens = _monthly_open_panel(
        ["2023-12", "2024-01", "2024-02", "2024-03", "2024-04"],
        {
            "A": [100.0, 101.0, np.nan, 103.0, 104.0],
            "B": [100.0, 101.0, 102.0, 103.0, 104.0],
        },
    )
    with pytest.raises(ValueError, match="held return"):
        backtest_next_period(targets, close=opens, open_prices=opens)


def test_target_month_gaps_are_rejected_not_time_compressed():
    targets = pd.DataFrame(
        {"A": [0.5, 0.5]}, index=pd.to_datetime(["2023-12-31", "2024-02-29"])
    )
    opens = _monthly_open_panel(
        ["2023-12", "2024-01", "2024-02", "2024-03"],
        {"A": [100.0, 101.0, 102.0, 103.0]},
    )
    with pytest.raises(ValueError, match="contiguous"):
        backtest_next_period(targets, close=opens, open_prices=opens)


def test_cagr_uses_elapsed_calendar_dates_not_observation_count():
    returns = pd.Series(
        [0.10, 0.10], index=pd.to_datetime(["2020-01-31", "2021-01-31"])
    )
    years = (pd.Timestamp("2021-02-01") - pd.Timestamp("2020-01-01")).days / 365.2425
    expected = 1.21 ** (1.0 / years) - 1.0
    assert _compound_cagr(returns) == pytest.approx(expected)


def test_artifact_writer_refuses_source_paths_and_reports_trial_count(tmp_path):
    close = _etf_prices()
    price_data = PriceData(close=close, open=close, source=None)
    benchmark_target = frozen_benchmark_targets(close)
    benchmark_backtest = backtest_next_period(
        benchmark_target.targets,
        close,
        open_prices=close,
        cost_bps_per_side=5.0,
        rebalance_band=FROZEN_BENCHMARK.rebalance_band,
        asset_weight_cap=FROZEN_BENCHMARK.asset_weight_cap,
        gross_weight_cap=FROZEN_BENCHMARK.gross_weight_cap,
    )
    run = TrialRun(
        name="../../benchmark",
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
    assert manifest["no_order"] is True
    assert manifest["production_writes"] is False
    assert manifest["trial_accounting"]["candidate_trials_executed"] == 0
    assert manifest["trial_artifact_slugs"]["../../benchmark"] == "benchmark"
    assert (written / "monthly_returns" / "benchmark.csv").is_file()
    assert not (tmp_path / "benchmark.csv").exists()
    with pytest.raises(ValueError, match="escapes"):
        _safe_child(written, "..", "escape.txt")


def test_trend_cli_output_is_confined_to_worktree_artifacts(tmp_path):
    allowed = ROOT / "artifacts" / "trend_v2" / "test-run"
    assert _resolve_project_artifact_output(allowed) == allowed.resolve()
    with pytest.raises(SystemExit, match="Refusing non-artifact output"):
        _resolve_project_artifact_output(tmp_path / "outside")


def test_stock_pit_audit_materializes_panels_and_requires_membership_source(tmp_path):
    months = pd.date_range("2024-01-31", periods=2, freq="ME")
    columns = ["A", "B"]
    desired = pd.DataFrame([[0.05, 0.05], [0.05, 0.05]], index=months, columns=columns)
    sector_panel = pd.DataFrame([["S1", "S2"], ["S1", "S2"]], index=months, columns=columns)
    scores = pd.DataFrame([[1.0, 2.0], [1.5, 2.5]], index=months, columns=columns)
    ranks = pd.DataFrame(1.0, index=months, columns=columns)
    membership_panel = pd.DataFrame(True, index=months, columns=columns)
    target = TargetResult(
        desired_targets=desired,
        signal_state=desired.gt(0.0),
        scores=scores,
        ranks=ranks,
        sector_panel=sector_panel,
        membership_panel=membership_panel,
    )
    monthly = pd.DataFrame(
        {
            "gross_return": [0.01, 0.02],
            "net_return": [0.009, 0.019],
            "cash_return": [0.0, 0.0],
            "turnover": [0.1, 0.0],
            "mandatory_cap_turnover": [0.0, 0.0],
            "cost": [0.001, 0.001],
            "gross_exposure": [0.1, 0.1],
            "net_exposure": [0.1, 0.1],
            "cash_weight": [0.9, 0.9],
            "execution": ["next_open_to_next_open"] * 2,
            "execution_boundary": ["x", "y"],
        },
        index=months,
    )
    backtest = BacktestResult(monthly, desired, desired, desired)
    benchmark = TrialRun(
        name="benchmark",
        family="frozen_benchmark_not_a_candidate_trial",
        specification={},
        targets=target,
        backtest=backtest,
        summary=performance_summary(backtest),
    )
    stock = TrialRun(
        name="stock",
        family="stock_cross_sectional_residual_trend_separate_family",
        specification={},
        targets=target,
        backtest=backtest,
        summary=performance_summary(backtest),
    )
    sector_source = tmp_path / "sectors.csv"
    member_source = tmp_path / "membership.csv"
    pd.DataFrame({"date": ["2024-01-01"], "ticker": ["A"], "sector": ["S1"]}).to_csv(
        sector_source, index=False
    )
    pd.DataFrame(
        {"date": ["2024-01-01"], "ticker": ["A"], "in_universe": [True]}
    ).to_csv(member_source, index=False)
    price_data = PriceData(
        close=pd.DataFrame({"SPY": [100.0]}, index=[pd.Timestamp("2024-01-02")]),
        open=None,
    )
    output = write_research_artifacts(
        tmp_path / "artifacts" / "pit_pass",
        price_data,
        [benchmark, stock],
        stock_family_requested=True,
        sector_history_source=sector_source,
        membership_history_source=member_source,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stock_pit_audit"]["pit_gate_passed"] is True
    assert manifest["stock_pit_audit"]["membership_history_provenance"]["sha256"]
    for name in (
        "sector_panel.parquet",
        "membership_panel.parquet",
        "scores.parquet",
        "sector_neutral_ranks.parquet",
        "classification_coverage.csv",
        "exact_universe.json",
    ):
        assert (output / "stock_pit_audit" / name).is_file()

    no_membership_target = TargetResult(
        desired_targets=desired,
        signal_state=desired.gt(0.0),
        scores=scores,
        ranks=ranks,
        sector_panel=sector_panel,
        membership_panel=None,
    )
    stock_no_membership = replace(stock, targets=no_membership_target)
    output_failed = write_research_artifacts(
        tmp_path / "artifacts" / "pit_failed",
        price_data,
        [benchmark, stock_no_membership],
        stock_family_requested=True,
        sector_history_source=sector_source,
    )
    failed = json.loads((output_failed / "manifest.json").read_text(encoding="utf-8"))
    assert failed["stock_pit_audit"]["pit_gate_passed"] is False
    assert any(
        "explicit historical membership" in reason
        for reason in failed["stock_pit_audit"]["reasons"]
    )
