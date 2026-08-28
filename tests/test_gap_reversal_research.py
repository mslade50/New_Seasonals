from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from research.intraday import EligibilityConfig
from research.intraday.diagnostics import day_cluster_statistics
from research.intraday.gap_reversal import (
    GAP_DOWN_LONG_TEMPLATE_ID,
    GAP_UP_SHORT_TEMPLATE_ID,
    SIGNAL_COLUMNS,
    _primary_daily,
    calculate_lagged_atr,
    candidate_slot_portfolios,
    run_gap_reversal_research,
    select_candidate_slots,
    selected_slot_concentration_summaries,
    simulate_gap_reversal_signals,
    write_gap_reversal_artifacts,
)
from research.intraday.streaming import (
    RawPriceDiscontinuityConfig,
    _frame_coverage_row,
)
from trading_calendar import TRADING_DAY

BAR_TIMES = pd.date_range("2026-01-05 09:30", "2026-01-05 15:45", freq="15min").time


def _dates(n: int = 20) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-05", periods=n, freq=TRADING_DAY).normalize()


def _frame(
    dates: pd.DatetimeIndex,
    *,
    overrides: dict[pd.Timestamp, dict[str, dict[str, float]]] | None = None,
) -> pd.DataFrame:
    overrides = overrides or {}
    rows: list[dict[str, float | pd.Timestamp]] = []
    for raw_day in dates:
        day = pd.Timestamp(raw_day).normalize()
        for bar_time in BAR_TIMES:
            clock = bar_time.strftime("%H:%M")
            spec = overrides.get(day, {}).get(clock, {})
            open_price = float(spec.get("open", 100.0))
            close_price = float(spec.get("close", 100.0))
            rows.append(
                {
                    "ts": day
                    + pd.Timedelta(hours=bar_time.hour, minutes=bar_time.minute),
                    "open": open_price,
                    "high": float(spec.get("high", max(open_price, close_price) + 1.0)),
                    "low": float(spec.get("low", min(open_price, close_price) - 1.0)),
                    "close": close_price,
                    "volume": float(spec.get("volume", 200_000.0)),
                }
            )
    return pd.DataFrame(rows)


def _eligibility() -> EligibilityConfig:
    return EligibilityConfig(
        lookback_sessions=3,
        min_history_sessions=2,
        min_price=1.0,
        min_median_dollar_volume=1.0,
        min_data_completeness=0.9,
    )


def _signal(
    *,
    ticker: str = "AAA",
    template_id: str = GAP_DOWN_LONG_TEMPLATE_ID,
    side: int = 1,
    day: pd.Timestamp | None = None,
    limit: float = 98.5,
    strength: float = 1.0,
) -> pd.DataFrame:
    day = pd.Timestamp(day or _dates()[-1]).normalize()
    record = {column: pd.NA for column in SIGNAL_COLUMNS}
    record.update(
        {
            "template_id": template_id,
            "ticker": ticker,
            "sector": "Technology",
            "trade_date": day,
            "side": side,
            "limit_price": limit,
            "signal_strength": strength,
            "gap_atr": -strength if side > 0 else strength,
        }
    )
    return pd.DataFrame([record])


def test_atr_is_simple_14_session_mean_lagged_through_t_minus_one():
    dates = _dates()
    signal_day = dates[-1]
    base = _frame(dates)
    expected = calculate_lagged_atr(base, dates)
    assert expected.loc[signal_day, "atr_14_lagged"] == pytest.approx(2.0)

    current_changed = base.copy()
    today = current_changed["ts"].dt.normalize().eq(signal_day)
    current_changed.loc[today, "high"] = 150.0
    current_changed.loc[today, "low"] = 50.0
    still_lagged = calculate_lagged_atr(current_changed, dates)
    assert still_lagged.loc[signal_day, "atr_14_lagged"] == pytest.approx(2.0)

    prior_changed = base.copy()
    prior = prior_changed["ts"].dt.normalize().eq(dates[-2])
    prior_changed.loc[prior, "high"] = 114.0
    prior_changed.loc[prior, "low"] = 86.0
    changed = calculate_lagged_atr(prior_changed, dates)
    assert changed.loc[signal_day, "atr_14_lagged"] > 2.0


def test_split_day_true_range_is_masked_and_atr_restores_on_session_plus_15():
    dates = _dates(36)
    split_index = 16
    overrides: dict[pd.Timestamp, dict[str, dict[str, float]]] = {}
    for day in dates[split_index:]:
        overrides[day] = {
            bar_time.strftime("%H:%M"): {
                "open": 50.0,
                "high": 51.0,
                "low": 49.0,
                "close": 50.0,
            }
            for bar_time in BAR_TIMES
        }
    frame = _frame(dates, overrides=overrides)
    prior_day = dates[split_index - 1]
    after_early_close = (
        frame["ts"].dt.normalize().eq(prior_day)
        & (frame["ts"].dt.time > pd.Timestamp("12:45").time())
    )
    frame = frame.loc[~after_early_close].reset_index(drop=True)
    daily = calculate_lagged_atr(frame, dates)
    split_day = dates[split_index]
    assert pd.isna(daily.loc[split_day, "prior_close_1545"])
    assert bool(daily.loc[split_day, "raw_price_discontinuity"])
    assert pd.isna(daily.loc[split_day, "daily_true_range"])
    assert daily.loc[split_day, "atr_14_lagged"] == pytest.approx(2.0)
    assert daily.loc[dates[split_index + 1 : split_index + 15], "atr_14_lagged"].isna().all()
    assert daily.loc[dates[split_index + 15], "atr_14_lagged"] == pytest.approx(2.0)


def test_coverage_handles_reindexed_nullable_session_flags_fail_closed(
    tmp_path: Path,
):
    expected_sessions = _dates()
    observed_sessions = expected_sessions[2:]
    aligned = calculate_lagged_atr(
        _frame(observed_sessions), expected_sessions
    )

    # This matches the real snapshot shape that triggered unary ``~`` on a
    # float NaN: calendar alignment promotes the original bool flags to object.
    assert aligned["is_exact_full_session"].dtype == object
    assert aligned["is_exact_full_session"].isna().sum() == 2
    coverage = _frame_coverage_row(
        ticker="AAA",
        role="candidate",
        path=tmp_path / "AAA_15min.parquet",
        input_hash="synthetic",
        daily=aligned,
        expected_sessions=expected_sessions,
        discontinuity_config=RawPriceDiscontinuityConfig(),
    )
    assert coverage["n_observed_expected_sessions"] == len(observed_sessions)
    assert coverage["n_missing_expected_sessions"] == 2
    assert coverage["n_exact_full_sessions"] == len(observed_sessions)
    assert coverage["n_observed_early_closes"] == 0
    assert coverage["n_partial_sessions"] == 0
    assert coverage["coverage_fraction"] == pytest.approx(
        len(observed_sessions) / len(expected_sessions)
    )


def test_gap_sign_assigns_distinct_co_primary_arm_and_side_ids(tmp_path: Path):
    dates = _dates()
    day = dates[-1]
    long_frame = _frame(
        dates,
        overrides={
            day: {
                "09:30": {"open": 99.0, "high": 100.0, "low": 98.8, "close": 99.0},
                "10:00": {"open": 99.0, "high": 99.2, "low": 98.4, "close": 99.0},
            }
        },
    )
    short_frame = _frame(
        dates,
        overrides={
            day: {
                "09:30": {"open": 101.0, "high": 102.0, "low": 100.8, "close": 101.0},
                "10:00": {"open": 101.0, "high": 102.6, "low": 100.8, "close": 101.0},
            }
        },
    )
    for ticker, frame in {"AAA": long_frame, "BBB": short_frame, "SPY": _frame(dates)}.items():
        frame.to_parquet(tmp_path / f"{ticker}_15min.parquet", index=False)
    metadata = pd.DataFrame(
        [
            {"ticker": "AAA", "sector": "Technology"},
            {"ticker": "BBB", "sector": "Financials"},
        ]
    )
    result = run_gap_reversal_research(
        tmp_path,
        metadata,
        ["AAA", "BBB"],
        eligibility_config=_eligibility(),
        bootstrap_reps=100,
    )
    on_day = result.signals.loc[result.signals["trade_date"].eq(day)]
    assert set(zip(on_day["template_id"], on_day["side"], strict=True)) == {
        (GAP_DOWN_LONG_TEMPLATE_ID, 1),
        (GAP_UP_SHORT_TEMPLATE_ID, -1),
    }
    assert on_day.set_index("ticker").loc["AAA", "limit_price"] == pytest.approx(98.5)
    assert on_day.set_index("ticker").loc["BBB", "limit_price"] == pytest.approx(102.5)


def test_first_touch_fills_exact_limit_and_exits_at_scheduled_close():
    day = _dates()[-1]
    bars = _frame(
        pd.DatetimeIndex([day]),
        overrides={
            day: {
                "09:45": {"open": 98.0, "high": 99.0, "low": 97.8, "close": 98.2},
                "15:45": {"open": 99.0, "high": 100.0, "low": 98.0, "close": 99.5},
            }
        },
    )
    trades, rejects = simulate_gap_reversal_signals(_signal(day=day), bars)
    assert rejects.empty
    trade = trades.iloc[0]
    assert trade["entry_price"] == pytest.approx(98.5)
    assert trade["fill_type"] == "activation_open_through_limit_exact_limit"
    assert trade["exit_price"] == pytest.approx(99.5)
    assert trade["gross_return"] == pytest.approx(99.5 / 98.5 - 1.0)


def test_opening_bar_touch_is_excluded_primary_but_included_optimistic_sensitivity():
    day = _dates()[-1]
    bars = _frame(
        pd.DatetimeIndex([day]),
        overrides={
            day: {
                "09:30": {"open": 99.0, "high": 99.2, "low": 98.4, "close": 99.0},
            }
        },
    )
    primary_trades, primary_rejects = simulate_gap_reversal_signals(
        _signal(day=day), bars
    )
    sensitivity_trades, sensitivity_rejects = simulate_gap_reversal_signals(
        _signal(day=day), bars, opening_bar_touch=True
    )
    assert primary_trades.empty
    assert primary_rejects["execution_status"].tolist() == [
        "limit_not_touched_before_1545"
    ]
    assert sensitivity_rejects.empty
    assert sensitivity_trades.iloc[0]["entry_price"] == pytest.approx(98.5)
    assert sensitivity_trades.iloc[0]["fill_type"] == (
        "optimistic_opening_bar_touch_exact_limit"
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing", "missing_required_execution_bar"),
        ("zero", "zero_volume_required_execution_bar"),
    ],
)
def test_missing_or_zero_volume_required_bar_fails_closed(mutation: str, expected: str):
    day = _dates()[-1]
    bars = _frame(pd.DatetimeIndex([day]))
    bad = bars["ts"].eq(day + pd.Timedelta(hours=11))
    if mutation == "missing":
        bars = bars.loc[~bad].reset_index(drop=True)
    else:
        bars.loc[bad, "volume"] = 0.0
    trades, rejects = simulate_gap_reversal_signals(_signal(day=day), bars)
    assert trades.empty
    assert rejects["execution_status"].tolist() == [expected]


def test_costs_apply_only_to_prefill_ranked_fills_and_lower_rank_cannot_substitute():
    day = _dates()[-1]
    signals = pd.concat(
        [
            _signal(ticker="A", day=day, strength=4.0),
            _signal(ticker="B", day=day, strength=3.0),
            _signal(ticker="C", day=day, strength=2.0),
            _signal(ticker="D", day=day, strength=1.0),
        ],
        ignore_index=True,
    )
    trades = signals.loc[signals["ticker"].isin(["A", "D"])].copy()
    trades["gross_return"] = [0.02, 0.50]
    daily, _ = candidate_slot_portfolios(
        signals,
        trades,
        pd.DatetimeIndex([day]),
        cost_grid_bps=(10.0,),
        slots=(3,),
        gap_thresholds_atr=(0.0,),
    )
    row = daily.iloc[0]
    assert row["slots_reserved"] == 3
    assert row["fills"] == 1
    assert row["unused_slots"] == 2
    assert row["slot_portfolio_return"] == pytest.approx((0.02 - 0.001) / 3.0)

    selected = select_candidate_slots(signals, trades)
    assert selected["ticker"].tolist() == ["A", "B", "C"]
    selected_fills = selected.loc[selected["filled"]]
    assert selected_fills["ticker"].tolist() == ["A"]
    assert "D" not in selected_fills["ticker"].tolist()
    ticker_summary, sector_summary = selected_slot_concentration_summaries(selected)
    assert ticker_summary["group"].tolist() == ["A"]
    assert sector_summary["group"].tolist() == ["Technology"]
    assert ticker_summary.iloc[0]["endpoint_contribution_sum"] == pytest.approx(
        (0.02 - 0.001) / 3.0
    )
    assert ticker_summary.iloc[0]["concentration_population"] == (
        "prefill_ranked_top3_filled_slots_10bps"
    )


def test_primary_inference_includes_zero_no_candidate_days_for_both_arms():
    sessions = _dates(3)
    signal = _signal(day=sessions[1], strength=1.0)
    trade = signal.copy()
    trade["gross_return"] = 0.03
    slot_daily, _ = candidate_slot_portfolios(
        signal,
        trade,
        sessions,
        cost_grid_bps=(10.0,),
        slots=(3,),
        gap_thresholds_atr=(0.0,),
    )
    primary = _primary_daily(slot_daily)
    long_days = primary.loc[
        primary["template_id"].eq(GAP_DOWN_LONG_TEMPLATE_ID)
    ].sort_values("trade_date")
    short_days = primary.loc[
        primary["template_id"].eq(GAP_UP_SHORT_TEMPLATE_ID)
    ].sort_values("trade_date")
    assert long_days["trade_date"].tolist() == list(sessions)
    assert long_days["daily_equal_notional_return"].tolist() == pytest.approx(
        [0.0, (0.03 - 0.001) / 3.0, 0.0]
    )
    assert short_days["trade_date"].tolist() == list(sessions)
    assert short_days["daily_equal_notional_return"].eq(0.0).all()

    stats = day_cluster_statistics(
        primary,
        cost_grid_bps=(10.0,),
        bootstrap_reps=100,
        primary_template_ids=(
            GAP_DOWN_LONG_TEMPLATE_ID,
            GAP_UP_SHORT_TEMPLATE_ID,
        ),
    ).set_index("template_id")
    assert stats.loc[GAP_DOWN_LONG_TEMPLATE_ID, "n_days"] == 3
    assert stats.loc[GAP_UP_SHORT_TEMPLATE_ID, "n_days"] == 3
    assert stats.loc[GAP_DOWN_LONG_TEMPLATE_ID, "mean_daily_return"] == pytest.approx(
        (0.03 - 0.001) / 9.0
    )


def test_rank_one_later_bad_tape_stays_selected_and_rank_four_cannot_substitute(
    tmp_path: Path,
):
    dates = _dates()
    signal_day = dates[-1]
    opening_prices = {"A": 96.0, "B": 97.0, "C": 98.0, "D": 99.0}
    metadata_rows: list[dict[str, str]] = []
    for ticker, opening_price in opening_prices.items():
        overrides = {
            signal_day: {
                "09:30": {
                    "open": opening_price,
                    "high": opening_price + 0.2,
                    "low": opening_price - 0.2,
                    "close": opening_price,
                },
                "10:00": {"open": 100.0, "high": 101.0, "low": 95.0, "close": 100.0},
            }
        }
        frame = _frame(dates, overrides=overrides)
        if ticker == "A":
            missing_later_bar = frame["ts"].eq(
                signal_day + pd.Timedelta(hours=11)
            )
            frame = frame.loc[~missing_later_bar].reset_index(drop=True)
        frame.to_parquet(tmp_path / f"{ticker}_15min.parquet", index=False)
        metadata_rows.append({"ticker": ticker, "sector": "Technology"})
    _frame(dates).to_parquet(tmp_path / "SPY_15min.parquet", index=False)

    result = run_gap_reversal_research(
        tmp_path,
        pd.DataFrame(metadata_rows),
        opening_prices,
        eligibility_config=_eligibility(),
        bootstrap_reps=100,
    )
    selected = result.primary_selected_candidates.loc[
        result.primary_selected_candidates["trade_date"].eq(signal_day)
        & result.primary_selected_candidates["template_id"].eq(
            GAP_DOWN_LONG_TEMPLATE_ID
        )
    ].sort_values("selected_rank")
    assert selected["ticker"].tolist() == ["A", "B", "C"]
    assert "D" not in selected["ticker"].tolist()
    rank_one = selected.iloc[0]
    assert rank_one["ticker"] == "A"
    assert not bool(rank_one["filled"])
    assert rank_one["execution_status"] == "missing_required_execution_bar"
    assert bool(rank_one["later_tape_quality_failure"])
    assert bool(
        result.selected_outcome_summary.loc[
            result.selected_outcome_summary["template_id"].eq(
                GAP_DOWN_LONG_TEMPLATE_ID
            ),
            "advance_blocked_by_selected_tape_quality",
        ].iloc[0]
    )


def test_non_full_spy_session_is_excluded_from_every_return_surface(tmp_path: Path):
    dates = _dates()
    signal_day = dates[-1]
    candidate = _frame(
        dates,
        overrides={
            signal_day: {
                "09:30": {"open": 99.0, "high": 99.2, "low": 98.8, "close": 99.0},
                "10:00": {"open": 99.0, "high": 99.2, "low": 98.4, "close": 99.0},
            }
        },
    )
    market = _frame(dates)
    missing_market_close = market["ts"].eq(
        signal_day + pd.Timedelta(hours=15, minutes=45)
    )
    market = market.loc[~missing_market_close].reset_index(drop=True)
    candidate.to_parquet(tmp_path / "AAA_15min.parquet", index=False)
    market.to_parquet(tmp_path / "SPY_15min.parquet", index=False)
    result = run_gap_reversal_research(
        tmp_path,
        pd.DataFrame([{"ticker": "AAA", "sector": "Technology"}]),
        ["AAA"],
        eligibility_config=_eligibility(),
        bootstrap_reps=100,
    )
    assert signal_day not in result.primary_sessions
    assert not result.signals["trade_date"].eq(signal_day).any()
    assert not result.trades["trade_date"].eq(signal_day).any()
    assert not result.cost_grid_trades["trade_date"].eq(signal_day).any()
    assert not result.conditional_daily_returns["trade_date"].eq(signal_day).any()


def test_writer_stays_under_artifacts_and_manifest_freezes_research_only(tmp_path: Path):
    dates = _dates()
    day = dates[-1]
    overrides = {
        day: {
            "09:30": {"open": 99.0, "high": 100.0, "low": 98.8, "close": 99.0},
            "10:00": {"open": 99.0, "high": 99.2, "low": 98.4, "close": 99.0},
        }
    }
    for ticker, frame in {
        "AAA": _frame(dates, overrides=overrides),
        "SPY": _frame(dates),
    }.items():
        frame.to_parquet(tmp_path / f"{ticker}_15min.parquet", index=False)
    result = run_gap_reversal_research(
        tmp_path,
        pd.DataFrame([{"ticker": "AAA", "sector": "Technology"}]),
        ["AAA"],
        eligibility_config=_eligibility(),
        bootstrap_reps=100,
    )
    repo_root = Path(__file__).resolve().parents[1]
    # Use a fresh ignored artifact path without deleting prior evidence.
    output = repo_root / "artifacts" / "gap-reversal-test-output" / uuid4().hex
    written = write_gap_reversal_artifacts(result, output)
    manifest = json.loads((written / "run_manifest.json").read_text(encoding="utf-8"))
    report = (written / "report.html").read_text(encoding="utf-8")
    assert "Required 20-bps robustness — both arms" in report
    assert GAP_DOWN_LONG_TEMPLATE_ID in report
    assert GAP_UP_SHORT_TEMPLATE_ID in report
    assert "Chronology / annual results at 10 and 20 bps" in report
    assert "Leave-one-year-out at 10 and 20 bps" in report
    assert "Primary selected-slot concentration" in report
    assert report.count(GAP_UP_SHORT_TEMPLATE_ID) >= 80
    assert (written / "primary_cohort_eligible_fills_10bps.parquet").is_file()
    assert (written / "primary_selected_candidates_10bps.parquet").is_file()
    assert (written / "primary_selected_fills_10bps.parquet").is_file()
    assert manifest["research_only"] is True
    assert manifest["production_writes"] is False
    assert manifest["primary_capacity_slots"] == 3
    assert manifest["holm_family_size"] == 2
    assert manifest["no_candidate_arm_days_included_as_zero_return"] is True
    assert manifest["n_primary_inference_sessions"] == len(result.primary_sessions)
    assert manifest["n_primary_cohort_eligible_fills"] == len(result.trades)
    assert set(pd.to_datetime(result.trades["trade_date"]).dt.normalize()).issubset(
        set(result.primary_sessions)
    )
    assert set(
        pd.to_datetime(result.conditional_daily_returns["trade_date"]).dt.normalize()
    ).issubset(set(result.primary_sessions))
    assert manifest["concentration_population"] == (
        "actual_fills_among_prefill_ranked_top3_per_arm_day_at_10bps"
    )
    assert manifest["lower_ranked_fills_excluded_from_concentration"] is True
    with pytest.raises(ValueError, match="artifact root"):
        write_gap_reversal_artifacts(result, tmp_path / "outside")
