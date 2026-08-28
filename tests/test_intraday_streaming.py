from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.intraday import (
    EligibilityConfig,
    generate_gap_first_hour_signals,
    generate_intraday_shock_signals,
    run_streaming_intraday_research,
    simulate_fixed_time_signals_audited,
    write_streaming_research_artifacts,
)
from trading_calendar import TRADING_DAY

BAR_TIMES = pd.date_range("2026-02-02 09:30", "2026-02-02 15:45", freq="15min").time


def _dates(n: int = 8) -> pd.DatetimeIndex:
    return pd.date_range("2026-02-02", periods=n, freq=TRADING_DAY).normalize()


def _frame(
    dates: pd.DatetimeIndex,
    *,
    overrides: dict[pd.Timestamp, dict[str, dict[str, float]]] | None = None,
    volume: float = 200_000.0,
) -> pd.DataFrame:
    overrides = overrides or {}
    rows: list[dict[str, float | pd.Timestamp]] = []
    previous_close = 100.0
    for raw_day in dates:
        day = pd.Timestamp(raw_day).normalize()
        day_overrides = overrides.get(day, {})
        for bar_time in BAR_TIMES:
            clock = bar_time.strftime("%H:%M")
            spec = day_overrides.get(clock, {})
            stamp = day + pd.Timedelta(hours=bar_time.hour, minutes=bar_time.minute)
            open_price = float(spec.get("open", previous_close))
            close_price = float(spec.get("close", open_price))
            rows.append(
                {
                    "ts": stamp,
                    "open": open_price,
                    "high": float(spec.get("high", max(open_price, close_price) + 0.05)),
                    "low": float(spec.get("low", min(open_price, close_price) - 0.05)),
                    "close": close_price,
                    "volume": float(spec.get("volume", volume)),
                }
            )
            previous_close = close_price
    return pd.DataFrame(rows)


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [{"ticker": "AAA", "sector": "Technology", "sector_proxy": "XLK"}]
    )


def _eligibility() -> EligibilityConfig:
    return EligibilityConfig(
        lookback_sessions=3,
        min_history_sessions=2,
        min_price=1.0,
        min_median_dollar_volume=1.0,
        min_data_completeness=0.90,
    )


def _frames(*, both_templates: bool = True) -> tuple[dict[str, pd.DataFrame], pd.Timestamp]:
    dates = _dates()
    signal_day = dates[-1]
    close_1300 = 104.0 if both_templates else 103.1
    asset_overrides = {
        signal_day: {
            "09:30": {"open": 102.0, "close": 102.2},
            "09:45": {"open": 102.2, "close": 102.4},
            "10:00": {"open": 102.4, "close": 102.6},
            "10:15": {"open": 102.6, "close": 103.1},
            "10:30": {"open": 103.1, "close": 103.2},
            "10:45": {"open": 103.2, "close": 103.2},
            "13:00": {"open": close_1300, "close": close_1300},
            "13:30": {"open": 104.0, "close": 104.0},
            "15:45": {"open": 104.0, "close": 104.5},
        }
    }
    return {
        "AAA": _frame(dates, overrides=asset_overrides),
        "SPY": _frame(dates),
        "XLK": _frame(dates),
    }, signal_day


def _write_frames(root: Path, frames: dict[str, pd.DataFrame]) -> None:
    for ticker, frame in frames.items():
        frame.to_parquet(root / f"{ticker}_15min.parquet", index=False)


def _old_signals(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    kwargs = {
        "metadata": _metadata(),
        "market_ticker": "SPY",
        "candidates": ["AAA"],
        "eligibility_config": _eligibility(),
    }
    return pd.concat(
        [
            generate_gap_first_hour_signals(frames, **kwargs),
            generate_intraday_shock_signals(frames, **kwargs),
        ],
        ignore_index=True,
    ).sort_values(["trade_date", "template_id", "ticker"], ignore_index=True)


def test_streaming_complete_fixture_matches_v0_signal_math_and_execution(tmp_path: Path):
    frames, _ = _frames()
    _write_frames(tmp_path, frames)
    old_signals = _old_signals(frames)
    old_simulation = simulate_fixed_time_signals_audited(
        old_signals, frames, round_trip_cost_bps=10.0
    )

    result = run_streaming_intraday_research(
        tmp_path,
        _metadata(),
        ["AAA"],
        eligibility_config=_eligibility(),
        bootstrap_reps=100,
    )

    assert len(old_signals) == len(result.signals) == 2
    exact_columns = [
        "template_id",
        "ticker",
        "sector",
        "sector_proxy",
        "trade_date",
        "side",
        "decision_ts",
        "feature_bar_ts",
        "feature_available_ts",
        "entry_bar_ts",
        "entry_ts",
        "exit_bar_ts",
        "exit_ts",
    ]
    pd.testing.assert_frame_equal(
        old_signals[exact_columns].reset_index(drop=True),
        result.signals[exact_columns].reset_index(drop=True),
        check_dtype=False,
    )
    numeric_columns = [
        "residual_gap",
        "residual_first_hour",
        "residual_shock",
        "signal_strength",
        "price_proxy",
        "median_dollar_volume",
        "data_completeness",
    ]
    pd.testing.assert_frame_equal(
        old_signals[numeric_columns].reset_index(drop=True),
        result.signals[numeric_columns].reset_index(drop=True),
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        old_simulation.trades[
            ["template_id", "ticker", "gross_return", "net_return"]
        ].reset_index(drop=True),
        result.trades[
            ["template_id", "ticker", "gross_return", "net_return"]
        ].reset_index(drop=True),
        check_dtype=False,
    )
    assert result.cost_grid_trades["cost_bps"].unique().tolist() == [
        5.0,
        10.0,
        15.0,
        20.0,
        30.0,
    ]
    assert len(result.day_cluster_stats) == 10
    primary = result.day_cluster_stats.loc[result.day_cluster_stats["primary_cost_case"]]
    assert set(primary["template_id"]) == set(result.signals["template_id"])
    assert result.capacity_summary["capacity_slots"].drop_duplicates().tolist() == [
        1,
        3,
        5,
        10,
    ]
    assert result.coverage_audit.loc[
        result.coverage_audit["ticker"].eq("AAA"), "n_zero_volume_bars"
    ].iloc[0] == 0


@pytest.mark.parametrize(
    ("clock", "expected_status"),
    [
        ("10:45", "missing_scheduled_entry_bar"),
        ("15:45", "missing_scheduled_exit_bar"),
    ],
)
def test_streaming_missing_execution_bar_matches_v0_audit(
    tmp_path: Path, clock: str, expected_status: str
):
    frames, signal_day = _frames(both_templates=False)
    drop = frames["AAA"]["ts"].eq(signal_day + pd.Timedelta(clock + ":00"))
    frames["AAA"] = frames["AAA"].loc[~drop].reset_index(drop=True)
    _write_frames(tmp_path, frames)
    old_signals = generate_gap_first_hour_signals(
        frames,
        _metadata(),
        candidates=["AAA"],
        eligibility_config=_eligibility(),
    )
    old = simulate_fixed_time_signals_audited(old_signals, frames)
    result = run_streaming_intraday_research(
        tmp_path,
        _metadata(),
        ["AAA"],
        eligibility_config=_eligibility(),
        bootstrap_reps=100,
    )
    assert len(old_signals) == len(result.signals) == 1
    assert old.execution_rejections["execution_status"].tolist() == [expected_status]
    assert result.execution_rejections["execution_status"].tolist() == [
        expected_status
    ]


@pytest.mark.parametrize("bad_clock", ["09:45", "10:00"])
def test_streaming_requires_complete_positive_volume_feature_window(
    tmp_path: Path, bad_clock: str
):
    frames, signal_day = _frames(both_templates=False)
    if bad_clock == "09:45":
        drop = frames["AAA"]["ts"].eq(
            signal_day + pd.Timedelta(hours=9, minutes=45)
        )
        frames["AAA"] = frames["AAA"].loc[~drop].reset_index(drop=True)
    else:
        zero = frames["AAA"]["ts"].eq(
            signal_day + pd.Timedelta(hours=10, minutes=0)
        )
        frames["AAA"].loc[zero, "volume"] = 0.0
    _write_frames(tmp_path, frames)

    # The old endpoint-only signal math still fires.  The streaming real-data
    # path adds the explicitly stricter full-window/positive-volume gate.
    old = generate_gap_first_hour_signals(
        frames,
        _metadata(),
        candidates=["AAA"],
        eligibility_config=_eligibility(),
    )
    result = run_streaming_intraday_research(
        tmp_path,
        _metadata(),
        ["AAA"],
        eligibility_config=_eligibility(),
        bootstrap_reps=100,
    )
    assert len(old) == 1
    assert result.signals.empty
    gap_audit = result.signal_generation_audit.loc[
        result.signal_generation_audit["template_id"].str.startswith("gap_")
    ].iloc[0]
    assert gap_audit["n_feature_quality_fail_after_calendar"] >= 1


@pytest.mark.parametrize(
    ("clock", "expected_status"),
    [
        ("10:45", "zero_volume_scheduled_entry_bar"),
        ("15:45", "zero_volume_scheduled_exit_bar"),
    ],
)
def test_streaming_and_v0_simulator_reject_zero_volume_execution_bars(
    tmp_path: Path, clock: str, expected_status: str
):
    frames, signal_day = _frames(both_templates=False)
    hour, minute = (int(value) for value in clock.split(":"))
    zero = frames["AAA"]["ts"].eq(
        signal_day + pd.Timedelta(hours=hour, minutes=minute)
    )
    frames["AAA"].loc[zero, "volume"] = 0.0
    _write_frames(tmp_path, frames)
    old_signals = generate_gap_first_hour_signals(
        frames,
        _metadata(),
        candidates=["AAA"],
        eligibility_config=_eligibility(),
    )
    old = simulate_fixed_time_signals_audited(old_signals, frames)
    result = run_streaming_intraday_research(
        tmp_path,
        _metadata(),
        ["AAA"],
        eligibility_config=_eligibility(),
        bootstrap_reps=100,
    )
    assert old.execution_rejections["execution_status"].tolist() == [expected_status]
    assert result.execution_rejections["execution_status"].tolist() == [
        expected_status
    ]


def test_common_split_factor_gap_is_flagged_and_filtered(tmp_path: Path):
    dates = _dates()
    signal_day = dates[-1]
    prior_day = dates[-2]
    overrides = {
        prior_day: {"15:45": {"open": 10.0, "close": 10.0}},
        signal_day: {
            "09:30": {"open": 100.0, "close": 100.2},
            "09:45": {"open": 100.2, "close": 100.4},
            "10:00": {"open": 100.4, "close": 100.6},
            "10:15": {"open": 100.6, "close": 101.0},
        },
    }
    frames = {
        "AAA": _frame(dates, overrides=overrides),
        "SPY": _frame(dates),
        "XLK": _frame(dates),
    }
    _write_frames(tmp_path, frames)
    old = generate_gap_first_hour_signals(
        frames,
        _metadata(),
        candidates=["AAA"],
        eligibility_config=_eligibility(),
    )
    result = run_streaming_intraday_research(
        tmp_path,
        _metadata(),
        ["AAA"],
        eligibility_config=_eligibility(),
        bootstrap_reps=100,
    )
    assert len(old) == 1
    assert result.signals.empty
    assert result.signal_rejections["signal_rejection_reason"].tolist() == [
        "raw_price_common_split_factor"
    ]
    assert result.signal_rejections.iloc[0]["asset_open_prior_close_ratio"] == pytest.approx(
        10.0
    )


def test_repo_calendar_surfaces_missing_all_inputs_and_classifies_half_day(
    tmp_path: Path,
):
    dates = _dates()
    missing_day = dates[3]
    early_day = dates[4]
    frames, _ = _frames(both_templates=False)
    for ticker in frames:
        keep = ~frames[ticker]["ts"].dt.normalize().eq(missing_day)
        frames[ticker] = frames[ticker].loc[keep].reset_index(drop=True)
    early_keep = ~(
        frames["SPY"]["ts"].dt.normalize().eq(early_day)
        & (frames["SPY"]["ts"].dt.time > pd.Timestamp("12:45").time())
    )
    frames["SPY"] = frames["SPY"].loc[early_keep].reset_index(drop=True)
    _write_frames(tmp_path, frames)
    result = run_streaming_intraday_research(
        tmp_path,
        _metadata(),
        ["AAA"],
        eligibility_config=_eligibility(),
        bootstrap_reps=100,
    )
    missing = result.market_calendar_audit.loc[
        result.market_calendar_audit["trade_date"].eq(missing_day)
    ].iloc[0]
    early = result.market_calendar_audit.loc[
        result.market_calendar_audit["trade_date"].eq(early_day)
    ].iloc[0]
    assert bool(missing["missing_from_all_loaded_inputs"])
    assert missing["market_session_status"] == "missing_market_proxy_session"
    assert early["market_session_status"] == "observed_early_close_excluded"


def test_missing_sector_proxy_excludes_only_affected_candidate(tmp_path: Path):
    frames, _ = _frames(both_templates=False)
    frames["BBB"] = frames["AAA"].copy()
    _write_frames(tmp_path, frames)
    metadata = pd.DataFrame(
        [
            {"ticker": "AAA", "sector": "Technology", "sector_proxy": "XLK"},
            {"ticker": "BBB", "sector": "Real Estate", "sector_proxy": "XLRE"},
        ]
    )
    result = run_streaming_intraday_research(
        tmp_path,
        metadata,
        ["AAA", "BBB"],
        eligibility_config=_eligibility(),
        bootstrap_reps=100,
    )
    assert result.loaded_candidate_tickers == ("AAA",)
    bbb = result.coverage_audit.loc[
        result.coverage_audit["ticker"].eq("BBB")
        & result.coverage_audit["role"].eq("candidate")
    ].iloc[0]
    assert bbb["exclusion_reason"] == "missing_sector_proxy_file:XLRE"
    assert "SPY" not in bbb["exclusion_reason"]


def test_streaming_writer_refuses_non_artifact_output(tmp_path: Path):
    frames, _ = _frames(both_templates=False)
    _write_frames(tmp_path, frames)
    result = run_streaming_intraday_research(
        tmp_path,
        _metadata(),
        ["AAA"],
        eligibility_config=_eligibility(),
        bootstrap_reps=100,
    )
    with pytest.raises(ValueError, match="artifact root"):
        write_streaming_research_artifacts(result, tmp_path / "output")
