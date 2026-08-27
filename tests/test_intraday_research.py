from __future__ import annotations

import json

import pandas as pd
import pytest

import scripts.run_intraday_research as intraday_cli
from research.intraday import (
    AmbiguousCapitalTieError,
    CapitalReuseConfig,
    EligibilityConfig,
    GapFirstHourConfig,
    IntradayDataError,
    IntradayShockConfig,
    LookaheadError,
    MissingExecutionBarError,
    apply_capital_feasibility,
    calculate_eligibility,
    generate_gap_first_hour_signals,
    generate_intraday_shock_signals,
    load_parquet_frames,
    normalize_bars,
    simulate_fixed_time_signals,
    simulate_fixed_time_signals_audited,
)

BAR_TIMES = pd.date_range("2026-01-02 09:30", "2026-01-02 15:45", freq="15min").time


def _frame(dates, *, overrides=None, volume=200_000.0):
    overrides = overrides or {}
    rows = []
    previous_close = 100.0
    for day in dates:
        day = pd.Timestamp(day)
        day_overrides = overrides.get(day.normalize(), {})
        for bar_time in BAR_TIMES:
            stamp = day.normalize() + pd.Timedelta(
                hours=bar_time.hour, minutes=bar_time.minute
            )
            spec = day_overrides.get(bar_time.strftime("%H:%M"), {})
            open_price = float(spec.get("open", previous_close))
            close_price = float(spec.get("close", open_price))
            high = float(spec.get("high", max(open_price, close_price) + 0.05))
            low = float(spec.get("low", min(open_price, close_price) - 0.05))
            rows.append(
                {
                    "ts": stamp,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close_price,
                    "volume": float(spec.get("volume", volume)),
                }
            )
            previous_close = close_price
    return pd.DataFrame(rows)


def _dates(n=12):
    return pd.bdate_range("2026-01-02", periods=n)


def _eligibility_config(**kwargs):
    base = {
        "lookback_sessions": 3,
        "min_history_sessions": 2,
        "min_price": 1.0,
        "min_median_dollar_volume": 1.0,
        "min_data_completeness": 0.90,
    }
    base.update(kwargs)
    return EligibilityConfig(**base)


def _metadata():
    return pd.DataFrame(
        [
            {"ticker": "AAA", "sector": "Technology", "sector_proxy": "XLK"},
            {"ticker": "SPY", "sector": "Market", "sector_proxy": "SPY"},
            {"ticker": "XLK", "sector": "Technology", "sector_proxy": "XLK"},
            {"ticker": "UNRELATED", "sector": "Energy", "sector_proxy": "XLE"},
        ]
    )


def _gap_frames():
    dates = _dates()
    signal_day = dates[-1].normalize()
    asset_overrides = {
        signal_day: {
            "09:30": {"open": 102.0, "close": 102.20},
            "09:45": {"open": 102.20, "close": 102.40},
            "10:00": {"open": 102.40, "close": 102.60},
            "10:15": {"open": 102.60, "close": 103.10},
            "10:30": {"open": 103.10, "close": 103.20},
            "15:45": {"open": 103.20, "close": 104.00},
        }
    }
    return {
        "AAA": _frame(dates, overrides=asset_overrides),
        "SPY": _frame(dates),
        "XLK": _frame(dates),
    }, signal_day


def _shock_frames():
    dates = _dates()
    signal_day = dates[-1].normalize()
    asset_overrides = {
        signal_day: {
            "12:30": {"open": 100.0, "close": 102.0},
            "12:45": {"open": 102.0, "close": 102.0},
            "13:00": {"open": 102.0, "close": 102.0},
            "13:15": {"open": 102.0, "close": 101.8},
            "15:45": {"open": 101.8, "close": 100.5},
        }
    }
    return {
        "AAA": _frame(dates, overrides=asset_overrides),
        "SPY": _frame(dates),
        "XLK": _frame(dates),
    }, signal_day


def test_regular_session_validation_rejects_after_hours():
    bad = _frame(_dates(1))
    bad.loc[0, "ts"] = pd.Timestamp("2026-01-02 08:00")
    with pytest.raises(IntradayDataError, match="non-regular-session"):
        normalize_bars(bad, ticker="AAA")


def test_bar_validation_rejects_non_finite_values():
    bad = _frame(_dates(1))
    bad.loc[0, "close"] = float("inf")
    bad.loc[0, "high"] = float("inf")
    with pytest.raises(IntradayDataError, match="non-finite"):
        normalize_bars(bad, ticker="AAA")


def test_eligibility_uses_only_completed_prior_sessions():
    dates = _dates(3)
    overrides = {
        dates[1].normalize(): {
            clock: {"volume": 10_000_000.0}
            for clock in (time.strftime("%H:%M") for time in BAR_TIMES)
        }
    }
    frame = _frame(dates, overrides=overrides, volume=1.0)
    config = EligibilityConfig(
        lookback_sessions=1,
        min_history_sessions=1,
        min_price=1.0,
        min_median_dollar_volume=1_000_000.0,
        min_data_completeness=0.95,
    )
    eligibility = calculate_eligibility({"AAA": frame}, config)
    day_two = eligibility.loc[eligibility["trade_date"].eq(dates[1].normalize())].iloc[
        0
    ]
    day_three = eligibility.loc[
        eligibility["trade_date"].eq(dates[2].normalize())
    ].iloc[0]
    assert not bool(day_two["eligible"])
    assert bool(day_three["eligible"])


def test_canonical_market_calendar_counts_wholly_missing_sessions():
    dates = _dates(5)
    frames = {"AAA": _frame(dates), "SPY": _frame(dates)}
    missing_day = dates[-2].normalize()
    frames["AAA"] = frames["AAA"].loc[
        ~frames["AAA"]["ts"].dt.normalize().eq(missing_day)
    ]
    eligibility = calculate_eligibility(
        frames,
        EligibilityConfig(
            lookback_sessions=1,
            min_history_sessions=1,
            min_price=1.0,
            min_median_dollar_volume=1.0,
            min_data_completeness=0.95,
        ),
        calendar_ticker="SPY",
    )
    missing_row = eligibility.loc[
        eligibility["ticker"].eq("AAA") & eligibility["trade_date"].eq(missing_day)
    ].iloc[0]
    next_row = eligibility.loc[
        eligibility["ticker"].eq("AAA")
        & eligibility["trade_date"].eq(dates[-1].normalize())
    ].iloc[0]
    assert missing_row["bars_in_session"] == 0
    assert not bool(next_row["eligible"])


def test_gap_template_clock_and_future_price_invariance():
    frames, signal_day = _gap_frames()
    kwargs = {
        "metadata": _metadata(),
        "market_ticker": "SPY",
        "candidates": ["AAA"],
        "eligibility_config": _eligibility_config(),
        "config": GapFirstHourConfig(),
    }
    original = generate_gap_first_hour_signals(frames, **kwargs)
    assert len(original) == 1
    row = original.iloc[0]
    assert row["trade_date"] == signal_day
    assert row["feature_bar_ts"].strftime("%H:%M") == "10:15"
    assert row["feature_available_ts"].strftime("%H:%M") == "10:30"
    assert row["decision_ts"].strftime("%H:%M") == "10:30"
    assert row["entry_ts"].strftime("%H:%M") == "10:45"
    assert row["decision_ts"] < row["entry_ts"]
    assert row["side"] == 1

    changed = {ticker: frame.copy() for ticker, frame in frames.items()}
    late = changed["AAA"]["ts"].eq(signal_day + pd.Timedelta(hours=15, minutes=45))
    changed["AAA"].loc[late, ["open", "high", "low", "close"]] = [
        50.0,
        50.1,
        49.9,
        50.0,
    ]
    rerun = generate_gap_first_hour_signals(changed, **kwargs)
    pd.testing.assert_frame_equal(
        original[
            ["ticker", "side", "decision_ts", "residual_gap", "residual_first_hour"]
        ],
        rerun[["ticker", "side", "decision_ts", "residual_gap", "residual_first_hour"]],
    )


def test_intraday_shock_is_cumulative_through_1300_and_enters_after_latency():
    frames, _ = _shock_frames()
    signals = generate_intraday_shock_signals(
        frames,
        _metadata(),
        market_ticker="SPY",
        candidates=["AAA"],
        eligibility_config=_eligibility_config(),
        config=IntradayShockConfig(),
    )
    assert len(signals) == 1
    row = signals.iloc[0]
    assert row["feature_bar_ts"].strftime("%H:%M") == "13:00"
    assert row["feature_available_ts"].strftime("%H:%M") == "13:15"
    assert row["decision_ts"].strftime("%H:%M") == "13:15"
    assert row["entry_ts"].strftime("%H:%M") == "13:30"
    assert row["asset_shock"] == pytest.approx(0.02)
    assert row["side"] == -1


def test_simulator_rejects_future_feature_and_deducts_costs():
    frames, _ = _gap_frames()
    signals = generate_gap_first_hour_signals(
        frames,
        _metadata(),
        market_ticker="SPY",
        candidates=["AAA"],
        eligibility_config=_eligibility_config(),
    )
    trades = simulate_fixed_time_signals(signals, frames, round_trip_cost_bps=10.0)
    assert len(trades) == 1
    assert trades.iloc[0]["net_return"] == pytest.approx(
        trades.iloc[0]["gross_return"] - 0.001
    )
    assert trades.iloc[0]["day_cluster"] == trades.iloc[0]["trade_date"]
    assert trades.iloc[0]["ticker_cluster"] == "AAA"
    assert trades.iloc[0]["sector_cluster"] == "Technology"
    with pytest.raises(ValueError, match="finite and non-negative"):
        simulate_fixed_time_signals(signals, frames, round_trip_cost_bps=float("nan"))

    leaked = signals.copy()
    leaked["feature_bar_ts"] = leaked["exit_bar_ts"]
    with pytest.raises(LookaheadError, match="feature availability"):
        simulate_fixed_time_signals(leaked, frames)

    zero_latency = signals.copy()
    zero_latency["entry_bar_ts"] = zero_latency["decision_ts"]
    zero_latency["entry_ts"] = zero_latency["decision_ts"]
    with pytest.raises(LookaheadError, match="strictly after"):
        simulate_fixed_time_signals(zero_latency, frames)

    wrong_day = signals.copy()
    wrong_day["feature_bar_ts"] = wrong_day["feature_bar_ts"] - pd.Timedelta(days=1)
    wrong_day["feature_available_ts"] = wrong_day["feature_bar_ts"] + pd.Timedelta(
        minutes=15
    )
    with pytest.raises(LookaheadError, match="trade_date"):
        simulate_fixed_time_signals(wrong_day, frames)


def test_missing_close_bar_does_not_select_the_signal_and_is_flagged_at_execution():
    frames, signal_day = _gap_frames()
    full = generate_gap_first_hour_signals(
        frames,
        _metadata(),
        market_ticker="SPY",
        candidates=["AAA"],
        eligibility_config=_eligibility_config(),
    )
    missing_close = {ticker: frame.copy() for ticker, frame in frames.items()}
    drop_bar = missing_close["AAA"]["ts"].eq(
        signal_day + pd.Timedelta(hours=15, minutes=45)
    )
    missing_close["AAA"] = missing_close["AAA"].loc[~drop_bar].reset_index(drop=True)
    rerun = generate_gap_first_hour_signals(
        missing_close,
        _metadata(),
        market_ticker="SPY",
        candidates=["AAA"],
        eligibility_config=_eligibility_config(),
    )
    pd.testing.assert_frame_equal(full, rerun)
    audited = simulate_fixed_time_signals_audited(rerun, missing_close)
    assert audited.trades.empty
    assert len(audited.execution_rejections) == 1
    assert (
        audited.execution_rejections.iloc[0]["execution_status"]
        == "missing_scheduled_exit_bar"
    )
    with pytest.raises(MissingExecutionBarError, match="missing_scheduled_exit_bar"):
        simulate_fixed_time_signals(rerun, missing_close)


def test_missing_future_entry_bar_does_not_change_signal_and_is_audited():
    frames, signal_day = _gap_frames()
    full = generate_gap_first_hour_signals(
        frames,
        _metadata(),
        market_ticker="SPY",
        candidates=["AAA"],
        eligibility_config=_eligibility_config(),
    )
    missing_entry = {ticker: frame.copy() for ticker, frame in frames.items()}
    drop_bar = missing_entry["AAA"]["ts"].eq(
        signal_day + pd.Timedelta(hours=10, minutes=45)
    )
    missing_entry["AAA"] = missing_entry["AAA"].loc[~drop_bar].reset_index(drop=True)
    rerun = generate_gap_first_hour_signals(
        missing_entry,
        _metadata(),
        market_ticker="SPY",
        candidates=["AAA"],
        eligibility_config=_eligibility_config(),
    )
    pd.testing.assert_frame_equal(full, rerun)
    audited = simulate_fixed_time_signals_audited(rerun, missing_entry)
    assert audited.trades.empty
    assert list(audited.execution_rejections["execution_status"]) == [
        "missing_scheduled_entry_bar"
    ]


def test_gap_requires_exact_same_prior_session_scheduled_close():
    frames, signal_day = _gap_frames()
    prior_day = _dates()[-2].normalize()
    missing_prior_close = {ticker: frame.copy() for ticker, frame in frames.items()}
    drop = missing_prior_close["AAA"]["ts"].eq(
        prior_day + pd.Timedelta(hours=15, minutes=45)
    )
    missing_prior_close["AAA"] = missing_prior_close["AAA"].loc[~drop]
    signals = generate_gap_first_hour_signals(
        missing_prior_close,
        _metadata(),
        market_ticker="SPY",
        candidates=["AAA"],
        eligibility_config=_eligibility_config(min_data_completeness=0.0),
    )
    assert signals.empty
    assert signal_day in set(frames["SPY"]["ts"].dt.normalize())


def test_gap_does_not_bridge_a_wholly_missing_market_proxy_session():
    frames, _ = _gap_frames()
    missing_day = _dates()[-2].normalize()
    no_market_session = {ticker: frame.copy() for ticker, frame in frames.items()}
    no_market_session["SPY"] = no_market_session["SPY"].loc[
        ~no_market_session["SPY"]["ts"].dt.normalize().eq(missing_day)
    ]
    signals = generate_gap_first_hour_signals(
        no_market_session,
        _metadata(),
        market_ticker="SPY",
        candidates=["AAA"],
        eligibility_config=_eligibility_config(min_data_completeness=0.0),
    )
    assert signals.empty


def test_missing_required_sector_proxy_fails_loudly():
    frames, _ = _gap_frames()
    frames.pop("XLK")
    with pytest.raises(ValueError, match="required sector proxy"):
        generate_gap_first_hour_signals(
            frames,
            _metadata(),
            market_ticker="SPY",
            candidates=["AAA"],
            eligibility_config=_eligibility_config(),
        )


def test_local_parquet_loader_accepts_existing_layout(tmp_path):
    frame = _frame(_dates(1))
    frame.to_parquet(tmp_path / "AAA_15min.parquet", index=False)
    loaded = load_parquet_frames(tmp_path, tickers=["AAA"])
    assert list(loaded) == ["AAA"]
    assert len(loaded["AAA"]) == 26
    with pytest.raises(FileNotFoundError, match="missing intraday"):
        load_parquet_frames(tmp_path, tickers=["MISSING"])


def test_explicit_candidate_missing_from_frames_fails_loudly():
    frames, _ = _gap_frames()
    with pytest.raises(ValueError, match="explicit candidate"):
        generate_gap_first_hour_signals(
            frames,
            _metadata(),
            candidates=["MISSING"],
            eligibility_config=_eligibility_config(),
        )


def _capital_trades():
    return pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "entry_ts": pd.Timestamp("2026-01-02 10:00"),
                "exit_ts": pd.Timestamp("2026-01-02 11:00"),
            },
            {
                "ticker": "BBB",
                "entry_ts": pd.Timestamp("2026-01-02 12:00"),
                "exit_ts": pd.Timestamp("2026-01-02 13:00"),
            },
        ]
    )


def test_capital_reuse_defaults_off_and_reports_rejected_trades():
    config = CapitalReuseConfig(
        default_notional_per_trade=60.0,
        starting_settled_cash=100.0,
    )
    result = apply_capital_feasibility(_capital_trades(), config)
    assert list(result.feasible_trades["ticker"]) == ["AAA"]
    assert list(result.rejected_trades["ticker"]) == ["BBB"]
    assert result.rejected_trades.iloc[0]["capital_rejection_reason"] == "settled_cash"
    assert not bool(result.audit["same_day_reuse_assumed"].any())
    with pytest.raises(ValueError, match="positive"):
        CapitalReuseConfig(starting_settled_cash=float("nan"))


def test_explicit_same_day_reuse_releases_capital_after_exit():
    config = CapitalReuseConfig(
        default_notional_per_trade=60.0,
        starting_settled_cash=100.0,
        same_day_reuse_allowed=True,
    )
    result = apply_capital_feasibility(_capital_trades(), config)
    assert list(result.feasible_trades["ticker"]) == ["AAA", "BBB"]
    assert result.rejected_trades.empty


def test_max_concurrent_notional_rejects_overlapping_trade():
    trades = _capital_trades()
    trades.loc[0, "exit_ts"] = pd.Timestamp("2026-01-02 15:00")
    config = CapitalReuseConfig(
        default_notional_per_trade=60.0,
        max_concurrent_notional=100.0,
    )
    result = apply_capital_feasibility(trades, config)
    assert list(result.rejected_trades["ticker"]) == ["BBB"]
    assert (
        result.rejected_trades.iloc[0]["capital_rejection_reason"]
        == "max_concurrent_notional"
    )


def test_oversubscribed_capital_tie_requires_priority_and_is_row_order_invariant():
    trades = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "entry_ts": pd.Timestamp("2026-01-02 10:00"),
                "exit_ts": pd.Timestamp("2026-01-02 15:00"),
                "priority": 1.0,
            },
            {
                "ticker": "BBB",
                "entry_ts": pd.Timestamp("2026-01-02 10:00"),
                "exit_ts": pd.Timestamp("2026-01-02 15:00"),
                "priority": 2.0,
            },
        ]
    )
    config = CapitalReuseConfig(
        default_notional_per_trade=60.0,
        max_concurrent_notional=100.0,
    )
    with pytest.raises(AmbiguousCapitalTieError, match="pre-registered priority"):
        apply_capital_feasibility(trades, config)
    tied = trades.assign(priority=1.0)
    with pytest.raises(AmbiguousCapitalTieError, match="is tied"):
        apply_capital_feasibility(tied, config, priority_column="priority")
    forward = apply_capital_feasibility(trades, config, priority_column="priority")
    reverse = apply_capital_feasibility(
        trades.iloc[::-1].reset_index(drop=True), config, priority_column="priority"
    )
    assert list(forward.feasible_trades["ticker"]) == ["BBB"]
    assert list(reverse.feasible_trades["ticker"]) == ["BBB"]


def test_cli_writes_only_research_artifacts_from_local_parquets(tmp_path, capsys):
    frames, _ = _gap_frames()
    data_dir = tmp_path / "intraday"
    artifacts_root = tmp_path / "artifacts"
    output_dir = artifacts_root / "output"
    data_dir.mkdir()
    for ticker, frame in frames.items():
        frame.to_parquet(data_dir / f"{ticker}_15min.parquet", index=False)
    sector_map = tmp_path / "sector_map.parquet"
    _metadata().to_parquet(sector_map, index=False)

    return_code = intraday_cli.main(
        [
            "--data-dir",
            str(data_dir),
            "--sector-map",
            str(sector_map),
            "--tickers",
            "AAA",
            "--min-dollar-volume",
            "1",
            "--per-trade-notional",
            "100",
            "--starting-settled-cash",
            "100",
            "--output-dir",
            str(output_dir),
        ],
        artifacts_root=artifacts_root,
    )
    assert return_code == 0
    assert "Research-only" in capsys.readouterr().out
    expected = {
        "eligibility.parquet",
        "signals.parquet",
        "trades.parquet",
        "execution_rejections.parquet",
        "summary.csv",
        "run_manifest.json",
        "capital_audit.parquet",
        "capital_feasible_trades.parquet",
        "capital_rejections.parquet",
        "capital_summary.csv",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["research_only"] is True
    assert manifest["no_order"] is True
    assert manifest["production_writes"] is False
    assert manifest["n_signals"] == 1
    assert manifest["n_trades"] == 1
    assert manifest["n_execution_rejected"] == 0
    assert manifest["n_capital_feasible"] == 1
    assert manifest["n_capital_rejected"] == 0
    assert manifest["capital_config"]["same_day_reuse_allowed"] is False


def test_cli_output_guard_rejects_paths_outside_artifacts_and_nonempty_runs(tmp_path):
    artifacts_root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    with pytest.raises(ValueError, match="must stay under"):
        intraday_cli._resolve_artifact_output(
            outside, artifacts_root=artifacts_root
        )

    occupied = artifacts_root / "occupied"
    occupied.mkdir(parents=True)
    (occupied / "existing.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        intraday_cli._resolve_artifact_output(
            occupied, artifacts_root=artifacts_root
        )
