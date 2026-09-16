from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from legend_etf.calendar import previous_session, rth_bar_starts
from legend_etf.etf_source import (
    evaluate_etf_setup,
    prepare_etf_signal_plan,
    signal_history,
)
from legend_etf.session import LegendSession
from legend_etf.storage import (
    StateStore,
    atomic_write_json,
    finalize_plan,
    validate_plan,
)
from research.legend_ema_backtest import BacktestConfig, build_daily, qualifies_setup

ET = "America/New_York"
ENTRY = "2026-09-16"


def history(entry=ENTRY, down=False):
    setup = previous_session(entry).date().isoformat()
    index = rth_bar_starts((pd.Timestamp(entry) - pd.Timedelta(days=20)).date(), setup)
    frame = pd.DataFrame(
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "volume": 10.0},
        index=index,
    )
    mask = frame.index.date == pd.Timestamp(setup).date()
    prices = np.linspace(110.0, 119.0, 26)
    for column in ("open", "high", "low", "close"):
        frame.loc[mask, column] = prices
    frame.iloc[-1, frame.columns.get_loc("high")] = 122.0
    if down:
        high, low = 200 - frame["low"], 200 - frame["high"]
        frame["open"], frame["close"] = 200 - frame["open"], 200 - frame["close"]
        frame["high"], frame["low"] = high, low
    return frame


def plan():
    return prepare_etf_signal_plan(
        histories={"SPY": history(), "QQQ": history(down=True)},
        entry_date=ENTRY,
        as_of=pd.Timestamp(f"{ENTRY} 08:45", tz=ET),
    )


@pytest.mark.parametrize("down", [False, True])
def test_native_setup_matches_original_etf_research_without_futures(down):
    frame = history(down=down)
    result = evaluate_etf_setup(frame, entry_date=ENTRY)
    reference = frame.reset_index(names="ts")
    reference["ema"] = reference["close"].ewm(span=20, adjust=False).mean()
    reference["ema_prev"] = reference["ema"].shift()
    reference["session"] = reference["ts"].dt.tz_localize(None).dt.normalize()
    daily = build_daily(reference)
    assert qualifies_setup(daily.iloc[-1], BacktestConfig()) == result["qualifies"]
    assert result["qualifies"]
    assert result["trend_ratio"] == pytest.approx(0.75)
    assert result["initial_ema"] == pytest.approx(reference.iloc[-1]["ema"])
    assert "instrument_id" not in result


def test_inclusive_touch_and_threshold_boundary():
    frame = history()
    frame.loc[frame.index[-1], ["close", "low"]] = 118.99
    assert (
        evaluate_etf_setup(frame, entry_date=ENTRY)["reason"]
        == "trend_ratio_below_threshold"
    )
    frame = history()
    index = frame.index[-4]
    frame.loc[index, "low"] = (
        frame["close"].ewm(span=20, adjust=False).mean().loc[index]
    )
    assert evaluate_etf_setup(frame, entry_date=ENTRY)["reason"] == "ema_touch"


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "naive", "bad_ohlc", "stale"]
)
def test_bad_history_fails_closed(damage):
    frame = history()
    if damage == "missing":
        frame = frame.drop(frame.index[-5])
    elif damage == "duplicate":
        frame = pd.concat([frame, frame.iloc[[-1]]])
    elif damage == "naive":
        frame.index = frame.index.tz_localize(None)
    elif damage == "bad_ohlc":
        frame.iloc[-1, frame.columns.get_loc("low")] = 1000
    else:
        frame = frame.iloc[:-26]
    with pytest.raises(ValueError):
        evaluate_etf_setup(frame, entry_date=ENTRY)


def test_current_session_cannot_leak_into_prior_setup():
    frame = history()
    after = frame.iloc[[-1]].copy()
    after.index = pd.DatetimeIndex([pd.Timestamp(f"{ENTRY} 09:30", tz=ET)])
    after[["open", "high", "low", "close"]] = 10000
    assert evaluate_etf_setup(
        pd.concat([frame, after]), entry_date=ENTRY
    ) == evaluate_etf_setup(frame, entry_date=ENTRY)


@pytest.mark.parametrize("entry", ["2026-03-09", "2026-03-10", "2026-09-08", "2026-11-02"])
def test_calendar_handles_dst_weekends_and_holidays(entry):
    result = evaluate_etf_setup(history(entry), entry_date=entry)
    assert result["qualifies"]
    assert result["setup_date"] == previous_session(entry).date().isoformat()


@pytest.mark.parametrize("entry", ["2026-11-27", "2026-11-30", "2026-09-19"])
def test_partial_setup_entry_and_closed_days_fail(entry):
    with pytest.raises(ValueError):
        signal_history(history(), entry)


def test_plan_is_spy_qqq_only_and_uses_completed_previous_close():
    payload = plan()
    validate_plan(payload, entry_date=ENTRY)
    assert [item["root"] for item in payload["markets"]] == ["SPY", "QQQ"]
    assert payload["data_as_of"] == "2026-09-15T16:00:00-04:00"
    assert payload["quoted_cost_usd"] == 0


@pytest.mark.parametrize(
    "damage", ["date", "duplicate", "source", "hash", "future", "ratio", "version"]
)
def test_mutated_native_plan_cannot_pass_validation(damage):
    payload = plan()
    if damage == "date":
        payload["setup_date"] = "2026-09-14"
    if damage == "duplicate":
        payload["markets"][1] = payload["markets"][0]
    if damage == "source":
        payload["schema"] = "ADJUSTED_LAST"
    if damage == "hash":
        payload["markets"][0]["history_sha256"] = "missing"
    if damage == "future":
        payload["created_at"] = "2026-09-16T09:31:00-04:00"
    if damage == "ratio":
        payload["markets"][0]["trend_ratio"] = 0.5
    if damage == "version":
        payload["strategy_version"] = "legend-etf-original-v1"
    payload = finalize_plan(payload)
    with pytest.raises((ValueError, TypeError)):
        validate_plan(payload, entry_date=ENTRY)


def test_native_plan_reaches_existing_preflight_without_contract_probe(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LEGEND_ETF_PRIMARY_ACCOUNT", "U123")
    target = tmp_path / "plan.json"
    atomic_write_json(target, plan())
    session = LegendSession(
        plan_path=target,
        state_store=StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl"),
        account_labels=["primary"],
        live_requested=False,
    )
    session.entry_date = ENTRY
    result = session.preflight(connect=False)
    assert result["qualified"] == ["SPY", "QQQ"]
    assert result["live"] is False


def test_runner_revalidates_signal_history_before_loading_trade_context(tmp_path):
    session = LegendSession(
        plan_path=tmp_path / "plan.json",
        state_store=StateStore(tmp_path / "state.json", tmp_path / "audit.jsonl"),
        account_labels=["primary"],
        live_requested=False,
    )
    session.entry_date = ENTRY
    session.plan = plan()
    changed = history()
    changed.iloc[-1, changed.columns.get_loc("high")] += 0.01
    session.feed = SimpleNamespace(
        stock=lambda symbol: symbol, historical_bars=lambda *args, **kwargs: changed
    )
    with pytest.raises(RuntimeError, match="changed after signal preparation"):
        session._load_market_contexts()


def test_qualified_native_setup_runs_existing_0931_1030_simulator():
    from legend_etf.core import simulate_etf_trade

    setup = evaluate_etf_setup(history(), entry_date=ENTRY)
    index = pd.date_range(f"{ENTRY} 09:30", f"{ENTRY} 15:59", freq="1min", tz=ET)
    price = setup["initial_ema"] - 5
    bars = pd.DataFrame(
        {
            "open": price,
            "high": price + 0.1,
            "low": price - 0.1,
            "close": price,
            "volume": 1.0,
        },
        index=index,
    )
    result = simulate_etf_trade(
        bars, entry_date=ENTRY, initial_ema=setup["initial_ema"]
    )
    assert result["traded"]
    assert result["exit_reason"] == "time_stop"
    assert result["exit_ts"] == pd.Timestamp(f"{ENTRY} 10:30", tz=ET)


def test_native_parity_evidence_requires_both_symbols_and_current_source(
    tmp_path, monkeypatch
):
    from legend_etf import reservations

    source = tmp_path / "input.parquet"
    source.write_bytes(b"test fixture only")
    stat = source.stat()
    record = {
        "path": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": reservations.file_sha256(source),
    }
    monkeypatch.setattr(
        reservations, "candidate_pipeline_attestation", lambda _: {"tree": "current"}
    )
    evidence = {
        "protocol": "legend-etf-native-candidate-parity-v1",
        "status": "pass",
        "completed_at": "2026-09-16T12:00:00Z",
        "runtime_seconds": 1.0,
        "range": {"start": "2012-01-01", "end": "2026-08-28"},
        "inputs": {symbol: record.copy() for symbol in ("SPY", "QQQ")},
        "counts": {
            symbol: {
                "evaluated": 3000,
                "blocked_history": 0,
                "reference": 100,
                "production": 100,
                "mismatches": 0,
                "max_ema_delta": 0.0,
                "max_ratio_delta": 0.0,
            }
            for symbol in ("SPY", "QQQ")
        },
        "candidate_pipeline": {"tree": "current"},
    }
    path = tmp_path / "evidence.json"
    atomic_write_json(path, evidence)
    reservations.validate_candidate_parity_evidence(path, legend_root=tmp_path)
    evidence["counts"]["QQQ"]["mismatches"] = 1
    atomic_write_json(path, evidence)
    with pytest.raises(RuntimeError, match="mismatched"):
        reservations.validate_candidate_parity_evidence(path, legend_root=tmp_path)
    evidence["counts"]["QQQ"]["mismatches"] = 0
    evidence["candidate_pipeline"] = {"tree": "old"}
    atomic_write_json(path, evidence)
    with pytest.raises(RuntimeError, match="stale"):
        reservations.validate_candidate_parity_evidence(path, legend_root=tmp_path)


def test_signal_cli_defaults_to_etf_only():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/prepare_legend_etf_signals.py"
    spec = importlib.util.spec_from_file_location("native_signal_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.make_parser().parse_args([]).source == "ibkr"
