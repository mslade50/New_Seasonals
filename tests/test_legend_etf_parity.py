"""Exact replay against the local, immutable research evidence when present."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from legend_etf.core import simulate_etf_trade

PRIMARY_VARIANT = "etf_geometry_dynamic_ema_ex_exdiv"


def _research_root() -> Path | None:
    for parent in [Path.cwd(), *Path.cwd().parents]:
        candidate = (
            parent
            / "artifacts"
            / "legend-etf-execution-backtest-20260901"
        )
        if (candidate / "candidates.csv").exists():
            return candidate
    return None


def _workspace_root(research: Path) -> Path:
    return research.parents[1]


def _ema_seeds(
    research: Path, candidates: pd.DataFrame
) -> dict[tuple[str, pd.Timestamp], float]:
    result: dict[tuple[str, pd.Timestamp], float] = {}
    data_root = _workspace_root(research) / "data" / "intraday"
    for symbol in candidates["etf"].unique():
        frame = pd.read_parquet(data_root / f"{symbol}_15min.parquet", columns=["ts", "close"])
        index = pd.DatetimeIndex(pd.to_datetime(frame["ts"]))
        if index.tz is None:
            index = index.tz_localize("America/New_York")
        else:
            index = index.tz_convert("America/New_York")
        ema = pd.Series(frame["close"].astype(float).to_numpy(), index=index).ewm(
            span=20, adjust=False, min_periods=20
        ).mean()
        for value in candidates.loc[candidates["etf"].eq(symbol), "entry_date"].unique():
            day = pd.Timestamp(value).normalize()
            start = day.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
            position = int(ema.index.searchsorted(start, side="left")) - 1
            if position >= 0 and pd.notna(ema.iloc[position]):
                result[(symbol, day)] = float(ema.iloc[position])
    return result


def test_production_execution_replays_all_342_golden_candidates():
    research = _research_root()
    if research is None:
        pytest.skip("local Legend research evidence is not present")
    candidates = pd.read_csv(
        research / "candidates.csv", parse_dates=["setup_date", "entry_date"]
    )
    expected = pd.read_csv(research / "trades_and_skips.csv")
    expected = expected.loc[expected["variant"].eq(PRIMARY_VARIANT)].copy()
    expected["setup_date"] = pd.to_datetime(expected["setup_date"])
    expected["entry_date"] = pd.to_datetime(expected["entry_date"])
    actions = pd.read_csv(research / "etf_corporate_actions.csv")
    actions["date"] = pd.to_datetime(actions["date"]).dt.normalize()
    dividends = {
        (row.etf, row.date): float(row.dividend)
        for row in actions.itertuples(index=False)
        if float(row.dividend) > 0
    }
    seeds = _ema_seeds(research, candidates)
    expected_by_key = {
        (row.root, row.entry_date): row for row in expected.itertuples(index=False)
    }

    actual_rows = []
    for candidate in candidates.itertuples(index=False):
        event_path = (
            research
            / "ibkr_1m_events"
            / candidate.root
            / f"{candidate.entry_date.date().isoformat()}.parquet"
        )
        minutes = pd.read_parquet(event_path)
        ex_dividend = (candidate.etf, candidate.entry_date) in dividends
        result = simulate_etf_trade(
            minutes,
            entry_date=candidate.entry_date,
            initial_ema=seeds[(candidate.etf, candidate.entry_date)],
            ex_dividend=ex_dividend,
        )
        expected_row = expected_by_key[(candidate.root, candidate.entry_date)]
        expected_reason = "" if pd.isna(expected_row.skip_reason) else expected_row.skip_reason
        assert result["traded"] == bool(expected_row.traded)
        assert result["skip_reason"] == expected_reason
        if result["traded"]:
            assert result["direction"] == int(expected_row.direction)
            assert result["side"] == expected_row.side
            assert result["initial_target"] == pytest.approx(expected_row.initial_target)
            assert pd.Timestamp(result["entry_ts"]) == pd.Timestamp(expected_row.entry_ts)
            assert result["entry_price"] == pytest.approx(expected_row.entry_price_raw)
            assert pd.Timestamp(result["exit_ts"]) == pd.Timestamp(expected_row.exit_ts)
            assert result["exit_price"] == pytest.approx(expected_row.exit_price_raw)
            assert result["exit_reason"] == expected_row.exit_reason
            if not np.isnan(expected_row.target_at_exit):
                assert result["target_at_exit"] == pytest.approx(expected_row.target_at_exit)
        actual_rows.append(
            {
                "root": candidate.root,
                "traded": bool(result["traded"]),
                "skip_reason": result["skip_reason"],
            }
        )

    actual = pd.DataFrame(actual_rows)
    assert len(actual) == 342
    assert int(actual["traded"].sum()) == 315
    assert actual.loc[actual["traded"]].groupby("root").size().to_dict() == {
        "ES": 101,
        "NQ": 131,
        "RTY": 83,
    }
    assert actual.loc[~actual["traded"], "skip_reason"].value_counts().to_dict() == {
        "opportunity_passed_in_0930_minute": 22,
        "ex_dividend": 5,
    }
