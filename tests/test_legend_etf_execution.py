from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from legend_etf.calendar import is_full_session, rth_bar_starts
from legend_etf.core import (
    active_ema_series,
    build_entry_day_revisions,
    entry_decision,
    prior_wilder_atr14,
    round_limit,
    simulate_etf_trade,
)
from legend_etf.session import (
    _seed_from_history,
    aggregate_five_second_minute,
    aggregate_trade_minute,
    et_timestamp,
    first_trade_at_or_after,
    validate_submission_tape,
)

ET = "America/New_York"


def _row(open_: float, high: float, low: float, close: float) -> pd.Series:
    return pd.Series({"open": open_, "high": high, "low": low, "close": close})


def _flat_day(day: str = "2026-09-01") -> pd.DataFrame:
    index = pd.date_range(f"{day} 09:30", f"{day} 15:59", freq="1min", tz=ET)
    frame = pd.DataFrame(
        {"open": 99.0, "high": 99.1, "low": 98.9, "close": 99.0, "volume": 1.0},
        index=index.tz_convert("UTC"),
    )
    return frame


def test_etf_geometry_sets_side_and_penny_away_limit():
    short = entry_decision(_row(101, 101.1, 100.5, 100.8), initial_ema=100)
    assert short.eligible
    assert short.direction == -1
    assert short.side == "short"
    assert short.initial_target == 100.0

    long = entry_decision(_row(99, 99.5, 98.8, 99.2), initial_ema=100.001)
    assert long.eligible
    assert long.direction == 1
    assert long.initial_target == 100.01
    assert round_limit(100.009, -1) == 100.0


def test_0930_touch_and_0931_through_target_skip():
    touched = entry_decision(_row(99, 100, 98.9, 99.5), initial_ema=100)
    assert not touched.eligible
    assert touched.reason == "opportunity_passed_in_0930_minute"

    through = entry_decision(
        _row(99, 99.9, 98.9, 99.5), initial_ema=100, decision_price=100
    )
    assert not through.eligible
    assert through.reason == "decision_price_through_target"


def test_dynamic_ema_activates_one_minute_after_completed_15m_bar():
    day = _flat_day()
    revisions = build_entry_day_revisions(day, entry_date="2026-09-01", initial_ema=100)
    index = pd.date_range("2026-09-01 09:45", "2026-09-01 09:46", freq="1min", tz=ET)
    active = active_ema_series(index, revisions, 100)
    expected = (2 / 21) * 99 + (19 / 21) * 100
    assert active.loc[pd.Timestamp("2026-09-01 09:45", tz=ET)] == 100
    assert active.loc[pd.Timestamp("2026-09-01 09:46", tz=ET)] == pytest.approx(expected)


def test_time_exit_uses_1030_open_and_never_1030_high_low():
    day = _flat_day()
    cutoff = pd.Timestamp("2026-09-01 10:30", tz=ET).tz_convert("UTC")
    day.loc[cutoff, ["open", "high", "low", "close"]] = [99.2, 101.0, 98.0, 100.5]
    result = simulate_etf_trade(day, entry_date="2026-09-01", initial_ema=100)
    assert result["traded"]
    assert result["exit_reason"] == "time_stop"
    assert result["exit_ts"] == pd.Timestamp("2026-09-01 10:30", tz=ET)
    assert result["exit_price"] == 99.2


def test_ex_dividend_fails_before_entry():
    result = simulate_etf_trade(
        _flat_day(), entry_date="2026-09-01", initial_ema=100, ex_dividend=True
    )
    assert not result["traded"]
    assert result["skip_reason"] == "ex_dividend"


def test_realtime_minute_requires_all_twelve_five_second_bars():
    start = pd.Timestamp("2026-09-01 09:30", tz=ET).tz_convert("UTC")
    index = pd.date_range(start, periods=12, freq="5s")
    frame = pd.DataFrame(
        {
            "open": np.arange(12) + 100,
            "high": np.arange(12) + 100.5,
            "low": np.arange(12) + 99.5,
            "close": np.arange(12) + 100.25,
            "volume": 1,
        },
        index=index,
    )
    minute = aggregate_five_second_minute(frame, start)
    assert minute["open"] == 100
    assert minute["close"] == 111.25
    assert minute["volume"] == 12
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_five_second_minute(frame.iloc[:-1], start)


def test_tick_tape_uses_first_0931_trade_and_blocks_a_pretransmit_cross():
    timestamps = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-09-01 09:30:00", tz=ET),
            pd.Timestamp("2026-09-01 09:30:45", tz=ET),
            pd.Timestamp("2026-09-01 09:30:59", tz=ET),
            pd.Timestamp("2026-09-01 09:31:00", tz=ET),
            pd.Timestamp("2026-09-01 09:31:01", tz=ET),
        ]
    ).tz_convert("UTC")
    frame = pd.DataFrame(
        {
            "sequence": range(5),
            "timestamp": timestamps,
            "price": [99.5, 99.2, 99.4, 99.45, 99.55],
            "size": [10, 12, 7, 15, 9],
        }
    )
    opening = aggregate_trade_minute(
        frame, pd.Timestamp("2026-09-01 09:30", tz=ET)
    )
    assert opening.to_dict() == {
        "open": 99.5,
        "high": 99.5,
        "low": 99.2,
        "close": 99.4,
        "volume": 29.0,
    }
    decision = first_trade_at_or_after(
        frame, pd.Timestamp("2026-09-01 09:31", tz=ET)
    )
    assert decision["price"] == 99.45
    now = pd.Timestamp("2026-09-01 09:31:01", tz=ET)
    price, _ = validate_submission_tape(
        frame,
        entry_date="2026-09-01",
        direction=1,
        target=100.0,
        now_et=now,
    )
    assert price == 99.55
    crossed = frame.copy()
    crossed.loc[crossed.index[-1], "price"] = 100.0
    with pytest.raises(RuntimeError, match="crossed"):
        validate_submission_tape(
            crossed,
            entry_date="2026-09-01",
            direction=1,
            target=100.0,
            now_et=now,
        )


def test_dst_mapping_and_known_half_day_block():
    assert et_timestamp("2026-07-06", "09:30").hour == 9
    assert et_timestamp("2026-07-06", "09:30").tz_convert("UTC").hour == 13
    assert et_timestamp("2026-12-07", "09:30").tz_convert("UTC").hour == 14
    assert is_full_session("2026-09-01")
    assert not is_full_session("2026-11-27")


def test_etf_ema_seed_requires_the_complete_calendar_rth_grid():
    grid = rth_bar_starts("2026-08-31", "2026-08-31")
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": np.linspace(100.0, 102.5, len(grid)),
            "volume": 1_000.0,
        },
        index=grid.tz_convert("UTC"),
    )
    assert np.isfinite(
        _seed_from_history(frame, "2026-09-01", "2026-08-31")
    )
    with pytest.raises(RuntimeError, match="exact XNYS"):
        _seed_from_history(
            frame.drop(frame.index[7]), "2026-09-01", "2026-08-31"
        )


def test_atr_requires_252_contiguous_xnys_sessions():
    daily_index = rth_bar_starts(
        "2025-01-02", "2026-08-31"
    ).tz_convert(ET).normalize().unique()
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
        },
        index=daily_index,
    )
    assert prior_wilder_atr14(
        frame,
        as_of_date="2026-09-01",
        expected_last_session="2026-08-31",
    ) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="exact XNYS"):
        prior_wilder_atr14(
            frame.drop(frame.index[100]),
            as_of_date="2026-09-01",
            expected_last_session="2026-08-31",
        )
