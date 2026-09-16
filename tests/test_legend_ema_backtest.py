import pandas as pd

from research.legend_ema_backtest import (
    BacktestConfig,
    limit_fill,
    qualifies_setup,
    session_touch_flags,
    simulate_trade,
)


def test_trade_target_uses_previous_completed_bar_without_lookahead():
    bars = pd.DataFrame(
        [
            {
                "ts": "2026-01-05 09:30",
                "open": 100.0,
                "high": 104.0,
                "low": 99.0,
                "close": 103.0,
                "ema_prev": 105.0,
                "ema": 101.0,
            },
            {
                "ts": "2026-01-05 09:45",
                "open": 103.0,
                "high": 104.0,
                "low": 100.0,
                "close": 101.0,
                "ema_prev": 101.0,
                "ema": 101.0,
            },
        ]
    )
    result = simulate_trade(bars, entry=100.0, side=1, initial_target=105.0)
    assert result["exit_bar"] == 1
    assert result["exit"] == 103.0  # marketable limit gets opening improvement
    assert result["active_target"] == 101.0


def test_ambiguous_stop_and_target_bar_is_stop_first():
    bars = pd.DataFrame(
        [
            {
                "ts": "2026-01-05 09:30",
                "open": 100.0,
                "high": 106.0,
                "low": 94.0,
                "close": 101.0,
                "ema_prev": 105.0,
                "ema": 102.0,
            },
        ]
    )
    result = simulate_trade(
        bars,
        entry=100.0,
        side=1,
        initial_target=105.0,
        stop_distance=5.0,
    )
    assert result["exit_reason"] == "stop"
    assert result["exit"] == 95.0


def test_marketable_target_at_open_precedes_stop_touched_later():
    bars = pd.DataFrame(
        [
            {
                "ts": "2026-01-05 09:30",
                "open": 106.0,
                "high": 107.0,
                "low": 94.0,
                "close": 101.0,
                "ema_prev": 105.0,
                "ema": 102.0,
            },
        ]
    )
    result = simulate_trade(
        bars,
        entry=100.0,
        side=1,
        initial_target=105.0,
        stop_distance=5.0,
    )
    assert result["exit_reason"] == "ema_target"
    assert result["exit"] == 106.0


def test_limit_gap_through_receives_price_improvement():
    row = pd.Series({"open": 94.0, "high": 96.0, "low": 93.0})
    assert limit_fill(row, target=95.0, side=-1) == 94.0


def test_touch_is_inclusive_and_close_only_is_looser():
    group = pd.DataFrame(
        {
            "low": [99.0, 101.0],
            "high": [100.0, 103.0],
            "close": [101.0, 102.0],
            "ema": [100.0, 100.5],
            "ema_prev": [98.0, 100.0],
        }
    )
    flags = session_touch_flags(group)
    assert flags["no_touch_same_bar"] is False  # first high equals EMA
    assert flags["no_touch_close_only"] is True


def test_trend_threshold_is_fixed_absolute_body_fraction():
    row = pd.Series(
        {
            "full_session": True,
            "trend_ratio": 0.75,
            "no_touch_same_bar": True,
            "no_touch_prior_bar": True,
            "no_touch_close_only": True,
        }
    )
    assert qualifies_setup(row, BacktestConfig(trend_threshold=0.75))
    assert not qualifies_setup(row, BacktestConfig(trend_threshold=0.751))
