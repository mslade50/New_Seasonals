import numpy as np
import pandas as pd

from scripts.backtest_legend_ema_futures import (
    BASE_VARIANT,
    DEFAULT_VARIANTS,
    EOD_VARIANT,
    _active_ema_for_minutes,
    _limit_fill,
    _session_through_time_stop,
    max_drawdown,
    round_order_price,
    simulate_trade,
    summarize_trades,
)


def _row(open_, high, low, close):
    return pd.Series({"open": open_, "high": high, "low": low, "close": close})


def test_primary_executable_variant_adopts_universal_1030_time_stop():
    variants = {variant.name: variant for variant in DEFAULT_VARIANTS}

    assert BASE_VARIANT == "executable_time_stop_1030"
    assert variants[BASE_VARIANT].time_stop == "10:30"
    assert variants[BASE_VARIANT].fallback_exit == "last_minute_open"
    assert variants[EOD_VARIANT].time_stop is None


def test_limit_fill_handles_touched_and_marketable_limits():
    row = _row(100, 103, 99, 102)

    assert _limit_fill(row, 102, direction=1) == 102
    assert _limit_fill(row, 99.5, direction=1) == 100
    assert _limit_fill(row, 99.5, direction=-1) == 99.5
    assert _limit_fill(row, 101, direction=-1) == 100


def test_active_ema_uses_only_last_completed_fifteen_minute_bar():
    bar_index = pd.DatetimeIndex(
        ["2026-01-05 09:15", "2026-01-05 09:30"], tz="America/New_York"
    )
    bars = pd.DataFrame({"ema20_rth": [101.0, 102.0]}, index=bar_index)
    minute_index = pd.DatetimeIndex(
        ["2026-01-05 09:30", "2026-01-05 09:44", "2026-01-05 09:45"],
        tz="America/New_York",
    )

    active = _active_ema_for_minutes(bars, minute_index, "rth")

    assert active.tolist() == [101.0, 101.0, 102.0]

    latency_safe_index = pd.DatetimeIndex(
        ["2026-01-05 09:45", "2026-01-05 09:46"], tz="America/New_York"
    )
    latency_safe = _active_ema_for_minutes(
        bars, latency_safe_index, "rth", activation_delay_minutes=1
    )
    assert latency_safe.tolist() == [101.0, 102.0]


def test_order_rounding_uses_valid_conservative_ticks():
    assert round_order_price(101.13, 0.25, direction=1, order="limit") == 101.25
    assert round_order_price(101.13, 0.25, direction=-1, order="limit") == 101.0
    assert round_order_price(98.87, 0.25, direction=1, order="stop") == 98.75
    assert round_order_price(101.13, 0.25, direction=-1, order="stop") == 101.25


def test_trade_updates_target_and_exits_at_second_bar_ema():
    index = pd.date_range("2026-01-05 09:30", periods=3, freq="min", tz="America/New_York")
    session = pd.DataFrame(
        {
            "open": [100.0, 100.5, 101.0],
            "high": [100.5, 101.0, 102.5],
            "low": [99.5, 100.0, 100.5],
            "close": [100.25, 100.75, 102.0],
        },
        index=index,
    )
    active_ema = pd.Series([103.0, 103.0, 102.0], index=index)

    trade = simulate_trade(session, active_ema, direction=1)

    assert trade["exit_ts"] == index[2]
    assert trade["exit_price"] == 102.0
    assert trade["exit_reason"] == "ema_limit"
    assert trade["gross_points"] == 2.0


def test_same_minute_stop_target_ambiguity_is_conservative_stop_first():
    index = pd.date_range("2026-01-05 09:30", periods=1, freq="min", tz="America/New_York")
    session = pd.DataFrame(
        {"open": [100.0], "high": [102.0], "low": [98.0], "close": [101.0]},
        index=index,
    )
    active_ema = pd.Series([101.0], index=index)

    trade = simulate_trade(session, active_ema, direction=1, stop_distance=1.0)

    assert trade["exit_reason"] == "atr_stop"
    assert trade["exit_price"] == 99.0


def test_executable_fallback_uses_last_minute_open_not_known_close():
    index = pd.date_range("2026-01-05 15:58", periods=2, freq="min", tz="America/New_York")
    session = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [100.5, 102.0],
            "low": [99.5, 100.5],
            "close": [100.0, 102.0],
        },
        index=index,
    )
    active_ema = pd.Series([110.0, 110.0], index=index)

    trade = simulate_trade(
        session,
        active_ema,
        direction=1,
        fallback_exit="last_minute_open",
    )

    assert trade["exit_ts"] == index[-1]
    assert trade["exit_price"] == 101.0
    assert trade["exit_reason"] == "last_minute_open"


def test_time_stop_includes_cutoff_open_but_not_cutoff_high_low():
    index = pd.DatetimeIndex(
        [
            "2026-01-05 10:29",
            "2026-01-05 10:30",
            "2026-01-05 10:31",
        ],
        tz="America/New_York",
    )
    session = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [100.5, 110.0, 103.0],
            "low": [99.5, 100.5, 101.0],
            "close": [100.0, 109.0, 102.0],
        },
        index=index,
    )
    truncated = _session_through_time_stop(
        session,
        pd.Timestamp("2026-01-05"),
        "10:30",
    )
    active_ema = pd.Series(105.0, index=truncated.index)

    trade = simulate_trade(
        truncated,
        active_ema,
        direction=1,
        fallback_exit="last_minute_open",
    )

    assert truncated.index[-1] == index[1]
    assert trade["exit_ts"] == index[1]
    assert trade["exit_price"] == 101.0
    assert trade["exit_reason"] == "last_minute_open"


def test_max_drawdown_starts_from_zero_equity():
    assert max_drawdown([100, -30, -90, 20]) == -120
    assert max_drawdown([]) == 0


def test_summary_uses_net_pnl_and_returns():
    trades = pd.DataFrame(
        {
            "entry_ts": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            "entry_date": ["2026-01-01", "2026-01-02"],
            "symbol": ["NQ", "NQ"],
            "net_points": [2.0, -1.0],
            "pnl_dollars": [40.0, -20.0],
            "return_bps": [2.0, -1.0],
            "exit_reason": ["ema_limit", "close"],
            "roll_window": [False, False],
        }
    )

    summary = summarize_trades(trades)

    assert summary["trades"] == 2
    assert summary["win_rate_pct"] == 50
    assert summary["profit_factor"] == 2
    assert summary["total_pnl_dollars"] == 20
    assert np.isclose(summary["avg_return_bps"], 0.5)
