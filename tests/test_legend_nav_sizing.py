"""Accounting tests for the research-only native Legend NAV study."""
import pandas as pd
import pytest

from research.legend_nav_sizing import portfolio_paths, return_paths, research_atr
from legend_etf.calendar import session_labels


def fixture_trade(exit_reason="time_stop", exit_minute="10:30", exit_price=100):
    grid = pd.date_range("2024-01-03 09:30", "2024-01-03 10:30", freq="1min", tz="America/New_York")
    bars = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100., "volume": 100}, index=grid)
    trade = {"entry_ts": grid[1], "exit_ts": pd.Timestamp(f"2024-01-03 {exit_minute}", tz="America/New_York"),
             "entry_price": 100., "exit_price": exit_price, "exit_reason": exit_reason}
    return bars, trade


def test_same_day_weights_add_without_redistribution_and_cost_is_notional():
    index = pd.date_range("2024-01-03 09:31", periods=2, freq="1min")
    trades = pd.DataFrame({"entry_date": ["2024-01-03"] * 2, "weight": [.4, .3], "gross_return_bps": [100, -200]})
    paths = {0: (pd.Series(.01, index=index), pd.Series(-.01, index=index)),
             1: (pd.Series(-.02, index=index), pd.Series(-.03, index=index))}
    days, _ = portfolio_paths(trades, paths, "weight", 2)
    assert days.iloc[0]["return"] == pytest.approx(-.00214)
    solo, _ = portfolio_paths(trades.iloc[:1], paths, "weight", 2)
    assert solo.iloc[0]["return"] == pytest.approx(.00392)


def test_first_loss_is_drawdown_from_initial_nav_and_days_compound():
    trades = pd.DataFrame({"entry_date": ["2024-01-03", "2024-01-04"], "weight": [.4, .4], "gross_return_bps": [-100, 100]})
    paths = {i: (pd.Series([r], index=[pd.Timestamp(day)]), pd.Series([r], index=[pd.Timestamp(day)]))
             for i, (r, day) in enumerate([(-.01, "2024-01-03"), (.01, "2024-01-04")])}
    days, stats = portfolio_paths(trades, paths, "weight", 0)
    assert days.iloc[0].closed_drawdown == pytest.approx(-.004)
    assert days.iloc[-1].nav == pytest.approx(.996 * 1.004)
    assert stats["closed_max_drawdown_pct"] == pytest.approx(-.4)


@pytest.mark.parametrize("reason,minute,price", [("time_stop", "10:30", 100), ("etf_ema_limit", "09:35", 100)])
def test_opening_exit_does_not_include_later_low(reason, minute, price):
    bars, trade = fixture_trade(reason, minute, price)
    bars.loc[trade["exit_ts"]:, "low"] = 50
    _, lower = return_paths(bars, trade)
    assert lower.min() == pytest.approx(-.01)


def test_intrabar_limit_exit_low_is_conservative_but_later_minutes_excluded():
    bars, trade = fixture_trade("etf_ema_limit", "09:35", 101)
    bars.loc[trade["exit_ts"], "low"] = 98
    bars.loc[trade["exit_ts"] + pd.Timedelta(minutes=1):, "low"] = 50
    close, lower = return_paths(bars, trade)
    assert lower.min() == pytest.approx(-.02)
    assert close.iloc[-1] == pytest.approx(.01)


def test_atr_allows_verified_closure_rejects_real_gap_and_excludes_entry_day():
    dates = session_labels("2023-01-03", "2025-02-03").difference(pd.DatetimeIndex(["2025-01-09"]))
    bars = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.}, index=dates)
    bars.loc[pd.Timestamp("2025-02-03"), ["high", "close"]] = 10000
    assert research_atr(bars, "2025-02-03", "2025-01-31") == pytest.approx(2)
    with pytest.raises(ValueError, match="missing or unexpected"):
        research_atr(bars.drop(pd.Timestamp("2025-01-08")), "2025-02-03", "2025-01-31")
