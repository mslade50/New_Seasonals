import pandas as pd
import pytest

from scripts.review_dial_hedge import NAV, hysteresis, next_open_pnl


def test_new_hedge_does_not_capture_pre_entry_overnight_move():
    index = pd.date_range("2026-01-05", periods=3)
    exposure = pd.Series([0, -1, 0], index=index)
    opening = pd.Series([100, 90, 80], index=index)
    closing = pd.Series([100, 90, 80], index=index)
    pnl, cost = next_open_pnl(exposure, opening, closing, cost_bps=0)
    assert pnl.iloc[1] == 0
    assert pnl.iloc[2] == pytest.approx(10 / 90)
    assert cost.sum() == 0


def test_rebalance_and_terminal_exit_are_charged():
    index = pd.date_range("2026-01-05", periods=3)
    exposure = pd.Series([-0.2, -0.4, -0.1], index=index)
    price = pd.Series([100, 100, 100], index=index)
    pnl, cost = next_open_pnl(exposure, price, price, cost_bps=2)
    assert cost.sum() * NAV == pytest.approx((0.2 + 0.2 + 0.3 + 0.1) * NAV * 0.0002)
    assert pnl.sum() == -cost.sum()


def test_existing_thresholds_are_not_retuned():
    values = pd.Series([49, 50, 47, float("nan"), 45, 44.9, 49])
    assert hysteresis(values).tolist() == [False, True, True, True, True, False, False]
