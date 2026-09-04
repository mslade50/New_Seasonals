import pandas as pd
import pytest

from research.treasury_month_end.strategy import (
    instruction_for_session,
    pilot_order,
    schedule_from_sessions,
)


def test_schedule_uses_sixth_last_close_and_month_end_close():
    sessions = pd.bdate_range("2026-07-01", "2026-07-31")
    schedule = schedule_from_sessions(sessions)
    assert schedule.entry_date == sessions[-6]
    assert schedule.exit_date == sessions[-1]
    assert schedule.sessions_held == 5


def test_instructions_are_state_aware():
    sessions = pd.bdate_range("2026-07-01", "2026-07-31")
    schedule = schedule_from_sessions(sessions)
    assert instruction_for_session(schedule.entry_date, schedule, position_open=False) == "ENTER_LONG_MOC"
    assert instruction_for_session(schedule.entry_date, schedule, position_open=True) == "HOLD"
    assert instruction_for_session(schedule.exit_date, schedule, position_open=True) == "EXIT_LONG_MOC"
    assert instruction_for_session(sessions[0], schedule, position_open=False) == "FLAT"


def test_schedule_rejects_mixed_months():
    with pytest.raises(ValueError):
        schedule_from_sessions(["2026-07-31", "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"])


def test_pilot_order_caps_aggregate_tlt_exposure():
    order = pilot_order(
        account_value=750_000,
        tlt_price=100,
        existing_tlt_notional=120_000,
    )
    assert order.shares == 300
    assert order.incremental_notional == 30_000
    assert order.aggregate_tlt_nav_pct_after == pytest.approx(0.20)


def test_pilot_order_uses_ten_percent_when_capacity_is_open():
    order = pilot_order(account_value=750_000, tlt_price=100)
    assert order.shares == 750
    assert order.incremental_nav_pct == pytest.approx(0.10)

