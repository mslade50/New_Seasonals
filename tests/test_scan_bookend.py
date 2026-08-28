import datetime as dt

from daily_scan import _bookend_timing


def _times(hour: int):
    now = dt.datetime(2026, 8, 27, hour, 0)
    return (
        now,
        now.replace(hour=9, minute=30),
        now.replace(hour=16, minute=0),
    )


def test_delayed_am_backup_keeps_settled_data_contract():
    now, market_open, market_close = _times(10)
    assert _bookend_timing("am", now, market_open, market_close) == (True, False)


def test_pm_bookend_is_never_classified_as_premarket():
    now, market_open, market_close = _times(5)
    assert _bookend_timing("pm", now, market_open, market_close) == (False, False)


def test_auto_mode_preserves_wall_clock_manual_behavior():
    now, market_open, market_close = _times(10)
    assert _bookend_timing("auto", now, market_open, market_close) == (False, True)
