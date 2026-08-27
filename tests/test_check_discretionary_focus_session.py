import datetime as dt

from scripts import check_discretionary_focus_session as gate


def test_session_gate_uses_new_york_market_date() -> None:
    should_run, market_date = gate.session_gate(
        dt.datetime(2026, 9, 8, 12, 35, tzinfo=dt.timezone.utc)
    )
    assert should_run is True
    assert market_date == dt.date(2026, 9, 8)


def test_session_gate_skips_weekday_exchange_holiday_and_weekend() -> None:
    assert gate.is_nyse_session(dt.date(2026, 9, 7)) is False  # Labor Day
    assert gate.is_nyse_session(dt.date(2026, 9, 6)) is False  # Sunday
    assert gate.is_nyse_session(dt.date(2026, 9, 8)) is True


def test_scheduled_gate_selects_exactly_one_dst_aware_utc_slot() -> None:
    summer = dt.datetime(2026, 8, 26, 12, 42, tzinfo=dt.timezone.utc)
    assert gate.scheduled_session_gate(summer, "35 12 * * 1-5")[0] is True
    assert gate.scheduled_session_gate(summer, "35 13 * * 1-5")[0] is False

    winter = dt.datetime(2026, 1, 6, 13, 49, tzinfo=dt.timezone.utc)
    assert gate.scheduled_session_gate(winter, "35 12 * * 1-5")[0] is False
    assert gate.scheduled_session_gate(winter, "35 13 * * 1-5")[0] is True


def test_scheduled_gate_refuses_materially_early_or_late_runner_start() -> None:
    early = dt.datetime(2026, 8, 26, 12, 24, tzinfo=dt.timezone.utc)
    late = dt.datetime(2026, 8, 26, 13, 11, tzinfo=dt.timezone.utc)
    assert gate.scheduled_session_gate(early, "35 12 * * 1-5")[0] is False
    assert gate.scheduled_session_gate(late, "35 12 * * 1-5")[0] is False


def test_delivery_window_has_a_hard_premarket_cutoff() -> None:
    allowed = dt.datetime(2026, 8, 26, 13, 20, tzinfo=dt.timezone.utc)
    late = dt.datetime(2026, 8, 26, 13, 21, tzinfo=dt.timezone.utc)
    assert gate.delivery_window_gate(allowed)[0] is True
    assert gate.delivery_window_gate(late)[0] is False


def test_cli_emits_github_outputs(tmp_path) -> None:
    output = tmp_path / "github-output.txt"
    original = list(gate.sys.argv)
    gate.sys.argv = [
        "check_discretionary_focus_session.py",
        "--now",
        "2026-09-07T12:35:00Z",
        "--github-output",
        str(output),
        "--scheduled-cron",
        "35 12 * * 1-5",
    ]
    try:
        assert gate.main() == 0
    finally:
        gate.sys.argv = original

    assert output.read_text(encoding="utf-8").splitlines() == [
        "should_run=false",
        "market_date=2026-09-07",
    ]
