from datetime import datetime, timezone

import pytest

from scripts import finish_ep_morning as finish


@pytest.mark.parametrize("status,hour,minute,expected", [
    ("DEADLINE_MISSED", 13, 40, True), ("DEADLINE_MISSED", 13, 45, True),
    ("RESUME", 12, 40, False), ("DELIVERED", 13, 40, False),
    ("FAILURE_REPORTED", 13, 40, False), ("DELIVERY_UNCERTAIN", 13, 40, False),
    ("PAUSED_BY_USER", 13, 40, False), ("NOT_DUE", 13, 40, False),
    ("DEADLINE_MISSED", 14, 40, False), ("DEADLINE_MISSED", 13, 51, False),
])
def test_independent_deadline_respects_terminal_states_and_time(monkeypatch, tmp_path, status, hour, minute, expected):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 24, hour, minute, tzinfo=timezone.utc).astimezone(tz)
    monkeypatch.setattr(finish, "datetime", Clock)
    monkeypatch.setattr(finish, "ROOT", tmp_path)
    monkeypatch.setattr(finish, "inspect_morning", lambda *_: {"status": status})
    monkeypatch.setattr(finish, "resolve_email_settings", lambda **_: "test-settings")
    sent = []
    monkeypatch.setattr(finish, "deliver_once", lambda *args: sent.append(args))
    finish.main(["--env-file", str(tmp_path / "unused.env"), "--send"])
    assert bool(sent) is expected
    if sent:
        assert sent[0][0].kind == "failure"


def test_deadline_race_accepts_other_confirmed_sender(monkeypatch, tmp_path):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 24, 13, 40, tzinfo=timezone.utc).astimezone(tz)
    monkeypatch.setattr(finish, "datetime", Clock)
    monkeypatch.setattr(finish, "ROOT", tmp_path)
    states = iter(["DEADLINE_MISSED", "FAILURE_REPORTED"])
    monkeypatch.setattr(finish, "inspect_morning", lambda *_: {"status": next(states)})
    monkeypatch.setattr(finish, "resolve_email_settings", lambda **_: "test-settings")
    def race(*_): raise finish.EmailDeliveryError("other process sent first")
    monkeypatch.setattr(finish, "deliver_once", race)
    assert finish.main(["--env-file", str(tmp_path / "unused.env"), "--send"]) == 0
