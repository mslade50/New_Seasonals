from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from episodic_pivot import email_delivery as mail
from episodic_pivot import morning_completion as mc

TARGET = "2026-09-23"
NOW = datetime(2026, 9, 23, 12, 40, tzinfo=timezone.utc)


def receipt(root, *, kind="morning", status="SENT", target=TARGET, when=NOW, name="run"):
    path = root / ("email_failures" if kind == "failure" else name) / (
        f"{target}-morning-test.email-delivery.json" if kind == "failure" else "email_delivery.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"record_type": "EP_RESEARCH_EMAIL_DELIVERY_V1", "schema_version": 1,
             "status": status, "kind": kind, "sent_at": when.isoformat(), "recipient_count": 1,
             "source_sha256": "a" * 64, "delivery_id": "b" * 64, "research_only": True,
             "broker_route": "NONE", "order_submission_allowed": False,
             "metadata": {"target_session_date": target, "phase": "morning", "research_mode": mc.MODE}}
    path.write_text(json.dumps(value))
    return path


def test_sept23_regression_tests_and_research_queue_are_not_delivery(tmp_path):
    root = tmp_path / "ep"
    root.mkdir()
    queue = root / "queue.json"
    queue.write_text(json.dumps({"target_session_date": TARGET, "targets": ["candidate"]}))
    (root / "audit.txt").write_text("57 tests passed. Workflow already enforces completed research.")
    mc.checkpoint(root, TARGET, "RESEARCH", {"queue": queue}, now=NOW)
    result = mc.inspect_morning(root, TARGET, now=NOW)
    assert result["status"] == "RESUME"
    assert result["progress"]["stage"] == "RESEARCH"
    assert result["progress"]["artifacts"]["queue"]["path"] == str(queue)
    assert mc.inspect_morning(root, TARGET, now=NOW.replace(hour=13, minute=30))["status"] == "DEADLINE_MISSED"


def test_missing_checkpoint_cannot_hide_missing_delivery(tmp_path):
    result = mc.inspect_morning(tmp_path, TARGET, now=NOW)
    assert result["status"] == "RESUME" and result["progress"]["stage"] == "NOT_STARTED"
    assert not list(tmp_path.iterdir())


def test_explicit_user_pause_blocks_recovery_and_email(tmp_path, smtp):
    mc.checkpoint(tmp_path, TARGET, "PAUSED_BY_USER", {}, now=NOW)
    assert mc.inspect_morning(tmp_path, TARGET, now=NOW)["status"] == "PAUSED_BY_USER"
    assert not mc.claim_resume(tmp_path, TARGET, now=NOW)
    with pytest.raises(mail.EmailDeliveryError, match="paused"):
        mc.deliver_once(payload(tmp_path, kind="failure"), SETTINGS, tmp_path, now_fn=lambda: NOW)
    assert not smtp


@pytest.mark.parametrize("kind,expected", [("morning", "DELIVERED"), ("failure", "FAILURE_REPORTED")])
def test_receipt_is_completion_authority(tmp_path, kind, expected):
    receipt(tmp_path, kind=kind)
    assert mc.inspect_morning(tmp_path, TARGET, now=NOW)["status"] == expected
    assert mc.inspect_morning(tmp_path, TARGET, now=NOW.replace(hour=14))["status"] == expected
    assert not mc.claim_resume(tmp_path, TARGET, now=NOW)


@pytest.mark.parametrize("status", ["SENDING", "AMBIGUOUS", "FAILED", "DRY_RUN"])
def test_unconfirmed_email_blocks_automatic_retry(tmp_path, status):
    receipt(tmp_path, status=status)
    assert mc.inspect_morning(tmp_path, TARGET, now=NOW)["status"] == "DELIVERY_UNCERTAIN"
    assert not mc.claim_resume(tmp_path, TARGET, now=NOW)


@pytest.mark.parametrize("mutation", ["schema", "hash", "recipient", "orders", "future", "postopen", "mode", "corrupt"])
def test_invalid_receipt_cannot_claim_success(tmp_path, mutation):
    path = receipt(tmp_path)
    value = json.loads(path.read_text())
    if mutation == "schema": value["record_type"] = "CUSTOM_DONE"
    elif mutation == "hash": value["delivery_id"] = "test"
    elif mutation == "recipient": value["recipient_count"] = 0
    elif mutation == "orders": value["order_submission_allowed"] = True
    elif mutation == "future": value["sent_at"] = (NOW + timedelta(minutes=1)).isoformat()
    elif mutation == "postopen": value["sent_at"] = NOW.replace(hour=14).isoformat()
    elif mutation == "mode": value["metadata"]["research_mode"] = "OFFLINE"
    path.write_text("{" if mutation == "corrupt" else json.dumps(value))
    os.utime(path, (NOW.timestamp(), NOW.timestamp()))
    assert mc.inspect_morning(tmp_path, TARGET, now=NOW)["status"] == "DELIVERY_UNCERTAIN"


def test_prior_session_and_night_failure_do_not_satisfy_morning(tmp_path):
    receipt(tmp_path, target="2026-09-22", when=NOW - timedelta(days=1))
    path = receipt(tmp_path, kind="failure")
    value = json.loads(path.read_text())
    value["metadata"]["phase"] = "night"
    path.write_text(json.dumps(value))
    assert mc.inspect_morning(tmp_path, TARGET, now=NOW)["status"] == "RESUME"


def test_conflicting_receipts_require_reconciliation(tmp_path):
    receipt(tmp_path)
    receipt(tmp_path, kind="failure")
    assert mc.inspect_morning(tmp_path, TARGET, now=NOW)["status"] == "DELIVERY_UNCERTAIN"


@pytest.mark.parametrize("when,target", [(NOW.replace(hour=12, minute=19), TARGET),
    (datetime(2026, 9, 26, 12, 40, tzinfo=timezone.utc), "2026-09-26"),
    (datetime(2026, 12, 25, 14, 0, tzinfo=timezone.utc), "2026-12-25"), (NOW, "2026-09-22")])
def test_non_session_and_not_due_checks_are_read_only(tmp_path, when, target):
    assert mc.inspect_morning(tmp_path, target, now=when)["status"] == "NOT_DUE"
    assert not list(tmp_path.iterdir())


def test_checkpoint_preserves_paths_and_detects_changed_inputs(tmp_path):
    queue, notes = tmp_path / "q.json", tmp_path / "notes.json"
    queue.write_text("{}")
    notes.write_text("[]")
    root = tmp_path / "ep"
    mc.checkpoint(root, TARGET, "RESEARCH", {"queue": queue}, now=NOW)
    mc.checkpoint(root, TARGET, "RESEARCH", {"notes": notes}, now=NOW)
    queue.write_text('{"changed":true}')
    state = mc.progress(root, TARGET)
    assert set(state["artifacts"]) == {"queue", "notes"}
    assert state["changed_artifacts"] == ["queue"]
    assert mc.inspect_morning(root, TARGET, now=NOW)["status"] == "RESUME"
    with pytest.raises(ValueError):
        mc.checkpoint(root, TARGET, "DONE", {}, now=NOW)


def test_invalid_progress_does_not_override_confirmed_receipt(tmp_path):
    mc.checkpoint(tmp_path, TARGET, "DISCOVERY", {}, now=NOW)
    mc._journal(tmp_path, TARGET).write_text("bad")
    receipt(tmp_path)
    result = mc.inspect_morning(tmp_path, TARGET, now=NOW)
    assert result["status"] == "DELIVERED" and result["progress"]["stage"] == "INVALID_CHECKPOINT"


def test_recovery_claim_is_atomic_bounded_and_not_reused(tmp_path):
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _: mc.claim_resume(tmp_path, TARGET, now=NOW), range(2)))
    assert sorted(claims) == [False, True]
    assert not mc.claim_resume(tmp_path, TARGET, now=NOW + timedelta(minutes=10))
    assert mc.claim_resume(tmp_path, TARGET, now=NOW + timedelta(minutes=20))
    assert not mc.claim_resume(tmp_path, TARGET, now=NOW.replace(hour=13, minute=30))


def payload(root, name="run", kind="morning"):
    if kind == "failure":
        return mail.failure_payload(phase="morning", summary="Morning research did not finish before the deadline.",
                                    target_session_date=TARGET, output_root=root)
    return mail.EmailPayload(kind="morning", subject="EP test", html_body="test", plain_body="test",
                            attachments=(), receipt_path=root / name / "email_delivery.json", source_sha256="a" * 64,
                            metadata={"target_session_date": TARGET, "research_mode": mc.MODE, "generated_at": NOW.isoformat()})


@pytest.fixture
def smtp(monkeypatch):
    sent = []

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    class SMTP:
        def __init__(self, *_args, **_kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def ehlo(self): pass
        def starttls(self, **_kwargs): pass
        def login(self, *_args): pass
        def send_message(self, message):
            sent.append(message)
            return {}

    monkeypatch.setattr(mail, "datetime", Clock)
    monkeypatch.setattr(mail.smtplib, "SMTP", SMTP)
    return sent


SETTINGS = mail.EmailSettings(sender="sender@example.com", password="test-only", recipients=("recipient@example.com",))


def test_parallel_report_and_failure_can_send_only_once(tmp_path, smtp):
    def send(kind):
        try:
            return mc.deliver_once(payload(tmp_path, kind=kind), SETTINGS, tmp_path, now_fn=lambda: NOW)
        except mail.EmailDeliveryError:
            return "BLOCKED"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(send, ["morning", "failure"]))
    assert sorted(outcomes) == ["BLOCKED", "SENT"]
    assert len(smtp) == 1


def test_rebuilt_report_cannot_duplicate_session_delivery(tmp_path, smtp):
    first = payload(tmp_path)
    assert mc.deliver_once(first, SETTINGS, tmp_path, now_fn=lambda: NOW) == "SENT"
    assert mc.deliver_once(first, SETTINGS, tmp_path, now_fn=lambda: NOW) == "ALREADY_SENT"
    with pytest.raises(mail.EmailDeliveryError, match="second email"):
        mc.deliver_once(payload(tmp_path, "rebuilt"), SETTINGS, tmp_path, now_fn=lambda: NOW)
    changed_recipients = mail.EmailSettings(sender="sender@example.com", password="test-only", recipients=("other@example.com",))
    with pytest.raises(mail.EmailDeliveryError, match="different source or recipient"):
        mc.deliver_once(first, changed_recipients, tmp_path, now_fn=lambda: NOW)
    assert len(smtp) == 1


def test_ambiguous_receipt_blocks_report_and_failure(tmp_path, smtp):
    receipt(tmp_path, status="SENDING")
    for kind in ("morning", "failure"):
        with pytest.raises(mail.EmailDeliveryError, match="DELIVERY_UNCERTAIN"):
            mc.deliver_once(payload(tmp_path, kind=kind), SETTINGS, tmp_path, now_fn=lambda: NOW)
    assert not smtp


def test_cutoff_and_report_age_rechecked_inside_send_lock(tmp_path, smtp):
    with pytest.raises(mail.EmailDeliveryError, match="stale"):
        mc.deliver_once(payload(tmp_path), SETTINGS, tmp_path, now_fn=lambda: NOW.replace(hour=13, minute=30))
    with pytest.raises(mail.EmailDeliveryError, match="stale"):
        mc.deliver_once(payload(tmp_path), SETTINGS, tmp_path, now_fn=lambda: NOW + timedelta(minutes=31))
    assert not smtp


def test_slow_smtp_connection_cannot_cross_submission_deadline(tmp_path, smtp):
    times = iter([NOW, NOW, NOW.replace(hour=13, minute=30)])
    with pytest.raises(mail.EmailDeliveryError, match="stale"):
        mc.deliver_once(payload(tmp_path), SETTINGS, tmp_path, now_fn=lambda: next(times))
    assert not smtp
    assert not (tmp_path / "run/email_delivery.json").exists()


def test_failure_after_report_deadline_still_allowed(tmp_path, smtp):
    result = mc.deliver_once(payload(tmp_path, kind="failure"), SETTINGS, tmp_path,
                             now_fn=lambda: NOW.replace(hour=13, minute=40))
    assert result == "SENT" and len(smtp) == 1


def test_sept24_early_failure_cannot_end_retry_window(tmp_path, smtp):
    with pytest.raises(mail.EmailDeliveryError, match="deadline"):
        mc.deliver_once(payload(tmp_path, kind="failure"), SETTINGS, tmp_path,
                        now_fn=lambda: NOW.replace(minute=23))
    assert not smtp
    assert mc.inspect_morning(tmp_path, TARGET, now=NOW)["status"] == "RESUME"
    assert mc.claim_resume(tmp_path, TARGET, now=NOW)


def test_retry_checkpoint_retains_frozen_research(tmp_path):
    queue = tmp_path / "queue.json"
    queue.write_text("{}")
    mc.checkpoint(tmp_path, TARGET, "RESEARCH", {"queue": queue}, now=NOW)
    mc.checkpoint(tmp_path, TARGET, "RETRY_PENDING", {}, now=NOW)
    state = mc.inspect_morning(tmp_path, TARGET, now=NOW)
    assert state["status"] == "RESUME"
    assert state["progress"]["stage"] == "RETRY_PENDING"
    assert state["progress"]["artifacts"]["queue"]["path"] == str(queue)


def test_transport_disconnect_retains_claim_and_blocks_recovery(tmp_path, smtp, monkeypatch):
    def disconnect(_self, _message):
        raise mail.smtplib.SMTPServerDisconnected("synthetic DATA disconnect")

    monkeypatch.setattr(mail.smtplib.SMTP, "send_message", disconnect)
    with pytest.raises(mail.EmailDeliveryError, match="ambiguous"):
        mc.deliver_once(payload(tmp_path), SETTINGS, tmp_path, now_fn=lambda: NOW)
    assert mc.inspect_morning(tmp_path, TARGET, now=NOW)["status"] == "DELIVERY_UNCERTAIN"
    for kind in ("morning", "failure"):
        with pytest.raises(mail.EmailDeliveryError, match="second email"):
            mc.deliver_once(payload(tmp_path, kind=kind), SETTINGS, tmp_path, now_fn=lambda: NOW)


def test_quit_disconnect_after_acceptance_does_not_repeat_email(tmp_path, smtp, monkeypatch):
    def disconnect(*_args):
        raise mail.smtplib.SMTPServerDisconnected("synthetic QUIT disconnect")

    monkeypatch.setattr(mail.smtplib.SMTP, "__exit__", disconnect)
    assert mc.deliver_once(payload(tmp_path), SETTINGS, tmp_path, now_fn=lambda: NOW) == "SENT"
    assert mc.deliver_once(payload(tmp_path), SETTINGS, tmp_path, now_fn=lambda: NOW) == "ALREADY_SENT"
    assert len(smtp) == 1
