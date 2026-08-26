"""Focused guards for the deterministic Discretionary Focus mailer."""
from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path

import pytest

from scripts import send_discretionary_focus_email as sender

NOW = dt.datetime(2026, 8, 26, 12, 0, tzinfo=dt.timezone.utc)


@pytest.fixture()
def ready_payload() -> dict:
    return {
        "schema_version": "discretionary-focus.v1",
        "research_only": True,
        "quick_review_created": False,
        "live_actions_enabled": False,
        "order_staging_enabled": False,
        "status": "READY",
        "phase": "FINAL",
        "as_of": "2026-08-26",
        "valid_for": "2026-08-26",
        "generated_at": "2026-08-26T11:45:00Z",
        "expires_at": "2026-08-26T14:00:00Z",
        "focus": [
            {
                "rank": 1,
                "ticker": "AMPL",
                "company_name": "Amplitude",
                "why_now": "Acceleration is visible while the chart tightens.",
                "setup": "Higher lows beneath a defined pivot.",
                "trigger": {"condition": "Relative volume confirms the pivot."},
                "invalidation": {
                    "technical": "The base loses its latest higher low.",
                    "thesis_kill": "Growth re-acceleration fails to persist.",
                },
                "catalyst": "Recent guidance increased.",
                "priced_in": "A meaningful post-earnings rerating is already visible.",
                "next_proof": "RPO and ARR growth remain elevated.",
                "event_date": "2026-11-04",
                "earnings_td": 49,
                "technical": {
                    "observed_at": "2026-08-26T11:40:00Z",
                    "setup_gate": "PASS",
                    "liquidity_gate": "PASS",
                    "setup_quality": 4,
                },
                "sources": [
                    {
                        "source_id": "ampl-q2",
                        "label": "Amplitude Q2 results",
                        "url": "https://investors.amplitude.com/q2",
                        "as_of": "2026-08-05",
                        "primary": True,
                    }
                ],
            }
        ],
        "screen_summary": {
            "input_count": 4,
            "technical_pass_count": 3,
            "research_pass_count": 1,
            "selected_count": 1,
            "rejected_counts": {"event_risk": 2, "loose_setup": 1},
        },
        "provenance": {
            "screen_snapshot_id": "screen-20260826-am",
            "screen_captured_at": "2026-08-26T11:35:00Z",
            "research_snapshot_id": "research-20260826-am",
            "research_as_of": "2026-08-26T11:42:00Z",
            "policy_version": "discretionary-focus-policy.v1",
        },
    }


@pytest.fixture()
def no_setup_payload(ready_payload) -> dict:
    payload = copy.deepcopy(ready_payload)
    payload["status"] = "NO_QUALIFIED_SETUP"
    payload["focus"] = []
    payload["no_setup_reason"] = (
        "No candidate cleared every technical, evidence, and event gate."
    )
    payload["screen_summary"]["research_pass_count"] = 0
    payload["screen_summary"]["selected_count"] = 0
    payload["screen_summary"]["rejected_counts"]["research_gap"] = 1
    return payload


@pytest.fixture(autouse=True)
def email_env(monkeypatch):
    monkeypatch.setenv("EMAIL_USER", "sender@example.com")
    monkeypatch.setenv("EMAIL_PASS", "app-password")
    monkeypatch.setenv(
        "DISCRETIONARY_FOCUS_RECIPIENTS",
        "one@example.com, two@example.com",
    )


def test_ready_email_is_concise_escaped_and_research_only(ready_payload):
    ready_payload["focus"][0]["why_now"] = "Growth <b>accelerates</b>."
    digest = sender.canonical_digest(ready_payload)
    body = sender.render_html(ready_payload, digest)

    assert "AMPL" in body
    assert "Growth &lt;b&gt;accelerates&lt;/b&gt;." in body
    assert "Why now" in body and "Confirmation" in body and "Invalidation" in body
    assert "Amplitude Q2 results" in body
    assert "Research attention only" in body
    assert "not an investment recommendation" in body
    for forbidden in ("place an order", "order staging", "position sizing"):
        assert forbidden not in body.lower()


def test_no_setup_is_a_valid_visible_email(no_setup_payload):
    digest = sender.canonical_digest(no_setup_payload)
    subject = sender.subject_for(no_setup_payload)
    body = sender.render_html(no_setup_payload, digest)
    text = sender.render_text(no_setup_payload, digest)

    assert subject.endswith("NO QUALIFIED SETUP")
    assert "NO QUALIFIED SETUP" in body
    assert "Nothing has been forced" in body
    assert "NO QUALIFIED SETUP" in text


def test_delivery_is_exact_once_by_valid_for_and_digest(
    ready_payload, tmp_path, monkeypatch
):
    receipt = tmp_path / "receipt.json"
    calls = []
    monkeypatch.setattr(sender, "_smtp_send", lambda *a, **k: calls.append((a, k)))

    assert sender.deliver(ready_payload, receipt_path=receipt, now=NOW) == "sent"
    assert sender.deliver(ready_payload, receipt_path=receipt, now=NOW) == "skipped"
    assert len(calls) == 1

    saved = json.loads(receipt.read_text(encoding="utf-8"))
    assert saved["schema_version"] == sender.RECEIPT_SCHEMA
    assert len(saved["deliveries"]) == 1
    assert saved["deliveries"][0]["status"] == "sent"
    assert saved["deliveries"][0]["valid_for"] == ready_payload["valid_for"]


def test_force_send_bypasses_receipt_only(ready_payload, tmp_path, monkeypatch):
    receipt = tmp_path / "receipt.json"
    calls = []
    monkeypatch.setattr(sender, "_smtp_send", lambda *a, **k: calls.append(1))

    sender.deliver(ready_payload, receipt_path=receipt, now=NOW)
    sender.deliver(
        ready_payload, receipt_path=receipt, force_send=True, now=NOW
    )
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    assert len(calls) == 2
    assert len(saved["deliveries"]) == 2
    assert saved["deliveries"][1]["forced"] is True

    provisional = copy.deepcopy(ready_payload)
    provisional["phase"] = "PROVISIONAL"
    with pytest.raises(sender.FocusPayloadError, match="requires phase FINAL"):
        sender.deliver(
            provisional, receipt_path=receipt, force_send=True, now=NOW
        )
    assert len(calls) == 2


def test_stale_payload_never_becomes_no_setup(ready_payload, tmp_path, monkeypatch):
    ready_payload["expires_at"] = "2026-08-26T11:59:00Z"
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(
        sender,
        "_smtp_send",
        lambda *a, **k: pytest.fail("stale input must never touch SMTP"),
    )

    with pytest.raises(sender.FocusPayloadError, match="expired"):
        sender.deliver(
            ready_payload, receipt_path=receipt, force_send=True, now=NOW
        )
    assert not receipt.exists()


def test_smtp_failure_is_loud_receipted_and_retryable(
    ready_payload, tmp_path, monkeypatch
):
    receipt = tmp_path / "receipt.json"

    def fail(*args, **kwargs):
        raise sender.EmailDeliveryError("SMTP delivery failed: TimeoutError")

    monkeypatch.setattr(sender, "_smtp_send", fail)
    with pytest.raises(sender.EmailDeliveryError):
        sender.deliver(ready_payload, receipt_path=receipt, now=NOW)
    failed = json.loads(receipt.read_text(encoding="utf-8"))["deliveries"]
    assert len(failed) == 1 and failed[0]["status"] == "failed"

    calls = []
    monkeypatch.setattr(sender, "_smtp_send", lambda *a, **k: calls.append(1))
    assert sender.deliver(ready_payload, receipt_path=receipt, now=NOW) == "sent"
    assert calls == [1]


def test_ambiguous_prior_attempt_blocks_automatic_resend(
    ready_payload, tmp_path, monkeypatch
):
    receipt = tmp_path / "receipt.json"
    digest = sender.canonical_digest(ready_payload)
    receipt.write_text(
        json.dumps(
            {
                "schema_version": sender.RECEIPT_SCHEMA,
                "deliveries": [
                    {
                        "valid_for": ready_payload["valid_for"],
                        "digest": digest,
                        "status": "sending",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sender,
        "_smtp_send",
        lambda *a, **k: pytest.fail("ambiguous attempt must not auto-resend"),
    )

    with pytest.raises(sender.AmbiguousDeliveryError):
        sender.deliver(ready_payload, receipt_path=receipt, now=NOW)


def test_dry_run_validates_no_setup_without_smtp_or_receipt(
    no_setup_payload, tmp_path, monkeypatch, capsys
):
    input_path = tmp_path / "focus.json"
    receipt = tmp_path / "receipt.json"
    input_path.write_text(json.dumps(no_setup_payload), encoding="utf-8")
    monkeypatch.setattr(
        sender,
        "_smtp_send",
        lambda *a, **k: pytest.fail("dry-run must not touch SMTP"),
    )

    rc = sender.main(
        [
            "--input",
            str(input_path),
            "--receipt",
            str(receipt),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert "NO QUALIFIED SETUP" in capsys.readouterr().out
    assert not receipt.exists()


def test_missing_credentials_fail_before_receipt_claim(
    ready_payload, tmp_path, monkeypatch
):
    monkeypatch.delenv("EMAIL_USER")
    receipt = tmp_path / "receipt.json"
    with pytest.raises(sender.EmailConfigurationError):
        sender.deliver(ready_payload, receipt_path=receipt, now=NOW)
    assert not receipt.exists()


def test_corrupt_receipt_fails_closed(ready_payload, tmp_path, monkeypatch):
    receipt = tmp_path / "receipt.json"
    receipt.write_text("not-json", encoding="utf-8")
    monkeypatch.setattr(
        sender,
        "_smtp_send",
        lambda *a, **k: pytest.fail("corrupt receipt must not touch SMTP"),
    )
    with pytest.raises(sender.ReceiptError):
        sender.deliver(ready_payload, receipt_path=receipt, now=NOW)
