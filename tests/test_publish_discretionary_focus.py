import json
import datetime as dt
from pathlib import Path

import pytest

from scripts import publish_discretionary_focus as publisher

NOW = dt.datetime(2026, 8, 26, 13, 0, tzinfo=dt.timezone.utc)


def _payload() -> dict:
    return {
        "schema_version": "discretionary-focus.v1",
        "research_only": True,
        "quick_review_created": False,
        "live_actions_enabled": False,
        "order_staging_enabled": False,
        "phase": "FINAL",
        "status": "NO_QUALIFIED_SETUP",
        "as_of": "2026-08-25",
        "valid_for": "2026-08-26",
        "generated_at": "2026-08-26T08:35:00-04:00",
        "expires_at": "2026-08-26T16:15:00-04:00",
        "focus": [],
        "screen_summary": {"selected_count": 0},
        "provenance": {"price_basis": "ADJUSTED_RECOMPUTED_DAILY"},
    }


def test_publish_orders_history_before_current(tmp_path, monkeypatch) -> None:
    path = tmp_path / "focus.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    uploaded: list[str] = []

    def upload(local: str, key: str) -> bool:
        uploaded.append(key)
        return True

    monkeypatch.setattr(publisher.cache_io, "upload_from_local", upload)
    monkeypatch.setattr(
        publisher.cache_io,
        "head",
        lambda key: {"ContentLength": path.stat().st_size},
    )
    monkeypatch.setattr(
        publisher,
        "validate_payload",
        lambda payload, **kwargs: payload,
    )
    monkeypatch.setattr(
        publisher, "delivery_window_gate", lambda now: (True, dt.date(2026, 8, 26))
    )
    monkeypatch.setattr(publisher, "canonical_digest", lambda payload: "a" * 64)

    archive, current = publisher.publish_payload(path, now=NOW)

    assert archive == (
        "discretionary_focus/history/2026-08-26/final-" + "a" * 64 + ".json"
    )
    assert current == publisher.CURRENT_KEY
    assert uploaded == [archive, publisher.CURRENT_KEY]


def test_current_is_not_replaced_when_archive_upload_fails(tmp_path, monkeypatch) -> None:
    path = tmp_path / "focus.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    uploaded: list[str] = []

    def upload(local: str, key: str) -> bool:
        uploaded.append(key)
        return False

    monkeypatch.setattr(publisher.cache_io, "upload_from_local", upload)
    monkeypatch.setattr(
        publisher,
        "validate_payload",
        lambda payload, **kwargs: payload,
    )
    monkeypatch.setattr(
        publisher, "delivery_window_gate", lambda now: (True, dt.date(2026, 8, 26))
    )
    monkeypatch.setattr(publisher, "canonical_digest", lambda payload: "b" * 64)

    with pytest.raises(RuntimeError, match="upload failed"):
        publisher.publish_payload(path, now=NOW)
    assert len(uploaded) == 1
    assert publisher.CURRENT_KEY not in uploaded


def test_receipt_requires_delivery_identity(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"valid_for": "2026-08-26"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        publisher.publish_receipt(path)


def test_successful_sender_receipt_is_publishable(tmp_path, monkeypatch) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": publisher.RECEIPT_SCHEMA,
                "deliveries": [
                    {
                        "attempt_id": "attempt-1",
                        "valid_for": "2026-08-26",
                        "digest": "a" * 64,
                        "status": "sent",
                        "started_at": "2026-08-26T12:35:00Z",
                        "sent_at": "2026-08-26T12:36:00Z",
                        "recipients": ["one@example.com"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    uploaded = []
    monkeypatch.setattr(
        publisher,
        "_verified_upload",
        lambda local, key: uploaded.append((local, key)),
    )

    assert publisher.publish_receipt(path) == publisher.RECEIPT_KEY
    assert uploaded == [(path, publisher.RECEIPT_KEY)]


def test_non_sent_latest_receipt_is_never_persisted(tmp_path, monkeypatch) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": publisher.RECEIPT_SCHEMA,
                "deliveries": [
                    {
                        "attempt_id": "attempt-1",
                        "valid_for": "2026-08-26",
                        "digest": "a" * 64,
                        "status": "sending",
                        "started_at": "2026-08-26T12:35:00Z",
                        "recipients": ["one@example.com"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        publisher,
        "_verified_upload",
        lambda *args: pytest.fail("ambiguous receipt must not be uploaded"),
    )

    with pytest.raises(ValueError, match="missing|status sent"):
        publisher.publish_receipt(path)
