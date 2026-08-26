from __future__ import annotations

import hashlib
import json
import smtplib
from pathlib import Path
from typing import ClassVar

import pytest

import episodic_pivot.email_delivery as delivery
from episodic_pivot.email_delivery import (
    EmailDeliveryError,
    EmailPayload,
    EmailSettings,
    deliver_email,
    morning_payload,
    night_payload,
    resolve_email_settings,
)
from episodic_pivot.email_delivery import (
    test_payload as build_test_payload,
)
from episodic_pivot.schema import PremarketSnapshot


def _snapshot(**overrides: object) -> PremarketSnapshot:
    values: dict[str, object] = {
        "symbol": "TEST",
        "company_name": "Test <Systems>",
        "observed_at": "2026-08-24T21:10:00Z",
        "previous_close": 10.0,
        "last": 11.0,
        "bid": 0.0,
        "ask": 0.0,
        "premarket_volume": 500_000,
        "premarket_open": 11.0,
        "premarket_high": 11.0,
        "premarket_low": 11.0,
        "premarket_vwap": 0.0,
        "prior_two_day_low": 0.0,
        "atr_14": 0.0,
        "avg_volume_20": 0.0,
        "addv_63": 0.0,
        "market_data_status": "BROWSER_EXPORT",
        "tradeable": False,
        "source": "TRADINGVIEW_BROWSER_EXPORT",
        "session": "after_hours",
        "provider": "TRADINGVIEW",
        "saved_screen_id": "ep-after-hours-v1",
        "target_session_date": "2026-08-25",
        "reported_result_count": 1,
        "extracted_row_count": 1,
        "reported_change_pct": 10.0,
        "reported_move_dollars": 1.0,
    }
    values.update(overrides)
    return PremarketSnapshot(**values)  # type: ignore[arg-type]


def _write_night_import(tmp_path: Path) -> Path:
    path = tmp_path / "night.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider": "TRADINGVIEW",
                "saved_screen_id": "ep-after-hours-v1",
                "session": "after_hours",
                "captured_at": "2026-08-24T21:10:00Z",
                "target_session_date": "2026-08-25",
                "source_file": "C:/artifacts/TradingView.csv",
                "source_file_sha256": "a" * 64,
                "reported_result_count": 1,
                "extracted_row_count": 1,
                "result_count_verified": True,
                "snapshots": [_snapshot().to_dict()],
            }
        ),
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_morning_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "EP-RUN-TEST"
    run_dir.mkdir()
    files = {
        "candidates.json": json.dumps(
            [{"snapshot": _snapshot(session="premarket").to_dict()}]
        ),
        "report.html": (
            "<!doctype html><html><body><h1>Research only</h1>"
            "<p>Complete EP report; broker route NONE</p></body></html>"
        ),
        "report.md": "# Research only\n\nBroker route NONE\n",
        "decisions.json": "[{}]",
        "research_sizing_preview.json": "[{}]",
        "research_sizing_preview.csv": "symbol,preview_only\nTEST,true\n",
    }
    for name, content in files.items():
        (run_dir / name).write_text(content, encoding="utf-8")
    artifacts = {
        name: {
            "sha256": _sha256(run_dir / name),
            "size_bytes": (run_dir / name).stat().st_size,
        }
        for name in files
    }
    manifest = {
        "schema_version": 2,
        "run_id": run_dir.name,
        "counts": {
            "candidates": 1,
            "decisions": 1,
            "research_sizing_previews": 1,
        },
        "safety": {
            "research_only": True,
            "live_actions_enabled": False,
            "broker_route": "NONE",
            "order_submission_allowed": False,
            "order_staging_performed": False,
            "production_deployed": False,
        },
        "artifacts": artifacts,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def test_email_settings_use_explicit_env_then_recipient_fallback(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "EMAIL_USER=fallback@example.com\n"
        "EMAIL_PASS=fallback-secret\n"
        "RECIPIENTS=first@example.com, second@example.com\n",
        encoding="utf-8",
    )
    settings = resolve_email_settings(
        env_file=env_file,
        environ={
            "EMAIL_USER": "runtime@example.com",
            "EMAIL_PASS": "runtime-secret",
            "EP_RECIPIENTS": "ep@example.com,ep@example.com",
        },
    )
    assert settings == EmailSettings(
        sender="runtime@example.com",
        password="runtime-secret",
        recipients=("ep@example.com",),
    )

    fallback = resolve_email_settings(env_file=env_file, environ={})
    assert fallback.recipients == ("first@example.com", "second@example.com")


@pytest.mark.parametrize(
    "environ",
    [
        {},
        {"EMAIL_USER": "sender@example.com"},
        {"EMAIL_USER": "bad-address", "EMAIL_PASS": "secret"},
        {
            "EMAIL_USER": "sender@example.com",
            "EMAIL_PASS": "secret",
            "EP_RECIPIENTS": "ok@example.com\nBcc:evil@example.com",
        },
    ],
)
def test_email_settings_fail_closed_on_missing_or_unsafe_values(
    environ: dict[str, str],
) -> None:
    with pytest.raises(EmailDeliveryError):
        resolve_email_settings(environ=environ)


def test_night_payload_revalidates_count_and_escapes_html(tmp_path: Path) -> None:
    source = _write_night_import(tmp_path)
    payload = night_payload(source)

    assert payload.kind == "night"
    assert "2026-08-25" in payload.subject
    assert "1 nominees" in payload.subject
    assert payload.attachments == (source.resolve(),)
    assert "Test &lt;Systems&gt;" in payload.html_body
    assert "Test <Systems>" not in payload.html_body
    assert payload.metadata["screen_rows"] == 1

    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["reported_result_count"] = 2
    source.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(EmailDeliveryError, match="count mismatch"):
        night_payload(source)


def test_morning_payload_revalidates_manifest_and_attaches_audit_files(
    tmp_path: Path,
) -> None:
    run_dir = _write_morning_run(tmp_path)
    payload = morning_payload(run_dir)

    assert payload.kind == "morning"
    assert "2026-08-25" in payload.subject
    assert payload.metadata["research_sizing_previews"] == 1
    assert [path.name for path in payload.attachments] == [
        "report.html",
        "report.md",
        "research_sizing_preview.csv",
        "manifest.json",
    ]

    (run_dir / "report.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(EmailDeliveryError, match="digest mismatch"):
        morning_payload(run_dir)


def test_morning_payload_rejects_manifest_count_drift(tmp_path: Path) -> None:
    run_dir = _write_morning_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["decisions"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(EmailDeliveryError, match="count mismatch: decisions"):
        morning_payload(run_dir)


class _FakeSMTP:
    instances: ClassVar[list[_FakeSMTP]] = []

    def __init__(self, host: str, port: int, timeout: int) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.messages = []
        self.login_args: tuple[str, str] | None = None
        self.__class__.instances.append(self)

    def __enter__(self) -> _FakeSMTP:  # noqa: PYI034
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def ehlo(self) -> None:
        return None

    def starttls(self, *, context: object) -> None:
        assert context is not None

    def login(self, sender: str, password: str) -> None:
        self.login_args = (sender, password)

    def send_message(self, message: object) -> None:
        self.messages.append(message)


def test_successful_delivery_writes_non_sensitive_receipt_and_deduplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakeSMTP.instances.clear()
    monkeypatch.setattr(delivery.smtplib, "SMTP", _FakeSMTP)
    payload = build_test_payload(output_root=tmp_path)
    settings = EmailSettings(
        sender="sender@example.com",
        password="top-secret",
        recipients=("recipient@example.com",),
    )

    assert deliver_email(payload, settings, send=True) == "SENT"
    assert len(_FakeSMTP.instances) == 1
    assert _FakeSMTP.instances[0].login_args == (
        "sender@example.com",
        "top-secret",
    )
    assert len(_FakeSMTP.instances[0].messages) == 1
    receipt_text = payload.receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    assert receipt["status"] == "SENT"
    assert receipt["recipient_count"] == 1
    assert "top-secret" not in receipt_text
    assert "sender@example.com" not in receipt_text
    assert "recipient@example.com" not in receipt_text

    assert deliver_email(payload, settings, send=True) == "ALREADY_SENT"
    assert len(_FakeSMTP.instances) == 1


def test_conflicting_receipt_requires_explicit_resend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakeSMTP.instances.clear()
    monkeypatch.setattr(delivery.smtplib, "SMTP", _FakeSMTP)
    payload = build_test_payload(output_root=tmp_path)
    first = EmailSettings("sender@example.com", "secret", ("one@example.com",))
    changed = EmailSettings("sender@example.com", "secret", ("two@example.com",))
    assert deliver_email(payload, first, send=True) == "SENT"

    with pytest.raises(EmailDeliveryError, match="different source or recipient"):
        deliver_email(payload, changed, send=True)
    assert len(_FakeSMTP.instances) == 1

    assert deliver_email(payload, changed, send=True, resend=True) == "SENT"
    assert len(_FakeSMTP.instances) == 2


def test_smtp_failure_writes_no_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingSMTP(_FakeSMTP):
        def login(self, sender: str, password: str) -> None:
            raise smtplib.SMTPAuthenticationError(535, b"denied")

    monkeypatch.setattr(delivery.smtplib, "SMTP", FailingSMTP)
    payload = build_test_payload(output_root=tmp_path)
    settings = EmailSettings("sender@example.com", "secret", ("to@example.com",))

    with pytest.raises(EmailDeliveryError, match="no delivery receipt"):
        deliver_email(payload, settings, send=True)
    assert not payload.receipt_path.exists()


def test_dry_run_never_contacts_smtp_or_writes_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_smtp(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("SMTP must not be contacted by a dry run")

    monkeypatch.setattr(delivery.smtplib, "SMTP", unexpected_smtp)
    payload: EmailPayload = build_test_payload(output_root=tmp_path)
    settings = EmailSettings("sender@example.com", "secret", ("to@example.com",))

    assert deliver_email(payload, settings, send=False) == "DRY_RUN"
    assert not payload.receipt_path.exists()
