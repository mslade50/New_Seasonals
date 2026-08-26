"""Render and deliver the validated pre-market Discretionary Focus email.

The sender is deliberately downstream of ``discretionary_focus.contracts``.
It never turns a malformed, stale, provisional, or live payload into a benign
"no setup" message.  A normal run accepts only a current ``FINAL`` payload.

Delivery is at-most-once for a ``valid_for`` market session; the canonical
digest remains recorded for audit.  A receipt is claimed before SMTP is
contacted and marked sent only after Gmail accepts the message.  In automation,
``--persist-receipt-r2`` checkpoints that claim to R2 before SMTP.  A process
that dies in the ambiguous interval therefore leaves a ``sending`` claim and
later automatic runs refuse to resend it.  ``--force-send`` is the explicit
operator escape hatch, but it never bypasses validation, currentness, or the
FINAL-phase requirement.

Usage::

    python scripts/send_discretionary_focus_email.py \
      --input data/discretionary_focus_latest.json \
      --receipt data/discretionary_focus_email_receipt.json

Environment:
    EMAIL_USER / EMAIL_PASS
    DISCRETIONARY_FOCUS_RECIPIENTS (comma separated; defaults to McKinley)
    DISCRETIONARY_FOCUS_SITE_URL (optional private-site link override)
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import smtplib
import sys
import time
import uuid
from contextlib import contextmanager
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discretionary_focus.contracts import (  # noqa: E402
    FocusPayloadError,
    canonical_digest,
    validate_payload,
)
from scripts.check_discretionary_focus_session import delivery_window_gate  # noqa: E402


DEFAULT_RECIPIENTS = "mckinleyslade@gmail.com"
DEFAULT_SITE_URL = "https://seasonals-mslade.pages.dev/focus.html"
DEFAULT_RECEIPT = ROOT / "data" / "discretionary_focus_email_receipt.json"
RECEIPT_SCHEMA = "discretionary-focus-email-receipt.v1"
RECEIPT_R2_KEY = "discretionary_focus/email_receipt.json"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


class FocusEmailError(RuntimeError):
    """Base class for deterministic sender failures."""


class ReceiptError(FocusEmailError):
    """The delivery receipt is unreadable or cannot be updated safely."""


class AmbiguousDeliveryError(FocusEmailError):
    """A prior process may have sent the same message before it died."""


class EmailConfigurationError(FocusEmailError):
    """SMTP credentials or recipients are missing."""


class EmailDeliveryError(FocusEmailError):
    """SMTP rejected or could not deliver the message."""


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso_utc(value: dt.datetime | None = None) -> str:
    value = value or _utc_now()
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _plain_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        populated = [
            key for key, item in value.items() if item not in (None, "", [], {})
        ]
        if populated == ["condition"]:
            return _plain_value(value["condition"])
        preferred = [
            "condition",
            "technical",
            "thesis_kill",
            "live_state",
            "price_basis",
        ]
        pieces: list[str] = []
        seen: set[str] = set()
        for key in preferred + sorted(value):
            if key in seen or key not in value:
                continue
            seen.add(key)
            item = value[key]
            if item in (None, "", [], {}):
                continue
            if isinstance(item, (dict, list)):
                rendered = json.dumps(item, sort_keys=True, separators=(",", ":"))
            else:
                rendered = str(item)
            label = key.replace("_", " ")
            pieces.append(f"{label}: {rendered}")
        return " | ".join(pieces)
    if isinstance(value, list):
        return "; ".join(_plain_value(item) for item in value if item not in (None, ""))
    return str(value if value is not None else "").strip()


def _source_links(sources: list[dict[str, Any]]) -> str:
    links: list[str] = []
    for source in sources[:4]:
        label = _esc(source.get("label") or source.get("source_id") or "Source")
        url = str(source.get("url") or "").strip()
        as_of = source.get("as_of")
        suffix = f" ({_esc(as_of)})" if as_of else ""
        if url.startswith(("https://", "http://")):
            links.append(
                f'<a href="{_esc(url)}" style="color:#2457a6">{label}</a>{suffix}'
            )
        else:
            links.append(f"{label}{suffix}")
    return " &nbsp;·&nbsp; ".join(links)


def _http_url(value: Any) -> str:
    text = str(value or "").strip()
    return text if text.startswith(("https://", "http://")) else ""


def _summary_html(summary: dict[str, Any]) -> str:
    labels = (
        ("input_count", "Input"),
        ("technical_pass_count", "Technical pass"),
        ("research_pass_count", "Research pass"),
        ("selected_count", "Selected"),
    )
    bits = [f"<b>{_esc(label)}:</b> {_esc(summary[key])}" for key, label in labels if key in summary]
    return " &nbsp;|&nbsp; ".join(bits)


def subject_for(payload: dict[str, Any]) -> str:
    day = payload["valid_for"]
    if payload["status"] == "NO_QUALIFIED_SETUP":
        return f"Discretionary Focus - {day} - NO QUALIFIED SETUP"
    count = len(payload["focus"])
    noun = "name" if count == 1 else "names"
    return f"Discretionary Focus - {day} - {count} {noun}"


def render_html(payload: dict[str, Any], digest: str | None = None) -> str:
    """Render a concise, escaped research-attention briefing."""
    day = payload["valid_for"]
    site_url = os.environ.get("DISCRETIONARY_FOCUS_SITE_URL", DEFAULT_SITE_URL)
    site_link = ""
    if site_url.startswith(("https://", "http://")):
        site_link = (
            f'<a href="{_esc(site_url)}" style="display:inline-block;margin-top:10px;'
            'color:#2457a6;font-weight:600">Open the private site</a>'
        )
    live_url = _http_url((payload.get("provenance") or {}).get("tradingview_live_url"))
    live_link = (
        f'<a href="{_esc(live_url)}" style="display:inline-block;margin:10px 0 0 14px;'
        'color:#2457a6;font-weight:600">Open the Live RVOL screen</a>'
        if live_url
        else ""
    )

    summary = _summary_html(payload.get("screen_summary") or {})
    summary_block = (
        f'<div style="font-size:12px;color:#5f6670;margin:8px 0 14px">{summary}</div>'
        if summary
        else ""
    )

    if payload["status"] == "NO_QUALIFIED_SETUP":
        no_setup_reason = _esc(payload["no_setup_reason"])
        body = f"""
        <div style="border:1px solid #d5d9df;border-left:5px solid #6b7280;
                    border-radius:7px;padding:18px;margin:14px 0;background:#fafafa">
          <div style="font-size:20px;font-weight:750;color:#23272f">NO QUALIFIED SETUP</div>
          <p style="margin:8px 0 0;color:#4b5563;line-height:1.45">
            No candidate cleared every required technical and research gate for
            {_esc(day)}. Nothing has been forced into the focus list.
          </p>
          <p style="margin:8px 0 0;color:#4b5563;line-height:1.45">
            {no_setup_reason}
          </p>
        </div>"""
    else:
        cards: list[str] = []
        for item in payload["focus"]:
            event = _esc(item.get("event_date"))
            earnings = _esc(item.get("earnings_td"))
            sources = _source_links(item.get("sources") or [])
            source_block = (
                f'<div style="font-size:11px;color:#6b7280;margin-top:12px">{sources}</div>'
                if sources
                else ""
            )
            cards.append(
                f"""
                <div style="border:1px solid #d8dde6;border-radius:8px;padding:16px;
                            margin:14px 0;background:#fff">
                  <div style="font-size:12px;color:#68707c;text-transform:uppercase;
                              letter-spacing:.05em">Focus {int(item['rank'])}</div>
                  <div style="font-size:22px;font-weight:750;color:#111827;margin:2px 0 10px">
                    {_esc(item['ticker'])}
                    <span style="font-size:14px;font-weight:500;color:#6b7280">
                      {_esc(item['company_name'])}
                    </span>
                  </div>
                  <table role="presentation" style="border-collapse:collapse;width:100%;
                         font-size:13px;line-height:1.45;color:#2f3742">
                    {_field_row('Why now', item['why_now'])}
                    {_field_row('Setup', item['setup'])}
                    {_field_row('Confirmation', item['trigger'])}
                    {_field_row('Invalidation', item['invalidation'])}
                    {_field_row('Catalyst', item['catalyst'])}
                    {_field_row('Already reflected', item['priced_in'])}
                    {_field_row('Next proof', item['next_proof'])}
                    {_field_row('Event', f'{event} | earnings {earnings} trading days away')}
                  </table>
                  {source_block}
                </div>"""
            )
        body = "".join(cards)

    digest_note = f" | digest {_esc(digest[:12])}" if digest else ""
    return f"""<!doctype html>
<html><body style="margin:0;background:#f3f5f7;color:#1f2937">
<div style="max-width:680px;margin:0 auto;padding:22px;font-family:Segoe UI,Arial,sans-serif">
  <div style="font-size:12px;color:#68707c;text-transform:uppercase;letter-spacing:.07em">
    Research attention only
  </div>
  <h1 style="font-size:25px;margin:4px 0 6px;color:#111827">Discretionary Focus</h1>
  <div style="font-size:14px;color:#4b5563">Session {_esc(day)} · FINAL</div>
  {summary_block}
  {body}
  {site_link}{live_link}
  <div style="font-size:11px;color:#7b8492;margin-top:18px;padding-top:12px;
              border-top:1px solid #d8dde6">
    This is a research-priority briefing, not an investment recommendation.
    Generated {_esc(payload['generated_at'])}{digest_note}.
  </div>
</div></body></html>"""


def _field_row(label: str, value: Any) -> str:
    rendered = _esc(_plain_value(value))
    return (
        '<tr>'
        f'<td style="vertical-align:top;width:118px;padding:4px 10px 4px 0;'
        f'color:#667085;font-weight:650">{_esc(label)}</td>'
        f'<td style="vertical-align:top;padding:4px 0">{rendered}</td>'
        '</tr>'
    )


def render_text(payload: dict[str, Any], digest: str | None = None) -> str:
    lines = [
        "DISCRETIONARY FOCUS",
        f"Session {payload['valid_for']} | FINAL | research attention only",
        "",
    ]
    if payload["status"] == "NO_QUALIFIED_SETUP":
        lines.extend(
            [
                "NO QUALIFIED SETUP",
                "No candidate cleared every required technical and research gate. ",
                "Nothing has been forced into the focus list.",
                payload["no_setup_reason"],
            ]
        )
    else:
        for item in payload["focus"]:
            lines.extend(
                [
                    f"FOCUS {item['rank']}: {item['ticker']} - {item['company_name']}",
                    f"Why now: {_plain_value(item['why_now'])}",
                    f"Setup: {_plain_value(item['setup'])}",
                    f"Confirmation: {_plain_value(item['trigger'])}",
                    f"Invalidation: {_plain_value(item['invalidation'])}",
                    f"Catalyst: {_plain_value(item['catalyst'])}",
                    f"Already reflected: {_plain_value(item['priced_in'])}",
                    f"Next proof: {_plain_value(item['next_proof'])}",
                    f"Event: {item['event_date']} | earnings {item['earnings_td']} trading days away",
                    "",
                ]
            )
    site_url = os.environ.get("DISCRETIONARY_FOCUS_SITE_URL", DEFAULT_SITE_URL)
    live_url = _http_url((payload.get("provenance") or {}).get("tradingview_live_url"))
    lines.extend(
        [
            f"Private site: {site_url}",
            *([f"TradingView Live RVOL screen: {live_url}"] if live_url else []),
            "This is a research-priority briefing, not an investment recommendation.",
        ]
    )
    if digest:
        lines.append(f"Digest: {digest[:12]}")
    return "\n".join(lines)


def _email_configuration() -> tuple[str, str, list[str]]:
    sender = os.environ.get("EMAIL_USER", "").strip()
    password = os.environ.get("EMAIL_PASS", "")
    recipient_text = (
        os.environ.get("DISCRETIONARY_FOCUS_RECIPIENTS", "").strip()
        or DEFAULT_RECIPIENTS
    )
    recipients = [
        item.strip()
        for item in recipient_text.split(",")
        if item.strip()
    ]
    if not sender or not password:
        raise EmailConfigurationError("EMAIL_USER/EMAIL_PASS are required")
    if not recipients:
        raise EmailConfigurationError(
            "DISCRETIONARY_FOCUS_RECIPIENTS resolved to an empty list"
        )
    return sender, password, recipients


def _smtp_send(
    subject: str,
    html_body: str,
    text_body: str,
    *,
    sender: str,
    password: str,
    recipients: list[str],
) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    delivery_started = False
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(sender, password)
            delivery_started = True
            refused = server.sendmail(sender, recipients, msg.as_string())
            if refused:
                raise AmbiguousDeliveryError(
                    "SMTP accepted at least one recipient but refused another; "
                    "inspect delivery before any resend"
                )
    except AmbiguousDeliveryError:
        raise
    except (
        smtplib.SMTPRecipientsRefused,
        smtplib.SMTPSenderRefused,
        smtplib.SMTPDataError,
        smtplib.SMTPHeloError,
        smtplib.SMTPNotSupportedError,
    ) as exc:
        # These are explicit protocol rejections: Gmail did not accept DATA.
        raise EmailDeliveryError(
            f"SMTP delivery rejected: {type(exc).__name__}"
        ) from exc
    except (smtplib.SMTPException, OSError) as exc:
        if delivery_started:
            # A disconnect/timeout after sendmail starts can occur after the
            # server accepted DATA. Keep the durable claim at ``sending``.
            raise AmbiguousDeliveryError(
                f"SMTP outcome is ambiguous after delivery started: {type(exc).__name__}"
            ) from exc
        raise EmailDeliveryError(f"SMTP delivery failed: {type(exc).__name__}") from exc


def _empty_receipt() -> dict[str, Any]:
    return {"schema_version": RECEIPT_SCHEMA, "deliveries": []}


def _load_receipt(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_receipt()
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"cannot read receipt {path}: {type(exc).__name__}") from exc
    if not isinstance(receipt, dict) or receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ReceiptError(f"unsupported receipt schema in {path}")
    if not isinstance(receipt.get("deliveries"), list):
        raise ReceiptError(f"receipt deliveries must be a list in {path}")
    return receipt


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except OSError as exc:
        raise ReceiptError(f"cannot update receipt {path}: {type(exc).__name__}") from exc


def _persist_receipt_r2(path: Path) -> None:
    """Checkpoint the local receipt to R2 and verify the stored byte count."""
    import cache_io

    if not cache_io.is_configured():
        raise ReceiptError("R2 credentials are required to checkpoint the email receipt")
    if not cache_io.upload_from_local(str(path), RECEIPT_R2_KEY):
        raise ReceiptError("R2 email-receipt upload failed")
    metadata = cache_io.head(RECEIPT_R2_KEY)
    actual_size = int((metadata or {}).get("ContentLength") or -1)
    if actual_size != path.stat().st_size:
        raise ReceiptError("R2 email-receipt verification failed")


def _checkpoint_receipt(
    path: Path,
    receipt: dict[str, Any],
    *,
    persist_r2: bool,
) -> None:
    _write_receipt(path, receipt)
    if persist_r2:
        _persist_receipt_r2(path)


@contextmanager
def _receipt_lock(path: Path, timeout_seconds: float = 10.0) -> Iterator[None]:
    """Cross-platform advisory lock; the tiny lock file intentionally persists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    handle = lock_path.open("a+b")
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    acquired = False
    deadline = time.monotonic() + timeout_seconds
    try:
        if os.name == "nt":
            import msvcrt

            while time.monotonic() < deadline:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except OSError:
                    time.sleep(0.1)
        else:
            import fcntl

            while time.monotonic() < deadline:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    time.sleep(0.1)
        if not acquired:
            raise ReceiptError(f"timed out waiting for receipt lock {lock_path}")
        yield
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _session_deliveries(
    receipt: dict[str, Any], valid_for: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in receipt["deliveries"]
        if row.get("valid_for") == valid_for
    ]


def deliver(
    payload: dict[str, Any],
    *,
    receipt_path: Path,
    force_send: bool = False,
    persist_r2: bool = False,
    now: dt.datetime | None = None,
) -> str:
    """Validate, render, and deliver once. Return ``sent`` or ``skipped``."""
    normalized = validate_payload(payload, now=now, require_current=True)
    if normalized.get("phase") != "FINAL":
        raise FocusPayloadError("email delivery requires phase FINAL")
    delivery_now = now or _utc_now()
    inside_window, market_date = delivery_window_gate(delivery_now)
    if not inside_window or market_date.isoformat() != normalized["valid_for"]:
        raise FocusPayloadError(
            "email delivery is allowed only 08:25-09:20 New York on valid_for"
        )
    digest = canonical_digest(normalized)
    valid_for = normalized["valid_for"]
    subject = subject_for(normalized)
    html_body = render_html(normalized, digest)
    text_body = render_text(normalized, digest)
    sender, password, recipients = _email_configuration()

    with _receipt_lock(receipt_path):
        receipt = _load_receipt(receipt_path)
        session_rows = _session_deliveries(receipt, valid_for)
        if not force_send and any(row.get("status") == "sent" for row in session_rows):
            prior = next(row for row in reversed(session_rows) if row.get("status") == "sent")
            prior_digest = str(prior.get("digest") or "unknown")[:12]
            print(
                f"Focus email already sent for {valid_for} digest {prior_digest}; "
                f"current digest {digest[:12]} is not resent automatically."
            )
            return "skipped"
        if not force_send and any(row.get("status") == "sending" for row in session_rows):
            raise AmbiguousDeliveryError(
                f"prior delivery for {valid_for} is still "
                "marked sending; inspect before using --force-send"
            )

        attempt_id = uuid.uuid4().hex
        record: dict[str, Any] = {
            "attempt_id": attempt_id,
            "valid_for": valid_for,
            "digest": digest,
            "status": "sending",
            "started_at": _iso_utc(now),
            "forced": bool(force_send),
            "subject": subject,
            "recipients": recipients,
        }
        receipt["deliveries"].append(record)
        _checkpoint_receipt(receipt_path, receipt, persist_r2=persist_r2)

        try:
            _smtp_send(
                subject,
                html_body,
                text_body,
                sender=sender,
                password=password,
                recipients=recipients,
            )
        except AmbiguousDeliveryError:
            # Preserve the already-checkpointed ``sending`` state. A new
            # runner will refuse to resend until an operator inspects inboxes.
            raise
        except EmailDeliveryError as exc:
            record["status"] = "failed"
            record["completed_at"] = _iso_utc()
            record["error_type"] = type(exc.__cause__).__name__ if exc.__cause__ else type(exc).__name__
            _checkpoint_receipt(receipt_path, receipt, persist_r2=persist_r2)
            raise

        record["status"] = "sent"
        record["sent_at"] = _iso_utc()
        try:
            _checkpoint_receipt(receipt_path, receipt, persist_r2=persist_r2)
        except ReceiptError as exc:
            raise AmbiguousDeliveryError(
                "SMTP accepted the message but the sent receipt could not be "
                "checkpointed; inspect inboxes before any resend"
            ) from exc

    print(
        f"Discretionary Focus email sent to {', '.join(recipients)} "
        f"for {valid_for} digest {digest[:12]}."
    )
    return "sent"


def _load_input(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FocusEmailError(f"input does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise FocusEmailError(f"cannot read input {path}: {type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise FocusEmailError("focus input must be a JSON object")
    return payload


def main(
    argv: list[str] | None = None,
    *,
    now: dt.datetime | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Send one validated FINAL Discretionary Focus email"
    )
    parser.add_argument("--input", required=True, help="validated focus JSON payload")
    parser.add_argument(
        "--receipt",
        default=str(DEFAULT_RECEIPT),
        help="persistent exact-once delivery receipt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and render only; do not email or touch the receipt",
    )
    parser.add_argument(
        "--force-send",
        action="store_true",
        help="explicitly resend an already-receipted payload; validation still applies",
    )
    parser.add_argument(
        "--persist-receipt-r2",
        action="store_true",
        help="checkpoint sending/sent receipt states to R2 for runner-safe at-most-once delivery",
    )
    args = parser.parse_args(argv)

    try:
        payload = _load_input(Path(args.input))
        if args.dry_run:
            normalized = validate_payload(
                payload,
                now=now,
                require_current=True,
            )
            if normalized.get("phase") != "FINAL":
                raise FocusPayloadError("email delivery requires phase FINAL")
            digest = canonical_digest(normalized)
            # Render both forms in dry-run so escaping/shape errors surface.
            html_body = render_html(normalized, digest)
            text_body = render_text(normalized, digest)
            print(f"[dry-run] {subject_for(normalized)}")
            print(
                f"[dry-run] validated {len(normalized['focus'])} focus name(s); "
                f"html={len(html_body)} chars text={len(text_body)} chars "
                f"digest={digest[:12]}; no email or receipt write"
            )
            return 0
        deliver(
            payload,
            receipt_path=Path(args.receipt),
            force_send=bool(args.force_send),
            persist_r2=bool(args.persist_receipt_r2),
            now=now,
        )
        return 0
    except AmbiguousDeliveryError as exc:
        print(f"FOCUS EMAIL AMBIGUOUS: {exc}", file=sys.stderr)
        return 3
    except (EmailConfigurationError, EmailDeliveryError, ReceiptError) as exc:
        print(f"FOCUS EMAIL FAILED: {exc}", file=sys.stderr)
        return 1
    except (FocusPayloadError, ValueError) as exc:
        # FocusPayloadError is a ValueError; keep the explicit name here because
        # this boundary is safety-critical and should remain obvious in review.
        print(f"FOCUS EMAIL REFUSED: {exc}", file=sys.stderr)
        return 2
    except FocusEmailError as exc:
        print(f"FOCUS EMAIL REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
