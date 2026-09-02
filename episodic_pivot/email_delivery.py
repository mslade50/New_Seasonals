"""Email delivery for the research-only Episodic Pivot shadow workflow.

The mailer consumes already-validated local artifacts.  It never discovers
symbols, contacts a broker, or changes a research decision.  Successful sends
write a non-sensitive receipt so a replay cannot silently duplicate an email.
"""

from __future__ import annotations

import hashlib
import html
import json
import mimetypes
import os
import smtplib
import ssl
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path
from typing import Any, Literal

from .config import DEFAULT_POLICY
from .premarket import nominate_candidates
from .schema import PremarketSnapshot, parse_timestamp
from .tradingview import target_session_date as tradingview_target_session_date

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024


class EmailDeliveryError(RuntimeError):
    """Raised when an EP email cannot be validated or delivered."""


@dataclass(frozen=True)
class EmailSettings:
    sender: str
    password: str
    recipients: tuple[str, ...]


@dataclass(frozen=True)
class EmailPayload:
    kind: str
    subject: str
    html_body: str
    plain_body: str
    attachments: tuple[Path, ...]
    receipt_path: Path
    source_sha256: str
    metadata: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise EmailDeliveryError(f"expected a JSON object: {path}")
    return raw


def _json_list(path: Path) -> list[Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise EmailDeliveryError(f"expected a JSON list: {path}")
    return raw


def dotenv_values(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    path = path.resolve()
    if not path.exists():
        raise EmailDeliveryError(f"email env file does not exist: {path}")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _valid_address(value: str) -> str:
    address = value.strip()
    if not address or "\r" in address or "\n" in address:
        raise EmailDeliveryError("invalid email recipient")
    _, parsed = parseaddr(address)
    if parsed != address or "@" not in parsed:
        raise EmailDeliveryError("invalid email recipient")
    return parsed


def resolve_email_settings(
    *, env_file: Path | None = None, environ: Mapping[str, str] | None = None
) -> EmailSettings:
    runtime = environ if environ is not None else os.environ
    fallback = dotenv_values(env_file)

    def value(key: str) -> str:
        return str(runtime.get(key) or fallback.get(key) or "").strip()

    sender = _valid_address(value("EMAIL_USER")) if value("EMAIL_USER") else ""
    password = value("EMAIL_PASS")
    if not sender or not password:
        raise EmailDeliveryError(
            "EMAIL_USER/EMAIL_PASS are unavailable; EP email was not delivered"
        )
    raw_recipients = value("EP_RECIPIENTS") or value("RECIPIENTS") or sender
    recipients = tuple(
        dict.fromkeys(
            _valid_address(item) for item in raw_recipients.split(",") if item.strip()
        )
    )
    if not recipients:
        raise EmailDeliveryError("EP_RECIPIENTS resolved to an empty list")
    return EmailSettings(sender=sender, password=password, recipients=recipients)


def _fmt_price(value: object) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_int(value: object) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def _email_shell(title: str, dek: str, body: str) -> str:
    return f"""<!doctype html><html><body style="margin:0;background:#f4f6f8;color:#15202b">
<div style="display:none;max-height:0;overflow:hidden">{html.escape(dek)}</div>
<main style="max-width:920px;margin:auto;padding:28px 18px;font:14px/1.5 Segoe UI,Arial,sans-serif">
<div style="font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#64748b;font-weight:700">Episodic Pivot · Shadow Research</div>
<h1 style="font:700 30px/1.15 Georgia,serif;color:#102a43;margin:7px 0 8px">{html.escape(title)}</h1>
<p style="color:#64748b;margin:0 0 18px">{html.escape(dek)}</p>
<div style="border-left:5px solid #b45309;background:#fff7ed;padding:11px 14px;margin:16px 0"><b>Research only.</b> No order was staged, routed, approved, or transmitted.</div>
{body}
<p style="color:#64748b;font-size:11px;margin-top:24px">Automated EP shadow delivery · broker route NONE</p>
</main></body></html>"""


def night_payload(import_path: Path) -> EmailPayload:
    import_path = import_path.resolve()
    raw = _json_object(import_path)
    if raw.get("schema_version") != 1:
        raise EmailDeliveryError("unsupported TradingView import schema")
    if raw.get("provider") != "TRADINGVIEW" or raw.get("session") != "after_hours":
        raise EmailDeliveryError("night email requires a validated TradingView after-hours import")
    if raw.get("result_count_verified") is not True:
        raise EmailDeliveryError("TradingView displayed count was not verified")
    rows = raw.get("snapshots")
    if not isinstance(rows, list):
        raise EmailDeliveryError("TradingView import is missing snapshots")
    extracted = int(raw.get("extracted_row_count", -1))
    reported = int(raw.get("reported_result_count", -1))
    if extracted != len(rows) or extracted != reported:
        raise EmailDeliveryError("TradingView import count mismatch")
    target_date = str(raw.get("target_session_date") or "").strip()
    captured_at = str(raw.get("captured_at") or "").strip()
    screen_id = str(raw.get("saved_screen_id") or "").strip()
    source_file = str(raw.get("source_file") or "").strip()
    source_file_hash = str(raw.get("source_file_sha256") or "").strip().lower()
    valid_hash = len(source_file_hash) == 64 and all(
        character in "0123456789abcdef" for character in source_file_hash
    )
    if not target_date or not captured_at or not screen_id or not source_file or not valid_hash:
        raise EmailDeliveryError("TradingView import identity is incomplete")
    try:
        derived_target = tradingview_target_session_date(
            captured_at, session="after_hours"
        )
    except (TypeError, ValueError) as exc:
        raise EmailDeliveryError("TradingView capture time is invalid") from exc
    if derived_target.isoformat() != target_date:
        raise EmailDeliveryError("TradingView target session date does not match capture time")

    snapshots = [PremarketSnapshot.from_dict(item) for item in rows]
    captured_timestamp = parse_timestamp(captured_at)
    for snapshot in snapshots:
        if (
            snapshot.provider != "TRADINGVIEW"
            or snapshot.session != "after_hours"
            or snapshot.saved_screen_id != screen_id
            or snapshot.target_session_date != target_date
            or snapshot.reported_result_count != reported
            or snapshot.extracted_row_count != extracted
            or parse_timestamp(snapshot.observed_at) != captured_timestamp
        ):
            raise EmailDeliveryError("TradingView row identity does not match its import")
    as_of = max(
        (parse_timestamp(item.observed_at) for item in snapshots),
        default=parse_timestamp(captured_at),
    )
    candidates = nominate_candidates(
        snapshots,
        as_of=as_of,
        policy=DEFAULT_POLICY,
        apply_candidate_limit=False,
    )
    table_rows = []
    for candidate in candidates[:50]:
        snapshot = candidate.snapshot
        table_rows.append(
            "<tr>"
            f"<td style='padding:7px;border-bottom:1px solid #e2e8f0'><b>{html.escape(snapshot.symbol)}</b><br><span style='color:#64748b'>{html.escape(snapshot.company_name)}</span></td>"
            f"<td style='padding:7px;border-bottom:1px solid #e2e8f0;text-align:right'>{_fmt_price(snapshot.last)}</td>"
            f"<td style='padding:7px;border-bottom:1px solid #e2e8f0;text-align:right'>{snapshot.discovery_gap_pct:+.2f}%</td>"
            f"<td style='padding:7px;border-bottom:1px solid #e2e8f0;text-align:right'>{snapshot.discovery_move_dollars:+.2f}</td>"
            f"<td style='padding:7px;border-bottom:1px solid #e2e8f0;text-align:right'>{_fmt_int(snapshot.premarket_volume)}</td>"
            "</tr>"
        )
    empty = (
        "<p style='background:#fff;padding:14px;border:1px solid #dbe3ea'>No broad movers qualified.</p>"
        if not table_rows
        else ""
    )
    body = f"""
<div style="display:flex;gap:10px;flex-wrap:wrap;margin:16px 0">
  <div style="background:#fff;border:1px solid #dbe3ea;padding:10px 13px"><b>{extracted}</b><br><span style="color:#64748b">screen rows</span></div>
  <div style="background:#fff;border:1px solid #dbe3ea;padding:10px 13px"><b>{len(candidates)}</b><br><span style="color:#64748b">broad nominees</span></div>
</div>
<p><b>Target session:</b> {html.escape(target_date)}<br><b>Captured:</b> {html.escape(captured_at)}<br><b>Saved screen:</b> {html.escape(screen_id)}</p>
{empty}
<table style="width:100%;border-collapse:collapse;background:#fff"><thead><tr><th style="text-align:left;padding:7px">Symbol</th><th style="text-align:right;padding:7px">Price</th><th style="text-align:right;padding:7px">Move</th><th style="text-align:right;padding:7px">$ move</th><th style="text-align:right;padding:7px">AH volume</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<p style="color:#64748b;font-size:12px">Showing up to 50 broad nominees. The validated normalized import is attached.</p>"""
    source_hash = sha256_file(import_path)
    return EmailPayload(
        kind="night",
        subject=(
            f"[EP Shadow] Night Queue | {target_date} | "
            f"{len(candidates)} nominees ({extracted} rows)"
        ),
        html_body=_email_shell(
            "After-hours queue captured",
            f"Validated queue for the {target_date} NYSE session.",
            body,
        ),
        plain_body=(
            f"EP after-hours queue for {target_date}: {extracted} screen rows, "
            f"{len(candidates)} broad nominees. Research only; no broker action."
        ),
        attachments=(import_path,),
        receipt_path=import_path.with_name(f"{import_path.stem}.email-delivery.json"),
        source_sha256=source_hash,
        metadata={
            "target_session_date": target_date,
            "screen_rows": extracted,
            "broad_nominees": len(candidates),
        },
    )


def _validate_run_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = _json_object(manifest_path)
    if manifest.get("schema_version") != 2 or manifest.get("run_id") != run_dir.name:
        raise EmailDeliveryError("EP manifest identity does not match its run directory")
    safety = manifest.get("safety")
    if not isinstance(safety, dict):
        raise EmailDeliveryError("EP manifest is missing its safety record")
    required_safety = {
        "research_only": True,
        "live_actions_enabled": False,
        "broker_route": "NONE",
        "order_submission_allowed": False,
        "order_staging_performed": False,
        "production_deployed": False,
    }
    for key, expected in required_safety.items():
        if safety.get(key) != expected:
            raise EmailDeliveryError(f"EP manifest safety mismatch: {key}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise EmailDeliveryError("EP manifest is missing artifact hashes")
    required = {
        "candidates.json",
        "decisions.json",
        "research_sizing_preview.json",
        "research_sizing_preview.csv",
        "report.html",
        "report.md",
    }
    if not required.issubset(artifacts):
        raise EmailDeliveryError("EP manifest is missing email deliverables")
    for name, record in artifacts.items():
        if Path(name).name != name or not isinstance(record, dict):
            raise EmailDeliveryError("EP manifest contains an unsafe artifact path")
        path = run_dir / name
        if path.resolve().parent != run_dir or not path.is_file():
            raise EmailDeliveryError(f"EP artifact is missing: {name}")
        if sha256_file(path) != record.get("sha256"):
            raise EmailDeliveryError(f"EP artifact digest mismatch: {name}")
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise EmailDeliveryError(f"EP artifact size mismatch: {name}")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise EmailDeliveryError("EP manifest is missing run counts")
    count_sources = {
        "candidates": "candidates.json",
        "decisions": "decisions.json",
        "research_sizing_previews": "research_sizing_preview.json",
    }
    for count_name, artifact_name in count_sources.items():
        actual = len(_json_list(run_dir / artifact_name))
        if counts.get(count_name) != actual:
            raise EmailDeliveryError(f"EP manifest count mismatch: {count_name}")
    return manifest


def _target_date_from_candidates(run_dir: Path) -> str:
    raw = _json_list(run_dir / "candidates.json")
    dates = {
        str(item.get("snapshot", {}).get("target_session_date") or "").strip()
        for item in raw
        if isinstance(item, dict) and isinstance(item.get("snapshot"), dict)
    }
    dates.discard("")
    if len(dates) != 1:
        return "unknown-session"
    value = next(iter(dates))
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise EmailDeliveryError("EP candidate target session date is invalid") from exc


def morning_payload(run_dir: Path) -> EmailPayload:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise EmailDeliveryError(f"morning run directory does not exist: {run_dir}")
    manifest = _validate_run_manifest(run_dir)
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    target_date = _target_date_from_candidates(run_dir)
    candidates = int(counts.get("candidates", 0))
    decisions = int(counts.get("decisions", 0))
    previews = int(counts.get("research_sizing_previews", 0))
    atr_qualified = int(counts.get("atr_qualified", candidates))
    news_researched = int(counts.get("news_research_selected", candidates))
    execution_verified = int(counts.get("execution_data_verified", 0))
    report_path = run_dir / "report.html"
    report_html = report_path.read_text(encoding="utf-8")
    if "Research only" not in report_html or "broker route NONE" not in report_html:
        raise EmailDeliveryError("EP HTML report is missing its research-only sentinels")
    attachments = (
        report_path,
        run_dir / "report.md",
        run_dir / "research_sizing_preview.csv",
        run_dir / "manifest.json",
    )
    source_hash = sha256_file(run_dir / "manifest.json")
    return EmailPayload(
        kind="morning",
        subject=(
            f"[EP Shadow] Morning Candidates | {target_date} | "
            f"{news_researched} researched, {atr_qualified} ATR-qualified"
        ),
        html_body=report_html,
        plain_body=(
            f"EP morning shadow report for {target_date}: {candidates} broad movers, "
            f"{atr_qualified} ATR-qualified, {news_researched} news-researched, "
            f"{execution_verified} with fresh execution verification, and "
            f"{previews} non-executable previews across {decisions} decisions. "
            "The complete HTML report and audit files are attached."
        ),
        attachments=attachments,
        receipt_path=run_dir / "email_delivery.json",
        source_sha256=source_hash,
        metadata={
            "run_id": manifest.get("run_id"),
            "target_session_date": target_date,
            "candidates": candidates,
            "decisions": decisions,
            "research_sizing_previews": previews,
            "atr_qualified": atr_qualified,
            "news_research_selected": news_researched,
            "execution_data_verified": execution_verified,
        },
    )


def failure_payload(
    *,
    phase: str,
    summary: str,
    output_root: Path,
    target_session_date: str = "",
) -> EmailPayload:
    phase = phase.strip().lower()
    if phase not in {"night", "morning"}:
        raise EmailDeliveryError("failure phase must be night or morning")
    summary = summary.strip()
    if not summary or len(summary) > 4_000:
        raise EmailDeliveryError("failure summary must contain 1-4000 characters")
    target = target_session_date.strip()
    if target:
        try:
            target = date.fromisoformat(target).isoformat()
        except ValueError as exc:
            raise EmailDeliveryError("failure target session date must use YYYY-MM-DD") from exc
    else:
        target = "unknown-session"
    canonical = json.dumps(
        {"phase": phase, "summary": summary, "target_session_date": target},
        sort_keys=True,
    )
    source_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    receipt = (
        output_root.resolve()
        / "email_failures"
        / f"{target}-{phase}-{source_hash[:12]}.email-delivery.json"
    )
    body = (
        "<div style='background:#fff;border:1px solid #fecaca;padding:14px'>"
        f"<p><b>Phase:</b> {html.escape(phase.title())}<br>"
        f"<b>Target session:</b> {html.escape(target)}</p>"
        f"<p>{html.escape(summary)}</p></div>"
    )
    return EmailPayload(
        kind="failure",
        subject=f"[EP Shadow] FAILURE | {phase.title()} | {target}",
        html_body=_email_shell(
            "Shadow run failed closed",
            "No research result was promoted and no broker action occurred.",
            body,
        ),
        plain_body=(
            f"EP {phase} shadow run failed closed for {target}: {summary}. "
            "No broker action occurred."
        ),
        attachments=(),
        receipt_path=receipt,
        source_sha256=source_hash,
        metadata={"phase": phase, "target_session_date": target},
    )


def test_payload(*, output_root: Path) -> EmailPayload:
    day = datetime.now(timezone.utc).date().isoformat()
    source_hash = hashlib.sha256(f"EP_EMAIL_TEST|{day}".encode()).hexdigest()
    return EmailPayload(
        kind="test",
        subject="[EP Shadow] Email delivery configured",
        html_body=_email_shell(
            "Email delivery is ready",
            "This is a configuration test. It contains no market data.",
            "<p>The night queue and morning research report will now be delivered by email. Successful runs retain local audit artifacts; no order capability was enabled.</p>",
        ),
        plain_body=(
            "EP shadow email delivery is configured. This test contains no market data "
            "and enabled no order capability."
        ),
        attachments=(),
        receipt_path=(
            output_root.resolve() / "email_tests" / f"{day}.email-delivery.json"
        ),
        source_sha256=source_hash,
        metadata={"test_date_utc": day},
    )


def _delivery_id(payload: EmailPayload, settings: EmailSettings) -> str:
    recipient_digest = hashlib.sha256(
        "\n".join(sorted(address.lower() for address in settings.recipients)).encode()
    ).hexdigest()
    canonical = f"{payload.kind}|{payload.source_sha256}|{recipient_digest}"
    return hashlib.sha256(canonical.encode()).hexdigest()


def _existing_delivery(
    receipt_path: Path, delivery_id: str
) -> Literal["MISSING", "MATCH", "CONFLICT"]:
    if not receipt_path.exists():
        return "MISSING"
    try:
        receipt = _json_object(receipt_path)
    except (OSError, ValueError, json.JSONDecodeError, EmailDeliveryError) as exc:
        raise EmailDeliveryError(f"invalid existing email receipt: {receipt_path}") from exc
    if receipt.get("status") != "SENT":
        raise EmailDeliveryError(
            f"existing email receipt is not a successful delivery: {receipt_path}"
        )
    return "MATCH" if receipt.get("delivery_id") == delivery_id else "CONFLICT"


def _write_receipt(
    payload: EmailPayload, settings: EmailSettings, *, delivery_id: str
) -> None:
    receipt = {
        "schema_version": 1,
        "record_type": "EP_RESEARCH_EMAIL_DELIVERY_V1",
        "status": "SENT",
        "delivery_id": delivery_id,
        "kind": payload.kind,
        "subject": payload.subject,
        "source_sha256": payload.source_sha256,
        "sent_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "recipient_count": len(settings.recipients),
        "smtp_host": SMTP_HOST,
        "research_only": True,
        "broker_route": "NONE",
        "order_submission_allowed": False,
        "metadata": payload.metadata,
    }
    payload.receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = payload.receipt_path.with_suffix(payload.receipt_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(payload.receipt_path)


def deliver_email(
    payload: EmailPayload,
    settings: EmailSettings,
    *,
    send: bool,
    resend: bool = False,
) -> str:
    delivery_id = _delivery_id(payload, settings)
    existing = _existing_delivery(payload.receipt_path, delivery_id)
    if not resend:
        if existing == "MATCH":
            return "ALREADY_SENT"
        if existing == "CONFLICT":
            raise EmailDeliveryError(
                "a successful receipt exists for a different source or recipient set; "
                "use --resend only after reviewing the change"
            )
    if not send:
        return "DRY_RUN"
    if "\r" in payload.subject or "\n" in payload.subject:
        raise EmailDeliveryError("email subject contains a newline")
    attachment_bytes = sum(path.stat().st_size for path in payload.attachments)
    if attachment_bytes > MAX_ATTACHMENT_BYTES:
        raise EmailDeliveryError("EP email attachments exceed the 20 MB safety limit")

    message = EmailMessage()
    message["Subject"] = payload.subject
    message["From"] = settings.sender
    message["To"] = ", ".join(settings.recipients)
    message["X-EP-Research-Only"] = "true"
    message.set_content(payload.plain_body)
    message.add_alternative(payload.html_body, subtype="html")
    for path in payload.attachments:
        mime, _ = mimetypes.guess_type(path.name)
        maintype, subtype = (
            mime.split("/", 1) if mime else ("application", "octet-stream")
        )
        message.add_attachment(
            path.read_bytes(), maintype=maintype, subtype=subtype, filename=path.name
        )

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            server.login(settings.sender, settings.password)
            server.send_message(message)
    except (OSError, smtplib.SMTPException) as exc:
        raise EmailDeliveryError(
            f"EP email send failed ({type(exc).__name__}); no delivery receipt was written"
        ) from exc
    _write_receipt(payload, settings, delivery_id=delivery_id)
    return "SENT"


def payload_summary(payload: EmailPayload) -> dict[str, Any]:
    return {
        "kind": payload.kind,
        "subject": payload.subject,
        "attachment_names": [path.name for path in payload.attachments],
        "receipt_path": str(payload.receipt_path),
        "metadata": payload.metadata,
    }
