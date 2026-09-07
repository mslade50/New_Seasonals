"""Evaluate strategy research and send an email only when something qualifies."""
from __future__ import annotations

import argparse
import json
import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daily_pitch import smtp_credentials
from research.strategy_discovery.contracts import (
    ContractError,
    load_json,
    sha256_json,
)
from research.strategy_discovery.email_gate import (
    evaluate_email_package,
    render_email,
)
from research_delivery import (
    DeliveryNotSent,
    DeliveryUncertain,
    deliver_once,
)
from research_io import write_json

DEFAULT_DECISION = ROOT / "data" / "strategy_research" / "latest_decision.json"
DEFAULT_RECEIPTS = ROOT / "data" / "strategy_research" / "email_receipts.jsonl"
DEFAULT_RECIPIENT = "mckinleyslade@gmail.com"


def _send(sender: str, password: str, recipients: list[str], message: MIMEMultipart) -> None:
    server = None
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=30)
        server.starttls()
        server.login(sender, password)
    except Exception as exc:
        if server is not None:
            try:
                server.close()
            except (OSError, smtplib.SMTPException):
                pass
        raise DeliveryNotSent("SMTP failed before message submission") from exc
    try:
        refused = server.sendmail(sender, recipients, message.as_string())
        if refused:
            raise RuntimeError("recipient refusal makes delivery outcome incomplete")
    finally:
        try:
            server.quit()
        except (OSError, smtplib.SMTPException):
            try:
                server.close()
            except (OSError, smtplib.SMTPException):
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--family-fit", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--html-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = load_json(args.report)
        family_fit = load_json(args.family_fit)
        package = load_json(args.package)
        decision = evaluate_email_package(report, family_fit, package)
        decision["delivery_status"] = "PENDING"
        decision["delivery_id"] = None
        args.decision.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.decision, decision)
        if not decision["email_required"]:
            decision["delivery_status"] = "NO_EMAIL"
            write_json(args.decision, decision)
            print(
                f"NO_EMAIL: 0 of {decision['candidate_count']} candidate(s) passed the worthwhile gate"
            )
            return 0
        subject, body = render_email(decision)
        if args.html_out:
            args.html_out.parent.mkdir(parents=True, exist_ok=True)
            args.html_out.write_text(body, encoding="utf-8")
        if not args.send:
            decision["delivery_status"] = "DRY_RUN"
            write_json(args.decision, decision)
            print(f"DRY_RUN: {decision['eligible_count']} candidate(s) qualify; no email sent")
            return 0
        sender, password = smtp_credentials()
        recipients = [
            value.strip()
            for value in os.environ.get("STRATEGY_RESEARCH_RECIPIENTS", DEFAULT_RECIPIENT).split(",")
            if value.strip()
        ]
        if not sender or not password or not recipients:
            raise DeliveryNotSent("email credentials or recipients are missing")
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = sender
        message["To"] = ", ".join(recipients)
        message.attach(MIMEText(body, "html", "utf-8"))
        payload = {"subject": subject, "html": body, "decision_digest": sha256_json(decision)}
        result = deliver_once(
            args.receipt,
            {
                "product": "strategy-research",
                "discovery_run_id": decision["discovery_run_id"],
                "recipients": sorted(recipients),
            },
            payload,
            lambda: _send(sender, password, recipients, message),
        )
        decision["delivery_status"] = result["status"]
        decision["delivery_id"] = result["delivery_id"]
        write_json(args.decision, decision)
        print(f"EMAIL_{result['status']}: {decision['eligible_count']} candidate(s)")
    except (ContractError, DeliveryNotSent, DeliveryUncertain, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"STRATEGY RESEARCH FINALIZATION FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
