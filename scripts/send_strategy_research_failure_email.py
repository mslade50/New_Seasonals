"""Send a deduplicated operational alert for a failed strategy-research run."""
from __future__ import annotations

import argparse
import datetime as dt
import html
import os
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daily_pitch import smtp_credentials
from research_delivery import (
    DeliveryNotSent,
    DeliveryUncertain,
    deliver_once,
)
from scripts.finalize_strategy_research import _send

DEFAULT_RECEIPT = ROOT / "data" / "strategy_research" / "failure_email_receipts.jsonl"
DEFAULT_RECIPIENT = "mckinleyslade@gmail.com"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("source_collection", "research_agent", "completion_check"))
    parser.add_argument("--summary", required=True)
    parser.add_argument("--day", default=str(dt.datetime.now(dt.timezone.utc).astimezone().date()))
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args(argv)
    subject = f"Strategy research pipeline failed — {args.day}"
    body = (
        '<div style="font-family:Segoe UI,Arial,sans-serif">'
        f"<h2>{html.escape(subject)}</h2>"
        f"<p><b>Phase:</b> {html.escape(args.phase)}</p>"
        f"<p>{html.escape(args.summary)}</p>"
        "<p>No strategy-research finding email was sent. Inspect "
        "artifacts/strategy_research_agent/last_run.json for the run log before retrying.</p></div>"
    )
    if not args.send:
        print(f"DRY_RUN: {subject} ({args.phase})")
        return 0
    sender, password = smtp_credentials()
    recipients = [
        value.strip()
        for value in os.environ.get("STRATEGY_RESEARCH_RECIPIENTS", DEFAULT_RECIPIENT).split(",")
        if value.strip()
    ]
    if not sender or not password or not recipients:
        print("STRATEGY RESEARCH FAILURE ALERT NOT SENT: credentials or recipients missing", file=sys.stderr)
        return 2
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.attach(MIMEText(body, "html", "utf-8"))
    try:
        result = deliver_once(
            args.receipt,
            {"product": "strategy-research-failure", "day": args.day, "phase": args.phase, "recipients": sorted(recipients)},
            {"subject": subject, "html": body},
            lambda: _send(sender, password, recipients, message),
        )
    except (DeliveryNotSent, DeliveryUncertain, OSError, RuntimeError, ValueError) as exc:
        print(f"STRATEGY RESEARCH FAILURE ALERT NOT CONFIRMED: {exc}", file=sys.stderr)
        return 2
    print(f"FAILURE_ALERT_{result['status']}: {args.phase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
