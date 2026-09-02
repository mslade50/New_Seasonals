"""Deliver EP shadow artifacts by email; dry-run unless --send is supplied."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from episodic_pivot.email_delivery import (
    EmailDeliveryError,
    EmailPayload,
    deliver_email,
    failure_payload,
    morning_payload,
    night_payload,
    payload_summary,
    resolve_email_settings,
    test_payload,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Email EP shadow research artifacts (dry-run by default)"
    )
    parser.add_argument(
        "--kind", required=True, choices=("night", "morning", "failure", "test")
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        help="night import JSON or morning run directory",
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--failure-phase", choices=("night", "morning"))
    parser.add_argument("--failure-summary")
    parser.add_argument("--target-session-date", default="")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "episodic_pivot",
        help="receipt root for failure/test emails",
    )
    parser.add_argument("--send", action="store_true")
    parser.add_argument(
        "--resend",
        action="store_true",
        help="explicitly bypass a matching successful delivery receipt",
    )
    return parser


def _build_payload(args: argparse.Namespace) -> EmailPayload:
    if args.kind == "night":
        if args.artifact is None:
            raise EmailDeliveryError("--artifact is required for a night email")
        return night_payload(args.artifact)
    if args.kind == "morning":
        if args.artifact is None:
            raise EmailDeliveryError("--artifact is required for a morning email")
        return morning_payload(args.artifact)
    if args.kind == "failure":
        if not args.failure_phase or not args.failure_summary:
            raise EmailDeliveryError(
                "--failure-phase and --failure-summary are required for a failure email"
            )
        return failure_payload(
            phase=args.failure_phase,
            summary=args.failure_summary,
            target_session_date=args.target_session_date,
            output_root=args.output_root,
        )
    return test_payload(output_root=args.output_root)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = _build_payload(args)
        summary = payload_summary(payload)
        if not args.send:
            print(json.dumps({"delivery_status": "DRY_RUN", **summary}, indent=2))
            print("No email was sent. Add --send after reviewing this payload.")
            return 0
        settings = resolve_email_settings(env_file=args.env_file)
        status = deliver_email(
            payload, settings, send=True, resend=bool(args.resend)
        )
    except (EmailDeliveryError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"EP EMAIL DELIVERY FAILED: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "delivery_status": status,
                "recipient_count": len(settings.recipients),
                **summary,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
