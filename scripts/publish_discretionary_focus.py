"""Publish a validated Discretionary Focus payload or email receipt to R2.

For a focus payload, the immutable history object is uploaded and verified
before ``discretionary_focus/current.json`` is replaced.  The private-site
Function reads only that current key.  A failed publish therefore leaves the
previous payload in place, where its explicit expiry will make it fail closed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cache_io  # noqa: E402
from discretionary_focus.contracts import (  # noqa: E402
    canonical_digest,
    validate_payload,
)
from scripts.check_discretionary_focus_session import delivery_window_gate  # noqa: E402


CURRENT_KEY = "discretionary_focus/current.json"
RECEIPT_KEY = "discretionary_focus/email_receipt.json"
HISTORY_PREFIX = "discretionary_focus/history"
RECEIPT_SCHEMA = "discretionary-focus-email-receipt.v1"


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON at {path} must be an object")
    return value


def _verified_upload(path: Path, key: str) -> None:
    if not cache_io.upload_from_local(str(path), key):
        raise RuntimeError(f"R2 upload failed: {key}")
    metadata = cache_io.head(key)
    if not metadata:
        raise RuntimeError(f"R2 HEAD failed after upload: {key}")
    expected_size = path.stat().st_size
    actual_size = int(metadata.get("ContentLength") or -1)
    if actual_size != expected_size:
        raise RuntimeError(
            f"R2 size mismatch after upload for {key}: {actual_size} != {expected_size}"
        )


def publish_payload(
    path: Path,
    *,
    now: dt.datetime | None = None,
) -> tuple[str, str]:
    payload = validate_payload(_read_object(path), now=now, require_current=True)
    if payload.get("phase") != "FINAL":
        raise ValueError("only a current FINAL Focus payload may be published")
    allowed, market_date = delivery_window_gate(now)
    if not allowed or market_date.isoformat() != payload["valid_for"]:
        raise ValueError(
            "Focus publication is allowed only 08:25-09:20 New York on valid_for"
        )
    digest = canonical_digest(payload)
    valid_for = payload["valid_for"]
    phase = str(payload["phase"]).lower()
    archive_key = f"{HISTORY_PREFIX}/{valid_for}/{phase}-{digest}.json"
    _verified_upload(path, archive_key)
    _verified_upload(path, CURRENT_KEY)
    return archive_key, CURRENT_KEY


def publish_receipt(path: Path) -> str:
    receipt = _read_object(path)
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError(f"email receipt schema must be {RECEIPT_SCHEMA}")
    deliveries = receipt.get("deliveries")
    if not isinstance(deliveries, list) or not deliveries:
        raise ValueError("email receipt deliveries must be a non-empty list")

    latest = deliveries[-1]
    if not isinstance(latest, dict):
        raise ValueError("latest email delivery must be an object")
    required = {
        "attempt_id",
        "valid_for",
        "digest",
        "status",
        "started_at",
        "sent_at",
        "recipients",
    }
    missing = sorted(required - set(latest))
    if missing:
        raise ValueError(f"latest email delivery is missing: {', '.join(missing)}")
    if latest.get("status") != "sent":
        raise ValueError("latest email delivery must have status sent")
    digest = latest.get("digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("latest email delivery digest must be a SHA-256 hex string")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(
            "latest email delivery digest must be a SHA-256 hex string"
        ) from exc
    recipients = latest.get("recipients")
    if not isinstance(recipients, list) or not recipients or not all(
        isinstance(value, str) and value.strip() for value in recipients
    ):
        raise ValueError("latest email delivery recipients must be a non-empty text list")
    _verified_upload(path, RECEIPT_KEY)
    return RECEIPT_KEY


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="validated Focus payload")
    group.add_argument("--receipt-only", type=Path, help="successful email receipt")
    args = parser.parse_args()

    if not cache_io.is_configured():
        print("R2 credentials are required; nothing was published")
        return 1
    try:
        if args.input:
            archive, current = publish_payload(args.input)
            print(f"published {archive}")
            print(f"published {current}")
        else:
            print(f"published {publish_receipt(args.receipt_only)}")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Discretionary Focus publish failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
