"""Local single-writer delivery claims for optional research messages.

An interrupted/uncertain attempt requires provider reconciliation. Content changes
cannot silently bypass a retained daily intent. No transport runs at import.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from research_io import append_jsonl, file_lock, read_jsonl


class DeliveryUncertain(RuntimeError):
    pass


class DeliveryNotSent(RuntimeError):
    """Transport can prove the attempt failed before message submission."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def deliver_once(path: Path, identity: dict, payload: dict, send):
    delivery_id, content_digest = digest(identity), digest(payload)
    with file_lock(path):
        records = read_jsonl(path)
        for record in records:
            fields = {"schema_version", "delivery_id", "content_digest", "state", "at"}
            valid_hashes = all(isinstance(record.get(k), str) and re.fullmatch(r"[0-9a-f]{64}", record[k]) for k in ("delivery_id", "content_digest"))
            if set(record) != fields or not valid_hashes or record.get("schema_version") != "research-delivery.v1" or record.get("state") not in {"SENDING", "SENT", "NOT_SENT"}:
                raise ValueError("Invalid delivery receipt; evidence preserved")
            if datetime.fromisoformat(str(record["at"]).replace("Z", "+00:00")).tzinfo is None:
                raise ValueError("Invalid delivery receipt timestamp; evidence preserved")
        previous = next((r for r in reversed(records) if r.get("delivery_id") == delivery_id), None)
        if previous:
            if previous.get("content_digest") != content_digest:
                raise DeliveryUncertain("This delivery identity already has different content; reconcile its receipt before a new edition")
            if previous["state"] == "SENT":
                return {"status": "ALREADY_SENT", "delivery_id": delivery_id}
            if previous["state"] != "NOT_SENT":
                raise DeliveryUncertain("Delivery may already have been accepted; reconcile the provider before retrying")
        def record(state):
            append_jsonl(path, [{"schema_version": "research-delivery.v1", "delivery_id": delivery_id,
                "content_digest": content_digest, "state": state,
                "at": datetime.now(timezone.utc).isoformat()}])
        record("SENDING")
        try:
            send()
        except DeliveryNotSent:
            record("NOT_SENT")
            raise
        # Any other exception (including process termination) leaves SENDING.
        record("SENT")
        return {"status": "SENT", "delivery_id": delivery_id}
