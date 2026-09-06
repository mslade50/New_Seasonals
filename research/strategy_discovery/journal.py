"""Tamper-evident append-only journal for strategy-discovery runs."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_json, parse_timestamp, sha256_json

GENESIS = "GENESIS"
EVENT_TYPES = {
    "RUN",
    "SOURCE_CAPTURE",
    "CANDIDATE_OBSERVED",
    "VALIDATION_ATTACHED",
    "OWNER_TRANSITION",
}
RECORD_KEYS = {
    "sequence",
    "event_key",
    "event_type",
    "recorded_at",
    "payload",
    "prev_hash",
    "record_hash",
}


def _record_hash(record_without_hash: dict[str, Any]) -> str:
    return sha256_json(record_without_hash)


def load_journal(path: Path) -> list[dict[str, Any]]:
    """Read and verify every byte of an existing hash chain.

    A malformed final line is not ignored.  Silent recovery would turn a
    potentially truncated audit trail into apparently clean state.
    """

    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"journal {path}: unreadable ({exc})") from exc
    records: list[dict[str, Any]] = []
    expected_prev = GENESIS
    event_keys: dict[str, str] = {}
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            raise ContractError(f"journal {path}:{line_no}: blank lines are forbidden")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"journal {path}:{line_no}: invalid JSON ({exc.msg})") from exc
        if not isinstance(record, dict) or set(record) != RECORD_KEYS:
            raise ContractError(f"journal {path}:{line_no}: invalid record contract")
        if type(record["sequence"]) is not int or record["sequence"] != line_no:
            raise ContractError(f"journal {path}:{line_no}: non-contiguous sequence")
        if not isinstance(record["event_key"], str) or not record["event_key"]:
            raise ContractError(f"journal {path}:{line_no}: invalid event_key")
        if record["event_type"] not in EVENT_TYPES:
            raise ContractError(f"journal {path}:{line_no}: unknown event_type")
        parse_timestamp(record["recorded_at"], f"journal[{line_no}].recorded_at")
        if not isinstance(record["payload"], dict):
            raise ContractError(f"journal {path}:{line_no}: payload must be an object")
        if record["prev_hash"] != expected_prev:
            raise ContractError(f"journal {path}:{line_no}: broken prev_hash chain")
        body = {key: record[key] for key in RECORD_KEYS - {"record_hash"}}
        expected_hash = _record_hash(body)
        if record["record_hash"] != expected_hash:
            raise ContractError(f"journal {path}:{line_no}: record_hash mismatch")
        prior_hash = event_keys.get(record["event_key"])
        if prior_hash is not None and prior_hash != record["record_hash"]:
            raise ContractError(f"journal {path}:{line_no}: duplicate event_key")
        event_keys[record["event_key"]] = record["record_hash"]
        expected_prev = record["record_hash"]
        records.append(record)
    return records


def append_events(
    path: Path,
    events: Iterable[dict[str, Any]],
    *,
    recorded_at: str,
) -> int:
    """Append new idempotent events with one flushed OS append.

    Existing records are never rewritten.  Replaying the exact event is a
    no-op; reusing an event key for different content fails closed.
    """

    parse_timestamp(recorded_at, "recorded_at")
    existing = load_journal(path)
    by_key = {
        record["event_key"]: (record["event_type"], record["payload"])
        for record in existing
    }
    sequence = len(existing)
    prev_hash = existing[-1]["record_hash"] if existing else GENESIS
    fresh: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict) or set(event) != {"event_key", "event_type", "payload"}:
            raise ContractError("journal event must contain event_key, event_type, and payload")
        event_key = event["event_key"]
        if not isinstance(event_key, str) or not event_key:
            raise ContractError("journal event_key must be a non-empty string")
        if event["event_type"] not in EVENT_TYPES:
            raise ContractError(f"unknown journal event_type: {event['event_type']}")
        if not isinstance(event["payload"], dict):
            raise ContractError("journal payload must be an object")
        prior = by_key.get(event_key)
        current = (event["event_type"], event["payload"])
        if prior is not None:
            if prior != current:
                raise ContractError(f"journal event_key conflict: {event_key}")
            continue
        sequence += 1
        body = {
            "sequence": sequence,
            "event_key": event_key,
            "event_type": event["event_type"],
            "recorded_at": recorded_at,
            "payload": event["payload"],
            "prev_hash": prev_hash,
        }
        record = {**body, "record_hash": _record_hash(body)}
        fresh.append(record)
        by_key[event_key] = current
        prev_hash = record["record_hash"]
    if not fresh:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(canonical_json(record) + "\n" for record in fresh).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    descriptor = os.open(path, flags, 0o600)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(f"short journal append: {written}/{len(encoded)} bytes")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return len(fresh)


def latest_source_captures(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["event_type"] != "SOURCE_CAPTURE":
            continue
        payload = record["payload"]
        source_id = payload.get("source_id")
        if source_id:
            latest[str(source_id)] = payload
    return latest


def accepted_owner_transitions(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    accepted: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["event_type"] != "OWNER_TRANSITION":
            continue
        payload = record["payload"]
        fingerprint = payload.get("candidate_fingerprint")
        if fingerprint:
            accepted[str(fingerprint)] = payload
    return accepted


def accepted_validations(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Latest journaled validation artifact by artifact id."""

    accepted: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["event_type"] != "VALIDATION_ATTACHED":
            continue
        payload = record["payload"]
        artifact_id = payload.get("artifact_id")
        if artifact_id:
            accepted[str(artifact_id)] = payload
    return accepted
