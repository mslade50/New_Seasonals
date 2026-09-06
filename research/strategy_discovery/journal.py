"""Tamper-evident append-only journal for strategy-discovery runs."""

from __future__ import annotations

import json
import os
import secrets
import stat
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
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


@dataclass(frozen=True)
class _JournalLease:
    path: Path
    nonce: str


def _record_hash(record_without_hash: dict[str, Any]) -> str:
    return sha256_json(record_without_hash)


def _reject_journal_constant(value: str, *, path: Path, line_no: int) -> None:
    raise ContractError(
        f"journal {path}:{line_no}: non-finite numeric constant {value}"
    )


def is_symlink_or_reparse(path: Path) -> bool:
    """Return true for symlinks and Windows reparse-point aliases.

    ``Path.is_symlink`` covers ordinary links, but a Windows junction or
    another reparse-point type can still redirect an apparently local state
    path.  State files and lock files must never cross that boundary.
    """

    try:
        details = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ContractError(f"state path cannot be inspected: {path}") from exc
    attributes = getattr(details, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(details.st_mode) or bool(attributes & reparse_flag)


def load_journal(path: Path) -> list[dict[str, Any]]:
    """Read and verify every byte of an existing hash chain.

    A malformed final line is not ignored.  Silent recovery would turn a
    potentially truncated audit trail into apparently clean state.
    """

    if is_symlink_or_reparse(path):
        raise ContractError(f"journal must not be a symlink or reparse point: {path}")
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
            record = json.loads(
                line,
                parse_constant=partial(
                    _reject_journal_constant,
                    path=path,
                    line_no=line_no,
                ),
            )
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
    lock: _JournalLease | None = None,
) -> int:
    """Append new idempotent events with one flushed OS append.

    Existing records are never rewritten.  Replaying the exact event is a
    no-op; reusing an event key for different content fails closed.
    """

    canonical_lock_path = path.with_suffix(path.suffix + ".lock")
    if is_symlink_or_reparse(path):
        raise ContractError(f"journal must not be a symlink or reparse point: {path}")
    if lock is None:
        with exclusive_lock(canonical_lock_path) as acquired:
            return append_events(
                path,
                events,
                recorded_at=recorded_at,
                lock=acquired,
            )
    _validate_lock_lease(lock, canonical_lock_path)
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
    verified = load_journal(path)
    if len(verified) != len(existing) + len(fresh):
        raise ContractError("journal append verification found an unexpected record count")
    if verified[-1]["record_hash"] != fresh[-1]["record_hash"]:
        raise ContractError("journal append verification found an unexpected head hash")
    return len(fresh)


def _validate_lock_lease(lock: _JournalLease, expected_path: Path) -> None:
    if not isinstance(lock, _JournalLease):
        raise ContractError("journal append requires its canonical lock lease")
    if lock.path.resolve() != expected_path.resolve():
        raise ContractError("journal lock lease belongs to a different lock domain")
    try:
        payload = json.loads(lock.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("journal lock lease is no longer verifiable") from exc
    if payload.get("nonce") != lock.nonce:
        raise ContractError("journal lock lease ownership mismatch")


@contextmanager
def exclusive_lock(path: Path, *, timeout_seconds: float = 10.0):
    """Acquire a fail-closed cross-process lock by atomic create.

    Locks are never guessed stale or stolen. A crashed writer intentionally
    leaves a lock that requires an operator to inspect the journal and bundle
    before manually clearing it.
    """

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ContractError(f"transaction lock parent is unavailable: {path.parent}") from exc
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    deadline = time.monotonic() + timeout_seconds
    descriptor = -1
    while descriptor < 0:
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError as exc:
            if time.monotonic() >= deadline:
                raise ContractError(f"transaction lock unavailable: {path}") from exc
            time.sleep(0.05)
        except OSError as exc:
            raise ContractError(f"transaction lock cannot be created: {path}") from exc
    try:
        nonce = secrets.token_hex(16)
        payload = canonical_json(
            {"pid": os.getpid(), "lock_path": str(path), "nonce": nonce}
        ) + "\n"
        os.write(descriptor, payload.encode("utf-8"))
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        yield _JournalLease(path=path, nonce=nonce)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            current = path.read_text(encoding="utf-8")
        except OSError:
            current = None
        if current == payload:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def candidate_state_history(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    history: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["event_type"] != "CANDIDATE_OBSERVED":
            continue
        payload = record["payload"]
        fingerprint = payload.get("candidate_fingerprint")
        lifecycle = payload.get("lifecycle")
        if not fingerprint or not lifecycle:
            continue
        history.setdefault(str(fingerprint), []).append(
            {
                "lifecycle": str(lifecycle),
                "recorded_at": record["recorded_at"],
                "sequence": record["sequence"],
                "run_id": payload.get("run_id"),
                "research_spec_digests": payload.get("research_spec_digests", []),
            }
        )
    return history


def validation_record_times(records: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(record["payload"]["artifact_id"]): record["recorded_at"]
        for record in records
        if record["event_type"] == "VALIDATION_ATTACHED"
        and record["payload"].get("artifact_id")
    }


def latest_source_captures(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the latest COMPLETE capture per source as a cursor anchor.

    PARTIAL and UNKNOWN observations stay in the audit journal but cannot
    advance the accepted cursor chain.  This prevents a later replay from
    laundering a continuity failure into a complete capture.
    """

    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["event_type"] != "SOURCE_CAPTURE":
            continue
        payload = record["payload"]
        source_id = payload.get("source_id")
        if source_id and payload.get("status") == "COMPLETE":
            latest[str(source_id)] = payload
    return latest


def observed_source_captures(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index every journaled source observation by source and capture ID."""

    observed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if record["event_type"] != "SOURCE_CAPTURE":
            continue
        payload = record["payload"]
        source_id = payload.get("source_id")
        capture_id = payload.get("capture_id")
        if source_id and capture_id:
            observed[(str(source_id), str(capture_id))] = payload
    return observed


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
