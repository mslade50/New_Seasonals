"""Append-only, research-only registry shared by strategy research lanes.

The registry is deliberately separate from every production journal and state
file.  It never uploads, stages, allocates, or talks to a broker.  Callers must
provide the destination path explicitly so a test or scheduled research run
cannot silently contaminate an authoritative data file.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "research-experiment.v1"
KINDS = {
    "source",
    "hypothesis",
    "preregistration",
    "trial",
    "result",
    "disposition",
}

REQUIRED_FIELDS: dict[str, set[str]] = {
    "source": {"source_id", "source_type", "url", "retrieved_at", "content_hash"},
    "hypothesis": {
        "hypothesis_id",
        "source_ids",
        "claim",
        "mechanism",
        "instruments",
        "horizon",
        "first_rejection",
        "status",
    },
    "preregistration": {
        "experiment_id",
        "hypothesis_id",
        "family",
        "question",
        "universe",
        "signal",
        "decision_time",
        "entry_rule",
        "exit_rule",
        "cost_model",
        "validation_plan",
        "trial_budget",
        "promotion_gate",
        "kill_gate",
    },
    "trial": {"trial_id", "experiment_id", "parameter_set", "data_cutoff", "status"},
    "result": {"trial_id", "experiment_id", "metrics", "completed_at"},
    "disposition": {"experiment_id", "disposition", "reason", "decided_at"},
}

# These keys are executable instructions when they appear as top-level
# controls.  Research descriptions can discuss costs and hypothetical sizing,
# but the registry itself must never be consumable as an order ticket.
FORBIDDEN_KEYS = {
    "approve",
    "approved",
    "broker_account",
    "order_id",
    "order_type",
    "quantity",
    "shares",
    "stage_order",
    "submit_order",
    "target_weight",
}


class RegistryValidationError(ValueError):
    """Raised when a record violates the research registry contract."""


@contextmanager
def _registry_lock(registry_path: Path) -> Iterator[None]:
    """Take a non-blocking cross-process lock beside the append-only file."""

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry_path.with_name(f"{registry_path.name}.lock")
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RegistryValidationError(
                    f"registry is busy; retry after the active writer finishes: {registry_path}"
                ) from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RegistryValidationError(
                    f"registry is busy; retry after the active writer finishes: {registry_path}"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_utc(value: dt.datetime | None = None) -> str:
    stamp = value or utc_now()
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=dt.timezone.utc)
    return stamp.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def content_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def stable_id(prefix: str, payload: Any, length: int = 20) -> str:
    return f"{prefix}_{content_digest(payload)[:length]}"


def _non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _find_forbidden_keys(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).strip().lower()
            child_path = f"{path}.{key_text}" if path else key_text
            if key_text in FORBIDDEN_KEYS:
                found.append(child_path)
            found.extend(_find_forbidden_keys(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_keys(child, f"{path}[{index}]"))
    return found


def validate_record(record: Mapping[str, Any]) -> None:
    kind = str(record.get("kind", "")).strip()
    if kind not in KINDS:
        raise RegistryValidationError(f"unknown registry kind: {kind or '<missing>'}")

    missing = sorted(
        field for field in REQUIRED_FIELDS[kind] if not _non_empty(record.get(field))
    )
    if missing:
        raise RegistryValidationError(f"{kind} record missing required fields: {missing}")

    if record.get("research_only") is not True:
        raise RegistryValidationError("every registry record must set research_only=true")
    if record.get("no_order") is not True:
        raise RegistryValidationError("every registry record must set no_order=true")

    forbidden = _find_forbidden_keys(record)
    if forbidden:
        raise RegistryValidationError(
            "registry records cannot contain executable fields: " + ", ".join(forbidden)
        )

    if kind == "hypothesis":
        if not isinstance(record.get("source_ids"), list):
            raise RegistryValidationError("hypothesis source_ids must be a list")
        if not isinstance(record.get("instruments"), list):
            raise RegistryValidationError("hypothesis instruments must be a list")
    elif kind == "preregistration":
        budget = record.get("trial_budget")
        if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
            raise RegistryValidationError("trial_budget must be a positive integer")
    elif kind == "result" and not isinstance(record.get("metrics"), Mapping):
        raise RegistryValidationError("result metrics must be an object")


def prepare_record(
    record: Mapping[str, Any],
    *,
    clock: Callable[[], dt.datetime] = utc_now,
) -> dict[str, Any]:
    """Add immutable registry metadata and validate a caller record."""
    out = dict(record)
    out.setdefault("schema_version", SCHEMA_VERSION)
    out.setdefault("research_only", True)
    out.setdefault("no_order", True)
    out.setdefault("written_at", iso_utc(clock()))
    identity_payload = {
        key: value
        for key, value in out.items()
        if key not in {"record_id", "written_at"}
    }
    out.setdefault("record_id", stable_id("rec", identity_payload))
    if out["schema_version"] != SCHEMA_VERSION:
        raise RegistryValidationError(
            f"unsupported schema_version: {out['schema_version']}"
        )
    validate_record(out)
    return out


def _load_records_unlocked(
    registry_path: Path, *, skip_corrupt: bool = False
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        registry_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
            validate_record(record)
            records.append(record)
        except (json.JSONDecodeError, RegistryValidationError) as exc:
            if skip_corrupt:
                continue
            raise RegistryValidationError(
                f"invalid registry line {line_number} in {registry_path}: {exc}"
            ) from exc
    return records


def load_records(path: str | Path, *, skip_corrupt: bool = False) -> list[dict[str, Any]]:
    registry_path = Path(path)
    if not registry_path.exists():
        return []
    with _registry_lock(registry_path):
        return _load_records_unlocked(registry_path, skip_corrupt=skip_corrupt)


def append_records(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    clock: Callable[[], dt.datetime] = utc_now,
) -> int:
    """Append novel records and return the number written.

    Identical reruns are idempotent: an already-present ``record_id`` is
    skipped.  Changed research is a new record with a new content-derived id,
    preserving history instead of rewriting it.
    """
    registry_path = Path(path)
    prepared_records = [prepare_record(record, clock=clock) for record in records]
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with _registry_lock(registry_path):
        existing = (
            _load_records_unlocked(registry_path) if registry_path.exists() else []
        )
        known_ids = {str(record["record_id"]) for record in existing}
        pending: list[dict[str, Any]] = []
        for prepared in prepared_records:
            record_id = str(prepared["record_id"])
            if record_id in known_ids:
                continue
            pending.append(prepared)
            known_ids.add(record_id)

        if not pending:
            return 0

        with registry_path.open("a", encoding="utf-8", newline="\n") as handle:
            for record in pending:
                handle.write(canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return len(pending)


def hypothesis_fingerprints(records: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        str(record["hypothesis_fingerprint"])
        for record in records
        if record.get("kind") == "hypothesis" and record.get("hypothesis_fingerprint")
    }


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(records)
    kinds = Counter(str(record.get("kind", "unknown")) for record in materialized)
    trial_ids = {
        str(record["trial_id"])
        for record in materialized
        if record.get("kind") == "trial" and record.get("trial_id")
    }
    dispositions = Counter(
        str(record.get("disposition", "unknown"))
        for record in materialized
        if record.get("kind") == "disposition"
    )
    families = Counter(
        str(record.get("family", "unclassified"))
        for record in materialized
        if record.get("kind") == "preregistration"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "record_count": len(materialized),
        "by_kind": dict(sorted(kinds.items())),
        "registered_trials": len(trial_ids),
        "by_disposition": dict(sorted(dispositions.items())),
        "by_family": dict(sorted(families.items())),
        "research_only": True,
        "no_order": True,
    }
