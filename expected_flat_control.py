"""Offline expected-flat reconciliation primitives.

This module is deliberately isolated from every broker, executor, network,
storage, email, and scheduler integration.  It consumes a complete local
manifest, evaluates whether due signal tranches are flat, and renders
deterministic artifacts.  A producer can later adapt its own append-only
events into this schema, but this module must remain a pure control boundary.

The important safety property is fail-closed reporting: only an explicitly
authoritative LIVE run with complete sources and reconciled aggregate
positions may produce CLEAR.  DISABLED, FIXTURE, and SHADOW runs can expose an
ALERT, but can never be mistaken for operational clearance.
"""

from __future__ import annotations

import html
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from itertools import pairwise
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
ZERO = Decimal(0)


class ManifestError(ValueError):
    """The local manifest does not conform to the frozen schema."""


class RunMode(str, Enum):
    DISABLED = "DISABLED"
    FIXTURE = "FIXTURE"
    SHADOW = "SHADOW"
    LIVE = "LIVE"


class ControlState(str, Enum):
    CLEAR = "CLEAR"
    ALERT = "ALERT"
    UNKNOWN = "UNKNOWN"
    NOT_SCHEDULED = "NOT_SCHEDULED"


class ReasonCode(str, Enum):
    RUN_DISABLED = "RUN_DISABLED"
    NON_AUTHORITATIVE_RUN = "NON_AUTHORITATIVE_RUN"
    NO_DUE_OBLIGATIONS = "NO_DUE_OBLIGATIONS"
    OBLIGATION_NOT_DUE = "OBLIGATION_NOT_DUE"
    OBLIGATION_CANCELLED = "OBLIGATION_CANCELLED"
    RESIDUAL_POSITION_LONG = "RESIDUAL_POSITION_LONG"
    RESIDUAL_POSITION_SHORT = "RESIDUAL_POSITION_SHORT"
    POSITION_REVERSED = "POSITION_REVERSED"
    MISSING_REQUIRED_PRODUCER_RECEIPT = "MISSING_REQUIRED_PRODUCER_RECEIPT"
    DUPLICATE_PRODUCER_RECEIPT = "DUPLICATE_PRODUCER_RECEIPT"
    INCOMPLETE_PRODUCER_RECEIPT = "INCOMPLETE_PRODUCER_RECEIPT"
    STALE_PRODUCER_RECEIPT = "STALE_PRODUCER_RECEIPT"
    RECEIPT_EVENT_SET_MISMATCH = "RECEIPT_EVENT_SET_MISMATCH"
    RECEIPT_ZERO_OBLIGATION_MISMATCH = "RECEIPT_ZERO_OBLIGATION_MISMATCH"
    SOURCE_SESSION_MISMATCH = "SOURCE_SESSION_MISMATCH"
    SOURCE_SEQUENCE_INCOMPLETE = "SOURCE_SEQUENCE_INCOMPLETE"
    OBLIGATION_REVISION_GAP = "OBLIGATION_REVISION_GAP"
    CONFLICTING_DUPLICATE_REVISION = "CONFLICTING_DUPLICATE_REVISION"
    IMMUTABLE_IDENTITY_CHANGED = "IMMUTABLE_IDENTITY_CHANGED"
    NON_MONOTONIC_REVISION = "NON_MONOTONIC_REVISION"
    DUPLICATE_ACTIVE_OBLIGATION = "DUPLICATE_ACTIVE_OBLIGATION"
    MISSING_BASELINE = "MISSING_BASELINE"
    DUPLICATE_BASELINE = "DUPLICATE_BASELINE"
    FUTURE_BASELINE = "FUTURE_BASELINE"
    MISSING_POSITION_SNAPSHOT = "MISSING_POSITION_SNAPSHOT"
    DUPLICATE_POSITION_SNAPSHOT = "DUPLICATE_POSITION_SNAPSHOT"
    STALE_POSITION_SNAPSHOT = "STALE_POSITION_SNAPSHOT"
    POSITION_SNAPSHOT_PRECEDES_EVIDENCE = "POSITION_SNAPSHOT_PRECEDES_EVIDENCE"
    INCOMPLETE_EXECUTION_SOURCE = "INCOMPLETE_EXECUTION_SOURCE"
    STALE_EXECUTION_SOURCE = "STALE_EXECUTION_SOURCE"
    EXECUTION_EVENT_SET_MISMATCH = "EXECUTION_EVENT_SET_MISMATCH"
    EXECUTION_ZERO_EVENT_MISMATCH = "EXECUTION_ZERO_EVENT_MISMATCH"
    INCOMPLETE_BASELINE_SOURCE = "INCOMPLETE_BASELINE_SOURCE"
    STALE_BASELINE_SOURCE = "STALE_BASELINE_SOURCE"
    BASELINE_EVENT_SET_MISMATCH = "BASELINE_EVENT_SET_MISMATCH"
    BASELINE_ZERO_EVENT_MISMATCH = "BASELINE_ZERO_EVENT_MISMATCH"
    CONFLICTING_DUPLICATE_EXECUTION = "CONFLICTING_DUPLICATE_EXECUTION"
    CORRECTION_REVISION_GAP = "CORRECTION_REVISION_GAP"
    AMBIGUOUS_CORRECTION_FAMILY = "AMBIGUOUS_CORRECTION_FAMILY"
    INCOMPLETE_POSITION_SOURCE = "INCOMPLETE_POSITION_SOURCE"
    STALE_POSITION_SOURCE = "STALE_POSITION_SOURCE"
    POSITION_EVENT_SET_MISMATCH = "POSITION_EVENT_SET_MISMATCH"
    AMBIGUOUS_CONTRACT_FALLBACK = "AMBIGUOUS_CONTRACT_FALLBACK"
    CONTRACT_IDENTITY_CONFLICT = "CONTRACT_IDENTITY_CONFLICT"
    AGGREGATE_POSITION_MISMATCH = "AGGREGATE_POSITION_MISMATCH"
    FUTURE_SOURCE_EVENT = "FUTURE_SOURCE_EVENT"


class CorrectionAction(str, Enum):
    APPLY = "APPLY"
    VOID = "VOID"


@dataclass(frozen=True, order=True)
class Issue:
    reason: ReasonCode
    detail: str
    scope: str = "run"

    def to_dict(self) -> dict[str, str]:
        return {"reason": self.reason.value, "scope": self.scope, "detail": self.detail}


def _strict_object(
    raw: Any,
    *,
    where: str,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{where} must be an object")
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(raw))
    extra = sorted(set(raw) - allowed)
    if missing:
        raise ManifestError(f"{where} missing required field(s): {', '.join(missing)}")
    if extra:
        raise ManifestError(f"{where} has unknown field(s): {', '.join(extra)}")
    return raw


def _nonempty_string(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{where} must be a non-empty string")
    return value.strip()


def _integer(value: Any, *, where: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{where} must be an integer")
    if minimum is not None and value < minimum:
        raise ManifestError(f"{where} must be >= {minimum}")
    return value


def _boolean(value: Any, *, where: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{where} must be a boolean")
    return value


def _decimal(value: Any, *, where: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ManifestError(
            f"{where} must be a decimal string or integer (floats are forbidden)"
        )
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ManifestError(f"{where} is not a valid decimal") from exc
    if not result.is_finite():
        raise ManifestError(f"{where} must be finite")
    return result


def _timestamp(value: Any, *, where: str) -> datetime:
    text = _nonempty_string(value, where=where)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestError(f"{where} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ManifestError(f"{where} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _date(value: Any, *, where: str) -> date:
    text = _nonempty_string(value, where=where)
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ManifestError(f"{where} must be YYYY-MM-DD") from exc


def _string_list(value: Any, *, where: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ManifestError(f"{where} must be an array")
    result = tuple(_nonempty_string(item, where=f"{where}[]") for item in value)
    if len(set(result)) != len(result):
        raise ManifestError(f"{where} must not contain duplicates")
    return result


def _int_list(value: Any, *, where: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ManifestError(f"{where} must be an array")
    result = tuple(_integer(item, where=f"{where}[]", minimum=1) for item in value)
    if len(set(result)) != len(result):
        raise ManifestError(f"{where} must not contain duplicates")
    return result


def _enum(enum_type: type[Enum], value: Any, *, where: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ManifestError(f"{where} must be one of: {allowed}") from exc


def decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    if value == ZERO:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def timestamp_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Contract:
    symbol: str
    sec_type: str
    currency: str
    exchange: str
    con_id: int | None = None
    expiry: str = ""

    @classmethod
    def from_dict(cls, raw: Any, *, where: str) -> Contract:
        obj = _strict_object(
            raw,
            where=where,
            required=("symbol", "sec_type", "currency", "exchange"),
            optional=("con_id", "expiry"),
        )
        con_id_raw = obj.get("con_id")
        con_id = (
            None
            if con_id_raw is None
            else _integer(con_id_raw, where=f"{where}.con_id", minimum=1)
        )
        return cls(
            symbol=" ".join(
                _nonempty_string(obj["symbol"], where=f"{where}.symbol").upper().split()
            ),
            sec_type=" ".join(
                _nonempty_string(obj["sec_type"], where=f"{where}.sec_type")
                .upper()
                .split()
            ),
            currency=" ".join(
                _nonempty_string(obj["currency"], where=f"{where}.currency")
                .upper()
                .split()
            ),
            exchange=" ".join(
                _nonempty_string(obj["exchange"], where=f"{where}.exchange")
                .upper()
                .split()
            ),
            con_id=con_id,
            expiry="".join(str(obj.get("expiry", "")).upper().split()),
        )

    @property
    def fallback(self) -> tuple[str, str, str, str, str]:
        return (self.symbol, self.sec_type, self.currency, self.exchange, self.expiry)

    def label(self) -> str:
        expiry = f"/{self.expiry}" if self.expiry else ""
        suffix = (
            f" conId={self.con_id}"
            if self.con_id is not None
            else " normalized-fallback"
        )
        return f"{self.symbol}/{self.sec_type}/{self.currency}/{self.exchange}{expiry}{suffix}"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    run_mode: RunMode
    operationally_authoritative: bool
    as_of: datetime
    session_date: date
    required_producers: tuple[str, ...]
    max_position_age_seconds: int

    @classmethod
    def from_dict(cls, raw: Any) -> RunSpec:
        obj = _strict_object(
            raw,
            where="run",
            required=(
                "run_id",
                "run_mode",
                "operationally_authoritative",
                "as_of",
                "session_date",
                "required_producers",
                "max_position_age_seconds",
            ),
        )
        mode = _enum(RunMode, obj["run_mode"], where="run.run_mode")
        authoritative = _boolean(
            obj["operationally_authoritative"], where="run.operationally_authoritative"
        )
        if authoritative and mode is not RunMode.LIVE:
            raise ManifestError("only LIVE may set operationally_authoritative=true")
        required_producers = _string_list(
            obj["required_producers"], where="run.required_producers"
        )
        if mode is not RunMode.DISABLED and not required_producers:
            raise ManifestError(
                "run.required_producers must not be empty outside DISABLED mode"
            )
        return cls(
            run_id=_nonempty_string(obj["run_id"], where="run.run_id"),
            run_mode=mode,
            operationally_authoritative=authoritative,
            as_of=_timestamp(obj["as_of"], where="run.as_of"),
            session_date=_date(obj["session_date"], where="run.session_date"),
            required_producers=required_producers,
            max_position_age_seconds=_integer(
                obj["max_position_age_seconds"],
                where="run.max_position_age_seconds",
                minimum=1,
            ),
        )


@dataclass(frozen=True)
class SourceReceipt:
    source_id: str
    receipt_id: str
    session_date: date
    complete: bool
    complete_through: datetime
    zero_events: bool
    event_ids: tuple[str, ...]
    event_count: int
    source_sequence_first: int
    source_sequence_last: int
    source_sequences: tuple[int, ...]

    @classmethod
    def from_dict(cls, raw: Any, *, where: str) -> SourceReceipt:
        obj = _strict_object(
            raw,
            where=where,
            required=(
                "source_id",
                "receipt_id",
                "session_date",
                "complete",
                "complete_through",
                "zero_events",
                "event_ids",
                "event_count",
                "source_sequence_first",
                "source_sequence_last",
                "source_sequences",
            ),
        )
        sequences = _int_list(
            obj["source_sequences"], where=f"{where}.source_sequences"
        )
        return cls(
            source_id=_nonempty_string(obj["source_id"], where=f"{where}.source_id"),
            receipt_id=_nonempty_string(obj["receipt_id"], where=f"{where}.receipt_id"),
            session_date=_date(obj["session_date"], where=f"{where}.session_date"),
            complete=_boolean(obj["complete"], where=f"{where}.complete"),
            complete_through=_timestamp(
                obj["complete_through"], where=f"{where}.complete_through"
            ),
            zero_events=_boolean(obj["zero_events"], where=f"{where}.zero_events"),
            event_ids=_string_list(obj["event_ids"], where=f"{where}.event_ids"),
            event_count=_integer(
                obj["event_count"], where=f"{where}.event_count", minimum=0
            ),
            source_sequence_first=_integer(
                obj["source_sequence_first"],
                where=f"{where}.source_sequence_first",
                minimum=1,
            ),
            source_sequence_last=_integer(
                obj["source_sequence_last"],
                where=f"{where}.source_sequence_last",
                minimum=1,
            ),
            source_sequences=sequences,
        )


@dataclass(frozen=True)
class ObligationRevision:
    event_id: str
    revision: int
    producer_id: str
    source_sequence: int
    recorded_at: datetime
    session_date: date
    account: str
    signal_id: str
    tranche_id: str
    strategy: str
    contract: Contract
    expected_flat_by: datetime
    active: bool
    note: str = ""

    @classmethod
    def from_dict(cls, raw: Any, *, where: str) -> ObligationRevision:
        obj = _strict_object(
            raw,
            where=where,
            required=(
                "event_id",
                "revision",
                "producer_id",
                "source_sequence",
                "recorded_at",
                "session_date",
                "account",
                "signal_id",
                "tranche_id",
                "strategy",
                "contract",
                "expected_flat_by",
                "active",
            ),
            optional=("note",),
        )
        return cls(
            event_id=_nonempty_string(obj["event_id"], where=f"{where}.event_id"),
            revision=_integer(obj["revision"], where=f"{where}.revision", minimum=1),
            producer_id=_nonempty_string(
                obj["producer_id"], where=f"{where}.producer_id"
            ),
            source_sequence=_integer(
                obj["source_sequence"], where=f"{where}.source_sequence", minimum=1
            ),
            recorded_at=_timestamp(obj["recorded_at"], where=f"{where}.recorded_at"),
            session_date=_date(obj["session_date"], where=f"{where}.session_date"),
            account=_nonempty_string(obj["account"], where=f"{where}.account"),
            signal_id=_nonempty_string(obj["signal_id"], where=f"{where}.signal_id"),
            tranche_id=_nonempty_string(obj["tranche_id"], where=f"{where}.tranche_id"),
            strategy=_nonempty_string(obj["strategy"], where=f"{where}.strategy"),
            contract=Contract.from_dict(obj["contract"], where=f"{where}.contract"),
            expected_flat_by=_timestamp(
                obj["expected_flat_by"], where=f"{where}.expected_flat_by"
            ),
            active=_boolean(obj["active"], where=f"{where}.active"),
            note=str(obj.get("note", "")).strip(),
        )

    def immutable_identity(self) -> tuple[Any, ...]:
        return (
            self.event_id,
            self.producer_id,
            self.session_date,
            self.account,
            self.signal_id,
            self.tranche_id,
            self.strategy,
            self.contract,
        )


@dataclass(frozen=True)
class PositionBaseline:
    baseline_id: str
    source_sequence: int
    effective_at: datetime
    account: str
    signal_id: str
    tranche_id: str
    contract: Contract
    signed_qty: Decimal

    @classmethod
    def from_dict(cls, raw: Any, *, where: str) -> PositionBaseline:
        obj = _strict_object(
            raw,
            where=where,
            required=(
                "baseline_id",
                "source_sequence",
                "effective_at",
                "account",
                "signal_id",
                "tranche_id",
                "contract",
                "signed_qty",
            ),
        )
        return cls(
            baseline_id=_nonempty_string(
                obj["baseline_id"], where=f"{where}.baseline_id"
            ),
            source_sequence=_integer(
                obj["source_sequence"], where=f"{where}.source_sequence", minimum=1
            ),
            effective_at=_timestamp(obj["effective_at"], where=f"{where}.effective_at"),
            account=_nonempty_string(obj["account"], where=f"{where}.account"),
            signal_id=_nonempty_string(obj["signal_id"], where=f"{where}.signal_id"),
            tranche_id=_nonempty_string(obj["tranche_id"], where=f"{where}.tranche_id"),
            contract=Contract.from_dict(obj["contract"], where=f"{where}.contract"),
            signed_qty=_decimal(obj["signed_qty"], where=f"{where}.signed_qty"),
        )


@dataclass(frozen=True)
class ExecutionRevision:
    execution_id: str
    correction_family_id: str
    correction_revision: int
    correction_action: CorrectionAction
    source_sequence: int
    occurred_at: datetime
    account: str
    signal_id: str
    tranche_id: str
    contract: Contract
    signed_qty_delta: Decimal

    @classmethod
    def from_dict(cls, raw: Any, *, where: str) -> ExecutionRevision:
        obj = _strict_object(
            raw,
            where=where,
            required=(
                "execution_id",
                "correction_family_id",
                "correction_revision",
                "correction_action",
                "source_sequence",
                "occurred_at",
                "account",
                "signal_id",
                "tranche_id",
                "contract",
                "signed_qty_delta",
            ),
        )
        return cls(
            execution_id=_nonempty_string(
                obj["execution_id"], where=f"{where}.execution_id"
            ),
            correction_family_id=_nonempty_string(
                obj["correction_family_id"], where=f"{where}.correction_family_id"
            ),
            correction_revision=_integer(
                obj["correction_revision"],
                where=f"{where}.correction_revision",
                minimum=1,
            ),
            correction_action=_enum(
                CorrectionAction,
                obj["correction_action"],
                where=f"{where}.correction_action",
            ),
            source_sequence=_integer(
                obj["source_sequence"], where=f"{where}.source_sequence", minimum=1
            ),
            occurred_at=_timestamp(obj["occurred_at"], where=f"{where}.occurred_at"),
            account=_nonempty_string(obj["account"], where=f"{where}.account"),
            signal_id=_nonempty_string(obj["signal_id"], where=f"{where}.signal_id"),
            tranche_id=_nonempty_string(obj["tranche_id"], where=f"{where}.tranche_id"),
            contract=Contract.from_dict(obj["contract"], where=f"{where}.contract"),
            signed_qty_delta=_decimal(
                obj["signed_qty_delta"], where=f"{where}.signed_qty_delta"
            ),
        )

    def family_identity(self) -> tuple[Any, ...]:
        return (self.account, self.signal_id, self.tranche_id, self.contract)


@dataclass(frozen=True)
class AggregatePosition:
    snapshot_id: str
    source_sequence: int
    observed_at: datetime
    account: str
    contract: Contract
    signed_qty: Decimal

    @classmethod
    def from_dict(cls, raw: Any, *, where: str) -> AggregatePosition:
        obj = _strict_object(
            raw,
            where=where,
            required=(
                "snapshot_id",
                "source_sequence",
                "observed_at",
                "account",
                "contract",
                "signed_qty",
            ),
        )
        return cls(
            snapshot_id=_nonempty_string(
                obj["snapshot_id"], where=f"{where}.snapshot_id"
            ),
            source_sequence=_integer(
                obj["source_sequence"], where=f"{where}.source_sequence", minimum=1
            ),
            observed_at=_timestamp(obj["observed_at"], where=f"{where}.observed_at"),
            account=_nonempty_string(obj["account"], where=f"{where}.account"),
            contract=Contract.from_dict(obj["contract"], where=f"{where}.contract"),
            signed_qty=_decimal(obj["signed_qty"], where=f"{where}.signed_qty"),
        )


@dataclass(frozen=True)
class Manifest:
    run: RunSpec
    producer_receipts: tuple[SourceReceipt, ...]
    execution_receipt: SourceReceipt
    baseline_receipt: SourceReceipt
    position_receipt: SourceReceipt
    obligation_revisions: tuple[ObligationRevision, ...]
    position_baselines: tuple[PositionBaseline, ...]
    execution_revisions: tuple[ExecutionRevision, ...]
    aggregate_positions: tuple[AggregatePosition, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> Manifest:
        obj = _strict_object(
            raw,
            where="manifest",
            required=(
                "schema_version",
                "run",
                "producer_receipts",
                "execution_receipt",
                "baseline_receipt",
                "position_receipt",
                "obligation_revisions",
                "position_baselines",
                "execution_revisions",
                "aggregate_positions",
            ),
        )
        if _integer(obj["schema_version"], where="schema_version") != SCHEMA_VERSION:
            raise ManifestError(f"schema_version must be {SCHEMA_VERSION}")

        def parse_array(name: str, parser: Any) -> tuple[Any, ...]:
            values = obj[name]
            if not isinstance(values, list):
                raise ManifestError(f"{name} must be an array")
            return tuple(
                parser(item, where=f"{name}[{idx}]") for idx, item in enumerate(values)
            )

        return cls(
            run=RunSpec.from_dict(obj["run"]),
            producer_receipts=parse_array("producer_receipts", SourceReceipt.from_dict),
            execution_receipt=SourceReceipt.from_dict(
                obj["execution_receipt"], where="execution_receipt"
            ),
            baseline_receipt=SourceReceipt.from_dict(
                obj["baseline_receipt"], where="baseline_receipt"
            ),
            position_receipt=SourceReceipt.from_dict(
                obj["position_receipt"], where="position_receipt"
            ),
            obligation_revisions=parse_array(
                "obligation_revisions", ObligationRevision.from_dict
            ),
            position_baselines=parse_array(
                "position_baselines", PositionBaseline.from_dict
            ),
            execution_revisions=parse_array(
                "execution_revisions", ExecutionRevision.from_dict
            ),
            aggregate_positions=parse_array(
                "aggregate_positions", AggregatePosition.from_dict
            ),
        )


@dataclass(frozen=True)
class ObligationResult:
    event_id: str
    account: str
    signal_id: str
    tranche_id: str
    strategy: str
    contract: str
    expected_flat_by: datetime
    state: ControlState
    signed_residual_qty: Decimal | None
    issues: tuple[Issue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "account": self.account,
            "signal_id": self.signal_id,
            "tranche_id": self.tranche_id,
            "strategy": self.strategy,
            "contract": self.contract,
            "expected_flat_by": timestamp_text(self.expected_flat_by),
            "state": self.state.value,
            "signed_residual_qty": decimal_text(self.signed_residual_qty),
            "issues": [issue.to_dict() for issue in sorted(self.issues)],
        }


@dataclass(frozen=True)
class EvaluationReport:
    schema_version: int
    run_id: str
    run_mode: RunMode
    operationally_authoritative: bool
    as_of: datetime
    session_date: date
    state: ControlState
    headline: str
    what_changed: tuple[str, ...]
    required_action: tuple[str, ...]
    obligations: tuple[ObligationResult, ...]
    issues: tuple[Issue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_mode": self.run_mode.value,
            "operationally_authoritative": self.operationally_authoritative,
            "as_of": timestamp_text(self.as_of),
            "session_date": self.session_date.isoformat(),
            "state": self.state.value,
            "headline": self.headline,
            "what_changed": list(self.what_changed),
            "required_action": list(self.required_action),
            "obligations": [item.to_dict() for item in self.obligations],
            "issues": [issue.to_dict() for issue in sorted(self.issues)],
        }


def _receipt_issues(
    receipt: SourceReceipt,
    *,
    expected_ids: set[str],
    expected_sequences: set[int],
    as_of: datetime,
    scope: str,
    incomplete_reason: ReasonCode,
    stale_reason: ReasonCode,
    set_reason: ReasonCode,
    zero_reason: ReasonCode,
) -> list[Issue]:
    issues: list[Issue] = []
    if not receipt.complete:
        issues.append(
            Issue(
                incomplete_reason,
                f"{scope} receipt {receipt.receipt_id} is incomplete",
                scope,
            )
        )
    if receipt.complete_through < as_of:
        issues.append(
            Issue(
                stale_reason,
                f"{scope} complete_through={timestamp_text(receipt.complete_through)} precedes "
                f"as_of={timestamp_text(as_of)}",
                scope,
            )
        )
    if set(receipt.event_ids) != expected_ids or receipt.event_count != len(
        expected_ids
    ):
        issues.append(
            Issue(
                set_reason,
                f"{scope} receipt declares {receipt.event_count} event(s), but manifest has "
                f"{len(expected_ids)} unique event(s)",
                scope,
            )
        )
    if receipt.zero_events != (len(expected_ids) == 0):
        issues.append(
            Issue(
                zero_reason,
                f"{scope} zero_events flag disagrees with its event set",
                scope,
            )
        )
    first, last = receipt.source_sequence_first, receipt.source_sequence_last
    expected_range = tuple(range(first, last + 1)) if first <= last else ()
    sequences = set(receipt.source_sequences)
    if receipt.source_sequences != expected_range or not expected_sequences.issubset(
        sequences
    ):
        issues.append(
            Issue(
                ReasonCode.SOURCE_SEQUENCE_INCOMPLETE,
                f"{scope} source sequence coverage is incomplete or non-contiguous",
                scope,
            )
        )
    return issues


def _dedupe_exact(records: Sequence[Any], identity: Any) -> tuple[list[Any], bool]:
    """Deduplicate exact records by identity; report conflicting duplicates."""
    unique: dict[Any, Any] = {}
    conflict = False
    for record in records:
        key = identity(record)
        previous = unique.get(key)
        if previous is None:
            unique[key] = record
        elif previous != record:
            conflict = True
    return list(unique.values()), conflict


def _sequence_collision_issues(
    records: Sequence[Any],
    *,
    event_identity: Any,
    source_sequence: Any,
    scope: str,
) -> list[Issue]:
    """Detect two distinct append-log events claiming one source sequence."""
    sequence_events: dict[int, set[Any]] = {}
    for record in records:
        sequence_events.setdefault(source_sequence(record), set()).add(
            event_identity(record)
        )
    collisions = sorted(
        sequence
        for sequence, event_ids in sequence_events.items()
        if len(event_ids) > 1
    )
    if not collisions:
        return []
    return [
        Issue(
            ReasonCode.SOURCE_SEQUENCE_INCOMPLETE,
            f"distinct events collide on source sequence(s): {', '.join(map(str, collisions))}",
            scope,
        )
    ]


CanonicalContract = tuple[str, ...]


class ContractResolver:
    """Resolve exact conIds, or a fallback only when it names one contract."""

    def __init__(self, account_contracts: Iterable[tuple[str, Contract]]):
        self._fallback_conids: dict[tuple[str, tuple[str, ...]], set[int]] = {}
        self._conid_fallbacks: dict[tuple[str, int], set[tuple[str, ...]]] = {}
        for account, contract in account_contracts:
            fallback = contract.fallback
            if contract.con_id is not None:
                self._fallback_conids.setdefault((account, fallback), set()).add(
                    contract.con_id
                )
                self._conid_fallbacks.setdefault((account, contract.con_id), set()).add(
                    fallback
                )

    def resolve(
        self, account: str, contract: Contract
    ) -> tuple[CanonicalContract | None, Issue | None]:
        if contract.con_id is not None:
            fingerprints = self._conid_fallbacks.get((account, contract.con_id), set())
            if len(fingerprints) > 1:
                return None, Issue(
                    ReasonCode.CONTRACT_IDENTITY_CONFLICT,
                    f"account={account} conId={contract.con_id} has conflicting normalized identities",
                    f"account:{account}:conId:{contract.con_id}",
                )
            return ("CONID", account, str(contract.con_id)), None
        candidates = self._fallback_conids.get((account, contract.fallback), set())
        if len(candidates) > 1:
            return None, Issue(
                ReasonCode.AMBIGUOUS_CONTRACT_FALLBACK,
                f"account={account} fallback={contract.fallback!r} maps to multiple conIds",
                f"account:{account}:fallback:{'|'.join(contract.fallback)}",
            )
        if len(candidates) == 1:
            return ("CONID", account, str(next(iter(candidates)))), None
        return ("FALLBACK", account, *contract.fallback), None


def _latest_obligations(
    revisions: Sequence[ObligationRevision],
) -> tuple[dict[str, ObligationRevision], list[Issue]]:
    grouped: dict[str, list[ObligationRevision]] = {}
    for row in revisions:
        grouped.setdefault(row.event_id, []).append(row)
    latest: dict[str, ObligationRevision] = {}
    issues: list[Issue] = []
    for event_id, rows in sorted(grouped.items()):
        rows, conflict = _dedupe_exact(rows, lambda row: row.revision)
        scope = f"obligation:{event_id}"
        if conflict:
            issues.append(
                Issue(
                    ReasonCode.CONFLICTING_DUPLICATE_REVISION,
                    "same revision has conflicting payloads",
                    scope,
                )
            )
            continue
        rows.sort(key=lambda row: row.revision)
        if [row.revision for row in rows] != list(range(1, rows[-1].revision + 1)):
            issues.append(
                Issue(
                    ReasonCode.OBLIGATION_REVISION_GAP,
                    "revision sequence is not contiguous",
                    scope,
                )
            )
            continue
        identity = rows[0].immutable_identity()
        if any(row.immutable_identity() != identity for row in rows[1:]):
            issues.append(
                Issue(
                    ReasonCode.IMMUTABLE_IDENTITY_CHANGED,
                    "immutable identity changed across revisions",
                    scope,
                )
            )
            continue
        if any(
            later.recorded_at <= earlier.recorded_at
            or later.source_sequence <= earlier.source_sequence
            for earlier, later in pairwise(rows)
        ):
            issues.append(
                Issue(
                    ReasonCode.NON_MONOTONIC_REVISION,
                    "recorded_at and source_sequence must increase with every revision",
                    scope,
                )
            )
            continue
        latest[event_id] = rows[-1]
    return latest, issues


def _active_executions(
    revisions: Sequence[ExecutionRevision],
) -> tuple[list[ExecutionRevision], list[Issue], set[tuple[str, str, str]]]:
    by_exec, duplicate_conflict = _dedupe_exact(revisions, lambda row: row.execution_id)
    issues: list[Issue] = []
    affected: set[tuple[str, str, str]] = set()
    if duplicate_conflict:
        issues.append(
            Issue(
                ReasonCode.CONFLICTING_DUPLICATE_EXECUTION,
                "one execution_id has conflicting payloads",
                "executions",
            )
        )
        # Conflicting duplicate IDs cannot safely be localized before choosing
        # a payload, so every evaluated obligation must fail closed.
        affected.add(("*", "*", "*"))

    grouped: dict[str, list[ExecutionRevision]] = {}
    for row in by_exec:
        grouped.setdefault(row.correction_family_id, []).append(row)
    active: list[ExecutionRevision] = []
    for family_id, rows in sorted(grouped.items()):
        scope = f"correction-family:{family_id}"
        rows_by_revision: dict[int, list[ExecutionRevision]] = {}
        for row in rows:
            rows_by_revision.setdefault(row.correction_revision, []).append(row)
        if any(len(group) != 1 for group in rows_by_revision.values()):
            issues.append(
                Issue(
                    ReasonCode.AMBIGUOUS_CORRECTION_FAMILY,
                    "multiple executions claim one correction revision",
                    scope,
                )
            )
            for row in rows:
                affected.add((row.account, row.signal_id, row.tranche_id))
            continue
        ordered = [rows_by_revision[idx][0] for idx in sorted(rows_by_revision)]
        if [row.correction_revision for row in ordered] != list(
            range(1, ordered[-1].correction_revision + 1)
        ):
            issues.append(
                Issue(
                    ReasonCode.CORRECTION_REVISION_GAP,
                    "correction revisions are not contiguous",
                    scope,
                )
            )
            for row in ordered:
                affected.add((row.account, row.signal_id, row.tranche_id))
            continue
        identity = ordered[0].family_identity()
        if any(row.family_identity() != identity for row in ordered[1:]):
            issues.append(
                Issue(
                    ReasonCode.AMBIGUOUS_CORRECTION_FAMILY,
                    "correction family changes allocation identity",
                    scope,
                )
            )
            for row in ordered:
                affected.add((row.account, row.signal_id, row.tranche_id))
            continue
        if ordered[0].correction_action is not CorrectionAction.APPLY or any(
            row.correction_action is CorrectionAction.VOID
            and row.signed_qty_delta != ZERO
            for row in ordered
        ):
            issues.append(
                Issue(
                    ReasonCode.AMBIGUOUS_CORRECTION_FAMILY,
                    "correction family must begin with APPLY and every VOID must carry zero quantity",
                    scope,
                )
            )
            for row in ordered:
                affected.add((row.account, row.signal_id, row.tranche_id))
            continue
        if any(
            later.source_sequence <= earlier.source_sequence
            or later.occurred_at < earlier.occurred_at
            for earlier, later in pairwise(ordered)
        ):
            issues.append(
                Issue(
                    ReasonCode.AMBIGUOUS_CORRECTION_FAMILY,
                    "correction revisions are not monotonic in sequence/time",
                    scope,
                )
            )
            for row in ordered:
                affected.add((row.account, row.signal_id, row.tranche_id))
            continue
        latest = ordered[-1]
        if latest.correction_action is CorrectionAction.APPLY:
            active.append(latest)
    return active, issues, affected


def evaluate(manifest: Manifest) -> EvaluationReport:
    run = manifest.run
    global_issues: list[Issue] = []

    latest_obligations, revision_issues = _latest_obligations(
        manifest.obligation_revisions
    )
    global_issues.extend(revision_issues)
    for row in manifest.obligation_revisions:
        if row.session_date != run.session_date:
            global_issues.append(
                Issue(
                    ReasonCode.SOURCE_SESSION_MISMATCH,
                    f"obligation event session {row.session_date.isoformat()} != run session "
                    f"{run.session_date.isoformat()}",
                    f"obligation:{row.event_id}",
                )
            )
        if row.recorded_at > run.as_of:
            global_issues.append(
                Issue(
                    ReasonCode.FUTURE_SOURCE_EVENT,
                    "obligation revision was recorded after run as_of",
                    f"obligation:{row.event_id}",
                )
            )
    for row in manifest.execution_revisions:
        if row.occurred_at > run.as_of:
            global_issues.append(
                Issue(
                    ReasonCode.FUTURE_SOURCE_EVENT,
                    "execution revision occurred after run as_of",
                    f"execution:{row.execution_id}",
                )
            )
    for producer in run.required_producers:
        producer_rows = [
            row for row in manifest.obligation_revisions if row.producer_id == producer
        ]
        global_issues.extend(
            _sequence_collision_issues(
                producer_rows,
                event_identity=lambda row: (row.event_id, row.revision),
                source_sequence=lambda row: row.source_sequence,
                scope=f"producer:{producer}",
            )
        )
    global_issues.extend(
        _sequence_collision_issues(
            manifest.execution_revisions,
            event_identity=lambda row: row.execution_id,
            source_sequence=lambda row: row.source_sequence,
            scope="execution-source",
        )
    )
    global_issues.extend(
        _sequence_collision_issues(
            manifest.position_baselines,
            event_identity=lambda row: row.baseline_id,
            source_sequence=lambda row: row.source_sequence,
            scope="baseline-source",
        )
    )
    global_issues.extend(
        _sequence_collision_issues(
            manifest.aggregate_positions,
            event_identity=lambda row: row.snapshot_id,
            source_sequence=lambda row: row.source_sequence,
            scope="position-source",
        )
    )

    # A required producer proves both that it ran (including the zero-event
    # case) and that its complete event and sequence sets were supplied.
    receipts_by_producer: dict[str, list[SourceReceipt]] = {}
    for receipt in manifest.producer_receipts:
        receipts_by_producer.setdefault(receipt.source_id, []).append(receipt)
    for producer in run.required_producers:
        rows = receipts_by_producer.get(producer, [])
        scope = f"producer:{producer}"
        if not rows:
            global_issues.append(
                Issue(
                    ReasonCode.MISSING_REQUIRED_PRODUCER_RECEIPT,
                    "required producer has no receipt",
                    scope,
                )
            )
            continue
        if len(rows) != 1:
            global_issues.append(
                Issue(
                    ReasonCode.DUPLICATE_PRODUCER_RECEIPT,
                    "required producer has multiple receipts",
                    scope,
                )
            )
            continue
        receipt = rows[0]
        if receipt.session_date != run.session_date:
            global_issues.append(
                Issue(
                    ReasonCode.SOURCE_SESSION_MISMATCH,
                    f"receipt session {receipt.session_date.isoformat()} != run session {run.session_date.isoformat()}",
                    scope,
                )
            )
        producer_rows = [
            row for row in manifest.obligation_revisions if row.producer_id == producer
        ]
        expected_ids = {row.event_id for row in producer_rows}
        expected_sequences = {row.source_sequence for row in producer_rows}
        global_issues.extend(
            _receipt_issues(
                receipt,
                expected_ids=expected_ids,
                expected_sequences=expected_sequences,
                as_of=run.as_of,
                scope=scope,
                incomplete_reason=ReasonCode.INCOMPLETE_PRODUCER_RECEIPT,
                stale_reason=ReasonCode.STALE_PRODUCER_RECEIPT,
                set_reason=ReasonCode.RECEIPT_EVENT_SET_MISMATCH,
                zero_reason=ReasonCode.RECEIPT_ZERO_OBLIGATION_MISMATCH,
            )
        )

    # Events from undeclared producers are never silently trusted.
    undeclared = sorted(
        {row.producer_id for row in manifest.obligation_revisions}
        - set(run.required_producers)
    )
    for producer in undeclared:
        global_issues.append(
            Issue(
                ReasonCode.MISSING_REQUIRED_PRODUCER_RECEIPT,
                "obligation producer is not declared in required_producers",
                f"producer:{producer}",
            )
        )

    extra_receipts = sorted(set(receipts_by_producer) - set(run.required_producers))
    for producer in extra_receipts:
        global_issues.append(
            Issue(
                ReasonCode.MISSING_REQUIRED_PRODUCER_RECEIPT,
                "producer receipt is not declared in required_producers",
                f"producer:{producer}",
            )
        )

    for receipt, scope in (
        (manifest.execution_receipt, "execution-source"),
        (manifest.baseline_receipt, "baseline-source"),
        (manifest.position_receipt, "position-source"),
    ):
        if receipt.session_date != run.session_date:
            global_issues.append(
                Issue(
                    ReasonCode.SOURCE_SESSION_MISMATCH,
                    f"receipt session {receipt.session_date.isoformat()} != run session {run.session_date.isoformat()}",
                    scope,
                )
            )

    execution_ids = {row.execution_id for row in manifest.execution_revisions}
    execution_sequences = {row.source_sequence for row in manifest.execution_revisions}
    global_issues.extend(
        _receipt_issues(
            manifest.execution_receipt,
            expected_ids=execution_ids,
            expected_sequences=execution_sequences,
            as_of=run.as_of,
            scope="execution-source",
            incomplete_reason=ReasonCode.INCOMPLETE_EXECUTION_SOURCE,
            stale_reason=ReasonCode.STALE_EXECUTION_SOURCE,
            set_reason=ReasonCode.EXECUTION_EVENT_SET_MISMATCH,
            zero_reason=ReasonCode.EXECUTION_ZERO_EVENT_MISMATCH,
        )
    )

    baseline_ids = {row.baseline_id for row in manifest.position_baselines}
    baseline_sequences = {row.source_sequence for row in manifest.position_baselines}
    global_issues.extend(
        _receipt_issues(
            manifest.baseline_receipt,
            expected_ids=baseline_ids,
            expected_sequences=baseline_sequences,
            as_of=run.as_of,
            scope="baseline-source",
            incomplete_reason=ReasonCode.INCOMPLETE_BASELINE_SOURCE,
            stale_reason=ReasonCode.STALE_BASELINE_SOURCE,
            set_reason=ReasonCode.BASELINE_EVENT_SET_MISMATCH,
            zero_reason=ReasonCode.BASELINE_ZERO_EVENT_MISMATCH,
        )
    )

    position_ids = {row.snapshot_id for row in manifest.aggregate_positions}
    position_sequences = {row.source_sequence for row in manifest.aggregate_positions}
    global_issues.extend(
        _receipt_issues(
            manifest.position_receipt,
            expected_ids=position_ids,
            expected_sequences=position_sequences,
            as_of=run.as_of,
            scope="position-source",
            incomplete_reason=ReasonCode.INCOMPLETE_POSITION_SOURCE,
            stale_reason=ReasonCode.STALE_POSITION_SOURCE,
            set_reason=ReasonCode.POSITION_EVENT_SET_MISMATCH,
            zero_reason=ReasonCode.POSITION_EVENT_SET_MISMATCH,
        )
    )

    active_execs, correction_issues, correction_affected = _active_executions(
        manifest.execution_revisions
    )
    global_issues.extend(correction_issues)

    all_account_contracts: list[tuple[str, Contract]] = []
    all_account_contracts.extend(
        (row.account, row.contract) for row in manifest.obligation_revisions
    )
    all_account_contracts.extend(
        (row.account, row.contract) for row in manifest.position_baselines
    )
    all_account_contracts.extend(
        (row.account, row.contract) for row in manifest.execution_revisions
    )
    all_account_contracts.extend(
        (row.account, row.contract) for row in manifest.aggregate_positions
    )
    resolver = ContractResolver(all_account_contracts)

    resolution_issues: list[Issue] = []

    def resolve(account: str, contract: Contract) -> CanonicalContract | None:
        key, issue = resolver.resolve(account, contract)
        if issue is not None and issue not in resolution_issues:
            resolution_issues.append(issue)
        return key

    # Resolve every record up front so a latent conId/fallback conflict cannot
    # be hidden merely because one row is not the first lookup.
    for account, contract in all_account_contracts:
        resolve(account, contract)
    global_issues.extend(resolution_issues)

    baseline_by_allocation: dict[tuple[Any, ...], list[PositionBaseline]] = {}
    for baseline in manifest.position_baselines:
        contract_key = resolve(baseline.account, baseline.contract)
        if contract_key is None:
            continue
        key = (baseline.account, baseline.signal_id, baseline.tranche_id, contract_key)
        baseline_by_allocation.setdefault(key, []).append(baseline)

    exec_by_allocation: dict[tuple[Any, ...], list[ExecutionRevision]] = {}
    for execution in active_execs:
        if execution.occurred_at > run.as_of:
            continue
        contract_key = resolve(execution.account, execution.contract)
        if contract_key is None:
            continue
        key = (
            execution.account,
            execution.signal_id,
            execution.tranche_id,
            contract_key,
        )
        exec_by_allocation.setdefault(key, []).append(execution)

    aggregate_by_contract: dict[CanonicalContract, list[AggregatePosition]] = {}
    for position in manifest.aggregate_positions:
        contract_key = resolve(position.account, position.contract)
        if contract_key is None:
            continue
        aggregate_by_contract.setdefault(contract_key, []).append(position)

    # Compute allocation quantities once.  Executions at/before baseline time
    # are already represented by the baseline and must not be double-counted.
    allocation_qty: dict[tuple[Any, ...], Decimal] = {}
    allocation_issues: dict[tuple[Any, ...], list[Issue]] = {}
    for key in set(baseline_by_allocation) | set(exec_by_allocation):
        baselines = baseline_by_allocation.get(key, [])
        scope = f"allocation:{key[0]}:{key[1]}:{key[2]}"
        if not baselines:
            allocation_issues.setdefault(key, []).append(
                Issue(
                    ReasonCode.MISSING_BASELINE,
                    "execution allocation has no effective baseline",
                    scope,
                )
            )
            continue
        if len(baselines) != 1:
            allocation_issues.setdefault(key, []).append(
                Issue(
                    ReasonCode.DUPLICATE_BASELINE,
                    "allocation has more than one baseline",
                    scope,
                )
            )
            continue
        baseline = baselines[0]
        if baseline.effective_at > run.as_of:
            allocation_issues.setdefault(key, []).append(
                Issue(
                    ReasonCode.FUTURE_BASELINE,
                    "baseline effective_at is after run as_of",
                    scope,
                )
            )
            continue
        qty = baseline.signed_qty
        for execution in exec_by_allocation.get(key, []):
            if execution.occurred_at > baseline.effective_at:
                qty += execution.signed_qty_delta
        allocation_qty[key] = qty

    due_obligations = [
        row
        for row in latest_obligations.values()
        if row.active
        and row.session_date == run.session_date
        and row.expected_flat_by <= run.as_of
    ]
    due_by_allocation: dict[tuple[Any, ...], list[ObligationRevision]] = {}
    for row in due_obligations:
        contract_key = resolve(row.account, row.contract)
        if contract_key is not None:
            key = (row.account, row.signal_id, row.tranche_id, contract_key)
            due_by_allocation.setdefault(key, []).append(row)
    duplicate_due_allocations = {
        key for key, rows in due_by_allocation.items() if len(rows) > 1
    }
    for key in sorted(duplicate_due_allocations):
        global_issues.append(
            Issue(
                ReasonCode.DUPLICATE_ACTIVE_OBLIGATION,
                "multiple active due event_ids claim the same account/signal/tranche/contract",
                f"allocation:{key[0]}:{key[1]}:{key[2]}",
            )
        )
    touched_contracts: set[CanonicalContract] = set()
    for row in due_obligations:
        contract_key = resolve(row.account, row.contract)
        if contract_key is not None:
            touched_contracts.add(contract_key)

    contract_issues: dict[CanonicalContract, list[Issue]] = {}
    for contract_key in touched_contracts:
        positions = aggregate_by_contract.get(contract_key, [])
        scope = "contract:" + ":".join(contract_key[1:])
        if not positions:
            contract_issues.setdefault(contract_key, []).append(
                Issue(
                    ReasonCode.MISSING_POSITION_SNAPSHOT,
                    "touched contract has no aggregate position",
                    scope,
                )
            )
            continue
        if len(positions) != 1:
            contract_issues.setdefault(contract_key, []).append(
                Issue(
                    ReasonCode.DUPLICATE_POSITION_SNAPSHOT,
                    "touched contract has multiple snapshots",
                    scope,
                )
            )
            continue
        position = positions[0]
        age = (run.as_of - position.observed_at).total_seconds()
        if age < 0 or age > run.max_position_age_seconds:
            contract_issues.setdefault(contract_key, []).append(
                Issue(
                    ReasonCode.STALE_POSITION_SNAPSHOT,
                    f"position age is {int(age)}s; allowed range is 0..{run.max_position_age_seconds}s",
                    scope,
                )
            )
            continue
        evidence_times = [
            baseline.effective_at
            for key, rows in baseline_by_allocation.items()
            if key[3] == contract_key
            for baseline in rows
            if baseline.effective_at <= run.as_of
        ]
        evidence_times.extend(
            execution.occurred_at
            for key, rows in exec_by_allocation.items()
            if key[3] == contract_key
            for execution in rows
            if execution.occurred_at <= run.as_of
        )
        if evidence_times and position.observed_at < max(evidence_times):
            contract_issues.setdefault(contract_key, []).append(
                Issue(
                    ReasonCode.POSITION_SNAPSHOT_PRECEDES_EVIDENCE,
                    "aggregate position was observed before the latest applied baseline/execution evidence",
                    scope,
                )
            )
            continue
        attributed = sum(
            (qty for key, qty in allocation_qty.items() if key[3] == contract_key),
            ZERO,
        )
        # Any unresolved allocation on the touched contract also invalidates
        # the aggregate proof, even if the computable subtotal happens to fit.
        unresolved = [key for key in allocation_issues if key[3] == contract_key]
        if unresolved or attributed != position.signed_qty:
            contract_issues.setdefault(contract_key, []).append(
                Issue(
                    ReasonCode.AGGREGATE_POSITION_MISMATCH,
                    f"attributed signed qty {decimal_text(attributed)} != aggregate signed qty "
                    f"{decimal_text(position.signed_qty)}",
                    scope,
                )
            )

    hard_global_reasons = {
        ReasonCode.MISSING_REQUIRED_PRODUCER_RECEIPT,
        ReasonCode.DUPLICATE_PRODUCER_RECEIPT,
        ReasonCode.INCOMPLETE_PRODUCER_RECEIPT,
        ReasonCode.STALE_PRODUCER_RECEIPT,
        ReasonCode.RECEIPT_EVENT_SET_MISMATCH,
        ReasonCode.RECEIPT_ZERO_OBLIGATION_MISMATCH,
        ReasonCode.SOURCE_SESSION_MISMATCH,
        ReasonCode.SOURCE_SEQUENCE_INCOMPLETE,
        ReasonCode.OBLIGATION_REVISION_GAP,
        ReasonCode.CONFLICTING_DUPLICATE_REVISION,
        ReasonCode.IMMUTABLE_IDENTITY_CHANGED,
        ReasonCode.NON_MONOTONIC_REVISION,
        ReasonCode.INCOMPLETE_EXECUTION_SOURCE,
        ReasonCode.STALE_EXECUTION_SOURCE,
        ReasonCode.EXECUTION_EVENT_SET_MISMATCH,
        ReasonCode.EXECUTION_ZERO_EVENT_MISMATCH,
        ReasonCode.INCOMPLETE_BASELINE_SOURCE,
        ReasonCode.STALE_BASELINE_SOURCE,
        ReasonCode.BASELINE_EVENT_SET_MISMATCH,
        ReasonCode.BASELINE_ZERO_EVENT_MISMATCH,
        ReasonCode.INCOMPLETE_POSITION_SOURCE,
        ReasonCode.STALE_POSITION_SOURCE,
        ReasonCode.POSITION_EVENT_SET_MISMATCH,
        ReasonCode.FUTURE_SOURCE_EVENT,
        ReasonCode.CONFLICTING_DUPLICATE_EXECUTION,
    }
    hard_global = [
        issue for issue in global_issues if issue.reason in hard_global_reasons
    ]

    results: list[ObligationResult] = []
    for event_id, row in sorted(latest_obligations.items()):
        base_kwargs = {
            "event_id": event_id,
            "account": row.account,
            "signal_id": row.signal_id,
            "tranche_id": row.tranche_id,
            "strategy": row.strategy,
            "contract": row.contract.label(),
            "expected_flat_by": row.expected_flat_by,
        }
        if not row.active:
            results.append(
                ObligationResult(
                    **base_kwargs,
                    state=ControlState.NOT_SCHEDULED,
                    signed_residual_qty=None,
                    issues=(
                        Issue(
                            ReasonCode.OBLIGATION_CANCELLED,
                            "latest revision deactivates obligation",
                            f"obligation:{event_id}",
                        ),
                    ),
                )
            )
            continue
        if row.session_date != run.session_date or row.expected_flat_by > run.as_of:
            results.append(
                ObligationResult(
                    **base_kwargs,
                    state=ControlState.NOT_SCHEDULED,
                    signed_residual_qty=None,
                    issues=(
                        Issue(
                            ReasonCode.OBLIGATION_NOT_DUE,
                            "obligation is not due in this run",
                            f"obligation:{event_id}",
                        ),
                    ),
                )
            )
            continue

        item_issues = list(hard_global)
        contract_key = resolve(row.account, row.contract)
        if contract_key is None:
            item_issues.extend(resolution_issues)
            qty = None
        else:
            allocation_key = (row.account, row.signal_id, row.tranche_id, contract_key)
            if allocation_key in duplicate_due_allocations:
                item_issues.append(
                    Issue(
                        ReasonCode.DUPLICATE_ACTIVE_OBLIGATION,
                        "multiple active due event_ids claim this allocation",
                        f"obligation:{event_id}",
                    )
                )
            if allocation_key not in baseline_by_allocation:
                item_issues.append(
                    Issue(
                        ReasonCode.MISSING_BASELINE,
                        "due obligation has no effective signal/tranche baseline",
                        f"obligation:{event_id}",
                    )
                )
            item_issues.extend(allocation_issues.get(allocation_key, []))
            item_issues.extend(contract_issues.get(contract_key, []))
            if ("*", "*", "*") in correction_affected or (
                row.account,
                row.signal_id,
                row.tranche_id,
            ) in correction_affected:
                item_issues.extend(correction_issues)
            qty = allocation_qty.get(allocation_key)

        reversed_position = False
        if not item_issues and qty is not None:
            baseline_rows = baseline_by_allocation.get(
                (row.account, row.signal_id, row.tranche_id, contract_key), []
            )
            baseline_qty = (
                baseline_rows[0].signed_qty if len(baseline_rows) == 1 else None
            )
            reversed_position = (
                baseline_qty is not None
                and baseline_qty != ZERO
                and qty != ZERO
                and ((baseline_qty > ZERO) != (qty > ZERO))
            )

        if item_issues or qty is None:
            state = ControlState.UNKNOWN
        elif qty > ZERO:
            state = ControlState.ALERT
            item_issues.append(
                Issue(
                    ReasonCode.POSITION_REVERSED
                    if reversed_position
                    else ReasonCode.RESIDUAL_POSITION_LONG,
                    f"signed residual is {decimal_text(qty)}",
                    f"obligation:{event_id}",
                )
            )
        elif qty < ZERO:
            reason = (
                ReasonCode.POSITION_REVERSED
                if reversed_position
                else ReasonCode.RESIDUAL_POSITION_SHORT
            )
            state = ControlState.ALERT
            item_issues.append(
                Issue(
                    reason,
                    f"signed residual is {decimal_text(qty)}",
                    f"obligation:{event_id}",
                )
            )
        else:
            state = ControlState.CLEAR

        results.append(
            ObligationResult(
                **base_kwargs,
                state=state,
                signed_residual_qty=qty,
                issues=tuple(sorted(set(item_issues))),
            )
        )

    due_results = [
        item for item in results if item.state is not ControlState.NOT_SCHEDULED
    ]
    if run.run_mode is RunMode.DISABLED:
        overall = ControlState.NOT_SCHEDULED
        global_issues.append(
            Issue(ReasonCode.RUN_DISABLED, "control run_mode is DISABLED")
        )
    elif any(item.state is ControlState.ALERT for item in due_results):
        overall = ControlState.ALERT
    elif global_issues or any(
        item.state is ControlState.UNKNOWN for item in due_results
    ):
        overall = ControlState.UNKNOWN
    elif not due_results:
        if global_issues:
            overall = ControlState.UNKNOWN
        else:
            overall = ControlState.NOT_SCHEDULED
            global_issues.append(
                Issue(
                    ReasonCode.NO_DUE_OBLIGATIONS,
                    "no active obligation is due in this run",
                )
            )
    else:
        overall = ControlState.CLEAR

    if run.run_mode is not RunMode.LIVE or not run.operationally_authoritative:
        if overall is ControlState.CLEAR:
            overall = ControlState.UNKNOWN
        if run.run_mode is not RunMode.DISABLED:
            global_issues.append(
                Issue(
                    ReasonCode.NON_AUTHORITATIVE_RUN,
                    "offline/shadow evidence cannot provide operational clearance",
                )
            )

    non_authoritative = (
        run.run_mode is not RunMode.LIVE or not run.operationally_authoritative
    )
    if non_authoritative:
        headline = f"{run.run_mode.value} / NON-AUTHORITATIVE — DO NOT TREAT AS OPERATIONAL CLEARANCE"
    else:
        headline = f"LIVE / OPERATIONALLY AUTHORITATIVE — {overall.value}"

    due_count = len(due_obligations)
    alert_count = sum(item.state is ControlState.ALERT for item in results)
    unknown_count = sum(item.state is ControlState.UNKNOWN for item in results)
    clear_count = sum(item.state is ControlState.CLEAR for item in results)
    what_changed = (
        f"Evaluated {due_count} due expected-flat obligation(s) for {run.session_date.isoformat()}.",
        f"Result mix: {alert_count} ALERT, {unknown_count} UNKNOWN, {clear_count} CLEAR.",
        f"Recorded {len(global_issues)} run-level issue(s).",
    )
    if overall is ControlState.ALERT:
        required_action = (
            "Investigate every ALERT against the broker and execution audit trail immediately.",
            "Do not place corrective trades from this report; execution requires a separately authorized workflow.",
        )
    elif overall is ControlState.UNKNOWN:
        required_action = (
            "Resolve every UNKNOWN data-completeness or identity issue before drawing a flatness conclusion.",
            "Do not interpret this report as operational clearance.",
        )
    elif overall is ControlState.NOT_SCHEDULED:
        required_action = (
            "No operational flatness conclusion was scheduled for this run.",
            "Confirm the intended run mode and producer receipts before relying on the control.",
        )
    else:
        required_action = (
            "No residual position was found for the due obligations in this authoritative snapshot.",
            "Retain the artifacts and continue normal independent broker review.",
        )

    return EvaluationReport(
        schema_version=SCHEMA_VERSION,
        run_id=run.run_id,
        run_mode=run.run_mode,
        operationally_authoritative=run.operationally_authoritative,
        as_of=run.as_of,
        session_date=run.session_date,
        state=overall,
        headline=headline,
        what_changed=what_changed,
        required_action=required_action,
        obligations=tuple(results),
        issues=tuple(sorted(set(global_issues))),
    )


def render_json(report: EvaluationReport) -> str:
    return (
        json.dumps(report.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    )


def render_markdown(report: EvaluationReport) -> str:
    lines = [
        f"# Expected-Flat Control — {report.state.value}",
        "",
        f"> **{report.headline}**",
        "",
        "## What changed",
        "",
        *[f"- {item}" for item in report.what_changed],
        "",
        "## Required action",
        "",
        *[f"- {item}" for item in report.required_action],
        "",
        "## Run summary",
        "",
        f"- Run: `{report.run_id}`",
        f"- Mode: `{report.run_mode.value}`",
        f"- Operationally authoritative: `{str(report.operationally_authoritative).lower()}`",
        f"- Session: `{report.session_date.isoformat()}`",
        f"- As of: `{timestamp_text(report.as_of)}`",
        f"- State: **{report.state.value}**",
        "",
        "## Obligations",
        "",
    ]
    if not report.obligations:
        lines.append("No obligation events were supplied.")
    else:
        lines.extend(
            [
                "| State | Strategy | Account | Signal / tranche | Contract | Signed residual | Flat by |",
                "|---|---|---|---|---|---:|---|",
            ]
        )
        for item in report.obligations:
            lines.append(
                f"| {item.state.value} | {item.strategy} | {item.account} | "
                f"{item.signal_id} / {item.tranche_id} | {item.contract} | "
                f"{decimal_text(item.signed_residual_qty) or '—'} | {timestamp_text(item.expected_flat_by)} |"
            )
    lines.extend(["", "## Obligation findings", ""])
    item_findings = [
        (item, issue) for item in report.obligations for issue in item.issues
    ]
    if item_findings:
        lines.extend(
            f"- `{item.event_id}` — `{issue.reason.value}`: {issue.detail}"
            for item, issue in item_findings
        )
    else:
        lines.append("- None.")
    lines.extend(["", "## Run-level issues", ""])
    if report.issues:
        lines.extend(
            f"- `{issue.reason.value}` ({issue.scope}): {issue.detail}"
            for issue in report.issues
        )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "---",
            "This artifact is observational only. It cannot stage, place, cancel, or modify an order.",
            "",
        ]
    )
    return "\n".join(lines)


def render_html(report: EvaluationReport) -> str:
    esc = html.escape
    obligations = (
        "".join(
            "<tr>"
            f"<td>{esc(item.state.value)}</td>"
            f"<td>{esc(item.strategy)}</td>"
            f"<td>{esc(item.account)}</td>"
            f"<td>{esc(item.signal_id)} / {esc(item.tranche_id)}</td>"
            f"<td>{esc(item.contract)}</td>"
            f"<td class='num'>{esc(decimal_text(item.signed_residual_qty) or '—')}</td>"
            f"<td>{esc(timestamp_text(item.expected_flat_by))}</td>"
            "</tr>"
            for item in report.obligations
        )
        or "<tr><td colspan='7'>No obligation events were supplied.</td></tr>"
    )
    issues = (
        "".join(
            f"<li><code>{esc(issue.reason.value)}</code> ({esc(issue.scope)}): {esc(issue.detail)}</li>"
            for issue in report.issues
        )
        or "<li>None.</li>"
    )
    item_issues = (
        "".join(
            f"<li><code>{esc(item.event_id)}</code> — <code>{esc(issue.reason.value)}</code>: "
            f"{esc(issue.detail)}</li>"
            for item in report.obligations
            for issue in item.issues
        )
        or "<li>None.</li>"
    )
    changed = "".join(f"<li>{esc(item)}</li>" for item in report.what_changed)
    action = "".join(f"<li>{esc(item)}</li>" for item in report.required_action)
    banner_class = (
        "shadow"
        if not report.operationally_authoritative
        else report.state.value.lower()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Expected-Flat Control — {esc(report.state.value)}</title>
<style>
body {{ font: 15px/1.45 system-ui, sans-serif; color: #17202a; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; }}
.banner {{ border: 3px solid #7d3c98; background: #f4ecf7; padding: 1rem; font-weight: 800; font-size: 1.15rem; }}
.alert {{ border-color: #b03a2e; background: #fadbd8; }}
.clear {{ border-color: #1e8449; background: #d5f5e3; }}
section {{ margin-top: 1.5rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccd1d1; padding: .45rem; text-align: left; }}
th {{ background: #f2f3f4; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
code {{ white-space: nowrap; }}
</style>
</head>
<body>
<h1>Expected-Flat Control — {esc(report.state.value)}</h1>
<div class="banner {esc(banner_class)}">{esc(report.headline)}</div>
<section><h2>What changed</h2><ul>{changed}</ul></section>
<section><h2>Required action</h2><ul>{action}</ul></section>
<section><h2>Run summary</h2>
<p>Run <code>{esc(report.run_id)}</code> · mode <code>{esc(report.run_mode.value)}</code> · session
<code>{esc(report.session_date.isoformat())}</code> · as of <code>{esc(timestamp_text(report.as_of))}</code> ·
operationally authoritative <code>{str(report.operationally_authoritative).lower()}</code></p></section>
<section><h2>Obligations</h2><table><thead><tr><th>State</th><th>Strategy</th><th>Account</th>
<th>Signal / tranche</th><th>Contract</th><th>Signed residual</th><th>Flat by</th></tr></thead>
<tbody>{obligations}</tbody></table></section>
<section><h2>Obligation findings</h2><ul>{item_issues}</ul></section>
<section><h2>Run-level issues</h2><ul>{issues}</ul></section>
<hr><p><strong>Observational only.</strong> This artifact cannot stage, place, cancel, or modify an order.</p>
</body>
</html>
"""


_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]+")


def artifact_stem(report: EvaluationReport) -> str:
    safe_run_id = _SAFE_FILENAME.sub("_", report.run_id).strip("._") or "run"
    return f"expected_flat_{report.session_date.isoformat()}_{safe_run_id}"


def atomic_write_text(path: Path, content: str) -> None:
    """Write one artifact by fsync + same-directory atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        descriptor, temp_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temp_path = Path(temp_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def write_artifacts(report: EvaluationReport, output_dir: Path) -> dict[str, Path]:
    stem = artifact_stem(report)
    paths = {
        "json": Path(output_dir) / f"{stem}.json",
        "markdown": Path(output_dir) / f"{stem}.md",
        "html": Path(output_dir) / f"{stem}.html",
    }
    atomic_write_text(paths["json"], render_json(report))
    atomic_write_text(paths["markdown"], render_markdown(report))
    atomic_write_text(paths["html"], render_html(report))
    return paths


def load_manifest(path: Path) -> Manifest:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"could not read manifest {path}: {exc}") from exc
    return Manifest.from_dict(raw)
