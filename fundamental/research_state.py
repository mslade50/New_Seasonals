"""Versioned research-event and non-executable portfolio state helpers.

The daily screen does not know whether a thesis changed.  This module provides
the explicit bridge from append-only evidence/trigger observations to the
reversible DEEPEN/WATCH/PASS controls.  Missing state fails closed: it can
reduce automation, but it can never manufacture a review or portfolio action.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    CURRENT_ROOT,
    EVIDENCE_SCHEMA_VERSION,
    PORTFOLIO_SNAPSHOT_SCHEMA_VERSION,
    TRIGGER_SCHEMA_VERSION,
)


TRIGGER_STATE_PATH = CURRENT_ROOT / "triggers_latest.json"
EVIDENCE_STATE_PATH = CURRENT_ROOT / "thesis_evidence_latest.json"
RUN_MANIFEST_LATEST_PATH = CURRENT_ROOT / "fundamental_run_manifest_latest.json"
PORTFOLIO_SNAPSHOT_PATH = CURRENT_ROOT / "portfolio_snapshot_latest.json"

MATERIAL_CHANGE_LEVELS = {"THESIS_CHANGING", "DECISION_CHANGING"}
TRIGGER_KINDS = {"PROOF", "WARNING", "KILL", "REOPEN"}
TRIGGER_STATUSES = {"ARMED", "FIRED", "EXPIRED", "CANCELLED"}
EVIDENCE_DIRECTIONS = {"CONFIRM", "DISCONFIRM", "NEUTRAL"}
TRIGGER_COMPARATORS = {
    ">",
    ">=",
    "<",
    "<=",
    "==",
    "!=",
    "CROSS_ABOVE",
    "CROSS_BELOW",
}


def _read_json(path: Path) -> tuple[Any | None, str]:
    if not path.exists():
        return None, "MISSING"
    try:
        return json.loads(path.read_text(encoding="utf-8")), "AVAILABLE"
    except (OSError, json.JSONDecodeError):
        return None, "INVALID"


def _utc_timestamp(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return parsed if pd.notna(parsed) else None


def _cutoff(as_of: str | date) -> pd.Timestamp:
    parsed = pd.Timestamp(as_of)
    parsed = parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")
    return parsed.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)


def _source_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if pd.notna(result) else None


def _compare(comparator: str, observed: Any, threshold: Any, prior: Any = None) -> bool:
    op = str(comparator or "").strip().upper()
    aliases = {"GT": ">", "GTE": ">=", "LT": "<", "LTE": "<=", "EQ": "==", "NE": "!="}
    op = aliases.get(op, op)
    if op not in TRIGGER_COMPARATORS:
        return False
    observed_number = _number(observed)
    threshold_number = _number(threshold)
    prior_number = _number(prior)
    if observed_number is not None and threshold_number is not None:
        if op == ">":
            return observed_number > threshold_number
        if op == ">=":
            return observed_number >= threshold_number
        if op == "<":
            return observed_number < threshold_number
        if op == "<=":
            return observed_number <= threshold_number
        if op == "==":
            return observed_number == threshold_number
        if op == "!=":
            return observed_number != threshold_number
        if op == "CROSS_ABOVE":
            return prior_number is not None and prior_number < threshold_number <= observed_number
        if op == "CROSS_BELOW":
            return prior_number is not None and prior_number > threshold_number >= observed_number
    if op == "==":
        return str(observed) == str(threshold)
    if op == "!=":
        return str(observed) != str(threshold)
    return False


def evaluate_trigger_payload(payload: Any, *, as_of: str | date) -> dict[str, Any]:
    """Evaluate sourced trigger observations without guessing missing values."""
    schema_version = payload.get("schema_version") if isinstance(payload, dict) else None
    rows = payload.get("triggers") if isinstance(payload, dict) else None
    updated_at = _utc_timestamp(payload.get("updated_at")) if isinstance(payload, dict) else None
    if schema_version != TRIGGER_SCHEMA_VERSION or not isinstance(rows, list) or updated_at is None:
        return {
            "schema_version": schema_version,
            "status": "WRONG_SCHEMA" if schema_version != TRIGGER_SCHEMA_VERSION else "INVALID",
            "armed": 0,
            "fired": 0,
            "expired": 0,
            "cancelled": 0,
            "unevaluable": 0,
            "fired_tickers": [],
            "events": [],
        }
    cutoff = _cutoff(as_of)
    events: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        ticker = str(raw.get("ticker") or "").upper().strip()
        trigger_id = str(raw.get("trigger_id") or "").strip()
        kind = str(raw.get("kind") or "").upper().strip()
        metric = str(raw.get("metric") or "").strip()
        comparator = str(raw.get("comparator") or "").upper().strip()
        comparator = {"GT": ">", "GTE": ">=", "LT": "<", "LTE": "<=", "EQ": "==", "NE": "!="}.get(
            comparator, comparator
        )
        sources = _source_ids(raw.get("source_ids"))
        observed_at = _utc_timestamp(raw.get("observed_at") or raw.get("fired_at"))
        expires_at = _utc_timestamp(raw.get("expires_at"))
        explicit_status = str(raw.get("status") or "ARMED").upper().strip()
        evaluation = "ARMED"
        reason = "waiting for a sourced observation"
        structurally_valid = bool(
            ticker
            and len(trigger_id) >= 3
            and kind in TRIGGER_KINDS
            and len(metric) >= 2
            and comparator in TRIGGER_COMPARATORS
            and "threshold" in raw
            and raw.get("threshold") is not None
            and explicit_status in TRIGGER_STATUSES
            and sources
        )
        if not structurally_valid:
            evaluation, reason = "UNEVALUABLE", "trigger row fails the declared schema contract"
        elif explicit_status == "CANCELLED":
            evaluation, reason = "CANCELLED", "trigger was explicitly cancelled"
        elif explicit_status == "EXPIRED":
            evaluation, reason = "EXPIRED", "trigger was explicitly expired"
        elif expires_at is not None and expires_at < cutoff and explicit_status != "FIRED":
            evaluation, reason = "EXPIRED", "trigger expired before the run cutoff"
        elif explicit_status == "FIRED" and observed_at is not None and observed_at <= cutoff:
            evaluation, reason = "FIRED", "ledger explicitly records a sourced firing"
        elif explicit_status == "FIRED":
            evaluation, reason = "UNEVALUABLE", "fired trigger lacks an in-cutoff observation timestamp"
        elif observed_at is None or observed_at > cutoff:
            evaluation, reason = "ARMED", "no in-cutoff observation"
        elif raw.get("observed_value") in (None, ""):
            evaluation, reason = "UNEVALUABLE", "observation has no value"
        elif _compare(
            raw.get("comparator"),
            raw.get("observed_value"),
            raw.get("threshold"),
            raw.get("prior_value"),
        ):
            evaluation, reason = "FIRED", "sourced observation crossed the recorded threshold"
        else:
            evaluation, reason = "ARMED", "latest sourced observation did not cross the threshold"
        events.append({
            "ticker": ticker,
            "trigger_id": trigger_id,
            "kind": kind,
            "evaluation": evaluation,
            "reason": reason,
            "observed_at": observed_at.isoformat() if observed_at is not None else None,
            "source_ids": sources,
        })
    fired_tickers = sorted({row["ticker"] for row in events if row["evaluation"] == "FIRED"})
    return {
        "schema_version": schema_version,
        "status": "AVAILABLE",
        "armed": sum(row["evaluation"] == "ARMED" for row in events),
        "fired": sum(row["evaluation"] == "FIRED" for row in events),
        "expired": sum(row["evaluation"] == "EXPIRED" for row in events),
        "cancelled": sum(row["evaluation"] == "CANCELLED" for row in events),
        "unevaluable": sum(row["evaluation"] == "UNEVALUABLE" for row in events),
        "fired_tickers": fired_tickers,
        "events": events,
    }


def _material_changes(
    payload: Any,
    *,
    as_of: str | date,
    since: Any = None,
) -> dict[str, Any]:
    schema_version = payload.get("schema_version") if isinstance(payload, dict) else None
    rows = payload.get("evidence") if isinstance(payload, dict) else None
    updated_at = _utc_timestamp(payload.get("updated_at")) if isinstance(payload, dict) else None
    if schema_version != EVIDENCE_SCHEMA_VERSION or not isinstance(rows, list) or updated_at is None:
        return {
            "schema_version": schema_version,
            "status": "WRONG_SCHEMA" if schema_version != EVIDENCE_SCHEMA_VERSION else "INVALID",
            "changed_tickers": [],
            "material_rows": 0,
            "invalid_rows": 0,
        }
    cutoff = _cutoff(as_of)
    since_at = _utc_timestamp(since)
    changed: set[str] = set()
    material_rows = 0
    invalid_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            invalid_rows += 1
            continue
        materiality = str(row.get("materiality") or "").upper()
        direction = str(row.get("direction") or "").upper()
        evidence_id = str(row.get("evidence_id") or "").strip()
        claim_id = str(row.get("claim_id") or "").strip()
        claim = str(row.get("claim") or "").strip()
        source_id = str(row.get("source_id") or "").strip()
        ticker = str(row.get("ticker") or "").upper().strip()
        observed_at = _utc_timestamp(row.get("observed_at"))
        structurally_valid = bool(
            len(evidence_id) >= 3
            and ticker
            and len(claim_id) >= 3
            and len(claim) >= 8
            and direction in EVIDENCE_DIRECTIONS
            and materiality in {"CONTEXT", "MODEL_RELEVANT", *MATERIAL_CHANGE_LEVELS}
            and observed_at is not None
            and source_id
        )
        if not structurally_valid:
            invalid_rows += 1
            continue
        if materiality not in MATERIAL_CHANGE_LEVELS:
            continue
        explicitly_new = row.get("new_since_last_run") is True
        if observed_at > cutoff:
            continue
        if not explicitly_new and (since_at is None or observed_at <= since_at):
            continue
        material_rows += 1
        changed.add(ticker)
    return {
        "schema_version": schema_version,
        "status": "AVAILABLE_WITH_INVALID_ROWS" if invalid_rows else "AVAILABLE",
        "changed_tickers": sorted(changed),
        "material_rows": material_rows,
        "invalid_rows": invalid_rows,
        "since": since_at.isoformat() if since_at is not None else None,
    }


def load_research_event_state(
    *,
    as_of: str | date,
    trigger_path: str | Path = TRIGGER_STATE_PATH,
    evidence_path: str | Path = EVIDENCE_STATE_PATH,
    previous_manifest_path: str | Path = RUN_MANIFEST_LATEST_PATH,
) -> dict[str, Any]:
    trigger_payload, trigger_file_status = _read_json(Path(trigger_path))
    evidence_payload, evidence_file_status = _read_json(Path(evidence_path))
    previous_manifest, manifest_status = _read_json(Path(previous_manifest_path))
    previous_completed_at = (
        previous_manifest.get("completed_at") if isinstance(previous_manifest, dict) else None
    )
    triggers = (
        evaluate_trigger_payload(trigger_payload, as_of=as_of)
        if trigger_payload is not None
        else {"status": trigger_file_status, "fired_tickers": [], "events": [], "armed": 0,
              "fired": 0, "expired": 0, "cancelled": 0, "unevaluable": 0,
              "schema_version": None}
    )
    evidence = (
        _material_changes(evidence_payload, as_of=as_of, since=previous_completed_at)
        if evidence_payload is not None
        else {"status": evidence_file_status, "changed_tickers": [], "material_rows": 0,
              "invalid_rows": 0, "since": previous_completed_at, "schema_version": None}
    )
    return {
        "fired_trigger_tickers": set(triggers.get("fired_tickers", [])),
        "thesis_changed_tickers": set(evidence.get("changed_tickers", [])),
        "health": {
            "trigger_ledger": {
                "file_status": trigger_file_status,
                "expected_schema": TRIGGER_SCHEMA_VERSION,
                **triggers,
            },
            "evidence_ledger": {
                "file_status": evidence_file_status,
                "expected_schema": EVIDENCE_SCHEMA_VERSION,
                **evidence,
            },
            "previous_manifest": {
                "status": manifest_status,
                "completed_at": previous_completed_at,
            },
        },
    }


def load_portfolio_snapshot(
    path: str | Path = PORTFOLIO_SNAPSHOT_PATH,
    *,
    as_of: str | date | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Load manual/read-only context; absence must never imply zero holdings."""
    payload, status = _read_json(Path(path))
    if not isinstance(payload, dict):
        return None, {"status": status, "available": False}
    snapshot_at = _utc_timestamp(payload.get("as_of"))
    holdings = payload.get("holdings")
    valid = (
        payload.get("schema_version") == PORTFOLIO_SNAPSHOT_SCHEMA_VERSION
        and snapshot_at is not None
        and isinstance(holdings, list)
        and payload.get("nav") not in (None, "")
        and payload.get("read_only") is True
    )
    future = False
    if as_of is not None and snapshot_at is not None:
        future = snapshot_at > _cutoff(as_of)
        valid = valid and not future
    return (payload if valid else None), {
        "status": "CURRENT" if valid else "INVALID",
        "available": bool(valid),
        "as_of": snapshot_at.isoformat() if snapshot_at is not None else None,
        "future_dated": future,
        "holding_count": len(holdings) if isinstance(holdings, list) else None,
    }


__all__ = [
    "EVIDENCE_STATE_PATH",
    "PORTFOLIO_SNAPSHOT_PATH",
    "RUN_MANIFEST_LATEST_PATH",
    "TRIGGER_STATE_PATH",
    "evaluate_trigger_payload",
    "load_portfolio_snapshot",
    "load_research_event_state",
]
