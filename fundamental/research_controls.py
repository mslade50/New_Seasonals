"""Consume reversible private-site research controls without touching capital."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import CONTROL_STATE_MAX_AGE_DAYS


ALLOWED_ACTIONS = {"DEEPEN", "WATCH", "PASS", "CLEAR"}


def current_research_allowed(candidate: dict[str, Any] | None) -> bool:
    """Current inbox membership requires a current, unsuppressed candidate."""
    if not candidate:
        return False
    suppressed = candidate.get("research_suppressed", False)
    return not (pd.notna(suppressed) and bool(suppressed)) and candidate.get("research_eligible") is not False


def load_research_controls(
    path: str | Path,
    *,
    as_of: str | date,
    max_age_days: int = CONTROL_STATE_MAX_AGE_DAYS,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return {}, {"available": False, "status": "MISSING", "updated_at": None, "action_count": 0}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {"available": False, "status": "INVALID", "updated_at": None, "action_count": 0}
    actions = payload.get("actions") if isinstance(payload, dict) else None
    if not isinstance(actions, dict):
        return {}, {"available": False, "status": "INVALID", "updated_at": None, "action_count": 0}

    cleaned: dict[str, dict[str, Any]] = {}
    action_counts = {action: 0 for action in sorted(ALLOWED_ACTIONS)}
    for raw_ticker, raw_record in actions.items():
        ticker = str(raw_ticker or "").upper().strip()
        record = raw_record if isinstance(raw_record, dict) else {}
        action = str(record.get("action") or "").upper().strip()
        if ticker and action in ALLOWED_ACTIONS:
            action_counts[action] += 1
            # Some clients can persist CLEAR as a tombstone.  It removes an
            # override and must never become a research instruction itself.
            if action == "CLEAR":
                continue
            cleaned[ticker] = {
                "action": action,
                "updated_at": record.get("updated_at"),
                "as_of": record.get("as_of"),
            }

    updated = pd.to_datetime(payload.get("updated_at"), errors="coerce", utc=True)
    report_date = pd.Timestamp(as_of).tz_localize("UTC")
    age_days = int((report_date.normalize() - updated.normalize()).days) if pd.notna(updated) else None
    if age_days is None:
        status = "UNDATED"
    elif age_days < 0:
        status = "FUTURE_DATED"
    elif age_days > max_age_days:
        status = "STALE"
    else:
        status = "CURRENT"
    return cleaned, {
        "available": True,
        "status": status,
        "updated_at": payload.get("updated_at"),
        "age_days": age_days,
        "max_age_days": int(max_age_days),
        "action_count": len(cleaned),
        "action_counts": action_counts,
    }


def apply_research_controls(
    candidates: pd.DataFrame,
    controls: dict[str, dict[str, Any]],
    *,
    thesis_events: Iterable[dict[str, Any]] = (),
    trigger_events: Iterable[dict[str, Any]] = (),
    completed_control_requests: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Apply controls to research priority only.

    PASS and WATCH are reopened only by a caller-supplied material change or
    fired trigger.  No field produced here can create security readiness,
    allocation, an order, or a portfolio mutation.
    """
    result = candidates.copy()
    if result.empty:
        return result
    thesis_events = list(thesis_events)
    trigger_events = list(trigger_events)
    if "research_base_queue_priority" not in result:
        result["research_base_queue_priority"] = result.get("research_queue_priority", 0.0)
    result["research_queue_priority"] = result["research_base_queue_priority"]
    result["research_control"] = ""
    result["research_suppressed"] = False
    result["control_disposition"] = "NONE"
    result["control_updated_at"] = None

    for idx, row in result.iterrows():
        ticker = str(row.get("ticker") or "").upper()
        control = controls.get(ticker)
        if not control:
            continue
        action = str(control.get("action") or "").upper()
        result.at[idx, "research_control"] = action
        result.at[idx, "control_updated_at"] = control.get("updated_at")
        control_at = pd.to_datetime(control.get("updated_at"), errors="coerce", utc=True)
        def newer(event):
            stamp = pd.to_datetime(event.get("observed_at"), errors="coerce", utc=True)
            return (str(event.get("ticker", "")).upper() == ticker
                    and pd.notna(control_at) and pd.notna(stamp) and stamp > control_at)
        changed = any(newer(event) and event.get("materiality") in {"THESIS_CHANGING", "DECISION_CHANGING"}
                      for event in thesis_events)
        fired = any(newer(event) and event.get("evaluation") == "FIRED"
                    and event.get("kind") in {"PROOF", "REOPEN"} for event in trigger_events)
        if action == "DEEPEN":
            if (completed_control_requests or {}).get(ticker) == control.get("updated_at"):
                result.at[idx, "control_disposition"] = "COMPLETED_BOUNDED_DILIGENCE_PASS"
                continue
            result.at[idx, "control_disposition"] = "NEXT_BOUNDED_DILIGENCE_PASS"
            result.at[idx, "research_queue_priority"] = max(
                float(row.get("research_queue_priority") or 0.0), 10_000.0
            )
        elif action == "WATCH":
            reopened = fired or changed
            result.at[idx, "research_suppressed"] = not reopened
            result.at[idx, "control_disposition"] = (
                "REOPENED_BY_TRIGGER" if reopened else "WAIT_FOR_RECORDED_TRIGGER"
            )
        elif action == "PASS":
            reopened = changed
            result.at[idx, "research_suppressed"] = not reopened
            result.at[idx, "control_disposition"] = (
                "REOPENED_BY_THESIS_CHANGE" if reopened else "SUPPRESS_UNCHANGED_EVIDENCE"
            )

    result["screen_can_surface_review"] = False
    result = result.sort_values(
        ["research_suppressed", "research_queue_priority"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)
    return result


def completed_diligence_requests(
    decisions: list[dict[str, Any]], controls: dict[str, dict], *, as_of: str | date | None = None,
) -> dict[str, str]:
    """Consume only an explicitly completed pass for this exact request revision.

    Building a screen/report is not diligence completion. The completed
    underwrite records which request it answered and its completion timestamp.
    """
    completed = {}
    cutoff = (pd.Timestamp(as_of, tz="UTC") + pd.Timedelta(days=1)
              if as_of is not None else pd.Timestamp.now(tz="UTC"))
    for record in decisions:
        ticker = str(record.get("ticker", "")).upper()
        control = controls.get(ticker, {})
        revision = record.get("research_control_updated_at")
        started = pd.to_datetime(revision, errors="coerce", utc=True)
        finished = pd.to_datetime(record.get("completed_at"), errors="coerce", utc=True)
        if (control.get("action") == "DEEPEN" and revision == control.get("updated_at")
                and record.get("schema_version") == "fundamental-underwrite.v2"
                and pd.notna(started) and pd.notna(finished) and started <= finished < cutoff
                and record.get("decision") in {"QUICK_REVIEW", "WAIT_FOR_PROOF", "WAIT_FOR_EVENT", "PASS"}):
            completed[ticker] = revision
    return completed


__all__ = ["ALLOWED_ACTIONS", "apply_research_controls", "load_research_controls",
           "current_research_allowed", "completed_diligence_requests"]
