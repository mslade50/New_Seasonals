"""Consume reversible private-site research controls without touching capital."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import CONTROL_STATE_MAX_AGE_DAYS


ALLOWED_ACTIONS = {"DEEPEN", "WATCH", "PASS", "CLEAR"}


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
    thesis_changed_tickers: Iterable[str] = (),
    fired_trigger_tickers: Iterable[str] = (),
) -> pd.DataFrame:
    """Apply controls to research priority only.

    PASS and WATCH are reopened only by a caller-supplied material change or
    fired trigger.  No field produced here can create security readiness,
    allocation, an order, or a portfolio mutation.
    """
    result = candidates.copy()
    if result.empty:
        return result
    changed = {str(ticker).upper() for ticker in thesis_changed_tickers}
    fired = {str(ticker).upper() for ticker in fired_trigger_tickers}
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
        if action == "DEEPEN":
            result.at[idx, "control_disposition"] = "NEXT_BOUNDED_DILIGENCE_PASS"
            result.at[idx, "research_queue_priority"] = max(
                float(row.get("research_queue_priority") or 0.0), 10_000.0
            )
        elif action == "WATCH":
            reopened = ticker in fired or ticker in changed
            result.at[idx, "research_suppressed"] = not reopened
            result.at[idx, "control_disposition"] = (
                "REOPENED_BY_TRIGGER" if reopened else "WAIT_FOR_RECORDED_TRIGGER"
            )
        elif action == "PASS":
            reopened = ticker in changed
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


__all__ = ["ALLOWED_ACTIONS", "apply_research_controls", "load_research_controls"]
