"""Durable AM progress and receipt-based recovery; never certify research from progress."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from research_io import append_jsonl, file_lock, read_json
from trading_calendar import TRADING_DAY
from .schema import parse_timestamp

NY = ZoneInfo("America/New_York")
SCHEMA = "EP_MORNING_PROGRESS_V1"
STAGES = {"DISCOVERY", "RESEARCH", "REPORT_READY", "RETRY_PENDING", "PAUSED_BY_USER"}
MODE = "AGENT_GOOGLE_SEARCH_AND_READ"


def _clock(now=None):
    return parse_timestamp(now or datetime.now(timezone.utc))


def _journal(root: Path, target: str) -> Path:
    date.fromisoformat(target)
    return root.resolve() / "morning_sessions" / f"{target}.jsonl"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(root: Path, target: str, stage: str, paths: dict[str, Path], *, now=None) -> None:
    now = _clock(now)
    if stage not in STAGES or target != now.astimezone(NY).date().isoformat():
        raise ValueError("Checkpoint must describe today's active morning stage")
    frozen = {}
    for key, path in paths.items():
        path = Path(path).resolve()
        if not re.fullmatch(r"snapshot_\d+|queue|notes|reviews|report|short_queue|short_notes", key):
            raise ValueError("Unknown morning checkpoint artifact")
        if root.resolve().parent not in path.parents or not path.is_file():
            raise ValueError("Checkpoint file must exist under the runtime artifact root")
        frozen[key] = {"path": str(path), "sha256": _hash(path)}
    append_jsonl(_journal(root, target), [{"schema": SCHEMA, "session_date": target,
        "stage": stage, "updated_at": now.isoformat(), "artifacts": frozen}])


def progress(root: Path, target: str) -> dict:
    result = {"stage": "NOT_STARTED", "artifacts": {}, "changed_artifacts": []}
    journal = _journal(root, target)
    if not journal.exists():
        return result
    # Reads only; inspection must not create locks or modify retained evidence.
    records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        if (record.get("schema") != SCHEMA or record.get("session_date") != target
                or record.get("stage") not in STAGES):
            raise ValueError("Invalid morning checkpoint")
        parse_timestamp(record["updated_at"])
        result.update(stage=record["stage"], updated_at=record["updated_at"])
        result["artifacts"].update(record["artifacts"])
    for key, record in result["artifacts"].items():
        path = Path(record["path"]).resolve()
        if root.resolve().parent not in path.parents:
            raise ValueError("Checkpoint artifact escaped runtime artifacts")
        if not path.is_file() or _hash(path) != record["sha256"]:
            result["changed_artifacts"].append(key)
    return result


def delivery_outcome(root: Path, target: str, *, now=None) -> tuple[str | None, list[str]]:
    """Read the existing SMTP DATA-acceptance receipts, not a manually set done flag."""
    now = _clock(now)
    found, uncertain = [], []
    paths = list(root.rglob("email_delivery.json"))
    paths += list((root / "email_failures").glob(f"{target}-morning-*.email-delivery.json"))
    for path in sorted(set(paths)):
        try:
            receipt = read_json(path)
        except (ValueError, OSError):
            if (target in str(path) or target.replace("-", "") in str(path)
                    or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).astimezone(NY).date().isoformat() == target):
                uncertain.append(str(path))
            continue
        meta = receipt.get("metadata", {})
        if not isinstance(meta, dict):
            uncertain.append(str(path))
            continue
        if meta.get("target_session_date") != target:
            continue
        kind = receipt.get("kind")
        if kind not in {"morning", "failure"} or (kind == "failure" and meta.get("phase") != "morning"):
            continue
        valid = (receipt.get("record_type") == "EP_RESEARCH_EMAIL_DELIVERY_V1"
                 and receipt.get("schema_version") == 1 and receipt.get("research_only") is True
                 and receipt.get("broker_route") == "NONE" and receipt.get("order_submission_allowed") is False
                 and isinstance(receipt.get("recipient_count"), int) and receipt["recipient_count"] > 0
                 and all(re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(k, ""))) for k in ("source_sha256", "delivery_id")))
        try:
            sent = parse_timestamp(receipt["sent_at"])
            local = sent.astimezone(NY)
            valid = valid and local.date().isoformat() == target and sent <= now
            if kind == "morning":
                valid = valid and time(4) <= local.time() < time(9, 30) and meta.get("research_mode") == MODE
        except (ValueError, KeyError, TypeError):
            valid = False
        if not valid or receipt.get("status") != "SENT":
            uncertain.append(str(path))
        else:
            found.append(("DELIVERED" if kind == "morning" else "FAILURE_REPORTED", str(path)))
    if uncertain or len(found) > 1:
        return "DELIVERY_UNCERTAIN", uncertain + [path for _, path in found]
    if found:
        return found[0][0], [found[0][1]]
    return None, []


def inspect_morning(root: Path, target: str, *, now=None) -> dict:
    now = _clock(now)
    local = now.astimezone(NY)
    day = date.fromisoformat(target)
    result = {"session_date": target, "checked_at": now.isoformat(), "status": "NOT_DUE"}
    if day != local.date() or not TRADING_DAY.is_on_offset(pd.Timestamp(day)) or local.time() < time(8, 20):
        return result
    status, receipts = delivery_outcome(root, target, now=now)
    result.update(status=status or ("RESUME" if local.time() < time(9, 30) else "DEADLINE_MISSED"), receipts=receipts)
    try:
        result["progress"] = progress(root, target)
    except (OSError, ValueError, KeyError, TypeError):
        result["progress"] = {"stage": "INVALID_CHECKPOINT", "artifacts": {}}
        # A broken progress file cannot invalidate a confirmed send or authorize a resend.
    if not status and result["progress"]["stage"] == "PAUSED_BY_USER":
        result["status"] = "PAUSED_BY_USER"
    return result


def claim_resume(root: Path, target: str, *, now=None) -> bool:
    now = _clock(now)
    local = now.astimezone(NY)
    with file_lock(_journal(root, target)):
        if not time(8, 40) <= local.time() < time(9, 30) or inspect_morning(root, target, now=now)["status"] != "RESUME":
            return False
        slot = local.strftime("%H") + f"{(local.minute // 20) * 20:02d}"
        path = root / "morning_sessions" / f"{target}-resume-{slot}.json"
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump({"session_date": target, "claimed_at": now.isoformat()}, handle)
        except FileExistsError:
            return False
        return True


def deliver_once(payload, settings, root: Path, *, now_fn=None) -> str:
    """Serialize every AM report/failure attempt, including across different run dirs."""
    from .email_delivery import EmailDeliveryError, deliver_email

    now_fn = now_fn or (lambda: datetime.now(timezone.utc))
    target = payload.metadata.get("target_session_date", "")
    if payload.kind not in {"morning", "failure"} or (payload.kind == "failure" and payload.metadata.get("phase") != "morning"):
        raise EmailDeliveryError("Session guard only accepts morning reports/failures")
    if root.resolve() not in payload.receipt_path.resolve().parents:
        raise EmailDeliveryError("Morning receipt must remain inside the guarded artifact root")
    with file_lock(_journal(root, target)):
        now = _clock(now_fn())
        if target != now.astimezone(NY).date().isoformat() or not TRADING_DAY.is_on_offset(pd.Timestamp(target)):
            raise EmailDeliveryError("Guarded morning delivery requires today's NYSE session")
        outcome, receipts = delivery_outcome(root, target, now=now)
        if outcome:
            if outcome in {"DELIVERED", "FAILURE_REPORTED"} and receipts == [str(payload.receipt_path)]:
                # Retain the sender's exact source/recipient identity check.
                return deliver_email(payload, settings, send=True)
            raise EmailDeliveryError(f"Morning session already has outcome {outcome}; no automatic second email")
        if progress(root, target)["stage"] == "PAUSED_BY_USER":
            raise EmailDeliveryError("Morning session explicitly paused by user")
        def pre_send():
            instant = _clock(now_fn())
            if target != instant.astimezone(NY).date().isoformat():
                raise EmailDeliveryError("Morning delivery target date changed before submission")
            if payload.kind == "failure" and instant.astimezone(NY).time() < time(9, 30):
                raise EmailDeliveryError("Morning failure email deferred until the 09:30 ET deadline; checkpoint RETRY_PENDING")
            if payload.kind == "morning":
                generated = parse_timestamp(payload.metadata["generated_at"])
                if (payload.metadata.get("research_mode") != MODE or not time(4) <= instant.astimezone(NY).time() < time(9, 30)
                        or not 0 <= (instant - generated).total_seconds() <= 1800):
                    raise EmailDeliveryError("Morning research is stale or outside today's premarket")
        pre_send()
        # Existing sender durably writes SENDING before SMTP DATA, and retains
        # ambiguous outcomes. They block recovery even if this process crashes.
        return deliver_email(payload, settings, send=True, pre_send=pre_send)
