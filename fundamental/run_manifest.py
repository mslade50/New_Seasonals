"""Immutable run manifests and append-only state-transition records."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .config import (
    CURRENT_ROOT,
    EVIDENCE_SCHEMA_VERSION,
    POLICY_VERSION,
    ROOT,
    RUN_MANIFEST_SCHEMA_VERSION,
    RUN_ROOT,
    SCHEMA_VERSION,
    TRIGGER_SCHEMA_VERSION,
    UNDERWRITE_SCHEMA_VERSION,
)
from .storage import canonical_json_bytes, iso_utc, write_run_manifest


LATEST_MANIFEST_PATH = CURRENT_ROOT / "fundamental_run_manifest_latest.json"
TRANSITION_LOG_PATH = CURRENT_ROOT / "research_transitions.jsonl"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_sources(paths: dict[str, str | Path]) -> dict[str, dict[str, Any]]:
    frozen: dict[str, dict[str, Any]] = {}
    for label, raw_path in paths.items():
        path = Path(raw_path)
        if not path.exists():
            frozen[label] = {"path": str(path), "available": False}
            continue
        stat = path.stat()
        frozen[label] = {
            "path": str(path),
            "available": True,
            "size_bytes": int(stat.st_size),
            "modified_at": iso_utc_from_epoch(stat.st_mtime),
            "sha256": file_sha256(path),
        }
    return frozen


def iso_utc_from_epoch(value: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def git_code_state() -> dict[str, Any]:
    commands = {
        "commit": ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
        "status": [
            "git", "-c", f"safe.directory={ROOT.as_posix()}",
            "status", "--porcelain", "--untracked-files=no",
        ],
    }
    results: dict[str, str | None] = {}
    for name, command in commands.items():
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            results[name] = completed.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            results[name] = None
    return {
        "git_commit": results["commit"],
        "tracked_worktree_dirty": bool(results["status"]) if results["status"] is not None else None,
    }


def decision_states(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        ticker = str(record.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        validation = record.get("_validation") if isinstance(record.get("_validation"), dict) else {}
        evidence = record.get("evidence_ledger") if isinstance(record.get("evidence_ledger"), list) else []
        supporting_ids = sorted({
            str(item.get("evidence_id")).strip()
            for item in evidence
            if isinstance(item, dict)
            and str(item.get("direction") or "").upper() == "CONFIRM"
            and str(item.get("evidence_id") or "").strip()
        })
        disconfirming_ids = sorted({
            str(item.get("evidence_id")).strip()
            for item in evidence
            if isinstance(item, dict)
            and str(item.get("direction") or "").upper() == "DISCONFIRM"
            and str(item.get("evidence_id") or "").strip()
        })
        states[ticker] = {
            "underwrite_id": record.get("underwrite_id"),
            "issuer": record.get("company_name") or record.get("issuer"),
            "schema_version": record.get("schema_version", "legacy"),
            "decision": record.get("decision"),
            "security_readiness": record.get("security_readiness"),
            "as_of": record.get("as_of"),
            "review_ready": bool(validation.get("valid_for_quick_review", False)),
            "supporting_evidence_ids": supporting_ids,
            "disconfirming_evidence_ids": disconfirming_ids,
        }
    return dict(sorted(states.items()))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def append_decision_transitions(
    *,
    previous_manifest_path: str | Path,
    current_states: dict[str, dict[str, Any]],
    run_id: str,
    as_of: str,
    output_path: str | Path = TRANSITION_LOG_PATH,
) -> list[dict[str, Any]]:
    """Append only actual decision-state changes, idempotently by content ID."""
    previous = _read_json(Path(previous_manifest_path)).get("decision_states", {})
    previous = previous if isinstance(previous, dict) else {}
    tickers = sorted(set(previous) | set(current_states))
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        before = previous.get(ticker)
        after = current_states.get(ticker)
        if before == after:
            continue
        reason_code = (
            "INITIAL_STATE_CAPTURE" if before is None
            else "UNDERWRITE_REMOVED" if after is None
            else "DECISION_STATE_CHANGED"
        )
        rationale = {
            "INITIAL_STATE_CAPTURE": "First manifested decision state for this security.",
            "UNDERWRITE_REMOVED": "Security no longer appears in the manifested underwrite set.",
            "DECISION_STATE_CHANGED": "One or more manifested decision-state fields changed.",
        }[reason_code]
        evidence_state = after if isinstance(after, dict) else before if isinstance(before, dict) else {}
        core = {
            "run_id": run_id,
            "agent_run_id": run_id,
            "as_of": as_of,
            "source_freeze_at": as_of,
            "ticker": ticker,
            "issuer_security": {
                "ticker": ticker,
                "issuer": evidence_state.get("issuer"),
            },
            "before": before,
            "after": after,
            "prior_state": before,
            "new_state": after,
            "reason_code": reason_code,
            "rationale": rationale,
            "supporting_evidence_ids": evidence_state.get("supporting_evidence_ids", []),
            "disconfirming_evidence_ids": evidence_state.get("disconfirming_evidence_ids", []),
            "fired_trigger_id": None,
            "versions": {
                "policy": POLICY_VERSION,
                "data_schema": SCHEMA_VERSION,
                "underwrite_schema": UNDERWRITE_SCHEMA_VERSION,
                "trigger_schema": TRIGGER_SCHEMA_VERSION,
                "evidence_schema": EVIDENCE_SCHEMA_VERSION,
            },
            "authority": "deterministic_research_engine",
            "research_only": True,
            "live_actions_enabled": False,
        }
        transition_id = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
        rows.append({"transition_id": transition_id, "recorded_at": iso_utc(), **core})
    if not rows:
        return []
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("transition_id"):
                existing_ids.add(str(item["transition_id"]))
    new_rows = [row for row in rows if row["transition_id"] not in existing_ids]
    if new_rows:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in new_rows:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return new_rows


def write_sleeve_run_manifest(payload: dict[str, Any]) -> tuple[Path, Path]:
    manifest = dict(payload)
    manifest.setdefault("schema_version", RUN_MANIFEST_SCHEMA_VERSION)
    manifest.setdefault("versions", {})
    manifest["versions"].update({
        "policy": POLICY_VERSION,
        "data_schema": SCHEMA_VERSION,
        "underwrite_schema": UNDERWRITE_SCHEMA_VERSION,
        "run_manifest_schema": RUN_MANIFEST_SCHEMA_VERSION,
    })
    manifest["research_only"] = True
    manifest["publishing_performed"] = False
    manifest["live_actions_enabled"] = False
    manifest.setdefault("completion_scope", "OPERATIONAL_RUN_ONLY")
    decision_ready = int(manifest.get("underwrite_contract", {}).get("decision_ready", 0) or 0)
    manifest.setdefault(
        "investment_readiness",
        "DECISION_READY_PRESENT" if decision_ready > 0 else "NO_DECISION_READY",
    )
    run_id = str(manifest["run_id"])
    immutable_path = write_run_manifest(manifest, run_id=run_id)
    CURRENT_ROOT.mkdir(parents=True, exist_ok=True)
    latest = LATEST_MANIFEST_PATH
    temp = latest.with_name(f".{latest.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temp, latest)
    return immutable_path, latest


def record_visual_qa(
    *,
    manifest_path: str | Path = LATEST_MANIFEST_PATH,
    report_path: str | Path,
    passed: bool,
    notes: str,
) -> tuple[Path, Path]:
    """Attest a real browser inspection against the exact report digest."""
    current_path = Path(manifest_path)
    manifest = _read_json(current_path)
    if not manifest:
        raise ValueError(f"run manifest unavailable: {current_path}")
    report = Path(report_path)
    if not report.exists():
        raise ValueError(f"report unavailable: {report}")
    report_digest = file_sha256(report)
    expected = (
        manifest.get("outputs", {}).get("report", {}).get("sha256")
        if isinstance(manifest.get("outputs"), dict)
        else None
    )
    if expected and report_digest != expected:
        raise ValueError("report digest differs from the manifest; rebuild or inspect the manifested report")
    attestation = {
        "schema_version": "fundamental-visual-qa.v1",
        "run_id": manifest.get("run_id"),
        "recorded_at": iso_utc(),
        "status": "PASS" if passed else "FAIL",
        "report_path": str(report),
        "report_sha256": report_digest,
        "notes": str(notes).strip(),
    }
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    qa_path = RUN_ROOT / f"{manifest.get('run_id')}.qa.json"
    if qa_path.exists():
        prior = _read_json(qa_path)
        if prior.get("report_sha256") != report_digest or prior.get("status") != attestation["status"]:
            raise FileExistsError(f"visual QA attestation already exists and differs: {qa_path}")
    else:
        qa_path.write_text(json.dumps(attestation, indent=2, sort_keys=True), encoding="utf-8")
    manifest["qa"] = {
        "visual_status": attestation["status"],
        "attestation_path": str(qa_path),
        "recorded_at": attestation["recorded_at"],
    }
    manifest["completion_status"] = "COMPLETE" if passed else "FAILED"
    temp = current_path.with_name(f".{current_path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temp, current_path)
    return qa_path, current_path


__all__ = [
    "LATEST_MANIFEST_PATH",
    "TRANSITION_LOG_PATH",
    "append_decision_transitions",
    "decision_states",
    "file_sha256",
    "freeze_sources",
    "git_code_state",
    "record_visual_qa",
    "write_sleeve_run_manifest",
]
