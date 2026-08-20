"""Validate the safety and reproducibility fields of a fundamental run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fundamental.config import RUN_MANIFEST_SCHEMA_VERSION  # noqa: E402
from fundamental.run_manifest import LATEST_MANIFEST_PATH  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=LATEST_MANIFEST_PATH)
    parser.add_argument("--require-visual-qa", action="store_true")
    args = parser.parse_args()
    if not args.path.exists():
        raise SystemExit(f"run manifest unavailable: {args.path}")
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    errors = []
    if payload.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        errors.append("wrong run-manifest schema")
    if payload.get("research_only") is not True:
        errors.append("research_only must be true")
    if payload.get("publishing_performed") is not False:
        errors.append("publishing_performed must be false")
    if payload.get("live_actions_enabled") is not False:
        errors.append("live_actions_enabled must be false")
    if payload.get("completion_scope") != "OPERATIONAL_RUN_ONLY":
        errors.append("completion_scope must be OPERATIONAL_RUN_ONLY")
    expected_readiness = (
        "DECISION_READY_PRESENT"
        if int(payload.get("underwrite_contract", {}).get("decision_ready", 0) or 0) > 0
        else "NO_DECISION_READY"
    )
    if payload.get("investment_readiness") != expected_readiness:
        errors.append(f"investment_readiness must be {expected_readiness}")
    for field in ("run_id", "as_of", "started_at", "completed_at", "source_freeze", "coverage", "decision_states"):
        if payload.get(field) in (None, ""):
            errors.append(f"missing {field}")
    report = payload.get("outputs", {}).get("report", {})
    if not report.get("available") or not report.get("sha256"):
        errors.append("manifested report is unavailable or lacks a digest")
    if args.require_visual_qa and payload.get("qa", {}).get("visual_status") != "PASS":
        errors.append("visual QA has not passed")
    if errors:
        raise SystemExit("Run-manifest validation failed:\n- " + "\n- ".join(errors))
    print(
        f"Run manifest valid: {payload['run_id']} | "
        f"status={payload.get('completion_status')} ({payload.get('completion_scope')}) | "
        f"readiness={payload.get('investment_readiness')} | "
        f"visual={payload.get('qa', {}).get('visual_status')}"
    )


if __name__ == "__main__":
    main()
