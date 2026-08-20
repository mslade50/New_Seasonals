"""Fail-closed validation for fundamental underwrite decision records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fundamental.underwrite import (  # noqa: E402
    DECISION_STATUSES,
    validate_underwrite_record,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the v2 fundamental decision gate.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(ROOT / "data" / "fundamental" / "current" / "underwrite_decisions_latest.json"),
    )
    parser.add_argument("--as-of", default=None)
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"No decision file: {path}")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("decisions", payload if isinstance(payload, list) else [])
    if not isinstance(records, list):
        raise SystemExit("decision payload must be a list or {'decisions': [...]} object")

    decision_as_of = args.as_of or (payload.get("as_of") if isinstance(payload, dict) else None)
    failures = []
    for record in records:
        if not isinstance(record, dict):
            failures.append("non-object decision record")
            continue
        ticker = str(record.get("ticker") or "UNKNOWN").upper()
        status = str(record.get("decision") or "").upper()
        if status not in DECISION_STATUSES:
            failures.append(f"{ticker}: invalid decision {status!r}")
            continue
        result = validate_underwrite_record(record, decision_as_of=decision_as_of)
        passed = sum(bool(value) for value in result["gates"].values())
        total = len(result["gates"])
        print(
            f"{ticker}: {status} | contract={result['schema_version']} | "
            f"gates={passed}/{total} | review_ready={result['valid_for_quick_review']}"
        )
        if status == "QUICK_REVIEW" and not result["valid_for_quick_review"]:
            failures.extend(f"{ticker}: {error}" for error in result["errors"])

    if failures:
        raise SystemExit("Underwrite validation failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()
