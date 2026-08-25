"""Immutable-ish local run artifacts for replay and review."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from .config import EPPolicy
from .schema import RunResult, StagingPreview


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _csv_safe(value: Any) -> Any:
    """Prevent review CSV cells from becoming spreadsheet formulas."""

    if not isinstance(value, str):
        return value
    trimmed = value.lstrip(" \t\r\n")
    if trimmed.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _report(result: RunResult, policy: EPPolicy) -> str:
    by_decision: dict[str, int] = {}
    for decision in result.decisions:
        by_decision[decision.decision] = by_decision.get(decision.decision, 0) + 1
    lines = [
        f"# Episodic Pivot shadow run — {result.run_id}",
        "",
        f"Generated: {result.generated_at}",
        f"Policy: `{policy.policy_id}`",
        "Mode: shadow research; no broker, Sheets, R2, schedule, or production write.",
        "",
        "## Counts",
        "",
        f"- Nominations: {len(result.candidates)}",
        f"- Stageable previews: {len(result.previews)}",
    ]
    for label in sorted(by_decision):
        lines.append(f"- {label}: {by_decision[label]}")
    lines.extend(["", "## Candidate decisions", ""])
    if not result.decisions:
        lines.append("No nominations passed the broad premarket discovery screen.")
    for decision in result.decisions:
        blockers = ", ".join(decision.blockers) or "none"
        warnings = ", ".join(decision.warnings) or "none"
        lines.append(
            f"- **{decision.symbol}** — {decision.decision} / {decision.setup_type}; "
            f"catalyst={decision.catalyst.catalyst_type}; "
            f"materiality={decision.catalyst.materiality_score}/5; "
            f"blockers={blockers}; warnings={warnings}."
        )
    lines.extend(
        [
            "",
            "## Safety invariants",
            "",
            "- Every preview is a regular-hours DAY limit order; there is no market fallback.",
            "- Release requires a fresh quote, halt and contract rechecks, the 4%-25% opening band, and a 09:35 ET entry expiry.",
            "- `Approval` is blank and `Live_Eligible` is false.",
            "- A fetched, timestamped source document is required; a search snippet cannot confirm an EP.",
            "- This output is a review artifact, not an instruction to trade.",
            "",
        ]
    )
    return "\n".join(lines)


def write_run_artifacts(
    result: RunResult,
    *,
    policy: EPPolicy,
    output_dir: str | Path,
    input_files: dict[str, str | Path] | None = None,
    search_provider: str = "OFFLINE",
) -> Path:
    root = Path(output_dir).resolve()
    if root.exists():
        existing_manifest = root / "manifest.json"
        if not existing_manifest.exists():
            raise FileExistsError(f"refusing to reuse non-run directory: {root}")
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing.get("run_id") != result.run_id:
            raise FileExistsError(f"run directory belongs to another run: {root}")
    else:
        root.mkdir(parents=True, exist_ok=False)

    input_manifest = {
        name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
        for name, path in (input_files or {}).items()
    }
    manifest = {
        "schema_version": 1,
        "run_id": result.run_id,
        "generated_at": result.generated_at,
        "policy": policy.to_dict(),
        "search_provider": search_provider,
        "inputs": input_manifest,
        "counts": {
            "candidates": len(result.candidates),
            "decisions": len(result.decisions),
            "staging_previews": len(result.previews),
        },
        "safety": {
            "live_actions_enabled": False,
            "broker_contacted": False,
            "sheets_written": False,
            "r2_written": False,
            "production_deployed": False,
        },
    }
    _json_dump(root / "candidates.json", [item.to_dict() for item in result.candidates])
    _json_dump(
        root / "evidence.json",
        {
            candidate_id: [document.to_dict() for document in documents]
            for candidate_id, documents in result.documents_by_candidate.items()
        },
    )
    _json_dump(
        root / "evidence_by_symbol.json",
        {
            candidate.snapshot.symbol: [
                document.to_dict()
                for document in result.documents_by_candidate.get(candidate.candidate_id, [])
            ]
            for candidate in result.candidates
        },
    )
    _json_dump(root / "decisions.json", [item.to_dict() for item in result.decisions])
    preview_rows = [item.to_dict() for item in result.previews]
    _json_dump(root / "staging_preview.json", preview_rows)

    headers = (
        list(result.previews[0].to_dict())
        if result.previews
        else [item.name for item in fields(StagingPreview)]
    )
    with (root / "staging_preview.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in preview_rows:
            row = dict(row)
            row["evidence_urls"] = " | ".join(row.get("evidence_urls") or ())
            row["evidence_published_at"] = " | ".join(
                row.get("evidence_published_at") or ()
            )
            writer.writerow({key: _csv_safe(value) for key, value in row.items()})

    (root / "report.md").write_text(_report(result, policy), encoding="utf-8")
    artifact_names = (
        "candidates.json",
        "evidence.json",
        "evidence_by_symbol.json",
        "decisions.json",
        "staging_preview.json",
        "staging_preview.csv",
        "report.md",
    )
    manifest["artifacts"] = {
        name: {
            "sha256": sha256_file(root / name),
            "size_bytes": (root / name).stat().st_size,
        }
        for name in artifact_names
    }
    _json_dump(root / "manifest.json", manifest)
    return root
