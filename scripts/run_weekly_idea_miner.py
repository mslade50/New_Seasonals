"""Build a local, research-only weekly X/SSRN hypothesis inbox.

The script intentionally has no network client.  Source collection is a
separate step so the exact URL, text, and timestamps used by a run are frozen
and auditable.  Writes are opt-in and constrained to an ``artifacts`` tree.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = (ROOT / "artifacts").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiment_registry import (
    append_records,
    hypothesis_fingerprints,
    load_records,
    summarize,
)
from research.idea_miner.models import load_source_files
from research.idea_miner.pipeline import (
    build_weekly_queue,
    registry_records_for_queue,
)
from research.idea_miner.report import render_weekly_inbox


def _artifact_path(
    path: str | Path,
    *,
    artifacts_root: str | Path = ARTIFACTS_ROOT,
) -> Path:
    resolved = Path(path).expanduser().resolve()
    root = Path(artifacts_root).expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"research outputs must stay under this worktree's artifacts root "
            f"({root}): {resolved}"
        ) from exc
    return resolved


def _as_of_date(value: str) -> str:
    try:
        return dt.date.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise ValueError("--as-of must be an ISO calendar date (YYYY-MM-DD)") from exc


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def run(
    args: argparse.Namespace,
    *,
    artifacts_root: str | Path = ARTIFACTS_ROOT,
) -> dict[str, Any]:
    as_of = _as_of_date(args.as_of)
    output_dir = _artifact_path(args.output_dir, artifacts_root=artifacts_root)
    registry_path = (
        _artifact_path(args.registry, artifacts_root=artifacts_root)
        if args.registry
        else None
    )
    prior_records = load_records(registry_path) if registry_path else []
    source_records = load_source_files(args.input)
    queue = build_weekly_queue(
        source_records,
        as_of=as_of,
        prior_fingerprints=hypothesis_fingerprints(prior_records),
        max_candidates=args.max_candidates,
        max_per_archetype=args.max_per_archetype,
    )

    result: dict[str, Any] = {
        "as_of": as_of,
        "research_only": True,
        "no_order": True,
        "production_writes": False,
        "write_requested": bool(args.write),
        "coverage": queue["coverage"],
        "funnel": queue["funnel"],
        "outputs": {},
        "registry_records_appended": 0,
    }
    if not args.write:
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"weekly_idea_inbox_{as_of}"
    json_path = output_dir / f"{stem}.json"
    html_path = output_dir / f"{stem}.html"
    sources_path = output_dir / f"source_snapshot_{as_of}.jsonl"
    manifest_path = output_dir / f"{stem}_manifest.json"
    run_paths = (json_path, html_path, sources_path, manifest_path)
    existing = [path for path in run_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing weekly run artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    queue_json = _json(queue)
    html_text = render_weekly_inbox(queue)
    source_jsonl = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        for row in queue["sources"]
    )
    _atomic_write(json_path, queue_json)
    _atomic_write(html_path, html_text)
    _atomic_write(sources_path, source_jsonl)

    appended = 0
    if registry_path:
        appended = append_records(registry_path, registry_records_for_queue(queue))

    manifest = {
        "schema_version": "weekly-hypothesis-run.v1",
        "as_of": as_of,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "research_only": True,
        "no_order": True,
        "production_writes": False,
        "network_fetch": False,
        "input_paths": [str(Path(path).resolve()) for path in args.input],
        "outputs": {
            "queue_json": {"path": str(json_path), "sha256": _sha256_text(queue_json)},
            "report_html": {"path": str(html_path), "sha256": _sha256_text(html_text)},
            "source_snapshot": {"path": str(sources_path), "sha256": _sha256_text(source_jsonl)},
            "registry": str(registry_path) if registry_path else None,
        },
        "registry_records_appended": appended,
        "registry_summary": summarize(load_records(registry_path)) if registry_path else None,
    }
    _atomic_write(manifest_path, _json(manifest))

    result["registry_records_appended"] = appended
    result["outputs"] = {
        "queue_json": str(json_path),
        "report_html": str(html_path),
        "source_snapshot": str(sources_path),
        "manifest": str(manifest_path),
        "registry": str(registry_path) if registry_path else None,
    }
    return result


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", action="append", required=True, help="local .json/.jsonl/.csv source records; repeatable")
    ap.add_argument("--output-dir", required=True, help="explicit artifacts directory")
    ap.add_argument("--registry", help="optional append-only registry path under artifacts")
    ap.add_argument(
        "--as-of",
        default=dt.datetime.now(dt.timezone.utc).date().isoformat(),
    )
    ap.add_argument("--max-candidates", type=int, default=5)
    ap.add_argument("--max-per-archetype", type=int, default=2)
    ap.add_argument("--write", action="store_true", help="write local artifacts; otherwise compute and print only")
    return ap


def main() -> int:
    args = parser().parse_args()
    try:
        result = run(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(_json(result), end="")
    if not args.write:
        print("DRY RUN: no files or registry records were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
