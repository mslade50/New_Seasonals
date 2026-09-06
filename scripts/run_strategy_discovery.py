"""Run the offline/file-only strategy-discovery foundation.

This executable cannot browse X.  A separately reviewed, read-only collector
must first create the strict local JSON/JSONL snapshots passed here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.strategy_discovery.contracts import (
    ContractError,
    load_json,
    load_jsonl,
    validate_report,
)
from research.strategy_discovery.journal import (
    append_events,
    exclusive_lock,
    load_journal,
)
from research.strategy_discovery.pipeline import run_discovery
from research.strategy_discovery.render import (
    publish_immutable_bundle,
    publish_latest_pointer,
)

DEFAULT_APPROVED_OUTPUT_ROOT = ROOT / "artifacts" / "strategy_discovery"


def _local_output_dir(raw: str, approved_root: Path) -> Path:
    if "://" in raw or raw.startswith(("\\\\", "//")):
        raise ContractError("output-dir must be a local filesystem path")
    requested = Path(raw)
    if requested.exists() and requested.is_symlink():
        raise ContractError("output-dir must not be a symbolic link")
    output_dir = requested.resolve()
    root = approved_root.resolve()
    try:
        output_dir.relative_to(root)
    except ValueError as exc:
        raise ContractError(
            f"output-dir must be within the approved local root: {root}"
        ) from exc
    return output_dir


def _input_paths(args: argparse.Namespace) -> list[Path]:
    values = [
        args.config,
        args.source_manifest,
        args.items,
        args.strategy_catalog,
        args.dead_end_catalog,
        args.validation_artifacts,
        args.owner_transitions,
    ]
    return [Path(value).resolve() for value in values if value]


def _reject_input_output_collisions(input_paths: list[Path], output_dir: Path) -> None:
    for input_path in input_paths:
        try:
            input_path.relative_to(output_dir)
        except ValueError:
            continue
        raise ContractError(
            f"input snapshot must not live inside output-dir: {input_path}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate local X discovery snapshots and render a research-only report."
    )
    parser.add_argument("--config", required=True, help="Strict run config JSON.")
    parser.add_argument("--source-manifest", required=True, help="Strict source coverage manifest JSON.")
    parser.add_argument("--items", required=True, help="Captured source items JSONL.")
    parser.add_argument("--strategy-catalog", required=True, help="Digested strategy-book snapshot JSON.")
    parser.add_argument("--dead-end-catalog", required=True, help="Digested rejected-research snapshot JSON.")
    parser.add_argument("--validation-artifacts", help="Optional reproducible-research artifact manifest JSON.")
    parser.add_argument("--owner-transitions", help="Optional explicit human transition JSONL.")
    parser.add_argument("--output-dir", required=True, help="Local output directory; owns its journal.jsonl.")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    approved_output_root: Path | None = None,
    lock_timeout_seconds: float = 10.0,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        approved_root = (approved_output_root or DEFAULT_APPROVED_OUTPUT_ROOT).resolve()
        output_dir = _local_output_dir(args.output_dir, approved_root)
        inputs = _input_paths(args)
        _reject_input_output_collisions(inputs, output_dir)
        journal_path = output_dir / "journal.jsonl"
        config = load_json(Path(args.config))
        manifest = load_json(Path(args.source_manifest))
        items = load_jsonl(Path(args.items))
        strategy_catalog = load_json(Path(args.strategy_catalog))
        dead_end_catalog = load_json(Path(args.dead_end_catalog))
        validation_artifacts = (
            load_json(Path(args.validation_artifacts)) if args.validation_artifacts else None
        )
        owner_transitions = (
            load_jsonl(Path(args.owner_transitions)) if args.owner_transitions else None
        )
        artifact_root = approved_root / "validation_artifacts"
        if validation_artifacts:
            artifact_paths = [
                (artifact_root / artifact["artifact_path"]).resolve()
                for artifact in validation_artifacts.get("artifacts", [])
                if isinstance(artifact, dict) and isinstance(artifact.get("artifact_path"), str)
            ]
            _reject_input_output_collisions(artifact_paths, output_dir)

        transaction_lock = output_dir / ".transaction.lock"
        with exclusive_lock(
            transaction_lock,
            timeout_seconds=lock_timeout_seconds,
        ):
            journal = load_journal(journal_path)
            report, events = run_discovery(
                config_raw=config,
                manifest_raw=manifest,
                items_raw=items,
                strategy_catalog_raw=strategy_catalog,
                dead_end_catalog_raw=dead_end_catalog,
                journal_records=journal,
                validation_artifacts_raw=validation_artifacts,
                owner_transitions_raw=owner_transitions,
                artifact_root=artifact_root,
            )
            validate_report(report)
            paths, bundle_manifest = publish_immutable_bundle(output_dir, report)
            appended = append_events(
                journal_path,
                events,
                recorded_at=report["as_of"],
                lock_held=True,
            )
            verified_journal = load_journal(journal_path)
            latest_path = publish_latest_pointer(
                output_dir,
                report,
                bundle_manifest,
                verified_journal,
            )
    except ContractError as exc:
        print(f"STRATEGY DISCOVERY BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(
        f"strategy discovery {report['run_mode']} {report['completeness']}: "
        f"{report['summary']['candidate_count']} candidate(s), "
        f"{report['summary']['new_research_ready']} research-ready"
    )
    print(f"journal: {journal_path} ({appended} new event(s))")
    for path in paths:
        print(f"report: {path}")
    print(f"latest: {latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
