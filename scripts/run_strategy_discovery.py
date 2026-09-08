"""Run the offline/file-only strategy-discovery foundation.

This executable cannot browse X.  A separately reviewed, read-only collector
must first create the strict local JSON/JSONL snapshots passed here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
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
    is_symlink_or_reparse,
    load_journal,
)
from research.strategy_discovery.pipeline import run_discovery
from research.strategy_discovery.family_fit import assess_family_fit, publish_family_fit
from research.strategy_discovery.render import (
    publish_immutable_bundle,
    publish_latest_pointer,
)

DEFAULT_APPROVED_OUTPUT_ROOT = ROOT / "artifacts" / "strategy_discovery"


def _local_output_dir(raw: str, approved_root: Path) -> Path:
    if "://" in raw or raw.startswith(("\\\\", "//")):
        raise ContractError("output-dir must be a local filesystem path")
    requested = Path(raw)
    if is_symlink_or_reparse(requested):
        raise ContractError("output-dir must not be a symbolic link")
    if requested.exists() and not requested.is_dir():
        raise ContractError("output-dir must be a directory, not an existing file")
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
        args.family_catalog,
        args.candidate_families,
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


def _reject_state_path_escape(path: Path, output_dir: Path) -> None:
    if is_symlink_or_reparse(path):
        raise ContractError(f"state path must not be a symlink or reparse point: {path}")
    if path.exists() and path.resolve().parent != output_dir.resolve():
        raise ContractError(f"state path escapes output-dir: {path}")


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
    parser.add_argument("--family-catalog", help="Optional source-backed algorithm family catalog; requires candidate-families.")
    parser.add_argument("--candidate-families", help="Explicit candidate fingerprint-to-family profiles; requires family-catalog.")
    parser.add_argument("--output-dir", required=True, help="Local output directory; owns its journal.jsonl.")
    parser.add_argument("--preflight", action="store_true", help="Validate all inputs and proposal gates without publishing or journaling.")
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
        if bool(args.family_catalog) != bool(args.candidate_families):
            raise ContractError("family-catalog and candidate-families must be supplied together")
        family_catalog = load_json(Path(args.family_catalog)) if args.family_catalog else None
        candidate_families = load_json(Path(args.candidate_families)) if args.candidate_families else None
        family_path = None
        journal_path = output_dir / "journal.jsonl"
        _reject_state_path_escape(journal_path, output_dir)
        config = load_json(Path(args.config))
        strict = os.environ.get("STRATEGY_RESEARCH_STRICT_PREFLIGHT") == "1"
        if strict:
            from research.strategy_discovery.contracts import parse_timestamp
            if parse_timestamp(config["as_of"], "config.as_of") > dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=5):
                raise ContractError("scheduled research as_of is in the future; read the actual UTC clock")
            from scripts.strategy_research_checkpoint import active_output
            if output_dir != active_output(ROOT):
                raise ContractError("scheduled research must use the active journal checkpoint")
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
        roots_overlap = False
        for child, parent in (
            (artifact_root.resolve(), output_dir),
            (output_dir, artifact_root.resolve()),
        ):
            try:
                child.relative_to(parent)
            except ValueError:
                continue
            roots_overlap = True
        if roots_overlap:
            raise ContractError(
                "output-dir and approved validation-artifact root must be disjoint"
            )
        if isinstance(validation_artifacts, dict):
            artifact_paths = [
                (artifact_root / artifact["artifact_path"]).resolve()
                for artifact in validation_artifacts.get("artifacts", [])
                if isinstance(artifact, dict) and isinstance(artifact.get("artifact_path"), str)
            ]
            _reject_input_output_collisions(artifact_paths, output_dir)

        # Use the journal's single writer lock so CLI transactions and any
        # direct append_events caller cannot allocate from the same head.
        transaction_lock = journal_path.with_suffix(journal_path.suffix + ".lock")
        with exclusive_lock(
            transaction_lock,
            timeout_seconds=lock_timeout_seconds,
        ) as journal_lock:
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
            # Validate the entire prospective transaction before publishing
            # immutable reports, including capture-identity conflicts.
            append_events(journal_path, events, recorded_at=report["as_of"],
                          lock=journal_lock, dry_run=True)
            if args.preflight or strict:
                malformed = [c for c in report["candidates"] if c["disposition"] == "NEEDS_SPEC"]
                if malformed:
                    raise ContractError("proposal preflight failed before journal writes: " +
                                        "; ".join(str(c.get("name", c["fingerprint"])) + ": " +
                                                  ", ".join(g["gate"] + "=" + g["reason"] for g in c["gates"] if g["status"] == "FAIL")
                                                  for c in malformed))
            if args.preflight:
                print(f"PREFLIGHT OK: {report['summary']['candidate_count']} candidate(s); no journal or report written")
                return 0
            assessment = assess_family_fit(report, family_catalog, candidate_families) if family_catalog is not None else None
            paths, bundle_manifest = publish_immutable_bundle(output_dir, report)
            if assessment is not None:
                family_path = publish_family_fit(output_dir, assessment)
            appended = append_events(
                journal_path,
                events,
                recorded_at=report["as_of"],
                lock=journal_lock,
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
    print(f"family fit: {family_path}" if family_path else "family fit: not assessed (no algorithm catalog/profiles supplied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
