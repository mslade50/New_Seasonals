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
)
from research.strategy_discovery.journal import (
    append_events,
    load_journal,
)
from research.strategy_discovery.pipeline import run_discovery
from research.strategy_discovery.render import write_report_bundle


def _local_output_dir(raw: str) -> Path:
    if "://" in raw:
        raise ContractError("output-dir must be a local filesystem path")
    return Path(raw).resolve()


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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_dir = _local_output_dir(args.output_dir)
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
        )
        appended = append_events(
            journal_path,
            events,
            recorded_at=report["as_of"],
        )
        paths = write_report_bundle(output_dir, report)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
