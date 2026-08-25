"""Run the EP process in shadow mode from a timestamped snapshot JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from episodic_pivot.config import DEFAULT_POLICY  # noqa: E402
from episodic_pivot.manifest import sha256_file, write_run_artifacts  # noqa: E402
from episodic_pivot.news import (  # noqa: E402
    GoogleCustomSearchProvider,
    GoogleNewsRssProvider,
)
from episodic_pivot.pipeline import run_shadow_pipeline  # noqa: E402
from episodic_pivot.schema import NewsDocument, PremarketSnapshot, parse_timestamp  # noqa: E402


_NY = ZoneInfo("America/New_York")


def _load_snapshots(path: Path) -> list[PremarketSnapshot]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("snapshots", []) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError("snapshot JSON must be a list or {'snapshots': [...]} object")
    return [PremarketSnapshot.from_dict(row) for row in rows]


def _load_documents(path: Path | None) -> dict[str, list[NewsDocument]] | None:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("evidence JSON must map ticker to a list of documents")
    return {
        str(symbol).upper(): [NewsDocument.from_dict(item) for item in items]
        for symbol, items in raw.items()
    }


def _verify_evidence_manifest(evidence: Path, manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = str(manifest.get("run_id", ""))
    provider = str(manifest.get("search_provider", ""))
    if not run_id or manifest_path.parent.name != run_id:
        raise ValueError("evidence manifest path/run_id mismatch")
    if not provider or provider.upper().startswith("OFFLINE"):
        raise ValueError("evidence provenance must originate from a network research run")
    expected = (
        (manifest.get("artifacts") or {})
        .get("evidence_by_symbol.json", {})
        .get("sha256")
    )
    if not expected or expected != sha256_file(evidence):
        raise ValueError("evidence digest does not match the source run manifest")
    safety = manifest.get("safety") or {}
    if safety.get("live_actions_enabled") is not False:
        raise ValueError("source evidence manifest does not prove shadow mode")
    return run_id


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EP shadow research: candidate -> news -> decision -> local review artifact"
    )
    parser.add_argument(
        "--snapshot",
        required=True,
        type=Path,
        action="append",
        help="normalized snapshot JSON; repeat to merge after-hours and premarket observations",
    )
    parser.add_argument("--evidence", type=Path, help="offline actual-document evidence JSON")
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        help="manifest.json from the network run that produced --evidence",
    )
    parser.add_argument(
        "--news-mode",
        choices=("offline", "google-news", "google-cse"),
        default="offline",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="required for a Google news mode; permits read-only search/fetch requests",
    )
    parser.add_argument("--as-of", help="timezone-aware decision timestamp")
    parser.add_argument("--target-session-date", help="regular-session date under review")
    parser.add_argument(
        "--run-research",
        action="store_true",
        help="run research and write local artifacts; never enables broker or production actions",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "episodic_pivot",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.news_mode != "offline" and args.run_research and not args.allow_network:
        raise SystemExit("Google news modes require the explicit --allow-network flag")
    if args.news_mode == "offline" and args.evidence is None:
        print("Note: offline mode without --evidence will leave every catalyst unconfirmed.")
    if args.evidence_manifest and not args.evidence:
        raise SystemExit("--evidence-manifest requires --evidence")

    snapshots = [
        snapshot
        for snapshot_path in args.snapshot
        for snapshot in _load_snapshots(snapshot_path)
    ]
    as_of = args.as_of or datetime.now(timezone.utc)
    as_of_dt = parse_timestamp(as_of)
    snapshot_dates = {
        snapshot.target_session_date
        for snapshot in snapshots
        if snapshot.target_session_date
    }
    target_session_date = (
        args.target_session_date
        or (next(iter(snapshot_dates)) if len(snapshot_dates) == 1 else None)
        or as_of_dt.astimezone(_NY).date().isoformat()
    )
    if snapshot_dates and snapshot_dates != {target_session_date}:
        raise SystemExit(
            "--target-session-date must match every snapshot target_session_date"
        )

    documents = _load_documents(args.evidence)
    evidence_source_run_id = None
    if args.evidence_manifest:
        try:
            evidence_source_run_id = _verify_evidence_manifest(
                args.evidence.resolve(), args.evidence_manifest.resolve()
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid evidence provenance: {exc}") from exc
    elif documents is not None:
        print(
            "Warning: offline evidence has no verified network-run manifest; "
            "it will remain UNVERIFIED_REPLAY and cannot create a preview."
        )
    if not args.run_research:
        print(
            f"Dry run: validated {len(snapshots)} snapshot row(s) for "
            f"{target_session_date}; planned news mode={args.news_mode}."
        )
        print("No network request or file write was performed. Use --run-research for local research artifacts.")
        return 0
    provider = None
    if args.news_mode == "google-news":
        provider = GoogleNewsRssProvider()
    elif args.news_mode == "google-cse":
        provider = GoogleCustomSearchProvider()

    result = run_shadow_pipeline(
        snapshots,
        as_of=as_of_dt,
        target_session_date=target_session_date,
        policy=DEFAULT_POLICY,
        offline_documents=documents,
        offline_documents_verified=bool(evidence_source_run_id),
        search_provider=provider,
    )
    output_root = args.output_root.resolve()
    allowed_root = (ROOT / "artifacts").resolve()
    if output_root != allowed_root and allowed_root not in output_root.parents:
        raise SystemExit("--output-root must stay under this worktree's artifacts directory")
    run_dir = output_root / result.run_id
    input_files = {
        f"snapshot_{index}": path
        for index, path in enumerate(args.snapshot, start=1)
    }
    if args.evidence:
        input_files["evidence"] = args.evidence
    if args.evidence_manifest:
        input_files["evidence_manifest"] = args.evidence_manifest
    written = write_run_artifacts(
        result,
        policy=DEFAULT_POLICY,
        output_dir=run_dir,
        input_files=input_files,
        search_provider=(
            provider.name
            if provider
            else (
                f"OFFLINE_VERIFIED:{evidence_source_run_id}"
                if evidence_source_run_id
                else "OFFLINE_UNVERIFIED"
            )
        ),
    )
    print(
        f"{result.run_id}: {len(result.candidates)} nomination(s), "
        f"{len(result.previews)} research sizing preview(s)"
    )
    print(f"Review artifacts: {written}")
    print("Safety: no broker, Sheets, R2, schedule, or production write was attempted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
