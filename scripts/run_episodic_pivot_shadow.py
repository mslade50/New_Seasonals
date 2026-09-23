"""Run the EP process in shadow mode from a timestamped snapshot JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from episodic_pivot.config import DEFAULT_POLICY
from episodic_pivot.daily_prices import YFINANCE_DAILY_PRICE_BASIS
from episodic_pivot.manifest import sha256_file, write_run_artifacts
from episodic_pivot.news import (
    DirectArticleSearchProvider,
    GoogleCustomSearchProvider,
    GoogleNewsRssProvider,
)
from episodic_pivot.pipeline import run_shadow_pipeline
from episodic_pivot.reviewed_news import MODE, apply_review, make_queue
from episodic_pivot.schema import (
    NewsDocument,
    PremarketSnapshot,
    parse_timestamp,
)

_NY = ZoneInfo("America/New_York")
_YFINANCE_RECORD_TYPE = "EP_YFINANCE_DAILY_ENRICHMENT_V1"
_IBKR_RECORD_TYPE = "EP_IBKR_PREMARKET_CAPTURE_V1"
_REFRESH_TARGET_RECORD_TYPE = "EP_RESEARCH_QUOTE_REFRESH_TARGETS_V1"


def _valid_sha256(value: object) -> bool:
    token = str(value or "").strip().lower()
    return len(token) == 64 and all(
        character in "0123456789abcdef" for character in token
    )


def _verify_input_hashes(records: object) -> None:
    if not isinstance(records, list) or not records:
        raise ValueError("snapshot wrapper is missing source input provenance")
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("snapshot input provenance records must be objects")
        source = Path(str(record.get("path") or ""))
        expected = record.get("sha256")
        if (
            not source.is_absolute()
            or not source.is_file()
            or not _valid_sha256(expected)
        ):
            raise ValueError("snapshot source input provenance is incomplete")
        if sha256_file(source) != str(expected).lower():
            raise ValueError("snapshot source input digest mismatch")


def _validate_yfinance_wrapper(raw: dict, snapshots: list[PremarketSnapshot]) -> None:
    request = raw.get("request")
    safety = raw.get("safety")
    if (
        raw.get("provider") != "YFINANCE"
        or raw.get("mode") != "YFINANCE_DAILY_RESEARCH_ONLY"
        or raw.get("daily_price_basis") != YFINANCE_DAILY_PRICE_BASIS
        or not isinstance(request, dict)
        or request.get("auto_adjust") is not True
        or request.get("repair") is not True
        or request.get("event_session_excluded") is not True
        or request.get("local_price_cache_used") is not False
        or not isinstance(safety, dict)
        or safety.get("research_only") is not True
        or safety.get("broker_contacted") is not False
        or safety.get("order_submission_allowed") is not False
        or safety.get("order_staging_performed") is not False
    ):
        raise ValueError("yfinance snapshot wrapper provenance is invalid")
    _verify_input_hashes(raw.get("inputs"))
    target_date = str(raw.get("target_session_date") or "")
    if not target_date or any(
        row.target_session_date != target_date for row in snapshots
    ):
        raise ValueError("yfinance snapshot target-session provenance is invalid")


def _is_positive_port(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 65535


def _validate_frozen_premarket_verification(
    raw_row: dict, snapshot: PremarketSnapshot
) -> None:
    keys = {
        "premarket_move_verification_status",
        "premarket_move_verification_source",
        "premarket_move_verified_at",
    }
    present = keys.intersection(raw_row)
    if not present:
        return
    if present != keys:
        raise ValueError("IBKR row premarket verification evidence is incomplete")
    source = str(raw_row["premarket_move_verification_source"] or "").strip()
    if (
        str(raw_row["premarket_move_verification_status"] or "").strip() != "VERIFIED"
        or source != snapshot.source
    ):
        raise ValueError("IBKR row premarket verification evidence is invalid")
    try:
        verified_at = datetime.fromisoformat(
            str(raw_row["premarket_move_verified_at"]).strip().replace("Z", "+00:00")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "IBKR row premarket verification timestamp is invalid"
        ) from exc
    if verified_at.tzinfo is None or verified_at.utcoffset() != timedelta(0):
        raise ValueError("IBKR row premarket verification timestamp must be UTC")


def _load_hashed_manifest_record(record: object) -> tuple[str, Path, dict]:
    if not isinstance(record, dict):
        raise TypeError("IBKR source manifest records must be objects")
    run_id = str(record.get("run_id") or "").strip()
    manifest_path = Path(str(record.get("path") or ""))
    expected_sha = str(record.get("sha256") or "").strip().lower()
    if (
        not run_id
        or not manifest_path.is_absolute()
        or not manifest_path.is_file()
        or manifest_path.name != "manifest.json"
        or manifest_path.resolve().parent.name != run_id
        or not _valid_sha256(expected_sha)
        or sha256_file(manifest_path) != expected_sha
    ):
        raise ValueError("IBKR source manifest path or digest is invalid")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("IBKR source manifest must be an object")
    safety = manifest.get("safety")
    provider = str(manifest.get("search_provider") or "").strip().upper()
    if (
        manifest.get("schema_version") != 2
        or manifest.get("run_id") != run_id
        or not provider
        or provider.startswith("OFFLINE")
        or not isinstance(safety, dict)
        or safety.get("research_only") is not True
        or safety.get("live_actions_enabled") is not False
        or str(safety.get("broker_route") or "").strip().upper() != "NONE"
        or safety.get("order_submission_allowed") is not False
        or safety.get("order_staging_performed") is not False
        or safety.get("broker_contacted") is not False
        or safety.get("sheets_written") is not False
        or safety.get("r2_written") is not False
        or safety.get("publishing_performed") is not False
        or safety.get("production_deployed") is not False
    ):
        raise ValueError(
            "IBKR source manifest identity, network provenance, or safety is invalid"
        )
    return run_id, manifest_path.resolve(), manifest


def _validate_research_refresh_chain(raw: dict, target_date: str) -> None:
    inputs = raw.get("inputs")
    if not isinstance(inputs, list):
        raise TypeError("IBKR snapshot inputs must be a list")
    refresh_inputs = [
        record
        for record in inputs
        if isinstance(record, dict)
        and str(record.get("record_type") or "").strip().upper()
        == _REFRESH_TARGET_RECORD_TYPE
    ]
    if not refresh_inputs:
        return

    manifest_records = raw.get("source_manifests")
    if not isinstance(manifest_records, list) or not manifest_records:
        raise ValueError("research refresh IBKR wrapper requires source_manifests")
    manifests: dict[str, tuple[Path, dict]] = {}
    for record in manifest_records:
        run_id, manifest_path, manifest = _load_hashed_manifest_record(record)
        if run_id in manifests:
            raise ValueError("duplicate IBKR source manifest run_id")
        manifests[run_id] = (manifest_path, manifest)

    referenced_run_ids: set[str] = set()
    for input_record in refresh_inputs:
        target_path = Path(str(input_record.get("path") or ""))
        target = json.loads(target_path.read_text(encoding="utf-8"))
        if not isinstance(target, dict):
            raise TypeError("research refresh target must be an object")
        source_run_id = str(target.get("source_run_id") or "").strip()
        if (
            target.get("schema_version") != 1
            or target.get("record_type") != _REFRESH_TARGET_RECORD_TYPE
            or target.get("target_session_date") != target_date
            or not source_run_id.startswith(f"EP-RUN-{target_date}-")
            or target.get("research_only") is not True
            or str(target.get("broker_route") or "").strip().upper() != "NONE"
            or target.get("order_submission_allowed") is not False
        ):
            raise ValueError(
                "research refresh target identity, session, or safety is invalid"
            )
        source_record = manifests.get(source_run_id)
        if source_record is None:
            raise ValueError("research refresh target has no matching source manifest")
        manifest_path, manifest = source_record
        artifact = (manifest.get("artifacts") or {}).get("refresh_targets.json")
        if not isinstance(artifact, dict):
            raise TypeError("source manifest is missing refresh_targets.json")
        if (
            target_path.resolve() != manifest_path.parent / "refresh_targets.json"
            or artifact.get("sha256") != sha256_file(target_path)
            or artifact.get("size_bytes") != target_path.stat().st_size
        ):
            raise ValueError("research refresh target artifact provenance is invalid")
        referenced_run_ids.add(source_run_id)

    if referenced_run_ids != set(manifests):
        raise ValueError(
            "IBKR source manifests do not exactly match refresh target runs"
        )


def _validate_ibkr_wrapper(
    raw: dict,
    raw_rows: list[dict],
    snapshots: list[PremarketSnapshot],
) -> None:
    connection = raw.get("connection")
    coverage = raw.get("coverage")
    selected_port = (
        connection.get("selected_port") if isinstance(connection, dict) else None
    )
    if (
        raw.get("record_type") != _IBKR_RECORD_TYPE
        or raw.get("provider") != "IBKR"
        or raw.get("mode") != "IBKR_READ_ONLY_SHADOW"
        or not isinstance(connection, dict)
        or connection.get("connected") is not True
        or connection.get("readonly_requested") is not True
        or connection.get("readonly") is not True
        or not _is_positive_port(selected_port)
        or connection.get("port") != selected_port
        or not isinstance(coverage, dict)
        or coverage.get("mode") != "TARGETED_TRADINGVIEW_CANDIDATES"
    ):
        raise ValueError("IBKR snapshot wrapper provenance is invalid")
    _verify_input_hashes(raw.get("inputs"))
    target_date = str(raw.get("target_session_date") or "")
    if not target_date or any(
        row.target_session_date != target_date for row in snapshots
    ):
        raise ValueError("IBKR snapshot target-session provenance is invalid")
    _validate_research_refresh_chain(raw, target_date)
    for raw_row, snapshot in zip(raw_rows, snapshots, strict=True):
        if (
            snapshot.provider != "IBKR"
            or snapshot.source != "IBKR_TARGETED_READ_ONLY"
            or snapshot.session != "premarket"
            or snapshot.target_session_date != target_date
        ):
            raise ValueError("IBKR snapshot row provenance is invalid")
        _validate_frozen_premarket_verification(raw_row, snapshot)


def _load_snapshots(path: Path) -> tuple[list[PremarketSnapshot], str, tuple[str, ...]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("snapshots", []) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise TypeError("snapshot JSON must be a list or {'snapshots': [...]} object")
    snapshots = [PremarketSnapshot.from_dict(row) for row in rows]
    wrapper_warnings = raw.get("warnings", []) if isinstance(raw, dict) else []
    if not isinstance(wrapper_warnings, list) or any(
        not isinstance(item, str) or not item.strip() for item in wrapper_warnings
    ):
        raise ValueError("snapshot wrapper warnings are invalid")
    warnings = {item.strip().upper() for item in wrapper_warnings}
    record_type = str(raw.get("record_type") or "") if isinstance(raw, dict) else ""
    if record_type == _YFINANCE_RECORD_TYPE:
        _validate_yfinance_wrapper(raw, snapshots)
        return snapshots, "YFINANCE", tuple(sorted(warnings))
    if record_type == _IBKR_RECORD_TYPE:
        if any(not isinstance(row, dict) for row in rows):
            raise TypeError("IBKR snapshot rows must be objects")
        _validate_ibkr_wrapper(raw, rows, snapshots)
        coverage = raw.get("coverage") or {}
        if coverage.get("input_candidate_complete") is not True:
            warnings.add("IBKR_PARTIAL_EXECUTION_REFRESH")
        return snapshots, "IBKR", tuple(sorted(warnings))
    return snapshots, "UNVERIFIED", tuple(sorted(warnings))


def _load_documents(path: Path | None) -> dict[str, list[NewsDocument]] | None:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("evidence JSON must map ticker to a list of documents")
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
        raise ValueError(
            "evidence provenance must originate from a network research run"
        )
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
        help=(
            "validated yfinance enrichment; repeat with a targeted read-only "
            "IBKR refresh when available"
        ),
    )
    parser.add_argument(
        "--evidence", type=Path, help="offline actual-document evidence JSON"
    )
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        help="manifest.json from the network run that produced --evidence",
    )
    parser.add_argument(
        "--news-mode",
        choices=("offline", "google-news", "google-cse", "agent-reviewed"),
        default="offline",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="required for a Google news mode; permits read-only search/fetch requests",
    )
    parser.add_argument("--as-of", help="timezone-aware decision timestamp")
    parser.add_argument(
        "--prepare-google-review",
        type=Path,
        help="write a bounded search queue under artifacts; no news requests",
    )
    parser.add_argument(
        "--reviews", type=Path, help="completed search-and-read review packet"
    )
    parser.add_argument(
        "--target-session-date", help="regular-session date under review"
    )
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
    parser.add_argument("--track-morning", action="store_true", help="checkpoint this scheduled morning's immutable research inputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.news_mode in {"google-news", "google-cse"}
        and args.run_research
        and not args.allow_network
    ):
        raise SystemExit("Google news modes require the explicit --allow-network flag")
    if bool(args.reviews) != (args.news_mode == "agent-reviewed"):
        raise SystemExit(
            "--reviews and --news-mode agent-reviewed must be used together"
        )
    if (args.reviews or args.prepare_google_review) and (
        args.evidence or args.evidence_manifest or args.allow_network
    ):
        raise SystemExit(
            "search-and-read mode cannot mix legacy evidence/network modes"
        )
    if args.prepare_google_review and (args.reviews or args.news_mode != "offline"):
        raise SystemExit("queue preparation requires offline mode without reviews")
    review_packet = (
        json.loads(args.reviews.read_text(encoding="utf-8")) if args.reviews else None
    )
    if args.news_mode == "offline" and args.evidence is None:
        print(
            "Note: offline mode without --evidence will leave every catalyst unconfirmed."
        )
    if args.evidence_manifest and not args.evidence:
        raise SystemExit("--evidence-manifest requires --evidence")

    loaded = [
        _load_snapshots(snapshot_path.resolve()) for snapshot_path in args.snapshot
    ]
    snapshots = [snapshot for rows, _kind, _warnings in loaded for snapshot in rows]
    snapshot_kinds = {kind for _rows, kind, _warnings in loaded}
    run_warnings = tuple(
        sorted({warning for _rows, _kind, warnings in loaded for warning in warnings})
    )
    if args.run_research and "YFINANCE" not in snapshot_kinds:
        raise SystemExit(
            "research runs require a validated EP_YFINANCE_DAILY_ENRICHMENT_V1 snapshot"
        )
    if args.run_research and "UNVERIFIED" in snapshot_kinds:
        raise SystemExit("research runs reject UNVERIFIED snapshot inputs")
    as_of = args.as_of or datetime.now(timezone.utc)
    as_of_dt = parse_timestamp(as_of)
    scan_at = (
        parse_timestamp(review_packet["queue"]["prepared_at"])
        if review_packet
        else as_of_dt
    )
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
        print(
            "No network request or file write was performed. Use --run-research for local research artifacts."
        )
        return 0
    provider = None
    if args.news_mode == "google-news":
        provider = GoogleNewsRssProvider()
    elif args.news_mode == "google-cse":
        provider = GoogleCustomSearchProvider()
    if provider is not None:
        provider = DirectArticleSearchProvider(
            provider,
            metadata_cache=str(
                ROOT / "artifacts" / "episodic_pivot" / "yfinance-news-metadata"
            ),
        )

    result = run_shadow_pipeline(
        snapshots,
        as_of=scan_at,
        target_session_date=target_session_date,
        policy=DEFAULT_POLICY,
        offline_documents=documents,
        offline_documents_verified=bool(evidence_source_run_id),
        search_provider=provider,
        run_warnings=run_warnings,
    )
    if args.prepare_google_review:
        path = args.prepare_google_review.resolve()
        if (ROOT / "artifacts").resolve() not in path.parents:
            raise SystemExit("review queue must stay under this worktree's artifacts")
        queue = make_queue(
            result.candidates,
            prepared_at=result.generated_at,
            target_session_date=target_session_date,
            policy=DEFAULT_POLICY,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(queue, handle, indent=2)
        if args.track_morning:
            from episodic_pivot.morning_completion import checkpoint
            checkpoint(ROOT / "artifacts" / "episodic_pivot", target_session_date, "RESEARCH",
                       {"queue": path, **{f"snapshot_{i}": p for i, p in enumerate(args.snapshot, 1)}})
        print(
            f"Google research queue prepared: {len(queue['targets'])} targets; no news or email sent."
        )
        return 0
    if review_packet is not None:
        result = apply_review(
            result, review_packet, decision_at=as_of_dt, policy=DEFAULT_POLICY
        )
    output_root = args.output_root.resolve()
    allowed_root = (ROOT / "artifacts").resolve()
    if output_root != allowed_root and allowed_root not in output_root.parents:
        raise SystemExit(
            "--output-root must stay under this worktree's artifacts directory"
        )
    run_dir = output_root / result.run_id
    input_files = {
        f"snapshot_{index}": path for index, path in enumerate(args.snapshot, start=1)
    }
    if args.evidence:
        input_files["evidence"] = args.evidence
    if args.evidence_manifest:
        input_files["evidence_manifest"] = args.evidence_manifest
    if args.reviews:
        input_files["agent_reviews"] = args.reviews
    written = write_run_artifacts(
        result,
        policy=DEFAULT_POLICY,
        output_dir=run_dir,
        input_files=input_files,
        search_provider=(
            MODE
            if review_packet is not None
            else provider.name
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
    if args.track_morning:
        from episodic_pivot.morning_completion import checkpoint
        checkpoint(ROOT / "artifacts" / "episodic_pivot", target_session_date, "REPORT_READY",
                   {"report": run_dir / "manifest.json", **({"reviews": args.reviews} if args.reviews else {})})
    print("Safety: no broker, Sheets, R2, schedule, or production write was attempted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
