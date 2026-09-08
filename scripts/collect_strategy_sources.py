"""Capture bounded X/SSRN inputs without advancing cursors prematurely.

``collect`` writes an immutable source bundle and records it as pending.
``acknowledge`` advances only complete source cursors after the discovery run
has durably accepted that exact bundle.  A pending bundle is replayed on the
next collection attempt, which prevents an agent failure from skipping posts
or papers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.strategy_discovery.contracts import (
    canonical_json,
    load_json,
    load_jsonl,
    validate_config,
    validate_item,
    validate_manifest,
)
from research.strategy_discovery.journal import load_journal
from research.strategy_discovery.pipeline import _capture_digest
from research.strategy_discovery.source_collectors import (
    CollectorError,
    accepted_from_manifest,
    bundle_digest,
    collect_bundle,
    empty_state,
    validate_source_config,
    validate_state,
)
from research_io import file_lock, read_json, write_json

DEFAULT_CONFIG = ROOT / "config" / "strategy_research_sources.json"
DEFAULT_STATE = ROOT / "data" / "strategy_source_cursors.json"
DEFAULT_CAPTURE_ROOT = ROOT / "artifacts" / "strategy_discovery" / "source_captures"
DEFAULT_JOURNAL = ROOT / "artifacts" / "strategy_discovery" / "daily" / "journal.jsonl"
DEFAULT_DECISION = ROOT / "data" / "strategy_research" / "latest_decision.json"
FILES = {
    "manifest": "source_manifest.json",
    "items": "items.raw.jsonl",
    "config": "discovery_config.json",
    "telemetry": "telemetry.json",
}


def _load_environment(env_file: Path | None) -> dict[str, str]:
    values: dict[str, str] = {}
    candidate = env_file or ROOT / ".env"
    if candidate.exists():
        from dotenv import dotenv_values

        values.update({str(k): str(v) for k, v in dotenv_values(candidate).items() if v is not None})
    values.update(os.environ)
    return values


def _load_state(path: Path) -> dict:
    if not path.exists():
        return empty_state()
    return validate_state(read_json(path))


def _write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _read_bundle(capture_dir: Path) -> tuple[dict, list[dict], dict, dict, str]:
    manifest = load_json(capture_dir / FILES["manifest"])
    items = load_jsonl(capture_dir / FILES["items"])
    config = load_json(capture_dir / FILES["config"])
    telemetry = load_json(capture_dir / FILES["telemetry"])
    validate_manifest(manifest)
    validate_config(config)
    for index, item in enumerate(items):
        validate_item(item, index)
    digest = bundle_digest(manifest, items, config, telemetry)
    if capture_dir.name != digest:
        raise CollectorError("capture directory does not match its immutable bundle digest")
    return manifest, items, config, telemetry, digest


def _pending_capture(state: dict, capture_root: Path) -> Path | None:
    pending = state["pending"]
    if pending is None:
        return None
    candidate = (capture_root / pending["capture_dir"]).resolve()
    try:
        candidate.relative_to(capture_root.resolve())
    except ValueError as exc:
        raise CollectorError("pending capture path escapes the approved capture root") from exc
    *_, digest = _read_bundle(candidate)
    if digest != pending["bundle_digest"]:
        raise CollectorError("pending capture digest mismatch; evidence preserved")
    return candidate


def collect(config_path: Path, state_path: Path, capture_root: Path, env_file: Path | None) -> Path:
    with file_lock(state_path):
        state = _load_state(state_path)
        pending = _pending_capture(state, capture_root)
        if pending is not None:
            print(f"PENDING {pending}")
            return pending
        source_config = validate_source_config(load_json(config_path))
        manifest, items, config, telemetry = collect_bundle(
            source_config,
            state,
            environment=_load_environment(env_file),
        )
        digest = bundle_digest(manifest, items, config, telemetry)
        capture_dir = capture_root / digest
        if capture_dir.exists():
            _read_bundle(capture_dir)
        else:
            capture_dir.mkdir(parents=True, exist_ok=False)
            _write_text_exclusive(capture_dir / FILES["manifest"], canonical_json(manifest) + "\n")
            _write_text_exclusive(
                capture_dir / FILES["items"],
                "".join(canonical_json(item) + "\n" for item in items),
            )
            _write_text_exclusive(capture_dir / FILES["config"], canonical_json(config) + "\n")
            _write_text_exclusive(capture_dir / FILES["telemetry"], canonical_json(telemetry) + "\n")
            _read_bundle(capture_dir)
        state["pending"] = {
            "capture_dir": capture_dir.name,
            "bundle_digest": digest,
            "created_at": config["as_of"],
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(state_path, state)
    print(f"CAPTURED {capture_dir}")
    return capture_dir


def _normalized_items_for_capture(raw_items: list[dict], normalized_path: Path | None) -> list[dict]:
    """Load journal inputs while proving they preserve the immutable capture.

    Discovery may add only ``claims`` and ``strategy_proposal``.  Every native
    source field must remain byte-for-byte equivalent after canonical JSON
    normalization, otherwise acknowledgement fails closed.
    """
    if normalized_path is None:
        return raw_items
    normalized_items = load_jsonl(normalized_path)
    for index, item in enumerate(normalized_items):
        validate_item(item, index)

    def by_id(items: list[dict], label: str) -> dict[str, dict]:
        indexed: dict[str, dict] = {}
        for item in items:
            item_id = item["item_id"]
            if item_id in indexed:
                raise CollectorError(f"{label} items contain duplicate item_id {item_id}")
            indexed[item_id] = item
        return indexed

    raw_by_id = by_id(raw_items, "raw")
    normalized_by_id = by_id(normalized_items, "normalized")
    if raw_by_id.keys() != normalized_by_id.keys():
        raise CollectorError("normalized items do not contain the exact immutable capture item set")
    enrichment_fields = {"claims", "strategy_proposal"}
    for item_id, raw_item in raw_by_id.items():
        normalized_item = normalized_by_id[item_id]
        raw_native = {key: value for key, value in raw_item.items() if key not in enrichment_fields}
        normalized_native = {
            key: value for key, value in normalized_item.items() if key not in enrichment_fields
        }
        if canonical_json(raw_native) != canonical_json(normalized_native):
            raise CollectorError(
                f"normalized item {item_id} changed an immutable source field"
            )
    return normalized_items


def _verify_capture_journaled(
    manifest: dict,
    items: list[dict],
    journal_path: Path,
) -> dict[str, dict]:
    records = load_journal(journal_path)
    source_events = [record["payload"] for record in records if record["event_type"] == "SOURCE_CAPTURE"]
    matched_events: dict[str, dict] = {}
    for source in manifest["sources"]:
        source_items = [
            item
            for item in items
            if item["source_id"] == source["source_id"] and item["capture_id"] == source["capture_id"]
        ]
        digest = _capture_digest(source, source_items)
        matched = next((
            event
            for event in reversed(source_events)
            if event.get("source_id") == source["source_id"]
            and event.get("capture_id") == source["capture_id"]
            and event.get("capture_digest") == digest
        ), None)
        if matched is None:
            raise CollectorError(
                f"capture is not committed in the discovery journal: {source['source_id']}"
            )
        matched_events[source["source_id"]] = matched
    return matched_events


def acknowledge(
    state_path: Path,
    capture_root: Path,
    capture_dir_raw: Path,
    journal_path: Path = DEFAULT_JOURNAL,
    decision_path: Path = DEFAULT_DECISION,
    normalized_items_path: Path | None = None,
) -> dict:
    with file_lock(state_path):
        state = _load_state(state_path)
        pending_dir = _pending_capture(state, capture_root)
        if pending_dir is None:
            raise CollectorError("no source capture is pending acknowledgement")
        supplied = capture_dir_raw.resolve()
        if supplied != pending_dir:
            raise CollectorError("acknowledgement does not match the pending capture")
        manifest, items, _, _, digest = _read_bundle(supplied)
        if digest != state["pending"]["bundle_digest"]:
            raise CollectorError("acknowledgement digest differs from pending state")
        journal_items = _normalized_items_for_capture(items, normalized_items_path)
        journal_events = _verify_capture_journaled(manifest, journal_items, journal_path)
        sources_complete = all(
            journal_events[source["source_id"]]["status"] == "COMPLETE"
            for source in manifest["sources"]
        )
        if sources_complete:
            if not decision_path.is_file():
                raise CollectorError("complete capture lacks a final research decision")
            decision = read_json(decision_path)
            if (
                decision.get("schema_version") != "strategy-research-email-decision.v1"
                or decision.get("source_bundle_digest") != digest
                or decision.get("positions_used") is not False
            ):
                raise CollectorError("final research decision does not bind the pending source capture")
            delivery = decision.get("delivery_status")
            acceptable = {"SENT", "ALREADY_SENT"} if decision.get("email_required") else {"NO_EMAIL"}
            if delivery not in acceptable:
                raise CollectorError("final research decision lacks a terminal delivery outcome")
        complete_manifest = {
            **manifest,
            "sources": [
                source
                for source in manifest["sources"]
                if journal_events[source["source_id"]]["status"] == "COMPLETE"
            ],
        }
        advances = accepted_from_manifest(complete_manifest, capture_digest=digest)
        state["accepted"].update(advances)
        state["pending"] = None
        write_json(state_path, state)
    print(f"ACKNOWLEDGED {supplied} ({len(advances)} complete source cursor(s) advanced)")
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--journal", type=Path, default=None)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("collect")
    ack = sub.add_parser("acknowledge")
    ack.add_argument("--capture-dir", type=Path, required=True)
    ack.add_argument(
        "--normalized-items",
        type=Path,
        help="normalized JSONL committed to the discovery journal",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "collect":
            collect(args.config, args.state, args.capture_root, args.env_file)
        else:
            from scripts.strategy_research_checkpoint import active_output
            acknowledge(
                args.state,
                args.capture_root,
                args.capture_dir,
                args.journal or active_output(ROOT) / "journal.jsonl",
                args.decision,
                args.normalized_items,
            )
    except (CollectorError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"SOURCE COLLECTION FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
