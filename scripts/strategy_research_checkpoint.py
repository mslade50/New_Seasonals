"""Preserved-checkpoint replay for a final, unvalidated normalization failure.

The old journal, capture, and cursors are never changed. Recovery creates a new
journal branch at the last committed research checkpoint and records the full
original journal hash, omitted failed transaction, pending bundle, and reason.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_io import file_lock, read_json, write_json
from research.strategy_discovery.contracts import ContractError
from research.strategy_discovery.journal import load_journal, latest_source_captures


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    path.relative_to(base.resolve())
    return path


def active_output(root: Path = ROOT) -> Path:
    base = root / "artifacts/strategy_discovery"
    pointer = base / "active_checkpoint.json"
    if not pointer.exists():
        return (base / "daily").resolve()
    record = read_json(pointer)
    if record.get("schema_version") != "strategy-research-checkpoint.v1":
        raise ContractError("unsupported research checkpoint")
    source = _inside(base, record["source_journal"])
    if _sha(source) != record["source_journal_sha256"]:
        raise ContractError("preserved research journal changed after checkpoint recovery")
    output = _inside(base, record["output_dir"])
    journal = load_journal(output / "journal.jsonl")
    count = record["checkpoint_sequence"]
    if len(journal) < count or (count and journal[count - 1]["record_hash"] != record["checkpoint_hash"]):
        raise ContractError("recovery journal no longer extends its preserved checkpoint")
    return output


def recovery_prefix(records, pending_capture_ids, accepted):
    starts = [i for i, record in enumerate(records) if record["event_type"] == "RUN"]
    if not starts:
        raise ContractError("no failed transaction to recover")
    cut = starts[-1]
    failed = records[cut:]
    sources = [r["payload"] for r in failed if r["event_type"] == "SOURCE_CAPTURE"]
    if {r["capture_id"] for r in sources} != set(pending_capture_ids):
        raise ContractError("last transaction is not the pending capture")
    candidates = [r["payload"] for r in failed if r["event_type"] == "CANDIDATE_OBSERVED"]
    if not candidates or any(r["lifecycle"] != "DISCOVERED" or r["disposition"] != "NEEDS_SPEC" for r in candidates):
        raise ContractError("only an unvalidated NEEDS_SPEC transaction can be replayed")
    if any(r["event_type"] not in {"RUN", "SOURCE_CAPTURE", "CANDIDATE_OBSERVED"} for r in failed):
        raise ContractError("cannot recover a transaction containing validation or owner authority")
    anchors = latest_source_captures(records[:cut])
    if set(anchors) != set(accepted):
        raise ContractError("checkpoint sources do not match accepted collector cursors")
    for source, cursor in accepted.items():
        anchor = anchors[source]
        if (anchor["capture_id"] != cursor["capture_id"] or anchor["cursor_out"] != cursor["cursor_out"]
                or anchor["window"]["end"] != cursor["window_end"]):
            raise ContractError("checkpoint would move accepted source cursors")
    return cut


def recover(root: Path, expected_bundle: str, reason: str) -> Path:
    from scripts.collect_strategy_sources import _load_state, _pending_capture, _read_bundle
    if not reason.strip():
        raise ContractError("an operator recovery reason is required")
    base = root / "artifacts/strategy_discovery"
    state_path = root / "data/strategy_source_cursors.json"
    # Same runner lock as scheduled research; no recovery under an active agent.
    with file_lock(root / "artifacts/strategy_research_agent/runner"), file_lock(state_path):
        state = _load_state(state_path)
        capture = _pending_capture(state, base / "source_captures")
        if capture is None or state["pending"]["bundle_digest"] != expected_bundle:
            raise ContractError("pending source bundle changed; recovery refused")
        manifest, *_ = _read_bundle(capture)
        old = active_output(root) / "journal.jsonl"
        records = load_journal(old)
        cut = recovery_prefix(records, [s["capture_id"] for s in manifest["sources"]], state["accepted"])
        decision = root / "data/strategy_research/latest_decision.json"
        if decision.exists() and read_json(decision).get("source_bundle_digest") == expected_bundle:
            raise ContractError("capture already has a decision; verify delivery before any recovery")
        original_bytes = old.read_bytes()
        output = base / ("daily-recovery-" + uuid.uuid4().hex[:12])
        output.mkdir(parents=True, exist_ok=False)
        prefix = b"".join([line for line in original_bytes.splitlines(keepends=True) if line.strip()][:cut])
        with (output / "journal.jsonl").open("xb") as handle:
            handle.write(prefix)
            handle.flush()
            import os
            os.fsync(handle.fileno())
        load_journal(output / "journal.jsonl")
        record = {"schema_version": "strategy-research-checkpoint.v1",
                  "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "reason": reason,
                  "pending_bundle": expected_bundle, "collector_state": state,
                  "source_journal": str(old.relative_to(base)),
                  "source_journal_sha256": hashlib.sha256(original_bytes).hexdigest(),
                  "checkpoint_sequence": cut,
                  "checkpoint_hash": records[cut - 1]["record_hash"] if cut else "GENESIS",
                  "failed_transaction": records[cut:], "output_dir": output.name}
        write_json(output / "recovery.json", record)
        write_json(base / "active_checkpoint.json", record)
        assert active_output(root) == output.resolve()
        print(f"Recovered research checkpoint: {output}; original journal and collector cursors preserved")
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover-bundle")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    try:
        if args.recover_bundle:
            recover(ROOT, args.recover_bundle, args.reason)
        else:
            print(active_output())
    except (OSError, ValueError, KeyError, TypeError, ContractError) as exc:
        print(f"Research checkpoint refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
