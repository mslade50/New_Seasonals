import copy
import datetime as dt
import json
from pathlib import Path

import pytest

from research.strategy_discovery.contracts import ContractError
from research.strategy_discovery.journal import append_events, load_journal
from scripts import collect_strategy_sources as collector
from scripts import run_strategy_discovery as cli
from scripts import strategy_research_checkpoint as checkpoint
from tests.test_strategy_discovery import config, manifest, source, item, proposal, catalog, run


def broken_item():
    value = proposal()
    value["entry"].update(order_type="MOO", timing="OPEN", price_rule="Resting limit ladder at fixed ATR spacing")
    return item(proposal_value=value)


def inputs(tmp_path, candidate):
    files = {"config": config(), "source-manifest": manifest(source(count=1, expected=1)),
             "strategy-catalog": catalog("STRATEGY_BOOK"), "dead-end-catalog": catalog("DEAD_ENDS")}
    args = []
    for flag, value in files.items():
        path = tmp_path / (flag + ".json")
        path.write_text(json.dumps(value))
        args += ["--" + flag, str(path)]
    path = tmp_path / "items.jsonl"
    path.write_text(json.dumps(candidate) + "\n")
    base = tmp_path / "artifacts/strategy_discovery"
    return args + ["--items", str(path), "--output-dir", str(base / "daily")], base


def test_preflight_rejects_bad_order_before_any_immutable_write(tmp_path, capsys):
    args, base = inputs(tmp_path, broken_item())
    assert cli.main(args + ["--preflight"], approved_output_root=base) == 2
    assert "ENTRY_EXIT_SPEC" in capsys.readouterr().err
    assert not (base / "daily/journal.jsonl").exists()
    assert not (base / "daily/runs").exists()
    (tmp_path / "items.jsonl").write_text(json.dumps(item()) + "\n")
    assert cli.main(args + ["--preflight"], approved_output_root=base) == 0
    assert not (base / "daily/journal.jsonl").exists()
    assert cli.main(args, approved_output_root=base) == 0


def test_scheduled_mode_enforces_preflight_without_optional_flag(tmp_path, monkeypatch):
    args, base = inputs(tmp_path, broken_item())
    monkeypatch.setenv("STRATEGY_RESEARCH_STRICT_PREFLIGHT", "1")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    assert cli.main(args, approved_output_root=base) == 2
    assert not (base / "daily/journal.jsonl").exists()


def test_scheduled_mode_rejects_a_guessed_future_clock(tmp_path, monkeypatch, capsys):
    args, base = inputs(tmp_path, item())
    cfg = config()
    cfg["as_of"] = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)).isoformat()
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("STRATEGY_RESEARCH_STRICT_PREFLIGHT", "1")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    assert cli.main(args, approved_output_root=base) == 2
    assert "future" in capsys.readouterr().err
    assert not (base / "daily/journal.jsonl").exists()


def seed_failure(tmp_path, monkeypatch):
    base = tmp_path / "artifacts/strategy_discovery"
    old = base / "daily/journal.jsonl"
    report, events = run(items=[broken_item()])
    append_events(old, events, recorded_at=report["as_of"])
    state = {"schema_version": "strategy-source-cursors.v1", "accepted": {},
             "pending": {"capture_dir": "a" * 64, "bundle_digest": "a" * 64, "created_at": report["as_of"]}}
    path = tmp_path / "data/strategy_source_cursors.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(state))
    monkeypatch.setattr(collector, "_pending_capture", lambda *args: base / "source_captures" / ("a" * 64))
    monkeypatch.setattr(collector, "_read_bundle", lambda *args: (manifest(source(count=1, expected=1)), [], {}, {}, "a" * 64))
    return old, path


def test_checkpoint_recovery_preserves_evidence_and_can_replay_corrected_capture(tmp_path, monkeypatch):
    old, state = seed_failure(tmp_path, monkeypatch)
    old_bytes, state_bytes = old.read_bytes(), state.read_bytes()
    new = checkpoint.recover(tmp_path, "a" * 64, "Fix encoding after failed preregistration")
    assert old.read_bytes() == old_bytes
    assert state.read_bytes() == state_bytes
    assert checkpoint.active_output(tmp_path) == new
    report, events = run(items=[item()], journal=load_journal(new / "journal.jsonl"))
    append_events(new / "journal.jsonl", events, recorded_at=report["as_of"])
    assert report["summary"]["new_research_ready"] == 1
    assert checkpoint.active_output(tmp_path) == new
    old.write_bytes(old_bytes + b"\n")
    with pytest.raises(ContractError, match="preserved research journal changed"):
        checkpoint.active_output(tmp_path)


def test_checkpoint_refuses_changed_pending_bundle_and_existing_decision(tmp_path, monkeypatch):
    old, state = seed_failure(tmp_path, monkeypatch)
    with pytest.raises(ContractError, match="pending source bundle changed"):
        checkpoint.recover(tmp_path, "b" * 64, "reason")
    decision = tmp_path / "data/strategy_research/latest_decision.json"
    decision.parent.mkdir()
    decision.write_text(json.dumps({"source_bundle_digest": "a" * 64}))
    with pytest.raises(ContractError, match="already has a decision"):
        checkpoint.recover(tmp_path, "a" * 64, "reason")


def test_checkpoint_refuses_ready_candidates_or_mismatched_accepted_cursors(tmp_path):
    old = tmp_path / "journal.jsonl"
    report, events = run()
    append_events(old, events, recorded_at=report["as_of"])
    with pytest.raises(ContractError, match="NEEDS_SPEC"):
        checkpoint.recovery_prefix(load_journal(old), {"capture-1"}, {})
    records = copy.deepcopy(load_journal(old))
    for record in records:
        if record["event_type"] == "CANDIDATE_OBSERVED":
            record["payload"].update(lifecycle="DISCOVERED", disposition="NEEDS_SPEC")
    with pytest.raises(ContractError, match="accepted collector cursors"):
        checkpoint.recovery_prefix(records, {"capture-1"}, {"x:unexpected": {}})
