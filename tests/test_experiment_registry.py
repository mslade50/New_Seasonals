import datetime as dt
import json

import pytest

from research.experiment_registry import (
    RegistryValidationError,
    append_records,
    load_records,
    prepare_record,
    summarize,
)

NOW = dt.datetime(2026, 8, 27, 13, 0, tzinfo=dt.timezone.utc)


def _clock():
    return NOW


def source_record(**overrides):
    row = {
        "kind": "source",
        "source_id": "src_1",
        "source_type": "ssrn",
        "url": "https://papers.ssrn.com/example",
        "title": "An intraday effect",
        "retrieved_at": "2026-08-27T12:00:00Z",
        "content_hash": "abc123",
        "research_only": True,
        "no_order": True,
    }
    row.update(overrides)
    return row


def preregistration(**overrides):
    row = {
        "kind": "preregistration",
        "experiment_id": "exp_1",
        "hypothesis_id": "hyp_1",
        "family": "intraday",
        "question": "Does a residual gap continue after the first hour?",
        "universe": "point-in-time execution-grade US equities",
        "signal": "overnight residual plus first-hour residual",
        "decision_time": "10:30 America/New_York",
        "entry_rule": "next 15-minute bar open",
        "exit_rule": "15:45 close",
        "cost_model": "10 bps round trip",
        "validation_plan": "held-out years and tickers, clustered by day",
        "trial_budget": 6,
        "promotion_gate": "positive held-out expectancy after 2x costs",
        "kill_gate": "non-positive held-out expectancy after base costs",
        "research_only": True,
        "no_order": True,
    }
    row.update(overrides)
    return row


def test_registry_append_is_idempotent_and_append_only(tmp_path):
    path = tmp_path / "artifacts" / "registry.jsonl"
    assert append_records(path, [source_record()], clock=_clock) == 1
    assert append_records(path, [source_record()], clock=_clock) == 0
    assert append_records(path, [source_record(title="Revised source title")], clock=_clock) == 1

    rows = load_records(path)
    assert len(rows) == 2
    assert rows[0]["written_at"] == "2026-08-27T13:00:00Z"
    assert rows[0]["record_id"] != rows[1]["record_id"]
    assert all(row["research_only"] and row["no_order"] for row in rows)


def test_preregistration_requires_trial_budget_promotion_and_kill_gates():
    prepare_record(preregistration(), clock=_clock)

    bad = preregistration()
    del bad["kill_gate"]
    with pytest.raises(RegistryValidationError, match="kill_gate"):
        prepare_record(bad, clock=_clock)

    with pytest.raises(RegistryValidationError, match="positive integer"):
        prepare_record(preregistration(trial_budget=0), clock=_clock)


def test_registry_rejects_executable_order_fields():
    with pytest.raises(RegistryValidationError, match="quantity"):
        prepare_record(source_record(metadata={"quantity": 100}), clock=_clock)


def test_registry_fails_closed_on_corrupt_history(tmp_path):
    path = tmp_path / "artifacts" / "registry.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(prepare_record(source_record(), clock=_clock)) + "\nnot-json\n")
    with pytest.raises(RegistryValidationError, match="line 2"):
        load_records(path)
    assert len(load_records(path, skip_corrupt=True)) == 1


def test_registry_summary_counts_trials_without_implying_promotion():
    trial = {
        "kind": "trial",
        "trial_id": "trial_1",
        "experiment_id": "exp_1",
        "parameter_set": {"gap_z": 1.5},
        "data_cutoff": "2024-12-31",
        "status": "complete",
        "research_only": True,
        "no_order": True,
    }
    result = summarize([prepare_record(preregistration(), clock=_clock), prepare_record(trial, clock=_clock)])
    assert result["registered_trials"] == 1
    assert result["by_kind"] == {"preregistration": 1, "trial": 1}
    assert result["research_only"] is True
    assert result["no_order"] is True

