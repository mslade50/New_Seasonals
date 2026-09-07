"""Offline algorithm-level family fit; current inventory is never an input."""
import copy
import json
from pathlib import Path

import pytest

from tests.test_strategy_discovery import config, run, catalog, manifest, source, item
from scripts.build_algorithm_family_catalog import catalog_from_source, build_catalog, source_literal, ROOT, REGISTRY
from scripts.run_strategy_discovery import main as cli_main
from research.strategy_discovery.contracts import ContractError, load_json, sha256_json
from research.strategy_discovery.family_fit import (
    ACTIVE_STATUS, assess_family_fit, horizon_bucket, publish_family_fit, validate_family_catalog,
)

AS_OF = "2026-09-06T15:00:00Z"


def report():
    cfg = config()
    cfg["as_of"] = AS_OF
    return run(cfg=cfg)[0]


def profiles(value, *, behavior="MEAN_REVERSION", horizon="SHORT_TERM"):
    return {"schema_version": "candidate-family-profiles.v1", "candidate_profiles": {
        candidate["fingerprint"]: {"behavior": behavior, "markets": ["EQUITY_INDEX"],
            "horizon": horizon, "evidence_refs": ["Supplied explicit three-session SPY/QQQ reversal rules."]}
        for candidate in value["candidates"]}}


def test_native_catalog_keeps_cash_gated_trend_and_reference_sleeves_separate():
    value = catalog_from_source(as_of=AS_OF)
    active = [r for r in value["records"] if r["status"] == ACTIVE_STATUS]
    assert len(active) == 22
    assert not any(r["profile"]["behavior"] == "UNCLASSIFIED" for r in active)
    trend = next(r for r in active if r["name"] == "Monthly Trend")
    assert trend["cash_gate_supported"] is True
    assert len([r for r in active if r["sleeve"] == "EVENT"]) == 6
    assert all(r["status"] != ACTIVE_STATUS for r in value["records"] if r["sleeve"] == "REFERENCE")
    assert value["positions_used"] is False and value["runtime_verified_now"] is False
    assert "SPY" in next(r for r in active if r["name"] == "Indices Oversold Bounce")["instruments"]
    assert value == catalog_from_source(as_of=AS_OF)


def test_index_reversion_is_compared_to_algorithm_families_not_current_positions():
    value = report()
    fit = assess_family_fit(value, catalog_from_source(as_of=AS_OF), profiles(value))
    candidate = fit["candidates"][0]
    assert candidate["fit_status"] == "ACTIVE_FAMILY_OVERLAP"
    names = {row["name"] for row in candidate["active_matches"]}
    assert {"Indices Oversold Bounce", "SPY QQQ MonFri Reversion", "Monthly Weak Close"} <= names
    assert candidate["email_eligible"] is False
    assert any(row["direction_relationship"] == "DIFFERENT_OR_BOTH" for row in candidate["active_matches"])
    assert fit["discovery_report_digest"] == sha256_json(value)


def test_paper_and_planned_ideas_are_only_references():
    value = report()
    fit = assess_family_fit(value, catalog_from_source(as_of=AS_OF), profiles(value, behavior="MOMENTUM"))
    candidate = fit["candidates"][0]
    assert candidate["fit_status"] == "NO_ACTIVE_FAMILY_MATCH"
    assert "Dial-gated SPY" in {row["name"] for row in candidate["reference_matches"]}
    assert candidate["email_eligible"] is False


def test_unclassified_new_strategy_makes_baseline_incomplete():
    from strategy_config import STRATEGY_BOOK
    book = copy.deepcopy(STRATEGY_BOOK)
    book.append({**copy.deepcopy(book[0]), "name": "New unreviewed algo"})
    native = catalog_from_source(as_of=AS_OF)
    cat = build_catalog(book, source_literal(ROOT / "event_sleeve.py", "EVENT_SLEEVE"),
        source_literal(ROOT / "trend_sleeve.py", "TREND_UNIVERSE"), load_json(ROOT / REGISTRY), native["source_digests"], as_of=AS_OF)
    value = report()
    assert assess_family_fit(value, cat, profiles(value))["candidates"][0]["fit_status"] == "INCOMPLETE_BASELINE"


def test_missing_classification_does_not_claim_fit():
    value = report()
    result = assess_family_fit(value, catalog_from_source(as_of=AS_OF),
        {"schema_version": "candidate-family-profiles.v1", "candidate_profiles": {}})
    assert result["candidates"][0]["fit_status"] == "NEEDS_CLASSIFICATION"


def test_stale_operating_observation_blocks_conclusive_fit():
    value = report()
    cat = catalog_from_source(as_of=AS_OF)
    cat["checked_at"] = "2026-08-01T14:40:00Z"
    fit = assess_family_fit(value, cat, profiles(value))
    assert fit["operating_status_stale"] is True
    assert fit["candidates"][0]["fit_status"] == "INCOMPLETE_BASELINE"


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(positions={"SPY": 50}),
    lambda c: c.update(runtime_verified_now=True),
    lambda c: c.update(records_digest="0" * 64),
    lambda c: c.update(checked_at="2026-09-07T00:00:00Z"),
    lambda c: c.update(source_digests={"strategy_config.py": "unknown"}),
])
def test_catalog_rejects_unproven_or_tampered_inputs(mutation):
    value = catalog_from_source(as_of=AS_OF)
    mutation(value)
    with pytest.raises(ContractError):
        validate_family_catalog(value)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True, "3"])
def test_horizon_requires_finite_session_count(bad):
    with pytest.raises(ContractError):
        horizon_bucket(bad)


@pytest.mark.parametrize("mutation", [
    lambda p: p["candidate_profiles"].update({"unknown": next(iter(p["candidate_profiles"].values()))}),
    lambda p: next(iter(p["candidate_profiles"].values())).update(horizon="INTRADAY"),
    lambda p: next(iter(p["candidate_profiles"].values())).update(markets=[["EQUITY_INDEX"]]),
    lambda p: next(iter(p["candidate_profiles"].values())).update(evidence_refs=[]),
])
def test_profiles_require_matching_identity_horizon_and_evidence(mutation):
    value = report()
    annotation = profiles(value)
    mutation(annotation)
    with pytest.raises(ContractError):
        assess_family_fit(value, catalog_from_source(as_of=AS_OF), annotation)


def test_companion_is_immutable_and_digest_bound(tmp_path):
    value = report()
    fit = assess_family_fit(value, catalog_from_source(as_of=AS_OF), profiles(value))
    path = publish_family_fit(tmp_path, fit)
    assert sha256_json(fit) in path.name
    assert publish_family_fit(tmp_path, fit) == path
    path.write_text("damaged evidence", encoding="utf-8")
    with pytest.raises(ContractError, match="immutable"):
        publish_family_fit(tmp_path, fit)
    assert path.read_text() == "damaged evidence"


def test_cli_writes_companion_without_promoting_lifecycle(tmp_path):
    value = report()
    cfg = config()
    cfg["as_of"] = AS_OF
    inputs = {"config": cfg, "source-manifest": manifest(source()),
              "strategy-catalog": catalog("STRATEGY_BOOK"), "dead-end-catalog": catalog("DEAD_ENDS"),
              "family-catalog": catalog_from_source(as_of=AS_OF), "candidate-families": profiles(value)}
    args = []
    for key, payload in inputs.items():
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        args += [f"--{key}", str(path)]
    items = tmp_path / "items.jsonl"
    items.write_text(json.dumps(item()) + "\n", encoding="utf-8")
    output = tmp_path / "approved" / "daily"
    args += ["--items", str(items), "--output-dir", str(output)]
    assert cli_main(args, approved_output_root=tmp_path / "approved") == 0
    companions = list(output.glob("family-fit-*.json"))
    assert len(companions) == 1
    fit = json.loads(companions[0].read_text())
    assert fit["candidates"][0]["email_eligible"] is False
    latest = json.loads((output / "latest.json").read_text())
    emitted = json.loads((output / "runs" / latest["run_id"] / "strategy_discovery_report.json").read_text())
    assert fit["discovery_report_digest"] == sha256_json(emitted)
    assert emitted["candidates"][0]["lifecycle"] == value["candidates"][0]["lifecycle"]


def test_invalid_optional_profile_blocks_before_report_or_journal_write(tmp_path):
    cfg = config()
    cfg["as_of"] = AS_OF
    inputs = {"config": cfg, "source-manifest": manifest(source()),
              "strategy-catalog": catalog("STRATEGY_BOOK"), "dead-end-catalog": catalog("DEAD_ENDS"),
              "family-catalog": catalog_from_source(as_of=AS_OF),
              "candidate-families": {"schema_version": "candidate-family-profiles.v1", "candidate_profiles": {"wrong-identity": {}}}}
    args = []
    for key, payload in inputs.items():
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        args += [f"--{key}", str(path)]
    items = tmp_path / "items.jsonl"
    items.write_text(json.dumps(item()) + "\n", encoding="utf-8")
    output = tmp_path / "approved" / "daily"
    args += ["--items", str(items), "--output-dir", str(output)]
    assert cli_main(args, approved_output_root=tmp_path / "approved") == 2
    assert not (output / "journal.jsonl").exists()
    assert not (output / "latest.json").exists()
    assert not (output / "runs").exists()


@pytest.mark.parametrize("limit", [0, -1, True, 31, float("inf")])
def test_status_freshness_policy_cannot_be_disabled(limit):
    value = report()
    with pytest.raises(ContractError):
        assess_family_fit(value, catalog_from_source(as_of=AS_OF), profiles(value), max_status_age_days=limit)
