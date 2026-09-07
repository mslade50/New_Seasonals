from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.strategy_discovery.contracts import ContractError, sha256_json
from research.strategy_discovery.email_gate import evaluate_email_package, render_email
from research.strategy_discovery.family_fit import assess_family_fit
from research.strategy_discovery.journal import append_events, load_journal
from scripts import finalize_strategy_research as finalizer
from scripts.build_algorithm_family_catalog import catalog_from_source
from tests.test_strategy_discovery import artifact, config, run
from tests.test_strategy_family_fit import profiles


def metric(name, value, unit="value"):
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "sample_size": 60,
        "definition": f"Fixture definition for {name}.",
        "methodology": "Frozen decision-available costed portfolio replay.",
    }


def qualifying_metrics():
    values = {
        "sample_size": 60,
        "gross_mean_bps": 30,
        "net_mean_bps": 24,
        "median_net_bps": 15,
        "hit_rate_pct": 58,
        "worst_trade_bps": -180,
        "recent_net_mean_bps": 18,
        "neighbor_positive_count": 3,
        "neighbor_test_count": 4,
        "bootstrap_probability_mean_le_zero": 0.04,
        "active_book_daily_correlation": 0.20,
        "bad_day_co_loss_pct": 12,
        "incremental_portfolio_sharpe": 0.14,
        "marginal_capital_occupancy_pct": 9,
        "estimated_strategy_capacity_usd": 5_000_000,
        "round_trip_cost_bps": 5,
        "liquid_ibkr_instrument_count": 2,
        "total_instrument_count": 2,
        "point_in_time_universe_flag": 0,
    }
    return [metric(name, value) for name, value in values.items()]


def validated_inputs(tmp_path: Path, *, behavior="MEAN_REVERSION"):
    cfg = config(mode="LIVE")
    cfg["as_of"] = "2026-09-06T15:00:00Z"
    ready, events = run(cfg=cfg)
    journal_path = tmp_path / "journal.jsonl"
    append_events(journal_path, events, recorded_at=ready["as_of"])
    fp = ready["candidates"][0]["fingerprint"]
    artifacts = artifact(fp, tmp_path / "validation_artifacts")
    artifacts["artifacts"][0]["created_at"] = cfg["as_of"]
    artifacts["artifacts"][0]["metrics"] = qualifying_metrics()
    validated, _ = run(
        cfg=cfg,
        journal=load_journal(journal_path),
        artifacts=artifacts,
        artifact_root=tmp_path / "validation_artifacts",
    )
    fit = assess_family_fit(
        validated,
        catalog_from_source(as_of=cfg["as_of"]),
        profiles(validated, behavior=behavior),
    )
    package = {
        "schema_version": "strategy-research-email-package.v1",
        "as_of": cfg["as_of"],
        "discovery_run_id": validated["run_id"],
        "source_bundle_digest": "c" * 64,
        "discovery_report_digest": sha256_json(validated),
        "family_fit_digest": sha256_json(fit),
        "positions_used": False,
        "candidates": [
            {
                "fingerprint": fp,
                "writeup": {
                    "strategy": "Buy a precisely defined three-session liquid-index reversal.",
                    "validation": "The fixed-universe replay survived costs and recent-era checks.",
                    "why_it_fits": "Portfolio replay adds Sharpe with low active-book correlation.",
                    "implementation": "Research implementation uses SPY and QQQ with next-open entry.",
                    "risks_and_falsifiers": "Stop if net edge or marginal portfolio benefit decays.",
                },
            }
        ],
    }
    return validated, fit, package


def test_worthwhile_candidate_renders_full_email_and_excludes_positions(tmp_path: Path):
    report, fit, package = validated_inputs(tmp_path)
    decision = evaluate_email_package(report, fit, package)
    assert decision["email_required"] is True
    assert decision["positions_used"] is False
    assert decision["eligible_count"] == 1
    subject, body = render_email(decision)
    assert "Strategy research worth reviewing" in subject
    for heading in ("Strategy", "What our data showed", "Why it fits", "How to implement"):
        assert heading in body
    assert "Current positions were not used" in body
    assert "https://x.com/alpha/status/100" in body


def test_explicit_fixed_universe_does_not_require_a_false_point_in_time_claim(tmp_path: Path):
    report, fit, package = validated_inputs(tmp_path)
    candidate = report["candidates"][0]
    assert candidate["structure"]["universe_history"]["membership_mode"] == "FIXED_INSTRUMENTS"
    decision = evaluate_email_package(report, fit, package)
    assert decision["email_required"] is True


def test_universe_metric_must_match_the_journaled_membership_mode(tmp_path: Path):
    report, fit, package = validated_inputs(tmp_path)
    candidate = report["candidates"][0]
    target = next(
        row
        for row in candidate["internally_validated_metrics"]
        if row["name"] == "point_in_time_universe_flag"
    )
    target["value"] = 1
    package["discovery_report_digest"] = sha256_json(report)
    fit["discovery_report_digest"] = sha256_json(report)
    package["family_fit_digest"] = sha256_json(fit)
    decision = evaluate_email_package(report, fit, package)
    assert decision["email_required"] is False
    assert (
        "universe history is neither point-in-time nor an explicitly fixed instrument set"
        in decision["candidates"][0]["failed_gates"]
    )


def test_active_family_overlap_uses_stricter_incremental_fit_gate(tmp_path: Path):
    report, fit, package = validated_inputs(tmp_path)
    candidate = report["candidates"][0]
    target = next(row for row in candidate["internally_validated_metrics"] if row["name"] == "incremental_portfolio_sharpe")
    target["value"] = 0.08
    package["discovery_report_digest"] = sha256_json(report)
    fit["discovery_report_digest"] = sha256_json(report)
    package["family_fit_digest"] = sha256_json(fit)
    decision = evaluate_email_package(report, fit, package)
    assert decision["email_required"] is False
    assert "active-family overlap requires at least 0.10 incremental Sharpe" in decision["candidates"][0]["failed_gates"]
    with pytest.raises(ContractError, match="no worthwhile research"):
        render_email(decision)


def test_no_active_family_match_still_requires_empirical_portfolio_benefit(tmp_path: Path):
    report, fit, package = validated_inputs(tmp_path, behavior="MOMENTUM")
    assert fit["candidates"][0]["fit_status"] == "NO_ACTIVE_FAMILY_MATCH"
    target = next(row for row in report["candidates"][0]["internally_validated_metrics"] if row["name"] == "incremental_portfolio_sharpe")
    target["value"] = 0.01
    package["discovery_report_digest"] = sha256_json(report)
    fit["discovery_report_digest"] = sha256_json(report)
    package["family_fit_digest"] = sha256_json(fit)
    decision = evaluate_email_package(report, fit, package)
    assert not decision["email_required"]
    assert "incremental portfolio Sharpe improvement is below 0.05" in decision["candidates"][0]["failed_gates"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p.update(positions_used=True), "must not use current positions"),
        (lambda p: p.update(discovery_report_digest="0" * 64), "report digest mismatch"),
        (lambda p: p["candidates"][0]["writeup"].update(validation=""), "must be nonempty"),
    ],
)
def test_package_identity_and_writeup_fail_closed(tmp_path: Path, mutation, match):
    report, fit, package = validated_inputs(tmp_path)
    mutation(package)
    with pytest.raises(ContractError, match=match):
        evaluate_email_package(report, fit, package)


def test_family_fit_must_exactly_cover_report_and_preserve_research_boundary(tmp_path: Path):
    report, fit, package = validated_inputs(tmp_path)
    fit["candidates"] = []
    package["family_fit_digest"] = sha256_json(fit)
    with pytest.raises(ContractError, match="exactly cover"):
        evaluate_email_package(report, fit, package)

    report, fit, package = validated_inputs(tmp_path)
    fit["research_only"] = False
    package["family_fit_digest"] = sha256_json(fit)
    with pytest.raises(ContractError, match="research-only boundary"):
        evaluate_email_package(report, fit, package)


def test_invalid_metric_domains_fail_instead_of_becoming_a_quiet_day(tmp_path: Path):
    report, fit, package = validated_inputs(tmp_path)
    target = next(
        row
        for row in report["candidates"][0]["internally_validated_metrics"]
        if row["name"] == "liquid_ibkr_instrument_count"
    )
    target["value"] = 3
    package["discovery_report_digest"] = sha256_json(report)
    fit["discovery_report_digest"] = sha256_json(report)
    package["family_fit_digest"] = sha256_json(fit)
    with pytest.raises(ContractError, match="exceeds total"):
        evaluate_email_package(report, fit, package)


def test_unfinished_research_or_family_baseline_cannot_become_a_quiet_day(tmp_path: Path):
    report, fit, package = validated_inputs(tmp_path)
    report["candidates"][0]["lifecycle"] = "RESEARCH_READY"
    report["candidates"][0]["internally_validated_metrics"] = []
    package["discovery_report_digest"] = sha256_json(report)
    fit["discovery_report_digest"] = sha256_json(report)
    package["family_fit_digest"] = sha256_json(fit)
    with pytest.raises(ContractError, match="lacks a journaled"):
        evaluate_email_package(report, fit, package)

    report, fit, package = validated_inputs(tmp_path)
    fit["candidates"][0]["fit_status"] = "INCOMPLETE_BASELINE"
    package["family_fit_digest"] = sha256_json(fit)
    with pytest.raises(ContractError, match="incomplete algorithm-family baseline"):
        evaluate_email_package(report, fit, package)


def test_shadow_report_can_never_authorize_email(tmp_path: Path):
    report, fit, package = validated_inputs(tmp_path)
    report["run_mode"] = "SHADOW"
    package["discovery_report_digest"] = sha256_json(report)
    fit["discovery_report_digest"] = sha256_json(report)
    package["family_fit_digest"] = sha256_json(fit)
    with pytest.raises(ContractError, match="LIVE"):
        evaluate_email_package(report, fit, package)


def test_actionable_candidate_cannot_be_omitted_to_manufacture_a_quiet_day(tmp_path: Path):
    report, fit, package = validated_inputs(tmp_path)
    package["candidates"] = []
    with pytest.raises(ContractError, match="cover every actionable"):
        evaluate_email_package(report, fit, package)


def test_failed_worthwhile_gate_records_no_email_without_opening_smtp(tmp_path: Path, monkeypatch):
    report, fit, package = validated_inputs(tmp_path)
    target = next(row for row in report["candidates"][0]["internally_validated_metrics"] if row["name"] == "net_mean_bps")
    target["value"] = -1
    fit["discovery_report_digest"] = sha256_json(report)
    package["discovery_report_digest"] = sha256_json(report)
    package["family_fit_digest"] = sha256_json(fit)
    paths = {}
    for name, payload in (("report", report), ("fit", fit), ("package", package)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    monkeypatch.setattr(finalizer.smtplib, "SMTP", lambda *a, **k: (_ for _ in ()).throw(AssertionError("SMTP must stay closed")))
    decision = tmp_path / "decision.json"
    assert finalizer.main([
        "--report", str(paths["report"]), "--family-fit", str(paths["fit"]),
        "--package", str(paths["package"]), "--decision", str(decision), "--send",
    ]) == 0
    materialized = json.loads(decision.read_text())
    assert materialized["delivery_status"] == "NO_EMAIL"
    assert materialized["email_required"] is False
