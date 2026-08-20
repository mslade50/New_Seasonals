import json

import pandas as pd
import pytest

from fundamental.research_controls import apply_research_controls, load_research_controls
from fundamental.research_process import apply_research_routes, summarize_research_funnel
from fundamental.research_state import evaluate_trigger_payload, load_research_event_state
from fundamental.run_manifest import append_decision_transitions
from fundamental.underwrite import (
    is_surfaceable_quick_review,
    load_underwrite_decisions,
    validate_underwrite_record,
)


def test_complete_v2_underwrite_clears_every_promotion_gate(v2_underwrite_factory):
    record = v2_underwrite_factory()
    result = validate_underwrite_record(record, decision_as_of="2026-08-05")
    assert result["valid_for_quick_review"] is True
    assert all(result["gates"].values())
    assert result["derived"]["upside_downside_ratio"] == 2.0
    assert result["derived"]["realization_signals"] == {
        "revisions": True,
        "catalyst": True,
        "trend": True,
    }


def test_legacy_or_incomplete_record_cannot_create_quick_review(tmp_path):
    path = tmp_path / "decisions.json"
    path.write_text(
        json.dumps({
            "as_of": "2026-08-05",
            "decisions": [{
                "ticker": "AAA",
                "decision": "QUICK_REVIEW",
                "verdict": "A polished sentence is not an underwrite.",
            }],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="failed v2 promotion gates"):
        load_underwrite_decisions(path)


def test_stale_price_and_wrong_valuation_method_fail_closed(v2_underwrite_factory):
    record = v2_underwrite_factory()
    record["price_snapshot"]["as_of"] = "2026-07-01"
    record["valuation"]["primary_method"] = "price_to_book"
    result = validate_underwrite_record(record, decision_as_of="2026-08-05")
    assert result["gates"]["current_security_snapshot"] is False
    assert result["gates"]["archetype_valuation_methods"] is False
    assert result["valid_for_quick_review"] is False


def test_realization_requires_two_independent_signals(v2_underwrite_factory):
    record = v2_underwrite_factory()
    record["realization"] = {
        "revision_signal": "NEGATIVE",
        "observable_catalyst": True,
        "trend_state": "RED",
    }
    result = validate_underwrite_record(record)
    assert result["gates"]["realization_edge"] is False
    assert not is_surfaceable_quick_review(record)


def _screen_row(ticker="AAA", trend="RED"):
    row = {
        "ticker": ticker,
        "research_lane": "standard_company",
        "sector": "Industrials",
        "industry": "Machinery",
        "hard_exclusion_reason": "",
        "score_coverage_pct": 100.0,
        "research_score": 85.0,
        "trend_state": trend,
    }
    for metric in (
        "roic", "incremental_roic", "gross_profitability", "gross_margin_stability",
        "fcf_margin", "cash_conversion", "accrual_ratio", "fcf_positive_years",
        "sbc_to_revenue", "share_count_cagr_3y", "net_debt_to_ebitda",
        "fcf_yield", "earnings_yield", "revenue_growth_change", "fcf_margin_change",
    ):
        row[f"rank_{metric}"] = 80.0
    return row


def test_damaged_trend_does_not_erase_a_research_hypothesis():
    routed = apply_research_routes(pd.DataFrame([_screen_row()]))
    row = routed.iloc[0]
    assert row["research_route"] == "HYPOTHESIS_TEST"
    assert row["trend_state"] == "RED"
    assert not bool(row["screen_can_surface_review"])
    assert row["security_readiness"] == "NOT_DECISION_GRADE"


def test_site_controls_change_research_priority_only(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "version": 1,
        "updated_at": "2026-08-05T12:00:00Z",
        "actions": {
            "AAA": {"action": "DEEPEN", "updated_at": "2026-08-05T12:00:00Z"},
            "BBB": {"action": "PASS", "updated_at": "2026-08-05T12:00:00Z"},
            "CCC": {"action": "WATCH", "updated_at": "2026-08-05T12:00:00Z"},
        },
    }), encoding="utf-8")
    controls, health = load_research_controls(state, as_of="2026-08-05")
    candidates = apply_research_routes(pd.DataFrame([
        _screen_row("AAA"), _screen_row("BBB"), _screen_row("CCC")
    ]))
    result = apply_research_controls(candidates, controls)
    by_ticker = result.set_index("ticker")
    assert health["status"] == "CURRENT"
    assert by_ticker.loc["AAA", "research_queue_priority"] == 10_000.0
    assert bool(by_ticker.loc["BBB", "research_suppressed"])
    assert by_ticker.loc["CCC", "control_disposition"] == "WAIT_FOR_RECORDED_TRIGGER"
    assert by_ticker["screen_can_surface_review"].eq(False).all()


def test_clear_is_a_tombstone_and_stale_control_state_is_visible(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "version": 1,
        "updated_at": "2026-06-01T12:00:00Z",
        "actions": {
            "AAA": {"action": "CLEAR", "updated_at": "2026-06-01T12:00:00Z"},
            "BBB": {"action": "WATCH", "updated_at": "2026-06-01T12:00:00Z"},
        },
    }), encoding="utf-8")
    controls, health = load_research_controls(state, as_of="2026-08-05", max_age_days=30)
    assert "AAA" not in controls
    assert controls["BBB"]["action"] == "WATCH"
    assert health["status"] == "STALE"
    assert health["action_counts"]["CLEAR"] == 1


def test_pass_reopens_only_on_thesis_changing_evidence():
    candidates = apply_research_routes(pd.DataFrame([_screen_row("AAA")]))
    controls = {"AAA": {"action": "PASS", "updated_at": "2026-08-05T12:00:00Z"}}
    unchanged = apply_research_controls(candidates, controls)
    changed = apply_research_controls(candidates, controls, thesis_changed_tickers={"AAA"})
    assert bool(unchanged.iloc[0]["research_suppressed"])
    assert not bool(changed.iloc[0]["research_suppressed"])
    assert changed.iloc[0]["control_disposition"] == "REOPENED_BY_THESIS_CHANGE"


def test_funnel_counts_are_structured_and_mutually_exclusive():
    rows = pd.DataFrame([
        _screen_row("AAA"),
        {**_screen_row("BBB"), "hard_exclusion_reason": "Liquidity below floor."},
    ])
    routed = apply_research_routes(rows)
    summary = summarize_research_funnel(routed)
    assert summary["total"] == 2
    assert sum(summary["routes"].values()) == 2
    assert sum(summary["primary_gates"].values()) == 2
    assert summary["axes"]["business_quality"]["strong_70_plus"] == 2


def test_sourced_trigger_evaluator_handles_crossing_and_missing_evidence():
    result = evaluate_trigger_payload({
        "schema_version": "fundamental-trigger.v1",
        "updated_at": "2026-08-05T13:00:00Z",
        "triggers": [
            {
                "ticker": "AAA", "trigger_id": "margin-proof", "kind": "PROOF",
                "metric": "fcf_margin", "comparator": "CROSS_ABOVE", "threshold": 0.15,
                "prior_value": 0.14, "observed_value": 0.16,
                "observed_at": "2026-08-05T12:00:00Z", "status": "ARMED",
                "source_ids": ["10q-1"],
            },
            {
                "ticker": "BBB", "trigger_id": "unsourced", "kind": "REOPEN",
                "metric": "revenue", "comparator": ">=", "threshold": 10,
                "observed_value": 11, "observed_at": "2026-08-05T12:00:00Z",
                "status": "ARMED", "source_ids": [],
            },
        ],
    }, as_of="2026-08-05")
    assert result["fired_tickers"] == ["AAA"]
    assert result["fired"] == 1
    assert result["unevaluable"] == 1


def test_wrong_trigger_schema_cannot_fire_anything():
    result = evaluate_trigger_payload({
        "schema_version": "fundamental-trigger.v0",
        "updated_at": "2026-08-05T13:00:00Z",
        "triggers": [{
            "ticker": "AAA", "trigger_id": "fake-fire", "kind": "PROOF",
            "metric": "margin", "comparator": ">=", "threshold": 1,
            "observed_value": 2, "observed_at": "2026-08-05T12:00:00Z",
            "status": "FIRED", "source_ids": ["source-1"],
        }],
    }, as_of="2026-08-05")
    assert result["status"] == "WRONG_SCHEMA"
    assert result["fired_tickers"] == []


def test_event_state_reopens_only_new_sourced_material_evidence(tmp_path):
    trigger_path = tmp_path / "triggers.json"
    evidence_path = tmp_path / "evidence.json"
    manifest_path = tmp_path / "manifest.json"
    trigger_path.write_text(json.dumps({
        "schema_version": "fundamental-trigger.v1", "updated_at": "2026-08-05T12:00:00Z",
        "triggers": [],
    }), encoding="utf-8")
    evidence_path.write_text(json.dumps({
        "schema_version": "fundamental-evidence.v1", "updated_at": "2026-08-05T12:00:00Z",
        "evidence": [
            {"evidence_id": "evidence-aaa", "ticker": "AAA", "claim_id": "claim-aaa",
             "claim": "New evidence changes the thesis.", "direction": "CONFIRM",
             "source_id": "10q-new", "materiality": "THESIS_CHANGING",
             "observed_at": "2026-08-05T12:00:00Z"},
            {"evidence_id": "evidence-bbb", "ticker": "BBB", "claim_id": "claim-bbb",
             "claim": "Old evidence predates the last run.", "direction": "DISCONFIRM",
             "source_id": "10q-old", "materiality": "THESIS_CHANGING",
             "observed_at": "2026-08-01T12:00:00Z"},
        ],
    }), encoding="utf-8")
    manifest_path.write_text(json.dumps({"completed_at": "2026-08-04T12:00:00Z"}), encoding="utf-8")
    state = load_research_event_state(
        as_of="2026-08-05",
        trigger_path=trigger_path,
        evidence_path=evidence_path,
        previous_manifest_path=manifest_path,
    )
    assert state["thesis_changed_tickers"] == {"AAA"}


def test_decision_transitions_are_append_only_and_idempotent(tmp_path):
    previous = tmp_path / "manifest.json"
    log = tmp_path / "transitions.jsonl"
    previous.write_text(json.dumps({
        "decision_states": {"AAA": {"decision": "WAIT_FOR_PROOF"}},
    }), encoding="utf-8")
    current = {"AAA": {"decision": "PASS"}}
    first = append_decision_transitions(
        previous_manifest_path=previous,
        current_states=current,
        run_id="run-1",
        as_of="2026-08-05",
        output_path=log,
    )
    second = append_decision_transitions(
        previous_manifest_path=previous,
        current_states=current,
        run_id="run-1",
        as_of="2026-08-05",
        output_path=log,
    )
    assert len(first) == 1
    assert second == []
    assert len(log.read_text(encoding="utf-8").splitlines()) == 1
    row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert row["reason_code"] == "DECISION_STATE_CHANGED"
    assert row["prior_state"] == {"decision": "WAIT_FOR_PROOF"}
    assert row["new_state"] == {"decision": "PASS"}
    assert row["authority"] == "deterministic_research_engine"
    assert row["research_only"] is True
