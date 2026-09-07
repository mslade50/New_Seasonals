from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.strategy_discovery.contracts import ContractError, validate_catalog
from scripts.build_algorithm_family_catalog import catalog_from_source
from scripts.build_strategy_discovery_catalogs import build

AS_OF = "2026-09-07T12:00:00Z"


def test_native_catalog_is_fresh_complete_and_deterministic(tmp_path: Path):
    strategy, dead = build(AS_OF, tmp_path / "absent-dead-ends.json")
    validate_catalog(strategy, "STRATEGY_BOOK", "strategy")
    validate_catalog(dead, "DEAD_ENDS", "dead")
    assert len(strategy["records"]) == 15
    assert len({row["structural_fingerprint"] for row in strategy["records"]}) == 15
    assert all(row["active"] is True for row in strategy["records"])
    assert dead["records"] == []
    assert build(AS_OF, tmp_path / "absent-dead-ends.json") == (strategy, dead)


def test_structured_dead_ends_are_required_to_match_strict_contract(tmp_path: Path):
    path = tmp_path / "dead.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "strategy-dead-ends.v1",
                "records": [
                    {
                        "name": "Failed candidate",
                        "structural_fingerprint": "a" * 64,
                        "rejection_reason": "Costed recent-era return was negative.",
                        "decided_at": AS_OF,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _, dead = build(AS_OF, path)
    assert dead["records"][0]["name"] == "Failed candidate"
    path.write_text('{"schema_version":"wrong","records":[]}', encoding="utf-8")
    with pytest.raises(ContractError, match="registry"):
        build(AS_OF, path)


def test_family_catalog_can_refresh_configured_status_without_claiming_runtime():
    value = catalog_from_source(as_of=AS_OF, configured_status_current=True)
    assert value["checked_at"] == AS_OF
    assert value["runtime_verified_now"] is False
    assert value["positions_used"] is False


def test_scheduled_runner_has_collection_agent_and_completion_gates():
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts" / "run_strategy_research.bat").read_text(encoding="utf-8")
    assert runner.index("collect_strategy_sources.py") < runner.index("invoke_strategy_research_agent.ps1")
    assert runner.index("invoke_strategy_research_agent.ps1") < runner.index("check_strategy_research_run.py")
    assert "send_strategy_research_failure_email.py" in runner
    assert "--permission-mode', 'bypassPermissions'" in (root / "scripts" / "invoke_strategy_research_agent.ps1").read_text(encoding="utf-8")
    skill = (root / ".claude" / "skills" / "strategy-research" / "SKILL.md").read_text(encoding="utf-8")
    assert "Current positions" in skill or "current positions" in skill
    assert "finalize_strategy_research.py" in skill
    assert "Never send a stand-down" in skill


def test_failure_alert_dry_run_never_opens_smtp(monkeypatch):
    from scripts import send_strategy_research_failure_email as alert

    monkeypatch.setattr(alert, "_send", lambda *a, **k: (_ for _ in ()).throw(AssertionError("dry run cannot send")))
    assert alert.main(["--phase", "completion_check", "--summary", "Fixture failure"]) == 0


def test_completion_gate_rejects_unconfirmed_email_and_accepts_no_email(tmp_path: Path):
    from scripts import check_strategy_research_run as check

    marker = tmp_path / "marker"
    marker.write_text("start", encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"schema_version": "strategy-source-cursors.v1", "accepted": {}, "pending": None}), encoding="utf-8")
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({
        "schema_version": "strategy-research-email-decision.v1",
        "source_bundle_digest": "c" * 64,
        "positions_used": False,
        "email_required": True,
        "delivery_status": "PENDING",
        "eligible_count": 1,
        "candidate_count": 1,
    }), encoding="utf-8")
    assert check.main(["--state", str(state), "--decision", str(decision), "--marker", str(marker)]) == 2
    payload = json.loads(decision.read_text())
    payload.update(email_required=False, delivery_status="NO_EMAIL", eligible_count=0)
    decision.write_text(json.dumps(payload), encoding="utf-8")
    assert check.main(["--state", str(state), "--decision", str(decision), "--marker", str(marker)]) == 0
