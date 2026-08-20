"""Shared test fixtures. Currently the Daily Pitch survey scaffolding.

Since 2026-08-08 the publisher refuses ideas whose morning cannot be found on
disk: the day's `scratch/pitch_checks/<asof>/` folder must exist, hold stage
B1's surface map and real check scripts, and every evidence path must resolve
inside it. That means every payload fixture needs a real surveyed morning
behind it. Building one here keeps the two pitch test modules from drifting
into two different notions of what a surveyed morning looks like, and keeps
both of them off the repo's actual scratch directory.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def checks_root(tmp_path, monkeypatch):
    """An isolated checks root, installed as pitch_grammar's default so a test
    never passes or fails on what happens to be sitting in scratch/."""
    import pitch_grammar as pg
    root = tmp_path / "pitch_checks"
    root.mkdir()
    monkeypatch.setattr(pg, "CHECKS_ROOT", root)
    return root


@pytest.fixture()
def survey(checks_root):
    """Give a payload a surveyed morning: the day folder, the surface map, a
    check and a development script per idea, evidence pointed at them all.

    Returns the day directory. Pass `root=` to write into a different (e.g.
    yesterday's) folder, or `dev=False` to leave out the round-3 scripts.
    """
    import pitch_grammar as pg

    def _survey(payload, root=None, dev=True):
        day = Path(root or checks_root) / str(payload.get("asof", ""))
        day.mkdir(parents=True, exist_ok=True)
        (day / pg.SURFACE_MAP_NAME).write_text("# surface map\n",
                                               encoding="utf-8")
        fields = ("script", "dev_script") if dev else ("script",)
        for i, idea in enumerate(payload.get("ideas") or [], 1):
            evidence = idea.setdefault("evidence", {})
            for field in fields:
                script = day / f"idea{i}_{field}.py"
                script.write_text("# a real check\n", encoding="utf-8")
                evidence[field] = str(script)
        return day

    return _survey


@pytest.fixture()
def v2_underwrite_factory():
    """Build a fully decision-gated synthetic fundamental underwrite."""

    def _build(ticker="AAA", decision="QUICK_REVIEW", as_of="2026-08-05"):
        readiness = "REVIEW_READY" if decision == "QUICK_REVIEW" else "WAIT_FOR_PROOF"
        return {
            "schema_version": "fundamental-underwrite.v2",
            "underwrite_id": f"uw-{ticker.lower()}-20260805",
            "ticker": ticker,
            "as_of": as_of,
            "decision": decision,
            "business_model_lane": "general_operating_company",
            "idea_archetype": "quality_compounder",
            "company_thesis_status": "INTACT",
            "security_readiness": readiness,
            "verdict": "A completed underwrite found a defined expectations gap.",
            "price_snapshot": {
                "price": 10.0,
                "currency": "USD",
                "as_of": as_of,
                "diluted_shares": 100.0,
                "net_debt": 0.0,
                "enterprise_value": 1000.0,
                "source_ids": ["price"],
            },
            "variant_hypothesis": {
                "market_view": "The market expects the current margin pressure to persist.",
                "variant_view": "The market discounts a recoverable operating issue.",
                "why_market_wrong": "Temporary launch costs obscure improving customer economics.",
                "causal_chain": [
                    "Retention improves as the product cohort matures.",
                    "Higher retention lifts contribution margin and owner earnings.",
                ],
                "time_horizon": "Two years",
            },
            "expectations": {
                "implied_case": "The price implies no recovery in normalized owner earnings.",
                "guidance_bridge": "Management guidance permits a measured margin recovery.",
                "consensus_bridge": "Current consensus assumes only modest recovery.",
                "estimate_status": "CURRENT",
                "estimate_snapshot_as_of": as_of,
            },
            "operating_model": {
                "drivers": [
                    {"name": "Retention", "baseline": 90, "thesis_case": 93, "unit": "%", "source_ids": ["filing"]},
                    {"name": "FCF margin", "baseline": 12, "thesis_case": 16, "unit": "%", "source_ids": ["release"]},
                ]
            },
            "valuation": {
                "primary_method": "driver_dcf",
                "secondary_method": "reverse_dcf",
                "currency": "USD",
                "bear": 8.0,
                "base": 14.0,
                "bull": 18.0,
                "horizon_years": 2.0,
                "reverse_expectations": "The quote capitalizes flat owner earnings in perpetuity.",
                "source_ids": ["price", "filing"],
            },
            "realization": {
                "revision_signal": "POSITIVE",
                "observable_catalyst": True,
                "trend_state": "GREEN",
            },
            "downside": {
                "mechanism": "Retention fails to improve and fixed costs keep margins compressed.",
                "financing_and_dilution": "Net cash and a stable diluted share count limit financing risk.",
                "bear_case": "The bear case values the unchanged earnings stream at USD 8 per share.",
            },
            "proof_triggers": [
                {
                    "trigger_id": "retention-proof",
                    "metric": "Retention",
                    "comparator": ">=",
                    "threshold": 92,
                    "unit": "%",
                    "expected_by": "2026-11-05",
                    "source_ids": ["release"],
                }
            ],
            "kill_conditions": [
                {
                    "condition_id": "retention-break",
                    "metric": "Retention",
                    "comparator": "<",
                    "threshold": 88,
                    "unit": "%",
                    "consequence": "Break the operating thesis.",
                    "source_ids": ["filing"],
                },
                {
                    "condition_id": "fcf-break",
                    "metric": "FCF margin",
                    "comparator": "<",
                    "threshold": 10,
                    "unit": "%",
                    "consequence": "Re-underwrite normalized earnings and downside.",
                    "source_ids": ["release"],
                },
            ],
            "red_team": {
                "strongest_case": "The apparent recovery may be mix and timing rather than durable retention.",
                "evidence_ids": ["e3"],
                "unresolved_conflicts": [],
            },
            "evidence_ledger": [
                {"evidence_id": "e1", "claim": "Current security bridge", "direction": "NEUTRAL", "source_id": "price", "materiality": "HIGH"},
                {"evidence_id": "e2", "claim": "Reported retention baseline", "direction": "CONFIRMING", "source_id": "filing", "materiality": "HIGH"},
                {"evidence_id": "e3", "claim": "Margin pressure remains", "direction": "DISCONFIRMING", "source_id": "release", "materiality": "HIGH"},
            ],
            "sources": [
                {"source_id": "price", "label": "Market close", "url": "https://example.com/price", "source_type": "MARKET_DATA", "as_of": as_of, "primary": False, "use": "Price and enterprise-value bridge."},
                {"source_id": "filing", "label": "SEC filing", "url": "https://www.sec.gov/example", "source_type": "SEC_FILING", "as_of": as_of, "primary": True, "use": "Reported financials and share count."},
                {"source_id": "release", "label": "Issuer release", "url": "https://example.com/release", "source_type": "ISSUER_RELEASE", "as_of": as_of, "primary": True, "use": "Guidance, KPI, and catalyst evidence."},
            ],
            "missing_evidence": [],
            "next_review": {"reason": "Test the retention proof trigger.", "date_or_trigger": "2026-11-05 results"},
            "review_request": "Choose READY LIST, WATCH, or PASS for research tracking only.",
            "live_actions_enabled": False,
        }

    return _build
