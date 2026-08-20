"""Canonical columns and validation helpers for sleeve artifacts."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


SOURCE_LABELS = {
    "fact_source_reported",
    "fact_provider_standardized",
    "derived_calculation",
    "issuer_management_claim",
    "analyst_interpretation",
    "assumption_user_provided",
    "assumption_inferred",
    "estimate_consensus",
    "stale_source",
    "contradicted_source",
    "missing_required_source",
    "unknown",
}

FMP_METADATA_COLUMNS = [
    "ticker",
    "endpoint",
    "source_name",
    "source_label",
    "source_url",
    "fetched_at",
    "snapshot_as_of",
    "payload_digest",
]

CANDIDATE_REQUIRED_COLUMNS = [
    "ticker",
    "company_name",
    "sector",
    "research_priority",
    "research_score",
    "score_coverage_pct",
    "trend_state",
    "actionability",
    "variant_wedge",
    "why_now",
    "first_rejection",
    "what_makes_investable",
    "what_kills_it",
    "next_workflow",
    "implementation_readiness",
    "source_posture",
    "research_route",
    "primary_gate_code",
    "primary_gate_reason",
    "business_model_lane",
    "candidate_archetype",
    "research_queue_priority",
    "screen_can_surface_review",
    "expectations_status",
    "security_readiness",
    "as_of",
]


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def validate_candidate_frame(frame: pd.DataFrame) -> None:
    require_columns(frame, CANDIDATE_REQUIRED_COLUMNS, "candidate frame")
    bad = set(frame["research_priority"].dropna().astype(str)) - {
        "A - immediate research candidate",
        "B - watchlist / needs trigger",
        "C - screen flag only",
        "Reject",
    }
    if bad:
        raise ValueError(f"unexpected research priorities: {sorted(bad)}")
    bad_routes = set(frame["research_route"].dropna().astype(str)) - {
        "HYPOTHESIS_TEST",
        "WATCH_FOR_CHANGE",
        "SPECIALIST_MODEL",
        "EVIDENCE_GAP",
        "BACKGROUND",
        "REJECT",
    }
    if bad_routes:
        raise ValueError(f"unexpected research routes: {sorted(bad_routes)}")
    if frame["screen_can_surface_review"].fillna(False).astype(bool).any():
        raise ValueError("a screen route cannot surface a reader-facing review")
    if not frame["security_readiness"].astype(str).eq("NOT_DECISION_GRADE").all():
        raise ValueError("screen candidates must remain not decision grade")
    if frame["implementation_readiness"].astype(str).str.lower().eq("ready").any():
        raise ValueError("phase-one candidates cannot be implementation ready")

