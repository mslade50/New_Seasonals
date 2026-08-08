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
    if frame["implementation_readiness"].astype(str).str.lower().eq("ready").any():
        raise ValueError("phase-one candidates cannot be implementation ready")

