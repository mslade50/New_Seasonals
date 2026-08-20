"""Deterministic v2 routing for the broad fundamental research funnel.

The screen exists to maximize recall and allocate research effort.  It cannot
create a reader-facing decision.  Security readiness is established only by a
validated, source-linked underwrite in :mod:`fundamental.underwrite`.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .config import SCREEN_AXIS_METRICS


SPECIALIST_LANES = {
    "financials_specialist",
    "real_estate_specialist",
    "biotech_pipeline_specialist",
}

RESEARCH_ROUTES = {
    "HYPOTHESIS_TEST",
    "WATCH_FOR_CHANGE",
    "SPECIALIST_MODEL",
    "EVIDENCE_GAP",
    "BACKGROUND",
    "REJECT",
}

ROUTE_PRECEDENCE = {
    "HYPOTHESIS_TEST": 0,
    "WATCH_FOR_CHANGE": 1,
    "SPECIALIST_MODEL": 2,
    "EVIDENCE_GAP": 3,
    "BACKGROUND": 4,
    "REJECT": 5,
}


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _axis_score(row: pd.Series, metrics: tuple[str, ...]) -> float:
    """Return the transparent mean percentile for the available axis inputs."""
    values = [_number(row.get(f"rank_{metric}")) for metric in metrics]
    present = [value for value in values if np.isfinite(value)]
    return round(float(np.mean(present)), 1) if present else float("nan")


def _business_model_lane(row: pd.Series) -> str:
    research_lane = str(row.get("research_lane") or "").lower()
    sector = str(row.get("sector") or "").lower()
    industry = str(row.get("industry") or "").lower()
    text = f"{sector} {industry}"
    if research_lane == "financials_specialist":
        if "insurance" in text:
            return "insurance"
        if "asset management" in text or "financial data" in text:
            return "asset_manager_or_financial_platform"
        return "bank_or_lender"
    if research_lane == "real_estate_specialist":
        return "reit"
    if research_lane == "biotech_pipeline_specialist":
        return "precommercial_biotech"
    if any(term in text for term in ("gold", "silver", "copper", "mining", "oil & gas", "uranium")):
        return "commodity_producer"
    if any(term in text for term in ("software", "internet content", "information technology")):
        return "recurring_revenue_or_software"
    if any(term in text for term in ("marketplace", "payments", "credit services", "exchange")):
        return "transaction_network_or_marketplace"
    if any(term in text for term in ("restaurant", "retail", "apparel", "consumer", "leisure", "travel")):
        return "consumer_or_unit_model"
    if any(term in text for term in ("industrial", "machinery", "semiconductor", "steel", "chemical")):
        return "capital_intensive_or_cyclical"
    if "utility" in text:
        return "utility_or_infrastructure"
    if any(term in text for term in ("pharmaceutical", "medical device", "healthcare")):
        return "commercial_healthcare"
    return "general_operating_company"


def _candidate_archetype(row: pd.Series) -> str:
    lane = str(row.get("business_model_lane") or "")
    if lane == "bank_or_lender" or lane in {"insurance", "asset_manager_or_financial_platform"}:
        return "financial"
    if lane == "reit":
        return "reit"
    if lane == "precommercial_biotech":
        return "biotech"
    if lane == "commodity_producer":
        return "commodity"

    quality = _number(row.get("axis_business_quality"))
    value = _number(row.get("axis_valuation_support"))
    change = _number(row.get("axis_fundamental_change"))
    trend = str(row.get("trend_state") or "UNKNOWN").upper()
    if np.isfinite(change) and change >= 75:
        return "revision_inflection"
    if np.isfinite(quality) and quality >= 75:
        if trend == "RED" or (np.isfinite(value) and value >= 65):
            return "derated_quality"
        return "quality_compounder"
    if np.isfinite(value) and value >= 75:
        return "cash_yield_discount"
    if lane == "capital_intensive_or_cyclical":
        return "cyclical_normalization"
    return "unclassified_hypothesis"


def _research_route(row: pd.Series) -> tuple[str, str, str]:
    hard_reason = str(row.get("hard_exclusion_reason") or "").strip()
    if hard_reason:
        return "REJECT", "eligibility", hard_reason
    if str(row.get("research_lane") or "") in SPECIALIST_LANES:
        return (
            "SPECIALIST_MODEL",
            "specialist_model",
            "Baseline is current, but this business requires its dedicated economics and valuation model.",
        )
    if _number(row.get("score_coverage_pct")) < 70:
        return (
            "EVIDENCE_GAP",
            "financial_history",
            "Comparable financial history is too incomplete for reliable discovery routing.",
        )

    quality = _number(row.get("axis_business_quality"))
    value = _number(row.get("axis_valuation_support"))
    change = _number(row.get("axis_fundamental_change"))
    owner = _number(row.get("axis_owner_alignment"))

    hypothesis_test = (
        (value >= 60 and (quality >= 55 or change >= 70))
        or (quality >= 80 and value >= 45)
        or (change >= 80 and value >= 45)
    )
    if hypothesis_test:
        return (
            "HYPOTHESIS_TEST",
            "hypothesis_test",
            "The independent screen axes justify testing an expectations gap; they do not establish one.",
        )
    if (
        (quality >= 70 and value < 45)
        or (value >= 70 and quality < 55)
        or change >= 65
        or owner < 30
    ):
        return (
            "WATCH_FOR_CHANGE",
            "missing_trigger",
            "One useful dimension is present, but valuation, business quality, ownership economics, or change is not aligned.",
        )
    return (
        "BACKGROUND",
        "no_compelling_setup",
        "No sufficiently differentiated quality, valuation, or change setup is visible in the current baseline.",
    )


def _queue_priority(row: pd.Series) -> float:
    """Rank research effort only; never represent expected return or readiness."""
    axis_values = [
        _number(row.get("axis_business_quality")),
        _number(row.get("axis_valuation_support")),
        _number(row.get("axis_fundamental_change")),
        _number(row.get("axis_owner_alignment")),
    ]
    usable = [value for value in axis_values if np.isfinite(value)]
    base = float(np.mean(sorted(usable, reverse=True)[:2])) if usable else 0.0
    trend_bonus = {"GREEN": 5.0, "AMBER": 1.0, "RED": -3.0}.get(
        str(row.get("trend_state") or "UNKNOWN").upper(), 0.0
    )
    evidence_bonus = min(_number(row.get("score_coverage_pct")) / 20.0, 5.0)
    if not np.isfinite(evidence_bonus):
        evidence_bonus = 0.0
    return round(base + trend_bonus + evidence_bonus, 1)


def _secular_trend_context(row: pd.Series) -> str:
    above = row.get("above_sma1000")
    slope = _number(row.get("sma1000_slope_60d"))
    if pd.isna(above) or not np.isfinite(slope):
        return "UNAVAILABLE"
    if bool(above) and slope > 0:
        return "CONSTRUCTIVE"
    if not bool(above) and slope < 0:
        return "DAMAGED"
    return "MIXED"


def apply_research_routes(frame: pd.DataFrame) -> pd.DataFrame:
    """Add v2 lanes, independent axes, and structured gate results."""
    result = frame.copy()
    if result.empty:
        return result
    for axis, metrics in SCREEN_AXIS_METRICS.items():
        result[f"axis_{axis}"] = result.apply(lambda row: _axis_score(row, metrics), axis=1)
    result["business_model_lane"] = result.apply(_business_model_lane, axis=1)
    result["candidate_archetype"] = result.apply(_candidate_archetype, axis=1)

    routed = result.apply(_research_route, axis=1, result_type="expand")
    routed.columns = ["research_route", "primary_gate_code", "primary_gate_reason"]
    result = pd.concat([result, routed], axis=1)
    result["research_queue_priority"] = result.apply(_queue_priority, axis=1)
    result["secular_trend_context"] = result.apply(_secular_trend_context, axis=1)
    result["screen_can_surface_review"] = False
    result["trend_role"] = (
        "Timing and value-trap evidence only; trend never proves the company or security thesis."
    )
    result["expectations_status"] = "UNTESTED"
    result["security_readiness"] = "NOT_DECISION_GRADE"

    # Preserve the v1 column for compatible consumers, but derive it from the
    # structured v2 route.  A is a diligence instruction, never a recommendation.
    priority_map = {
        "HYPOTHESIS_TEST": "A - immediate research candidate",
        "WATCH_FOR_CHANGE": "B - watchlist / needs trigger",
        "SPECIALIST_MODEL": "C - screen flag only",
        "EVIDENCE_GAP": "C - screen flag only",
        "BACKGROUND": "C - screen flag only",
        "REJECT": "Reject",
    }
    result["research_priority"] = result["research_route"].map(priority_map)
    result["first_rejection"] = result["primary_gate_reason"]
    result["variant_wedge"] = (
        "UNTESTED: discovery data cannot establish what the market has mispriced."
    )
    result["why_now"] = result.apply(
        lambda row: (
            "Test the candidate archetype against current expectations and an observable evidence path."
            if row.get("research_route") == "HYPOTHESIS_TEST"
            else row.get("primary_gate_reason")
        ),
        axis=1,
    )
    result["what_makes_investable"] = (
        "A validated v2 underwrite with current primary evidence, quantified price-implied expectations, "
        "archetype-specific valuation, a realization edge, downside, and falsifiers."
    )
    result["implementation_readiness"] = (
        "Not implementation-ready - screen output cannot create a review or capital action"
    )
    result["_route_order"] = result["research_route"].map(ROUTE_PRECEDENCE)
    result = result.sort_values(
        ["_route_order", "research_queue_priority", "research_score"],
        ascending=[True, False, False],
        na_position="last",
    ).drop(columns="_route_order")
    return result.reset_index(drop=True)


def summarize_research_funnel(frame: pd.DataFrame) -> dict[str, Any]:
    """Return exact, mutually exclusive counts from structured gate fields."""
    if frame.empty:
        return {"total": 0, "routes": {}, "primary_gates": {}, "axes": {}}
    routes = Counter(frame["research_route"].fillna("UNKNOWN").astype(str))
    gates = Counter(frame["primary_gate_code"].fillna("unknown").astype(str))
    axes: dict[str, dict[str, int]] = {}
    for axis in SCREEN_AXIS_METRICS:
        column = pd.to_numeric(frame.get(f"axis_{axis}"), errors="coerce")
        axes[axis] = {
            "available": int(column.notna().sum()),
            "strong_70_plus": int(column.ge(70).sum()),
            "weak_below_30": int(column.lt(30).sum()),
        }
    return {
        "total": int(len(frame)),
        "routes": dict(sorted(routes.items())),
        "primary_gates": dict(sorted(gates.items())),
        "axes": axes,
        "method": (
            "Routes and primary gates are mutually exclusive. Axis counts overlap and are discovery diagnostics only."
        ),
    }


__all__ = [
    "RESEARCH_ROUTES",
    "apply_research_routes",
    "summarize_research_funnel",
]
