"""Build the deliberately narrow private-site view of fundamental research.

The underlying research system keeps a broad universe and full evidence trail.
This module is the reader-facing adapter: it exposes only proven quick reviews,
at most three active research names, compact lenses, and audit counts.  It never
emits the background candidate queue or any capital/order instruction.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any


QUICK_REVIEW = "QUICK_REVIEW"
ACTIVE_RESEARCH = {"WAIT_FOR_PROOF", "WAIT_FOR_EVENT"}
MAX_VISIBLE_NAMES = 3
SPECIALIST_LANES = {
    "financials_specialist",
    "real_estate_specialist",
    "biotech_pipeline_specialist",
}

PASS_REASON_META = {
    "valuation_expectations": (
        "Valuation / expectations unproven",
        "The screen is interesting, but it does not yet prove what is mispriced or what the current price already assumes.",
    ),
    "specialist_underwriting": (
        "Specialist model required",
        "Financials, REITs and biotech stay in dedicated lanes instead of being forced through a generic-company score.",
    ),
    "trend_damaged": (
        "Damaged trend",
        "Price confirmation is weak enough to block advancement until the 200-day setup improves.",
    ),
    "leverage": (
        "Leverage needs a stress test",
        "Debt and normalized cash flow are the first unresolved downside questions.",
    ),
    "dilution": (
        "Per-share dilution",
        "Share-count growth is high enough to question whether business growth reaches each owner.",
    ),
    "coverage_eligibility": (
        "Coverage / eligibility gap",
        "History, data comparability, listing structure or another baseline gate is incomplete.",
    ),
    "cash_generation": (
        "Free cash flow not positive",
        "The latest period does not yet support a positive free-cash-flow screen.",
    ),
    "other": (
        "Other unresolved issue",
        "A less common first-rejection reason is keeping the company in the background queue.",
    ),
}


def _read_json(path: str | Path) -> dict[str, Any] | None:
    source = Path(path)
    if not source.exists():
        return None
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := _text(item))]


def _trend_label(candidate: dict[str, Any]) -> str:
    state = _text(candidate.get("trend_state")).upper()
    return state if state in {"GREEN", "AMBER", "RED"} else "UNAVAILABLE"


def _company_tags(
    ticker: str,
    candidate: dict[str, Any],
    circle_by_ticker: dict[str, dict[str, Any]],
    founder_tickers: set[str],
) -> dict[str, Any]:
    circle = circle_by_ticker.get(ticker, {})
    founder_led = ticker in founder_tickers or bool(circle.get("founder_led"))
    return {
        "trend_state": _trend_label(candidate),
        "product_circle": bool(circle),
        "product_fit_score": circle.get("fit_score") if circle else None,
        "product_basis": _text(circle.get("basis")) if circle else "",
        "founder_led": founder_led,
    }


def _source_rows(value: Any) -> list[dict[str, str]]:
    rows = []
    for raw in value if isinstance(value, list) else []:
        if not isinstance(raw, dict):
            continue
        rows.append(
            {
                "label": _text(raw.get("label") or raw.get("source")),
                "url": _text(raw.get("url")),
                "as_of": _text(raw.get("as_of")),
                "use": _text(raw.get("use")),
            }
        )
    return rows


def _pass_reason_category(candidate: dict[str, Any]) -> str:
    reason = _text(candidate.get("first_rejection")).lower()
    lane = _text(candidate.get("research_lane")).lower()
    if lane in SPECIALIST_LANES or "baseline covered" in reason:
        return "specialist_underwriting"
    if "expectations and valuation" in reason or "variant view" in reason:
        return "valuation_expectations"
    if "trend is damaged" in reason or "200-day trend recovers" in reason:
        return "trend_damaged"
    if "leverage" in reason or "maturities" in reason:
        return "leverage"
    if "dilution" in reason or "share count" in reason:
        return "dilution"
    if "free cash flow is not positive" in reason:
        return "cash_generation"
    if any(
        phrase in reason
        for phrase in (
            "insufficient comparable",
            "fewer than four",
            "duplicate issuer",
            "unavailable",
            "below the phase-one universe floor",
            "liquidity",
            "special situations",
        )
    ):
        return "coverage_eligibility"
    return "other"


def _pass_summary(
    candidates: list[dict[str, Any]],
    underwritten_tickers: set[str],
) -> dict[str, Any]:
    background = [
        row
        for row in candidates
        if _text(row.get("ticker")).upper() not in underwritten_tickers
    ]
    counts = Counter(_pass_reason_category(row) for row in background)
    reasons = []
    for key, count in counts.most_common():
        label, explanation = PASS_REASON_META[key]
        reasons.append(
            {
                "key": key,
                "label": label,
                "count": int(count),
                "pct": round((count / len(background) * 100.0), 1) if background else 0.0,
                "explanation": explanation,
            }
        )

    trend_counts = Counter(
        _trend_label(row) for row in background
    )
    full_confirmation = int(trend_counts.get("GREEN", 0))
    return {
        "background_count": len(background),
        "reason_method": "One primary reason per company; categories are mutually exclusive and add to the background queue.",
        "reasons": reasons,
        "trend_overlay": {
            "green": full_confirmation,
            "amber": int(trend_counts.get("AMBER", 0)),
            "red": int(trend_counts.get("RED", 0)),
            "unavailable": int(trend_counts.get("UNAVAILABLE", 0)),
            "without_full_confirmation": len(background) - full_confirmation,
            "method": "Trend is a separate, overlapping lens; amber and red lack full 200-day confirmation.",
        },
    }


def _review_row(
    decision: dict[str, Any],
    candidate: dict[str, Any],
    circle_by_ticker: dict[str, dict[str, Any]],
    founder_tickers: set[str],
) -> dict[str, Any]:
    ticker = _text(decision.get("ticker")).upper()
    proof = _strings(decision.get("proof_required"))
    kills = _strings(decision.get("kill_conditions"))
    return {
        "ticker": ticker,
        "company_name": _text(candidate.get("company_name")) or ticker,
        "decision": QUICK_REVIEW,
        "verdict": _text(decision.get("verdict")),
        "mispricing": _text(decision.get("mispricing")),
        "priced_in": _text(decision.get("priced_in")),
        "valuation": _text(decision.get("valuation")),
        "downside": kills[0] if kills else "Downside mechanism still needs definition.",
        "proof_trigger": proof[0] if proof else "No observable proof trigger has been recorded.",
        "proof_required": proof,
        "kill_conditions": kills,
        "next_review": _text(decision.get("next_review")),
        "price_as_of": _text(decision.get("price_as_of")),
        "exact_decision": "Choose whether research should DEEPEN, WATCH, or PASS.",
        "sources": _source_rows(decision.get("sources")),
        **_company_tags(ticker, candidate, circle_by_ticker, founder_tickers),
    }


def _active_row(
    decision: dict[str, Any],
    candidate: dict[str, Any],
    circle_by_ticker: dict[str, dict[str, Any]],
    founder_tickers: set[str],
) -> dict[str, Any]:
    ticker = _text(decision.get("ticker")).upper()
    return {
        "ticker": ticker,
        "company_name": _text(candidate.get("company_name")) or ticker,
        "decision": _text(decision.get("decision")),
        "verdict": _text(decision.get("verdict")),
        "next_review": _text(decision.get("next_review")),
        "price_as_of": _text(decision.get("price_as_of")),
        **_company_tags(ticker, candidate, circle_by_ticker, founder_tickers),
    }


def build_fundamental_site_payload(
    daily_path: str | Path,
    company_maps_path: str | Path,
) -> dict[str, Any] | None:
    """Return the compact site payload, or ``None`` when no daily brief exists."""
    daily = _read_json(daily_path)
    if daily is None:
        return None
    maps = _read_json(company_maps_path) or {}

    candidates = daily.get("candidates") if isinstance(daily.get("candidates"), list) else []
    candidate_by_ticker = {
        _text(row.get("ticker")).upper(): row
        for row in candidates
        if isinstance(row, dict) and _text(row.get("ticker"))
    }
    circle_rows = maps.get("circle_rows") if isinstance(maps.get("circle_rows"), list) else []
    circle_by_ticker = {
        _text(row.get("ticker")).upper(): row
        for row in circle_rows
        if isinstance(row, dict) and _text(row.get("ticker"))
    }
    founder_rows = maps.get("founder_rows") if isinstance(maps.get("founder_rows"), list) else []
    founder_tickers = {
        _text(row.get("ticker")).upper()
        for row in founder_rows
        if isinstance(row, dict) and _text(row.get("ticker"))
    }

    decisions = daily.get("underwrite_decisions")
    decisions = decisions if isinstance(decisions, list) else []
    valid_decisions = [row for row in decisions if isinstance(row, dict)]
    underwritten_tickers = {
        _text(row.get("ticker")).upper()
        for row in valid_decisions
        if _text(row.get("ticker"))
    }

    quick_reviews = []
    active_research = []
    for decision in valid_decisions:
        ticker = _text(decision.get("ticker")).upper()
        if not ticker:
            continue
        candidate = candidate_by_ticker.get(ticker, {})
        status = _text(decision.get("decision")).upper()
        if status == QUICK_REVIEW and len(quick_reviews) < MAX_VISIBLE_NAMES:
            quick_reviews.append(
                _review_row(decision, candidate, circle_by_ticker, founder_tickers)
            )
        elif status in ACTIVE_RESEARCH and len(active_research) < MAX_VISIBLE_NAMES:
            active_research.append(
                _active_row(decision, candidate, circle_by_ticker, founder_tickers)
            )

    health = daily.get("health") if isinstance(daily.get("health"), dict) else {}
    universe = health.get("universe") if isinstance(health.get("universe"), dict) else {}
    maps_meta = maps.get("meta") if isinstance(maps.get("meta"), dict) else {}
    sources = _source_rows(health.get("sources"))

    return {
        "version": 1,
        "as_of": _text(health.get("as_of") or maps_meta.get("as_of")),
        "status": QUICK_REVIEW if quick_reviews else "NO_REVIEW",
        "reviews": quick_reviews,
        "active_research": active_research,
        "portfolio": {
            "position_count": 0,
            "max_positions": 10,
            "capital_allocated_pct": 0.0,
            "capital_cap_pct": 30.0,
            "tracking_posture": "Manual allocation; no broker or order connection.",
        },
        "lenses": {
            "product_circle_count": int(maps_meta.get("circle_count") or 0),
            "founder_led_count": int(maps_meta.get("founder_active") or 0),
            "founder_product_overlap_count": len(maps_meta.get("founder_circle_overlap") or []),
            "product_excluded_count": int(maps_meta.get("circle_excluded_count") or 0),
            "trend_rule": "200-day trend is confirmation, never a substitute for valuation or thesis proof.",
        },
        "audit": {
            "discovered": int(universe.get("discovered") or 0),
            "research_eligible": int(universe.get("research_eligible") or 0),
            "fundamental_covered": int(universe.get("fundamental_covered") or 0),
            "sec_covered": int(universe.get("sec_covered") or 0),
            "scored_candidates": int(universe.get("scored_candidates") or 0),
            "completed_underwrites": len(valid_decisions),
            "pass_summary": _pass_summary(candidates, underwritten_tickers),
            "sources": sources,
            "gaps": _strings(health.get("gaps")),
        },
        "research_actions_enabled": True,
        "live_actions_enabled": False,
    }


__all__ = ["build_fundamental_site_payload"]
