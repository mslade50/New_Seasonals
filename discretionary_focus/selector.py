"""Hard-gated, deterministic selector for the 0--2 name Focus list.

This module ranks research attention only.  It does not create a fundamental
review, portfolio proposal, sizing instruction, order ticket, or trade signal.
The producer owns source collection and normalization; this selector owns the
fail-closed gates, causal-cluster de-duplication, and attention cap.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from typing import Any, Mapping
from urllib.parse import urlparse

from .contracts import MAX_FOCUS_NAMES, RAW_PRICE_BASIS


_UNPROVEN_WEDGES = {
    "",
    "none",
    "n/a",
    "na",
    "unknown",
    "untested",
    "unproven",
}

_PRICE_LEVEL_KEYS = {
    "price",
    "pivot",
    "level",
    "trigger_price",
    "stop_price",
    "invalidation_price",
}


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _truthy_gate(value: Any) -> bool:
    if value is True:
        return True
    return _text(value).upper() in {"PASS", "ADVANCE", "ADVANCE_FOCUS", "FOCUS"}


def _finite(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    try:
        if float(value) != result:
            return None
    except (TypeError, ValueError):
        return None
    return result


def _gate_value(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _wedge_is_specific(value: Any) -> bool:
    text = _text(value)
    if text.lower() in _UNPROVEN_WEDGES:
        return False
    lowered = text.lower()
    return not lowered.startswith(("untested", "unproven", "screen cannot"))


def _price_expression_safe(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if not isinstance(value, Mapping) or not _text(value.get("condition")):
        return False
    has_numeric_price = any(
        key in value
        and not isinstance(value[key], bool)
        and isinstance(value[key], (int, float))
        and math.isfinite(float(value[key]))
        for key in _PRICE_LEVEL_KEYS
    )
    return not has_numeric_price or value.get("price_basis") == RAW_PRICE_BASIS


def _sources_valid(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    primary = False
    for source in value:
        if not isinstance(source, Mapping):
            return False
        if not all(
            _text(source.get(field))
            for field in ("source_id", "label", "url", "as_of")
        ):
            return False
        if urlparse(_text(source.get("url"))).scheme not in {"http", "https"}:
            return False
        if not isinstance(source.get("primary"), bool):
            return False
        primary = primary or source["primary"]
    return primary


def _build_technical(row: Mapping[str, Any]) -> dict[str, Any]:
    supplied = row.get("technical")
    technical = deepcopy(dict(supplied)) if isinstance(supplied, Mapping) else {}
    technical.setdefault("observed_at", row.get("observed_at"))
    technical["setup_gate"] = "PASS"
    technical["liquidity_gate"] = "PASS"
    technical.setdefault("setup_quality", row.get("setup_quality"))
    return technical


def _build_invalidation(combined: Mapping[str, Any]) -> dict[str, Any]:
    kill = _text(combined.get("kill_condition") or combined.get("thesis_kill"))
    supplied = combined.get("invalidation")
    if isinstance(supplied, Mapping) and "technical" in supplied:
        result = deepcopy(dict(supplied))
        result["thesis_kill"] = _text(result.get("thesis_kill")) or kill
        return result
    return {"technical": deepcopy(supplied), "thesis_kill": kill}


def _ranking_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Lexicographic PM priority; intentionally not a blended sector score."""

    return (
        _finite(row.get("attention_rank"), float("inf")),
        -_finite(row.get("setup_quality"), -1.0),
        -_finite(row.get("catalyst_quality"), -1.0),
        -_finite(row.get("source_quality"), -1.0),
        _finite(row.get("screen_rank"), float("inf")),
        _text(row.get("ticker")).upper(),
    )


def _candidate_card(combined: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rank": 0,
        "ticker": _text(combined.get("ticker")).upper(),
        "company_name": _text(combined.get("company_name")),
        "why_now": _text(combined.get("why_now")),
        "setup": _text(combined.get("setup")),
        "trigger": deepcopy(combined.get("trigger")),
        "invalidation": _build_invalidation(combined),
        "catalyst": _text(combined.get("catalyst")),
        "priced_in": _text(combined.get("priced_in")),
        "next_proof": _text(combined.get("next_proof")),
        "event_date": combined.get("event_date"),
        "earnings_td": combined.get("earnings_td"),
        "technical": _build_technical(combined),
        "sources": deepcopy(combined.get("sources")),
    }


def _first_gate_failure(combined: Mapping[str, Any]) -> str | None:
    if not _truthy_gate(
        _gate_value(combined, "technical_gate", "technical_pass", "setup_gate")
    ):
        return "technical_gate"
    if not _truthy_gate(
        _gate_value(combined, "liquidity_gate", "liquidity_pass")
    ):
        return "liquidity_gate"
    quality = _finite(combined.get("setup_quality"), -1.0)
    if quality < 0:
        return "setup_quality_missing"
    if not _truthy_gate(
        _gate_value(combined, "research_gate", "research_pass", "research_disposition")
    ):
        return "research_gate"
    earnings_td = _integer(combined.get("earnings_td"))
    if earnings_td is None:
        return "earnings_missing"
    if abs(earnings_td) <= 5 or earnings_td < 0:
        return "earnings_window"
    if not _text(combined.get("event_date")):
        return "earnings_missing"
    if not _text(combined.get("catalyst")):
        return "catalyst_missing"
    if not _wedge_is_specific(combined.get("variant_wedge")):
        return "variant_wedge_missing"
    if not _text(combined.get("priced_in")):
        return "priced_in_missing"
    if not _text(combined.get("kill_condition") or combined.get("thesis_kill")):
        return "kill_condition_missing"
    if not _text(combined.get("causal_cluster")):
        return "causal_cluster_missing"
    if combined.get("sources") in (None, []):
        return "sources_missing"
    if not _sources_valid(combined.get("sources")):
        return "sources_invalid"
    for field in ("company_name", "why_now", "setup", "next_proof"):
        if not _text(combined.get(field)):
            return f"{field}_missing"
    if combined.get("trigger") in (None, "", {}):
        return "trigger_missing"
    if not _price_expression_safe(combined.get("trigger")):
        return "trigger_invalid"
    if combined.get("invalidation") in (None, "", {}):
        return "invalidation_missing"
    invalidation = _build_invalidation(combined)
    if not _price_expression_safe(invalidation.get("technical")):
        return "invalidation_invalid"
    if combined.get("unresolved_financing_risk") is True:
        return "financing_risk"
    if combined.get("unresolved_dilution_risk") is True:
        return "dilution_risk"
    if combined.get("unresolved_restatement_risk") is True:
        return "restatement_risk"
    return None


def select_focus(
    technical_rows: list[dict],
    research_by_ticker: dict[str, dict],
    *,
    max_names: int = MAX_FOCUS_NAMES,
) -> tuple[list[dict], dict]:
    """Select at most two research-attention cards and return audit counts.

    Research rows override same-named narrative fields from the normalized
    technical rows.  Every promotion gate is explicit and fail-closed.  The
    returned cards deliberately omit the variant wedge and causal cluster:
    those are selection evidence, while the public v1 card carries the
    investor-readable ``why_now`` and ``priced_in`` fields.
    """

    if isinstance(max_names, bool) or not isinstance(max_names, int):
        raise ValueError("max_names must be an integer")
    if not 0 <= max_names <= MAX_FOCUS_NAMES:
        raise ValueError(f"max_names must be between 0 and {MAX_FOCUS_NAMES}")
    if not isinstance(technical_rows, list):
        raise TypeError("technical_rows must be a list")
    if not isinstance(research_by_ticker, dict):
        raise TypeError("research_by_ticker must be a dict")

    rejected: Counter[str] = Counter()
    normalized_research = {
        _text(ticker).upper(): row
        for ticker, row in research_by_ticker.items()
        if _text(ticker) and isinstance(row, Mapping)
    }
    ticker_counts = Counter(
        _text(row.get("ticker")).upper()
        for row in technical_rows
        if isinstance(row, Mapping) and _text(row.get("ticker"))
    )

    technical_pass_count = 0
    research_pass_count = 0
    survivors: list[dict[str, Any]] = []
    for raw in technical_rows:
        if not isinstance(raw, Mapping):
            rejected["invalid_row"] += 1
            continue
        ticker = _text(raw.get("ticker")).upper()
        if not ticker:
            rejected["ticker_missing"] += 1
            continue
        if ticker_counts[ticker] > 1:
            rejected["duplicate_ticker"] += 1
            continue

        research = normalized_research.get(ticker, {})
        combined = {**deepcopy(dict(raw)), **deepcopy(dict(research)), "ticker": ticker}

        technical_ok = _truthy_gate(
            _gate_value(combined, "technical_gate", "technical_pass", "setup_gate")
        ) and _truthy_gate(
            _gate_value(combined, "liquidity_gate", "liquidity_pass")
        ) and _finite(combined.get("setup_quality"), -1.0) >= 0
        if technical_ok:
            technical_pass_count += 1

        failure = _first_gate_failure(combined)
        if failure:
            rejected[failure] += 1
            continue
        card = _candidate_card(combined)
        # Full card validation needs producer-owned dates.  Validate every
        # source/level/narrative field later in validate_payload; here retain
        # the normalized evidence needed for deterministic ranking.
        research_pass_count += 1
        survivors.append(
            {
                "combined": combined,
                "card": card,
                "cluster": _text(combined.get("causal_cluster")).lower(),
            }
        )

    survivors.sort(key=lambda item: _ranking_key(item["combined"]))
    clustered: list[dict[str, Any]] = []
    seen_clusters: set[str] = set()
    for item in survivors:
        if item["cluster"] in seen_clusters:
            rejected["causal_cluster_duplicate"] += 1
            continue
        seen_clusters.add(item["cluster"])
        clustered.append(item)

    selected_items = clustered[:max_names]
    rejected["attention_cap"] += max(len(clustered) - len(selected_items), 0)
    selected = []
    for rank, item in enumerate(selected_items, start=1):
        card = item["card"]
        card["rank"] = rank
        selected.append(card)

    summary = {
        "input_count": len(technical_rows),
        "technical_pass_count": technical_pass_count,
        "research_pass_count": research_pass_count,
        "selected_count": len(selected),
        "rejected_counts": dict(sorted(rejected.items())),
    }
    return selected, summary


__all__ = ["select_focus"]
