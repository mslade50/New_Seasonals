"""Deterministic strategy-discovery classification pipeline."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

from .contracts import (
    COMPLETENESS,
    CONDITION_OPERATORS,
    ORDER_TYPES,
    ContractError,
    canonical_json,
    parse_timestamp,
    sha256_json,
    validate_catalog,
    validate_config,
    validate_item,
    validate_manifest,
    validate_transition,
    validate_validation_artifacts,
)
from .journal import (
    accepted_owner_transitions,
    accepted_validations,
    candidate_state_history,
    committed_run_id,
    latest_source_captures,
    observed_source_captures,
    transaction_commitment_digest,
    validate_journal_records,
    validation_record_times,
)

STATUS_ORDER = {"COMPLETE": 0, "PARTIAL": 1, "UNKNOWN": 2}
PROCESSOR_VERSION = "1.0.6"
INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(?:all\s+|any\s+|the\s+|previous\s+)*instructions", re.IGNORECASE),
    re.compile(r"(?:system|developer)\s+prompt", re.IGNORECASE),
    re.compile(r"reveal.{0,40}(?:secret|password|api[- ]?key|token)", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:run|execute)\s+(?:this\s+)?(?:command|powershell|cmd|bash|curl|python)", re.IGNORECASE),
    re.compile(r"<\s*script\b", re.IGNORECASE),
    re.compile(r"begin\s+(?:system|developer)\s+message", re.IGNORECASE),
)
PLACEHOLDER_VALUES = {
    "",
    "missing",
    "n/a",
    "na",
    "none",
    "not available",
    "not applicable",
    "not_applicable",
    "tbd",
    "unknown",
}
def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, bool):
        return value
    if type(value) is int:
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, list):
        return sorted((_normalize_value(v) for v in value), key=canonical_json)
    if isinstance(value, dict):
        return {key: _normalize_value(value[key]) for key in sorted(value)}
    return value


def _is_substantive(value: str | None) -> bool:
    return value is not None and _normalize_text(value) not in PLACEHOLDER_VALUES


def _value_is_substantive(value: Any) -> bool:
    if isinstance(value, str):
        return _is_substantive(value)
    if isinstance(value, list):
        return bool(value) and all(_value_is_substantive(member) for member in value)
    return value is not None


def structural_spec(proposal: dict[str, Any]) -> dict[str, Any]:
    """Return the execution structure used for cross-source deduplication.

    Names, prose, claimed performance, costs, and borrow assumptions are
    intentionally excluded: changing any of those does not create a new
    signal/entry/exit strategy.
    """

    normalized_conditions = [
        {
            "field": _normalize_text(condition["field"]),
            "operator": _normalize_text(condition["operator"]),
            "value": _normalize_value(condition["value"]),
            "unit": _normalize_text(condition["unit"]) if condition["unit"] else None,
            "lookback_sessions": condition["lookback_sessions"],
        }
        for condition in proposal["signal"]["conditions"]
    ]
    conditions = {
        canonical_json(condition): condition for condition in normalized_conditions
    }
    requirements = [
        {key: _normalize_text(requirement[key]) for key in ("field", "frequency", "availability")}
        for requirement in proposal["data_requirements"]
    ]
    return {
        "direction": proposal["direction"],
        "universe": {
            "asset_class": proposal["universe"]["asset_class"],
            "scope": _normalize_text(proposal["universe"]["scope"]),
            "instruments": sorted({ticker.upper() for ticker in proposal["universe"]["instruments"]}),
        },
        "universe_history": deepcopy(proposal["universe_history"]),
        "signal": {
            "observation_timing": proposal["signal"]["observation_timing"],
            "decision_lead_minutes": proposal["signal"]["decision_lead_minutes"],
            "conditions": [conditions[key] for key in sorted(conditions)],
        },
        "entry": {
            "session_offset": proposal["entry"]["session_offset"],
            "timing": proposal["entry"]["timing"],
            "order_type": _normalize_text(proposal["entry"]["order_type"]),
            "price_rule": _normalize_text(proposal["entry"]["price_rule"])
            if proposal["entry"]["price_rule"]
            else None,
        },
        "exit": {
            "time_stop_sessions": proposal["exit"]["time_stop_sessions"],
            "stop_rule": _normalize_text(proposal["exit"]["stop_rule"])
            if proposal["exit"]["stop_rule"]
            else None,
            "target_rule": _normalize_text(proposal["exit"]["target_rule"])
            if proposal["exit"]["target_rule"]
            else None,
        },
        "data_requirements": sorted(requirements, key=canonical_json),
    }


def structural_fingerprint(proposal: dict[str, Any]) -> str:
    fingerprint_spec = structural_spec(proposal)
    # Data vendor/field phrasing belongs to feasibility, not strategy identity.
    # Two descriptions of the same executable signal must still deduplicate.
    fingerprint_spec.pop("data_requirements")
    fingerprint_spec.pop("universe_history")
    return sha256_json(fingerprint_spec)


def research_spec(proposal: dict[str, Any]) -> dict[str, Any]:
    """Canonical validation target, separate from broad structural identity.

    Structural fingerprints intentionally deduplicate marketing variants. This
    stricter projection binds every field that changes what would be executed,
    researched, costed, falsified, or judged investable. Source narrative
    (name, thesis, why-now, wedge, and portfolio-fit prose) remains provenance:
    the reproducible artifact does not validate marketing wording.
    """

    spec = {
        key: deepcopy(proposal[key])
        for key in (
            "falsifiers",
            "direction",
            "universe",
            "universe_history",
            "signal",
            "entry",
            "exit",
            "data_requirements",
            "costs",
            "borrow",
            "capacity",
            "investable_if",
            "explicit_unknowns",
        )
    }
    normalized = _normalize_value(spec)
    structural = structural_spec(proposal)
    normalized["signal"]["conditions"] = structural["signal"]["conditions"]
    normalized["universe"]["instruments"] = structural["universe"]["instruments"]
    requirements = {
        canonical_json(requirement): requirement
        for requirement in normalized["data_requirements"]
    }
    normalized["data_requirements"] = [
        requirements[key] for key in sorted(requirements)
    ]
    for key in ("falsifiers", "investable_if", "explicit_unknowns"):
        normalized[key] = sorted(set(normalized[key]))
    return normalized


def research_spec_digest(proposal: dict[str, Any]) -> str:
    return sha256_json(research_spec(proposal))


def _contains_prompt_injection(item: dict[str, Any]) -> tuple[bool, str | None]:
    text = canonical_json(
        {
            "text": item["text"],
            "claims": item["claims"],
            "proposal": item["strategy_proposal"],
        }
    )
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            return True, pattern.pattern
    return False, None


def feasibility_gates(
    proposal: dict[str, Any],
    *,
    prompt_injection: tuple[bool, str | None],
) -> list[dict[str, str]]:
    gates: list[dict[str, str]] = []

    def gate(name: str, passed: bool, reason: str) -> None:
        gates.append({"gate": name, "status": "PASS" if passed else "FAIL", "reason": reason})

    injected, pattern = prompt_injection
    gate(
        "PROMPT_INJECTION",
        not injected,
        "No instruction-like payload detected."
        if not injected
        else f"Quarantined untrusted instruction pattern: {pattern}",
    )
    conditions = proposal["signal"]["conditions"]
    conditions_ok = bool(conditions) and all(
        _is_substantive(condition["field"])
        and _normalize_text(condition["operator"]) in CONDITION_OPERATORS
        and _value_is_substantive(condition["value"])
        and (
            condition["unit"] is None
            or _is_substantive(condition["unit"])
        )
        for condition in conditions
    )
    gate(
        "SIGNAL_SPEC",
        conditions_ok,
        "Signal has deterministic, non-placeholder structured conditions."
        if conditions_ok
        else "Signal fields/operators/values are missing, unsupported, or placeholders.",
    )
    exit_spec = proposal["exit"]
    bounded_exit = exit_spec["time_stop_sessions"] is not None or any(
        _is_substantive(exit_spec[key]) for key in ("stop_rule", "target_rule")
    )
    entry = proposal["entry"]
    order_type = entry["order_type"]
    price_rule_required = order_type in {"LIMIT", "LOC", "LOO", "STOP", "STOP_LIMIT"}
    price_rule_forbidden = order_type in {"MARKET", "MOC", "MOO"}
    order_timing_ok = not (
        (order_type in {"MOO", "LOO"} and entry["timing"] != "OPEN")
        or (order_type in {"MOC", "LOC"} and entry["timing"] != "CLOSE")
    )
    entry_spec_ok = (
        order_type in ORDER_TYPES
        and order_timing_ok
        and (entry["price_rule"] is None or _is_substantive(entry["price_rule"]))
        and (not price_rule_required or _is_substantive(entry["price_rule"]))
        and (not price_rule_forbidden or entry["price_rule"] is None)
    )
    gate(
        "ENTRY_EXIT_SPEC",
        bounded_exit and entry_spec_ok,
        "Entry order and at least one bounded exit rule are specified."
        if bounded_exit and entry_spec_ok
        else "Entry order or bounded time-stop/stop/target specification is missing or placeholder.",
    )
    signal = proposal["signal"]
    impossible_close = (
        signal["observation_timing"] in {"CLOSE", "CLOSE_FINAL"}
        and entry["session_offset"] == 0
    )
    same_session = entry["session_offset"] == 0
    observation_timing = signal["observation_timing"]
    entry_timing = entry["timing"]
    impossible_earlier_boundary = same_session and (
        (observation_timing == "PREOPEN" and entry_timing == "PREOPEN")
        or (
            observation_timing == "OPEN"
            and entry_timing in {"PREOPEN", "OPEN"}
        )
        or (
            observation_timing == "INTRADAY"
            and entry_timing in {"PREOPEN", "OPEN"}
        )
    )
    insufficient_lead = (
        signal["observation_timing"] == "INTRADAY"
        and same_session
        and entry["timing"] == "CLOSE"
        and (signal["decision_lead_minutes"] is None or signal["decision_lead_minutes"] < 5)
    )
    ambiguous_intraday_order = (
        signal["observation_timing"] == "INTRADAY"
        and same_session
        and entry["timing"] == "INTRADAY"
    )
    timing_ok = (
        not impossible_close
        and not impossible_earlier_boundary
        and not insufficient_lead
        and not ambiguous_intraday_order
    )
    timing_reason = "Signal is available before the proposed execution boundary."
    if impossible_close:
        timing_reason = "Final-close data cannot cause an entry at that same close."
    elif impossible_earlier_boundary:
        timing_reason = (
            "A signal cannot cause an entry at the same or an already-fixed "
            "same-session execution boundary."
        )
    elif insufficient_lead:
        timing_reason = "Same-session close execution needs at least five minutes of declared decision lead."
    elif ambiguous_intraday_order:
        timing_reason = (
            "Same-session intraday observation and entry need explicit ordered clocks; "
            "the current schema does not provide them."
        )
    gate("TIMING_FEASIBILITY", timing_ok, timing_reason)

    rationale_values = [
        proposal["name"],
        proposal["thesis"],
        proposal["why_now"],
        proposal["variant_wedge"],
        proposal["portfolio_fit_hypothesis"],
        proposal["universe"]["scope"],
    ]
    rationale_ok = all(_is_substantive(value) for value in rationale_values)
    gate(
        "RESEARCH_RATIONALE",
        rationale_ok,
        "Name, thesis, why-now, wedge, portfolio role, and universe scope are substantive."
        if rationale_ok
        else "Decision-useful strategy rationale contains a missing or placeholder value.",
    )
    falsifiers_ok = bool(proposal["falsifiers"]) and all(
        _is_substantive(value) for value in proposal["falsifiers"]
    )
    gate(
        "FALSIFIERS",
        falsifiers_ok,
        "Every declared kill criterion is substantive."
        if falsifiers_ok
        else "Every falsifier/kill criterion must be substantive, not a placeholder.",
    )

    costs = proposal["costs"]
    costs_ok = bool(
        costs
        and 0 <= costs["commission_bps"] <= 10_000
        and 0 <= costs["slippage_bps"] <= 10_000
        and _is_substantive(costs["market_impact_model"])
    )
    gate(
        "COST_MODEL",
        costs_ok,
        "Commission, slippage, and market-impact assumptions are explicit."
        if costs_ok
        else "Commission, slippage, and a market-impact model are required before research.",
    )
    direction_requires_borrow = proposal["direction"] in {"SHORT", "BOTH"}
    borrow = proposal["borrow"]
    borrow_ok = True
    borrow_reason = "Borrow is not required for this long-only proposal."
    if direction_requires_borrow:
        borrow_ok = bool(
            borrow
            and borrow["required"] is True
            and _is_substantive(borrow["availability_check"])
            and borrow["fee_assumption_bps_annual"] is not None
            and 0 <= borrow["fee_assumption_bps_annual"] <= 1_000_000
        )
        borrow_reason = (
            "Borrow availability and fee assumptions are explicit."
            if borrow_ok
            else "Short exposure requires an availability check and annualized borrow-fee assumption."
        )
    elif borrow is None:
        borrow_ok = False
        borrow_reason = "Borrow applicability must be stated explicitly, even when not required."
    gate("BORROW", borrow_ok, borrow_reason)

    requirements = proposal["data_requirements"]
    requirements_ok = bool(requirements) and all(
        _is_substantive(requirement[field])
        for requirement in requirements
        for field in ("field", "frequency", "availability")
    )
    gate(
        "DATA_FEASIBILITY",
        requirements_ok,
        "Required fields, cadence, and availability are substantively declared."
        if requirements_ok
        else "Data requirements are missing or contain placeholder field/cadence/availability values.",
    )
    history = proposal["universe_history"]
    membership_mode = history["membership_mode"]
    pit_ok = (
        membership_mode == "POINT_IN_TIME"
        and history["includes_delisted"] is True
        and history["models_delisting_returns"] is True
        and _is_substantive(history["evidence_reference"])
    ) or (
        membership_mode == "FIXED_INSTRUMENTS"
        and bool(proposal["universe"]["instruments"])
        and all(
            _is_substantive(instrument)
            for instrument in proposal["universe"]["instruments"]
        )
        and _is_substantive(history["evidence_reference"])
    )
    pit_reason = (
        "Universe uses point-in-time membership with delisted securities and "
        "delisting returns, or an explicit fixed-instrument set."
        if pit_ok
        else "Current/static membership or missing delisting controls cannot support research readiness."
    )
    gate("PIT_UNIVERSE_AND_DELISTING", pit_ok, pit_reason)

    capacity = proposal["capacity"]
    capacity_ok = bool(
        capacity
        and capacity["median_daily_dollar_volume_usd"] > 0
        and capacity["median_daily_dollar_volume_usd"] <= 10**15
        and 0 < capacity["max_participation_rate_pct"] <= 100
        and capacity["estimated_strategy_capacity_usd"] is not None
        and capacity["estimated_strategy_capacity_usd"] > 0
        and capacity["estimated_strategy_capacity_usd"] <= 10**15
        and _is_substantive(capacity["methodology"])
    )
    investment_case_ok = (
        _is_substantive(proposal["why_now"])
        and _is_substantive(proposal["variant_wedge"])
        and bool(proposal["investable_if"])
        and all(_is_substantive(value) for value in proposal["investable_if"])
        and bool(proposal["explicit_unknowns"])
        and all(_is_substantive(value) for value in proposal["explicit_unknowns"])
        and bool(proposal["downstream_workflow"])
        and all(_is_substantive(value) for value in proposal["downstream_workflow"])
    )
    gate(
        "CAPACITY_AND_INVESTABILITY",
        capacity_ok and investment_case_ok,
        (
            "Numeric capacity assumptions, why-now, variant wedge, investability "
            "conditions, explicit unknowns, and downstream research workflow are declared."
            if capacity_ok and investment_case_ok
            else "Capacity and the decision-useful investment case are incomplete or placeholders."
        ),
    )
    return gates


def _capture_digest(source: dict[str, Any], items: list[dict[str, Any]]) -> str:
    return sha256_json(
        {
            "source": source,
            "items": sorted(items, key=lambda item: (item["item_id"], canonical_json(item))),
        }
    )


def _cursor_anchor(previous: dict[str, Any]) -> dict[str, Any]:
    """Project only the accepted prior state used for cursor continuity."""

    return {
        "source_id": previous["source_id"],
        "capture_id": previous["capture_id"],
        "capture_digest": previous["capture_digest"],
        "cursor_out": previous["cursor_out"],
        "status": previous["status"],
    }


def _source_coverage(
    source: dict[str, Any],
    source_items: list[dict[str, Any]],
    *,
    as_of: Any,
    max_age_hours: float,
    previous: dict[str, Any] | None,
    prior_observation: dict[str, Any] | None,
    provider: str,
    provider_version: str,
) -> dict[str, Any]:
    findings: list[str] = []
    status = "COMPLETE"

    def worsen(new_status: str, finding: str) -> None:
        nonlocal status
        if STATUS_ORDER[new_status] > STATUS_ORDER[status]:
            status = new_status
        findings.append(finding)

    actual_count = len(source_items)
    if source["provider_status"] == "ERROR":
        worsen("UNKNOWN", "Provider reported ERROR; zero items is not evidence of no posts.")
    elif source["provider_status"] == "PARTIAL":
        worsen("PARTIAL", "Provider explicitly reported partial capture.")
    if source["observed_item_count"] != actual_count:
        worsen(
            "UNKNOWN",
            f"Manifest observed {source['observed_item_count']} item(s), file contains {actual_count}.",
        )
    expected = source["expected_item_count"]
    if expected is not None and expected != actual_count:
        worsen("PARTIAL", f"Expected {expected} item(s), observed {actual_count}.")
    if actual_count < source["expected_min_items"]:
        worsen(
            "PARTIAL",
            f"Observed {actual_count}, below minimum {source['expected_min_items']}.",
        )
    if not source["cursor"]["exhausted"]:
        worsen("PARTIAL", "Provider cursor is not exhausted for the requested window.")
    window_end = parse_timestamp(source["window"]["end"], "source.window.end")
    captured_at = parse_timestamp(source["captured_at"], "source.captured_at")
    if captured_at > as_of:
        worsen("UNKNOWN", "Capture timestamp is after report as_of.")
    if window_end > as_of:
        worsen("UNKNOWN", "Capture window ends after report as_of.")
    elif as_of - window_end > timedelta(hours=max_age_hours):
        worsen("PARTIAL", f"Capture window is older than {max_age_hours:g} hours.")
    for item in source_items:
        created = parse_timestamp(item["created_at"], "item.created_at")
        item_captured = parse_timestamp(item["captured_at"], "item.captured_at")
        start = parse_timestamp(source["window"]["start"], "source.window.start")
        end = parse_timestamp(source["window"]["end"], "source.window.end")
        if created < start or created > end:
            worsen("UNKNOWN", f"Item {item['item_id']} falls outside the declared capture window.")
        if item_captured > captured_at:
            worsen("UNKNOWN", f"Item {item['item_id']} was captured after its manifest snapshot.")
    digest = _capture_digest(source, source_items)
    continuity_context: dict[str, Any]
    exact_replay = (
        prior_observation is not None
        and prior_observation.get("capture_digest") == digest
    )
    if exact_replay and isinstance(prior_observation.get("continuity_context"), dict):
        # Replay the original continuity decision, rather than treating an
        # already-observed capture as its own predecessor.
        continuity_context = deepcopy(prior_observation["continuity_context"])
    elif previous is not None and previous.get("capture_id") != source["capture_id"]:
        continuity_context = {
            "basis": "ACCEPTED_CAPTURE",
            "anchor": _cursor_anchor(previous),
        }
    else:
        continuity_context = {"basis": "GENESIS", "anchor": None}

    if prior_observation is not None and prior_observation.get("capture_digest") != digest:
        worsen("UNKNOWN", "Replayed capture_id has different content.")

    anchor = continuity_context["anchor"]
    if continuity_context["basis"] == "ACCEPTED_CAPTURE":
        if source["cursor"]["in"] != anchor["cursor_out"]:
            worsen(
                "UNKNOWN",
                "cursor.in does not continue the latest journaled cursor.out.",
            )
    elif source["cursor"]["in"] is not None:
        worsen("UNKNOWN", "First local capture starts from an unverifiable non-null cursor.")
    if prior_observation is not None and prior_observation.get("status") in {
        "PARTIAL",
        "UNKNOWN",
    }:
        prior_status = prior_observation["status"]
        if STATUS_ORDER[prior_status] > STATUS_ORDER[status]:
            worsen(
                prior_status,
                f"Prior immutable observation classified this capture {prior_status}; "
                "same-ID replay cannot improve it.",
            )
    if not findings:
        findings.append(
            f"Expected {expected if expected is not None else 'unspecified exact count'}; "
            f"observed {actual_count}; provider OK; cursor exhausted."
        )
    return {
        "source_id": source["source_id"],
        "platform": "X",
        "discovery_only": True,
        "provider": provider,
        "provider_version": provider_version,
        "provider_status": source["provider_status"],
        "locator": deepcopy(source["locator"]),
        "capture_id": source["capture_id"],
        "captured_at": source["captured_at"],
        "window": deepcopy(source["window"]),
        "cursor_in": source["cursor"]["in"],
        "cursor_out": source["cursor"]["out"],
        "cursor_exhausted": source["cursor"]["exhausted"],
        "expected_item_count": expected,
        "expected_min_items": source["expected_min_items"],
        "manifest_observed_item_count": source["observed_item_count"],
        "file_observed_item_count": actual_count,
        "status": status,
        "findings": findings,
        "capture_digest": digest,
        "continuity_context": continuity_context,
    }


def _catalog_health(catalog: dict[str, Any], as_of: Any, max_age_days: int) -> dict[str, Any]:
    generated = parse_timestamp(catalog["generated_at"], "catalog.generated_at")
    snapshot_as_of = parse_timestamp(catalog["as_of"], "catalog.as_of")
    if generated > as_of or snapshot_as_of > as_of:
        status = "UNKNOWN"
        finding = "Catalog generation or data timestamp is after report as_of."
    elif (
        as_of - generated > timedelta(days=max_age_days)
        or as_of - snapshot_as_of > timedelta(days=max_age_days)
    ):
        status = "PARTIAL"
        finding = f"Catalog generation or data timestamp is older than {max_age_days} days."
    else:
        status = "COMPLETE"
        finding = "Digest verified and freshness policy passed."
    return {
        "snapshot_id": catalog["snapshot_id"],
        "catalog_type": catalog["catalog_type"],
        "generated_at": catalog["generated_at"],
        "as_of": catalog["as_of"],
        "records_digest": catalog["records_digest"],
        "record_count": len(catalog["records"]),
        "status": status,
        "finding": finding,
    }


def _native_content(item: dict[str, Any]) -> dict[str, Any]:
    """Stable native-post projection, excluding observation metadata."""

    return {
        key: deepcopy(item[key])
        for key in (
            "platform",
            "kind",
            "post_id",
            "canonical_post_id",
            "thread_id",
            "parent_post_id",
            "quoted_post_id",
            "reposted_post_id",
            "author_handle",
            "created_at",
            "text",
            "claims",
            "strategy_proposal",
        )
    }


def _observation(
    item: dict[str, Any],
    coverage_by_capture: dict[str, dict[str, Any]],
    content_hash: str,
) -> dict[str, Any]:
    coverage = coverage_by_capture[item["capture_id"]]
    return {
        "item_id": item["item_id"],
        "source_id": item["source_id"],
        "capture_id": item["capture_id"],
        "captured_at": item["captured_at"],
        "permalink": item["permalink"],
        "source_locator": deepcopy(coverage["locator"]),
        "provider": coverage["provider"],
        "provider_version": coverage["provider_version"],
        "provider_status": coverage["provider_status"],
        "source_window": deepcopy(coverage["window"]),
        "capture_digest": coverage["capture_digest"],
        "native_content_hash": content_hash,
    }


def _deduplicate_items(
    items: list[dict[str, Any]],
    coverage_by_capture: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]], list[dict[str, Any]]]:
    by_post: dict[str, dict[str, Any]] = {}
    observation_ids: dict[str, list[str]] = defaultdict(list)
    conflicting_sources: dict[str, set[str]] = defaultdict(set)
    for item in sorted(
        items,
        key=lambda row: (
            row["created_at"],
            row["post_id"],
            row["source_id"],
            row["capture_id"],
            row["item_id"],
        ),
    ):
        post_id = item["post_id"]
        native = _native_content(item)
        content_hash = sha256_json(native)
        observation = _observation(item, coverage_by_capture, content_hash)
        current = by_post.get(post_id)
        observation_ids[post_id].append(item["item_id"])
        if current is None:
            canonical = deepcopy(item)
            canonical["_native_content_hash"] = content_hash
            canonical["_observations"] = [observation]
            by_post[post_id] = canonical
            continue
        if current["_native_content_hash"] == content_hash:
            current["_observations"].append(observation)
            continue
        conflicting_sources[post_id].update(
            observation_row["source_id"] for observation_row in current["_observations"]
        )
        conflicting_sources[post_id].add(item["source_id"])
    conflicts = [
        {
            "post_id": post_id,
            "source_ids": sorted(source_ids),
            "reason": "same native post_id has conflicting content or lineage",
        }
        for post_id, source_ids in sorted(conflicting_sources.items())
    ]
    conflict_ids = set(conflicting_sources)
    # A content conflict is never resolved by observation order. Drop every
    # version so a divergent post cannot become RESEARCH_READY.
    unique = sorted(
        (row for post_id, row in by_post.items() if post_id not in conflict_ids),
        key=lambda row: (row["created_at"], row["post_id"]),
    )
    duplicates = {
        key: value for key, value in observation_ids.items() if len(value) > 1
    }
    return unique, duplicates, conflicts


def _candidate_groups(
    items: list[dict[str, Any]],
    strategy_catalog: dict[str, Any],
    dead_end_catalog: dict[str, Any],
    artifacts: list[dict[str, Any]],
    transitions: dict[str, dict[str, Any]],
    promotion_allowed: bool,
    promotion_reason: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        proposal = item["strategy_proposal"]
        if proposal is None or item["kind"] == "REPOST":
            continue
        grouped[structural_fingerprint(proposal)].append(item)
    strategy_by_fp = {record["structural_fingerprint"]: record for record in strategy_catalog["records"]}
    dead_by_fp = {record["structural_fingerprint"]: record for record in dead_end_catalog["records"]}
    artifacts_by_fp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        artifacts_by_fp[artifact["candidate_fingerprint"]].append(artifact)

    candidates: list[dict[str, Any]] = []
    for fingerprint in sorted(grouped):
        group = sorted(grouped[fingerprint], key=lambda row: (row["created_at"], row["post_id"]))
        group_spec_digests = sorted(
            {research_spec_digest(item["strategy_proposal"]) for item in group}
        )
        primary = group[0]
        proposal = primary["strategy_proposal"]
        injection_results = [_contains_prompt_injection(item) for item in group]
        injected = next((result for result in injection_results if result[0]), (False, None))
        per_item_gates = [
            feasibility_gates(
                item["strategy_proposal"],
                prompt_injection=injection_result,
            )
            for item, injection_result in zip(
                group,
                injection_results,
                strict=True,
            )
        ]
        gates: list[dict[str, str]] = []
        for gate_index, primary_gate in enumerate(per_item_gates[0]):
            observed = [item_gates[gate_index] for item_gates in per_item_gates]
            if any(gate["gate"] != primary_gate["gate"] for gate in observed):
                raise AssertionError("feasibility gate ordering diverged")
            failures = sorted(
                {gate["reason"] for gate in observed if gate["status"] == "FAIL"}
            )
            gates.append(
                {
                    "gate": primary_gate["gate"],
                    "status": "FAIL" if failures else "PASS",
                    "reason": "; ".join(failures) if failures else primary_gate["reason"],
                }
            )
        gates.append(
            {
                "gate": "RESEARCH_SPEC_CONSISTENCY",
                "status": "PASS" if len(group_spec_digests) == 1 else "FAIL",
                "reason": (
                    "All observations agree on one exact validation research spec."
                    if len(group_spec_digests) == 1
                    else "Structurally similar observations disagree on validation-relevant fields."
                ),
            }
        )
        gates.append(
            {
                "gate": "SOURCE_AND_CATALOG_COMPLETENESS",
                "status": "PASS" if promotion_allowed else "FAIL",
                "reason": promotion_reason,
            }
        )
        all_gates_pass = all(gate["status"] == "PASS" for gate in gates)

        if injected[0]:
            disposition = "QUARANTINED"
            first_rejection = next(
                {"gate": gate["gate"], "reason": gate["reason"]}
                for gate in gates
                if gate["status"] == "FAIL"
            )
        elif fingerprint in strategy_by_fp:
            disposition = "KNOWN_STRATEGY"
            first_rejection = {
                "gate": "CATALOG_DEDUPE",
                "reason": f"Matches active strategy: {strategy_by_fp[fingerprint]['name']}",
            }
        elif fingerprint in dead_by_fp:
            disposition = "KNOWN_DEAD_END"
            first_rejection = {
                "gate": "DEAD_END_DEDUPE",
                "reason": dead_by_fp[fingerprint]["rejection_reason"],
            }
        elif not promotion_allowed:
            disposition = "NEEDS_COVERAGE"
            first_rejection = {
                "gate": "SOURCE_AND_CATALOG_COMPLETENESS",
                "reason": promotion_reason,
            }
        elif not all_gates_pass:
            disposition = "NEEDS_SPEC"
            first_rejection = next(
                {"gate": gate["gate"], "reason": gate["reason"]}
                for gate in gates
                if gate["status"] == "FAIL"
            )
        else:
            disposition = "NEW_RESEARCH_CANDIDATE"
            first_rejection = None

        lifecycle = "RESEARCH_READY" if disposition == "NEW_RESEARCH_CANDIDATE" else "DISCOVERED"
        selected_spec_digest = (
            group_spec_digests[0] if len(group_spec_digests) == 1 else None
        )
        attached = (
            sorted(
                (
                    artifact
                    for artifact in artifacts_by_fp.get(fingerprint, [])
                    if artifact["research_spec_digest"] == selected_spec_digest
                ),
                key=lambda row: row["artifact_id"],
            )
            if disposition == "NEW_RESEARCH_CANDIDATE"
            else []
        )
        if attached:
            # Attachment authority was checked against a *prior* journaled
            # RESEARCH_READY event before candidate construction. This run
            # must independently pass every coverage and feasibility gate.
            lifecycle = "VALIDATED_RESEARCH"
        transition = (
            transitions.get(fingerprint)
            if disposition == "NEW_RESEARCH_CANDIDATE"
            else None
        )
        if transition:
            if lifecycle != "VALIDATED_RESEARCH":
                raise ContractError(
                    f"OWNER_REVIEW transition for {fingerprint} requires reproducible validation"
                )
            lifecycle = "OWNER_REVIEW"

        metrics: list[dict[str, Any]] = []
        claims: list[dict[str, Any]] = []
        provenance: list[dict[str, Any]] = []
        for item in group:
            source_ids = sorted(
                {observation["source_id"] for observation in item["_observations"]}
            )
            for observation in item["_observations"]:
                provenance.append(
                    {
                    **deepcopy(observation),
                    "post_id": item["post_id"],
                    "canonical_post_id": item["canonical_post_id"],
                    "thread_id": item["thread_id"],
                    "parent_post_id": item["parent_post_id"],
                    "quoted_post_id": item["quoted_post_id"],
                    "reposted_post_id": item["reposted_post_id"],
                    "kind": item["kind"],
                    "author_handle": item["author_handle"],
                    "created_at": item["created_at"],
                    }
                )
            for claim in item["claims"]:
                claims.append(
                    {
                        "source_id": source_ids[0],
                        "source_ids": source_ids,
                        "post_id": item["post_id"],
                        **deepcopy(claim),
                    }
                )
                for metric in claim["metrics"]:
                    metrics.append(
                        {
                            "source_id": source_ids[0],
                            "source_ids": source_ids,
                            "post_id": item["post_id"],
                            "claim_id": claim["claim_id"],
                            "evidence_class": "SOURCE_CLAIMED",
                            **deepcopy(metric),
                        }
                    )
        internal_metrics = [
            {"artifact_id": artifact["artifact_id"], **deepcopy(metric)}
            for artifact in attached
            for metric in artifact["metrics"]
        ]
        next_step = (
            "Owner may review the reproducible research artifact."
            if lifecycle == "OWNER_REVIEW"
            else "Record an explicit human OWNER_REVIEW transition."
            if lifecycle == "VALIDATED_RESEARCH"
            else "Build an independent point-in-time, costed, reproducible research artifact."
            if lifecycle == "RESEARCH_READY"
            else f"Resolve {first_rejection['gate']}: {first_rejection['reason']}"
        )
        candidates.append(
            {
                "fingerprint": fingerprint,
                "research_spec_digests": group_spec_digests,
                "name": proposal["name"],
                "aliases": sorted({item["strategy_proposal"]["name"] for item in group}),
                "thesis": proposal["thesis"],
                "why_now": proposal["why_now"],
                "variant_wedge": proposal["variant_wedge"],
                "portfolio_fit_hypotheses": sorted(
                    {
                        item["strategy_proposal"]["portfolio_fit_hypothesis"]
                        for item in group
                    }
                ),
                "falsifiers": sorted(
                    {
                        falsifier
                        for item in group
                        for falsifier in item["strategy_proposal"]["falsifiers"]
                    }
                ),
                "structure": structural_spec(proposal),
                "disposition": disposition,
                "lifecycle": lifecycle,
                "automatic_lifecycle_ceiling": "RESEARCH_READY",
                "gates": gates,
                "first_rejection": first_rejection,
                "source_claims": sorted(claims, key=lambda row: (row["source_id"], row["post_id"], row["claim_id"])),
                "source_claimed_metrics": sorted(
                    metrics,
                    key=lambda row: (row["source_id"], row["post_id"], row["name"]),
                ),
                "validation_artifacts": attached,
                "internally_validated_metrics": internal_metrics,
                "edge_status": "INTERNALLY_VALIDATED" if attached else "SOURCE_CLAIMS_ONLY",
                "provenance": sorted(provenance, key=lambda row: (row["source_id"], row["post_id"])),
                "duplicate_proposal_count": len(group),
                "observation_count": len(provenance),
                "research_assumptions": {
                    "costs": deepcopy(proposal["costs"]),
                    "borrow": deepcopy(proposal["borrow"]),
                    "capacity": deepcopy(proposal["capacity"]),
                },
                "investable_if": deepcopy(proposal["investable_if"]),
                "explicit_unknowns": deepcopy(proposal["explicit_unknowns"]),
                "downstream_workflow": deepcopy(proposal["downstream_workflow"]),
                "actionability": (
                    "RESEARCH_ACTIONABLE"
                    if disposition == "NEW_RESEARCH_CANDIDATE" and all_gates_pass
                    else "BLOCKED"
                ),
                "next_research_step": next_step,
                "trading_actions_enabled": False,
                "operationally_authoritative": False,
            }
        )
    return candidates


def _overall_status(statuses: list[str]) -> str:
    if not statuses:
        return "UNKNOWN"
    return max(statuses, key=lambda value: STATUS_ORDER[value])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifact_file(
    artifact: dict[str, Any],
    *,
    artifact_root: Path | None,
    as_of: Any,
) -> None:
    if artifact_root is None:
        raise ContractError("an approved artifact root is required for validation artifacts")
    root = artifact_root.resolve()
    candidate = (root / artifact["artifact_path"]).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ContractError(
            f"validation artifact escapes approved root: {artifact['artifact_path']}"
        ) from exc
    if not candidate.is_file():
        raise ContractError(f"validation artifact is not a regular local file: {candidate}")
    actual = _file_sha256(candidate)
    if actual != artifact["sha256"]:
        raise ContractError(
            f"validation artifact SHA-256 mismatch for {artifact['artifact_id']}"
        )
    created_at = parse_timestamp(artifact["created_at"], "artifact.created_at")
    if created_at > as_of:
        raise ContractError(
            f"validation artifact {artifact['artifact_id']} is after report as_of"
        )


def _state_times(
    history: dict[str, list[dict[str, Any]]],
    fingerprint: str,
    lifecycle: str,
) -> list[Any]:
    return [
        parse_timestamp(row["recorded_at"], "journal.candidate.recorded_at")
        for row in history.get(fingerprint, [])
        if row["lifecycle"] == lifecycle
    ]


def run_discovery(
    *,
    config_raw: Any,
    manifest_raw: Any,
    items_raw: list[Any],
    strategy_catalog_raw: Any,
    dead_end_catalog_raw: Any,
    journal_records: list[dict[str, Any]],
    validation_artifacts_raw: Any | None = None,
    owner_transitions_raw: list[Any] | None = None,
    artifact_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate local snapshots and return a report plus journal events."""

    journal_records = validate_journal_records(journal_records)
    config = validate_config(config_raw)
    manifest = validate_manifest(manifest_raw)
    strategy_catalog = validate_catalog(strategy_catalog_raw, "STRATEGY_BOOK", "strategy_catalog")
    dead_end_catalog = validate_catalog(dead_end_catalog_raw, "DEAD_ENDS", "dead_end_catalog")
    items = [validate_item(raw, i) for i, raw in enumerate(items_raw)]
    item_ids = [item["item_id"] for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ContractError("item_id values must be globally unique within a capture bundle")
    artifact_wrapper = (
        validate_validation_artifacts(validation_artifacts_raw)
        if validation_artifacts_raw is not None
        else {"schema_version": "1.0", "artifacts": []}
    )
    new_transitions = [
        validate_transition(raw, i) for i, raw in enumerate(owner_transitions_raw or [])
    ]
    transition_ids: set[str] = set()
    for transition in new_transitions:
        if transition["transition_id"] in transition_ids:
            raise ContractError(f"duplicate transition_id: {transition['transition_id']}")
        transition_ids.add(transition["transition_id"])

    as_of = parse_timestamp(config["as_of"], "config.as_of")
    mode = config["run_mode"]
    if mode == "DISABLED" and (items or manifest["sources"]):
        raise ContractError("DISABLED mode requires an empty source manifest and empty item file")

    allowed_source_ids = set(config["required_source_ids"])
    manifest_ids = {source["source_id"] for source in manifest["sources"]}
    extra_sources = sorted(manifest_ids - allowed_source_ids)
    if extra_sources:
        raise ContractError(f"manifest contains unapproved source_id(s): {extra_sources}")
    for source in manifest["sources"]:
        approved_locator = config["source_locator_allowlist"][source["source_id"]]
        if source["locator"] != approved_locator:
            raise ContractError(
                f"source {source['source_id']} locator does not match the approved registry"
            )

    by_capture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_by_capture = {source["capture_id"]: source for source in manifest["sources"]}
    for item in items:
        source = source_by_capture.get(item["capture_id"])
        if source is None:
            raise ContractError(f"item {item['item_id']} references unknown capture_id")
        if item["source_id"] != source["source_id"]:
            raise ContractError(f"item {item['item_id']} source_id does not match its capture")
        by_capture[item["capture_id"]].append(item)

    prior_captures = latest_source_captures(journal_records)
    prior_observations = observed_source_captures(journal_records)
    coverage: list[dict[str, Any]] = []
    for source in sorted(manifest["sources"], key=lambda row: row["source_id"]):
        coverage.append(
            _source_coverage(
                source,
                by_capture.get(source["capture_id"], []),
                as_of=as_of,
                max_age_hours=float(config["source_max_age_hours"]),
                previous=prior_captures.get(source["source_id"]),
                prior_observation=prior_observations.get(
                    (source["source_id"], source["capture_id"])
                ),
                provider=manifest["provider"],
                provider_version=manifest["provider_version"],
            )
        )
    for source_id in sorted(set(config["required_source_ids"]) - manifest_ids):
        coverage.append(
            {
                "source_id": source_id,
                "platform": "X",
                "discovery_only": True,
                "provider": manifest["provider"],
                "provider_version": manifest["provider_version"],
                "provider_status": "MISSING",
                "locator": deepcopy(config["source_locator_allowlist"][source_id]),
                "capture_id": None,
                "captured_at": None,
                "window": None,
                "cursor_in": None,
                "cursor_out": None,
                "cursor_exhausted": False,
                "expected_item_count": None,
                "expected_min_items": None,
                "manifest_observed_item_count": None,
                "file_observed_item_count": 0,
                "status": "UNKNOWN",
                "findings": ["Required source is absent from the capture manifest."],
                "capture_digest": None,
                "continuity_context": {"basis": "GENESIS", "anchor": None},
            }
        )

    coverage_by_capture = {
        row["capture_id"]: row for row in coverage if row["capture_id"] is not None
    }
    unique_items, duplicate_posts, item_conflicts = _deduplicate_items(
        items,
        coverage_by_capture,
    )
    if item_conflicts:
        conflict_sources = {
            source_id
            for conflict in item_conflicts
            for source_id in conflict["source_ids"]
        }
        for row in coverage:
            if row["source_id"] in conflict_sources:
                row["status"] = "UNKNOWN"
                row["findings"].append("Conflicting payloads share a post_id; absence/presence is unsafe.")

    catalog_health = [
        _catalog_health(strategy_catalog, as_of, config["catalog_max_age_days"]),
        _catalog_health(dead_end_catalog, as_of, config["catalog_max_age_days"]),
    ]
    statuses = [row["status"] for row in coverage] + [
        row["status"] for row in catalog_health
    ]
    completeness = _overall_status(statuses)
    if mode == "DISABLED":
        completeness = "UNKNOWN"
    if completeness not in COMPLETENESS:
        raise AssertionError("invalid internal completeness state")
    promotion_allowed = completeness == "COMPLETE" and mode in {
        "FIXTURE",
        "SHADOW",
        "LIVE",
    }
    if artifact_wrapper["artifacts"] and not promotion_allowed:
        raise ContractError(
            "validation artifacts require a COMPLETE enabled discovery run"
        )
    if new_transitions and not promotion_allowed:
        raise ContractError(
            "OWNER_REVIEW transitions require a COMPLETE enabled discovery run"
        )
    promotion_reason = (
        "All configured source windows and injected catalogs are COMPLETE."
        if promotion_allowed
        else f"Run completeness is {completeness}; automatic research promotion is blocked."
    )

    current_specs_by_fp: dict[str, set[str]] = defaultdict(set)
    for item in unique_items:
        if item["strategy_proposal"] is None or item["kind"] == "REPOST":
            continue
        fingerprint = structural_fingerprint(item["strategy_proposal"])
        current_specs_by_fp[fingerprint].add(
            research_spec_digest(item["strategy_proposal"])
        )
    current_fingerprints = set(current_specs_by_fp)
    orphan_artifacts = sorted(
        artifact["artifact_id"]
        for artifact in artifact_wrapper["artifacts"]
        if artifact["candidate_fingerprint"] not in current_fingerprints
        or current_specs_by_fp[artifact["candidate_fingerprint"]]
        != {artifact["research_spec_digest"]}
    )
    if orphan_artifacts:
        raise ContractError(f"validation artifacts target unknown candidates: {orphan_artifacts}")

    history = candidate_state_history(journal_records)
    prior_validation_records = {
        str(record["payload"].get("artifact_id")): record
        for record in journal_records
        if record["event_type"] == "VALIDATION_ATTACHED"
        and record["payload"].get("artifact_id")
    }
    prior_transition_records = {
        str(record["payload"].get("transition_id")): record
        for record in journal_records
        if record["event_type"] == "OWNER_TRANSITION"
        and record["payload"].get("transition_id")
    }
    prior_validation_run_ids: dict[str, str] = {}
    prior_transition_run_ids: dict[str, str] = {}
    journal_run_id: str | None = None
    for record in journal_records:
        if record["event_type"] == "RUN":
            journal_run_id = record["payload"]["run_id"]
        elif record["event_type"] == "VALIDATION_ATTACHED":
            if journal_run_id is None:  # pragma: no cover - journal validation owns this
                raise ContractError("validation artifact lacks its journal RUN")
            prior_validation_run_ids[record["payload"]["artifact_id"]] = journal_run_id
        elif record["event_type"] == "OWNER_TRANSITION":
            if journal_run_id is None:  # pragma: no cover - journal validation owns this
                raise ContractError("owner transition lacks its journal RUN")
            prior_transition_run_ids[record["payload"]["transition_id"]] = journal_run_id
    for record in [*prior_validation_records.values(), *prior_transition_records.values()]:
        if parse_timestamp(record["recorded_at"], "journal.recorded_at") > as_of:
            raise ContractError("prior lifecycle journal event is after report as_of")

    prior_artifact_all = accepted_validations(journal_records)
    if prior_artifact_all:
        validate_validation_artifacts(
            {"schema_version": "1.0", "artifacts": list(prior_artifact_all.values())}
        )
    # A journaled validation is authoritative only if the same candidate was
    # observed RESEARCH_READY in an earlier journal sequence, and no relevant
    # clock points into the report's future.
    for artifact_id, artifact in prior_artifact_all.items():
        validation_record = prior_validation_records.get(artifact_id)
        if validation_record is None:
            raise ContractError(f"validation artifact lacks journal metadata: {artifact_id}")
        artifact_created = parse_timestamp(artifact["created_at"], "artifact.created_at")
        validation_recorded = parse_timestamp(
            validation_record["recorded_at"],
            "journal.validation.recorded_at",
        )
        ready_rows = [
            row
            for row in history.get(artifact["candidate_fingerprint"], [])
            if row["lifecycle"] == "RESEARCH_READY"
            and row["sequence"] < validation_record["sequence"]
            and artifact["research_spec_digest"] in row["research_spec_digests"]
            and parse_timestamp(row["recorded_at"], "journal.candidate.recorded_at")
            <= artifact_created
        ]
        if not ready_rows:
            raise ContractError(
                f"journaled validation {artifact_id} lacks a prior RESEARCH_READY preregistration"
            )
        if artifact_created > validation_recorded:
            raise ContractError(
                f"journaled validation {artifact_id} was recorded before artifact creation"
            )
        if artifact_created > as_of or validation_recorded > as_of:
            raise ContractError(f"journaled validation {artifact_id} is after report as_of")
        _verify_artifact_file(artifact, artifact_root=artifact_root, as_of=as_of)

    prior_artifacts = {
        artifact_id: artifact
        for artifact_id, artifact in prior_artifact_all.items()
        if artifact.get("candidate_fingerprint") in current_fingerprints
        and current_specs_by_fp[artifact["candidate_fingerprint"]]
        == {artifact.get("research_spec_digest")}
    }
    all_artifacts = dict(prior_artifacts)
    for artifact in artifact_wrapper["artifacts"]:
        fingerprint = artifact["candidate_fingerprint"]
        ready_rows = [
            row
            for row in history.get(fingerprint, [])
            if row["lifecycle"] == "RESEARCH_READY"
            and artifact["research_spec_digest"] in row["research_spec_digests"]
        ]
        if not ready_rows:
            raise ContractError(
                f"validation artifact {artifact['artifact_id']} requires a prior journaled "
                "RESEARCH_READY observation"
            )
        created_at = parse_timestamp(artifact["created_at"], "artifact.created_at")
        if not any(
            parse_timestamp(row["recorded_at"], "journal.candidate.recorded_at")
            <= created_at
            for row in ready_rows
        ):
            raise ContractError(
                f"validation artifact {artifact['artifact_id']} predates candidate preregistration"
            )
        prior = all_artifacts.get(artifact["artifact_id"])
        if prior is not None and prior != artifact:
            raise ContractError(f"conflicting validation artifact id: {artifact['artifact_id']}")
        all_artifacts[artifact["artifact_id"]] = artifact
    for artifact in artifact_wrapper["artifacts"]:
        _verify_artifact_file(artifact, artifact_root=artifact_root, as_of=as_of)

    prior_transition_payloads = accepted_owner_transitions(journal_records)
    transitions: dict[str, dict[str, Any]] = {}
    for index, payload in enumerate(prior_transition_payloads.values()):
        validated = validate_transition(payload, index)
        transition_record = prior_transition_records[validated["transition_id"]]
        transition_at = parse_timestamp(validated["recorded_at"], "transition.recorded_at")
        transition_journaled = parse_timestamp(
            transition_record["recorded_at"],
            "journal.transition.recorded_at",
        )
        if transition_at > transition_journaled:
            raise ContractError(
                f"journaled OWNER_REVIEW transition {validated['transition_id']} "
                "was journaled before its recorded_at"
            )
        if transition_at > as_of:
            raise ContractError(
                f"journaled OWNER_REVIEW transition {validated['transition_id']} is after report as_of"
            )
        prior_validated_rows = [
            row
            for row in history.get(validated["candidate_fingerprint"], [])
            if row["lifecycle"] == "VALIDATED_RESEARCH"
            and row["sequence"] < transition_record["sequence"]
            and validated["research_spec_digest"] in row["research_spec_digests"]
            and parse_timestamp(row["recorded_at"], "journal.candidate.recorded_at")
            <= transition_at
        ]
        validation_rows = [
            record
            for record in prior_validation_records.values()
            if record["payload"].get("candidate_fingerprint")
            == validated["candidate_fingerprint"]
            and record["payload"].get("research_spec_digest")
            == validated["research_spec_digest"]
            and record["sequence"] < transition_record["sequence"]
            and parse_timestamp(record["recorded_at"], "journal.validation.recorded_at")
            <= transition_at
        ]
        if not prior_validated_rows or not validation_rows:
            raise ContractError(
                f"journaled OWNER_REVIEW transition {validated['transition_id']} lacks a "
                "prior validated run and accepted validation"
            )
        if (
            validated["candidate_fingerprint"] in current_fingerprints
            and current_specs_by_fp[validated["candidate_fingerprint"]]
            == {validated["research_spec_digest"]}
        ):
            transitions[validated["candidate_fingerprint"]] = validated

    prior_validation_times = validation_record_times(journal_records)
    prior_artifacts_by_fp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in prior_artifacts.values():
        prior_artifacts_by_fp[artifact["candidate_fingerprint"]].append(artifact)

    for transition in new_transitions:
        fingerprint = transition["candidate_fingerprint"]
        if fingerprint not in current_fingerprints:
            raise ContractError(
                f"OWNER_REVIEW transition targets unknown candidate: {fingerprint}"
            )
        if current_specs_by_fp[fingerprint] != {transition["research_spec_digest"]}:
            raise ContractError(
                f"OWNER_REVIEW transition targets an unregistered current research spec: "
                f"{transition['research_spec_digest']}"
            )
        transition_at = parse_timestamp(transition["recorded_at"], "transition.recorded_at")
        if transition_at > as_of:
            raise ContractError(
                f"OWNER_REVIEW transition {transition['transition_id']} is after report as_of"
            )
        validated_times = [
            parse_timestamp(row["recorded_at"], "journal.candidate.recorded_at")
            for row in history.get(fingerprint, [])
            if row["lifecycle"] == "VALIDATED_RESEARCH"
            and transition["research_spec_digest"] in row["research_spec_digests"]
        ]
        accepted_for_candidate = [
            artifact
            for artifact in prior_artifacts_by_fp.get(fingerprint, [])
            if artifact["research_spec_digest"] == transition["research_spec_digest"]
        ]
        if not validated_times or not accepted_for_candidate:
            raise ContractError(
                f"OWNER_REVIEW transition for {fingerprint} requires a prior journaled "
                "VALIDATED_RESEARCH observation and accepted validation artifact"
            )
        validation_times = [
            parse_timestamp(
                prior_validation_times[artifact["artifact_id"]],
                "journal.validation.recorded_at",
            )
            for artifact in accepted_for_candidate
            if artifact["artifact_id"] in prior_validation_times
        ]
        if not validation_times or transition_at < max(max(validated_times), max(validation_times)):
            raise ContractError(
                f"OWNER_REVIEW transition {transition['transition_id']} violates lifecycle order"
            )
        prior = transitions.get(fingerprint)
        if prior is not None and prior != transition:
            raise ContractError(f"conflicting OWNER_REVIEW transition for {fingerprint}")
        transitions[fingerprint] = transition
    candidates = _candidate_groups(
        unique_items,
        strategy_catalog,
        dead_end_catalog,
        list(all_artifacts.values()),
        transitions,
        promotion_allowed,
        promotion_reason,
    )
    attached_artifact_ids = {
        artifact["artifact_id"]
        for candidate in candidates
        for artifact in candidate["validation_artifacts"]
    }
    unattached_artifacts = sorted(
        artifact["artifact_id"]
        for artifact in artifact_wrapper["artifacts"]
        if artifact["artifact_id"] not in attached_artifact_ids
    )
    if unattached_artifacts:
        raise ContractError(
            "validation artifacts do not attach to an eligible current candidate: "
            f"{unattached_artifacts}"
        )
    owner_candidate_specs = {
        (candidate["fingerprint"], candidate["research_spec_digests"][0])
        for candidate in candidates
        if candidate["lifecycle"] == "OWNER_REVIEW"
    }
    unattached_transitions = sorted(
        transition["transition_id"]
        for transition in new_transitions
        if (
            transition["candidate_fingerprint"],
            transition["research_spec_digest"],
        )
        not in owner_candidate_specs
    )
    if unattached_transitions:
        raise ContractError(
            "OWNER_REVIEW transitions do not attach to an eligible current candidate: "
            f"{unattached_transitions}"
        )
    normalized_config = deepcopy(config)
    normalized_config["required_source_ids"] = sorted(config["required_source_ids"])
    normalized_manifest = deepcopy(manifest)
    normalized_manifest["sources"] = sorted(
        manifest["sources"],
        key=lambda row: (row["source_id"], row["capture_id"], canonical_json(row)),
    )
    normalized_items = sorted(
        (deepcopy(item) for item in items),
        key=lambda row: (
            row["post_id"],
            row["source_id"],
            row["capture_id"],
            row["item_id"],
            canonical_json(row),
        ),
    )
    authoritative_input_material = {
        "processor_version": PROCESSOR_VERSION,
        "config": normalized_config,
        "manifest": normalized_manifest,
        "items": normalized_items,
        "strategy_catalog": {
            "snapshot_id": strategy_catalog["snapshot_id"],
            "catalog_type": strategy_catalog["catalog_type"],
            "generated_at": strategy_catalog["generated_at"],
            "as_of": strategy_catalog["as_of"],
            "digest": strategy_catalog["records_digest"],
        },
        "dead_end_catalog": {
            "snapshot_id": dead_end_catalog["snapshot_id"],
            "catalog_type": dead_end_catalog["catalog_type"],
            "generated_at": dead_end_catalog["generated_at"],
            "as_of": dead_end_catalog["as_of"],
            "digest": dead_end_catalog["records_digest"],
        },
        "validation_artifacts": {
            "schema_version": "1.0",
            "artifacts": sorted(all_artifacts.values(), key=lambda row: row["artifact_id"]),
        },
        "owner_transitions": sorted(transitions.values(), key=lambda row: row["transition_id"]),
        "coverage_continuity_context": [
            {
                "source_id": row["source_id"],
                "continuity_context": deepcopy(row["continuity_context"]),
            }
            for row in sorted(coverage, key=lambda value: value["source_id"])
        ],
    }
    input_material_digest = sha256_json(authoritative_input_material)
    summary = {
        "raw_item_count": len(items),
        "canonical_item_count": len(unique_items),
        "repost_count": sum(item["kind"] == "REPOST" for item in unique_items),
        "duplicate_post_ids": len(duplicate_posts),
        "conflicting_post_ids": len(item_conflicts),
        "candidate_count": len(candidates),
        "new_research_ready": sum(
            candidate["disposition"] == "NEW_RESEARCH_CANDIDATE"
            and candidate["lifecycle"] == "RESEARCH_READY"
            for candidate in candidates
        ),
        "validated_research": sum(candidate["lifecycle"] == "VALIDATED_RESEARCH" for candidate in candidates),
        "owner_review": sum(candidate["lifecycle"] == "OWNER_REVIEW" for candidate in candidates),
        "needs_spec": sum(candidate["disposition"] == "NEEDS_SPEC" for candidate in candidates),
        "needs_coverage": sum(
            candidate["disposition"] == "NEEDS_COVERAGE" for candidate in candidates
        ),
        "quarantined": sum(candidate["disposition"] == "QUARANTINED" for candidate in candidates),
        "known_or_dead_end": sum(
            candidate["disposition"] in {"KNOWN_STRATEGY", "KNOWN_DEAD_END"}
            for candidate in candidates
        ),
    }
    source_event_payloads = [
        {
            "source_id": row["source_id"],
            "capture_id": row["capture_id"],
            "capture_digest": row["capture_digest"],
            "captured_at": row["captured_at"],
            "provider": row["provider"],
            "provider_version": row["provider_version"],
            "provider_status": row["provider_status"],
            "locator": deepcopy(row["locator"]),
            "cursor_in": row["cursor_in"],
            "cursor_out": row["cursor_out"],
            "window": deepcopy(row["window"]),
            "status": row["status"],
            "continuity_context": deepcopy(row["continuity_context"]),
        }
        for row in coverage
        if row["capture_id"] is not None
    ]
    validation_event_payloads = sorted(
        (deepcopy(artifact) for artifact in artifact_wrapper["artifacts"]),
        key=lambda row: row["artifact_id"],
    )
    transition_event_payloads = sorted(
        (deepcopy(transition) for transition in new_transitions),
        key=lambda row: row["transition_id"],
    )
    candidate_event_payloads = [
        {
            "candidate_fingerprint": candidate["fingerprint"],
            "research_spec_digests": deepcopy(candidate["research_spec_digests"]),
            "disposition": candidate["disposition"],
            "lifecycle": candidate["lifecycle"],
            "source_post_ids": [row["post_id"] for row in candidate["provenance"]],
        }
        for candidate in candidates
    ]
    run_commitment_payload = {
        "processor_version": PROCESSOR_VERSION,
        "run_mode": mode,
        "as_of": config["as_of"],
        "completeness": completeness,
        "required_source_ids": sorted(config["required_source_ids"]),
        "summary": summary,
        "input_material_digest": input_material_digest,
    }
    transaction_commitment = transaction_commitment_digest(
        run_commitment_payload,
        sources=source_event_payloads,
        validations=validation_event_payloads,
        transitions=transition_event_payloads,
        candidates=candidate_event_payloads,
    )
    run_material = {
        "input_material_digest": input_material_digest,
        "transaction_commitment": transaction_commitment,
    }
    run_id = committed_run_id(**run_material)
    redundant_artifacts = sorted(
        artifact["artifact_id"]
        for artifact in validation_event_payloads
        if artifact["artifact_id"] in prior_validation_run_ids
        and prior_validation_run_ids[artifact["artifact_id"]] != run_id
    )
    if redundant_artifacts:
        raise ContractError(
            "already-journaled validation artifacts may only be resubmitted for "
            f"an exact same-run replay: {redundant_artifacts}"
        )
    redundant_transitions = sorted(
        transition["transition_id"]
        for transition in transition_event_payloads
        if transition["transition_id"] in prior_transition_run_ids
        and prior_transition_run_ids[transition["transition_id"]] != run_id
    )
    if redundant_transitions:
        raise ContractError(
            "already-journaled owner transitions may only be resubmitted for "
            f"an exact same-run replay: {redundant_transitions}"
        )
    report = {
        "schema_version": "1.0",
        "processor_version": PROCESSOR_VERSION,
        "report_type": "STRATEGY_DISCOVERY",
        "run_id": run_id,
        "run_mode": mode,
        "as_of": config["as_of"],
        "title": config["report_title"],
        "completeness": completeness,
        "authority": {
            "research_only": True,
            "x_is_discovery_only": True,
            "automatic_lifecycle_ceiling": "RESEARCH_READY",
            "operationally_authoritative": False,
            "trading_actions_enabled": False,
            "strategy_mutation_enabled": False,
        },
        "source_coverage": sorted(coverage, key=lambda row: row["source_id"]),
        "catalog_health": sorted(catalog_health, key=lambda row: row["catalog_type"]),
        "item_normalization": {
            "duplicate_post_ids": duplicate_posts,
            "conflicts": item_conflicts,
            "reposts_are_lineage_only": True,
            "quotes_preserve_quote_post_identity": True,
            "thread_replies_preserve_post_identity": True,
            "source_registry_digest": sha256_json(config["source_locator_allowlist"]),
        },
        "summary": summary,
        "candidates": candidates,
        "limitations": [
            "X content is untrusted discovery input, never evidence of edge.",
            "Source-claimed metrics are not recomputed or promoted.",
            "No network, email, storage, scheduler, strategy, order, or broker action occurs here.",
            (
                "COMPLETE describes configured source-window capture, not completeness "
                "of all X or all possible strategies."
            ),
        ],
    }

    events: list[dict[str, Any]] = [
        {
            "event_key": f"run:{run_id}",
            "event_type": "RUN",
            "payload": {
                "run_id": run_id,
                "processor_version": PROCESSOR_VERSION,
                "run_mode": mode,
                "as_of": config["as_of"],
                "completeness": completeness,
                "required_source_ids": sorted(config["required_source_ids"]),
                "summary": summary,
                "input_material_digest": input_material_digest,
                "transaction_commitment": transaction_commitment,
            },
        }
    ]
    for payload in source_event_payloads:
        events.append(
            {
                "event_key": (
                    f"source_capture:{run_id}:{payload['source_id']}:{payload['capture_id']}"
                ),
                "event_type": "SOURCE_CAPTURE",
                "payload": {
                    "run_id": run_id,
                    **deepcopy(payload),
                },
            }
        )
    # Authority inputs precede the observation whose lifecycle they support.
    # Both inputs themselves require evidence from an earlier run, so this
    # ordering cannot create a first-run shortcut.
    for artifact in validation_event_payloads:
        events.append(
            {
                "event_key": f"validation:{artifact['artifact_id']}",
                "event_type": "VALIDATION_ATTACHED",
                "payload": deepcopy(artifact),
            }
        )
    for transition in transition_event_payloads:
        events.append(
            {
                "event_key": f"owner_transition:{transition['transition_id']}",
                "event_type": "OWNER_TRANSITION",
                "payload": deepcopy(transition),
            }
        )
    for payload in candidate_event_payloads:
        events.append(
            {
                "event_key": f"candidate:{run_id}:{payload['candidate_fingerprint']}",
                "event_type": "CANDIDATE_OBSERVED",
                "payload": {
                    "run_id": run_id,
                    **deepcopy(payload),
                },
            }
        )
    return report, events
