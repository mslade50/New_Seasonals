"""Deterministic strategy-discovery classification pipeline."""

from __future__ import annotations

import re
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from typing import Any

from .contracts import (
    COMPLETENESS,
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
from .journal import accepted_owner_transitions, accepted_validations, latest_source_captures


STATUS_ORDER = {"COMPLETE": 0, "PARTIAL": 1, "UNKNOWN": 2}
INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(?:all\s+|any\s+|the\s+|previous\s+)*instructions", re.I),
    re.compile(r"(?:system|developer)\s+prompt", re.I),
    re.compile(r"reveal.{0,40}(?:secret|password|api[- ]?key|token)", re.I | re.S),
    re.compile(r"(?:run|execute)\s+(?:this\s+)?(?:command|powershell|cmd|bash|curl|python)", re.I),
    re.compile(r"<\s*script\b", re.I),
    re.compile(r"begin\s+(?:system|developer)\s+message", re.I),
)


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, list):
        return sorted((_normalize_value(v) for v in value), key=canonical_json)
    if isinstance(value, dict):
        return {key: _normalize_value(value[key]) for key in sorted(value)}
    return value


def structural_spec(proposal: dict[str, Any]) -> dict[str, Any]:
    """Return the execution structure used for cross-source deduplication.

    Names, prose, claimed performance, costs, and borrow assumptions are
    intentionally excluded: changing any of those does not create a new
    signal/entry/exit strategy.
    """

    conditions = [
        {
            "field": _normalize_text(condition["field"]),
            "operator": _normalize_text(condition["operator"]),
            "value": _normalize_value(condition["value"]),
            "unit": _normalize_text(condition["unit"]) if condition["unit"] else None,
            "lookback_sessions": condition["lookback_sessions"],
        }
        for condition in proposal["signal"]["conditions"]
    ]
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
        "signal": {
            "observation_timing": proposal["signal"]["observation_timing"],
            "decision_lead_minutes": proposal["signal"]["decision_lead_minutes"],
            "conditions": sorted(conditions, key=canonical_json),
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
    return sha256_json(fingerprint_spec)


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
    gate(
        "SIGNAL_SPEC",
        bool(conditions),
        "Signal has deterministic structured conditions."
        if conditions
        else "No structured signal condition was supplied.",
    )
    exit_spec = proposal["exit"]
    bounded_exit = any(exit_spec[key] is not None for key in ("time_stop_sessions", "stop_rule", "target_rule"))
    gate(
        "ENTRY_EXIT_SPEC",
        bounded_exit,
        "Entry and at least one bounded exit rule are specified."
        if bounded_exit
        else "No time stop, stop rule, or target rule was supplied.",
    )
    signal = proposal["signal"]
    entry = proposal["entry"]
    impossible_close = (
        signal["observation_timing"] in {"CLOSE", "CLOSE_FINAL"}
        and entry["session_offset"] == 0
    )
    impossible_open = (
        signal["observation_timing"] in {"OPEN", "INTRADAY"}
        and entry["session_offset"] == 0
        and entry["timing"] == "OPEN"
    )
    insufficient_lead = (
        signal["observation_timing"] == "INTRADAY"
        and entry["session_offset"] == 0
        and entry["timing"] == "CLOSE"
        and (signal["decision_lead_minutes"] is None or signal["decision_lead_minutes"] < 5)
    )
    timing_ok = not impossible_close and not impossible_open and not insufficient_lead
    timing_reason = "Signal is available before the proposed execution boundary."
    if impossible_close:
        timing_reason = "Final-close data cannot cause an entry at that same close."
    elif impossible_open:
        timing_reason = "Open or intraday observations cannot cause an entry at the already-fixed same-session open."
    elif insufficient_lead:
        timing_reason = "Same-session close execution needs at least five minutes of declared decision lead."
    gate("TIMING_FEASIBILITY", timing_ok, timing_reason)

    costs = proposal["costs"]
    invalid_sentinels = {"none", "n/a", "na", "not applicable", "missing", "unknown"}
    costs_ok = bool(
        costs
        and costs["market_impact_model"].strip()
        and _normalize_text(costs["market_impact_model"]) not in invalid_sentinels
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
            and borrow["availability_check"].strip()
            and _normalize_text(borrow["availability_check"]) not in invalid_sentinels
            and borrow["fee_assumption_bps_annual"] is not None
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
    gate(
        "DATA_FEASIBILITY",
        bool(requirements),
        "Required fields, cadence, and availability are declared."
        if requirements
        else "No data requirements were supplied.",
    )
    return gates


def _capture_digest(source: dict[str, Any], items: list[dict[str, Any]]) -> str:
    return sha256_json({"source": source, "items": sorted(items, key=lambda item: item["item_id"])})


def _source_coverage(
    source: dict[str, Any],
    source_items: list[dict[str, Any]],
    *,
    as_of: Any,
    max_age_hours: float,
    previous: dict[str, Any] | None,
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
    if previous:
        if previous.get("capture_id") == source["capture_id"]:
            if previous.get("capture_digest") != digest:
                worsen("UNKNOWN", "Replayed capture_id has different content.")
        elif source["cursor"]["in"] != previous.get("cursor_out"):
            worsen(
                "UNKNOWN",
                "cursor.in does not continue the latest journaled cursor.out.",
            )
    elif source["cursor"]["in"] is not None:
        worsen("UNKNOWN", "First local capture starts from an unverifiable non-null cursor.")
    if not findings:
        findings.append(
            f"Expected {expected if expected is not None else 'unspecified exact count'}; "
            f"observed {actual_count}; provider OK; cursor exhausted."
        )
    return {
        "source_id": source["source_id"],
        "platform": "X",
        "discovery_only": True,
        "capture_id": source["capture_id"],
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
        "records_digest": catalog["records_digest"],
        "record_count": len(catalog["records"]),
        "status": status,
        "finding": finding,
    }


def _deduplicate_items(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]], list[dict[str, str]]]:
    by_post: dict[str, dict[str, Any]] = {}
    duplicate_ids: dict[str, list[str]] = defaultdict(list)
    conflicts: list[dict[str, str]] = []
    for item in sorted(items, key=lambda row: (row["created_at"], row["post_id"], row["item_id"])):
        post_id = item["post_id"]
        current = by_post.get(post_id)
        if current is None:
            by_post[post_id] = item
            duplicate_ids[post_id].append(item["item_id"])
            continue
        if canonical_json(current) == canonical_json(item):
            duplicate_ids[post_id].append(item["item_id"])
            continue
        conflicts.append(
            {
                "post_id": post_id,
                "source_id": item["source_id"],
                "reason": "same post_id has conflicting canonical payloads",
            }
        )
    conflict_ids = {conflict["post_id"] for conflict in conflicts}
    # A conflict is not resolved by order.  Drop every version so a corrupt or
    # replay-divergent post can never become RESEARCH_READY.
    unique = sorted(
        (row for post_id, row in by_post.items() if post_id not in conflict_ids),
        key=lambda row: (row["created_at"], row["post_id"]),
    )
    return unique, {key: value for key, value in duplicate_ids.items() if len(value) > 1}, conflicts


def _candidate_groups(
    items: list[dict[str, Any]],
    strategy_catalog: dict[str, Any],
    dead_end_catalog: dict[str, Any],
    artifacts: list[dict[str, Any]],
    transitions: dict[str, dict[str, Any]],
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
        primary = group[0]
        proposal = primary["strategy_proposal"]
        injection_results = [_contains_prompt_injection(item) for item in group]
        injected = next((result for result in injection_results if result[0]), (False, None))
        gates = feasibility_gates(proposal, prompt_injection=injected)
        all_gates_pass = all(gate["status"] == "PASS" for gate in gates)

        if fingerprint in strategy_by_fp:
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
        elif injected[0]:
            disposition = "QUARANTINED"
            first_rejection = next(
                {"gate": gate["gate"], "reason": gate["reason"]}
                for gate in gates
                if gate["status"] == "FAIL"
            )
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
        attached = sorted(artifacts_by_fp.get(fingerprint, []), key=lambda row: row["artifact_id"])
        if attached and lifecycle != "RESEARCH_READY":
            raise ContractError(
                f"validation artifact targets candidate {fingerprint} before it is RESEARCH_READY"
            )
        if attached:
            lifecycle = "VALIDATED_RESEARCH"
        transition = transitions.get(fingerprint)
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
            provenance.append(
                {
                    "source_id": item["source_id"],
                    "post_id": item["post_id"],
                    "canonical_post_id": item["canonical_post_id"],
                    "thread_id": item["thread_id"],
                    "kind": item["kind"],
                    "author_handle": item["author_handle"],
                    "created_at": item["created_at"],
                    "permalink": item["permalink"],
                }
            )
            for claim in item["claims"]:
                claims.append(
                    {
                        "source_id": item["source_id"],
                        "post_id": item["post_id"],
                        **deepcopy(claim),
                    }
                )
                for metric in claim["metrics"]:
                    metrics.append(
                        {
                            "source_id": item["source_id"],
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
                "name": proposal["name"],
                "aliases": sorted({item["strategy_proposal"]["name"] for item in group}),
                "thesis": proposal["thesis"],
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
                "next_research_step": next_step,
                "trading_actions_enabled": False,
            }
        )
    return candidates


def _overall_status(statuses: list[str]) -> str:
    if not statuses:
        return "UNKNOWN"
    return max(statuses, key=lambda value: STATUS_ORDER[value])


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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate local snapshots and return a report plus journal events."""

    config = validate_config(config_raw)
    manifest = validate_manifest(manifest_raw)
    strategy_catalog = validate_catalog(strategy_catalog_raw, "STRATEGY_BOOK", "strategy_catalog")
    dead_end_catalog = validate_catalog(dead_end_catalog_raw, "DEAD_ENDS", "dead_end_catalog")
    items = [validate_item(raw, i) for i, raw in enumerate(items_raw)]
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
    coverage: list[dict[str, Any]] = []
    for source in sorted(manifest["sources"], key=lambda row: row["source_id"]):
        coverage.append(
            _source_coverage(
                source,
                by_capture.get(source["capture_id"], []),
                as_of=as_of,
                max_age_hours=float(config["source_max_age_hours"]),
                previous=prior_captures.get(source["source_id"]),
            )
        )
    manifest_ids = {source["source_id"] for source in manifest["sources"]}
    for source_id in sorted(set(config["required_source_ids"]) - manifest_ids):
        coverage.append(
            {
                "source_id": source_id,
                "platform": "X",
                "discovery_only": True,
                "capture_id": None,
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
            }
        )

    unique_items, duplicate_posts, item_conflicts = _deduplicate_items(items)
    if item_conflicts:
        conflict_sources = {conflict["source_id"] for conflict in item_conflicts}
        for row in coverage:
            if row["source_id"] in conflict_sources:
                row["status"] = "UNKNOWN"
                row["findings"].append("Conflicting payloads share a post_id; absence/presence is unsafe.")

    prior_artifacts = accepted_validations(journal_records)
    if prior_artifacts:
        validate_validation_artifacts(
            {"schema_version": "1.0", "artifacts": list(prior_artifacts.values())}
        )
    all_artifacts = dict(prior_artifacts)
    for artifact in artifact_wrapper["artifacts"]:
        prior = all_artifacts.get(artifact["artifact_id"])
        if prior is not None and prior != artifact:
            raise ContractError(f"conflicting validation artifact id: {artifact['artifact_id']}")
        all_artifacts[artifact["artifact_id"]] = artifact
    prior_transition_payloads = accepted_owner_transitions(journal_records)
    transitions = {
        validate_transition(payload, index)["candidate_fingerprint"]: payload
        for index, payload in enumerate(prior_transition_payloads.values())
    }
    for transition in new_transitions:
        fingerprint = transition["candidate_fingerprint"]
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
    )
    candidate_fingerprints = {candidate["fingerprint"] for candidate in candidates}
    orphan_artifacts = sorted(
        artifact["artifact_id"]
        for artifact in artifact_wrapper["artifacts"]
        if artifact["candidate_fingerprint"] not in candidate_fingerprints
    )
    if orphan_artifacts:
        raise ContractError(f"validation artifacts target unknown candidates: {orphan_artifacts}")
    orphan_transitions = sorted(
        {transition["candidate_fingerprint"] for transition in new_transitions}
        - candidate_fingerprints
    )
    if orphan_transitions:
        raise ContractError(f"OWNER_REVIEW transitions target unknown candidates: {orphan_transitions}")

    catalog_health = [
        _catalog_health(strategy_catalog, as_of, config["catalog_max_age_days"]),
        _catalog_health(dead_end_catalog, as_of, config["catalog_max_age_days"]),
    ]
    statuses = [row["status"] for row in coverage] + [row["status"] for row in catalog_health]
    completeness = _overall_status(statuses)
    if mode == "DISABLED":
        completeness = "UNKNOWN"
    if completeness not in COMPLETENESS:
        raise AssertionError("invalid internal completeness state")

    run_material = {
        "config": config,
        "manifest": manifest,
        "items": items,
        "strategy_catalog": {
            "snapshot_id": strategy_catalog["snapshot_id"],
            "digest": strategy_catalog["records_digest"],
        },
        "dead_end_catalog": {
            "snapshot_id": dead_end_catalog["snapshot_id"],
            "digest": dead_end_catalog["records_digest"],
        },
        "validation_artifacts": {
            "schema_version": "1.0",
            "artifacts": sorted(all_artifacts.values(), key=lambda row: row["artifact_id"]),
        },
        "owner_transitions": sorted(transitions.values(), key=lambda row: row["transition_id"]),
    }
    run_id = sha256_json(run_material)
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
        "quarantined": sum(candidate["disposition"] == "QUARANTINED" for candidate in candidates),
        "known_or_dead_end": sum(
            candidate["disposition"] in {"KNOWN_STRATEGY", "KNOWN_DEAD_END"}
            for candidate in candidates
        ),
    }
    report = {
        "schema_version": "1.0",
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
        },
        "summary": summary,
        "candidates": candidates,
        "limitations": [
            "X content is untrusted discovery input, never evidence of edge.",
            "Source-claimed metrics are not recomputed or promoted.",
            "No network, email, storage, scheduler, strategy, order, or broker action occurs here.",
            "COMPLETE describes configured source-window capture, not completeness "
            "of all X or all possible strategies.",
        ],
    }

    events: list[dict[str, Any]] = [
        {
            "event_key": f"run:{run_id}",
            "event_type": "RUN",
            "payload": {
                "run_id": run_id,
                "run_mode": mode,
                "as_of": config["as_of"],
                "completeness": completeness,
                "summary": summary,
            },
        }
    ]
    for row in coverage:
        if row["capture_id"] is None:
            continue
        events.append(
            {
                "event_key": f"source_capture:{row['capture_id']}",
                "event_type": "SOURCE_CAPTURE",
                "payload": {
                    "source_id": row["source_id"],
                    "capture_id": row["capture_id"],
                    "capture_digest": row["capture_digest"],
                    "cursor_in": row["cursor_in"],
                    "cursor_out": row["cursor_out"],
                    "window": row["window"],
                    "status": row["status"],
                },
            }
        )
    for candidate in candidates:
        events.append(
            {
                "event_key": f"candidate:{run_id}:{candidate['fingerprint']}",
                "event_type": "CANDIDATE_OBSERVED",
                "payload": {
                    "run_id": run_id,
                    "candidate_fingerprint": candidate["fingerprint"],
                    "disposition": candidate["disposition"],
                    "lifecycle": candidate["lifecycle"],
                    "source_post_ids": [row["post_id"] for row in candidate["provenance"]],
                },
            }
        )
    for artifact in sorted(artifact_wrapper["artifacts"], key=lambda row: row["artifact_id"]):
        events.append(
            {
                "event_key": f"validation:{artifact['artifact_id']}",
                "event_type": "VALIDATION_ATTACHED",
                "payload": deepcopy(artifact),
            }
        )
    for transition in sorted(new_transitions, key=lambda row: row["transition_id"]):
        events.append(
            {
                "event_key": f"owner_transition:{transition['transition_id']}",
                "event_type": "OWNER_TRANSITION",
                "payload": deepcopy(transition),
            }
        )
    return report, events
