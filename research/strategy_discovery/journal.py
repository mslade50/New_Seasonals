"""Tamper-evident append-only journal for strategy-discovery runs."""

from __future__ import annotations

import json
import os
import secrets
import stat
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from .contracts import (
    COMPLETENESS,
    LIFECYCLES,
    LOCATOR_KINDS,
    PROVIDER_STATUSES,
    RUN_MODES,
    SHA256_RE,
    ContractError,
    canonical_json,
    parse_timestamp,
    sha256_json,
    validate_transition,
    validate_validation_artifacts,
)

GENESIS = "GENESIS"
EVENT_TYPES = {
    "RUN",
    "SOURCE_CAPTURE",
    "CANDIDATE_OBSERVED",
    "VALIDATION_ATTACHED",
    "OWNER_TRANSITION",
}
RECORD_KEYS = {
    "sequence",
    "event_key",
    "event_type",
    "recorded_at",
    "payload",
    "prev_hash",
    "record_hash",
}
DISPOSITIONS = {
    "KNOWN_STRATEGY",
    "KNOWN_DEAD_END",
    "QUARANTINED",
    "NEEDS_COVERAGE",
    "NEEDS_SPEC",
    "NEW_RESEARCH_CANDIDATE",
}
SUMMARY_KEYS = {
    "raw_item_count",
    "canonical_item_count",
    "repost_count",
    "duplicate_post_ids",
    "conflicting_post_ids",
    "candidate_count",
    "new_research_ready",
    "validated_research",
    "owner_review",
    "needs_spec",
    "needs_coverage",
    "quarantined",
    "known_or_dead_end",
}
TRANSACTION_RUN_KEYS = (
    "processor_version",
    "run_mode",
    "as_of",
    "completeness",
    "required_source_ids",
    "summary",
    "input_material_digest",
)


@dataclass(frozen=True)
class _JournalLease:
    path: Path
    nonce: str


def _record_hash(record_without_hash: dict[str, Any]) -> str:
    return sha256_json(record_without_hash)


def committed_run_id(
    input_material_digest: str,
    transaction_commitment: str,
) -> str:
    """Derive a run identity from both inputs and the emitted transaction."""

    return sha256_json(
        {
            "input_material_digest": input_material_digest,
            "transaction_commitment": transaction_commitment,
        }
    )


def transaction_commitment_digest(
    run_payload: dict[str, Any],
    *,
    sources: Iterable[dict[str, Any]],
    validations: Iterable[dict[str, Any]],
    transitions: Iterable[dict[str, Any]],
    candidates: Iterable[dict[str, Any]],
) -> str:
    """Commit to authority-relevant run metadata and every emitted child.

    Child ``run_id`` fields and event keys are deliberately excluded because
    they are derived only after this digest.  Array order is canonicalized so
    a semantically identical transaction has one commitment.
    """

    run_projection = {key: run_payload[key] for key in TRANSACTION_RUN_KEYS}

    def child_projection(payload: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in payload.items() if key != "run_id"}

    def canonical_children(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        projected = [child_projection(row) for row in rows]
        return sorted(projected, key=canonical_json)

    return sha256_json(
        {
            "schema_version": "1.0",
            "run": run_projection,
            "source_captures": canonical_children(sources),
            "validation_artifacts": canonical_children(validations),
            "owner_transitions": canonical_children(transitions),
            "candidate_observations": canonical_children(candidates),
        }
    )


def _exact_object(value: Any, keys: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{path} must be an object")
    missing = sorted(keys - value.keys())
    extra = sorted(value.keys() - keys)
    if missing or extra:
        raise ContractError(
            f"{path} has invalid fields; missing={missing}, extra={extra}"
        )
    return value


def _journal_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{path} must be a non-empty string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ContractError(f"{path} must contain valid Unicode") from exc
    return value


def _journal_digest(value: Any, path: str) -> str:
    digest = _journal_string(value, path)
    if not SHA256_RE.fullmatch(digest):
        raise ContractError(f"{path} must be a lowercase SHA-256")
    return digest


def _optional_journal_string(value: Any, path: str) -> str | None:
    return None if value is None else _journal_string(value, path)


def _validate_continuity_context(
    raw: Any,
    *,
    source_id: str,
    capture_id: str,
) -> None:
    context = _exact_object(raw, {"basis", "anchor"}, "SOURCE_CAPTURE.continuity_context")
    basis = context["basis"]
    if basis not in {"GENESIS", "ACCEPTED_CAPTURE"}:
        raise ContractError("SOURCE_CAPTURE.continuity_context.basis is invalid")
    if basis == "GENESIS":
        if context["anchor"] is not None:
            raise ContractError("GENESIS continuity context must have a null anchor")
        return
    anchor = _exact_object(
        context["anchor"],
        {"source_id", "capture_id", "capture_digest", "cursor_out", "status"},
        "SOURCE_CAPTURE.continuity_context.anchor",
    )
    if _journal_string(anchor["source_id"], "continuity anchor source_id") != source_id:
        raise ContractError("continuity anchor source_id must match the capture source")
    anchor_capture = _journal_string(anchor["capture_id"], "continuity anchor capture_id")
    if anchor_capture == capture_id:
        raise ContractError("continuity anchor cannot reference the current capture")
    _journal_digest(anchor["capture_digest"], "continuity anchor capture_digest")
    _optional_journal_string(anchor["cursor_out"], "continuity anchor cursor_out")
    if anchor["status"] != "COMPLETE":
        raise ContractError("continuity anchor status must be COMPLETE")


def _validate_journal_event(
    event_key: Any,
    event_type: Any,
    payload_raw: Any,
    *,
    recorded_at: str,
) -> None:
    event_key = _journal_string(event_key, "journal event_key")
    recorded_time = parse_timestamp(recorded_at, "journal recorded_at")
    payload = payload_raw
    if event_type == "RUN":
        payload = _exact_object(
            payload,
            {
                "run_id",
                "processor_version",
                "run_mode",
                "as_of",
                "completeness",
                "required_source_ids",
                "summary",
                "input_material_digest",
                "transaction_commitment",
            },
            "RUN payload",
        )
        run_id = _journal_digest(payload["run_id"], "RUN.run_id")
        input_material_digest = _journal_digest(
            payload["input_material_digest"],
            "RUN.input_material_digest",
        )
        transaction_commitment = _journal_digest(
            payload["transaction_commitment"],
            "RUN.transaction_commitment",
        )
        if run_id != committed_run_id(input_material_digest, transaction_commitment):
            raise ContractError(
                "RUN.run_id must bind input material and transaction commitment"
            )
        if event_key != f"run:{run_id}":
            raise ContractError("RUN event_key must bind its run_id")
        _journal_string(payload["processor_version"], "RUN.processor_version")
        if payload["run_mode"] not in RUN_MODES:
            raise ContractError("RUN.run_mode is invalid")
        if payload["completeness"] not in COMPLETENESS:
            raise ContractError("RUN.completeness is invalid")
        if parse_timestamp(payload["as_of"], "RUN.as_of") > recorded_time:
            raise ContractError("RUN.as_of cannot be after journal recorded_at")
        source_ids = payload["required_source_ids"]
        if not isinstance(source_ids, list):
            raise ContractError("RUN.required_source_ids must be an array")
        for index, source_id in enumerate(source_ids):
            _journal_string(source_id, f"RUN.required_source_ids[{index}]")
        if source_ids != sorted(set(source_ids)):
            raise ContractError("RUN.required_source_ids must be sorted and unique")
        if payload["run_mode"] == "DISABLED" and source_ids:
            raise ContractError("DISABLED RUN cannot declare required sources")
        if payload["run_mode"] != "DISABLED" and not source_ids:
            raise ContractError("enabled RUN requires at least one required source")
        summary = _exact_object(payload["summary"], SUMMARY_KEYS, "RUN.summary")
        if any(type(value) is not int or value < 0 for value in summary.values()):
            raise ContractError("RUN.summary values must be non-negative integers")
        if summary["canonical_item_count"] > summary["raw_item_count"]:
            raise ContractError("RUN.summary canonical items cannot exceed raw items")
        if summary["repost_count"] > summary["canonical_item_count"]:
            raise ContractError("RUN.summary reposts cannot exceed canonical items")
        if summary["candidate_count"] > summary["canonical_item_count"]:
            raise ContractError("RUN.summary candidates cannot exceed canonical items")
        if summary["duplicate_post_ids"] > summary["raw_item_count"]:
            raise ContractError("RUN.summary duplicate IDs cannot exceed raw items")
        if summary["conflicting_post_ids"] > summary["raw_item_count"]:
            raise ContractError("RUN.summary conflicts cannot exceed raw items")
        return
    if event_type == "SOURCE_CAPTURE":
        payload = _exact_object(
            payload,
            {
                "run_id",
                "source_id",
                "capture_id",
                "capture_digest",
                "captured_at",
                "provider",
                "provider_version",
                "provider_status",
                "locator",
                "cursor_in",
                "cursor_out",
                "window",
                "status",
                "continuity_context",
            },
            "SOURCE_CAPTURE payload",
        )
        run_id = _journal_digest(payload["run_id"], "SOURCE_CAPTURE.run_id")
        source_id = _journal_string(payload["source_id"], "SOURCE_CAPTURE.source_id")
        capture_id = _journal_string(payload["capture_id"], "SOURCE_CAPTURE.capture_id")
        if event_key != f"source_capture:{run_id}:{source_id}:{capture_id}":
            raise ContractError(
                "SOURCE_CAPTURE event_key must bind run_id, source_id, and capture_id"
            )
        _journal_digest(payload["capture_digest"], "SOURCE_CAPTURE.capture_digest")
        captured_at = parse_timestamp(payload["captured_at"], "SOURCE_CAPTURE.captured_at")
        if captured_at > recorded_time:
            raise ContractError("SOURCE_CAPTURE.captured_at cannot postdate journal recorded_at")
        _journal_string(payload["provider"], "SOURCE_CAPTURE.provider")
        _journal_string(payload["provider_version"], "SOURCE_CAPTURE.provider_version")
        if payload["provider_status"] not in PROVIDER_STATUSES:
            raise ContractError("SOURCE_CAPTURE.provider_status is invalid")
        locator = _exact_object(payload["locator"], {"kind", "value"}, "SOURCE_CAPTURE.locator")
        if locator["kind"] not in LOCATOR_KINDS:
            raise ContractError("SOURCE_CAPTURE.locator.kind is invalid")
        _journal_string(locator["value"], "SOURCE_CAPTURE.locator.value")
        cursor_in = _optional_journal_string(payload["cursor_in"], "SOURCE_CAPTURE.cursor_in")
        _optional_journal_string(payload["cursor_out"], "SOURCE_CAPTURE.cursor_out")
        window = _exact_object(payload["window"], {"start", "end"}, "SOURCE_CAPTURE.window")
        window_start = parse_timestamp(window["start"], "SOURCE_CAPTURE.window.start")
        window_end = parse_timestamp(window["end"], "SOURCE_CAPTURE.window.end")
        if window_end < window_start:
            raise ContractError("SOURCE_CAPTURE window end precedes start")
        if captured_at < window_end:
            raise ContractError("SOURCE_CAPTURE.captured_at must be on or after window end")
        if payload["status"] not in COMPLETENESS:
            raise ContractError("SOURCE_CAPTURE.status is invalid")
        _validate_continuity_context(
            payload["continuity_context"],
            source_id=source_id,
            capture_id=capture_id,
        )
        context = payload["continuity_context"]
        if payload["status"] == "COMPLETE":
            if payload["provider_status"] != "OK":
                raise ContractError("COMPLETE source capture requires provider_status OK")
            if context["basis"] == "GENESIS" and cursor_in is not None:
                raise ContractError("COMPLETE genesis capture requires cursor_in null")
            if (
                context["basis"] == "ACCEPTED_CAPTURE"
                and cursor_in != context["anchor"]["cursor_out"]
            ):
                raise ContractError("COMPLETE source capture violates cursor continuity")
        return
    if event_type == "CANDIDATE_OBSERVED":
        payload = _exact_object(
            payload,
            {
                "run_id",
                "candidate_fingerprint",
                "research_spec_digests",
                "disposition",
                "lifecycle",
                "source_post_ids",
            },
            "CANDIDATE_OBSERVED payload",
        )
        run_id = _journal_digest(payload["run_id"], "CANDIDATE_OBSERVED.run_id")
        fingerprint = _journal_digest(
            payload["candidate_fingerprint"],
            "CANDIDATE_OBSERVED.candidate_fingerprint",
        )
        if event_key != f"candidate:{run_id}:{fingerprint}":
            raise ContractError("CANDIDATE_OBSERVED event_key must bind run and candidate")
        digests = payload["research_spec_digests"]
        if not isinstance(digests, list) or not digests:
            raise ContractError("CANDIDATE_OBSERVED research specs must be a non-empty array")
        for index, digest in enumerate(digests):
            _journal_digest(digest, f"CANDIDATE_OBSERVED.research_spec_digests[{index}]")
        if len(digests) != len(set(digests)):
            raise ContractError("CANDIDATE_OBSERVED research specs must be unique")
        if payload["disposition"] not in DISPOSITIONS:
            raise ContractError("CANDIDATE_OBSERVED disposition is invalid")
        if payload["lifecycle"] not in LIFECYCLES:
            raise ContractError("CANDIDATE_OBSERVED lifecycle is invalid")
        if (
            (payload["lifecycle"] == "DISCOVERED")
            != (payload["disposition"] != "NEW_RESEARCH_CANDIDATE")
        ):
            raise ContractError(
                "candidate disposition and lifecycle authority must agree"
            )
        source_post_ids = payload["source_post_ids"]
        if not isinstance(source_post_ids, list) or not source_post_ids:
            raise ContractError("CANDIDATE_OBSERVED source_post_ids must be non-empty")
        for index, post_id in enumerate(source_post_ids):
            _journal_string(post_id, f"CANDIDATE_OBSERVED.source_post_ids[{index}]")
        return
    if event_type == "VALIDATION_ATTACHED":
        artifact = validate_validation_artifacts(
            {"schema_version": "1.0", "artifacts": [payload]}
        )["artifacts"][0]
        if event_key != f"validation:{artifact['artifact_id']}":
            raise ContractError("VALIDATION_ATTACHED event_key must bind artifact_id")
        if parse_timestamp(artifact["created_at"], "validation.created_at") > recorded_time:
            raise ContractError("validation artifact cannot postdate journal recorded_at")
        return
    if event_type == "OWNER_TRANSITION":
        transition = validate_transition(payload, 0)
        if event_key != f"owner_transition:{transition['transition_id']}":
            raise ContractError("OWNER_TRANSITION event_key must bind transition_id")
        if parse_timestamp(transition["recorded_at"], "transition.recorded_at") > recorded_time:
            raise ContractError("owner transition cannot postdate journal recorded_at")
        return
    raise ContractError(f"unknown journal event_type: {event_type}")


def _derived_candidate_summary(
    candidates: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "candidate_count": len(candidates),
        "new_research_ready": sum(
            candidate["lifecycle"] == "RESEARCH_READY" for candidate in candidates
        ),
        "validated_research": sum(
            candidate["lifecycle"] == "VALIDATED_RESEARCH"
            for candidate in candidates
        ),
        "owner_review": sum(
            candidate["lifecycle"] == "OWNER_REVIEW" for candidate in candidates
        ),
        "needs_spec": sum(
            candidate["disposition"] == "NEEDS_SPEC" for candidate in candidates
        ),
        "needs_coverage": sum(
            candidate["disposition"] == "NEEDS_COVERAGE"
            for candidate in candidates
        ),
        "quarantined": sum(
            candidate["disposition"] == "QUARANTINED" for candidate in candidates
        ),
        "known_or_dead_end": sum(
            candidate["disposition"] in {"KNOWN_STRATEGY", "KNOWN_DEAD_END"}
            for candidate in candidates
        ),
    }


def _finalize_run_group(
    run: dict[str, Any] | None,
    sources: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> tuple[
    set[tuple[str, str, str]],
    set[tuple[str, str, str]],
    set[tuple[str, str]],
    set[tuple[str, str]],
]:
    """Reconcile one complete journal transaction before it can seed state."""

    if run is None:
        if sources or validations or transitions or candidates:
            raise ContractError("journal child event lacks a RUN transaction")
        return set(), set(), set(), set()

    mode = run["run_mode"]
    completeness = run["completeness"]
    enabled = mode != "DISABLED"
    authority_candidates = [
        candidate
        for candidate in candidates
        if candidate["lifecycle"] != "DISCOVERED"
    ]
    if (validations or transitions or authority_candidates) and (
        not enabled or completeness != "COMPLETE"
    ):
        raise ContractError(
            "authority-bearing journal events require a COMPLETE enabled RUN"
        )

    if not enabled:
        if completeness != "UNKNOWN":
            raise ContractError("DISABLED RUN completeness must be UNKNOWN")
        if sources or validations or transitions or candidates:
            raise ContractError("DISABLED RUN cannot contain child events")
        if any(run["summary"].values()):
            raise ContractError("DISABLED RUN summary must be all zero")
    elif completeness in {"COMPLETE", "PARTIAL"} and not sources:
        raise ContractError(f"{completeness} enabled RUN requires source observations")

    required_source_ids = set(run["required_source_ids"])
    observed_source_ids = [source["source_id"] for source in sources]
    if len(observed_source_ids) != len(set(observed_source_ids)):
        raise ContractError("RUN cannot contain duplicate source observation siblings")
    if not set(observed_source_ids) <= required_source_ids:
        raise ContractError("RUN contains an unregistered source observation sibling")
    if completeness in {"COMPLETE", "PARTIAL"} and set(
        observed_source_ids
    ) != required_source_ids:
        raise ContractError(
            f"{completeness} RUN must observe every required source sibling"
        )
    source_statuses = {source["status"] for source in sources}
    if completeness == "COMPLETE" and source_statuses != {"COMPLETE"}:
        raise ContractError(
            "COMPLETE RUN requires every child source observation to be COMPLETE"
        )
    if completeness == "PARTIAL" and "UNKNOWN" in source_statuses:
        raise ContractError("PARTIAL RUN cannot contain an UNKNOWN source observation")

    derived = _derived_candidate_summary(candidates)
    for key, value in derived.items():
        if run["summary"][key] != value:
            raise ContractError(
                f"RUN.summary.{key} does not match CANDIDATE_OBSERVED children"
            )

    candidate_lifecycles: dict[tuple[str, str], str] = {}
    for candidate in candidates:
        if len(candidate["research_spec_digests"]) != 1:
            continue
        spec = (
            candidate["candidate_fingerprint"],
            candidate["research_spec_digests"][0],
        )
        candidate_lifecycles[spec] = candidate["lifecycle"]
    validation_specs_current = {
        (artifact["candidate_fingerprint"], artifact["research_spec_digest"])
        for artifact in validations
    }
    transition_specs_current = {
        (transition["candidate_fingerprint"], transition["research_spec_digest"])
        for transition in transitions
    }
    if any(
        candidate_lifecycles.get(spec)
        not in {"VALIDATED_RESEARCH", "OWNER_REVIEW"}
        for spec in validation_specs_current
    ):
        raise ContractError(
            "validation event lacks a matching validated candidate in its RUN"
        )
    if any(
        candidate_lifecycles.get(spec) != "OWNER_REVIEW"
        for spec in transition_specs_current
    ):
        raise ContractError(
            "owner transition lacks a matching OWNER_REVIEW candidate in its RUN"
        )

    actual_commitment = transaction_commitment_digest(
        run,
        sources=sources,
        validations=validations,
        transitions=transitions,
        candidates=candidates,
    )
    if actual_commitment != run["transaction_commitment"]:
        raise ContractError(
            "RUN transaction commitment does not match its emitted child events"
        )

    run_id = run["run_id"]
    ready = {
        (candidate["candidate_fingerprint"], candidate["research_spec_digests"][0])
        for candidate in candidates
        if candidate["lifecycle"] == "RESEARCH_READY"
    }
    validated = {
        (candidate["candidate_fingerprint"], candidate["research_spec_digests"][0])
        for candidate in candidates
        if candidate["lifecycle"] == "VALIDATED_RESEARCH"
    }
    return (
        {(fingerprint, digest, run_id) for fingerprint, digest in ready},
        {(fingerprint, digest, run_id) for fingerprint, digest in validated},
        validation_specs_current,
        transition_specs_current,
    )


def _validate_event_sequence(records: list[dict[str, Any]]) -> None:
    """Validate provenance and lifecycle ordering across journal events.

    The hash chain proves byte continuity, while this pass proves that records
    cannot become authority-bearing state merely by carrying a familiar event
    name.  Source and candidate observations bind a prior RUN; cursor anchors
    bind an earlier accepted source observation; and higher lifecycle states
    require their exact-spec predecessor evidence.
    """

    runs: dict[str, dict[str, Any]] = {}
    run_recorded_at: dict[str, Any] = {}
    source_observations: dict[tuple[str, str], dict[str, Any]] = {}
    source_observation_runs: dict[tuple[str, str], str] = {}
    latest_accepted: dict[str, dict[str, Any]] = {}
    ready_spec_runs: dict[tuple[str, str], set[str]] = {}
    validated_spec_runs: dict[tuple[str, str], set[str]] = {}
    validation_specs: set[tuple[str, str]] = set()
    owner_specs: set[tuple[str, str]] = set()
    previous_recorded_at = None
    current_run_id: str | None = None
    current_run: dict[str, Any] | None = None
    current_sources: list[dict[str, Any]] = []
    current_validations: list[dict[str, Any]] = []
    current_transitions: list[dict[str, Any]] = []
    current_candidates: list[dict[str, Any]] = []
    current_phase = -1
    phase_by_type = {
        "RUN": 0,
        "SOURCE_CAPTURE": 1,
        "VALIDATION_ATTACHED": 2,
        "OWNER_TRANSITION": 3,
        "CANDIDATE_OBSERVED": 4,
    }

    def finalize_current() -> None:
        nonlocal current_run
        nonlocal current_sources
        nonlocal current_validations
        nonlocal current_transitions
        nonlocal current_candidates
        ready, validated, validations, transitions = _finalize_run_group(
            current_run,
            current_sources,
            current_validations,
            current_transitions,
            current_candidates,
        )
        for fingerprint, digest, run_id in ready:
            ready_spec_runs.setdefault((fingerprint, digest), set()).add(run_id)
        for fingerprint, digest, run_id in validated:
            validated_spec_runs.setdefault((fingerprint, digest), set()).add(run_id)
        validation_specs.update(validations)
        owner_specs.update(transitions)
        current_run = None
        current_sources = []
        current_validations = []
        current_transitions = []
        current_candidates = []

    for record in records:
        event_type = record["event_type"]
        payload = record["payload"]
        recorded_at = parse_timestamp(record["recorded_at"], "journal.recorded_at")
        if previous_recorded_at is not None and recorded_at < previous_recorded_at:
            raise ContractError("journal recorded_at values must be non-decreasing")
        previous_recorded_at = recorded_at
        _validate_journal_event(
            record["event_key"],
            event_type,
            payload,
            recorded_at=record["recorded_at"],
        )
        phase = phase_by_type[event_type]
        if event_type == "RUN":
            current_phase = phase
        elif phase < current_phase:
            raise ContractError("journal events are out of transaction order")
        else:
            current_phase = phase

        if event_type == "RUN":
            finalize_current()
            runs[payload["run_id"]] = payload
            run_recorded_at[payload["run_id"]] = recorded_at
            current_run_id = payload["run_id"]
            current_run = payload
            continue

        if event_type in {"SOURCE_CAPTURE", "CANDIDATE_OBSERVED"}:
            run_id = payload["run_id"]
            if run_id not in runs:
                raise ContractError(f"{event_type} references an unknown or later RUN")
            if run_id != current_run_id:
                raise ContractError(f"{event_type} must bind the current open RUN")
            if recorded_at < run_recorded_at[run_id]:
                raise ContractError(f"{event_type} predates its referenced RUN")

        if event_type == "SOURCE_CAPTURE":
            run_as_of = parse_timestamp(runs[payload["run_id"]]["as_of"], "RUN.as_of")
            if (
                payload["status"] == "COMPLETE"
                and parse_timestamp(payload["window"]["end"], "SOURCE_CAPTURE.window.end")
                > run_as_of
            ):
                raise ContractError("COMPLETE source capture window cannot postdate RUN.as_of")
            identity = (payload["source_id"], payload["capture_id"])
            prior_observation = source_observations.get(identity)
            content = {key: value for key, value in payload.items() if key != "run_id"}
            if prior_observation is not None and prior_observation != content:
                raise ContractError(
                    "source capture identity was reused with conflicting content"
                )
            if prior_observation is None:
                source_observations[identity] = content
                source_observation_runs[identity] = payload["run_id"]

            context = payload["continuity_context"]
            anchor = context["anchor"]
            if context["basis"] == "ACCEPTED_CAPTURE":
                anchor_identity = (payload["source_id"], anchor["capture_id"])
                accepted_anchor = source_observations.get(anchor_identity)
                if accepted_anchor is None or accepted_anchor["status"] != "COMPLETE":
                    raise ContractError(
                        "source continuity anchor does not reference a prior accepted capture"
                    )
                if source_observation_runs[anchor_identity] == payload["run_id"]:
                    raise ContractError(
                        "source continuity anchor must come from a prior run"
                    )
                for key in ("capture_digest", "cursor_out", "status"):
                    if anchor[key] != accepted_anchor[key]:
                        raise ContractError(
                            "source continuity anchor conflicts with its prior capture"
                        )

            if payload["status"] == "COMPLETE" and prior_observation is None:
                current = latest_accepted.get(payload["source_id"])
                if current is None:
                    if context["basis"] != "GENESIS":
                        raise ContractError(
                            "first accepted source capture must use GENESIS continuity"
                        )
                else:
                    if context["basis"] != "ACCEPTED_CAPTURE":
                        raise ContractError(
                            "new accepted source capture must continue the accepted head"
                        )
                    for key in ("capture_id", "capture_digest", "cursor_out", "status"):
                        if anchor[key] != current[key]:
                            raise ContractError(
                                "new accepted source capture does not continue the accepted head"
                            )
                latest_accepted[payload["source_id"]] = content
            current_sources.append(payload)
            continue

        if event_type == "VALIDATION_ATTACHED":
            if current_run_id is None:
                raise ContractError("validation artifact lacks a current RUN")
            run = runs[current_run_id]
            if run["completeness"] != "COMPLETE" or run["run_mode"] == "DISABLED":
                raise ContractError(
                    "validation artifact requires a COMPLETE enabled RUN"
                )
            if (
                parse_timestamp(payload["created_at"], "validation.created_at")
                > parse_timestamp(run["as_of"], "RUN.as_of")
            ):
                raise ContractError("validation artifact cannot postdate RUN.as_of")
            spec = (
                payload["candidate_fingerprint"],
                payload["research_spec_digest"],
            )
            if current_run_id is None or not any(
                run_id != current_run_id for run_id in ready_spec_runs.get(spec, set())
            ):
                raise ContractError(
                    "validation artifact lacks a prior-run exact-spec RESEARCH_READY observation"
                )
            current_validations.append(payload)
            continue

        if event_type == "OWNER_TRANSITION":
            if current_run_id is None:
                raise ContractError("owner transition lacks a current RUN")
            run = runs[current_run_id]
            if run["completeness"] != "COMPLETE" or run["run_mode"] == "DISABLED":
                raise ContractError(
                    "owner transition requires a COMPLETE enabled RUN"
                )
            if (
                parse_timestamp(payload["recorded_at"], "transition.recorded_at")
                > parse_timestamp(run["as_of"], "RUN.as_of")
            ):
                raise ContractError("owner transition cannot postdate RUN.as_of")
            spec = (
                payload["candidate_fingerprint"],
                payload["research_spec_digest"],
            )
            if (
                current_run_id is None
                or not any(
                    run_id != current_run_id
                    for run_id in validated_spec_runs.get(spec, set())
                )
                or spec not in validation_specs
            ):
                raise ContractError(
                    "owner transition lacks prior-run exact-spec validated research evidence"
                )
            current_transition_specs = {
                (
                    transition["candidate_fingerprint"],
                    transition["research_spec_digest"],
                )
                for transition in current_transitions
            }
            if spec in owner_specs or spec in current_transition_specs:
                raise ContractError("owner transition duplicates an accepted exact-spec decision")
            current_transitions.append(payload)
            continue

        if event_type == "CANDIDATE_OBSERVED":
            lifecycle = payload["lifecycle"]
            specs = [
                (payload["candidate_fingerprint"], digest)
                for digest in payload["research_spec_digests"]
            ]
            if lifecycle != "DISCOVERED" and len(specs) != 1:
                raise ContractError(
                    "authority-bearing candidate lifecycle requires one exact research spec"
                )
            if lifecycle != "DISCOVERED":
                run = runs[payload["run_id"]]
                if run["completeness"] != "COMPLETE" or run["run_mode"] == "DISABLED":
                    raise ContractError(
                        "authority-bearing candidate requires a COMPLETE enabled RUN"
                    )
            if lifecycle == "VALIDATED_RESEARCH":
                current_validation_specs = {
                    (
                        artifact["candidate_fingerprint"],
                        artifact["research_spec_digest"],
                    )
                    for artifact in current_validations
                }
                if any(
                    spec not in ready_spec_runs
                    or spec not in validation_specs | current_validation_specs
                    for spec in specs
                ):
                    raise ContractError(
                        "VALIDATED_RESEARCH lacks prior exact-spec preregistration and validation"
                    )
            elif lifecycle == "OWNER_REVIEW":
                current_transition_specs = {
                    (
                        transition["candidate_fingerprint"],
                        transition["research_spec_digest"],
                    )
                    for transition in current_transitions
                }
                if any(
                    spec not in validated_spec_runs
                    or spec not in owner_specs | current_transition_specs
                    for spec in specs
                ):
                    raise ContractError(
                        "OWNER_REVIEW lacks prior exact-spec validation and human transition"
                    )
            current_candidates.append(payload)
            continue

    finalize_current()


def _reject_journal_constant(value: str, *, path: Path, line_no: int) -> None:
    raise ContractError(
        f"journal {path}:{line_no}: non-finite numeric constant {value}"
    )


def is_symlink_or_reparse(path: Path) -> bool:
    """Return true for symlinks and Windows reparse-point aliases.

    ``Path.is_symlink`` covers ordinary links, but a Windows junction or
    another reparse-point type can still redirect an apparently local state
    path.  State files and lock files must never cross that boundary.
    """

    try:
        details = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ContractError(f"state path cannot be inspected: {path}") from exc
    attributes = getattr(details, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(details.st_mode) or bool(attributes & reparse_flag)


def load_journal(path: Path) -> list[dict[str, Any]]:
    """Read and verify every byte of an existing hash chain.

    A malformed final line is not ignored.  Silent recovery would turn a
    potentially truncated audit trail into apparently clean state.
    """

    if is_symlink_or_reparse(path):
        raise ContractError(f"journal must not be a symlink or reparse point: {path}")
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"journal {path}: unreadable ({exc})") from exc
    records: list[dict[str, Any]] = []
    expected_prev = GENESIS
    event_keys: dict[str, str] = {}
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            raise ContractError(f"journal {path}:{line_no}: blank lines are forbidden")
        try:
            record = json.loads(
                line,
                parse_constant=partial(
                    _reject_journal_constant,
                    path=path,
                    line_no=line_no,
                ),
            )
        except ValueError as exc:
            detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            raise ContractError(
                f"journal {path}:{line_no}: invalid JSON ({detail})"
            ) from exc
        if not isinstance(record, dict) or set(record) != RECORD_KEYS:
            raise ContractError(f"journal {path}:{line_no}: invalid record contract")
        if type(record["sequence"]) is not int or record["sequence"] != line_no:
            raise ContractError(f"journal {path}:{line_no}: non-contiguous sequence")
        if not isinstance(record["event_key"], str) or not record["event_key"]:
            raise ContractError(f"journal {path}:{line_no}: invalid event_key")
        if record["event_type"] not in EVENT_TYPES:
            raise ContractError(f"journal {path}:{line_no}: unknown event_type")
        parse_timestamp(record["recorded_at"], f"journal[{line_no}].recorded_at")
        if not isinstance(record["payload"], dict):
            raise ContractError(f"journal {path}:{line_no}: payload must be an object")
        if record["prev_hash"] != expected_prev:
            raise ContractError(f"journal {path}:{line_no}: broken prev_hash chain")
        body = {key: record[key] for key in RECORD_KEYS - {"record_hash"}}
        expected_hash = _record_hash(body)
        if record["record_hash"] != expected_hash:
            raise ContractError(f"journal {path}:{line_no}: record_hash mismatch")
        prior_hash = event_keys.get(record["event_key"])
        if prior_hash is not None and prior_hash != record["record_hash"]:
            raise ContractError(f"journal {path}:{line_no}: duplicate event_key")
        event_keys[record["event_key"]] = record["record_hash"]
        expected_prev = record["record_hash"]
        records.append(record)
    _validate_event_sequence(records)
    return records


def validate_journal_records(records_raw: Any) -> list[dict[str, Any]]:
    """Validate an in-memory journal exactly as the file reader would.

    This closes the library-call boundary: callers cannot bypass the hash,
    payload, or cross-event checks by passing hand-built records directly to
    ``run_discovery``.
    """

    if not isinstance(records_raw, list):
        raise ContractError("journal records must be an array")
    expected_prev = GENESIS
    event_keys: dict[str, str] = {}
    for index, record in enumerate(records_raw, 1):
        if not isinstance(record, dict) or set(record) != RECORD_KEYS:
            raise ContractError(f"journal[{index}] has an invalid record contract")
        if type(record["sequence"]) is not int or record["sequence"] != index:
            raise ContractError(f"journal[{index}] has a non-contiguous sequence")
        if not isinstance(record["event_key"], str) or not record["event_key"]:
            raise ContractError(f"journal[{index}] has an invalid event_key")
        if record["event_type"] not in EVENT_TYPES:
            raise ContractError(f"journal[{index}] has an unknown event_type")
        parse_timestamp(record["recorded_at"], f"journal[{index}].recorded_at")
        if not isinstance(record["payload"], dict):
            raise ContractError(f"journal[{index}] payload must be an object")
        if record["prev_hash"] != expected_prev:
            raise ContractError(f"journal[{index}] has a broken prev_hash chain")
        body = {key: record[key] for key in RECORD_KEYS - {"record_hash"}}
        expected_hash = _record_hash(body)
        if record["record_hash"] != expected_hash:
            raise ContractError(f"journal[{index}] has a record_hash mismatch")
        prior_hash = event_keys.get(record["event_key"])
        if prior_hash is not None and prior_hash != record["record_hash"]:
            raise ContractError(f"journal[{index}] reuses an event_key")
        event_keys[record["event_key"]] = record["record_hash"]
        expected_prev = record["record_hash"]
    _validate_event_sequence(records_raw)
    return records_raw


def append_events(
    path: Path,
    events: Iterable[dict[str, Any]],
    *,
    recorded_at: str,
    lock: _JournalLease | None = None,
) -> int:
    """Append new idempotent events with one flushed OS append.

    Existing records are never rewritten.  Replaying the exact event is a
    no-op; reusing an event key for different content fails closed.
    """

    event_list = list(events)
    canonical_lock_path = path.with_suffix(path.suffix + ".lock")
    if is_symlink_or_reparse(path):
        raise ContractError(f"journal must not be a symlink or reparse point: {path}")
    if lock is None:
        with exclusive_lock(canonical_lock_path) as acquired:
            return append_events(
                path,
                event_list,
                recorded_at=recorded_at,
                lock=acquired,
            )
    _validate_lock_lease(lock, canonical_lock_path)
    parse_timestamp(recorded_at, "recorded_at")
    existing = load_journal(path)
    by_key = {
        record["event_key"]: (record["event_type"], record["payload"])
        for record in existing
    }
    existing_keys = set(by_key)
    sequence = len(existing)
    prev_hash = existing[-1]["record_hash"] if existing else GENESIS
    fresh: list[dict[str, Any]] = []
    batch_run_id: str | None = None
    for index, event in enumerate(event_list):
        if not isinstance(event, dict) or set(event) != {"event_key", "event_type", "payload"}:
            raise ContractError("journal event must contain event_key, event_type, and payload")
        event_key = event["event_key"]
        if not isinstance(event_key, str) or not event_key:
            raise ContractError("journal event_key must be a non-empty string")
        if event["event_type"] not in EVENT_TYPES:
            raise ContractError(f"unknown journal event_type: {event['event_type']}")
        if not isinstance(event["payload"], dict):
            raise ContractError("journal payload must be an object")
        _validate_journal_event(
            event_key,
            event["event_type"],
            event["payload"],
            recorded_at=recorded_at,
        )
        if index == 0:
            if event["event_type"] != "RUN":
                raise ContractError("journal transaction must begin with exactly one RUN event")
            batch_run_id = event["payload"]["run_id"]
        elif event["event_type"] == "RUN":
            raise ContractError("journal transaction may contain exactly one RUN event")
        if (
            event["event_type"] in {"SOURCE_CAPTURE", "CANDIDATE_OBSERVED"}
            and event["payload"]["run_id"] != batch_run_id
        ):
            raise ContractError("journal child event must bind its transaction RUN")
        prior = by_key.get(event_key)
        current = (event["event_type"], event["payload"])
        if prior is not None:
            if prior != current:
                raise ContractError(f"journal event_key conflict: {event_key}")
            continue
        sequence += 1
        body = {
            "sequence": sequence,
            "event_key": event_key,
            "event_type": event["event_type"],
            "recorded_at": recorded_at,
            "payload": event["payload"],
            "prev_hash": prev_hash,
        }
        record = {**body, "record_hash": _record_hash(body)}
        fresh.append(record)
        by_key[event_key] = current
        prev_hash = record["record_hash"]
    if fresh and event_list[0]["event_key"] in existing_keys:
        raise ContractError("journal transaction cannot backfill an already recorded RUN")
    validate_journal_records([*existing, *fresh])
    if not fresh:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(canonical_json(record) + "\n" for record in fresh).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    descriptor = os.open(path, flags, 0o600)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(f"short journal append: {written}/{len(encoded)} bytes")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    verified = load_journal(path)
    if len(verified) != len(existing) + len(fresh):
        raise ContractError("journal append verification found an unexpected record count")
    if verified[-1]["record_hash"] != fresh[-1]["record_hash"]:
        raise ContractError("journal append verification found an unexpected head hash")
    return len(fresh)


def _validate_lock_lease(lock: _JournalLease, expected_path: Path) -> None:
    if not isinstance(lock, _JournalLease):
        raise ContractError("journal append requires its canonical lock lease")
    if lock.path.resolve() != expected_path.resolve():
        raise ContractError("journal lock lease belongs to a different lock domain")
    try:
        payload = json.loads(lock.path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError("journal lock lease is no longer verifiable") from exc
    if payload.get("nonce") != lock.nonce:
        raise ContractError("journal lock lease ownership mismatch")


@contextmanager
def exclusive_lock(path: Path, *, timeout_seconds: float = 10.0):
    """Acquire a fail-closed cross-process lock by atomic create.

    Locks are never guessed stale or stolen. A crashed writer intentionally
    leaves a lock that requires an operator to inspect the journal and bundle
    before manually clearing it.
    """

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ContractError(f"transaction lock parent is unavailable: {path.parent}") from exc
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    deadline = time.monotonic() + timeout_seconds
    descriptor = -1
    payload: str | None = None
    while descriptor < 0:
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError as exc:
            if time.monotonic() >= deadline:
                raise ContractError(f"transaction lock unavailable: {path}") from exc
            time.sleep(0.05)
        except OSError as exc:
            raise ContractError(f"transaction lock cannot be created: {path}") from exc
    try:
        nonce = secrets.token_hex(16)
        payload = canonical_json(
            {"pid": os.getpid(), "lock_path": str(path), "nonce": nonce}
        ) + "\n"
        payload_bytes = payload.encode("utf-8")
        written = os.write(descriptor, payload_bytes)
        if written != len(payload_bytes):
            raise OSError(f"short lock write: {written}/{len(payload_bytes)} bytes")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        yield _JournalLease(path=path, nonce=nonce)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            current = path.read_text(encoding="utf-8")
        except OSError:
            current = None
        if payload is not None and current == payload:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def candidate_state_history(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    history: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["event_type"] != "CANDIDATE_OBSERVED":
            continue
        payload = record["payload"]
        fingerprint = payload.get("candidate_fingerprint")
        lifecycle = payload.get("lifecycle")
        if not fingerprint or not lifecycle:
            continue
        history.setdefault(str(fingerprint), []).append(
            {
                "lifecycle": str(lifecycle),
                "recorded_at": record["recorded_at"],
                "sequence": record["sequence"],
                "run_id": payload.get("run_id"),
                "research_spec_digests": payload.get("research_spec_digests", []),
            }
        )
    return history


def validation_record_times(records: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(record["payload"]["artifact_id"]): record["recorded_at"]
        for record in records
        if record["event_type"] == "VALIDATION_ATTACHED"
        and record["payload"].get("artifact_id")
    }


def latest_source_captures(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the latest COMPLETE capture per source as a cursor anchor.

    PARTIAL and UNKNOWN observations stay in the audit journal but cannot
    advance the accepted cursor chain.  This prevents a later replay from
    laundering a continuity failure into a complete capture.
    """

    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["event_type"] != "SOURCE_CAPTURE":
            continue
        payload = record["payload"]
        source_id = payload.get("source_id")
        if source_id and payload.get("status") == "COMPLETE":
            latest[str(source_id)] = payload
    return latest


def observed_source_captures(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index every journaled source observation by source and capture ID."""

    observed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if record["event_type"] != "SOURCE_CAPTURE":
            continue
        payload = record["payload"]
        source_id = payload.get("source_id")
        capture_id = payload.get("capture_id")
        if source_id and capture_id:
            observed[(str(source_id), str(capture_id))] = payload
    return observed


def accepted_owner_transitions(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    accepted: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["event_type"] != "OWNER_TRANSITION":
            continue
        payload = record["payload"]
        fingerprint = payload.get("candidate_fingerprint")
        if fingerprint:
            accepted[str(fingerprint)] = payload
    return accepted


def accepted_validations(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Latest journaled validation artifact by artifact id."""

    accepted: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["event_type"] != "VALIDATION_ATTACHED":
            continue
        payload = record["payload"]
        artifact_id = payload.get("artifact_id")
        if artifact_id:
            accepted[str(artifact_id)] = payload
    return accepted
