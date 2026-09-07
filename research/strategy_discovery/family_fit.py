"""Strategy-family overlap from supplied profiles and an algorithm catalog.

No holdings, NAV, broker, price provider or account state is an input. Family
overlap determines the next research question, never evidence of profitability.
The result is a separate digest-bound companion to the validated V1 report;
it cannot promote its lifecycle or authorize email/trading.
"""
from __future__ import annotations

import math
import os
import re
from datetime import timedelta
from pathlib import Path

from .contracts import (
    ContractError,
    canonical_json,
    parse_timestamp,
    sha256_json,
    validate_report,
)
from .journal import is_symlink_or_reparse

BEHAVIORS = {"MEAN_REVERSION", "MOMENTUM", "CALENDAR", "VOLATILITY_CARRY", "HEDGE", "RELATIVE_VALUE", "FLOW", "UNCLASSIFIED"}
MARKETS = {"EQUITY_INDEX", "SINGLE_EQUITY", "SECTOR_EQUITY", "RATES", "COMMODITIES", "FX", "VOLATILITY"}
HORIZONS = {"INTRADAY", "SHORT_TERM", "SWING", "LONG_TERM", "VARIABLE"}
ACTIVE_STATUS = "PREVIOUSLY_OBSERVED_ACTIVE"
REFERENCE_STATUSES = {"PAPER", "NOT_ACTIVATED", "RESEARCH_ONLY", "UNVERIFIED"}
FIT_STATUSES = {
    "NEEDS_CLASSIFICATION",
    "INCOMPLETE_BASELINE",
    "ACTIVE_FAMILY_OVERLAP",
    "NO_ACTIVE_FAMILY_MATCH",
}


def horizon_bucket(sessions):
    if sessions is None:
        return "VARIABLE"
    if isinstance(sessions, bool) or not isinstance(sessions, (int, float)) or not math.isfinite(sessions) or sessions < 0:
        raise ContractError("holding horizon must be a nonnegative session count")
    return "INTRADAY" if sessions == 0 else "SHORT_TERM" if sessions <= 10 else "SWING" if sessions <= 63 else "LONG_TERM"


def validate_profile(profile):
    if not isinstance(profile, dict) or set(profile) != {"behavior", "markets", "horizon", "evidence_refs"}:
        raise ContractError("family profile must contain behavior, markets, horizon and evidence_refs only")
    if not isinstance(profile["behavior"], str) or not isinstance(profile["horizon"], str) or profile["behavior"] not in BEHAVIORS or profile["horizon"] not in HORIZONS:
        raise ContractError("unknown strategy-family behavior or horizon")
    if not isinstance(profile["markets"], list) or not profile["markets"] or not all(isinstance(x, str) and x in MARKETS for x in profile["markets"]):
        raise ContractError("family profile must name supported liquid IBKR market categories")
    if not isinstance(profile["evidence_refs"], list) or not profile["evidence_refs"] or not all(isinstance(x, str) and x.strip() for x in profile["evidence_refs"]):
        raise ContractError("family classification requires source/spec evidence references")
    return profile


def validate_family_catalog(catalog):
    fields = {"schema_version", "as_of", "checked_at", "operating_evidence", "source_digests", "records", "records_digest", "runtime_verified_now", "positions_used"}
    if not isinstance(catalog, dict) or set(catalog) != fields or catalog["schema_version"] != "algorithm-family-catalog.v1":
        raise ContractError("invalid algorithm family catalog contract")
    if catalog["positions_used"] is not False or catalog["runtime_verified_now"] is not False:
        raise ContractError("family catalog cannot assert current positions or fresh runtime verification")
    as_of = parse_timestamp(catalog["as_of"], "catalog.as_of")
    checked = parse_timestamp(catalog["checked_at"], "catalog.checked_at")
    if checked > as_of:
        raise ContractError("operating-status observation is after the catalog cutoff")
    def digest(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
    if not isinstance(catalog["operating_evidence"], str) or not catalog["operating_evidence"].strip():
        raise ContractError("catalog requires operating-status evidence")
    if not isinstance(catalog["source_digests"], dict) or not catalog["source_digests"] or not all(isinstance(k, str) and k.strip() and digest(v) for k, v in catalog["source_digests"].items()):
        raise ContractError("catalog requires digested source provenance")
    if not isinstance(catalog["records"], list) or sha256_json(catalog["records"]) != catalog["records_digest"]:
        raise ContractError("algorithm family catalog digest mismatch")
    seen = set()
    for row in catalog["records"]:
        expected = {"name", "sleeve", "status", "direction", "instruments", "profile", "configuration_digest", "cash_gate_supported"}
        if not isinstance(row, dict) or set(row) != expected or not isinstance(row["name"], str) or not row["name"].strip() or row["name"] in seen:
            raise ContractError("invalid or duplicate algorithm family catalog row")
        seen.add(row["name"])
        if not isinstance(row["status"], str) or not isinstance(row["direction"], str) or row["status"] not in {ACTIVE_STATUS, *REFERENCE_STATUSES} or row["direction"] not in {"LONG", "SHORT", "BOTH"}:
            raise ContractError("unknown algorithm operational status/direction")
        if not isinstance(row["sleeve"], str) or not row["sleeve"].strip() or not isinstance(row["cash_gate_supported"], bool) or not digest(row["configuration_digest"]):
            raise ContractError("invalid algorithm configuration provenance")
        if not isinstance(row["instruments"], list) or not all(isinstance(x, str) and x.strip() for x in row["instruments"]):
            raise ContractError("algorithm instruments must be a string list")
        validate_profile(row["profile"])
    return catalog


def assess_family_fit(report, catalog, annotations, *, max_status_age_days=30):
    if isinstance(max_status_age_days, bool) or not isinstance(max_status_age_days, int) or not 1 <= max_status_age_days <= 30:
        raise ContractError("operating-status age limit must be 1 to 30 days")
    validate_report(report)
    validate_family_catalog(catalog)
    if not isinstance(annotations, dict) or set(annotations) != {"schema_version", "candidate_profiles"} or annotations["schema_version"] != "candidate-family-profiles.v1":
        raise ContractError("invalid candidate family-profile wrapper")
    profiles = annotations["candidate_profiles"]
    if not isinstance(profiles, dict):
        raise ContractError("candidate_profiles must be keyed by discovery fingerprint")
    fingerprints = {row["fingerprint"] for row in report["candidates"]}
    if set(profiles) - fingerprints:
        raise ContractError("family annotation references a candidate outside this discovery report")
    for profile in profiles.values():
        validate_profile(profile)
    as_of = parse_timestamp(report["as_of"], "report.as_of")
    checked = parse_timestamp(catalog["checked_at"], "catalog.checked_at")
    generated = parse_timestamp(catalog["as_of"], "catalog.as_of")
    if generated > as_of or checked > as_of:
        raise ContractError("family catalog is after the research decision cutoff")
    status_stale = as_of - checked > timedelta(days=max_status_age_days)
    active = [r for r in catalog["records"] if r["status"] == ACTIVE_STATUS]
    references = [r for r in catalog["records"] if r["status"] != ACTIVE_STATUS]
    incomplete = not active or any(r["profile"]["behavior"] == "UNCLASSIFIED" for r in active)
    results = []
    for candidate in report["candidates"]:
        profile = profiles.get(candidate["fingerprint"])
        if profile and profile["horizon"] != horizon_bucket(candidate["structure"]["exit"]["time_stop_sessions"]):
            raise ContractError("family horizon contradicts the candidate's bounded holding rule")
        def matches(rows, candidate_profile, candidate_direction):
            if not candidate_profile or candidate_profile["behavior"] == "UNCLASSIFIED":
                return []
            return [{"name": r["name"], "status": r["status"], "sleeve": r["sleeve"],
                     "direction_relationship": "SAME" if r["direction"] == candidate_direction else "DIFFERENT_OR_BOTH"}
                    for r in rows if r["profile"]["behavior"] == candidate_profile["behavior"]
                    and set(r["profile"]["markets"]) & set(candidate_profile["markets"])
                    and (r["profile"]["horizon"] == candidate_profile["horizon"] or "VARIABLE" in {r["profile"]["horizon"], candidate_profile["horizon"]})]
        direction = candidate["structure"]["direction"]
        peers = matches(active, profile, direction)
        reference_peers = matches(references, profile, direction)
        if not profile or profile["behavior"] == "UNCLASSIFIED":
            status = "NEEDS_CLASSIFICATION"
        elif status_stale or incomplete or report["completeness"] != "COMPLETE":
            status = "INCOMPLETE_BASELINE"
        else:
            status = "ACTIVE_FAMILY_OVERLAP" if peers else "NO_ACTIVE_FAMILY_MATCH"
        results.append({"fingerprint": candidate["fingerprint"], "name": candidate["name"], "fit_status": status,
                        "active_matches": peers, "reference_matches": reference_peers,
                        "classification": profile, "email_eligible": False,
                        "required_validation": [
                            "Replay the complete strategy with decision-available data and realistic costs.",
                            "Compare strategy-level return streams, bad-period co-losses and signal overlap against active algorithmic peers.",
                            "Measure marginal capital occupancy and portfolio improvement under the same historical risk limits.",
                            "Verify liquid instruments, IBKR availability, and an implementable entry/exit schedule.",
                        ]})
    return {"schema_version": "strategy-family-fit.v1", "discovery_run_id": report["run_id"],
            "discovery_report_digest": sha256_json(report), "catalog_digest": sha256_json(catalog),
            "profiles_digest": sha256_json(annotations), "as_of": report["as_of"],
            "operating_status_checked_at": catalog["checked_at"], "operating_status_stale": status_stale,
            "operating_status_max_age_days": max_status_age_days,
            "runtime_verified_now": False, "positions_used": False, "research_only": True,
            "active_algorithm_count": len(active), "reference_algorithm_count": len(references),
            "candidates": results, "limitations": [
                "Classification uses supplied source/spec annotations; prose is not automatically interpreted.",
                "Family overlap is not a measured correlation, proof of edge, or approval to implement.",
                "No active-family match is only a research lead; it does not establish diversification.",
                "Paper and unactivated sleeves are references, including when they share a family.",
                "Active sleeves remain in the baseline while cash-gated; current positions are irrelevant.",
            ]}


def validate_family_fit_assessment(assessment, report):
    """Validate the digest-bound family companion before an email decision."""
    validate_report(report)
    fields = {
        "schema_version", "discovery_run_id", "discovery_report_digest",
        "catalog_digest", "profiles_digest", "as_of",
        "operating_status_checked_at", "operating_status_stale",
        "operating_status_max_age_days", "runtime_verified_now",
        "positions_used", "research_only", "active_algorithm_count",
        "reference_algorithm_count", "candidates", "limitations",
    }
    if (
        not isinstance(assessment, dict)
        or set(assessment) != fields
        or assessment.get("schema_version") != "strategy-family-fit.v1"
    ):
        raise ContractError("invalid strategy family-fit assessment contract")
    if assessment["discovery_run_id"] != report["run_id"]:
        raise ContractError("family-fit assessment references a different discovery run")
    if assessment["discovery_report_digest"] != sha256_json(report):
        raise ContractError("family-fit assessment does not bind the supplied discovery report")
    digest = lambda value: isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
    if not digest(assessment["catalog_digest"]) or not digest(assessment["profiles_digest"]):
        raise ContractError("family-fit assessment provenance digest is invalid")
    if assessment["as_of"] != report["as_of"]:
        raise ContractError("family-fit assessment cutoff differs from the discovery report")
    as_of = parse_timestamp(assessment["as_of"], "family_fit.as_of")
    checked = parse_timestamp(
        assessment["operating_status_checked_at"],
        "family_fit.operating_status_checked_at",
    )
    if checked > as_of:
        raise ContractError("family-fit operating status is after the decision cutoff")
    if not isinstance(assessment["operating_status_stale"], bool):
        raise ContractError("family-fit stale flag must be boolean")
    age_limit = assessment["operating_status_max_age_days"]
    if type(age_limit) is not int or not 1 <= age_limit <= 30:
        raise ContractError("family-fit operating-status age limit is invalid")
    if (
        assessment["runtime_verified_now"] is not False
        or assessment["positions_used"] is not False
        or assessment["research_only"] is not True
    ):
        raise ContractError("family-fit assessment violates its research-only boundary")
    for key in ("active_algorithm_count", "reference_algorithm_count"):
        if type(assessment[key]) is not int or assessment[key] < 0:
            raise ContractError(f"family-fit {key} must be a nonnegative integer")
    if assessment["active_algorithm_count"] < 1:
        raise ContractError("family-fit assessment has no active algorithm baseline")
    limitations = assessment["limitations"]
    if not isinstance(limitations, list) or not limitations or not all(
        isinstance(value, str) and value.strip() for value in limitations
    ):
        raise ContractError("family-fit limitations are invalid")

    report_rows = {row["fingerprint"]: row for row in report["candidates"]}
    rows = assessment["candidates"]
    if not isinstance(rows, list):
        raise ContractError("family-fit candidates must be a list")
    seen = set()
    expected_row_fields = {
        "fingerprint", "name", "fit_status", "active_matches",
        "reference_matches", "classification", "email_eligible",
        "required_validation",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_row_fields:
            raise ContractError("invalid family-fit candidate row")
        fingerprint = row["fingerprint"]
        source = report_rows.get(fingerprint)
        if source is None or fingerprint in seen or row["name"] != source["name"]:
            raise ContractError("family-fit candidate identity/coverage is invalid")
        seen.add(fingerprint)
        if row["fit_status"] not in FIT_STATUSES or row["email_eligible"] is not False:
            raise ContractError("family-fit candidate status is invalid")
        profile = row["classification"]
        if profile is not None:
            validate_profile(profile)
        required = row["required_validation"]
        if not isinstance(required, list) or not required or not all(
            isinstance(value, str) and value.strip() for value in required
        ):
            raise ContractError("family-fit required validation is invalid")
        for key, allowed_statuses in (
            ("active_matches", {ACTIVE_STATUS}),
            ("reference_matches", REFERENCE_STATUSES),
        ):
            matches = row[key]
            if not isinstance(matches, list):
                raise ContractError(f"family-fit {key} must be a list")
            for match in matches:
                if (
                    not isinstance(match, dict)
                    or set(match) != {"name", "status", "sleeve", "direction_relationship"}
                    or not isinstance(match["name"], str)
                    or not match["name"].strip()
                    or not isinstance(match["sleeve"], str)
                    or not match["sleeve"].strip()
                    or match["status"] not in allowed_statuses
                    or match["direction_relationship"] not in {"SAME", "DIFFERENT_OR_BOTH"}
                ):
                    raise ContractError(f"family-fit {key} row is invalid")
    if seen != set(report_rows):
        raise ContractError("family-fit candidates do not exactly cover the discovery report")
    return assessment


def publish_family_fit(output_dir: Path, assessment: dict) -> Path:
    return _publish_companion(output_dir, assessment, "family-fit")


def publish_family_catalog(output_dir: Path, catalog: dict) -> Path:
    validate_family_catalog(catalog)
    return _publish_companion(output_dir, catalog, "algorithm-family-catalog")


def _publish_companion(output_dir: Path, assessment: dict, prefix: str) -> Path:
    path = output_dir / f"{prefix}-{sha256_json(assessment)}.json"
    if is_symlink_or_reparse(path):
        raise ContractError("family-fit companion must not be a symlink or reparse point")
    content = canonical_json(assessment) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ContractError("immutable family-fit companion differs")
        return path
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return path
