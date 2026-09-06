"""Strict JSON contracts for the offline strategy-discovery boundary."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
RUN_MODES = {"DISABLED", "FIXTURE", "SHADOW", "LIVE"}
COMPLETENESS = {"COMPLETE", "PARTIAL", "UNKNOWN"}
ITEM_KINDS = {"POST", "REPLY", "QUOTE", "REPOST"}
CLAIM_TYPES = {
    "STRATEGY_LOGIC",
    "ENTRY",
    "EXIT",
    "METRIC",
    "RISK",
    "IMPLEMENTATION",
}
DIRECTIONS = {"LONG", "SHORT", "BOTH"}
TIMINGS = {"PREOPEN", "OPEN", "INTRADAY", "CLOSE", "CLOSE_FINAL"}
LOCATOR_KINDS = {"ACCOUNT", "LIST", "SEARCH"}
PROVIDER_STATUSES = {"OK", "PARTIAL", "ERROR"}
LIFECYCLES = {"DISCOVERED", "RESEARCH_READY", "VALIDATED_RESEARCH", "OWNER_REVIEW"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
POST_ID_RE = re.compile(r"^[A-Za-z0-9:_-]+$")
HANDLE_RE = re.compile(r"^@[A-Za-z0-9_]{1,30}$")


class ContractError(ValueError):
    """Raised when an input cannot safely cross the research boundary."""


def canonical_json(value: Any) -> str:
    """Stable JSON used for digests, fingerprints, and journal hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _where(path: str, message: str) -> ContractError:
    return ContractError(f"{path}: {message}")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _where(path, "must be an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise _where(path, "must be an array")
    return value


def _strict(
    value: dict[str, Any],
    path: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing:
        raise _where(path, f"missing field(s): {', '.join(missing)}")
    if unknown:
        raise _where(path, f"unknown field(s): {', '.join(unknown)}")


def _string(value: Any, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise _where(path, "must be a string")
    if nonempty and not value.strip():
        raise _where(path, "must not be empty")
    return value


def _bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise _where(path, "must be a boolean")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise _where(path, "must be an integer")
    if minimum is not None and value < minimum:
        raise _where(path, f"must be >= {minimum}")
    return value


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float):
        raise _where(path, "must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise _where(path, "must be finite")
    if minimum is not None and result < minimum:
        raise _where(path, f"must be >= {minimum}")
    return result


def _enum(value: Any, allowed: set[str], path: str) -> str:
    result = _string(value, path)
    if result not in allowed:
        raise _where(path, f"must be one of {sorted(allowed)}")
    return result


def parse_timestamp(value: Any, path: str) -> datetime:
    text = _string(value, path)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _where(path, "must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _where(path, "must include a UTC offset")
    return parsed


def validate_config(raw: Any) -> dict[str, Any]:
    config = _object(raw, "config")
    _strict(
        config,
        "config",
        {
            "schema_version",
            "run_mode",
            "as_of",
            "source_max_age_hours",
            "catalog_max_age_days",
            "required_source_ids",
            "report_title",
            "policy",
        },
    )
    if config["schema_version"] != SCHEMA_VERSION:
        raise _where("config.schema_version", f"must equal {SCHEMA_VERSION}")
    _enum(config["run_mode"], RUN_MODES, "config.run_mode")
    parse_timestamp(config["as_of"], "config.as_of")
    _number(config["source_max_age_hours"], "config.source_max_age_hours", minimum=0)
    _integer(config["catalog_max_age_days"], "config.catalog_max_age_days", minimum=0)
    required_sources = _list(config["required_source_ids"], "config.required_source_ids")
    for i, source_id in enumerate(required_sources):
        _string(source_id, f"config.required_source_ids[{i}]")
    if len(required_sources) != len(set(required_sources)):
        raise _where("config.required_source_ids", "must not contain duplicates")
    if config["run_mode"] != "DISABLED" and not required_sources:
        raise _where(
            "config.required_source_ids",
            "must include at least one source unless run_mode is DISABLED",
        )
    _string(config["report_title"], "config.report_title")
    policy = _object(config["policy"], "config.policy")
    _strict(policy, "config.policy", {"auto_lifecycle_ceiling", "x_discovery_only"})
    if policy["auto_lifecycle_ceiling"] != "RESEARCH_READY":
        raise _where(
            "config.policy.auto_lifecycle_ceiling",
            "must equal RESEARCH_READY",
        )
    if policy["x_discovery_only"] is not True:
        raise _where("config.policy.x_discovery_only", "must be true")
    return config


def validate_manifest(raw: Any) -> dict[str, Any]:
    manifest = _object(raw, "manifest")
    _strict(
        manifest,
        "manifest",
        {"schema_version", "provider", "provider_version", "sources"},
    )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise _where("manifest.schema_version", f"must equal {SCHEMA_VERSION}")
    _string(manifest["provider"], "manifest.provider")
    _string(manifest["provider_version"], "manifest.provider_version")
    sources = _list(manifest["sources"], "manifest.sources")
    seen: set[str] = set()
    captures: set[str] = set()
    for i, raw_source in enumerate(sources):
        path = f"manifest.sources[{i}]"
        source = _object(raw_source, path)
        _strict(
            source,
            path,
            {
                "source_id",
                "platform",
                "discovery_only",
                "locator",
                "capture_id",
                "captured_at",
                "window",
                "cursor",
                "provider_status",
                "expected_item_count",
                "expected_min_items",
                "observed_item_count",
            },
        )
        source_id = _string(source["source_id"], f"{path}.source_id")
        if source_id in seen:
            raise _where(f"{path}.source_id", "must be unique")
        seen.add(source_id)
        capture_id = _string(source["capture_id"], f"{path}.capture_id")
        if capture_id in captures:
            raise _where(f"{path}.capture_id", "must be globally unique")
        captures.add(capture_id)
        if source["platform"] != "X":
            raise _where(f"{path}.platform", "must equal X")
        if source["discovery_only"] is not True:
            raise _where(f"{path}.discovery_only", "must be true")
        locator = _object(source["locator"], f"{path}.locator")
        _strict(locator, f"{path}.locator", {"kind", "value"})
        _enum(locator["kind"], LOCATOR_KINDS, f"{path}.locator.kind")
        _string(locator["value"], f"{path}.locator.value")
        parse_timestamp(source["captured_at"], f"{path}.captured_at")
        window = _object(source["window"], f"{path}.window")
        _strict(window, f"{path}.window", {"start", "end"})
        start = parse_timestamp(window["start"], f"{path}.window.start")
        end = parse_timestamp(window["end"], f"{path}.window.end")
        if end < start:
            raise _where(f"{path}.window", "end must be >= start")
        captured_at = parse_timestamp(source["captured_at"], f"{path}.captured_at")
        if captured_at < end:
            raise _where(f"{path}.captured_at", "must be on or after window.end")
        cursor = _object(source["cursor"], f"{path}.cursor")
        _strict(cursor, f"{path}.cursor", {"in", "out", "exhausted"})
        for key in ("in", "out"):
            if cursor[key] is not None:
                _string(cursor[key], f"{path}.cursor.{key}")
        _bool(cursor["exhausted"], f"{path}.cursor.exhausted")
        _enum(source["provider_status"], PROVIDER_STATUSES, f"{path}.provider_status")
        if source["expected_item_count"] is not None:
            _integer(
                source["expected_item_count"],
                f"{path}.expected_item_count",
                minimum=0,
            )
        _integer(source["expected_min_items"], f"{path}.expected_min_items", minimum=0)
        _integer(source["observed_item_count"], f"{path}.observed_item_count", minimum=0)
    return manifest


def _validate_metric(raw: Any, path: str, *, internal: bool) -> None:
    metric = _object(raw, path)
    base = {"name", "value", "unit", "sample_size", "definition"}
    if internal:
        required = base | {"methodology"}
    else:
        required = base | {"period_start", "period_end"}
    _strict(metric, path, required)
    _string(metric["name"], f"{path}.name")
    if type(metric["value"]) not in (int, float, str):
        raise _where(f"{path}.value", "must be a number or source text")
    if type(metric["value"]) in (int, float):
        _number(metric["value"], f"{path}.value")
    else:
        _string(metric["value"], f"{path}.value")
    _string(metric["unit"], f"{path}.unit")
    if metric["sample_size"] is not None:
        _integer(metric["sample_size"], f"{path}.sample_size", minimum=0)
    _string(metric["definition"], f"{path}.definition")
    if internal:
        _string(metric["methodology"], f"{path}.methodology")
    else:
        for key in ("period_start", "period_end"):
            if metric[key] is not None:
                _string(metric[key], f"{path}.{key}")


def _validate_proposal(raw: Any, path: str) -> None:
    proposal = _object(raw, path)
    _strict(
        proposal,
        path,
        {
            "proposal_id",
            "name",
            "thesis",
            "portfolio_fit_hypothesis",
            "falsifiers",
            "direction",
            "universe",
            "signal",
            "entry",
            "exit",
            "data_requirements",
            "costs",
            "borrow",
        },
    )
    for key in ("proposal_id", "name", "thesis", "portfolio_fit_hypothesis"):
        _string(proposal[key], f"{path}.{key}")
    falsifiers = _list(proposal["falsifiers"], f"{path}.falsifiers")
    if not falsifiers:
        raise _where(f"{path}.falsifiers", "must not be empty")
    for i, falsifier in enumerate(falsifiers):
        _string(falsifier, f"{path}.falsifiers[{i}]")
    _enum(proposal["direction"], DIRECTIONS, f"{path}.direction")

    universe = _object(proposal["universe"], f"{path}.universe")
    _strict(universe, f"{path}.universe", {"asset_class", "scope", "instruments"})
    if universe["asset_class"] != "LISTED_EQUITY":
        raise _where(f"{path}.universe.asset_class", "must equal LISTED_EQUITY")
    _string(universe["scope"], f"{path}.universe.scope")
    instruments = _list(universe["instruments"], f"{path}.universe.instruments")
    for i, ticker in enumerate(instruments):
        _string(ticker, f"{path}.universe.instruments[{i}]")

    signal = _object(proposal["signal"], f"{path}.signal")
    _strict(
        signal,
        f"{path}.signal",
        {"observation_timing", "decision_lead_minutes", "conditions"},
    )
    _enum(signal["observation_timing"], TIMINGS, f"{path}.signal.observation_timing")
    if signal["decision_lead_minutes"] is not None:
        _integer(
            signal["decision_lead_minutes"],
            f"{path}.signal.decision_lead_minutes",
            minimum=0,
        )
    conditions = _list(signal["conditions"], f"{path}.signal.conditions")
    for i, raw_condition in enumerate(conditions):
        cpath = f"{path}.signal.conditions[{i}]"
        condition = _object(raw_condition, cpath)
        _strict(condition, cpath, {"field", "operator", "value", "unit", "lookback_sessions"})
        _string(condition["field"], f"{cpath}.field")
        _string(condition["operator"], f"{cpath}.operator")
        if not isinstance(condition["value"], (str, int, float, bool, list)):
            raise _where(f"{cpath}.value", "must be scalar or array")
        if isinstance(condition["value"], list):
            for j, value in enumerate(condition["value"]):
                if not isinstance(value, (str, int, float, bool)):
                    raise _where(f"{cpath}.value[{j}]", "must be scalar")
        if condition["unit"] is not None:
            _string(condition["unit"], f"{cpath}.unit")
        if condition["lookback_sessions"] is not None:
            _integer(condition["lookback_sessions"], f"{cpath}.lookback_sessions", minimum=0)

    entry = _object(proposal["entry"], f"{path}.entry")
    _strict(entry, f"{path}.entry", {"session_offset", "timing", "order_type", "price_rule"})
    _integer(entry["session_offset"], f"{path}.entry.session_offset", minimum=0)
    _enum(entry["timing"], TIMINGS - {"CLOSE_FINAL"}, f"{path}.entry.timing")
    _string(entry["order_type"], f"{path}.entry.order_type")
    if entry["price_rule"] is not None:
        _string(entry["price_rule"], f"{path}.entry.price_rule")

    exit_spec = _object(proposal["exit"], f"{path}.exit")
    _strict(exit_spec, f"{path}.exit", {"time_stop_sessions", "stop_rule", "target_rule"})
    if exit_spec["time_stop_sessions"] is not None:
        _integer(exit_spec["time_stop_sessions"], f"{path}.exit.time_stop_sessions", minimum=1)
    for key in ("stop_rule", "target_rule"):
        if exit_spec[key] is not None:
            _string(exit_spec[key], f"{path}.exit.{key}")

    data_requirements = _list(proposal["data_requirements"], f"{path}.data_requirements")
    for i, raw_requirement in enumerate(data_requirements):
        rpath = f"{path}.data_requirements[{i}]"
        requirement = _object(raw_requirement, rpath)
        _strict(requirement, rpath, {"field", "frequency", "availability"})
        for key in ("field", "frequency", "availability"):
            _string(requirement[key], f"{rpath}.{key}")

    if proposal["costs"] is not None:
        costs = _object(proposal["costs"], f"{path}.costs")
        _strict(costs, f"{path}.costs", {"commission_bps", "slippage_bps", "market_impact_model"})
        _number(costs["commission_bps"], f"{path}.costs.commission_bps", minimum=0)
        _number(costs["slippage_bps"], f"{path}.costs.slippage_bps", minimum=0)
        _string(costs["market_impact_model"], f"{path}.costs.market_impact_model")

    if proposal["borrow"] is not None:
        borrow = _object(proposal["borrow"], f"{path}.borrow")
        _strict(borrow, f"{path}.borrow", {"required", "availability_check", "fee_assumption_bps_annual"})
        _bool(borrow["required"], f"{path}.borrow.required")
        _string(borrow["availability_check"], f"{path}.borrow.availability_check")
        if borrow["fee_assumption_bps_annual"] is not None:
            _number(
                borrow["fee_assumption_bps_annual"],
                f"{path}.borrow.fee_assumption_bps_annual",
                minimum=0,
            )


def validate_item(raw: Any, index: int) -> dict[str, Any]:
    path = f"items[{index}]"
    item = _object(raw, path)
    _strict(
        item,
        path,
        {
            "schema_version",
            "item_id",
            "capture_id",
            "source_id",
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
            "captured_at",
            "permalink",
            "text",
            "claims",
            "strategy_proposal",
        },
    )
    if item["schema_version"] != SCHEMA_VERSION:
        raise _where(f"{path}.schema_version", f"must equal {SCHEMA_VERSION}")
    for key in ("item_id", "capture_id", "source_id", "post_id", "canonical_post_id", "author_handle", "permalink"):
        _string(item[key], f"{path}.{key}")
    if not POST_ID_RE.fullmatch(item["post_id"]):
        raise _where(f"{path}.post_id", "contains unsupported characters")
    if item["platform"] != "X":
        raise _where(f"{path}.platform", "must equal X")
    kind = _enum(item["kind"], ITEM_KINDS, f"{path}.kind")
    for key in ("thread_id", "parent_post_id", "quoted_post_id", "reposted_post_id"):
        if item[key] is not None:
            _string(item[key], f"{path}.{key}")
    if kind == "POST":
        if item["canonical_post_id"] != item["post_id"]:
            raise _where(f"{path}.canonical_post_id", "POST must canonicalize to post_id")
        if any(item[key] is not None for key in ("parent_post_id", "quoted_post_id", "reposted_post_id")):
            raise _where(path, "POST cannot carry reply/quote/repost lineage")
    elif kind == "REPLY":
        if not item["thread_id"] or not item["parent_post_id"]:
            raise _where(path, "REPLY requires thread_id and parent_post_id")
        if item["canonical_post_id"] != item["post_id"]:
            raise _where(f"{path}.canonical_post_id", "REPLY must preserve its own post_id")
        if item["quoted_post_id"] is not None or item["reposted_post_id"] is not None:
            raise _where(path, "REPLY cannot also be a quote or repost")
    elif kind == "QUOTE":
        if not item["quoted_post_id"]:
            raise _where(path, "QUOTE requires quoted_post_id")
        if item["canonical_post_id"] != item["post_id"]:
            raise _where(f"{path}.canonical_post_id", "QUOTE must preserve its own post_id")
        if item["parent_post_id"] is not None or item["reposted_post_id"] is not None:
            raise _where(path, "QUOTE cannot also be a reply or repost")
    elif kind == "REPOST":
        if not item["reposted_post_id"]:
            raise _where(path, "REPOST requires reposted_post_id")
        if item["canonical_post_id"] != item["reposted_post_id"]:
            raise _where(f"{path}.canonical_post_id", "REPOST must canonicalize to reposted_post_id")
        if item["parent_post_id"] is not None or item["quoted_post_id"] is not None:
            raise _where(path, "REPOST cannot also be a reply or quote")
        if item["strategy_proposal"] is not None or item["claims"]:
            raise _where(path, "REPOST cannot introduce claims or a proposal")
        if item["text"].strip():
            raise _where(f"{path}.text", "REPOST must not duplicate original text")
    created_at = parse_timestamp(item["created_at"], f"{path}.created_at")
    captured_at = parse_timestamp(item["captured_at"], f"{path}.captured_at")
    if captured_at < created_at:
        raise _where(f"{path}.captured_at", "must be on or after created_at")
    if not HANDLE_RE.fullmatch(item["author_handle"]):
        raise _where(f"{path}.author_handle", "must be a canonical X handle beginning with @")
    if not re.fullmatch(r"https://(?:www\.)?(?:x\.com|twitter\.com)/[^\s]+", item["permalink"]):
        raise _where(f"{path}.permalink", "must be an https X/Twitter permalink")
    _string(item["text"], f"{path}.text", nonempty=False)
    claims = _list(item["claims"], f"{path}.claims")
    claim_ids: set[str] = set()
    for i, raw_claim in enumerate(claims):
        cpath = f"{path}.claims[{i}]"
        claim = _object(raw_claim, cpath)
        _strict(claim, cpath, {"claim_id", "claim_type", "text", "evidence_class", "metrics"})
        claim_id = _string(claim["claim_id"], f"{cpath}.claim_id")
        if claim_id in claim_ids:
            raise _where(f"{cpath}.claim_id", "must be unique within the item")
        claim_ids.add(claim_id)
        _enum(claim["claim_type"], CLAIM_TYPES, f"{cpath}.claim_type")
        _string(claim["text"], f"{cpath}.text")
        if claim["evidence_class"] != "SOURCE_CLAIMED":
            raise _where(
                f"{cpath}.evidence_class",
                "X-derived claims must equal SOURCE_CLAIMED",
            )
        for j, metric in enumerate(_list(claim["metrics"], f"{cpath}.metrics")):
            _validate_metric(metric, f"{cpath}.metrics[{j}]", internal=False)
    if item["strategy_proposal"] is not None:
        _validate_proposal(item["strategy_proposal"], f"{path}.strategy_proposal")
    return item


def validate_catalog(raw: Any, expected_type: str, path: str) -> dict[str, Any]:
    catalog = _object(raw, path)
    _strict(
        catalog,
        path,
        {
            "schema_version",
            "snapshot_id",
            "catalog_type",
            "generated_at",
            "as_of",
            "records_digest",
            "records",
        },
    )
    if catalog["schema_version"] != SCHEMA_VERSION:
        raise _where(f"{path}.schema_version", f"must equal {SCHEMA_VERSION}")
    _string(catalog["snapshot_id"], f"{path}.snapshot_id")
    if catalog["catalog_type"] != expected_type:
        raise _where(f"{path}.catalog_type", f"must equal {expected_type}")
    parse_timestamp(catalog["generated_at"], f"{path}.generated_at")
    parse_timestamp(catalog["as_of"], f"{path}.as_of")
    digest = _string(catalog["records_digest"], f"{path}.records_digest")
    if not SHA256_RE.fullmatch(digest):
        raise _where(f"{path}.records_digest", "must be a lowercase SHA-256")
    records = _list(catalog["records"], f"{path}.records")
    if sha256_json(records) != digest:
        raise _where(f"{path}.records_digest", "does not match canonical records")
    fingerprints: set[str] = set()
    for i, raw_record in enumerate(records):
        rpath = f"{path}.records[{i}]"
        record = _object(raw_record, rpath)
        if expected_type == "STRATEGY_BOOK":
            _strict(record, rpath, {"name", "structural_fingerprint", "active"})
            _bool(record["active"], f"{rpath}.active")
        else:
            _strict(
                record,
                rpath,
                {"name", "structural_fingerprint", "rejection_reason", "decided_at"},
            )
            _string(record["rejection_reason"], f"{rpath}.rejection_reason")
            parse_timestamp(record["decided_at"], f"{rpath}.decided_at")
        _string(record["name"], f"{rpath}.name")
        fingerprint = _string(record["structural_fingerprint"], f"{rpath}.structural_fingerprint")
        if not SHA256_RE.fullmatch(fingerprint):
            raise _where(f"{rpath}.structural_fingerprint", "must be a lowercase SHA-256")
        if fingerprint in fingerprints:
            raise _where(f"{rpath}.structural_fingerprint", "must be unique")
        fingerprints.add(fingerprint)
    return catalog


def validate_validation_artifacts(raw: Any) -> dict[str, Any]:
    wrapper = _object(raw, "validation_artifacts")
    _strict(wrapper, "validation_artifacts", {"schema_version", "artifacts"})
    if wrapper["schema_version"] != SCHEMA_VERSION:
        raise _where("validation_artifacts.schema_version", f"must equal {SCHEMA_VERSION}")
    ids: set[str] = set()
    for i, raw_artifact in enumerate(_list(wrapper["artifacts"], "validation_artifacts.artifacts")):
        path = f"validation_artifacts.artifacts[{i}]"
        artifact = _object(raw_artifact, path)
        _strict(
            artifact,
            path,
            {
                "artifact_id",
                "candidate_fingerprint",
                "artifact_type",
                "artifact_path",
                "sha256",
                "created_at",
                "reproduce_command",
                "code_revision",
                "data_snapshot_digests",
                "methodology",
                "validation_status",
                "metrics",
            },
        )
        artifact_id = _string(artifact["artifact_id"], f"{path}.artifact_id")
        if artifact_id in ids:
            raise _where(f"{path}.artifact_id", "must be unique")
        ids.add(artifact_id)
        for key in ("candidate_fingerprint", "sha256"):
            value = _string(artifact[key], f"{path}.{key}")
            if not SHA256_RE.fullmatch(value):
                raise _where(f"{path}.{key}", "must be a lowercase SHA-256")
        if artifact["artifact_type"] != "REPRODUCIBLE_RESEARCH":
            raise _where(f"{path}.artifact_type", "must equal REPRODUCIBLE_RESEARCH")
        artifact_path = _string(artifact["artifact_path"], f"{path}.artifact_path")
        if "://" in artifact_path:
            raise _where(f"{path}.artifact_path", "must be a local artifact reference")
        parse_timestamp(artifact["created_at"], f"{path}.created_at")
        command = _list(artifact["reproduce_command"], f"{path}.reproduce_command")
        if not command:
            raise _where(f"{path}.reproduce_command", "must not be empty")
        for j, token in enumerate(command):
            _string(token, f"{path}.reproduce_command[{j}]")
        _string(artifact["code_revision"], f"{path}.code_revision")
        digests = _list(artifact["data_snapshot_digests"], f"{path}.data_snapshot_digests")
        if not digests:
            raise _where(f"{path}.data_snapshot_digests", "must not be empty")
        for j, value in enumerate(digests):
            digest = _string(value, f"{path}.data_snapshot_digests[{j}]")
            if not SHA256_RE.fullmatch(digest):
                raise _where(f"{path}.data_snapshot_digests[{j}]", "must be a lowercase SHA-256")
        _string(artifact["methodology"], f"{path}.methodology")
        if artifact["validation_status"] != "PASSED":
            raise _where(f"{path}.validation_status", "must equal PASSED")
        metrics = _list(artifact["metrics"], f"{path}.metrics")
        if not metrics:
            raise _where(f"{path}.metrics", "must not be empty")
        for j, metric in enumerate(metrics):
            _validate_metric(metric, f"{path}.metrics[{j}]", internal=True)
    return wrapper


def validate_transition(raw: Any, index: int) -> dict[str, Any]:
    path = f"owner_transitions[{index}]"
    transition = _object(raw, path)
    _strict(
        transition,
        path,
        {
            "transition_id",
            "candidate_fingerprint",
            "actor_type",
            "actor",
            "from_state",
            "to_state",
            "reason",
            "recorded_at",
        },
    )
    for key in ("transition_id", "actor", "reason"):
        _string(transition[key], f"{path}.{key}")
    fingerprint = _string(transition["candidate_fingerprint"], f"{path}.candidate_fingerprint")
    if not SHA256_RE.fullmatch(fingerprint):
        raise _where(f"{path}.candidate_fingerprint", "must be a lowercase SHA-256")
    if transition["actor_type"] != "HUMAN":
        raise _where(f"{path}.actor_type", "must equal HUMAN")
    if transition["from_state"] != "VALIDATED_RESEARCH":
        raise _where(f"{path}.from_state", "must equal VALIDATED_RESEARCH")
    if transition["to_state"] != "OWNER_REVIEW":
        raise _where(f"{path}.to_state", "must equal OWNER_REVIEW")
    parse_timestamp(transition["recorded_at"], f"{path}.recorded_at")
    return transition


def validate_report(report_raw: Any) -> dict[str, Any]:
    """Validate the deterministic output contract before anything is written."""

    report = _object(report_raw, "report")
    _strict(
        report,
        "report",
        {
            "schema_version",
            "report_type",
            "run_id",
            "run_mode",
            "as_of",
            "title",
            "completeness",
            "authority",
            "source_coverage",
            "catalog_health",
            "item_normalization",
            "summary",
            "candidates",
            "limitations",
        },
    )
    if report["schema_version"] != SCHEMA_VERSION:
        raise _where("report.schema_version", f"must equal {SCHEMA_VERSION}")
    if report["report_type"] != "STRATEGY_DISCOVERY":
        raise _where("report.report_type", "must equal STRATEGY_DISCOVERY")
    run_id = _string(report["run_id"], "report.run_id")
    if not SHA256_RE.fullmatch(run_id):
        raise _where("report.run_id", "must be a lowercase SHA-256")
    _enum(report["run_mode"], RUN_MODES, "report.run_mode")
    parse_timestamp(report["as_of"], "report.as_of")
    _string(report["title"], "report.title")
    _enum(report["completeness"], COMPLETENESS, "report.completeness")
    authority = _object(report["authority"], "report.authority")
    _strict(
        authority,
        "report.authority",
        {
            "research_only",
            "x_is_discovery_only",
            "automatic_lifecycle_ceiling",
            "trading_actions_enabled",
            "strategy_mutation_enabled",
        },
    )
    if authority != {
        "research_only": True,
        "x_is_discovery_only": True,
        "automatic_lifecycle_ceiling": "RESEARCH_READY",
        "trading_actions_enabled": False,
        "strategy_mutation_enabled": False,
    }:
        raise _where("report.authority", "violates the research-only authority boundary")
    _list(report["source_coverage"], "report.source_coverage")
    _list(report["catalog_health"], "report.catalog_health")
    _object(report["item_normalization"], "report.item_normalization")
    _object(report["summary"], "report.summary")
    candidates = _list(report["candidates"], "report.candidates")
    candidate_keys = {
        "fingerprint",
        "name",
        "aliases",
        "thesis",
        "portfolio_fit_hypotheses",
        "falsifiers",
        "structure",
        "disposition",
        "lifecycle",
        "automatic_lifecycle_ceiling",
        "gates",
        "first_rejection",
        "source_claims",
        "source_claimed_metrics",
        "validation_artifacts",
        "internally_validated_metrics",
        "edge_status",
        "provenance",
        "duplicate_proposal_count",
        "next_research_step",
        "trading_actions_enabled",
    }
    for i, raw_candidate in enumerate(candidates):
        path = f"report.candidates[{i}]"
        candidate = _object(raw_candidate, path)
        _strict(candidate, path, candidate_keys)
        fingerprint = _string(candidate["fingerprint"], f"{path}.fingerprint")
        if not SHA256_RE.fullmatch(fingerprint):
            raise _where(f"{path}.fingerprint", "must be a lowercase SHA-256")
        _string(candidate["name"], f"{path}.name")
        _string(candidate["thesis"], f"{path}.thesis")
        fit_hypotheses = _list(
            candidate["portfolio_fit_hypotheses"],
            f"{path}.portfolio_fit_hypotheses",
        )
        if not fit_hypotheses:
            raise _where(f"{path}.portfolio_fit_hypotheses", "must not be empty")
        for j, hypothesis in enumerate(fit_hypotheses):
            _string(hypothesis, f"{path}.portfolio_fit_hypotheses[{j}]")
        falsifiers = _list(candidate["falsifiers"], f"{path}.falsifiers")
        if not falsifiers:
            raise _where(f"{path}.falsifiers", "must not be empty")
        for j, falsifier in enumerate(falsifiers):
            _string(falsifier, f"{path}.falsifiers[{j}]")
        _object(candidate["structure"], f"{path}.structure")
        _enum(candidate["lifecycle"], LIFECYCLES, f"{path}.lifecycle")
        if candidate["automatic_lifecycle_ceiling"] != "RESEARCH_READY":
            raise _where(f"{path}.automatic_lifecycle_ceiling", "must equal RESEARCH_READY")
        if candidate["trading_actions_enabled"] is not False:
            raise _where(f"{path}.trading_actions_enabled", "must be false")
        if candidate["edge_status"] == "SOURCE_CLAIMS_ONLY" and candidate["internally_validated_metrics"]:
            raise _where(f"{path}.edge_status", "cannot label internal metrics as source-only")
        if candidate["lifecycle"] in {"VALIDATED_RESEARCH", "OWNER_REVIEW"} and not candidate["validation_artifacts"]:
            raise _where(f"{path}.lifecycle", "requires a reproducible validation artifact")
        for metric in candidate["source_claimed_metrics"]:
            if metric.get("evidence_class") != "SOURCE_CLAIMED":
                raise _where(f"{path}.source_claimed_metrics", "must remain SOURCE_CLAIMED")
    limitations = _list(report["limitations"], "report.limitations")
    for i, limitation in enumerate(limitations):
        _string(limitation, f"report.limitations[{i}]")
    return report


def load_json(path: Path) -> Any:
    _validate_local_json_path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: unreadable JSON ({exc})") from exc


def load_jsonl(path: Path) -> list[Any]:
    _validate_local_json_path(path, jsonl=True)
    records: list[Any] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"{path}: unreadable JSONL ({exc})") from exc
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ContractError(f"{path}:{i}: invalid JSON ({exc.msg})") from exc
    return records


def _validate_local_json_path(path: Path, *, jsonl: bool = False) -> None:
    text = str(path)
    # pathlib normalizes ``https://`` to ``https:\\`` on Windows, so detect
    # schemes after that normalization while excluding one-letter drive names.
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]+:[\\/]{1,2}", text):
        raise ContractError(f"{path}: URLs are forbidden; use a local file snapshot")
    allowed = {".jsonl"} if jsonl else {".json"}
    if path.suffix.lower() not in allowed:
        raise ContractError(f"{path}: only {sorted(allowed)} input files are accepted")
    if not path.is_file():
        raise ContractError(f"{path}: local input file does not exist")
