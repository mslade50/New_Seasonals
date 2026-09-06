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
MEMBERSHIP_MODES = {"POINT_IN_TIME", "FIXED_INSTRUMENTS", "CURRENT_STATIC"}
CONDITION_OPERATORS = {
    "!=",
    "<",
    "<=",
    "==",
    ">",
    ">=",
    "between",
    "crosses_above",
    "crosses_below",
    "in",
    "not_in",
}
ORDER_TYPES = {
    "LIMIT",
    "LOC",
    "LOO",
    "MARKET",
    "MOC",
    "MOO",
    "STOP",
    "STOP_LIMIT",
}
LIFECYCLES = {"DISCOVERED", "RESEARCH_READY", "VALIDATED_RESEARCH", "OWNER_REVIEW"}
MAX_ABS_NUMERIC = 10**18
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
POST_ID_RE = re.compile(r"^[A-Za-z0-9:_-]+$")
HANDLE_RE = re.compile(r"^@[A-Za-z0-9_]{1,30}$")
FIELD_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
PERMALINK_RE = re.compile(
    r"^https://(?:www\.)?(?:x\.com|twitter\.com)/"
    r"(?P<handle>[A-Za-z0-9_]{1,30})/status/"
    r"(?P<post_id>[A-Za-z0-9:_-]+)(?:[/?#].*)?$"
)


class ContractError(ValueError):
    """Raised when an input cannot safely cross the research boundary."""


def canonical_json(value: Any) -> str:
    """Stable JSON used for digests, fingerprints, and journal hashes."""

    try:
        result = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        result.encode("utf-8", errors="strict")
    except (OverflowError, TypeError, UnicodeError, ValueError) as exc:
        raise ContractError("value cannot be encoded as canonical UTF-8 JSON") from exc
    return result


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _reject_nonfinite(value: Any, path: str) -> None:
    if isinstance(value, str):
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise _where(path, "must contain valid Unicode scalar text") from exc
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise _where(path, "NaN and Infinity are forbidden")
        if abs(value) > MAX_ABS_NUMERIC:
            raise _where(path, f"numeric magnitude must be <= {MAX_ABS_NUMERIC}")
    elif type(value) is int and abs(value) > MAX_ABS_NUMERIC:
        raise _where(path, f"numeric magnitude must be <= {MAX_ABS_NUMERIC}")
    if isinstance(value, list):
        for i, child in enumerate(value):
            _reject_nonfinite(child, f"{path}[{i}]")
    elif isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise _where(path, "object keys must be strings")
            try:
                key.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise _where(path, "object key must contain valid Unicode scalar text") from exc
            _reject_nonfinite(child, f"{path}.{key}")


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
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise _where(path, "must contain valid Unicode scalar text") from exc
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


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> int | float:
    if type(value) not in (int, float):
        raise _where(path, "must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise _where(path, "must be finite")
    if abs(value) > MAX_ABS_NUMERIC:
        raise _where(path, f"numeric magnitude must be <= {MAX_ABS_NUMERIC}")
    if minimum is not None and value < minimum:
        raise _where(path, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise _where(path, f"must be <= {maximum}")
    return value


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
    _reject_nonfinite(raw, "config")
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
            "source_locator_allowlist",
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
    locator_allowlist = _object(
        config["source_locator_allowlist"],
        "config.source_locator_allowlist",
    )
    if set(locator_allowlist) != set(required_sources):
        raise _where(
            "config.source_locator_allowlist",
            "keys must exactly match required_source_ids",
        )
    for source_id, raw_locator in locator_allowlist.items():
        lpath = f"config.source_locator_allowlist.{source_id}"
        locator = _object(raw_locator, lpath)
        _strict(locator, lpath, {"kind", "value"})
        _enum(locator["kind"], LOCATOR_KINDS, f"{lpath}.kind")
        _string(locator["value"], f"{lpath}.value")
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
    _reject_nonfinite(raw, "manifest")
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


def _validate_condition_value(operator: str, value: Any, path: str) -> None:
    """Enforce an unambiguous operator/value grammar."""

    scalar_types = (str, int, float, bool)

    def validate_scalar(raw: Any, scalar_path: str) -> None:
        if not isinstance(raw, scalar_types):
            raise _where(scalar_path, "must be a scalar")
        if type(raw) in (int, float):
            _number(
                raw,
                scalar_path,
                minimum=-MAX_ABS_NUMERIC,
                maximum=MAX_ABS_NUMERIC,
            )
        elif isinstance(raw, str):
            _string(raw, scalar_path)

    def value_domain(raw: Any) -> str:
        if isinstance(raw, bool):
            return "boolean"
        if type(raw) in (int, float):
            return "number"
        return "text"

    if operator in {"between", "in", "not_in"}:
        values = _list(value, path)
        if operator == "between" and len(values) != 2:
            raise _where(path, "between requires exactly two ordered bounds")
        if operator in {"in", "not_in"} and not values:
            raise _where(path, f"{operator} requires at least one member")
        for index, member in enumerate(values):
            validate_scalar(member, f"{path}[{index}]")
        domains = {value_domain(member) for member in values}
        if len(domains) != 1:
            raise _where(path, f"{operator} members must have one homogeneous type")
        if operator == "between":
            if domains != {"number"}:
                raise _where(path, "between requires two numeric bounds")
            if values[0] >= values[1]:
                raise _where(path, "between bounds must be strictly increasing")
        return
    if isinstance(value, list):
        raise _where(path, f"operator {operator} requires one scalar value")
    validate_scalar(value, path)
    if (
        operator in {"<", "<=", ">", ">=", "crosses_above", "crosses_below"}
        and type(value) not in (int, float)
    ):
        raise _where(path, f"operator {operator} requires a numeric value")


def _validate_proposal(raw: Any, path: str) -> None:
    proposal = _object(raw, path)
    _strict(
        proposal,
        path,
        {
            "proposal_id",
            "name",
            "thesis",
            "why_now",
            "variant_wedge",
            "portfolio_fit_hypothesis",
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
            "downstream_workflow",
        },
    )
    for key in (
        "proposal_id",
        "name",
        "thesis",
        "why_now",
        "variant_wedge",
        "portfolio_fit_hypothesis",
    ):
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

    universe_history = _object(proposal["universe_history"], f"{path}.universe_history")
    _strict(
        universe_history,
        f"{path}.universe_history",
        {
            "membership_mode",
            "includes_delisted",
            "models_delisting_returns",
            "evidence_reference",
        },
    )
    _enum(
        universe_history["membership_mode"],
        MEMBERSHIP_MODES,
        f"{path}.universe_history.membership_mode",
    )
    _bool(universe_history["includes_delisted"], f"{path}.universe_history.includes_delisted")
    _bool(
        universe_history["models_delisting_returns"],
        f"{path}.universe_history.models_delisting_returns",
    )
    _string(
        universe_history["evidence_reference"],
        f"{path}.universe_history.evidence_reference",
    )

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
        field = _string(condition["field"], f"{cpath}.field")
        if not FIELD_IDENTIFIER_RE.fullmatch(field):
            raise _where(
                f"{cpath}.field",
                "must be a bounded machine-readable field identifier",
            )
        operator = _enum(
            condition["operator"],
            CONDITION_OPERATORS,
            f"{cpath}.operator",
        )
        _validate_condition_value(operator, condition["value"], f"{cpath}.value")
        if condition["unit"] is not None:
            _string(condition["unit"], f"{cpath}.unit")
        if condition["lookback_sessions"] is not None:
            _integer(condition["lookback_sessions"], f"{cpath}.lookback_sessions", minimum=0)

    entry = _object(proposal["entry"], f"{path}.entry")
    _strict(entry, f"{path}.entry", {"session_offset", "timing", "order_type", "price_rule"})
    _integer(entry["session_offset"], f"{path}.entry.session_offset", minimum=0)
    _enum(entry["timing"], TIMINGS - {"CLOSE_FINAL"}, f"{path}.entry.timing")
    _enum(entry["order_type"], ORDER_TYPES, f"{path}.entry.order_type")
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

    if proposal["capacity"] is not None:
        capacity = _object(proposal["capacity"], f"{path}.capacity")
        _strict(
            capacity,
            f"{path}.capacity",
            {
                "median_daily_dollar_volume_usd",
                "max_participation_rate_pct",
                "estimated_strategy_capacity_usd",
                "methodology",
            },
        )
        _number(
            capacity["median_daily_dollar_volume_usd"],
            f"{path}.capacity.median_daily_dollar_volume_usd",
            minimum=0,
        )
        participation = _number(
            capacity["max_participation_rate_pct"],
            f"{path}.capacity.max_participation_rate_pct",
            minimum=0,
        )
        if participation > 100:
            raise _where(
                f"{path}.capacity.max_participation_rate_pct",
                "must be <= 100",
            )
        if capacity["estimated_strategy_capacity_usd"] is not None:
            _number(
                capacity["estimated_strategy_capacity_usd"],
                f"{path}.capacity.estimated_strategy_capacity_usd",
                minimum=0,
            )
        _string(capacity["methodology"], f"{path}.capacity.methodology")

    for key in ("investable_if", "explicit_unknowns", "downstream_workflow"):
        values = _list(proposal[key], f"{path}.{key}")
        for i, value in enumerate(values):
            _string(value, f"{path}.{key}[{i}]")


def validate_item(raw: Any, index: int) -> dict[str, Any]:
    path = f"items[{index}]"
    _reject_nonfinite(raw, path)
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
    item_text = _string(item["text"], f"{path}.text", nonempty=False)
    claims = _list(item["claims"], f"{path}.claims")
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
        if item["strategy_proposal"] is not None or claims:
            raise _where(path, "REPOST cannot introduce claims or a proposal")
        if item_text.strip():
            raise _where(f"{path}.text", "REPOST must not duplicate original text")
    created_at = parse_timestamp(item["created_at"], f"{path}.created_at")
    captured_at = parse_timestamp(item["captured_at"], f"{path}.captured_at")
    if captured_at < created_at:
        raise _where(f"{path}.captured_at", "must be on or after created_at")
    if not HANDLE_RE.fullmatch(item["author_handle"]):
        raise _where(f"{path}.author_handle", "must be a canonical X handle beginning with @")
    permalink_match = PERMALINK_RE.fullmatch(item["permalink"])
    if permalink_match is None:
        raise _where(
            f"{path}.permalink",
            "must be an https X/Twitter status permalink",
        )
    if permalink_match.group("handle").casefold() != item["author_handle"][1:].casefold():
        raise _where(
            f"{path}.permalink",
            "status handle must match author_handle",
        )
    if permalink_match.group("post_id") != item["post_id"]:
        raise _where(
            f"{path}.permalink",
            "status id must match post_id",
        )
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
    _reject_nonfinite(raw, path)
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
    catalog_as_of = parse_timestamp(catalog["as_of"], f"{path}.as_of")
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
            active = _bool(record["active"], f"{rpath}.active")
            if active is not True:
                raise _where(
                    f"{rpath}.active",
                    "must be true in the active strategy-book snapshot",
                )
        else:
            _strict(
                record,
                rpath,
                {"name", "structural_fingerprint", "rejection_reason", "decided_at"},
            )
            _string(record["rejection_reason"], f"{rpath}.rejection_reason")
            decided_at = parse_timestamp(record["decided_at"], f"{rpath}.decided_at")
            if decided_at > catalog_as_of:
                raise _where(
                    f"{rpath}.decided_at",
                    "must be on or before catalog.as_of",
                )
        _string(record["name"], f"{rpath}.name")
        fingerprint = _string(record["structural_fingerprint"], f"{rpath}.structural_fingerprint")
        if not SHA256_RE.fullmatch(fingerprint):
            raise _where(f"{rpath}.structural_fingerprint", "must be a lowercase SHA-256")
        if fingerprint in fingerprints:
            raise _where(f"{rpath}.structural_fingerprint", "must be unique")
        fingerprints.add(fingerprint)
    return catalog


def validate_validation_artifacts(raw: Any) -> dict[str, Any]:
    _reject_nonfinite(raw, "validation_artifacts")
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
                "research_spec_digest",
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
        for key in ("candidate_fingerprint", "research_spec_digest", "sha256"):
            value = _string(artifact[key], f"{path}.{key}")
            if not SHA256_RE.fullmatch(value):
                raise _where(f"{path}.{key}", "must be a lowercase SHA-256")
        if artifact["artifact_type"] != "REPRODUCIBLE_RESEARCH":
            raise _where(f"{path}.artifact_type", "must equal REPRODUCIBLE_RESEARCH")
        artifact_path = _string(artifact["artifact_path"], f"{path}.artifact_path")
        if "://" in artifact_path:
            raise _where(f"{path}.artifact_path", "must be a local artifact reference")
        artifact_ref = Path(artifact_path)
        if artifact_ref.is_absolute() or ".." in artifact_ref.parts:
            raise _where(
                f"{path}.artifact_path",
                "must be a traversal-free path relative to the approved artifact root",
            )
        if artifact_ref.suffix.lower() not in {".json", ".jsonl"}:
            raise _where(f"{path}.artifact_path", "must reference JSON or JSONL")
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
    _reject_nonfinite(raw, path)
    transition = _object(raw, path)
    _strict(
        transition,
        path,
        {
            "transition_id",
            "candidate_fingerprint",
            "research_spec_digest",
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
    spec_digest = _string(transition["research_spec_digest"], f"{path}.research_spec_digest")
    if not SHA256_RE.fullmatch(spec_digest):
        raise _where(f"{path}.research_spec_digest", "must be a lowercase SHA-256")
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

    _reject_nonfinite(report_raw, "report")
    report = _object(report_raw, "report")
    _strict(
        report,
        "report",
        {
            "schema_version",
            "processor_version",
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
    _string(report["processor_version"], "report.processor_version")
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
            "operationally_authoritative",
            "trading_actions_enabled",
            "strategy_mutation_enabled",
        },
    )
    if authority != {
        "research_only": True,
        "x_is_discovery_only": True,
        "automatic_lifecycle_ceiling": "RESEARCH_READY",
        "operationally_authoritative": False,
        "trading_actions_enabled": False,
        "strategy_mutation_enabled": False,
    }:
        raise _where("report.authority", "violates the research-only authority boundary")
    coverage_keys = {
        "source_id",
        "platform",
        "discovery_only",
        "provider",
        "provider_version",
        "provider_status",
        "locator",
        "capture_id",
        "captured_at",
        "window",
        "cursor_in",
        "cursor_out",
        "cursor_exhausted",
        "expected_item_count",
        "expected_min_items",
        "manifest_observed_item_count",
        "file_observed_item_count",
        "status",
        "findings",
        "capture_digest",
        "continuity_context",
    }
    for i, raw_coverage in enumerate(
        _list(report["source_coverage"], "report.source_coverage")
    ):
        path = f"report.source_coverage[{i}]"
        coverage = _object(raw_coverage, path)
        _strict(coverage, path, coverage_keys)
        _string(coverage["source_id"], f"{path}.source_id")
        if coverage["platform"] != "X" or coverage["discovery_only"] is not True:
            raise _where(path, "source coverage must remain X discovery-only")
        _string(coverage["provider"], f"{path}.provider")
        _string(coverage["provider_version"], f"{path}.provider_version")
        if coverage["provider_status"] not in PROVIDER_STATUSES | {"MISSING"}:
            raise _where(f"{path}.provider_status", "invalid provider status")
        locator = _object(coverage["locator"], f"{path}.locator")
        _strict(locator, f"{path}.locator", {"kind", "value"})
        _enum(locator["kind"], LOCATOR_KINDS, f"{path}.locator.kind")
        _string(locator["value"], f"{path}.locator.value")
        _enum(coverage["status"], COMPLETENESS, f"{path}.status")
        if coverage["capture_digest"] is not None:
            digest = _string(coverage["capture_digest"], f"{path}.capture_digest")
            if not SHA256_RE.fullmatch(digest):
                raise _where(f"{path}.capture_digest", "must be a lowercase SHA-256")
        continuity = _object(
            coverage["continuity_context"],
            f"{path}.continuity_context",
        )
        _strict(
            continuity,
            f"{path}.continuity_context",
            {"basis", "anchor"},
        )
        if continuity["basis"] not in {"GENESIS", "ACCEPTED_CAPTURE"}:
            raise _where(
                f"{path}.continuity_context.basis",
                "must be GENESIS or ACCEPTED_CAPTURE",
            )
        anchor = continuity["anchor"]
        if continuity["basis"] == "GENESIS":
            if anchor is not None:
                raise _where(
                    f"{path}.continuity_context.anchor",
                    "must be null for GENESIS",
                )
        else:
            anchor_path = f"{path}.continuity_context.anchor"
            anchor = _object(anchor, anchor_path)
            _strict(
                anchor,
                anchor_path,
                {"source_id", "capture_id", "capture_digest", "cursor_out", "status"},
            )
            _string(anchor["source_id"], f"{anchor_path}.source_id")
            _string(anchor["capture_id"], f"{anchor_path}.capture_id")
            anchor_digest = _string(
                anchor["capture_digest"],
                f"{anchor_path}.capture_digest",
            )
            if not SHA256_RE.fullmatch(anchor_digest):
                raise _where(
                    f"{anchor_path}.capture_digest",
                    "must be a lowercase SHA-256",
                )
            if anchor["cursor_out"] is not None:
                _string(anchor["cursor_out"], f"{anchor_path}.cursor_out")
            if anchor["status"] != "COMPLETE":
                raise _where(f"{anchor_path}.status", "must equal COMPLETE")
    catalog_health = _list(report["catalog_health"], "report.catalog_health")
    if len(catalog_health) != 2:
        raise _where(
            "report.catalog_health",
            "must contain exactly the strategy-book and dead-end snapshots",
        )
    catalog_types: set[str] = set()
    for index, raw_catalog in enumerate(catalog_health):
        path = f"report.catalog_health[{index}]"
        catalog = _object(raw_catalog, path)
        _strict(
            catalog,
            path,
            {
                "snapshot_id",
                "catalog_type",
                "generated_at",
                "as_of",
                "records_digest",
                "record_count",
                "status",
                "finding",
            },
        )
        _string(catalog["snapshot_id"], f"{path}.snapshot_id")
        catalog_type = _enum(
            catalog["catalog_type"],
            {"STRATEGY_BOOK", "DEAD_ENDS"},
            f"{path}.catalog_type",
        )
        if catalog_type in catalog_types:
            raise _where(f"{path}.catalog_type", "must be unique")
        catalog_types.add(catalog_type)
        parse_timestamp(catalog["generated_at"], f"{path}.generated_at")
        parse_timestamp(catalog["as_of"], f"{path}.as_of")
        digest = _string(catalog["records_digest"], f"{path}.records_digest")
        if not SHA256_RE.fullmatch(digest):
            raise _where(f"{path}.records_digest", "must be a lowercase SHA-256")
        _integer(catalog["record_count"], f"{path}.record_count", minimum=0)
        _enum(catalog["status"], COMPLETENESS, f"{path}.status")
        _string(catalog["finding"], f"{path}.finding")
    item_normalization = _object(report["item_normalization"], "report.item_normalization")
    _strict(
        item_normalization,
        "report.item_normalization",
        {
            "duplicate_post_ids",
            "conflicts",
            "reposts_are_lineage_only",
            "quotes_preserve_quote_post_identity",
            "thread_replies_preserve_post_identity",
            "source_registry_digest",
        },
    )
    registry_digest = _string(
        item_normalization["source_registry_digest"],
        "report.item_normalization.source_registry_digest",
    )
    if not SHA256_RE.fullmatch(registry_digest):
        raise _where(
            "report.item_normalization.source_registry_digest",
            "must be a lowercase SHA-256",
        )
    summary = _object(report["summary"], "report.summary")
    _strict(
        summary,
        "report.summary",
        {
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
        },
    )
    for key, value in summary.items():
        _integer(value, f"report.summary.{key}", minimum=0)
    candidates = _list(report["candidates"], "report.candidates")
    candidate_keys = {
        "fingerprint",
        "research_spec_digests",
        "name",
        "aliases",
        "thesis",
        "why_now",
        "variant_wedge",
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
        "observation_count",
        "research_assumptions",
        "investable_if",
        "explicit_unknowns",
        "downstream_workflow",
        "actionability",
        "next_research_step",
        "trading_actions_enabled",
        "operationally_authoritative",
    }
    for i, raw_candidate in enumerate(candidates):
        path = f"report.candidates[{i}]"
        candidate = _object(raw_candidate, path)
        _strict(candidate, path, candidate_keys)
        fingerprint = _string(candidate["fingerprint"], f"{path}.fingerprint")
        if not SHA256_RE.fullmatch(fingerprint):
            raise _where(f"{path}.fingerprint", "must be a lowercase SHA-256")
        research_spec_digests = _list(
            candidate["research_spec_digests"],
            f"{path}.research_spec_digests",
        )
        if not research_spec_digests:
            raise _where(f"{path}.research_spec_digests", "must not be empty")
        for j, digest in enumerate(research_spec_digests):
            value = _string(digest, f"{path}.research_spec_digests[{j}]")
            if not SHA256_RE.fullmatch(value):
                raise _where(
                    f"{path}.research_spec_digests[{j}]",
                    "must be a lowercase SHA-256",
                )
        _string(candidate["name"], f"{path}.name")
        _string(candidate["thesis"], f"{path}.thesis")
        _string(candidate["why_now"], f"{path}.why_now")
        _string(candidate["variant_wedge"], f"{path}.variant_wedge")
        aliases = _list(candidate["aliases"], f"{path}.aliases")
        for j, alias in enumerate(aliases):
            _string(alias, f"{path}.aliases[{j}]")
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
        assumptions = _object(
            candidate["research_assumptions"],
            f"{path}.research_assumptions",
        )
        _strict(assumptions, f"{path}.research_assumptions", {"costs", "borrow", "capacity"})
        if assumptions["costs"] is not None:
            costs = _object(assumptions["costs"], f"{path}.research_assumptions.costs")
            _strict(
                costs,
                f"{path}.research_assumptions.costs",
                {"commission_bps", "slippage_bps", "market_impact_model"},
            )
            _number(costs["commission_bps"], f"{path}.research_assumptions.costs.commission_bps", minimum=0)
            _number(costs["slippage_bps"], f"{path}.research_assumptions.costs.slippage_bps", minimum=0)
            _string(costs["market_impact_model"], f"{path}.research_assumptions.costs.market_impact_model")
        if assumptions["borrow"] is not None:
            borrow = _object(assumptions["borrow"], f"{path}.research_assumptions.borrow")
            _strict(
                borrow,
                f"{path}.research_assumptions.borrow",
                {"required", "availability_check", "fee_assumption_bps_annual"},
            )
            _bool(borrow["required"], f"{path}.research_assumptions.borrow.required")
            _string(borrow["availability_check"], f"{path}.research_assumptions.borrow.availability_check")
            if borrow["fee_assumption_bps_annual"] is not None:
                _number(
                    borrow["fee_assumption_bps_annual"],
                    f"{path}.research_assumptions.borrow.fee_assumption_bps_annual",
                    minimum=0,
                )
        if assumptions["capacity"] is not None:
            capacity = _object(assumptions["capacity"], f"{path}.research_assumptions.capacity")
            _strict(
                capacity,
                f"{path}.research_assumptions.capacity",
                {
                    "median_daily_dollar_volume_usd",
                    "max_participation_rate_pct",
                    "estimated_strategy_capacity_usd",
                    "methodology",
                },
            )
            _number(
                capacity["median_daily_dollar_volume_usd"],
                f"{path}.research_assumptions.capacity.median_daily_dollar_volume_usd",
                minimum=0,
            )
            _number(
                capacity["max_participation_rate_pct"],
                f"{path}.research_assumptions.capacity.max_participation_rate_pct",
                minimum=0,
            )
            if capacity["estimated_strategy_capacity_usd"] is not None:
                _number(
                    capacity["estimated_strategy_capacity_usd"],
                    f"{path}.research_assumptions.capacity.estimated_strategy_capacity_usd",
                    minimum=0,
                )
            _string(
                capacity["methodology"],
                f"{path}.research_assumptions.capacity.methodology",
            )
        for field in ("investable_if", "explicit_unknowns", "downstream_workflow"):
            values = _list(candidate[field], f"{path}.{field}")
            for j, value in enumerate(values):
                _string(value, f"{path}.{field}[{j}]")
        if candidate["actionability"] not in {"RESEARCH_ACTIONABLE", "BLOCKED"}:
            raise _where(f"{path}.actionability", "must be RESEARCH_ACTIONABLE or BLOCKED")
        _enum(candidate["lifecycle"], LIFECYCLES, f"{path}.lifecycle")
        if candidate["automatic_lifecycle_ceiling"] != "RESEARCH_READY":
            raise _where(f"{path}.automatic_lifecycle_ceiling", "must equal RESEARCH_READY")
        if candidate["trading_actions_enabled"] is not False:
            raise _where(f"{path}.trading_actions_enabled", "must be false")
        if candidate["operationally_authoritative"] is not False:
            raise _where(f"{path}.operationally_authoritative", "must be false")
        if candidate["edge_status"] not in {"SOURCE_CLAIMS_ONLY", "INTERNALLY_VALIDATED"}:
            raise _where(f"{path}.edge_status", "invalid edge status")
        gates = _list(candidate["gates"], f"{path}.gates")
        if not gates:
            raise _where(f"{path}.gates", "must not be empty")
        for j, raw_gate in enumerate(gates):
            gate_path = f"{path}.gates[{j}]"
            gate = _object(raw_gate, gate_path)
            _strict(gate, gate_path, {"gate", "status", "reason"})
            _string(gate["gate"], f"{gate_path}.gate")
            if gate["status"] not in {"PASS", "FAIL"}:
                raise _where(f"{gate_path}.status", "must be PASS or FAIL")
            _string(gate["reason"], f"{gate_path}.reason")
        rejection = candidate["first_rejection"]
        if rejection is not None:
            rejection_path = f"{path}.first_rejection"
            rejection = _object(rejection, rejection_path)
            _strict(rejection, rejection_path, {"gate", "reason"})
            _string(rejection["gate"], f"{rejection_path}.gate")
            _string(rejection["reason"], f"{rejection_path}.reason")
        source_claims = _list(candidate["source_claims"], f"{path}.source_claims")
        for j, raw_claim in enumerate(source_claims):
            claim_path = f"{path}.source_claims[{j}]"
            claim = _object(raw_claim, claim_path)
            _strict(
                claim,
                claim_path,
                {
                    "source_id",
                    "source_ids",
                    "post_id",
                    "claim_id",
                    "claim_type",
                    "text",
                    "evidence_class",
                    "metrics",
                },
            )
            for key in ("source_id", "post_id", "claim_id", "text"):
                _string(claim[key], f"{claim_path}.{key}")
            source_ids = _list(claim["source_ids"], f"{claim_path}.source_ids")
            if not source_ids:
                raise _where(f"{claim_path}.source_ids", "must not be empty")
            for k, source_id in enumerate(source_ids):
                _string(source_id, f"{claim_path}.source_ids[{k}]")
            _enum(claim["claim_type"], CLAIM_TYPES, f"{claim_path}.claim_type")
            if claim["evidence_class"] != "SOURCE_CLAIMED":
                raise _where(f"{claim_path}.evidence_class", "must equal SOURCE_CLAIMED")
            for k, metric in enumerate(_list(claim["metrics"], f"{claim_path}.metrics")):
                _validate_metric(metric, f"{claim_path}.metrics[{k}]", internal=False)
        artifacts = _list(
            candidate["validation_artifacts"],
            f"{path}.validation_artifacts",
        )
        validate_validation_artifacts(
            {"schema_version": SCHEMA_VERSION, "artifacts": artifacts}
        )
        artifact_ids = {artifact["artifact_id"] for artifact in artifacts}
        internal_metrics = _list(
            candidate["internally_validated_metrics"],
            f"{path}.internally_validated_metrics",
        )
        for j, raw_metric in enumerate(internal_metrics):
            metric_path = f"{path}.internally_validated_metrics[{j}]"
            metric = _object(raw_metric, metric_path)
            artifact_id = _string(metric.get("artifact_id"), f"{metric_path}.artifact_id")
            if artifact_id not in artifact_ids:
                raise _where(
                    f"{metric_path}.artifact_id",
                    "must reference an attached validation artifact",
                )
            _validate_metric(
                {key: value for key, value in metric.items() if key != "artifact_id"},
                metric_path,
                internal=True,
            )
        if candidate["edge_status"] == "SOURCE_CLAIMS_ONLY" and internal_metrics:
            raise _where(f"{path}.edge_status", "cannot label internal metrics as source-only")
        if candidate["edge_status"] == "INTERNALLY_VALIDATED" and not artifacts:
            raise _where(f"{path}.edge_status", "requires a validation artifact")
        if candidate["lifecycle"] in {"VALIDATED_RESEARCH", "OWNER_REVIEW"} and not artifacts:
            raise _where(f"{path}.lifecycle", "requires a reproducible validation artifact")
        source_metrics = _list(
            candidate["source_claimed_metrics"],
            f"{path}.source_claimed_metrics",
        )
        source_metric_keys = {
            "source_id",
            "source_ids",
            "post_id",
            "claim_id",
            "evidence_class",
            "name",
            "value",
            "unit",
            "sample_size",
            "definition",
            "period_start",
            "period_end",
        }
        for j, raw_metric in enumerate(source_metrics):
            metric_path = f"{path}.source_claimed_metrics[{j}]"
            metric = _object(raw_metric, metric_path)
            _strict(metric, metric_path, source_metric_keys)
            if metric["evidence_class"] != "SOURCE_CLAIMED":
                raise _where(metric_path, "must remain SOURCE_CLAIMED")
            for key in ("source_id", "post_id", "claim_id"):
                _string(metric[key], f"{metric_path}.{key}")
            source_ids = _list(metric["source_ids"], f"{metric_path}.source_ids")
            if not source_ids:
                raise _where(f"{metric_path}.source_ids", "must not be empty")
            for k, source_id in enumerate(source_ids):
                _string(source_id, f"{metric_path}.source_ids[{k}]")
            _validate_metric(
                {
                    key: value
                    for key, value in metric.items()
                    if key
                    not in {
                        "source_id",
                        "source_ids",
                        "post_id",
                        "claim_id",
                        "evidence_class",
                    }
                },
                metric_path,
                internal=False,
            )
        provenance_keys = {
            "item_id",
            "source_id",
            "capture_id",
            "captured_at",
            "permalink",
            "source_locator",
            "provider",
            "provider_version",
            "provider_status",
            "source_window",
            "capture_digest",
            "native_content_hash",
            "post_id",
            "canonical_post_id",
            "thread_id",
            "parent_post_id",
            "quoted_post_id",
            "reposted_post_id",
            "kind",
            "author_handle",
            "created_at",
        }
        provenance_rows = _list(candidate["provenance"], f"{path}.provenance")
        for j, raw_provenance in enumerate(provenance_rows):
            ppath = f"{path}.provenance[{j}]"
            provenance = _object(raw_provenance, ppath)
            _strict(provenance, ppath, provenance_keys)
            for key in (
                "item_id",
                "source_id",
                "capture_id",
                "captured_at",
                "permalink",
                "provider",
                "provider_version",
                "provider_status",
                "capture_digest",
                "native_content_hash",
                "post_id",
                "canonical_post_id",
                "kind",
                "author_handle",
                "created_at",
            ):
                _string(provenance[key], f"{ppath}.{key}")
            _object(provenance["source_locator"], f"{ppath}.source_locator")
            _object(provenance["source_window"], f"{ppath}.source_window")
        _integer(candidate["duplicate_proposal_count"], f"{path}.duplicate_proposal_count", minimum=1)
        observation_count = _integer(
            candidate["observation_count"],
            f"{path}.observation_count",
            minimum=1,
        )
        if observation_count != len(provenance_rows):
            raise _where(f"{path}.observation_count", "must equal provenance row count")
    limitations = _list(report["limitations"], "report.limitations")
    for i, limitation in enumerate(limitations):
        _string(limitation, f"report.limitations[{i}]")
    return report


def load_json(path: Path) -> Any:
    _validate_local_json_path(path)
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_json_object,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ContractError(f"{path}: unreadable JSON ({exc})") from exc


def load_jsonl(path: Path) -> list[Any]:
    _validate_local_json_path(path, jsonl=True)
    records: list[Any] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"{path}: unreadable JSONL ({exc})") from exc
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(
                json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_strict_json_object,
                )
            )
        except (UnicodeError, ValueError) as exc:
            detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            raise ContractError(f"{path}:{i}: invalid JSON ({detail})") from exc
    return records


def _validate_local_json_path(path: Path, *, jsonl: bool = False) -> None:
    text = str(path)
    # pathlib normalizes ``https://`` to ``https:\\`` on Windows, so detect
    # schemes after that normalization while excluding one-letter drive names.
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]+:[\\/]{1,2}", text):
        raise ContractError(f"{path}: URLs are forbidden; use a local file snapshot")
    if text.startswith(("\\\\", "//")):
        raise ContractError(f"{path}: UNC/network paths are forbidden")
    allowed = {".jsonl"} if jsonl else {".json"}
    if path.suffix.lower() not in allowed:
        raise ContractError(f"{path}: only {sorted(allowed)} input files are accepted")
    if not path.is_file():
        raise ContractError(f"{path}: local input file does not exist")


def _reject_json_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON numeric constant is forbidden: {value}")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON object key is forbidden: {key!r}")
        result[key] = value
    return result
