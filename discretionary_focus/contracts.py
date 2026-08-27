"""Strict contracts for the research-only discretionary focus payload.

The payload is consumed by the private-site and email presentation layers.  It
is deliberately incapable of expressing an order, position size, allocation,
or fundamental ``QUICK_REVIEW`` decision.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import re
from typing import Any, Mapping
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


SCHEMA_VERSION = "discretionary-focus.v1"
PHASES = {"PROVISIONAL", "FINAL"}
STATUSES = {"READY", "NO_QUALIFIED_SETUP"}
MAX_FOCUS_NAMES = 2
RAW_PRICE_BASIS = "RAW_AS_TRADED"
ET = ZoneInfo("America/New_York")

# Friday's close is still the immediately prior completed session on Monday,
# and a three-day exchange weekend can put that bar almost 89 hours behind a
# pre-market run. The producer separately enforces exact prior-session data;
# this wall-clock bound is a second sanity check, not the freshness policy.
SCREEN_MAX_AGE = timedelta(hours=96)
RESEARCH_MAX_AGE = timedelta(hours=36)
LIVE_OBSERVATION_MAX_AGE = timedelta(minutes=5)
FUTURE_CLOCK_TOLERANCE = timedelta(minutes=5)
SOURCE_MAX_AGE = timedelta(days=550)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.^/-]{0,19}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

_TOP_LEVEL_KEYS = {
    "schema_version",
    "research_only",
    "quick_review_created",
    "live_actions_enabled",
    "order_staging_enabled",
    "status",
    "phase",
    "as_of",
    "valid_for",
    "generated_at",
    "expires_at",
    "focus",
    "screen_summary",
    "provenance",
    "no_setup_reason",
}

_CARD_KEYS = {
    "rank",
    "ticker",
    "company_name",
    "why_now",
    "setup",
    "trigger",
    "invalidation",
    "catalyst",
    "priced_in",
    "next_proof",
    "event_date",
    "earnings_td",
    "technical",
    "sources",
}

# These keys describe execution or a fundamental-sleeve decision.  Reject them
# anywhere in the payload rather than trusting a presentation layer to hide
# them.  Technical price levels are allowed separately with an explicit raw
# as-traded basis.
_FORBIDDEN_KEYS = {
    "action",
    "action_id",
    "allocation",
    "approval_status",
    "approved_for_capital",
    "broker",
    "decision",
    "dry_run_required",
    "limit_order",
    "notional",
    "order",
    "order_id",
    "order_type",
    "position_size",
    "position_size_pct",
    "proposed_weight_pct",
    "quantity",
    "quick_review",
    "risk_amt",
    "risk_bps",
    "shares",
    "side",
    "tif",
}

_PRICE_LEVEL_KEYS = {
    "price",
    "pivot",
    "level",
    "trigger_price",
    "stop_price",
    "invalidation_price",
}


class FocusPayloadError(ValueError):
    """A discretionary-focus payload failed its research-only contract."""


def _fail(path: str, message: str) -> None:
    raise FocusPayloadError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, "must be a list")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string")
    return value.strip()


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "must be an integer")
    if minimum is not None and value < minimum:
        _fail(path, f"must be >= {minimum}")
    return value


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(path, "must be finite")
    if minimum is not None and result < minimum:
        _fail(path, f"must be >= {minimum}")
    return result


def _date(value: Any, path: str) -> date:
    if not isinstance(value, str) or not _DATE_RE.fullmatch(value):
        _fail(path, "must be an ISO date (YYYY-MM-DD)")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise FocusPayloadError(f"{path}: invalid ISO date") from exc


def _datetime(value: Any, path: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a timezone-aware ISO datetime")
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        result = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise FocusPayloadError(f"{path}: invalid ISO datetime") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        _fail(path, "must include a timezone offset")
    return result.astimezone(timezone.utc)


def _now(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _datetime(value, "now")
    if not isinstance(value, datetime):
        _fail("now", "must be a datetime, ISO datetime, or None")
    if value.tzinfo is None or value.utcoffset() is None:
        _fail("now", "must include a timezone offset")
    return value.astimezone(timezone.utc)


def _reject_forbidden_keys(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).strip().lower()
            if key in _FORBIDDEN_KEYS:
                _fail(f"{path}.{raw_key}", "execution or QUICK_REVIEW field is forbidden")
            _reject_forbidden_keys(nested, f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_forbidden_keys(nested, f"{path}[{index}]")


def _validate_price_expression(value: Any, path: str) -> None:
    if isinstance(value, str):
        _text(value, path)
        return
    expression = _mapping(value, path)
    _text(expression.get("condition"), f"{path}.condition")
    has_numeric_price = False
    for key in _PRICE_LEVEL_KEYS:
        if key not in expression or expression[key] is None:
            continue
        _number(expression[key], f"{path}.{key}")
        has_numeric_price = True
    if has_numeric_price and expression.get("price_basis") != RAW_PRICE_BASIS:
        _fail(
            f"{path}.price_basis",
            f"numeric frozen levels require {RAW_PRICE_BASIS}",
        )


def _validate_invalidation(value: Any, path: str) -> None:
    invalidation = _mapping(value, path)
    if "technical" not in invalidation:
        _fail(f"{path}.technical", "is required")
    _validate_price_expression(invalidation["technical"], f"{path}.technical")
    _text(invalidation.get("thesis_kill"), f"{path}.thesis_kill")


def _validate_source(value: Any, path: str, *, generated_at: datetime) -> bool:
    source = _mapping(value, path)
    _text(source.get("source_id"), f"{path}.source_id")
    _text(source.get("label"), f"{path}.label")
    url = _text(source.get("url"), f"{path}.url")
    if urlparse(url).scheme not in {"http", "https"}:
        _fail(f"{path}.url", "must be an http(s) URL")
    source_as_of = source.get("as_of")
    if isinstance(source_as_of, str) and _DATE_RE.fullmatch(source_as_of):
        source_date = _date(source_as_of, f"{path}.as_of")
        generated_date = generated_at.astimezone(ET).date()
        if source_date > generated_date:
            _fail(f"{path}.as_of", "cannot be after generated_at")
        if generated_date - source_date > SOURCE_MAX_AGE:
            _fail(f"{path}.as_of", "is too stale")
    else:
        source_time = _datetime(source_as_of, f"{path}.as_of")
        if source_time > generated_at + FUTURE_CLOCK_TOLERANCE:
            _fail(f"{path}.as_of", "cannot be after generated_at")
        if generated_at - source_time > SOURCE_MAX_AGE:
            _fail(f"{path}.as_of", "is too stale")
    if not isinstance(source.get("primary"), bool):
        _fail(f"{path}.primary", "must be boolean")
    return bool(source["primary"])


def _validate_technical(
    value: Any,
    path: str,
    *,
    phase: str,
    generated_at: datetime,
) -> None:
    technical = _mapping(value, path)
    observed_at = _datetime(technical.get("observed_at"), f"{path}.observed_at")
    if observed_at > generated_at + FUTURE_CLOCK_TOLERANCE:
        _fail(f"{path}.observed_at", "cannot be after generated_at")
    if generated_at - observed_at > SCREEN_MAX_AGE:
        _fail(f"{path}.observed_at", f"is stale for phase {phase}")
    if technical.get("setup_gate") != "PASS":
        _fail(f"{path}.setup_gate", "must be PASS for a selected name")
    if technical.get("liquidity_gate") != "PASS":
        _fail(f"{path}.liquidity_gate", "must be PASS for a selected name")
    quality = _number(technical.get("setup_quality"), f"{path}.setup_quality", minimum=0)
    if quality > 100:
        _fail(f"{path}.setup_quality", "must be <= 100")


def _validate_focus_card(
    value: Any,
    path: str,
    *,
    expected_rank: int,
    valid_for: date,
    phase: str,
    generated_at: datetime,
) -> str:
    card = _mapping(value, path)
    unknown = set(card) - _CARD_KEYS
    if unknown:
        _fail(path, f"unexpected fields: {sorted(unknown)}")
    missing = _CARD_KEYS - set(card)
    if missing:
        _fail(path, f"missing fields: {sorted(missing)}")

    rank = _integer(card.get("rank"), f"{path}.rank", minimum=1)
    if rank != expected_rank:
        _fail(f"{path}.rank", f"must be contiguous and equal {expected_rank}")
    ticker = _text(card.get("ticker"), f"{path}.ticker")
    if ticker != ticker.upper() or not _TICKER_RE.fullmatch(ticker):
        _fail(f"{path}.ticker", "must be an uppercase normalized ticker")

    for field in (
        "company_name",
        "why_now",
        "setup",
        "catalyst",
        "priced_in",
        "next_proof",
    ):
        _text(card.get(field), f"{path}.{field}")

    _validate_price_expression(card.get("trigger"), f"{path}.trigger")
    _validate_invalidation(card.get("invalidation"), f"{path}.invalidation")
    event_date = _date(card.get("event_date"), f"{path}.event_date")
    earnings_td = _integer(card.get("earnings_td"), f"{path}.earnings_td")
    if earnings_td <= 5:
        _fail(f"{path}.earnings_td", "must be > 5 trading days")
    if event_date < valid_for:
        _fail(f"{path}.event_date", "cannot precede the focus session")

    _validate_technical(
        card.get("technical"),
        f"{path}.technical",
        phase=phase,
        generated_at=generated_at,
    )
    sources = _list(card.get("sources"), f"{path}.sources")
    if not sources:
        _fail(f"{path}.sources", "must contain at least one source")
    primary_count = sum(
        _validate_source(
            source,
            f"{path}.sources[{index}]",
            generated_at=generated_at,
        )
        for index, source in enumerate(sources)
    )
    if primary_count < 1:
        _fail(f"{path}.sources", "must contain at least one primary source")
    return ticker


def _validate_screen_summary(value: Any, path: str, selected_count: int) -> None:
    summary = _mapping(value, path)
    for field in (
        "input_count",
        "technical_pass_count",
        "research_pass_count",
        "selected_count",
    ):
        _integer(summary.get(field), f"{path}.{field}", minimum=0)
    if summary["selected_count"] != selected_count:
        _fail(f"{path}.selected_count", "must equal len(focus)")
    if summary["technical_pass_count"] > summary["input_count"]:
        _fail(f"{path}.technical_pass_count", "cannot exceed input_count")
    if summary["research_pass_count"] > summary["technical_pass_count"]:
        _fail(f"{path}.research_pass_count", "cannot exceed technical_pass_count")
    if summary["selected_count"] > summary["research_pass_count"]:
        _fail(f"{path}.selected_count", "cannot exceed research_pass_count")
    rejected = _mapping(summary.get("rejected_counts"), f"{path}.rejected_counts")
    rejected_total = 0
    for key, count in rejected.items():
        _text(str(key), f"{path}.rejected_counts key")
        _integer(count, f"{path}.rejected_counts.{key}", minimum=0)
        rejected_total += count
    if rejected_total + selected_count != summary["input_count"]:
        _fail(
            f"{path}.rejected_counts",
            "counts plus selected_count must equal input_count",
        )


def _validate_provenance(
    value: Any,
    path: str,
    *,
    phase: str,
    generated_at: datetime,
) -> None:
    provenance = _mapping(value, path)
    for field in ("screen_snapshot_id", "research_snapshot_id", "policy_version"):
        _text(provenance.get(field), f"{path}.{field}")
    screen_at = _datetime(
        provenance.get("screen_captured_at"), f"{path}.screen_captured_at"
    )
    research_at = _datetime(
        provenance.get("research_as_of"), f"{path}.research_as_of"
    )
    for name, timestamp in (("screen_captured_at", screen_at), ("research_as_of", research_at)):
        if timestamp > generated_at + FUTURE_CLOCK_TOLERANCE:
            _fail(f"{path}.{name}", "cannot be after generated_at")
    if generated_at - screen_at > SCREEN_MAX_AGE:
        _fail(f"{path}.screen_captured_at", f"is stale for phase {phase}")
    if generated_at - research_at > RESEARCH_MAX_AGE:
        _fail(f"{path}.research_as_of", "is stale")
    for digest_field in ("screen_digest", "research_digest"):
        if digest_field in provenance:
            digest = _text(provenance[digest_field], f"{path}.{digest_field}")
            if not _HEX64_RE.fullmatch(digest):
                _fail(f"{path}.{digest_field}", "must be a lowercase SHA-256 digest")


def validate_payload(
    payload: Mapping[str, Any],
    *,
    now: datetime | str | None = None,
    require_current: bool = False,
) -> dict[str, Any]:
    """Return a defensive copy after validating the v1 focus contract.

    ``now`` is optional so archived payloads can be structurally inspected.
    Current-site and email consumers should pass their clock; expired or
    implausibly future-dated payloads then fail closed.
    """

    document = deepcopy(dict(_mapping(payload, "payload")))
    _reject_forbidden_keys(document)
    unknown = set(document) - _TOP_LEVEL_KEYS
    if unknown:
        _fail("payload", f"unexpected fields: {sorted(unknown)}")
    required = _TOP_LEVEL_KEYS - {"no_setup_reason"}
    missing = required - set(document)
    if missing:
        _fail("payload", f"missing fields: {sorted(missing)}")

    if document.get("schema_version") != SCHEMA_VERSION:
        _fail("payload.schema_version", f"must be {SCHEMA_VERSION}")
    safety = {
        "research_only": True,
        "quick_review_created": False,
        "live_actions_enabled": False,
        "order_staging_enabled": False,
    }
    for field, expected in safety.items():
        if document.get(field) is not expected:
            _fail(f"payload.{field}", f"must be {str(expected).lower()}")

    status = _text(document.get("status"), "payload.status")
    if status not in STATUSES:
        _fail("payload.status", f"must be one of {sorted(STATUSES)}")
    phase = _text(document.get("phase"), "payload.phase")
    if phase not in PHASES:
        _fail("payload.phase", f"must be one of {sorted(PHASES)}")
    as_of = _date(document.get("as_of"), "payload.as_of")
    valid_for = _date(document.get("valid_for"), "payload.valid_for")
    if as_of > valid_for:
        _fail("payload.as_of", "cannot be after valid_for")

    generated_at = _datetime(document.get("generated_at"), "payload.generated_at")
    expires_at = _datetime(document.get("expires_at"), "payload.expires_at")
    if expires_at <= generated_at:
        _fail("payload.expires_at", "must be after generated_at")
    generated_local = generated_at.astimezone(ET)
    expires_local = expires_at.astimezone(ET)
    if phase == "FINAL" and generated_local.date() != valid_for:
        _fail("payload.generated_at", "must be generated on valid_for in New York")
    allowed_expiry_times = {datetime.min.time().replace(hour=13, minute=15), datetime.min.time().replace(hour=16, minute=15)}
    expiry_clock = expires_local.time().replace(tzinfo=None)
    if expires_local.date() != valid_for or expiry_clock not in allowed_expiry_times:
        _fail(
            "payload.expires_at",
            "must be 15 minutes after a regular or early XNYS close on valid_for",
        )
    if not isinstance(require_current, bool):
        _fail("require_current", "must be boolean")
    current = _now(now)
    if require_current and current is None:
        current = datetime.now(timezone.utc)
    if current is not None:
        if generated_at > current + FUTURE_CLOCK_TOLERANCE:
            _fail("payload.generated_at", "is implausibly in the future")
        if current >= expires_at:
            _fail("payload.expires_at", "payload is expired")
        market_date = current.astimezone(ET).date()
        if require_current and market_date != valid_for:
            _fail(
                "payload.valid_for",
                f"current delivery requires today's session ({market_date})",
            )

    focus = _list(document.get("focus"), "payload.focus")
    if len(focus) > MAX_FOCUS_NAMES:
        _fail("payload.focus", f"may contain at most {MAX_FOCUS_NAMES} names")
    if status == "READY" and not focus:
        _fail("payload.focus", "READY requires one or two names")
    if status == "NO_QUALIFIED_SETUP" and focus:
        _fail("payload.focus", "NO_QUALIFIED_SETUP requires an empty list")
    if status == "NO_QUALIFIED_SETUP":
        _text(document.get("no_setup_reason"), "payload.no_setup_reason")
    elif "no_setup_reason" in document and document["no_setup_reason"]:
        _fail("payload.no_setup_reason", "is allowed only for NO_QUALIFIED_SETUP")

    tickers = [
        _validate_focus_card(
            card,
            f"payload.focus[{index}]",
            expected_rank=index + 1,
            valid_for=valid_for,
            phase=phase,
            generated_at=generated_at,
        )
        for index, card in enumerate(focus)
    ]
    if len(tickers) != len(set(tickers)):
        _fail("payload.focus", "duplicate tickers are forbidden")

    _validate_screen_summary(document.get("screen_summary"), "payload.screen_summary", len(focus))
    _validate_provenance(
        document.get("provenance"),
        "payload.provenance",
        phase=phase,
        generated_at=generated_at,
    )
    return document


def canonical_digest(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest for exact-once delivery and provenance."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "FocusPayloadError",
    "MAX_FOCUS_NAMES",
    "PHASES",
    "RAW_PRICE_BASIS",
    "SCHEMA_VERSION",
    "STATUSES",
    "canonical_digest",
    "validate_payload",
]
