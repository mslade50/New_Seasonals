"""Pure premarket discovery rules for EP nominations."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, time
from zoneinfo import ZoneInfo

from .config import EPPolicy
from .schema import Candidate, PremarketSnapshot, parse_timestamp

_NY = ZoneInfo("America/New_York")
_TRADINGVIEW_PREMARKET_SCREEN_IDS = {"yftOvM3e"}
PREMARKET_MOVE_VERIFIED = "VERIFIED"
PREMARKET_MOVE_UNVERIFIED = "UNVERIFIED"


def _candidate_id(snapshot: PremarketSnapshot, policy_id: str) -> str:
    session = (
        snapshot.target_session_date
        or parse_timestamp(snapshot.observed_at).astimezone(_NY).date().isoformat()
    )
    symbol = snapshot.symbol
    seed = f"{policy_id}|{session}|{symbol.upper()}".encode()
    return f"EP-{session}-{symbol.upper()}-{hashlib.sha256(seed).hexdigest()[:10]}"


def _positive_number(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def premarket_move_is_verified(
    snapshot: PremarketSnapshot,
    *,
    as_of: str | datetime | None = None,
    max_age_seconds: int = 900,
    future_tolerance_seconds: int = 2,
    require_fresh_at_as_of: bool = False,
) -> bool:
    """Return whether TV or IBKR verified the move in the target premarket.

    A successful verification may be frozen on the snapshot.  Frozen evidence
    is checked against the source observation at ``premarket_move_verified_at``
    and does not expire merely because ATR/news work finishes later.  Boundary
    ingestion can set ``require_fresh_at_as_of`` to independently prove that a
    row was still fresh when its enclosing artifact finished capture.
    """

    if snapshot.session.lower() != "premarket" or not snapshot.target_session_date:
        return False
    try:
        target_date = datetime.fromisoformat(snapshot.target_session_date).date()
    except ValueError:
        return False
    provider = snapshot.provider.strip().upper()
    source = snapshot.source.strip().upper()
    if provider == "TRADINGVIEW":
        bulk = source == "TRADINGVIEW_PUBLIC_BULK"
        if (
            (source != "TRADINGVIEW_BROWSER_EXPORT" and not bulk)
            or (not bulk and snapshot.saved_screen_id not in _TRADINGVIEW_PREMARKET_SCREEN_IDS)
            or (
                snapshot.reported_change_pct is None
                and snapshot.reported_move_dollars is None
            )
        ):
            return False
        observed_value = snapshot.observed_at
        if bulk and (not snapshot.source_file_sha256 or not snapshot.reported_result_count
                     or snapshot.reported_result_count != snapshot.extracted_row_count):
            return False
        expected_verification_source = source
    elif provider == "IBKR":
        if source != "IBKR_TARGETED_READ_ONLY":
            return False
        if snapshot.market_data_status.strip().upper() != "LIVE":
            return False
        if not snapshot.premarket_metrics_at:
            return False
        observed_value = snapshot.premarket_metrics_at
        expected_verification_source = "IBKR_TARGETED_READ_ONLY"
    else:
        return False
    try:
        observed_utc = parse_timestamp(observed_value)
    except (TypeError, ValueError):
        return False
    observed = observed_utc.astimezone(_NY)
    session_matches = observed.date() == target_date and time(
        4, 0
    ) <= observed.time().replace(tzinfo=None) < time(9, 30)
    if not session_matches:
        return False

    def fresh_at(value: str | datetime) -> bool:
        try:
            checked_at = parse_timestamp(value)
        except (TypeError, ValueError):
            return False
        age_seconds = (checked_at - observed_utc).total_seconds()
        return (
            age_seconds >= -future_tolerance_seconds and age_seconds <= max_age_seconds
        )

    status = snapshot.premarket_move_verification_status.strip().upper()
    verification_source = snapshot.premarket_move_verification_source.strip().upper()
    verified_at = snapshot.premarket_move_verified_at
    has_persisted_verification = (
        status not in {"", PREMARKET_MOVE_UNVERIFIED}
        or bool(verification_source)
        or verified_at is not None
    )
    if has_persisted_verification:
        if (
            status != PREMARKET_MOVE_VERIFIED
            or verification_source != expected_verification_source
            or not verified_at
        ):
            return False
        try:
            verified_local = parse_timestamp(verified_at).astimezone(_NY)
        except (TypeError, ValueError):
            return False
        if not (
            verified_local.date() == target_date
            and time(4, 0) <= verified_local.time().replace(tzinfo=None) < time(9, 30)
            and fresh_at(verified_at)
        ):
            return False
        if not require_fresh_at_as_of:
            return True

    if as_of is None:
        return not require_fresh_at_as_of and not has_persisted_verification
    if require_fresh_at_as_of:
        try:
            as_of_local = parse_timestamp(as_of).astimezone(_NY)
        except (TypeError, ValueError):
            return False
        if not (
            as_of_local.date() == target_date
            and time(4, 0) <= as_of_local.time().replace(tzinfo=None) < time(9, 30)
        ):
            return False
    return fresh_at(as_of)


def nominate_candidates(
    snapshots: list[PremarketSnapshot],
    *,
    as_of: str | datetime,
    policy: EPPolicy,
    apply_candidate_limit: bool = True,
    require_verified_premarket_move: bool = False,
) -> list[Candidate]:
    """Return broad EP research nominations, newest snapshot per symbol.

    Data-quality problems are retained as warnings so the report explains why a
    visible mover was not preview-eligible.  Basic move/price/volume failures are not
    nominations at all.
    """

    decision_at = parse_timestamp(as_of)
    rules = policy.discovery
    latest: dict[str, PremarketSnapshot] = {}
    for snapshot in snapshots:
        # The gate is an OR across independently verified morning sources.
        # Filter first so a newer partial/frozen IBKR row cannot erase a valid
        # TradingView observation for the same symbol during deduplication.
        if require_verified_premarket_move and not premarket_move_is_verified(
            snapshot,
            as_of=decision_at,
            max_age_seconds=rules.premarket_metrics_max_age_seconds,
            future_tolerance_seconds=rules.future_timestamp_tolerance_seconds,
        ):
            continue
        existing = latest.get(snapshot.symbol)
        if existing is None or parse_timestamp(snapshot.observed_at) > parse_timestamp(
            existing.observed_at
        ):
            latest[snapshot.symbol] = snapshot

    out: list[Candidate] = []
    for symbol in sorted(latest):
        snapshot = latest[symbol]
        if not all(
            _positive_number(v)
            for v in (
                snapshot.previous_close,
                snapshot.last,
                snapshot.premarket_volume,
            )
        ):
            continue
        if snapshot.last < rules.min_price:
            continue
        if snapshot.premarket_volume < rules.min_premarket_volume:
            continue

        discovery_gap_pct = snapshot.discovery_gap_pct
        discovery_move_dollars = snapshot.discovery_move_dollars
        direction_ok = (
            discovery_gap_pct > 0
            if rules.long_only
            else (discovery_gap_pct != 0 or discovery_move_dollars != 0)
        )
        move_ok = abs(discovery_gap_pct) >= rules.min_abs_gap_pct
        if not (direction_ok and move_ok):
            continue

        reasons = ["PREMARKET_VOLUME_THRESHOLD", "PRICE_THRESHOLD"]
        if abs(discovery_gap_pct) >= rules.min_abs_gap_pct:
            reasons.append("SESSION_PERCENT_MOVE_THRESHOLD")
        if snapshot.premarket_volume >= 8_900_000:
            reasons.append("EP9M_VOLUME_DISCOVERY")

        warnings: list[str] = []
        age = (decision_at - parse_timestamp(snapshot.observed_at)).total_seconds()
        if age < -rules.future_timestamp_tolerance_seconds:
            warnings.append("SNAPSHOT_FROM_FUTURE")
        elif age > rules.quote_max_age_seconds:
            warnings.append("STALE_MARKET_DATA")
        if snapshot.market_data_status.upper() != "LIVE":
            warnings.append("NON_LIVE_MARKET_DATA")
        halt_status = snapshot.halt_status.upper().strip()
        if snapshot.halted or halt_status in {"GENERAL_HALT", "VOLATILITY_HALT"}:
            warnings.append("HALTED")
        elif halt_status != "NOT_HALTED":
            warnings.append("HALT_STATUS_UNKNOWN")
        if not snapshot.tradeable:
            warnings.append("NOT_TRADEABLE")
        if discovery_gap_pct < 0:
            warnings.append("BEARISH_RESEARCH_ONLY")
        if discovery_gap_pct > policy.execution.extension_warning_gap_pct:
            warnings.append("EXTENDED_GAP")
        if discovery_gap_pct > policy.execution.max_immediate_gap_pct:
            warnings.append("DELAYED_EP_PREFERRED")

        out.append(
            Candidate(
                candidate_id=_candidate_id(snapshot, policy.policy_id),
                snapshot=snapshot,
                discovery_reasons=tuple(reasons),
                discovery_warnings=tuple(warnings),
            )
        )
    out.sort(
        key=lambda item: (
            -item.snapshot.premarket_volume,
            -item.snapshot.premarket_dollar_volume,
            item.snapshot.symbol,
        )
    )
    return out[: rules.max_candidates] if apply_candidate_limit else out
