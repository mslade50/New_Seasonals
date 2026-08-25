"""Pure premarket discovery rules for EP nominations."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import EPPolicy
from .schema import Candidate, PremarketSnapshot, parse_timestamp


_NY = ZoneInfo("America/New_York")


def _candidate_id(symbol: str, observed_at: str, policy_id: str) -> str:
    session = parse_timestamp(observed_at).astimezone(_NY).date().isoformat()
    seed = f"{policy_id}|{session}|{symbol.upper()}".encode("utf-8")
    return f"EP-{session}-{symbol.upper()}-{hashlib.sha256(seed).hexdigest()[:10]}"


def _positive_number(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def nominate_candidates(
    snapshots: list[PremarketSnapshot],
    *,
    as_of: str | datetime,
    policy: EPPolicy,
) -> list[Candidate]:
    """Return broad EP research nominations, newest snapshot per symbol.

    Data-quality problems are retained as warnings so the report explains why a
    visible mover was not stageable.  Basic move/price/volume failures are not
    nominations at all.
    """

    decision_at = parse_timestamp(as_of)
    latest: dict[str, PremarketSnapshot] = {}
    for snapshot in snapshots:
        existing = latest.get(snapshot.symbol)
        if existing is None or parse_timestamp(snapshot.observed_at) > parse_timestamp(
            existing.observed_at
        ):
            latest[snapshot.symbol] = snapshot

    out: list[Candidate] = []
    rules = policy.discovery
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

        direction_ok = snapshot.gap_pct > 0 if rules.long_only else snapshot.gap_pct != 0
        move_ok = (
            abs(snapshot.gap_pct) >= rules.min_abs_gap_pct
            or abs(snapshot.move_dollars) >= rules.min_abs_move_dollars
        )
        if not (direction_ok and move_ok):
            continue

        reasons = [
            "PREMARKET_MOVE_THRESHOLD",
            "PREMARKET_VOLUME_THRESHOLD",
            "PRICE_THRESHOLD",
        ]
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
        if snapshot.gap_pct >= policy.execution.extension_warning_gap_pct:
            warnings.append("EXTENDED_GAP")
        if snapshot.gap_pct > policy.execution.max_immediate_gap_pct:
            warnings.append("DELAYED_EP_PREFERRED")

        out.append(
            Candidate(
                candidate_id=_candidate_id(symbol, snapshot.observed_at, policy.policy_id),
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
    return out[: rules.max_candidates]
