"""Deterministic EP classification and research-preview gates."""

from __future__ import annotations

import math
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from .config import EPPolicy
from .schema import Candidate, CatalystAssessment, QualificationDecision, parse_timestamp


_NY = ZoneInfo("America/New_York")


_CLASSIC_TYPES = {
    "EARNINGS_GUIDANCE",
    "EARNINGS",
    "REGULATORY_APPROVAL",
    "CLINICAL_DATA",
    "MATERIAL_CONTRACT",
    "PRODUCT_TECHNOLOGY",
}

_HARD_REJECT_ADVERSE = {
    "FIXED_PRICE_TAKEOVER",
    "REVERSE_SPLIT",
    "BANKRUPTCY_OR_GOING_CONCERN",
}
_BEARISH_RESEARCH_ADVERSE = {
    "GUIDANCE_CUT_OR_WITHDRAWAL",
    "CLINICAL_OR_REGULATORY_FAILURE",
    "RESTATEMENT_OR_RECALL",
    "INVESTIGATION",
}
_DIRECT_LONG_BLOCK_ADVERSE = {"DILUTION_OR_OFFERING"}
_LONG_ENTRY_BLOCKERS = {
    "BELOW_STAGE_PRICE",
    "MOVE_IS_DISCOVERY_ONLY",
    "CURRENT_ASK_BELOW_STAGE_GAP",
    "GAP_TOO_EXTENDED_FOR_IMMEDIATE_ENTRY",
    "PRICE_TOO_FAR_ABOVE_PREMARKET_VWAP",
    "INVALID_INITIAL_STOP_REFERENCE",
}


def _valid_positive(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def qualify_candidate(
    candidate: Candidate,
    catalyst: CatalystAssessment,
    *,
    policy: EPPolicy,
    decision_at: str | datetime,
    target_session_date: date | str | None = None,
) -> QualificationDecision:
    snap = candidate.snapshot
    bearish = snap.discovery_gap_pct < 0 or snap.discovery_move_dollars < 0
    blockers: list[str] = []
    warnings = list(candidate.discovery_warnings)
    execution = policy.execution
    decision_timestamp = parse_timestamp(decision_at)
    local_decision = decision_timestamp.astimezone(_NY)
    target_date = (
        target_session_date
        if isinstance(target_session_date, date)
        else date.fromisoformat(target_session_date)
        if target_session_date
        else local_decision.date()
    )
    if local_decision.date() > target_date or (
        local_decision.date() == target_date and local_decision.time() >= time(9, 35)
    ):
        blockers.append("ENTRY_WINDOW_EXPIRED")
    if not snap.premarket_metrics_at:
        blockers.append("MISSING_PREMARKET_METRICS_TIMESTAMP")
    else:
        metrics_age = (
            decision_timestamp - parse_timestamp(snap.premarket_metrics_at)
        ).total_seconds()
        if metrics_age < -policy.discovery.future_timestamp_tolerance_seconds:
            blockers.append("PREMARKET_METRICS_FROM_FUTURE")
        elif metrics_age > policy.discovery.premarket_metrics_max_age_seconds:
            blockers.append("STALE_PREMARKET_METRICS")

    for warning in candidate.discovery_warnings:
        if warning in {
            "SNAPSHOT_FROM_FUTURE",
            "STALE_MARKET_DATA",
            "NON_LIVE_MARKET_DATA",
            "HALTED",
            "HALT_STATUS_UNKNOWN",
            "NOT_TRADEABLE",
        }:
            blockers.append(warning)

    if not all(_valid_positive(v) for v in (snap.bid, snap.ask, snap.premarket_vwap)):
        blockers.append("MISSING_EXECUTABLE_QUOTE")
    elif snap.ask < snap.bid:
        blockers.append("CROSSED_QUOTE")
    if snap.last < execution.min_stage_price:
        blockers.append("BELOW_STAGE_PRICE")
    if not isinstance(snap.contract_con_id, int) or snap.contract_con_id <= 0:
        blockers.append("UNRESOLVED_IB_CONTRACT")
    if snap.contract_identity_status != "UNIQUE_IBKR_MATCH":
        blockers.append("UNRESOLVED_CONTRACT_IDENTITY")
    if snap.resolved_symbol.strip().upper() != snap.symbol:
        blockers.append("CONTRACT_SYMBOL_MISMATCH")
    if snap.contract_sec_type != "STK" or snap.contract_currency != "USD":
        blockers.append("INVALID_CONTRACT_CLASS")
    if not snap.primary_exchange.strip() or snap.primary_exchange.upper() == "SMART":
        blockers.append("MISSING_PRIMARY_EXCHANGE")
    valid_exchanges = {
        item.strip().upper() for item in snap.valid_exchanges.split(",") if item.strip()
    }
    if "SMART" not in valid_exchanges:
        blockers.append("SMART_ROUTING_UNAVAILABLE")
    if not _valid_positive(snap.quote_previous_close):
        blockers.append("MISSING_QUOTE_PREVIOUS_CLOSE")
    else:
        basis_mismatch_pct = 100.0 * abs(
            float(snap.quote_previous_close) / snap.previous_close - 1.0
        )
        if basis_mismatch_pct > policy.discovery.max_prior_close_basis_mismatch_pct:
            blockers.append("PRIOR_CLOSE_BASIS_MISMATCH")
    if snap.gap_pct < execution.min_stage_gap_pct:
        blockers.append("MOVE_IS_DISCOVERY_ONLY")
    ask_gap_pct = 100.0 * (snap.ask / snap.previous_close - 1.0) if snap.previous_close > 0 else float("-inf")
    if ask_gap_pct < execution.min_stage_gap_pct:
        blockers.append("CURRENT_ASK_BELOW_STAGE_GAP")
    if max(snap.gap_pct, ask_gap_pct) > execution.max_immediate_gap_pct:
        blockers.append("GAP_TOO_EXTENDED_FOR_IMMEDIATE_ENTRY")
    if snap.premarket_dollar_volume < execution.min_premarket_dollar_volume:
        blockers.append("INSUFFICIENT_PREMARKET_DOLLAR_VOLUME")
    if snap.spread_bps > execution.max_spread_bps:
        blockers.append("SPREAD_TOO_WIDE")
    if _valid_positive(snap.premarket_vwap):
        chase_pct = 100.0 * (snap.ask / snap.premarket_vwap - 1.0)
        if chase_pct > execution.max_chase_above_premarket_vwap_pct:
            blockers.append("PRICE_TOO_FAR_ABOVE_PREMARKET_VWAP")
    if not _valid_positive(snap.prior_two_day_low) or snap.prior_two_day_low >= snap.ask:
        blockers.append("INVALID_INITIAL_STOP_REFERENCE")
    if not _valid_positive(snap.addv_63):
        blockers.append("MISSING_ADDV_CAPACITY")
    if snap.ask_size <= 0:
        blockers.append("MISSING_DISPLAYED_ASK_SIZE")

    if snap.prior_63d_return_pct is not None and snap.prior_63d_return_pct > 20:
        warnings.append("NOT_NEGLECTED_PRIOR_63D_RUNUP")
    if snap.sessions_since_prior_ep is not None and snap.sessions_since_prior_ep < 126:
        warnings.append("NOT_FIRST_EP_IN_SIX_MONTHS")
    if (
        snap.prior_63d_return_pct is not None
        and snap.prior_63d_return_pct > 20
        and snap.sessions_since_prior_ep is not None
        and snap.sessions_since_prior_ep < 126
    ):
        blockers.append("LATE_CYCLE_RUNUP_AND_RECENT_EP")
    if snap.market_cap is not None and snap.market_cap > 10_000_000_000:
        warnings.append("LARGE_CAP_INSTITUTIONAL_VARIANT")

    if catalyst.status == "ADVERSE":
        adverse = set(catalyst.adverse_flags or ("ADVERSE_CATALYST",))
        blockers.extend(adverse)
        if adverse & _HARD_REJECT_ADVERSE:
            decision = "REJECT"
            setup_type = "CORPORATE_ACTION_OR_DISTRESS_REJECT"
        elif bearish and adverse & _BEARISH_RESEARCH_ADVERSE:
            decision = "WATCH"
            setup_type = "BEARISH_EP_RESEARCH"
        elif adverse & _DIRECT_LONG_BLOCK_ADVERSE:
            decision = "WATCH"
            setup_type = "CATALYST_WITH_FINANCING_RISK"
        else:
            decision = "WATCH"
            setup_type = "ADVERSE_EVENT_RESEARCH"
    elif catalyst.status != "CONFIRMED":
        blockers.append("CATALYST_NOT_CONFIRMED")
        decision = "WATCH"
        setup_type = (
            "EP9M_STORY_WATCH"
            if "EP9M_VOLUME_DISCOVERY" in candidate.discovery_reasons
            else "UNCONFIRMED_MOVER"
        )
    elif catalyst.catalyst_type == "M_AND_A":
        blockers.append("M_AND_A_REQUIRES_MANUAL_REVALUATION")
        decision = "WATCH"
        setup_type = "CLASSIC_EP_REVIEW"
    elif catalyst.catalyst_type == "ANALYST_ACTION":
        blockers.append("ANALYST_ACTION_NOT_AUTOMATICALLY_CLASSIC")
        decision = "WATCH"
        setup_type = "STORY_EP_WATCH"
    elif catalyst.catalyst_type not in _CLASSIC_TYPES:
        blockers.append("CATALYST_CLASS_NOT_STAGE_ENABLED")
        decision = "WATCH"
        setup_type = "STORY_EP_WATCH"
    elif not catalyst.primary_source_confirmed:
        blockers.append("PRIMARY_SOURCE_NOT_VERIFIED")
        decision = "WATCH"
        setup_type = "CLASSIC_EP_REVIEW"
    elif not catalyst.publication_time_verified:
        blockers.append("PUBLICATION_TIME_NOT_VERIFIED")
        decision = "WATCH"
        setup_type = "CLASSIC_EP_REVIEW"
    elif not catalyst.trajectory_change_verified:
        blockers.append("TRAJECTORY_CHANGE_NOT_VERIFIED")
        decision = "WATCH"
        setup_type = "CLASSIC_EP_REVIEW"
    elif catalyst.materiality_score < (
        4 if catalyst.catalyst_type in {"EARNINGS", "MATERIAL_CONTRACT", "PRODUCT_TECHNOLOGY"} else 3
    ):
        blockers.append("MATERIALITY_SCORE_BELOW_STAGE_GATE")
        decision = "WATCH"
        setup_type = "CLASSIC_EP_REVIEW"
    else:
        setup_type = (
            "EP9M_CATALYST"
            if "EP9M_VOLUME_DISCOVERY" in candidate.discovery_reasons
            else "CLASSIC_EP"
        )
        decision = "RESEARCH_PREVIEW_ELIGIBLE" if not blockers else "WATCH"

    if "GAP_TOO_EXTENDED_FOR_IMMEDIATE_ENTRY" in blockers and catalyst.status == "CONFIRMED":
        setup_type = "EXTENDED_GAP_DEP_CANDIDATE"
        decision = "WATCH"

    # Negative movers belong in the research funnel, but this shadow version
    # deliberately has no borrow, locate, SSR, or short-execution contract.
    if bearish:
        blockers.append("BEARISH_EXECUTION_NOT_IMPLEMENTED")
        if decision != "REJECT":
            decision = "WATCH"
            setup_type = "BEARISH_EP_RESEARCH"
        blockers = [item for item in blockers if item not in _LONG_ENTRY_BLOCKERS]

    return QualificationDecision(
        candidate_id=candidate.candidate_id,
        symbol=snap.symbol,
        decision=decision,
        setup_type=setup_type,
        catalyst=catalyst,
        blockers=tuple(sorted(set(blockers))),
        warnings=tuple(sorted(set(warnings))),
    )
