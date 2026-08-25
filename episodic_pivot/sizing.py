"""Liquidity-aware sizing for deliberately non-executable EP research previews."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date

from .config import EPPolicy
from .schema import Candidate, QualificationDecision, ResearchSizingPreview


@dataclass(frozen=True)
class SizingOutcome:
    preview: ResearchSizingPreview | None
    blockers: tuple[str, ...] = ()


def apply_daily_preview_caps(
    previews: list[ResearchSizingPreview], *, policy: EPPolicy
) -> list[ResearchSizingPreview]:
    """Pro-rata hypothetical previews to daily research risk/notional caps."""

    if not previews:
        return []
    rules = policy.execution
    total_risk = sum(item.modeled_risk_dollars for item in previews)
    total_notional = sum(item.hypothetical_notional for item in previews)
    risk_cap = rules.account_value * rules.max_daily_risk_bps / 10_000.0
    notional_cap = (
        rules.account_value * rules.max_daily_notional_pct_of_account / 100.0
    )
    factors = {
        "DAILY_RISK": risk_cap / total_risk if total_risk > 0 else 1.0,
        "DAILY_NOTIONAL": notional_cap / total_notional if total_notional > 0 else 1.0,
    }
    binding, factor = min(factors.items(), key=lambda item: item[1])
    factor = min(1.0, factor)
    if factor >= 1.0:
        return previews

    adjusted: list[ResearchSizingPreview] = []
    for item in previews:
        quantity = math.floor(item.max_preview_shares * factor)
        if quantity < 1:
            continue
        risk_dollars = quantity * item.modeled_risk_per_share
        adjusted.append(
            replace(
                item,
                max_preview_shares=quantity,
                modeled_risk_dollars=round(risk_dollars, 2),
                modeled_risk_bps=round(10_000.0 * risk_dollars / rules.account_value, 4),
                hypothetical_notional=round(quantity * item.reference_entry_price, 2),
                binding_constraint=f"{item.binding_constraint}+{binding}",
            )
        )
    return adjusted


def _round_up_cent(value: float) -> float:
    return math.ceil(value * 100.0 - 1e-9) / 100.0


def _round_down_cent(value: float) -> float:
    return math.floor(value * 100.0 + 1e-9) / 100.0


def build_research_sizing_preview(
    candidate: Candidate,
    decision: QualificationDecision,
    *,
    policy: EPPolicy,
    target_session_date: date | str,
) -> SizingOutcome:
    if decision.decision != "RESEARCH_PREVIEW_ELIGIBLE":
        return SizingOutcome(None, ("DECISION_NOT_PREVIEW_ELIGIBLE",))

    snap = candidate.snapshot
    rules = policy.execution
    blockers: list[str] = []
    if snap.ask <= 0 or snap.bid <= 0 or snap.ask < snap.bid:
        blockers.append("INVALID_QUOTE")
    if snap.premarket_vwap <= 0 or snap.addv_63 <= 0 or snap.premarket_volume <= 0:
        blockers.append("MISSING_LIQUIDITY_INPUT")
    if snap.ask_size <= 0:
        blockers.append("MISSING_DISPLAYED_ASK_SIZE")
    if blockers:
        return SizingOutcome(None, tuple(blockers))

    stop = _round_down_cent(
        snap.prior_two_day_low * (1.0 - rules.stop_buffer_bps / 10_000.0)
    )
    if stop <= 0 or stop >= snap.ask:
        return SizingOutcome(None, ("INVALID_INITIAL_STOP",))
    raw_stop_pct = 100.0 * (snap.ask - stop) / snap.ask
    if raw_stop_pct > rules.max_stop_distance_pct:
        return SizingOutcome(None, ("STOP_DISTANCE_TOO_WIDE",))

    risk_bps = (
        rules.ep9m_catalyst_risk_bps
        if decision.setup_type == "EP9M_CATALYST"
        else rules.classic_risk_bps
    )
    risk_budget = rules.account_value * risk_bps / 10_000.0
    half_spread_bps = max(0.0, snap.spread_bps / 2.0)

    # First pass uses the ask so impact can be estimated from the naturally
    # binding risk/liquidity cap rather than from an arbitrary maximum order.
    stressed_exit_per_share = snap.ask * rules.stressed_exit_slippage_bps / 10_000.0
    gap_stress_per_share = snap.ask * rules.event_gap_stress_pct / 100.0
    provisional_risk_per_share = (
        snap.ask - stop + stressed_exit_per_share + gap_stress_per_share
    )
    caps = {
        "RISK": math.floor(risk_budget / provisional_risk_per_share),
        "MAX_NOTIONAL": math.floor(
            (rules.account_value * rules.max_notional_pct_of_account / 100.0) / snap.ask
        ),
        "ADDV_PARTICIPATION": math.floor(
            (snap.addv_63 * rules.addv_participation) / snap.ask
        ),
        "PREMARKET_VOLUME_PARTICIPATION": math.floor(
            snap.premarket_volume * rules.premarket_volume_participation
        ),
        "DISPLAYED_ASK_PARTICIPATION": math.floor(
            snap.ask_size * rules.displayed_size_participation
        ),
        "ABSOLUTE_QUANTITY": rules.max_quantity,
    }
    provisional_qty = min(caps.values())
    if provisional_qty < 1:
        binding = min(caps, key=caps.get)
        return SizingOutcome(None, (f"ZERO_QUANTITY:{binding}",))

    participation = provisional_qty * snap.ask / snap.addv_63
    impact_bps = rules.impact_coefficient_bps * math.sqrt(max(0.0, participation))
    expected_slippage_bps = half_spread_bps + rules.base_entry_slippage_bps + impact_bps
    if expected_slippage_bps > rules.max_entry_slippage_bps:
        return SizingOutcome(None, ("EXPECTED_ENTRY_SLIPPAGE_TOO_HIGH",))

    entry_limit = _round_up_cent(
        snap.ask * (1.0 + expected_slippage_bps / 10_000.0)
    )
    activation_max_price = math.floor(
        snap.previous_close * (1.0 + rules.max_immediate_gap_pct / 100.0) * 100.0
    ) / 100.0
    if entry_limit > activation_max_price:
        return SizingOutcome(None, ("LIMIT_EXCEEDS_MAX_OPENING_GAP",))
    modeled_risk_per_share = (
        entry_limit
        - stop
        + entry_limit * rules.stressed_exit_slippage_bps / 10_000.0
        + entry_limit * rules.event_gap_stress_pct / 100.0
    )
    caps["RISK"] = math.floor(risk_budget / modeled_risk_per_share)
    caps["MAX_NOTIONAL"] = math.floor(
        (rules.account_value * rules.max_notional_pct_of_account / 100.0) / entry_limit
    )
    caps["ADDV_PARTICIPATION"] = math.floor(
        (snap.addv_63 * rules.addv_participation) / entry_limit
    )
    quantity = min(caps.values())
    binding_cap = min(caps, key=caps.get)
    if quantity < 1:
        return SizingOutcome(None, (f"ZERO_QUANTITY:{binding_cap}",))

    risk_dollars = quantity * modeled_risk_per_share
    session_date = (
        target_session_date.isoformat()
        if isinstance(target_session_date, date)
        else str(target_session_date)
    )
    preview = ResearchSizingPreview(
        candidate_id=candidate.candidate_id,
        symbol=snap.symbol,
        research_direction="LONG",
        max_preview_shares=quantity,
        reference_entry_price=entry_limit,
        hypothetical_stop_price=stop,
        modeled_risk_per_share=round(modeled_risk_per_share, 4),
        modeled_risk_dollars=round(risk_dollars, 2),
        modeled_risk_bps=round(10_000.0 * risk_dollars / rules.account_value, 4),
        hypothetical_notional=round(quantity * entry_limit, 2),
        expected_entry_slippage_bps=round(expected_slippage_bps, 2),
        max_entry_slippage_bps=rules.max_entry_slippage_bps,
        stressed_exit_slippage_bps=rules.stressed_exit_slippage_bps,
        binding_constraint=binding_cap,
        setup_type=decision.setup_type,
        catalyst_type=decision.catalyst.catalyst_type,
        catalyst_summary=decision.catalyst.summary,
        catalyst_confidence=decision.catalyst.confidence,
        materiality_score=decision.catalyst.materiality_score,
        evidence_urls=decision.catalyst.evidence_urls,
        evidence_published_at=decision.catalyst.evidence_published_at,
        contract_con_id=snap.contract_con_id,
        primary_exchange=snap.primary_exchange,
        reference_activation_min_price=_round_up_cent(
            snap.previous_close * (1.0 + rules.min_stage_gap_pct / 100.0)
        ),
        reference_activation_max_price=activation_max_price,
        max_reference_gap_pct=rules.max_immediate_gap_pct,
        reference_entry_window_end_et=rules.reference_entry_window_end_et,
        target_session_date=session_date,
        quote_revalidation_required=True,
        halt_revalidation_required=True,
        gap_revalidation_required=True,
        contract_revalidation_required=True,
    )
    return SizingOutcome(preview)
