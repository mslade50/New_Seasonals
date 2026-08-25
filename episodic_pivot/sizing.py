"""Liquidity-aware sizing for non-executable EP order previews."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date

from .config import EPPolicy
from .schema import Candidate, QualificationDecision, StagingPreview


@dataclass(frozen=True)
class SizingOutcome:
    preview: StagingPreview | None
    blockers: tuple[str, ...] = ()


def apply_daily_preview_caps(
    previews: list[StagingPreview], *, policy: EPPolicy
) -> list[StagingPreview]:
    """Pro-rata shadow previews to portfolio-level daily risk/notional caps."""

    if not previews:
        return []
    rules = policy.execution
    total_risk = sum(item.risk_dollars for item in previews)
    total_notional = sum(item.notional for item in previews)
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

    adjusted: list[StagingPreview] = []
    for item in previews:
        quantity = math.floor(item.quantity * factor)
        if quantity < 1:
            continue
        risk_dollars = quantity * item.risk_per_share
        adjusted.append(
            replace(
                item,
                quantity=quantity,
                risk_dollars=round(risk_dollars, 2),
                risk_bps=round(10_000.0 * risk_dollars / rules.account_value, 4),
                notional=round(quantity * item.entry_limit, 2),
                binding_cap=f"{item.binding_cap}+{binding}",
            )
        )
    return adjusted


def _round_up_cent(value: float) -> float:
    return math.ceil(value * 100.0 - 1e-9) / 100.0


def _round_down_cent(value: float) -> float:
    return math.floor(value * 100.0 + 1e-9) / 100.0


def build_staging_preview(
    candidate: Candidate,
    decision: QualificationDecision,
    *,
    policy: EPPolicy,
    execute_on: date | str,
) -> SizingOutcome:
    if decision.decision != "STAGEABLE":
        return SizingOutcome(None, ("DECISION_NOT_STAGEABLE",))

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
    execute_date = execute_on.isoformat() if isinstance(execute_on, date) else str(execute_on)
    preview = StagingPreview(
        candidate_id=candidate.candidate_id,
        symbol=snap.symbol,
        action="BUY",
        quantity=quantity,
        order_type=rules.order_type,
        tif=rules.tif,
        regular_hours_only=rules.regular_hours_only,
        entry_limit=entry_limit,
        initial_stop=stop,
        risk_per_share=round(modeled_risk_per_share, 4),
        risk_dollars=round(risk_dollars, 2),
        risk_bps=round(10_000.0 * risk_dollars / rules.account_value, 4),
        notional=round(quantity * entry_limit, 2),
        expected_entry_slippage_bps=round(expected_slippage_bps, 2),
        max_entry_slippage_bps=rules.max_entry_slippage_bps,
        stressed_exit_slippage_bps=rules.stressed_exit_slippage_bps,
        binding_cap=binding_cap,
        setup_type=decision.setup_type,
        catalyst_type=decision.catalyst.catalyst_type,
        catalyst_summary=decision.catalyst.summary,
        catalyst_confidence=decision.catalyst.confidence,
        materiality_score=decision.catalyst.materiality_score,
        evidence_urls=decision.catalyst.evidence_urls,
        evidence_published_at=decision.catalyst.evidence_published_at,
        contract_con_id=snap.contract_con_id,
        primary_exchange=snap.primary_exchange,
        activation_min_price=_round_up_cent(
            snap.previous_close * (1.0 + rules.min_stage_gap_pct / 100.0)
        ),
        activation_max_price=activation_max_price,
        max_opening_gap_pct=rules.max_immediate_gap_pct,
        entry_window_end_et="09:35:00",
        requires_fresh_quote_at_release=True,
        requires_halt_recheck_at_release=True,
        requires_opening_gap_recheck=True,
        requires_contract_recheck_at_release=True,
        execute_on=execute_date,
    )
    return SizingOutcome(preview)
