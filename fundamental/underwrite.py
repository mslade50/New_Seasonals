"""Durable, PM-facing decision records for completed company underwrites."""

from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    ARCHETYPE_VALUATION_METHODS,
    UNDERWRITE_POLICY,
    UNDERWRITE_SCHEMA_VERSION,
)


DECISION_STATUSES = {"QUICK_REVIEW", "WAIT_FOR_PROOF", "WAIT_FOR_EVENT", "PASS"}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text_value(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _date_value(value: Any) -> date | None:
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError):
        return None


def _dated_age_days(older: Any, newer: Any) -> int | None:
    old_date = _date_value(older)
    new_date = _date_value(newer)
    return (new_date - old_date).days if old_date and new_date else None


def _source_ids(value: Any) -> set[str]:
    return {_text_value(item) for item in _sequence(value) if _text_value(item)}


def validate_underwrite_record(
    record: dict[str, Any],
    *,
    decision_as_of: str | date | None = None,
) -> dict[str, Any]:
    """Evaluate the deterministic v2 promotion gates.

    The function deliberately returns every failure instead of stopping at the
    first one.  This makes an incomplete underwrite useful as a research work
    queue while ensuring it cannot be promoted by polished prose.
    """
    errors: list[str] = []
    warnings: list[str] = []
    gates: dict[str, bool] = {}
    version = _text_value(record.get("schema_version"))
    if version != UNDERWRITE_SCHEMA_VERSION:
        return {
            "schema_version": version or "legacy",
            "valid_for_quick_review": False,
            "gates": {"v2_contract": False},
            "errors": ["QUICK_REVIEW requires the fundamental-underwrite.v2 contract"],
            "warnings": [],
            "derived": {},
        }

    required_text = (
        "underwrite_id",
        "ticker",
        "as_of",
        "decision",
        "business_model_lane",
        "idea_archetype",
        "company_thesis_status",
        "security_readiness",
        "verdict",
        "review_request",
    )
    missing_text = [field for field in required_text if not _text_value(record.get(field))]
    gates["required_identity"] = not missing_text
    if missing_text:
        errors.append(f"missing required fields: {', '.join(missing_text)}")

    as_of = decision_as_of or record.get("as_of")
    as_of_date = _date_value(as_of)
    gates["decision_date"] = as_of_date is not None
    if as_of_date is None:
        errors.append("as_of must be a valid date")

    price = _mapping(record.get("price_snapshot"))
    price_value = _finite(price.get("price"))
    price_age = _dated_age_days(price.get("as_of"), as_of)
    price_complete = (
        price_value > 0
        and _finite(price.get("diluted_shares")) > 0
        and np.isfinite(_finite(price.get("net_debt")))
        and np.isfinite(_finite(price.get("enterprise_value")))
        and bool(_text_value(price.get("currency")))
        and bool(_source_ids(price.get("source_ids")))
    )
    gates["current_security_snapshot"] = bool(
        price_complete
        and price_age is not None
        and 0 <= price_age <= UNDERWRITE_POLICY.current_price_max_age_days
    )
    if not price_complete:
        errors.append("price, diluted shares, net debt, enterprise value, currency, and source are required")
    elif price_age is None or price_age < 0 or price_age > UNDERWRITE_POLICY.current_price_max_age_days:
        errors.append("price snapshot is missing, future-dated, or stale")

    variant = _mapping(record.get("variant_hypothesis"))
    causal_chain = [_text_value(item) for item in _sequence(variant.get("causal_chain"))]
    variant_complete = all(
        _text_value(variant.get(field))
        for field in ("market_view", "variant_view", "why_market_wrong", "time_horizon")
    ) and len([item for item in causal_chain if item]) >= UNDERWRITE_POLICY.min_causal_links
    gates["falsifiable_variant"] = variant_complete
    if not variant_complete:
        errors.append("variant hypothesis must distinguish the market view and include a causal chain")

    expectations = _mapping(record.get("expectations"))
    estimate_status = _text_value(expectations.get("estimate_status")).upper()
    expectations_complete = all(
        _text_value(expectations.get(field))
        for field in ("implied_case", "guidance_bridge", "consensus_bridge")
    ) and estimate_status in {"CURRENT", "UNAVAILABLE_EXPLAINED"}
    if estimate_status == "CURRENT":
        estimate_age = _dated_age_days(expectations.get("estimate_snapshot_as_of"), as_of)
        expectations_complete = bool(
            expectations_complete
            and estimate_age is not None
            and 0 <= estimate_age <= UNDERWRITE_POLICY.estimate_snapshot_max_age_days
        )
    gates["expectations_bridge"] = expectations_complete
    if not expectations_complete:
        errors.append("price-implied expectations, guidance, and current or explained consensus are required")

    drivers = _sequence(_mapping(record.get("operating_model")).get("drivers"))
    driver_complete = 2 <= len(drivers) <= 5 and all(
        isinstance(driver, dict)
        and _text_value(driver.get("name"))
        and driver.get("baseline") is not None
        and driver.get("thesis_case") is not None
        and _text_value(driver.get("unit"))
        and _source_ids(driver.get("source_ids"))
        for driver in drivers
    )
    gates["causal_operating_model"] = driver_complete
    if not driver_complete:
        errors.append("operating model requires two to five sourced causal drivers")

    valuation = _mapping(record.get("valuation"))
    archetype = _text_value(record.get("idea_archetype"))
    allowed_methods = set(ARCHETYPE_VALUATION_METHODS.get(archetype, ()))
    primary_method = _text_value(valuation.get("primary_method"))
    secondary_method = _text_value(valuation.get("secondary_method"))
    method_gate = bool(
        allowed_methods
        and primary_method in allowed_methods
        and secondary_method in allowed_methods
        and primary_method != secondary_method
    )
    gates["archetype_valuation_methods"] = method_gate
    if not method_gate:
        errors.append("valuation methods must be two distinct methods allowed for the idea archetype")

    bear = _finite(valuation.get("bear"))
    base = _finite(valuation.get("base"))
    bull = _finite(valuation.get("bull"))
    horizon = _finite(valuation.get("horizon_years"))
    values_ordered = bool(
        price_value > 0
        and bear > 0
        and bear < base < bull
        and horizon > 0
        and _text_value(valuation.get("currency"))
        and _text_value(valuation.get("reverse_expectations"))
        and _source_ids(valuation.get("source_ids"))
    )
    gates["valuation_cases"] = values_ordered
    if not values_ordered:
        errors.append("valuation needs sourced, ordered bear/base/bull cases and reverse expectations")

    base_discount = (base / price_value - 1.0) if values_ordered else float("nan")
    base_cagr = ((base / price_value) ** (1.0 / horizon) - 1.0) if values_ordered else float("nan")
    bear_downside = max((price_value - bear) / price_value, 0.0) if values_ordered else float("nan")
    reward_risk = (
        max(base - price_value, 0.0) / max(price_value - bear, 1e-9)
        if values_ordered else float("nan")
    )
    return_gate = bool(
        values_ordered
        and (
            base_cagr >= UNDERWRITE_POLICY.min_base_case_cagr
            or base_discount >= UNDERWRITE_POLICY.min_discount_to_base_value
        )
        and reward_risk >= UNDERWRITE_POLICY.min_upside_downside_ratio
        and bear_downside <= UNDERWRITE_POLICY.max_bear_case_downside
    )
    gates["valuation_skew"] = return_gate
    if values_ordered and not return_gate:
        errors.append("base return, upside/downside, or bear-case loss does not clear the v2 research hurdle")

    realization = _mapping(record.get("realization"))
    realization_signals = {
        "revisions": _text_value(realization.get("revision_signal")).upper()
        in {"POSITIVE", "REVERSING"},
        "catalyst": realization.get("observable_catalyst") is True,
        "trend": _text_value(realization.get("trend_state")).upper() == "GREEN",
    }
    gates["realization_edge"] = sum(realization_signals.values()) >= 2
    if not gates["realization_edge"]:
        errors.append("at least two of revisions, an observable catalyst, and green trend are required")

    downside = _mapping(record.get("downside"))
    downside_complete = all(
        _text_value(downside.get(field))
        for field in ("mechanism", "financing_and_dilution", "bear_case")
    )
    gates["downside_and_financing"] = downside_complete
    if not downside_complete:
        errors.append("downside mechanism, financing/dilution, and bear case are required")

    triggers = _sequence(record.get("proof_triggers"))
    trigger_complete = len(triggers) >= UNDERWRITE_POLICY.min_proof_triggers and all(
        isinstance(trigger, dict)
        and all(
            trigger.get(field) not in (None, "")
            for field in ("trigger_id", "metric", "comparator", "threshold", "expected_by")
        )
        and _source_ids(trigger.get("source_ids"))
        for trigger in triggers
    )
    gates["observable_proof"] = trigger_complete
    if not trigger_complete:
        errors.append("at least one sourced, dated, measurable proof trigger is required")

    kills = _sequence(record.get("kill_conditions"))
    kill_complete = len(kills) >= UNDERWRITE_POLICY.min_kill_conditions and all(
        isinstance(condition, dict)
        and all(
            condition.get(field) not in (None, "")
            for field in ("condition_id", "metric", "comparator", "threshold", "consequence")
        )
        and _source_ids(condition.get("source_ids"))
        for condition in kills
    )
    gates["measurable_falsifiers"] = kill_complete
    if not kill_complete:
        errors.append("at least two sourced, measurable kill conditions are required")

    sources = [source for source in _sequence(record.get("sources")) if isinstance(source, dict)]
    source_id_list = [_text_value(source.get("source_id")) for source in sources]
    source_ids = {source_id for source_id in source_id_list if source_id}
    primary_sources = [source for source in sources if source.get("primary") is True]
    unique_sources = len(source_ids) == len(source_id_list) and len(source_ids) >= UNDERWRITE_POLICY.min_evidence_items
    current_primary = sum(
        1
        for source in primary_sources
        if (age := _dated_age_days(source.get("as_of"), as_of)) is not None
        and 0 <= age <= UNDERWRITE_POLICY.primary_source_max_age_days
    )
    source_complete = bool(
        unique_sources
        and current_primary >= UNDERWRITE_POLICY.min_primary_sources
        and all(
            _text_value(source.get(field))
            for source in sources
            for field in ("source_id", "label", "url", "source_type", "as_of", "use")
        )
    )
    gates["source_integrity"] = source_complete
    if not source_complete:
        errors.append("sources need unique IDs, complete metadata, and at least two current primary sources")

    evidence = [item for item in _sequence(record.get("evidence_ledger")) if isinstance(item, dict)]
    evidence_ids = {_text_value(item.get("evidence_id")) for item in evidence if _text_value(item.get("evidence_id"))}
    evidence_complete = bool(
        len(evidence) >= UNDERWRITE_POLICY.min_evidence_items
        and len(evidence_ids) == len(evidence)
        and all(
            _text_value(item.get(field))
            for item in evidence
            for field in ("evidence_id", "claim", "direction", "source_id", "materiality")
        )
        and all(_text_value(item.get("source_id")) in source_ids for item in evidence)
    )
    gates["claim_linked_evidence"] = evidence_complete
    if not evidence_complete:
        errors.append("evidence ledger must contain at least three uniquely identified, source-linked claims")

    referenced_source_ids = set()
    referenced_source_ids |= _source_ids(price.get("source_ids"))
    referenced_source_ids |= _source_ids(valuation.get("source_ids"))
    for driver in drivers:
        referenced_source_ids |= _source_ids(_mapping(driver).get("source_ids"))
    for trigger in triggers:
        referenced_source_ids |= _source_ids(_mapping(trigger).get("source_ids"))
    for condition in kills:
        referenced_source_ids |= _source_ids(_mapping(condition).get("source_ids"))
    gates["source_references_resolve"] = referenced_source_ids.issubset(source_ids)
    if not gates["source_references_resolve"]:
        errors.append("one or more model, price, trigger, or falsifier source IDs do not resolve")

    red_team = _mapping(record.get("red_team"))
    unresolved_conflicts = _sequence(red_team.get("unresolved_conflicts"))
    red_team_complete = bool(
        _text_value(red_team.get("strongest_case"))
        and _source_ids(red_team.get("evidence_ids"))
        and _source_ids(red_team.get("evidence_ids")).issubset(evidence_ids)
        and not unresolved_conflicts
    )
    gates["red_team_clear"] = red_team_complete
    if not red_team_complete:
        errors.append("strongest opposing case must be sourced and decision-critical conflicts resolved")

    missing = [item for item in _sequence(record.get("missing_evidence")) if isinstance(item, dict)]
    blocking_missing = [item for item in missing if item.get("blocker") is True]
    gates["no_blocking_missing_evidence"] = not blocking_missing
    if blocking_missing:
        errors.append("decision-blocking evidence is still missing")

    next_review = _mapping(record.get("next_review"))
    gates["next_review_defined"] = bool(
        _text_value(next_review.get("reason")) and _text_value(next_review.get("date_or_trigger"))
    )
    if not gates["next_review_defined"]:
        errors.append("next review must name a reason and date or observable trigger")

    state_gate = bool(
        _text_value(record.get("security_readiness")).upper() == "REVIEW_READY"
        and _text_value(record.get("company_thesis_status")).upper()
        in {"STRENGTHENING", "INTACT", "WATCH"}
        and record.get("live_actions_enabled") is False
    )
    gates["state_and_safety"] = state_gate
    if not state_gate:
        errors.append("review-ready state, a tested company thesis, and live_actions_enabled=false are required")

    if _text_value(record.get("decision")).upper() != "QUICK_REVIEW":
        warnings.append("record is an in-progress or pass decision; promotion gates are diagnostic")

    valid_for_quick = bool(
        _text_value(record.get("decision")).upper() == "QUICK_REVIEW"
        and all(gates.values())
        and not errors
    )
    return {
        "schema_version": version,
        "valid_for_quick_review": valid_for_quick,
        "gates": gates,
        "errors": errors,
        "warnings": warnings,
        "derived": {
            "price_age_days": price_age,
            "base_discount_pct": round(base_discount * 100.0, 1) if np.isfinite(base_discount) else None,
            "base_case_cagr_pct": round(base_cagr * 100.0, 1) if np.isfinite(base_cagr) else None,
            "bear_downside_pct": round(bear_downside * 100.0, 1) if np.isfinite(bear_downside) else None,
            "upside_downside_ratio": round(reward_risk, 2) if np.isfinite(reward_risk) else None,
            "realization_signals": realization_signals,
        },
    }


def _trigger_text(trigger: dict[str, Any]) -> str:
    unit = f" {trigger.get('unit')}" if trigger.get("unit") else ""
    return (
        f"{_text_value(trigger.get('metric'))} {_text_value(trigger.get('comparator'))} "
        f"{_text_value(trigger.get('threshold'))}{unit} by {_text_value(trigger.get('expected_by'))}."
    ).strip()


def _kill_text(condition: dict[str, Any]) -> str:
    unit = f" {condition.get('unit')}" if condition.get("unit") else ""
    return (
        f"{_text_value(condition.get('metric'))} {_text_value(condition.get('comparator'))} "
        f"{_text_value(condition.get('threshold'))}{unit}: {_text_value(condition.get('consequence'))}"
    ).strip()


def normalize_underwrite_record(record: dict[str, Any]) -> dict[str, Any]:
    """Add the flat display fields used by the existing report and site adapters."""
    if _text_value(record.get("schema_version")) != UNDERWRITE_SCHEMA_VERSION:
        return record
    price = _mapping(record.get("price_snapshot"))
    variant = _mapping(record.get("variant_hypothesis"))
    expectations = _mapping(record.get("expectations"))
    valuation = _mapping(record.get("valuation"))
    realization = _mapping(record.get("realization"))
    downside = _mapping(record.get("downside"))
    next_review = _mapping(record.get("next_review"))
    currency = _text_value(price.get("currency"))
    range_text = (
        f"{valuation.get('primary_method')} with {valuation.get('secondary_method')}; "
        f"bear/base/bull {currency} {valuation.get('bear')}/{valuation.get('base')}/{valuation.get('bull')} "
        f"over {valuation.get('horizon_years')} years. {_text_value(valuation.get('reverse_expectations'))}"
    )
    sources = [
        {
            "source_id": source.get("source_id"),
            "label": source.get("label"),
            "url": source.get("url"),
            "as_of": source.get("as_of"),
            "use": source.get("use"),
        }
        for source in _sequence(record.get("sources"))
        if isinstance(source, dict)
    ]
    normalized = {
        **record,
        "mispricing": _text_value(variant.get("variant_view")),
        "priced_in": _text_value(expectations.get("implied_case")),
        "valuation_summary": range_text,
        "trend_summary": (
            f"{_text_value(realization.get('trend_state'))} trend; revision signal "
            f"{_text_value(realization.get('revision_signal'))}."
        ),
        "price_as_of": f"{price.get('as_of')} close {currency} {price.get('price')}",
        "proof_required": [
            _trigger_text(trigger)
            for trigger in _sequence(record.get("proof_triggers"))
            if isinstance(trigger, dict)
        ],
        "kill_condition_summaries": [
            _kill_text(condition)
            for condition in _sequence(record.get("kill_conditions"))
            if isinstance(condition, dict)
        ],
        "next_review_summary": (
            f"{_text_value(next_review.get('reason'))} - {_text_value(next_review.get('date_or_trigger'))}"
        ).strip(" -"),
        "downside_summary": _text_value(downside.get("mechanism")),
        "source_rows": sources,
    }
    return normalized


def is_surfaceable_quick_review(
    record: dict[str, Any], *, decision_as_of: str | date | None = None
) -> bool:
    return validate_underwrite_record(record, decision_as_of=decision_as_of)[
        "valid_for_quick_review"
    ]


def load_underwrite_decisions(path: Path) -> list[dict[str, Any]]:
    """Load and validate the latest human/agent-authored underwriting decisions."""
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("decisions", payload if isinstance(payload, list) else [])
    if not isinstance(records, list):
        raise ValueError("underwrite decisions must be a list or a {'decisions': [...]} object")
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        ticker = str(record.get("ticker") or "").upper().strip()
        status = str(record.get("decision") or "").upper().strip()
        if not ticker or status not in DECISION_STATUSES:
            raise ValueError(f"invalid underwrite decision: ticker={ticker!r}, decision={status!r}")
        if ticker in seen:
            raise ValueError(f"duplicate underwrite decision for {ticker}")
        seen.add(ticker)
        normalized = {**record, "ticker": ticker, "decision": status}
        validation = validate_underwrite_record(
            normalized,
            decision_as_of=payload.get("as_of") if isinstance(payload, dict) else None,
        )
        if status == "QUICK_REVIEW" and not validation["valid_for_quick_review"]:
            detail = "; ".join(validation["errors"][:6])
            raise ValueError(f"{ticker} QUICK_REVIEW failed v2 promotion gates: {detail}")
        normalized = normalize_underwrite_record(normalized)
        normalized["_validation"] = validation
        cleaned.append(normalized)
    return cleaned


def _esc(value: Any) -> str:
    return html.escape("—" if value is None else str(value))


def _items(values: list[str] | None) -> str:
    return "".join(f"<li>{_esc(value)}</li>" for value in (values or []))


def render_underwrite(record: dict[str, Any], candidate: pd.Series, output_path: Path) -> Path:
    """Render one source-backed decision page; it intentionally has no execution path."""
    record = normalize_underwrite_record(record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status = str(record["decision"])
    status_label = status.replace("_", " ")
    source_rows = "".join(
        "<tr>"
        f"<td><a href='{_esc(source.get('url'))}'>{_esc(source.get('label'))}</a></td>"
        f"<td>{_esc(source.get('as_of'))}</td><td>{_esc(source.get('use'))}</td>"
        "</tr>"
        for source in (record.get("source_rows") or record.get("sources") or [])
    )
    validation = record.get("_validation") or validate_underwrite_record(record)
    gate_rows = "".join(
        f"<tr><td>{_esc(str(name).replace('_', ' ').title())}</td>"
        f"<td>{'PASS' if passed else 'BLOCK'}</td></tr>"
        for name, passed in (validation.get("gates") or {}).items()
    )
    driver_rows = "".join(
        f"<tr><td>{_esc(driver.get('name'))}</td><td>{_esc(driver.get('baseline'))} {_esc(driver.get('unit'))}</td>"
        f"<td>{_esc(driver.get('thesis_case'))} {_esc(driver.get('unit'))}</td></tr>"
        for driver in _sequence(_mapping(record.get("operating_model")).get("drivers"))
        if isinstance(driver, dict)
    )
    red_team = _mapping(record.get("red_team"))
    v2_sections = ""
    if _text_value(record.get("schema_version")) == UNDERWRITE_SCHEMA_VERSION:
        v2_sections = f"""
<section class="grid">
<div class="panel"><h2>Causal operating model</h2><table><thead><tr><th>Driver</th><th>Baseline</th><th>Thesis case</th></tr></thead><tbody>{driver_rows}</tbody></table></div>
<div class="panel"><h2>Strongest opposing case</h2><p>{_esc(red_team.get('strongest_case'))}</p></div>
</section>
<section class="panel"><h2>Deterministic promotion gates</h2><table><thead><tr><th>Gate</th><th>Result</th></tr></thead><tbody>{gate_rows}</tbody></table></section>
"""
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(record['ticker'])} — Fundamental underwrite</title>
<style>
:root{{--bg:#0a1020;--panel:#121b2e;--panel2:#18243b;--text:#ecf1fa;--muted:#9ba9bd;--line:#293955;--blue:#78baff;--green:#67dca8;--amber:#ffd070;--red:#ff9a9a}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(180deg,#09101d,#0d1628 45%,#0a1020);color:var(--text);font:15px/1.55 Inter,Segoe UI,Arial,sans-serif}}main{{max-width:1050px;margin:auto;padding:34px 26px}}a{{color:var(--blue)}}
.eyebrow{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.7px}}h1{{font-size:36px;line-height:1.1;margin:6px 0}}h2{{font-size:19px;margin:27px 0 10px}}p{{margin:6px 0}}.muted{{color:var(--muted)}}
.warning,.decision,.panel{{border-radius:12px;padding:16px 18px}}.warning{{background:#302817;border:1px solid #8b6b2c;color:#ffdc8a;margin:16px 0}}.decision{{background:linear-gradient(135deg,#16243b,#111a2d);border:1px solid #365172}}.decision b{{display:block;font-size:23px;margin:2px 0 7px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.panel{{background:rgba(18,27,46,.96);border:1px solid var(--line)}}.panel h2{{margin-top:0}}ul{{margin:5px 0;padding-left:20px}}li{{margin:6px 0}}table{{border-collapse:collapse;width:100%}}th{{text-align:left;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.4px;padding:8px;border-bottom:1px solid var(--line)}}td{{padding:9px 8px;border-bottom:1px solid #22304a;vertical-align:top}}footer{{color:var(--muted);font-size:12px;margin-top:28px}}
@media(max-width:720px){{main{{padding:22px 15px}}h1{{font-size:30px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="eyebrow">Completed fundamental underwrite · {_esc(record.get('as_of'))}</div>
<h1>{_esc(record['ticker'])} <span class="muted">{_esc(candidate.get('company_name'))}</span></h1>
<div class="muted">Screen score {_esc(candidate.get('research_score'))} · {_esc(candidate.get('trend_state'))} trend · price snapshot {_esc(record.get('price_as_of'))}</div>
<div class="warning">Research only. No allocation, position, order, or broker action is generated here.</div>
<section class="decision"><div class="eyebrow">Decision</div><b>{_esc(status_label)}</b><p>{_esc(record.get('verdict'))}</p></section>

<section class="grid">
<div class="panel"><h2>What could be mispriced</h2><p>{_esc(record.get('mispricing'))}</p></div>
<div class="panel"><h2>What appears priced in</h2><p>{_esc(record.get('priced_in'))}</p></div>
<div class="panel"><h2>Valuation read</h2><p>{_esc(record.get('valuation_summary') or record.get('valuation'))}</p></div>
<div class="panel"><h2>Trend and timing</h2><p>{_esc(record.get('trend_summary') or record.get('trend'))}</p></div>
</section>

<section class="grid">
<div class="panel"><h2>Proof required</h2><ul>{_items(record.get('proof_required'))}</ul></div>
<div class="panel"><h2>Downside and kill conditions</h2><ul>{_items(record.get('kill_condition_summaries') or record.get('kill_conditions'))}</ul></div>
</section>

<section class="panel"><h2>Next review</h2><p>{_esc(record.get('next_review_summary') or record.get('next_review'))}</p></section>
{v2_sections}
<h2>Primary-source ledger</h2><section class="panel"><table><thead><tr><th>Source</th><th>As of</th><th>Used for</th></tr></thead><tbody>{source_rows}</tbody></table></section>
<footer><a href="../fundamental_daily.html">← Back to the daily brief</a> · Underwrite state: {_esc(status_label)} · Live actions disabled.</footer>
</main></body></html>"""
    output_path.write_text(page, encoding="utf-8")
    return output_path


def build_underwrite_pack(
    decisions: list[dict[str, Any]],
    candidates: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    """Render current decisions and return report-relative links by ticker."""
    links: dict[str, str] = {}
    if not decisions or candidates.empty:
        return links
    indexed = candidates.drop_duplicates("ticker").set_index("ticker")
    for record in decisions:
        validation = validate_underwrite_record(record)
        if str(record.get("decision") or "").upper() == "QUICK_REVIEW" and not validation["valid_for_quick_review"]:
            raise ValueError(f"{record.get('ticker')} QUICK_REVIEW cannot render before v2 gates pass")
        record = normalize_underwrite_record(record)
        record["_validation"] = validation
        ticker = record["ticker"]
        if ticker not in indexed.index:
            continue
        path = render_underwrite(record, indexed.loc[ticker], output_dir / f"{ticker}.html")
        links[ticker] = f"underwrites/{path.name}"
    return links
