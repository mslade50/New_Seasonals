"""Deterministic triage from external source records to hypothesis cards."""

from __future__ import annotations

import re
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from research.experiment_registry import content_digest, stable_id
from research.idea_miner.models import SourceRecord, dedupe_sources


ARCHETYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "intraday": ("intraday", "opening", "overnight", "vwap", "minute", "hour", "auction"),
    "trend": ("trend", "momentum", "breakout", "moving average", "time-series"),
    "event": ("earnings", "filing", "announcement", "event", "rebalance", "index inclusion"),
    "fundamental": ("valuation", "revenue", "margin", "cash flow", "estimate", "fundamental"),
    "portfolio": ("portfolio", "correlation", "volatility", "risk", "diversification", "factor"),
}


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _first_sentence(text: str, limit: int = 420) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    match = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
    return match[:limit].rstrip()


def classify_archetype(record: SourceRecord) -> str:
    haystack = _normalize(" ".join((record.title, record.text, " ".join(record.tags))))
    matches = {
        archetype: sum(1 for term in terms if _normalize(term) in haystack)
        for archetype, terms in ARCHETYPE_KEYWORDS.items()
    }
    best = max(matches, key=lambda key: (matches[key], key))
    return best if matches[best] else "other"


def hypothesis_fingerprint(
    claim: str,
    mechanism: str,
    instruments: Iterable[str],
    horizon: str,
) -> str:
    return content_digest(
        {
            "claim": _normalize(claim),
            "mechanism": _normalize(mechanism),
            "instruments": sorted(_normalize(value) for value in instruments),
            "horizon": _normalize(horizon),
        }
    )


@dataclass(frozen=True)
class HypothesisCard:
    hypothesis_id: str
    hypothesis_fingerprint: str
    source_id: str
    source_type: str
    source_url: str
    source_title: str
    archetype: str
    claim: str
    mechanism: str
    instruments: tuple[str, ...]
    horizon: str
    falsifiable_test: str
    data_requirements: tuple[str, ...]
    actionability: str
    variant_wedge: str
    why_now: str
    first_rejection: str
    what_would_make_it_researchable: str
    what_would_kill_it: str
    next_workflow: str
    status: str
    duplicate_prior: bool
    priority_components: Mapping[str, int]
    research_priority_score: int
    research_only: bool = True
    no_order: bool = True

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["instruments"] = list(self.instruments)
        out["data_requirements"] = list(self.data_requirements)
        out["priority_components"] = dict(self.priority_components)
        return out

    def registry_record(self, as_of: str) -> dict[str, Any]:
        return {
            "kind": "hypothesis",
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_fingerprint": self.hypothesis_fingerprint,
            "source_ids": [self.source_id],
            "source_type": self.source_type,
            "source_url": self.source_url,
            "archetype": self.archetype,
            "claim": self.claim,
            "mechanism": self.mechanism,
            "instruments": list(self.instruments),
            "horizon": self.horizon,
            "falsifiable_test": self.falsifiable_test,
            "data_requirements": list(self.data_requirements),
            "actionability": self.actionability,
            "variant_wedge": self.variant_wedge,
            "why_now": self.why_now,
            "first_rejection": self.first_rejection,
            "what_would_kill_it": self.what_would_kill_it,
            "next_workflow": self.next_workflow,
            "status": self.status,
            "as_of": as_of,
            "research_priority_score": self.research_priority_score,
            "research_only": True,
            "no_order": True,
        }


def build_card(record: SourceRecord, prior_fingerprints: set[str]) -> HypothesisCard:
    claim = record.claim or _first_sentence(record.text) or record.title
    mechanism_supplied = bool(record.mechanism)
    mechanism = record.mechanism or "Unresolved: the source has not established a causal mechanism."
    instruments_supplied = bool(record.instruments)
    instruments = record.instruments or ("UNRESOLVED",)
    horizon_supplied = bool(record.horizon)
    horizon = record.horizon or "UNRESOLVED"
    test_supplied = bool(record.test_idea)
    falsifiable_test = record.test_idea or (
        "Translate the claim into a point-in-time signal, fixed decision time, "
        "fixed cost model, and held-out test before running variants."
    )
    data_requirements = record.data_requirements or ("point-in-time prices and eligibility",)
    first_rejection = record.first_rejection or (
        "Reject at intake if the claim cannot be measured with decision-available data "
        "or the mechanism cannot be stated independently of the backtest result."
    )
    fingerprint = hypothesis_fingerprint(claim, mechanism, instruments, horizon)
    duplicate_prior = fingerprint in prior_fingerprints

    components = {
        "claim_clarity": 2 if record.claim else (1 if claim else 0),
        "mechanism_clarity": 2 if mechanism_supplied else 0,
        "test_specificity": 2 if test_supplied else 0,
        "instrument_horizon_definition": int(instruments_supplied) + int(horizon_supplied),
        "novelty": 0 if duplicate_prior else 2,
    }
    score = sum(components.values())

    missing: list[str] = []
    if not mechanism_supplied:
        missing.append("mechanism")
    if not instruments_supplied:
        missing.append("instruments")
    if not horizon_supplied:
        missing.append("horizon")
    if not test_supplied:
        missing.append("test specification")
    if duplicate_prior:
        status = "duplicate_prior_hypothesis"
        next_workflow = "registry_review"
    elif missing:
        status = "needs_source_diligence"
        next_workflow = "source_diligence"
    else:
        status = "advance_to_preregistration"
        next_workflow = "preregistered_quant_test"

    return HypothesisCard(
        hypothesis_id=stable_id("hyp", {"fingerprint": fingerprint, "source": record.source_id}),
        hypothesis_fingerprint=fingerprint,
        source_id=record.source_id,
        source_type=record.source_type,
        source_url=record.url,
        source_title=record.title,
        archetype=classify_archetype(record),
        claim=claim,
        mechanism=mechanism,
        instruments=instruments,
        horizon=horizon,
        falsifiable_test=falsifiable_test,
        data_requirements=data_requirements,
        actionability=(
            "Research only: register the bounded test; no position or execution action."
            if status == "advance_to_preregistration"
            else "Research only: resolve source gaps before any test."
        ),
        variant_wedge=(
            record.variant_wedge
            or "Unresolved: the source claim has not yet been separated from what is already known or priced."
        ),
        why_now=(
            record.why_now
            or "New to this intake window; no market catalyst or current mispricing is inferred."
        ),
        first_rejection=first_rejection,
        what_would_make_it_researchable=(
            "Resolve " + ", ".join(missing) + "." if missing and not duplicate_prior
            else "Register a bounded trial grid, point-in-time universe, costs, and holdout gates."
        ),
        what_would_kill_it=(record.what_would_kill or (
            "Independent holdout expectancy is non-positive after stressed costs, "
            "or the effect disappears after day/event clustering."
        )),
        next_workflow=next_workflow,
        status=status,
        duplicate_prior=duplicate_prior,
        priority_components=components,
        research_priority_score=score,
    )


def _round_robin_select(
    cards: Iterable[HypothesisCard],
    limit: int,
    max_per_archetype: int,
) -> list[HypothesisCard]:
    lanes: dict[str, deque[HypothesisCard]] = defaultdict(deque)
    for card in sorted(
        cards,
        key=lambda item: (-item.research_priority_score, item.source_type, item.hypothesis_id),
    ):
        if card.duplicate_prior:
            continue
        lanes[card.archetype].append(card)
    ordered_lanes = sorted(lanes)
    selected: list[HypothesisCard] = []
    lane_counts: Counter[str] = Counter()
    while len(selected) < limit and ordered_lanes:
        progressed = False
        for lane in list(ordered_lanes):
            queue = lanes[lane]
            if not queue or lane_counts[lane] >= max_per_archetype:
                ordered_lanes.remove(lane)
                continue
            selected.append(queue.popleft())
            lane_counts[lane] += 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def build_weekly_queue(
    records: Iterable[SourceRecord],
    *,
    as_of: str,
    prior_fingerprints: set[str] | None = None,
    max_candidates: int = 5,
    max_per_archetype: int = 2,
) -> dict[str, Any]:
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if max_per_archetype <= 0:
        raise ValueError("max_per_archetype must be positive")
    unique, duplicate_sources = dedupe_sources(records)
    cards = [build_card(record, prior_fingerprints or set()) for record in unique]
    selected = _round_robin_select(cards, max_candidates, max_per_archetype)
    status_counts = Counter(card.status for card in cards)
    archetype_counts = Counter(card.archetype for card in cards)
    source_type_counts = Counter(record.source_type for record in unique)
    selected_ids = {card.hypothesis_id for card in selected}

    return {
        "schema_version": "weekly-hypothesis-inbox.v1",
        "as_of": as_of,
        "research_only": True,
        "no_order": True,
        "coverage": {
            "source_rows_received": len(unique) + duplicate_sources,
            "unique_sources": len(unique),
            "duplicate_sources": duplicate_sources,
            "source_types": dict(sorted(source_type_counts.items())),
        },
        "funnel": {
            "hypotheses_created": len(cards),
            "selected_for_review": len(selected),
            "duplicate_prior": status_counts.get("duplicate_prior_hypothesis", 0),
            "ready_for_preregistration": status_counts.get("advance_to_preregistration", 0),
            "needs_source_diligence": status_counts.get("needs_source_diligence", 0),
            "by_archetype": dict(sorted(archetype_counts.items())),
        },
        "selected": [card.as_dict() for card in selected],
        "all_hypotheses": [
            {**card.as_dict(), "selected_for_review": card.hypothesis_id in selected_ids}
            for card in sorted(cards, key=lambda item: item.hypothesis_id)
        ],
        "sources": [record.as_dict() for record in unique],
        "methodology": {
            "selection": "bounded archetype round-robin; score allocates research attention only",
            "max_candidates": max_candidates,
            "max_per_archetype": max_per_archetype,
            "network_fetch": False,
            "promotion": "none; preregistration and independent validation required",
        },
    }


def registry_records_for_queue(queue: Mapping[str, Any]) -> list[dict[str, Any]]:
    as_of = str(queue["as_of"])
    source_rows = [SourceRecord.from_mapping(row) for row in queue.get("sources", [])]
    source_records = [row.registry_record() for row in source_rows]
    hypotheses: list[dict[str, Any]] = []
    for card in queue.get("all_hypotheses", []):
        hypotheses.append(
            {
                "kind": "hypothesis",
                "hypothesis_id": card["hypothesis_id"],
                "hypothesis_fingerprint": card["hypothesis_fingerprint"],
                "source_ids": [card["source_id"]],
                "source_type": card["source_type"],
                "source_url": card["source_url"],
                "archetype": card["archetype"],
                "claim": card["claim"],
                "mechanism": card["mechanism"],
                "instruments": list(card["instruments"]),
                "horizon": card["horizon"],
                "falsifiable_test": card["falsifiable_test"],
                "data_requirements": list(card["data_requirements"]),
                "actionability": card["actionability"],
                "variant_wedge": card["variant_wedge"],
                "why_now": card["why_now"],
                "first_rejection": card["first_rejection"],
                "what_would_kill_it": card["what_would_kill_it"],
                "next_workflow": card["next_workflow"],
                "status": card["status"],
                "as_of": as_of,
                "research_priority_score": card["research_priority_score"],
                "research_only": True,
                "no_order": True,
            }
        )
    return source_records + hypotheses
