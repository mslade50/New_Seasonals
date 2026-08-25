"""End-to-end shadow EP workflow orchestration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date, datetime

from .config import EPPolicy
from .news import (
    ArticleFetcher,
    SearchProvider,
    assess_catalyst,
    normalize_evidence_document,
)
from .premarket import nominate_candidates
from .qualify import qualify_candidate
from .schema import (
    NewsDocument,
    PremarketSnapshot,
    RunResult,
    iso_utc,
    parse_timestamp,
    utc_now,
)
from .sizing import apply_daily_preview_caps, build_research_sizing_preview


def _run_id(
    snapshots: list[PremarketSnapshot],
    *,
    as_of: str | datetime,
    policy: EPPolicy,
    documents_by_candidate: dict[str, list[NewsDocument]],
) -> str:
    normalized_inputs = json.dumps(
        {
            "snapshots": [
                snapshot.to_dict()
                for snapshot in sorted(snapshots, key=lambda x: x.symbol)
            ],
            "documents": {
                candidate_id: [document.to_dict() for document in documents]
                for candidate_id, documents in sorted(documents_by_candidate.items())
            },
            "policy": policy.to_dict(),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    seed = f"{iso_utc(as_of)}|{normalized_inputs}".encode("utf-8")
    day = parse_timestamp(as_of).date().isoformat()
    return f"EP-RUN-{day}-{hashlib.sha256(seed).hexdigest()[:12]}"


def run_shadow_pipeline(
    snapshots: list[PremarketSnapshot],
    *,
    as_of: str | datetime,
    target_session_date: date | str,
    policy: EPPolicy,
    offline_documents: dict[str, list[NewsDocument]] | None = None,
    offline_documents_verified: bool = False,
    search_provider: SearchProvider | None = None,
    article_fetcher: ArticleFetcher | None = None,
) -> RunResult:
    """Run the research pipeline without any external write or broker action."""

    if policy.live_actions_enabled:
        raise ValueError("live actions are forbidden in the shadow pipeline")
    if search_provider is not None and offline_documents:
        raise ValueError("choose offline evidence or a search provider, not both")

    scan_at = iso_utc(as_of)
    initial_candidates = nominate_candidates(snapshots, as_of=scan_at, policy=policy)
    fetcher = article_fetcher or ArticleFetcher()
    documents_by_candidate: dict[str, list[NewsDocument]] = {}

    for candidate in initial_candidates:
        symbol = candidate.snapshot.symbol
        if offline_documents is not None:
            documents = []
            for document in offline_documents.get(symbol, []):
                normalized = normalize_evidence_document(document)
                if not offline_documents_verified:
                    normalized = replace(
                        normalized,
                        fetch_status="UNVERIFIED_REPLAY",
                    )
                documents.append(normalized)
        elif search_provider is not None:
            try:
                hits = search_provider.search(
                    symbol=symbol,
                    company_name=candidate.snapshot.company_name,
                    as_of=parse_timestamp(scan_at),
                    limit=policy.news.max_documents_per_candidate,
                )
                documents = [
                    normalize_evidence_document(fetcher.fetch(hit))
                    for hit in hits[: policy.news.max_documents_per_candidate]
                ]
            except Exception as exc:
                documents = [
                    NewsDocument(
                        title="",
                        url="",
                        canonical_url="",
                        publisher=getattr(search_provider, "name", "SEARCH"),
                        published_at=None,
                        retrieved_at=iso_utc(utc_now()),
                        text_excerpt="",
                        text_sha256="",
                        source_tier="SEARCH",
                        fetch_status=f"SEARCH_FAILED:{type(exc).__name__}",
                    )
                ]
        else:
            documents = []

        documents_by_candidate[candidate.candidate_id] = documents

    # Network research takes time.  The actual post-research decision timestamp
    # is used to recheck quote freshness; a slow run correctly produces WATCH
    # decisions until the user captures a fresh snapshot and replays the evidence.
    decision_at = iso_utc(utc_now()) if search_provider is not None else scan_at
    candidates = nominate_candidates(snapshots, as_of=decision_at, policy=policy)
    decisions = []
    previews = []
    for candidate in candidates:
        symbol = candidate.snapshot.symbol
        documents = documents_by_candidate.get(candidate.candidate_id, [])
        catalyst = assess_catalyst(
            documents,
            decision_at=decision_at,
            policy=policy.news,
            symbol=symbol,
            company_name=candidate.snapshot.company_name,
            first_trigger_at=(
                candidate.snapshot.first_trigger_at
                or (
                    candidate.snapshot.observed_at
                    if candidate.snapshot.source.upper().startswith("IBKR")
                    else None
                )
            ),
            target_session_date=target_session_date,
        )
        decision = qualify_candidate(
            candidate,
            catalyst,
            policy=policy,
            decision_at=decision_at,
            target_session_date=target_session_date,
        )
        outcome = build_research_sizing_preview(
            candidate,
            decision,
            policy=policy,
            target_session_date=target_session_date,
        )
        if outcome.preview is not None:
            previews.append(outcome.preview)
        elif decision.decision == "RESEARCH_PREVIEW_ELIGIBLE":
            decision = replace(
                decision,
                decision="WATCH",
                blockers=tuple(
                    sorted(
                        set(decision.blockers)
                        | {f"SIZING:{blocker}" for blocker in outcome.blockers}
                    )
                ),
            )
        decisions.append(decision)

    pre_cap_candidate_ids = {preview.candidate_id for preview in previews}
    previews = apply_daily_preview_caps(previews, policy=policy)
    kept_candidate_ids = {preview.candidate_id for preview in previews}
    dropped_by_daily_cap = pre_cap_candidate_ids - kept_candidate_ids
    if dropped_by_daily_cap:
        decisions = [
            replace(
                decision,
                decision="WATCH",
                blockers=tuple(
                    sorted(set(decision.blockers) | {"SIZING:DAILY_CAP_ZERO_QUANTITY"})
                ),
            )
            if decision.candidate_id in dropped_by_daily_cap
            else decision
            for decision in decisions
        ]

    return RunResult(
        run_id=_run_id(
            snapshots,
            as_of=decision_at,
            policy=policy,
            documents_by_candidate=documents_by_candidate,
        ),
        generated_at=decision_at,
        candidates=candidates,
        documents_by_candidate=documents_by_candidate,
        decisions=decisions,
        previews=previews,
    )
