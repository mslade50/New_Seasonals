"""Auditable agent search-and-read research, separate from execution approval.

The agent owns semantic judgment and source attribution. These checks establish
record integrity, time/issuer binding and tape eligibility, NOT independent proof
that the agent read a page correctly. Hashes must never be described as such proof.
No network, broker, email or execution actions live in this module.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
from dataclasses import replace
from datetime import date, datetime, time
from urllib.parse import parse_qs, urlencode, urlsplit
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from .config import EPPolicy
from .pipeline import _research_prior_atr_blocker
from .premarket import nominate_candidates
from .schema import Candidate, CatalystAssessment, RunResult, iso_utc, parse_timestamp

MODE = "AGENT_GOOGLE_SEARCH_AND_READ"
PACKET_TYPE = "EP_SEARCH_READ_REVIEW_V2"
QUEUE_TYPE = "EP_SEARCH_READ_QUEUE_V2"
_NY = ZoneInfo("America/New_York")
_STATUSES = {"QUALIFIED", "REJECTED", "NO_VERIFIED_CATALYST", "UNRESOLVED"}
_KINDS = {
    "EARNINGS",
    "EARNINGS_GUIDANCE",
    "REGULATORY_APPROVAL",
    "CLINICAL_DATA",
    "MATERIAL_CONTRACT",
    "PRODUCT_TECHNOLOGY",
    "OTHER_MATERIAL_BUSINESS_EVENT",
}


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def _text(record: dict, key: str, minimum: int = 1) -> str:
    value = record.get(key)
    if not isinstance(value, str) or len(value.strip()) < minimum:
        raise ValueError(f"review record requires {key}")
    return value.strip()


def _https(value: str) -> str:
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or not host
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or "." not in host
        or host.endswith((".local", ".internal", ".localhost"))
    ):
        raise ValueError("review source must be a public HTTPS article URL")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return host
    raise ValueError("review source cannot use a literal IP address")


def _within(value: str, start: datetime, end: datetime) -> datetime:
    stamp = parse_timestamp(value)
    if not start <= stamp <= end:
        raise ValueError("review timestamp is outside the session/review window")
    return stamp


def _event_within(value: str, start: datetime, opened: datetime) -> None:
    # An article visibly dated this morning and read before the open needs no
    # invented clock time. A prior-day date alone cannot prove after-close news.
    if len(value) == 10:
        day = date.fromisoformat(value)
        lower = datetime.combine(day, time(), tzinfo=_NY)
        if day != opened.astimezone(_NY).date() or lower < start:
            raise ValueError("date-only evidence cannot establish the event window")
    else:
        _within(value, start, opened)


def _window(session: str, decision_at: str) -> tuple[datetime, datetime]:
    day = date.fromisoformat(session)
    end = parse_timestamp(decision_at)
    if end.astimezone(_NY).date() != day or end.astimezone(_NY).time() >= time(9, 30):
        raise ValueError("search-and-read decision must be in the target premarket")
    calendar = xcals.get_calendar("XNYS")
    label = pd.Timestamp(day)
    if not calendar.is_session(label):
        raise ValueError("review target must be an NYSE session")
    start = calendar.session_close(calendar.previous_session(label)).to_pydatetime()
    return start, end


def make_queue(
    candidates: list[Candidate],
    *,
    prepared_at: str,
    target_session_date: str,
    policy: EPPolicy,
) -> dict:
    """Every eligible positive mover is reviewed; no partial top-N cutoff."""
    start, end = _window(target_session_date, prepared_at)
    if any(c.snapshot.target_session_date != target_session_date for c in candidates):
        raise ValueError("queue contains a different target session")
    eligible = [
        c
        for c in candidates
        if c.snapshot.discovery_gap_pct >= policy.discovery.min_abs_gap_pct
        and _research_prior_atr_blocker(c.snapshot, policy=policy) is None
        and nominate_candidates(
            [c.snapshot],
            as_of=end,
            policy=policy,
            apply_candidate_limit=False,
            require_verified_premarket_move=True,
        )
    ]
    # Stable across input ordering; liquid names first, then move, then ticker.
    eligible.sort(
        key=lambda c: (
            -c.snapshot.premarket_dollar_volume,
            -c.snapshot.discovery_gap_pct,
            c.snapshot.symbol,
        )
    )
    targets = []
    for candidate in eligible:
        snap = candidate.snapshot
        queries = [
            f'"{snap.company_name}" {snap.symbol} stock news {target_session_date}',
            f'"{snap.company_name}" press release after:{start.date()} before:{day_after(target_session_date)}',
        ]
        targets.append(
            {
                "candidate_id": candidate.candidate_id,
                "symbol": snap.symbol,
                "company_name": snap.company_name,
                "snapshot_sha256": digest(snap.to_dict()),
                "google_queries": queries,
                "google_urls": [
                    "https://www.google.com/search?" + urlencode({"q": q})
                    for q in queries
                ],
            }
        )
    queue = {
        "record_type": QUEUE_TYPE,
        "prepared_at": iso_utc(end),
        "target_session_date": target_session_date,
        "window_start": iso_utc(start),
        "targets": targets,
        "eligible_total": len(eligible),
        "unresearched_by_cap": max(0, len(eligible) - len(targets)),
    }
    return {**queue, "queue_sha256": digest(queue)}


def day_after(session: str) -> str:
    from datetime import timedelta

    return (date.fromisoformat(session) + timedelta(days=1)).isoformat()


def _validate_source(
    source: dict,
    *,
    start: datetime,
    reviewed_at: datetime,
    prepared_at: datetime,
    qualified: bool,
) -> None:
    host = _https(_text(source, "url"))
    if host in {"google.com", "www.google.com", "news.google.com"}:
        raise ValueError("search results are not article evidence")
    if source.get("capture_kind") != "ARTICLE_BODY":
        raise ValueError("review requires an opened article, not a snippet")
    _text(source, "observation_ref")
    _text(source, "title")
    content = _text(source, "content", 80)
    if source.get("content_sha256") != hashlib.sha256(content.encode()).hexdigest():
        raise ValueError("review article text digest mismatch")
    opened = _within(_text(source, "opened_at"), prepared_at, reviewed_at)
    # A rejected event may be old. Its page must still have been opened this run.
    if not qualified:
        return
    if source.get("source_kind") not in {
        "ISSUER",
        "REGULATOR",
        "ISSUER_WIRE",
        "EDITORIAL",
    }:
        raise ValueError("source authority must be explicitly assessed")
    _text(source, "authority_basis", 20)
    _event_within(_text(source, "published_at"), start, opened)
    _event_within(_text(source, "announced_at"), start, opened)
    if (
        source.get("event_status") != "ANNOUNCED"
        or source.get("event_relationship") != "DIRECT_ISSUER"
    ):
        raise ValueError("only announced direct-issuer events can qualify")
    for key in ("issuer_quote", "event_quote", "time_quote"):
        quote = _text(source, key, 10)
        if quote not in content:
            raise ValueError(f"{key} is not present in opened article text")


def validate_packet(
    packet: dict, candidates: list[Candidate], *, decision_at: str, policy: EPPolicy
) -> dict[str, CatalystAssessment]:
    """Rebuild assessments from the retained review, never from qualified flags."""
    if (
        packet.get("record_type") != PACKET_TYPE
        or packet.get("reviewer") != "CODEX_SEARCH_AND_READ"
    ):
        raise ValueError("unrecognized agent research record")
    queue = packet.get("queue")
    if not isinstance(queue, dict):
        raise TypeError("review has no prepared queue")
    expected = make_queue(
        candidates,
        prepared_at=queue["prepared_at"],
        target_session_date=queue["target_session_date"],
        policy=policy,
    )
    if queue != expected:
        raise ValueError("review queue does not match verified market inputs")
    start, end = _window(queue["target_session_date"], decision_at)
    prepared = _within(queue["prepared_at"], start, end)
    targets = {c["candidate_id"]: c for c in queue["targets"]}
    reviews = packet.get("reviews")
    if not isinstance(reviews, list) or len(reviews) > len(targets):
        raise ValueError("invalid bounded review list")
    assessments = {}
    for review in reviews:
        if not isinstance(review, dict):
            raise TypeError("review must be an object")
        cid = review.get("candidate_id")
        if cid not in targets or cid in assessments:
            raise ValueError("review has duplicate or out-of-queue identity")
        target = targets[cid]
        if any(review.get(k) != target[k] for k in ("symbol", "company_name")):
            raise ValueError("review issuer identity mismatch")
        status = review.get("status")
        if status not in _STATUSES:
            raise ValueError("invalid review disposition")
        if status != "UNRESOLVED" and review.get("research_complete") is not True:
            raise ValueError("terminal disposition requires completed research")
        reviewed_at = _within(_text(review, "reviewed_at"), prepared, end)
        searches = review.get("searches")
        if not isinstance(searches, list) or not 1 <= len(searches) <= 4:
            raise ValueError("each disposition needs observed Google search evidence")
        for search in searches:
            query = _text(search, "query", 5)
            host = _https(_text(search, "url"))
            url = urlsplit(search["url"])
            if (
                host not in {"google.com", "www.google.com"}
                or url.path != "/search"
                or parse_qs(url.query).get("q") != [query]
            ):
                raise ValueError(
                    "search observation must identify the actual Google query"
                )
            if (
                target["symbol"].lower() not in query.lower()
                and target["company_name"].lower() not in query.lower()
            ):
                raise ValueError("search query is not bound to this issuer")
            _text(search, "observation_ref")
            _within(_text(search, "searched_at"), prepared, reviewed_at)
            if search.get("outcome") not in {
                "RESULTS_READ",
                "NO_RELEVANT_RESULTS",
                "BLOCKED",
            }:
                raise ValueError("invalid search observation outcome")
        sources = review.get("sources", [])
        if not isinstance(sources, list) or len(sources) > 4:
            raise ValueError("invalid source list")
        qualified = status == "QUALIFIED"
        if status in {"QUALIFIED", "REJECTED"} and (
            not sources or not any(s["outcome"] == "RESULTS_READ" for s in searches)
        ):
            raise ValueError("qualification/rejection requires opened source evidence")
        if status == "NO_VERIFIED_CATALYST":
            completed = [s for s in searches if s["outcome"] != "BLOCKED"]
            if len({s["query"].casefold() for s in completed}) < 2 or not {
                "COMPANY_NEWS",
                "PRIMARY_ANNOUNCEMENT",
            }.issubset({s.get("purpose") for s in completed}):
                raise ValueError(
                    "negative conclusion requires completed news and primary-announcement searches"
                )
        for source in sources:
            _validate_source(
                source,
                start=start,
                reviewed_at=reviewed_at,
                prepared_at=prepared,
                qualified=qualified,
            )
        reason = _text(review, "reason", 20)
        if qualified:
            summary = _text(review, "business_change", 20)
            materiality = _text(review, "materiality_reason", 20)
            if review.get("catalyst_type") not in _KINDS:
                raise ValueError("qualified review needs a business catalyst type")
            if (
                review.get("contradictions_checked") is not True
                or review.get("adverse_flags") != []
            ):
                raise ValueError(
                    "qualified review must resolve contradictions/adverse events"
                )
            assessments[cid] = CatalystAssessment(
                status="WATCH",
                catalyst_type=review["catalyst_type"],
                summary=summary,
                confidence="AGENT_REVIEWED",
                materiality_score=3,
                materiality_signals=("AGENT_ASSESSED_BUSINESS_CATALYST",),
                evidence_urls=tuple(s["url"] for s in sources),
                evidence_published_at=tuple(s["published_at"] for s in sources),
                reason_codes=("AGENT_RESEARCH_ONLY_NOT_EXECUTION_APPROVAL",),
                publication_time_verified=True,
                trajectory_change_verified=True,
                research_news_qualified=True,
                research_news_basis=f"Google search and opened-source agent review. {materiality} {reason}",
                research_news_excerpt=" | ".join(s["event_quote"] for s in sources),
            )
        else:
            assessments[cid] = CatalystAssessment(
                status="UNCONFIRMED",
                catalyst_type="NONE",
                summary=reason,
                confidence="LOW",
                reason_codes=(f"AGENT_REVIEW_{status}",),
            )
    for cid in targets.keys() - assessments.keys():
        assessments[cid] = CatalystAssessment(
            status="UNCONFIRMED",
            catalyst_type="NONE",
            confidence="LOW",
            summary="Search-and-read review was not completed; coverage is unresolved.",
            reason_codes=("AGENT_REVIEW_UNRESOLVED",),
        )
    return assessments


def require_complete_research(packet: dict) -> None:
    """Email is a final deliverable, never a progress report or partial list.

    Call after validate_packet so identities and each terminal disposition have
    already been checked. Unresolved audit artifacts remain resumable locally.
    """
    targets = {t["candidate_id"] for t in packet["queue"]["targets"]}
    reviews = packet["reviews"]
    if (
        {r["candidate_id"] for r in reviews} != targets
        or packet["queue"]["unresearched_by_cap"]
        or any(
            r["status"] == "UNRESOLVED" or r.get("research_complete") is not True
            for r in reviews
        )
    ):
        raise ValueError(
            "candidate email requires completed research for every eligible mover"
        )


def apply_review(
    base: RunResult, packet: dict, *, decision_at: str, policy: EPPolicy
) -> RunResult:
    assessments = validate_packet(
        packet, base.candidates, decision_at=decision_at, policy=policy
    )
    decisions = []
    for previous in base.decisions:
        catalyst = assessments.get(previous.candidate_id)
        if catalyst is None:
            decisions.append(
                replace(
                    previous,
                    decision="WATCH",
                    blockers=("NEWS_RESEARCH_OUTSIDE_LONG_ELIGIBILITY",),
                )
            )
        else:
            decisions.append(
                replace(
                    previous,
                    catalyst=catalyst,
                    decision="WATCH",
                    setup_type="AGENT_REVIEWED_CATALYST",
                    blockers=("RESEARCH_ONLY_NO_EXECUTION_REVIEW",),
                )
            )
    warnings = set(base.warnings)
    if packet["queue"]["unresearched_by_cap"]:
        warnings.add("NEWS_RESEARCH_CAP_REACHED_INCOMPLETE_COVERAGE")
    seed = digest(
        {"base_run": base.run_id, "packet": packet, "decision_at": iso_utc(decision_at)}
    )
    return replace(
        base,
        run_id=f"EP-RUN-{packet['queue']['target_session_date']}-{seed[:12]}",
        generated_at=iso_utc(decision_at),
        decisions=decisions,
        previews=[],
        review_packet=packet,
        warnings=tuple(sorted(warnings)),
    )
