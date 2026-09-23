"""Boundary tests for agent-reviewed research; no network, mail or broker calls."""

import copy
import hashlib
import json
from datetime import datetime
from urllib.parse import urlencode

import pytest

from episodic_pivot.config import DEFAULT_POLICY
from episodic_pivot.email_delivery import EmailDeliveryError, morning_payload
from episodic_pivot.manifest import sha256_file, write_run_artifacts
from episodic_pivot.pipeline import run_shadow_pipeline
from episodic_pivot.reviewed_news import MODE, PACKET_TYPE, apply_review, make_queue
from tests.test_episodic_pivot import AS_OF, _snapshot

DECISION = "2026-08-24T12:36:00Z"
CONTENT = (
    "August 24, 2026 at 8:00 AM EDT. Test Systems Inc. announced today that it "
    "has received FDA approval for its lead product. The approval permits "
    "commercial sales beginning this quarter, creating the company's first product revenue."
)


def base(snapshots=None):
    return run_shadow_pipeline(
        snapshots or [_snapshot()],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
    )


def packet(result=None):
    result = result or base()
    queue = make_queue(
        result.candidates,
        prepared_at=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
    )
    target = queue["targets"][0]
    query = target["google_queries"][0]
    return {
        "record_type": PACKET_TYPE,
        "reviewer": "CODEX_SEARCH_AND_READ",
        "queue": queue,
        "reviews": [
            {
                **{k: target[k] for k in ("candidate_id", "symbol", "company_name")},
                "status": "QUALIFIED",
                "research_complete": True,
                "reviewed_at": "2026-08-24T12:35:00Z",
                "searches": [
                    {
                        "query": query,
                        "url": "https://www.google.com/search?"
                        + urlencode({"q": query}),
                        "searched_at": "2026-08-24T12:32:00Z",
                        "observation_ref": "browser-observation-1",
                        "outcome": "RESULTS_READ",
                    }
                ],
                "reason": "The release is on the issuer's own investor-relations site and announces a new approval today.",
                "business_change": "The company received approval for its first commercial product.",
                "materiality_reason": "Approval opens a new revenue source rather than an ordinary trading update.",
                "catalyst_type": "REGULATORY_APPROVAL",
                "contradictions_checked": True,
                "adverse_flags": [],
                "sources": [
                    {
                        "url": "https://ir.testsystems.example/news/approval",
                        "title": "Test Systems approval announcement",
                        "source_kind": "ISSUER",
                        "authority_basis": "Investor-relations release linked from the company's verified corporate website.",
                        "capture_kind": "ARTICLE_BODY",
                        "opened_at": "2026-08-24T12:33:00Z",
                        "observation_ref": "browser-observation-2",
                        "content": CONTENT,
                        "content_sha256": hashlib.sha256(CONTENT.encode()).hexdigest(),
                        "published_at": "2026-08-24T12:00:00Z",
                        "announced_at": "2026-08-24T12:00:00Z",
                        "event_status": "ANNOUNCED",
                        "event_relationship": "DIRECT_ISSUER",
                        "issuer_quote": "Test Systems Inc.",
                        "event_quote": "has received FDA approval for its lead product.",
                        "time_quote": "August 24, 2026 at 8:00 AM EDT.",
                    }
                ],
            }
        ],
    }


def complete(p=None, b=None):
    b = b or base()
    return apply_review(b, p or packet(b), decision_at=DECISION, policy=DEFAULT_POLICY)


def write(tmp_path, result):
    return write_run_artifacts(
        result,
        policy=DEFAULT_POLICY,
        output_dir=tmp_path / result.run_id,
        search_provider=MODE,
    )


def rehash(root, name):
    path = root / "manifest.json"
    m = json.loads(path.read_text())
    m["artifacts"][name] = {
        "sha256": sha256_file(root / name),
        "size_bytes": (root / name).stat().st_size,
    }
    path.write_text(json.dumps(m))


def test_one_original_issuer_source_passes_without_domain_allowlist(tmp_path):
    result = complete()
    assert result.decisions[0].catalyst.research_news_qualified
    assert result.decisions[0].catalyst.status == "WATCH"
    assert not result.decisions[0].catalyst.primary_source_confirmed
    assert result.previews == []
    root = write(tmp_path, result)
    mail = morning_payload(root)
    assert "1 news-qualified" in mail.subject
    assert "first commercial product" in mail.html_body
    assert "https://ir.testsystems.example/news/approval" in mail.html_body
    assert "No fetched source" not in mail.html_body
    assert {p.name for p in mail.attachments} == {
        "report.html",
        "report.md",
        "manifest.json",
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("announced_at", "2026-08-21T18:00:00Z"),  # fresh article about stale event
        ("announced_at", "2026-08-24T13:00:00Z"),
        ("published_at", "2026-08-21T18:00:00Z"),
        ("opened_at", "2026-08-24T12:37:00Z"),
        ("opened_at", "2026-08-24T12:00:00Z"),
        ("event_status", "EXPECTED"),
        ("event_relationship", "PEER"),
        ("capture_kind", "SEARCH_SNIPPET"),
        ("content_sha256", "bad"),
        ("event_quote", "This claim never appeared in the source text"),
        ("time_quote", "Invented publication timestamp"),
        ("authority_basis", ""),
        ("source_kind", "UNKNOWN"),
        ("url", "https://news.google.com/rss/articles/wrapper"),
        ("url", "http://example.com/article"),
        ("url", "https://127.0.0.1/article"),
        ("url", "https://user:password@example.com/article"),
    ],
)
def test_bad_evidence_cannot_qualify(key, value):
    p = packet()
    p["reviews"][0]["sources"][0][key] = value
    with pytest.raises(ValueError):
        complete(p)


@pytest.mark.parametrize(
    "key,value",
    [
        ("symbol", "PEER"),
        ("company_name", "Other Company"),
        ("candidate_id", "unknown"),
        ("status", "APPROVED"),
        ("reason", ""),
        ("sources", []),
        ("searches", []),
        ("contradictions_checked", False),
        ("adverse_flags", ["DILUTION_OR_OFFERING"]),
        ("materiality_reason", ""),
    ],
)
def test_incomplete_or_wrong_issuer_review_fails(key, value):
    p = packet()
    p["reviews"][0][key] = value
    with pytest.raises(ValueError):
        complete(p)


def test_duplicate_and_out_of_queue_reviews_fail():
    p = packet()
    p["reviews"].append(copy.deepcopy(p["reviews"][0]))
    with pytest.raises(ValueError):
        complete(p)


def test_missing_and_rejected_are_not_conflated(tmp_path):
    p = packet()
    p["reviews"] = []
    result = complete(p)
    with pytest.raises(EmailDeliveryError):
        morning_payload(write(tmp_path, result))
    p = packet()
    p["reviews"][0]["status"] = "REJECTED"
    p["reviews"][0]["reason"] = (
        "Read the source: this is a retrospective event, not a new announcement."
    )
    p["reviews"][0]["sources"][0]["announced_at"] = "2026-08-01T12:00:00Z"
    result = complete(p)
    mail = morning_payload(write(tmp_path, result))
    assert "news coverage incomplete" not in mail.subject
    assert "Reviewed and excluded: 1" in mail.html_body
    assert "TEST" not in mail.html_body


def test_blocked_google_is_unresolved_not_a_candidate(tmp_path):
    p = packet()
    review = p["reviews"][0]
    review.update(
        status="UNRESOLVED",
        sources=[],
        reason="Google blocked the query; source evidence was not available.",
    )
    review["searches"][0]["outcome"] = "BLOCKED"
    with pytest.raises(EmailDeliveryError):
        morning_payload(write(tmp_path, complete(p)))


def test_queue_does_not_spend_budget_on_negative_or_low_atr():
    b = base(
        [
            _snapshot(symbol="DOWN", last=8),
            _snapshot(symbol="LOW", atr_14=0.2),
            _snapshot(),
        ]
    )
    q = make_queue(
        b.candidates,
        prepared_at=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
    )
    assert [c["symbol"] for c in q["targets"]] == ["TEST"]


@pytest.mark.parametrize(
    "overrides", [{"last": 10.49}, {"premarket_volume": 99_999}, {"atr_14": 0.4}]
)
def test_market_thresholds_remain_mandatory(overrides):
    b = base([_snapshot(**overrides)])
    q = make_queue(
        b.candidates,
        prepared_at=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
    )
    assert q["targets"] == []


def test_queue_market_fingerprint_cannot_be_swapped():
    p = packet()
    b = base([_snapshot(premarket_volume=200_000)])
    with pytest.raises(ValueError, match="queue"):
        complete(p, b)


def test_every_eligible_mover_is_selected_without_top_25_cutoff():
    b = base([_snapshot(symbol=f"S{i}") for i in range(28)])
    q = make_queue(
        b.candidates,
        prepared_at=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
    )
    assert len(q["targets"]) == 28 and q["unresearched_by_cap"] == 0


def test_post_open_or_wrong_session_decisions_fail():
    for decision in ("2026-08-24T13:30:00Z", "2026-08-25T12:00:00Z"):
        with pytest.raises(ValueError):
            apply_review(base(), packet(), decision_at=decision, policy=DEFAULT_POLICY)


@pytest.mark.parametrize(
    "filename", ["report.html", "report.md", "agent_reviews.json", "decisions.json"]
)
def test_sender_checks_records_not_just_manifest_hashes(tmp_path, filename):
    root = write(tmp_path, complete())
    p = root / filename
    content = p.read_text(encoding="utf-8")
    p.write_text(
        content.replace("first commercial product", "invented new claim"),
        encoding="utf-8",
    )
    rehash(root, filename)
    with pytest.raises(EmailDeliveryError):
        morning_payload(root)


def test_review_changes_run_identity():
    p = packet()
    first = complete(p)
    p["reviews"][0]["materiality_reason"] += " Explicit source-backed reasoning."
    assert complete(p).run_id != first.run_id


def test_agent_review_flag_rejects_legacy_send(tmp_path):
    from scripts.send_episodic_pivot_email import main

    root = write_run_artifacts(
        base(), policy=DEFAULT_POLICY, output_dir=tmp_path / base().run_id
    )
    assert (
        main(["--kind", "morning", "--artifact", str(root), "--require-agent-review"])
        == 2
    )


def test_date_only_today_does_not_invent_a_clock_time(tmp_path):
    p = packet()
    p["reviews"][0]["sources"][0].update(
        published_at="2026-08-24", announced_at="2026-08-24"
    )
    assert (
        "announced 2026-08-24;"
        in morning_payload(write(tmp_path, complete(p))).html_body
    )
    p["reviews"][0]["sources"][0]["announced_at"] = "2026-08-21"
    with pytest.raises(ValueError, match="date-only"):
        complete(p)


def test_empty_market_queue_retains_session_and_is_valid_zero(tmp_path):
    b = run_shadow_pipeline(
        [], as_of=AS_OF, target_session_date="2026-08-24", policy=DEFAULT_POLICY
    )
    q = make_queue(
        [], prepared_at=AS_OF, target_session_date="2026-08-24", policy=DEFAULT_POLICY
    )
    p = {
        "record_type": PACKET_TYPE,
        "reviewer": "CODEX_SEARCH_AND_READ",
        "queue": q,
        "reviews": [],
    }
    mail = morning_payload(write(tmp_path, complete(p, b)))
    assert mail.metadata["target_session_date"] == "2026-08-24"
    assert "0 news-qualified" in mail.subject
    assert "news coverage incomplete" not in mail.subject


@pytest.mark.parametrize("track_morning", [False, True])
def test_prepare_seal_complete_and_dry_email_cli(tmp_path, monkeypatch, track_morning):
    from episodic_pivot import morning_completion as completion
    from scripts import run_episodic_pivot_shadow as runner
    from scripts import seal_ep_news_reviews as seal
    from scripts.send_episodic_pivot_email import main as email

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(seal, "ROOT", tmp_path)
    monkeypatch.setattr(completion, "_clock", lambda _now=None: datetime.fromisoformat(DECISION.replace("Z", "+00:00")))
    monkeypatch.setattr(
        runner, "_load_snapshots", lambda path: ([_snapshot()], "YFINANCE", ())
    )
    source = tmp_path / "artifacts" / "input.json"
    source.parent.mkdir()
    source.write_text("{}")
    output = tmp_path / "artifacts"
    queue = output / "queue.json"
    assert (
        runner.main(
            [
                "--snapshot",
                str(source),
                "--prepare-google-review",
                str(queue),
                "--as-of",
                AS_OF,
                "--run-research",
                *(["--track-morning"] if track_morning else []),
            ]
        )
        == 0
    )
    assert json.loads(queue.read_text()) == packet()["queue"]
    if track_morning:
        state = completion.progress(output / "episodic_pivot", "2026-08-24")
        assert state["stage"] == "RESEARCH" and set(state["artifacts"]) == {"queue", "snapshot_1"}
    notes = output / "notes.json"
    notes.write_text(json.dumps(packet()["reviews"]))
    reviewed = output / "reviews.json"
    assert (
        seal.main(
            ["--queue", str(queue), "--notes", str(notes), "--output", str(reviewed)]
        )
        == 0
    )
    assert (
        runner.main(
            [
                "--snapshot",
                str(source),
                "--news-mode",
                "agent-reviewed",
                "--reviews",
                str(reviewed),
                "--as-of",
                DECISION,
                "--output-root",
                str(output),
                "--run-research",
                *(["--track-morning"] if track_morning else []),
            ]
        )
        == 0
    )
    runs = list(output.glob("EP-RUN-*"))
    assert len(runs) == 1
    manifest = json.loads((runs[0] / "manifest.json").read_text())
    assert manifest["search_provider"] == MODE
    if track_morning:
        state = completion.progress(output / "episodic_pivot", "2026-08-24")
        assert state["stage"] == "REPORT_READY" and set(state["artifacts"]) == {"queue", "snapshot_1", "reviews", "report"}
    assert (
        email(
            ["--kind", "morning", "--artifact", str(runs[0]), "--require-agent-review"]
        )
        == 0
    )


def test_historical_review_cannot_be_sent_today(tmp_path, monkeypatch):
    from scripts import send_episodic_pivot_email as sender

    root = write(tmp_path, complete())
    monkeypatch.setattr(
        sender,
        "resolve_email_settings",
        lambda **kw: pytest.fail("credentials must not load"),
    )
    assert (
        sender.main(
            [
                "--kind",
                "morning",
                "--artifact",
                str(root),
                "--require-agent-review",
                "--send",
            ]
        )
        == 2
    )
