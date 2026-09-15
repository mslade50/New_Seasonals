import hashlib
import json

import pytest

from episodic_pivot.config import DEFAULT_POLICY
from episodic_pivot.email_delivery import EmailDeliveryError, morning_payload
from episodic_pivot.manifest import write_run_artifacts
from episodic_pivot.news import DirectArticleSearchProvider, assess_catalyst
from episodic_pivot.pipeline import run_shadow_pipeline
from episodic_pivot.schema import NewsDocument, NewsHit, parse_timestamp
from tests.test_episodic_pivot import AS_OF, _snapshot

BODY = (
    "Test Systems raised guidance after quarterly earnings beat estimates. "
    "The company raised its full-year revenue forecast by 20% and reported record revenue. "
) * 4


def document(body=BODY, domain="sec.gov", **kwargs):
    data = {
        "title": "Test Systems raises guidance",
        "url": f"https://{domain}/release",
        "canonical_url": f"https://{domain}/release",
        "publisher": domain,
        "published_at": "2026-08-24T12:00:00Z",
        "retrieved_at": AS_OF,
        "text_excerpt": body,
        "text_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "source_tier": "SECONDARY",
        "fetch_status": "FETCHED",
        "published_at_provenance": "PAGE_METADATA",
    }
    data.update(kwargs)
    return NewsDocument(**data)


def assess(docs, **kwargs):
    params = {
        "decision_at": AS_OF,
        "policy": DEFAULT_POLICY.news,
        "symbol": "TEST",
        "company_name": "Test Systems",
        "target_session_date": "2026-08-24",
        "first_trigger_at": "2026-08-24T12:15:00Z",
    }
    params.update(kwargs)
    return assess_catalyst(docs, **params)


@pytest.mark.parametrize(
    "domain", ["sec.gov", "globenewswire.com", "businesswire.com", "prnewswire.com"]
)
def test_material_body_with_fresh_source_qualifies_for_research(domain):
    result = assess([document(domain=domain)])
    assert result.research_news_qualified
    assert "raised guidance" in result.research_news_excerpt
    if domain != "sec.gov":
        assert not result.primary_source_confirmed
        assert result.status == "WATCH"  # no weakening of execution standard


def test_research_does_not_need_exact_first_trigger_but_sizing_still_does():
    result = assess([document()], first_trigger_at=None)
    assert result.research_news_qualified
    assert result.status == "WATCH"
    assert "MISSING_FIRST_TRIGGER_TIMESTAMP" in result.reason_codes


@pytest.mark.parametrize(
    "change",
    [
        {"fetch_status": "FETCH_FAILED:ValueError"},
        {"fetch_status": "UNVERIFIED_REPLAY"},
        {"text_sha256": "bad"},
        {"published_at_provenance": "SEARCH_FALLBACK"},
        {"published_at": "2026-08-21T18:00:00Z"},  # before prior NYSE close
        {"published_at": "2026-08-24T12:20:00Z"},  # after trigger
        {"published_at": "2026-08-24T14:00:00Z"},
        {"retrieved_at": "2026-08-24T14:00:00Z"},
        {"url": "https://news.google.com/wrapper"},
    ],
)
def test_missing_stale_tampered_or_late_evidence_never_qualifies(change):
    assert not assess([document(**change)]).research_news_qualified


@pytest.mark.parametrize(
    "body",
    [
        "Test Systems shares rose 20% on heavy volume without news. " * 12,
        "Test Systems is a popular company discussed by investors. " * 12,
        "Test Systems received a higher price target from an analyst. " * 12,
        "Test Systems announced a new product without financial projections. " * 12,
        "Test Systems reports quarterly earnings that met expectations. " * 12,
        "Other Holdings raised guidance after earnings beat estimates. " * 12,
        "Test Systems shares rallied after a report that Other Holdings raised guidance after earnings beat estimates. "
        * 8,
        "Test Systems shares rallied while Other Holdings raised guidance after earnings beat estimates. "
        * 8,
        "Test Systems previously raised guidance after earnings beat estimates. " * 8,
        "Test Systems could raise guidance if earnings beat estimates. " * 8,
        BODY + "Test Systems announced a public offering.",
        BODY + "Test Systems entered an all-cash transaction.",
    ],
)
def test_headline_alone_generic_movers_peer_news_and_adverse_events_fail(body):
    assert not assess([document(body=body)]).research_news_qualified


def test_secondary_needs_independent_reputable_actual_documents():
    assert not assess([document(domain="reuters.com")]).research_news_qualified
    assert assess(
        [
            document(domain="reuters.com"),
            document(domain="apnews.com", body=BODY + " Additional source reporting."),
        ]
    ).research_news_qualified
    assert not assess(
        [
            document(domain="reuters.com"),
            document(domain="news.reuters.com", body=BODY + " Syndication."),
        ]
    ).research_news_qualified
    assert not assess(
        [
            document(domain="example.com"),
            document(domain="example.org", body=BODY + " Another blog."),
        ]
    ).research_news_qualified


def run(docs, *, verified=True):
    return run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": docs},
        offline_documents_verified=verified,
    )


def test_unconfirmed_mover_never_leaks_to_html_markdown_or_email_attachments(tmp_path):
    result = run([])
    root = write_run_artifacts(
        result, policy=DEFAULT_POLICY, output_dir=tmp_path / result.run_id
    )
    assert len(json.loads((root / "decisions.json").read_text())) == 1
    assert json.loads((root / "news_qualified.json").read_text()) == []
    payload = morning_payload(root)
    assert "0 news-qualified" in payload.subject
    assert "news coverage incomplete" in payload.subject
    for name in ("report.html", "report.md"):
        text = (root / name).read_text(encoding="utf-8")
        assert "TEST" not in text
        assert "No news-qualified EP candidates" in text
    assert {p.name for p in payload.attachments} == {
        "report.html",
        "report.md",
        "manifest.json",
    }


def test_qualified_report_contains_business_fact_source_and_no_auto_trade(tmp_path):
    result = run([document()])
    root = write_run_artifacts(
        result, policy=DEFAULT_POLICY, output_dir=tmp_path / result.run_id
    )
    payload = morning_payload(root)
    assert "1 news-qualified" in payload.subject
    assert "raised guidance" in payload.html_body
    assert "https://sec.gov/release" in payload.html_body
    assert "Research only" in payload.html_body


def test_sender_rejects_legacy_report_without_news_gate(tmp_path):
    result = run([])
    root = write_run_artifacts(
        result, policy=DEFAULT_POLICY, output_dir=tmp_path / result.run_id
    )
    path = root / "manifest.json"
    raw = json.loads(path.read_text())
    raw["artifacts"].pop("news_qualified.json")
    path.write_text(json.dumps(raw))
    with pytest.raises(EmailDeliveryError, match="missing email deliverables"):
        morning_payload(root)


def rewrite_hashed_artifact(root, name, value):
    path = root / name
    path.write_text(json.dumps(value), encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][name] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    "mutation",
    ["missing", "generic", "stale", "future", "peer", "bad_hash", "headline_only"],
)
def test_sender_independently_rejects_bad_sources_even_with_qualified_flags_and_updated_manifest(
    tmp_path, mutation
):
    result = run([document()])
    root = write_run_artifacts(
        result, policy=DEFAULT_POLICY, output_dir=tmp_path / result.run_id
    )
    assert "1 news-qualified" in morning_payload(root).subject
    evidence = json.loads((root / "evidence.json").read_text())
    key = next(iter(evidence))
    doc = evidence[key][0]
    if mutation == "missing":
        evidence[key] = []
    elif mutation in {"generic", "peer", "headline_only"}:
        doc["text_excerpt"] = {
            "generic": "Test Systems shares rose 20% without news. " * 12,
            "peer": "Other Holdings raised guidance after earnings beat estimates. "
            * 12,
            "headline_only": "",
        }[mutation]
        doc["text_sha256"] = hashlib.sha256(doc["text_excerpt"].encode()).hexdigest()
    elif mutation in {"stale", "future"}:
        doc["published_at"] = (
            "2026-08-21T18:00:00Z" if mutation == "stale" else "2026-08-24T14:00:00Z"
        )
    else:
        doc["text_sha256"] = "bad"
    rewrite_hashed_artifact(root, "evidence.json", evidence)
    with pytest.raises(EmailDeliveryError, match="independent pre-email vetting"):
        morning_payload(root)


@pytest.mark.parametrize(
    "change",
    [
        {"reported_change_pct": 4.999},
        {"reported_change_pct": -5.0},
        {"premarket_volume": 99999},
    ],
)
def test_sender_rejects_below_threshold_candidates_even_with_qualified_flags(
    tmp_path, change
):
    result = run([document()])
    root = write_run_artifacts(
        result, policy=DEFAULT_POLICY, output_dir=tmp_path / result.run_id
    )
    candidates = json.loads((root / "candidates.json").read_text())
    candidates[0]["snapshot"].update(change)
    rewrite_hashed_artifact(root, "candidates.json", candidates)
    with pytest.raises(EmailDeliveryError, match="independent pre-email vetting"):
        morning_payload(root)


def test_url_fallback_is_bounded_and_never_promotes_summaries(monkeypatch):
    class Primary:
        name = "GOOGLE_NEWS_RSS"

        def search(self, **kwargs):
            return [
                NewsHit(
                    title="wrapper",
                    url="https://news.google.com/articles/id",
                    published_at=None,
                )
            ]

    class Ticker:
        def get_news(self, **kwargs):
            assert kwargs == {"count": 2, "tab": "all"}
            return [
                {
                    "content": {
                        "title": "News",
                        "canonicalUrl": {"url": "https://sec.gov/release"},
                        "summary": BODY,
                        "pubDate": "2026-08-24T12:00:00Z",
                    }
                }
            ] * 2

    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", lambda symbol: Ticker())
    monkeypatch.setattr(yfinance, "set_tz_cache_location", lambda path: None)
    provider = DirectArticleSearchProvider(Primary(), metadata_cache="artifacts/test")
    hits = provider.search(
        symbol="TEST",
        company_name="Test Systems",
        as_of=parse_timestamp(AS_OF),
        limit=2,
    )
    assert len(hits) == 2
    assert hits[0].url == "https://sec.gov/release"
    assert hits[0].snippet == ""
    assert hits[1].url.startswith("https://news.google.com")
