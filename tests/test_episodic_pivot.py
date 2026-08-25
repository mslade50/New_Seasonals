from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import fields, replace
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

import episodic_pivot.schema as ep_schema
from episodic_pivot.config import DEFAULT_POLICY
from episodic_pivot.historical import (
    attach_benchmark_outcomes,
    clustered_outcome_summary,
    horizon_comparison_summary,
    study_ticker,
)
from episodic_pivot.historical_news import (
    SECSubmissionArchive,
    bind_fmp_evidence,
    bind_sec_evidence,
    blind_events,
    classify_event_text,
    classify_fmp_source_quality,
    classify_trajectory,
    normalize_fmp_news,
    normalize_sec_submissions,
    summarize_event_evidence,
)
from episodic_pivot.manifest import write_run_artifacts
from episodic_pivot.news import (
    ArticleFetcher,
    _document_mentions_candidate,
    _PinnedHTTPSConnection,
    _read_response_body,
    _validate_public_http_url,
    assess_catalyst,
    normalize_evidence_document,
    source_tier,
)
from episodic_pivot.pipeline import run_shadow_pipeline
from episodic_pivot.premarket import nominate_candidates
from episodic_pivot.qualify import qualify_candidate
from episodic_pivot.schema import (
    NewsDocument,
    NewsHit,
    PremarketSnapshot,
    ResearchSizingPreview,
    RunResult,
)
from scripts.capture_ep_premarket_ibkr import (
    _DAILY_PRICE_BASIS,
    _DAILY_WHAT_TO_SHOW,
    _daily_metrics,
    _halt_status,
    _load_target_rows,
    _premarket_metrics,
    _round_robin_keys,
)
from scripts.run_episodic_pivot_shadow import _verify_evidence_manifest
from trading_calendar import TRADING_DAY

AS_OF = "2026-08-24T12:31:00Z"
FIRST_TRIGGER = "2026-08-24T12:30:30Z"


def _trading_dates(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq=TRADING_DAY)


def _snapshot(**overrides) -> PremarketSnapshot:
    values = {
        "symbol": "TEST",
        "company_name": "Test Systems Inc.",
        "observed_at": "2026-08-24T12:30:30Z",
        "previous_close": 10.0,
        "quote_previous_close": overrides.get("previous_close", 10.0),
        "last": 11.0,
        "bid": 10.98,
        "ask": 11.0,
        "bid_size": 3_000,
        "ask_size": 2_000,
        "premarket_volume": 500_000,
        "premarket_open": 10.5,
        "premarket_high": 11.1,
        "premarket_low": 10.4,
        "premarket_vwap": 10.90,
        "prior_two_day_low": 9.80,
        "atr_14": 0.50,
        "avg_volume_20": 1_000_000,
        "addv_63": 10_000_000,
        "market_cap": 800_000_000,
        "float_shares": 20_000_000,
        "prior_63d_return_pct": -8.0,
        "sessions_since_prior_ep": 300,
        "market_data_status": "LIVE",
        "premarket_metrics_at": "2026-08-24T12:30:00Z",
        "halt_status": "NOT_HALTED",
        "tradeable": True,
        "daily_price_basis": "IBKR_ADJUSTED_LAST",
        "contract_con_id": 123456,
        "primary_exchange": "NASDAQ",
        "contract_identity_status": "UNIQUE_IBKR_MATCH",
        "resolved_symbol": overrides.get("symbol", "TEST"),
        "contract_sec_type": "STK",
        "contract_currency": "USD",
        "valid_exchanges": "SMART,NASDAQ",
        "allowed_order_types": "LMT,MKT,STP",
    }
    values.update(overrides)
    return PremarketSnapshot(**values)


def _document(**overrides) -> NewsDocument:
    default_text = (
        "Test Systems reports quarterly results and raised guidance after revenue grew "
        "well above prior expectations. Management described durable demand, a larger "
        "order backlog, improved margins, and a higher full-year outlook. "
    ) * 4
    text = overrides.pop("text_excerpt", default_text)
    values = {
        "title": "Test Systems reports earnings and raises guidance",
        "url": "https://www.sec.gov/Archives/edgar/data/123456/test-systems-quarterly-results",
        "canonical_url": "https://www.sec.gov/Archives/edgar/data/123456/test-systems-quarterly-results",
        "publisher": "Test Systems",
        "published_at": "2026-08-24T12:00:00Z",
        "retrieved_at": AS_OF,
        "text_excerpt": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_tier": "PRIMARY_REGULATOR",
        "fetch_status": "FETCHED",
        "catalyst_types": ("EARNINGS_GUIDANCE",),
        "adverse_flags": (),
        "published_at_provenance": "SEC_ACCEPTED_AT",
    }
    values.update(overrides)
    return NewsDocument(**values)


def test_discovery_uses_move_or_dollar_branch_but_always_requires_volume_and_price():
    pct_branch = _snapshot(symbol="PCT", last=10.25, bid=10.23, ask=10.25)
    dollar_branch = _snapshot(
        symbol="DOLLAR",
        previous_close=100,
        last=101,
        bid=100.98,
        ask=101,
        premarket_vwap=100.8,
    )
    too_thin = _snapshot(symbol="THIN", premarket_volume=99_999)
    too_cheap = _snapshot(symbol="CHEAP", last=0.9, bid=0.89, ask=0.90)

    candidates = nominate_candidates(
        [pct_branch, dollar_branch, too_thin, too_cheap],
        as_of=AS_OF,
        policy=DEFAULT_POLICY,
    )
    assert [item.snapshot.symbol for item in candidates] == ["DOLLAR", "PCT"]


def test_stale_or_delayed_snapshot_is_visible_but_not_stageable():
    snapshot = _snapshot(
        observed_at="2026-08-24T12:20:00Z", market_data_status="DELAYED"
    )
    candidate = nominate_candidates([snapshot], as_of=AS_OF, policy=DEFAULT_POLICY)[0]
    catalyst = assess_catalyst(
        [_document()],
        decision_at=AS_OF,
        policy=DEFAULT_POLICY.news,
        first_trigger_at=FIRST_TRIGGER,
    )
    decision = qualify_candidate(
        candidate, catalyst, policy=DEFAULT_POLICY, decision_at=AS_OF
    )

    assert "STALE_MARKET_DATA" in candidate.discovery_warnings
    assert "NON_LIVE_MARKET_DATA" in candidate.discovery_warnings
    assert decision.decision == "WATCH"
    assert "STALE_MARKET_DATA" in decision.blockers


def test_future_snapshot_uses_small_clock_skew_not_staleness_window():
    snapshot = _snapshot(observed_at="2026-08-24T12:32:29Z")
    candidate = nominate_candidates([snapshot], as_of=AS_OF, policy=DEFAULT_POLICY)[0]
    assert "SNAPSHOT_FROM_FUTURE" in candidate.discovery_warnings


def test_search_snippet_or_failed_fetch_cannot_confirm_an_ep():
    failed = _document(
        fetch_status="FETCH_FAILED:ValueError",
        text_excerpt="",
        text_sha256="",
        source_tier="SEARCH_WRAPPER",
    )
    result = assess_catalyst(
        [failed],
        decision_at=AS_OF,
        policy=DEFAULT_POLICY.news,
        first_trigger_at=FIRST_TRIGGER,
    )
    assert result.status == "UNCONFIRMED"
    assert "NO_ACTUAL_SOURCE_EVIDENCE" in result.reason_codes


def test_primary_actual_document_confirms_and_future_document_does_not():
    confirmed = assess_catalyst(
        [_document()],
        decision_at=AS_OF,
        policy=DEFAULT_POLICY.news,
        first_trigger_at=FIRST_TRIGGER,
    )
    future = assess_catalyst(
        [_document(published_at="2026-08-24T12:40:00Z")],
        decision_at=AS_OF,
        policy=DEFAULT_POLICY.news,
        first_trigger_at=FIRST_TRIGGER,
    )
    assert confirmed.status == "CONFIRMED"
    assert confirmed.catalyst_type == "EARNINGS_GUIDANCE"
    assert future.status == "UNCONFIRMED"
    assert "POST_DECISION_SOURCE" in future.reason_codes


@pytest.mark.parametrize(
    "url",
    (
        "https://www.businesswire.com/news/home/test-systems-quarterly-results",
        "https://www.reuters.com/markets/test-systems-quarterly-results",
    ),
)
def test_wire_or_single_reputable_article_cannot_auto_confirm_classic(url):
    document = _document(
        url=url,
        canonical_url=url,
        published_at_provenance="PAGE_METADATA",
    )
    assessment = assess_catalyst(
        [document],
        decision_at=AS_OF,
        policy=DEFAULT_POLICY.news,
        first_trigger_at=FIRST_TRIGGER,
        symbol="TEST",
        company_name="Test Systems Inc.",
    )
    assert assessment.status == "WATCH"
    assert assessment.primary_source_confirmed is False
    assert "PRIMARY_SOURCE_NOT_VERIFIED" in assessment.reason_codes


def test_source_after_price_trigger_and_search_fallback_time_cannot_confirm():
    after_trigger = assess_catalyst(
        [_document(published_at="2026-08-24T12:30:40Z")],
        decision_at=AS_OF,
        policy=DEFAULT_POLICY.news,
        first_trigger_at=FIRST_TRIGGER,
        symbol="TEST",
        company_name="Test Systems Inc.",
    )
    fallback_time = assess_catalyst(
        [_document(published_at_provenance="SEARCH_FALLBACK")],
        decision_at=AS_OF,
        policy=DEFAULT_POLICY.news,
        first_trigger_at=FIRST_TRIGGER,
        symbol="TEST",
        company_name="Test Systems Inc.",
    )
    assert after_trigger.status == "UNCONFIRMED"
    assert "SOURCE_AFTER_FIRST_PRICE_TRIGGER" in after_trigger.reason_codes
    assert fallback_time.status == "WATCH"
    assert fallback_time.publication_time_verified is False
    assert "UNVERIFIED_PUBLICATION_TIMESTAMP" in fallback_time.reason_codes


def test_document_retrieved_after_decision_is_not_point_in_time_evidence():
    result = assess_catalyst(
        [_document(retrieved_at="2026-08-24T12:40:00Z")],
        decision_at=AS_OF,
        policy=DEFAULT_POLICY.news,
        first_trigger_at=FIRST_TRIGGER,
    )
    assert result.status == "UNCONFIRMED"
    assert "POST_DECISION_RETRIEVAL" in result.reason_codes


def test_article_retrieved_at_is_fetch_completion_clock(monkeypatch):
    completed = pd.Timestamp("2026-08-24T12:32:00Z").to_pydatetime()
    body = (
        "<html><article>"
        + (
            "Test Systems raised guidance after quarterly results and durable demand. "
            * 10
        )
        + "</article></html>"
    )
    monkeypatch.setattr(
        "episodic_pivot.news._fetch_public_html",
        lambda *args, **kwargs: (
            "https://www.businesswire.com/news/home/test",
            "text/html",
            body,
        ),
    )
    monkeypatch.setattr("episodic_pivot.news.utc_now", lambda: completed)
    document = ArticleFetcher().fetch(
        NewsHit(
            title="Test Systems raises guidance",
            url="https://www.businesswire.com/news/home/test",
            published_at="2026-08-24T12:00:00Z",
        )
    )
    assert document.retrieved_at == "2026-08-24T12:32:00Z"


def test_adverse_offering_never_reaches_research_sizing():
    doc = _document(
        text_excerpt=(
            "Test Systems announced a registered direct public offering that will issue "
            "new common shares and dilute existing owners. The financing terms, placement "
            "agent, expected proceeds, and closing conditions were disclosed by the company. "
        )
        * 4
    )
    result = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        target_session_date=date(2026, 8, 24),
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [doc]},
        offline_documents_verified=True,
    )
    assert result.decisions[0].decision == "WATCH"
    assert result.decisions[0].setup_type == "CATALYST_WITH_FINANCING_RISK"
    assert result.previews == []


def test_failed_clinical_endpoint_is_adverse_not_a_positive_ep():
    text = (
        "Test Systems said its phase 3 clinical trial failed the primary endpoint. The "
        "company will discontinue the program after the study did not meet the primary "
        "endpoint, and management is evaluating remaining cash and strategic options. "
    ) * 4
    document = _document(
        title="Test Systems phase 3 trial fails primary endpoint",
        text_excerpt=text,
    )
    result = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [document]},
        offline_documents_verified=True,
    )
    assert result.decisions[0].decision == "WATCH"
    assert "CLINICAL_OR_REGULATORY_FAILURE" in result.decisions[0].blockers
    assert result.previews == []


def test_end_to_end_research_sizing_is_capped_and_never_executable(tmp_path):
    result = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        target_session_date=date(2026, 8, 24),
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document()]},
        offline_documents_verified=True,
    )
    assert result.decisions[0].decision == "RESEARCH_PREVIEW_ELIGIBLE"
    preview = result.previews[0]
    assert preview.max_preview_shares > 0
    assert preview.binding_constraint in {
        "RISK",
        "MAX_NOTIONAL",
        "ADDV_PARTICIPATION",
        "PREMARKET_VOLUME_PARTICIPATION",
        "DISPLAYED_ASK_PARTICIPATION",
        "ABSOLUTE_QUANTITY",
    }
    assert preview.preview_only is True
    assert preview.executable is False
    assert preview.broker_route == "NONE"
    assert preview.order_submission_allowed is False
    assert preview.production_eligible is False
    assert preview.reference_activation_min_price == 10.40
    assert preview.reference_activation_max_price == 12.50
    assert preview.max_reference_gap_pct == 25.0
    assert preview.reference_entry_window_end_et == "09:35:00"
    assert preview.quote_revalidation_required is True
    assert preview.halt_revalidation_required is True
    assert preview.gap_revalidation_required is True
    assert preview.modeled_risk_bps <= DEFAULT_POLICY.execution.classic_risk_bps

    input_path = tmp_path / "input.json"
    input_path.write_text("{}", encoding="utf-8")
    output = write_run_artifacts(
        result,
        policy=DEFAULT_POLICY,
        output_dir=tmp_path / "run",
        input_files={"snapshot": input_path},
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["safety"]["research_only"] is True
    assert manifest["safety"]["live_actions_enabled"] is False
    assert manifest["safety"]["order_submission_allowed"] is False
    assert manifest["safety"]["order_staging_performed"] is False
    assert (output / "research_sizing_preview.csv").exists()
    assert (output / "report.html").exists()
    assert not (output / "staging_preview.csv").exists()


def test_multiple_previews_are_scaled_to_daily_portfolio_risk_cap():
    snapshots = [_snapshot(symbol=f"TST{i}") for i in range(6)]
    documents = {item.symbol: [_document()] for item in snapshots}
    result = run_shadow_pipeline(
        snapshots,
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents=documents,
        offline_documents_verified=True,
    )
    assert len(result.previews) == 6
    assert sum(item.modeled_risk_bps for item in result.previews) <= (
        DEFAULT_POLICY.execution.max_daily_risk_bps + 0.01
    )
    assert all("DAILY_RISK" in item.binding_constraint for item in result.previews)


def test_daily_cap_zero_quantity_is_visible_in_decision():
    snapshots = [_snapshot(symbol=f"ONE{i}", ask_size=4) for i in range(6)]
    documents = {item.symbol: [_document()] for item in snapshots}
    tiny_daily_cap_policy = replace(
        DEFAULT_POLICY,
        execution=replace(DEFAULT_POLICY.execution, max_daily_risk_bps=0.01),
    )
    result = run_shadow_pipeline(
        snapshots,
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=tiny_daily_cap_policy,
        offline_documents=documents,
        offline_documents_verified=True,
    )
    assert result.previews == []
    assert all(decision.decision == "WATCH" for decision in result.decisions)
    assert all(
        "SIZING:DAILY_CAP_ZERO_QUANTITY" in decision.blockers
        for decision in result.decisions
    )


def test_search_failure_fails_candidate_closed_without_aborting_run():
    class BrokenSearch:
        name = "BROKEN"

        def search(self, **kwargs):
            raise RuntimeError("provider unavailable")

    result = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        search_provider=BrokenSearch(),
    )
    assert result.decisions[0].decision == "WATCH"
    document = next(iter(result.documents_by_candidate.values()))[0]
    assert document.fetch_status == "SEARCH_FAILED:RuntimeError"


def test_prior_atr_gate_precedes_news_search_and_research_cap():
    class CountingSearch:
        name = "COUNTING"

        def __init__(self):
            self.symbols = []

        def search(self, **kwargs):
            self.symbols.append(kwargs["symbol"])
            return []

    search = CountingSearch()
    policy = replace(
        DEFAULT_POLICY,
        discovery=replace(DEFAULT_POLICY.discovery, max_candidates=1),
    )
    snapshots = [
        _snapshot(symbol="MISSING", atr_14=0.0, premarket_volume=4_000_000),
        _snapshot(symbol="EXACT", atr_14=0.40, premarket_volume=3_000_000),
        _snapshot(symbol="HIGH", atr_14=0.50, premarket_volume=2_000_000),
        _snapshot(symbol="HIGH2", atr_14=0.60, premarket_volume=1_000_000),
    ]
    result = run_shadow_pipeline(
        snapshots,
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=policy,
        search_provider=search,
    )

    assert search.symbols == ["HIGH"]
    assert {candidate.snapshot.symbol for candidate in result.candidates} == {
        "MISSING",
        "EXACT",
        "HIGH",
        "HIGH2",
    }
    decisions = {
        candidate.snapshot.symbol: decision
        for candidate, decision in zip(result.candidates, result.decisions)
    }
    assert {
        "PRIOR_ATR_UNRESOLVED",
        "NEWS_RESEARCH_SKIPPED_PRIOR_ATR",
    } <= set(decisions["MISSING"].blockers)
    assert {
        "PRIOR_ATR_PCT_AT_OR_BELOW_FLOOR",
        "NEWS_RESEARCH_SKIPPED_PRIOR_ATR",
    } <= set(decisions["EXACT"].blockers)
    assert "NEWS_RESEARCH_NOT_SELECTED_BY_CAP" in decisions["HIGH2"].blockers


def test_network_research_rechecks_quote_age_after_fetch(monkeypatch):
    class EmptySearch:
        name = "EMPTY"

        def search(self, **kwargs):
            return []

    monkeypatch.setattr(
        "episodic_pivot.pipeline.utc_now",
        lambda: pd.Timestamp("2026-08-24T12:35:00Z").to_pydatetime(),
    )
    result = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        search_provider=EmptySearch(),
    )
    assert "STALE_MARKET_DATA" in result.decisions[0].blockers


def test_entry_window_is_enforced_not_just_written_to_preview():
    late_as_of = "2026-08-24T13:36:00Z"  # 09:36 America/New_York
    result = run_shadow_pipeline(
        [_snapshot(observed_at="2026-08-24T13:35:30Z")],
        as_of=late_as_of,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document()]},
        offline_documents_verified=True,
    )
    assert result.decisions[0].decision == "WATCH"
    assert "ENTRY_WINDOW_EXPIRED" in result.decisions[0].blockers
    assert result.previews == []


def test_offline_evidence_hash_and_labels_are_recomputed_not_trusted():
    mislabeled = _document(
        text_sha256="0" * 64,
        catalyst_types=("REGULATORY_APPROVAL",),
        source_tier="PRIMARY",
    )
    result = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [mislabeled]},
        offline_documents_verified=True,
    )
    archived = next(iter(result.documents_by_candidate.values()))[0]
    assert archived.fetch_status == "INVALID_EVIDENCE_HASH"
    assert result.decisions[0].decision == "WATCH"
    assert result.previews == []


def test_unverified_offline_replay_cannot_create_a_preview():
    result = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document()]},
    )
    archived = next(iter(result.documents_by_candidate.values()))[0]
    assert archived.fetch_status == "UNVERIFIED_REPLAY"
    assert result.decisions[0].decision == "WATCH"
    assert result.previews == []


def test_offline_evidence_manifest_must_match_network_run_digest(tmp_path):
    run_id = "EP-RUN-2026-08-24-source"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    evidence = run_dir / "evidence_by_symbol.json"
    evidence.write_text('{"TEST": []}\n', encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    manifest = run_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "search_provider": "GOOGLE_CSE",
                "safety": {"live_actions_enabled": False},
                "artifacts": {"evidence_by_symbol.json": {"sha256": digest}},
            }
        ),
        encoding="utf-8",
    )
    assert _verify_evidence_manifest(evidence, manifest) == run_id
    evidence.write_text('{"TEST": ["tampered"]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        _verify_evidence_manifest(evidence, manifest)


def test_review_csv_escapes_formula_title_but_json_keeps_raw_audit_value(tmp_path):
    malicious_title = (
        '=HYPERLINK("https://evil.example","Test Systems raises guidance")'
    )
    result = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document(title=malicious_title)]},
        offline_documents_verified=True,
    )
    output = write_run_artifacts(
        result,
        policy=DEFAULT_POLICY,
        output_dir=tmp_path / result.run_id,
    )
    with (output / "research_sizing_preview.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["catalyst_summary"].startswith("'=")
    raw = json.loads(
        (output / "research_sizing_preview.json").read_text(encoding="utf-8")
    )
    assert raw[0]["catalyst_summary"].startswith("=")


def test_empty_and_nonempty_preview_csv_have_identical_headers(tmp_path):
    stageable = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document()]},
        offline_documents_verified=True,
    )
    empty = run_shadow_pipeline(
        [_snapshot(symbol="EMPTY")],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"EMPTY": []},
        offline_documents_verified=True,
    )
    stage_dir = write_run_artifacts(
        stageable,
        policy=DEFAULT_POLICY,
        output_dir=tmp_path / "stageable",
    )
    empty_dir = write_run_artifacts(
        empty,
        policy=DEFAULT_POLICY,
        output_dir=tmp_path / "empty",
    )
    stage_header = (
        (stage_dir / "research_sizing_preview.csv")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    empty_header = (
        (empty_dir / "research_sizing_preview.csv")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert stage_header == empty_header


def test_unknown_press_release_path_is_not_primary_authority():
    assert (
        source_tier("https://stockpromo.example/press-releases/test-systems")
        == "SECONDARY"
    )


def test_page_controlled_canonical_cannot_elevate_source_authority():
    document = _document(
        url="https://stockpromo.example/article/test-systems",
        canonical_url="https://www.businesswire.com/news/home/fake-canonical",
        source_tier="PRIMARY",
    )
    assert normalize_evidence_document(document).source_tier == "SECONDARY"


def test_common_word_company_or_ticker_does_not_establish_identity():
    document = _document(
        title="Retailers target higher revenue",
        text_excerpt=(
            "Retailers target higher revenue as consumers return to stores. " * 12
        ),
    )
    assert not _document_mentions_candidate(
        document, symbol="TGT", company_name="Target Corporation"
    )


def test_article_fetch_rejects_loopback_targets_before_request():
    with pytest.raises(ValueError, match="non-public"):
        _validate_public_http_url("https://127.0.0.1/internal")


def test_article_connection_uses_once_vetted_ip_without_second_dns(monkeypatch):
    dns_calls = []

    def fake_getaddrinfo(host, port, **kwargs):
        dns_calls.append((host, port))
        return [(2, 1, 6, "", ("93.184.216.34", port))]

    connection_targets = []
    sentinel = object()

    def fake_create_connection(target, timeout, **kwargs):
        connection_targets.append(target)
        return sentinel

    monkeypatch.setattr("episodic_pivot.news.socket.getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(
        "urllib3.connection.connection.create_connection", fake_create_connection
    )
    vetted = _validate_public_http_url("https://example.com/article")
    connection = _PinnedHTTPSConnection(
        host="example.com", port=443, pinned_ip=vetted[0]
    )
    assert connection._new_conn() is sentinel
    assert dns_calls == [("example.com", 443)]
    assert connection_targets == [("93.184.216.34", 443)]
    assert connection.host == "example.com"  # TLS SNI/certificate identity.


def test_article_body_reader_enforces_absolute_deadline_during_slow_drip(monkeypatch):
    clock = iter([0.0, 0.4, 1.1])
    monkeypatch.setattr(
        "episodic_pivot.news.monotonic_time.monotonic", lambda: next(clock)
    )

    class Raw:
        def read1(self, size):
            return b"x"

    class Sock:
        def settimeout(self, timeout):
            self.timeout = timeout

    response = SimpleNamespace(_fp=Raw(), _connection=SimpleNamespace(sock=Sock()))
    with pytest.raises(TimeoutError, match="wall-clock"):
        _read_response_body(response, max_bytes=100, deadline_monotonic=1.0)


def test_halt_telemetry_unknown_is_not_treated_as_clear():
    assert _halt_status(float("nan")) == ("UNKNOWN", None)
    assert _halt_status(-1)[0] == "UNKNOWN"
    assert _halt_status(0) == ("NOT_HALTED", 0.0)
    assert _halt_status(1)[0] == "GENERAL_HALT"
    assert _halt_status(2)[0] == "VOLATILITY_HALT"


def test_scanner_sample_selection_round_robins_sources():
    keys = {
        "TOP_PERC_GAIN": ["G1", "G2", "G3"],
        "HOT_BY_VOLUME": ["V1", "V2"],
        "MOST_ACTIVE": ["A1", "A2"],
    }
    assert _round_robin_keys(keys, list(keys), 5) == [
        "G1",
        "V1",
        "A1",
        "G2",
        "V2",
    ]


def test_stale_premarket_bar_timestamp_blocks_stageability():
    result = run_shadow_pipeline(
        [_snapshot(premarket_metrics_at="2026-08-24T12:00:00Z")],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document()]},
        offline_documents_verified=True,
    )
    assert result.decisions[0].decision == "WATCH"
    assert "STALE_PREMARKET_METRICS" in result.decisions[0].blockers


def test_previous_close_basis_mismatch_blocks_phantom_gap():
    result = run_shadow_pipeline(
        [_snapshot(quote_previous_close=5.0)],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document()]},
        offline_documents_verified=True,
    )
    assert result.decisions[0].decision == "WATCH"
    assert "PRIOR_CLOSE_BASIS_MISMATCH" in result.decisions[0].blockers


def test_unknown_halt_or_unresolved_contract_cannot_be_stageable():
    result = run_shadow_pipeline(
        [
            _snapshot(
                halt_status="UNKNOWN",
                contract_con_id=None,
                primary_exchange="",
            )
        ],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document()]},
        offline_documents_verified=True,
    )
    assert result.decisions[0].decision == "WATCH"
    assert {
        "HALT_STATUS_UNKNOWN",
        "UNRESOLVED_IB_CONTRACT",
        "MISSING_PRIMARY_EXCHANGE",
    }.issubset(result.decisions[0].blockers)
    assert result.previews == []


def test_ask_above_25_percent_routes_to_delayed_even_if_last_is_below_cap():
    result = run_shadow_pipeline(
        [_snapshot(last=12.40, bid=12.99, ask=13.00, premarket_vwap=12.75)],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document()]},
        offline_documents_verified=True,
    )
    assert result.decisions[0].decision == "WATCH"
    assert result.decisions[0].setup_type == "EXTENDED_GAP_DEP_CANDIDATE"
    assert "GAP_TOO_EXTENDED_FOR_IMMEDIATE_ENTRY" in result.decisions[0].blockers
    assert result.previews == []


def test_publication_after_decision_and_entry_window_cannot_confirm():
    decision_at = "2026-08-24T13:34:00Z"  # 09:34 ET.
    result = assess_catalyst(
        [
            _document(
                published_at="2026-08-24T13:36:00Z",
                retrieved_at="2026-08-24T13:36:00Z",
            )
        ],
        decision_at=decision_at,
        policy=DEFAULT_POLICY.news,
        symbol="TEST",
        company_name="Test Systems Inc.",
        first_trigger_at=FIRST_TRIGGER,
    )
    assert result.status == "UNCONFIRMED"
    assert "POST_DECISION_RETRIEVAL" in result.reason_codes


def test_two_secondary_sources_about_different_catalysts_do_not_corroborate():
    earnings = _document(
        url="https://alpha.example/test-earnings",
        canonical_url="https://alpha.example/test-earnings",
    )
    product_text = (
        "Test Systems launches a new product for enterprise customers. The commercial "
        "launch includes pricing, distribution partners, availability, and management's "
        "description of the addressable market and product roadmap. "
    ) * 5
    product = _document(
        title="Test Systems launches enterprise product",
        url="https://beta.example/test-product",
        canonical_url="https://beta.example/test-product",
        text_excerpt=product_text,
    )
    assessment = assess_catalyst(
        [earnings, product],
        decision_at=AS_OF,
        policy=DEFAULT_POLICY.news,
        symbol="TEST",
        company_name="Test Systems Inc.",
        first_trigger_at=FIRST_TRIGGER,
    )
    assert assessment.status == "WATCH"
    assert "PRIMARY_SOURCE_NOT_VERIFIED" in assessment.reason_codes


def test_trial_enrollment_is_watch_not_positive_clinical_data():
    text = (
        "Test Systems announced the first patient enrolled in a phase 2 clinical trial. "
        "The study has initiated enrollment, but the company reported no efficacy data, "
        "endpoint result, statistical result, or regulatory approval. "
    ) * 5
    result = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={
            "TEST": [
                _document(
                    title="Test Systems enrolls first patient in phase 2 trial",
                    text_excerpt=text,
                )
            ]
        },
        offline_documents_verified=True,
    )
    assert result.decisions[0].decision == "WATCH"
    assert result.decisions[0].catalyst.catalyst_type == "CLINICAL_TRIAL_UPDATE"
    assert "CATALYST_NOT_CONFIRMED" in result.decisions[0].blockers


def test_negative_trial_results_cannot_be_labeled_positive_clinical_data():
    text = (
        "Test Systems reported that trial results showed no statistically significant "
        "difference from placebo and failed to achieve the primary endpoint. Management "
        "will review the dataset before deciding whether to continue the program. "
    ) * 5
    result = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={
            "TEST": [
                _document(
                    title="Test Systems trial fails primary endpoint",
                    text_excerpt=text,
                )
            ]
        },
        offline_documents_verified=True,
    )
    assert result.decisions[0].decision == "WATCH"
    assert "CLINICAL_OR_REGULATORY_FAILURE" in result.decisions[0].blockers
    assert result.previews == []


def test_negated_raised_guidance_cannot_confirm_guidance_catalyst():
    text = (
        "Test Systems has not raised guidance and repeated its prior annual outlook. "
        "Revenue was unchanged, management announced no new commercial milestone, and "
        "the company declined to provide a higher forecast. "
    ) * 5
    result = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={
            "TEST": [
                _document(
                    title="Test Systems repeats prior outlook",
                    text_excerpt=text,
                )
            ]
        },
        offline_documents_verified=True,
    )
    assert result.decisions[0].decision == "WATCH"
    assert result.decisions[0].catalyst.catalyst_type != "EARNINGS_GUIDANCE"
    assert result.previews == []


def test_two_company_roundup_does_not_assign_other_issuers_catalyst():
    text = (
        "Test Systems shares traded higher in early activity after an otherwise quiet "
        "session. The company made no new announcement and repeated its existing outlook. "
        "Test Systems' filing contained routine administrative disclosures and no change "
        "to its expected business trajectory. A separate market update covered unrelated "
        "issuers. Other Holdings raised guidance after reporting revenue above expectations "
        "and a record backlog, while a third company announced a product launch. "
    )
    document = _document(
        title="Morning movers include Test Systems and Other Holdings",
        url="https://www.reuters.com/markets/morning-movers-test-other",
        canonical_url="https://www.reuters.com/markets/morning-movers-test-other",
        text_excerpt=text,
    )
    assessment = assess_catalyst(
        [document],
        decision_at=AS_OF,
        policy=DEFAULT_POLICY.news,
        symbol="TEST",
        company_name="Test Systems Inc.",
        first_trigger_at=FIRST_TRIGGER,
    )
    assert assessment.status == "WATCH"
    assert assessment.catalyst_type == "UNCLASSIFIED"


def test_preceding_other_company_sentence_cannot_confirm_candidate():
    text = (
        "Other Holdings raised guidance after beating estimates and reporting record "
        "revenue. Test Systems rose in premarket trading on no company announcement. "
        "Test Systems repeated its old outlook and disclosed no commercial milestone. "
    ) * 3
    document = _document(
        title="Premarket roundup for Other Holdings and Test Systems",
        url="https://www.reuters.com/markets/roundup-other-test",
        canonical_url="https://www.reuters.com/markets/roundup-other-test",
        text_excerpt=text,
    )
    assessment = assess_catalyst(
        [document],
        decision_at=AS_OF,
        policy=DEFAULT_POLICY.news,
        symbol="TEST",
        company_name="Test Systems Inc.",
        first_trigger_at=FIRST_TRIGGER,
    )
    assert assessment.status == "WATCH"
    assert assessment.catalyst_type == "UNCLASSIFIED"


def test_ordinary_quarterly_results_are_not_mislabeled_as_raised_guidance():
    text = (
        "Test Systems published quarterly results. Revenue was nearly unchanged and the "
        "company repeated its prior outlook without raising guidance or announcing a new "
        "commercial milestone. The release contains the standard income statement and "
        "balance sheet tables for the quarter. "
    ) * 3
    document = _document(
        title="Test Systems publishes ordinary quarterly results",
        text_excerpt=text,
    )
    result = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [document]},
        offline_documents_verified=True,
    )
    assert result.decisions[0].catalyst.catalyst_type == "EARNINGS"
    assert result.decisions[0].decision == "WATCH"
    assert "CATALYST_NOT_CONFIRMED" in result.decisions[0].blockers


def test_eps_substring_in_keeps_or_steps_is_not_an_earnings_catalyst():
    text = (
        "Test Systems keeps taking steps toward a multi-year product vision. Revenue grew "
        "$1 million in an internal estimate, but the article is a retrospective profile "
        "and contains no new corporate announcement. "
    ) * 4
    document = _document(
        title="Test Systems keeps taking steps on product vision",
        text_excerpt=text,
    )
    assessment = assess_catalyst(
        [document],
        decision_at=AS_OF,
        policy=DEFAULT_POLICY.news,
        symbol="TEST",
        company_name="Test Systems Inc.",
        first_trigger_at=FIRST_TRIGGER,
    )
    assert assessment.status == "WATCH"
    assert assessment.catalyst_type == "UNCLASSIFIED"


def test_sizing_failure_is_propagated_into_candidate_decision():
    result = run_shadow_pipeline(
        [_snapshot(prior_two_day_low=7.0)],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document()]},
        offline_documents_verified=True,
    )
    assert result.decisions[0].decision == "WATCH"
    assert "SIZING:STOP_DISTANCE_TOO_WIDE" in result.decisions[0].blockers
    assert result.previews == []


def test_gap_over_25_percent_routes_to_delayed_watch():
    result = run_shadow_pipeline(
        [
            _snapshot(
                last=13.0,
                bid=12.98,
                ask=13.0,
                premarket_vwap=12.9,
                premarket_high=13.1,
            )
        ],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document()]},
        offline_documents_verified=True,
    )
    assert result.decisions[0].setup_type == "EXTENDED_GAP_DEP_CANDIDATE"
    assert result.previews == []


def test_policy_cannot_enable_live_actions():
    try:
        replace(DEFAULT_POLICY, live_actions_enabled=True)
    except ValueError as exc:
        assert "cannot enable live actions" in str(exc)
    else:
        raise AssertionError("live policy unexpectedly constructed")


def test_exact_extension_boundaries_and_negative_movers_are_researchable():
    snapshots = [
        _snapshot(symbol="G20", last=12.0, bid=11.99, ask=12.0),
        _snapshot(symbol="G2001", last=12.001, bid=12.0, ask=12.01),
        _snapshot(symbol="G25", last=12.5, bid=12.49, ask=12.5),
        _snapshot(symbol="G2501", last=12.501, bid=12.50, ask=12.51),
        _snapshot(symbol="DOWN", last=9.0, bid=8.99, ask=9.0, premarket_vwap=9.0),
    ]
    candidates = nominate_candidates(snapshots, as_of=AS_OF, policy=DEFAULT_POLICY)
    by_symbol = {item.snapshot.symbol: item for item in candidates}
    assert "EXTENDED_GAP" not in by_symbol["G20"].discovery_warnings
    assert "EXTENDED_GAP" in by_symbol["G2001"].discovery_warnings
    assert "DELAYED_EP_PREFERRED" not in by_symbol["G25"].discovery_warnings
    assert "DELAYED_EP_PREFERRED" in by_symbol["G2501"].discovery_warnings
    assert "BEARISH_RESEARCH_ONLY" in by_symbol["DOWN"].discovery_warnings


def test_guidance_cut_negative_mover_routes_to_bearish_research_only():
    text = (
        "Test Systems lowers guidance after demand weakened. The company cuts guidance, "
        "reduces its outlook, and described a material deterioration in revenue and margins. "
    ) * 5
    result = run_shadow_pipeline(
        [_snapshot(last=9.0, bid=8.99, ask=9.0, premarket_vwap=9.0)],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={
            "TEST": [
                _document(
                    title="Test Systems cuts guidance",
                    text_excerpt=text,
                )
            ]
        },
        offline_documents_verified=True,
    )
    decision = result.decisions[0]
    assert decision.decision == "WATCH"
    assert decision.setup_type == "BEARISH_EP_RESEARCH"
    assert "BEARISH_EXECUTION_NOT_IMPLEMENTED" in decision.blockers
    assert result.previews == []


def test_targeted_ibkr_input_preserves_symbol_identity_and_session(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(
        json.dumps(
            {
                "snapshots": [
                    {
                        "symbol": "TEST",
                        "screen_exchange": "NASDAQ",
                        "saved_screen_id": "screen-1",
                        "target_session_date": "2026-08-24",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rows, session = _load_target_rows(path)
    assert session == "2026-08-24"
    assert rows == [
        {
            "symbol": "TEST",
            "expected_primary_exchange": "NASDAQ",
            "source_screen_id": "screen-1",
        }
    ]


def test_ibkr_five_minute_bars_record_first_actual_trigger_timestamp():
    bars = [
        SimpleNamespace(
            date=pd.Timestamp("2026-08-24T04:00:00", tz="America/New_York"),
            open=10.0,
            high=10.15,
            low=10.0,
            close=10.10,
            volume=50_000,
            barCount=100,
        ),
        SimpleNamespace(
            date=pd.Timestamp("2026-08-24T04:05:00", tz="America/New_York"),
            open=10.10,
            high=10.25,
            low=10.10,
            close=10.20,
            volume=60_000,
            barCount=120,
        ),
    ]
    metrics = _premarket_metrics(bars, date(2026, 8, 24), previous_close=10.0)
    assert metrics["first_trigger_at"] == "2026-08-24T08:05:00Z"


def _normalized_field(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def test_research_sizing_contract_is_incompatible_with_live_order_rows():
    research = {_normalized_field(item.name) for item in fields(ResearchSizingPreview)}
    live = {
        _normalized_field(name)
        for name in (
            "Symbol",
            "Action",
            "Quantity",
            "Order_Type",
            "TIF",
            "Limit_Price",
            "Manual_Limit",
            "Strategy_Ref",
            "Trade_Direction",
            "Risk_Amt",
            "Risk_Bps",
            "Approval",
            "Execute_On",
            "Transmit",
        )
    }
    assert research & live == {"symbol"}
    assert not hasattr(ep_schema, "StagingPreview")


def test_every_research_preview_safety_sentinel_is_immutable():
    result = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document()]},
        offline_documents_verified=True,
    )
    preview = result.previews[0]
    mutations = (
        {"preview_only": False},
        {"executable": True},
        {"broker_route": "SMART"},
        {"order_submission_allowed": True},
        {"human_review_required": False},
        {"production_eligible": True},
        {"live_actions_enabled": True},
    )
    for mutation in mutations:
        with pytest.raises(ValueError):
            replace(preview, **mutation)

    payload = preview.to_dict()
    payload.pop("record_type")
    payload["action"] = "BUY"
    with pytest.raises(TypeError):
        ResearchSizingPreview(**payload)


def test_manifest_rejects_order_shaped_preview_before_creating_directory(tmp_path):
    result = RunResult(
        run_id="EP-RUN-invalid",
        generated_at=AS_OF,
        previews=[SimpleNamespace(symbol="TEST", action="BUY", quantity=100)],
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(TypeError, match="ResearchSizingPreview"):
        write_run_artifacts(result, policy=DEFAULT_POLICY, output_dir=output)
    assert not output.exists()


def test_historical_proxy_uses_prior_volume_and_suppresses_repeat_events():
    dates = _trading_dates("2025-01-02", periods=300)
    close = pd.Series(10.0, index=range(len(dates)))
    open_ = close.copy()
    high = close * 1.02
    low = close * 0.98
    volume = pd.Series(1_000_000.0, index=range(len(dates)))

    first = 150
    second = 200
    for index in (first, second):
        open_.iloc[index] = 11.5
        high.iloc[index] = 12.5
        low.iloc[index] = 11.2
        close.iloc[index] = 12.0
        volume.iloc[index] = 3_000_000
        # Reset the following bar so the synthetic series remains neglected.
        open_.iloc[index + 1] = 10.0
        high.iloc[index + 1] = 10.2
        low.iloc[index + 1] = 9.8
        close.iloc[index + 1] = 10.0

    frame = pd.DataFrame(
        {
            "date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )
    result = study_ticker(
        "SYN",
        frame,
        policy=replace(DEFAULT_POLICY.historical, min_prior_atr_pct=0.0),
        include_outcomes=True,
    )
    assert len(result.events) == 1
    event = result.events.iloc[0]
    assert event["date"] == dates[first]
    assert event["prior_addv_63"] == 10_000_000
    assert event["event_rvol_20"] == 3.0
    assert result.counts["volume_confirmed_ex_post"] == 2
    assert result.counts["first_confirmed_in_126"] == 1
    assert event["next_open_to_close_1d_pct"] == 0.0


@pytest.mark.parametrize(
    ("prior_range", "expected_events"),
    [
        (1.0, 0),  # exactly 4.0% is excluded: the rule is strictly greater than 4%.
        (1.125, 1),
    ],
)
def test_historical_atr_gate_uses_strict_prior_session_threshold(
    prior_range, expected_events
):
    dates = _trading_dates("2025-01-02", periods=170)
    close = pd.Series(25.0, index=range(len(dates)))
    open_ = close.copy()
    high = close + prior_range / 2.0
    low = close - prior_range / 2.0
    volume = pd.Series(1_000_000.0, index=range(len(dates)))
    event_index = 150
    open_.iloc[event_index] = 28.0
    high.iloc[event_index] = 29.0
    low.iloc[event_index] = 27.5
    close.iloc[event_index] = 28.5
    volume.iloc[event_index] = 3_000_000.0

    frame = pd.DataFrame(
        {
            "date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )
    result = study_ticker("ATR", frame, policy=DEFAULT_POLICY.historical)

    assert result.counts["open_observable"] == 1
    assert result.counts["strict_with_neglect_pre_atr"] == 1
    assert result.counts["strict_with_neglect_and_prior_atr"] == expected_events
    assert len(result.events) == expected_events
    if expected_events:
        assert result.events.iloc[0]["prior_atr_pct_14"] == pytest.approx(4.5)
        assert bool(result.events.iloc[0]["prior_atr_window_clean"])


def test_historical_atr_gate_cannot_see_event_day_range():
    dates = _trading_dates("2025-01-02", periods=170)
    close = pd.Series(25.0, index=range(len(dates)))
    open_ = close.copy()
    high = close + 0.25
    low = close - 0.25
    volume = pd.Series(1_000_000.0, index=range(len(dates)))
    event_index = 150
    open_.iloc[event_index] = 28.0
    high.iloc[event_index] = 40.0
    low.iloc[event_index] = 20.0
    close.iloc[event_index] = 29.0
    volume.iloc[event_index] = 3_000_000.0

    frame = pd.DataFrame(
        {
            "date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )
    result = study_ticker("NOLEAK", frame, policy=DEFAULT_POLICY.historical)

    assert result.counts["open_observable"] == 1
    assert result.counts["strict_with_neglect_pre_atr"] == 1
    assert result.counts["strict_with_neglect_and_prior_atr"] == 0
    assert result.events.empty


def test_historical_atr_gate_requires_consecutive_nyse_source_sessions():
    dates = _trading_dates("2025-01-02", periods=170)
    close = pd.Series(25.0, index=range(len(dates)))
    frame = pd.DataFrame(
        {
            "date": dates,
            "Open": close.copy(),
            "High": close + 0.625,
            "Low": close - 0.625,
            "Close": close.copy(),
            "Volume": 1_000_000.0,
        }
    )
    event_index = 150
    frame.loc[event_index, ["Open", "High", "Low", "Close", "Volume"]] = [
        28.0,
        29.0,
        27.5,
        28.5,
        3_000_000.0,
    ]
    frame = frame.drop(index=event_index - 5).reset_index(drop=True)

    result = study_ticker("CALGAP", frame, policy=DEFAULT_POLICY.historical)

    assert result.counts["strict_with_neglect_pre_atr"] == 1
    assert result.counts["strict_prior_atr_calendar_incomplete"] == 1
    assert result.counts["strict_with_neglect_and_prior_atr"] == 0
    assert result.events.empty


def test_low_atr_confirmed_gap_still_resets_first_event_clock():
    dates = _trading_dates("2025-01-02", periods=300)
    close = pd.Series(25.0, index=range(len(dates)))
    open_ = close.copy()
    high = close + 0.25
    low = close - 0.25
    volume = pd.Series(1_000_000.0, index=range(len(dates)))
    first, second = 150, 200
    for event_index in (first, second):
        open_.iloc[event_index] = 28.0
        high.iloc[event_index] = 29.0
        low.iloc[event_index] = 27.5
        close.iloc[event_index] = 28.5
        volume.iloc[event_index] = 3_000_000.0
        open_.iloc[event_index + 1] = 25.0
        high.iloc[event_index + 1] = 25.25
        low.iloc[event_index + 1] = 24.75
        close.iloc[event_index + 1] = 25.0
    high.iloc[second - 14 : second] = 25.625
    low.iloc[second - 14 : second] = 24.375

    frame = pd.DataFrame(
        {
            "date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )
    result = study_ticker("CLOCK", frame, policy=DEFAULT_POLICY.historical)

    assert result.counts["volume_confirmed_ex_post"] == 2
    assert result.counts["first_confirmed_in_126"] == 1
    assert result.counts["strict_with_neglect_pre_atr"] == 1
    assert result.counts["strict_with_neglect_and_prior_atr"] == 0
    assert result.events.empty


def test_future_price_mutation_cannot_change_event_inclusion_or_quality():
    dates = _trading_dates("2025-01-02", periods=180)
    close = pd.Series(25.0, index=range(len(dates)))
    frame = pd.DataFrame(
        {
            "date": dates,
            "Open": close.copy(),
            "High": close + 0.625,
            "Low": close - 0.625,
            "Close": close.copy(),
            "Volume": 1_000_000.0,
        }
    )
    event_index = 150
    frame.loc[event_index, ["Open", "High", "Low", "Close", "Volume"]] = [
        28.0,
        29.0,
        27.5,
        28.5,
        3_000_000.0,
    ]
    mutated = frame.copy()
    mutated.loc[event_index + 2, ["Open", "High", "Low", "Close"]] = [
        60.0,
        65.0,
        55.0,
        60.0,
    ]

    baseline = study_ticker("PREFIX", frame, policy=DEFAULT_POLICY.historical)
    changed = study_ticker("PREFIX", mutated, policy=DEFAULT_POLICY.historical)

    assert baseline.events[["date", "prior_atr_pct_14", "prior_window_clean"]].equals(
        changed.events[["date", "prior_atr_pct_14", "prior_window_clean"]]
    )


def test_legitimate_extreme_event_gap_is_not_censored_as_bad_price_data():
    dates = _trading_dates("2025-01-02", periods=170)
    close = pd.Series(25.0, index=range(len(dates)))
    frame = pd.DataFrame(
        {
            "date": dates,
            "Open": close.copy(),
            "High": close + 0.625,
            "Low": close - 0.625,
            "Close": close.copy(),
            "Volume": 1_000_000.0,
        }
    )
    event_index = 150
    frame.loc[event_index, ["Open", "High", "Low", "Close", "Volume"]] = [
        40.0,
        41.0,
        24.5,
        25.0,
        3_000_000.0,
    ]

    result = study_ticker("EXTREME", frame, policy=DEFAULT_POLICY.historical)

    assert len(result.events) == 1
    assert result.events.iloc[0]["gap_pct"] == pytest.approx(60.0)
    assert bool(result.events.iloc[0]["prior_window_clean"])
    assert bool(result.events.iloc[0]["basis_review_cleared"])
    assert not bool(result.events.iloc[0]["event_half_double_review_required"])
    assert result.counts["anomalies"] == 1


def test_coherent_doubling_event_is_kept_but_flagged_for_basis_review():
    dates = _trading_dates("2025-01-02", periods=170)
    close = pd.Series(25.0, index=range(len(dates)))
    frame = pd.DataFrame(
        {
            "date": dates,
            "Open": close.copy(),
            "High": close + 0.625,
            "Low": close - 0.625,
            "Close": close.copy(),
            "Volume": 1_000_000.0,
        }
    )
    event_index = 150
    frame.loc[event_index, ["Open", "High", "Low", "Close", "Volume"]] = [
        50.0,
        51.0,
        49.0,
        50.0,
        3_000_000.0,
    ]

    result = study_ticker("DOUBLE", frame, policy=DEFAULT_POLICY.historical)

    assert len(result.events) == 1
    assert result.events.iloc[0]["gap_pct"] == pytest.approx(100.0)
    assert bool(result.events.iloc[0]["prior_window_clean"])
    assert not bool(result.events.iloc[0]["basis_review_cleared"])
    assert bool(result.events.iloc[0]["event_half_double_review_required"])


def test_historical_news_labeler_is_blind_to_forward_outcomes():
    raw = pd.DataFrame(
        {
            "ticker": ["TEST"],
            "date": [pd.Timestamp("2026-08-24")],
            "previous_session": [pd.Timestamp("2026-08-21")],
            "previous_close": [10.0],
            "gap_pct": [10.0],
            "prior_atr_pct_14": [5.0],
            "excess_next_open_to_close_20d_pct": [99.0],
            "mfe_60d_pct": [150.0],
        }
    )
    blinded = blind_events(raw)
    assert "event_id" in blinded
    assert not any(
        "excess" in column or column.startswith("mfe_") for column in blinded
    )


def _sec_payload(accepted_at: str) -> dict:
    return {
        "filings": {
            "recent": {
                "accessionNumber": ["0000123456-26-000001"],
                "filingDate": ["2026-08-24"],
                "reportDate": ["2026-06-30"],
                "acceptanceDateTime": [accepted_at],
                "form": ["8-K"],
                "items": ["2.02,9.01"],
                "primaryDocument": ["results.htm"],
                "primaryDocDescription": ["Quarterly financial results"],
            }
        }
    }


@pytest.mark.parametrize(
    ("accepted_at", "timing_status"),
    [
        ("2026-08-24T13:26:59Z", "PREOPEN_SEC_ASSUMED_PUBLIC"),
        ("2026-08-24T13:27:01Z", "POST_OPEN_CONTEXT"),
    ],
)
def test_sec_public_time_uses_conservative_three_minute_boundary(
    accepted_at, timing_status
):
    event = blind_events(
        pd.DataFrame(
            {
                "ticker": ["TEST"],
                "date": ["2026-08-24"],
                "previous_session": ["2026-08-21"],
            }
        )
    ).iloc[0]
    filings = normalize_sec_submissions(
        [("CIK0000123456.json", _sec_payload(accepted_at))],
        ticker="TEST",
        cik=123456,
    )
    bound = bind_sec_evidence(filings, event)
    assert bound.iloc[0]["timing_status"] == timing_status
    assert "EARNINGS_GUIDANCE" in bound.iloc[0]["event_types"]


def test_sec_timing_uses_actual_xnys_early_close():
    event = blind_events(
        pd.DataFrame(
            {
                "ticker": ["TEST"],
                "date": ["2026-11-30"],
                "previous_session": ["2026-11-27"],
            }
        )
    ).iloc[0]
    payload = _sec_payload("2026-11-27T18:05:00Z")
    payload["filings"]["recent"]["filingDate"] = ["2026-11-27"]
    filings = normalize_sec_submissions(
        [("CIK0000123456.json", payload)],
        ticker="TEST",
        cik=123456,
    )
    bound = bind_sec_evidence(filings, event)
    assert bound.iloc[0]["timing_status"] == "PREOPEN_SEC_ASSUMED_PUBLIC"


def test_sec_synthetic_midnight_timestamp_cannot_prove_causality():
    event = blind_events(
        pd.DataFrame(
            {
                "ticker": ["TEST"],
                "date": ["2000-08-24"],
                "previous_session": ["2000-08-23"],
            }
        )
    ).iloc[0]
    filings = normalize_sec_submissions(
        [("old.json", _sec_payload("2000-08-24T04:00:00Z"))],
        ticker="TEST",
        cik=123456,
    )
    bound = bind_sec_evidence(filings, event)
    assert bound.iloc[0]["accepted_at_quality"] == "DATE_ONLY_OR_SYNTHETIC_MIDNIGHT"
    assert bound.iloc[0]["timing_status"] == "TIMING_UNRESOLVED"


def test_item_101_is_material_agreement_not_automatically_customer_contract():
    labels = classify_event_text("", form="8-K", items="1.01,9.01")
    assert "MATERIAL_AGREEMENT_UNCLASSIFIED" in labels
    assert "PRODUCT_CUSTOMER_CONTRACT" not in labels


def test_fmp_timezone_naive_news_is_deduped_timing_and_direction_unresolved():
    payload = [
        {
            "symbol": "TEST",
            "publishedDate": "2026-08-24 08:00:00",
            "title": "Test Systems reports quarterly results and raises guidance",
            "text": "Test Systems reports financial results and raises guidance.",
            "url": url,
            "site": "Example Wire",
        }
        for url in ("https://one.example/story", "https://two.example/copy")
    ]
    news = normalize_fmp_news(payload, ticker="TEST", company_name="Test Systems Inc.")
    event = blind_events(
        pd.DataFrame(
            {
                "ticker": ["TEST"],
                "date": ["2026-08-24"],
                "previous_session": ["2026-08-21"],
            }
        )
    ).iloc[0]
    bound = bind_fmp_evidence(news, event)
    assert len(bound) == 1
    assert bound.iloc[0]["published_at_quality"] == "PROVIDER_TIMEZONE_UNKNOWN"
    assert bound.iloc[0]["timing_status"] == "TIMING_UNRESOLVED"
    assert bound.iloc[0]["trajectory_signal"] == "TRAJECTORY_UNRESOLVED"


def test_low_signal_holdings_story_stays_context_only():
    news = normalize_fmp_news(
        [
            {
                "symbol": "TER",
                "publishedDate": "2026-07-29 08:00:00",
                "title": "Arrowstreet Capital Lowers Holdings in Teradyne, Inc.",
                "text": (
                    "The position change was disclosed in Securities and Exchange "
                    "Commission filings. Teradyne shares were unchanged."
                ),
                "url": "https://example.com/holdings-update",
                "site": "Example Aggregator",
            }
        ],
        ticker="TER",
        company_name="Teradyne Inc.",
    )
    assert news.iloc[0]["source_quality"] == "LOW_SIGNAL_HOLDINGS_UPDATE"
    assert "LEGAL_INVESTIGATION" not in news.iloc[0]["event_types"]
    assert news.iloc[0]["trajectory_signal"] == "TRAJECTORY_UNRESOLVED"

    events = blind_events(
        pd.DataFrame(
            {
                "ticker": ["TER"],
                "date": ["2026-07-29"],
                "previous_session": ["2026-07-28"],
            }
        )
    )
    bound = bind_fmp_evidence(news, events.iloc[0])
    provider = pd.DataFrame(
        [
            {
                "event_id": events.iloc[0]["event_id"],
                "provider": "FMP_STOCK_NEWS",
                "status": "FETCHED",
            }
        ]
    )
    summary = summarize_event_evidence(events, bound, provider)
    assert summary.iloc[0]["evidence_posture"] == "CONTEXT_ONLY_NOT_CAUSAL"
    assert summary.iloc[0]["primary_event_type"] == "CONTEXT_ONLY"
    assert summary.iloc[0]["trajectory_posture"] == "TRAJECTORY_UNRESOLVED"
    assert summary.iloc[0]["fmp_articles"] == 0
    assert summary.iloc[0]["fmp_articles_raw"] == 1


def test_historical_fmp_direction_is_disabled_and_law_firm_spam_is_low_signal():
    news = normalize_fmp_news(
        [
            {
                "symbol": "NVAX",
                "publishedDate": "2024-11-11 07:00:00",
                "title": "FDA removes clinical hold and Novavax resumes testing",
                "text": "The regulator cleared the company to resume its Phase 3 trial.",
                "url": "https://example.com/clinical-hold-removed",
                "site": "Example Wire",
            },
            {
                "symbol": "NVAX",
                "publishedDate": "2024-11-11 08:30:00",
                "title": (
                    "Opportunity to Lead Novavax Securities Fraud Lawsuit Before "
                    "Upcoming Deadline"
                ),
                "text": "Investors and purchasers may seek appointment as lead plaintiff.",
                "url": "https://example.com/deadline-solicitation",
                "site": "GlobeNewswire",
            },
            {
                "symbol": "NVAX",
                "publishedDate": "2024-11-11 08:00:00",
                "title": "Investigation commenced on behalf of Novavax shareholders",
                "text": "Shareholders who lost money should contact the law firm.",
                "url": "https://example.com/legal-solicitation",
                "site": "Example Wire",
            },
        ],
        ticker="NVAX",
        company_name="Novavax Inc.",
    ).set_index("url")
    resolved = news.loc["https://example.com/clinical-hold-removed"]
    solicitation = news.loc["https://example.com/legal-solicitation"]
    deadline = news.loc["https://example.com/deadline-solicitation"]
    assert resolved["trajectory_signal"] == "TRAJECTORY_UNRESOLVED"
    assert solicitation["source_quality"] == "LOW_SIGNAL_LEGAL_SOLICITATION"
    assert deadline["source_quality"] == "LOW_SIGNAL_LEGAL_SOLICITATION"
    assert deadline["trajectory_signal"] == "TRAJECTORY_UNRESOLVED"


@pytest.mark.parametrize(
    "title",
    [
        "Test Systems did not beat estimates",
        "Test Systems reports no FDA approval",
        "Test Systems has not raised guidance",
    ],
)
def test_historical_trajectory_phrases_are_negation_aware(title):
    news = normalize_fmp_news(
        [
            {
                "symbol": "TEST",
                "publishedDate": "2026-08-24 08:00:00",
                "title": title,
                "text": "Test Systems provided an update.",
                "url": "https://example.com/negated-claim",
                "site": "Example Wire",
            }
        ],
        ticker="TEST",
        company_name="Test Systems Inc.",
    )
    assert news.iloc[0]["trajectory_signal"] == "TRAJECTORY_UNRESOLVED"


def test_resolved_legal_charges_are_not_adverse():
    assert (
        classify_trajectory(
            "Fraud charges in the case are dismissed",
            ("LEGAL_INVESTIGATION",),
            infer_structural_adverse=False,
        )
        == "TRAJECTORY_UNRESOLVED"
    )


@pytest.mark.parametrize(
    "title",
    [
        "Contact Levi & Korsinsky by May 12 Deadline to Join Class Action",
        "Kahn Swick & Foti Reminds Investors of Lead Plaintiff Deadline",
        "Kirby McInerney LLP Announces Investigation Into Securities Fraud",
        "Gainey McKenna & Egleston Announces a Class Action Lawsuit Has Been Filed",
        "Holzer & Holzer Investigation Alert: Click Here to Learn More",
        "Wohl & Fruchter Investigates Potential Securities Fraud",
        "Scott+Scott Attorneys at Law LLP: CLICK HERE TO LEARN MORE",
        "Faruqi & Faruqi, LLP Reminds Investors of #ClassAction",
    ],
)
def test_residual_law_firm_calls_to_action_are_low_signal(title):
    quality = classify_fmp_source_quality(
        title,
        "If you suffered a loss, contact counsel to seek a potential recovery.",
        publisher="Accesswire",
    )
    assert quality == "LOW_SIGNAL_LEGAL_SOLICITATION"


def test_unrelated_company_bankruptcy_is_not_bound_to_candidate_context():
    news = normalize_fmp_news(
        [
            {
                "symbol": "GME",
                "publishedDate": "2023-11-29 10:18:38",
                "title": "GameStop shares attract renewed trader interest",
                "text": "Bed Bath and Beyond previously filed for bankruptcy.",
                "url": "https://example.com/roundup",
                "site": "Example News",
            }
        ],
        ticker="GME",
        company_name="GameStop Corp.",
    )
    assert "DISTRESS_RESTRUCTURING" not in news.iloc[0]["event_types"]
    assert news.iloc[0]["trajectory_signal"] == "TRAJECTORY_UNRESOLVED"


def test_same_sentence_other_company_bankruptcy_is_not_attributed_to_candidate():
    news = normalize_fmp_news(
        [
            {
                "symbol": "ULCC",
                "publishedDate": "2024-10-04 11:31:16",
                "title": (
                    "Frontier, JetBlue Stocks Soar on Report of Spirit Airlines "
                    "Bankruptcy Talks"
                ),
                "text": (
                    "Shares of Frontier Group Holdings (ULCC) took off on a report "
                    "that rival Spirit Airlines is considering a bankruptcy filing."
                ),
                "url": "https://example.com/frontier-spirit-report",
                "site": "Example News",
            }
        ],
        ticker="ULCC",
        company_name="Frontier Group Holdings, Inc.",
    )
    assert "DISTRESS_RESTRUCTURING" not in news.iloc[0]["event_types"]
    assert news.iloc[0]["issuer_relevant"]


def test_primary_sec_evidence_cannot_be_overwritten_by_unresolved_secondary_news():
    events = blind_events(
        pd.DataFrame(
            {
                "ticker": ["TEST"],
                "date": ["2026-08-24"],
                "previous_session": ["2026-08-21"],
            }
        )
    )
    payload = _sec_payload("2026-08-24T13:15:00Z")
    payload["filings"]["recent"]["items"] = ["3.02,9.01"]
    payload["filings"]["recent"]["primaryDocDescription"] = ["Private placement"]
    sec = bind_sec_evidence(
        normalize_sec_submissions(
            [("CIK0000123456.json", payload)],
            ticker="TEST",
            cik=123456,
        ),
        events.iloc[0],
    )
    sec["identifier_quality"] = "POINT_IN_TIME_TICKER_CIK_VALIDATED"
    fmp = bind_fmp_evidence(
        normalize_fmp_news(
            [
                {
                    "symbol": "TEST",
                    "publishedDate": "2026-08-24 09:00:00",
                    "title": "Test Systems raises guidance and beats estimates",
                    "text": "Test Systems reported stronger results.",
                    "url": "https://example.com/positive-context",
                    "site": "Example Wire",
                }
            ],
            ticker="TEST",
            company_name="Test Systems Inc.",
        ),
        events.iloc[0],
    )
    provider = pd.DataFrame(
        [
            {
                "event_id": events.iloc[0]["event_id"],
                "provider": "SEC_EDGAR",
                "status": "FETCHED",
            },
            {
                "event_id": events.iloc[0]["event_id"],
                "provider": "FMP_STOCK_NEWS",
                "status": "FETCHED",
            },
        ]
    )
    summary = summarize_event_evidence(events, pd.concat([sec, fmp]), provider).iloc[0]
    assert (
        summary["evidence_posture"] == "PRIMARY_PREOPEN_SEC_ASSUMED_PUBLIC_CLASSIFIED"
    )
    assert summary["primary_event_type"] == "FINANCING_DILUTION"
    assert summary["trajectory_posture"] == "ADVERSE_OR_DILUTIVE"
    assert summary["secondary_context_event_type"] == "EARNINGS_GUIDANCE"
    assert summary["secondary_context_trajectory_posture"] == "TRAJECTORY_UNRESOLVED"


def test_current_cik_mapping_cannot_create_primary_historical_evidence():
    events = blind_events(
        pd.DataFrame(
            {
                "ticker": ["TEST"],
                "date": ["2026-08-24"],
                "previous_session": ["2026-08-21"],
            }
        )
    )
    payload = _sec_payload("2026-08-24T13:15:00Z")
    sec = bind_sec_evidence(
        normalize_sec_submissions(
            [("CIK0000123456.json", payload)],
            ticker="TEST",
            cik=123456,
        ),
        events.iloc[0],
    )
    sec["identifier_quality"] = "CURRENT_FMP_PROFILE_CIK"
    provider = pd.DataFrame(
        [
            {
                "event_id": events.iloc[0]["event_id"],
                "provider": "SEC_EDGAR",
                "status": "FETCHED",
            }
        ]
    )
    summary = summarize_event_evidence(events, sec, provider).iloc[0]
    assert (
        summary["evidence_posture"]
        == "PREOPEN_SEC_ASSUMED_PUBLIC_IDENTITY_UNRESOLVED_CLASSIFIED"
    )
    assert summary["primary_event_type"] == "IDENTITY_UNRESOLVED"
    assert summary["trajectory_posture"] == "TRAJECTORY_UNRESOLVED"
    assert summary["preopen_sec_event_type"] == "EARNINGS_GUIDANCE"


def test_sec_normalized_cache_is_ticker_and_raw_digest_safe(tmp_path):
    cache_root = tmp_path / "news-cache"
    raw_path = cache_root / "sec" / "CIK0000123456.json"
    raw_path.parent.mkdir(parents=True)

    def write_payload(payload):
        raw_path.write_text(
            json.dumps(
                {
                    "source_url": "https://data.sec.gov/submissions/CIK0000123456.json",
                    "retrieved_at": "2026-08-25T12:00:00Z",
                    "payload": payload,
                }
            ),
            encoding="utf-8",
        )

    first = _sec_payload("2026-08-24T13:15:00Z")
    write_payload(first)
    archive = SECSubmissionArchive(
        "Researcher test@example.com",
        cache_root,
        cache_only=True,
    )
    aaa, _ = archive.filings(
        ticker="AAA",
        cik=123456,
        start=date(2026, 8, 20),
        end=date(2026, 8, 25),
    )
    bbb, _ = archive.filings(
        ticker="BBB",
        cik=123456,
        start=date(2026, 8, 20),
        end=date(2026, 8, 25),
    )
    assert set(aaa["ticker"]) == {"AAA"}
    assert set(bbb["ticker"]) == {"BBB"}

    changed = _sec_payload("2026-08-24T13:16:00Z")
    changed["filings"]["recent"]["accessionNumber"] = ["0000123456-26-000099"]
    write_payload(changed)
    refreshed, _ = archive.filings(
        ticker="AAA",
        cik=123456,
        start=date(2026, 8, 20),
        end=date(2026, 8, 25),
    )
    assert refreshed.iloc[0]["accession_number"] == "0000123456-26-000099"
    assert len(list((cache_root / "sec-normalized").glob("*.parquet"))) == 3


def test_empty_news_coverage_is_unresolved_not_no_catalyst():
    events = blind_events(
        pd.DataFrame(
            {
                "ticker": ["TEST"],
                "date": ["2026-08-24"],
                "previous_session": ["2026-08-21"],
            }
        )
    )
    provider = pd.DataFrame(
        [
            {
                "event_id": events.iloc[0]["event_id"],
                "provider": "SEC_EDGAR",
                "status": "FETCHED",
                "rows": 0,
            },
            {
                "event_id": events.iloc[0]["event_id"],
                "provider": "FMP_STOCK_NEWS",
                "status": "FETCHED",
                "rows": 0,
            },
        ]
    )
    summary = summarize_event_evidence(events, pd.DataFrame(), provider)
    assert summary.iloc[0]["evidence_posture"] == "COVERAGE_UNRESOLVED"
    assert summary.iloc[0]["primary_event_type"] == "COVERAGE_UNRESOLVED"


@pytest.mark.parametrize(
    ("atr_14", "expected_blocker"),
    [
        (0.0, "PRIOR_ATR_UNRESOLVED"),
        (0.40, "PRIOR_ATR_PCT_AT_OR_BELOW_FLOOR"),
        (0.40001, None),
    ],
)
def test_research_preview_requires_prior_atr_pct_strictly_above_four(
    atr_14, expected_blocker
):
    result = run_shadow_pipeline(
        [_snapshot(atr_14=atr_14)],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document()]},
        offline_documents_verified=True,
    )
    blockers = set(result.decisions[0].blockers)
    if expected_blocker:
        assert expected_blocker in blockers
        assert result.previews == []
    else:
        assert "PRIOR_ATR_UNRESOLVED" not in blockers
        assert "PRIOR_ATR_PCT_AT_OR_BELOW_FLOOR" not in blockers
        assert len(result.previews) == 1


def test_research_preview_requires_verified_adjusted_daily_atr_basis():
    result = run_shadow_pipeline(
        [_snapshot(daily_price_basis="UNVERIFIED")],
        as_of=AS_OF,
        target_session_date="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document()]},
        offline_documents_verified=True,
    )
    assert "PRIOR_ATR_PRICE_BASIS_UNVERIFIED" in result.decisions[0].blockers
    assert "NEWS_RESEARCH_SKIPPED_PRIOR_ATR" in result.decisions[0].blockers
    assert result.documents_by_candidate[result.candidates[0].candidate_id] == []
    assert result.previews == []


def test_ibkr_daily_atr_matches_simple_prior_session_method_and_fails_dirty_basis():
    dates = _trading_dates("2026-05-01", periods=130)
    bars = [
        SimpleNamespace(
            date=day.date(),
            open=25.0,
            high=25.5625,
            low=24.4375,
            close=25.0,
            volume=1_000_000,
        )
        for day in dates
    ]
    metrics = _daily_metrics(bars, (dates[-1] + TRADING_DAY).date())
    assert metrics["atr_14"] == pytest.approx(1.125)
    assert 100.0 * metrics["atr_14"] / metrics["previous_close"] == pytest.approx(4.5)

    bars[-5] = SimpleNamespace(
        date=bars[-5].date,
        open=12.5,
        high=12.8,
        low=12.2,
        close=12.5,
        volume=1_000_000,
    )
    with pytest.raises(ValueError, match="unclean 15-bar ATR source window"):
        _daily_metrics(bars, (dates[-1] + TRADING_DAY).date())


def test_ibkr_daily_atr_uses_adjusted_basis_and_matches_historical_method():
    assert _DAILY_WHAT_TO_SHOW == "ADJUSTED_LAST"
    assert _DAILY_PRICE_BASIS == "IBKR_ADJUSTED_LAST"
    dates = _trading_dates("2025-01-02", periods=140)
    event_index = 130

    # On an adjusted basis the $3 cash distribution at source index 125 has
    # already been applied to earlier bars, so it cannot manufacture >4% ATR.
    adjusted_close = pd.Series(47.0, index=range(len(dates)))
    adjusted = pd.DataFrame(
        {
            "date": dates,
            "Open": adjusted_close.copy(),
            "High": adjusted_close + 0.9,
            "Low": adjusted_close - 0.9,
            "Close": adjusted_close.copy(),
            "Volume": 1_000_000.0,
        }
    )
    adjusted.loc[event_index, ["Open", "High", "Low", "Close", "Volume"]] = [
        52.0,
        53.0,
        51.5,
        52.5,
        3_000_000.0,
    ]
    bars = [
        SimpleNamespace(
            date=row.date.date(),
            open=row.Open,
            high=row.High,
            low=row.Low,
            close=row.Close,
            volume=row.Volume,
        )
        for row in adjusted.iloc[:event_index].itertuples(index=False)
    ]
    live = _daily_metrics(bars, dates[event_index].date())
    history = study_ticker("DIV", adjusted, policy=DEFAULT_POLICY.historical)

    assert 100.0 * live["atr_14"] / live["previous_close"] == pytest.approx(
        100.0 * 1.8 / 47.0
    )
    assert history.counts["strict_with_neglect_pre_atr"] == 1
    assert history.counts["strict_prior_atr_at_or_below_floor"] == 1
    assert history.events.empty

    # The same economics on split-only/raw-like bars would include the cash
    # drop in true range and cross the threshold, demonstrating why TRADES is
    # not an acceptable live basis for parity.
    raw_bars = list(bars)
    for index in range(125):
        raw_bars[index] = SimpleNamespace(
            date=raw_bars[index].date,
            open=50.0,
            high=50.9,
            low=49.1,
            close=50.0,
            volume=1_000_000.0,
        )
    raw = _daily_metrics(raw_bars, dates[event_index].date())
    assert 100.0 * raw["atr_14"] / raw["previous_close"] > 4.0


def test_ibkr_daily_metrics_require_same_126_bar_history_as_census():
    dates = _trading_dates("2026-01-02", periods=125)
    bars = [
        SimpleNamespace(
            date=day.date(),
            open=25.0,
            high=25.5,
            low=24.5,
            close=25.0,
            volume=1_000_000,
        )
        for day in dates
    ]
    with pytest.raises(ValueError, match="fewer than 126 completed daily bars"):
        _daily_metrics(bars, (dates[-1] + TRADING_DAY).date())


def test_ibkr_daily_atr_fails_closed_on_missing_or_stale_source_bar():
    dates = _trading_dates("2026-05-01", periods=130)
    bars = [
        SimpleNamespace(
            date=day.date(),
            open=25.0,
            high=25.5,
            low=24.5,
            close=25.0,
            volume=1_000_000,
        )
        for day in dates
    ]
    session_date = (dates[-1] + TRADING_DAY).date()

    missing = list(bars)
    missing[-3] = SimpleNamespace(
        date=missing[-3].date,
        open=25.0,
        high=float("nan"),
        low=24.5,
        close=25.0,
        volume=1_000_000,
    )
    with pytest.raises(ValueError, match="unclean 15-bar ATR source window"):
        _daily_metrics(missing, session_date)

    with pytest.raises(
        ValueError, match="stale or incomplete 15-bar ATR source window"
    ):
        _daily_metrics(bars[:-1], session_date)


def test_historical_audit_flags_half_price_basis_cliff():
    dates = _trading_dates("2025-01-02", periods=140)
    close = pd.Series(10.0, index=range(len(dates)))
    close.iloc[130] = 5.0
    frame = pd.DataFrame(
        {
            "date": dates,
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000,
        }
    )
    result = study_ticker("CLIFF", frame, policy=DEFAULT_POLICY.historical)
    assert result.anomalies["half_or_double_cliff"].any()
    assert result.counts["anomalies"] >= 1


def test_benchmark_outcomes_are_attached_without_changing_event_return():
    dates = _trading_dates("2026-01-02", periods=70)
    benchmark = pd.DataFrame(
        {
            "date": dates,
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": pd.Series(range(100, 170), dtype=float),
            "Volume": 1_000_000,
        }
    )
    event = pd.DataFrame(
        {
            "date": [dates[0]],
            "open_to_close_1d_pct": [10.0],
            "next_open_to_close_1d_pct": [9.0],
        }
    )
    attached = attach_benchmark_outcomes(
        event, benchmark, policy=DEFAULT_POLICY.historical
    )
    assert attached.loc[0, "open_to_close_1d_pct"] == 10.0
    assert pd.notna(attached.loc[0, "excess_open_to_close_1d_pct"])


def test_clustered_summary_bootstraps_dates_or_issuers_not_rows():
    events = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-05"]),
            "ticker": ["A", "B", "A"],
            "prior_window_clean": [True, True, True],
            "basis_review_cleared": [True, True, True],
            "excess_next_open_to_close_5d_pct": [10.0, -10.0, 3.0],
        }
    )
    result = clustered_outcome_summary(
        events, cluster_column="date", bootstrap_samples=100, seed=1
    )
    assert result.loc[0, "n_events"] == 3
    assert result.loc[0, "n_clusters"] == 2


def test_horizon_comparison_reports_available_and_balanced_cohorts():
    events = pd.DataFrame(
        {
            "sample_period": ["DEVELOPMENT_1999_2019"] * 3,
            "prior_window_clean": [True, True, True],
            "basis_review_cleared": [True, True, True],
            "excess_next_open_to_close_5d_pct": [1.0, 2.0, 99.0],
            "excess_next_open_to_close_20d_pct": [2.0, 3.0, float("nan")],
            "excess_next_open_to_close_60d_pct": [3.0, 4.0, float("nan")],
        }
    )
    result = horizon_comparison_summary(events)
    all_rows = result[result["sample_period"].eq("ALL")]
    available_5 = all_rows[
        all_rows["cohort"].eq("AVAILABLE")
        & all_rows["horizon_sessions"].eq(5)
    ].iloc[0]
    balanced_5 = all_rows[
        all_rows["cohort"].eq("BALANCED")
        & all_rows["horizon_sessions"].eq(5)
    ].iloc[0]
    assert available_5["n"] == 3
    assert balanced_5["n"] == 2
    assert balanced_5["mean_pct"] == pytest.approx(1.5)
