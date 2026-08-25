from __future__ import annotations

import json
import hashlib
import csv
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from episodic_pivot.config import DEFAULT_POLICY
from episodic_pivot.historical import (
    attach_benchmark_outcomes,
    clustered_outcome_summary,
    study_ticker,
)
from episodic_pivot.manifest import write_run_artifacts
from episodic_pivot.news import (
    ArticleFetcher,
    _PinnedHTTPSConnection,
    _document_mentions_candidate,
    _read_response_body,
    _validate_public_http_url,
    assess_catalyst,
    normalize_evidence_document,
    source_tier,
)
from scripts.capture_ep_premarket_ibkr import _halt_status, _round_robin_keys
from scripts.run_episodic_pivot_shadow import _verify_evidence_manifest
from episodic_pivot.pipeline import run_shadow_pipeline
from episodic_pivot.premarket import nominate_candidates
from episodic_pivot.qualify import qualify_candidate
from episodic_pivot.schema import (
    NewsDocument,
    NewsHit,
    PremarketSnapshot,
)


AS_OF = "2026-08-24T12:31:00Z"


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
        "url": "https://www.businesswire.com/news/home/test-systems-quarterly-results",
        "canonical_url": "https://www.businesswire.com/news/home/test-systems-quarterly-results",
        "publisher": "Test Systems",
        "published_at": "2026-08-24T12:00:00Z",
        "retrieved_at": AS_OF,
        "text_excerpt": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_tier": "PRIMARY",
        "fetch_status": "FETCHED",
        "catalyst_types": ("EARNINGS_GUIDANCE",),
        "adverse_flags": (),
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
    catalyst = assess_catalyst([_document()], decision_at=AS_OF, policy=DEFAULT_POLICY.news)
    decision = qualify_candidate(
        candidate, catalyst, policy=DEFAULT_POLICY, decision_at=AS_OF
    )

    assert "STALE_MARKET_DATA" in candidate.discovery_warnings
    assert "NON_LIVE_MARKET_DATA" in candidate.discovery_warnings
    assert decision.decision == "WATCH"
    assert "STALE_MARKET_DATA" in decision.blockers


def test_future_snapshot_uses_small_clock_skew_not_staleness_window():
    snapshot = _snapshot(observed_at="2026-08-24T12:32:29Z")
    candidate = nominate_candidates(
        [snapshot], as_of=AS_OF, policy=DEFAULT_POLICY
    )[0]
    assert "SNAPSHOT_FROM_FUTURE" in candidate.discovery_warnings


def test_search_snippet_or_failed_fetch_cannot_confirm_an_ep():
    failed = _document(
        fetch_status="FETCH_FAILED:ValueError",
        text_excerpt="",
        text_sha256="",
        source_tier="SEARCH_WRAPPER",
    )
    result = assess_catalyst([failed], decision_at=AS_OF, policy=DEFAULT_POLICY.news)
    assert result.status == "UNCONFIRMED"
    assert "NO_ACTUAL_SOURCE_EVIDENCE" in result.reason_codes


def test_primary_actual_document_confirms_and_future_document_does_not():
    confirmed = assess_catalyst(
        [_document()], decision_at=AS_OF, policy=DEFAULT_POLICY.news
    )
    future = assess_catalyst(
        [_document(published_at="2026-08-24T12:40:00Z")],
        decision_at=AS_OF,
        policy=DEFAULT_POLICY.news,
    )
    assert confirmed.status == "CONFIRMED"
    assert confirmed.catalyst_type == "EARNINGS_GUIDANCE"
    assert future.status == "UNCONFIRMED"
    assert "POST_DECISION_SOURCE" in future.reason_codes


def test_document_retrieved_after_decision_is_not_point_in_time_evidence():
    result = assess_catalyst(
        [_document(retrieved_at="2026-08-24T12:40:00Z")],
        decision_at=AS_OF,
        policy=DEFAULT_POLICY.news,
    )
    assert result.status == "UNCONFIRMED"
    assert "POST_DECISION_RETRIEVAL" in result.reason_codes


def test_article_retrieved_at_is_fetch_completion_clock(monkeypatch):
    completed = pd.Timestamp("2026-08-24T12:32:00Z").to_pydatetime()
    body = "<html><article>" + (
        "Test Systems raised guidance after quarterly results and durable demand. " * 10
    ) + "</article></html>"
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


def test_adverse_offering_never_reaches_staging():
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
        execute_on=date(2026, 8, 24),
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [doc]},
        offline_documents_verified=True,
    )
    assert result.decisions[0].decision == "REJECT"
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
        execute_on="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [document]},
        offline_documents_verified=True,
    )
    assert result.decisions[0].decision == "REJECT"
    assert "CLINICAL_OR_REGULATORY_FAILURE" in result.decisions[0].blockers
    assert result.previews == []


def test_end_to_end_preview_is_capped_limit_only_and_never_live(tmp_path):
    result = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        execute_on=date(2026, 8, 24),
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document()]},
        offline_documents_verified=True,
    )
    assert result.decisions[0].decision == "STAGEABLE"
    preview = result.previews[0]
    assert preview.quantity > 0
    assert preview.binding_cap in {
        "RISK",
        "MAX_NOTIONAL",
        "ADDV_PARTICIPATION",
        "PREMARKET_VOLUME_PARTICIPATION",
        "DISPLAYED_ASK_PARTICIPATION",
        "ABSOLUTE_QUANTITY",
    }
    assert preview.order_type == "LMT"
    assert preview.regular_hours_only is True
    assert preview.approval == ""
    assert preview.live_eligible is False
    assert preview.activation_min_price == 10.40
    assert preview.activation_max_price == 12.50
    assert preview.max_opening_gap_pct == 25.0
    assert preview.entry_window_end_et == "09:35:00"
    assert preview.requires_fresh_quote_at_release is True
    assert preview.requires_halt_recheck_at_release is True
    assert preview.requires_opening_gap_recheck is True
    assert preview.risk_bps <= DEFAULT_POLICY.execution.classic_risk_bps

    input_path = tmp_path / "input.json"
    input_path.write_text("{}", encoding="utf-8")
    output = write_run_artifacts(
        result,
        policy=DEFAULT_POLICY,
        output_dir=tmp_path / "run",
        input_files={"snapshot": input_path},
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["safety"] == {
        "live_actions_enabled": False,
        "broker_contacted": False,
        "sheets_written": False,
        "r2_written": False,
        "production_deployed": False,
    }
    assert (output / "staging_preview.csv").exists()


def test_multiple_previews_are_scaled_to_daily_portfolio_risk_cap():
    snapshots = [_snapshot(symbol=f"TST{i}") for i in range(6)]
    documents = {item.symbol: [_document()] for item in snapshots}
    result = run_shadow_pipeline(
        snapshots,
        as_of=AS_OF,
        execute_on="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents=documents,
        offline_documents_verified=True,
    )
    assert len(result.previews) == 6
    assert sum(item.risk_bps for item in result.previews) <= (
        DEFAULT_POLICY.execution.max_daily_risk_bps + 0.01
    )
    assert all("DAILY_RISK" in item.binding_cap for item in result.previews)


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
        execute_on="2026-08-24",
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
        execute_on="2026-08-24",
        policy=DEFAULT_POLICY,
        search_provider=BrokenSearch(),
    )
    assert result.decisions[0].decision == "WATCH"
    document = next(iter(result.documents_by_candidate.values()))[0]
    assert document.fetch_status == "SEARCH_FAILED:RuntimeError"


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
        execute_on="2026-08-24",
        policy=DEFAULT_POLICY,
        search_provider=EmptySearch(),
    )
    assert "STALE_MARKET_DATA" in result.decisions[0].blockers


def test_entry_window_is_enforced_not_just_written_to_preview():
    late_as_of = "2026-08-24T13:36:00Z"  # 09:36 America/New_York
    result = run_shadow_pipeline(
        [_snapshot(observed_at="2026-08-24T13:35:30Z")],
        as_of=late_as_of,
        execute_on="2026-08-24",
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
        execute_on="2026-08-24",
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
        execute_on="2026-08-24",
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
                "artifacts": {
                    "evidence_by_symbol.json": {"sha256": digest}
                },
            }
        ),
        encoding="utf-8",
    )
    assert _verify_evidence_manifest(evidence, manifest) == run_id
    evidence.write_text('{"TEST": ["tampered"]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        _verify_evidence_manifest(evidence, manifest)


def test_review_csv_escapes_formula_title_but_json_keeps_raw_audit_value(tmp_path):
    malicious_title = '=HYPERLINK("https://evil.example","Test Systems raises guidance")'
    result = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        execute_on="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document(title=malicious_title)]},
        offline_documents_verified=True,
    )
    output = write_run_artifacts(
        result,
        policy=DEFAULT_POLICY,
        output_dir=tmp_path / result.run_id,
    )
    with (output / "staging_preview.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["catalyst_summary"].startswith("'=")
    raw = json.loads((output / "staging_preview.json").read_text(encoding="utf-8"))
    assert raw[0]["catalyst_summary"].startswith("=")


def test_empty_and_nonempty_preview_csv_have_identical_headers(tmp_path):
    stageable = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        execute_on="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document()]},
        offline_documents_verified=True,
    )
    empty = run_shadow_pipeline(
        [_snapshot(symbol="EMPTY")],
        as_of=AS_OF,
        execute_on="2026-08-24",
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
    stage_header = (stage_dir / "staging_preview.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    empty_header = (empty_dir / "staging_preview.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
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
        text_excerpt=("Retailers target higher revenue as consumers return to stores. " * 12),
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

    response = SimpleNamespace(
        _fp=Raw(), _connection=SimpleNamespace(sock=Sock())
    )
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
        execute_on="2026-08-24",
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
        execute_on="2026-08-24",
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
        execute_on="2026-08-24",
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
        execute_on="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document()]},
        offline_documents_verified=True,
    )
    assert result.decisions[0].decision == "WATCH"
    assert result.decisions[0].setup_type == "DELAYED_EP_WATCH"
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
    )
    assert assessment.status == "WATCH"
    assert "INSUFFICIENT_SOURCE_AUTHORITY" in assessment.reason_codes


def test_trial_enrollment_is_watch_not_positive_clinical_data():
    text = (
        "Test Systems announced the first patient enrolled in a phase 2 clinical trial. "
        "The study has initiated enrollment, but the company reported no efficacy data, "
        "endpoint result, statistical result, or regulatory approval. "
    ) * 5
    result = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        execute_on="2026-08-24",
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
    assert "CATALYST_CLASS_NOT_STAGE_ENABLED" in result.decisions[0].blockers


def test_negative_trial_results_cannot_be_labeled_positive_clinical_data():
    text = (
        "Test Systems reported that trial results showed no statistically significant "
        "difference from placebo and failed to achieve the primary endpoint. Management "
        "will review the dataset before deciding whether to continue the program. "
    ) * 5
    result = run_shadow_pipeline(
        [_snapshot()],
        as_of=AS_OF,
        execute_on="2026-08-24",
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
    assert result.decisions[0].decision == "REJECT"
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
        execute_on="2026-08-24",
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
        execute_on="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [document]},
        offline_documents_verified=True,
    )
    assert result.decisions[0].catalyst.catalyst_type == "EARNINGS"
    assert result.decisions[0].decision == "WATCH"
    assert "MATERIALITY_SCORE_BELOW_STAGE_GATE" in result.decisions[0].blockers


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
    )
    assert assessment.status == "WATCH"
    assert assessment.catalyst_type == "UNCLASSIFIED"


def test_sizing_failure_is_propagated_into_candidate_decision():
    result = run_shadow_pipeline(
        [_snapshot(prior_two_day_low=7.0)],
        as_of=AS_OF,
        execute_on="2026-08-24",
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
        execute_on="2026-08-24",
        policy=DEFAULT_POLICY,
        offline_documents={"TEST": [_document()]},
        offline_documents_verified=True,
    )
    assert result.decisions[0].setup_type == "DELAYED_EP_WATCH"
    assert result.previews == []


def test_policy_cannot_enable_live_actions():
    try:
        replace(DEFAULT_POLICY, live_actions_enabled=True)
    except ValueError as exc:
        assert "cannot enable live actions" in str(exc)
    else:
        raise AssertionError("live policy unexpectedly constructed")


def test_historical_proxy_uses_prior_volume_and_suppresses_repeat_events():
    dates = pd.bdate_range("2025-01-02", periods=300)
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
        policy=DEFAULT_POLICY.historical,
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


def test_historical_audit_flags_half_price_basis_cliff():
    dates = pd.bdate_range("2025-01-02", periods=140)
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
    dates = pd.bdate_range("2026-01-02", periods=70)
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
            "data_quality_clean": [True, True, True],
            "excess_next_open_to_close_5d_pct": [10.0, -10.0, 3.0],
        }
    )
    result = clustered_outcome_summary(
        events, cluster_column="date", bootstrap_samples=100, seed=1
    )
    assert result.loc[0, "n_events"] == 3
    assert result.loc[0, "n_clusters"] == 2
