from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from discretionary_focus.contracts import (
    FocusPayloadError,
    canonical_digest,
    validate_payload,
)
from discretionary_focus.selector import select_focus


def _source(source_id: str = "ir-1") -> dict:
    return {
        "source_id": source_id,
        "label": "Issuer quarterly results",
        "url": "https://example.com/investors/results",
        "as_of": "2026-08-25",
        "primary": True,
    }


def _card(ticker: str = "AMPL", rank: int = 1) -> dict:
    return {
        "rank": rank,
        "ticker": ticker,
        "company_name": f"{ticker} Incorporated",
        "why_now": "Reported growth accelerated while the chart formed a tighter shelf.",
        "setup": "Tight consolidation within three percent of the prior high.",
        "trigger": "Break the shelf with RVOL-at-Time at or above 2.0 and price above the open.",
        "invalidation": {
            "technical": "Lose the shelf low on expanding volume.",
            "thesis_kill": "Forward growth guidance falls below the current market-implied path.",
        },
        "catalyst": "Raised forward revenue guidance after the latest report.",
        "priced_in": "The price already discounts continued growth, but not another guide raise.",
        "next_proof": "Next monthly KPI or earnings update confirms the raised trajectory.",
        "event_date": "2026-09-10",
        "earnings_td": 9,
        "technical": {
            "observed_at": "2026-08-26T20:15:00Z",
            "setup_gate": "PASS",
            "liquidity_gate": "PASS",
            "setup_quality": 88,
        },
        "sources": [_source()],
    }


def _payload(*cards: dict, phase: str = "FINAL") -> dict:
    ready = bool(cards)
    payload = {
        "schema_version": "discretionary-focus.v1",
        "research_only": True,
        "quick_review_created": False,
        "live_actions_enabled": False,
        "order_staging_enabled": False,
        "status": "READY" if ready else "NO_QUALIFIED_SETUP",
        "phase": phase,
        "as_of": "2026-08-26",
        "valid_for": "2026-08-27",
        "generated_at": "2026-08-27T12:45:00Z",
        "expires_at": "2026-08-27T20:15:00Z",
        "focus": list(cards),
        "screen_summary": {
            "input_count": 4,
            "technical_pass_count": 3,
            "research_pass_count": len(cards),
            "selected_count": len(cards),
            "rejected_counts": {
                "earnings_window": 1,
                "research_gate": 3 - len(cards),
            },
        },
        "provenance": {
            "screen_snapshot_id": "tv-armed-20260826",
            "screen_captured_at": "2026-08-26T20:15:00Z",
            "research_snapshot_id": "research-20260827",
            "research_as_of": "2026-08-27T12:30:00Z",
            "policy_version": "discretionary-focus-policy.v1",
        },
    }
    if not ready:
        payload["no_setup_reason"] = "No candidate cleared every technical, evidence, and event gate."
    return payload


def _technical(ticker: str, *, quality: int = 80, cluster: str = "software") -> dict:
    return {
        "ticker": ticker,
        "company_name": f"{ticker} Incorporated",
        "technical_gate": "PASS",
        "liquidity_gate": "PASS",
        "setup_quality": quality,
        "screen_rank": 1,
        "observed_at": "2026-08-26T20:15:00Z",
        "setup": "Tight shelf beneath the prior high.",
        "trigger": "Break the shelf with RVOL-at-Time at or above 2.0.",
        "invalidation": "Lose the shelf low on expanding volume.",
        "event_date": "2026-09-10",
        "earnings_td": 9,
        "causal_cluster": cluster,
    }


def _research(ticker: str, *, attention_rank: int = 1, cluster: str = "software") -> dict:
    return {
        "ticker": ticker,
        "research_gate": "PASS",
        "attention_rank": attention_rank,
        "company_name": f"{ticker} Incorporated",
        "why_now": "Fundamental change is reaching reported revenue and guidance.",
        "catalyst": "The company raised forward guidance after the latest report.",
        "variant_wedge": "Consensus underweights the durability of the new growth cohort.",
        "priced_in": "The stock discounts the current guide but not another upward revision.",
        "next_proof": "The next KPI update must sustain the raised operating trajectory.",
        "source_current": True,
        "catalyst_reaches_economics": True,
        "unresolved_financing_risk": False,
        "unresolved_dilution_risk": False,
        "unresolved_restatement_risk": False,
        "kill_condition": "Forward growth falls below the market-implied path.",
        "causal_cluster": cluster,
        "sources": [_source(f"{ticker.lower()}-ir")],
    }


def test_validate_ready_payload_and_digest_is_canonical() -> None:
    payload = _payload(_card())
    validated = validate_payload(
        payload,
        now=datetime(2026, 8, 27, 13, 0, tzinfo=timezone.utc),
    )
    assert validated == payload
    reordered = {key: payload[key] for key in reversed(payload)}
    assert canonical_digest(reordered) == canonical_digest(payload)
    assert len(canonical_digest(payload)) == 64


def test_no_qualified_setup_is_an_explicit_empty_payload() -> None:
    payload = validate_payload(_payload())
    assert payload["status"] == "NO_QUALIFIED_SETUP"
    assert payload["focus"] == []
    assert payload["no_setup_reason"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("research_only", False),
        ("quick_review_created", True),
        ("live_actions_enabled", True),
        ("order_staging_enabled", True),
    ],
)
def test_safety_booleans_fail_closed(field: str, value: bool) -> None:
    payload = _payload(_card())
    payload[field] = value
    with pytest.raises(FocusPayloadError, match=field):
        validate_payload(payload)


def test_payload_rejects_more_than_two_names_and_duplicate_tickers() -> None:
    with pytest.raises(FocusPayloadError, match="at most 2"):
        validate_payload(_payload(_card("AAA", 1), _card("BBB", 2), _card("CCC", 3)))

    duplicate = _payload(_card("AAA", 1), _card("AAA", 2))
    with pytest.raises(FocusPayloadError, match="duplicate tickers"):
        validate_payload(duplicate)


def test_payload_rejects_earnings_inside_five_days_or_missing_sources() -> None:
    payload = _payload(_card())
    payload["focus"][0]["earnings_td"] = 5
    with pytest.raises(FocusPayloadError, match="must be > 5"):
        validate_payload(payload)

    payload = _payload(_card())
    payload["focus"][0]["sources"] = []
    with pytest.raises(FocusPayloadError, match="at least one source"):
        validate_payload(payload)


def test_numeric_frozen_levels_require_raw_as_traded_basis() -> None:
    payload = _payload(_card())
    payload["focus"][0]["trigger"] = {"condition": "Cross the pivot", "price": 42.5}
    with pytest.raises(FocusPayloadError, match="RAW_AS_TRADED"):
        validate_payload(payload)

    payload["focus"][0]["trigger"]["price_basis"] = "RAW_AS_TRADED"
    validate_payload(payload)


def test_payload_rejects_execution_fields_at_any_depth() -> None:
    payload = _payload(_card())
    payload["focus"][0]["technical"]["position_size"] = "1% NAV"
    with pytest.raises(FocusPayloadError, match="execution or QUICK_REVIEW"):
        validate_payload(payload)


def test_current_validation_rejects_expired_and_unsupported_phase_payloads() -> None:
    payload = _payload(_card())
    with pytest.raises(FocusPayloadError, match="expired"):
        validate_payload(
            payload,
            now=datetime(2026, 8, 27, 20, 16, tzinfo=timezone.utc),
        )

    live = _payload(_card(), phase="LIVE")
    with pytest.raises(FocusPayloadError, match="must be one of"):
        validate_payload(live)


def test_current_validation_rejects_early_future_session_delivery() -> None:
    payload = _payload(_card(), phase="PROVISIONAL")
    payload["generated_at"] = "2026-08-26T13:00:00Z"
    payload["focus"][0]["technical"]["observed_at"] = "2026-08-26T12:55:00Z"
    payload["provenance"]["screen_captured_at"] = "2026-08-26T12:55:00Z"
    payload["provenance"]["research_as_of"] = "2026-08-26T12:55:00Z"
    with pytest.raises(FocusPayloadError, match="current delivery requires today's session"):
        validate_payload(
            payload,
            now=datetime(2026, 8, 26, 13, 1, tzinfo=timezone.utc),
            require_current=True,
        )


def test_current_session_date_uses_new_york_not_utc() -> None:
    payload = _payload(_card(), phase="PROVISIONAL")
    payload["generated_at"] = "2026-08-27T00:25:00Z"
    payload["focus"][0]["technical"]["observed_at"] = "2026-08-27T00:20:00Z"
    payload["provenance"]["screen_captured_at"] = "2026-08-27T00:20:00Z"
    payload["provenance"]["research_as_of"] = "2026-08-27T00:20:00Z"
    with pytest.raises(FocusPayloadError, match="2026-08-26"):
        validate_payload(
            payload,
            now=datetime(2026, 8, 27, 0, 30, tzinfo=timezone.utc),
            require_current=True,
        )


def test_selector_hard_gates_then_prioritizes_the_best_setup() -> None:
    technical = [
        _technical("AMPL", quality=90, cluster="software"),
        _technical("GTLB", quality=95, cluster="software"),
        _technical("EAT", quality=75, cluster="restaurants"),
    ]
    research = {
        "AMPL": _research("AMPL", attention_rank=1, cluster="software"),
        "GTLB": _research("GTLB", attention_rank=2, cluster="software"),
        "EAT": _research("EAT", attention_rank=3, cluster="restaurants"),
    }

    selected, summary = select_focus(technical, research)
    assert [row["ticker"] for row in selected] == ["GTLB", "EAT"]
    assert [row["rank"] for row in selected] == [1, 2]
    assert summary["input_count"] == 3
    assert summary["technical_pass_count"] == 3
    assert summary["research_pass_count"] == 3
    assert summary["selected_count"] == 2
    assert summary["rejected_counts"]["causal_cluster_duplicate"] == 1

    payload = _payload(*selected)
    payload["screen_summary"] = summary
    validate_payload(payload)


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (lambda t, r: t.update(technical_gate="FAIL"), "technical_gate"),
        (lambda t, r: t.pop("earnings_td"), "earnings_missing"),
        (lambda t, r: t.update(earnings_td=4), "earnings_window"),
        (lambda t, r: r.update(catalyst=""), "catalyst_missing"),
        (lambda t, r: r.update(source_current=False), "source_stale_or_unknown"),
        (
            lambda t, r: r.update(catalyst_reaches_economics=False),
            "economic_link_unproven",
        ),
        (lambda t, r: r.update(variant_wedge="UNTESTED"), "variant_wedge_missing"),
        (lambda t, r: r.update(kill_condition=""), "kill_condition_missing"),
        (lambda t, r: r.update(sources=[]), "sources_missing"),
        (
            lambda t, r: r["sources"][0].update(primary=False),
            "sources_invalid",
        ),
        (
            lambda t, r: t.update(trigger={"condition": "Cross pivot", "price": 42.5}),
            "trigger_invalid",
        ),
        (lambda t, r: r.update(unresolved_financing_risk=True), "financing_risk"),
        (lambda t, r: r.pop("unresolved_dilution_risk"), "dilution_risk"),
        (lambda t, r: t.update(setup_quality=59), "setup_quality"),
    ],
)
def test_selector_fails_closed_on_each_required_gate(mutator, reason: str) -> None:
    technical = _technical("AMPL")
    research = _research("AMPL")
    mutator(technical, research)
    selected, summary = select_focus([technical], {"AMPL": research})
    assert selected == []
    assert summary["rejected_counts"][reason] == 1


def test_selector_rejects_ambiguous_duplicate_ticker_rows() -> None:
    row = _technical("AMPL")
    selected, summary = select_focus(
        [row, deepcopy(row)],
        {"AMPL": _research("AMPL")},
    )
    assert selected == []
    assert summary["rejected_counts"]["duplicate_ticker"] == 2


def test_research_cannot_overwrite_failed_technical_or_earnings_truth() -> None:
    failed_chart = _technical("AMPL")
    failed_chart["technical_gate"] = "FAIL"
    attempted_rescue = _research("AMPL") | {
        "technical_gate": "PASS",
        "liquidity_gate": "PASS",
        "setup_quality": 100,
    }
    selected, summary = select_focus([failed_chart], {"AMPL": attempted_rescue})
    assert selected == []
    assert summary["rejected_counts"]["technical_gate"] == 1

    near_event = _technical("AMPL")
    near_event["earnings_td"] = 3
    attempted_rescue = _research("AMPL") | {
        "earnings_td": 30,
        "event_date": "2026-12-01",
    }
    selected, summary = select_focus([near_event], {"AMPL": attempted_rescue})
    assert selected == []
    assert summary["rejected_counts"]["earnings_window"] == 1


def test_selector_caps_names_without_turning_scores_into_recommendations() -> None:
    technical = [
        _technical("AAA", quality=90, cluster="a"),
        _technical("BBB", quality=80, cluster="b"),
        _technical("CCC", quality=70, cluster="c"),
    ]
    research = {
        ticker: _research(ticker, attention_rank=rank, cluster=ticker.lower())
        for rank, ticker in enumerate(("AAA", "BBB", "CCC"), start=1)
    }
    selected, summary = select_focus(technical, research)
    assert [row["ticker"] for row in selected] == ["AAA", "BBB"]
    assert summary["rejected_counts"]["attention_cap"] == 1
    assert not any("decision" in row or "position_size" in row for row in selected)


def test_selector_rejects_max_names_above_two() -> None:
    with pytest.raises(ValueError, match="between 0 and 2"):
        select_focus([], {}, max_names=3)
