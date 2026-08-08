import json
from pathlib import Path

from fundamental.site_payload import build_fundamental_site_payload


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _daily(decisions: list[dict], candidates: list[dict] | None = None) -> dict:
    return {
        "health": {
            "as_of": "2026-08-05",
            "universe": {
                "discovered": 2001,
                "research_eligible": 1149,
                "fundamental_covered": 1149,
                "sec_covered": 20,
                "scored_candidates": 1149,
            },
            "sources": [{"source": "SEC", "as_of": "2026-08-05"}],
            "gaps": ["History remains survivorship-biased."],
        },
        "candidates": candidates or [],
        "underwrite_decisions": decisions,
        "live_actions_enabled": False,
    }


def _decision(ticker: str, status: str) -> dict:
    return {
        "ticker": ticker,
        "decision": status,
        "verdict": f"{ticker} verdict",
        "mispricing": f"{ticker} mispricing",
        "priced_in": f"{ticker} priced in",
        "valuation": f"{ticker} valuation",
        "proof_required": [f"{ticker} proof"],
        "kill_conditions": [f"{ticker} kill"],
        "next_review": f"{ticker} next review",
        "price_as_of": "2026-08-04 close $10.00",
    }


def _maps() -> dict:
    return {
        "meta": {
            "as_of": "2026-08-05",
            "circle_count": 31,
            "founder_active": 59,
            "circle_excluded_count": 15,
            "founder_circle_overlap": ["AAA"],
        },
        "circle_rows": [
            {"ticker": "AAA", "fit_score": 9, "basis": "Consumer product", "founder_led": True}
        ],
        "founder_rows": [{"ticker": "AAA"}],
    }


def test_no_review_keeps_only_three_active_names(tmp_path):
    decisions = [_decision(ticker, "WAIT_FOR_PROOF") for ticker in ["AAA", "BBB", "CCC", "DDD"]]
    candidates = [
        {"ticker": ticker, "company_name": f"Company {ticker}", "trend_state": "AMBER"}
        for ticker in ["AAA", "BBB", "CCC", "DDD"]
    ]
    payload = build_fundamental_site_payload(
        _write(tmp_path / "daily.json", _daily(decisions, candidates)),
        _write(tmp_path / "maps.json", _maps()),
    )

    assert payload["status"] == "NO_REVIEW"
    assert payload["reviews"] == []
    assert [row["ticker"] for row in payload["active_research"]] == ["AAA", "BBB", "CCC"]
    assert payload["active_research"][0]["product_circle"] is True
    assert payload["active_research"][0]["founder_led"] is True
    assert payload["live_actions_enabled"] is False


def test_quick_reviews_are_complete_and_capped(tmp_path):
    decisions = [_decision(ticker, "QUICK_REVIEW") for ticker in ["AAA", "BBB", "CCC", "DDD"]]
    candidates = [
        {"ticker": ticker, "company_name": f"Company {ticker}", "trend_state": "GREEN"}
        for ticker in ["AAA", "BBB", "CCC", "DDD"]
    ]
    payload = build_fundamental_site_payload(
        _write(tmp_path / "daily.json", _daily(decisions, candidates)),
        _write(tmp_path / "maps.json", _maps()),
    )

    assert payload["status"] == "QUICK_REVIEW"
    assert [row["ticker"] for row in payload["reviews"]] == ["AAA", "BBB", "CCC"]
    first = payload["reviews"][0]
    for key in [
        "verdict", "mispricing", "priced_in", "valuation", "downside",
        "proof_trigger", "kill_conditions", "exact_decision",
    ]:
        assert first[key]


def test_payload_never_exposes_background_candidate_queue(tmp_path):
    payload = build_fundamental_site_payload(
        _write(
            tmp_path / "daily.json",
            _daily(
                [_decision("AAA", "WAIT_FOR_EVENT")],
                [{"ticker": "AAA", "company_name": "Company AAA", "research_score": 99.9}],
            ),
        ),
        _write(tmp_path / "maps.json", _maps()),
    )

    assert "candidates" not in payload
    assert "research_score" not in json.dumps(payload)
    assert payload["audit"]["research_eligible"] == 1149
    assert payload["portfolio"]["max_positions"] == 10
    assert payload["portfolio"]["capital_cap_pct"] == 30.0


def test_pass_summary_is_high_level_mutually_exclusive_and_excludes_underwrites(tmp_path):
    candidates = [
        {"ticker": "AAA", "company_name": "Active", "trend_state": "RED",
         "first_rejection": "Price trend is damaged; no full-size entry until the 200-day trend recovers.",
         "research_lane": "standard_company"},
        {"ticker": "BBB", "trend_state": "GREEN",
         "first_rejection": "The screen cannot prove a variant view; expectations and valuation need a full underwrite.",
         "research_lane": "standard_company"},
        {"ticker": "CCC", "trend_state": "RED",
         "first_rejection": "Price trend is damaged; no full-size entry until the 200-day trend recovers.",
         "research_lane": "standard_company"},
        {"ticker": "DDD", "trend_state": "AMBER",
         "first_rejection": "Baseline covered; banks and financials require capital underwriting before ranking.",
         "research_lane": "financials_specialist"},
        {"ticker": "EEE", "trend_state": "GREEN",
         "first_rejection": "Leverage is the first downside question; stress normalized cash flow and maturities.",
         "research_lane": "standard_company"},
        {"ticker": "FFF", "trend_state": "AMBER",
         "first_rejection": "Per-share dilution is running above 3% annually.",
         "research_lane": "standard_company"},
        {"ticker": "GGG", "trend_state": "GREEN",
         "first_rejection": "Latest free cash flow is not positive.",
         "research_lane": "standard_company"},
        {"ticker": "HHH", "trend_state": "UNAVAILABLE",
         "first_rejection": "Fewer than four comparable annual statement periods are available.",
         "research_lane": "standard_company"},
    ]
    payload = build_fundamental_site_payload(
        _write(tmp_path / "daily.json", _daily([_decision("AAA", "WAIT_FOR_PROOF")], candidates)),
        _write(tmp_path / "maps.json", _maps()),
    )

    summary = payload["audit"]["pass_summary"]
    assert summary["background_count"] == 7
    assert sum(row["count"] for row in summary["reasons"]) == 7
    assert {row["key"] for row in summary["reasons"]} == {
        "valuation_expectations", "trend_damaged", "specialist_underwriting",
        "leverage", "dilution", "cash_generation", "coverage_eligibility",
    }
    assert summary["trend_overlay"]["without_full_confirmation"] == 4
    assert "AAA" not in json.dumps(summary)


def test_missing_daily_report_skips_payload(tmp_path):
    assert build_fundamental_site_payload(
        tmp_path / "missing.json",
        _write(tmp_path / "maps.json", _maps()),
    ) is None
