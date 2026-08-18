from pathlib import Path

import pandas as pd

import fundamental.company_maps as company_maps
from fundamental.company_maps import (
    build_company_maps_report,
    load_company_map_sources,
    prepare_founder_rows,
)


def test_reference_lists_are_broad_unique_and_research_only():
    founder, circle = load_company_map_sources()
    founder_tickers = {row["ticker"] for row in founder["active"]}
    circle_tickers = {row["ticker"] for row in circle["companies"]}
    excluded_tickers = {row["ticker"] for row in circle["excluded"]}
    assert len(founder_tickers) >= 50
    assert len(circle_tickers) >= 25
    assert {"NVDA", "META", "NET", "SHOP", "CPNG"}.issubset(founder_tickers)
    assert {"PCOR", "RDDT"}.isdisjoint(founder_tickers)
    assert {"COST", "CMG", "DUOL", "ABNB", "NFLX", "DASH"}.issubset(circle_tickers)
    assert {"CME", "IBKR", "NET", "MSFT", "GOOGL", "SNOW"}.issubset(excluded_tickers)
    assert not circle_tickers & excluded_tickers
    assert {"ABNB", "DASH", "DUOL", "RBLX", "ROKU", "YELP"}.issubset(
        founder_tickers & circle_tickers
    )
    assert "RDDT" in circle_tickers


def test_current_ceo_mismatch_holds_founder_record_for_recheck():
    source = {"active": [{
        "ticker": "AAA",
        "company_name": "Alpha",
        "founder_ceo": "Founding Person",
        "founder_role": "Founder & CEO",
        "source_date": "2026-08-01",
        "source_url": "https://example.com/source",
        "match_tokens": ["founding"],
    }]}
    profiles = pd.DataFrame([{
        "ticker": "AAA",
        "endpoint": "profile",
        "ceo": "Professional Successor",
        "snapshot_as_of": "2026-08-05",
    }])
    row = prepare_founder_rows(
        source,
        profiles,
        pd.DataFrame(),
        pd.DataFrame(),
        as_of="2026-08-05",
    )[0]
    assert row["verification_class"] == "mismatch"
    assert "recheck" in row["verification"].lower()


def test_company_map_html_is_additive_and_has_no_execution_path(tmp_path, monkeypatch):
    monkeypatch.setattr(company_maps, "CURRENT_ROOT", tmp_path / "current")
    report, support = build_company_maps_report(
        as_of="2026-08-05",
        output_path=Path(tmp_path) / "company_maps.html",
        fmp_frame=pd.DataFrame(),
        universe=pd.DataFrame(),
        candidates=pd.DataFrame(),
    )
    text = report.read_text(encoding="utf-8")
    payload = support.read_text(encoding="utf-8")
    assert "Founder-led companies and your circle of competence" in text
    assert "Current founder-CEO roster" in text
    assert "Best product-led starting points" in text
    assert "Familiar product, still an opaque business" in text
    assert "CME" in text
    assert "Live actions are disabled" in text
    assert '"circle_excluded"' in payload
    assert '"live_actions_enabled": false' in payload
    assert "/exec-command" not in text
    assert "Send live" not in text
