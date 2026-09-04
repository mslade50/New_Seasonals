"""Guards for the private site's all-portfolio-overlays-off comparison book."""

import inspect
from pathlib import Path

from daily_portfolio_report import build_full_strategy_book
from pages.strat_backtester import process_signals_fast
from scripts.build_trade_ledger import (
    OVERLAY_EXECUTION_KEYS,
    OVERLAY_LAB_SPECS,
    _overlay_spec_active,
    strip_portfolio_overlays,
)
from scripts.site_r2_pipeline import GENERATED_INPUTS
from strategy_config import STRATEGY_BOOK


ROOT = Path(__file__).resolve().parents[1]


def test_overlay_free_book_keeps_core_book_and_removes_every_named_overlay():
    production = build_full_strategy_book()
    clean = strip_portfolio_overlays(production)

    assert clean is not production
    assert [s["name"] for s in clean] == [s["name"] for s in production]
    assert [s["universe_tickers"] for s in clean] == [
        s["universe_tickers"] for s in production
    ]
    native_risk = {
        s["name"]: s["execution"]["risk_bps"] for s in STRATEGY_BOOK
    }

    for strat in clean:
        settings = strat["settings"]
        execution = strat["execution"]
        assert settings.get("dial_filters") == []
        assert settings.get("use_t1_gap_kill") is not True
        assert OVERLAY_EXECUTION_KEYS.isdisjoint(execution)
        assert execution["risk_bps"] == native_risk[strat["name"]]

        original = next(
            item for item in production
            if item["name"] == strat["name"]
            and item["universe_tickers"] == strat["universe_tickers"]
        )
        for core_key in (
            "hold_days", "stop_atr", "tgt_atr", "use_stop_loss",
            "use_take_profit",
        ):
            assert execution.get(core_key) == original["execution"].get(core_key)


def test_engine_and_cloud_bundle_expose_overlay_free_controls():
    assert "portfolio_overlays_enabled" in inspect.signature(
        process_signals_fast).parameters
    assert "portfolio_overlay_names" in inspect.signature(
        process_signals_fast).parameters
    generated = {item.name: item for item in GENERATED_INPUTS}
    assert generated["ledger_overlay_free"].required is True
    assert generated["ledger_overlay_free_daily"].required is True
    assert generated["overlay_lab"].required is True


def test_overlay_lab_has_unique_controls_for_every_current_overlay_family():
    ids = [spec["id"] for spec in OVERLAY_LAB_SPECS]
    assert len(ids) == len(set(ids)) == 16
    production = build_full_strategy_book()
    assert all(_overlay_spec_active(spec, production) for spec in OVERLAY_LAB_SPECS)
    assert {
        "risk_dial_gates", "fragility_pc_sizing", "earnings_blackouts",
        "ovs_path2_sizing", "cross_strategy_overlap",
        "overflow_risk_override", "per_strategy_daily_cap",
    }.issubset(ids)


def test_portfolio_has_one_click_production_and_overlay_free_modes():
    html = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "site" / "assets" / "portfolio.js").read_text(encoding="utf-8")

    assert 'id="portfolioMode"' in html
    assert 'data-production-only' in html
    assert 'data-overlay-only' in html
    assert 'id="overlayLabControls"' in html
    assert 'id="overlayLabChart"' in html
    assert '["production", "Production overlays"]' in js
    assert '["overlay_free", "All overlays off"]' in js
    assert 'url.searchParams.set("book", "overlay_free")' in js
    assert 'url.searchParams.delete("book")' in js
    assert 'data/overlay_free/' in js
    assert 'All overlays off (fixed)' in js
    assert 'Current production (fixed)' in js
    assert 'Custom selection' in js
