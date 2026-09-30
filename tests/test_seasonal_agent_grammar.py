"""Guards for the Daily Seasonal grammar extensions (pitch_grammar.py,
product="seasonal"). Live rule: docs/claude_ref/daily_seasonal.md.

Each extension is product-gated, so every test here has a pitch twin proving
the pitch rules did not move.
"""
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pitch_grammar as pg  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "pitch_ideas_fixture.json"


@pytest.fixture()
def seasonal_root(tmp_path, monkeypatch):
    root = tmp_path / "seasonal_checks"
    root.mkdir()
    monkeypatch.setattr(pg, "SEASONAL_CHECKS_ROOT", root)
    return root


def seasonal_idea(**over) -> dict:
    idea = copy.deepcopy(json.loads(FIXTURE.read_text(encoding="utf-8"))["ideas"][0])
    idea["horizon_td"] = 42
    idea["exit"] = {"time_td": 42, "time_order": "MOC", "stop_atr": 2.0}
    idea["sizing"] = {"mode": "risk_bps", "risk_bps": 30, "stop_atr_for_sizing": 2.0}
    idea["evidence"]["script"] = "x.py"
    idea["evidence"]["dev_script"] = "y.py"
    idea.update(over)
    return idea


def errors_for(idea, product="seasonal"):
    return pg.validate_idea(idea, "idea 1", product)


# --- horizon ----------------------------------------------------------------
def test_seasonal_horizon_cap_is_63():
    assert pg.MAX_HORIZON_TD_BY_PRODUCT["seasonal"] == 63
    idea = seasonal_idea(horizon_td=63, exit={"time_td": 63, "stop_atr": 2.0})
    assert errors_for(idea) == []
    idea = seasonal_idea(horizon_td=64, exit={"time_td": 63, "stop_atr": 2.0})
    assert any("horizon_td" in e for e in errors_for(idea))


def test_pitch_horizon_cap_unchanged():
    assert pg.MAX_HORIZON_TD_BY_PRODUCT["pitch"] == pg.MAX_HORIZON_TD == 63
    idea = seasonal_idea(horizon_td=64, sizing=None)
    assert any("horizon_td" in e for e in errors_for(idea, "pitch"))


def test_time_td_bounded_by_horizon():
    idea = seasonal_idea(horizon_td=20, exit={"time_td": 21, "stop_atr": 2.0})
    assert any("exceeds the stated horizon" in e for e in errors_for(idea))


# --- trail ------------------------------------------------------------------
def test_trail_valid_with_and_without_a_stop():
    with_stop = seasonal_idea(exit={"time_td": 42, "stop_atr": 2.0,
                                    "trail": {"arm_atr": 1.5, "trail_atr": 1.0}})
    assert errors_for(with_stop) == []
    no_stop = seasonal_idea(exit={"time_td": 42,
                                  "trail": {"arm_atr": 1.5, "trail_atr": 1.0}},
                            sizing={"risk_bps": 30, "stop_atr_for_sizing": 3.0})
    assert errors_for(no_stop) == []


@pytest.mark.parametrize("trail", [
    {"arm_atr": 0, "trail_atr": 1.0},
    {"arm_atr": 1.5, "trail_atr": -1},
    {"arm_atr": 1.5},
    {"arm_atr": "1.5", "trail_atr": 1.0},
    {"arm_atr": 1.5, "trail_atr": 1.0, "step": 2},
    [1.5, 1.0],
])
def test_bad_trails_are_refused(trail):
    idea = seasonal_idea(exit={"time_td": 42, "stop_atr": 2.0, "trail": trail})
    assert any("trail" in e for e in errors_for(idea))


def test_trail_is_illegal_on_the_pitch():
    idea = seasonal_idea(horizon_td=5, sizing=None,
                         exit={"time_td": 5, "trail": {"arm_atr": 1, "trail_atr": 1}})
    assert any("Daily Seasonal extension" in e for e in errors_for(idea, "pitch"))


def test_a_trail_makes_the_idea_manual():
    idea = seasonal_idea(entry={"type": "LIMIT", "anchor": "CLOSE", "atr_mult": -0.3},
                         exit={"time_td": 42, "trail": {"arm_atr": 1.5, "trail_atr": 1.0}})
    pass_name, reason = pg.auto_placement(idea)
    assert pass_name == "manual" and "trail" in reason
    assert "trail 1 ATR" in pg.exit_label(idea)


# --- sizing -----------------------------------------------------------------
@pytest.mark.parametrize("bps,ok", [(15, True), (30, True), (50, True),
                                    (14.9, False), (50.1, False), (0, False)])
def test_seasonal_risk_bps_band(bps, ok):
    idea = seasonal_idea(sizing={"risk_bps": bps, "stop_atr_for_sizing": 2.0})
    assert (errors_for(idea) == []) is ok


def test_risk_bps_defaults_to_30_when_omitted():
    idea = seasonal_idea(sizing={"stop_atr_for_sizing": 2.0})
    assert errors_for(idea) == []
    prices = pd.DataFrame({
        "ticker": "GLD", "date": pd.bdate_range(end="2026-09-29", periods=60),
        "Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0, "Volume": 1e6})
    prices = pd.concat([prices, prices.assign(ticker="SLV")])
    ctx = {t: pg.price_context(prices, t, pd.Timestamp("2026-09-30"))
           for t in ("GLD", "SLV")}
    rows = pg.build_orders(idea, ctx, "2026-09-30", "2026-09-30-S1",
                           product="seasonal")
    assert rows[0]["Sizing_Note"].startswith("risk_bps 30")
    assert rows[0]["Scan_Source"] == "Seasonal_Agent"
    assert sum(r["Risk_Amt"] for r in rows) <= pg.ACCOUNT_VALUE * 30 / 1e4 + 1


def test_seasonal_sizing_requires_stop_atr_for_sizing():
    idea = seasonal_idea(sizing={"risk_bps": 30})
    assert any("stop_atr_for_sizing is required" in e for e in errors_for(idea))
    assert any("sizing is required" in e for e in errors_for(seasonal_idea(sizing=None)))
    idea = seasonal_idea(sizing={"risk_bps": 30, "stop_atr_for_sizing": 0.8})
    assert any("1 ATR floor" in e for e in errors_for(idea))
    idea = seasonal_idea(sizing={"mode": "nav_pct", "nav_pct": 0.1,
                                 "stop_atr_for_sizing": 2.0})
    assert any("risk_bps only" in e for e in errors_for(idea))


@pytest.mark.parametrize("stop,ok", [(3.0, True), (4.5, True), (2.99, False),
                                     (1.0, False)])
def test_time_only_exit_needs_a_3_atr_catastrophe_distance(stop, ok):
    idea = seasonal_idea(exit={"time_td": 42},
                         sizing={"risk_bps": 30, "stop_atr_for_sizing": stop})
    errs = errors_for(idea)
    assert (errs == []) is ok
    if not ok:
        assert any("catastrophe" in e for e in errs)


def test_pitch_caps_still_bind_a_seasonal_slate():
    rows = [{"Idea_Id": f"i{i}", "Risk_Amt": pg.ACCOUNT_VALUE * 55 / 1e4}
            for i in range(3)]
    errs = pg.check_risk_budget(rows)
    assert any("150 bps" in e for e in errs)


# --- pitch defaults unchanged -------------------------------------------------
def test_pitch_sizing_rules_unchanged():
    idea = seasonal_idea(horizon_td=5, exit={"time_td": 5}, sizing=None)
    assert errors_for(idea, "pitch") == []            # sizing optional
    idea["sizing"] = {"risk_bps": 80}                  # pitch allows up to 100
    assert errors_for(idea, "pitch") == []
    idea["sizing"] = {"risk_bps": 120}
    assert any("(0, 100]" in e for e in errors_for(idea, "pitch"))
    assert pg.validate_idea(idea, "idea 1") == errors_for(idea, "pitch")


def test_pitch_rows_carry_no_trail_columns():
    idea = seasonal_idea(horizon_td=5, exit={"time_td": 5}, sizing=None)
    prices = pd.DataFrame({
        "ticker": "GLD", "date": pd.bdate_range(end="2026-09-29", periods=60),
        "Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0, "Volume": 1e6})
    prices = pd.concat([prices, prices.assign(ticker="SLV")])
    ctx = {t: pg.price_context(prices, t, pd.Timestamp("2026-09-30"))
           for t in ("GLD", "SLV")}
    rows = pg.build_orders(idea, ctx, "2026-09-30", "2026-09-30-1")
    assert "Trail_ATR" not in rows[0] and rows[0]["Scan_Source"] == "Pitch"


def test_unknown_product_is_refused():
    assert pg.validate_payload({"asof": "2026-09-30", "ideas": []},
                               product="posts")[0].startswith("unknown product")


def test_survey_root_follows_the_product(seasonal_root, checks_root):
    payload = {"asof": "2026-09-30", "ideas": [seasonal_idea()]}
    errs = pg.validate_survey_evidence(payload, product="seasonal")
    assert any(str(seasonal_root) in e for e in errs)
    errs = pg.validate_survey_evidence(payload)
    assert any(str(checks_root) in e for e in errs)
    assert pg.default_checks_root("pitch") == checks_root
