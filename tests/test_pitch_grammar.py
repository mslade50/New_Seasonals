"""Guards for pitch_grammar.py — the Daily Pitch hard requirements.

Spec: daily_pitch_agent_spec_2026-08-06.html sections 4 and 5. These are the
rules an idea cannot be published without, so each one gets a test that fails
if somebody loosens the grammar instead of fixing an idea.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pitch_grammar as pg  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "pitch_ideas_fixture.json"


@pytest.fixture()
def payload():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture()
def idea(payload):
    return copy.deepcopy(payload["ideas"][0])


def synth_prices(ticker="TEST", periods=400, close=100.0, atr_pct=1.0):
    """Flat-ish bars with a constant true range, so Wilder-14 converges to a
    known ATR and derived levels are checkable by hand."""
    idx = pd.bdate_range(end="2026-08-05", periods=periods)
    band = close * atr_pct / 100.0
    return pd.DataFrame({
        "ticker": ticker, "date": idx,
        "Open": close, "High": close + band / 2, "Low": close - band / 2,
        "Close": close, "Volume": 1e6,
    })


def contexts_for(*tickers, **kw):
    prices = pd.concat([synth_prices(t, **kw) for t in tickers])
    return {t: pg.price_context(prices, t, pd.Timestamp("2026-08-06"))
            for t in tickers}


# ---------------------------------------------------------------------------
# product rules
# ---------------------------------------------------------------------------
def test_product_rules_frozen():
    assert pg.IDEA_COUNT == 3
    assert pg.MAX_GRADE_C == 1
    assert pg.REPEAT_BLOCK_TD == 10
    assert pg.DEFAULT_RISK_BPS == 30.0
    assert pg.ENTRY_TYPES == {"MOO", "MOC", "LIMIT"}
    assert pg.EXIT_ORDER_TYPES == {"MOC", "MOO"}


def test_fixture_payload_is_valid(payload):
    assert pg.validate_payload(payload) == []


def test_exactly_three_ideas(payload):
    payload["ideas"] = payload["ideas"][:2]
    assert any("exactly 3 ideas" in e for e in pg.validate_payload(payload))


def test_one_grade_c_cap(payload):
    payload["ideas"][0]["grade"] = "C"
    payload["ideas"][0]["evidence"]["n"] = 9
    errors = pg.validate_payload(payload)
    assert any("grade-C" in e for e in errors)


def test_time_stop_is_mandatory(idea):
    idea["exit"].pop("time_td")
    assert any("time_td is required" in e for e in pg.validate_idea(idea, "x"))


def test_time_stop_cannot_exceed_horizon(idea):
    idea["exit"]["time_td"] = idea["horizon_td"] + 1
    assert any("exceeds the stated horizon" in e
               for e in pg.validate_idea(idea, "x"))


@pytest.mark.parametrize("entry", [
    {"type": "GTC"},
    {"type": "LIMIT", "atr_mult": -0.25},                      # no anchor
    {"type": "LIMIT", "anchor": "MIDPOINT", "atr_mult": -0.25},
    {"type": "LIMIT", "anchor": "CLOSE"},                      # no atr_mult
    {"type": "LIMIT", "anchor": "CLOSE", "atr_mult": -0.25, "fill_window_td": 0},
])
def test_entry_vocabulary_is_closed(idea, entry):
    idea["entry"] = entry
    assert pg.validate_idea(idea, "x")


def test_evidence_and_survived_are_required(idea):
    for field in ("summary", "script", "n", "control", "era_note"):
        broken = copy.deepcopy(idea)
        broken["evidence"].pop(field)
        assert pg.validate_idea(broken, "x"), f"missing evidence.{field} passed"
    broken = copy.deepcopy(idea)
    broken.pop("survived")
    assert any("survived" in e for e in pg.validate_idea(broken, "x"))


@pytest.mark.parametrize("grade,n,ok", [
    ("A", 80, True), ("A", 30, False),
    ("B", 22, True), ("B", 9, False),
    ("C", 9, True), ("C", 40, False),
])
def test_grade_matches_sample_size(idea, grade, n, ok):
    idea["grade"], idea["evidence"]["n"] = grade, n
    errors = [e for e in pg.validate_idea(idea, "x") if "grade" in e]
    assert (errors == []) is ok


def test_novelty_axis_must_be_known(idea):
    idea["novelty_axis"] = "vibes"
    assert any("novelty_axis" in e for e in pg.validate_idea(idea, "x"))


def test_futures_leg_needs_contract_proxy_and_multiplier(payload):
    leg = payload["ideas"][2]["legs"][0]
    assert leg["sec_type"] == "FUT"
    for field in ("contract", "proxy_ticker", "multiplier"):
        broken = copy.deepcopy(payload["ideas"][2])
        broken["legs"][0].pop(field)
        assert pg.validate_idea(broken, "x"), f"FUT leg without {field} passed"


# ---------------------------------------------------------------------------
# repetition control
# ---------------------------------------------------------------------------
def test_fingerprint_ignores_prose_but_not_structure(idea):
    reworded = copy.deepcopy(idea)
    reworded["title"] = "Completely different words about the same trade"
    reworded["thesis"] = "Different prose entirely."
    assert pg.fingerprint(reworded) == pg.fingerprint(idea)

    flipped = copy.deepcopy(idea)
    flipped["legs"][0]["side"] = "SHORT"
    assert pg.fingerprint(flipped) != pg.fingerprint(idea)


def test_repeat_inside_the_window_needs_changed_since(payload):
    fp = pg.fingerprint(payload["ideas"][0])
    errors = pg.validate_payload(payload, {fp: "2026-08-04"})
    assert any("repetition window" in e for e in errors)

    payload["ideas"][0]["changed_since"] = "The dial crossed 50 since then."
    assert pg.validate_payload(payload, {fp: "2026-08-04"}) == []


def test_two_ideas_cannot_be_the_same_trade(payload):
    payload["ideas"][1] = copy.deepcopy(payload["ideas"][0])
    assert any("duplicates idea" in e for e in pg.validate_payload(payload))


# ---------------------------------------------------------------------------
# ATR and sizing
# ---------------------------------------------------------------------------
def test_wilder_atr_matches_the_risk_tab_generator():
    from scripts.build_atr_downside_stats import wilder_atr as reference
    rng = np.random.default_rng(11)
    close = 100 + np.cumsum(rng.normal(0, 1, 300))
    high, low = close + 1.2, close - 0.9
    ours = pg.wilder_atr(high, low, close)
    theirs = reference(high, low, close)
    assert np.allclose(ours[20:], theirs[20:], equal_nan=True)


def test_risk_sizing_hits_the_stated_dollar_risk(idea):
    ctx = contexts_for("GLD", "SLV", close=100.0, atr_pct=2.0)
    rows = pg.build_orders(idea, ctx, "2026-08-06", "t-1",
                           account_value=1_000_000)
    # 30 bps of 1m = $3,000, split evenly, each leg sized on 1.0 ATR = $2.
    assert [r["Quantity"] for r in rows] == [750, 750]
    assert sum(r["Risk_Amt"] for r in rows) == pytest.approx(3000.0, abs=1.0)


def test_futures_multiplier_scales_size_and_notional(payload):
    idea = copy.deepcopy(payload["ideas"][2])
    ctx = contexts_for("DX-Y.NYB", close=100.0, atr_pct=1.0)
    rows = pg.build_orders(idea, ctx, "2026-08-06", "t-3",
                           account_value=1_000_000)
    # 20 bps of 1m = $2,000; 1 ATR = 1.0 point x $1,000 = $1,000 per contract.
    assert rows[0]["Quantity"] == 2
    assert rows[0]["Notional"] == pytest.approx(2 * 100.0 * 1000)


def test_limit_and_bracket_levels_come_off_the_close(payload):
    idea = copy.deepcopy(payload["ideas"][1])   # XLU, close -0.25 ATR
    ctx = contexts_for("XLU", close=100.0, atr_pct=2.0)
    row = pg.build_orders(idea, ctx, "2026-08-06", "t-2")[0]
    assert row["Limit_Price"] == pytest.approx(99.5)     # 100 - 0.25 x 2
    assert row["Stop_Price"] == pytest.approx(97.0)      # limit - 1.25 x 2
    assert row["Target_Price"] == pytest.approx(103.5)   # limit + 2.0 x 2


def test_short_leg_bracket_flips(payload):
    idea = copy.deepcopy(payload["ideas"][1])
    idea["legs"][0]["side"] = "SHORT"
    ctx = contexts_for("XLU", close=100.0, atr_pct=2.0)
    row = pg.build_orders(idea, ctx, "2026-08-06", "t-2")[0]
    assert row["Action"] == "SELL_SHORT"
    assert row["Stop_Price"] == pytest.approx(102.0)
    assert row["Target_Price"] == pytest.approx(95.5)


def test_sizing_stop_defaults_by_horizon():
    assert pg.sizing_stop_atr(5) == 1.0
    assert pg.sizing_stop_atr(10) == 1.3
    assert pg.sizing_stop_atr(21) == 1.6


def test_unplaceable_size_raises(idea):
    ctx = contexts_for("GLD", "SLV", close=100.0, atr_pct=2.0)
    with pytest.raises(pg.PitchGrammarError):
        pg.build_orders(idea, ctx, "2026-08-06", "t-1", account_value=100.0)


def test_risk_budget_bounds_are_atr_denominated(idea):
    ctx = contexts_for("GLD", "SLV", close=100.0, atr_pct=2.0)
    rows = pg.build_orders(idea, ctx, "2026-08-06", "t-1")
    assert pg.check_risk_budget(rows) == []
    fat = [{**r, "Risk_Amt": 9_000.0} for r in rows]
    assert any("per-idea bound" in e for e in pg.check_risk_budget(fat))


# ---------------------------------------------------------------------------
# placement routing
# ---------------------------------------------------------------------------
def test_close_anchored_limit_is_auction_placeable(payload):
    assert pg.auto_placement(payload["ideas"][1])[0] == "auction"


def test_moc_with_a_time_exit_only_is_auction_placeable(payload):
    assert pg.auto_placement(payload["ideas"][0])[0] == "auction"


def test_moc_with_a_price_stop_is_manual(idea):
    idea["exit"]["stop_atr"] = 1.0
    assert pg.auto_placement(idea)[0] == "manual"


def test_open_anchored_limit_routes_to_the_open_pass(idea):
    idea["entry"] = {"type": "LIMIT", "anchor": "OPEN", "atr_mult": -0.25,
                     "fill_window_td": 1}
    assert pg.auto_placement(idea)[0] == "open"


def test_futures_leg_is_manual(payload):
    assert pg.auto_placement(payload["ideas"][2])[0] == "manual"


def test_open_anchored_row_carries_no_fabricated_prices(idea):
    idea["entry"] = {"type": "LIMIT", "anchor": "OPEN", "atr_mult": -0.25,
                     "fill_window_td": 1}
    idea["exit"]["stop_atr"] = 1.0
    ctx = contexts_for("GLD", "SLV", close=100.0, atr_pct=2.0)
    for row in pg.build_orders(idea, ctx, "2026-08-06", "t-1"):
        assert row["Limit_Price"] == ""
        assert row["Stop_Price"] == ""


def test_multi_session_fill_window_stamps_gtd(idea):
    idea["entry"] = {"type": "LIMIT", "anchor": "CLOSE", "atr_mult": -0.25,
                     "fill_window_td": 3}
    ctx = contexts_for("GLD", "SLV", close=100.0, atr_pct=2.0)
    rows = pg.build_orders(idea, ctx, "2026-08-06", "t-1")
    assert {r["TIF"] for r in rows} == {"GTD"}
    assert rows[0]["Entry_Expire_Date"] == "2026-08-10"   # T+0 counts


def test_exit_dates_use_the_nyse_calendar(idea):
    ctx = contexts_for("GLD", "SLV", close=100.0, atr_pct=2.0)
    row = pg.build_orders(idea, ctx, "2026-08-06", "t-1")[0]
    assert row["Time_Exit_Date"] == "2026-08-13"    # 5 sessions after Thu 8/6
    assert pg.time_exit_date("2026-07-01", 4) == pd.Timestamp("2026-07-08")


# ---------------------------------------------------------------------------
# stand-down: the morning that ships nothing
#
# 2026-08-07 killed 24 candidates and had nowhere to put the verdict, so a real
# all-kill result looked exactly like a crashed task. These guard the shape of
# the replacement AND its cost: standing down must stay harder than shipping,
# or it quietly becomes the way out of every hard morning.
# ---------------------------------------------------------------------------
@pytest.fixture()
def stand_down(tmp_path):
    checks = tmp_path / "checks"
    checks.mkdir()
    (checks / "c1_probe.py").write_text("# a real check\n", encoding="utf-8")
    (checks / pg.SURFACE_MAP_NAME).write_text("# surface\n", encoding="utf-8")
    return {
        "asof": "2026-08-07",
        "ideas": [],
        "stand_down": {
            "reason": "x" * pg.STAND_DOWN_MIN_REASON,
            "candidates_considered": 24,
            "axes": ["relative_value", "inversion", "event_fingerprint",
                     "flow_mechanics"],
            "asset_classes": ["us_large", "rates", "gold_miners", "energy"],
            "checks_dir": str(checks),
            "closest": [{"title": "Short TLT at the 52w low",
                         "decisive": "excess +1.263%, Welch t=2.10",
                         "why_died": "8 of 12 episodes are 2021-22"}],
        },
        "killed": [{"title": f"k{i}", "reason": "died",
                    "novelty_axis": "inversion"}
                   for i in range(pg.STAND_DOWN_MIN_KILLED)],
    }


def test_stand_down_floors_frozen():
    assert pg.STAND_DOWN_MIN_CANDIDATES == 8
    assert pg.STAND_DOWN_MIN_AXES == 4
    assert pg.STAND_DOWN_MIN_KILLED == 6
    assert pg.STAND_DOWN_MIN_REASON == 120


def test_valid_stand_down_passes(stand_down):
    assert pg.is_stand_down(stand_down)
    assert pg.validate_payload(stand_down) == []


def test_stand_down_does_not_need_three_ideas(stand_down):
    assert not any("exactly 3" in e for e in pg.validate_payload(stand_down))


def test_stand_down_cannot_also_ship_ideas(payload, stand_down):
    stand_down["ideas"] = payload["ideas"]
    assert any("ships nothing" in e for e in pg.validate_payload(stand_down))


def test_thin_sweep_cannot_stand_down(stand_down):
    stand_down["stand_down"]["candidates_considered"] = 3
    assert any("candidates_considered" in e
               for e in pg.validate_payload(stand_down))


def test_single_axis_sweep_cannot_stand_down(stand_down):
    stand_down["stand_down"]["axes"] = ["relative_value"]
    assert any("axes" in e for e in pg.validate_payload(stand_down))


def test_duplicate_axes_do_not_count(stand_down):
    stand_down["stand_down"]["axes"] = ["inversion", "Inversion", " inversion",
                                        "inversion"]
    assert any("axes" in e for e in pg.validate_payload(stand_down))


def test_stand_down_needs_named_kills(stand_down):
    stand_down["killed"] = stand_down["killed"][:2]
    assert any("payload.killed" in e for e in pg.validate_payload(stand_down))


def test_stand_down_kills_need_a_reason(stand_down):
    stand_down["killed"][0]["reason"] = ""
    assert any("reason is required" in e
               for e in pg.validate_payload(stand_down))


def test_one_line_reason_is_not_a_verdict(stand_down):
    stand_down["stand_down"]["reason"] = "nothing worked today"
    assert any("reason" in e for e in pg.validate_payload(stand_down))


def test_stand_down_needs_checks_on_disk(stand_down, tmp_path):
    stand_down["stand_down"]["checks_dir"] = str(tmp_path / "nope")
    assert any("does not exist" in e for e in pg.validate_payload(stand_down))


def test_empty_checks_dir_is_not_evidence(stand_down, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    stand_down["stand_down"]["checks_dir"] = str(empty)
    assert any("no .py checks" in e for e in pg.validate_payload(stand_down))


def test_near_misses_are_required(stand_down):
    stand_down["stand_down"]["closest"] = []
    assert any("closest" in e for e in pg.validate_payload(stand_down))


@pytest.mark.parametrize("field", ["title", "decisive", "why_died"])
def test_near_miss_needs_the_number_it_turned_on(stand_down, field):
    stand_down["stand_down"]["closest"][0][field] = ""
    assert any(field in e for e in pg.validate_payload(stand_down))


def test_stand_down_asof_still_required(stand_down):
    stand_down["asof"] = ""
    assert any("asof" in e for e in pg.validate_payload(stand_down))


# --- coverage: axis variety is not asset coverage --------------------------
# 2026-08-07 hit seven novelty axes and still ran every calendar-anchored
# check on SPY. These two floors are the difference between "I thought about
# it several ways" and "I looked at the whole surface".
def test_coverage_floors_frozen():
    assert pg.STAND_DOWN_MIN_ASSET_CLASSES == 4
    assert pg.SURFACE_MAP_NAME == "00_surface_map.md"


def test_an_all_equity_sweep_cannot_declare_the_market_empty(stand_down):
    stand_down["stand_down"]["asset_classes"] = ["us_large"]
    assert any("asset_classes" in e for e in pg.validate_payload(stand_down))


def test_missing_asset_classes_is_an_error_not_a_default(stand_down):
    stand_down["stand_down"].pop("asset_classes")
    assert any("asset_classes" in e for e in pg.validate_payload(stand_down))


def test_duplicate_asset_classes_do_not_count(stand_down):
    stand_down["stand_down"]["asset_classes"] = ["rates", "Rates", " rates",
                                                 "RATES"]
    assert any("asset_classes" in e for e in pg.validate_payload(stand_down))


def test_many_axes_do_not_substitute_for_coverage(stand_down):
    stand_down["stand_down"]["axes"] = sorted(pg.NOVELTY_AXES)   # all seven
    stand_down["stand_down"]["asset_classes"] = ["us_large", "us_small"]
    errors = pg.validate_payload(stand_down)
    assert not any("axes" in e and "asset" not in e for e in errors)
    assert any("asset_classes" in e for e in errors)


def test_stand_down_needs_the_surface_map(stand_down, tmp_path):
    checks = Path(stand_down["stand_down"]["checks_dir"])
    (checks / pg.SURFACE_MAP_NAME).unlink()
    assert any(pg.SURFACE_MAP_NAME in e for e in pg.validate_payload(stand_down))


def test_checks_without_a_map_are_a_sweep_without_a_survey(stand_down):
    """Both diagnoses fire independently, so a run that wrote neither sees
    both lines rather than only the first."""
    checks = Path(stand_down["stand_down"]["checks_dir"])
    (checks / pg.SURFACE_MAP_NAME).unlink()
    (checks / "c1_probe.py").unlink()
    errors = pg.validate_payload(stand_down)
    assert any("no .py checks" in e for e in errors)
    assert any(pg.SURFACE_MAP_NAME in e for e in errors)


# --- directed ideas --------------------------------------------------------
# The exactly-three rule stops the AGENT hedging down to one safe idea. It was
# never meant to block McKinley adding one he asked for, which is what happened
# on 2026-08-07 after he overruled a kill.
def test_directed_field_frozen():
    assert pg.DIRECTED_FIELD == "directed_by"


def test_a_directed_publish_may_carry_one_idea(payload):
    payload["ideas"] = payload["ideas"][:1]
    payload["ideas"][0]["directed_by"] = "mckinley"
    assert pg.validate_payload(payload) == []


def test_a_directed_publish_still_caps_at_three(payload):
    payload["ideas"] = payload["ideas"] + [copy.deepcopy(payload["ideas"][0])]
    for i in payload["ideas"]:
        i["directed_by"] = "mckinley"
    assert any("1 to 3 ideas" in e for e in pg.validate_payload(payload))


def test_an_undirected_single_idea_still_fails(payload):
    payload["ideas"] = payload["ideas"][:1]
    assert any("exactly 3 ideas" in e for e in pg.validate_payload(payload))


def test_a_blank_directed_by_does_not_unlock_the_count(payload):
    payload["ideas"] = payload["ideas"][:1]
    payload["ideas"][0]["directed_by"] = "   "
    assert any("exactly 3 ideas" in e for e in pg.validate_payload(payload))


def test_directed_does_not_relax_any_other_rule(payload):
    """The count is the ONLY thing directed_by changes."""
    payload["ideas"] = payload["ideas"][:1]
    payload["ideas"][0]["directed_by"] = "mckinley"
    for field in ("survived", "what_kills_it", "overlap"):
        broken = copy.deepcopy(payload)
        broken["ideas"][0].pop(field, None)
        assert pg.validate_payload(broken), f"directed idea passed without {field}"
    broken = copy.deepcopy(payload)
    broken["ideas"][0]["exit"].pop("time_td")
    assert pg.validate_payload(broken), "directed idea passed without a time stop"


def test_directed_ideas_helper_finds_them(payload):
    assert pg.directed_ideas(payload) == []
    payload["ideas"][1]["directed_by"] = "mckinley"
    assert len(pg.directed_ideas(payload)) == 1
