"""Trade Console guards (spec: RISK_TRADE_CONSOLE_2026-07-16.md step 4).

Covers: class precedence + N-floor fallthrough, fingerprint/vintage
degradation, rendered-string regex guards (no order-shaped language, no +EV
without the CI gate, footer disclaimer), schema, and the 400-day vintage
tripwire on the COMMITTED stats file — the one kill criterion with an
executor (this test suite)."""
import datetime as dt
import json
import os
import re
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_risk_json import (
    TC_STATS_PATH,
    _render_trade_console,
    _tc_structure_line,
)
from scripts.build_trade_console_stats import (
    CLASS_PRECEDENCE,
    MIN_READABLE,
    class_fingerprint,
    classify_row,
    tri_state_any,
)


def _stats(built_days_ago=1, dl_n=13, flip=False):
    built = (dt.datetime.utcnow() - dt.timedelta(days=built_days_ago))
    ev = {
        "n_priced": dl_n,
        "spread_ret_on_cost": {"mean": 0.31 if flip else -0.078,
                               "median": -0.6,
                               "ci5_95": [0.05, 0.6] if flip else [-0.87, 0.83]},
        "spread_cost_pct_notional_mean": 1.6,
        "spread_hits": 3, "spread_oracle_mean": 0.04,
        "tail_ret_on_cost": {"mean": -0.33, "median": -1.0,
                             "ci5_95": [-1.0, 0.4]},
        "tail_cost_pct_notional_mean": 0.4,
        "tail_hits": 0, "tail_oracle_mean": 0.2,
    }
    def cls(n, med, pdd10, years=None):
        return {"n_episodes": n, "episode_years": years or [2016, 2020, 2026],
                "n_distinct_years": 3, "max_year_share": 0.3,
                "fwd21_mean_pct": 0.5, "fwd63_mean_pct": 1.6,
                "fwd63_median_pct": med, "fwd63_p10_pct": -11.2,
                "fwd63_p90_pct": 12.0, "fwd63_mean_ci_pct": [-3.0, 6.0],
                "fwd63_drop_best_mean_pct": 0.4,
                "p_fwd63_le_m5": 0.15, "p_fwd63_le_m10": 0.08,
                "p_dd5_63td": 0.29, "p_dd10_63td": pdd10,
                "p_vix28_63td": 0.3, "episode_t_vs_baseline": -0.9,
                "days_in_class": 200, "readable": n >= MIN_READABLE,
                "structures": ev}
    return {
        "built_utc": built.strftime("%Y-%m-%d %H:%M UTC"),
        "data_through": built.strftime("%Y-%m-%d"),
        "class_set_version": "v1 (frozen 2026-07-16, screened post-hoc, one-shot)",
        "fingerprint": "testfp",
        "min_readable_episodes": MIN_READABLE,
        "dispersion_episodes": 5,
        "baseline": cls(37, 4.7, 0.135),
        "classes": {
            "BEAR_2PLUS": cls(17, 3.5, 0.06),
            "DL_TAIL": cls(dl_n, 3.8, 0.231),
            "AR_DRAWDOWN": cls(15, 3.9, 0.20),
            "SRD_TAIL": cls(11, 3.1, 0.18),
            "DA_SOFT": cls(7, 2.9, 0.14),
            "VRC_CONTEXT": cls(15, 4.0, 0.067),
            "NONE": cls(37, 4.7, 0.135),
        },
    }


def _row(**over):
    base = {"n_bear_any": 0, "any_DL": False, "any_AR": False, "any_SRD": False,
            "any_DA": False, "any_VRC": False, "any_DISP": False,
            "near_52w_high": True, "above_200d": True}
    base.update(over)
    return pd.Series(base)


def test_classify_precedence_first_match_wins():
    assert classify_row(_row(n_bear_any=2, any_DL=True)) == "BEAR_2PLUS"
    assert classify_row(_row(n_bear_any=1, any_DL=True)) == "DL_TAIL"
    assert classify_row(_row(n_bear_any=1, any_VRC=True)) == "VRC_CONTEXT"
    assert classify_row(_row()) == "NONE"
    assert CLASS_PRECEDENCE[0] == "BEAR_2PLUS" and CLASS_PRECEDENCE[-1] == "NONE"


def test_n_floor_falls_through_to_none():
    tc = _render_trade_console(_stats(), _row(n_bear_any=1, any_SRD=True),
                               [("Seasonal Rank Divergence", 2)], "2026-07-16", 0)
    assert tc["class_id"] == "NONE"          # SRD has 11 < 12 episodes
    assert tc["state"] == "ok"
    assert "nothing to do" in tc["action_line"]


def test_dl_card_renders_expected_shape():
    tc = _render_trade_console(_stats(), _row(n_bear_any=1, any_DL=True),
                               [("Defensive Leadership", 3)], "2026-07-16", 0)
    assert tc["class_id"] == "DL_TAIL"
    assert tc["headline"].startswith("ELEVATED LEFT TAIL")
    assert "LOW SAMPLE" in tc["headline"]     # 13 episodes -> low-sample tier
    assert "fired 3 sessions ago (now off)" in tc["fired_line"]
    assert "52-week high" in tc["fired_line"]
    assert "13 prior episodes" in tc["dist_line"]
    assert "MODEL-PRICED" in tc["structure_line"]
    assert "held to expiry" in tc["structure_line"]
    assert "expired worthless 13 of 13" in tc["structure_line"]
    assert tc["action_line"].startswith("Historical read:")
    assert tc["flip_gate_passed"] is False
    assert any("Display-only" in c for c in tc["caveats"])
    assert any("multiple-comparisons" in c for c in tc["caveats"])


def test_no_order_shaped_language_and_no_unearned_ev():
    """Regex on RENDERED strings, not source (spec: grep-the-source is theater)."""
    for row, fired in [
        (_row(n_bear_any=1, any_DL=True), [("Defensive Leadership", 0)]),
        (_row(n_bear_any=2, any_DL=True, any_AR=True),
         [("Defensive Leadership", 0), ("Low Absorption Ratio", 1)]),
        (_row(), []),
        (_row(n_bear_any=1, any_VRC=True), [("VIX Range Compression", 0)]),
    ]:
        tc = _render_trade_console(_stats(), row, fired, "2026-07-16", 0)
        rendered = " ".join(str(v) for k, v in tc.items() if k != "caveats")
        rendered += " " + " ".join(tc.get("caveats", []))
        after_prefix = rendered.replace("Historical read:", "")
        assert not re.search(r"\b(buy|sell|open|short)\b", after_prefix,
                             re.IGNORECASE), rendered
        assert not re.search(r"historically \+EV", rendered), \
            "flip-gate language without the CI gate"
        assert not re.search(r"\b\d{3,4}\s?(strike|expiry)\b", rendered)


def test_flip_gate_emits_ev_language_only_when_ci_clears():
    line, flip = _tc_structure_line(_stats(flip=True)["classes"]["DL_TAIL"])
    assert flip is True and "+EV" in line and "MODEL-PRICED" in line
    line, flip = _tc_structure_line(_stats()["classes"]["DL_TAIL"])
    assert flip is False and "+EV" not in line
    assert "neither structure has cleared cost" in line


def test_degraded_states_render_reason():
    stale = _render_trade_console(_stats(built_days_ago=200),
                                  _row(n_bear_any=1, any_DL=True),
                                  [("Defensive Leadership", 0)], "2026-07-16", 0)
    assert stale["state"] == "degraded"
    assert "structure_line" not in stale       # structure block withheld
    assert stale["fired_line"]                 # fired line still renders
    silent = _render_trade_console(_stats(), _row(), [], "2026-07-10", 5)
    assert silent["state"] == "silent" and "stale" in silent["reason"]


def test_dispersion_alone_yields_none_plus_note():
    tc = _render_trade_console(_stats(), _row(n_bear_any=1, any_DISP=True),
                               [("Dispersion", 0)], "2026-07-16", 0)
    assert tc["class_id"] == "NONE"
    assert "too few to condition on" in tc.get("extra_line", "")


def test_tri_state_any_recency_window():
    idx = pd.bdate_range("2026-01-05", periods=20)
    h = pd.Series(False, index=idx)
    h.iloc[5] = True
    out = tri_state_any(h, idx)
    assert bool(out.iloc[5]) and bool(out.iloc[9])   # within 5 sessions
    assert not bool(out.iloc[10])                     # 5 sessions elapsed


def test_committed_stats_file_vintage_and_schema():
    """The 400-day tripwire — the only kill criterion with an executor."""
    if not os.path.exists(TC_STATS_PATH):
        pytest.skip("stats file not present in this checkout")
    with open(TC_STATS_PATH, encoding="utf-8") as f:
        stats = json.load(f)
    built = dt.datetime.strptime(stats["built_utc"][:10], "%Y-%m-%d")
    age = (dt.datetime.utcnow() - built).days
    assert age <= 400, (
        f"trade_console_stats.json is {age} days old — regenerate via "
        f"scripts/build_trade_console_stats.py after a deliberate review, "
        f"or retire the console")
    assert stats["fingerprint"] == class_fingerprint([
        "Distribution Dominance", "VIX Range Compression",
        "Defensive Leadership", "Pre-FOMC Rally", "Low Absorption Ratio",
        "Seasonal Rank Divergence", "Dispersion"])
    for cid in ("BEAR_2PLUS", "DL_TAIL", "NONE"):
        assert cid in stats["classes"]
