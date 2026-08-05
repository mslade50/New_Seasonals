"""P/C fear-conditioned fragility bands (2026-08-05) — config + selection
+ lag-1 + fail-closed guards.

The rule (prereg: scratch/ultracode_research/family_pc_fear_band_prereg_
2026-08-05.md, shipped rev 3): the 6 dip-buy family band carriers select
their frag band TABLE by the lag-1 equity P/C fear state. Fear ON ->
1.25x below dial 50 / 1.0x above; fear OFF -> 1.0x / ZERO; stale P/C ->
the incumbent frag_risk_bands (0.25x at >=50). The multiplier set is
CLOSED: {1.25, 1.0, 0.25, 0.0}.
"""
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pc_fear
import strategy_config as sc


FAMILY = ["Weak Close Decent Sznls", "SPY QQQ MonFri Reversion", "Monday Dip",
          "Indices Oversold Bounce", "3x Bear ETF Overbot Fade",
          "Monthly Weak Close"]


# ---------------------------------------------------------------------------
# Config invariants
# ---------------------------------------------------------------------------

def test_carriers_are_exactly_the_frag_band_family():
    pc = {s["name"] for s in sc.STRATEGY_BOOK
          if s.get("execution", {}).get("pc_fear_bands")}
    frag = {s["name"] for s in sc.STRATEGY_BOOK
            if s.get("execution", {}).get("frag_risk_bands")}
    assert pc == frag == set(FAMILY)


def test_all_carriers_share_the_single_constant():
    # Equality, not identity: the GRM import-time scaling deep-copies the
    # execution dicts. The invariant is that no carrier diverges from
    # PC_FEAR_BANDS (and that the tables are NOT GRM-scaled — pure mults).
    for s in sc.STRATEGY_BOOK:
        spec = s.get("execution", {}).get("pc_fear_bands")
        if spec is not None:
            assert spec == sc.PC_FEAR_BANDS, s["name"]


def test_tables_match_prereg_and_multiplier_set_is_closed():
    assert sc.PC_FEAR_BANDS["on"] == [[0, 50, 1.25], [50, 999, 1.0]]
    assert sc.PC_FEAR_BANDS["off"] == [[0, 50, 1.0], [50, 999, 0.0]]
    mults = {m for tbl in sc.PC_FEAR_BANDS.values() for _, _, m in tbl}
    incumbent = {m for s in sc.STRATEGY_BOOK
                 for _, _, m in s.get("execution", {}).get("frag_risk_bands", [])}
    assert mults | incumbent <= {1.25, 1.0, 0.25, 0.0}


def test_incumbent_frag_bands_unchanged():
    for s in sc.STRATEGY_BOOK:
        if s["name"] in FAMILY:
            assert s["execution"]["frag_risk_bands"] == [[50, 999, 0.25]], s["name"]


# ---------------------------------------------------------------------------
# Table selection + band arithmetic
# ---------------------------------------------------------------------------

EXEC = {"frag_risk_bands": [[50, 999, 0.25]],
        "pc_fear_bands": sc.PC_FEAR_BANDS}


@pytest.mark.parametrize("state,dial,expected", [
    ("on", 20.0, 1.25),   # boost below 50
    ("on", 50.0, 1.0),    # full size at/above 50 (boundary: 50 is hi-frag)
    ("on", 80.0, 1.0),
    ("off", 20.0, 1.0),   # normal below 50
    ("off", 50.0, 0.0),   # zeroed at/above 50
    ("off", 999.5, 1.0),  # above table range -> no match -> 1.0 (dial can't
                          # exceed 100 in practice; documents the semantics)
    ("stale", 20.0, 1.0),   # incumbent table
    ("stale", 50.0, 0.25),  # incumbent 0.25x — exactly the pre-2026-08-05 book
])
def test_selection_matrix(state, dial, expected):
    bands = pc_fear.select_bands(EXEC, state)
    assert pc_fear.band_mult(bands, dial) == expected


def test_missing_dial_score_is_fail_open_full_size():
    for state in ("on", "off", "stale"):
        assert pc_fear.band_mult(pc_fear.select_bands(EXEC, state), None) == 1.0


def test_non_carrier_keeps_plain_frag_bands():
    plain = {"frag_risk_bands": [[50, 999, 0.25]]}
    for state in ("on", "off", "stale"):
        assert pc_fear.select_bands(plain, state) == [[50, 999, 0.25]]
    assert pc_fear.select_bands({}, "on") is None
    assert pc_fear.select_bands(None, "on") is None


# ---------------------------------------------------------------------------
# Fear state: lag-1 + staleness (synthetic cache)
# ---------------------------------------------------------------------------

@pytest.fixture()
def synth_cache(tmp_path):
    """300 bdays of equity P/C ending 2026-07-31 (Fri): flat 0.60 with the
    last 15 days at 1.20 -> 10d MA percentile pins at 100 (fear ON)."""
    idx = pd.bdate_range(end="2026-07-31", periods=300)
    vals = np.full(len(idx), 0.60)
    vals[-15:] = 1.20
    df = pd.DataFrame({"equity": vals}, index=idx)
    df.index.name = "date"
    p = str(tmp_path / "pc.parquet")
    df.to_parquet(p)
    return p


def test_lag1_and_staleness(synth_cache):
    # Mon 8/3: newest eligible row <= Fri 7/31 -> age 1, fresh, fear ON
    st = pc_fear.fear_state_asof(dt.date(2026, 8, 3), cache_path=synth_cache)
    assert st["state"] == "on" and st["age_bd"] == 1
    assert st["data_date"] == dt.date(2026, 7, 31)
    assert st["pct"] > pc_fear.FEAR_PCT_THRESHOLD

    # Fri 7/31 itself: lag-1 excludes the same-day row -> data through Thu 7/30
    st2 = pc_fear.fear_state_asof(dt.date(2026, 7, 31), cache_path=synth_cache)
    assert st2["data_date"] == dt.date(2026, 7, 30) and st2["age_bd"] == 1

    # Thu 8/6: age 4 bd > STALE_BD 3 -> stale, pct still reported for audit
    st3 = pc_fear.fear_state_asof(dt.date(2026, 8, 6), cache_path=synth_cache)
    assert st3["state"] == "stale" and st3["age_bd"] == 4
    assert st3["pct"] is not None


def test_low_pc_is_fear_off(tmp_path):
    idx = pd.bdate_range(end="2026-07-31", periods=300)
    vals = np.full(len(idx), 0.60)
    vals[-15:] = 0.30  # collapsing P/C = complacency, never fear
    df = pd.DataFrame({"equity": vals}, index=idx)
    df.index.name = "date"
    p = str(tmp_path / "pc_low.parquet")
    df.to_parquet(p)
    st = pc_fear.fear_state_asof(dt.date(2026, 8, 3), cache_path=p)
    assert st["state"] == "off" and st["pct"] < 15


def test_missing_cache_is_stale():
    st = pc_fear.fear_state_asof(dt.date(2026, 8, 3),
                                 cache_path="Z:/nonexistent/pc.parquet")
    assert st == {"state": "stale", "pct": None, "data_date": None,
                  "age_bd": None}


# ---------------------------------------------------------------------------
# Live/engine parity of the selection layer
# ---------------------------------------------------------------------------

def test_daily_scan_frag_band_mult_matches_helper():
    """daily_scan.frag_band_mult must be a thin wrapper over the same
    select_bands + band_mult the engine uses."""
    import importlib.util
    spec = importlib.util.find_spec("daily_scan")
    if spec is None:
        pytest.skip("daily_scan not importable in this environment")
    import daily_scan
    for state, dial, expected in [
        ("on", 20.0, 1.25), ("off", 55.0, 0.0), ("stale", 55.0, 0.25),
    ]:
        got = daily_scan.frag_band_mult(EXEC, dial, pc_state={"state": state})
        assert got == expected, (state, dial)
    # no pc_state passed -> stale semantics (fail closed)
    assert daily_scan.frag_band_mult(EXEC, 55.0) == 0.25
    # non-carrier unchanged
    assert daily_scan.frag_band_mult({"frag_risk_bands": [[50, 999, 0.25]]},
                                     55.0) == 0.25
    assert daily_scan.frag_band_mult({}, 55.0) == 1.0
