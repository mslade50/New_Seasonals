"""3x Bear ETF Overbot Fade: config invariants + same-day signal de-rate.

Guards the 2026-07-07 carve-out (bear-equity 3x names moved from the generic
3x ETF Overbot Fade into a looser bear-only fade) and the shared de-rate rule
(strategy_config.same_day_derate_mult) applied by daily_scan post-pass 5c and
strat_backtester sizing 3b4. Evidence: scratch/lev3x_fade_class_study.py,
lev3x_fade_bear_episodes.py, lev3x_fade_bear_sizing_rule.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _NoOp:
    def __getattr__(self, name): return self
    def __call__(self, *a, **k): return self
    def __bool__(self): return False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **k):
        def deco(fn): return fn
        return deco
    cache_resource = cache_data


sys.modules['streamlit'] = _NoOp()

from strategy_config import (
    STRATEGY_BOOK, LEV3X_ALL, LEV3X_BEAR_EQ, GLOBAL_RISK_MULTIPLIER,
    same_day_derate_mult,
)

BEAR = "3x Bear ETF Overbot Fade"
PARENT = "3x ETF Overbot Fade"


def _strat(name):
    for s in STRATEGY_BOOK:
        if s["name"] == name:
            return s
    raise AssertionError(f"strategy {name!r} not in STRATEGY_BOOK")


def test_universes_disjoint_and_cover_lev3x():
    bear = _strat(BEAR)
    parent = _strat(PARENT)
    b, p = set(bear["universe_tickers"]), set(parent["universe_tickers"])
    assert b == set(LEV3X_BEAR_EQ)
    assert not (b & p), "bear/parent universes overlap -> same-day cross-fire"
    assert b | p == set(LEV3X_ALL), "carve-out must partition LEV3X_ALL exactly"


def test_bear_filters_looser_but_lt_leader_filter_intact():
    pf = {f["window"]: f for f in _strat(BEAR)["settings"]["perf_filters"]}
    for w in (2, 5, 10, 21):
        assert pf[w]["thresh"] == 80.0 and pf[w]["logic"] == ">"
        assert pf[w]["consecutive"] == 1
    # 126/252 < 65 is LOAD-BEARING (keeps the fade from shorting sustained
    # bear markets: dropping it collapsed avgR +0.66 -> +0.28 with -14R/-16R
    # years in 2020/2022). Must never be relaxed or removed.
    for w in (126, 252):
        assert pf[w]["thresh"] == 65.0 and pf[w]["logic"] == "<"


def test_bear_execution_overlays():
    exe = _strat(BEAR)["execution"]
    # 25 bps nominal, GRM-scaled at import like the rest of the book
    assert exe["risk_bps"] == 25 * GLOBAL_RISK_MULTIPLIER
    assert exe["frag_risk_bands"] == [[50, 999, 0.25]]
    assert exe["same_day_signal_derate"] == 0.10
    assert exe["same_day_derate_floor"] == 0.30
    assert exe["use_stop_loss"] is False and exe["use_take_profit"] is False
    assert exe["hold_days"] == 2


def test_parent_keeps_strict_filters():
    pf = {f["window"]: f for f in _strat(PARENT)["settings"]["perf_filters"]}
    for w in (2, 5, 10):
        assert pf[w]["thresh"] == 85.0
    assert pf[21]["thresh"] == 85.0 and pf[21]["consecutive"] == 3


def test_derate_mult_formula():
    exe = _strat(BEAR)["execution"]
    assert same_day_derate_mult(exe, 1) == 1.0
    assert same_day_derate_mult(exe, 2) == 0.9
    assert same_day_derate_mult(exe, 3) == 0.8
    assert same_day_derate_mult(exe, 5) == 0.6
    # floor engages at n >= 8 (1 - 0.1*7 = 0.30)
    assert same_day_derate_mult(exe, 8) == 0.30
    assert same_day_derate_mult(exe, 20) == 0.30


def test_derate_noop_without_field():
    assert same_day_derate_mult({}, 5) == 1.0
    assert same_day_derate_mult(_strat(PARENT)["execution"], 5) == 1.0


def test_derate_default_floor():
    assert same_day_derate_mult({"same_day_signal_derate": 0.10}, 99) == 0.30


def test_bear_strat_is_the_only_derate_carrier():
    carriers = [s["name"] for s in STRATEGY_BOOK
                if s["execution"].get("same_day_signal_derate")]
    assert carriers == [BEAR]
