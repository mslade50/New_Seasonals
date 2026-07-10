"""3x Leader Gap Fade: config invariants (added 2026-07-10).

Guards the leader-expansion strategy: fade a 3x ETF whose 252d rank is at a
>95th-percentile EXTREME (the inverse of the other fades' <65 exclusion)
when it is short-horizon overbought AND still gapping up 0.25 ATR at the
T+1 open, filled at Open + 0.75 ATR, 2-day time exit, NO stop. Bull-equity
names are excluded structurally (every selectivity layer made them worse —
strictest cell 0-for-7). Evidence: scratch/lev3x_fade_leader_expansion.py,
_stops.py, _entries.py, _ovs_entry.py, _class_split.py, _bulleq_strict.py,
_validation.py (episode t=2.17, LOYO floor 1.55, drop-best +9.4R @ t=1.79).
"""
import json
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
    STRATEGY_BOOK, LEV3X_ALL, LEV3X_BEAR_EQ, LEV3X_BULL_EQ,
    GLOBAL_RISK_MULTIPLIER,
)

LEADER = "3x Leader Gap Fade"
BEAR = "3x Bear ETF Overbot Fade"
PARENT = "3x ETF Overbot Fade"


def _strat(name):
    for s in STRATEGY_BOOK:
        if s["name"] == name:
            return s
    raise AssertionError(f"strategy {name!r} not in STRATEGY_BOOK")


def test_universe_is_lev3x_minus_bull_eq():
    s = _strat(LEADER)
    u = set(s["universe_tickers"])
    assert u == set(LEV3X_ALL) - set(LEV3X_BULL_EQ)
    assert len(u) == 21
    # bull-eq exclusion is structural — never re-add without re-running
    # scratch/lev3x_fade_leader_bulleq_strict.py (strictest cell 0-for-7)
    assert not (u & set(LEV3X_BULL_EQ))
    # bear-eq + bonds are the core; both must be present
    assert set(LEV3X_BEAR_EQ) <= u
    assert {'TMF', 'TMV'} <= u


def test_bull_eq_constant_partitions_lev3x():
    # LEV3X_BULL_EQ + leader-fade universe must partition LEV3X_ALL exactly,
    # so a ticker added to LEV3X_ALL lands in exactly one side by decision,
    # not by accident.
    assert set(LEV3X_BULL_EQ) <= set(LEV3X_ALL)
    assert set(LEV3X_BULL_EQ) | set(_strat(LEADER)["universe_tickers"]) == set(LEV3X_ALL)
    assert not (set(LEV3X_BULL_EQ) & set(LEV3X_BEAR_EQ))


def test_252d_filters_mutually_exclusive_with_other_fades():
    """No same-day same-ticker cross-fire is possible by construction:
    the other fades require 252d rank < 65, this one requires > 95."""
    def _f252(name):
        return [f for f in _strat(name)["settings"]["perf_filters"]
                if f["window"] == 252]

    leader = _f252(LEADER)
    assert len(leader) == 1
    assert leader[0]["logic"] == ">" and leader[0]["thresh"] == 95.0

    for other in (BEAR, PARENT):
        f = _f252(other)
        assert len(f) == 1
        assert f[0]["logic"] == "<" and f[0]["thresh"] == 65.0
        # <65 and >95 cannot both hold for one ticker on one day
        assert f[0]["thresh"] <= leader[0]["thresh"]


def test_short_horizon_filters_match_bear_fade():
    pf = {f["window"]: f for f in _strat(LEADER)["settings"]["perf_filters"]}
    assert set(pf) == {2, 5, 10, 21, 252}, "no 126d filter — leader config drops it"
    for w in (2, 5, 10, 21):
        assert pf[w]["logic"] == ">" and pf[w]["thresh"] == 80.0
        assert pf[w]["consecutive"] == 1


def test_gap_gate_stamped_for_order_staging():
    """The gap gate resolves LIVE at the T+1 open (order_staging generic
    T1_Open_Filters gate, fail-closed), never at scan time. The spec must
    stay JSON-serializable and reference only stamped columns (Close)."""
    s = _strat(LEADER)["settings"]
    assert s["use_t1_open_filter"] is True
    filts = s["t1_open_filters"]
    assert filts == [{'reference': 'Close', 'atr_offset': 0.25, 'logic': '>'}]
    json.dumps(filts)  # must survive the scanner's JSON stamp


def test_entry_and_exit_conventions():
    s = _strat(LEADER)
    assert s["settings"]["entry_type"] == "Limit (Open +/- 0.75 ATR)", \
        "OVS entry convention — engine parses the 0.75 multiplier from this string"
    assert s["settings"]["trade_direction"] == "Short"
    assert s["settings"]["max_one_pos"] is True
    exe = s["execution"]
    assert exe["hold_days"] == 2
    # NO stop: tested 1.0/1.5/2.0 ATR day-1 and day-2 armed, all destroyed
    # the edge (scratch/lev3x_fade_leader_stops.py: non-bull +23.7R -> -39.8R
    # at 1.0 ATR). The demanding entry is the risk control.
    assert exe["use_stop_loss"] is False
    assert exe["use_take_profit"] is False


def test_sizing_and_no_overlays():
    exe = _strat(LEADER)["execution"]
    # 25 bps nominal, GRM-scaled at import like the rest of the book
    assert exe["risk_bps"] == 25 * GLOBAL_RISK_MULTIPLIER
    # Deliberately EXEMPT from the FAMILY4 fragility throttle and the
    # same-day de-rate (2026-07-10 decision): the edge lives on exactly the
    # high-fragility multi-signal days those overlays would cut, and the
    # generic per-strategy daily cap (engine cap_bps=250 default +
    # order_staging PER_STRAT_DAILY_CAP_DOLLARS override at 250 bps) bounds
    # the many-signal days instead.
    assert "frag_risk_bands" not in exe
    assert "same_day_signal_derate" not in exe
    assert "ladder_multipliers" not in exe
    assert "cycle_risk_mults" not in exe
