"""D3.3 clone clamps: config, overlap resolution and engine/live ordering.

The cross-strategy rule is an absolute staged-risk clamp. The IOB rule is a
second, multiplicative 0.5x cut only when both members of its exact two-index
universe fire. Both operate on staged signals, before fill outcomes are known.
"""
import copy
import inspect
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "pages"))


class _NoOp:
    def __getattr__(self, name):
        def f(*a, **k):
            return self
        return f
    def __call__(self, *a, **k): return self
    def __bool__(self): return False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **k):
        def deco(fn): return fn
        return deco
    cache_resource = cache_data


sys.modules["streamlit"] = _NoOp()

import daily_scan
import strategy_config as sc
from pages import strat_backtester as sb


IOB = "Indices Oversold Bounce"
MONFRI = "SPY QQQ MonFri Reversion"
BEAR = "3x Bear ETF Overbot Fade"

EXPECTED_PAIRS = {
    frozenset((IOB, MONFRI)),
    frozenset(("Monday Dip", "Weak Close Decent Sznls")),
    frozenset((MONFRI, "Weak Close Decent Sznls")),
    frozenset(("Monthly Weak Close", MONFRI)),
    frozenset(("Monthly Weak Close", IOB)),
    frozenset(("Monday Dip", IOB)),
}


def _production_strategy(name):
    return next(s for s in sc.STRATEGY_BOOK if s["name"] == name)


def _frame():
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    frame = pd.DataFrame({
        "Open": [100.0] * 5,
        "High": [101.0] * 5,
        "Low": [99.0] * 5,
        "Close": [100.0] * 5,
        "Volume": [1_000_000.0] * 5,
    }, index=dates)
    frame["ATR"] = 2.0
    frame["RangePct"] = 0.02
    frame["vol_ratio"] = 1.0
    frame["Sznl"] = 50.0
    return frame


def _signal_data(tickers):
    row = {
        "atr": 2.0, "close": 100.0, "open": 100.0,
        "high": 101.0, "low": 99.0, "vol_ratio": 1.0,
        "sznl": 50.0, "range_pct": 2.0,
        "atr_sznl_5d": 50.0, "rank_ret_126d": 50.0,
        "rank_ret_252d": 50.0,
    }
    return {(ticker, 0): dict(row) for ticker in tickers}


def _strategy(name, tickers, risk_bps=60.0, same_day=None):
    execution = {
        "risk_bps": risk_bps,
        "slippage_bps": 0,
        "stop_atr": 1.0,
        "tgt_atr": 0.0,
        "hold_days": 2,
        "use_stop_loss": False,
        "use_take_profit": False,
    }
    if same_day is not None:
        execution.update({
            "same_day_signal_derate": same_day,
            "same_day_derate_floor": same_day,
        })
    return {
        "name": name,
        "settings": {
            "trade_direction": "Long",
            "entry_type": "T+1 Open",
            "max_one_pos": False,
        },
        "execution": execution,
        "universe_tickers": list(tickers),
    }


def _run_engine(monkeypatch, strategies, candidate_specs, overrides):
    monkeypatch.setattr(sc, "CROSS_STRATEGY_OVERLAP_OVERRIDES", overrides)
    tickers = sorted({ticker for ticker, _ in candidate_specs})
    frames = {ticker: _frame() for ticker in tickers}
    signal_date = next(iter(frames.values())).index[0]
    candidates = [
        (int(signal_date.value), ticker, ticker, strat_idx, 0)
        for ticker, strat_idx in candidate_specs
    ]
    return sb.process_signals_fast(
        candidates,
        _signal_data(tickers),
        frames,
        strategies,
        starting_equity=100_000,
        flat_sizing=True,
        cap_bps=0,
    )


def test_pair_table_is_exact_and_grm_scaled_once():
    got = {
        frozenset(override["strategies"])
        for override in sc.CROSS_STRATEGY_OVERLAP_OVERRIDES
    }
    assert got == EXPECTED_PAIRS
    assert len(sc.CROSS_STRATEGY_OVERLAP_OVERRIDES) == 6
    assert all(
        override["risk_bps_when_overlapping"] == 20 * sc.GLOBAL_RISK_MULTIPLIER
        for override in sc.CROSS_STRATEGY_OVERLAP_OVERRIDES
    )


def test_iob_clone_config_is_exact_and_monfri_is_untouched():
    iob = _production_strategy(IOB)
    monfri = _production_strategy(MONFRI)
    assert iob["universe_tickers"] == ["^GSPC", "^NDX"]
    assert iob["execution"]["same_day_signal_derate"] == 0.5
    assert iob["execution"]["same_day_derate_floor"] == 0.5
    assert "same_day_signal_derate" not in monfri["execution"]
    carriers = {
        s["name"] for s in sc.STRATEGY_BOOK
        if s["execution"].get("same_day_signal_derate")
    }
    assert carriers == {BEAR, IOB}
    assert sc.same_day_derate_mult(iob["execution"], 1) == 1.0
    assert sc.same_day_derate_mult(iob["execution"], 2) == 0.5


def test_overlap_resolver_is_strategy_specific_min_and_order_invariant():
    day = pd.Timestamp("2024-01-02")
    fired = {(day, "SPY"): {"A", "B", "C"}}
    overrides = [
        {"strategies": ("A", "B"), "risk_bps_when_overlapping": 30},
        {"strategies": ("B", "C"), "risk_bps_when_overlapping": 20},
    ]
    expected = {
        (day, "SPY", "A"): 30.0,
        (day, "SPY", "B"): 20.0,
        (day, "SPY", "C"): 20.0,
    }
    assert sc.resolve_cross_strategy_overlap_clamps(fired, overrides) == expected
    assert sc.resolve_cross_strategy_overlap_clamps(
        fired, list(reversed(overrides))) == expected


def test_engine_triple_collision_clamps_every_participant(monkeypatch):
    strategies = [
        _strategy("A", ["TEST"]),
        _strategy("B", ["TEST"]),
        _strategy("C", ["TEST"]),
    ]
    overrides = [
        {"strategies": ("A", "B"), "risk_bps_when_overlapping": 30},
        {"strategies": ("B", "C"), "risk_bps_when_overlapping": 20},
    ]
    for ordered in (overrides, list(reversed(overrides))):
        result = _run_engine(
            monkeypatch, copy.deepcopy(strategies),
            [("TEST", 0), ("TEST", 1), ("TEST", 2)], ordered)
        risk = result.groupby("Strategy")["Risk $"].sum().to_dict()
        assert risk == {"A": 300.0, "B": 200.0, "C": 200.0}


def test_engine_alias_overlap_then_iob_clone(monkeypatch):
    strategies = [
        _strategy(IOB, ["^GSPC", "^NDX"], same_day=0.5),
        _strategy(MONFRI, ["SPY"]),
    ]
    overrides = [{
        "strategies": (IOB, MONFRI),
        "risk_bps_when_overlapping": 30,
    }]
    result = _run_engine(
        monkeypatch, strategies,
        [("^GSPC", 0), ("^NDX", 0), ("SPY", 1)], overrides)
    risk = {
        (row["Strategy"], row["Ticker"]): row["Risk $"]
        for _, row in result.iterrows()
    }
    # ^GSPC aliases to SPY: absolute 30-bps clamp, then IOB 0.5x = 15 bps.
    assert risk[(IOB, "^GSPC")] == 150.0
    # The unmatched QQQ clone gets only IOB's 0.5x: 60 -> 30 bps.
    assert risk[(IOB, "^NDX")] == 300.0
    # MonFri receives the absolute clamp but never IOB's clone multiplier.
    assert risk[(MONFRI, "SPY")] == 300.0


def test_absolute_clamp_never_raises_a_row(monkeypatch):
    strategies = [
        _strategy("A", ["TEST"], risk_bps=20),
        _strategy("B", ["TEST"], risk_bps=20),
    ]
    overrides = [{
        "strategies": ("A", "B"),
        "risk_bps_when_overlapping": 30,
    }]
    result = _run_engine(
        monkeypatch, strategies, [("TEST", 0), ("TEST", 1)], overrides)
    assert result.groupby("Strategy")["Risk $"].sum().to_dict() == {
        "A": 200.0, "B": 200.0,
    }


def test_scan_and_engine_use_shared_resolver_before_same_day_multiplier():
    scan_src = inspect.getsource(daily_scan.run_daily_scan)
    engine_src = inspect.getsource(sb.process_signals_fast)
    assert scan_src.index("resolve_cross_strategy_overlap_clamps(") < scan_src.index("_derate_execs =")
    assert engine_src.index("resolve_cross_strategy_overlap_clamps(") < engine_src.index("same_day_derate_mult(execution")
    assert "(_s.get('Strategy_Name'), _s.get('Scan_Source'))" in scan_src
    assert "SPOT_TO_TRADEABLE.get(_tkr, _tkr)" in scan_src
    assert "SPOT_TO_TRADEABLE.get(_c[2], _c[2])" in engine_src
