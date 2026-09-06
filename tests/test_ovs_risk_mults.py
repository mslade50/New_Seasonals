"""D3.5 OVS rank-mean and liquid-tier sizing parity guards."""

import inspect
import math
import os
import sys

import pandas as pd
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "pages"))


class _NoOp:
    def __getattr__(self, name):
        def f(*a, **k):
            return self
        return f

    def __call__(self, *a, **k):
        return self

    def __bool__(self):
        return False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cache_data(self, *a, **k):
        def deco(fn):
            return fn
        return deco

    cache_resource = cache_data


sys.modules["streamlit"] = _NoOp()

import daily_scan
import strategy_config as sc
from pages import strat_backtester as sb


OVS = "Overbot Vol Spike"
RANK_CONFIG = {
    "windows": [2, 5, 10, 21],
    "threshold": 94.0,
    "below_mult": 0.7,
    "cycle_exempt": [2],
}
TIER_CONFIG = {"Liquid": 0.7}


def _production_ovs():
    return next(strategy for strategy in sc.STRATEGY_BOOK
                if strategy["name"] == OVS)


def _execution(path2_cap_pct=0.75):
    return {
        "risk_bps": 40.0,
        "slippage_bps": 0,
        "stop_atr": 1.0,
        "tgt_atr": 2.0,
        "hold_days": 2,
        "use_stop_loss": False,
        "use_take_profit": False,
        "path1_bps": 40.0,
        "path2_bps": 8.0,
        "path2_daily_cap_pct": path2_cap_pct,
        "cycle_risk_mults": {2: 0.75},
        "rank_mean_risk": dict(RANK_CONFIG),
        "tier_risk_mults": dict(TIER_CONFIG),
    }


def _strategy(tickers, path2_cap_pct=0.75):
    return {
        "name": OVS,
        "settings": {
            "trade_direction": "Short",
            "entry_type": "Limit (Open +/- 0.75 ATR)",
            "max_one_pos": False,
        },
        "execution": _execution(path2_cap_pct),
        "universe_tickers": list(tickers),
    }


def _frame(year, gap_open):
    dates = pd.date_range(f"{year}-01-04", periods=5, freq="B")
    frame = pd.DataFrame(
        {
            "Open": [100.0, gap_open, 101.0, 100.5, 100.0],
            "High": [101.0, 104.0, 103.0, 102.0, 101.0],
            "Low": [99.0, 100.0, 99.0, 99.0, 99.0],
            "Close": [100.0, 102.0, 101.0, 100.5, 100.0],
            "Volume": [1_000_000.0] * 5,
        },
        index=dates,
    )
    frame["ATR"] = 2.0
    frame["RangePct"] = 0.02
    frame["vol_ratio"] = 1.0
    frame["Sznl"] = 50.0
    for window in (2, 5, 10, 21, 126, 252):
        frame[f"rank_ret_{window}d"] = 96.0
    return frame


def _signal_row(ranks):
    row = {
        "atr": 2.0,
        "close": 100.0,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "vol_ratio": 1.0,
        "sznl": 50.0,
        "range_pct": 2.0,
        "atr_sznl_5d": 50.0,
        "rank_ret_126d": 50.0,
        "rank_ret_252d": 50.0,
    }
    row.update({f"rank_ret_{window}d": float(value)
                for window, value in ranks.items()})
    return row


def _run_one(ticker, year, ranks, path):
    gap_open = 101.0 if path == "P1" else 100.25
    frame = _frame(year, gap_open)
    candidates = [(int(frame.index[0].value), ticker, ticker, 0, 0)]
    result = sb.process_signals_fast(
        candidates,
        {(ticker, 0): _signal_row(ranks)},
        {ticker: frame},
        [_strategy([ticker])],
        starting_equity=100_000,
        flat_sizing=True,
        cap_bps=0,
    )
    assert len(result) == 1
    return result.iloc[0]


def test_exact_carriers_configs_and_no_grm_scaling():
    rank_carriers = {
        strategy["name"]: strategy["execution"]["rank_mean_risk"]
        for strategy in sc.STRATEGY_BOOK
        if strategy["execution"].get("rank_mean_risk")
    }
    tier_carriers = {
        strategy["name"]: strategy["execution"]["tier_risk_mults"]
        for strategy in sc.STRATEGY_BOOK
        if strategy["execution"].get("tier_risk_mults")
    }
    assert rank_carriers == {OVS: RANK_CONFIG}
    assert tier_carriers == {OVS: TIER_CONFIG}
    assert _production_ovs()["execution"]["rank_mean_risk"] == RANK_CONFIG
    assert _production_ovs()["execution"]["tier_risk_mults"] == TIER_CONFIG


def test_production_before_cap_effective_bps_contract():
    execution = _production_ovs()["execution"]
    p1 = execution["path1_bps"]
    p2 = execution["path2_bps"]
    tier = sc.tier_risk_mult(execution, "Liquid")
    bottom, _, _ = sc.rank_mean_risk_decision(
        execution, {2: 93.0, 5: 93.0, 10: 93.0, 21: 93.0}, 2027
    )
    midterm_bottom, _, exempt = sc.rank_mean_risk_decision(
        execution, {2: 93.0, 5: 93.0, 10: 93.0, 21: 93.0}, 2026
    )
    cycle = execution["cycle_risk_mults"][2]

    assert (p1 * tier, p2 * tier) == pytest.approx((42.0, 8.4))
    assert (p1 * tier * bottom, p2 * tier * bottom) == pytest.approx(
        (29.4, 5.88)
    )
    assert exempt and midterm_bottom == 1.0
    assert (p1 * tier * cycle, p2 * tier * cycle) == pytest.approx(
        (31.5, 6.3)
    )
    assert (
        p1 * sc.tier_risk_mult(execution, "Overflow"),
        p2 * sc.tier_risk_mult(execution, "Overflow"),
    ) == pytest.approx((60.0, 12.0))


def test_rank_mean_strict_boundary_midterm_and_missing_fallback():
    execution = _execution()
    bottom = {2: 93.999, 5: 93.999, 10: 93.999, 21: 93.999}
    edge = {2: 94.0, 5: 94.0, 10: 94.0, 21: 94.0}
    assert sc.rank_mean_risk_decision(execution, bottom, 2027) == (
        0.7, 93.999, False)
    assert sc.rank_mean_risk_decision(execution, edge, 2027) == (
        1.0, 94.0, False)
    assert sc.rank_mean_risk_decision(execution, bottom, 2026) == (
        1.0, 93.999, True)
    assert sc.rank_mean_risk_decision(execution, {2: 93.0}, 2027) == (
        1.0, None, False)
    nonfinite = dict(edge)
    nonfinite[21] = math.nan
    assert sc.rank_mean_risk_decision(execution, nonfinite, 2027) == (
        1.0, None, False)
    assert sc.rank_mean_risk_decision({}, edge, 2027) == (1.0, None, False)


def test_tier_helper_liquid_point_seven_overflow_and_missing_full():
    execution = _execution()
    assert sc.tier_risk_mult(execution, "Liquid") == 0.7
    assert sc.tier_risk_mult(execution, "Overflow") == 1.0
    assert sc.tier_risk_mult({}, "Liquid") == 1.0
    with pytest.raises(ValueError):
        sc.tier_risk_mult({"tier_risk_mults": {"Liquid": 0}}, "Liquid")


@pytest.mark.parametrize("path,path_mult", [("P1", 1.0), ("P2", 0.2)])
def test_engine_liquid_bottom_composes_on_both_paths(path, path_mult):
    row = _run_one("SPY", 2027, {2: 93, 5: 93, 10: 93, 21: 93}, path)
    assert row["Risk $"] == pytest.approx(400.0 * 0.7 * 0.7 * path_mult)


@pytest.mark.parametrize("path,path_mult", [("P1", 1.0), ("P2", 0.2)])
def test_engine_overflow_bottom_has_no_liquid_cut_on_both_paths(path, path_mult):
    row = _run_one("TEST", 2027, {2: 93, 5: 93, 10: 93, 21: 93}, path)
    assert row["Risk $"] == pytest.approx(400.0 * 0.7 * path_mult)


def test_engine_midterm_exempts_extremity_but_keeps_cycle_and_tier():
    row = _run_one("SPY", 2026, {2: 93, 5: 93, 10: 93, 21: 93}, "P1")
    assert row["Risk $"] == pytest.approx(400.0 * 0.75 * 0.7)


def test_p2_cap_prepass_uses_tier_and_extremity_once_on_mixed_day():
    tickers = ["SPY", "TEST"]
    frames = {ticker: _frame(2027, 100.25) for ticker in tickers}
    ranks = {
        "SPY": {2: 93, 5: 93, 10: 93, 21: 93},
        "TEST": {2: 96, 5: 96, 10: 96, 21: 96},
    }
    candidates = [
        (int(frames[ticker].index[0].value), ticker, ticker, 0, 0)
        for ticker in tickers
    ]
    result = sb.process_signals_fast(
        candidates,
        {(ticker, 0): _signal_row(ranks[ticker]) for ticker in tickers},
        frames,
        [_strategy(tickers, path2_cap_pct=0.10)],
        starting_equity=100_000,
        flat_sizing=True,
        cap_bps=0,
    )
    risk = result.groupby("Ticker")["Risk $"].sum()
    # Pre-cap staged P2 risk: SPY 39.2, TEST 80. Fixed cap = $100, so both
    # receive the same 100/119.2 pro-rata factor.
    assert risk.sum() == pytest.approx(100.0)
    assert risk["SPY"] == pytest.approx(39.2 * 100.0 / 119.2)
    assert risk["TEST"] == pytest.approx(80.0 * 100.0 / 119.2)


def test_point_in_time_snapshot_and_scan_engine_order_contracts():
    generate = inspect.getsource(sb.generate_candidates_fast)
    engine = inspect.getsource(sb.process_signals_fast)
    scan = inspect.getsource(daily_scan.run_daily_scan)
    for window in (2, 5, 10, 21):
        assert f"'rank_ret_{window}d'" in generate
    scan_tier = scan.index("_trm = tier_risk_mult(")
    scan_rank = scan.index("rank_mean_risk_decision(", scan_tier)
    assert scan_tier < scan_rank < scan.index("shares = int(risk / dist)")
    assert "if _tier_rule:" in scan[scan_tier:scan_rank]
    assert "Tier {_scan_source}: {_trm:.2f}x" in scan[scan_tier:scan_rank]
    engine_tier = engine.index("_tier_m = tier_risk_mult(execution")
    engine_rank = engine.index("rank_mean_risk_decision(", engine_tier)
    assert engine_tier < engine_rank < engine.index("shares = int(base_risk / dist)")
    p2 = engine.index("# OVS path-2 contribution")
    assert engine.index("_tier_m = tier_risk_mult(_exe", p2) < engine.index(
        "_p2_risk = _base_risk_p1 * _p2_mult", p2)
    assert engine.index("rank_mean_risk_decision(", p2) < engine.index(
        "_p2_risk = _base_risk_p1 * _p2_mult", p2)


def test_noncarrier_and_existing_ovs_contracts_unchanged():
    production = _production_ovs()
    ovs = production["execution"]
    assert ovs["path1_bps"] == 40 * sc.GLOBAL_RISK_MULTIPLIER
    assert ovs["path2_bps"] == 8 * sc.GLOBAL_RISK_MULTIPLIER
    assert ovs["path2_daily_cap_pct"] == 0.75 * sc.GLOBAL_RISK_MULTIPLIER
    assert ovs["cycle_risk_mults"] == {2: 0.75}
    assert not [
        strategy["name"] for strategy in sc.STRATEGY_BOOK
        if strategy["name"] != OVS
        and (strategy["execution"].get("rank_mean_risk")
             or strategy["execution"].get("tier_risk_mults"))
    ]
    assert "0.7x liquid-tier" in production["description"]
    assert "0.7x non-midterm bottom-extremity" in production["description"]
    assert "Same scheme for liquid and overflow universes" not in str(production)
    module_source = inspect.getsource(sb)
    assert "Liquid and overflow tickers receive identical OVS sizing" not in module_source
    assert "liquid rows receive the 0.7x tier multiplier" in module_source
