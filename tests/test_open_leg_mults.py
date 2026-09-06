"""D3.4 WCDS/LT Trend solo-add sizing and scan/engine parity guards."""

import inspect
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


WCDS = "Weak Close Decent Sznls"
LT = "LT Trend ST OS"
EXPECTED = {"none_open": 0.8, "adds": 1.2}


def _production(name):
    return next(s for s in sc.STRATEGY_BOOK if s["name"] == name)


def _frame(lows=None, periods=9):
    dates = pd.date_range("2024-01-02", periods=periods, freq="B")
    frame = pd.DataFrame(
        {
            "Open": [100.0] * periods,
            "High": [101.0] * periods,
            "Low": list(lows) if lows is not None else [99.0] * periods,
            "Close": [100.0] * periods,
            "Volume": [1_000_000.0] * periods,
        },
        index=dates,
    )
    frame["ATR"] = 2.0
    frame["RangePct"] = 0.02
    frame["vol_ratio"] = 1.0
    frame["Sznl"] = 50.0
    return frame


def _strategy(name, entry_type="T+1 Open", hold_days=4):
    return {
        "name": name,
        "settings": {
            "trade_direction": "Long",
            "entry_type": entry_type,
            "max_one_pos": False,
        },
        "execution": {
            "risk_bps": 60.0,
            "slippage_bps": 0,
            "stop_atr": 1.0,
            "tgt_atr": 0.0,
            "hold_days": hold_days,
            "fill_window_days": 4,
            "use_stop_loss": False,
            "use_take_profit": False,
            "open_leg_mults": dict(EXPECTED),
        },
        "universe_tickers": [],
    }


def _signal_row():
    return {
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


def _run(strategies, specs, frames=None):
    """specs are (ticker, strategy_index, signal_index)."""
    tickers = sorted({ticker for ticker, _, _ in specs})
    frames = frames or {ticker: _frame() for ticker in tickers}
    candidates = [
        (int(frames[ticker].index[idx].value), ticker, ticker, strat_idx, idx)
        for ticker, strat_idx, idx in specs
    ]
    signal_data = {(ticker, idx): _signal_row() for ticker, _, idx in specs}
    return sb.process_signals_fast(
        candidates,
        signal_data,
        frames,
        strategies,
        starting_equity=100_000,
        flat_sizing=True,
        cap_bps=0,
    )


def test_carrier_set_and_configs_are_exact():
    carriers = {
        s["name"]: s["execution"]["open_leg_mults"]
        for s in sc.STRATEGY_BOOK
        if s["execution"].get("open_leg_mults")
    }
    assert carriers == {WCDS: EXPECTED, LT: EXPECTED}


def test_open_leg_helper_boundaries_and_validation():
    execution = {"open_leg_mults": dict(EXPECTED)}
    assert sc.open_leg_mult({}, 99, 99) == 1.0
    assert sc.open_leg_mult(execution, 1, 0) == 0.8
    assert sc.open_leg_mult(execution, 2, 0) == 1.2
    assert sc.open_leg_mult(execution, 1, 1) == 1.2
    assert sc.open_leg_mult(execution, 2, 5) == 1.2
    with pytest.raises(ValueError):
        sc.open_leg_mult(execution, -1, 0)
    with pytest.raises(ValueError):
        sc.open_leg_mult({"open_leg_mults": {"none_open": 0, "adds": 1.2}}, 1, 0)


def test_engine_true_solo_is_point_eight():
    result = _run([_strategy(WCDS)], [("AAA", 0, 0)])
    assert result[["Strategy", "Risk $"]].to_dict("records") == [
        {"Strategy": WCDS, "Risk $": 480.0}
    ]


def test_engine_two_staged_candidates_are_both_one_point_two():
    result = _run(
        [_strategy(WCDS)],
        [("AAA", 0, 0), ("BBB", 0, 0)],
    )
    assert result.set_index("Ticker")["Risk $"].to_dict() == {
        "AAA": 720.0,
        "BBB": 720.0,
    }


def test_engine_prior_open_leg_on_different_ticker_is_one_point_two():
    result = _run(
        [_strategy(LT, hold_days=5)],
        [("AAA", 0, 0), ("BBB", 0, 2)],
    )
    risk = result.set_index("Ticker")["Risk $"].to_dict()
    assert risk["AAA"] == 480.0
    assert risk["BBB"] == 720.0


def test_engine_working_future_fill_does_not_count_as_open():
    # AAA is staged on day 0 but cannot fill until day 3. BBB is the only
    # staged signal on day 1, so AAA is merely working—not a prior open leg.
    frames = {
        "AAA": _frame([100, 100, 100, 99, 99, 99, 99, 99, 99]),
        "BBB": _frame([100, 100, 99, 99, 99, 99, 99, 99, 99]),
    }
    strategy = _strategy(LT, "Limit Order -0.25 ATR (Persistent)", hold_days=5)
    result = _run([strategy], [("AAA", 0, 0), ("BBB", 0, 1)], frames)
    assert result.set_index("Ticker")["Risk $"].to_dict() == {
        "AAA": 480.0,
        "BBB": 480.0,
    }


def test_engine_failed_fill_does_not_demote_other_cluster_row():
    frames = {
        "AAA": _frame([100] * 9),  # never reaches close - 0.25 ATR
        "BBB": _frame([100, 99, 99, 99, 99, 99, 99, 99, 99]),
    }
    strategy = _strategy(WCDS, "Limit Order -0.25 ATR (Persistent)")
    result = _run([strategy], [("AAA", 0, 0), ("BBB", 0, 0)], frames)
    assert result[["Ticker", "Risk $"]].to_dict("records") == [
        {"Ticker": "BBB", "Risk $": 720.0}
    ]


def test_duplicate_strategy_passes_keep_cluster_count_tier_local():
    strategies = [_strategy(LT), _strategy(LT)]
    result = _run(strategies, [("AAA", 0, 0), ("BBB", 1, 0)])
    assert result.set_index("Ticker")["Risk $"].to_dict() == {
        "AAA": 480.0,
        "BBB": 480.0,
    }


def test_scan_and_engine_order_and_counting_contracts():
    scan = inspect.getsource(daily_scan.run_daily_scan)
    scan_postpass = inspect.getsource(daily_scan.apply_open_leg_sizing)
    engine = inspect.getsource(sb.process_signals_fast)
    assert scan.index("apply_open_leg_sizing(") < scan.index("resolve_cross_strategy_overlap_clamps(")
    assert engine.index("open_leg_mult(execution") < engine.index("base_risk = min(")
    assert "(_s.get('Strategy_Name'), _s.get('Scan_Source'))" in scan
    assert "all_signals, _open_leg_execs, _open_legs_by_strat" in scan
    assert "prior_open_by_strategy.get(name, 0)" in scan_postpass
    assert "p['strat_name'] == strat_name" in engine
    assert "p['entry_date'] <= signal_date < p['exit_date']" in engine
    assert "int(target_risk / stop_distance)" in scan_postpass
    assert "shares > hard_cap" in scan_postpass


def _scan_row(stop_distance=2.7, hard_cap=None):
    return {
        "Strategy_Name": WCDS,
        "Scan_Source": "Liquid",
        "Shares": int(600.0 / stop_distance),
        "Risk_Amt": 600.0,
        "Notional": 0.0,
        "Entry": 100.0,
        "Sizing_Notes": "Standard (1.0x)",
        "_Open_Leg_Base_Risk": 600.0,
        "_Open_Leg_Stop_Distance": stop_distance,
        "_Open_Leg_Hard_Share_Cap": hard_cap,
    }


def test_scanner_multiplies_risk_then_floors_shares_like_engine():
    rows = [_scan_row()]
    daily_scan.apply_open_leg_sizing(
        rows, {WCDS: _strategy(WCDS)["execution"]}, {})
    assert rows[0]["Risk_Amt"] == 480.0
    assert rows[0]["Shares"] == int(480.0 / 2.7) == 177
    assert not any(key.startswith("_Open_Leg_") for key in rows[0])


def test_scanner_cluster_never_reexpands_an_earlier_adv_hard_cap():
    rows = [_scan_row(hard_cap=100), _scan_row(hard_cap=100)]
    daily_scan.apply_open_leg_sizing(
        rows, {WCDS: _strategy(WCDS)["execution"]}, {})
    assert [row["Shares"] for row in rows] == [100, 100]
    assert [row["Risk_Amt"] for row in rows] == [270.0, 270.0]
    assert all("hard share cap retained: 100" in row["Sizing_Notes"]
               for row in rows)


@pytest.mark.parametrize("ceiling_kind", ["ADV", "concurrent-notional"])
def test_scanner_cluster_respects_a_pre_multiplier_slack_ceiling(ceiling_kind):
    # $600 / $6 is 100 base shares. The 110 ceiling is initially slack but
    # must bind the later 1.2x target (120 shares).
    rows = [
        _scan_row(stop_distance=6.0, hard_cap=110),
        _scan_row(stop_distance=6.0, hard_cap=110),
    ]
    daily_scan.apply_open_leg_sizing(
        rows, {WCDS: _strategy(WCDS)["execution"]}, {})
    assert [row["Shares"] for row in rows] == [110, 110], ceiling_kind
    assert [row["Risk_Amt"] for row in rows] == [660.0, 660.0], ceiling_kind


def test_main_loop_records_slack_adv_and_concurrent_notional_ceilings():
    scan = inspect.getsource(daily_scan.run_daily_scan)
    adv = scan.index("_cap = adv_share_cap(")
    assert scan.index("_open_leg_hard_share_cap = (", adv) < scan.index(
        "if _cap < shares:", adv)
    tnc = scan.index("_tnc_share_cap = int(")
    assert scan.index("_open_leg_hard_share_cap = (", tnc) < scan.index(
        "if _tnc_open + _tnc_new > _tnc_cap:", tnc)


def test_dormant_ticker_ladder_remains_without_carriers():
    assert not [
        s["name"] for s in sc.STRATEGY_BOOK
        if s["execution"].get("ladder_multipliers")
    ]
