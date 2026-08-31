from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import research.primary_breakout.engine as breakout_engine
from research.primary_breakout.engine import (
    BacktestConfig,
    build_stock_universe,
    compute_features,
    monthly_block_bootstrap_sharpe,
    run_backtest,
)
from research.primary_breakout.report import render_report


def _frame(close, high=None, low=None, open_=None, dates=None):
    close = np.asarray(close, dtype=float)
    dates = dates if dates is not None else pd.bdate_range("2024-01-02", periods=len(close))
    high = np.asarray(high if high is not None else close + 1.0, dtype=float)
    low = np.asarray(low if low is not None else close - 1.0, dtype=float)
    open_ = np.asarray(open_ if open_ is not None else close, dtype=float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1_000_000.0},
        index=pd.DatetimeIndex(dates),
    )


def _config(**overrides):
    base = BacktestConfig(
        start_date="2024-01-08",
        end_date="2024-01-10",
        initial_equity=100.0,
        breakout_window=2,
        roc_window=2,
        ck_atr_period=2,
        ck_atr_multiple=3.0,
        ck_stop_period=2,
        risk_fraction=0.50,
        max_name_fraction=1.0,
        max_gross_fraction=1.0,
        max_open_risk_fraction=1.0,
        cost_bps=0.0,
    )
    return replace(base, **overrides)


def _rank_panel(gap_stop=False):
    dates = pd.bdate_range("2024-01-02", periods=7)
    a_close = [9, 9, 9, 11, 11.5, 12, 12]
    b_close = [19, 19, 19, 24, 24.5, 25, 25]
    a = _frame(
        a_close,
        high=[10, 10, 10, 11.2, 12, 12.5, 12.5],
        low=[8, 8, 8, 10.8, 11.1, 11.7, 11.8],
        open_=[9, 9, 9, 11, 11.4, 12, 12],
        dates=dates,
    )
    b_open = [19, 19, 19, 24, 24.2, 10 if gap_stop else 25, 25]
    b_low = [18, 18, 18, 23.8, 24.0, 9 if gap_stop else 24.7, 24.8]
    b = _frame(
        b_close,
        high=[20, 20, 20, 24.2, 25, 25.5, 25.5],
        low=b_low,
        open_=b_open,
        dates=dates,
    )
    spy = _frame(
        [100, 100, 100, 101, 101, 102, 102],
        high=[101, 101, 101, 102, 102, 103, 103],
        low=[99, 99, 99, 100, 100, 101, 101],
        dates=dates,
    )
    return {"A": a, "B": b, "SPY": spy}, dates


def test_breakout_excludes_today_and_fresh_fires_once():
    bars = _frame(
        [9, 9, 9, 9, 12.5, 12, 12],
        high=[10, 10, 10, 10, 13, 13, 13],
        low=[8, 8, 8, 8, 11.5, 11.5, 11.5],
    )
    features = compute_features(bars, _config(breakout_window=3, roc_window=3))
    assert features["PriorBreakoutHigh"].iloc[3] == 10
    assert not bool(features["Breakout"].iloc[3])
    assert bool(features["FreshBreakout"].iloc[4])
    assert bool(features["Breakout"].iloc[4])
    assert np.isclose(features["ROC"].iloc[4], 12.5 / 9 - 1)

    mutated = bars.copy()
    mutated.iloc[-1, mutated.columns.get_loc("High")] = 10_000
    mutated.iloc[-1, mutated.columns.get_loc("Close")] = 10_000
    mutated_features = compute_features(mutated, _config(breakout_window=3, roc_window=3))
    pd.testing.assert_frame_equal(features.iloc[:-1], mutated_features.iloc[:-1])


def test_missing_prior_breakout_state_cannot_create_fresh_signal():
    bars = _frame(
        [9, 9, 11, 12, 12],
        high=[10, 10, 12, 13, 13],
        low=[8, 8, 10, 11, 11],
    )
    features = compute_features(bars, _config(breakout_window=2, roc_window=2))
    assert bool(features["Breakout"].iloc[2])
    assert not bool(features["FreshBreakout"].iloc[2])


def test_signal_while_held_is_discarded_even_if_position_gaps_out_next_open():
    dates = pd.bdate_range("2024-01-02", periods=8)
    stock = _frame(
        [9, 9, 9, 11, 9, 13.5, 3, 3],
        high=[10, 10, 10, 12, 11, 14, 4, 4],
        low=[8, 8, 8, 10, 8.5, 11, 2, 2],
        open_=[9, 9, 9, 11, 10, 13, 3, 3],
        dates=dates,
    )
    spy = _frame(
        [100, 100, 101, 101, 102, 102, 102, 102],
        dates=dates,
    )
    config = _config(start_date="2024-01-05", end_date="2024-01-11")
    result = run_backtest({"A": stock, "SPY": spy}, dates, ["A"], config)
    rejected = result.candidates[
        (result.candidates["Ticker"] == "A")
        & (result.candidates["Reason"] == "already_held")
    ]
    assert not rejected.empty
    assert len(result.candidates[result.candidates["Status"] == "Entered"]) == 1


def test_wilder_chande_kroll_formula_is_hand_checkable():
    bars = _frame(
        [10, 11, 12, 13, 14, 15],
        high=[11, 12, 13, 14, 15, 16],
        low=[9, 10, 11, 12, 13, 14],
    )
    features = compute_features(
        bars,
        _config(ck_atr_period=2, ck_atr_multiple=1.0, ck_stop_period=2),
    )
    assert np.isclose(features["ATR"].iloc[2], 2.0)
    assert np.isclose(features["CKPreliminary"].iloc[2], 11.0)
    assert np.isclose(features["CKStop"].iloc[3], 12.0)


def test_same_day_candidates_are_admitted_by_roc_not_ticker():
    panel, dates = _rank_panel()
    result = run_backtest(
        panel,
        dates,
        ["A", "B"],
        _config(max_gross_fraction=0.30),
    )
    entered = result.candidates[result.candidates["Status"] == "Entered"]
    assert list(entered["Ticker"]) == ["B"]
    rejected_a = result.candidates[result.candidates["Ticker"] == "A"].iloc[0]
    assert rejected_a["Status"] == "Rejected"
    assert "gross_cap" in rejected_a["Reason"]


def test_gap_through_stop_exits_at_open_less_cost():
    panel, dates = _rank_panel(gap_stop=True)
    config = _config(max_gross_fraction=0.30, cost_bps=10.0)
    result = run_backtest(panel, dates, ["A", "B"], config)
    trade = result.trades[result.trades["Ticker"] == "B"].iloc[0]
    assert trade["ExitReason"] == "StopGap"
    assert trade["RawExitPrice"] == 10.0
    assert np.isclose(trade["ExitPrice"], 9.99)


def test_close_computed_stop_activates_next_session_and_never_loosens(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=6)
    stock = _frame(
        [95, 96, 110, 108, 107, 107],
        high=[97, 98, 112, 111, 109, 109],
        low=[93, 94, 95, 106, 104, 105],
        open_=[95, 96, 100, 110, 108, 107],
        dates=dates,
    )
    spy = _frame([100, 100, 101, 101, 101, 101], dates=dates)

    def fake_features(bars, _config):
        features = bars.copy()
        features["ROC"] = [np.nan, 0.5, 0.6, 0.4, 0.3, 0.2]
        features["CKStop"] = [np.nan, 90.0, 105.0, 80.0, 80.0, 80.0]
        features["FreshBreakout"] = [False, True, False, False, False, False]
        return features

    monkeypatch.setattr(breakout_engine, "compute_features", fake_features)
    config = _config(
        start_date=dates[2].date().isoformat(),
        end_date=dates[4].date().isoformat(),
        initial_equity=100_000.0,
    )
    result = run_backtest({"A": stock, "SPY": spy}, dates, ["A"], config)
    trade = result.trades.iloc[0]
    assert trade["EntryDate"] == dates[2]
    assert trade["ExitDate"] == dates[4]
    assert trade["RawExitPrice"] == 105.0
    assert trade["ExitReason"] == "StopTouch"


def test_terminal_equity_reconciles_to_trade_pnl_and_caps():
    panel, dates = _rank_panel(gap_stop=True)
    config = _config(initial_equity=100_000.0, max_gross_fraction=0.30)
    result = run_backtest(panel, dates, ["A", "B"], config)
    expected = config.initial_equity + result.trades["PnL"].sum()
    assert np.isclose(result.equity["Equity"].iloc[-1], expected)
    assert result.equity["Cash"].min() >= -1e-9
    # Admission caps do not force a rebalance after prices appreciate, but the
    # cash-only engine must never create leverage.
    assert result.equity["GrossPct"].max() <= 1.0 + 1e-9


def test_entry_risk_is_at_most_half_percent_before_caps():
    panel, dates = _rank_panel()
    config = _config(initial_equity=100_000.0, risk_fraction=0.005)
    result = run_backtest(panel, dates, ["B"], config)
    entered = result.candidates[result.candidates["Status"] == "Entered"].iloc[0]
    assert entered["TargetRisk"] == 500.0
    assert 0 < entered["InitialRisk"] <= entered["TargetRisk"]


def test_stock_universe_excludes_declared_instruments():
    stocks, excluded = build_stock_universe(["AAPL", "SPY", "GLD", "MSFT"], ["SPY", "GLD"])
    assert stocks == ["AAPL", "MSFT"]
    assert excluded == ["GLD", "SPY"]


def test_constant_monthly_returns_produce_undefined_bootstrap_not_error():
    dates = pd.date_range("2020-01-31", periods=36, freq="ME")
    result = monthly_block_bootstrap_sharpe(pd.DataFrame({"Return": 0.0}, index=dates))
    assert all(np.isnan(value) for value in result.values())


def test_report_surfaces_spec_gates_and_reliance_boundary():
    panel, dates = _rank_panel()
    result = run_backtest(panel, dates, ["A", "B"], _config())
    robustness = pd.DataFrame(
        [
            {
                "Variant": "Primary",
                "Primary": True,
                "CAGR": result.metrics["cagr"],
                "Sharpe": result.metrics["sharpe"],
                "MaxDD": result.metrics["max_drawdown"],
                "Trades": result.metrics["trades"],
                "AverageGross": result.metrics["average_gross"],
            }
        ]
    )
    periods = pd.DataFrame(
        [
            {
                "Period": "Synthetic",
                "cagr": result.metrics["cagr"],
                "sharpe": result.metrics["sharpe"],
                "max_drawdown": result.metrics["max_drawdown"],
                "spy_cagr": result.metrics["spy_cagr"],
                "ew_cagr": result.metrics["ew_cagr"],
            }
        ]
    )
    output = Path("artifacts/test_primary_breakout_report/report.html").resolve()
    render_report(
        result,
        robustness,
        periods,
        {"sharpe_p05": -1.0, "sharpe_p95": 1.0},
        {"synthetic_gate": False},
        ["SPY"],
        "universe-hash",
        "data-hash",
        output,
    )
    text = output.read_text(encoding="utf-8")
    assert "Frozen specification" in text
    assert "Static-universe bias" in text
    assert "DO NOT ADVANCE" in text
    assert "production" in text.lower()
    assert "<script" not in text.lower()
