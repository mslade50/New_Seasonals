"""Offline regressions; extract UI functions without running Streamlit entrypoints."""
import ast
import datetime
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(os.environ.get("AUDIT_SOURCE_ROOT", Path(__file__).resolve().parents[1]))


def functions(path, names, **extra):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    scope = dict(pd=pd, np=np, os=os, datetime=datetime, **extra)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), scope)
    return scope


@pytest.mark.parametrize("horizon", [5, 10, 21, 63, 126, 252])
def test_seasonal_ranks_ignore_unknown_outcomes(horizon):
    scope = functions("build_atr_seasonal_ranks.py", ["prepare_ticker_data", "compute_ranks_for_year"],
                      ATR_WINDOW=14, FWD_WINDOWS=[5, 10, 21, 63, 126, 252], MAX_DAY_COUNT=251)
    idx = pd.bdate_range("2000-01-01", "2026-12-31")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(42).normal(0.0002, .008, len(idx))))
    prices = pd.DataFrame({"Close": close, "High": close * 1.01, "Low": close * .99}, index=idx)
    altered = prices.copy()
    altered.loc["2025":] *= 2
    rank = lambda p: scope["compute_ranks_for_year"](scope["prepare_ticker_data"](p), 2025)
    pd.testing.assert_series_equal(rank(prices)[f"atr_sznl_{horizon}d"], rank(altered)[f"atr_sznl_{horizon}d"])
    pd.testing.assert_series_equal(rank(prices)[f"atr_sznl_{horizon}d"], rank(prices.loc[:"2024"])[f"atr_sznl_{horizon}d"])


def test_annual_stats_reconcile_first_days_and_single_observation_year():
    fn = functions("pages/strat_backtester.py", ["calculate_annual_stats"])["calculate_annual_stats"]
    pnl = pd.Series([-100., 20., -50., 10., 30.], index=pd.to_datetime(["2025-01-02", "2025-01-03", "2026-01-02", "2026-01-03", "2027-01-04"]))
    result = fn(pnl, 1000).set_index("Year")
    assert result.loc[2025, "Total Return ($)"] == -80
    assert result.loc[2026, "Total Return ($)"] == -40
    assert result.loc[2027, "Total Return ($)"] == 30
    assert result.loc["Total", "Total Return ($)"] == pnl.sum()
    assert result.loc[2025, "Max Drawdown"] == pytest.approx(-.1)
    assert result.loc[2026, "Max Drawdown"] == pytest.approx(-50 / 920)


def test_single_strategy_drawdown_includes_initial_capital():
    fn = functions('pages/backtester.py', ['compute_portfolio_stats'])['compute_portfolio_stats']
    equity = pd.DataFrame({'Equity_Close':[900.,920.], 'Equity_High':[1000.,920.],
                           'Equity_Low':[895.,900.], 'InMarket':[True,True]},
                          index=pd.to_datetime(['2025-01-02','2025-01-03']))
    result = fn(equity, 1000)
    assert result['TotalReturn_Pct'] == pytest.approx(-8.)
    assert result['MaxDD_Pct'] == pytest.approx(-10.)
    assert result['MaxDD_Low_Pct'] == pytest.approx(-10.5)
    assert result['DDStillOngoing']


def test_trailing_windows_are_calendar_months():
    fn = functions("daily_portfolio_report.py", ["calculate_trailing_strategy_stats"])["calculate_trailing_strategy_stats"]
    today = pd.Timestamp("2026-09-06")
    df = pd.DataFrame({"Entry Date": [today-pd.Timedelta(days=d) for d in [1, 75, 150, 300, -1]], "Strategy": ["A"]*5, "PnL": [1.]*5})
    result = fn(df, as_of=today)
    assert [int(result[key].iloc[0]["Trades"]) for key in ["3M", "6M", "12M"]] == [2, 3, 4]


def test_rebuild_accepts_both_yfinance_column_orientations():
    fn = functions("scripts/build_master_prices.py", ["_normalize_ticker_df"])["_normalize_ticker_df"]
    for columns in [[("SPY", "Close"), ("SPY", "Open")], [("Close", "SPY"), ("Open", "SPY")]]:
        frame = pd.DataFrame([[100., 99.]], index=pd.to_datetime(["2026-01-02"]), columns=pd.MultiIndex.from_tuples(columns))
        assert fn(frame)["Close"].iloc[0] == 100


@pytest.mark.parametrize("values", [[], [np.nan]*300, [100.]*299+[np.nan], [100.]*299+[np.inf]])
def test_price_context_unknown_is_json_null(values):
    fn = functions("pages/risk_dashboard_v2.py", ["compute_price_context"])["compute_price_context"]
    ctx = fn(pd.Series(values, index=pd.bdate_range("2025-01-01", periods=len(values)), dtype=float))
    assert ctx["regime_label"] == "Insufficient data"
    assert ctx["price"] is None
    json.dumps(ctx, allow_nan=False)


def test_price_context_banner_renders_missing_price():
    from types import SimpleNamespace
    output = []
    scope = functions('pages/risk_dashboard_v2.py', ['compute_price_context', 'render_price_context'],
                      st=SimpleNamespace(markdown=lambda text, **kwargs: output.append(text)))
    context = scope['compute_price_context'](pd.Series(dtype=float))
    scope['render_price_context'](context)
    assert 'SPY: N/A' in output[0]


def test_compounding_uses_only_profits_realized_before_entry():
    fn = functions("pages/fragility_sizing_lab.py", ["replay_equity"], _assign_regime=lambda _: "neutral")["replay_equity"]
    df = pd.DataFrame({"Ticker": ["EARLY", "LATE", "AFTER"], "Strategy": ["A"]*3,
                       "Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-08"]),
                       "Entry Date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-09"]),
                       "Exit Date": pd.to_datetime(["2026-01-08", "2026-01-06", "2026-01-10"]),
                       "Price": [10.]*3, "Exit Price": [11.]*3, "Action": ["BUY"]*3,
                       "Risk bps": [1000.]*3, "ATR": [1.]*3, "stop_atr": [1.]*3})
    out = fn(df, None, 1000, 1, 1, 100, ["A"], scale_with_equity=True).set_index("Ticker")
    assert out.loc["EARLY", "Orig Shares"] == 100
    assert out.loc["LATE", "Orig Shares"] == 100
    assert out.loc["AFTER", "Orig Shares"] == 120


def test_dispersion_does_not_invent_returns_at_missing_endpoints():
    fn = functions("abs_return_dispersion.py", ["compute_dispersion_series"])["compute_dispersion_series"]
    frame = pd.DataFrame({"SPY": [100.,100.,100.], "A": [100.,100.,110.], "B": [100.,np.nan,np.nan]}, index=pd.bdate_range("2026-01-01", periods=3))
    result = fn(frame, window=1)
    assert result.iloc[-1]["n_constituents"] == 1


def test_rotation_ui_names_the_close_execution_it_models():
    source = (ROOT / "pages/rotation_backtester.py").read_text(encoding="utf-8")
    assert "Execute next-session close (T+1 lag)" in source
    assert "Execute next-day open" not in source
