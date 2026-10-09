import ast
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from indicators import calculate_indicators
from scripts.backtest_inside_day_breakout import (
    ExitSpec, buy_stop_fill, one_position_mask, signal_mask, simulate, valid_ohlc,
)


def fixture():
    meta = pd.DataFrame({'ticker': ['X'], 'entry': [100.], 'atr': [2.],
                         'signal_low': [97.], 'signal_idx': [0], 'entry_idx': [1]})
    paths = {c: np.array([a], dtype=float) for c, a in {
        'Open': [99, 100, 101], 'High': [104, 104, 103],
        'Low': [95, 99, 100], 'Close': [101, 102, 102]}.items()}
    return meta, paths


def test_buy_stop_touch_gap_and_expiry():
    fills = buy_stop_fill(np.array([100, 100, 100]), np.array([99, 102, 99]),
                          np.array([100, 103, 99.9]))
    np.testing.assert_allclose(fills[:2], [100, 102])
    assert np.isnan(fills[2])


def test_invalid_bars_are_detected_including_close_outside_range():
    bars = pd.DataFrame({'Open': [100., 100., 100.], 'High': [102., 102., 102.],
                         'Low': [99., 0., 99.], 'Close': [101., 101., 103.]})
    assert valid_ohlc(bars).tolist() == [True, False, False]


def test_time_stop_and_entry_day_arming_follow_existing_framework():
    m, p = fixture()
    r = simulate(m, p, ExitSpec(2, None, None), cost_bps=0)
    assert r.exit_day.iloc[0] == 2
    assert r.exit.iloc[0] == 102
    r = simulate(m, p, ExitSpec(2, 1., None), cost_bps=0)
    assert r.reason.iloc[0] == 'Time'  # entry-day low is ignored in primary
    r = simulate(m, p, ExitSpec(2, 1., None), entry_day=True, cost_bps=0)
    assert r.exit.iloc[0] == 98  # pre-trigger open is not a valid stop fill


def test_stop_first_collision_and_gap_fills():
    m, p = fixture()
    p['Low'][0, 1] = 97
    assert simulate(m, p, ExitSpec(2, 1., 1.), cost_bps=0).exit.iloc[0] == 98
    p['Open'][0, 1] = 96
    assert simulate(m, p, ExitSpec(2, 1., 1.), cost_bps=0).exit.iloc[0] == 96
    p['Open'][0, 1], p['Low'][0, 1], p['High'][0, 1] = 105, 104, 106
    assert simulate(m, p, ExitSpec(2, None, 1.), cost_bps=0).exit.iloc[0] == 105


def test_signal_low_r_target_and_cost():
    m, p = fixture()
    r = simulate(m, p, ExitSpec(2, 'signal_low', 1., True), cost_bps=5)
    assert r.exit.iloc[0] == 103
    np.testing.assert_allclose(r.net_r.iloc[0], (103 * .9995 - 100 * 1.0005) / 3)
    np.testing.assert_allclose(r.net_atr.iloc[0], (103 * .9995 - 100 * 1.0005) / 2)


def test_zero_risk_and_unavailable_atr_cannot_produce_statistics():
    m, p = fixture()
    m['signal_low'] = 100.
    with pytest.raises(ValueError, match='finite and positive'):
        simulate(m, p, ExitSpec(2, 'signal_low', 1., True))
    m['atr'] = np.nan
    with pytest.raises(ValueError, match='finite and positive'):
        simulate(m, p, ExitSpec(2, None, None))


def test_one_position_rejects_signal_on_exit_date():
    m = pd.DataFrame({'ticker': ['X', 'X', 'Y', 'X'], 'signal_idx': [0, 3, 0, 4],
                      'entry_idx': [1, 4, 1, 5]})
    r = pd.DataFrame({'exit_day': [2, 2, 2, 2]})
    assert one_position_mask(m, r).tolist() == [True, False, True, True]


def test_signal_boundaries_strict_inside_day_and_optional_126():
    df = pd.DataFrame({'rank_ret_252d': [50, 90, 90], 'rank_ret_126d': [49, 49, 50],
                       'rank_ret_5d': [94, 94, 95], 'rank_ret_10d': [94]*3,
                       'rank_ret_21d': [94]*3, 'High': [103, 102, 102],
                       'Low': [95, 96, 97], 'Close': [100]*3,
                       'SMA10': [99]*3, 'EMA21': [99]*3})
    assert signal_mask(df, False).tolist() == [False, True, False]
    assert not signal_mask(df, True).any()


def test_shared_indicator_prefix_is_causal():
    rng = np.random.default_rng(22)
    c = 100 * np.exp(np.cumsum(rng.normal(0, .01, 650)))
    raw = pd.DataFrame({'Open': c, 'Close': c, 'High': c * 1.02,
                        'Low': c * .98, 'Volume': np.full(650, 1e6)},
                       index=pd.bdate_range('2000-01-03', periods=650))
    short = calculate_indicators(raw.iloc[:600], {}, 'X')
    full = calculate_indicators(raw, {}, 'X')
    columns = ['SMA10', 'EMA21', 'ATR'] + [f'rank_ret_{w}d' for w in (5, 10, 21, 126, 252)]
    pd.testing.assert_frame_equal(short[columns], full.iloc[:600][columns])
    assert short.rank_ret_252d.iloc[:503].isna().all()
    assert pd.notna(short.rank_ret_252d.iloc[503])


@pytest.mark.parametrize('stop,target,gap', [(None, None, False), (1., 1., False),
                                           (1., 1., True), (None, 1., True)])
def test_exit_parity_with_existing_backtester_function(stop, target, gap):
    # Load the actual function without importing the Streamlit page or invoking
    # its UI/cache/network setup. Signals/ATR are precomputed to isolate exits.
    source = Path(__file__).resolve().parents[1] / 'pages/backtester.py'
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_engine')
    widget = SimpleNamespace(text=lambda *a: None, progress=lambda *a: None, empty=lambda: None)
    env = {'pd': pd, 'np': np, 're': re, 'MARKET_TICKER': '^GSPC', 'VIX_TICKER': '^VIX',
           'st': SimpleNamespace(progress=lambda *a: widget, empty=lambda: widget),
           'calculate_indicators': lambda df, *a: df.copy()}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), env)
    m, p = fixture()
    p['Open'][0, 0] = 100.
    if stop is not None:
        p['Low'][0, 1] = 97.
    if gap:
        p['Open'][0, 1] = 96. if stop is not None else 105.
    data = pd.DataFrame({c: np.full(104, 100.) for c in ('Open', 'High', 'Low', 'Close')},
                        index=pd.bdate_range('2000-01-03', periods=104))
    for c in p:
        data.loc[data.index[101:104], c] = p[c][0]
    data['ATR'], data['ATR_Pct'], data['age_years'], data['vol_ma'] = 2., 2., 1., 1e6
    data['rank_ret_5d'] = 100.
    data.loc[data.index[100], 'rank_ret_5d'] = 0.
    params = {'backtest_start_date': '2000-01-01', 'entry_type': 'T+1 Open',
              'min_price': 0, 'min_vol': 0, 'min_age': 0, 'max_age': 100,
              'min_atr_pct': 0, 'max_atr_pct': 100, 'holding_days': 2,
              'stop_atr': stop or 1., 'tgt_atr': target or 1.,
              'use_stop_loss': stop is not None, 'use_take_profit': target is not None,
              'slippage_bps': 5, 'perf_filters': [{'window': 5, 'logic': '<', 'thresh': 10, 'consecutive': 1}]}
    actual, rejected, _ = env['run_engine']({'X': data}, params, {})
    assert len(actual) == 1 and rejected.empty
    expected = simulate(m, p, ExitSpec(2, stop, target))
    assert actual.Exit.iloc[0] == expected.exit.iloc[0]
    assert actual.Type.iloc[0] == expected.reason.iloc[0]
    np.testing.assert_allclose(actual.R.iloc[0], expected.net_r.iloc[0])
