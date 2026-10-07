import numpy as np
import pandas as pd

from indicators import calculate_indicators, impulse_momentum_features
from scripts.backtest_momentum_impulse import entry_prices, signal_selection
from tests.test_smooth_momentum import prices


def test_impulse_ends_before_pullback_and_uses_then_known_atr():
    raw, market, hours = prices()
    df = calculate_indicators(raw, {}, 'X')
    f = impulse_momentum_features(df, market, hours)
    expected = (df.Close.iloc[-4] - df.Close.iloc[-9]) / df.ATR.iloc[-4]
    assert np.isclose(f.impulse5_before3_atr.iloc[-1], expected)
    prefix = impulse_momentum_features(calculate_indicators(raw.iloc[:300], {}, 'X'), market.iloc[:300], hours.iloc[:300])
    pd.testing.assert_frame_equal(prefix, f.iloc[:300])


def test_entry_expiry_gap_and_untouched_orders():
    m = pd.DataFrame({'signal_close': [100.]*3, 'atr': [2.]*3, 'box_high3': [102.]*3})
    # Later bars touch both levels but may not rescue an expired order.
    paths = {'Open': np.array([[101., 100.], [105., 100.], [98., 100.]]),
             'High': np.array([[101.5, 110.], [106., 110.], [103., 110.]]),
             'Low': np.array([[100., 90.], [104., 90.], [97., 90.]])}
    fill = entry_prices(m, paths, {'entry': 'stop', 'family': 'flag5'})
    np.testing.assert_allclose(fill, [np.nan, 105., 102.], equal_nan=True)
    fill = entry_prices(m, paths, {'entry': 'limit', 'family': 'break21'})
    np.testing.assert_allclose(fill, [np.nan, np.nan, 98.], equal_nan=True)


def test_daily_ranking_cannot_use_tomorrow_availability_or_future_dates():
    f = pd.DataFrame({'ticker': ['A','B'], 'signal_date': pd.to_datetime(['2020-01-02']*2),
                      'trend_stack': True, 'above_ema21': True, 'beta126': 1.5,
                      'efficiency63': [.2,.3], 'largest_step_share63': .1,
                      'relative_return126': .1, 'dollar_volume63': 30_000_000., 'signal_close': 100.,
                      'market_above200': True, 'move21_atr': 5., 'pullback2_atr': -.75,
                      'day_range_atr': 1., 'quiet_volume_rate_ratio': .5, 'event_idx': [0,-1]})
    p = {'family': 'dip2', 'market': True, 'top': 1}
    assert signal_selection(f, p).tolist() == [False, True]
    later = f.iloc[:1].copy()
    later.signal_date = pd.Timestamp('2020-01-03')
    assert signal_selection(pd.concat([f,later], ignore_index=True), p)[:2].tolist() == [False, True]
