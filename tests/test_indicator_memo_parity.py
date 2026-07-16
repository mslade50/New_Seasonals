"""Indicator memo parity guard (2026-07-16).

daily_scan.memoized_indicators shares one indicator frame per (ticker,
market-source, ref-config) across the whole strategy loop (was 5.9x
redundant, ~14 min of the pre-market critical path). Safe only while:
  1. the memoized frame is byte-identical to a fresh compute (incl. the
     folded-in atr_sznl merge)
  2. distinct market/ref configs get distinct cache entries
  3. nothing downstream mutates the shared frame (check_signal is read-only)

Real-data proof at ship time: scratch/verify_indicator_memo.py — 100%
frame + check_signal parity and zero post-run mutations across the full
strategy book. This synthetic test keeps the contract enforced in CI.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import daily_scan as ds


def _ticker_df(seed, n=320):
    rng = np.random.RandomState(seed)
    dates = pd.date_range('2024-01-02', periods=n, freq='B')
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        'Open': close + rng.normal(0, 0.3, n),
        'High': close + abs(rng.normal(0, 0.8, n)),
        'Low': close - abs(rng.normal(0, 0.8, n)),
        'Close': close,
        'Volume': rng.randint(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)


def _fresh(df, t_key, market_series):
    calc = ds.calculate_indicators(df.copy(), {}, t_key, market_series, None,
                                   ref_ticker_ranks=None, xsec_rank_matrices=None)
    return calc


def test_memo_matches_fresh_and_reuses():
    df = _ticker_df(1)
    mkt = _ticker_df(2)
    ms = mkt['Close'] > mkt['Close'].rolling(200).mean()
    memo = {}
    a = ds.memoized_indicators(memo, ('TEST', 'SPY', None), df, {}, 'TEST',
                               ms, None, None, None, {})
    b = ds.memoized_indicators(memo, ('TEST', 'SPY', None), df, {}, 'TEST',
                               ms, None, None, None, {})
    assert a is b, "same key must return the SAME cached frame"
    pd.testing.assert_frame_equal(a, _fresh(df, 'TEST', ms))


def test_distinct_market_config_gets_distinct_entry():
    df = _ticker_df(1)
    m1 = _ticker_df(2)
    m2 = _ticker_df(3)
    ms1 = m1['Close'] > m1['Close'].rolling(200).mean()
    ms2 = m2['Close'] > m2['Close'].rolling(200).mean()
    memo = {}
    a = ds.memoized_indicators(memo, ('TEST', 'SPY', None), df, {}, 'TEST',
                               ms1, None, None, None, {})
    b = ds.memoized_indicators(memo, ('TEST', '^GSPC', None), df, {}, 'TEST',
                               ms2, None, None, None, {})
    assert len(memo) == 2
    assert a is not b


def test_atr_sznl_merge_folded_into_cache():
    df = _ticker_df(4)
    ranks = pd.DataFrame(
        {c: 50.0 for c in ds.ATR_SZNL_COLS},
        index=df.index.normalize(),
    )
    memo = {}
    got = ds.memoized_indicators(memo, ('TEST', None, None), df, {}, 'TEST',
                                 None, None, None, None, {'TEST': ranks})
    for c in ds.ATR_SZNL_COLS:
        if c in ranks.columns:
            assert c in got.columns and float(got[c].iloc[-1]) == 50.0


def test_check_signal_does_not_mutate_cached_frame():
    df = _ticker_df(5)
    memo = {}
    got = ds.memoized_indicators(memo, ('TEST', None, None), df, {}, 'TEST',
                                 None, None, None, None, {})
    snapshot = got.copy(deep=True)
    settings = {
        'trade_direction': 'Long', 'min_price': 1.0, 'min_vol': 1,
        'perf_filters': [{'window': 5, 'logic': '<', 'thresh': 99.0, 'consecutive': 1}],
        'use_sznl': False, 'sznl_logic': '<', 'sznl_thresh': 15.0,
        'use_52w': False, '52w_type': 'New High',
        'use_vol': True, 'vol_thresh': 0.0,
        'ma_consec_filters': [], 'xsec_filters': [],
    }
    ds.check_signal(got, settings, {}, ticker='TEST')
    pd.testing.assert_frame_equal(got, snapshot)
