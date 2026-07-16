"""atr_sznl filter parity: engine mask vs scan check_signal (2026-07-16).

The two implementations diverged: the engine failed OPEN on a missing rank
column (booking ledger trades that live could never stage — the scan fails
closed) and ignored the 'consecutive' key. Both halves now share the scan's
semantics; this pins them.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'pages'))


class _NoOp:
    def __getattr__(self, name):
        def f(*a, **k):
            return self
        return f
    def __call__(self, *a, **k): return self
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **k):
        def deco(fn): return fn
        return deco
    cache_resource = cache_data


sys.modules['streamlit'] = _NoOp()

from strat_backtester import get_historical_mask


def _base_df(n=300, with_rank=True, rank_vals=None):
    dates = pd.date_range('2024-01-02', periods=n, freq='B')
    df = pd.DataFrame({
        'Open': 100.0, 'High': 101.0, 'Low': 99.0, 'Close': 100.0,
        'Volume': 1e6, 'ATR': 2.0, 'RangePct': 50.0,
        # columns get_historical_mask reads unconditionally on some paths —
        # neutral values so only the atr_sznl filter drives the mask
        'ATR_Pct': 2.0, 'DayOfWeekVal': 2, 'Market_Above_SMA200': True,
        'Mkt_Sznl_Ref': 50.0, 'NextOpen': 100.0, 'PrevHigh': 101.0,
        'PrevLow': 99.0, 'SMA200': 95.0, 'Sznl': 50.0, 'VIX_Value': 18.0,
        'age_years': 10.0, 'is_52w_high': False, 'is_52w_low': False,
        'is_ath': False, 'range_in_atr': 1.0, 'vol_ma': 1e6,
        'vol_ratio': 1.0, 'vol_ratio_10d_rank': 50.0,
    }, index=dates)
    if with_rank:
        df['atr_sznl_5d'] = 50.0 if rank_vals is None else rank_vals
    return df


def _params(consecutive=1):
    # minimal mask params: only the atr_sznl filter active
    return {
        'trade_direction': 'Short',
        'atr_sznl_filters': [
            {'window': 5, 'logic': '>', 'thresh': 40.0,
             'thresh_max': 100.0, 'consecutive': consecutive},
        ],
        'min_price': 0.0, 'min_vol': 0,
    }


def test_missing_rank_column_fails_closed():
    df = _base_df(with_rank=False)
    mask = get_historical_mask(df, _params(), {}, ticker_name='TEST')
    assert not mask.any(), (
        "engine must fail CLOSED on a missing atr_sznl column — the scan can "
        "never stage these signals (daily_scan check_signal 4b returns False)"
    )


def test_present_rank_column_passes():
    df = _base_df(with_rank=True)  # 50 > 40 everywhere
    mask = get_historical_mask(df, _params(), {}, ticker_name='TEST')
    assert mask.any()


def test_consecutive_honored():
    n = 300
    vals = np.full(n, 30.0)
    vals[-1] = 50.0          # only the last day satisfies > 40
    df = _base_df(n=n, rank_vals=vals)
    m1 = get_historical_mask(df, _params(consecutive=1), {}, ticker_name='TEST')
    m3 = get_historical_mask(df, _params(consecutive=3), {}, ticker_name='TEST')
    assert bool(m1.iloc[-1]), "single-day satisfaction passes at consecutive=1"
    assert not bool(m3.iloc[-1]), (
        "one qualifying day must NOT pass consecutive=3 — matches daily_scan's "
        "rolling(consec).sum()==consec semantics"
    )
    vals3 = np.full(n, 30.0)
    vals3[-3:] = 50.0
    df3 = _base_df(n=n, rank_vals=vals3)
    m3b = get_historical_mask(df3, _params(consecutive=3), {}, ticker_name='TEST')
    assert bool(m3b.iloc[-1]), "three consecutive qualifying days pass consecutive=3"
