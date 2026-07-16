"""filters.py consolidation guards (2026-07-16).

daily_scan.check_signal and strat_backtester.get_historical_mask both
delegate to filters.evaluate_filter_mask — one implementation, so
scan/ledger filter parity is structural. These tests pin the unified
semantics and the two deliberate mode differences. Ship-time proof on real
data: scratch/verify_filters_consolidation.py (100% scan parity vs the
retired implementation; every engine diff a documented scan-truth
correction).
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

import filters
from filters import check_signal_live, evaluate_filter_mask


def _frame(n=300, **overrides):
    dates = pd.date_range('2024-01-02', periods=n, freq='B')
    base = {
        'Open': 100.0, 'High': 101.0, 'Low': 99.0, 'Close': 100.0,
        'Volume': 1e6, 'ATR': 2.0, 'RangePct': 0.5, 'ATR_Pct': 2.0,
        'DayOfWeekVal': 2, 'Market_Above_SMA200': True, 'Mkt_Sznl_Ref': 50.0,
        'NextOpen': 100.0, 'PrevHigh': 101.0, 'PrevLow': 99.0, 'SMA200': 95.0,
        'Sznl': 50.0, 'VIX_Value': 18.0, 'age_years': 10.0,
        'is_52w_high': False, 'is_52w_low': False, 'is_ath': False,
        'range_in_atr': 1.0, 'vol_ma': 1e6, 'vol_ratio': 1.0,
        'vol_ratio_10d_rank': 50.0, 'today_return_atr': 0.0,
    }
    base.update(overrides)
    df = pd.DataFrame(base, index=dates)
    df['DayOfWeekVal'] = df.index.dayofweek
    return df


BASE = {'trade_direction': 'Long', 'min_price': 0.0, 'min_vol': 0,
        'use_sznl': False, 'use_52w': False, 'use_vol': False,
        'ma_consec_filters': [], 'xsec_filters': []}


def test_delegations_share_one_implementation():
    from strat_backtester import get_historical_mask
    import daily_scan as ds
    df = _frame()
    params = dict(BASE, min_atr_pct=0.5)
    mask_engine = get_historical_mask(df, params, {}, ticker_name='TEST')
    mask_shared = evaluate_filter_mask(df, params, ticker_name='TEST', mode='backtest')
    pd.testing.assert_series_equal(mask_engine, mask_shared)
    assert ds.check_signal(df, params, {}, ticker='TEST') == \
        check_signal_live(df, params, ticker='TEST')


def test_etf_atr_exemption_waives_min_floor_only():
    df = _frame(ATR_Pct=0.1)   # below a 0.2 floor
    params = dict(BASE, min_atr_pct=0.2, max_atr_pct=10.0)
    assert not check_signal_live(df, params, ticker='XLE')
    assert check_signal_live(df, params, ticker='SPY'), \
        "SPY is ETF_ATR_EXEMPT — the engine used to reject these rows live takes"
    # max still applies to exempt names
    df_hi = _frame(ATR_Pct=99.0)
    assert not check_signal_live(df_hi, dict(BASE, max_atr_pct=10.0), ticker='SPY')


def test_live_strips_t1_gates_backtest_keeps_them():
    # NextOpen NaN on the last row (live reality) — a T1 gate would reject
    df = _frame()
    df.loc[df.index[-1], 'NextOpen'] = np.nan
    params = dict(BASE, use_t1_open_filter=True,
                  t1_open_filters=[{'reference': 'Close', 'atr_offset': 0.25, 'logic': '>'}])
    assert check_signal_live(df, params, ticker='TEST'), \
        "scan must stamp T1 gates, not evaluate them (order_staging enforces live)"
    mask = evaluate_filter_mask(df, params, ticker_name='TEST', mode='backtest')
    assert not bool(mask.iloc[-1]), "engine keeps the NextOpen gate"


def test_dial_mode_split(monkeypatch):
    params = dict(BASE, dial_filters=[{'dial': '63d', 'window': 1, 'logic': '<', 'thresh': 50.0}])
    df = _frame()
    # missing fragility cache: live fails closed, backtest passes through
    monkeypatch.setitem(filters._FRAG_DF_CACHE, 'loaded', None)
    assert not check_signal_live(df, params, ticker='TEST')
    mask = evaluate_filter_mask(df, params, ticker_name='TEST', mode='backtest')
    assert bool(mask.iloc[-1])
    # stale cache: live fails closed even with the column present
    stale = pd.DataFrame({'63d': [40.0]},
                         index=[df.index[-1] - pd.Timedelta(days=30)])
    monkeypatch.setitem(filters._FRAG_DF_CACHE, 'loaded', stale)
    assert not check_signal_live(df, params, ticker='TEST')
    # fresh cache under threshold: live passes
    fresh = pd.DataFrame({'63d': [40.0]}, index=[df.index[-1]])
    monkeypatch.setitem(filters._FRAG_DF_CACHE, 'loaded', fresh)
    assert check_signal_live(df, params, ticker='TEST')


def test_vix_missing_column_fails_toward_reject():
    df = _frame().drop(columns=['VIX_Value'])
    assert not check_signal_live(df, dict(BASE, use_vix_filter=True, vix_min=10, vix_max=100), ticker='T')
    assert check_signal_live(df, dict(BASE, use_vix_filter=True, vix_min=0, vix_max=100), ticker='T')


def test_or_group_fails_closed_when_unresolvable():
    df = _frame()
    params = dict(BASE, or_filter_groups=[[{'type': 'nonsense', 'window': 5,
                                            'logic': '>', 'thresh': 50}]])
    assert not check_signal_live(df, params, ticker='T')


def test_month_filter_in_shared_mask():
    df = _frame()  # ends mid-2025; last row month known
    last_month = df.index[-1].month
    ok = dict(BASE, use_month_filter=True, allowed_months=[last_month])
    bad = dict(BASE, use_month_filter=True,
               allowed_months=[m for m in range(1, 13) if m != last_month])
    assert check_signal_live(df, ok, ticker='T')
    assert not check_signal_live(df, bad, ticker='T')


def test_dow_empty_allowed_rejects():
    df = _frame()
    assert not check_signal_live(df, dict(BASE, use_dow_filter=True, allowed_days=[]), ticker='T')
