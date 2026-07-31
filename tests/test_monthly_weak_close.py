"""Monthly Weak Close pilot (2026-07-31): filter semantics + config invariants.

The monthly weak-close range filter (filters.use_month_range_pos) passes only
on a month's last trading day when the month closed at/below
month_range_pos_max of its high-low range. Evidence:
scratch/monthly_weak_close_mr*.py.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _NoOp:
    def __getattr__(self, name): return self
    def __call__(self, *a, **k): return self
    def __bool__(self): return False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **k):
        def deco(fn): return fn
        return deco
    cache_resource = cache_data


sys.modules.setdefault('streamlit', _NoOp())

from filters import evaluate_filter_mask
from strategy_config import GLOBAL_RISK_MULTIPLIER, STRATEGY_BOOK

PARAMS = {'use_month_range_pos': True, 'month_range_pos_max': 0.15,
          'trend_filter': 'None'}


def _df(end='2026-06-30', start='2026-04-01'):
    idx = pd.bdate_range(start, end)
    return pd.DataFrame({'High': 110.0, 'Low': 100.0, 'Close': 105.0,
                         'ATR': 1.0, 'vol_ma': 1e6, 'age_years': 10.0},
                        index=idx)


def _month_end_rows(df):
    per = df.index.to_period('M')
    return df.index[np.r_[per[1:] != per[:-1], True]]


def test_fires_only_on_weak_month_end():
    df = _df()
    may_end = pd.Timestamp('2026-05-29')
    df.loc[may_end, 'Close'] = 101.0          # pos 0.10
    df.loc['2026-05-15', 'Close'] = 100.5     # weaker, but mid-month
    mask = evaluate_filter_mask(df, PARAMS)
    m = pd.Series(mask, index=df.index)
    assert m.loc[may_end]
    assert not m.loc['2026-05-15']
    assert not m.loc['2026-04-30']            # month-end, pos 0.5
    assert m.sum() == 1


def test_final_row_uses_calendar_month_end():
    # df ending exactly on a month's last trading day: final row can fire
    df = _df(end='2026-06-30')
    df.loc['2026-06-30', 'Close'] = 100.9
    assert np.asarray(evaluate_filter_mask(df, PARAMS))[-1]
    # same weak close but the df ends mid-month: must not fire
    df2 = _df(end='2026-06-29')
    df2.loc['2026-06-29', 'Close'] = 100.9
    assert not np.asarray(evaluate_filter_mask(df2, PARAMS))[-1]


def test_threshold_boundary_inclusive():
    df = _df()
    df.loc['2026-05-29', 'Close'] = 101.5     # pos exactly 0.15
    assert pd.Series(evaluate_filter_mask(df, PARAMS),
                     index=df.index).loc['2026-05-29']
    df.loc['2026-05-29', 'Close'] = 101.6     # pos 0.16
    assert not pd.Series(evaluate_filter_mask(df, PARAMS),
                         index=df.index).loc['2026-05-29']


def test_degenerate_range_rejects():
    df = _df()
    may = df.index.to_period('M') == pd.Period('2026-05')
    df.loc[may, ['High', 'Low', 'Close']] = 100.0
    m = pd.Series(evaluate_filter_mask(df, PARAMS), index=df.index)
    assert not m[may].any()


def test_month_range_uses_intramonth_extremes_not_closes():
    # close ends near the LOW of the month's high-low range even though the
    # close itself never dipped: High spike early in the month widens the
    # range, so pos is computed off High/Low, not closes
    df = _df()
    df.loc['2026-05-04', 'High'] = 140.0      # range now 100..140
    df.loc['2026-05-29', 'Close'] = 105.0     # pos = 5/40 = 0.125
    m = pd.Series(evaluate_filter_mask(df, PARAMS), index=df.index)
    assert m.loc['2026-05-29']


def _strat():
    for s in STRATEGY_BOOK:
        if s["name"] == "Monthly Weak Close":
            return s
    raise AssertionError("Monthly Weak Close not in STRATEGY_BOOK")


def test_config_invariants():
    s = _strat()
    st, ex = s["settings"], s["execution"]
    assert s["universe_tickers"] == ['SPY', 'QQQ']
    assert st["use_month_range_pos"] is True
    assert st["month_range_pos_max"] == 0.15
    assert st["trend_filter"] == "Price > 200 SMA"
    assert st["trade_direction"] == "Long"
    assert st["entry_type"] == "Limit Order -0.25 ATR (Persistent)"
    assert st["max_one_pos"] is True
    assert ex["risk_bps"] == 30 * GLOBAL_RISK_MULTIPLIER  # nominal 30, GRM-scaled at import
    assert ex["hold_days"] == 5
    assert ex["tgt_atr"] == 2.0
    assert ex["fill_window_days"] == 2
    assert ex["use_stop_loss"] is False          # no stop by design
    assert ex["use_take_profit"] is True
    assert ex["stop_atr"] == 1.0                 # sizing risk unit only
    assert ex["frag_risk_bands"] == [[50, 999, 0.25]]  # FAMILY4 analogy


def test_single_carrier_of_month_range_filter():
    carriers = [s["name"] for s in STRATEGY_BOOK
                if s["settings"].get("use_month_range_pos")]
    assert carriers == ["Monthly Weak Close"]
