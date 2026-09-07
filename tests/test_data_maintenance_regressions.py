"""Offline behavior checks for the review's ancillary data/engine findings."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd

from tests.test_data_audit_regressions import functions


def mtm_inputs():
    trades = pd.DataFrame({'Ticker':['SPY'], 'Action':['BUY'], 'Shares':[10.],
                           'Entry Date':pd.to_datetime(['2026-01-16']),
                           'Exit Date':pd.to_datetime(['2026-01-20']),
                           'Price':[100.], 'PnL':[20.]})
    prices = {'SPY':pd.DataFrame({'Close':[101., 103.]},
                                index=pd.to_datetime(['2026-01-16','2026-01-20']))}
    return trades, prices


def test_mtm_is_bounded_by_inputs_and_excludes_exchange_holidays():
    fn = functions('pages/strat_backtester.py',['get_daily_mtm_series'])['get_daily_mtm_series']
    trades,prices = mtm_inputs()
    pnl = fn(trades,prices)
    assert list(pnl.index) == list(prices['SPY'].index)
    assert pnl.tolist() == [10.,10.]
    assert pnl.sum() == trades.PnL.sum()


def test_mtm_asof_does_not_realize_future_exit_and_start_preserves_daily_marks():
    fn = functions('pages/strat_backtester.py',['get_daily_mtm_series'])['get_daily_mtm_series']
    trades,prices = mtm_inputs()
    assert fn(trades,prices,end_date='2026-01-16').tolist() == [10.]
    assert fn(trades,prices,start_date='2026-01-20').tolist() == [10.]
    assert fn(trades,prices,end_date='2026-01-15').empty


def test_mtm_accepts_explicit_other_market_sessions():
    fn = functions('pages/strat_backtester.py',['get_daily_mtm_series'])['get_daily_mtm_series']
    trades,prices = mtm_inputs()
    sessions = pd.to_datetime(['2026-01-16','2026-01-17','2026-01-20'])
    prices['SPY'] = pd.DataFrame({'Close':[101.,101.5,103.]},index=sessions)
    pnl = fn(trades,prices,session_dates=sessions)
    assert list(pnl.index) == list(sessions)
    assert pnl.tolist() == [10.,5.,5.]


def test_reference_tickers_and_additional_windows_remain_independent(tmp_path, monkeypatch):
    from filters import evaluate_filter_mask
    scope = functions('pages/strat_backtester.py',[
        '_indicator_cache_has_required_schema','_indicator_cache_path',
        'precompute_all_indicators','get_historical_mask'],
        parent_dir=str(tmp_path),INDICATOR_CACHE_VERSION='test',
        _INDICATOR_CACHE_REQUIRED_COLUMNS={'Close'},ATR_SZNL_COLS=[],
        ThreadPoolExecutor=ThreadPoolExecutor,as_completed=as_completed,
        evaluate_filter_mask=evaluate_filter_mask)
    monkeypatch.setitem(sys.modules,'cache_io',SimpleNamespace(is_configured=lambda:False))
    def calculate(frame, seasonal, ticker, *args, **kwargs):
        return frame.assign(vol_ma=10000.,age_years=10.,
                            rank_ret_5d=10. if ticker=='IWM' else 90.,rank_ret_10d=80.)
    scope['calculate_indicators'] = calculate
    index = pd.bdate_range('2025-01-01',periods=260)
    frames = {ticker:pd.DataFrame({'Close':np.full(len(index),100.)},index=index)
              for ticker in ['A','IWM','DIA']}
    book = [{'universe_tickers':['A'],'settings':{'use_ref_ticker_filter':True,
              'ref_ticker':ref,'ref_filters':[{'window':window,'logic':'>','thresh':50.}]}}
            for ref,window in [('IWM',5),('DIA',5),('IWM',10)]]
    processed = scope['precompute_all_indicators'](frames,book,{},None)
    observed = [bool(scope['get_historical_mask'](processed['A'],s['settings'],{},'A').iloc[-1]) for s in book]
    assert observed == [False,True,True]
    # Reading one strategy must not mutate another strategy's reference input.
    assert 'Ref_rank_ret_5d' not in processed['A']
