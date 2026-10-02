"""Offline tests: no production jobs, broker imports, or network calls."""
import ast
import copy
import datetime
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REFERENCE = Path(os.environ.get('PA_REFERENCE_ROOT', ROOT))
if REFERENCE != ROOT:
    sys.path.append(str(REFERENCE))
from pa_portfolio import build_pa_payload, execution_candidates


def settings():
    return {'risk_multiplier': 1.3, 'per_strategy_daily_cap_bps': 250}


def trade(**overrides):
    row = {'Signal Date':'2024-01-02', 'Entry Date':'2024-01-03',
           'Exit Date':'2024-01-05', 'Ticker':'TEST', 'Strategy':'Example',
           'Action':'BUY', 'Shares_flat':945, 'Entry Price':100,
           'Exit Price':103, 'Risk_flat_750k':9450, 'Exit Type':'Time'}
    return {**row, **overrides}


def inputs():
    dates = pd.bdate_range('2024-01-02', periods=5)
    prices = {'TEST':pd.DataFrame({'Close':[100,101,99,104,105]},index=dates)}
    return dates, prices


def test_exact_execution_pnl_and_calendar_zero_days():
    dates, prices = inputs()
    data = build_pa_payload(pd.DataFrame([trade()]), prices, dates, 750000, settings())
    t = data['trades'][0]
    assert t['primary_qty'] == 945
    assert t['unit_pnl'] == [1, -2, 4]  # exact exit 103, not close 104
    assert t['stage'] == 1 and t['entry'] == 1 and t['exit'] == 3
    assert len(data['dates']) == 5  # retains zero PnL sessions
    assert 'pa_equity' not in json.dumps(data)


def test_short_and_signal_close_execution():
    dates, prices = inputs()
    t = trade(**{'Signal Date':'2024-01-03','Action':'SELL SHORT'})
    data = build_pa_payload(pd.DataFrame([t]),prices,dates,750000,settings())
    assert data['trades'][0]['unit_pnl'] == [-1,2,-4]
    assert data['trades'][0]['stage'] == 1


def test_gtc_orders_size_on_first_staging_session():
    dates, prices = inputs()
    t = trade(**{'Entry Date':'2024-01-04'})
    data = build_pa_payload(pd.DataFrame([t]),prices,dates,750000,settings())
    assert data['trades'][0]['stage'] == 1
    assert data['trades'][0]['entry'] == 2


def test_unfinished_position_is_labeled_asof_mark():
    dates, prices = inputs()
    t = trade(**{'Time Stop':'2024-01-10'})
    data = build_pa_payload(pd.DataFrame([t]),prices,dates,750000,settings())
    assert data['trades'][0]['exit_type'] == 'As-of mark'


@pytest.mark.parametrize('problem', ['missing','stale','anchor','build','multiplier'])
def test_new_deploy_requires_current_pa_payload(tmp_path, problem):
    import importlib.util
    spec = importlib.util.spec_from_file_location('pa_freshness_fixtures', ROOT/'tests/test_site_freshness_gate.py')
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    _site, _write, _stamped = fixtures._site, fixtures._write, fixtures._stamped
    from scripts.validate_site_freshness import validate_site
    _site(tmp_path)
    meta_path = tmp_path/'data/meta.json'
    meta = json.loads(meta_path.read_text())
    meta.update(pa_portfolio_version=1, account_value=750000)
    meta['payloads']['pa_portfolio'] = True
    _write(meta_path, meta)
    pa = _stamped({'version':1, 'config':{'primary_anchor':750000,'risk_multiplier':1.3},
                   'dates':['2026-08-05'], 'asof':'2026-08-05', 'trades':[{'fixture':True}]})
    if problem == 'stale': pa['asof']='2026-08-04'
    if problem == 'anchor': pa['config']['primary_anchor']=500000
    if problem == 'build': pa['_site_build_id']='wrong'
    if problem == 'multiplier': pa['config']['risk_multiplier']=None
    if problem != 'missing': _write(tmp_path/'data/pa_portfolio.json',pa)
    problems = validate_site(str(tmp_path))
    assert any('PA' in p or 'pa_portfolio' in p for p in problems)


@pytest.mark.parametrize('problem', ['missing_close','missing_ticker','fractional_qty','missing_session'])
def test_reject_incomplete_or_nonexecutable_basis(problem):
    dates, prices = inputs()
    t = trade()
    if problem == 'missing_close': prices['TEST'].iloc[2,0] = np.nan
    if problem == 'missing_ticker': prices = {}
    if problem == 'fractional_qty': t['Shares_flat'] = 1.5
    if problem == 'missing_session': dates = dates.delete(3)
    with pytest.raises(ValueError): build_pa_payload(pd.DataFrame([t]),prices,dates,750000,settings())


def test_execution_alias_uses_etf_geometry_and_deduplicates():
    dates, prices = inputs()
    f = prices['TEST'].assign(Open=50,High=55,Low=49,ATR=2,RangePct=.03)
    candidates = [(dates[0].value,'^GSPC','^GSPC',0,0),
                  (dates[0].value,'SPY','SPY',0,0)]
    c, rows = execution_candidates(candidates,{}, {'SPY':f}, {'^GSPC':'SPY'})
    assert c == [(dates[0].value,'SPY','SPY',0,0)]
    assert rows[('SPY',0)]['atr'] == 2
    assert rows[('SPY',0)]['range_pct'] == 3
    with pytest.raises(ValueError): execution_candidates(candidates,{}, {},{'^GSPC':'SPY'})


def engine_functions(path):
    """Compile only pure function definitions, skipping UI/import side effects.

    Nonfixture overlays have no data here. No refresh/staging entry point is
    imported or executed. The engine's order/exit/cap paths run unchanged.
    """
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    funcs = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    pending = ['process_signals_fast']; selected = set()
    overrides = {'frag_band_mult_at','_sector_map'}
    while pending:
        name = pending.pop()
        if name in selected or name in overrides: continue
        selected.add(name)
        pending.extend(n.id for n in ast.walk(funcs[name])
                       if isinstance(n,ast.Name) and n.id in funcs and n.id not in selected)
    env = {'pd':pd, 'np':np, 'datetime':datetime, 'os':os,
           'TRADING_DAY':pd.offsets.BusinessDay(), 'STOP_SLIP_BPS':3., 'STOP_GAP_SLIP_BPS':10.,
           'SPOT_TO_TRADEABLE':{}, 'GLOBAL_RISK_MULTIPLIER':1., 'OVERFLOW_RISK_OVERRIDES':{},
           'same_day_derate_mult':lambda *args:1., 'frag_band_mult_at':lambda *args,**kwargs:1.,
           '_sector_map':lambda:{}, 'load_earnings_dates_map':lambda:{}}
    for name in selected: funcs[name].decorator_list = []
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in selected]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env


def fixture():
    dates = pd.bdate_range('2024-01-02',periods=5)
    f = pd.DataFrame({'Open':[100,101,100.6,100.5,100],
       'High':[101,103.5,101.5,101.5,102], 'Low':[99,101,100.4,100,100],
       'Close':[100,102,100.8,101,100.5]},index=dates)
    f = f.assign(ATR=2, RangePct=.02, vol_ratio=1, Sznl=50,
                 atr_sznl_5d=50, rank_ret_126d=50, rank_ret_252d=50)
    book = [{'name':'Overbot Vol Spike',
             'settings':{'trade_direction':'Short','entry_type':'Limit (Open +/- 0.75 ATR)','max_one_pos':False},
             'execution':{'risk_bps':40,'slippage_bps':2,'stop_atr':1,'tgt_atr':2,'hold_days':2,
                 'use_stop_loss':False,'use_take_profit':True,'path1_bps':40,'path2_bps':8,
                 'path2_daily_cap_pct':.75,'scaleout_near_frac':.4,'scaleout_near_tgt_atr':1},
             'universe_tickers':['TEST']}]
    rows = {('TEST',0):{'atr':2.,'close':100.,'open':100.,'high':101.,'low':99.,
           'vol_ratio':1.,'sznl':50,'range_pct':2.,'atr_sznl_5d':50.,
           'rank_ret_126d':50.,'rank_ret_252d':50.}}
    return [(dates[0].value,'TEST','TEST',0,0)],rows,{'TEST':f},book


def test_fresh_engine_pa_single_target_and_main_unchanged(monkeypatch):
    # The two modules imported inside the function are pure sizing/config;
    # fixtures explicitly have no overlap overrides or OLV reservations.
    monkeypatch.setitem(sys.modules,'strategy_config',types.SimpleNamespace(CROSS_STRATEGY_OVERLAP_OVERRIDES=[]))
    source_root = REFERENCE
    sys.path.append(str(source_root))  # pure olv_sizing module only
    original = engine_functions(source_root/'pages/strat_backtester.py')
    review = engine_functions(ROOT/'pages/strat_backtester.py')
    c, rows, processed, book = fixture()
    main_before = original['process_signals_fast'](copy.deepcopy(c),rows,processed,book,100000,flat_sizing=True)
    main_after = review['process_signals_fast'](copy.deepcopy(c),rows,processed,book,100000,flat_sizing=True)
    pd.testing.assert_frame_equal(main_before,main_after)
    assert list(main_after['Shares']) == [80,120]
    pa_book = copy.deepcopy(book); pa_book[0]['execution']['scaleout_near_frac'] = 0
    pa = review['process_signals_fast'](copy.deepcopy(c),rows,processed,pa_book,100000,
                                       flat_sizing=True,staging_quantity_floor=True)
    assert len(pa) == 1 and pa.iloc[0]['Shares'] == 200
    assert pa.iloc[0]['Exit Price'] == 101
    assert pa.iloc[0]['PnL'] == 300


def test_daily_cap_floors_pa_basis_without_changing_main():
    env = engine_functions(ROOT/'pages/strat_backtester.py')
    frame = pd.DataFrame([{'Strategy':'Example','Shares':7,'Price':100,
                          'Exit Price':102,'Action':'BUY','PnL':14,'Risk $':70.,'_Sizing ID':'test'}])
    main, pa = frame.copy(),frame.copy()
    env['_apply_daily_risk_scale'](main,[0],.55)
    env['_apply_daily_risk_scale'](pa,[0],.55,staging_quantity_floor=True)
    assert main.iloc[0]['Shares'] == 4
    assert pa.iloc[0]['Shares'] == 3 and pa.iloc[0]['PnL'] == 6
