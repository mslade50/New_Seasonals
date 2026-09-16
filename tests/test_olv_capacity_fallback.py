"""Exercise the scanner wiring, not just the reference-attribution helper."""
import ast
import copy
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
import olv_capacity as cap
from tagged_inventory import TaggedInventory

AT = '2026-09-14T20:05:30Z'
STRATEGY = cap.STRATEGY


def book(at=AT):
    return {'capacity_query_started_at':(pd.Timestamp(at)-pd.Timedelta(seconds=3)).isoformat(),
            'capacity_query_completed_at':at,
            'accounts': [dict(key='primary', broker_account='TEST', error=None,
        nlv=600000, orders_source_at=pd.Timestamp(at).timestamp()-2,
        fills_complete=True, fills_source_at=int((pd.Timestamp(at).timestamp()-1)*1000),
        fills_query_from=pd.Timestamp(at).tz_convert('America/New_York').normalize().isoformat(),
        positions=[dict(symbol='SNA', account='TEST', con_id=42, sec_type='STK',
                        currency='USD', position=700, market_value=280000)],
        orders=[dict(symbol='SNA', account='TEST', con_id=42, sec_type='STK', currency='USD',
                     action='BUY', order_ref='SNA|BUY|Oversold Low Volume|2026-09-14',
                     parent_id=0, perm_id=99, order_type='LMT', status='Submitted',
                     qty=200, filled=50, remaining=150, lmt=100)], fills=[])]}


def scanner_blocks():
    tree=ast.parse((Path(__file__).resolve().parents[1]/'daily_scan.py').read_text(encoding='utf-8'))
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_daily_scan')
    start=next(i for i,n in enumerate(run.body) if isinstance(n,ast.ImportFrom) and n.module=='olv_capacity')
    end=next(i for i,n in enumerate(run.body[start:],start) if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='ladder_strats' for t in n.targets))
    branch=next(n for n in ast.walk(run) if isinstance(n,ast.If)
                and ast.unparse(n.test).startswith('_tnc and shares > 0'))
    wrapped=ast.For(target=ast.Name(id='_once',ctx=ast.Store()),iter=ast.List(elts=[ast.Constant(0)],ctx=ast.Load()),
                    body=[branch],orelse=[])
    return compile(ast.fix_missing_locations(ast.Module(body=run.body[start:end]+[wrapped],type_ignores=[])),
                   'actual-scanner-cap-path','exec')


def run_scanner(monkeypatch, *, b=None, known=True, exempt=False, when=AT, saved=None, inventory=None, clock_at=None, ticker='SNA'):
    real_loader=cap.load_capacity
    def load(**kw):
        def read(key):
            if saved is None:raise FileNotFoundError()
            return saved
        def fetch(*_):
            if not known:raise ConnectionError()
            return b if b is not None else book()
        kw.setdefault('now',clock_at or when)
        return real_loader(**kw,reader=read,book_loader=fetch)
    monkeypatch.setattr(cap,'load_capacity',load)
    inventory=inventory or TaggedInventory(reasons=['bridge refuses'])
    context=dict(_actual_inventory=inventory,_cap_strats={STRATEGY},now_eastern=pd.Timestamp(when),
        is_intraday_partial=False,error_tickers=[],_tnc={'pct_nav':.5,'exempt':['SNA'] if exempt else []},
        shares=100,_include_pending=True,t_clean=ticker,strat={'name':STRATEGY},entry=100.25,
        _entry_offset_atr=.25,atr=1.,dist=1.25,risk=125.,sizing_note='',print=lambda *a:None)
    exec(scanner_blocks(),context)
    return context


def test_actual_scanner_caps_unknown_inventory_with_partial_pending_order(monkeypatch):
    inv=TaggedInventory(reasons=['bridge refuses'])
    result=run_scanner(monkeypatch,inventory=inv)
    assert result['_capacity'].known and result['shares']==50
    assert result['risk']==62.5
    assert result['_pending_notionals'][('SNA',STRATEGY)]==20000
    assert '100 -> 50 sh' in result['sizing_note']
    assert inv.status=='unknown' and not inv.exit_metadata_known and inv.tranches==[]


def test_known_exit_inventory_cannot_hide_other_same_ticker_holdings(monkeypatch):
    import actual_inventory_io
    # Exit attribution omits holdings from other strategies; the broker actually
    # holds $280k. With $15k pending and $600k NAV, only $5k is available.
    inv=TaggedInventory(status='known',asof_utc=AT)
    monkeypatch.setattr(actual_inventory_io,'load_primary_nav',lambda *a,**k:600000)
    monkeypatch.setattr(actual_inventory_io,'load_pending_entry_notionals',lambda *a,**k:{})
    result=run_scanner(monkeypatch,inventory=inv)
    assert result['shares']==50
    assert result['_capacity'].held[('SNA',STRATEGY)]==280000


@pytest.mark.parametrize('exempt,available,expected',[(True,True,100),(False,False,100)])
def test_exemption_and_owner_fail_open_policy(monkeypatch,exempt,available,expected):
    result=run_scanner(monkeypatch,exempt=exempt,known=available)
    assert result['shares']==expected
    if not available:assert any('bypassed' in reason for _,reason in result['error_tickers'])


def test_sizing_capture_serves_morning_with_broker_offline(monkeypatch):
    saved=cap.make_snapshot(book(),now=AT)
    result=run_scanner(monkeypatch,known=False,saved=saved,when='2026-09-15T08:15:00Z')
    assert result['shares']==50 and result['_capacity'].source=='prior_close_conservative'


def test_flat_book_is_known_zero_usage():
    b=book();b['accounts'][0].update(positions=[],orders=[])
    result=cap.from_book(b,now=AT)
    assert result.known and result.held=={} and result.pending=={}


@pytest.mark.parametrize('fault',[
    lambda a:a.update(nlv=float('nan')),
    lambda a:a.update(nlv=0),
    lambda a:a.update(orders_source_at=None),
    lambda a:a.update(orders_source_at=pd.Timestamp(AT).timestamp()+1),
    lambda a:a.update(orders_source_at=pd.Timestamp(AT).timestamp()-91),
    lambda a:a.update(positions=None),
    lambda a:a.update(orders=None),
    lambda a:a.update(broker_account=''),
    lambda a:a.update(error='failed'),
    lambda a:a['positions'][0].update(account='PA'),
    lambda a:a['positions'][0].update(market_value=float('inf')),
    lambda a:a['positions'][0].update(market_value=None),
    lambda a:a['positions'][0].update(market_value=0),
    lambda a:a['positions'][0].update(position=float('nan')),
    lambda a:a['positions'].append(copy.deepcopy(a['positions'][0])),
    lambda a:a['orders'][0].update(account='PA'),
    lambda a:a['orders'][0].update(remaining=None),
    lambda a:a['orders'][0].update(remaining=100),
    lambda a:a['orders'][0].update(lmt=float('nan')),
    lambda a:a['orders'].append(copy.deepcopy(a['orders'][0])),
])
def test_bad_capacity_never_pretends_to_be_known(fault):
    b=book();fault(b['accounts'][0])
    result=cap.load_capacity(now=AT,book_loader=lambda *_:b)
    assert not result.known and result.nav is None and not result.held and not result.pending


def test_unmatched_and_old_tags_do_not_free_same_symbol_capacity():
    b=book();b['accounts'][0]['orders']=[]
    result=cap.from_book(b,now=AT)
    assert result.held[('SNA',STRATEGY)]==280000


def test_delayed_scanner_uses_capacity_clock_not_its_start_time(monkeypatch):
    result=run_scanner(monkeypatch,when='2026-09-14T19:40:00Z',clock_at=AT)
    assert result['_capacity'].known and result['shares']==50


@pytest.mark.parametrize('broker_symbol',['BRK B','BRK.B','BRK-B'])
def test_share_class_holdings_use_scanner_symbol_key(monkeypatch,broker_symbol):
    b=book();b['accounts'][0]['orders']=[]
    b['accounts'][0]['positions'][0].update(symbol=broker_symbol,market_value=295000)
    result=run_scanner(monkeypatch,b=b,ticker='BRK-B')
    assert result['_capacity'].known and result['shares']==50


def test_same_symbol_option_order_does_not_disable_stock_capacity(monkeypatch):
    b=book()
    b['accounts'][0]['orders'].append(dict(symbol='SNA',sec_type='OPT',con_id=5555,
                                         action='BUY',status='Submitted',order_ref='manual option'))
    result=run_scanner(monkeypatch,b=b)
    assert result['_capacity'].known and result['shares']==50


def test_live_clock_is_sampled_after_broker_query(monkeypatch):
    # Source completes after the read begins; validation must use completion.
    class Clock:
        def __init__(self):self.calls=0
        def __call__(self):
            self.calls+=1
            return pd.Timestamp(AT)-pd.Timedelta(seconds=10) if self.calls==1 else pd.Timestamp(AT)
    clock=Clock()
    monkeypatch.setattr(cap,'utc_now',clock)
    result=cap.load_capacity(book_loader=lambda:book())
    assert result.known and clock.calls==2


def test_collection_fill_cannot_disappear_between_holdings_and_orders(monkeypatch):
    b=book();a=b['accounts'][0]
    # The 150 remaining shares fill after positions were copied. The order
    # disappears, but its $15k must not disappear from used capacity.
    a['orders']=[]
    a['fills']=[dict(exec_id='fill.01',account='TEST',con_id=42,symbol='SNA',sec_type='STK',currency='USD',
                     side='BOT',qty=150,price=100,time=(pd.Timestamp(AT)-pd.Timedelta(seconds=2)).isoformat())]
    result=run_scanner(monkeypatch,b=b)
    assert result['_capacity'].held[('SNA',STRATEGY)]==295000
    assert result['shares']==50


@pytest.mark.parametrize('fault',[
    lambda b:b.pop('capacity_query_started_at'),
    lambda b:b['accounts'][0].update(fills_complete=False),
    lambda b:b['accounts'][0].update(fills_source_at=None),
    lambda b:b['accounts'][0].update(fills_query_from=None),
    lambda b:b['accounts'][0].update(fills_error='failed'),
    lambda b:b['accounts'][0].update(fills=None),
    lambda b:b['accounts'][0]['orders'][0].update(con_id=43),
])
def test_missing_collection_proof_or_contract_mismatch_refuses(fault):
    b=book();fault(b)
    assert not cap.load_capacity(now=AT,book_loader=lambda:b).known


@pytest.mark.parametrize('fault',[
    lambda f:f.update(account='OTHER'),
    lambda f:f.update(con_id=43),
    lambda f:f.update(qty=float('nan')),
    lambda f:f.update(price=0),
    lambda f:f.update(side='UNKNOWN'),
    lambda f:f.update(time=None),
])
def test_invalid_collection_fill_never_becomes_zero(fault):
    b=book();f=dict(exec_id='fill.01',account='TEST',con_id=42,symbol='SNA',sec_type='STK',currency='USD',
                  side='BOT',qty=150,price=100,time=(pd.Timestamp(AT)-pd.Timedelta(seconds=2)).isoformat())
    fault(f);b['accounts'][0]['fills']=[f]
    assert not cap.load_capacity(now=AT,book_loader=lambda:b).known


@pytest.mark.parametrize('when',['2026-09-15T13:30:00Z','2026-09-16T08:15:00Z','2026-09-14T20:04:00Z'])
def test_closing_snapshot_cannot_outlive_next_open_or_travel_backwards(when):
    with pytest.raises(ValueError):cap.read_snapshot(cap.make_snapshot(book(),now=AT),now=when)


@pytest.mark.parametrize('at,when',[
    ('2026-09-04T20:05:30Z','2026-09-08T08:15:00Z'),
    ('2026-11-27T18:05:30Z','2026-11-30T09:15:00Z'),
])
def test_weekend_holiday_and_half_day_capture(at,when):
    saved=cap.make_snapshot(book(at),now=at)
    assert cap.read_snapshot(saved,now=when).known


@pytest.mark.parametrize('exit_failure',['unknown','exception','invalid_known'])
def test_capture_publishes_capacity_even_when_exit_bridge_refuses(monkeypatch,tmp_path,capsys,exit_failure):
    import scripts.capture_closing_inventory as capture
    import cache_io
    writes={}
    class Client:
        def put_object(self,**kw):writes[kw['Key']]=kw['Body']
    monkeypatch.setattr(cache_io,'_client',lambda:Client())
    monkeypatch.setattr(cache_io,'_r2_creds',lambda:{'R2_BUCKET':'test'})
    monkeypatch.setattr(cap,'query_local_book',lambda:book())
    inv=TaggedInventory(reasons=['bridge refuses'])
    def load(**_):
        if exit_failure=='exception':raise ValueError('fixture failed')
        if exit_failure=='invalid_known':return TaggedInventory(status='known')
        return inv
    monkeypatch.setattr(capture,'load_actual_inventory',load)
    capture.main(['--publish','--capacity-output',str(tmp_path/'capacity.json'),
                  '--output',str(tmp_path/'inventory.json')],now=AT)
    saved=json.loads(writes['ops/olv_capacity/2026-09-14.json'])
    assert cap.read_snapshot(saved,now='2026-09-15T08:15:00Z').known
    assert not (tmp_path/'inventory.json').exists()
    assert '"inventory": "unknown"' in capsys.readouterr().out
    assert not any(k.startswith('ops/olv_closing_inventory/') for k in writes)
