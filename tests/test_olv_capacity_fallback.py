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
    return {'accounts': [dict(key='primary', broker_account='TEST', error=None,
        nlv=600000, orders_source_at=pd.Timestamp(at).timestamp(),
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


def run_scanner(monkeypatch, *, b=None, known=True, exempt=False, when=AT, saved=None, inventory=None):
    real_loader=cap.load_capacity
    def load(inv,**kw):
        def read(key):
            if saved is None:raise FileNotFoundError()
            return saved
        def fetch(*_):
            if not known:raise ConnectionError()
            return b if b is not None else book()
        return real_loader(inv,**kw,reader=read,book_loader=fetch)
    monkeypatch.setattr(cap,'load_capacity',load)
    inventory=inventory or TaggedInventory(reasons=['bridge refuses'])
    context=dict(_actual_inventory=inventory,_cap_strats={STRATEGY},now_eastern=pd.Timestamp(when),
        is_intraday_partial=False,error_tickers=[],_tnc={'pct_nav':.5,'exempt':['SNA'] if exempt else []},
        shares=100,_include_pending=True,t_clean='SNA',strat={'name':STRATEGY},entry=100.25,
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
    result=cap.load_capacity(TaggedInventory(),now=AT,book_loader=lambda *_:b)
    assert not result.known and result.nav is None and not result.held and not result.pending


def test_unmatched_and_old_tags_do_not_free_same_symbol_capacity():
    b=book();b['accounts'][0]['orders']=[]
    result=cap.from_book(b,now=AT)
    assert result.held[('SNA',STRATEGY)]==280000


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


def test_capture_publishes_capacity_even_when_exit_bridge_refuses(monkeypatch,tmp_path,capsys):
    import scripts.capture_closing_inventory as capture
    import cache_io
    writes={}
    class Client:
        def put_object(self,**kw):writes[kw['Key']]=kw['Body']
    monkeypatch.setattr(cache_io,'_client',lambda:Client())
    monkeypatch.setattr(cache_io,'_r2_creds',lambda:{'R2_BUCKET':'test'})
    monkeypatch.setattr(cap,'query_local_book',lambda:book())
    inv=TaggedInventory(reasons=['bridge refuses'])
    monkeypatch.setattr(capture,'load_actual_inventory',lambda **_:inv)
    capture.main(['--publish','--capacity-output',str(tmp_path/'capacity.json'),
                  '--output',str(tmp_path/'inventory.json')],now=AT)
    saved=json.loads(writes['ops/olv_capacity/2026-09-14.json'])
    assert cap.read_snapshot(saved,now='2026-09-15T08:15:00Z').known
    assert not (tmp_path/'inventory.json').exists()
    assert '"inventory": "unknown"' in capsys.readouterr().out
    assert not any(k.startswith('ops/olv_closing_inventory/') for k in writes)
