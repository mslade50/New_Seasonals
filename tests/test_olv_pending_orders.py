import datetime as dt
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from olv_sizing import pending_entry_notionals, clip_quantity, ModelReservations
from actual_inventory_io import load_pending_entry_notionals
from tagged_inventory import TaggedInventory

NOW = dt.datetime(2026,9,9,20,tzinfo=dt.timezone.utc)


def order(**changes):
    row = dict(symbol='TEST',action='BUY',sec_type='STK',currency='USD',
        order_type='LMT',parent_id=0,account='PRIMARY',con_id=42,perm_id=701,
        remaining=40,filled=60,qty=100,lmt=99.5,status='Submitted',
        order_ref='TEST|BUY|Oversold Low Volume|2026-09-08')
    row.update(changes)
    return row


def book(rows=None):
    return {'at':NOW.timestamp(), 'accounts':[{'key':'primary','broker_account':'PRIMARY',
        'error':None,'orders_source_at':NOW.timestamp()-10,'orders':[order()] if rows is None else rows}]}


def read(value):
    return pending_entry_notionals(value,'PRIMARY',asof=NOW.isoformat())


def test_partial_fill_reserves_only_remaining_and_dead_orders_release():
    assert read(book()) == {('TEST','Oversold Low Volume'):3980}
    assert read(book([order(status='Cancelled')])) == {}
    assert read(book([order(status='Filled')])) == {}
    assert read(book([order(status='PendingCancel')])) == {('TEST','Oversold Low Volume'):3980}
    assert read(book([order(action='SELL')])) == {}


@pytest.mark.parametrize('changes',[
    {'remaining':None},{'remaining':float('nan')},{'remaining':-1},
    {'remaining':39},{'filled':None},{'con_id':42.5},{'perm_id':0},
    {'account':'PA'},{'currency':'CAD'},{'status':None},{'lmt':0},
    {'symbol':'OTHER'},{'order_type':'MKT'},
])
def test_bad_order_data_is_unknown_not_zero(changes):
    with pytest.raises((ValueError,TypeError,KeyError)):
        read(book([order(**changes)]))


def test_old_feed_missing_remaining_is_not_accepted():
    row=order(); del row['remaining']
    with pytest.raises(KeyError): read(book([row]))


def test_stale_account_cannot_hide_behind_recent_other_account():
    value=book();value['accounts'][0]['orders_source_at']-=100
    with pytest.raises(ValueError,match='stale'):read(value)


def test_duplicate_orders_or_contract_aliases_reject():
    with pytest.raises(ValueError):read(book([order(),order()]))
    with pytest.raises(ValueError):read(book([order(),order(perm_id=702,con_id=43)]))


def test_unknown_inventory_never_reads_pending_orders():
    with pytest.raises(ValueError):
        load_pending_entry_notionals(TaggedInventory(), book_loader=lambda *args:pytest.fail('unexpected read'))


def test_adapter_rejects_book_newer_than_fill_inventory():
    inventory=TaggedInventory(status='known',broker_account='PRIMARY',
        asof_utc=(NOW-dt.timedelta(seconds=20)).isoformat())
    with pytest.raises(ValueError,match='caught up'):
        load_pending_entry_notionals(inventory,asof=NOW.isoformat(),book_loader=lambda *a:book())
    inventory.asof_utc=NOW.isoformat()
    assert load_pending_entry_notionals(inventory,asof=NOW.isoformat(),book_loader=lambda *a:book()) == read(book())


def test_millisecond_observation_matches_iso_receipt_without_float_nanoseconds():
    observed=NOW-dt.timedelta(seconds=10,milliseconds=1)
    inventory=TaggedInventory(status='known',broker_account='PRIMARY',asof_utc=observed.isoformat())
    value=book();value['accounts'][0]['orders_source_at']=observed.timestamp()
    assert load_pending_entry_notionals(inventory,asof=NOW.isoformat(),book_loader=lambda *a:value)==read(value)
    value['accounts'][0]['orders_source_at']+=.001
    with pytest.raises(ValueError,match='caught up'):
        load_pending_entry_notionals(inventory,asof=NOW.isoformat(),book_loader=lambda *a:value)


def test_past_expiry_still_reserves_until_broker_confirms_cancel():
    assert read(book([order(good_till='20260908 15:59:00 US/Eastern')])) == read(book())


def test_quantity_never_uses_more_than_remaining_budget():
    assert clip_quantity(100,99.5,20000,19900)==1
    assert clip_quantity(100,99.5,20000,20001)==0
    with pytest.raises(ValueError):clip_quantity(100,0,20000,0)


def test_model_reservation_moves_from_pending_to_filled_then_releases():
    ledger=ModelReservations()
    item=ledger.reserve(key='TEST',signal_date=1,expiry=4,quantity=100,limit=100,
        equity=100000,fill_date=3,fill_price=90,exit_date=6)
    assert ledger.used_fraction('TEST',2,lambda _:1)==.1
    assert ledger.used_fraction('TEST',3,lambda _:1)==.09
    assert ledger.used_fraction('TEST',6,lambda _:1)==0
    item['fill_date']=None
    assert ledger.used_fraction('TEST',3,lambda _:.5)==.05
    assert ledger.used_fraction('TEST',4,lambda _:1)==0


@pytest.mark.parametrize('known,expected',[(True,50),(False,100)])
def test_actual_scanner_block_combines_held_and_pending(known,expected):
    # Execute only the arithmetic block; never invoke daily_scan.main.
    source=(Path(__file__).resolve().parents[1]/'daily_scan.py').read_text(encoding='utf-8')
    start=source.index("                    _tnc = strat['execution'].get('ticker_notional_cap')")
    end=source.index('                    entry_mode =',start)
    import textwrap
    body=textwrap.dedent(source[start:end])
    wrapper='for _fixture in [0]:\n'+textwrap.indent(body,'    ')
    ns=dict(strat={'name':'Oversold Low Volume','execution':{'ticker_notional_cap':
        {'pct_nav':.2,'exempt':[],'include_pending':True}}},shares=100,entry=100.,atr=2.,
        _actual_inventory=SimpleNamespace(status='known'),_pending_capacity_known=known,
        _pending_notionals={('TEST','Oversold Low Volume'):5000.},
        open_notionals={('TEST','Oversold Low Volume'):10000.},t_clean='TEST',
        ACCOUNT_VALUE=750000,_primary_nav=100000,dist=2.5,risk=250.,sizing_note='',_entry_offset_atr=.25)
    exec(compile(ast.parse(wrapper),'<scanner-cap>','exec'),ns)
    assert ns['shares']==expected
    if known:
        assert ns['_pending_notionals'][('TEST','Oversold Low Volume')]==9975.
