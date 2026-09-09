import copy
import pytest
from inventory_reconciliation import inventory_reconciliation

STRATEGY='Oversold Low Volume'
REF=f'SPY|BUY|{STRATEGY}|2026-09-01'


def book():
    order=dict(account='P',con_id=42,symbol='SPY',action='SELL',status='Submitted',
               order_ref=REF,oca_group='group',client_id=1,parent_id=0,qty=100,
               filled=20,remaining=80,order_type='LMT',lmt=110)
    return {'accounts':[{'key':'primary','broker_account':'P',
        'positions':[dict(account='P',con_id=42,symbol='SPY',position=80)],
        'orders':[order,dict(order,order_type='MKT',good_after='20260916 15:59:00 US/Eastern')]}]}


def report(value=None):
    return inventory_reconciliation(value or book(),[],strategies={STRATEGY})


def test_oca_siblings_count_once_using_remaining_not_original_quantity():
    result=report()
    assert len(result['tranches'])==1
    assert result['tranches'][0]['claimed_qty']==80
    assert result['balances'][0]['residual']==0
    assert result['status']=='review_required'


def test_legacy_snapshot_is_never_labeled_verified_quantity():
    value=book()
    for order in value['accounts'][0]['orders']:
        order.pop('remaining');order.pop('filled')
    result=report(value)
    assert result['tranches'][0]['quantity_basis']=='total_quantity_unverified'
    assert result['balances'][0]['residual']==-20


def test_option_contract_sharing_symbol_remains_separate():
    value=book()
    value['accounts'][0]['positions'].append(dict(account='P',con_id=43,symbol='SPY',position=-2))
    result=report(value)
    assert len(result['balances'])==2 and result['balances'][1]['residual']==-2


def test_conflicting_siblings_require_review():
    value=book();value['accounts'][0]['orders'][1]['remaining']=79
    result=report(value)
    assert result['issues'] and not result['tranches']


def test_working_entry_children_are_not_counted_as_held():
    value=book();account=value['accounts'][0]
    for order in account['orders']:order['parent_id']=7
    account['orders'].append(dict(account['orders'][0],order_id=7,parent_id=0,action='BUY'))
    assert not report(value)['tranches']


def test_cross_account_evidence_rejected_without_mutating_input():
    value=book();value['accounts'][0]['positions'][0]['account']='OTHER'
    before=copy.deepcopy(value)
    with pytest.raises(ValueError,match='account'):report(value)
    assert value==before
