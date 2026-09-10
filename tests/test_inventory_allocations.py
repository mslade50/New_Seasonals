import copy
import pytest
from tests.test_tagged_inventory import seed, fill, build


def assigned():
    start = seed()
    start['execution_allocations'] = {'fixture.01': {
        'review': copy.deepcopy(start['review']),
        'allocations': [{'tranche_id':'test-tranche','qty':20}]}}
    return start


def manual(**kwargs):
    return fill(strategy='', order_ref='', **kwargs)


def test_unassigned_tws_sale_stays_discretionary():
    result=build(fills=[manual()])
    assert result.status=='known' and result.tranches[0]['signed_qty']==100


def test_only_explicitly_assigned_sale_changes_algo_inventory():
    result=build(start=assigned(), fills=[manual()])
    assert result.status=='known' and result.tranches[0]['signed_qty']==80
    assert result.tranches[0]['atr']==3 and result.tranches[0]['entry_price']==100


@pytest.mark.parametrize('change', [
    lambda a:a.update(review={}),
    lambda a:a['allocations'][0].update(qty=19),
    lambda a:a['allocations'][0].update(tranche_id='missing'),
    lambda a:a['allocations'].append(dict(a['allocations'][0])),
])
def test_invalid_assignment_never_becomes_known(change):
    start=assigned();change(start['execution_allocations']['fixture.01'])
    assert build(start=start,fills=[manual()]).status=='unknown'


def test_correction_requires_review_of_exact_execution_revision():
    result=build(start=assigned(),fills=[manual(),manual(exec_id='fixture.02',qty=10)])
    assert result.status=='unknown' and 'corrected' in result.reasons[0]


def test_wrong_contract_or_account_cannot_be_allocated():
    assert build(start=assigned(),fills=[manual(con_id=43)]).status=='unknown'
    assert build(start=assigned(),fills=[manual(account='OTHER')]).status=='unknown'


def test_assigned_add_weights_cost_without_changing_exit_metadata():
    result=build(start=assigned(),fills=[manual(side='BOT')])
    assert result.status=='known' and result.tranches[0]['signed_qty']==120
    assert result.tranches[0]['entry_price']==pytest.approx((10000+2100)/120)
    assert result.tranches[0]['atr']==3
