import hashlib,io,json
import pandas as pd
import pytest
from actual_inventory_io import load_actual_inventory,load_pending_entry_notionals
from tests.test_tagged_inventory import seed,STRATEGY


def fixture(tmp_path):
    now=pd.Timestamp.now(tz='UTC');through=now-pd.Timedelta(seconds=10)
    opening=seed();opening['asof_utc']=(now-pd.Timedelta(minutes=5)).isoformat()
    opening['positions'][0]['entry_order_ref']=f'SPY|BUY|{STRATEGY}|2026-09-01'
    path=tmp_path/'seed.json';path.write_text(json.dumps(opening))
    primary=dict(complete=True,broker_account='TEST_PRIMARY',continuous_from=(now-pd.Timedelta(hours=1)).isoformat(),
                 complete_through=through.isoformat(),source_at=through.isoformat(),received_at=through.isoformat())
    account=dict(key='primary',broker_account='TEST_PRIMARY',error=None,
                 orders_source_at=(through-pd.Timedelta(seconds=1)).timestamp(),orders=[],
                 positions=[dict(account='TEST_PRIMARY',con_id=42,position=100)])
    payload=dict(fills=[],book={'accounts':[account]},completeness={'accounts':{'primary':primary}})
    return opening,path,payload,now


def read(path,payload,now,**kwargs):
    return load_actual_inventory(seed_path=path,fills_loader=lambda *a:payload,asof=now,
                                 algo_strategies={STRATEGY},**kwargs)


def test_live_coverage_inside_seed_needs_no_historical_download(tmp_path):
    _,path,payload,now=fixture(tmp_path)
    result=read(path,payload,now,canonical_loader=lambda:pytest.fail('unnecessary historical read'))
    assert result.status=='known' and result.tranches[0]['signed_qty']==100
    assert load_pending_entry_notionals(result,asof=now)=={}
    assert result.observed_book==payload['book']


@pytest.mark.parametrize('change',[
    lambda p:p['completeness']['accounts']['primary'].update(complete=False),
    lambda p:p['completeness']['accounts']['primary'].update(continuous_from=None),
    lambda p:p['book']['accounts'][0].update(broker_account='PA'),
    lambda p:p['book']['accounts'][0]['positions'][0].update(position=80),
    lambda p:p['book']['accounts'][0].update(orders_source_at=pd.Timestamp.now(tz='UTC').timestamp()),
])
def test_unknown_inputs_do_not_activate_inventory(tmp_path,change):
    _,path,payload,now=fixture(tmp_path);change(payload)
    assert read(path,payload,now).status=='unknown'


@pytest.mark.parametrize('gap,bad_hash',[(False,False),(True,False),(False,True)])
def test_older_seed_requires_verified_overlapping_canonical_history(tmp_path,gap,bad_hash):
    opening,path,payload,now=fixture(tmp_path)
    opening['asof_utc']=(now-pd.Timedelta(days=2)).isoformat();path.write_text(json.dumps(opening))
    end=now-pd.Timedelta(hours=2 if gap else .5)
    prior=dict(payload['completeness']['accounts']['primary'],continuous_from=opening['asof_utc'],complete_through=end.isoformat())
    stream=io.BytesIO();pd.DataFrame(columns=['account_key']).to_parquet(stream,index=False);body=stream.getvalue()
    status={'complete':True,'canonical_sha256':'bad' if bad_hash else hashlib.sha256(body).hexdigest(),
            'completeness':{'accounts':{'primary':prior}}}
    result=read(path,payload,now,canonical_loader=lambda:(status,body))
    assert result.status==('unknown' if gap or bad_hash else 'known')


def test_untagged_discretionary_sale_is_not_assigned_to_algo(tmp_path):
    _,path,payload,now=fixture(tmp_path)
    payload['fills']=[dict(exec_id='manual.01',account_key='primary',account='TEST_PRIMARY',
        symbol='SPY',con_id=42,sec_type='STK',currency='USD',side='SLD',qty=20,price=105,
        time=(now-pd.Timedelta(minutes=1)).isoformat(),order_ref='')]
    result=read(path,payload,now)
    assert result.status=='known' and result.tranches[0]['signed_qty']==100
    payload['book']['accounts'][0]['positions'][0]['position']=80
    assert read(path,payload,now).status=='unknown'


def test_gateway_scope_never_attests_other_algorithms(tmp_path):
    _,path,payload,now=fixture(tmp_path)
    primary=payload['completeness']['accounts']['primary']
    primary.update(olv_continuous_from=primary['continuous_from'],
                   olv_coverage={'scope':'OLV_US_STK_NON_OVERNIGHT'},
                   continuous_from=(now-pd.Timedelta(minutes=1)).isoformat())
    assert read(path,payload,now).status=='known'
    result=load_actual_inventory(seed_path=path,fills_loader=lambda *a:payload,asof=now,
        algo_strategies={STRATEGY,'another algorithm'},canonical_loader=lambda:(_ for _ in ()).throw(ValueError('gap')))
    assert result.status=='unknown'
