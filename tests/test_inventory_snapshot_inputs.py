import datetime as dt
import json
import pytest
from broker_runtime.inventory_snapshot_inputs import query_start,entry_metadata,gateway_olv_coverage


def gateway(tmp_path, at='2026-09-08T10:00:00+00:00'):
    path=tmp_path/'gateway.json'
    policy=dict(source='gateway',broker_account='P',timezone='America/New_York',lookback_days=1,
                strategy_scope=['Oversold Low Volume'],review={'status':'approved'})
    path.write_text(json.dumps(policy))
    snapshot=dict(broker_account='P',fills_complete=True,
                  fills_source_at=dt.datetime.fromisoformat(at).timestamp()*1000,orders=[],fills=[])
    return path,policy,snapshot


def test_gateway_cannot_attest_tws_history(tmp_path):
    path,policy,_=gateway(tmp_path)
    now=dt.datetime(2026,9,8,10,tzinfo=dt.timezone.utc)
    assert query_start(now,path,'P')=='2026-09-08T04:00:00+00:00'
    for change in [dict(lookback_days=7),dict(settings_path='tws.xml')]:
        path.write_text(json.dumps(dict(policy,**change)))
        with pytest.raises(ValueError,match='Gateway'):query_start(now,path,'P')


@pytest.mark.parametrize('at,cutoff',[
    ('2026-09-08T10:00:00+00:00','2026-09-05T00:00:00+00:00'),
    ('2026-09-06T10:00:00+00:00','2026-09-05T00:00:00+00:00'),
    ('2026-11-28T10:00:00+00:00','2026-11-28T01:00:00+00:00'),
])
def test_gateway_uses_previous_session_including_holidays_and_halfdays(tmp_path,at,cutoff):
    path,_,snapshot=gateway(tmp_path,at)
    assert gateway_olv_coverage(snapshot,path)['prior_session_close']==cutoff


@pytest.mark.parametrize('change',[dict(contract_exchange='OVERNIGHT'),dict(tif='OVERNIGHT'),dict(sec_type='OPT')])
def test_gateway_rejects_unsupported_olv_orders(tmp_path,change):
    path,_,snapshot=gateway(tmp_path)
    order=dict(order_ref='SPY|BUY|Oversold Low Volume|2026-09-08',sec_type='STK',currency='USD',contract_exchange='SMART',tif='GTC')
    snapshot['orders']=[order]
    assert gateway_olv_coverage(snapshot,path)['scope']=='OLV_US_STK_NON_OVERNIGHT'
    snapshot['orders']=[dict(order,**change)]
    with pytest.raises(ValueError,match='route'):gateway_olv_coverage(snapshot,path)


@pytest.mark.parametrize('change',[dict(time='2026-09-08T03:00:00-04:00'),dict(time='2026-09-08T20:00:00-04:00'),
                                  dict(time='2026-09-08T10:00:00'),dict(exchange='IBEOS'),dict(account='PA')])
def test_gateway_rejects_unsupported_execution_scope(tmp_path,change):
    path,_,snapshot=gateway(tmp_path)
    fill=dict(order_ref='SPY|BUY|Oversold Low Volume|2026-09-08',sec_type='STK',currency='USD',
              account='P',exchange='NYSE',time='2026-09-08T04:00:00-04:00')
    snapshot['fills']=[fill]
    assert gateway_olv_coverage(snapshot,path)['scope']
    snapshot['fills']=[dict(fill,**change)]
    with pytest.raises(ValueError):gateway_olv_coverage(snapshot,path)


def test_query_lookback_needs_reviewed_account_and_actual_settings(tmp_path):
    now=dt.datetime(2026,9,9,15,tzinfo=dt.timezone.utc)
    path=tmp_path/'policy.json';settings=tmp_path/'tws.xml'
    settings.write_text('<root tradeLogShowLastNDays="7"/>')
    policy=dict(broker_account='P',timezone='America/New_York',lookback_days=7,
                settings_path=str(settings),review={'status':'approved'})
    path.write_text(json.dumps(policy))
    assert query_start(now,path,'P')=='2026-09-03T04:00:00+00:00'
    settings.write_text('<root tradeLogShowLastNDays="1"/>')
    with pytest.raises(ValueError,match='no longer'):query_start(now,path,'P')
    with pytest.raises(ValueError,match='unreviewed'):query_start(now,path,'OTHER')
    policy.pop('settings_path');path.write_text(json.dumps(policy))
    assert query_start(now,path,'P')=='2026-09-09T04:00:00+00:00'


def test_frozen_metadata_requires_actual_matching_exit_bracket(tmp_path):
    path=tmp_path/'staged.csv'
    path.write_text('Symbol,Strategy_Ref,Staged_Date,Used_ATR,Target_Price,Exit_Condition_Time\nSPY,Oversold Low Volume,2026-09-09,2,105,2026-09-23 15:59:00\n')
    ref='SPY|BUY|Oversold Low Volume|2026-09-09'
    target=dict(order_ref=ref,account='P',action='SELL',order_type='LMT',lmt=105,oca_group='one',con_id=42,sec_type='STK',currency='USD')
    time=dict(target,order_type='MKT',good_after='20260923 15:59:00 US/Eastern')
    account=dict(broker_account='P',orders=[target,time])
    result=entry_metadata(path,account)
    assert result[ref]['atr']==2 and result[ref]['metadata_con_id']==42
    assert result[ref]['exit_deadline_utc']=='2026-09-23T19:59:00+00:00'
    target['lmt']=106
    with pytest.raises(ValueError,match='match'):entry_metadata(path,account)
