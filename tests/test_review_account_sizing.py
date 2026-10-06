"""Pure sizing and isolated fake-broker account separation, never live I/O."""
import copy
import datetime as dt
from types import SimpleNamespace
import uuid

import pandas as pd
import pytest
import pitch_grammar as grammar
import review_publish as publisher
from broker_runtime import review_execution as C, review_execution_runtime as R, review_sizing as S
from tests.test_agent_review_execution import command, config, fake_runtime, NOW, DAY, ROW


def snapshot(account='primary', nlv=100000):
    return {'account':account,'broker_account':'MOCK_'+account,'currency':'USD',
            'source':'fresh_broker_account_summary','request_completed':True,'observed_at':NOW.isoformat(),
            'nlv':nlv,'buying_power':200000,'available_funds':50000,'excess_liquidity':50000}


@pytest.mark.parametrize('product,mode,stop', [('pitch','risk_bps',1),('pitch','risk_bps',None),('pitch','nav_pct',1),('seasonal','risk_bps',3)])
@pytest.mark.parametrize('account,nlv,mult',[('primary',750000,1),('primary',237000,1),('pa',169143.92,1),('pa',169143.92,1.3)])
def test_actual_grammar_parity_with_normalized_weights_and_account_equity(tmp_path,product,mode,stop,account,nlv,mult):
    idea={'grade':'A','horizon_td':7,'entry':{'type':'LIMIT','anchor':'CLOSE','atr_mult':-.5},
          'exit':{'stop_atr':stop,'target_atr':2,'time_td':7,'time_order':'MOO'},
          'legs':[{'ticker':'XLE','side':'LONG','weight':2},{'ticker':'TLT','side':'SHORT','weight':1}],
          'sizing':{'mode':mode,'risk_bps':30,'nav_pct':.05}}
    if product=='seasonal':idea['sizing']['stop_atr_for_sizing']=3
    contexts={k:{'atr':a,'close':c,'date':pd.Timestamp('2026-10-05')} for k,a,c in [('XLE',2,100),('TLT',4,80)]}
    original=grammar.build_orders(idea,contexts,DAY,DAY+'-1',750000,product)
    expected=grammar.build_orders(idea,contexts,DAY,DAY+'-1',nlv*mult,product)
    spec=S.instruction(idea,original,product)
    proposal={'product':product,'orders':original,'source_sizing':spec,
              'account_proposals':{account:{'account':account,'status':'requires_fresh_account_preview','sizing_hash':C.frozen(spec)['hash']}}}
    cfg=config(tmp_path/'unused');cfg['risk_multipliers'][account]=mult
    result,summary=S.size(proposal,snapshot(account,nlv),S.policy(cfg,product,account))
    assert [r['row']['Quantity'] for r in result]==[r['Quantity'] for r in expected]
    assert [r['sizing']['reference_quantity'] for r in result]==[r['Quantity'] for r in original]
    assert summary['account_multiplier']==mult and summary['systematic_grm_applied'] is False
    assert original==proposal['orders']


@pytest.mark.parametrize('change', [lambda s:s.update(nlv=None),lambda s:s.update(nlv=0),lambda s:s.update(nlv=float('nan')),
    lambda s:s.update(currency='EUR'),lambda s:s.update(account='pa'),lambda s:s.update(broker_account='OTHER'),
    lambda s:s.update(request_completed=False),lambda s:s.update(source='cached'),
    lambda s:s.update(observed_at=(NOW-dt.timedelta(seconds=61)).isoformat()),
    lambda s:s.update(observed_at=(NOW+dt.timedelta(seconds=1)).isoformat())])
def test_missing_wrong_stale_equity_cannot_substitute_reference_or_other_account(change):
    value=snapshot();change(value)
    with pytest.raises((ValueError,TypeError)):S.equity(value,'primary','MOCK_primary',NOW)


@pytest.mark.parametrize('product',['pitch','seasonal'])
def test_zero_leg_blocks_entire_pa_idea_and_leaves_primary_possible(tmp_path,product):
    cfg=config(tmp_path/'unused');proposal=command(product,[ROW,{**ROW,'Ticker':'TLT','Leg':2}])['payload']['proposal']['payload']
    result,_=S.size(proposal,snapshot(),S.policy(cfg,product,'primary'));assert len(result)==2
    with pytest.raises(ValueError,match='rounds to zero'):
        S.size(proposal,snapshot('pa',100),S.policy(cfg,product,'pa'))
    assert [r['Quantity'] for r in proposal['orders']]==[10,10]


@pytest.mark.parametrize('field,value',[('Sec_Type','FUT'),('Multiplier',1000),('Contract','derivative contract')])
def test_unsupported_contract_cannot_use_stock_lot_or_multiplier(tmp_path,field,value):
    proposal=copy.deepcopy(command()['payload']['proposal']['payload']);proposal['orders'][0][field]=value
    if field=='Multiplier':
        proposal['source_sizing']['legs'][0]['multiplier']=value
        proposal['account_proposals']['primary']['sizing_hash']=C.frozen(proposal['source_sizing'])['hash']
    with pytest.raises(ValueError,match='instrument unavailable'):
        S.size(proposal,snapshot(),S.policy(config(tmp_path/'unused'),'pitch','primary'))


@pytest.mark.parametrize('product',['pitch','seasonal'])
def test_dual_accounts_own_quantities_keys_retries_and_partial_results(tmp_path,product):
    journal=C.Journal.initialize(tmp_path/'review.sqlite');cfg=config(journal.path)
    made={}
    shared=command(product,[ROW,{**ROW,'Leg':2,'Ticker':'TLT'}])
    for account in S.ACCOUNTS:
        g,ib=fake_runtime(tmp_path);cmd=copy.deepcopy(shared);cmd.update(account=account,id=str(uuid.uuid4()))
        ib.nlvs['pa']=20000  # two-leg PA idea floors to 1 per leg; Primary to 5
        plan=R.preflight(g,ib,cmd,cfg,NOW);journal.save_preview(plan)
        cmd.update(id=str(uuid.uuid4()),dry_run=False);cmd['payload'].update(operation='execute',plan_hash=plan['hash'],confirmed=True,non_atomic_ack=True,risk_ack=True)
        if account=='pa':ib.fail='TLT'
        rec=R.execute_batch(g,ib,cmd,cfg,journal,NOW)
        assert len(ib.calls)==2
        repeated=R.execute_batch(g,ib,cmd,cfg,journal,NOW);assert len(ib.calls)==2 and repeated['key']==rec['key']
        made[account]=(g,ib,cmd,plan,rec)
    primary=made['primary'];pa=made['pa']
    assert primary[4]['state']=='working' and pa[4]['state']=='needs_reconciliation'
    assert [l['payload']['quantity'] for l in primary[3]['payload']['legs']]==[5,5]
    assert [l['payload']['quantity'] for l in pa[3]['payload']['legs']]==[1,1]
    assert primary[4]['key']!=pa[4]['key']
    wrong={'account':'primary','payload':{'product':product,'actor':'qa-human','run_key':pa[4]['key']}}
    with pytest.raises(ValueError):R.reconcile_batch(primary[0],primary[1],wrong,journal)
    assert journal.get(primary[4]['key'])['state']=='working'
    # Same source/account cannot re-enter under a changed proposal version.
    payload=copy.deepcopy(primary[3]['payload']);payload['proposal_hash']='new-version'
    changed=C.frozen(payload)
    with pytest.raises(ValueError,match='permanent source/account'):
        journal.claim(C.run_key(product,DAY+'-1','primary','new-version'),str(uuid.uuid4()),changed)


@pytest.mark.parametrize('change,reason', [('buying_power', 'buying power'),('available_funds','margin'),('excess_liquidity','margin')])
def test_capacity_is_per_account_and_does_not_borrow_other_budget(tmp_path,change,reason):
    journal=C.Journal.initialize(tmp_path/'review.sqlite');cfg=config(journal.path);g,ib=fake_runtime(tmp_path)
    original=g['_review_account_snapshot']
    def evidence(ib,logical,broker):
        s=original(ib,logical,broker);s[change]=1;return s
    g['_review_account_snapshot']=evidence
    with pytest.raises(ValueError,match=reason):R.preflight(g,ib,command(),cfg,NOW)
    assert not ib.calls


@pytest.mark.parametrize('field,value',[('permission_verified',False),('quantity',999),('broker_account','OTHER'),('initial_margin_change',float('inf'))])
def test_instrument_permission_and_capacity_identity_fail_closed(tmp_path,field,value):
    journal=C.Journal.initialize(tmp_path/'review.sqlite');g,ib=fake_runtime(tmp_path);old=g['_review_capacity']
    g['_review_capacity']=lambda *a:{**old(*a),field:value}
    with pytest.raises(ValueError):R.preflight(g,ib,command(),config(journal.path),NOW)
    assert not ib.calls


def test_existing_inventory_and_changed_balance_require_independent_decision(tmp_path):
    journal=C.Journal.initialize(tmp_path/'review.sqlite');cfg=config(journal.path);g,ib=fake_runtime(tmp_path)
    original=g['_review_capture'];g['_review_capture']=lambda *a:{**original(*a),'position':1}
    with pytest.raises(ValueError,match='existing position'):R.preflight(g,ib,command(),cfg,NOW)
    g['_review_capture']=original;cmd=command();plan=R.preflight(g,ib,cmd,cfg,NOW);journal.save_preview(plan)
    ib.nlvs['primary']=101000
    cmd.update(id=str(uuid.uuid4()),dry_run=False);cmd['payload'].update(operation='execute',plan_hash=plan['hash'],confirmed=True,non_atomic_ack=True,risk_ack=True)
    with pytest.raises(ValueError,match='changed'):R.execute_batch(g,ib,cmd,cfg,journal,NOW)
    assert not ib.calls


def test_fresh_summary_request_id_filters_other_subscriptions_and_releases_query():
    calls=[];wrapper=SimpleNamespace(accountSummary=lambda *a:calls.append(a),startReq=lambda rid:rid)
    ib=SimpleNamespace(wrapper=wrapper);tags={'NetLiquidation':100000,'AvailableFunds':10000,'BuyingPower':200000,'ExcessLiquidity':12000}
    def query(rid,*args):
        for tag,value in tags.items():
            wrapper.accountSummary(1,'MOCK_primary',tag,'999','USD')
            wrapper.accountSummary(rid,'OTHER',tag,'444','USD')
            wrapper.accountSummary(rid,'MOCK_primary',tag,str(value),'USD')
    cancelled=[];ib.client=SimpleNamespace(getReqId=lambda:42,reqAccountSummary=query,cancelAccountSummary=cancelled.append)
    completed=[];ib._run=lambda request:completed.append(request)
    g={'_review_clock':lambda:NOW};before=wrapper.accountSummary
    result=R._account_snapshot(g,ib,'primary','MOCK_primary')
    assert result['nlv']==100000 and completed==[42] and cancelled==[42] and wrapper.accountSummary is before
    assert len(calls)==12


def test_daily_risk_ledger_is_agent_account_and_date_scoped(tmp_path):
    journal=C.Journal.initialize(tmp_path/'review.sqlite');cfg=config(journal.path);g,ib=fake_runtime(tmp_path);plan=R.preflight(g,ib,command(),cfg,NOW)
    payload=copy.deepcopy(plan['payload']);payload['source_idea_id']=DAY+'-previous';payload['risk_usd']=1300
    prior=C.frozen(payload);key=C.run_key('pitch',payload['source_idea_id'],'primary',payload['proposal_hash'])
    journal.claim(key,str(uuid.uuid4()),prior)
    with pytest.raises(ValueError,match='staged-day'):R.preflight(g,ib,command(),cfg,NOW)
    assert journal.allocated_risk('pitch','pa',DAY)==0 and journal.allocated_risk('seasonal','primary',DAY)==0
    cfg['risk_multipliers']['pa']=None
    with pytest.raises(ValueError,match='unconfigured'):R.preflight(g,ib,command(account='pa'),cfg,NOW)
    assert not ib.calls


def test_publisher_seals_both_accounts_without_changing_original_quantities():
    source=command()['payload']['proposal']['payload'];orders=source['orders']
    idea={'idea_id':DAY+'-1','title':'fixture','thesis':'offline only','horizon_td':7,'orders':orders,
          'legs':[{'ticker':'XLE','side':'LONG'}],'sizing':{'risk_bps':30,'stop_atr_for_sizing':15},'exit':{'stop_atr':1}}
    receipt={'status':'sent','date':DAY,'delivery_id':'fixture','delivery_digest':'same','verdict_digest':'same','sent_at':DAY+'T09:00:00Z'}
    record=publisher.build_record(product='pitch',asof=DAY,ideas=[idea],receipt=receipt,verdict_digest='same',account_value=750000)
    payload=next(iter(record['proposals'].values()))['payload']
    assert payload['orders']==orders and set(payload['account_proposals'])=={'primary','pa'}
    assert all(v['status']=='requires_fresh_account_preview' for v in payload['account_proposals'].values())
    assert payload['source_sizing_canonical']==C.canonical(payload['source_sizing'])
