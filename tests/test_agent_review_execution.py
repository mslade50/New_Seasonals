"""Offline whole-idea lifecycle tests. No production script imports or sockets."""
import ast
import copy
import datetime as dt
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid
import math
import itertools

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from broker_runtime import review_execution as C
from broker_runtime import review_execution_runtime as R
from broker_runtime import prepare_review_execution as P
from broker_runtime import review_sizing as S

NOW=dt.datetime(2026,10,6,14,tzinfo=dt.timezone.utc)
DAY='2026-10-06'
ROW={'Idea_Id':DAY+'-1','Leg':1,'Ticker':'XLE','Sec_Type':'STK','Contract':'','Proxy_Ticker':'',
     'Action':'BUY','Quantity':10,'Entry_Type':'LIMIT','Entry_Anchor':'CLOSE','Entry_Offset_ATR':-0.5,
     'Order_Type':'LMT','TIF':'DAY','Limit_Price':100,'Stop_Price':98,'Target_Price':104,
     'Stop_ATR':1,'Target_ATR':2,'ATR':2,'Execute_On':DAY,'Time_Exit_Date':'2026-10-13','Time_Exit_Order':'MOO'}


def command(product='pitch', rows=None, account='primary'):
    payload={'schema':1,'product':product,'source_idea_id':DAY+'-1','source_date':DAY,
             'account':'primary' if product=='pitch' else 'Unassigned - manual account selection required',
             'published_at':DAY+'T09:00:00Z','review_deadline':DAY+'T20:00:00Z','orders':copy.deepcopy(rows or [ROW])}
    payload['orders'] = [{**r, 'ATR': r.get('ATR', 2), 'Ref_Close': r.get('Ref_Close', 100), 'Ref_Date': '2026-10-05', 'Multiplier': r.get('Multiplier', 1)} for r in payload['orders']]
    source={'horizon_td':7,'sizing':{'mode':'risk_bps','risk_bps':30,'stop_atr_for_sizing':15},'exit':{'stop_atr':1},
            'legs':[{'ticker':r['Ticker'],'side':'LONG' if r['Action']=='BUY' else 'SHORT','multiplier':r.get('Multiplier',1)} for r in payload['orders']]}
    payload['source_sizing']=S.instruction(source,payload['orders'],product)
    payload['source_sizing_canonical']=C.canonical(payload['source_sizing'])
    payload['account_proposals']={a:{'account':a,'status':'requires_fresh_account_preview','sizing_hash':C.frozen(payload['source_sizing'])['hash']} for a in S.ACCOUNTS}
    e=C.frozen(payload);e['id']=f"{product}:{payload['source_idea_id']}:{e['hash'][:16]}"
    review={'id':str(uuid.uuid4()),'proposal_id':e['id'],'proposal_hash':e['hash'],
            'decision':'approve_review','scope':'human_review_only','execution':'not_submitted','actor':'qa-human','at':DAY+'T13:00:00Z'}
    return {'id':str(uuid.uuid4()),'type':'review_execution','account':account,'dry_run':True,
            'created_at':NOW.timestamp()*1000,'expires_at':(NOW.timestamp()+60)*1000,
            'payload':{'operation':'preview','product':product,'actor':'qa-human','proposal':e,'review':review,
                       'delivery_id':'qa-delivery','current_at_authorization':True}}


def config(path):
    return {'accounts':{'pitch':['primary','pa'],'seasonal':['primary','pa']},'risk_multipliers':{'primary':1,'pa':1},'preview_enabled':True,'live_enabled':True,
            'max_risk_bps':100,'db':str(path)}


class FakeIB:
    """No socket or IB library. All order/fill state lives in this object."""
    def __init__(self):
        self.rows={};self.calls=[];self.fail=None;self.crash=False;self.reject_preflight=None
        self.errorEvent=SimpleNamespace();self.connections=[];self.positions=[];self.nlvs={'primary':100000,'pa':10000}
    def reqAccountSummary(self):return []
    def sleep(self,value):pass
    def qualifyContracts(self,c):c.conId={'XLE':11,'TLT':22}.get(c.symbol,33);c.secType='STK';c.currency='USD';c.multiplier='';return [c]
    def reqHistoricalData(self,*args,**kwargs):return [SimpleNamespace(date=NOW.replace(hour=13,minute=30),open=102)]
    def capture(self,account,con_id):
        return {'account':account,'con_id':con_id,'at':NOW.timestamp(),'position':0,
                'orders':copy.deepcopy(self.rows.get(con_id,[])),'completed':[],'executions':[]}


def broker_rows(leg,*,filled=0):
    native=leg['payload'];pid=100+leg['con_id']*10;qty=native['quantity'];account=leg['broker_account']
    def row(oid,typ,action,parent=0,limit=0,stop=0,good_after='',status='Submitted',filled=0):
        return {'identity':[account,leg['con_id'],123,oid,10000+oid],'status':status,'filled':filled,
                'qty':qty,'action':action,'ref':leg['ref'],'parent':parent,'limit':limit,'stop':stop,
                'order_type':typ,'oca_group':f'OCA_{pid}' if parent else '', 'oca_type':1,
                'tif':('OPG' if native['entry_type']=='MOO' else 'GTD' if native['expiry'] else 'DAY') if not parent else 'GTC',
                'good_after':good_after,'good_till':(leg['timing']['parent_good_till'] or '') if not parent else '',
                'outside_rth':typ=='MKT' and bool(parent),'transmit':False}
    parent=row(pid,'LMT' if native['entry_type']=='LMT' else 'MKT' if native['entry_type']=='MOO' else 'MOC',native['action'],limit=native['entry'],status='Filled' if filled==qty else 'Submitted',filled=filled)
    children=[];action='SELL' if native['action']=='BUY' else 'BUY'
    if native['target'] is not None:children.append(row(pid+1,'LMT',action,pid,limit=native['target'],status='PreSubmitted'))
    if native['stop'] is not None:children.append(row(pid+2,'STP',action,pid,stop=native['stop'],good_after=leg['timing']['stop_good_after'] or '',status='PreSubmitted'))
    children.append(row(pid+3,'MKT',action,pid,good_after=leg['timing']['time_good_after'],status='PreSubmitted'))
    return [parent]+children


def fake_runtime(tmp_path):
    ib=FakeIB();account='DU_NONEXECUTING_QA'
    def entry(ib,p,logical):
        account='DU_NONEXECUTING_QA' if logical=='primary' else 'DU_NONEXECUTING_PA_QA'
        if p.get('_review_preflight_only'):
            if ib.reject_preflight==p['symbol']:return {'ok':False,'state':'rejected','detail':'fixture preflight rejected'}
            c=R.preflight_context(payload=p,broker_account=account,contract_id=11 if p['symbol']=='XLE' else 22,
                                 quantity=p['quantity'],entry=p['entry'],stop=p['stop'],target=p['target'],
                                 risk=abs(p['entry']-(p['stop'] or p['entry']-6))*p['quantity'],nlv=ib.nlvs[logical],
                                 stop_gat='20261007 09:30:00 America/New_York' if p['stop'] else None,
                                 time_gat='20261013 09:30:00 America/New_York',parent_gtd=None)
            return {'ok':True,'state':'executed','detail':'fake preflight','fill':{'review_preflight':c}}
        ib.calls.append(p['symbol'])
        if ib.fail==p['symbol']:return {'ok':False,'state':'rejected','detail':'fixture broker rejected','fill':None}
        context=R.preflight_context(payload=p,broker_account=account,contract_id=p['_review_expected_con_id'],
                                   quantity=p['quantity'],entry=p['entry'],stop=p['stop'],target=p['target'],risk=20,nlv=ib.nlvs[logical],
                                   stop_gat='20261007 09:30:00 America/New_York' if p['stop'] else None,
                                   time_gat='20261013 09:30:00 America/New_York',parent_gtd=None)
        leg={**context,'ref':R._reference(p)};rows=broker_rows(leg);ib.rows[leg['con_id']]=rows
        if ib.crash:raise RuntimeError('fixture loss AFTER broker received the chain')
        return {'ok':True,'state':'executed','detail':'broker acknowledged, not filled', 'fill':{'order_ids':[r['identity'][3] for r in rows]}}
    g={'_THIS_DIR':str(tmp_path),'LIVE_ENABLED':True,'LIVE_ACCOUNTS':{'primary','pa'},'LIVE_TYPES':{'entry_bracket','review_execution'},
       'LIVE_MAX_QTY':1000,'_max_notional':lambda acct:100000,'_resolve_broker_account':lambda ib,acct:account if acct=='primary' else 'DU_NONEXECUTING_PA_QA',
       '_do_entry_bracket':entry,'_review_capture':lambda ib,acct,cid:ib.capture(acct,cid),
       '_review_clock':lambda:NOW,'Stock':lambda sym,*args:SimpleNamespace(symbol=sym,conId=0)}
    g['_review_account_snapshot']=lambda ib,logical,broker:{'account':logical,'broker_account':broker,'currency':'USD',
        'source':'fresh_broker_account_summary','request_completed':True,'observed_at':NOW.isoformat(),
        'nlv':ib.nlvs[logical],'available_funds':50000,'buying_power':200000,'excess_liquidity':50000}
    g['_review_capacity']=lambda ib,c,p,broker:{'broker_account':broker,'con_id':c.conId,'quantity':p['quantity'],
        'permission_verified':True,'initial_margin_change':100,'maintenance_margin_change':100,'observed_at':NOW.isoformat()}
    return g,ib


def prepared(tmp_path,product='pitch',rows=None):
    path=tmp_path/'review.sqlite';journal=C.Journal.initialize(path);cfg=config(path);g,ib=fake_runtime(tmp_path)
    cmd=command(product,rows);plan=R.preflight(g,ib,cmd,cfg,NOW);journal.save_preview(plan)
    execute=copy.deepcopy(cmd);execute['id']=str(uuid.uuid4());execute['dry_run']=False
    execute['payload'].update(operation='execute',plan_hash=plan['hash'],confirmed=True,non_atomic_ack=True,risk_ack=True)
    return g,ib,cfg,journal,plan,execute


@pytest.mark.parametrize('product',['pitch','seasonal'])
def test_complete_preview_working_partial_fill_full_fill_and_exit(tmp_path,product):
    g,ib,cfg,journal,plan,cmd=prepared(tmp_path,product)
    assert not ib.calls
    record=R.execute_batch(g,ib,cmd,cfg,journal,NOW);assert record['state']=='working' and len(ib.calls)==1
    leg=plan['payload']['legs'][0];ib.rows[leg['con_id']][0]['filled']=3
    rec={'account':'primary','payload':{'product':product,'actor':'qa-human','run_key':record['key']}}
    assert R.reconcile_batch(g,ib,rec,journal)['state']=='partially_filled'
    ib.rows[leg['con_id']][0].update(status='Filled',filled=10)
    assert R.reconcile_batch(g,ib,rec,journal)['state']=='filled'
    ib.rows[leg['con_id']][1].update(status='Filled',filled=10)
    for row in ib.rows[leg['con_id']][2:]:row['status']='Cancelled'
    assert R.reconcile_batch(g,ib,rec,journal)['state']=='closed'
    assert len(ib.calls)==1


@pytest.mark.parametrize('field,value',[('Quantity',True),('Quantity',0),('Quantity',1.5),('Quantity',float('nan')),('Ticker','../../bad'),('Sec_Type','FUT'),('Proxy_Ticker','SPY'),('Manual_Only','TRUE'),('Trail_ATR',1),('Entry_Type','MKT'),('Time_Exit_Date',DAY),('Time_Exit_Order','MKT'),('Stop_Price',101)])
def test_ambiguous_or_unsupported_instruction_blocks_every_leg(field,value):
    with pytest.raises((ValueError,TypeError)):C.native_payload({**ROW,field:value},'pitch',DAY+'-1',DAY)


@pytest.mark.parametrize('field,value',[('confirmed',False),('plan_hash',''),('current_at_authorization',False),('actor','')])
def test_execution_needs_explicit_frozen_confirmation(tmp_path,field,value):
    g,ib,cfg,journal,plan,cmd=prepared(tmp_path);cmd['payload'][field]=value
    with pytest.raises(ValueError):R.execute_batch(g,ib,cmd,cfg,journal,NOW)
    assert not ib.calls


def test_unassigned_seasonal_account_never_defaults_to_primary():
    cmd=command('seasonal')
    with pytest.raises(ValueError,match='unassigned'):C.validate_request(cmd,{'pitch':['primary','pa'],'seasonal':[]},NOW)


def test_actor_account_hash_review_and_expiry_fail_closed(tmp_path):
    g,ib,cfg,journal,plan,cmd=prepared(tmp_path)
    changes=[lambda c:c.update(account='pa'),lambda c:c['payload'].update(actor='other'),
             lambda c:c['payload']['proposal']['payload']['orders'][0].update(Quantity=11),
             lambda c:c['payload']['review'].update(decision='reject'),lambda c:c.update(expires_at=NOW.timestamp()*1000)]
    for change in changes:
        bad=copy.deepcopy(cmd);change(bad)
        with pytest.raises(ValueError):R.execute_batch(g,ib,bad,cfg,journal,NOW)
    assert not ib.calls


def test_open_rule_is_exact_and_broker_open_is_required():
    row={**ROW,'Entry_Anchor':'OPEN','Entry_Offset_ATR':-0.5}
    with pytest.raises(ValueError):C.native_payload(row,'pitch',DAY+'-1',DAY)
    p=C.native_payload(row,'pitch',DAY+'-1',DAY,102)
    assert (p['entry'],p['stop'],p['target'])==(101,99,105)


def test_multi_leg_all_preflight_then_non_atomic_confirmation(tmp_path):
    rows=[ROW,{**ROW,'Leg':2,'Ticker':'TLT'}]
    g,ib,cfg,journal,plan,cmd=prepared(tmp_path,rows=rows)
    cmd['payload']['non_atomic_ack']=False
    with pytest.raises(ValueError,match='non-atomic'):R.execute_batch(g,ib,cmd,cfg,journal,NOW)
    cmd['payload']['non_atomic_ack']=True;ib.reject_preflight='TLT'
    with pytest.raises(ValueError,match='preflight rejected'):R.execute_batch(g,ib,cmd,cfg,journal,NOW)
    assert not ib.calls
    ib.reject_preflight=None
    assert R.execute_batch(g,ib,cmd,cfg,journal,NOW)['state']=='working'
    assert ib.calls==['XLE','TLT']


def test_partial_idea_rejection_and_retry_do_not_place_remaining_legs(tmp_path):
    g,ib,cfg,journal,plan,cmd=prepared(tmp_path,rows=[ROW,{**ROW,'Leg':2,'Ticker':'TLT'}]);ib.fail='TLT'
    result=R.execute_batch(g,ib,cmd,cfg,journal,NOW)
    assert result['state']=='needs_reconciliation' and ib.calls==['XLE','TLT']
    ib.fail=None;assert R.execute_batch(g,ib,cmd,cfg,journal,NOW)['state']=='needs_reconciliation'
    assert ib.calls==['XLE','TLT']


def test_crash_after_broker_delivery_is_reconciled_without_resuming(tmp_path):
    g,ib,cfg,journal,plan,cmd=prepared(tmp_path,rows=[ROW,{**ROW,'Leg':2,'Ticker':'TLT'}]);ib.crash=True
    with pytest.raises(RuntimeError):R.execute_batch(g,ib,cmd,cfg,journal,NOW)
    key=C.run_key('pitch',DAY+'-1','primary',plan['payload']['proposal_hash']);saved=C.Journal(journal.path).get(key)
    assert saved['legs'][0]['state']=='submitting' and saved['legs'][1]['state']=='not_sent'
    ib.crash=False;assert R.execute_batch(g,ib,cmd,cfg,journal,NOW)['state']=='needs_reconciliation'
    result=R.reconcile_batch(g,ib,{'account':'primary','payload':{'product':'pitch','actor':'qa-human','run_key':key}},journal)
    assert result['legs'][0]['state']=='working' and result['legs'][1]['state']=='not_sent' and ib.calls==['XLE']


def test_permanent_claim_rejects_new_uuid_and_new_proposal_version(tmp_path):
    g,ib,cfg,journal,plan,cmd=prepared(tmp_path);R.execute_batch(g,ib,cmd,cfg,journal,NOW)
    new=copy.deepcopy(cmd);new['id']=str(uuid.uuid4())
    with pytest.raises(ValueError,match='permanent'):R.execute_batch(g,ib,new,cfg,journal,NOW)
    assert len(ib.calls)==1


def test_changed_contract_account_prices_or_risk_cannot_use_old_preview(tmp_path):
    g,ib,cfg,journal,plan,cmd=prepared(tmp_path)
    g['_resolve_broker_account']=lambda ib,account:'DU_OTHER_QA'
    with pytest.raises(ValueError):R.execute_batch(g,ib,cmd,cfg,journal,NOW)
    assert not ib.calls


def test_whole_idea_aggregate_risk_cap(tmp_path):
    g,ib=fake_runtime(tmp_path);cfg=config(tmp_path/'fake');cfg['max_risk_bps']=3
    with pytest.raises(ValueError,match='whole-idea.*risk'):R.preflight(g,ib,command(rows=[ROW,{**ROW,'Leg':2,'Ticker':'TLT'}]),cfg,NOW)
    assert not ib.calls


def test_missing_corrupt_journal_and_nested_operation_lock(tmp_path):
    with pytest.raises(ValueError,match='missing'):C.Journal(tmp_path/'missing').get('x')
    path=tmp_path/'corrupt';path.write_text('not sqlite')
    with pytest.raises(Exception):C.Journal(path).get('x')
    journal=C.Journal.initialize(tmp_path/'journal')
    with journal.operation():
        with pytest.raises((OSError,BlockingIOError)): 
            with C.Journal(journal.path).operation():pass


def test_short_dedup_checks_legacy_and_site_order_tags(tmp_path):
    g,ib=fake_runtime(tmp_path);row={**ROW,'Action':'SELL_SHORT','Stop_Price':102,'Target_Price':96}
    ref='XLE|SELL|Pitch-'+DAY+'-1|'+DAY
    ib.rows[11]=[{'ref':ref}]
    with pytest.raises(ValueError,match='already exists'):R.preflight(g,ib,command(rows=[row]),config(tmp_path/'x'),NOW)


def test_adapter_gates_never_arm_legacy_paths(tmp_path):
    g,ib=fake_runtime(tmp_path);cmd=command();cfg=config(tmp_path/'x');cfg['preview_enabled']=False
    with pytest.raises(ValueError,match='disabled'):R.gate(g,cmd,cfg)
    cfg['preview_enabled']=True;(tmp_path/'pitch_moo_enabled.flag').write_text('QA only')
    with pytest.raises(ValueError,match='legacy'):R.gate(g,cmd,cfg)
    (tmp_path/'pitch_moo_enabled.flag').unlink()
    cmd['payload']['operation']='execute';cmd['dry_run']=False
    cfg['live_enabled']=False
    with pytest.raises(ValueError,match='disabled'):R.gate(g,cmd,cfg)
    cfg['live_enabled']=True;g['LIVE_TYPES']={'entry_bracket'}
    with pytest.raises(ValueError,match='type gates'):R.gate(g,cmd,cfg)


@pytest.mark.parametrize('mutation',[lambda r:r[0].update(tif='OPG'),lambda r:r[1].update(limit=999),lambda r:r[1].update(tif='DAY'),lambda r:r[2].update(good_after='wrong'),lambda r:r[3].update(outside_rth=False),lambda r:r.pop(),lambda r:r[0]['identity'].__setitem__(0,'DU_OTHER')])
def test_acknowledgment_requires_entire_exact_bracket(tmp_path,mutation):
    g,ib,cfg,journal,plan,cmd=prepared(tmp_path);leg=plan['payload']['legs'][0]
    ib.rows[leg['con_id']]=broker_rows(leg);mutation(ib.rows[leg['con_id']])
    assert R.evidence_leg(g,ib,plan['payload'],leg,{})['state'] in {'unknown','unprotected'}


def test_old_broker_fills_are_not_inferred_from_missing_evidence(tmp_path):
    g,ib,cfg,journal,plan,cmd=prepared(tmp_path);leg=plan['payload']['legs'][0]
    ib.rows[leg['con_id']]=broker_rows(leg);ib.rows[leg['con_id']][0].update(status='Cancelled',filled=None)
    assert R.evidence_leg(g,ib,plan['payload'],leg,{})['state']=='unknown'


def test_real_executor_json_exit_protocol_is_captured():
    def old_executor(*args):print(json.dumps({'ok':True,'state':'executed','detail':'fixture','fill':{'x':1}}));return 0
    assert R._invoke_entry({'_do_entry_bracket':old_executor},None,{},'primary')['fill']=={'x':1}


def test_preparer_is_hash_pinned_and_never_installs(tmp_path):
    source=Path(r'C:\Users\McKinley Slade\OneDrive\trading_ibkr')
    if not source.exists():pytest.skip('reviewed external source not present; portable contract/lifecycle tests still run')
    output=tmp_path/'candidate';result=P.prepare(source,output)
    assert not result['installed'] and not result['armed']
    assert set(result['candidate_hashes'])=={'exec_agent.py','execute_order.py','broker_reconciliation.py','review_execution.py','review_execution_runtime.py','review_sizing.py'}
    for name in result['candidate_hashes']:compile((output/name).read_bytes(),name,'exec')
    with pytest.raises(ValueError):P.prepare(source,source/'should-never-write')


@pytest.mark.parametrize('product',['pitch','seasonal'])
def test_actual_patched_executor_and_native_bracket_with_fake_broker(tmp_path,monkeypatch,product):
    """Execute only inspected AST functions, never a production module import."""
    source=Path(r'C:\Users\McKinley Slade\OneDrive\trading_ibkr\execute_order.py')
    if not source.exists():pytest.skip('external executor absent; portable lifecycle tests run above')
    original=source.read_text(encoding='utf-8-sig')
    patched=P.patch_executor(original)
    tree=ast.parse(patched)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'build_bracket','_do_entry_bracket','main'}]
    assert len(nodes)==3
    def order(kind,action,qty,price=None):
        return SimpleNamespace(orderType=kind,action=action,totalQuantity=qty,lmtPrice=price if kind=='LMT' else 0,
                               auxPrice=price if kind=='STP' else 0,tif='',parentId=0,goodAfterTime='',goodTillDate='',
                               outsideRth=False,account='',orderRef='',ocaGroup='',ocaType=0,permId=0)
    counter=itertools.count(100)
    g,ib=fake_runtime(tmp_path)
    ib.client=SimpleNamespace(getReqId=lambda:next(counter))
    placements=[]
    def place(ib,contract,order,**kwargs):
        placements.append((order,kwargs));order.permId=10000+order.orderId
        return SimpleNamespace(order=order,orderStatus=SimpleNamespace(status='Submitted'))
    g.update(math=math,json=json,sys=SimpleNamespace(argv=[]),
       _out=lambda ok,state,detail,fill=None:{'ok':ok,'state':state,'detail':detail,'fill':fill},
       build_order_ref=lambda sym,action,strategy,date:(f'{sym}|{action}|{strategy}|{date}',None),
       LimitOrder=lambda a,q,p:order('LMT',a,q,p),StopOrder=lambda a,q,p:order('STP',a,q,p),
       MarketOrder=lambda a,q:order('MKT',a,q),
       _nlv=lambda ib,account:100000,_atr_estimate=lambda *args:(2,100),RISK_ACK_BPS=50,
       _exit_timing=lambda p:('20261007 09:30:00 America/New_York','09:30:00'),
       _stop_arm_problem=lambda *args:None,
       _time_exit_deadline=lambda *args:'20261013 09:30:00 America/New_York',
       _execution_deadline=lambda *args:'20261008 16:00:00 America/New_York',
       guarded_place_order=place,_placement_problem=lambda *args,**kwargs:None,
       _command_signal=lambda p,suffix:p['_command_id']+'|'+suffix)
    monkeypatch.setitem(sys.modules,'review_execution_runtime',R)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'isolated_executor_ast','exec'),g)
    native=C.native_payload(ROW,product,DAY+'-1',DAY)
    payload=dict(native,_broker_account='DU_NONEXECUTING_QA',_command_id='qa',_review_preflight_only=True)
    # First exercise real main -> adapter -> all-leg preflight -> native entry
    # with fake broker dependencies. No adapter stub or context injection.
    class Event:
        def __iadd__(self,handler):return self
    ib.errorEvent=Event()
    ib.connect=lambda *args,**kwargs:ib.connections.append(kwargs)
    ib.disconnect=lambda:None
    db=tmp_path/'real-main-qa.sqlite';C.Journal.initialize(db)
    monkeypatch.setattr(R,'configuration',lambda:config(db))
    g.update(IB=lambda:ib,PORTS={'primary':('mock-local-only',0,1)},_on_err=lambda *args:None)
    real_command=command(product)
    g['sys'].argv=['offline-executor',json.dumps(real_command)]
    assert '_COMMAND_TYPE' not in g
    preview=g['main']()
    assert preview['ok'] and preview['fill']['review_execution']['state']=='preview'
    assert ib.connections==[{'clientId':1,'timeout':8,'readonly':True}]
    assert not placements
    assert preview['fill']['review_execution']['plan']['payload']['legs'][0]['ref']==R._reference(native)
    g.pop('_COMMAND_TYPE')
    # Then check native bracket placement on fake orders through the same main.
    invoked=[]
    def adapter(env,cmd):
        invoked.append(env.get('_COMMAND_TYPE'))
        return env['_do_entry_bracket'](ib,cmd['payload'],'primary')
    monkeypatch.setattr(R,'run_executor',adapter)
    def through_main(payload):
        g['sys'].argv=['offline-executor',json.dumps({'type':'review_execution','payload':payload})]
        return g['main']()
    assert '_COMMAND_TYPE' not in g
    context=through_main(payload)
    assert invoked==['review_execution']
    assert context['ok'] and context['fill']['review_preflight']['risk_usd']==20
    assert not placements
    payload.pop('_review_preflight_only');payload['_review_expected_con_id']=11
    result=through_main(payload)
    assert result['ok'] and len(placements)==4
    parent=placements[0][0];children=[x[0] for x in placements[1:]]
    assert parent.tif=='DAY' and parent.lmtPrice==100 and not parent.transmit
    assert parent.orderRef==R._reference(native) and all(o.orderRef==parent.orderRef for o in children)
    assert [o.orderType for o in children]==['LMT','STP','MKT']
    assert [o.transmit for o in children]==[False,False,True]
    assert all(o.parentId==parent.orderId and o.totalQuantity==10 and o.account=='DU_NONEXECUTING_QA' for o in children)
    assert children[0].lmtPrice==104 and children[1].auxPrice==98
    assert children[1].goodAfterTime.startswith('20261007') and children[2].goodAfterTime.startswith('20261013')
    assert all(o.ocaGroup==children[0].ocaGroup for o in children)
    assert placements[0][1]['mutation_kind']=='entry' and all(v['mutation_kind']=='protective' for _,v in placements[1:])
    placements.clear();payload['_review_expected_con_id']=999
    assert not through_main(payload)['ok'] and not placements


def test_configuration_defaults_are_disabled_with_both_accounts_and_agent_parity(monkeypatch):
    for key in ('REVIEW_EXECUTION_PREVIEW_ENABLED','REVIEW_EXECUTION_LIVE_ENABLED','REVIEW_EXECUTION_SEASONAL_ACCOUNT','REVIEW_EXECUTION_PITCH_ACCOUNT','REVIEW_EXECUTION_DB','REVIEW_EXECUTION_PA_RISK_MULTIPLIER'):
        monkeypatch.delenv(key,raising=False)
    cfg=R.configuration()
    assert not cfg['preview_enabled'] and not cfg['live_enabled']
    assert cfg['accounts']=={'pitch':['primary','pa'],'seasonal':['primary','pa']} and cfg['db'] is None
    assert cfg['risk_multipliers']=={'primary':1,'pa':1}


def test_executor_connects_readonly_for_preview_and_reconciliation(tmp_path,monkeypatch):
    g,ib=fake_runtime(tmp_path);path=tmp_path/'journal';C.Journal.initialize(path);cfg=config(path)
    monkeypatch.setattr(R,'configuration',lambda:cfg)
    class Event:
        def __iadd__(self,handler):return self
    ib.errorEvent=Event();ib.connect=lambda *args,**kwargs:ib.connections.append(kwargs)
    ib.disconnect=lambda:None
    g.update(IB=lambda:ib,PORTS={'primary':('mock-local-only',0,1)},_on_err=lambda *args:None,
             _out=lambda ok,state,detail,fill=None:{'ok':ok,'state':state,'detail':detail,'fill':fill})
    preview_command=command()
    preview=R.run_executor(g,preview_command)
    assert preview['ok'] and ib.connections[-1]['readonly'] is True and not ib.calls
    plan=preview['fill']['review_execution']['plan']
    execute=copy.deepcopy(preview_command);execute['id']=str(uuid.uuid4());execute['dry_run']=False
    execute['payload'].update(operation='execute',confirmed=True,plan_hash=plan['hash'])
    result=R.run_executor(g,execute)
    assert result['ok'] and ib.connections[-1]['readonly'] is False and ib.calls==['XLE']
    reconcile={'id':str(uuid.uuid4()),'type':'review_execution','account':'primary','dry_run':True,
               'expires_at':(NOW.timestamp()+60)*1000,
               'payload':{'operation':'reconcile','product':'pitch','actor':'qa-human','run_key':C.run_key('pitch',DAY+'-1','primary',plan['payload']['proposal_hash'])}}
    assert R.run_executor(g,reconcile)['ok'] and ib.connections[-1]['readonly'] is True
    assert ib.calls==['XLE']
    cfg['live_enabled']=False
    assert not R.run_executor(g,execute)['ok'] and len(ib.connections)==3


def test_patched_agent_verifies_signature_before_review_adapter(tmp_path,monkeypatch):
    source=Path(r'C:\Users\McKinley Slade\OneDrive\trading_ibkr\exec_agent.py')
    if not source.exists():pytest.skip('external executor absent')
    tree=ast.parse(P.patch_agent(source.read_text(encoding='utf-8-sig')))
    node=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='_handle_command')
    import asyncio,time
    called=[]
    async def adapter(g,cmd):called.append(cmd);return {'ok':True,'state':'preview','detail':'mock only'}
    monkeypatch.setattr(R,'handle_agent',adapter);monkeypatch.setitem(sys.modules,'review_execution_runtime',R)
    g={'time':time,'json':json,'log':lambda *args:None,'_verify':lambda signed,sig:sig=='valid'}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'isolated_agent_ast','exec'),g)
    class Socket:
        def __init__(self):self.messages=[]
        async def send(self,text):self.messages.append(json.loads(text))
    ws=Socket();cmd=command();cmd['expires_at']=(time.time()+60)*1000
    asyncio.run(g['_handle_command'](ws,json.dumps(cmd),'invalid'))
    assert not called and ws.messages[-1]['state']=='rejected'
    asyncio.run(g['_handle_command'](ws,json.dumps(cmd),'valid'))
    assert called==[cmd] and ws.messages[-1]['state']=='preview'
