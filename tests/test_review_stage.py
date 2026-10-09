"""Single approval -> automatic sizing/staging. Fake brokers only, no sockets."""
import copy
import asyncio
import datetime as dt
import json
import uuid

import pytest

from broker_runtime import review_execution as C, review_execution_runtime as R
from tests.test_agent_review_execution import command, config, fake_runtime, NOW, ROW


def stage_command(product='pitch', account='primary', rows=None):
    cmd=command(product, rows, account)
    source=copy.deepcopy(cmd['payload']['proposal']['payload'])
    source['execution_accounts']=['primary','pa']
    envelope=C.frozen(source)
    envelope['id']=f"{product}:{source['source_idea_id']}:{envelope['hash'][:16]}"
    cmd['payload']['proposal']=envelope
    cmd['payload']['review'].update(proposal_id=envelope['id'],proposal_hash=envelope['hash'],
        scope='review_and_stage',execution='queued',accounts=['primary','pa'])
    cmd['payload']['operation']='stage'
    cmd['dry_run']=False
    return cmd


@pytest.mark.parametrize('product',['pitch','seasonal'])
@pytest.mark.parametrize('account,nlv,expected',[('primary',200000,20),('pa',20000,2)])
def test_one_approval_sizes_and_stages_without_a_browser_preview(tmp_path,product,account,nlv,expected):
    g,ib=fake_runtime(tmp_path);ib.nlvs[account]=nlv
    journal=C.Journal.initialize(tmp_path/'journal');cfg=config(journal.path)
    cmd=stage_command(product,account)
    result=R.execute_batch(g,ib,cmd,cfg,journal,NOW)
    assert result['state']=='working' and ib.calls==['XLE']
    plan=result['plan']['payload']
    assert plan['account']==account and plan['legs'][0]['payload']['quantity']==expected
    assert plan['sizing']['account_multiplier']==1
    assert 'plan_hash' not in cmd['payload'] and 'confirmed' not in cmd['payload']
    assert R.execute_batch(g,ib,cmd,cfg,journal,NOW)==result
    assert ib.calls==['XLE']


@pytest.mark.parametrize('change',[
    lambda c:c['payload']['review'].update(scope='human_review_only',execution='not_submitted'),
    lambda c:c['payload']['review'].update(accounts=['primary']),
    lambda c:c['payload']['review'].update(actor='other'),
    lambda c:c['payload']['review'].update(decision='reject'),
    lambda c:c.update(dry_run=True),
])
def test_old_review_or_incomplete_authorization_cannot_stage(tmp_path,change):
    g,ib=fake_runtime(tmp_path);journal=C.Journal.initialize(tmp_path/'journal')
    cmd=stage_command();change(cmd)
    with pytest.raises(ValueError):R.execute_batch(g,ib,cmd,config(journal.path),journal,NOW)
    assert not ib.calls


def test_every_leg_is_checked_before_any_submission(tmp_path):
    g,ib=fake_runtime(tmp_path);ib.reject_preflight='TLT'
    journal=C.Journal.initialize(tmp_path/'journal')
    cmd=stage_command(rows=[ROW,{**ROW,'Ticker':'TLT','Leg':2}])
    with pytest.raises(ValueError,match='preflight rejected'):
        R.execute_batch(g,ib,cmd,config(journal.path),journal,NOW)
    assert not ib.calls


def test_broker_failure_never_retries_partial_idea(tmp_path):
    g,ib=fake_runtime(tmp_path);ib.fail='TLT'
    journal=C.Journal.initialize(tmp_path/'journal')
    cmd=stage_command(rows=[ROW,{**ROW,'Ticker':'TLT','Leg':2}])
    first=R.execute_batch(g,ib,cmd,config(journal.path),journal,NOW)
    assert first['state']=='needs_reconciliation' and ib.calls==['XLE','TLT']
    ib.fail=None
    assert R.execute_batch(g,ib,cmd,config(journal.path),journal,NOW)==first
    assert ib.calls==['XLE','TLT']
    other=copy.deepcopy(cmd);other['id']=str(uuid.uuid4())
    with pytest.raises(ValueError,match='permanent'):R.execute_batch(g,ib,other,config(journal.path),journal,NOW)
    assert ib.calls==['XLE','TLT']


def test_both_account_jobs_execute_independently_under_one_approval(tmp_path):
    journal=C.Journal.initialize(tmp_path/'journal');cfg=config(journal.path)
    primary=stage_command();pa=copy.deepcopy(primary)
    pa.update(account='pa',id=str(uuid.uuid4()))
    event=primary['payload']['review']
    for cmd in [primary,pa]:
        g,ib=fake_runtime(tmp_path)
        assert cmd['payload']['review']==event
        result=R.execute_batch(g,ib,cmd,cfg,journal,NOW)
        assert result['state']=='working' and ib.calls==['XLE']
    assert primary['payload']['proposal']['payload']['orders'][0]['Quantity']==10


def test_single_approval_uses_existing_live_gates(tmp_path,monkeypatch):
    g,ib=fake_runtime(tmp_path);journal=C.Journal.initialize(tmp_path/'journal');cfg=config(journal.path)
    monkeypatch.setattr(R,'configuration',lambda:cfg)
    cmd=stage_command();cfg['live_enabled']=False
    with pytest.raises(ValueError,match='live adapter disabled'):R.gate(g,cmd,cfg)
    cfg['live_enabled']=True;g['LIVE_ENABLED']=False
    with pytest.raises(ValueError,match='live gate'):R.gate(g,cmd,cfg)
    assert not ib.calls


def test_runtime_connects_and_submits_directly_for_single_approval(tmp_path,monkeypatch):
    g,ib=fake_runtime(tmp_path);journal=C.Journal.initialize(tmp_path/'journal');cfg=config(journal.path)
    monkeypatch.setattr(R,'configuration',lambda:cfg)
    class Event:
        def __iadd__(self,fn):return self
    ib.errorEvent=Event();ib.connect=lambda *args,**kwargs:ib.connections.append(kwargs)
    ib.disconnect=lambda:None
    g.update(IB=lambda:ib,PORTS={'primary':('fake-no-network',0,1)},_on_err=lambda *args:None,
        _out=lambda ok,state,detail,fill=None:dict(ok=ok,state=state,detail=detail,fill=fill))
    result=R.run_executor(g,stage_command())
    assert result['ok'] and result['fill']['review_execution']['state']=='working'
    assert ib.connections==[{'clientId':1,'timeout':8,'readonly':False}]
    assert ib.calls==['XLE']


def test_moc_source_stays_an_auction_order(tmp_path):
    g,ib=fake_runtime(tmp_path);journal=C.Journal.initialize(tmp_path/'journal')
    row={**ROW,'Entry_Type':'MOC','Order_Type':'MKT','TIF':'MOC','Entry_Anchor':'',
         'Stop_Price':'','Stop_ATR':'','Target_Price':'','Target_ATR':''}
    cmd=stage_command('seasonal',rows=[row])
    result=R.execute_batch(g,ib,cmd,config(journal.path),journal,NOW)
    assert result['state']=='working'
    assert result['plan']['payload']['legs'][0]['payload']['entry_type']=='MOC'


def test_open_limit_is_queued_premarket_and_runs_after_restart_without_a_second_click(tmp_path,monkeypatch):
    g,ib=fake_runtime(tmp_path);journal=C.Journal.initialize(tmp_path/'journal');cfg=config(journal.path)
    monkeypatch.setattr(R,'configuration',lambda:cfg)
    early=NOW.replace(hour=10);g['_review_clock']=lambda:early
    cmd=stage_command(rows=[{**ROW,'Entry_Anchor':'OPEN'}])
    cmd['payload']['review']['at']=early.isoformat()
    invoked=[]
    async def execute(command):
        invoked.append(command['id'])
        record=R.execute_batch(g,ib,command,cfg,C.Journal(journal.path),g['_review_clock']())
        return {'ok':True,'detail':'fake staging','fill':{'review_execution':{'state':record['state'],'record':record}}}
    g['_execute_live']=execute
    first=asyncio.run(R.handle_agent(g,cmd))
    assert first['state']=='scheduled' and not invoked and not ib.calls
    # Simulate a new process reading the same persistent journal.
    restarted=C.Journal(journal.path)
    assert not restarted.due_staging(early)
    g['_review_clock']=lambda:NOW
    assert restarted.due_staging(g['_review_clock']())[0]==cmd
    class Socket:
        def __init__(self):self.messages=[]
        async def send(self,value):self.messages.append(json.loads(value))
    ws=Socket()
    asyncio.run(R.process_due_staging(g,ws))
    assert invoked==[cmd['id']] and ib.calls==['XLE']
    assert ws.messages[-1]['state']=='working' and ws.messages[-1]['id']==cmd['id']
    asyncio.run(R.process_due_staging(g,ws))
    assert invoked==[cmd['id']] and len(ws.messages)==2  # retry result until persisted ack
    assert ws.messages[0]['review_receipt']==ws.messages[1]['review_receipt']
    R.acknowledge_staging({'of':'review_result','id':cmd['id'],'receipt':'wrong'})
    assert restarted.unreported_staging()
    R.acknowledge_staging({'of':'review_result','id':cmd['id'],'receipt':ws.messages[-1]['review_receipt']})
    asyncio.run(R.process_due_staging(g,ws))
    assert invoked==[cmd['id']] and len(ws.messages)==2


def test_scheduled_result_survives_websocket_loss_without_resubmission(tmp_path):
    journal=C.Journal.initialize(tmp_path/'journal');cmd=stage_command()
    journal.schedule(cmd,NOW);assert journal.start_scheduled(cmd['id'])
    result={'ok':True,'state':'working','detail':'fixture'}
    journal.finish_scheduled(cmd['id'],result)
    restored=C.Journal(journal.path)
    assert restored.unreported_staging()==[(cmd['id'],result)]
    assert not restored.due_staging(NOW+dt.timedelta(minutes=1))
    restored.reported_staging(cmd['id'])
    assert not restored.unreported_staging()


def test_restart_during_scheduled_submission_reports_unknown_and_never_retries(tmp_path):
    journal=C.Journal.initialize(tmp_path/'journal');cmd=stage_command()
    journal.schedule(cmd,NOW);assert journal.start_scheduled(cmd['id'])
    restarted=C.Journal(journal.path);restarted.interrupt_staging()
    assert not restarted.due_staging(NOW)
    assert restarted.unreported_staging()[0][1]['state']=='unknown'


def test_missed_open_approval_expires_instead_of_running_next_day(tmp_path,monkeypatch):
    g,ib=fake_runtime(tmp_path);journal=C.Journal.initialize(tmp_path/'journal');cfg=config(journal.path)
    monkeypatch.setattr(R,'configuration',lambda:cfg)
    cmd=stage_command(rows=[{**ROW,'Entry_Anchor':'OPEN'}]);journal.schedule(cmd,NOW)
    g['_review_clock']=lambda:NOW+dt.timedelta(days=1)
    class Socket:
        async def send(self,value):self.result=json.loads(value)
    ws=Socket();asyncio.run(R.process_due_staging(g,ws))
    assert ws.result['state']=='rejected' and not ib.calls


def test_prior_journal_migrates_without_losing_permanent_claims(tmp_path):
    import sqlite3
    journal=C.Journal.initialize(tmp_path/'journal')
    with sqlite3.connect(journal.path) as db:
        db.execute('DROP TABLE scheduled')  # only this test-created database
        db.execute('UPDATE meta SET schema_version=1')
        db.execute('INSERT INTO runs VALUES (?,?,?)',('fixture-key','fixture-id','{"fixture":true}'))
    restored=C.Journal(journal.path)
    assert restored.get('fixture-key')=={'fixture':True}
    assert restored.unreported_staging()==[]
    with sqlite3.connect(journal.path) as db:
        assert db.execute('SELECT schema_version FROM meta').fetchall()==[(2,)]
