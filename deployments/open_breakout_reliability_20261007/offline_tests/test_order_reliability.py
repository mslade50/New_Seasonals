"""Deterministic offline reliability cases. No native owner or broker connection."""
import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta, timezone
import itertools

import pytest
from test_existing_open_breakout import config, setup_service, tick, at, manifest
from open_breakout.order_policy import validate_policy
from open_breakout.order_reliability import Lifecycle, body_hash
from open_breakout.replay import SimBroker
from open_breakout.service import Service
from open_breakout.store import Store


def policy():
    # Test inputs only. This is explicitly not a paper qualification.
    return dict(schema=1,scope='offline_fixture',qualification_id='offline-Oct6-Oct7',
                source_sha256='0'*64,soft_ack_seconds=3.,hard_ack_seconds=12.,
                cancel_seconds=15.,snapshot_seconds=5.,protection_ack_seconds=3.,observed_protection_ack_max_seconds=2.,observed_ack_max_seconds=8.,approved=True)


def enabled(c):
    return replace(c,entry_order_type='stop_limit',allow_price_only_pause=True,
                   order_reliability_enabled=True,order_reliability_policy=policy())


def body(side=1):
    return dict(kind='STP LMT',side=side,qty=8,stop=7855.5 if side==1 else 7821.,
                limit=7856. if side==1 else 7820.5,tif='GTD',oca='OB-ES-1',oca_type=1,
                account='DU_TEST',ref=f'MES|BUY|OpenBreakout|2026-10-07|ES-1-ENTRY')


@pytest.fixture
def ledger(tmp_path):
    now=[at('09:30:01').astimezone(timezone.utc)]
    store=Store(tmp_path/'ledger.sqlite','fixture','2026-10-07',lock=False)
    life=Lifecycle(store,'DU_TEST',927481,'2026-10-07',lambda:now[0])
    life.intent(15663,'ES','ENTRY',815824257,body(),371.8);life.send_started(15663)
    yield life,store,now
    store.close()


def event(life,kind='orderStatus',status='Submitted',**changes):
    r=life.rows()['15663']
    return dict(source='ibkr_wire',kind=kind,id=15663,account='DU_TEST',client_id=927481,
                session='2026-10-07',con_id=815824257,ref=r['ref'],perm_id=1323944217,
                received_at=life.clock().isoformat(),status=status,
                filled=0,remaining=8,body_sha256=r['body_sha256'],**changes)


def echo(life):
    life.native(event(life,'openOrder'));life.native(event(life))


def snapshot(life,now,orders=True):
    r=life.rows()['15663'];b=r['body']
    return dict(execution_health=dict(source='ibkr_responses',account='DU_TEST',client_id=927481,
                session='2026-10-07',execution_since='2026-10-07',started_at=now[0].isoformat(),
                received_at=now[0].isoformat(),roundtrip_seconds=0.,connected=True,healthy=True,
                fresh_orders=True,fresh_positions=True,fresh_executions=True,fresh_completed=True,
                callback_stable=True),own={},own_exec_ids=[],positions={},
                orders=[dict(id=15663,account='DU_TEST',client_id=927481,con_id=815824257,
                ref=r['ref'],perm_id=1323944217,remaining=8,**{k:b[k] for k in
                ('kind','side','tif','oca','oca_type','stop','limit')})] if orders else [])


@pytest.mark.parametrize('field,value',[('scope','offline_fixture'),('approved',False),
    ('hard_ack_seconds',3.),('hard_ack_seconds',61.),('hard_ack_seconds',7.),
    ('cancel_seconds',31.),('snapshot_seconds',31.),('soft_ack_seconds',0.),
    ('observed_ack_max_seconds',float('nan')),('source_sha256','x')])
def test_live_policy_rejects_unqualified_or_unbounded(field,value):
    p=policy();p['scope']='paper_qualified';p['paper_evidence_sha256']='0'*64;p[field]=value
    with pytest.raises(ValueError):validate_policy(p,'live')


def test_disabled_policy_has_no_invented_live_grace():
    with pytest.raises(ValueError):validate_policy(None,'live')
    assert validate_policy(policy(),'shadow')['scope']=='offline_fixture'


@pytest.mark.parametrize('sequence',list(itertools.permutations(('PendingSubmit','Submitted','Submitted','PendingSubmit'))))
def test_duplicate_reordered_status_requires_echo_and_cancel_never_rearms(ledger,sequence):
    life,store,now=ledger
    for status in sequence:life.native(event(life,status=status))
    assert not life.accepted(life.rows()['15663'])
    life.native(event(life,'openOrder'))
    assert life.cancel_started(15663,'Oct7-first-cause')
    for status in sequence:life.native(event(life,status=status))
    row=life.rows()['15663']
    assert row['cancel_started'] and row['cancel_cause']=='Oct7-first-cause'
    assert not life.accepted(row) and not life.cancel_started(15663,'duplicate')
    assert row['risk']==371.8


@pytest.mark.parametrize('field,value',[('account','other'),('client_id',1),('session','2026-10-08'),
    ('con_id',1),('ref','other'),('perm_id',0),('source','sdk_event')])
def test_wrong_native_identity_is_rejected(ledger,field,value):
    life,_,_=ledger;e=event(life);e[field]=value
    with pytest.raises(ValueError):life.native(e)
    assert not life.accepted(life.rows()['15663'])


@pytest.mark.parametrize('missing',['fresh_orders','fresh_positions','fresh_executions','fresh_completed','callback_stable'])
def test_partial_response_cycle_cannot_clear_account_block(ledger,missing):
    life,_,now=ledger;echo(life);s=snapshot(life,now);s['execution_health'][missing]=False
    assert not life.reconcile(s,life.revision,{815824257:0},[])[0]


def test_snapshot_omission_never_proves_flat_even_after_cancel(ledger):
    life,_,now=ledger;echo(life);life.cancel_started(15663,'cause')
    assert not life.reconcile(snapshot(life,now,False),life.revision,{815824257:0},[])[0]
    assert not life.rows()['15663']['closed']
    life.native(event(life,status='Cancelled'))
    now[0]+=timedelta(microseconds=1)
    assert life.reconcile(snapshot(life,now,False),life.revision,{815824257:0},[])[0]
    assert life.rows()['15663']['closed']
    life.native(event(life,status='Submitted'))
    assert life.rows()['15663']['closed'] and life.rows()['15663']['cancel_started']


@pytest.mark.parametrize('crash',['after-intent','after-send-fence','after-cancel-fence'])
def test_crash_restart_never_authorizes_send_or_cancel_retransmit(ledger,crash):
    life,store,_=ledger
    if crash=='after-intent':
        life.intent(15664,'ES','ENTRY',815824257,body(-1),371.8)
    if crash=='after-cancel-fence':life.cancel_started(15663,'first')
    restarted=Lifecycle(store,'DU_TEST',927481,'2026-10-07',life.clock)
    with pytest.raises(ValueError):restarted.send_started(15663)
    if crash=='after-cancel-fence':assert not restarted.cancel_started(15663,'second')
    assert restarted.rows()['15663']['risk']==371.8


def test_fill_generation_and_execution_identity_fence(ledger):
    life,_,now=ledger;echo(life);s=snapshot(life,now);fence=life.revision
    assert life.fill(15663,'A.B.C.01',2,7855.5)
    assert not life.fill(15663,'A.B.C.01',2,7855.5)
    assert not life.reconcile(s,fence,{815824257:2},['A.B.C'])[0]
    with pytest.raises(ValueError):life.fill(15663,'A.B.C.02',2,7855.5)
    assert life.rows()['15663']['filled']==2


def test_partial_protector_revision_requires_current_echo(ledger):
    life,_,_=ledger;echo(life)
    old=event(life,'openOrder')
    new={**body(),'qty':10,'stop':7856.}
    life.modify_started(15663,new)
    life.native(old);life.native(event(life))
    assert not life.accepted(life.rows()['15663'])
    life.native(event(life,'openOrder'))
    assert life.accepted(life.rows()['15663'])


def test_unexpected_current_native_echo_revokes_prior_acceptance(ledger):
    life,_,_=ledger;echo(life);assert life.accepted(life.rows()['15663'])
    changed=event(life,'openOrder');changed['body_sha256']='f'*64
    life.native(changed)
    assert not life.accepted(life.rows()['15663'])
    assert life.rows()['15663']['echo_mismatch'] and life.rows()['15663']['echo'] is None


class WireFixture(SimBroker):
    """Native-shaped callbacks, explicitly inert. Not paper/exchange qualification."""
    def __init__(self,*args):
        super().__init__(*args)
        self.native_order_callback=lambda e:None
        self.lifecycle_revision=lambda:0
        self.cancel_calls=[];self.send_calls=[];self.filled={}
        self.pending_cancel=False;self.accept_entries=True;self.echo_entries=True
        self.immediate=False;self.protection_failure=False;self.snapshot_hook=None

    def native(self,oid,kind='orderStatus',status=None,body_override=None):
        o=self.orders[oid];b=o['body'] if body_override is None else body_override
        self.native_order_callback(dict(source='ibkr_wire',kind=kind,id=oid,
            account=self.config.account,client_id=self.config.client_id,session=self.clock().astimezone(NY).date().isoformat(),
            con_id=o['market'].execution.con_id,ref=b['ref'],perm_id=10000+oid,
            received_at=self.clock().astimezone(timezone.utc).isoformat(),
            status=status or o['status'],body_sha256=body_hash(b),filled=self.filled.get(oid,0),
            remaining=max(0,b['qty']-self.filled.get(oid,0))))
        # Quantity is explicit native-shaped evidence, never inferred from SDK status.

    def send(self,oid,market,b):
        self.send_calls.append((oid,deepcopy(b)))
        self.orders[oid]=dict(body=dict(b),market=market,status='Submitted')
        if b['kind']=='STP' and self.protection_failure:raise RuntimeError('fixture protective send failure')
        entry=b['kind']=='STP LMT'
        if entry and not self.accept_entries:self.orders[oid]['status']='PendingSubmit'
        self.status_callback(oid,'PendingSubmit') # local SDK observation proves nothing
        if not entry or self.echo_entries:self.native(oid,'openOrder')
        self.native(oid)
        if entry and self.immediate:self.execute(oid,1,b['stop'])

    def cancel(self,oid):
        self.cancel_calls.append(oid)
        self.orders[oid]['status']='PendingCancel' if self.pending_cancel else 'Cancelled'
        self.status_callback(oid,self.orders[oid]['status'])
        self.native(oid)

    def execute(self,oid,qty,price):
        o=self.orders[oid];b=o['body'];self.execution+=1
        execution=f'FIXTURE-{self.execution}'
        self.record(o['market'].execution.con_id,b['side']*qty,b['ref'],execution)
        self.filled[oid]=self.filled.get(oid,0)+qty
        self.fill_callback(oid,execution,qty,price)
        o['status']='Filled' if self.filled[oid]>=b['qty'] else 'Submitted'
        self.native(oid)
        # OCA sibling cancellation follows real execution, including a partial.
        if b.get('oca'):
            for other,x in list(self.orders.items()):
                if other!=oid and x['body'].get('oca')==b['oca'] and x['status'] in ('Submitted','PendingSubmit'):
                    x['status']='Cancelled';self.native(other)

    async def snapshot(self):
        # A fresh complete read occurs after terminal callbacks, in virtual time.
        self.advance[0]+=timedelta(microseconds=10)
        s=await super().snapshot()
        s['orders']=[dict(id=oid,account=self.config.account,client_id=self.config.client_id,
            con_id=o['market'].execution.con_id,ref=o['body']['ref'],perm_id=10000+oid,
            remaining=o['body']['qty']-self.filled.get(oid,0),
            **{k:o['body'][k] for k in ('kind','side','tif','oca','oca_type','stop','limit') if k in o['body']})
            for oid,o in self.orders.items() if o['status'] in ('Submitted','PendingSubmit','PendingCancel')]
        s['execution_health']=dict(source='ibkr_responses',account=self.config.account,
            client_id=self.config.client_id,session=self.clock().astimezone(NY).date().isoformat(),
            execution_since=self.clock().astimezone(NY).date().isoformat(),
            received_at=self.clock().isoformat(),started_at=self.clock().astimezone(timezone.utc).isoformat(),
            roundtrip_seconds=0.,connected=True,healthy=True,fresh_orders=True,fresh_positions=True,
            fresh_executions=not bool(self.executions_error),fresh_completed=True,callback_stable=True,
            lifecycle_revision=self.lifecycle_revision())
        if self.snapshot_hook:self.snapshot_hook(s)
        return s


from open_breakout.strategy import NY


async def parked(c,tmp,**options):
    service,b,store,now=setup_service(enabled(c),tmp,WireFixture);b.advance=now
    async def sleep(seconds):
        now[0]+=timedelta(seconds=seconds)
        await asyncio.sleep(0)
    service.reliability_sleep=sleep
    for k,v in options.items():setattr(b,k,v)
    await tick(service,b,now,20000,'09:30:00')
    return service,b,store,now


@pytest.mark.parametrize('partial',[1,2,3])
def test_late_partial_fill_during_halt_keeps_single_current_protector(config,tmp_path,partial):
    async def run():
        s,b,st,now=await parked(config,tmp_path)
        try:
            state=s.states['NQ'];oid=state.entry_orders[0];risk=st.get('daily_reserved')
            s.halt('Oct7 ACK failure');b.pending_cancel=True
            await s.cancel_resting_entries()
            for i in range(partial):b.execute(oid,1,20010.+i*.25)
            await s.drain()
            stops=[x for x in st.orders().values() if x['role']=='STOP']
            assert len(stops)==1 and state.qty==partial
            assert b.orders[state.stop_order]['body']['qty']==partial
            assert s.lifecycle.accepted(s.lifecycle.rows()[str(state.stop_order)])
            assert state.stop_order not in b.cancel_calls
            assert len(b.cancel_calls)==len(set(b.cancel_calls))
            assert st.get('initiating_cause')['reason']=='Oct7 ACK failure'
            assert st.get('daily_reserved')==risk and state.attempts==1
            assert not s.reliability_admission()
        finally:st.close()
    asyncio.run(run())


def test_immediate_fill_has_intents_before_send_and_no_duplicate_protector(config,tmp_path):
    async def run():
        s,b,st,now=await parked(config,tmp_path,immediate=True)
        try:
            state=s.states['NQ']
            assert state.qty==1 and state.stop_order
            assert len([o for o in st.orders().values() if o['role']=='STOP'])==1
            assert all(str(oid) in s.lifecycle.rows() for oid in state.entry_orders)
            assert s.lifecycle.rows()[str(state.entry_orders[1])]['closed']
            assert state.stop_order not in b.cancel_calls
        finally:st.close()
    asyncio.run(run())


def test_unknown_execution_cycle_blocks_other_market_and_protection_failure_halts(config,tmp_path):
    async def run():
        s,b,st,now=await parked(config,tmp_path)
        try:
            b.executions_error='missing executions';assert not await s.reliability_snapshot()
            assert s.owner_block and not s.reliability_admission()
            b.protection_failure=True;b.execute(s.states['NQ'].entry_orders[0],1,20010.)
            await s.drain()
            assert s.halted and s.states['NQ'].qty==1
            assert st.fill_ids() and not s.reliability_admission()
        finally:st.close()
    asyncio.run(run())


@pytest.mark.parametrize('bounded',[False,True])
def test_per_market_wait_releases_lock_but_unknown_exposure_blocks_NQ(config,tmp_path,bounded):
    async def run():
        cfg=replace(config,markets=tuple(replace(m,risk_bps=15.) if m.name=='ES' else m for m in config.markets))
        s,b,st,now=setup_service(enabled(cfg),tmp_path,WireFixture);b.advance=now
        gate=asyncio.Event()
        s.reliability_sleep=lambda _:gate.wait()
        b.accept_entries=False;b.echo_entries=bounded
        try:
            b.update_quote('ES',4999.75,5000.,now[0]);s.tick('ES',now[0],5000.)
            for _ in range(100):
                if s.states['ES'].entry_orders:break
                await asyncio.sleep(0)
            es=s.states['ES'];assert es.entry_orders and not s.entry_lock.locked()
            b.accept_entries=True;b.echo_entries=True
            b.update_quote('NQ',19999.75,20000.,now[0]);s.tick('NQ',now[0],20000.)
            for _ in range(100):
                if s.states['NQ'].entry_orders or s.states['NQ'].phase=='BLOCKED':break
                await asyncio.sleep(0)
            nq=s.states['NQ']
            if bounded:
                assert len(nq.entry_orders)==2 and nq.attempts==1
            else:
                assert not nq.entry_orders and nq.attempts==0 and nq.phase=='BLOCKED'
            assert sum(x.planned_risk for x in s.states.values())<=config.shadow_equity*config.max_open_risk_bps/10000
            assert st.get('daily_reserved')<=config.shadow_equity*config.max_daily_risk_bps/10000
            for oid in es.entry_orders:
                b.orders[oid]['status']='Submitted';b.native(oid,'openOrder');b.native(oid)
            gate.set();await s.drain()
            assert es.attempts==1
        finally:
            gate.set()
            for task in list(s.tasks):task.cancel()
            await asyncio.gather(*list(s.tasks),return_exceptions=True)
            st.close()
    asyncio.run(run())


def test_Oct7_pending_cancel_and_OCA_sequence_preserves_first_cause(config,tmp_path):
    async def run():
        s,b,st,now=await parked(config,tmp_path)
        try:
            state=s.states['NQ'];entries=list(state.entry_orders);risk=st.get('daily_reserved')
            s.halt('ORDER_ACCEPTANCE_TIMEOUT:ES:15663');b.pending_cancel=True
            await s.cancel_resting_entries()
            # Oct7: duplicate cancel refusal -> SDK synthetic Cancelled -> native Inactive/OCA error.
            for oid in entries:
                s.status(oid,'Cancelled');s.order_error(oid,10148,'state PendingCancel')
                assert not s.lifecycle.rows()[str(oid)]['closed']
                b.orders[oid]['status']='Inactive';b.native(oid)
            s.order_error(entries[1],201,'OCA order cancel')
            for _ in range(4):await s.cancel_resting_entries()
            assert len(b.cancel_calls)==2 and len(set(b.cancel_calls))==2
            assert all(s.lifecycle.rows()[str(o)]['closed'] for o in entries)
            s.halt('RECONCILE:later symptom')
            assert st.get('halt_reason')=='ORDER_ACCEPTANCE_TIMEOUT:ES:15663'
            assert st.get('initiating_cause')['reason']==st.get('halt_reason')
            assert state.attempts==1 and st.get('daily_reserved')==risk and not state.long_armed and not state.short_armed
        finally:st.close()
    asyncio.run(run())


@pytest.mark.parametrize('age',[599.999,600.])
def test_Oct6_price_gap_exact_grace_boundary_retains_protection_and_risk(config,tmp_path,age):
    async def run():
        s,b,st,now=await parked(config,tmp_path)
        try:
            state=s.states['NQ'];risk=st.get('daily_reserved');entries=state.entry_orders.copy()
            start=now[0];now[0]=at('09:30:41');await s.watchdog()
            assert s.price_paused and not s.halted
            now[0]=start+timedelta(seconds=age);await s.watchdog();await s.drain()
            assert s.halted==(age>=600)
            assert len(b.cancel_calls)==(len(entries) if age>=600 else 0)
            assert state.attempts==1 and st.get('daily_reserved')==risk
            assert not st.fill_ids()
        finally:st.close()
    asyncio.run(run())


def test_real_SDK_decoder_hook_excludes_synthetic_error_status(config):
    async def run():
        from ib_insync import IB, Contract, Order, OrderStatus, OrderState, Trade
        from open_breakout.ibkr import IBKR
        b=IBKR(enabled(config),session='2026-10-07')
        b.ib.client.clientId=config.client_id
        b.ib.wrapper.clientId=config.client_id
        m=config.markets[0];intent={**body(), 'account':config.account}
        order=b._order(15663,intent);order.permId=1323944217
        contract=Contract(conId=m.execution.con_id)
        trade=Trade(contract,order,OrderStatus(orderId=15663,status='PendingSubmit'))
        b.trades[15663]=trade;b._intent_bodies[15663]=intent
        key=b.ib.wrapper.orderKey(config.client_id,15663,order.permId)
        b.ib.wrapper.trades[key]=trade;b.ib.wrapper.permId2Trade[order.permId]=trade
        events=[];sdk=[];b.native_order_callback=events.append
        b.ib.orderStatusEvent+=lambda t:sdk.append(t.orderStatus.status)
        b._install_wire_hooks()
        b.ib.wrapper.openOrder(15663,contract,order,OrderState(status='Submitted'))
        b.ib.wrapper.orderStatus(15663,'Submitted',0,8,0,order.permId,0,0,config.client_id,'',0)
        assert [e['kind'] for e in events]==['openOrder','orderStatus']
        assert events[0]['body_sha256']==body_hash(intent)
        before=len(events)
        b.ib.wrapper.error(15663,10148,'cannot cancel; state PendingCancel','')
        assert sdk[-1]=='Cancelled' and len(events)==before
        # Existing decoder resolves the wrapper method dynamically; the hook is on the wire route.
        handler=b.ib.client.decoder.handlers[3]
        handler(['3','15663','Cancelled','0','8','0',str(order.permId),'0','0',str(config.client_id),'','0'])
        assert events[-1]['status']=='Cancelled' and len(events)==before+1
    asyncio.run(run())


@pytest.mark.parametrize('status',['Cancelled','Inactive','ApiCancelled'])
def test_native_protector_disappearance_never_clears_account_block(config,tmp_path,status):
    async def run():
        s,b,st,now=await parked(config,tmp_path)
        try:
            state=s.states['NQ'];b.execute(state.entry_orders[0],1,20010.);await s.drain()
            stop=state.stop_order;b.orders[stop]['status']=status;b.native(stop)
            await s.drain();await s.watchdog()
            assert state.qty==1 and s.halted and s.owner_block
            assert not s.reliability_admission()
            assert 'PROTECT' in st.get('initiating_cause')['reason']
            assert stop not in b.cancel_calls
        finally:st.close()
    asyncio.run(run())


def test_lifecycle_and_fill_journal_transaction_survive_crash_gap(config,tmp_path,monkeypatch):
    async def run():
        s,b,st,now=await parked(config,tmp_path)
        try:
            state=s.states['NQ'];oid=state.entry_orders[0];original=st.add_fill
            def crash(*a):raise RuntimeError('fixture crash inside journal commit')
            monkeypatch.setattr(st,'add_fill',crash)
            b.execute(oid,1,20010.)
            assert not st.fill_ids() and s.lifecycle.rows()[str(oid)]['filled']==0 and not state.qty
            monkeypatch.setattr(st,'add_fill',original)
            s.fill(oid,'FIXTURE-1',1,20010.);await s.drain()
            assert len(st.fill_ids())==1 and state.qty==1
            assert s.lifecycle.rows()[str(oid)]['filled']==1
            assert len([o for o in st.orders().values() if o['role']=='STOP'])==1
            assert b.orders[state.stop_order]['body']['qty']==1
        finally:st.close()
    asyncio.run(run())


def test_opposed_OCA_net_zero_cleans_only_exact_owned_exits_after_full_proof(config,tmp_path):
    async def run():
        s,b,st,now=await parked(config,tmp_path)
        try:
            state=s.states['NQ'];buy,sell=state.entry_orders
            b.execute(buy,1,20010.);await s.drain()
            stop,timed=state.stop_order,state.time_order;risk=st.get('daily_reserved')
            b.execute(sell,1,19990.)
            b.orders[sell]['status']='Cancelled';b.native(sell)
            await s.drain()
            assert state.qty==0 and state.phase=='BLOCKED' and s.halted
            assert stop in b.cancel_calls and timed in b.cancel_calls
            assert b.status(stop)==b.status(timed)=='Cancelled'
            assert all(s.lifecycle.rows()[str(o)]['closed'] for o in (stop,timed))
            assert st.get('daily_reserved')==risk and state.attempts==1
            assert not s.reliability_admission()
            assert st.get('initiating_cause')['reason'].startswith('OPPOSED_OCA_EXECUTION')
            assert s.lifecycle.rows()[str(stop)]['flat_cleanup_proof_sha256']
        finally:st.close()
    asyncio.run(run())


@pytest.mark.parametrize('halted',[False,True])
def test_reliability_and_price_pause_recover_only_price_latch(config,tmp_path,halted):
    async def run():
        s,b,st,now=await parked(config,tmp_path)
        try:
            risk=st.get('daily_reserved');count=len(st.orders())
            now[0]=at('09:30:41');await s.watchdog();assert s.price_paused
            if halted:s.halt('separate safety incident');await s.drain()
            now[0]=at('09:30:42');s.tick('NQ',now[0],20000.)
            b.update_quote('NQ',19999.75,20000.,now[0]);await s.watchdog()
            assert s.price_paused==halted and s.halted==halted
            assert st.get('daily_reserved')==risk and len(st.orders())==count
            assert s.states['NQ'].attempts==1
            if halted:assert st.get('initiating_cause')['reason']=='separate safety incident'
        finally:st.close()
    asyncio.run(run())


def test_actual_adapter_quarantines_only_exact_owned_cancel_consequences(config,tmp_path):
    async def run():
        from types import SimpleNamespace as NS
        from open_breakout.ibkr import IBKR
        s,b,st,now=await parked(config,tmp_path)
        try:
            oid=s.states['NQ'].entry_orders[0];b.pending_cancel=True
            s.halt('ACK initiating cause');await s.cancel_resting_entries()
            adapter=IBKR.__new__(IBKR);adapter.config=s.config;adapter.healthy=True
            adapter.trades={oid:NS()};adapter.diagnostic_callback=lambda _:None
            adapter.order_error_callback=s.order_error;adapter.order_error_policy=s.reliability_error_policy
            adapter.halt_callback=s.halt
            adapter._error(oid,10148,'cannot cancel; state PendingCancel',None)
            assert adapter.healthy and s.owner_block and not s.reliability_admission()
            b.orders[oid]['status']='Inactive';b.native(oid)
            adapter._error(oid,201,'Order rejected - reason:OCA order cancel',None)
            assert adapter.healthy
            # Unknown/unrequested rejection retains the existing fatal policy.
            adapter._error(999999,201,'unrelated rejection',None)
            assert not adapter.healthy and st.get('initiating_cause')['reason']=='ACK initiating cause'
            assert len(b.cancel_calls)==len(set(b.cancel_calls))
        finally:st.close()
    asyncio.run(run())


@pytest.mark.parametrize('crash',[False,True])
def test_actual_sdk_execution_first_binds_owner_and_redelivers_failed_commit(config,tmp_path,monkeypatch,crash):
    async def run():
        from ib_insync import Contract,OrderStatus,Trade,Execution
        from open_breakout.ibkr import IBKR
        s,b,st,now=await parked(config,tmp_path)
        try:
            state=s.states['NQ'];oid=state.entry_orders[0];intent=st.orders()[oid]['body']
            adapter=IBKR(s.config,session=s.manifest['day'])
            adapter.ib.client.clientId=s.config.client_id;adapter.ib.wrapper.clientId=s.config.client_id
            contract=Contract(conId=s.markets['NQ'].execution.con_id)
            order=adapter._order(oid,intent);assert order.permId==0
            trade=Trade(contract,order,OrderStatus(orderId=oid,status='PendingSubmit'))
            adapter.trades[oid]=trade;adapter._intent_bodies[oid]=intent
            adapter.ib.wrapper.trades[adapter.ib.wrapper.orderKey(s.config.client_id,oid,0)]=trade
            adapter.native_order_callback=s.reliability_native;adapter.fill_callback=s.fill;adapter.halt_callback=s.halt
            adapter.ib.execDetailsEvent+=adapter._fill
            # Reset the fixture's earlier echoes to reproduce actual execDetails first.
            rows=s.lifecycle.rows();rows[str(oid)].update(perm_id=0,native=None,echo=None)
            st.set(s.lifecycle.KEY,rows)
            execution=Execution(execId='SDK.EXEC.FIRST.01',orderId=oid,clientId=s.config.client_id,
                acctNumber=s.config.account,orderRef=intent['ref'],permId=10000+oid,
                shares=1,price=20010.,side='BOT',cumQty=1,time=now[0])
            original=st.add_fill
            if crash:monkeypatch.setattr(st,'add_fill',lambda *a:(_ for _ in ()).throw(RuntimeError('commit failure')))
            adapter.ib.wrapper.execDetails(-1,contract,execution)
            assert trade.order.permId==execution.permId
            assert not s.lifecycle.accepted(s.lifecycle.rows()[str(oid)])
            if crash:
                assert execution.execId not in adapter._delivered and not st.fill_ids()
                monkeypatch.setattr(st,'add_fill',original)
                # SDK suppresses a duplicate event, so reqExecutions must redeliver it.
                async def responses(_):return [adapter.ib.wrapper.fills[execution.execId]]
                adapter.ib.reqExecutionsAsync=responses
                monkeypatch.setattr('open_breakout.ibkr.session_start',lambda *a:now[0].replace(hour=0,minute=0,second=0,microsecond=0))
                await adapter._execution_books(force=True)
            assert execution.execId in adapter._delivered and len(st.fill_ids())==1
            assert state.qty==1 and state.stop_order
            assert len([x for x in st.orders().values() if x['role']=='STOP'])==1
            assert b.orders[state.stop_order]['body']['qty']==1
        finally:
            for task in list(s.tasks):task.cancel()
            await asyncio.gather(*list(s.tasks),return_exceptions=True)
            st.close()
    asyncio.run(run())


def test_normal_stop_to_flat_uses_durable_exit_cancel_and_deadline(config,tmp_path):
    async def run():
        s,b,st,now=await parked(config,tmp_path)
        try:
            state=s.states['NQ'];b.execute(state.entry_orders[0],1,20010.);await s.drain()
            stop,timed=state.stop_order,state.time_order;b.pending_cancel=True
            # Prevent fixture OCA magic: native sibling remains PendingCancel.
            b.orders[timed]['status']='PendingCancel'
            b.execute(stop,1,state.stop);await s.drain()
            assert not state.qty and timed in b.cancel_calls
            row=s.lifecycle.rows()[str(timed)]
            assert row['cancel_started'] and row['flat_cleanup_proof_sha256'] and not row['closed']
            s.status(timed,'Cancelled');s.fill(stop,'FIXTURE-2',1,state.stop)
            count=b.cancel_calls.count(timed)
            first_deadline=st.get('flat_exit_cleanup')['NQ']['started_at']
            now[0]+=timedelta(seconds=10)
            await s.reliability_flat_exit_cleanup('NQ')
            assert b.cancel_calls.count(timed)==count==1
            assert st.get('flat_exit_cleanup')['NQ']['started_at']==first_deadline
            now[0]+=timedelta(seconds=6);await s.watchdog();await s.drain()
            assert s.halted and st.get('flat_exit_cleanup') and not s.reliability_admission()
            b.orders[timed]['status']='Cancelled';b.native(timed);await s.watchdog()
            assert s.lifecycle.rows()[str(timed)]['closed'] and not st.get('flat_exit_cleanup')
            assert b.cancel_calls.count(timed)==1
        finally:st.close()
    asyncio.run(run())


def test_rejected_protection_never_uses_unqualified_emergency_flatten(config,tmp_path):
    async def run():
        s,b,st,now=await parked(config,tmp_path)
        try:
            state=s.states['NQ'];b.execute(state.entry_orders[0],1,20010.);await s.drain()
            stop,timed=state.stop_order,state.time_order;before=len(b.send_calls)
            s.status(stop,'Cancelled');s.order_error(stop,201,'explicit rejection')
            await s._emergency_flatten('NQ','fixture rejected stop');await s.drain()
            assert s.halted and state.qty==1 and len(b.send_calls)==before
            assert not any(x['role']=='FLATTEN' for x in st.orders().values())
            assert stop not in b.cancel_calls and timed not in b.cancel_calls
        finally:st.close()
    asyncio.run(run())


def test_native_partial_exit_OCA_reduction_requires_manual_revision_proof(config,tmp_path):
    async def run():
        s,b,st,now=await parked(config,tmp_path)
        try:
            state=s.states['NQ'];b.execute(state.entry_orders[0],3,20010.);await s.drain()
            stop,timed=state.stop_order,state.time_order
            # Keep both exits working while the broker reduces the sibling qty.
            original_oca=b.orders[timed]['body']['oca'];b.orders[timed]['body']['oca']='fixture-isolate'
            b.execute(stop,1,state.stop)
            b.orders[timed]['body'].update(oca=original_oca,qty=2)
            b.native(timed,'openOrder');b.native(timed)
            await s.watchdog();await s.drain()
            assert state.qty==2 and s.halted and s.owner_block
            assert not s.reliability_admission()
            assert stop not in b.cancel_calls and timed not in b.cancel_calls
            assert not any(x['role']=='FLATTEN' for x in st.orders().values())
            assert b.orders[timed]['body']['qty']==2
        finally:st.close()
    asyncio.run(run())


def test_late_known_exit_reversal_is_journaled_once_and_loud_manual(config,tmp_path):
    async def run():
        s,b,st,now=await parked(config,tmp_path)
        try:
            state=s.states['NQ'];b.execute(state.entry_orders[0],1,20010.);await s.drain()
            stop,timed=state.stop_order,state.time_order
            b.execute(stop,1,state.stop);await s.drain()
            assert not state.qty
            b.execute(timed,1,20000.);await s.drain()
            assert state.qty==1 and state.side==-1 and s.halted
            assert len(st.fill_ids())==3 and s.lifecycle.rows()[str(timed)]['filled']==1
            assert s.fill(timed,'FIXTURE-3',1,20000.) is True and state.qty==1
            assert not s.reliability_admission()
        finally:st.close()
    asyncio.run(run())


def test_same_entry_late_partial_after_native_closed_exits_gets_new_protection(config,tmp_path):
    async def run():
        s,b,st,now=await parked(config,tmp_path)
        try:
            state=s.states['NQ'];entry=state.entry_orders[0]
            b.execute(entry,1,20010.);await s.drain()
            old_stop,old_time=state.stop_order,state.time_order
            b.execute(old_stop,1,state.stop);await s.drain()
            assert not state.qty and all(s.lifecycle.rows()[str(o)]['closed'] for o in (old_stop,old_time))
            b.execute(entry,1,20011.);await s.drain()
            assert state.qty==1 and state.stop_order not in (old_stop,old_time)
            assert s.lifecycle.accepted(s.lifecycle.rows()[str(state.stop_order)])
            assert b.orders[state.stop_order]['body']['qty']==1 and state.stop_order not in b.cancel_calls
            assert len([x for x in st.orders().values() if x['role']=='STOP'])==2
            assert s.halted and not s.reliability_admission()
            assert len(st.fill_ids())==3
        finally:st.close()
    asyncio.run(run())
