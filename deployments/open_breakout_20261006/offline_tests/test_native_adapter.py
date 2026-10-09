"""IB-shaped async broker contract, original Service/Store callbacks; no network.

The simulated same-ID order-type atomicity is an explicit test assumption,
not evidence that IBKR supports it. Default/unsafe paths preserve protection.
"""
import asyncio
import copy
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace as NS

import pytest

from intraday_coordination import Ledger, LegendSignal, Scope, EntryBlocked
from open_breakout.native_coordination import AsyncOwnerCoordinator, NativeOwnerAdapter, OfflineCloseQualification
from open_breakout.replay import SimBroker
from test_existing_open_breakout import config, setup_service, tick, at, DAY


class NativeAPI:
    def __init__(self,owner):
        self.owner=owner;self.omit_completed=set();self.omit_open=set();self.hide_fills=set();self.delay=0
        self.read_hook=None;self.timeout=False
        self.client=NS(clientId=owner.config.client_id);self.connected=True

    def isConnected(self):return self.connected

    async def read(self,kind):
        self.owner.calls.append(('read',kind))
        await asyncio.sleep(self.delay)
        if self.timeout:raise TimeoutError('read unavailable')
        if self.read_hook:
            hook=self.read_hook;self.read_hook=None;hook()

    async def reqAllOpenOrdersAsync(self):
        await self.read('open')
        return [copy.deepcopy(t) for oid,t in self.owner.trades.items()
                if oid not in self.omit_open and t.orderStatus.status not in {'Filled','Cancelled','ApiCancelled','Inactive','REJECTED'}]

    async def reqCompletedOrdersAsync(self,apiOnly):
        assert apiOnly is False
        await self.read('completed')
        result=[]
        for oid,t in self.owner.trades.items():
            if oid in self.omit_completed or t.orderStatus.status not in {'Filled','Cancelled','ApiCancelled','Inactive','REJECTED'}:continue
            # Faithful installed SDK shape: completedOrder does not populate API
            # client/order IDs, status filled counters, fills or trade log.
            row=copy.deepcopy(t);row.order.filledQuantity=t.orderStatus.filled
            row.order.orderId=row.order.clientId=0;row.orderStatus.filled=0;row.fills=[];row.log=[]
            result.append(row)
        return result

    async def reqExecutionsAsync(self,filter):
        assert filter.acctCode==self.owner.config.account
        await self.read('executions')
        return [copy.deepcopy(f) for f in self.owner.native_fills if f.execution.execId not in self.hide_fills]

    async def reqPositionsAsync(self):
        await self.read('positions')
        return [NS(account=self.owner.config.account,contract=NS(conId=cid),position=qty)
                for cid,qty in {**self.owner.positions}.items() if qty]

    def connect(self,*args,**kwargs):raise AssertionError('No broker connection allowed')
    async def connectAsync(self,*args,**kwargs):raise AssertionError('No broker connection allowed')


class NativeMock(SimBroker):
    """Native objects and cumulative fill/OCA counters through source callbacks."""
    def __init__(self,config,clock):
        super().__init__(config,clock)
        self.trades={};self.native_fills=[];self.calls=[];self.ib=NativeAPI(self)
        self.snapshot_lock=asyncio.Lock();self._exec_generation=0;self.offline_fixture=True
        self.modify_mode='full';self.before_modify=None;self.before_cancel=None;self.lose_ack=False

    def status(self,oid):return self.trades[oid].orderStatus.status if oid in self.trades else 'UNKNOWN'

    def send(self,oid,market,body):
        if oid in self.trades:
            old=self.trades[oid]
            if self.before_modify:
                hook=self.before_modify;self.before_modify=None;hook()
            if self.status(oid) in {'Filled','Cancelled','ApiCancelled','Inactive','REJECTED'}:
                self.calls.append(('terminal_modify_rejected',oid))
                raise RuntimeError('Same native identity cannot reopen terminal order')
            if body['kind']=='MKT' and old.order.orderType=='STP' and self.modify_mode=='refuse':
                self.calls.append(('modify_refused',oid))
                old.log.append(NS(errorCode=201));return
            self.calls.append(('modify',oid,body['kind'],body['qty']))
            assert body['ref']==old.order.orderRef and body['oca']==old.order.ocaGroup
            assert body['qty']==old.order.totalQuantity  # never resize/recreate residual in this handoff
            old.order.orderType=body['kind']
            old.order.auxPrice=body.get('stop',0)
            trade=old
        else:
            self.calls.append(('new',oid,body['kind'],body['qty']))
            o=NS(account=self.config.account,clientId=self.config.client_id,orderId=oid,permId=10000+oid,
                 action='BUY' if body['side']==1 else 'SELL',totalQuantity=body['qty'],orderType=body['kind'],
                 orderRef=body['ref'],tif=body['tif'],auxPrice=body.get('stop',0),
                 ocaGroup=body.get('oca',''),ocaType=body.get('oca_type',2 if body.get('oca') else 0),goodAfterTime=body.get('good_after',''))
            c=NS(conId=market.execution.con_id,symbol=market.execution.symbol)
            trade=NS(order=o,contract=c,orderStatus=NS(status='Submitted',filled=0),fills=[],log=[])
            self.trades[oid]=trade
        self.orders[oid]=dict(body=body.copy(),market=market,status=trade.orderStatus.status)
        self.status_callback(oid,'Submitted')
        if body['kind']=='MKT' and not body.get('good_after'):
            remaining=trade.order.totalQuantity-trade.orderStatus.filled
            if self.modify_mode=='full':self.execute(oid,remaining,19990)
            elif self.modify_mode=='partial':self.execute(oid,min(1,remaining),19990)
            elif self.modify_mode=='reject':
                trade.orderStatus.status='Inactive';trade.log.append(NS(errorCode=201))
                self.status_callback(oid,'Inactive')
            if self.lose_ack:raise TimeoutError('lost acknowledgement after native acceptance')

    def _fill(self,trade,fill):
        self._exec_generation+=1
        e=fill.execution
        self.fill_callback(e.orderId,e.execId,e.shares,e.price)

    def execute(self,oid,qty,price,*,deliver=True):
        trade=self.trades[oid];o=trade.order
        assert self.status(oid)=='Submitted'
        assert 0<qty<=o.totalQuantity-trade.orderStatus.filled
        self.execution+=1
        e=NS(acctNumber=o.account,clientId=o.clientId,orderId=oid,permId=o.permId,orderRef=o.orderRef,
             shares=qty,side='BOT' if o.action=='BUY' else 'SLD',price=price,time=self.clock(),
             execId=f'NATIVE-{self.execution}.01')
        fill=NS(execution=e,contract=trade.contract)
        self.native_fills.append(fill);trade.fills.append(fill);trade.orderStatus.filled+=qty
        self.record(trade.contract.conId,qty if o.action=='BUY' else -qty,o.orderRef,e.execId)
        if o.ocaGroup:
            for other_id,sibling in list(self.trades.items()):
                if other_id==oid or sibling.order.ocaGroup!=o.ocaGroup or self.status(other_id)!='Submitted':continue
                if o.ocaType==2:
                    sibling.order.totalQuantity=max(sibling.orderStatus.filled,sibling.order.totalQuantity-qty)
                    if sibling.order.totalQuantity==sibling.orderStatus.filled:
                        sibling.orderStatus.status='Cancelled';self.orders[other_id]['status']='Cancelled'
                        self.status_callback(other_id,'Cancelled')
                elif trade.orderStatus.filled==o.totalQuantity:
                    sibling.orderStatus.status='Cancelled';self.orders[other_id]['status']='Cancelled'
                    self.status_callback(other_id,'Cancelled')
        if deliver:self._fill(trade,fill)
        trade.orderStatus.status='Filled' if trade.orderStatus.filled==o.totalQuantity else 'Submitted'
        self.orders[oid]['status']=trade.orderStatus.status;self.status_callback(oid,trade.orderStatus.status)

    def cancel(self,oid):
        if self.before_cancel:
            hook=self.before_cancel;self.before_cancel=None;hook(oid)
        self.calls.append(('cancel',oid))
        t=self.trades[oid]
        if self.status(oid)=='Filled':return
        t.orderStatus.status='Cancelled';self.orders[oid]['status']='Cancelled';self.status_callback(oid,'Cancelled')
        for other_id,other in list(self.trades.items()):
            if other_id!=oid and other.order.ocaGroup==t.order.ocaGroup and t.order.ocaGroup and self.status(other_id)=='Submitted':
                other.orderStatus.status='Cancelled';self.orders[other_id]['status']='Cancelled'
                self.status_callback(other_id,'Cancelled')


QUALIFIED=OfflineCloseQualification(True,True,True,True)


async def setup(config,tmp_path,*,filled=2,qualified=False):
    service,b,store,now=setup_service(replace(config,entry_order_type='stop_limit'),tmp_path,NativeMock)
    ledger=Ledger(tmp_path/'native-coord.sqlite',enabled=True);service.coordination=ledger
    adapter=NativeOwnerAdapter(service,qualification=QUALIFIED if qualified else None,
                              filter_factory=lambda account:NS(acctCode=account))
    bridge=AsyncOwnerCoordinator(adapter);service.coordination_backend=bridge
    await tick(service,b,now,20000,'09:30:00')
    row=next(oid for oid,o in store.orders().items() if o['role']=='ENTRY' and o['body']['side']==-1)
    if filled:
        b.execute(row,filled,19990);await service.drain()
    now[0]=at('09:31:00')
    scope=Scope.for_symbol(DAY,config.account,'MNQ')
    ledger.publish(LegendSignal(scope,1,'legend-native',now[0],True,True,True,2),now[0])
    b.calls.clear()
    return service,b,store,now,scope,bridge


async def advance(bridge,scope,now,until='done',count=15):
    result=[]
    for _ in range(count):
        result.append(await bridge.poll(scope,now[0]))
        await bridge.adapter.service.drain()
        if result[-1] in {until,'reconcile'}:break
    return result


def test_default_native_adapter_keeps_both_exits_and_reports_exact_missing_guarantee(config,tmp_path):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path)
        try:
            stop,time=s.states['NQ'].stop_order,s.states['NQ'].time_order
            assert await advance(c,scope,now)==['protected_prepare','reconcile']
            assert b.status(stop)==b.status(time)=='Submitted'
            assert not [x for x in b.calls if x[0] in {'modify','new','cancel'}]
            assert not s.coordination.legend_ready(scope,now[0])
        finally:store.close()
    asyncio.run(run())


def test_native_unfilled_cancel_requires_terminal_receipts_and_can_settle(config,tmp_path):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,filled=0)
        try:
            assert (await advance(c,scope,now))[-1]=='done',s.coordination.read(scope)['operation']
            assert s.coordination.legend_ready(scope,now[0])
            cancelled=[x[1] for x in b.calls if x[0]=='cancel']
            assert len(cancelled)==1 and store.orders()[cancelled[0]]['body']['side']==-1
            assert not [x for x in b.calls if x[0] in {'modify','new'}]
        finally:store.close()
    asyncio.run(run())


@pytest.mark.parametrize('unsafe',['missing_terminal','hidden_fill','inactive','manual','netted','read_timeout','correction'])
def test_native_unsafe_evidence_never_mutates_protection(config,tmp_path,unsafe):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,qualified=True)
        try:
            state=s.states['NQ'];stop=state.stop_order;before=state.qty
            entry=next(oid for oid,row in store.orders().items() if row['role']=='ENTRY' and row['body']['side']==-1)
            if unsafe=='missing_terminal':b.ib.omit_completed.add(entry)
            if unsafe=='hidden_fill':b.ib.hide_fills.add(b.native_fills[0].execution.execId)
            if unsafe=='inactive':b.trades[stop].orderStatus.status='Inactive'
            if unsafe=='manual':s.manual_markets.add('NQ')
            if unsafe=='netted':b.positions[scope and s.markets['NQ'].execution.con_id]=0
            if unsafe=='read_timeout':b.ib.timeout=True
            if unsafe=='correction':
                corrected=copy.deepcopy(b.native_fills[0]);corrected.execution.execId=corrected.execution.execId.rsplit('.',1)[0]+'.02'
                corrected.execution.shares=1;b.native_fills.append(corrected)
            assert (await advance(c,scope,now))[-1]=='reconcile'
            assert b.trades[stop].order.orderType=='STP'
            assert not [x for x in b.calls if x[0] in {'modify','new','cancel'}]
            assert state.qty==before
        finally:store.close()
    asyncio.run(run())


def test_fixture_qualified_same_id_close_has_no_exit_cancel_or_new_market_order(config,tmp_path):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,qualified=True)
        try:
            stop=s.states['NQ'].stop_order;identity=copy.deepcopy(b.trades[stop].order.permId)
            assert (await advance(c,scope,now))[-1]=='done'
            assert s.states['NQ'].qty==0 and s.coordination.legend_ready(scope,now[0])
            assert ('modify',stop,'MKT',2) in b.calls
            assert not [x for x in b.calls if x[0]=='new']
            assert not [x for x in b.calls if x[0]=='cancel' and x[1]==stop]
            assert b.trades[stop].order.permId==identity
        finally:store.close()
    asyncio.run(run())


@pytest.mark.parametrize('full',[False,True])
def test_server_stop_fill_at_modify_race_never_resizes_or_reopens(config,tmp_path,full):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,qualified=True)
        try:
            stop=s.states['NQ'].stop_order
            b.before_modify=lambda:b.execute(stop,2 if full else 1,19990)
            result=await advance(c,scope,now)
            if full:
                assert result[-1]=='reconcile'
                assert await c.recover(scope,now[0])=='cleanup'
                assert (await advance(c,scope,now))[-1]=='done'
                assert not [x for x in b.calls if x[0]=='modify']
            else:
                assert result[-1]=='done' and ('modify',stop,'MKT',2) in b.calls
            assert s.states['NQ'].qty==0 and not [x for x in b.calls if x[0]=='new']
        finally:store.close()
    asyncio.run(run())


def test_partial_close_restart_finishes_without_repeat_modification(config,tmp_path):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,qualified=True)
        try:
            b.modify_mode='partial';stop=s.states['NQ'].stop_order
            assert (await advance(c,scope,now,until='protected_working'))[-1]=='protected_working'
            assert s.states['NQ'].qty==1
            restarted=AsyncOwnerCoordinator(c.adapter)
            assert await restarted.poll(scope,now[0])=='protected_working'
            b.execute(stop,1,19990);await s.drain()
            assert (await advance(restarted,scope,now))[-1]=='done'
            assert len([x for x in b.calls if x[0]=='modify'])==1
        finally:store.close()
    asyncio.run(run())


@pytest.mark.parametrize('mode',['refuse','reject'])
def test_refused_or_rejected_close_preserves_or_restores_verified_stop_once(config,tmp_path,mode):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,qualified=True)
        try:
            b.modify_mode=mode;old_stop=s.states['NQ'].stop_order
            assert (await advance(c,scope,now))[-1]=='reconcile'
            stop=s.states['NQ'].stop_order
            assert b.trades[stop].order.orderType=='STP' and b.status(stop)=='Submitted'
            assert s.states['NQ'].qty==2 and not s.coordination.legend_ready(scope,now[0])
            restores=[x for x in b.calls if x[0]=='new']
            assert len(restores)==(1 if mode=='reject' else 0)
            if mode=='refuse':assert stop==old_stop
            assert await c.poll(scope,now[0])=='reconcile'
            assert len([x for x in b.calls if x[0]=='new'])==len(restores)
        finally:store.close()
    asyncio.run(run())


def test_lost_ack_and_late_terminal_receipt_recovers_without_duplicate(config,tmp_path):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,qualified=True)
        try:
            b.modify_mode='none';b.lose_ack=True;stop=s.states['NQ'].stop_order
            assert (await advance(c,scope,now))[-1]=='reconcile'
            assert await c.recover(scope,now[0])=='reconcile'
            b.execute(stop,2,19990);await s.drain()
            assert await c.recover(scope,now[0])=='cleanup'
            assert (await advance(c,scope,now))[-1]=='done'
            assert len([x for x in b.calls if x[0]=='modify'])==1
        finally:store.close()
    asyncio.run(run())


def test_partial_close_timeout_does_not_restore_against_working_close(config,tmp_path):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,qualified=True)
        try:
            b.modify_mode='partial'
            await advance(c,scope,now,until='protected_working')
            now[0]+=timedelta(seconds=4)
            assert await c.poll(scope,now[0])=='reconcile'
            assert s.states['NQ'].qty==1
            assert len([x for x in b.calls if x[0]=='modify'])==1 and not [x for x in b.calls if x[0]=='new']
        finally:store.close()
    asyncio.run(run())


def test_async_refresh_does_not_block_callbacks_or_hold_coordination_lock(config,tmp_path):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,filled=0)
        try:
            b.ib.delay=.01;progress=[]
            async def callback():
                await asyncio.sleep(.005)
                assert s.coordination.read(scope)['signal']
                progress.append('callback ran during async read')
            await asyncio.gather(c.poll(scope,now[0]),callback())
            assert progress
        finally:store.close()
    asyncio.run(run())


def test_mock_qualification_cannot_enable_native_nonfixture_transport(config,tmp_path):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,qualified=True)
        try:
            b.offline_fixture=False
            assert (await advance(c,scope,now))[-1]=='reconcile'
            assert not [x for x in b.calls if x[0] in {'new','modify','cancel'}]
        finally:store.close()
    asyncio.run(run())


def test_cancel_fill_race_returns_actual_owner_fill_to_original_protection(config,tmp_path):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,filled=0)
        try:
            b.before_cancel=lambda oid:b.execute(oid,1,19990)
            result=await advance(c,scope,now)
            assert result[-1]=='reconcile'  # default cannot safely auto close filled exposure
            state=s.states['NQ']
            assert state.qty==1 and state.side==-1
            assert b.status(state.stop_order)==b.status(state.time_order)=='Submitted'
            assert not [x for x in b.calls if x[0]=='modify']
        finally:store.close()
    asyncio.run(run())


def test_pending_cancel_and_disappeared_order_are_never_settlement(config,tmp_path):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,filled=0)
        try:
            original=b.cancel
            def pending(oid):
                b.calls.append(('cancel',oid));b.trades[oid].orderStatus.status='PendingCancel'
            b.cancel=pending
            assert await c.poll(scope,now[0])=='cancel_entries'
            assert not s.coordination.legend_ready(scope,now[0])
            now[0]+=timedelta(seconds=4)
            assert await c.poll(scope,now[0])=='reconcile'
            assert len([x for x in b.calls if x[0]=='cancel'])==1
            b.cancel=original
        finally:store.close()
    asyncio.run(run())


def test_missing_completed_cancel_receipt_blocks_unfilled_settlement(config,tmp_path):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,filled=0)
        try:
            assert await c.poll(scope,now[0])=='cancel_entries'
            b.ib.omit_completed={oid for oid in b.trades if b.status(oid)=='Cancelled'}
            assert await c.poll(scope,now[0])=='reconcile'
            assert not s.coordination.legend_ready(scope,now[0])
        finally:store.close()
    asyncio.run(run())


def test_unreported_cancel_race_execution_blocks_settlement_until_owner_fill_caught_up(config,tmp_path):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,filled=0)
        try:
            entry=next(oid for oid,row in store.orders().items() if row['role']=='ENTRY' and row['body']['side']==-1)
            # Native execution query races ahead of its live callback. The
            # adapter must deliver the ACTUAL execution through Service.fill.
            b.execute(entry,1,19990,deliver=False)
            result=await advance(c,scope,now)
            assert result[-1]=='reconcile'
            state=s.states['NQ']
            assert state.qty==1 and b.status(state.stop_order)=='Submitted'
            assert not [x for x in b.calls if x[0]=='modify']
        finally:store.close()
    asyncio.run(run())


def test_partial_close_then_terminal_cancel_restores_only_verified_residual(config,tmp_path):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,qualified=True)
        try:
            b.modify_mode='partial';stop=s.states['NQ'].stop_order
            await advance(c,scope,now,until='protected_working')
            assert s.states['NQ'].qty==1
            b.cancel(stop)  # simulated terminal cancellation; never a real broker call
            assert (await advance(c,scope,now))[-1]=='reconcile'
            restored=s.states['NQ'].stop_order
            assert restored!=stop and b.status(restored)=='Submitted'
            assert b.trades[restored].order.totalQuantity==1 and b.trades[restored].order.orderType=='STP'
            assert len([x for x in b.calls if x[0]=='modify'])==1
            assert len([x for x in b.calls if x[0]=='new'])==1
            assert not s.coordination.legend_ready(scope,now[0])
        finally:store.close()
    asyncio.run(run())


@pytest.mark.parametrize('event',['revoked','broker_rejected'])
def test_unconfigured_legend_failure_policy_preserves_latch_and_original_exits(config,tmp_path,event):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,qualified=True)
        try:
            s.coordination.signal_status(scope,'legend-native',event)
            assert await c.poll(scope,now[0])=='reconcile'
            assert not s.coordination.allowed(scope,-1)
            assert b.trades[s.states['NQ'].stop_order].order.orderType=='STP'
            assert not [x for x in b.calls if x[0] in {'modify','new','cancel'}]
        finally:store.close()
    asyncio.run(run())


def test_direct_exit_cancel_is_refused_while_actual_owner_qty_remains(config,tmp_path):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path)
        try:
            snap=await c.adapter.refresh(scope)
            stop=next(v for v in snap.orders if v.role=='STOP')
            with pytest.raises(RuntimeError,match='never cancels protection'):
                c.adapter.cancel_exact(scope,stop)
            assert b.status(stop.order_id)=='Submitted'
        finally:store.close()
    asyncio.run(run())


def test_same_id_close_preserves_cumulative_total_after_prior_stop_partial(config,tmp_path):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,qualified=True)
        try:
            stop=s.states['NQ'].stop_order
            b.execute(stop,1,19990);await s.drain()
            assert s.states['NQ'].qty==1 and b.trades[stop].orderStatus.filled==1
            b.calls.clear()
            assert (await advance(c,scope,now))[-1]=='done'
            assert ('modify',stop,'MKT',2) in b.calls  # original total, not residual 1
            assert b.trades[stop].orderStatus.filled==2 and s.states['NQ'].qty==0
        finally:store.close()
    asyncio.run(run())


@pytest.mark.parametrize('missing',list(QUALIFIED.__dataclass_fields__))
def test_each_unproved_atomicity_requirement_keeps_protection(config,tmp_path,missing):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,qualified=True)
        try:
            c.adapter.qualification=replace(QUALIFIED,**{missing:False})
            assert (await advance(c,scope,now))[-1]=='reconcile'
            assert b.trades[s.states['NQ'].stop_order].order.orderType=='STP'
            assert not [x for x in b.calls if x[0] in {'new','modify','cancel'}]
        finally:store.close()
    asyncio.run(run())


def test_manual_handoff_during_partial_close_prevents_restoration_or_new_close(config,tmp_path):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,qualified=True)
        try:
            b.modify_mode='partial';stop=s.states['NQ'].stop_order
            await advance(c,scope,now,until='protected_working')
            s.operator_order(b.trades[stop])
            assert await c.poll(scope,now[0])=='reconcile'
            assert not [x for x in b.calls if x[0]=='new']
            assert len([x for x in b.calls if x[0]=='modify'])==1
        finally:store.close()
    asyncio.run(run())


@pytest.mark.parametrize('problem',['other_client','disconnected'])
def test_owner_connection_identity_is_verified_without_reconnecting(config,tmp_path,problem):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,qualified=True)
        try:
            if problem=='other_client':b.ib.client.clientId+=1
            else:b.ib.connected=False
            assert (await advance(c,scope,now))[-1]=='reconcile'
            assert not [x for x in b.calls if x[0] in {'new','modify','cancel'}]
        finally:store.close()
    asyncio.run(run())


def test_explicitly_denied_local_entry_is_no_send_evidence_not_fake_broker_cancellation(config,tmp_path):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,filled=0)
        try:
            original=next(row for row in store.orders().values() if row['role']=='ENTRY' and row['body']['side']==-1)
            oid=b.next_id();store.order(oid,'NQ','ENTRY',original['body'])
            with pytest.raises(EntryBlocked):s._coord_entry_send(oid,s.markets['NQ'],original['body'])
            assert oid not in b.trades
            assert (await advance(c,scope,now))[-1]=='done'
        finally:store.close()
    asyncio.run(run())


@pytest.mark.parametrize('edit',['stop_price','time_activation','time_quantity'])
def test_manual_protection_edit_between_prepare_and_send_preserves_existing_orders(config,tmp_path,edit):
    async def run():
        s,b,store,now,scope,c=await setup(config,tmp_path,qualified=True)
        try:
            assert await c.poll(scope,now[0])=='protected_prepare'
            assert await c.poll(scope,now[0])=='protected_submit'
            state=s.states['NQ']
            if edit=='stop_price':b.trades[state.stop_order].order.auxPrice+=1
            elif edit=='time_activation':b.trades[state.time_order].order.goodAfterTime=''
            else:b.trades[state.time_order].order.totalQuantity+=1
            assert await c.poll(scope,now[0])=='reconcile'
            assert b.trades[state.stop_order].order.orderType=='STP'
            assert not [x for x in b.calls if x[0] in {'modify','new','cancel'}]
        finally:store.close()
    asyncio.run(run())
