"""Cross-feature regressions on inert transports; no broker connection or commands."""
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace as NS

import pytest

from intraday_coordination import Ledger, LegendSignal, Scope
from open_breakout.ibkr import IBKR
from open_breakout.native_coordination import NativeOwnerAdapter, AsyncOwnerCoordinator
from open_breakout.replay import SimBroker
from open_breakout.service import Service
from test_existing_open_breakout import config, setup_service, tick, at, DAY, manifest
from test_native_adapter import NativeMock, QUALIFIED


class Cascade(SimBroker):
    def cancel(self, oid):
        group=self.orders[oid]['body'].get('oca')
        for other,row in list(self.orders.items()):
            if other==oid or (group and row['body'].get('oca')==group):
                super().cancel(other)


class NativePrices(NativeMock):
    """Use the real enabled snapshot/proof code on the inert native-shaped API."""
    snapshot=IBKR.snapshot
    _execution_books=IBKR._execution_books
    def __init__(self, config, clock):
        super().__init__(config,clock)
        self._exec_books=None;self._exec_at=0.;self._delivered=set();self.healthy=True

    def _fill(self, trade, fill):
        if fill.execution.execId in self._delivered:return
        self._delivered.add(fill.execution.execId)
        super()._fill(trade,fill)

    def send(self, oid, market, body):
        if (oid in self.trades and body['kind']=='STP' and self.trades[oid].order.orderType=='STP'):
            # The original owner fill handler resizes protection on each actual
            # entry partial. This is separate from the same-ID close prototype,
            # whose MKT modification must preserve the original native total.
            self.trades[oid].order.totalQuantity=body['qty']
        super().send(oid,market,body)
        self.trades[oid].order.lmtPrice=body.get('limit',0.)
        self.trades[oid].order.goodTillDate=body.get('good_till','')


async def armed(config, tmp_path, monkeypatch, *, native=False, qualified=False, broker_class=None):
    cfg=replace(config,entry_order_type='stop_limit',allow_price_only_pause=True)
    service,b,store,now=setup_service(cfg,tmp_path,broker_class or (NativePrices if native else Cascade))
    service.coordination=Ledger(tmp_path/'combined-coord.sqlite',enabled=True)
    if native:
        import open_breakout.ibkr as module
        class ClockDateTime(datetime):
            @classmethod
            def now(cls,tz=None):return now[0].astimezone(tz) if tz else now[0].replace(tzinfo=None)
        monkeypatch.setattr(module,'datetime',ClockDateTime)
        monkeypatch.setattr(module,'session_start',lambda:at('00:00:00'))
        adapter=NativeOwnerAdapter(service,qualification=QUALIFIED if qualified else None,
                                  filter_factory=lambda account:NS(acctCode=account))
        service.coordination_backend=AsyncOwnerCoordinator(adapter)
    await tick(service,b,now,20000.,'09:30:00');await service.watchdog()
    assert not service.halted and not service.price_paused and len(store.orders())==2
    return service,b,store,now


def claim(service, now, *, side=1, account=None, family='NQ', status=None):
    scope=Scope(DAY,account or service.config.account,family)
    signal=LegendSignal(scope,side,f'combined:{scope.key}:{side}',now[0],True,True,True,2)
    service.coordination.publish(signal,now[0])
    if status:service.coordination.signal_status(scope,signal.signal_id,status)
    return signal


async def settled(service):
    for _ in range(4):await service.watchdog();await service.drain()


async def recover(service,b,now,stamp='09:31:01',price=20000.):
    now[0]=at(stamp)
    b.update_quote('NQ',price-.25,price,now[0]);service.tick('NQ',now[0],price)
    await service.drain();await service.watchdog();await service.drain()


@pytest.mark.parametrize('native',[False,True])
@pytest.mark.parametrize('side',[-1,1])
def test_simultaneous_claim_and_price_gap_recovery_keeps_only_eligible_direction(config,tmp_path,monkeypatch,native,side):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch,native=native)
        try:
            opening=s.states['NQ'].opening;budget=store.get('daily_reserved')
            now[0]=at('09:31:00');sig=claim(s,now,side=side)
            await settled(s)
            assert s.price_paused and not s.halted and s.states['NQ'].phase=='FLAT'
            assert not any(row['status']=='Submitted' for row in b.orders.values())
            assert s.states['NQ'].attempts==1 and store.get('daily_reserved')==budget
            assert not s.states['NQ'].long_armed and not s.states['NQ'].short_armed
            await recover(s,b,now)
            assert not s.price_paused and not s.halted and len(store.orders())==2
            await tick(s,b,now,20000.,'09:31:02')
            working=[row['body'] for row in b.orders.values() if row['status']=='Submitted' and row['body']['kind']=='STP LMT']
            assert len(working)==1 and working[0]['side']==side
            assert s.states['NQ'].opening==opening and s.states['NQ'].attempts==2
            assert store.get('daily_reserved')>budget and not s.coordination.allowed(sig.scope,-side)
            assert (not s.states['NQ'].short_armed if side==1 else not s.states['NQ'].long_armed)
        finally:store.close()
    asyncio.run(run())


@pytest.mark.parametrize('status',['revoked','broker_rejected','delivery_unknown'])
def test_recovery_does_not_release_legend_latch_or_native_reconciliation_pause(config,tmp_path,monkeypatch,status):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch,native=True)
        try:
            now[0]=at('09:31:00');sig=claim(s,now,status=status)
            await settled(s)
            assert s.price_paused and 'NQ' in s.coord_reconcile
            count=len(store.orders());budget=store.get('daily_reserved')
            await recover(s,b,now)
            assert not s.price_paused and 'NQ' in s.coord_reconcile
            await tick(s,b,now,20000.,'09:31:02')
            assert len(store.orders())==count and store.get('daily_reserved')==budget
            assert not s._coord_allowed('NQ',1) and not s._coord_allowed('NQ',-1)
            assert not s.coordination.allowed(sig.scope,-1) and not s.coordination.legend_ready(sig.scope,now[0])
            assert s.coordination_backend.adapter.qualification is None
        finally:store.close()
    asyncio.run(run())


@pytest.mark.parametrize('kind',['account','family','day'])
def test_claim_scope_is_exact_and_does_not_cancel_other_account_family_or_day(config,tmp_path,monkeypatch,kind):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch,broker_class=SimBroker)
        try:
            now[0]=at('09:31:00')
            if kind=='day':
                older=now[0]-timedelta(days=1)
                scope=Scope(older.date().isoformat(),s.config.account,'NQ')
                s.coordination.publish(LegendSignal(scope,1,'yesterday',older,True,True,True,1),older)
            else:claim(s,now,account='PA_TEST' if kind=='account' else None,family='ES' if kind=='family' else 'NQ')
            await s.watchdog();await s.drain()
            assert s.price_paused and not s.halted
            assert all(b.status(oid)=='Submitted' for oid in s.states['NQ'].entry_orders)
            assert s._coord_allowed('NQ',-1)
        finally:store.close()
    asyncio.run(run())


def test_queries_and_claims_do_not_reset_price_grace(config,tmp_path,monkeypatch):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch,broker_class=SimBroker)
        try:
            start=now[0];original=dict(s.price_observed)
            now[0]=at('09:31:00');claim(s,now)
            await s.watchdog();await s.drain()
            buy=next(oid for oid,row in b.orders.items() if row['body']['side']==1)
            assert b.status(buy)=='Submitted'
            now[0]=start+timedelta(seconds=599.999)
            await s.watchdog();await s.drain()
            assert not s.halted and b.status(buy)=='Submitted' and s.price_observed==original
            now[0]=start+timedelta(seconds=600)
            await s.watchdog();await s.drain()
            assert s.halted and b.status(buy)=='Cancelled'
            assert 'PRICE_OUTAGE_GRACE_EXPIRED' in store.get('halt_reason')
        finally:store.close()
    asyncio.run(run())


def test_actual_partial_query_fill_during_gap_and_claim_protects_once_without_qualifying_close(config,tmp_path,monkeypatch):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch,native=True)
        try:
            sell=next(oid for oid,row in store.orders().items() if row['body']['side']==-1)
            budget=store.get('daily_reserved');now[0]=at('09:30:41')
            b.execute(sell,1,19990.,deliver=False)
            assert not s.states['NQ'].qty
            now[0]=at('09:31:00');sig=claim(s,now)
            await settled(s)
            state=s.states['NQ'];stop,timed=state.stop_order,state.time_order
            assert s.price_paused and state.qty==1 and state.side==-1
            assert b.status(stop)==b.status(timed)=='Submitted'
            assert b.trades[stop].order.orderType=='STP' and b.trades[stop].order.totalQuantity==1
            assert len(store.fill_ids())==1 and 'NQ' in s.coord_reconcile
            assert store.get('daily_reserved')==budget and state.attempts==1
            assert not s.coordination.legend_ready(sig.scope,now[0]) and not b.calls.count(('modify',stop,'MKT',1))
            await recover(s,b,now,price=19995.)
            await s.watchdog();await s.drain()
            assert not s.price_paused and state.qty==1 and state.stop_order==stop and state.time_order==timed
            assert len(store.fill_ids())==1 and s.coordination_backend.adapter.qualified() is False
            assert not [row for row in b.calls if row[0]=='modify']
        finally:store.close()
    asyncio.run(run())


@pytest.mark.parametrize('native',[False,True])
def test_claim_cancel_fill_race_during_price_gap_preserves_stop_time_and_blocks_new_entries(config,tmp_path,monkeypatch,native):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch,native=native)
        try:
            sell=next(oid for oid,row in store.orders().items() if row['body']['side']==-1)
            if native:
                b.before_cancel=lambda oid:b.execute(oid,1,19990.) if oid==sell else None
            else:
                original=b.cancel
                def racing(oid):
                    if oid==sell and b.status(oid)=='Submitted':b.execute(oid,1,19990.)
                    original(oid)
                b.cancel=racing
            now[0]=at('09:31:00');claim(s,now)
            await settled(s)
            state=s.states['NQ'];stop,timed=state.stop_order,state.time_order
            assert state.qty==1 and state.side==-1 and 'NQ' in s.coord_reconcile
            assert b.status(stop)==b.status(timed)=='Submitted'
            count=len(store.orders());await recover(s,b,now)
            await tick(s,b,now,20000.,'09:31:02')
            assert len(store.orders())==count and state.stop_order==stop and state.time_order==timed
            assert not s.working_entries(state) and not [r for r in store.orders().values() if r['role']=='COORD_CLOSE']
        finally:store.close()
    asyncio.run(run())


@pytest.mark.parametrize('failure',['client','disconnect','executions'])
def test_execution_failure_has_priority_over_price_grace_and_legend_claim(config,tmp_path,monkeypatch,failure):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch,native=True)
        try:
            now[0]=at('09:31:00');claim(s,now)
            if failure=='client':b.ib.client.clientId+=1
            elif failure=='disconnect':b.ib.connected=False
            else:
                async def unavailable(filter):raise TimeoutError('fresh executions unavailable')
                b.ib.reqExecutionsAsync=unavailable
            await s.watchdog();await s.drain()
            assert s.halted and not s.working_entries(s.states['NQ'])
            assert 'PRICE_OUTAGE_GRACE_EXPIRED' not in store.get('halt_reason')
            count=len(store.orders());await recover(s,b,now)
            assert s.halted and len(store.orders())==count
            assert not [row for row in b.calls if row[0]=='modify']
        finally:store.close()
    asyncio.run(run())


def test_missing_completed_receipt_remains_manual_block_after_price_recovery(config,tmp_path,monkeypatch):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch,native=True)
        try:
            now[0]=at('09:31:00');claim(s,now);await s.watchdog();await s.drain()
            b.ib.omit_completed.add(s.states['NQ'].entry_orders[0])
            await s.watchdog();await s.drain()
            assert 'NQ' in s.coord_reconcile
            count=len(store.orders());budget=store.get('daily_reserved')
            await recover(s,b,now);await tick(s,b,now,20000.,'09:31:02')
            assert not s.price_paused and 'NQ' in s.coord_reconcile and len(store.orders())==count
            assert store.get('daily_reserved')==budget and not s._coord_allowed('NQ',1)
        finally:store.close()
    asyncio.run(run())


@pytest.mark.parametrize('reason',['MANUAL_HALT','PROTECTION_UNCERTAIN','PRICE_OUTAGE_GRACE_EXPIRED:NQ:trade'])
def test_hard_halt_never_begins_fixture_qualified_stop_to_market_close(config,tmp_path,monkeypatch,reason):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch,native=True,qualified=True)
        try:
            sell=next(oid for oid,row in store.orders().items() if row['body']['side']==-1)
            b.execute(sell,1,19990.);await s.drain()
            stop,timed=s.states['NQ'].stop_order,s.states['NQ'].time_order
            now[0]=at('09:31:00');sig=claim(s,now)
            bridge=s.coordination_backend
            assert await bridge.poll(sig.scope,now[0])=='protected_prepare'
            assert await bridge.poll(sig.scope,now[0])=='protected_submit'
            s.halt(reason);await s.drain();b.calls.clear()
            assert await bridge.poll(sig.scope,now[0])=='reconcile'
            assert b.status(stop)==b.status(timed)=='Submitted' and b.trades[stop].order.orderType=='STP'
            assert not [row for row in b.calls if row[0] in {'modify','new','cancel'}]
        finally:store.close()
    asyncio.run(run())


def test_price_grace_expiry_precedes_awaited_qualified_coordination_close(config,tmp_path,monkeypatch):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch,native=True,qualified=True)
        try:
            sell=next(oid for oid,row in store.orders().items() if row['body']['side']==-1)
            b.execute(sell,1,19990.);await s.drain()
            stop=s.states['NQ'].stop_order;now[0]=at('09:31:00');sig=claim(s,now)
            bridge=s.coordination_backend
            assert await bridge.poll(sig.scope,now[0])=='protected_prepare'
            assert await bridge.poll(sig.scope,now[0])=='protected_submit'
            now[0]=at('09:40:00');b.calls.clear();await s.watchdog();await s.drain()
            assert s.halted and 'PRICE_OUTAGE_GRACE_EXPIRED' in store.get('halt_reason')
            assert b.status(stop)=='Submitted' and b.trades[stop].order.orderType=='STP'
            assert not [row for row in b.calls if row[0]=='modify']
        finally:store.close()
    asyncio.run(run())


def test_restart_keeps_legend_latch_risk_and_protection_without_resubmission(config,tmp_path,monkeypatch):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch,native=True)
        try:
            sell=next(oid for oid,row in store.orders().items() if row['body']['side']==-1)
            b.execute(sell,1,19990.);await s.drain()
            now[0]=at('09:31:00');sig=claim(s,now);await settled(s)
            old=s.states['NQ'];count=len(store.orders());budget=store.get('daily_reserved')
            restarted=Service(s.config,manifest(s.config),store,b,lambda:now[0],coordination=s.coordination)
            await restarted.drain();await restarted.watchdog();await restarted.drain()
            assert restarted.halted and 'RESTART_REQUIRES_READ_ONLY_RECONCILIATION' in store.get('halt_reason')
            assert len(store.orders())==count and store.get('daily_reserved')==budget
            assert restarted.states['NQ'].attempts==old.attempts and restarted.states['NQ'].opening==old.opening
            assert 'NQ' in restarted.coord_reconcile and not restarted._coord_allowed('NQ',-1)
            assert b.status(old.stop_order)==b.status(old.time_order)=='Submitted'
            assert not restarted.coordination.legend_ready(sig.scope,now[0])
        finally:store.close()
    asyncio.run(run())


def test_poll_and_snapshot_concurrency_is_bounded_and_uses_one_owner_lock(config,tmp_path,monkeypatch):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch,native=True)
        try:
            now[0]=at('09:31:00');claim(s,now)
            active=[0];peak=[0];original=b.ib.read
            async def tracked(kind):
                active[0]+=1;peak[0]=max(peak[0],active[0])
                try:
                    await asyncio.sleep(.005);await original(kind)
                finally:active[0]-=1
            b.ib.read=tracked
            first=asyncio.create_task(s._coord_poll());await asyncio.sleep(0)
            for _ in range(25):s._coord_queue_poll()
            assert len(s.tasks)<=1
            await asyncio.gather(first,s.watchdog(),*(s._coord_poll() for _ in range(12)))
            await s.drain()
            assert peak[0]==1 and not s.coord_poll_lock.locked() and not s.coord_poll_queued
            assert s.price_paused and not s.halted
        finally:store.close()
    asyncio.run(run())


@pytest.mark.parametrize('gap',[False,True])
def test_atomic_legend_publication_and_price_race_after_planning_cannot_leak_opposite_entry(config,tmp_path,monkeypatch,gap):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch)
        try:
            await s.cancel_resting_entries();now[0]=at('09:31:00')
            b.update_quote('NQ',19999.75,20000.,now[0]);s.tick('NQ',now[0],20000.)
            # The stream-gap callback pauses. Clear only after fresh execution evidence.
            await s.drain();await s.watchdog()
            original=s.coordination.submit_entry;once=[True]
            def publish_before_send(scope,side,token,identity,submit):
                if once[0]:
                    once[0]=False;claim(s,now,side=-1)
                    if gap:
                        now[0]+=timedelta(seconds=4)
                        b.update_quote('NQ',19999.75,20000.,now[0])
                return original(scope,side,token,identity,submit)
            s.coordination.submit_entry=publish_before_send
            budget=store.get('daily_reserved');old_ids=set(b.orders)
            await tick(s,b,now,20000.,'09:31:01')
            new=[row['body'] for oid,row in b.orders.items() if oid not in old_ids]
            assert not any(row['side']==1 for row in new)
            assert (not new and s.price_paused) if gap else (len(new)==1 and new[0]['side']==-1)
            assert s.states['NQ'].attempts==2 and store.get('daily_reserved')>budget
            assert not s.coordination.allowed(s._coord_scope('NQ'),1)
        finally:store.close()
    asyncio.run(run())


def test_queried_first_partial_and_second_partial_at_cancel_resize_protection_once(config,tmp_path,monkeypatch):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch,native=True)
        try:
            sell=next(oid for oid,row in store.orders().items() if row['body']['side']==-1)
            now[0]=at('09:30:41');b.execute(sell,1,19990.,deliver=False)
            # The first sibling cancel can cancel this entry through OCA. Its
            # second execution arrives before that cancel is acknowledged.
            b.before_cancel=lambda oid:b.execute(sell,1,19989.75)
            now[0]=at('09:31:00');claim(s,now)
            await settled(s)
            state=s.states['NQ']
            assert state.qty==2 and state.entry==19989.875 and len(store.fill_ids())==2
            assert b.trades[state.stop_order].order.totalQuantity==2 and b.trades[state.time_order].order.totalQuantity==2
            assert b.status(state.stop_order)==b.status(state.time_order)=='Submitted'
            assert 'NQ' in s.coord_reconcile and not [row for row in b.calls if row[0]=='modify' and row[2]=='MKT']
            await recover(s,b,now,price=19995.)
            assert not s.price_paused and state.qty==2 and len(store.fill_ids())==2
        finally:store.close()
    asyncio.run(run())


@pytest.mark.parametrize('limit',['attempts','risk'])
def test_price_recovery_and_same_direction_eligibility_cannot_reset_entry_limits(config,tmp_path,monkeypatch,limit):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch,native=True)
        try:
            now[0]=at('09:31:00');claim(s,now);await settled(s)
            state=s.states['NQ'];count=len(store.orders())
            if limit=='attempts':state.attempts=3;store.save(state)
            else:store.set('daily_reserved',s.config.shadow_equity*s.config.max_daily_risk_bps/10000)
            budget=store.get('daily_reserved');attempts=state.attempts
            await recover(s,b,now);await tick(s,b,now,20000.,'09:31:02')
            assert not s.price_paused and len(store.orders())==count
            assert state.attempts==attempts and store.get('daily_reserved')==budget
            assert s.coordination.allowed(s._coord_scope('NQ'),1) and not s.coordination.allowed(s._coord_scope('NQ'),-1)
        finally:store.close()
    asyncio.run(run())


def test_recovery_after_entry_cutoff_cannot_repark_or_extend_legend_deadline(config,tmp_path,monkeypatch):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch,native=True)
        try:
            now[0]=at('09:31:00');sig=claim(s,now);await settled(s)
            count=len(store.orders());budget=store.get('daily_reserved')
            # Genuine callbacks keep each outage below 600 seconds, while recovery
            # waits for verified execution evidence until after the entry cutoff.
            while now[0]+timedelta(seconds=500)<at('11:29:59'):
                now[0]+=timedelta(seconds=500)
                b.update_quote('NQ',19999.75,20000.,now[0]);s.tick('NQ',now[0],20000.)
                await s.drain()
            await recover(s,b,now,stamp='11:30:01')
            await tick(s,b,now,20000.,'11:30:02')
            assert not s.halted and not s.price_paused and len(store.orders())==count
            assert s.states['NQ'].attempts==1 and store.get('daily_reserved')==budget
            assert not s.coordination.legend_ready(sig.scope,now[0])
        finally:store.close()
    asyncio.run(run())


def test_manual_market_control_survives_price_recovery_and_claim(config,tmp_path,monkeypatch):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch,native=True)
        try:
            now[0]=at('09:30:41');await s.watchdog();assert s.price_paused
            s.manual_markets.add('NQ');store.set('manual_markets',['NQ'])
            now[0]=at('09:31:00');claim(s,now);b.calls.clear()
            count=len(store.orders());await s.watchdog();await recover(s,b,now)
            assert 'NQ' in s.manual_markets and 'NQ' in s.coord_reconcile
            assert len(store.orders())==count and not [row for row in b.calls if row[0] in {'new','modify','cancel'}]
            assert all(b.status(oid)=='Submitted' for oid in s.states['NQ'].entry_orders)
        finally:store.close()
    asyncio.run(run())


def test_price_callback_expiry_during_owner_read_blocks_pending_qualified_close(config,tmp_path,monkeypatch):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch,native=True,qualified=True)
        try:
            sell=next(oid for oid,row in store.orders().items() if row['body']['side']==-1)
            b.execute(sell,1,19990.);await s.drain()
            stop=s.states['NQ'].stop_order;now[0]=at('09:31:00');sig=claim(s,now)
            bridge=s.coordination_backend
            assert await bridge.poll(sig.scope,now[0])=='protected_prepare'
            assert await bridge.poll(sig.scope,now[0])=='protected_submit'
            def price_arrives_late():
                now[0]=at('09:40:00')
                b.update_quote('NQ',19994.75,19995.,now[0])
            b.ib.read_hook=price_arrives_late;b.calls.clear()
            assert await bridge.poll(sig.scope,at('09:31:00'))=='reconcile'
            await s.drain()
            assert s.halted and 'PRICE_OUTAGE_GRACE_EXPIRED' in store.get('halt_reason')
            assert b.status(stop)=='Submitted' and b.trades[stop].order.orderType=='STP'
            assert not [row for row in b.calls if row[0]=='modify']
        finally:store.close()
    asyncio.run(run())
