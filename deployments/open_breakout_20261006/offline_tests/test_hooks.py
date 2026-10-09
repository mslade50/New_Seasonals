import asyncio
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace as NS

import pytest

from intraday_coordination import Ledger, LegendSignal, Scope
from legend.coordination_bridge import coordinate_before_entry, record_legend_result
from test_coordination import Broker, CloseController, at as coord_at, scope, signal, ledger
from test_existing_open_breakout import config, setup_service, tick, at, DAY


def make_signal(service, market='NQ', side=1):
    return LegendSignal(Scope.for_symbol(DAY, service.config.account, service.markets[market].execution.symbol),
                        side, f'Legend:{market}:{side}', at('09:31:00'), True, True, True, 2)


@pytest.mark.parametrize('side', [-1, 1])
def test_actual_resting_service_cancels_only_opposite_and_keeps_same_eligible(config, tmp_path, side):
    async def run():
        service, broker, store, now = setup_service(replace(config, entry_order_type='stop_limit'), tmp_path)
        service.coordination = Ledger(tmp_path/'coord.db', enabled=True)
        try:
            await tick(service, broker, now, 20000, '09:30:00')
            before = store.orders()
            entries = {o['body']['side']: oid for oid, o in before.items() if o['role'] == 'ENTRY'}
            assert len(entries) == 2  # 09:30 timing stays unchanged
            now[0] = at('09:31:00')
            service.coordination.publish(make_signal(service, side=side), now[0])
            await service._coord_poll()
            assert broker.status(entries[-side]) == 'Cancelled'
            assert broker.status(entries[side]) == 'Submitted'
            assert not service.halted
            price = 20000 + side*10
            broker.update_quote('NQ', price-.25, price, now[0])
            broker.update_trade('NQ', price, now[0])
            await service.drain()
            assert service.states['NQ'].qty > 0 and service.states['NQ'].side == side
            assert not service.halted
        finally:
            store.close()
    asyncio.run(run())


def test_oca_cascade_can_repark_same_side_but_never_opposite(config, tmp_path):
    async def run():
        service, b, store, now = setup_service(replace(config, entry_order_type='stop_limit'), tmp_path)
        service.coordination = Ledger(tmp_path/'coord.db', enabled=True)
        try:
            await tick(service, b, now, 20000, '09:30:00')
            # Preserve the existing stream-health guard through the 09:31 handoff.
            for second in range(1, 60):
                await tick(service, b, now, 20000, f'09:30:{second:02d}')
            original = b.cancel
            def cascade(oid):
                group = b.orders[oid]['body'].get('oca')
                for key, order in list(b.orders.items()):
                    if order['body'].get('oca') == group:
                        original(key)
            b.cancel = cascade
            now[0] = at('09:31:00')
            service.coordination.publish(make_signal(service), now[0])
            await service._coord_poll()
            assert service.states['NQ'].phase == 'FLAT' and not service.halted
            await tick(service, b, now, 20000, '09:31:01')
            working = [o for o in b.orders.values() if o['status'] == 'Submitted' and o['body']['kind'] == 'STP LMT']
            assert working and {o['body']['side'] for o in working} == {1}
            scope_key = make_signal(service).scope
            service.coordination.signal_status(scope_key, make_signal(service).signal_id, 'revoked')
            await tick(service, b, now, 19980, '09:31:02')
            assert not service.states['NQ'].qty and not service.coordination.allowed(scope_key, -1)
        finally:
            store.close()
    asyncio.run(run())


def test_source_cancel_fill_race_keeps_protection_without_unqualified_close(config, tmp_path):
    async def run():
        service, b, store, now = setup_service(replace(config, entry_order_type='stop_limit'), tmp_path)
        service.coordination = Ledger(tmp_path/'coord.db', enabled=True)
        try:
            await tick(service, b, now, 20000, '09:30:00')
            opposite = next(oid for oid, o in store.orders().items() if o['role']=='ENTRY' and o['body']['side']==-1)
            original = b.cancel
            def racing_cancel(oid):
                if oid == opposite and b.status(oid) == 'Submitted':
                    b.execute(oid, 1, 19990)
                original(oid)
            b.cancel = racing_cancel
            now[0] = at('09:31:00')
            service.coordination.publish(make_signal(service), now[0])
            await service._coord_poll(); await service.drain()
            s = service.states['NQ']
            assert s.qty == 1 and s.side == -1
            assert b.status(s.stop_order) == b.status(s.time_order) == 'Submitted'
            assert 'NQ' in service.coord_reconcile
            assert not any(o['role']=='COORD_CLOSE' for o in store.orders().values())
        finally:
            store.close()
    asyncio.run(run())


class Clock:
    def __init__(self): self.at = coord_at()
    def now(self): return self.at
    def sleep(self, seconds): self.at += timedelta(seconds=seconds)


def legend_args():
    return (NS(account='PRIMARY_TEST', allow_longs=True, allow_shorts=False),
            NS(symbol='MES', forced=False, setup=NS(qualified=True), qty=2, sig='final'),
            NS(side='BUY'), NS(strategy='Legend_EMA', transmit_deadline=coord_at('09:31:20')))


@pytest.mark.parametrize('invalid', ['unarmed', 'forced', 'unqualified', 'zero', 'test_window', 'short_disabled'])
def test_legend_bridge_does_not_publish_candidates(tmp_path, invalid):
    ledger = Ledger(tmp_path/'coord.db', enabled=True); cfg, st, dec, sched = legend_args(); clock=Clock()
    armed = invalid != 'unarmed'
    if invalid=='forced': st.forced=True
    if invalid=='unqualified': st.setup.qualified=False
    if invalid=='zero': st.qty=0
    if invalid=='test_window': sched.strategy='Legend_EMA_TEST'
    if invalid=='short_disabled': dec.side='SELL_SHORT'
    assert not coordinate_before_entry(ledger,cfg,st,dec,sched,clock,armed=armed,sleeper=clock.sleep)
    assert ledger.read(scope())['signal'] is None


def test_disabled_bridge_preserves_decision_timing():
    cfg, st, dec, sched = legend_args(); clock = Clock(); before=clock.now()
    assert coordinate_before_entry(None,cfg,st,dec,sched,clock,armed=True,sleeper=clock.sleep)
    assert clock.now() == before


def test_legend_waits_for_settled_breakout_within_existing_deadline(tmp_path):
    ledger=Ledger(tmp_path/'coord.db',enabled=True); b=Broker(); b.add_entry(ledger, filled=1)
    cfg,st,dec,sched=legend_args(); clock=Clock(); controller=CloseController(ledger,b)
    def sleep(seconds):
        clock.sleep(seconds); b.snapshot_at=clock.now(); controller.step(scope(),clock.now())
    assert coordinate_before_entry(ledger,cfg,st,dec,sched,clock,armed=True,sleeper=sleep)
    assert ledger.read(scope())['operation']['phase']=='done' and b.owned==0
    assert clock.now() <= sched.transmit_deadline


def test_legend_never_transmits_after_unsettled_deadline(tmp_path):
    ledger=Ledger(tmp_path/'coord.db',enabled=True); cfg,st,dec,sched=legend_args(); clock=Clock()
    assert not coordinate_before_entry(ledger,cfg,st,dec,sched,clock,armed=True,sleeper=clock.sleep)
    assert ledger.read(scope())['signal']['status']=='delivery_unknown'
    assert not ledger.allowed(scope(),-1)


def test_late_fill_after_settlement_invalidates_readiness(ledger):
    b=Broker(); row=b.add_entry(ledger); ledger.publish(signal(),coord_at()); controller=CloseController(ledger,b)
    from test_coordination import advance
    assert advance(controller)[-1]=='done'
    b.orders[row.token]=replace(b.orders[row.token],status='Submitted',remaining=1)
    b.fill(row.token,1)
    assert controller.step(scope(),coord_at())=='reconcile'
    assert not ledger.legend_ready(scope(),coord_at())


@pytest.mark.parametrize('result_status,lifecycle', [('REJECTED','broker_rejected'), ('PLACE_FAILED','delivery_unknown')])
def test_legend_source_result_retains_accepted_latch(tmp_path,result_status,lifecycle):
    ledger=Ledger(tmp_path/'coord.db',enabled=True); cfg,st,dec,sched=legend_args(); clock=Clock()
    accepted=LegendSignal(scope(),1,st.sig,clock.now(),True,True,True,2)
    ledger.publish(accepted,clock.now())
    record_legend_result(ledger,cfg,st,{'status':result_status},clock)
    assert ledger.read(scope())['signal']['status']==lifecycle
    assert ledger.allowed(scope(),1) and not ledger.allowed(scope(),-1)
