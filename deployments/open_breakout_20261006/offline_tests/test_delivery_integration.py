"""Production-payload races: filled breakout, feed loss, stop fill, Legend and recovery."""
import asyncio
import copy
from types import SimpleNamespace as NS

import pytest

from test_combined import armed, claim, recover, settled
from test_existing_open_breakout import config, at


@pytest.mark.parametrize('side', [-1, 1])
@pytest.mark.parametrize('foreign_scope', ['account', 'family'])
@pytest.mark.parametrize('stop_before_claim', [False, True])
def test_unqualified_filled_handoff_keeps_protection_and_foreign_orders_through_recovery(
        config, tmp_path, monkeypatch, side, foreign_scope, stop_before_claim):
    async def run():
        s,b,store,now=await armed(config,tmp_path,monkeypatch,native=True)
        try:
            entry=next(oid for oid,row in store.orders().items() if row['body']['side']==-side)
            foreign=copy.deepcopy(b.trades[entry])
            foreign.order.orderId=900; foreign.order.permId=10900; foreign.order.ocaGroup='foreign-only'
            if foreign_scope=='account':
                foreign.order.account='OTHER_ACCOUNT'
            else:
                foreign.contract=NS(conId=999,symbol='MES')
                foreign.order.orderRef=foreign.order.orderRef.replace('MNQ|','MES|')
            b.trades[900]=foreign
            b.orders[900]=dict(body=dict(b.orders[entry]['body'],oca='foreign-only'),market=b.orders[entry]['market'],status='Submitted')
            now[0]=at('09:30:40'); b.execute(entry,2,19990.)
            await s.drain()
            state=s.states['NQ']; stop,timed=state.stop_order,state.time_order
            assert state.qty==2 and b.status(stop)==b.status(timed)=='Submitted'
            attempts=state.attempts; risk=store.get('daily_reserved'); opening=state.opening
            if stop_before_claim:
                b.execute(stop,1,state.stop); await s.drain()
            now[0]=at('09:31:00'); signal=claim(s,now,side=side)
            await settled(s)
            assert s.price_paused and 'NQ' in s.coord_reconcile
            assert s.coordination_backend.adapter.qualification is None
            assert not s.coordination.legend_ready(signal.scope,now[0])
            if not stop_before_claim:
                b.execute(stop,1,state.stop); await s.drain()
            # Duplicate actual callback may arrive after the terminal/partial receipt.
            fill=b.native_fills[-1].execution
            s.fill(fill.orderId,fill.execId,fill.shares,fill.price); await s.drain()
            assert state.qty==1 and len(store.fill_ids())==2
            assert b.status(stop)==b.status(timed)=='Submitted'
            assert b.trades[stop].order.orderType=='STP'
            await recover(s,b,now,price=state.stop-side*5.)
            assert not s.price_paused and 'NQ' in s.coord_reconcile
            assert state.opening==opening and state.attempts==attempts and store.get('daily_reserved')==risk
            assert state.qty==1 and not s.coordination.allowed(signal.scope,-side)
            assert b.status(900)=='Submitted'
            assert not any(row[0]=='modify' and row[2]=='MKT' for row in b.calls)
            # Price recovery and stop settlement never release the durable opposing latch.
            b.execute(stop,1,state.stop); await s.drain(); await settled(s)
            assert state.qty==0 and b.status(900)=='Submitted'
            assert not s.coordination.allowed(signal.scope,-side)
            assert state.attempts==attempts and store.get('daily_reserved')==risk
        finally:
            store.close()
    asyncio.run(run())
